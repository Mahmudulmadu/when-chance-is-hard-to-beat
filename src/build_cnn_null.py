# -*- coding: utf-8 -*-
"""A deep baseline under the permutation null -- notebook 28.

The standing objection to this project is "only Random Forest". The standing
answer (notebooks 22-23) is that with eleven units the null band reaches 74 %,
so no architecture can be ranked. Both are arguments. This is the measurement:
one standard 1D-CNN -- WDCNN (Zhang, Peng, Li, Chen & Zhang 2017), the
wide-first-kernel network that is the default deep baseline in bearing
diagnosis -- put through exactly the tests the forest was put through.

  * **the exact permutation null** (nb22): every OR/IR assignment across the
    eleven units that preserves the class counts, C(11,3)=165 on XJTU-SY and
    C(11,5)=462 on Paderborn, each scored bearing-held-out;
  * **the deliberately-incapable control** (nb20/22): the same network fed
    crops whose samples have been shuffled in time, which keeps every
    amplitude statistic and destroys every waveform;
  * **the bearing-identity fingerprint** (nb21): can the network tell six
    nominally identical healthy Paderborn bearings apart, run-held-out, against
    16.7 % chance -- and, secondarily, fifteen XJTU-SY units from their first
    twenty healthy snapshots against 6.7 %.

What keeps this fair to both sides:

  * **No early stopping, no validation set.** A fixed epoch budget. Stopping on
    the held-out unit is the leakage notebook 05 caught in a different form.
  * **Paired seeds.** The weight init and the crop draws depend on the fold, not
    on the labelling, so the true labelling and all 164 / 461 permuted ones start
    from the same weights and see the same crops. The null is then a paired
    comparison rather than a noisy one.
  * **Identical scoring to nb22.** Logits averaged over non-overlapping crops of
    a snapshot (XJTU) or run (Paderborn), argmax, per-unit accuracy, mean over
    units. The forest's 94.1 % / 85.5 % were computed the same way.
  * **Per-crop standardisation.** Each 2 048-sample crop is z-scored, so the
    network sees waveform shape rather than gain -- the usual WDCNN practice,
    and it removes the easiest fingerprint (RMS) before the test asks about
    fingerprinting.

Usage:
  python src/build_cnn_null.py --rig xjtu --mode time            # seconds per fit
  python src/build_cnn_null.py --rig xjtu --mode null            # true + all permutations
  python src/build_cnn_null.py --rig pu   --mode null [--n-perm 200]
  python src/build_cnn_null.py --rig xjtu --mode control         # shuffled-sample control
  python src/build_cnn_null.py --rig pu   --mode fingerprint
  python src/build_cnn_null.py --rig pu   --mode null --input orders [--model speccnn|rf]
  python src/build_cnn_null.py --rig xjtu --mode true --epochs 24 --per-epoch 1536 --tag big
  python src/build_cnn_null.py --rig xjtu --mode trainfit        # train vs held-out, same fit

Run everything that is compared against the null at --threads 5, the thread count
the nulls use: CPU torch is not bit-stable across thread counts, and the true
labelling moved 35.9 % -> 33.2 % on Paderborn between 3 and 5 threads.
Writes data/processed/cnn_null_<rig>.csv, cnn_control_<rig>.csv, cnn_fingerprint_<rig>.csv;
the spectrum models add a suffix, e.g. cnn_null_pu_orders_speccnn.csv.

The spectrum probe (`--input orders`): the same folds, seeds, scoring and
caching applied to the order-domain envelope spectrum from
src/build_envelope_orders.py, with a small SpecCNN or a forest on all 576 bins.
It exists to answer "you gave the network the wrong representation" -- see the
notebook for why the answer is the unit count, not the input.
"""
import argparse
import itertools
import json
import pathlib
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"
CROP = 2048
FAULT_XJ = {"Bearing1_1": "OR", "Bearing1_2": "OR", "Bearing1_3": "OR",
            "Bearing1_4": "cage", "Bearing1_5": "IR+OR", "Bearing2_1": "IR",
            "Bearing2_2": "OR", "Bearing2_3": "cage", "Bearing2_4": "OR",
            "Bearing2_5": "OR", "Bearing3_1": "OR", "Bearing3_2": "compound",
            "Bearing3_3": "IR", "Bearing3_4": "IR", "Bearing3_5": "OR"}


# ----------------------------------------------------------------------------- model
class WDCNN(nn.Module):
    """Zhang et al. 2017: wide first kernel (64, stride 16), then four 3-wide
    conv blocks, each followed by BN, ReLU and max-pool 2; FC 100; softmax."""

    def __init__(self, n_classes):
        super().__init__()
        chans = [(1, 16, 64, 16, 24), (16, 32, 3, 1, 1), (32, 64, 3, 1, 1),
                 (64, 64, 3, 1, 1), (64, 64, 3, 1, 0)]
        blocks = []
        for ci, co, k, s, p in chans:
            blocks += [nn.Conv1d(ci, co, k, stride=s, padding=p), nn.BatchNorm1d(co),
                       nn.ReLU(), nn.MaxPool1d(2)]
        self.features = nn.Sequential(*blocks)
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 1, CROP)).numel()
        self.head = nn.Sequential(nn.Linear(flat, 100), nn.BatchNorm1d(100), nn.ReLU(),
                                  nn.Linear(100, n_classes))

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class SpecCNN(nn.Module):
    """The spectrum probe: three 9-wide conv blocks over the 576-bin order-domain
    envelope spectrum (src/build_envelope_orders.py), FC 100. Not WDCNN -- a
    second, smaller model for a second, smaller input."""

    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 9, padding=4), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 9, padding=4), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 9, padding=4), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(4))
        self.head = nn.Sequential(nn.Linear(256, 100), nn.BatchNorm1d(100), nn.ReLU(),
                                  nn.Linear(100, n_classes))

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def zscore(x):
    return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)


# ----------------------------------------------------------------------------- data
class Rig:
    """`input='raw'` is the decimated waveform cache (crops); `input='orders'` is
    the order-domain envelope spectrum (one 576-bin row per snapshot/run)."""

    def __init__(self, name, input="raw"):
        self.name, self.input = name, input
        src = "orders" if input == "orders_cal" else input
        self.X = np.load(OUT / f"{src}_{name}.npy", mmap_mode="r")
        self.meta = pd.read_csv(OUT / f"{src}_{name}_meta.csv")
        self.n_crops = self.X.shape[1] // CROP if input == "raw" else 1
        if input == "orders_cal":
            # each row minus its own unit's healthy reference -- nb22's forest features
            # carry exactly this calibration on XJTU-SY; Paderborn has no healthy history
            if name != "xjtu":
                raise SystemExit("orders_cal needs a healthy history per unit: XJTU-SY only (nb19)")
            B = np.load(OUT / "orders_xjtu_baseline.npy")
            idx = {u: i for i, u in enumerate(pd.read_csv(OUT / "orders_xjtu_baseline_units.csv")["unit"])}
            self.X = np.asarray(self.X, np.float32) - B[self.meta["unit"].map(idx).to_numpy()]
        self.units = list(dict.fromkeys(self.meta["unit"]))
        self.rows = {u: self.meta.index[self.meta["unit"] == u].to_numpy() for u in self.units}

    def train_crops(self, rows, n, rng):
        """n random crops from random rows, as a float32 tensor (n, 1, CROP)."""
        r = rng.choice(rows, n)
        s = rng.integers(0, self.X.shape[1] - CROP + 1, n)
        out = np.empty((n, CROP), np.float32)
        for i, (ri, si) in enumerate(zip(r, s)):
            out[i] = self.X[ri, si:si + CROP]
        return torch.from_numpy(out).unsqueeze(1)

    def eval_crops(self, rows):
        """All non-overlapping crops of the given rows: (len(rows)*n_crops, 1, CROP)."""
        a = np.asarray(self.X[rows, :self.n_crops * CROP], np.float32)
        return torch.from_numpy(a.reshape(len(rows) * self.n_crops, 1, CROP))


# ----------------------------------------------------------------------------- one fit
def fit_predict_orders(rig, tr_rows, tr_y, te_rows, n_classes, cfg, seed, shuffle=False):
    """The spectrum branch: whole 576-bin rows, no crops, dB/10 as the input scale
    (floor-relative, so no per-sample standardisation). `shuffle` permutes the
    bins, the spectral analogue of shuffling samples in time."""
    Xd = np.array(rig.X, dtype=np.float32) / 10.0        # a writable copy, off the mmap
    if cfg.get("model") == "rf":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=2)
        return m.fit(Xd[tr_rows], tr_y).predict(Xd[te_rows])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    perm = torch.randperm(Xd.shape[1]) if shuffle else None
    Xt = torch.from_numpy(Xd).unsqueeze(1)
    if perm is not None:
        Xt = Xt[:, :, perm]
    m = SpecCNN(n_classes)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    y_of = dict(zip(tr_rows, tr_y))
    steps = cfg["per_epoch"] // cfg["batch"]
    m.train()
    for _ in range(cfg["epochs"]):
        for _ in range(steps):
            r = rng.choice(tr_rows, cfg["batch"])
            yb = torch.tensor([y_of[i] for i in r])
            opt.zero_grad()
            F.cross_entropy(m(Xt[r]), yb).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        return m(Xt[te_rows]).argmax(1).numpy()


def fit_predict(rig, tr_rows, tr_y, te_rows, n_classes, cfg, seed, shuffle=False):
    """Train on (tr_rows, tr_y); return the per-row predicted class on te_rows."""
    if rig.input in ("orders", "orders_cal"):
        return fit_predict_orders(rig, tr_rows, tr_y, te_rows, n_classes, cfg, seed, shuffle)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = WDCNN(n_classes)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    y_of = dict(zip(tr_rows, tr_y))
    steps = cfg["per_epoch"] // cfg["batch"]
    m.train()
    for _ in range(cfg["epochs"]):
        for _ in range(steps):
            r = rng.choice(tr_rows, cfg["batch"])
            s = rng.integers(0, rig.X.shape[1] - CROP + 1, cfg["batch"])
            xb = np.empty((cfg["batch"], CROP), np.float32)
            for i, (ri, si) in enumerate(zip(r, s)):
                xb[i] = rig.X[ri, si:si + CROP]
            xb = torch.from_numpy(xb).unsqueeze(1)
            if shuffle:                               # the incapable control
                xb = xb[:, :, torch.randperm(CROP)]
            yb = torch.tensor([y_of[i] for i in r])
            opt.zero_grad()
            F.cross_entropy(m(zscore(xb)), yb).backward()
            opt.step()
    m.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(te_rows), cfg["eval_rows"]):
            rows = te_rows[i:i + cfg["eval_rows"]]
            xb = rig.eval_crops(rows)
            if shuffle:
                xb = xb[:, :, torch.randperm(CROP)]
            logit = m(zscore(xb)).view(len(rows), rig.n_crops, n_classes).mean(1)
            preds.append(logit.argmax(1).numpy())
    return np.concatenate(preds)


def score(rig, units, lab, cfg, shuffle=False, classes=("OR", "IR")):
    """nb22's score: bearing-held-out accuracy under an arbitrary labelling."""
    cls = {c: i for i, c in enumerate(classes)}
    per = {}
    for fi, h in enumerate(units):
        tr = [o for o in units if o != h]
        if len({lab[o] for o in tr}) < 2:
            continue
        tr_rows = np.concatenate([rig.rows[o] for o in tr])
        tr_y = np.concatenate([[cls[lab[o]]] * len(rig.rows[o]) for o in tr])
        p = fit_predict(rig, tr_rows, tr_y, rig.rows[h], len(classes), cfg,
                        seed=cfg["seed"] + fi, shuffle=shuffle)
        per[h] = float((p == cls[lab[h]]).mean())
    return float(np.mean(list(per.values()))), per


# ----------------------------------------------------------------------------- modes
def outpath(kind, rig, cfg):
    """raw/wdcnn keeps the original names; the spectrum models get a suffix, and
    `--tag` appends one more (e.g. `big` for the 4x-budget run)."""
    tag = "" if (rig.input == "raw" and cfg.get("model", "wdcnn") == "wdcnn") \
        else f"_{rig.input}_{cfg.get('model', 'speccnn')}"
    if cfg.get("tag"):
        tag += f"_{cfg['tag']}"
    return OUT / f"cnn_{kind}_{rig.name}{tag}.csv"


def race_units(rig):
    if rig.name == "xjtu":
        units = rig.units
        truth = {u: FAULT_XJ[u] for u in units}
    else:
        units = [u for u in rig.units if rig.meta.loc[rig.rows[u][0], "domain"] == "real"]
        truth = {u: rig.meta.loc[rig.rows[u][0], "fault"] for u in units}
    return units, truth


def mode_time(rig, cfg):
    units, truth = race_units(rig)
    h = units[0]
    tr = [o for o in units if o != h]
    tr_rows = np.concatenate([rig.rows[o] for o in tr])
    tr_y = np.concatenate([[0 if truth[o] == "OR" else 1] * len(rig.rows[o]) for o in tr])
    t0 = time.time()
    fit_predict(rig, tr_rows, tr_y, rig.rows[h], 2, cfg, seed=cfg["seed"])
    dt = time.time() - t0
    n_assign = {"xjtu": 165, "pu": 462}[rig.name]
    print(f"{rig.name}: one fit = {dt:.1f}s  (train rows {len(tr_rows)}, "
          f"held-out {len(rig.rows[h])} x {rig.n_crops} crops)")
    print(f"  exact null = {n_assign} assignments x {len(units)} folds = "
          f"{n_assign * len(units)} fits = {n_assign * len(units) * dt / 3600:.1f} h")


def mode_null(rig, cfg, n_perm):
    path = outpath("null", rig, cfg)
    units, truth = race_units(rig)
    minority = min(set(truth.values()), key=lambda c: sum(v == c for v in truth.values()))
    other = next(c for c in set(truth.values()) if c != minority)
    n_min = sum(v == minority for v in truth.values())
    combos = list(itertools.combinations(units, n_min))
    true_combo = tuple(u for u in units if truth[u] == minority)
    nulls = [c for c in combos if c != true_combo]
    if n_perm and n_perm < len(nulls):
        rng = np.random.default_rng(cfg["seed"])
        nulls = [nulls[i] for i in sorted(rng.choice(len(nulls), n_perm, replace=False))]
    todo = [true_combo] + nulls
    done = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done_keys = set(done["combo"]) if len(done) else set()
    print(f"{rig.name}: {len(units)} units, minority {minority} x{n_min}, "
          f"{len(combos)} assignments, running {len(todo)} ({len(done_keys)} cached)", flush=True)
    t0, rows = time.time(), []
    for i, combo in enumerate(todo):
        key = "|".join(combo)
        if key in done_keys:
            continue
        lab = {u: (minority if u in combo else other) for u in units}
        acc, per = score(rig, units, lab, cfg)
        rows.append(dict(rig=rig.name, combo=key, is_true=(combo == true_combo),
                         acc=acc, per_unit=json.dumps(per)))
        if len(rows) % 5 == 0 or combo == true_combo:
            pd.concat([done, pd.DataFrame(rows)]).to_csv(path, index=False)
            el = time.time() - t0
            left = sum("|".join(c) not in done_keys for c in todo[i + 1:])   # uncached only
            print(f"  {i + 1}/{len(todo)}  acc={acc:.3f}  {'TRUE' if combo == true_combo else ''}"
                  f"  ({el / 60:.0f} min, ~{el / max(len(rows), 1) * left / 60:.0f} min left)",
                  flush=True)
    pd.concat([done, pd.DataFrame(rows)]).to_csv(path, index=False)
    print(f"wrote {path.name}")


def mode_true(rig, cfg):
    """The true labelling alone -- for a budget or input the null was not run on.
    With the same seed, threads and budget it reproduces the null's own true row."""
    units, truth = race_units(rig)
    acc, per = score(rig, units, truth, cfg)
    steps = cfg["epochs"] * (cfg["per_epoch"] // cfg["batch"])
    pd.DataFrame([dict(rig=rig.name, steps=steps, acc=acc, per_unit=json.dumps(per))]).to_csv(
        outpath("true", rig, cfg), index=False)
    print(f"{rig.name} true labelling ({steps} steps): {acc:.3f}")


def mode_trainfit(rig, cfg):
    """Accuracy on a subsample of each fold's training units against its held-out
    unit, from the same fit: the memorisation check. 20 training rows per unit on
    XJTU-SY (some units have only 22), 10 of 20 runs per unit on Paderborn."""
    units, truth = race_units(rig)
    cls = {"OR": 0, "IR": 1}
    k = 20 if rig.name == "xjtu" else 10
    rng = np.random.default_rng(1)
    rows = []
    for fi, h in enumerate(units):
        tr = [o for o in units if o != h]
        tr_rows = np.concatenate([rig.rows[o] for o in tr])
        tr_y = np.concatenate([[cls[truth[o]]] * len(rig.rows[o]) for o in tr])
        sub = np.concatenate([rng.choice(rig.rows[o], min(k, len(rig.rows[o])), replace=False)
                              for o in tr])
        p = fit_predict(rig, tr_rows, tr_y, np.concatenate([sub, rig.rows[h]]), 2, cfg,
                        seed=cfg["seed"] + fi)
        ysub = np.array([cls[truth[rig.meta.loc[r, "unit"]]] for r in sub])
        rows.append(dict(rig=rig.name, held=h, train_acc=float((p[:len(sub)] == ysub).mean()),
                         heldout_acc=float((p[len(sub):] == cls[truth[h]]).mean())))
        print(f"  {h}: train {rows[-1]['train_acc']:.2f}  held-out {rows[-1]['heldout_acc']:.2f}",
              flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(outpath("trainfit", rig, cfg), index=False)
    print(f"{rig.name}: train {d['train_acc'].mean():.3f} vs held-out {d['heldout_acc'].mean():.3f}")


def mode_control(rig, cfg):
    units, truth = race_units(rig)
    acc, per = score(rig, units, truth, cfg, shuffle=True)
    pd.DataFrame([dict(rig=rig.name, acc=acc, per_unit=json.dumps(per))]).to_csv(
        outpath("control", rig, cfg), index=False)
    print(f"{rig.name} shuffled-sample control: {acc:.3f}")


def mode_fingerprint(rig, cfg):
    rows = []
    if rig.name == "pu":
        units = [u for u in rig.units if rig.meta.loc[rig.rows[u][0], "domain"] == "healthy"]
        folds = [list(range(1 + 5 * k, 6 + 5 * k)) for k in range(4)]       # runs held out
        key = "run"
    else:
        # XJTU: the cache holds post-K rows only, so the healthy fingerprint uses
        # the first 20 cached snapshots (K..K+19) as a pseudo-healthy early stretch
        units = rig.units
        folds = [list(range(20 + 5 * k, 25 + 5 * k)) for k in range(4)]
        key = "snapshot"
    cls = {u: i for i, u in enumerate(units)}
    for fi, held in enumerate(folds):
        m = rig.meta
        tr_rows = m.index[m["unit"].isin(units) & ~m[key].isin(held)
                          & (m[key] < 40 if rig.name == "xjtu" else True)].to_numpy()
        te_rows = m.index[m["unit"].isin(units) & m[key].isin(held)].to_numpy()
        tr_y = np.array([cls[u] for u in m.loc[tr_rows, "unit"]])
        p = fit_predict(rig, tr_rows, tr_y, te_rows, len(units), cfg, seed=cfg["seed"] + fi)
        te_y = np.array([cls[u] for u in m.loc[te_rows, "unit"]])
        rows.append(dict(rig=rig.name, fold=fi, n_train=len(tr_rows), n_test=len(te_rows),
                         acc=float((p == te_y).mean()), chance=1 / len(units)))
        print(f"  fold {fi}: {rows[-1]['acc']:.3f}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(outpath("fingerprint", rig, cfg), index=False)
    print(f"{rig.name} identity fingerprint: {d['acc'].mean():.3f} vs chance {1 / len(units):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", choices=["xjtu", "pu"], required=True)
    ap.add_argument("--mode", choices=["time", "null", "true", "control", "fingerprint", "trainfit"],
                    required=True)
    ap.add_argument("--tag", default="", help="suffix for the output file, e.g. big")
    ap.add_argument("--input", choices=["raw", "orders", "orders_cal"], default="raw",
                    help="raw waveform crops (WDCNN), order-domain envelope spectra, or those "
                         "spectra minus each unit's own healthy reference (XJTU-SY only)")
    ap.add_argument("--model", choices=["wdcnn", "speccnn", "rf"], default=None,
                    help="default: wdcnn for raw, speccnn for orders")
    ap.add_argument("--n-perm", type=int, default=0, help="0 = exact (all assignments)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--per-epoch", type=int, default=768)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval-rows", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model = a.model or ("wdcnn" if a.input == "raw" else "speccnn")
    if a.input == "raw" and model != "wdcnn":
        ap.error("raw input runs WDCNN only")
    cfg = dict(epochs=a.epochs, per_epoch=a.per_epoch, batch=a.batch,
               eval_rows=a.eval_rows, seed=a.seed, model=model, tag=a.tag)
    rig = Rig(a.rig, a.input)
    {"time": lambda: mode_time(rig, cfg), "null": lambda: mode_null(rig, cfg, a.n_perm),
     "true": lambda: mode_true(rig, cfg), "control": lambda: mode_control(rig, cfg),
     "fingerprint": lambda: mode_fingerprint(rig, cfg),
     "trainfit": lambda: mode_trainfit(rig, cfg)}[a.mode]()
