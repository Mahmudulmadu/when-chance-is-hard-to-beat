# -*- coding: utf-8 -*-
"""The unit-count power curve on XJTU-SY -- notebook 24.

Notebook 23 measured, on Paderborn, that an attribution claim needs about eight
units. That is itself a single-rig claim, and notebooks 20 and 21 showed exactly
how those fail: the load-zone channel cleared its null on XJTU-SY and vanished on
Paderborn. So the eight-unit figure has to be checked on the second rig before it
is quoted.

Two things differ here and both matter:

  * **The effect is larger.** Notebook 22 put geometry at 94.1 % against a null
    centred at 59 % on XJTU-SY, against 85.5 % / 44 % on Paderborn. A larger
    effect should be detectable with fewer units, so the threshold may be lower.
  * **The minority class is tiny.** XJTU-SY has 8 outer-race and only **3
    inner-race** units. A subset needs at least two of each class, otherwise
    leave-one-out leaves a single-class training set -- so every valid subset
    reuses two or three of the *same three* inner-race bearings, and the curve's
    points are far from independent.

Three channels are run so the curve carries controls rather than only positives.
Notebook 22 established geometry and load-zone are both real here (p = 0.0061
each); notebook 18's non-geometry channel reached 64.3 %, inside the null band,
and serves as the negative.

Usage:  python src/build_power_curve_xjtu.py
Writes data/processed/power_curve_xjtu.csv

    CORRECTION (notebook 24): the null must EXCLUDE the true labelling. When the
    true assignment is drawn into a sampled null, its score equals the true score,
    so p >= 1/N and the subset can never reach p < 0.05. That happens 71 % of the
    time on XJTU-SY at n=8 (k=2, C(8,2)=28 drawn 20) against 29 % on Paderborn,
    so it biases against exactly the datasets with a small minority class. The
    null is now the non-true labellings and p uses the standard
    (1 + #{null >= true}) / (1 + N) form, whose floor is 1/21 = 0.048.
"""
import argparse
import itertools
import os
import pathlib
import time
import warnings

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"
# n_jobs=1 and 15 trees for the same reasons as the Paderborn curve: measured
# 33 ms/fit against 48 with n_jobs=-1 and 227 with 100 trees. Both the true and
# the null side use the same forest, so this costs precision, not comparability.
RF = dict(n_estimators=15, random_state=0, n_jobs=1)
# Referee M3: see the note in build_power_curve.py. 30 subsets per size, and the
# subset's null enumerated exactly wherever C(n,k)-1 <= 40.
N_SUBSETS, PERM_CAP = 30, 40
K, W = 20, 10
RNG = np.random.default_rng(0)

FAULT = {"Bearing1_1": "OR", "Bearing1_2": "OR", "Bearing1_3": "OR",
         "Bearing1_4": "cage", "Bearing1_5": "IR+OR", "Bearing2_1": "IR",
         "Bearing2_2": "OR", "Bearing2_3": "cage", "Bearing2_4": "OR",
         "Bearing2_5": "OR", "Bearing3_1": "OR", "Bearing3_2": "compound",
         "Bearing3_3": "IR", "Bearing3_4": "IR", "Bearing3_5": "OR"}
GEO = ["H_BPFO", "H_BPFI", "H_FR", "H_BPFO_h2", "H_BPFI_h2", "H_BSF", "H_FTF",
       "H_BPFI_sb1", "H_BPFO_sb1"]


class Exceedance:
    def fit(self, A):
        self.med = np.median(A, axis=0)
        self.mad = np.maximum(1.4826 * np.median(np.abs(A - self.med), axis=0), 0.1)
        return self

    def score(self, A):
        return ((A - self.med) / self.mad).max(axis=1)


def load():
    env = pd.read_csv(OUT / "xjtu_envelope_features.csv")
    lz = pd.read_csv(OUT / "xjtu_loadzone_features.csv")
    ng = pd.read_csv(OUT / "xjtu_nongeometry_features.csv")
    d = env.merge(lz, on=["Bearing", "Snapshot"], how="inner")
    d = d.merge(ng, on=["Bearing", "Snapshot"], how="inner", suffixes=("", "_n"))
    race = [b for b in sorted(d["Bearing"].unique()) if FAULT[b] in ("OR", "IR")]
    G = {b: d[d["Bearing"] == b].sort_values("Snapshot").reset_index(drop=True)
         for b in race}
    lzf = [c for c in lz.columns if c.startswith("LZ_") and c != "LZ_frdev"]
    ngf = [c for c in ng.columns if c.startswith("H_") and c not in GEO]

    def post(feats):
        out = {}
        for b in race:
            A = G[b][["H_RMS", "H_Kurt"]].to_numpy()
            z = Exceedance().fit(A[:K]).score(A)
            c = np.convolve((z[K:] > np.quantile(z[:K], 0.99)).astype(float),
                            np.ones(W) / W, "valid")
            a = int(np.flatnonzero(c >= 0.5)[0]) + K
            M = G[b][feats].to_numpy()
            out[b] = (M - np.median(M[:K], axis=0))[a:]
        return out

    return race, {b: FAULT[b] for b in race}, {
        "geometry": post(GEO), "load-zone": post(lzf), "non-geometry": post(ngf)}


def score(P, units, lab):
    acc = []
    for h in units:
        tr = [o for o in units if o != h]
        y = np.concatenate([[lab[o]] * len(P[o]) for o in tr])
        if len(set(y)) < 2:
            continue
        m = RandomForestClassifier(**RF).fit(np.vstack([P[o] for o in tr]), y)
        acc.append((m.predict(P[h]) == lab[h]).mean())
    return float(np.mean(acc)) if acc else np.nan


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subsets", type=int, default=N_SUBSETS)
    ap.add_argument("--perm-cap", type=int, default=PERM_CAP,
                    help="enumerate the subset's null exactly when C(n,k)-1 <= this")
    ap.add_argument("--force", action="store_true", help="overwrite an existing curve")
    a = ap.parse_args()
    N_SUBSETS, PERM_CAP = a.n_subsets, a.perm_cap
    path = OUT / "power_curve_xjtu.csv"
    if path.exists() and not a.force:
        print(f"{path.name} already built: {len(pd.read_csv(path))} rows (--force to redo)")
        raise SystemExit
    race, truth, chans = load()
    print(f"{len(race)} race units: "
          f"{sum(v == 'OR' for v in truth.values())} OR, "
          f"{sum(v == 'IR' for v in truth.values())} IR", flush=True)
    rows, t0 = [], time.time()
    for n in range(4, len(race) + 1):
        # a subset needs >=2 of each class or leave-one-out leaves one class behind
        valid = [s for s in itertools.combinations(race, n)
                 if sum(truth[b] == "IR" for b in s) >= 2
                 and sum(truth[b] == "OR" for b in s) >= 2]
        if len(valid) > N_SUBSETS:
            idx = RNG.choice(len(valid), N_SUBSETS, replace=False)
            valid = [valid[i] for i in idx]
        for sub in valid:
            sub = list(sub)
            k = sum(truth[c] == "IR" for c in sub)
            true_combo = tuple(c for c in sub if truth[c] == "IR")
            perms = [p for p in itertools.combinations(sub, k) if p != true_combo]
            exact = len(perms) <= PERM_CAP          # the whole null, so p is exact here
            if not exact:
                pidx = RNG.choice(len(perms), PERM_CAP, replace=False)
                perms = [perms[i] for i in pidx]
            for cname, P in chans.items():
                true_acc = score(P, sub, truth)
                null = [score(P, sub, {c: ("IR" if c in combo else "OR") for c in sub})
                        for combo in perms]
                null = [v for v in null if v == v]
                if not null:
                    continue
                rows.append(dict(n=n, channel=cname, subset="|".join(sub), true=true_acc,
                                 n_minority=k, null_mean=float(np.mean(null)),
                                 null_95=float(np.quantile(null, 0.95)),
                                 null_sd=float(np.std(null)),
                                 p=float((1 + np.sum(np.array(null) >= true_acc))
                                         / (1 + len(null))),
                                 n_perm=len(null), exact=exact))
        pd.DataFrame(rows).to_csv(path, index=False)   # partial write: survive a kill
        print(f"  n={n}: {len(valid)} valid subsets ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path.name}: {len(rows)} rows in {(time.time() - t0) / 60:.1f} min")
