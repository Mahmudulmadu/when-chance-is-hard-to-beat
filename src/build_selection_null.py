# -*- coding: utf-8 -*-
"""Selection-aware exact null for the Paderborn geometry result (paper section 4.2).

The exact null in `build_permutation_nulls.py` holds the Paderborn configuration
fixed: the 400 N condition (N15_M07_F04) and the carriers + sidebands feature set.
Both were chosen in notebook 11, which compared twelve configurations (six feature
sets at each of two radial forces) on the same eleven real-damage bearings the
null is computed on, using their race labels. A null that holds the winner fixed
does not account for that choice.

This one repeats the choice under every labelling. For each of the 462
class-preserving labellings of the eleven real units it scores all twelve
configurations exactly as notebook 11 did (a 300-tree forest trained on the eight
artificial-damage bearings, scored on the eleven real ones under that labelling),
keeps the best, and then scores the paper's bearing-held-out protocol with the
configuration it kept. Under the true labels the choice reproduces notebook 11's
winner (87.3 % artificial -> real), so the true labelling keeps the cached 85.5 %;
every other labelling gets the configuration that would have been picked had its
labels been the true ones.

It corrects only the last step of the exploration: notebooks 09 and 10 had already
compared loads before notebook 11, and nothing here repeats those steps.

Usage:  python src/build_selection_null.py      (about 20 minutes on 8 cores)
Writes data/processed/selection_null_pu.csv
"""
import itertools
import pathlib
import time
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
from sklearn.ensemble import RandomForestClassifier   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"
HI, LO = "N15_M07_F10", "N15_M07_F04"                # 1000 N and 400 N radial force
SETS = {                                             # notebook 11's six feature sets
    "carriers (BPFO, BPFI, FR)": ["BPFO", "BPFI", "FR"],
    "contrast alone": ["BPFI_minus_BPFO"],
    "carriers + harmonics": ["BPFO", "BPFI", "FR", "HARM_BPFO", "HARM_BPFI"],
    "carriers + sidebands": ["BPFO", "BPFI", "FR", "BPFI_sb1", "BPFO_sb1", "BPFI_sb2", "BPFI_h2_sb1"],
    "sideband ratios only": ["SBR_BPFI", "SBR_BPFO", "SBR_diff"],
    "everything": ["BPFO", "BPFI", "FR", "BPFO_h2", "BPFI_h2", "BPFI_sb1", "BPFO_sb1",
                   "BPFI_sb2", "BPFI_h2_sb1", "SBR_BPFI", "SBR_BPFO", "SBR_diff",
                   "BPFI_minus_BPFO", "HARM_BPFO", "HARM_BPFI"],
}
CONFIGS = [(cond, name) for cond in (HI, LO) for name in SETS]
CHOSEN = (LO, "carriers + sidebands")               # what notebook 11 picked
RF_SELECT = dict(n_estimators=300, random_state=0, n_jobs=-1)   # notebook 11's forest
RF_NULL = dict(n_estimators=100, random_state=0, n_jobs=-1)     # build_permutation_nulls.py's


def label(cfg):
    return f"{cfg[1]} @{'1000 N' if cfg[0] == HI else '400 N'}"


def held_out(P, units, lab):
    """The paper's protocol: bearing-held-out, mean over units of per-unit accuracy."""
    acc = []
    for h in units:
        tr = [o for o in units if o != h]
        y = np.concatenate([[lab[o]] * len(P[o]) for o in tr])
        m = RandomForestClassifier(**RF_NULL).fit(np.vstack([P[o] for o in tr]), y)
        acc.append(float((m.predict(P[h]) == lab[h]).mean()))
    return float(np.mean(acc))


def main():
    t0 = time.time()
    sb = {c: pd.read_csv(OUT / f"pu_{c}_sidebands.csv") for c in (HI, LO)}
    lz = pd.read_csv(OUT / f"pu_{LO}_loadzone_features.csv")
    key = ["Code", "Condition", "Domain", "Run"]
    real = {HI: sb[HI][sb[HI]["Domain"] == "real"],
            LO: sb[LO].merge(lz, on=key, suffixes=("", "_l")).query("Domain == 'real'")}
    units = sorted(real[LO]["Code"].unique())
    truth = {u: real[LO][real[LO]["Code"] == u]["Condition"].iloc[0] for u in units}
    assert len(units) == 11

    # Notebook 11's selection score: train on the artificial bearings, predict the real ones.
    pred = {}
    for cond, name in CONFIGS:
        d = sb[cond]
        art = d[d["Domain"] == "artificial"]
        m = RandomForestClassifier(**RF_SELECT).fit(art[SETS[name]], art["Condition"])
        pred[(cond, name)] = {u: m.predict(d[d["Code"] == u][SETS[name]]) for u in units}

    def selection_score(cfg, lab):
        return float(np.mean([(pred[cfg][u] == lab[u]).mean() for u in units]))

    print("selection score under the true labels (notebook 11):")
    for cfg in CONFIGS:
        print(f"  {label(cfg):34s} {selection_score(cfg, truth):.3f}")
    assert max(CONFIGS, key=lambda c: selection_score(c, truth)) == CHOSEN

    nulls = pd.read_csv(OUT / "permutation_nulls.csv")
    fixed = nulls[(nulls.rig == "Paderborn") & (nulls.channel == "geometry")].reset_index(drop=True)
    combos = list(itertools.combinations(units, 5))          # the five outer-race positions
    assert len(fixed) == len(combos) == 462
    labs = [{u: ("OR" if u in cb else "IR") for u in units} for cb in combos]
    true_idx = [i for i, lab in enumerate(labs) if lab == truth]
    assert len(true_idx) == 1 and bool(fixed.loc[true_idx[0], "is_true"])

    features = lambda cfg: {u: real[cfg[0]][real[cfg[0]]["Code"] == u][SETS[cfg[1]]].to_numpy()
                            for u in units}
    P_chosen = features(CHOSEN)
    for i in (0, 1):                                         # harness: reproduce two cached rows
        a = held_out(P_chosen, units, labs[i])
        assert abs(a - fixed.loc[i, "acc"]) < 1e-12, (i, a, fixed.loc[i, "acc"])

    picks, accs, cache = [], [], {}
    for i, lab in enumerate(labs):
        best = max(CONFIGS, key=lambda c: selection_score(c, lab))   # first maximum, as nb11
        picks.append(label(best))
        if best == CHOSEN:
            accs.append(float(fixed.loc[i, "acc"]))
        else:
            if best not in cache:
                cache[best] = features(best)
            accs.append(held_out(cache[best], units, lab))
        if i % 50 == 0:
            print(f"  {i}/462  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    out = pd.DataFrame(dict(combo=["|".join(c) for c in combos], pick=picks, acc_selection=accs,
                            acc_fixed=fixed["acc"], is_true=fixed["is_true"]))
    out.to_csv(OUT / "selection_null_pu.csv", index=False)
    t = float(out.loc[out.is_true, "acc_selection"].iloc[0])
    print(f"\ntrue labelling {t:.4f}; p fixed {(out.acc_fixed >= t).mean():.4f} "
          f"({int((out.acc_fixed >= t).sum())}/462), p selection-aware {(out.acc_selection >= t).mean():.4f} "
          f"({int((out.acc_selection >= t).sum())}/462)")
    print(f"wrote selection_null_pu.csv in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
