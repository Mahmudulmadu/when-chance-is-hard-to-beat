# -*- coding: utf-8 -*-
"""Exact bearing-level label-permutation nulls for stage-2 attribution (notebook 22).

Every attribution number in this project is a bearing-held-out accuracy over
about eleven units. Notebook 20 showed that a feature which cannot possibly work
(`LZ_frdev`, a speed reading) scored 81 %, and notebook 21 showed that both
feature families fingerprint individual healthy bearings at ~90 % against 16.7 %
chance. Neither observation says how much of a class result is real.

This script does that. We reassign the OR/IR labels across the bearings in every
distinct way that preserves the class counts, re-run the same bearing-held-out
protocol on each, and compare the true labelling against that null. The number of
assignments is small enough to enumerate exactly:

    XJTU-SY   11 race units, 8 OR / 3 IR  ->  C(11,3) = 165
    Paderborn 11 real damaged, 5 OR / 6 IR ->  C(11,5) = 462

so the p-values are exact, with a floor of 1/165 = 0.0061 and 1/462 = 0.0022.

Usage:  python src/build_permutation_nulls.py
Writes data/processed/permutation_nulls.csv
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
RF = dict(n_estimators=100, random_state=0, n_jobs=-1)   # 100 keeps 627 fits tractable
K = 20

FAULT = {"Bearing1_1": "OR", "Bearing1_2": "OR", "Bearing1_3": "OR",
         "Bearing1_4": "cage", "Bearing1_5": "IR+OR", "Bearing2_1": "IR",
         "Bearing2_2": "OR", "Bearing2_3": "cage", "Bearing2_4": "OR",
         "Bearing2_5": "OR", "Bearing3_1": "OR", "Bearing3_2": "compound",
         "Bearing3_3": "IR", "Bearing3_4": "IR", "Bearing3_5": "OR"}
XJ_GEO = ["H_BPFO", "H_BPFI", "H_FR", "H_BPFO_h2", "H_BPFI_h2", "H_BSF", "H_FTF",
          "H_BPFI_sb1", "H_BPFO_sb1"]
PU_GEO = ["BPFO", "BPFI", "FR", "BPFI_sb1", "BPFO_sb1", "BPFI_sb2", "BPFI_h2_sb1"]
PU_COND = "N15_M07_F04"


def xjtu_matrices():
    env = pd.read_csv(OUT / "xjtu_envelope_features.csv")
    lz = pd.read_csv(OUT / "xjtu_loadzone_features.csv")
    d = env.merge(lz, on=["Bearing", "Snapshot"], how="inner")
    race = [b for b in sorted(d["Bearing"].unique()) if FAULT[b] in ("OR", "IR")]
    G = {b: d[d["Bearing"] == b].sort_values("Snapshot").reset_index(drop=True)
         for b in race}
    lzf = [c for c in lz.columns if c.startswith("LZ_") and c != "LZ_frdev"]

    def post(feats):
        return {b: (G[b][feats].to_numpy()
                    - np.median(G[b][feats].to_numpy()[:K], axis=0))[K:] for b in race}

    return race, {b: FAULT[b] for b in race}, {"geometry": post(XJ_GEO),
                                               "load-zone": post(lzf)}


def paderborn_matrices():
    sb = pd.read_csv(OUT / f"pu_{PU_COND}_sidebands.csv")
    lz = pd.read_csv(OUT / f"pu_{PU_COND}_loadzone_features.csv")
    key = ["Code", "Condition", "Domain", "Run"]
    d = sb.merge(lz, on=key, suffixes=("", "_l"))
    d = d[d["Domain"] == "real"]
    codes = sorted(d["Code"].unique())
    lzf = [c for c in lz.columns if c.startswith("LZ_") and c != "LZ_frdev"]
    truth = {c: d[d["Code"] == c]["Condition"].iloc[0] for c in codes}
    return codes, truth, {"geometry": {c: d[d["Code"] == c][PU_GEO].to_numpy()
                                       for c in codes},
                          "load-zone": {c: d[d["Code"] == c][lzf].to_numpy()
                                        for c in codes}}


def score(P, units, lab):
    """Bearing-held-out mean accuracy under an arbitrary label assignment."""
    acc = []
    for h in units:
        tr = [o for o in units if o != h]
        y = np.concatenate([[lab[o]] * len(P[o]) for o in tr])
        if len(set(y)) < 2:
            continue
        m = RandomForestClassifier(**RF).fit(np.vstack([P[o] for o in tr]), y)
        acc.append((m.predict(P[h]) == lab[h]).mean())
    return float(np.mean(acc))


if __name__ == "__main__":
    path = OUT / "permutation_nulls.csv"
    if path.exists():
        print(f"{path.name} already built: {len(pd.read_csv(path))} rows")
        raise SystemExit
    rows = []
    for rig, loader in (("XJTU-SY", xjtu_matrices), ("Paderborn", paderborn_matrices)):
        units, truth, chans = loader()
        minority = min(set(truth.values()), key=lambda c: sum(v == c for v in truth.values()))
        n_min = sum(v == minority for v in truth.values())
        combos = list(itertools.combinations(units, n_min))
        print(f"{rig}: {len(units)} units, minority class {minority} x{n_min} -> "
              f"{len(combos)} assignments", flush=True)
        for cname, P in chans.items():
            t0 = time.time()
            for combo in combos:
                lab = {u: (minority if u in combo else
                           next(c for c in set(truth.values()) if c != minority))
                       for u in units}
                is_true = all(lab[u] == truth[u] for u in units)
                rows.append(dict(rig=rig, channel=cname, acc=score(P, units, lab),
                                 is_true=is_true))
            print(f"  {cname}: {len(combos)} fits in {(time.time() - t0) / 60:.1f} min",
                  flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path.name}: {len(rows)} rows")
