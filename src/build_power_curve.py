# -*- coding: utf-8 -*-
"""How many units does an attribution claim need? -- notebook 23.

Three notebooks have reached the same constraint from different directions:
attribution priced at "about six real units" (nb14), only 6 of 11 units clearing
the RUL floor's safety target (nb17), and a label-permutation null whose 95th
percentile reaches 69-74 % on eleven units (nb22). None of them answers the
question numerically: **at what number of units does a real effect become
distinguishable from chance?**

This measures it. For each unit count n, draw subsets of n bearings, and within
each subset compute both

  * the accuracy under the TRUE labels, and
  * the null band from permuting labels inside that same subset,

using the identical bearing-held-out protocol. Where the true curve separates
from the null band is the answer, and the width of the band at each n is the
smallest effect that study size could ever resolve.

Paderborn is used because its 20 runs per bearing keep 50k random-forest fits
tractable, and because notebook 22 established that its geometry channel carries
a genuinely real effect (85.5 %, p = 0.0043) to serve as the signal being
detected, alongside its load-zone channel which does not (p = 0.147).

Usage:  python src/build_power_curve.py
Writes data/processed/power_curve.csv

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

# joblib workers re-emit a UserWarning on every fit; the filter has to reach the
# worker processes, so it goes in the environment as well as the interpreter
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"
COND = "N15_M07_F04"
GEO = ["BPFO", "BPFI", "FR", "BPFI_sb1", "BPFO_sb1", "BPFI_sb2", "BPFI_h2_sb1"]
# n_jobs=1 deliberately: these fits are ~200 rows, where thread setup costs more
# than the parallelism saves (measured: 179 ms with n_jobs=-1 against 120 with 1).
# 15 trees rather than the protocol's 300 keeps ~25k fits to about 25 minutes, and
# reproduces notebook 22 at n=11 to within two points (geometry 85.0 % against
# 85.5 %, load-zone 60.5 % against 62.7 %) -- the curve measures SEPARATION between
# the true labels and their null, and both sides use the same forest.
RF = dict(n_estimators=15, random_state=0, n_jobs=1)
# Referee M3: 12 subsets per size cannot support a point estimate, and N_PERM=20 puts
# the per-subset p-floor at 1/21 = 0.048, so "detected" meant beating every draw. The
# defaults are now 30 subsets and a null enumerated exactly wherever C(n,k)-1 <= 40.
N_SUBSETS, PERM_CAP = 30, 40
RNG = np.random.default_rng(0)


def load():
    sb = pd.read_csv(OUT / f"pu_{COND}_sidebands.csv")
    lz = pd.read_csv(OUT / f"pu_{COND}_loadzone_features.csv")
    key = ["Code", "Condition", "Domain", "Run"]
    d = sb.merge(lz, on=key, suffixes=("", "_l")).query("Domain == 'real'")
    lzf = [c for c in lz.columns if c.startswith("LZ_") and c != "LZ_frdev"]
    codes = sorted(d["Code"].unique())
    truth = {c: d[d["Code"] == c]["Condition"].iloc[0] for c in codes}
    mats = {"geometry": {c: d[d["Code"] == c][GEO].to_numpy() for c in codes},
            "load-zone": {c: d[d["Code"] == c][lzf].to_numpy() for c in codes}}
    return codes, truth, mats


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
    path = OUT / "power_curve.csv"
    if path.exists() and not a.force:
        print(f"{path.name} already built: {len(pd.read_csv(path))} rows (--force to redo)")
        raise SystemExit
    codes, truth, mats = load()
    classes = sorted(set(truth.values()))
    rows = []
    t0 = time.time()
    for n in range(4, len(codes) + 1):
        subsets = list(itertools.combinations(codes, n))
        if len(subsets) > N_SUBSETS:
            idx = RNG.choice(len(subsets), N_SUBSETS, replace=False)
            subsets = [subsets[i] for i in idx]
        for sub in subsets:
            sub = list(sub)
            if len({truth[c] for c in sub}) < 2:
                continue
            k = sum(truth[c] == classes[0] for c in sub)
            true_combo = tuple(c for c in sub if truth[c] == classes[0])
            perms = [p for p in itertools.combinations(sub, k) if p != true_combo]
            exact = len(perms) <= PERM_CAP          # the whole null, so p is exact here
            if not exact:
                pidx = RNG.choice(len(perms), PERM_CAP, replace=False)
                perms = [perms[i] for i in pidx]
            for cname, P in mats.items():
                true_acc = score(P, sub, truth)
                null = [score(P, sub, {c: (classes[0] if c in combo else classes[1])
                                       for c in sub}) for combo in perms]
                null = [v for v in null if v == v]
                rows.append(dict(n=n, channel=cname, subset="|".join(sub), true=true_acc,
                                 n_minority=k, null_mean=float(np.mean(null)),
                                 null_95=float(np.quantile(null, 0.95)),
                                 null_sd=float(np.std(null)),
                                 p=float((1 + np.sum(np.array(null) >= true_acc))
                                         / (1 + len(null))),
                                 n_perm=len(null), exact=exact))
        pd.DataFrame(rows).to_csv(path, index=False)   # partial write: survive a kill
        print(f"  n={n}: {len(subsets)} subsets done ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path.name}: {len(rows)} rows in {(time.time()-t0)/60:.1f} min")
