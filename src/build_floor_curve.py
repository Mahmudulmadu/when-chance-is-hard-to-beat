# -*- coding: utf-8 -*-
"""Per-unit safety of the RUL floor against the number of training units (notebook 26).

Notebooks 22-25 tested how far this project's attribution numbers hold: an
exact permutation null (nb22), a unit-count power curve on each rig (nb23, nb24),
and a reliability layer whose interval came out as [0 %, 100 %] (nb25). The one
positive remaining-life result, RUL reported as a floor, has not been through
any of these.

Two standing open items ask for the same measurement. Notebook 16's is a RUL
learning curve over units. Notebook 17's is more specific, because it corrected
its own main result: one global q0.10 floor is 91 % safe on the median unit, but
only 6 of 11 units individually clear 90 % safety, and Bearing2_4, the shortest
life in the set, is 18 % safe. A floor applies to each unit separately, so
whether "at least n minutes" can be said to an engineer depends on how that
fraction of units moves with the size of the training fleet.

This measures it. For each training-set size n, draw subsets of the 15 bearings,
fit notebook 16's forest on those units only, and score every held-out unit on
the pair notebook 17 used:

  * safety: the share of that unit's post-alarm minutes where the floor
    does not overstate the life left (q <= y), and
  * give-up: the median (y - q) / y, how much of the remaining life the
    floor gives up for that safety.

Both controls from the standing rule run alongside, because safety alone is
trivial to reach: a floor of zero minutes is 100 % safe and gives up everything,
so the forest has to beat the controls on the pair:

  * prior floor: the same quantile of the training units' post-alarm RUL
    distribution, ignoring the features entirely. This is the physically
    incapable control: it cannot know anything about the unit in front of it.
  * permuted: the same forest fitted after shuffling RUL across all
    training rows, destroying the feature-to-lifetime map. The null control.

n=14 is exhaustive (15 subsets, one per omitted unit), so the right-hand end of
the curve reproduces notebook 17's leave-one-bearing-out numbers exactly. This is
the build's own correctness check.

Usage:  python src/build_floor_curve.py
Writes data/processed/floor_curve.csv
"""
import itertools
import os
import pathlib
import time
import warnings

# sklearn's joblib workers re-emit a UserWarning on every parallel fit; the filter
# has to reach the worker processes, so it goes in the environment as well
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"

FAULT = {"Bearing1_1": "OR", "Bearing1_2": "OR", "Bearing1_3": "OR",
         "Bearing1_4": "cage", "Bearing1_5": "IR+OR", "Bearing2_1": "IR",
         "Bearing2_2": "OR", "Bearing2_3": "cage", "Bearing2_4": "OR",
         "Bearing2_5": "OR", "Bearing3_1": "OR", "Bearing3_2": "compound",
         "Bearing3_3": "IR", "Bearing3_4": "IR", "Bearing3_5": "OR"}
ENV = ["H_BPFO", "H_BPFI", "H_FR", "H_BPFO_h2", "H_BPFI_h2", "H_BSF", "H_FTF",
       "H_BPFI_sb1", "H_BPFO_sb1"]
DETECT = ["H_RMS", "H_Kurt"]
K, W = 20, 10
# notebook 16's forest, unchanged: 300 trees is what the quantile is cut from, and
# a 60-tree check moved held-out safety by 0.000-0.010, so the time saved does not justify
# a smaller ensemble here (176 fits at ~6 s is ~18 min)
RFR = dict(n_estimators=300, random_state=0, n_jobs=-1, min_samples_leaf=2)
QS = np.array([0.02, 0.06, 0.10, 0.14, 0.20, 0.30, 0.40, 0.50])
N_SUBSETS = 16                       # >= C(15,14) = 15, so n=14 is exhaustive
RNG = np.random.default_rng(0)


class Exceedance:
    '''Robust per-feature z over the unit's own healthy baseline (notebook 07).'''

    def fit(self, A):
        self.med = np.median(A, axis=0)
        self.mad = np.maximum(1.4826 * np.median(np.abs(A - self.med), axis=0), 0.1)
        return self

    def score(self, A):
        return ((A - self.med) / self.mad).max(axis=1)


def load():
    ds = pd.read_csv(OUT / "xjtu_envelope_features.csv")
    bear = sorted(ds["Bearing"].unique())
    G = {b: ds[ds["Bearing"] == b].sort_values("Snapshot").reset_index(drop=True)
         for b in bear}
    N = {b: len(G[b]) for b in bear}
    alarm, anom = {}, {}
    for b in bear:                                   # notebook 14's stage-1 alarm
        A = G[b][DETECT].to_numpy()
        z = Exceedance().fit(A[:K]).score(A)
        anom[b] = z
        c = np.convolve((z[K:] > np.quantile(z[:K], 0.99)).astype(float),
                        np.ones(W) / W, "valid")
        alarm[b] = int(np.flatnonzero(c >= 0.5)[0]) + K

    def frame(b):
        '''Notebook 16's per-minute state vector.'''
        n, t = N[b], np.arange(N[b])
        A, R, z = G[b][ENV].to_numpy(), G[b][DETECT].to_numpy(), anom[b]
        slope = np.zeros(n)
        for i in range(n):
            lo = max(0, i - 9)
            if i - lo >= 2:
                slope[i] = np.polyfit(np.arange(lo, i + 1), z[lo:i + 1], 1)[0]
        f = pd.DataFrame(A - np.median(A[:K], axis=0), columns=[c + "_d" for c in ENV])
        dr = R - np.median(R[:K], axis=0)
        f["RMS_d"], f["Kurt_d"] = dr[:, 0], dr[:, 1]
        f["anom"], f["anom_slope"] = z, slope
        f["elapsed"] = t
        f["since_alarm"] = np.maximum(t - alarm[b], 0)
        f["alarmed"] = (t >= alarm[b]).astype(int)
        f["fr_Hz"] = G[b]["fr_Hz"].to_numpy()
        f["RUL"] = (n - 1) - t
        f["Bearing"] = b
        return f

    DF = pd.concat([frame(b) for b in bear], ignore_index=True)
    feats = [c for c in DF.columns if c not in ("RUL", "Bearing")]
    return bear, DF, feats, alarm


def pair(floor, y, post):
    '''Notebook 17's safety / give-up pair over a unit's post-alarm minutes.'''
    f, yy = floor[post], y[post]
    k = yy > 0
    return (float((f <= yy).mean()),
            float(np.median((yy[k] - f[k]) / yy[k])) if k.any() else np.nan)


if __name__ == "__main__":
    path = OUT / "floor_curve.csv"
    if path.exists():
        print(f"{path.name} already built: {len(pd.read_csv(path))} rows")
        raise SystemExit
    BEAR, DF, FEATS, ALARM = load()
    Y = {b: DF.loc[DF["Bearing"] == b, "RUL"].to_numpy() for b in BEAR}
    POST = {b: DF.loc[DF["Bearing"] == b, "elapsed"].to_numpy() >= ALARM[b] for b in BEAR}
    X = {b: DF.loc[DF["Bearing"] == b, FEATS].to_numpy() for b in BEAR}
    print(f"{len(BEAR)} units, {len(DF)} snapshots, {len(FEATS)} features", flush=True)

    rows, t0 = [], time.time()
    for n in range(4, len(BEAR) + 1):
        subsets = list(itertools.combinations(BEAR, n))
        if len(subsets) > N_SUBSETS:
            idx = RNG.choice(len(subsets), N_SUBSETS, replace=False)
            subsets = [subsets[i] for i in idx]
        for sid, sub in enumerate(subsets):
            sub = list(sub)
            held = [b for b in BEAR if b not in sub]
            tr = DF["Bearing"].isin(sub)
            Xtr, ytr = DF.loc[tr, FEATS], DF.loc[tr, "RUL"].to_numpy()
            # the prior floor: quantiles of the training fleet's post-alarm RUL,
            # which is what a floor can promise knowing nothing about this unit
            pool = np.concatenate([Y[b][POST[b]] for b in sub])
            prior = {q: float(np.quantile(pool, q)) for q in QS}

            models = {"forest": RandomForestRegressor(**RFR).fit(Xtr, ytr),
                      "permuted": RandomForestRegressor(**RFR).fit(
                          Xtr, RNG.permutation(ytr))}
            for h in held:
                base = dict(n=n, subset=sid, n_rows=int(tr.sum()), unit=h,
                            fault=FAULT[h], life=len(Y[h]), post_n=int(POST[h].sum()))
                for name, m in models.items():
                    trees = np.stack([t.predict(X[h]) for t in m.estimators_])
                    for q in QS:
                        s, g = pair(np.quantile(trees, q, axis=0), Y[h], POST[h])
                        rows.append(dict(base, scheme=name, q=float(q),
                                         safety=s, giveup=g))
                for q in QS:
                    s, g = pair(np.full(len(Y[h]), prior[q]), Y[h], POST[h])
                    rows.append(dict(base, scheme="prior", q=float(q),
                                     safety=s, giveup=g))
        print(f"  n={n}: {len(subsets)} subsets ({time.time() - t0:.0f}s)", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(path, index=False)
    print(f"wrote {path.name}: {len(d)} rows in {(time.time() - t0) / 60:.1f} min")
