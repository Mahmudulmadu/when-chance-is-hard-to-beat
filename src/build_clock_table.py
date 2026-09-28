# -*- coding: utf-8 -*-
"""What does an RUL number beat? -- notebook 27.

Notebook 16 found that the two-channel state is worse than the clock it contains
over a whole trajectory (132 minutes against 120) and better only after the
alarm, and it stated -- in prose -- that predicting remaining life on
run-to-failure data is the same act as predicting total lifetime. Both claims
are load-bearing for the paper and neither is currently a table.

This builds the table. Every estimator is scored against the same held-out unit
under one leave-one-bearing-out pass, over five scopes, with the identity
checked numerically rather than asserted.

Estimators, in increasing order of what they are allowed to know:

  * **zero**       -- RUL = 0. Trivially safe, gives up everything. The floor of
                      the scale: any model that cannot beat it is not a model.
  * **survived**   -- RUL = minutes already run, i.e. "it will last twice as long
                      as it has". Uses the clock and nothing else, and exists
                      because notebook 16 observed that both its estimators were
                      "flattered by a floor they get for free" without ever
                      scoring that floor on its own.
  * **med_all**    -- median lifetime of the training units, minus minutes run.
  * **med_cond**   -- the same, restricted to units at this operating point.
                      An engineer knows the load and the speed, so this is the
                      strongest baseline that reads no vibration at all.
  * **clock**      -- a forest given only (elapsed, fr_Hz). Notebook 16's
                      learned clock: the same two facts, fitted rather than
                      assumed.
  * **state**      -- notebook 16's forest over all 17 features.
  * **clock_post / state_post** -- both refitted on post-alarm rows only, which
                      is how stage 2 would actually be deployed.

Scopes: whole trajectory, post-alarm, the single snapshot at the alarm, and each
fifth of life. Notebook 16 reported fifths for the state alone; the gap to a
clock is what makes a fifth interpretable, so every estimator is binned here.

The identity: RUL = (L-1) - t with t known exactly, so any RUL estimate implies a
lifetime estimate L_hat = RUL_hat + t + 1, and the two errors are the same
number. The build records both and their difference, so the claim is checked at
every one of the 9216 snapshots rather than argued.

Usage:  python src/build_clock_table.py
Writes data/processed/clock_table.csv   (per snapshot, per estimator)
       data/processed/clock_identity.csv (the identity check, per unit)
"""
import os
import pathlib
import time
import warnings

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
RF = dict(n_estimators=300, random_state=0, n_jobs=-1, min_samples_leaf=2)
CLOCK = ["elapsed", "fr_Hz"]                 # notebook 16's learned clock


class Exceedance:
    '''Robust per-feature z over the unit's own healthy baseline (notebook 07).'''

    def fit(self, A):
        self.med = np.median(A, axis=0)
        self.mad = np.maximum(1.4826 * np.median(np.abs(A - self.med), axis=0), 0.1)
        return self

    def score(self, A):
        return ((A - self.med) / self.mad).max(axis=1)


def load():
    '''Notebook 16's frame, unchanged -- this build must not move its numbers.'''
    ds = pd.read_csv(OUT / "xjtu_envelope_features.csv")
    bear = sorted(ds["Bearing"].unique())
    G = {b: ds[ds["Bearing"] == b].sort_values("Snapshot").reset_index(drop=True)
         for b in bear}
    N = {b: len(G[b]) for b in bear}
    cond = {b: G[b]["Cond"].iloc[0] for b in bear}
    alarm, anom = {}, {}
    for b in bear:
        A = G[b][DETECT].to_numpy()
        z = Exceedance().fit(A[:K]).score(A)
        anom[b] = z
        c = np.convolve((z[K:] > np.quantile(z[:K], 0.99)).astype(float),
                        np.ones(W) / W, "valid")
        alarm[b] = int(np.flatnonzero(c >= 0.5)[0]) + K

    def frame(b):
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
    return bear, DF, feats, alarm, N, cond


if __name__ == "__main__":
    path, ipath = OUT / "clock_table.csv", OUT / "clock_identity.csv"
    if path.exists():
        print(f"{path.name} already built: {len(pd.read_csv(path))} rows")
        raise SystemExit
    BEAR, DF, FEATS, ALARM, N, COND = load()
    print(f"{len(BEAR)} units, {len(DF)} snapshots, {len(FEATS)} features", flush=True)

    rows, ident, t0 = [], [], time.time()
    for held in BEAR:
        tr = DF["Bearing"].isin([o for o in BEAR if o != held])
        hm = DF["Bearing"] == held
        trp = tr & (DF["alarmed"] == 1)                  # post-alarm training rows
        hd = DF.loc[hm]
        t = hd["elapsed"].to_numpy().astype(float)
        y = hd["RUL"].to_numpy().astype(float)
        L = float(N[held])
        same = [N[o] for o in BEAR if o != held and COND[o] == COND[held]]

        P = {
            "zero": np.zeros_like(t),
            "survived": t.copy(),
            "med_all": np.maximum(np.median([N[o] for o in BEAR if o != held]) - 1 - t, 0),
            "med_cond": np.maximum(np.median(same) - 1 - t, 0),
            "clock": RandomForestRegressor(**RF).fit(
                DF.loc[tr, CLOCK], DF.loc[tr, "RUL"]).predict(hd[CLOCK]),
            "state": RandomForestRegressor(**RF).fit(
                DF.loc[tr, FEATS], DF.loc[tr, "RUL"]).predict(hd[FEATS]),
            "clock_post": RandomForestRegressor(**RF).fit(
                DF.loc[trp, CLOCK], DF.loc[trp, "RUL"]).predict(hd[CLOCK]),
            "state_post": RandomForestRegressor(**RF).fit(
                DF.loc[trp, FEATS], DF.loc[trp, "RUL"]).predict(hd[FEATS]),
        }

        for name, p in P.items():
            # the identity: an RUL estimate is a lifetime estimate, and the two
            # errors are the same number at every snapshot
            e_rul = np.abs(p - y)
            e_life = np.abs((p + t + 1) - L)
            ident.append(dict(unit=held, estimator=name,
                              max_abs_diff=float(np.max(np.abs(e_rul - e_life))),
                              mae_rul=float(e_rul.mean()), mae_life=float(e_life.mean())))
            rows.append(pd.DataFrame(dict(
                unit=held, fault=FAULT[held], cond=COND[held], life=L,
                estimator=name, t=t, pos=t / (L - 1), y=y, pred=p,
                post=(t >= ALARM[held]), at_alarm=(t == ALARM[held]),
                err=p - y, abs_err=e_rul, abs_err_life=e_life)))
        print(f"  {held}: {len(P)} estimators ({time.time() - t0:.0f}s)", flush=True)

    d = pd.concat(rows, ignore_index=True)
    d.to_csv(path, index=False)
    pd.DataFrame(ident).to_csv(ipath, index=False)
    print(f"wrote {path.name}: {len(d)} rows in {(time.time() - t0) / 60:.1f} min")
    print(f"wrote {ipath.name}: identity max|diff| = "
          f"{pd.DataFrame(ident)['max_abs_diff'].max():.10f}")
