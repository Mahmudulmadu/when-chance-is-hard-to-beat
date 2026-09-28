# -*- coding: utf-8 -*-
"""The exact null for the XJTU-SY non-geometry channel -- notebooks 18/24.

Table 1's non-geometry row was priced by the study-size curve (notebook 24),
whose n=11 point draws **20** of the 164 non-true labellings. The other rows on
that rig are exact over all 165. A paper whose abstract says its p-values are
exact rather than sampled cannot leave that row sampled, so this enumerates it.

Everything except the number of labellings is held at the curve's own settings,
because the true accuracy (62.3 %) is the curve's and must not move:

  * the curve's `load()` -- alarm-anchored windows, per-unit median subtraction;
  * the curve's 15-tree forest, n_jobs=1, random_state=0;
  * the curve's leave-one-bearing-out scoring.

All three channels are enumerated, not only the negative one: geometry and
load-zone are already exact under notebook 22's 100-tree forest, so scoring them
again here says whether the verdicts survive the change of forest.

p is the exact form of section 4.2, #{a(L) >= a(true)} / 165, which for a
complete enumeration equals the curve's (1 + #{null >= true}) / (1 + N).

Usage:  python src/build_nongeometry_null.py
Writes data/processed/nongeometry_null_xjtu.csv
"""
import itertools
import pathlib
import time

import numpy as np
import pandas as pd

import build_power_curve_xjtu as PC24

OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed"

if __name__ == "__main__":
    path = OUT / "nongeometry_null_xjtu.csv"
    race, truth, chans = PC24.load()
    minority = "IR"
    combos = list(itertools.combinations([b for b in race], sum(v == minority for v in truth.values())))
    true_combo = tuple(b for b in race if truth[b] == minority)
    assert true_combo in combos and len(combos) == 165, (len(combos), true_combo)
    rows, t0 = [], time.time()
    for cname, P in chans.items():
        for combo in combos:
            lab = {b: (minority if b in combo else "OR") for b in race}
            rows.append(dict(rig="XJTU-SY", channel=cname, combo="|".join(combo),
                             is_true=(combo == true_combo), acc=PC24.score(P, race, lab)))
        d = pd.DataFrame(rows)
        d = d[d.channel == cname]
        t = float(d.loc[d.is_true, "acc"].iloc[0])
        print(f"  {cname}: true {t:.4f}  null 95th {d.loc[~d.is_true, 'acc'].quantile(0.95):.4f}  "
              f"p = {float((d.acc >= t).mean()):.4f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path.name}: {len(rows)} rows")
