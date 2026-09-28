# -*- coding: utf-8 -*-
"""The deliberately incapable control for the load-zone channel -- notebook 21.

Notebook 20's `LZ_frdev` scored 81 % on nothing at all, and the check that caught
it was asking whether a feature could physically carry what it appeared to carry.
The project adopted that as a standing control: whenever a feature family is
added, run one that *cannot* work and see what it gets.

This is that control for the load-zone fold. It is identical to
`build_loadzone_paderborn.py` in every respect except the fold rate: it folds the
envelope at **1.37 x shaft rate**, which is not synchronous with the shaft and is
not a bearing order, so it cannot carry load-zone modulation. Whatever it scores
is a measurement of something other than the physics the channel claims.

Result (notebook 21): 53.6 % artificial->real and 44.1 % among real damaged
bearings, against the true fold's 74.5 % and 65.5 % -- the control fails as it
should, so the load-zone channel is measuring genuine shaft-locked content.

Usage:  python src/build_decoy_fold.py
Writes data/processed/pu_<cond>_decoy_fold.csv
"""
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, hilbert

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "paderborn"
OUT = ROOT / "data" / "processed"
COND = "N15_M07_F04"
FS, WIN, N_KEEP, NBIN = 64_000, 65_536, 245_760, 64
SOS = butter(4, (1000, 20000), btype="bandpass", fs=FS, output="sos")
DECOY = 1.37                               # not 1.0, and not a bearing order

BEARINGS = ([(c, "OR", "artificial") for c in ("KA01", "KA03", "KA05", "KA07")]
            + [(c, "IR", "artificial") for c in ("KI01", "KI03", "KI05", "KI07")]
            + [(c, "OR", "real") for c in ("KA04", "KA15", "KA16", "KA22", "KA30")]
            + [(c, "IR", "real") for c in ("KI04", "KI14", "KI16", "KI17", "KI18", "KI21")])


def decoy_features(x, fold_rate):
    """Same statistics as the load-zone fold, computed at a non-shaft rate."""
    e = np.abs(hilbert(sosfiltfilt(SOS, x)))
    e = e - e.mean()
    rms = np.sqrt(np.mean(e ** 2)) + 1e-30
    t = np.arange(len(e)) / FS
    best = (-1.0, None)
    for f in np.arange(fold_rate * 0.97, fold_rate * 1.03, 0.02):
        idx = ((t * f) % 1.0 * NBIN).astype(int)
        p = np.zeros(NBIN)
        c = np.zeros(NBIN)
        np.add.at(p, idx, e)
        np.add.at(c, idx, 1)
        p /= np.maximum(c, 1)
        s = float(np.mean(p ** 2) / rms ** 2)
        if s > best[0]:
            best = (s, p)
    sync, prof = best
    P = np.abs(np.fft.rfft(prof)) ** 2
    tot = P[1:].sum() + 1e-30
    pk = prof - prof.mean()
    return {"DC_sync": sync,
            "DC_depth": float((prof.max() - prof.min()) / rms),
            "DC_h1": float(P[1] / tot), "DC_h2": float(P[2] / tot),
            "DC_h3": float(P[3] / tot),
            "DC_kurt": float(np.mean(pk ** 4) / (np.mean(pk ** 2) ** 2 + 1e-30) - 3.0),
            "DC_conc": float(np.max(np.abs(pk)) / (np.mean(np.abs(pk)) + 1e-30))}


def one_bearing(args):
    code, cls, dom = args
    rows = []
    for run in range(1, 21):
        p = RAW / code / f"{COND}_{code}_{run}.mat"
        if not p.exists():
            continue
        ch = loadmat(p)[p.stem]["Y"][0, 0][0]
        c = {str(x["Name"][0]): x["Data"].squeeze() for x in ch}
        v = c["vibration_1"][:N_KEEP].astype(np.float64)
        fr = float(np.atleast_1d(c["speed"]).mean()) / 60.0
        acc = {}
        for s in range(N_KEEP // WIN):
            for k, val in decoy_features(v[s * WIN:(s + 1) * WIN], DECOY * fr).items():
                acc.setdefault(k, []).append(val)
        r = {"Code": code, "Condition": cls, "Domain": dom, "Run": run}
        r.update({k: float(np.mean(vals)) for k, vals in acc.items()})
        rows.append(r)
    return rows


if __name__ == "__main__":
    path = OUT / f"pu_{COND}_decoy_fold.csv"
    if path.exists():
        print(f"{path.name} already built: {len(pd.read_csv(path))} runs")
    else:
        t0 = time.time()
        rows = []
        with ProcessPoolExecutor(max_workers=6) as pool:
            for i, out in enumerate(pool.map(one_bearing, BEARINGS), 1):
                rows.extend(out)
                print(f"  [{i}/{len(BEARINGS)}] {out[0]['Code']} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        d = pd.DataFrame(rows)
        d.to_csv(path, index=False)
        print(f"wrote {path.name}: {len(d)} runs in {(time.time() - t0) / 60:.1f} min")
