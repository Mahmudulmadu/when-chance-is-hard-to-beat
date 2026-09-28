# -*- coding: utf-8 -*-
"""Shaft-synchronous (load-zone) envelope features for XJTU-SY -- notebook 20.

An inner-race defect rotates with the shaft and passes through the loaded arc
once per revolution, so its impulses are amplitude-modulated at 1x shaft rate and
phase-locked to the shaft; an outer-race defect, stationary relative to a fixed
load direction, is not. Folding the envelope onto one shaft revolution and
averaging over the ~45-51 revolutions in a snapshot reinforces what is
phase-locked and averages out what is not.

These features use the SHAFT RATE but no bearing geometry -- no BPFO, BPFI, BSF
or FTF, and nothing derived from pitch diameter, ball diameter or ball count.

CAUTION (notebook 20, finding 4): LZ_frdev is a pure speed reading and acts as a
proxy for the operating condition, which is confounded with class in this dataset
(every race unit at 35 Hz is outer race). Quarantine it, and check any new
feature set with leave-one-condition-out.

Only each unit's healthy baseline and its post-alarm snapshots are extracted,
which is about a fifth of the 9216 and keeps this to ~15 min on six cores.

Usage:  python src/build_loadzone_features.py
Writes data/processed/xjtu_loadzone_features.csv
"""

import pathlib
import time

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import butter, sosfiltfilt, hilbert
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "xjtu"
OUT = ROOT / "data" / "processed"
FS, WIN, K, NBIN = 25_600, 32_768, 20, 64
SOS = butter(4, (1000, 12000), btype="bandpass", fs=FS, output="sos")
COND = {"35Hz12kN": 35.0, "37.5Hz11kN": 37.5, "40Hz10kN": 40.0}
FAULT = {"Bearing1_1": "OR", "Bearing1_2": "OR", "Bearing1_3": "OR",
         "Bearing1_4": "cage", "Bearing1_5": "IR+OR", "Bearing2_1": "IR",
         "Bearing2_2": "OR", "Bearing2_3": "cage", "Bearing2_4": "OR",
         "Bearing2_5": "OR", "Bearing3_1": "OR", "Bearing3_2": "compound",
         "Bearing3_3": "IR", "Bearing3_4": "IR", "Bearing3_5": "OR"}


def lz_features(x, f0):
    """Shaft-synchronous envelope features, with fr found by search.

    The FFT bin spacing (0.78 Hz) is far too coarse to fold on -- at 40 Hz it
    drifts a full revolution over the record -- so fr is chosen as the value
    that maximises the synchronous energy, on a grid fine enough to hold drift
    under an eighth of a revolution.
    """
    e = np.abs(hilbert(sosfiltfilt(SOS, x)))
    e = e - e.mean()
    rms = np.sqrt(np.mean(e ** 2)) + 1e-30
    t = np.arange(len(e)) / FS
    best = (-1.0, f0, None)
    for fr in np.arange(f0 * 0.965, f0 * 1.035, 0.04):
        idx = ((t * fr) % 1.0 * NBIN).astype(int)
        p = np.zeros(NBIN)
        c = np.zeros(NBIN)
        np.add.at(p, idx, e)
        np.add.at(c, idx, 1)
        p /= np.maximum(c, 1)
        s = float(np.mean(p ** 2) / rms ** 2)
        if s > best[0]:
            best = (s, fr, p)
    sync, fr, prof = best
    P = np.abs(np.fft.rfft(prof)) ** 2
    tot = P[1:].sum() + 1e-30
    pk = prof - prof.mean()
    return {"LZ_sync": sync,
            "LZ_depth": float((prof.max() - prof.min()) / rms),
            "LZ_h1": float(P[1] / tot), "LZ_h2": float(P[2] / tot),
            "LZ_h3": float(P[3] / tot),
            "LZ_kurt": float(np.mean(pk ** 4) / (np.mean(pk ** 2) ** 2 + 1e-30) - 3.0),
            "LZ_conc": float(np.max(np.abs(pk)) / (np.mean(np.abs(pk)) + 1e-30)),
            "LZ_frdev": float(fr / f0 - 1.0)}


def one_bearing(args):
    cond, name, keep = args
    d = RAW / cond / name
    rows = []
    for s in keep:
        p = d / f"{s}.csv"
        if not p.exists():
            continue
        x = pd.read_csv(p).to_numpy(dtype=np.float64)
        if len(x) < WIN:
            x = np.pad(x, ((0, WIN - len(x)), (0, 0)))
        r = {"Bearing": name, "Snapshot": s}
        r.update(lz_features(x[:WIN, 0], COND[cond]))
        rows.append(r)
    return rows


if __name__ == "__main__":
    env = pd.read_csv(OUT / "xjtu_envelope_features.csv")
    BEAR = sorted(env["Bearing"].unique())
    G = {b: env[env["Bearing"] == b].sort_values("Snapshot").reset_index(drop=True)
         for b in BEAR}

    class Exceedance:
        def fit(self, A):
            self.med = np.median(A, axis=0)
            self.mad = np.maximum(1.4826 * np.median(np.abs(A - self.med), axis=0), 0.1)
            return self

        def score(self, A):
            return ((A - self.med) / self.mad).max(axis=1)

    ALARM = {}
    for b in BEAR:
        A = G[b][["H_RMS", "H_Kurt"]].to_numpy()
        z = Exceedance().fit(A[:K]).score(A)
        c = np.convolve((z[K:] > np.quantile(z[:K], 0.99)).astype(float),
                        np.ones(10) / 10, "valid")
        ALARM[b] = int(np.flatnonzero(c >= 0.5)[0]) + K

    path = OUT / "xjtu_loadzone_features.csv"
    if path.exists():
        LZ = pd.read_csv(path)
        print(f"{path.name} already built: {len(LZ)} snapshots")
    else:
        jobs = []
        for b in BEAR:
            cond = G[b]["Cond"].iloc[0]
            snaps = G[b]["Snapshot"].to_numpy()
            keep = sorted(set(snaps[:K].tolist()) | set(snaps[ALARM[b]:].tolist()))
            jobs.append((cond, b, keep))
        t0 = time.time()
        rows = []
        with ProcessPoolExecutor(max_workers=6) as pool:
            for i, out in enumerate(pool.map(one_bearing, jobs), 1):
                rows.extend(out)
                print(f"  [{i}/15] {out[0]['Bearing']}: {len(out)} snapshots "
                      f"({time.time() - t0:.0f}s)", flush=True)
        LZ = pd.DataFrame(rows)
        LZ.to_csv(path, index=False)
        print(f"wrote {path.name}: {len(LZ)} snapshots in {(time.time()-t0)/60:.1f} min")

    print("done -- run notebooks/20_loadzone_modulation.ipynb for the analysis")
