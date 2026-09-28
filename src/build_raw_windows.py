# -*- coding: utf-8 -*-
"""Raw vibration cache for the deep baseline -- notebook 28.

Every model in this project so far reads a feature table. The deep baseline
reads the waveform, so it needs the waveform in a form that 1 815 + 5 082
training runs can pull crops from without touching 6 000 CSV and MAT files
each time.

Both rigs are decimated to the neighbourhood of WDCNN's native 12 kHz (Zhang et
al. 2017 on CWRU), so that a 2 048-sample crop spans a few shaft revolutions on
either rig rather than a fraction of one:

    XJTU-SY    25.6 kHz / 2 -> 12.8 kHz   2 048 samples = 160 ms = 5.6-6.4 rev
    Paderborn  64.0 kHz / 4 -> 16.0 kHz   2 048 samples = 128 ms = 3.2 rev

Decimation is an anti-aliased FIR, zero-phase. Nothing else is done to the
signal here; per-crop standardisation is the model's business, not the cache's.

What is cached:

  * XJTU-SY: the 11 race units (OR/IR), horizontal channel, snapshots K..N-1 --
    the same post-baseline rows notebook 22's `xjtu_matrices()` scores.
  * Paderborn: the 11 real damaged and 6 healthy bearings at N15_M07_F04, all 20
    runs each, `vibration_1` truncated to 245 760 samples as in notebooks 11-21.
    The healthy runs are for the bearing-identity (fingerprint) test.

Usage:  python src/build_raw_windows.py
Writes data/processed/raw_xjtu.npy  + raw_xjtu_meta.csv   (float16, ~190 MB)
       data/processed/raw_pu.npy    + raw_pu_meta.csv     (float16, ~42 MB)
"""
import pathlib
import time
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import decimate

warnings.simplefilter("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_XJ = ROOT / "data" / "raw" / "xjtu"
RAW_PU = ROOT / "data" / "raw" / "paderborn"
OUT = ROOT / "data" / "processed"

FAULT = {"Bearing1_1": "OR", "Bearing1_2": "OR", "Bearing1_3": "OR",
         "Bearing1_4": "cage", "Bearing1_5": "IR+OR", "Bearing2_1": "IR",
         "Bearing2_2": "OR", "Bearing2_3": "cage", "Bearing2_4": "OR",
         "Bearing2_5": "OR", "Bearing3_1": "OR", "Bearing3_2": "compound",
         "Bearing3_3": "IR", "Bearing3_4": "IR", "Bearing3_5": "OR"}
COND_DIR = {"Bearing1": "35Hz12kN", "Bearing2": "37.5Hz11kN", "Bearing3": "40Hz10kN"}
K = 20                                        # healthy baseline rows, as everywhere
XJ_Q, XJ_LEN = 2, 32_768 // 2                 # decimation factor, samples kept

PU_COND = "N15_M07_F04"
PU_BEARINGS = ([(f"K00{i}", "Healthy", "healthy") for i in range(1, 7)]
               + [(c, "OR", "real") for c in ("KA04", "KA15", "KA16", "KA22", "KA30")]
               + [(c, "IR", "real") for c in ("KI04", "KI14", "KI16", "KI17", "KI18", "KI21")])
PU_Q, PU_KEEP = 4, 245_760
PU_LEN = PU_KEEP // PU_Q


def build_xjtu():
    race = [b for b in sorted(FAULT) if FAULT[b] in ("OR", "IR")]
    meta, X = [], []
    for b in race:
        d = RAW_XJ / COND_DIR[b[:8]] / b
        n = len(list(d.glob("*.csv")))
        for s in range(K, n):                 # files are 1-indexed: snapshot s -> s+1.csv
            x = pd.read_csv(d / f"{s + 1}.csv").iloc[:, 0].to_numpy(np.float64)
            X.append(decimate(x, XJ_Q, ftype="fir", zero_phase=True)[:XJ_LEN].astype(np.float16))
            meta.append(dict(row=len(meta), unit=b, fault=FAULT[b], cond=COND_DIR[b[:8]],
                             snapshot=s, life=n))
        print(f"  {b}: {n - K} snapshots", flush=True)
    return np.stack(X), pd.DataFrame(meta)


def pu_channels(p):
    p = pathlib.Path(p)
    ch = loadmat(p)[p.stem]["Y"][0, 0][0]
    return {str(x["Name"][0]): x["Data"].squeeze() for x in ch}


def build_pu():
    meta, X = [], []
    for code, cond, dom in PU_BEARINGS:
        for run in range(1, 21):
            v = pu_channels(RAW_PU / code / f"{PU_COND}_{code}_{run}.mat")["vibration_1"]
            v = v[:PU_KEEP].astype(np.float64)
            X.append(decimate(v, PU_Q, ftype="fir", zero_phase=True)[:PU_LEN].astype(np.float16))
            meta.append(dict(row=len(meta), unit=code, fault=cond, domain=dom, run=run))
        print(f"  {code}: 20 runs", flush=True)
    return np.stack(X), pd.DataFrame(meta)


if __name__ == "__main__":
    for name, fn, fs in (("xjtu", build_xjtu, 25_600 // XJ_Q), ("pu", build_pu, 64_000 // PU_Q)):
        path = OUT / f"raw_{name}.npy"
        if path.exists():
            print(f"{path.name} already built: {np.load(path, mmap_mode='r').shape}")
            continue
        t0 = time.time()
        print(f"{name}: decimated to {fs} Hz", flush=True)
        X, M = fn()
        M["fs_hz"] = fs
        np.save(path, X)
        M.to_csv(OUT / f"raw_{name}_meta.csv", index=False)
        print(f"wrote {path.name}: {X.shape} float16, {X.nbytes / 1e6:.0f} MB "
              f"in {(time.time() - t0) / 60:.1f} min")
