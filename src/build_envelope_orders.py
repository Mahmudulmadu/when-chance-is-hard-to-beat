# -*- coding: utf-8 -*-
"""Order-domain envelope spectra for the spectrum-CNN probe -- notebook 28.

The raw-waveform WDCNN reads bearing identity and operating point, not race.
The obvious objection is that a waveform at three shaft speeds is the wrong
representation, and that a network given what the forest is given -- the
envelope spectrum on a shaft-order axis -- would find the race. This caches
that representation so the objection can be tested rather than argued.

The spectrum is computed exactly as the feature builders compute it (notebook
13 for XJTU-SY, notebook 11 for Paderborn): band-pass to the resonance band,
Hilbert envelope, Hann window, power spectrum. It is then

  * expressed in dB relative to the same 10-400 Hz floor the peak features use,
    so the network sees floor-relative levels, not gain; and
  * resampled onto a shaft-order axis 0.5-12.0 in steps of 0.02 (575 bins),
    which puts BPFO at 3.08, BPFI at 4.92 and the shaft rate at 1.0 for every
    snapshot regardless of speed. The bearing-frequency peaks the forest reads
    are therefore at fixed indices.

For XJTU-SY it also caches each race unit's healthy reference spectrum (median
of its first 20 snapshots), so the probe can be run with the per-unit
calibration the forest's features carry. Paderborn records no healthy history
for a damaged bearing (nb19), so it has no such reference.

Usage:  python src/build_envelope_orders.py
Writes data/processed/orders_xjtu.npy (5793 x 576) + orders_xjtu_meta.csv
       data/processed/orders_pu.npy   (340 x 576)  + orders_pu_meta.csv
       data/processed/orders_xjtu_baseline.npy (11 x 576) + orders_xjtu_baseline_units.csv
"""
import pathlib
import time
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, hilbert, get_window

warnings.simplefilter("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_XJ = ROOT / "data" / "raw" / "xjtu"
RAW_PU = ROOT / "data" / "raw" / "paderborn"
OUT = ROOT / "data" / "processed"
ORDERS = np.arange(0.5, 12.0 + 1e-9, 0.02)           # 575 bins

FAULT = {"Bearing1_1": "OR", "Bearing1_2": "OR", "Bearing1_3": "OR",
         "Bearing1_4": "cage", "Bearing1_5": "IR+OR", "Bearing2_1": "IR",
         "Bearing2_2": "OR", "Bearing2_3": "cage", "Bearing2_4": "OR",
         "Bearing2_5": "OR", "Bearing3_1": "OR", "Bearing3_2": "compound",
         "Bearing3_3": "IR", "Bearing3_4": "IR", "Bearing3_5": "OR"}
COND_DIR = {"Bearing1": "35Hz12kN", "Bearing2": "37.5Hz11kN", "Bearing3": "40Hz10kN"}
K = 20

# --- XJTU-SY, as notebook 13
XJ_FS, XJ_WIN = 25_600, 32_768
XJ_HANN = get_window("hann", XJ_WIN)
XJ_F = np.fft.rfftfreq(XJ_WIN, 1 / XJ_FS)
XJ_FLOOR = (XJ_F > 10) & (XJ_F < 400)
XJ_SOS = butter(4, (1000, 12000), btype="bandpass", fs=XJ_FS, output="sos")

# --- Paderborn, as notebook 11
PU_COND = "N15_M07_F04"
PU_FS, PU_KEEP, PU_WIN = 64_000, 245_760, 65_536
PU_SUB = PU_KEEP // PU_WIN
PU_HANN = get_window("hann", PU_WIN)
PU_F = np.fft.rfftfreq(PU_WIN, 1 / PU_FS)
PU_FLOOR = (PU_F > 10) & (PU_F < 400)
PU_SOS = butter(4, (1000, 20000), btype="bandpass", fs=PU_FS, output="sos")
PU_BEARINGS = ([(f"K00{i}", "Healthy", "healthy") for i in range(1, 7)]
               + [(c, "OR", "real") for c in ("KA04", "KA15", "KA16", "KA22", "KA30")]
               + [(c, "IR", "real") for c in ("KI04", "KI14", "KI16", "KI17", "KI18", "KI21")])


def to_orders(spec, F, floor_mask, fr):
    """Floor-relative dB, resampled onto the shaft-order axis."""
    db = 10 * np.log10(spec / np.median(spec[floor_mask]))
    return np.interp(ORDERS * fr, F, db).astype(np.float32)


def xjtu_order_row(b, s, fr_of):
    """Cached index s (0-based) is file (s+1).csv, which the feature table calls
    Snapshot s+1. The first build looked the speed up at Snapshot s; fr_Hz is
    constant within every XJTU-SY unit, so that cache is byte-identical to this."""
    x = pd.read_csv(RAW_XJ / COND_DIR[b[:8]] / b / f"{s + 1}.csv").iloc[:, 0].to_numpy(np.float64)
    e = np.abs(hilbert(sosfiltfilt(XJ_SOS, x)))
    spec = np.abs(np.fft.rfft((e - e.mean()) * XJ_HANN)) ** 2
    return to_orders(spec, XJ_F, XJ_FLOOR, fr_of[(b, s + 1)])


def xjtu_speeds():
    feats = pd.read_csv(OUT / "xjtu_envelope_features.csv")[["Bearing", "Snapshot", "fr_Hz"]]
    return {(b, s): f for b, s, f in feats.itertuples(index=False)}


def build_xjtu():
    fr_of = xjtu_speeds()
    race = [b for b in sorted(FAULT) if FAULT[b] in ("OR", "IR")]
    meta, X = [], []
    for b in race:
        n = len(list((RAW_XJ / COND_DIR[b[:8]] / b).glob("*.csv")))
        for s in range(K, n):
            X.append(xjtu_order_row(b, s, fr_of))
            meta.append(dict(row=len(meta), unit=b, fault=FAULT[b], cond=COND_DIR[b[:8]],
                             snapshot=s, life=n, fr_Hz=fr_of[(b, s + 1)]))
        print(f"  {b}: {n - K} snapshots", flush=True)
    return np.stack(X), pd.DataFrame(meta)


def build_xjtu_baseline():
    """Each race unit's healthy reference: the per-bin median of its first K order
    spectra (files 1..K), in dB. Subtracting it is what nb22's `xjtu_matrices()`
    does to the forest's geometry features -- nb13's per-unit calibration."""
    fr_of = xjtu_speeds()
    race = [b for b in sorted(FAULT) if FAULT[b] in ("OR", "IR")]
    B = np.stack([np.median(np.stack([xjtu_order_row(b, s, fr_of) for s in range(K)]), axis=0)
                  for b in race]).astype(np.float32)
    return B, pd.DataFrame(dict(unit=race))


def pu_channels(p):
    p = pathlib.Path(p)
    ch = loadmat(p)[p.stem]["Y"][0, 0][0]
    return {str(x["Name"][0]): x["Data"].squeeze() for x in ch}


def build_pu():
    meta, X = [], []
    for code, cond, dom in PU_BEARINGS:
        for run in range(1, 21):
            c = pu_channels(RAW_PU / code / f"{PU_COND}_{code}_{run}.mat")
            v = c["vibration_1"][:PU_KEEP].astype(np.float64)
            fr = float(np.mean(c["speed"])) / 60
            e = np.abs(hilbert(sosfiltfilt(PU_SOS, v)))
            w = (e - e.mean())[: PU_SUB * PU_WIN].reshape(PU_SUB, PU_WIN)
            w = w - w.mean(axis=1, keepdims=True)
            spec = (np.abs(np.fft.rfft(w * PU_HANN, axis=1)) ** 2).mean(axis=0)
            X.append(to_orders(spec, PU_F, PU_FLOOR, fr))
            meta.append(dict(row=len(meta), unit=code, fault=cond, domain=dom, run=run, fr_Hz=fr))
        print(f"  {code}: 20 runs", flush=True)
    return np.stack(X), pd.DataFrame(meta)


if __name__ == "__main__":
    for name, fn in (("xjtu", build_xjtu), ("pu", build_pu)):
        path = OUT / f"orders_{name}.npy"
        if path.exists():
            print(f"{path.name} already built: {np.load(path, mmap_mode='r').shape}")
            continue
        t0 = time.time()
        X, M = fn()
        np.save(path, X)
        M.to_csv(OUT / f"orders_{name}_meta.csv", index=False)
        print(f"wrote {path.name}: {X.shape} float32, {X.nbytes / 1e6:.1f} MB "
              f"in {(time.time() - t0) / 60:.1f} min", flush=True)
    path = OUT / "orders_xjtu_baseline.npy"
    if path.exists():
        print(f"{path.name} already built: {np.load(path).shape}")
    else:
        B, U = build_xjtu_baseline()
        np.save(path, B)
        U.to_csv(OUT / "orders_xjtu_baseline_units.csv", index=False)
        print(f"wrote {path.name}: {B.shape} (median of files 1..{K} per race unit)")
