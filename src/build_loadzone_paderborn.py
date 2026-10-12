# -*- coding: utf-8 -*-
"""Shaft-synchronous (load-zone) envelope features for Paderborn (notebook 21).

Notebook 20 measured, on XJTU-SY, that race attribution works from the shaft rate
without bearing geometry: seven dimensionless features from folding the envelope
at shaft rate reached 90.0 % of snapshots and 11/11 units with no BPFO, BPFI, BSF
or FTF. That claim rests on one rig and eleven units.

Paderborn is the check: 25 bearings, artificial and real damage, a documented
shaft speed recorded per run, and an established bearing-held-out protocol
(notebook 05). Nothing new has to be measured; the existing recordings are enough.

Differences from the XJTU build that matter:
  - 64 kHz rather than 25.6 kHz, so the resonance band runs to 20 kHz
  - each 1.024 s window holds ~25 shaft revolutions against XJTU's ~51, so the
    synchronous average has about half the averaging gain
  - the shaft rate is recorded per run, so the search is a refinement of a
    measured value rather than of a nominal one

Usage:  python src/build_loadzone_paderborn.py
Writes data/processed/pu_<cond>_loadzone_features.csv
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
COND = "N15_M07_F04"                       # notebook 11's best attribution point
FS, WIN, N_KEEP, NBIN = 64_000, 65_536, 245_760, 64
SOS = butter(4, (1000, 20000), btype="bandpass", fs=FS, output="sos")

BEARINGS = ([(f"K00{i}", "Healthy", "healthy") for i in range(1, 7)]
            + [(c, "OR", "artificial") for c in ("KA01", "KA03", "KA05", "KA07")]
            + [(c, "IR", "artificial") for c in ("KI01", "KI03", "KI05", "KI07")]
            + [(c, "OR", "real") for c in ("KA04", "KA15", "KA16", "KA22", "KA30")]
            + [(c, "IR", "real") for c in ("KI04", "KI14", "KI16", "KI17", "KI18", "KI21")])


def lz_features(x, f0):
    """Envelope folded onto one shaft revolution and averaged over all of them.

    The FFT bin spacing is far too coarse to fold on, so fr is chosen as the
    value maximising the synchronous energy, on a grid fine enough to hold phase
    drift under an eighth of a revolution across the record.

    NOTE (notebook 20, finding 4): LZ_frdev is a pure speed reading and is not a
    usable feature: it proxies how stably the search locks and scored 81 % on
    nothing at all. It is written out for diagnosis only; quarantine it.
    """
    e = np.abs(hilbert(sosfiltfilt(SOS, x)))
    e = e - e.mean()
    rms = np.sqrt(np.mean(e ** 2)) + 1e-30
    t = np.arange(len(e)) / FS
    best = (-1.0, f0, None)
    for fr in np.arange(f0 * 0.97, f0 * 1.03, 0.02):
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
    code, cls, dom = args
    rows = []
    for run in range(1, 21):
        p = RAW / code / f"{COND}_{code}_{run}.mat"
        if not p.exists():
            continue
        ch = loadmat(p)[p.stem]["Y"][0, 0][0]
        c = {str(x["Name"][0]): x["Data"].squeeze() for x in ch}
        v = c["vibration_1"][:N_KEEP].astype(np.float64)
        f0 = float(c["speed"].mean()) / 60.0        # recorded, not nominal
        acc = {}
        for s in range(N_KEEP // WIN):
            for k, val in lz_features(v[s * WIN:(s + 1) * WIN], f0).items():
                acc.setdefault(k, []).append(val)
        r = {"Code": code, "Condition": cls, "Domain": dom, "Run": run, "fr_Hz": f0}
        r.update({k: float(np.mean(vals)) for k, vals in acc.items()})
        rows.append(r)
    return rows


if __name__ == "__main__":
    path = OUT / f"pu_{COND}_loadzone_features.csv"
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
        print(f"wrote {path.name}: {len(d)} runs x {d.shape[1]} cols "
              f"in {(time.time() - t0) / 60:.1f} min")
