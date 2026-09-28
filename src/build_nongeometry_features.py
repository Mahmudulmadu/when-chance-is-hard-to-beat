# -*- coding: utf-8 -*-
"""Non-geometry vibration features for XJTU-SY and Paderborn (notebook 18).

The defining constraint, and the whole point of the channel: **no feature may
reference BPFO, BPFI, BSF, FTF or the shaft rate.** Everything here is either a
property of *where in absolute Hz* the energy sits, or a property of the *shape*
of the spectrum -- never of a repetition rate predicted from bearing geometry.

Three notebooks arrived at this requirement independently:
  nb12  detection   -- KA15 is silent at BPFO, KA22 is off-frequency (order 5.5)
  nb15  reject      -- Bearing1_4 raises nothing at any order
  nb17  attribution -- early inner race is confidently mislabelled outer race

Feature families, per channel:
  SK*    fast kurtogram over a log-spaced filterbank. The band that maximises
         envelope kurtosis is where the impulsiveness lives; its centre
         frequency is the resonance-tracking feature.
  shape  centroid, spread, entropy, flatness, 95 % rolloff of the raw spectrum
  env*   entropy and flatness of the envelope spectrum inside the most
         impulsive band -- peaky means a geometry-locked repetition, diffuse
         means something else. This is the one that could catch KA22.
  oct*   fixed-Hz octave energies, geometry-free by construction

Usage:  python src/build_nongeometry_features.py xjtu|paderborn
Writes data/processed/{xjtu,pu_<cond>}_nongeometry_features.csv
"""
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"

XJTU_RAW = ROOT / "data" / "raw" / "xjtu"
PU_RAW = ROOT / "data" / "raw" / "paderborn"
XJTU_COND = {"35Hz12kN": 35.0, "37.5Hz11kN": 37.5, "40Hz10kN": 40.0}
PU_COND = "N15_M07_F04"                      # notebook 11's best attribution point

N_BAND, N_OCT = 12, 8


def kurt(a):
    """Fisher kurtosis, without scipy's per-call policy wrapper (50 % of runtime)."""
    d = a - a.mean()
    m2 = np.mean(d * d)
    if m2 <= 0:
        return 0.0
    return float(np.mean(d ** 4) / (m2 * m2) - 3.0)


def spec_shape(spec, f):
    """Geometry-free descriptors of a spectrum's shape."""
    tot = spec.sum()
    if tot <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    p = spec / tot
    c = float((f * p).sum())
    sd = float(np.sqrt(np.maximum(((f - c) ** 2 * p).sum(), 0.0)))
    ent = float(-(p * np.log(p + 1e-30)).sum() / np.log(len(p)))
    flat = float(np.exp(np.mean(np.log(spec + 1e-30))) / max(spec.mean(), 1e-30))
    roll = float(f[min(int(np.searchsorted(np.cumsum(p), 0.95)), len(f) - 1)])
    return c, sd, ent, flat, roll


class Extractor:
    """Holds the per-dataset frequency grid so it is built once, not per snapshot."""

    def __init__(self, fs, win, band_lo, band_hi, oct_lo, oct_hi):
        self.fs, self.win = fs, win
        self.F = np.fft.rfftfreq(win, 1 / fs)
        self.T = win / fs
        be = np.geomspace(band_lo, band_hi, N_BAND + 1)
        self.bands = [(be[i], be[i + 1]) for i in range(N_BAND)]
        self.bidx = [np.flatnonzero((self.F >= lo) & (self.F < hi)) for lo, hi in self.bands]
        self.bctr = [float(np.sqrt(lo * hi)) for lo, hi in self.bands]
        oe = np.geomspace(oct_lo, oct_hi, N_OCT + 1)
        self.oidx = [np.flatnonzero((self.F >= oe[i]) & (self.F < oe[i + 1]))
                     for i in range(N_OCT)]

    def band_env(self, X, k):
        """Complex envelope of band k, decimated to the band's own Nyquist.

        Only the band's bins are ifft'd, into the smallest power-of-two array
        that holds them, so this costs O(band width) rather than O(win).
        """
        idx = self.bidx[k]
        nb = len(idx)
        if nb < 8:
            return None
        L = 1 << int(np.ceil(np.log2(nb)))
        Y = np.zeros(L, dtype=complex)
        Y[:nb] = X[idx]
        return np.abs(np.fft.ifft(Y))

    def features(self, x, tag):
        X = np.fft.rfft(x)
        spec = np.abs(X) ** 2
        r = {}
        ks, envs = [], []
        for k in range(N_BAND):
            e = self.band_env(X, k)
            envs.append(e)
            ks.append(kurt(e) if e is not None else np.nan)
        ks = np.asarray(ks, dtype=float)
        j = int(np.nanargmax(ks)) if np.isfinite(ks).any() else 0
        r[f"{tag}_SKmax"] = float(np.nanmax(ks)) if np.isfinite(ks).any() else 0.0
        r[f"{tag}_SKband"] = self.bctr[j]                 # resonance tracking
        r[f"{tag}_SKspread"] = float(np.nanstd(ks))
        r[f"{tag}_SKlow"] = float(np.nanmean(ks[:N_BAND // 2]))
        r[f"{tag}_SKhigh"] = float(np.nanmean(ks[N_BAND // 2:]))
        c, sd, ent, flat, roll = spec_shape(spec, self.F)
        r[f"{tag}_cent"], r[f"{tag}_spread"] = c, sd
        r[f"{tag}_ent"], r[f"{tag}_flat"], r[f"{tag}_roll"] = ent, flat, roll
        # envelope-spectrum diffuseness inside the most impulsive band
        e = envs[j]
        if e is None:
            r[f"{tag}_envent"] = r[f"{tag}_envflat"] = 0.0
        else:
            v = e - e.mean()
            es = np.abs(np.fft.rfft(v)) ** 2
            fe = np.fft.rfftfreq(len(v), self.T / len(v))
            m = (fe > 5) & (fe < min(1000.0, fe[-1]))
            if m.sum() > 8:
                _, _, eent, eflat, _ = spec_shape(es[m], fe[m])
            else:
                eent = eflat = 0.0
            r[f"{tag}_envent"], r[f"{tag}_envflat"] = eent, eflat
        for i in range(N_OCT):
            s = spec[self.oidx[i]].sum() if len(self.oidx[i]) else 0.0
            r[f"{tag}_oct{i}"] = float(10 * np.log10(s + 1e-30))
        return r


# ----------------------------------------------------------------------- XJTU
XJ = dict(fs=25_600, win=32_768, band_lo=500, band_hi=12_000, oct_lo=250, oct_hi=12_000)


def xjtu_bearing(args):
    cond, bdir = args
    ex = Extractor(**XJ)
    rows = []
    for f in sorted(pathlib.Path(bdir).glob("*.csv"), key=lambda p: int(p.stem)):
        a = pd.read_csv(f).to_numpy(dtype=np.float64)
        if len(a) < ex.win:
            a = np.pad(a, ((0, ex.win - len(a)), (0, 0)))
        r = {"Cond": cond, "Bearing": pathlib.Path(bdir).name, "Snapshot": int(f.stem)}
        r.update(ex.features(a[:ex.win, 0], "H"))
        r.update(ex.features(a[:ex.win, 1], "V"))
        rows.append(r)
    return rows


def build_xjtu():
    path = OUT / "xjtu_nongeometry_features.csv"
    if path.exists():
        print(f"{path.name} already built")
        return
    jobs = [(c, str(b)) for c in XJTU_COND
            for b in sorted((XJTU_RAW / c).iterdir()) if b.is_dir()]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i, out in enumerate(pool.map(xjtu_bearing, jobs), 1):
            rows.extend(out)
            print(f"  [{i}/{len(jobs)}] {out[0]['Bearing']}: {len(out)} snapshots "
                  f"({time.time() - t0:.0f}s)", flush=True)
    d = pd.DataFrame(rows).sort_values(["Cond", "Bearing", "Snapshot"])
    d.to_csv(path, index=False)
    print(f"wrote {path} -- {d.shape[0]} snapshots x {d.shape[1]} cols "
          f"in {(time.time() - t0) / 60:.1f} min")


# ------------------------------------------------------------------ Paderborn
PU = dict(fs=64_000, win=65_536, band_lo=1_000, band_hi=30_000, oct_lo=500, oct_hi=30_000)
PU_BEARINGS = ([(f"K00{i}", "Healthy", "healthy") for i in range(1, 7)]
               + [(c, "OR", "artificial") for c in ("KA01", "KA03", "KA05", "KA07")]
               + [(c, "IR", "artificial") for c in ("KI01", "KI03", "KI05", "KI07")]
               + [(c, "OR", "real") for c in ("KA04", "KA15", "KA16", "KA22", "KA30")]
               + [(c, "IR", "real") for c in ("KI04", "KI14", "KI16", "KI17", "KI18", "KI21")])


def pu_bearing(args):
    from scipy.io import loadmat
    code, cls, dom, cond = args
    ex = Extractor(**PU)
    n_keep = 245_760
    rows = []
    for run in range(1, 21):
        p = PU_RAW / code / f"{cond}_{code}_{run}.mat"
        ch = loadmat(p)[p.stem]["Y"][0, 0][0]
        c = {str(x["Name"][0]): x["Data"].squeeze() for x in ch}
        v = c["vibration_1"][:n_keep].astype(np.float64)
        n_sub = n_keep // ex.win
        acc = {}
        for s in range(n_sub):                    # average over sub-windows, as nb03/11 do
            f = ex.features(v[s * ex.win:(s + 1) * ex.win], "H")
            for k, val in f.items():
                acc.setdefault(k, []).append(val)
        r = {"Code": code, "Condition": cls, "Domain": dom, "Run": run}
        r.update({k: float(np.mean(vals)) for k, vals in acc.items()})
        rows.append(r)
    return rows


def build_pu():
    path = OUT / f"pu_{PU_COND}_nongeometry_features.csv"
    if path.exists():
        print(f"{path.name} already built")
        return
    jobs = [(c, cl, d, PU_COND) for c, cl, d in PU_BEARINGS]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i, out in enumerate(pool.map(pu_bearing, jobs), 1):
            rows.extend(out)
            print(f"  [{i}/{len(jobs)}] {out[0]['Code']} ({time.time() - t0:.0f}s)",
                  flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(path, index=False)
    print(f"wrote {path} -- {d.shape[0]} runs x {d.shape[1]} cols "
          f"in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "xjtu"
    if which in ("xjtu", "both"):
        build_xjtu()
    if which in ("paderborn", "both"):
        build_pu()
