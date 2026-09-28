# -*- coding: utf-8 -*-
"""Assemble every number notebook 28 quotes -- notebook 28.

The CNN results come from `src/build_cnn_null.py`, run at --threads 5 (the thread
count the nulls use). This script reads those outputs, recomputes the forest
references under the *same* splits the network was scored on, adds the two
label-only baselines, and checks that every quantity scored twice came out the
same both times. It writes nothing new to the raw caches and trains no network.

Inputs (data/processed/):
  cnn_null_<rig>.csv                       exact WDCNN null; its true row is the 144-step result
  cnn_true_<rig>_big.csv                   WDCNN at 4x budget
  cnn_control_<rig>.csv                    WDCNN, samples shuffled in time
  cnn_fingerprint_<rig>.csv                WDCNN bearing identity
  cnn_true_<rig>_orders_{speccnn,rf}.csv   spectrum probe, true labelling
  cnn_fingerprint_<rig>_orders_speccnn.csv spectrum probe, identity
  cnn_true_xjtu_orders_cal_{speccnn,rf}.csv spectrum probe with per-unit calibration (XJTU-SY only)
  cnn_null_<rig>_orders_{speccnn,rf}.csv   exact spectrum nulls, both rigs (462 / 165), optional

Usage:  python src/build_cnn_summary.py
Writes data/processed/cnn_summary.csv and data/processed/speccnn_probe.csv
"""
import json
import pathlib
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
from sklearn.ensemble import RandomForestClassifier   # noqa: E402

import build_permutation_nulls as N22                  # nb22's matrices and forest

OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed"
RIGS = ("xjtu", "pu")
FAULT_XJ = {"Bearing1_1": "OR", "Bearing1_2": "OR", "Bearing1_3": "OR", "Bearing2_1": "IR",
            "Bearing2_2": "OR", "Bearing2_4": "OR", "Bearing2_5": "OR", "Bearing3_1": "OR",
            "Bearing3_3": "IR", "Bearing3_4": "IR", "Bearing3_5": "OR"}
XJ_ENV = ["H_BPFO", "H_BPFI", "H_FR", "H_BPFO_h2", "H_BPFI_h2", "H_BSF", "H_FTF",
          "H_BPFI_sb1", "H_BPFO_sb1"]


def row(rig, model, budget, acc, per=None, n_units=11, note=""):
    per = per or {}
    return dict(rig=rig, model=model, budget=budget, acc=float(acc),
                units_ge_half=sum(v >= 0.5 for v in per.values()) if per else np.nan,
                n_units=n_units, per_unit=json.dumps(per) if per else "", note=note)


def one(path):
    d = pd.read_csv(path)
    assert len(d) == 1, f"{path.name}: expected one row, got {len(d)}"
    return d.iloc[0]


def forest_geometry_per_unit():
    """nb22's forest, true labelling, per unit -- the split the network was scored on."""
    out = {}
    for rig, loader in (("xjtu", N22.xjtu_matrices), ("pu", N22.paderborn_matrices)):
        units, truth, chans = loader()
        P, per = chans["geometry"], {}
        for h in units:
            tr = [o for o in units if o != h]
            y = np.concatenate([[truth[o]] * len(P[o]) for o in tr])
            m = RandomForestClassifier(**N22.RF).fit(np.vstack([P[o] for o in tr]), y)
            per[h] = float((m.predict(P[h]) == truth[h]).mean())
        out[rig] = per
    return out


def forest_fingerprint():
    """The forest on the identity task, under exactly the folds mode_fingerprint uses."""
    rf = dict(n_estimators=100, random_state=0, n_jobs=-1)
    pu = pd.read_csv(OUT / f"pu_{N22.PU_COND}_sidebands.csv")
    h = pu[pu["Domain"] == "healthy"]
    acc_pu = []
    for k in range(4):
        held = list(range(1 + 5 * k, 6 + 5 * k))
        te, tr = h[h["Run"].isin(held)], h[~h["Run"].isin(held)]
        m = RandomForestClassifier(**rf).fit(tr[N22.PU_GEO], tr["Code"])
        acc_pu.append(float((m.predict(te[N22.PU_GEO]) == te["Code"]).mean()))
    # The feature table's Snapshot is the 1-based file number; the raw cache's snapshot
    # index is 0-based (file s+1). mode_fingerprint uses cache indices 20..39, i.e.
    # files 21..40, so the forest is scored on Snapshot 21..40 in the same four blocks.
    x = pd.read_csv(OUT / "xjtu_envelope_features.csv")
    e = x[x["Bearing"].isin(FAULT_XJ) & (x["Snapshot"] >= 21) & (x["Snapshot"] <= 40)]
    acc_xj = {}
    for name, feats in (("geometry", XJ_ENV), ("geometry + RMS/kurtosis", XJ_ENV + ["H_RMS", "H_Kurt"])):
        a = []
        for k in range(4):
            held = list(range(21 + 5 * k, 26 + 5 * k))
            te, tr = e[e["Snapshot"].isin(held)], e[~e["Snapshot"].isin(held)]
            m = RandomForestClassifier(**rf).fit(tr[feats], tr["Bearing"])
            a.append(float((m.predict(te[feats]) == te["Bearing"]).mean()))
        acc_xj[name] = float(np.mean(a))
    return float(np.mean(acc_pu)), acc_xj


if __name__ == "__main__":
    rows, checks = [], []
    rf_geo = forest_geometry_per_unit()
    rf_fp_pu, rf_fp_xj = forest_fingerprint()

    for rig in RIGS:
        null = pd.read_csv(OUT / f"cnn_null_{rig}.csv")
        t = null[null["is_true"]]
        assert len(t) == 1, f"cnn_null_{rig}.csv: expected one true row, got {len(t)}"
        rows.append(row(rig, "WDCNN raw waveform", "144 steps", t.iloc[0]["acc"],
                        json.loads(t.iloc[0]["per_unit"])))
        b = one(OUT / f"cnn_true_{rig}_big.csv")
        rows.append(row(rig, "WDCNN raw waveform", "576 steps (4x)", b["acc"], json.loads(b["per_unit"])))
        c = one(OUT / f"cnn_control_{rig}.csv")
        rows.append(row(rig, "WDCNN, samples shuffled in time", "144 steps", c["acc"],
                        json.loads(c["per_unit"]), note="incapable control"))
        per = rf_geo[rig]
        rows.append(row(rig, "Random Forest, geometry features", "nb22", np.mean(list(per.values())), per))
        fp = pd.read_csv(OUT / f"cnn_fingerprint_{rig}.csv")
        rows.append(row(rig, "WDCNN bearing-identity fingerprint", "144 steps", fp["acc"].mean(),
                        n_units=int(round(1 / fp["chance"].iloc[0])),
                        note=f"chance {fp['chance'].iloc[0]:.3f}; folds {fp['acc'].round(3).tolist()}"))
    rows.append(row("pu", "Random Forest bearing-identity fingerprint", "nb22 forest", rf_fp_pu,
                    n_units=6, note="geometry features, same run-held-out folds"))
    rows.append(row("xjtu", "Random Forest bearing-identity fingerprint", "nb22 forest",
                    rf_fp_xj["geometry + RMS/kurtosis"], n_units=11,
                    note=f"geometry + RMS/kurtosis, same snapshot folds; geometry alone "
                         f"{rf_fp_xj['geometry']:.3f}"))

    # label-only baselines on XJTU-SY: no signal is read
    cond = {u: u[:8] for u in FAULT_XJ}
    maj = {h: float(Counter(FAULT_XJ[o] for o in FAULT_XJ if o != h).most_common(1)[0][0] == FAULT_XJ[h])
           for h in FAULT_XJ}
    rows.append(row("xjtu", "always OR (majority class)", "labels only", np.mean(list(maj.values())), maj))
    orc = {h: float(Counter(FAULT_XJ[o] for o in FAULT_XJ if o != h and cond[o] == cond[h])
                    .most_common(1)[0][0] == FAULT_XJ[h]) for h in FAULT_XJ}
    rows.append(row("xjtu", "condition-majority oracle", "labels only", np.mean(list(orc.values())), orc,
                    note="majority race among the OTHER units at the same operating point"))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "cnn_summary.csv", index=False)

    probe = []
    for rig in RIGS:
        for key, name in (("speccnn", "spectrum-CNN"), ("rf", "RF on 576 order bins")):
            t = one(OUT / f"cnn_true_{rig}_orders_{key}.csv")
            per = json.loads(t["per_unit"])
            probe.append(dict(rig=rig, model=name, acc=float(t["acc"]),
                              units_ge_half=sum(v >= 0.5 for v in per.values()), per_unit=t["per_unit"]))
            npath = OUT / f"cnn_null_{rig}_orders_{key}.csv"
            if npath.exists():                          # scored twice -> must agree exactly
                nt = pd.read_csv(npath).query("is_true").iloc[0]
                checks.append((f"{rig} {name}: true row of the null == mode true",
                               abs(float(nt["acc"]) - float(t["acc"])) < 1e-12))
        fp = pd.read_csv(OUT / f"cnn_fingerprint_{rig}_orders_speccnn.csv")
        probe.append(dict(rig=rig, model="spectrum-CNN identity fingerprint", acc=float(fp["acc"].mean()),
                          units_ge_half=np.nan,
                          per_unit=json.dumps(dict(folds=fp["acc"].tolist(), chance=float(fp["chance"].iloc[0])))))
    # XJTU-SY only: the same spectra minus each unit's own healthy reference, which is the
    # calibration nb22's forest features carry there (Paderborn has no healthy history)
    for key, name in (("speccnn", "spectrum-CNN, per-unit calibrated"),
                      ("rf", "RF on 576 order bins, per-unit calibrated")):
        t = one(OUT / f"cnn_true_xjtu_orders_cal_{key}.csv")
        per = json.loads(t["per_unit"])
        probe.append(dict(rig="xjtu", model=name, acc=float(t["acc"]),
                          units_ge_half=sum(v >= 0.5 for v in per.values()), per_unit=t["per_unit"]))
    pd.DataFrame(probe).to_csv(OUT / "speccnn_probe.csv", index=False)

    print(S[["rig", "model", "budget", "acc", "units_ge_half", "n_units"]].to_string(index=False))
    print()
    print(pd.DataFrame(probe)[["rig", "model", "acc", "units_ge_half"]].to_string(index=False))
    print()
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    assert all(ok for _, ok in checks), "a quantity scored twice disagreed"
    print("wrote cnn_summary.csv, speccnn_probe.csv")
