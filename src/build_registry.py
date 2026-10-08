# -*- coding: utf-8 -*-
"""The results registry -- notebook 29.

The paper's argument is that a number in this field is not evidence until it has
been priced against something: a permuted-label null, a degenerate baseline, or
chance. This applies that rule to the project's own numbers and writes them to
one table, `results/registry.csv`, so every claim the paper makes can be read
with its comparator beside it.

Two kinds of row, and the `source` column says which:

  * **recomputed** -- the value is recomputed here from a cached result file, so
    it cannot drift from the notebook that produced it. Most rows are these.
  * **quoted (nbNN)** -- the value is typed from a notebook whose intermediate
    results were never cached to CSV. These are listed so the paper does not
    silently mix the two, and each names the notebook to re-run.

Two columns qualify the attribution family: `p_selection_aware`, the Paderborn geometry p with
notebook 11's configuration choice repeated under every labelling (build_selection_null.py),
and `q_bh_selection_aware`, the Benjamini-Hochberg adjustment with that p substituted.

Usage:  python src/build_registry.py
Writes results/registry.csv
"""
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = pathlib.Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"
OUT = ROOT / "results"
rows = []


def add(nb, rig, n_units, claim, value, comparator, comp_value, p=np.nan, source="recomputed", note="",
        p_selection_aware=np.nan):
    rows.append(dict(notebook=nb, rig=rig, n_units=n_units, claim=claim, value=value,
                     comparator=comparator, comparator_value=comp_value, p=p,
                     verdict=("" if pd.isna(p) else ("clears its null" if p < 0.05 else "inside its null")),
                     source=source, note=note, p_selection_aware=p_selection_aware))


# ---- attribution, exact permuted-label nulls (nb22; the accuracies are nb11/14/20/21)
NB = {("XJTU-SY", "geometry"): "nb14/22", ("Paderborn", "geometry"): "nb11/22",
      ("XJTU-SY", "load-zone"): "nb20/22", ("Paderborn", "load-zone"): "nb21/22"}
nb22 = d = pd.read_csv(P / "permutation_nulls.csv")
# The Paderborn geometry configuration (400 N, carriers + sidebands) was chosen in nb11 on the
# same eleven units. build_selection_null.py repeats that choice under every labelling, and the
# row carries the resulting p beside the fixed-configuration one (paper section 4.2).
sel = pd.read_csv(P / "selection_null_pu.csv")
for (rig, ch), g in d.groupby(["rig", "channel"]):
    t = float(g.loc[g.is_true, "acc"].iloc[0])
    null = g.loc[~g.is_true, "acc"]
    note, p_sel = "bearing-held-out, all assignments enumerated", np.nan
    if (rig, ch) == ("Paderborn", "geometry"):
        ts = float(sel.loc[sel.is_true, "acc_selection"].iloc[0])
        assert abs(ts - t) < 1e-9, (ts, t)
        k = int((sel.acc_selection >= ts).sum())
        p_sel = round(k / len(sel), 4)
        note += (f"; with nb11's choice among 12 configurations repeated under every labelling, "
                 f"p = {k}/{len(sel)} and the null's 95th pct is "
                 f"{sel.loc[~sel.is_true, 'acc_selection'].quantile(0.95):.4f}")
    add(NB[(rig, ch)], rig, 11, f"race attribution, {ch} features", round(t, 4),
        f"exact permuted-label null, 95th pct (N={len(g)})", round(float(null.quantile(0.95)), 4),
        round(float((g.acc >= t).mean()), 4), note=note, p_selection_aware=p_sel)

# ---- the non-geometry channel: the study-size curve's own 15-tree forest and windows, with
# the null enumerated over all 165 assignments instead of the curve's 20 draws (nb18/24)
ng = pd.read_csv(P / "nongeometry_null_xjtu.csv").query("channel == 'non-geometry'")
t = float(ng.loc[ng.is_true, "acc"].iloc[0])
null = ng.loc[~ng.is_true, "acc"]
add("nb18/24", "XJTU-SY", 11, "race attribution, non-geometry features", round(t, 4),
    f"exact permuted-label null, 95th pct (N={len(ng)})", round(float(null.quantile(0.95)), 4),
    round(float((ng.acc >= t).mean()), 4),
    note="15-tree forest of the study-size curve, not nb22's 100-tree fit")

# ---- deep baseline and the spectrum probe, each against its own null (nb28)
for f, nb_, rig, claim in (("cnn_null_xjtu", "nb28", "XJTU-SY", "race attribution, WDCNN on raw waveform"),
                           ("cnn_null_pu", "nb28", "Paderborn", "race attribution, WDCNN on raw waveform"),
                           ("cnn_null_pu_orders_speccnn", "nb28", "Paderborn", "race attribution, spectrum-CNN on 576 order bins"),
                           ("cnn_null_pu_orders_rf", "nb28", "Paderborn", "race attribution, forest on 576 order bins"),
                           ("cnn_null_xjtu_orders_speccnn", "nb28", "XJTU-SY", "race attribution, spectrum-CNN on 576 order bins"),
                           ("cnn_null_xjtu_orders_rf", "nb28", "XJTU-SY", "race attribution, forest on 576 order bins")):
    g = pd.read_csv(P / f"{f}.csv")
    t = float(g.loc[g.is_true, "acc"].iloc[0])
    null = g.loc[~g.is_true, "acc"]
    exact = len(g) in (165, 462)
    add(nb_, rig, 11, claim, round(t, 4),
        f"{'exact' if exact else 'sampled'} permuted-label null, 95th pct (N={len(g)})",
        round(float(null.quantile(0.95)), 4), round(float((1 + (null >= t).sum()) / (1 + len(null))), 4),
        note="the null is the model's own; it is 4-7 points wider than the forest's")

# ---- what a model can do instead of diagnosis: identity, and an incapable control (nb28)
S = pd.read_csv(P / "cnn_summary.csv")


def pick(rig, model, budget=None):
    s = S[(S.rig == rig) & (S.model == model)]
    if budget is not None:
        s = s[s.budget == budget]
    assert len(s) == 1, f"{rig}/{model}/{budget}: {len(s)} rows"
    return s.iloc[0]


for rig, key in (("XJTU-SY", "xjtu"), ("Paderborn", "pu")):
    fp = pick(key, "WDCNN bearing-identity fingerprint")
    add("nb28", rig, int(fp.n_units), "telling individual bearings apart, WDCNN", round(float(fp.acc), 4),
        "chance (1 / units)", round(1 / int(fp.n_units), 4), note="same network, same budget that fails at the race")
    ct = pick(key, "WDCNN, samples shuffled in time")
    intact = pick(key, "WDCNN raw waveform", "144 steps")
    add("nb28", rig, 11, "race attribution, WDCNN with samples shuffled in time", round(float(ct.acc), 4),
        "the same network on the intact waveform", round(float(intact.acc), 4),
        note="deliberately incapable control: destroys the waveform, keeps the amplitudes")

# ---- remaining life: every estimator against the degenerate baseline (nb27)
c = pd.read_csv(P / "clock_table.csv")
pa = c[c.post]
mae = pa.groupby(["estimator", "unit"]).abs_err.mean().groupby("estimator").median()
s = pa[pa.y > 0].copy()
s["rel"] = s.abs_err / s.y
rel = s.groupby(["estimator", "unit"]).rel.median().groupby("estimator").median()
for est, label in (("state", "remaining life, two-channel state"), ("clock", "remaining life, learned clock"),
                   ("state_post", "remaining life, two-channel state fitted post-alarm")):
    add("nb16/27", "XJTU-SY", 15, f"{label} (median per-unit MAE, post-alarm, minutes)", round(float(mae[est]), 1),
        "predicting zero life left", round(float(mae["zero"]), 1),
        note=f"scale-free (error / life left): {rel[est]:.2f} against {rel['zero']:.2f} for the constant")
res = (c.abs_err - c.abs_err_life).abs()
add("nb27", "XJTU-SY", 15, "largest |RUL error - lifetime error| over all estimator-snapshots",
    float(f"{res.max():.3e}"), "float64 step (1 ULP) at the longest lifetime",
    float(f"{np.spacing(c.life.max()):.3e}"), note="the two are the same claim written twice")

# ---- the remaining-life floor and its two controls (nb17/26)
f = pd.read_csv(P / "floor_curve.csv")
x = f[(f.n == 14) & (np.isclose(f.q, 0.10))]
g = {k: v for k, v in x.groupby("scheme")}
add("nb17/26", "XJTU-SY", 15, "remaining-life floor, median per-unit safety (q0.10)",
    round(float(g["forest"].safety.median()), 3), "the same floor with lifetimes permuted",
    round(float(g["permuted"].safety.median()), 3),
    note=f"give-up {g['forest'].giveup.median():.2f}; share of bearings at 90 % safety "
         f"{(g['forest'].safety >= 0.9).mean():.3f} against {(g['prior'].safety >= 0.9).mean():.3f} for a lifetime prior")

# ---- how many units a claim needs (nb23/24). Referee M3: 12 subsets per size could not
# support a point estimate, so the curves were recomputed at 30, with each subset's null
# enumerated exactly wherever it is small enough. The detection rate now carries a
# Clopper-Pearson interval -- optimistic, because the subsets share bearings heavily, which
# the note quantifies.
def cp(k, n, conf=0.95):
    a = (1 - conf) / 2
    return (0.0 if k == 0 else beta.ppf(a, k, n - k + 1),
            1.0 if k == n else beta.ppf(1 - a, k + 1, n - k))


CONTROL = {"Paderborn": "load-zone", "XJTU-SY": "non-geometry"}       # inside their own full-set nulls
for rig, f_ in (("Paderborn", "power_curve.csv"), ("XJTU-SY", "power_curve_xjtu.csv")):
    allc = pd.read_csv(P / f_)
    pcv = allc.query("channel == 'geometry'")
    ctrl = allc[allc.channel == CONTROL[rig]]
    reach = 1 / (ctrl.n_perm + 1) < 0.05            # subsets whose exact null can return p < 0.05
    kc, kr = int((ctrl.p < 0.05).sum()), int((ctrl[reach].p < 0.05).sum())
    det = {}
    for n_, gn in pcv.groupby("n"):
        k, m = int((gn.p < 0.05).sum()), len(gn)
        det[int(n_)] = (k, m, k / m, cp(k, m), 1 / (gn.n_perm.median() + 1))
    first = int(min(n_ for n_, v in det.items() if v[2] >= 0.8))
    k, m, rate, (lo, hi), _ = det[first]
    reachable = [n_ for n_, v in det.items() if v[4] <= 0.05]
    add("nb23/24", rig, first, "units needed before the geometry features reach p < 0.05 in 80 % of subsets",
        first, "units used by every attribution result in this project", 11,
        note=f"{k} of {m} subsets at n={first}, {rate:.0%} (Clopper-Pearson 95 % CI "
             f"[{lo:.0%}, {hi:.0%}], and the subsets share units so it is optimistic); "
             f"rate by unit count " + ", ".join(f"n={n_}: {v[2]:.0%}" for n_, v in det.items())
             + f"; p < 0.05 is unreachable by construction below n={min(reachable)}, where the "
               f"per-subset floor exceeds 0.05; negative control ({CONTROL[rig]}) significant in "
               f"{kc} of {len(ctrl)} subsets, {kr} of {int(reach.sum())} ({kr / reach.sum():.1%}) "
               f"where p < 0.05 is reachable")

# ---- the reject option: sixteen rule-and-feature combinations, none usable (nb15)
rj = pd.read_csv(P / "reject_option.csv")
best = rj.loc[rj.auroc.idxmax()]
add("nb15", "XJTU-SY", 15, "rejecting naturally grown out-of-scope damage, best of 16 rules",
    round(float(best.auroc), 3), "a rule that reads nothing (chance)", 0.5,
    note=f"{best['rule']} on '{best.feature_set}'; the 16 span {rj.auroc.min():.3f}-{rj.auroc.max():.3f} "
         f"and {int((rj.auroc < 0.5).sum())} are below chance; at a 5 % in-scope budget the best rejects "
         f"{rj.loc[rj.auroc.idxmax(), 'rejected_at_95']:.1%} of out-of-scope snapshots")
sup = pd.read_csv(P / "reject_option_supervised.csv")
add("nb15", "XJTU-SY", 15, "in-scope vs out-of-scope with the labels handed over, units correct",
    f"{int(sup.correct.sum())} of {len(sup)}", "a model that always answers 'in scope'",
    f"{int((sup.scope == 'in scope').sum())} of {len(sup)}",
    note="the one unit of margin is the combined inner+outer fault, where 'out of scope' is least defensible")
wc = pd.read_csv(P / "reject_option_wrongcall.csv")
add("nb15", "XJTU-SY", 11, "confidence separating a wrong in-scope call from a right one",
    round(float(wc.loc[wc.pooling.str.startswith("80"), "auroc"].iloc[0]), 3), "chance", 0.5,
    note=f"the question that does have an answer; {wc.loc[~wc.pooling.str.startswith('80'), 'auroc'].iloc[0]:.3f} "
         f"if every snapshot is pooled, where three bearings supply 71 % of them")

# ---- the six claims that used to be quoted. Phase 2 of docs/submission-checklist.md had
# nb13, nb17, nb20, nb21 and nb25 write their intermediates out, so they are recomputed here
# like everything else and Limitation 6 is gone.
det = pd.read_csv(P / "xjtu_detection_own_vs_pooled.csv")
add("nb13", "XJTU-SY", len(det), "faults detected, per-unit healthy baseline",
    f"{int(det['own'].notna().sum())} of {len(det)}", "a pooled 14-unit fleet baseline",
    f"{int(det['foreign'].notna().sum())} of {len(det)}",
    note=f"median detection lead {det['own'].median():.1%} of life against "
         f"{det['foreign'].median():.1%}; pooling inflates the reference 2.91x at shaft rate")

nest = pd.read_csv(P / "xjtu_contract_nested.csv")
r20 = nest.iloc[(nest.silence - 0.20).abs().idxmin()]
add("nb17", "XJTU-SY", int(r20.votes), "spoken outer-race calls that are wrong, at a 20 % silence budget",
    f"{100 * r20.OR:.2f} %", "spoken inner-race calls under the same rule", f"{100 * r20.IR:.2f} %",
    note=f"the rule that hedges {r20.silence:.1%} of snapshots; pooled it reads {100 * r20.resid:.2f} %, "
         f"which is why the rate must be quoted per element")

iv = pd.read_csv(P / "xjtu_reliability_intervals.csv")
k = int((iv.realised == 0.0).sum())
add("nb25", "XJTU-SY", len(iv), "early outer-race calls that are an undetected inner race",
    f"{k} of {len(iv)} units", "80 % between-unit interval", "[0 %, 100 %]",
    note=f"{int((iv.realised == 1.0).sum())} units right through their whole early window and {k} wrong "
         f"through all of theirs; a per-snapshot 9.4 % is false precision over correlated minutes")

sch = pd.read_csv(P / "xjtu_reliability_schemes.csv").set_index("scheme")
key, perm = sch.loc["element x phase"], sch.loc["element x phase (permuted)"]
add("nb25", "XJTU-SY", 11, "calibration error when the phase key is replaced by noise",
    f"{key.MAE - perm.MAE:+.3f} in the favourable direction", "discrimination (AUROC) over the same swap",
    f"{key.AUROC:.3f} -> {perm.AUROC:.3f}",
    note=f"calibration alone cannot detect a dead key; a matched random key reaches "
         f"{sch.loc['element x random', 'AUROC']:.3f}")

lz = pd.read_csv(P / "xjtu_loadzone_channels.csv")
fr = lz[lz.name.str.startswith("LZ_frdev")].iloc[0]
add("nb20/22", "XJTU-SY", 11, "race attribution, LZ_frdev alone (a speed reading)",
    round(float(fr.lobo), 4), "exact permuted-label null, 95th pct",
    round(float(nb22.loc[(nb22.rig == "XJTU-SY") & (nb22.channel == "geometry") & ~nb22.is_true,
                         "acc"].quantile(0.95)), 4),
    note=f"clears the null on {int(fr.lobo_v)} of 11 units and cannot diagnose a race: both controls "
         f"are needed")

cr = pd.read_csv(P / "pu_loadzone_crossrig.csv")
decoy = cr[cr.name.str.contains("DECOY")].iloc[0]
true_fold = cr[cr.name.str.startswith("load-zone")].iloc[0]
add("nb21", "Paderborn", 11, "race attribution, decoy fold at 1.37x shaft rate",
    round(float(decoy.hro), 4), "the true shaft-rate fold", round(float(true_fold.hro), 4),
    note=f"deliberately incapable control, passes: {int(decoy.vhro)} of 11 units against "
         f"{int(true_fold.vhro)} of 11")

def bh(R):
    """Referee M2. Benjamini-Hochberg across the attribution tests, which are one family:
    the same question (does this channel carry the race?) asked of every channel, model and
    rig. Reported beside the raw p rather than instead of it -- a corrected value answers a
    different question, and both belong in the table. Rows outside the family get NaN."""
    fam = R.p.notna() & R.claim.str.startswith("race attribution")         & R.comparator.str.contains("null", na=False)
    q = R.loc[fam, "p"].sort_values()
    m = len(q)
    adj = (q.to_numpy() * m / np.arange(1, m + 1))[::-1]
    adj = np.minimum.accumulate(adj)[::-1]                  # step-up: enforce monotonicity
    R["q_bh"] = np.nan
    R.loc[q.index, "q_bh"] = np.round(np.minimum(adj, 1.0), 4)
    # The same family with the Paderborn geometry p replaced by its selection-aware value (§4.2).
    qs = R.loc[fam, "p_selection_aware"].fillna(R.loc[fam, "p"]).sort_values()
    adj = (qs.to_numpy() * m / np.arange(1, m + 1))[::-1]
    adj = np.minimum.accumulate(adj)[::-1]
    R["q_bh_selection_aware"] = np.nan
    R.loc[qs.index, "q_bh_selection_aware"] = np.round(np.minimum(adj, 1.0), 4)
    return R, m


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    R, m_fam = bh(pd.DataFrame(rows))
    R.to_csv(OUT / "registry.csv", index=False)
    print(f"wrote results/registry.csv: {len(R)} claims "
          f"({(R.source == 'recomputed').sum()} recomputed, {(R.source != 'recomputed').sum()} quoted)")
    print(R.groupby("verdict").size().to_string())
    f = R[R.q_bh.notna()].sort_values("p")
    print()
    print(f"Benjamini-Hochberg over the {m_fam} attribution tests (FDR 5 %):")
    print(f[["rig", "claim", "p", "q_bh", "verdict"]].to_string(index=False, max_colwidth=52))
    print(f"survive at FDR 5 %: {int((f.q_bh < 0.05).sum())} of {m_fam}; "
          f"clear at raw 0.05: {int((f.p < 0.05).sum())}")
    s_ = f[f.q_bh_selection_aware < 0.05]
    print(f"with the Paderborn configuration choice repeated under every labelling: {len(s_)} of {m_fam} "
          f"survive, at q = " + ", ".join(f"{v:.4f}" for v in sorted(s_.q_bh_selection_aware)))
