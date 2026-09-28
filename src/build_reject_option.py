# -*- coding: utf-8 -*-
"""Can stage 2 say "I do not recognise this"? -- notebook 15.

XJTU-SY has fifteen bearings, of which eleven failed by inner or outer race --
the two classes the attribution stage has -- and four failed some other way: two
cages, one compound fault, one combined inner+outer. Those four are the closest
thing this project has to naturally grown out-of-scope damage, and a reject
option is the standard remedy: score how unlike the training classes a snapshot
is, and refuse to answer above a threshold.

This caches the measurement notebook 15 made interactively, because the paper now
reports it (section 5.8) and every number the paper reports should be recomputable
from a script rather than read off a notebook.

Held at the notebook's settings, since the point is to reproduce it: 300-tree
forest, random_state 0; the per-unit alarm from the RMS/kurtosis exceedance of
notebook 07; features as change from each unit's own first twenty snapshots;
post-alarm rows only; 80 snapshots per unit when pooling, evenly spaced, so a
2 538-minute unit does not outvote a 42-minute one.

Four rules, each scoring HIGH for "less like the training classes": model
confidence (1 - max class probability), Mahalanobis distance to the nearer class
centre, mean 10-nearest-neighbour distance, one-class forest. Four feature sets:
the nine envelope order bands, plus RMS/kurtosis, plus the vertical
accelerometer, plus everything measured. Sixteen combinations.

The comparator, in the form section 4.4 demands of any such claim: a rule that
reads nothing scores 0.500 AUROC, and for the supervised version below, a model
that always answers "in scope" gets 11 of the 15 units right.

Usage:  python src/build_reject_option.py
Writes data/processed/reject_option.csv and reject_option_supervised.csv
"""
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
from sklearn.covariance import LedoitWolf                              # noqa: E402
from sklearn.ensemble import IsolationForest, RandomForestClassifier   # noqa: E402
from sklearn.metrics import roc_auc_score                              # noqa: E402
from sklearn.neighbors import NearestNeighbors                         # noqa: E402
from sklearn.preprocessing import StandardScaler                       # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed"
FAULT = {"Bearing1_1": "OR", "Bearing1_2": "OR", "Bearing1_3": "OR",
         "Bearing1_4": "cage", "Bearing1_5": "IR+OR", "Bearing2_1": "IR",
         "Bearing2_2": "OR", "Bearing2_3": "cage", "Bearing2_4": "OR",
         "Bearing2_5": "OR", "Bearing3_1": "OR", "Bearing3_2": "compound",
         "Bearing3_3": "IR", "Bearing3_4": "IR", "Bearing3_5": "OR"}
ENV = ["H_BPFO", "H_BPFI", "H_FR", "H_BPFO_h2", "H_BPFI_h2", "H_BSF", "H_FTF",
       "H_BPFI_sb1", "H_BPFO_sb1"]
DETECT = ["H_RMS", "H_Kurt"]
V = [c.replace("H_", "V_") for c in ENV]
SETS = {"envelope orders (nb14)": ENV,
        "+ RMS and kurtosis": ENV + DETECT,
        "+ the vertical channel": ENV + V,
        "everything measured": ENV + V + DETECT + ["V_RMS", "V_Kurt"]}
RULES = ["model confidence", "Mahalanobis to class centre",
         "10-nearest-neighbour distance", "one-class forest"]
RF = dict(n_estimators=300, random_state=0, n_jobs=-1)
K, W, CAP = 20, 10, 80


class Exceedance:
    """Robust per-feature z over the unit's own healthy baseline (notebook 07)."""

    def fit(self, A):
        self.med = np.median(A, axis=0)
        self.mad = np.maximum(1.4826 * np.median(np.abs(A - self.med), axis=0), 0.1)
        return self

    def score(self, A):
        return ((A - self.med) / self.mad).max(axis=1)


def fit_rules(X, y):
    """Notebook 14's two-class model, plus four ways of asking "is this one of mine?"."""
    clf = RandomForestClassifier(**RF).fit(X, y)
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    cls = sorted(np.unique(y))
    ctr = {c: (Z[y == c].mean(axis=0),
               np.linalg.pinv(LedoitWolf().fit(Z[y == c]).covariance_)) for c in cls}
    nn = {c: NearestNeighbors(n_neighbors=10).fit(Z[y == c]) for c in cls}
    oc = {c: IsolationForest(n_estimators=200, random_state=0).fit(Z[y == c]) for c in cls}

    def score(A):
        Za = sc.transform(A)
        s = {"model confidence": 1 - clf.predict_proba(A).max(axis=1)}
        s["Mahalanobis to class centre"] = np.column_stack(
            [np.sqrt(np.einsum("ij,jk,ik->i", Za - ctr[c][0], ctr[c][1], Za - ctr[c][0]))
             for c in cls]).min(axis=1)
        s["10-nearest-neighbour distance"] = np.column_stack(
            [nn[c].kneighbors(Za)[0].mean(axis=1) for c in cls]).min(axis=1)
        s["one-class forest"] = np.column_stack(
            [-oc[c].decision_function(Za) for c in cls]).min(axis=1)
        return clf.predict(A), s

    return score


def cap(a, n=CAP):
    """Even-spaced thinning, so the longest-lived unit does not outvote the shortest."""
    return a if len(a) <= n else a[np.linspace(0, len(a) - 1, n).astype(int)]


def post(G, alarm, bear, feats):
    """Each unit's post-alarm rows as change from its own first K snapshots."""
    out = {}
    for b in bear:
        A = G[b][feats].to_numpy()
        out[b] = (A - np.median(A[:K], axis=0))[alarm[b]:]
    return out


if __name__ == "__main__":
    ds = pd.read_csv(OUT / "xjtu_envelope_features.csv")
    bear = sorted(ds["Bearing"].unique())
    race = [b for b in bear if FAULT[b] in ("OR", "IR")]
    other = [b for b in bear if b not in race]
    G = {b: ds[ds["Bearing"] == b].sort_values("Snapshot").reset_index(drop=True) for b in bear}

    alarm = {}
    for b in bear:
        A = G[b][DETECT].to_numpy()
        z = Exceedance().fit(A[:K]).score(A)
        c = np.convolve((z[K:] > np.quantile(z[:K], 0.99)).astype(float),
                        np.ones(W) / W, "valid")
        hit = np.flatnonzero(c >= 0.5)
        alarm[b] = None if hit.size == 0 else int(hit[0]) + K
    print(f"{len(race)} race units in scope, {len(other)} out of scope: "
          + ", ".join(f"{b.replace('Bearing', '')} ({FAULT[b]})" for b in other), flush=True)

    rows = []
    for name, feats in SETS.items():
        P = post(G, alarm, bear, feats)
        ins = {r: [] for r in RULES}
        oos = {r: {b: [] for b in other} for r in RULES}
        for held in race:
            tr = [o for o in race if o != held]
            score = fit_rules(np.vstack([P[o] for o in tr]),
                              np.concatenate([[FAULT[o]] * len(P[o]) for o in tr]))
            _, s = score(P[held])
            for r in RULES:
                ins[r].append(cap(s[r]))
            for b in other:
                _, s2 = score(P[b])
                for r in RULES:
                    oos[r][b].append(s2[r])
        for r in RULES:
            i = np.concatenate(ins[r])
            o = np.concatenate([cap(np.mean(oos[r][b], axis=0)) for b in other])
            truth = np.r_[np.zeros(len(i)), np.ones(len(o))]
            rows.append(dict(feature_set=name, rule=r,
                             auroc=float(roc_auc_score(truth, np.r_[i, o])),
                             rejected_at_95=float((o > np.quantile(i, 0.95)).mean()),
                             n_in=len(i), n_out=len(o)))
        print(f"  {name}: "
              + ", ".join(f"{d['rule'].split()[0]} {d['auroc']:.3f}" for d in rows[-4:]),
              flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "reject_option.csv", index=False)

    # The other question the same score can answer: not "am I outside my classes?" but
    # "am I about to get this wrong inside them?" -- scored on in-scope snapshots only,
    # confidence against whether the call was wrong.
    # Scored twice on purpose. Notebook 15 pooled every in-scope snapshot, which lets the
    # three longest-lived bearings supply most of the rows -- the composition problem this
    # project reports elsewhere. The thinned version is the one consistent with that rule,
    # so both are cached and the paper leads with the thinned one.
    P = post(G, alarm, bear, ENV)
    cf, wr = {True: [], False: []}, {True: [], False: []}
    for held in race:
        tr = [o for o in race if o != held]
        score = fit_rules(np.vstack([P[o] for o in tr]),
                          np.concatenate([[FAULT[o]] * len(P[o]) for o in tr]))
        pred, s = score(P[held])
        for thin in (True, False):
            f = cap if thin else (lambda a: a)
            cf[thin].append(f(s["model confidence"]))
            wr[thin].append(f((pred != FAULT[held]).astype(float)))
    wrong_auroc = {thin: float(roc_auc_score(np.concatenate(wr[thin]), np.concatenate(cf[thin])))
                   for thin in (True, False)}

    # The generous version: hand the labels over and ask in-scope vs out-of-scope directly.
    PC = {b: cap(v) for b, v in post(G, alarm, bear, ENV).items()}
    scope = {b: ("in scope" if b in race else "out of scope") for b in bear}
    sup = []
    for held in bear:
        tr = [o for o in bear if o != held]
        m = RandomForestClassifier(**RF, class_weight="balanced").fit(
            np.vstack([PC[o] for o in tr]),
            np.concatenate([[scope[o]] * len(PC[o]) for o in tr]))
        frac = float((m.predict(PC[held]) == "out of scope").mean())
        sup.append(dict(bearing=held, fault=FAULT[held], scope=scope[held], out_frac=frac,
                        call="out of scope" if frac > 0.5 else "in scope"))
    s = pd.DataFrame(sup)
    s["correct"] = s["call"] == s["scope"]
    s.to_csv(OUT / "reject_option_supervised.csv", index=False)

    print()
    print(d.pivot(index="rule", columns="feature_set", values="auroc").round(3).to_string())
    print(f"\n16 combinations span {d.auroc.min():.3f} to {d.auroc.max():.3f}; the best is "
          f"{d.loc[d.auroc.idxmax(), 'rule']} on {d.loc[d.auroc.idxmax(), 'feature_set']}")
    base = d[d.feature_set == "envelope orders (nb14)"]
    print("at a 5 % in-scope rejection budget: "
          + ", ".join(f"{r.rule} {r.rejected_at_95:.1%}" for _, r in base.iterrows()))
    print(f"below chance: {int((d.auroc < 0.5).sum())} of {len(d)} combinations")
    oos_conf = d.loc[(d.feature_set == "envelope orders (nb14)")
                     & (d.rule == "model confidence"), "auroc"].iloc[0]
    print(f"confidence separating a WRONG in-scope call from a right one: "
          f"{wrong_auroc[True]:.3f} AUROC thinned, {wrong_auroc[False]:.3f} pooled over every "
          f"snapshot (nb15's figure), against {oos_conf:.3f} for out-of-scope vs in-scope")
    pd.DataFrame([dict(quantity="confidence: wrong in-scope call vs right one",
                       pooling="80 snapshots per unit", auroc=wrong_auroc[True],
                       comparator="chance", comparator_value=0.5),
                  dict(quantity="confidence: wrong in-scope call vs right one",
                       pooling="every snapshot (nb15)", auroc=wrong_auroc[False],
                       comparator="chance", comparator_value=0.5)]).to_csv(
        OUT / "reject_option_wrongcall.csv", index=False)
    print(f"supervised, labels handed over: {int(s.correct.sum())} of {len(s)} units correct, "
          f"against {int((s.scope == 'in scope').sum())} of {len(s)} for always-in-scope")
    print("out-of-scope units caught: "
          + str(s[(s.scope == "out of scope") & s.correct].bearing.tolist()))
    print("wrote reject_option.csv, reject_option_supervised.csv")
