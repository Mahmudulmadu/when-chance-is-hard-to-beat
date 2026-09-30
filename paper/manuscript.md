# When chance is hard to beat: exact permutation nulls and degenerate baselines for bearing fault diagnosis and remaining useful life

**Md. Mahmudul Hasan**  
Independent Researcher, Dhaka, Bangladesh · mahmudulmadu@gmail.com

---

## Abstract

Bearing fault diagnosis and remaining-useful-life (RUL) prediction are evaluated largely on public datasets of ten to twenty bearings. Leakage from window-level splitting is now well documented, but less is known about the reference points needed to interpret results from such small datasets. For diagnosis, we compute exact permutation nulls by enumerating all 165 (XJTU-SY) and 462 (Paderborn) class-preserving bearing labellings. With eleven bearings, meaningless labellings reach 69.1–76.4 % accuracy at the 95th percentile, depending on the model and its input. Six of our own eleven attribution results fall inside their null; of the five that clear it, only three survive a Benjamini–Hochberg correction. A standard deep baseline stays inside its null on both rigs (*p* = 0.79, 0.72) yet identifies individual bearings at 99.5 % and 95.8 %: it learns the bearing, not the fault. Study-size analysis shows that detecting a real effect reliably takes nine bearings on Paderborn and at most seven on XJTU-SY; below seven, the exact test cannot return *p* < 0.05.

For RUL, we show that on run-to-failure data remaining-life error and total-lifetime error are exactly the same quantity. After the alarm, a constant "no life left" prediction beats every trained estimator on mean absolute error, with our state-based estimator 7.1 times worse in the final fifth of life. RUL error magnitude alone therefore cannot support a remaining-life claim; what can be defended is a lower bound. We distil five reporting rules and publish a registry checking all 31 of our claims against their comparators.

**Keywords:** rolling-element bearings; permutation test; study size; remaining useful life; evaluation methodology; negative controls; condition monitoring

**Highlights**

- On eleven bearings, meaningless labellings reach 69-76 % accuracy (95th percentile)
- The null belongs to the model and its input, moving 7 points on one rig's labellings
- On run-to-failure data an RUL error is a lifetime error, to machine epsilon
- A constant "no life left" has lower post-alarm MAE than every trained estimator
- Six of our own eleven attribution results fall inside their own null, and we say so

---

## 1. Introduction

Data-driven bearing diagnosis is a mature field, and much of its evaluation relies on a small number of public datasets, each containing on the order of ten to twenty physical bearings. Accuracies above 95 % are frequently reported [23–29], and remaining-useful-life (RUL) predictions are reported as errors in units of time or cycles [36, 38, 39].

The protocols behind these numbers have received sustained attention. Splitting windows rather than bearings inflates results, because segments from one recording appear in both training and test sets. Of 41 reviewed studies using the CWRU dataset published between 2008 and 2020, 40 used designs susceptible to this similarity bias; of 55 bearing-diagnosis papers published between 2020 and 2024, only six employed rigorous data splitting, as summarised by [3]. The problem is not peculiar to this field: a survey of machine-learning-based science found leakage in 17 fields, collectively affecting 294 papers [31]. Bearing-wise partitioning is now widely regarded as a minimum standard [3, 4, 5, 6], and Hendriks et al. [4] showed that the conventional CWRU protocol lets a convolutional network learn features tied to *specific bearings* rather than to faults.

This work assumes a clean protocol with bearing-level splits and asks a further question: what reference point should the resulting accuracy or error be compared against? We consider two, and in the bearing literature reviewed here neither is standard practice.

The first is **chance at the actual study size**. With eleven bearings and a two-class problem, a classifier that has learned nothing about bearing faults can still score well, because the eleven units differ in ways that can align with an arbitrary grouping of them — the operating point they ran at, the severity of their damage, or simply their individual identity. How well it can score is an empirical question with an exact answer: enumerate every class-preserving reassignment of the labels and re-run the identical protocol.

The second is a **degenerate baseline for quantities reported as errors**. On run-to-failure data the elapsed time is known exactly, so predicting remaining life is arithmetically equivalent to predicting total lifetime. An RUL error is therefore a statement about how well a bearing's total life was estimated, and it should be compared with estimators that predict lifetime without reading any sensor.

We measure both on two public rigs and apply them to our own diagnosis and prognosis pipeline, reporting the results that fail these tests alongside those that pass.

**Contributions.**

1. **An exact permutation null for bearing-held-out classification**, computed over every class-preserving labelling on two rigs. On eleven units its 95th percentile lies between 69.1 % and 76.4 %, depending on the model and its input, and an accuracy that does not exceed the threshold for its own model and input does not constitute evidence of fault-specific attribution (§5.1).
2. **Evidence that the null belongs to the model and its input, not to the dataset.** Scored on identical labellings, the null's 95th percentile spans 7.3 points across five model-and-input combinations on one rig — highest for a deep network on raw waveforms, lowest for a small network on the order spectrum — so a threshold measured for one model should not be assumed to hold for another (§5.5).
3. **A study-size curve** showing that a real effect is detected in at least 80 % of subsets from nine units on Paderborn and from seven on XJTU-SY, with an interval on each rate and negative controls detected in 2.6 % and 1.6 % of subsets. The XJTU-SY figure is an upper bound: with six or fewer units the exact test cannot return *p* < 0.05, so no smaller count can be tested (§5.4).
4. **The identity RUL ≡ lifetime prediction**, verified to machine epsilon, and the comparison it calls for: after the alarm, a degenerate constant ("no life left") has a lower median per-unit mean absolute error than every trained estimator we built (§5.6).
5. **Five reporting rules and a public claim registry** that checks all 31 of our own claims against their comparators. Six of our eleven attribution results fall inside their null, and two more clear it only before multiplicity correction (§6, §5.2).

---

## 2. Related work

**Evaluation protocol and leakage.** The limitations of window-level splitting on CWRU and Paderborn are well documented [3, 4, 5, 13], and bearing-wise partitioning is widely recommended as the correction. Earlier, Smith and Randall [19] applied three established diagnostic techniques to the entire CWRU data set and made recommendations on how it should be used as a benchmark. Hendriks et al. [4] showed that the conventional domain-shift framework on CWRU places the same physical bearings in training and test, and that networks trained that way learn bearing-specific features; they proposed splits with independent sets of bearings. Recent benchmarks fix source-target scenarios and enforce recording-level separation [6, 14], and Vieira et al. [3] find that the number of distinct bearings in the training set is a decisive factor for robust performance. **We assume that correction rather than repeating it**: every result below uses bearing-held-out evaluation, and we ask what the resulting number must exceed.

**Shortcut learning.** A model can solve a benchmark through a correlate of the label rather than the intended signal. This is a general phenomenon [8], described across domains as shortcut learning, spurious correlation or the Clever Hans effect [15], and Hendriks et al. [4] identified its bearing-identity form in this field. Our addition is to measure it: we report identity recognition as a number beside accuracy, and show that the problem persists under bearing-held-out evaluation, the remedy proposed in [4]. A standard 1D convolutional network evaluated that way stays inside its null on both rigs, while the same architecture distinguishes individual bearings at 95.8–99.5 % (§5.5).

**Bearing diagnostics and the features used here.** The envelope-spectrum analysis underlying our geometry features is standard practice [20], as are spectral kurtosis and the kurtogram for selecting a demodulation band [21, 22]; our non-geometry channel is built from the latter. The deep-learning literature these methods are compared against is large and reports high accuracy on the same public data [23–30].

**Chance level at small sample sizes.** The problem we address has been studied in neuroscience, where small samples are common. Combrisson and Jerbi [32] showed that the theoretical chance level of a classification problem is not the empirical one when samples are few, and published look-up tables of the accuracy a classifier must reach before it beats chance at a given sample size; Varoquaux [33] showed that cross-validation error bars at typical neuroimaging sample sizes are far wider than usually reported; and Button et al. [34] set out the consequences of low statistical power for a literature's reliability. We import that reasoning into condition monitoring, where the unit of analysis is the bearing and the counts are smaller still.

**Permutation testing.** Label-permutation tests of classifier performance are well established. Ojala and Garriga [7] formalise a test in which the labels are permuted to ask whether a classifier has found real class structure, and Combrisson and Jerbi [32] use permutation tests to assign significance to decoding accuracy in small-sample brain-signal studies. In the condition-monitoring literature reviewed here, we did not find the test applied at *bearing level* on these benchmarks, where class counts are small enough to enumerate every labelling (C(11,3) = 165, C(11,5) = 462), so that *p*-values are exact rather than Monte-Carlo estimates.

**RUL evaluation and uncertainty.** Prognostics has its own methodological reviews: of the approaches developed on the C-MAPSS benchmark [16], of prognostic performance metrics [37], and of the full pipeline from data acquisition to RUL prediction [36]. Deep models have been applied to RUL prediction on the C-MAPSS turbofan data [38] and on bearing run-to-failure data [39], and the FEMTO/PRONOSTIA platform [35] provides a further bearing benchmark. Health-indicator construction, on which our alarm and floor both depend, has been studied in its own right [43], and the reliability literature has examined how prognostic outputs feed maintenance decisions [44, 45, 46]. Uncertainty in RUL prediction has been reviewed recently [42], and methods for it include conformal prediction [40] and uncertainty-aware prognostic models [41]. Some recent bearing-RUL studies report calibrated or conformal intervals rather than point estimates [9, 10]; Yang et al. [10] also report coverage per bearing and per operating regime rather than pooled, and find conditional failures that a pooled figure hides. Simple baselines do appear: Kumar et al. [17] report a linear-trend baseline that beats both an LSTM and an MLP on an asymmetric score on an XJTU-SY bearing. In the literature reviewed here, however, we did not find the *degenerate* baselines used in this paper — estimators that read no sensor — nor a statement of the identity between remaining-life and total-lifetime prediction on run-to-failure data.

**Fleet versus per-unit references.** Fleet-based anomaly detection compares a machine with similar machines operating alongside it, so that no historical data set is needed for each machine [11]. We report a boundary condition for such references on run-to-failure data: a reference pooled over the other XJTU-SY bearings detects 8 of the 15 bearings, and only one of the three inner-race units, where a reference built from each unit's own healthy period detects all 15.

**Deep baselines.** As a deep baseline we use the wide-first-kernel 1D convolutional network (WDCNN) of Zhang et al. [12], which was designed for raw vibration signals. We keep its architecture unmodified and score it identically to the classical baseline, under the same null.

**Open-set recognition.** Open-set methods for rejecting unknown fault types have reported F1 scores in the high nineties on seeded-fault data [18]. Our negative result concerns *naturally grown* damage: across sixteen rule-and-feature combinations and two rigs, no rejection rule exceeded 0.660 AUROC, because on these features the out-of-scope units fall within the distribution of the trained classes (§5.8).

## 3. Data

We use two public datasets, chosen because they differ in the two ways that matter for this study: how the damage arose, and whether each unit has a healthy history.

**Paderborn (KAt) bearing data centre** [1]. The data set covers 6203-type bearings on an electromechanical drive train, sampled at 64 kHz, with vibration, two motor phase currents, force, torque, speed and temperature. We use 25 bearings: **6 healthy, 8 with artificial damage** (EDM, electric engraver, drilling) and **11 with real damage** grown in accelerated lifetime tests (5 outer-race, 6 inner-race). Each bearing has twenty recordings of 4 s at each operating point. Damage is documented per unit — mode, symptom, component, extent, arrangement, characteristic and cause.

**XJTU-SY run-to-failure data** [2]. Fifteen LDK UER204 bearings were run from new to failure under three operating conditions (35 Hz/12 kN, 37.5 Hz/11 kN, 40 Hz/10 kN), sampled at 25.6 kHz in 1.28 s snapshots once per minute, 9 216 snapshots in total. Lifetimes span **42 to 2 538 minutes, a factor of 60**. Eleven units failed at a single race (8 outer, 3 inner) and are the attribution set; the remaining four (two cage, one compound, one combined inner + outer) are used as out-of-scope units.

**Label provenance.** Every label used here was checked against its source document rather than taken from code: all 15 XJTU-SY fault labels and file counts match Table 2 of the dataset introduction, and all 25 Paderborn units match their per-unit damage sheets on class and on artificial-versus-real. Two units labelled inner-race (KI04, KI14) carry a documented secondary outer-ring indentation; we retain the dataset's convention, which labels multiple damage by its largest component.

**Why both rigs.** Paderborn supplies seeded and real damage but no healthy history for a damaged unit. XJTU-SY supplies a healthy-to-failed trajectory per unit but only three inner-race units. Neither alone can support the claims below, and three of our results changed or narrowed when moved to the second rig.

---

## 4. Method

### 4.1 Protocol

All classification is **bearing-held-out**: a model is trained on all units but one and scored on every snapshot or recording of the held-out unit; the reported accuracy is the mean over units of the per-unit accuracy. Reporting per-unit means rather than per-snapshot means matters here, because lifetimes vary 60-fold and three XJTU-SY bearings supply 71 % of all snapshots.

### 4.2 Exact permutation null

For a class-preserving labelling *L* of the units, let *a*(*L*) be the bearing-held-out accuracy under the identical protocol. We enumerate **every** *L*: 165 on XJTU-SY (8 outer / 3 inner) and 462 on Paderborn (5 outer / 6 inner). The reported *p* is (#{*a*(*L*) ≥ *a*(true)}) / *N*, exact by construction, with floors of 1/165 = 0.0061 and 1/462 = 0.0022. Because the enumeration is complete, this estimator and the sampled form (1 + #{null ≥ true}) / (1 + *N*) that §4.5 uses for the study-size curve coincide exactly — the true labelling is the +1 in the numerator. The two differ only when the enumeration is partial, and no result reported here is partial; every *p* in Table 1 is over the full 165 or 462. The null's 95th percentile is the threshold an accuracy must exceed before it counts as evidence that the channel carries the fault class.

**Multiplicity.** Table 1 reports eleven tests of a single question — does this channel, on this model and this rig, carry the race? — so they are one family, and we report Benjamini–Hochberg *q*-values across all eleven beside the raw *p* rather than instead of it, because the two answer different questions. Three results survive at FDR 5 %, and two of the five that clear at α = 0.05 do not. The more important consequence is that **the design's floor bounds what any correction can establish here.** A Bonferroni threshold for eleven tests is α = 0.0045, which neither XJTU-SY result can reach — not because the effects are weak but because 1/165 = 0.0061 is the smallest value the exact test can return on that rig. Eleven units admit only 165 labellings, and no correction can add resolution that the design does not have. We therefore treat the floor as the binding limit on this study and report the correction beside it.

Seeds are **paired**: model initialisation and data sampling depend on the fold, not on the labelling, so the true labelling and every permuted one start from the same state.

### 4.3 Deliberately incapable controls

A permutation null asks whether the class structure is real. It does not ask whether a feature can carry what its name suggests. For that we run features constructed to be incapable: a shaft-speed reading in place of a diagnostic feature; a synchronous fold at 1.37 × shaft rate, which cannot carry load-zone modulation; and, for the deep baseline, crops whose samples are shuffled in time, destroying the waveform while preserving every amplitude statistic. **Both tests are required** — §5.3 gives a case where each catches what the other misses.

### 4.4 Degenerate baselines for errors

For any quantity reported in minutes we score, alongside the model, estimators that read no sensor: *zero* (no life remaining), *survived* (as long again as it has already run), the training fleet's median lifetime minus elapsed time, the same restricted to the unit's operating point, and a forest given only elapsed time and shaft rate.

### 4.5 Study-size curve

For each unit count *n* we draw **30 subsets** of *n* bearings at random, or take every subset where fewer than 30 exist (11 at *n* = 10, one at *n* = 11), and discard any subset that contains a single class. In each we score the true labelling and compare it with a null permuted **inside that same subset**, so the reference reflects the size being tested. Within a subset the null is enumerated **exhaustively** wherever it has at most 40 non-true labellings and sampled at 40 above that; *p* = (1 + #{null ≥ true}) / (1 + N), which at full enumeration is the exact form of §4.2. A subset detects the effect when *p* < 0.05, and the study-size threshold is the smallest *n* at which at least 80 % of subsets detect it. Detection rates are reported with a Clopper–Pearson 95 % interval, and that interval is **optimistic**: subsets of the same eleven bearings overlap heavily, and at *n* = 9 each bearing appears in as many as 28 of the 30 subsets.

Two properties of this design bound what it can show, and both are easy to mistake for findings.

**The per-subset *p* has a floor that moves with *n*.** It is 1/41 = 0.024 from *n* = 8 upward, but at *n* ≤ 6 the entire null inside a subset is smaller than twenty labellings and the smallest attainable *p* exceeds 0.05 — 0.067 at *n* = 6, 0.10 at *n* = 5, 0.25 at *n* = 4. **No subset of six or fewer units can reach significance at any effect size**, so a zero detection rate there is a statement about the design and not about the channel. At *n* = 4 on Paderborn the true accuracy already exceeds its own null band, by 2.0 points, and not one of 29 subsets is significant.

**The null must exclude the true labelling.** An earlier version of this analysis sampled without excluding it; when the true assignment is drawn into its own null, *p* ≥ 1/N and the subset can never reach significance. The bias scales as 1/C(n,k) and therefore penalises small minority classes hardest — 71 % contamination on XJTU-SY at *n* = 8 against 29 % on Paderborn — and it reversed the apparent ordering of the two rigs.

### 4.6 Identity test (fingerprinting)

Any model that can recognise *which unit* a signal came from can use identity as a proxy for class when units per class are few. We therefore score each learned model on a task with no fault content: distinguishing individual bearings, held out by recording or snapshot block, against a chance rate of 1/units.

### 4.7 Calibration with discrimination and resolution

Reliability statements are scored on three axes, never calibration alone: calibration error, discrimination (AUROC) and resolution. §5.7 shows why.

### 4.8 Reproducibility

Every result is produced by a cached build script and re-derived from the cache; all 31 claims in the registry are recomputed at load time, none transcribed. CPU PyTorch is **not bit-stable across thread counts** — the accuracy of one true labelling moved from 35.9 % to 33.2 % between 3 and 5 threads — so all numbers compared against a null were produced at a single thread count, and every quantity computed twice is asserted to agree exactly.

---

## 5. Results

### 5.1 The null band on eleven units

On eleven units, the 95th percentile of bearing-held-out accuracy over labellings with no physical meaning is **69.1 % to 76.4 %**, depending on the rig, the model and the input (Fig. 1). For the forests — the hand-built feature sets and the forest on 576 order bins — the band is 69.6–73.9 %; for WDCNN on raw waveforms, 76.3–76.4 %; for the small spectrum-CNN, 69.1–73.5 %.

The practical consequence is that **a bearing-held-out accuracy anywhere in that range cannot be read as evidence of diagnosis until its own null has been computed.** The same correction has been urged in neuroscience, where sample-size-specific thresholds have been proposed in place of theoretical chance levels [32].

### 5.2 Our own results against their nulls

Of the eleven attribution numbers we report, **five clear their null and six do not** (Fig. 1, Table 1). Every *p* below is exact over the full enumeration — 165 assignments on XJTU-SY, 462 on Paderborn:

| Channel | Rig | Accuracy | Null 95th | *p* | BH *q* | Verdict |
|---|---|---|---|---|---|---|
| Geometry (envelope order bands) | XJTU-SY | 94.1 % | 71.9 % | 0.0061 | 0.022 | clears |
| Load-zone (shaft-synchronous fold) | XJTU-SY | 89.7 % | 73.9 % | 0.0061 | 0.022 | clears |
| Geometry | Paderborn | 85.5 % | 69.6 % | 0.0043 | 0.022 | clears |
| Forest on 576 order-spectrum bins | Paderborn | 78.6 % | 71.4 % | 0.0195 | 0.054 | clears at α, not at FDR 5 % |
| Spectrum-CNN on 576 order bins | Paderborn | 70.9 % | 69.1 % | 0.0368 | 0.081 | clears at α, not at FDR 5 % |
| Load-zone | Paderborn | 62.7 % | 71.4 % | 0.147 | 0.270 | **inside** |
| Non-geometry (resonance/spectral shape) | XJTU-SY | 62.3 % | 71.1 % | 0.442 | 0.608 | **inside** |
| Spectrum-CNN on 576 order bins | XJTU-SY | 60.4 % | 73.5 % | 0.376 | 0.591 | **inside** |
| Forest on 576 order bins | XJTU-SY | 58.6 % | 71.6 % | 0.697 | 0.788 | **inside** |
| WDCNN on raw waveform | XJTU-SY | 48.9 % | 76.3 % | 0.788 | 0.788 | **inside** |
| WDCNN on raw waveform | Paderborn | 33.2 % | 76.4 % | 0.723 | 0.788 | **inside** |
| *Always outer race* (reads no signal) | XJTU-SY | 72.7 % | — | — | — | label-only baseline |

The last row is not a result but the comparator the table would otherwise omit: on XJTU-SY the majority class alone scores **72.7 %**, above four of the six XJTU-SY results and within that rig's 71.1–76.3 % range of null thresholds. An accuracy in this table has to be read against it as well as against its own null.

The load-zone channel is the instructive case. It clears its null at the test's floor on XJTU-SY (*p* = 0.0061) and sits inside it on Paderborn (*p* = 0.147). Without the test, the Paderborn result would naturally be described as "underperforms geometry"; with it, the accurate description is **"no demonstrable signal"**. The two statements support very different conclusions.

Under Benjamini–Hochberg across all eleven tests (§4.2), **three of the five results that clear at α = 0.05 survive at FDR 5 %**: the geometry channel on both rigs and the load-zone channel on XJTU-SY, all at *q* = 0.022. The two Paderborn order-spectrum results do not (*q* = 0.054 and 0.081). We report them as clearing their own null and failing the correction, because both statements are true.

The two XJTU-SY channels that clear sit at the test's 1/165 floor, so the test is saturated there: it rejects the null for both but cannot rank the two effects against each other. The two rows nearest the 0.05 line — the Paderborn spectrum models at 0.0195 and 0.0368 — are 9 and 17 labellings out of 462 respectively; both were sampled at *N* = 100 in an earlier version of this work, where they read 0.0198 and 0.0396, and we completed the enumeration so that no *p* in the table is a sampled estimate.

### 5.3 A null is necessary, not sufficient

Two results show that the permutation test cannot be the only control.

`LZ_frdev` is a feature that records how stably a frequency search converged — a pure shaft-speed reading, incapable of distinguishing an inner from an outer race. It scores **81.3 %**, above every null threshold in Table 1. It was caught only by asking whether the quantity could physically carry what its name implied: its accuracy comes from the estimator's arbitrary 0.04 Hz search grid, on which all three inner-race units returned the same grid point.

Conversely, a synchronous fold at **1.37 × shaft rate**, built to be incapable, scores 44.1 % against 65.5 % for the same fold at the true shaft rate. The control behaves as it should, indicating that the true fold measures shaft-locked content, even on the rig where that channel fails its null.

**A permutation null answers "is the class structure real?"; an incapable control answers "can this feature carry what it claims?" Neither subsumes the other.**

### 5.4 How many units a claim needs

The band narrows as units are added, so power comes as much from the null tightening as from accuracy rising — something a plain learning curve cannot show. Each unit count is scored over up to 30 subsets (§4.5). On Paderborn, the geometry channel's effect, which clears its null on the full set, is detected in **12 of 30 subsets at *n* = 7** (40 %, 95 % CI [23 %, 59 %]), **21 of 30 at *n* = 8** (70 %, [51 %, 85 %]) and **28 of 30 at *n* = 9** (93 %, [78 %, 99 %]), where the mean margin over the null reaches **+13.3 points**; every subset is detected at *n* = 10 and *n* = 11 (Fig. 2).

On XJTU-SY, where the effect is larger, detection is **complete from *n* = 7** — 30 of 30, and 100 % at every larger size. That figure has to be read with the design's floor beside it. Below *n* = 7 the exact null inside a subset cannot return *p* < 0.05 at all (§4.5), and XJTU-SY's detection rate jumps from 0 % at *n* = 6 to 100 % at *n* = 7, the step at which the smallest attainable *p* first falls below 0.05. **Seven is therefore an upper bound on what that rig needs, not a measurement of it**: at *n* = 5 and *n* = 6 the mean margin over the null is already **+20.1 and +22.3 points** with not one subset significant. Paderborn's nine is a measurement, because its floor becomes reachable at *n* = 7 and the detection rate then climbs over the next two unit counts rather than saturating at once.

The thresholds are therefore **nine units on Paderborn and at most seven on XJTU-SY**, and they should not be averaged into one number: the first is a measurement and the second a bound. The eleven units both datasets provide clear the bar for the strongest channel on each rig — by two units on Paderborn, and by at least four on XJTU-SY, where the bar itself is an upper bound. The negative controls behave as expected: Paderborn's load-zone channel is significant in **5 of 190** subsets across all unit counts and XJTU-SY's non-geometry channel in **3 of 192** — 2.6 % and 1.6 %, below the 5 % expected by chance from a correctly calibrated test — and both sit below their own nulls at nearly every size. XJTU-SY's load-zone channel, which does clear its null on that rig, needs **eight**.

Two limitations apply to these thresholds. The subsets at a given size are drawn from the same eleven bearings and are far from independent — at *n* = 9 each bearing appears in as many as 28 of the 30 subsets — so the intervals above are optimistic, and the threshold is itself estimated from eleven units. Separately, "how many units until an effect is *detectable*" is a harder question than "how many until accuracy *saturates*", which we measured at about six — the two should not be conflated.

### 5.5 A deep baseline under the same null

WDCNN [12] on decimated raw vibration, evaluated bearing-held-out with a fixed training budget, no early stopping and paired seeds, reaches **48.9 %** (XJTU-SY) and **33.2 %** (Paderborn), against 94.1 % and 85.5 % for the forest on geometry features with identical folds, and sits **inside its own null on both rigs** (*p* = 0.788, 0.723; Fig. 3).

The failure is not due to undertraining. Quadrupling the budget leaves the same units wrong. Within each fold it reaches **97.5 %** (XJTU-SY) and **99.4 %** (Paderborn) on a held-out sample of its own *training* units — at least 90 % in every one of 22 folds — while scoring 48.9 % and 33.2 % on the unit it did not see.

**Its calls follow the bearing and its operating point rather than the fault.** That a network *can* learn bearing-specific features was shown by Hendriks et al. [4], whose remedy was splits with independent bearings. We adopt that remedy — every number here is bearing-held-out — and the behaviour persists in measurable form: the same network at the same budget distinguishes individual bearings at **99.5 %** (chance 9.1 %) and **95.8 %** (chance 16.7 %). Shuffling every sample in time — destroying the waveform while keeping the amplitudes — does not reduce race accuracy (61.4 % against 48.9 % on XJTU-SY, 32.3 % against 33.2 % on Paderborn). On XJTU-SY its held-out calls match a label-only rule ("the majority race among other units at the same speed and load") on **11 of 11 units**, and across all 165 labellings its accuracy is correlated with operating-point alignment (Spearman +0.26, *p* = 9 × 10⁻⁴) where the forest's is not (−0.06, *p* = 0.44; Fig. 4).

Two findings extend beyond this one architecture, as the second network shows. First, **the null belongs to the model and its input**, not to the dataset. On the identical 462 Paderborn labellings the 95th percentile runs from 69.1 % for the small spectrum-CNN through 69.6 % and 71.4 % for the forests to **76.4 %** for WDCNN on the raw waveform — a spread of 7.3 points; on XJTU-SY's 165 it runs 71.1–76.3 %. WDCNN's null is the widest on both rigs, plausibly because higher capacity on the least-processed input finds more ways to fit an arbitrary grouping of eleven units. The ordering does not simply separate deep from classical models, however: the **narrowest** null in the study also belongs to a network, the spectrum-CNN on Paderborn. A threshold borrowed from another model can therefore be wrong by up to seven points, in either direction. Second, in the network's Paderborn null one labelling other than the true race labels scores **95.0 %** — so a bearing-held-out accuracy of that size, from a standard deep model on eleven units, is not by itself evidence of diagnosis.

Finally, at this study size the choice of **input** mattered more than the choice of architecture. Given the envelope spectrum on a shaft-order axis instead of the raw waveform, both a small CNN (70.9 %, *p* = 0.0368) and a forest on the same 576 bins (78.6 %, *p* = 0.0195) clear their nulls on Paderborn. On XJTU-SY both fail their own exact nulls — 60.4 % at *p* = 0.376 and 58.6 % at *p* = 0.697, and per-unit calibration moves them only to 62.4 % and 57.9 % — so the failure on that rig is measured against each model's own null rather than inferred from other models' thresholds. Only the physics-selected features clear the null on **both** rigs.

### 5.6 Remaining life: an identity and a constant baseline

**RUL prediction is total-lifetime prediction.** Because elapsed time is known exactly, RUL = (L − 1) − t, and any RUL estimate implies a lifetime estimate with the same error. Across 73 728 estimator-snapshots the largest discrepancy between the two errors is **9.1 × 10⁻¹³ minutes**, two units in the last place of float64 at the longest lifetime. This is a property of the benchmark, not of a model, and it means a reported RUL error should be benchmarked against estimators that predict lifetime without reading a sensor.

Against those, even our best state-based estimator has a higher post-alarm median per-unit MAE than the constant (Table 2):

| Estimator | MAE (min) | Error ÷ life remaining |
|---|---|---|
| **Zero (no life left)** | **51** | 1.00 |
| Two-channel state, fitted post-alarm | 57 | **0.79** |
| Two-channel state | 78 | 1.26 |
| Learned clock (elapsed + shaft rate) | 89 | 1.74 |
| Fleet-median lifetime at this condition | 82 | 1.00 |

Predicting *zero* gives the lowest MAE of the five estimators, and in the **final fifth of life** the state-based model's error is **7.1 times** the constant's (113 min against 16; Fig. 5), the period in which a maintenance decision is most likely to be taken. Only the post-alarm-fitted model does better than the constant on the scale-free measure (0.79 against 1.00), and then only on 8 of 15 units.

We do not conclude that remaining life is unpredictable. We conclude that **post-alarm MAE on this benchmark cannot support a remaining-life claim**, because a constant that reads no sensor achieves the lowest MAE, and that what can be supported is a **bound**. For the median unit, a lower bound taken from the 10 % quantile of the forest's per-tree predictions does not overstate remaining life on 91 % of its post-alarm snapshots, against 19 % for the point estimate, at the cost of understating it by 63 %. Its permuted-lifetime control falls to 17.5 % median safety, so the bound's safety comes from the lifetime information it was trained on rather than from caution alone.

That bound, however, is a **scope condition rather than a rate**. Its safety is correlated with the ratio of a bearing's post-alarm window to its training fleet's median (ρ = +0.757 over 1 055 fleet–bearing pairs): the share of pairs reaching 90 % safety rises from 2.7 % below a ratio of 0.5 to 92.9 % above a ratio of 2, crossing near parity. The floor holds for a bearing roughly when the fleet does not outlive it. **Two of fifteen bearings never reach 90 % safety in any fleet, and none is 90 % safe in every fleet.** Fleet size does not fix this: the share of bearings reaching 90 % safety moves only from 44.9 % to 53.3 % as the fleet grows from 4 to 14 bearings (Spearman ρ = +0.32, *p* = 0.33 over the eleven fleet sizes). Per-fleet safety correlates **+0.05 with bearing count but −0.21 with snapshot count**, which we attribute to three long-lived bearings supplying most of the snapshots; at a fixed fleet of twelve bearings the correlation with snapshot count is −0.44. More snapshots from long-lived bearings make the floor less safe, not more.

### 5.7 Calibration error does not detect an uninformative key

We built and scored a reliability layer that attaches to each race call the rate at which such calls are wrong, read from a table keyed by the called fault element and the phase of post-alarm life. Permuting the phase labels within each unit — destroying the key's content — moved calibration error by **0.002 in the favourable direction** while discrimination fell from an AUROC of 0.703 to 0.466, with a matched random key at 0.371; six keyed schemes lay within a 0.008-wide band of calibration error while their AUROC ranged over 0.33.

**A well-calibrated statement can be uninformative.** Calibration should therefore be reported with discrimination and resolution beside it.

The same analysis also illustrates the per-unit problem. A rate of 9.4 % wrong early calls describes no bearing in the dataset: eight units are right through 100 % of their early window and three are wrong through 100 % of theirs. Read per unit it is 3 of 11 (Clopper–Pearson 90 % CI [7.9 %, 56.4 %]); an 80 % between-unit interval is **[0 %, 100 %]**: it covers all 11 held-out units and carries no information.

### 5.8 Rejecting naturally grown out-of-scope damage

§2 notes open-set F1 scores in the high nineties on seeded-fault data. The measurement below is why we do not claim a reject option for our pipeline.

XJTU-SY provides a natural test: of its fifteen bearings, eleven failed at an inner or outer race — the two classes the attribution stage knows — and four failed in other ways: two at the cage, one with a compound fault and one with combined inner- and outer-race damage. Those four are naturally grown out-of-scope damage, which seeded-fault benchmarks cannot offer. We score four standard rejection rules on them, leave-one-bearing-out, on post-alarm snapshots expressed as change from each unit's own healthy baseline, over four feature sets (Table 3). Every rule is oriented so that a high score means *less like the training classes*, and the comparator is chance: a rule that reads nothing scores 0.500.

| Rule | 9 envelope orders | + RMS, kurtosis | + vertical channel | everything measured |
|---|---|---|---|---|
| Model confidence | 0.613 | 0.526 | **0.660** | 0.561 |
| 10-nearest-neighbour distance | 0.523 | 0.458 | 0.539 | 0.496 |
| One-class forest | 0.480 | 0.457 | 0.509 | 0.523 |
| Mahalanobis to class centre | 0.463 | 0.453 | 0.487 | 0.495 |

**Sixteen rule-and-feature combinations span 0.453 to 0.660**, and eight of the sixteen are below chance. At a practical operating point — rejecting 5 % of in-scope snapshots — model confidence rejects **2.3 %** of out-of-scope snapshots and the other three rules reject **none**. Attribution accuracy is unaffected (94.1 %), so the rules are not trading accuracy for caution; they simply separate the out-of-scope units poorly.

More features do not help: adding the second accelerometer moves the distance rules by 0.02–0.03, and adding the detection channel's RMS and kurtosis makes every rule worse. The reason is that, on these features, **the out-of-scope units are not out of distribution**. One cage unit raises BPFO by +15.9 dB where the outer-race group raises it +15.4 dB — on the features that define that class it *is* an outer-race signature. The other cage unit shows no appreciable rise in any feature, and a snapshot near the origin of this change-from-baseline space lies within the training data rather than far from it. Moreover, the FTF lift of the two cage units, +0.9 and +2.3 dB, lies inside the −2.6 to +4.0 dB spread of the eleven race units: the one feature whose physics is specific to the cage does not separate the two units that failed at the cage.

Supervision does not solve the problem either. A supervised in-scope-versus-out-of-scope classifier, trained leave-one-unit-out over all fifteen units, classifies **12 of 15 units correctly, against 11 of 15 for a model that always answers "in scope"**. The one-unit margin comes from the combined inner-and-outer unit, the case where "out of scope" is least defensible, since an outer-race call there is half right; neither cage unit nor the compound unit is detected. Even with labels, these features carry little information about out-of-scope damage.

The general finding is that **novelty relative to healthy data is measurable here, but novelty relative to known damage is not.** Holding the channel, the rules and the retention calibration fixed on Paderborn and changing only the reference population, distance from the *healthy* bearings flags five of eleven damaged units on most of their runs, with the per-unit rate spanning the full 0–100 %; distance from the *known damage classes* flags not one of the eleven on more than 2 % of its runs. Once a unit has alarmed it is far from healthy by construction, because that distance is what triggered the alarm. A reject option needs a second, different distance — from the known damage classes — and in a bearing-geometry feature space that distance carries little information, plausibly because damaged units resemble one another whatever the fault. A check on the other rig agrees: KA22, a real Paderborn outer-race bearing whose fault energy sits at order 5.5 rather than at BPFO, ranks 7th of 11 at the 37th novelty percentile — *less* novel than the average damaged bearing — and no threshold flags it without also flagging the others.

What the attribution stage can offer instead is narrower but useful: the same model confidence separates **a wrong call from a right one** at an AUROC of **0.724**, against 0.613 for separating out-of-scope from in-scope units. (Pooling every snapshot instead of 80 per unit gives 0.752; we quote the per-unit figure because three bearings supply 71 % of the snapshots, and §4.1 explains why they should not dominate the estimate.) The pipeline can therefore flag its likely errors within its scope, but it cannot detect that a unit lies outside that scope.

---

## 6. Five rules

Each rule corresponds to a case in which one of our own numbers would otherwise have been reported misleadingly.

1. **Report a permuted-label null *and* a deliberately incapable control.** Neither subsumes the other (§5.3).
2. **Compute the null for the model you are using, on the input you are giving it.** The 95th percentile moved 7.3 points across the five model-and-input combinations we scored on identical labellings, and the highest and the lowest both belong to networks (§5.5).
3. **Put a degenerate baseline beside any error reported in physical units.** On run-to-failure data a constant has a lower post-alarm mean absolute error than every trained estimator we built (§5.6). Prognostic metrics have been formalised [37]; this rule supplies the reference point against which they are read.
4. **Report a learned model's identity-recognition rate beside its accuracy.** At these study sizes, identity is a usable proxy for class (§5.5).
5. **Report calibration with discrimination and resolution, never alone** (§5.7).

Two further requirements concern how a result is read rather than which statistic accompanies it. **A per-unit guarantee cannot rest on a population statistic**: a median safety of 91 % over a population with no reliably safe member guarantees nothing to any individual unit (§5.6). And **a result pooled over units should be reported against both bearing count and snapshot count**, because the two can point in opposite directions (§5.6).

---

## 7. Limitations

We state these limitations by the same rules we apply to other results.

1. **Study size.** All results rest on 11–15 bearings. The study-size threshold is itself measured at that size, and its subsets reuse units, so it is an estimate with wide and unquantified uncertainty.

   An obvious objection follows: our own curve says nine units are needed, we have eleven, and so our conclusions may be as marginal as those we evaluate. We accept the arithmetic, and separate two kinds of claim. **Detecting an effect needs power.** Where we claim one, we rest it on an exact *p* at or near the test's floor — 0.0061 on XJTU-SY, which is the floor (1/165), and 0.0043 on Paderborn (2/462, against a floor of 1/462 = 0.0022) — and we decline to rank two real effects against each other (§5.2). The six results inside their null are reported as *no demonstrable signal*, never as no effect: at this size a failure to reject the null is weak evidence of absence, and we apply that to our own results. **The rest of the paper makes no effect-size claim at all.** The null band is an exhaustive enumeration of what these eleven units yield under a fixed protocol; the RUL ≡ lifetime identity is arithmetic, verified to 9.1 × 10⁻¹³ min; the identity-recognition rates and the degenerate-baseline comparison are measurements of this benchmark. Adding units would not sharpen those numbers, it would change the quantity being measured — that the band is a function of study size *is* the finding. Measuring a benchmark does not require the power that detecting an effect in a population does, and this asymmetry is why a paper about marginal study sizes can itself be written at one.

2. **Two architectures, one training regime.** We ran a standard wide-first-kernel 1D CNN on raw waveforms and a smaller convolutional network on order spectra, both from scratch, at a fixed budget, without pretraining, augmentation or transfer learning. Two architectures are enough to show that the null moves with the model and its input, and that the movement does not follow the deep/classical divide; they are not enough to characterise deep learning. The claims are about *these* models under *this* protocol. A pretrained or augmented model may behave differently, and we did not test it.
3. **Two laboratory rigs.** Both are accelerated test benches, not machines in service. We have no field data, and no data from a motor with an electrical fault.
4. **RUL on one dataset.** Every remaining-life result uses XJTU-SY. FEMTO/PRONOSTIA [35] and IMS are untested, and the lifetime-parity mechanism predicts that adding short-lived units would change the picture.
5. **Saturated tests.** Both XJTU-SY *p*-values sit at the 1/165 floor; the exact test cannot rank two real effects, and the floor also bounds multiplicity correction — a Bonferroni threshold for Table 1's eleven tests is 0.0045, below anything this design can return on that rig (§4.2).
6. **Feature families are ours.** We tested geometry, shaft-synchronous, resonance/spectral-shape and order-spectrum inputs. A channel we did not construct may behave differently.

---

## 8. Conclusion

Bearing diagnosis is evaluated on datasets whose size makes chance a serious competitor, and prognosis on a benchmark where elapsed time does much of the work. Measuring both reference points changes how reported numbers should be read.

On eleven bearings, a labelling with no physical meaning reaches 69.1–76.4 % bearing-held-out accuracy at the null's 95th percentile, and where in that range the threshold falls is a property of the model and its input, not of the dataset. A real effect is detected in at least 80 % of subsets from nine units on one rig and from seven on the other, where seven is an upper bound set by the design's floor rather than by the effect; the eleven units available clear that bar for the strongest channel on each rig, by two units on Paderborn and by at least four on XJTU-SY. On run-to-failure data an RUL error is a lifetime error, and after the alarm a constant has a lower mean absolute error than every trained estimator we built, most clearly in the final fifth of life.

None of this says the task is impossible; our physics-selected features clear their null on both rigs at *p* ≤ 0.0061 and survive correction across all eleven tests at *q* = 0.022, and a calibrated lower bound on remaining life survives its controls. It says that **a reported accuracy or error is not evidence until it has been compared with an appropriate reference point**, and it provides the tests and thresholds for that comparison. We apply them to all 31 of our own claims: six of our eleven attribution results fall inside their null, and two more clear it only before a correction for multiplicity.

---

## Figures

Exported at journal resolution to `paper/figures/` — vector PDF and 600 dpi RGB TIFF (LZW), flattened onto white — by the notebook that produces each one, so a figure cannot drift from its source. A greyscale proof of every figure is in `paper/figures/greyscale-proofs/`; all five were checked for legibility without colour.

| Fig. | Content | File | Source |
|---|---|---|---|
| 1 | The eleven attribution numbers against their own null's 95th percentile, and the `LZ_frdev` speed reading against the geometry null it clears | `fig1_attribution_vs_null` | `notebooks/29_methods_and_registry.ipynb` |
| 2 | Study-size curves, both rigs, with negative controls and the region where *p* < 0.05 is unreachable | `fig2_study_size_curves` | `notebooks/29_methods_and_registry.ipynb` |
| 3 | True labelling inside the deep baseline's null, both rigs | `fig3_deep_null_both_rigs` | `notebooks/28_deep_baseline_under_the_null.ipynb` |
| 4 | Network accuracy vs operating-point alignment, network against forest | `fig4_operating_point_alignment` | `notebooks/28_deep_baseline_under_the_null.ipynb` |
| 5 | Remaining-life estimators against the degenerate baseline, per fifth of life | `fig5_remaining_life_vs_constant` | `notebooks/27_xjtu_clock_table.ipynb` |

## Data and code availability

Both datasets are public [1, 2]. All results are produced by build scripts in `src/` and consumed by numbered notebooks; `results/registry.csv` lists all 31 claims with their comparators, and **every one of them is recomputed from a cached result at load time** rather than transcribed from a notebook, so no number in this paper can drift from the code that produced it.

## Declarations

**Funding.** This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors. The work was carried out independently, outside the author's employment, and received no institutional or corporate support.

**Declaration of competing interest.** The author declares that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

**Declaration of generative AI and AI-assisted technologies in the manuscript preparation process.** During the preparation of this work, the author used Claude (Anthropic) for language editing, drafting assistance, programming assistance, and discussion of methodological and interpretive questions. The author independently directed the research and made the final research decisions, and all AI-assisted outputs, code, analyses, interpretations, and reported results were reviewed and verified by the author. The author takes full responsibility for the accuracy and integrity of the published work.

## References

1. C. Lessmeier, J. K. Kimotho, D. Zimmer, W. Sextro. *Condition monitoring of bearing damage in electromechanical drive systems by using motor current signals of electric motors: a benchmark data set for data-driven classification.* PHM Society European Conference 3(1), 2016. DOI 10.36001/phme.2016.v3i1.1577.
2. B. Wang, Y. Lei, N. Li, N. Li. *A hybrid prognostics approach for estimating remaining useful life of rolling element bearings.* IEEE Transactions on Reliability 69(1):401-412, 2020. DOI 10.1109/TR.2018.2882682.
3. J. P. Vieira, V. A. Bauler, R. K. Rosa, D. Silva. *Towards a more realistic evaluation of machine learning models for bearing fault diagnosis.* Mechanical Systems and Signal Processing 258:114640, 2026. DOI 10.1016/j.ymssp.2026.114640. (Preprint arXiv:2509.22267, 2025.)
4. J. Hendriks, P. Dumond, D. Knox. *Towards better benchmarking using the CWRU bearing fault dataset.* Mechanical Systems and Signal Processing 169:108732, 2022. DOI 10.1016/j.ymssp.2021.108732.
5. O. Matania, R. Cohen, E. Bechhoefer, J. Bortman. *Test-training leakage in evaluation of machine learning algorithms for condition-based maintenance.* PHM Society European Conference 8(1):13, 2024. DOI 10.36001/phme.2024.v8i1.4125.
6. P. Knap, U. Jachymczyk, K. Lalik. *Leakage-safe, reproducible benchmarking for vibration-based fault diagnosis.* PHM Society European Conference 9(1):1-8, 2026. DOI 10.36001/phme.2026.v9i1.4924.
7. M. Ojala, G. C. Garriga. *Permutation tests for studying classifier performance.* Journal of Machine Learning Research 11(62):1833-1863, 2010.
8. R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, F. A. Wichmann. *Shortcut learning in deep neural networks.* Nature Machine Intelligence 2:665-673, 2020. DOI 10.1038/s42256-020-00257-z.
9. S. Wang, Y. Vidal, F. Pozo. *Uncertainty-aware bearing remaining useful life prediction based on conformal prediction.* PHM Society European Conference 9(1):1-10, 2026. DOI 10.36001/phme.2026.v9i1.4902.
10. S. Yang, J. Wang, Y. Wang. *Empirical calibration and conditional-reliability diagnostics for bearing RUL prediction under operating-regime shift.* arXiv:2607.08273, 2026.
11. K. Hendrickx, W. Meert, Y. Mollet, J. Gyselinck, B. Cornelis, K. Gryllias, J. Davis. *A general anomaly detection framework for fleet-based condition monitoring of machines.* Mechanical Systems and Signal Processing 139:106585, 2020. DOI 10.1016/j.ymssp.2019.106585.
12. W. Zhang, G. Peng, C. Li, Y. Chen, Z. Zhang. *A new deep learning model for fault diagnosis with good anti-noise and domain adaptation ability on raw vibration signals.* Sensors 17(2):425, 2017. DOI 10.3390/s17020425.
13. R. K. Rosa, D. Braga, D. Silva. *Benchmarking deep learning models for bearing fault diagnosis using the CWRU dataset: a multi-label approach.* arXiv:2407.14625, 2024.
14. J. Zhang, G. Shen. *Protocol and implementation sensitivity in raw-index-audited few-shot bearing fault diagnosis benchmarking: evidence from CWRU and HUSTbearing.* Machines 14(8):891, 2026. DOI 10.3390/machines14080891.
15. W. Ye, L. Jiang, E. Xie, G. Zheng, Y. Ma, X. Cao, et al. *The Clever Hans mirage: a comprehensive survey on spurious correlations in machine learning.* arXiv:2402.12715, 2024 (rev. 2025).
16. E. Ramasso, A. Saxena. *Review and analysis of algorithmic approaches developed for prognostics on CMAPSS dataset.* Annual Conference of the PHM Society 6(1), 2014. DOI 10.36001/phmconf.2014.v6i1.2512.
17. S. Kumar, G. Cirrincione, R. R. Kumar. *Bearing remaining useful life estimation using proximal policy optimization (PPO): validation on the XJTU-SY run-to-failure dataset.* Machines 14(6):672, 2026. DOI 10.3390/machines14060672.
18. J. Xu, Y. Wang, R. Xu, H. Wang, X. Zhou. *Research on open-set recognition methods for rolling bearing fault diagnosis.* Sensors 25(10):3019, 2025. DOI 10.3390/s25103019.
19. W. A. Smith, R. B. Randall. *Rolling element bearing diagnostics using the Case Western Reserve University data: a benchmark study.* Mechanical Systems and Signal Processing 64-65:100-131, 2015. DOI 10.1016/j.ymssp.2015.04.021.
20. R. B. Randall, J. Antoni. *Rolling element bearing diagnostics - a tutorial.* Mechanical Systems and Signal Processing 25(2):485-520, 2011. DOI 10.1016/j.ymssp.2010.07.017.
21. J. Antoni. *The spectral kurtosis: a useful tool for characterising non-stationary signals.* Mechanical Systems and Signal Processing 20(2):282-307, 2006. DOI 10.1016/j.ymssp.2004.09.001.
22. J. Antoni. *Fast computation of the kurtogram for the detection of transient faults.* Mechanical Systems and Signal Processing 21(1):108-124, 2007. DOI 10.1016/j.ymssp.2005.12.002.
23. Y. Lei, B. Yang, X. Jiang, F. Jia, N. Li, A. K. Nandi. *Applications of machine learning to machine fault diagnosis: a review and roadmap.* Mechanical Systems and Signal Processing 138:106587, 2020. DOI 10.1016/j.ymssp.2019.106587.
24. Z. Zhao, T. Li, J. Wu, C. Sun, S. Wang, R. Yan, X. Chen. *Deep learning algorithms for rotating machinery intelligent diagnosis: an open source benchmark study.* ISA Transactions 107:224-255, 2020. DOI 10.1016/j.isatra.2020.08.010.
25. F. Jia, Y. Lei, J. Lin, X. Zhou, N. Lu. *Deep neural networks: a promising tool for fault characteristic mining and intelligent diagnosis of rotating machinery with massive data.* Mechanical Systems and Signal Processing 72-73:303-315, 2016. DOI 10.1016/j.ymssp.2015.10.025.
26. L. Wen, X. Li, L. Gao, Y. Zhang. *A new convolutional neural network-based data-driven fault diagnosis method.* IEEE Transactions on Industrial Electronics 65(7):5990-5998, 2018. DOI 10.1109/TIE.2017.2774777.
27. H. Shao, H. Jiang, H. Zhao, F. Wang. *A novel deep autoencoder feature learning method for rotating machinery fault diagnosis.* Mechanical Systems and Signal Processing 95:187-204, 2017. DOI 10.1016/j.ymssp.2017.03.034.
28. M. Zhao, S. Zhong, X. Fu, B. Tang, M. Pecht. *Deep residual shrinkage networks for fault diagnosis.* IEEE Transactions on Industrial Informatics 16(7):4681-4690, 2020. DOI 10.1109/TII.2019.2943898.
29. X. Li, W. Zhang, Q. Ding. *Cross-domain fault diagnosis of rolling element bearings using deep generative neural networks.* IEEE Transactions on Industrial Electronics 66(7):5525-5534, 2019. DOI 10.1109/TIE.2018.2868023.
30. C. Li, S. Zhang, Y. Qin, E. Estupinan. *A systematic review of deep transfer learning for machinery fault diagnosis.* Neurocomputing 407:121-135, 2020. DOI 10.1016/j.neucom.2020.04.045.
31. S. Kapoor, A. Narayanan. *Leakage and the reproducibility crisis in machine-learning-based science.* Patterns 4(9):100804, 2023. DOI 10.1016/j.patter.2023.100804.
32. E. Combrisson, K. Jerbi. *Exceeding chance level by chance: the caveat of theoretical chance levels in brain signal classification and statistical assessment of decoding accuracy.* Journal of Neuroscience Methods 250:126-136, 2015. DOI 10.1016/j.jneumeth.2015.01.010.
33. G. Varoquaux. *Cross-validation failure: small sample sizes lead to large error bars.* NeuroImage 180:68-77, 2018. DOI 10.1016/j.neuroimage.2017.06.061.
34. K. S. Button, J. P. A. Ioannidis, C. Mokrysz, B. A. Nosek, J. Flint, E. S. J. Robinson, M. R. Munafo. *Power failure: why small sample size undermines the reliability of neuroscience.* Nature Reviews Neuroscience 14(5):365-376, 2013. DOI 10.1038/nrn3475.
35. P. Nectoux, R. Gouriveau, K. Medjaher, E. Ramasso, B. Chebel-Morello, N. Zerhouni, C. Varnier. *PRONOSTIA: an experimental platform for bearings accelerated degradation tests.* IEEE International Conference on Prognostics and Health Management (PHM'12), Denver, 2012. HAL hal-00719503.
36. Y. Lei, N. Li, L. Guo, N. Li, T. Yan, J. Lin. *Machinery health prognostics: a systematic review from data acquisition to RUL prediction.* Mechanical Systems and Signal Processing 104:799-834, 2018. DOI 10.1016/j.ymssp.2017.11.016.
37. A. Saxena, J. Celaya, B. Saha, S. Saha, K. Goebel. *Metrics for offline evaluation of prognostic performance.* International Journal of Prognostics and Health Management 1(1), 2010. DOI 10.36001/ijphm.2010.v1i1.1336.
38. X. Li, Q. Ding, J.-Q. Sun. *Remaining useful life estimation in prognostics using deep convolution neural networks.* Reliability Engineering & System Safety 172:1-11, 2018. DOI 10.1016/j.ress.2017.11.021.
39. J. Zhu, N. Chen, W. Peng. *Estimation of bearing remaining useful life based on multiscale convolutional neural network.* IEEE Transactions on Industrial Electronics 66(4):3208-3216, 2019. DOI 10.1109/TIE.2018.2844856.
40. A. N. Angelopoulos, S. Bates. *Conformal prediction: a gentle introduction.* Foundations and Trends in Machine Learning 16(4):494-591, 2023. DOI 10.1561/2200000101.
41. Z. Xu, E. Zio. *Uncertainty-aware prediction of remaining useful life in complex systems.* Annual Conference of the PHM Society 16(1), 2024. DOI 10.36001/phmconf.2024.v16i1.4178.
42. Y.-H. Lin, P.-C. Yan, E. Zio. *Recent advances in uncertainty analysis for prognostics and remaining useful life prediction: a review.* Reliability Engineering & System Safety 269:112110, 2026. DOI 10.1016/j.ress.2025.112110.
43. L. Guo, N. Li, F. Jia, Y. Lei, J. Lin. *A recurrent neural network based health indicator for remaining useful life prediction of bearings.* Neurocomputing 240:98-109, 2017. DOI 10.1016/j.neucom.2017.02.045.
44. E. Zio. *Prognostics and health management (PHM): where are we and where do we (need to) go in theory and practice.* Reliability Engineering & System Safety 218:108119, 2022. DOI 10.1016/j.ress.2021.108119.
45. T. P. K. Nguyen, K. Medjaher. *A new dynamic predictive maintenance framework using deep learning for failure prognostics.* Reliability Engineering & System Safety 188:251-262, 2019. DOI 10.1016/j.ress.2019.03.018.
46. A. Rivas, G. K. Delipei, I. Davis, S. Bhongale, J. Yang, J. Hou. *A component diagnostic and prognostic framework for pump bearings based on deep learning with data augmentation.* Reliability Engineering & System Safety 247:110121, 2024. DOI 10.1016/j.ress.2024.110121.
