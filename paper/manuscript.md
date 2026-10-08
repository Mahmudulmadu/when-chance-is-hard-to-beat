# When chance is hard to beat: exact permutation nulls and degenerate baselines for bearing fault diagnosis and remaining useful life

**Md. Mahmudul Hasan**  
Independent Researcher, Dhaka, Bangladesh · mahmudulmadu@gmail.com

---

## Abstract

Bearing fault diagnosis and remaining-useful-life (RUL) prediction are evaluated largely on public datasets of at most a few dozen bearings. Leakage from window-level splitting is well documented, but less is known about the reference points needed to interpret results from such small datasets. For diagnosis, we compute exact permutation nulls by enumerating all 165 (XJTU-SY) and 462 (Paderborn) class-preserving bearing labellings. With eleven bearings, meaningless labellings reach 69.1–76.4 % accuracy at the 95th percentile, depending on the model and its input. Six of our eleven attribution results fall inside their null; of the five that clear it, only three survive a Benjamini–Hochberg correction. A standard deep baseline stays inside its null on both rigs (*p* = 0.79, 0.72) yet identifies individual bearings at 99.5 % and 95.8 %, which suggests that it learns the bearings rather than the faults. Study-size analysis shows that detecting our strongest effect reliably takes nine bearings on Paderborn and at most seven on XJTU-SY; below seven, the exact test cannot return *p* < 0.05.

For RUL, we show that on run-to-failure data remaining-life error and total-lifetime error are exactly the same quantity. After the alarm, a constant "no life left" prediction has a lower mean absolute error than every trained estimator. RUL error magnitude alone therefore cannot support a remaining-life claim; what can be defended is a lower bound. We distil five reporting rules and publish a registry checking all 31 of our claims against their comparators.

**Keywords:** rolling-element bearings; permutation test; study size; remaining useful life; evaluation methodology; negative controls; condition monitoring

**Highlights**

- On eleven bearings, meaningless labellings reach 69-76 % accuracy (95th percentile)
- The null depends on the model and its input, moving 7 points on one rig's labellings
- On run-to-failure data an RUL error is a lifetime error, to machine epsilon
- A constant "no life left" has lower post-alarm MAE than every trained estimator
- Six of our own eleven attribution results fall inside their own null

---

## 1. Introduction

Data-driven bearing diagnosis is a mature field, and much of its evaluation relies on a small number of public datasets, each containing a few dozen physical bearings at most (15 in XJTU-SY [1] and 32 in Paderborn [2]). Accuracies above 95 % are frequently reported [3–5], and remaining-useful-life (RUL) predictions are commonly evaluated by an error in cycles or units of time, such as the root-mean-square or mean absolute error, often alongside an asymmetric score or a percentage error [6–8].

Several studies have examined the protocols behind these numbers. Splitting windows rather than bearings inflates results, because segments from one recording appear in both training and test sets. Earlier reviews summarised by [9] found that 40 of 41 studies using the CWRU dataset, published between 2008 and 2020, used designs susceptible to this bias, which they call similarity bias. Of 55 studies published between 2020 and 2024 that used the Paderborn vibration data, only six ensured that no bearing's data appeared in both the training and the test set [9, 10]. The problem is not peculiar to this field: a survey of machine-learning-based science found leakage in 17 fields, collectively affecting 294 papers [11]. Bearing-wise partitioning is now widely regarded as a minimum standard [9, 12–14], and Hendriks et al. [12] showed that the conventional CWRU protocol lets a convolutional network learn features tied to *specific bearings* rather than to faults.

This work assumes a clean protocol with bearing-level splits and asks what the resulting accuracy or error should be compared with. We consider two reference points. In the bearing literature reviewed here, neither is standard practice.

The first is *chance at the actual study size*. With eleven bearings and a two-class problem, a classifier that has learned nothing about bearing faults can still score well, because the eleven units differ in ways that can line up with an arbitrary grouping of them: the operating condition they ran at, the severity of their damage, or simply their individual identity. How well such a classifier can score can be measured exactly, by re-running the same protocol on every class-preserving reassignment of the labels.

The second is a *degenerate baseline for quantities reported as errors*. On run-to-failure data the elapsed time is known exactly, so predicting remaining life is arithmetically equivalent to predicting total lifetime [15]. An RUL error is therefore a statement about how well a bearing's total life was estimated, and it should be compared with estimators that predict lifetime without reading any sensor.

We measure both reference points on two public rigs and apply them to our own diagnosis and prognosis pipeline, and we report the results that fall short of them alongside those that clear them.

**Contributions.**

1. **An exact permutation null for bearing-held-out classification**, computed over every class-preserving labelling on two rigs. On eleven units its 95th percentile lies between 69.1 % and 76.4 %, depending on the rig, the model and its input, and an accuracy that does not exceed the threshold for its own rig, model and input does not constitute evidence of fault-specific attribution (§5.1).
2. **Evidence that the null depends on the model and its input, not only on the dataset.** Scored on identical labellings, the null's 95th percentile spans 7.3 points across five model-and-input combinations on one rig (highest for a deep network on raw waveforms, lowest for a small network on the order spectrum), so a threshold measured for one model should not be assumed to hold for another (§5.5).
3. **A study-size curve** for the geometry features, which clear their null with all eleven units on both rigs. Tested against a null permuted inside each subset, the true labelling gives *p* < 0.05 in at least 80 % of subsets from nine units on Paderborn and from seven on XJTU-SY, with an interval on each rate. The XJTU-SY figure is an upper bound: with six or fewer units the exact test cannot return *p* < 0.05, so no smaller count can be tested. From seven units upward, the negative controls, two channels that stay inside their null with all eleven units, give *p* < 0.05 in 4.9 % of subsets on Paderborn and 2.9 % on XJTU-SY (§5.4).
4. **The identity RUL ≡ lifetime prediction**, which follows from the standard definition of an RUL estimate [15] and which we verify to machine epsilon, and the comparison it calls for: after the alarm, a degenerate constant ("no life left") has a lower median per-unit mean absolute error than every trained estimator we built (§5.6).
5. **Five reporting rules and a public claim registry** that checks all 31 of our own claims against their comparators. Six of our eleven attribution results fall inside their null, and two more clear it only before multiplicity correction (§6, §5.2).

---

## 2. Related work

**Evaluation protocol and leakage.** The limitations of window-level splitting on CWRU and Paderborn are well documented [9, 12, 13, 16], and bearing-wise partitioning is widely recommended as the correction. Earlier, Smith and Randall [17] applied three established diagnostic techniques to the entire CWRU dataset and made recommendations on how it should be used as a benchmark. Hendriks et al. [12] showed that the conventional domain-shift framework on CWRU places the same physical bearings in training and test, and that networks trained that way learn bearing-specific features; they proposed splits with independent sets of bearings. Recent benchmarks fix source-target scenarios and enforce recording-level separation [14, 18], and Vieira et al. [9] find that the number of distinct bearings in the training set is a decisive factor for robust performance. We take bearing-wise partitioning as given and do not re-examine the leakage problem: every result below uses bearing-held-out evaluation, and we ask what the resulting number must exceed.

**Shortcut learning.** A model can solve a benchmark through a correlate of the label rather than the intended signal. This is a general phenomenon [19], described across domains as shortcut learning, spurious correlation or the Clever Hans effect [20], and Hendriks et al. [12] identified its bearing-identity form in this field. Our addition is to measure it: we report identity recognition as a number beside accuracy. Under bearing-held-out evaluation, the remedy proposed in [12], a standard 1D convolutional network stays inside its null on both rigs, while the same architecture distinguishes individual bearings at 95.8–99.5 % (§5.5).

**Bearing diagnostics and the features used here.** Our geometry features rest on envelope-spectrum analysis [21]. Our non-geometry channel uses spectral kurtosis and the kurtogram [22, 23] to find the most impulsive frequency band, describes the envelope spectrum in that band by its entropy and flatness, and adds spectral-shape and octave-band energy features; none of these features uses the bearing's fault frequencies or the shaft rate. Reviews cover machine learning for machinery fault diagnosis [24] and deep transfer learning for it [25]. Individual deep-learning methods reach accuracies above 95 % on CWRU [4, 5] and on two pump datasets [5], and an open-source benchmark reports accuracy above 95 % on six of its seven public datasets, including Paderborn and XJTU-SY [3]. In that benchmark each XJTU-SY bearing, and each of the thirteen Paderborn real-damage bearings it uses, is its own class [3], so accuracy there cannot separate recognising the fault from recognising the individual bearing. Further architectures address feature learning [26], noisy signals [27] and cross-domain diagnosis [28].

**Chance level at small sample sizes.** The problem we address has been studied in neuroscience, where small samples are common. Combrisson and Jerbi [29] showed that the theoretical chance level of a classification problem is not the empirical one when samples are few, and published a look-up table, based on the binomial distribution, of the minimum accuracy a classifier must reach to be significant at a given sample size, number of classes and *p*-value. Varoquaux [30] showed that at sample sizes common in neuroimaging, cross-validated accuracy has large error bars, about ±10 % for 100 samples, which the standard error across folds strongly underestimates. Button et al. [31] set out the consequences of low statistical power for a literature's reliability. We apply the same reasoning to condition monitoring, where the unit of analysis is the bearing and the number of units is smaller still.

**Permutation testing.** Label-permutation tests of classifier performance are well established. Ojala and Garriga [32] study a test in which the labels are permuted to ask whether a classifier has found real class structure; they note that the exact test, over every permutation, is computationally infeasible except on very small datasets, and approximate it by sampling. Combrisson and Jerbi [29] use sampled permutation tests to assign significance to decoding accuracy in small-sample brain-signal studies. In the condition-monitoring literature reviewed here, we did not find the test applied at *bearing level* on these benchmarks, where class counts are small enough to enumerate every labelling, $\binom{11}{3} = 165$ of them on XJTU-SY and $\binom{11}{5} = 462$ on Paderborn, so that *p*-values are exact rather than Monte-Carlo estimates.

**RUL evaluation and uncertainty.** Methodological reviews in prognostics cover the approaches developed on the C-MAPSS benchmark [6] and the full pipeline from data acquisition to RUL prediction [33]. Saxena et al. [15] proposed metrics for the offline evaluation of prognostic performance. Deep models have been applied to RUL prediction on the C-MAPSS turbofan data [7] and on bearing run-to-failure data from the FEMTO/PRONOSTIA platform [8, 34]. Health-indicator construction, on which our alarm and our lower bound on remaining life both depend, has been studied in its own right [35], and the reliability literature has examined how prognostic outputs feed maintenance decisions [36–38]. Uncertainty in RUL prediction has been reviewed recently [39], and methods for it include conformal prediction [40] and uncertainty-aware prognostic models [41]. Some recent bearing-RUL studies report calibrated or conformal intervals rather than point estimates [42, 43]; Yang et al. [43] also report coverage per bearing and per operating regime rather than pooled, and find conditional failures that a pooled figure hides. Simple baselines do appear: Kumar et al. [44] report a linear-trend baseline that beats both an LSTM and an MLP on an asymmetric score on an XJTU-SY bearing. The identity between remaining-life and total-lifetime prediction on run-to-failure data follows from the standard definition of an RUL estimate as the estimated end of life minus the current time [15]. In the literature reviewed here, however, we did not find it used to argue that an RUL error should be compared with estimators that read no sensor, nor did we find such *degenerate* baselines.

**Fleet versus per-unit references.** Fleet-based anomaly detection compares a machine with similar machines monitored at the same time, so no historical dataset is needed; it assumes that most machines in the fleet are healthy and that all run under similar conditions [45]. On run-to-failure data a related reference can be pooled from the healthy early life of other units. On XJTU-SY, a reference pooled from the first ten minutes of the other fourteen bearings raises an alarm on 8 of the 15 bearings and on only one of the three inner-race units, where a reference built from each unit's own first twenty minutes raises an alarm on all 15. In each count one alarm comes during the unit's healthy early life and is a false alarm rather than a detection: Bearing3_5 under the pooled reference and Bearing3_2 under its own.

**Deep baselines.** As a deep baseline we use the wide-first-kernel 1D convolutional network (WDCNN) of Zhang et al. [46], which was designed for raw vibration signals. We keep its architecture unmodified and score it with the same protocol and the same labellings as the classical baseline, against a null computed for the network itself.

**Open-set recognition.** Xu et al. [47] report normalised F1 scores of 0.97–1.00 for open-set bearing fault diagnosis on most CWRU tasks and of 1.00 on two of six XJTU-SY tasks, in a protocol where each class is a single bearing and the test windows of the known classes come from the same bearings as the training windows. Under bearing-held-out evaluation, across sixteen rule-and-feature combinations on XJTU-SY, no rejection rule exceeded an AUROC of 0.660, where 0.500 is chance, and a check on Paderborn agrees. On these features the out-of-scope units are poorly separated from the trained classes (§5.8).

## 3. Data

We use two public datasets that differ in two respects central to this study: how the damage arose, and whether each unit has an observed healthy-to-failure trajectory.

**Paderborn (KAt) bearing data centre** [2]. The dataset covers 6203-type bearings on an electromechanical drive train. It records two motor phase currents and the vibration of the bearing housing at 64 kHz, and radial force, load torque, speed and temperature at lower rates [2]; only the vibration signal is used here, with the speed signal giving the shaft rate. We use 25 of its 32 bearings: 6 healthy, 8 of the 12 with artificial damage and 11 of the 14 with real damage grown in accelerated lifetime tests (5 outer-race, 6 inner-race). The eight artificial bearings were chosen at the start of the project, before any result reported here, to give four per race with a similar mix of damage methods: on the outer race one EDM, two electric-engraver and one drilled bearing (KA01, KA03, KA05, KA07), and on the inner race, where [2] has no drilled bearing, one EDM and three electric-engraver bearings (KI01, KI03, KI05, KI07). The other four artificial bearings (KA06, KA08, KA09, KI08) were not used. As in [9], we leave out the three real-damage bearings that Table 5 of [2] lists with damage on both rings (KB23, KB24, KB27). The eleven real-damage units are the attribution set; the six healthy units are used for the identity test (§5.5) and as the healthy reference in §5.8, and the eight artificial units as the known-damage reference in §5.8. Each bearing was measured 20 times for 4 s at each of four operating conditions: 1 500 rpm with 0.7 Nm load torque and 1 000 N radial force, and three variants that each reduce one of these, to 900 rpm, 0.1 Nm or 400 N [2]. All Paderborn results in this paper use the 400 N condition (`N15_M07_F04` in the dataset's file names) and all 20 measurements of each bearing; this condition was chosen in exploratory work on the same units (§4.2). Damage is documented per unit in a fact sheet, by criteria that include mode, symptom, component, extent, arrangement, characteristic and cause [2].

**XJTU-SY run-to-failure data** [1]. Fifteen LDK UER204 bearings were run under three operating conditions (35 Hz/12 kN, 37.5 Hz/11 kN, 40 Hz/10 kN), each from a healthy state until the maximum amplitude of its vibration exceeded ten times the maximum amplitude of its healthy stage; this rule defines the lifetimes used in §5.6. Vibration was sampled at 25.6 kHz in 1.28 s snapshots once per minute, 9 216 snapshots in total. Each snapshot holds a horizontal and a vertical accelerometer channel; we use the horizontal channel throughout and add the vertical one only in §5.8. Lifetimes span 42 to 2 538 minutes, a factor of 60. Eleven units failed at a single race (8 outer, 3 inner) and are the attribution set; the remaining four (two cage, one compound, one combined inner + outer) are used as out-of-scope units (§5.8). All fifteen are used for remaining life (§5.6).

**Label provenance.** Every label used here was checked against its source document rather than taken from code: all 15 XJTU-SY fault labels and file counts match Table 2 of the dataset introduction, and all 25 Paderborn units match their fact sheets on class and on artificial-versus-real. Two units labelled inner-race (KI04, KI14) also carry particle-caused indentations on the outer-ring raceway, documented in their fact sheets but not in Table 5 of [2]; we keep the dataset's labels, which characterise multiple damage by the damage with the highest extent [2]; for both units this is the inner-ring pitting, the only one of their two damages given an extent level.

**Why both rigs.** Paderborn supplies seeded and real damage, but no recording of a damaged unit before it was damaged. XJTU-SY supplies an observed healthy-to-failure trajectory for each unit, but only three inner-race units. Neither alone can support the claims below, and three of the channels in Table 1 clear their null on one rig but not on the other.

---

## 4. Method

### 4.1 Protocol

All classification is bearing-held-out: a model is trained on all units but one and scored on every snapshot or recording of the held-out unit; the reported accuracy is the mean over units of the per-unit accuracy. Reporting per-unit means rather than per-snapshot means matters here, because lifetimes vary 60-fold and three XJTU-SY bearings supply 71 % of all snapshots.

### 4.2 Exact permutation null

For a class-preserving labelling *L* of the units, let *a*(*L*) be the bearing-held-out accuracy under the identical protocol, and let $L_0$ be the true labelling. We enumerate every *L*: 165 on XJTU-SY (8 outer / 3 inner) and 462 on Paderborn (5 outer / 6 inner). With $\mathcal{L}$ the set of all *N* labellings, $L_0$ included, the reported *p* is the share of them that score at least the true accuracy:

$$ p = \frac{1}{N} \sum_{L \in \mathcal{L}} \mathbf{1}\big[a(L) \ge a(L_0)\big] \tag{1} $$

where $\mathbf{1}[\cdot]$ is 1 when the condition inside it holds and 0 otherwise. This *p* is exact, and its smallest possible value, the floor of the test, is 1/165 = 0.0061 on XJTU-SY and 1/462 = 0.0022 on Paderborn. The study-size curve in §4.5 uses the sampled form, for a null of *M* labellings $L_1, \dots, L_M$ that does not contain $L_0$:

$$ p = \frac{1 + \sum_{m=1}^{M} \mathbf{1}\big[a(L_m) \ge a(L_0)\big]}{1 + M} \tag{2} $$

When that null is complete, *M* = *N* − 1 and Eqs. (1) and (2) give the same number, because $L_0$ itself supplies the 1 added to the count and the 1 added to *M*. They differ only when the null is a sample, and no *p* in Table 1 is: each is over the full 165 or 462. The null's 95th percentile is the threshold an accuracy must exceed before it counts as evidence that the channel carries the fault class.

**Multiplicity.** Table 1 reports eleven tests of the same question, whether a channel carries the race on a given model and rig, so they form one family. We report Benjamini–Hochberg *q*-values across all eleven beside the raw *p* rather than instead of it, because the two answer different questions. Three results survive at FDR 5 %, and two of the five that clear at α = 0.05 do not. More important, the design's floor limits what any correction can show here. A Bonferroni threshold for eleven tests is α = 0.0045. Neither XJTU-SY result can reach it, because 1/165 = 0.0061 is the smallest value the exact test can return on that rig, however large the effect. With eleven units there are only 165 labellings, and no correction can make the test finer than that. We therefore treat the floor as the binding limit on this study and report the correction beside the raw *p*.

**Selection before the null.** The features, and on Paderborn the operating condition, were developed in exploratory work on the same units that Table 1 scores, and the exact null holds the final configuration fixed, so every *p* in Table 1 is conditional on that configuration. The most explicit choice was made on Paderborn: twelve configurations, six feature sets at each of two radial forces (1 000 N and 400 N), were compared by the accuracy on the eleven real-damage units of a model trained on the eight artificial-damage bearings, and the envelope levels at BPFO, BPFI and shaft rate, with shaft-rate sidebands around BPFI, its second harmonic and BPFO, at 400 N, were chosen. When that choice is repeated under each of the 462 labellings, the Paderborn geometry result gives *p* = 5/462 = 0.011 instead of 0.0043, and the three results that survive the Benjamini–Hochberg correction still survive it, at *q* = 0.034, 0.034 and 0.040 instead of 0.022. This repeats only the last step of the exploration, so it is a partial correction. We did not compute it for XJTU-SY, where the one explicit choice was between two encodings, raw levels or change from each unit's own healthy baseline, that both attributed all eleven units correctly.

Seeds are paired: model initialisation and data sampling depend on the fold, not on the labelling, so the true labelling and every permuted one start from the same state.

### 4.3 Deliberately incapable controls

A permutation null asks whether the class structure is real. It does not ask whether a feature can carry what its name suggests. For that we run features constructed to be incapable: a shaft-speed reading in place of a diagnostic feature; the load-zone fold done at 1.37 × shaft rate instead of at the shaft rate, which is not locked to the shaft and so cannot carry load-zone modulation; and, for the deep baseline, crops whose samples are shuffled in time, destroying the waveform while preserving every amplitude statistic. Both tests are needed; §5.3 gives a case where each catches what the other misses.

### 4.4 Degenerate baselines for errors

For any quantity reported in minutes we score, alongside the model, estimators that read no sensor: *zero* (no life remaining), *survived* (as long again as it has already run), the training fleet's median lifetime minus elapsed time, the same restricted to units at the same operating condition, and a forest given only elapsed time and the shaft speed of the unit's operating condition (35, 37.5 or 40 Hz). That speed is a setting of the test, the same for the whole life of a unit, not a measurement, so none of these estimators reads a sensor.

### 4.5 Study-size curve

For each unit count *n* we draw 30 subsets of *n* bearings at random, or take every subset where fewer than 30 exist (11 at *n* = 10, one at *n* = 11), and discard any subset that contains a single class. In each we score the true labelling and compare it with a null permuted inside that same subset, so the reference reflects the size being tested. Within a subset the null is enumerated exhaustively wherever it has at most 40 non-true labellings and sampled at 40 above that. In either case *p* comes from Eq. (2), which equals Eq. (1) when the null is enumerated in full. A subset detects the effect when *p* < 0.05, and the study-size threshold is the smallest *n* at which at least 80 % of subsets detect it. Detection rates are reported with a Clopper–Pearson 95 % interval, and that interval is optimistic: subsets of the same eleven bearings overlap heavily, and at *n* = 9 each bearing appears in as many as 28 of the 30 subsets.

Two properties of this design limit what it can show, and both can be mistaken for findings.

**The per-subset *p* has a floor that moves with *n* and with the class split.** By Eq. (2), the smallest *p* a subset can return is 1/(1 + *M*). A subset whose null is sampled at 40 labellings has a floor of 1/41 = 0.024; a subset whose null is enumerated in full has a higher one, for example 1/28 = 0.036 for an XJTU-SY subset of eight units with two inner-race units. From seven units upward every subset in the curves has a floor between 0.024 and 0.048, so every one of them can reach *p* < 0.05. At six units or fewer every null has at most 19 labellings, and the floor is 0.05 or more: between 0.05 and 0.17 at *n* = 6, 0.10 and 0.20 at *n* = 5, and 0.17 and 0.25 at *n* = 4, depending on how the units split between the two classes. No subset of six or fewer units can reach significance at any effect size, so a detection rate of zero there says nothing about the channel. At *n* = 4 on Paderborn the true accuracy already exceeds its own null band, by 2.0 points, and not one of 29 subsets is significant.

**The null must exclude the true labelling.** An earlier version of this analysis did not. It drew 20 of a subset's $\binom{n}{k}$ labellings, *k* being the number of units in one class, without removing the true one, and took *p* as the share of the 20 that scored at least the true accuracy. When the true labelling was among the 20 it matched itself, so *p* was at least 1/20 = 0.05 and the subset could not reach significance. This happened with probability $20/\binom{n}{k}$, or every time when a subset had 20 labellings or fewer, so it hit subsets with a small class hardest: 20/28 = 71 % for an XJTU-SY subset of eight with two inner-race units, against 20/70 = 29 % for a Paderborn subset of eight split four and four. It penalised XJTU-SY more than Paderborn and reversed the apparent ordering of the two rigs.

### 4.6 Identity test (fingerprinting)

When each class has only a few units, a model that can tell the units apart could use a unit's identity in place of its fault class. A high identity score shows that this is possible for the model, not that the model does it; §5.5 gives further evidence for the deep baseline. We therefore score each learned model on a task with no fault content: telling individual bearings apart. On Paderborn the units are the six healthy bearings, and each fold holds out five of every bearing's 20 recordings. On XJTU-SY they are the eleven race units, and each fold holds out a block of five early snapshots from every unit. Every unit thus contributes the same number of test samples, so a model that ignores the signal scores 1/units, which is the chance rate we report.

### 4.7 Calibration with discrimination and resolution

Reliability statements are scored on three measures, never on calibration alone. *Calibration error* is the gap between the stated rate and the observed rate: for each held-out unit we put its snapshots into ten bins of the stated rate (width 0.1), take the absolute difference between the mean stated rate and the observed rate in each bin, and average these differences weighted by the number of snapshots. *Discrimination* is the AUROC of the stated rate for telling right calls from wrong ones. *Resolution* is the standard deviation of the stated rate across snapshots; it is zero for a statement that never changes. §5.7 shows why all three are needed.

### 4.8 Reproducibility

Every result is produced by a build script that caches its output, and all 31 claims in the registry are recomputed from those caches rather than copied by hand. In our CPU environment, PyTorch did not give identical results at different thread counts: the accuracy of one true labelling moved from 35.9 % to 33.2 % between 3 and 5 threads. All numbers compared against a null were therefore produced at a single thread count, and every quantity computed twice is checked to agree exactly.

---

## 5. Results

### 5.1 The null band on eleven units

On eleven units, the 95th percentile of bearing-held-out accuracy over labellings with no physical meaning is 69.1 % to 76.4 %, depending on the rig, the model and the input (Fig. 1). For the forests (the hand-built feature sets and the forest on 576 order bins), the band is 69.5–73.9 %; for WDCNN on raw waveforms, 76.3–76.4 %; for the small spectrum-CNN, 69.1–73.5 %.

In practice, on eleven units a bearing-held-out accuracy in this range may or may not be evidence of diagnosis. That depends on the null of the rig, model and input it came from, so that null has to be computed. The spectrum-CNN on Paderborn, for example, clears its null at 70.9 %, while the majority-class rule reaches 72.7 % on XJTU-SY without reading any signal (§5.2). A similar concern has been raised in neuroscience, where sample-size-specific thresholds have been proposed in place of theoretical chance levels [29].

### 5.2 Our own results against their nulls

Of the eleven attribution numbers we report, five clear their null and six do not (Fig. 1, Table 1). Every *p* below is exact, over all 165 assignments on XJTU-SY and all 462 on Paderborn:

| Channel | Rig | Accuracy | Null 95th | *p* | BH *q* | Verdict |
|---|---|---|---|---|---|---|
| Geometry (envelope order bands) | XJTU-SY | 94.1 % | 71.9 % | 0.0061 | 0.022 | clears |
| Load-zone (shaft-synchronous fold) | XJTU-SY | 89.7 % | 73.9 % | 0.0061 | 0.022 | clears |
| Geometry | Paderborn | 85.5 % | 69.5 % | 0.0043 | 0.022 | clears |
| Forest on 576 order-spectrum bins | Paderborn | 78.6 % | 71.4 % | 0.0195 | 0.054 | clears at α, not at FDR 5 % |
| Spectrum-CNN on 576 order bins | Paderborn | 70.9 % | 69.1 % | 0.0368 | 0.081 | clears at α, not at FDR 5 % |
| Load-zone | Paderborn | 62.7 % | 71.4 % | 0.147 | 0.270 | **inside** |
| Non-geometry (resonance/spectral shape) | XJTU-SY | 62.3 % | 71.1 % | 0.442 | 0.608 | **inside** |
| Spectrum-CNN on 576 order bins | XJTU-SY | 60.4 % | 73.5 % | 0.376 | 0.591 | **inside** |
| Forest on 576 order bins | XJTU-SY | 58.6 % | 71.6 % | 0.697 | 0.788 | **inside** |
| WDCNN on raw waveform | XJTU-SY | 48.9 % | 76.3 % | 0.788 | 0.788 | **inside** |
| WDCNN on raw waveform | Paderborn | 33.2 % | 76.4 % | 0.723 | 0.788 | **inside** |
| *Always outer race* (reads no signal) | XJTU-SY | 72.7 % | — | — | — | label-only baseline |

The last row is not a result. It is a comparator the table would otherwise leave out: on XJTU-SY the majority class alone scores 72.7 %, above four of the six XJTU-SY results and within that rig's 71.1–76.3 % range of null thresholds. An accuracy in this table has to be read against it as well as against its own null. The non-geometry row comes from the 15-tree forest used for the study-size curve, not from the 100-tree forest used for the geometry and load-zone rows.

The load-zone channel shows why the null matters. It clears its null at the test's floor on XJTU-SY (*p* = 0.0061) and sits inside it on Paderborn (*p* = 0.147). Without the test, the Paderborn result would be described as weaker than geometry; with it, the accurate description is "no demonstrable signal". The two descriptions lead to different conclusions.

Under Benjamini–Hochberg across all eleven tests (§4.2), three of the five results that clear at α = 0.05 survive at FDR 5 %: the geometry channel on both rigs and the load-zone channel on XJTU-SY, all at *q* = 0.022, or at *q* = 0.034–0.040 when the Paderborn configuration choice is repeated under every labelling (§4.2). The two Paderborn order-spectrum results do not (*q* = 0.054 and 0.081). We report them as clearing their own null and failing the correction, because both statements are true.

The two XJTU-SY channels that clear sit at the test's 1/165 floor, so the test is saturated there: it rejects the null for both but cannot rank the two effects against each other. The two rows nearest the 0.05 line, the Paderborn spectrum models at 0.0195 and 0.0368, correspond to 9 and 17 labellings out of 462; both were sampled at *N* = 100 in an earlier version of this work, where they read 0.0198 and 0.0396, and we completed the enumeration so that no *p* in the table is a sampled estimate.

### 5.3 A null is necessary, not sufficient

Two results show that the permutation test cannot be the only control.

`LZ_frdev` is the offset between the nominal shaft rate and the rate at which the load-zone fold found its peak. It is a speed reading, and it should not be able to tell an inner from an outer race. It scores 81.3 %, above every null threshold in Table 1. It was caught only by asking whether the quantity could carry what its name implied. Our best explanation is the search itself: on its 0.04 Hz grid, all three inner-race units returned the same grid point on every post-alarm snapshot, while the outer-race units moved between points. A stable lock may be a weak echo of a sharp inner-race peak, but it reaches the classifier through the grid rather than through a measurement of the fault, and it rests on three units.

Conversely, the same fold done at 1.37 × shaft rate, which is not locked to the shaft and was built to be incapable, scores 44.1 % against 65.5 % for the same fold at the true shaft rate, both with a 300-tree forest; with the 100-tree forest used for the null in Table 1, the true fold scores 62.7 %. The control behaves as it should, indicating that the true fold measures shaft-locked content, even on the rig where that channel fails its null.

The permutation null and the incapable control therefore answer different questions, and neither can replace the other.

### 5.4 How many units a claim needs

As units are added the null band narrows, so power comes from the narrowing null as well as from the rising accuracy. A plain learning curve does not show this. Each unit count is scored over up to 30 subsets (§4.5). On Paderborn, the geometry channel's effect, which clears its null on the full set, is detected in 12 of 30 subsets at *n* = 7 (40 %, 95 % CI [23 %, 59 %]), 21 of 30 at *n* = 8 (70 %, [51 %, 85 %]) and 28 of 30 at *n* = 9 (93 %, [78 %, 99 %]), where the mean margin over the null reaches +13.3 points; every subset is detected at *n* = 10 and *n* = 11 (Fig. 2).

On XJTU-SY, where the effect is larger, detection is complete from *n* = 7: 30 of 30 subsets, and 100 % at every larger size. That figure has to be read together with the design's floor. Below *n* = 7 the exact null inside a subset cannot return *p* < 0.05 at all (§4.5), and XJTU-SY's detection rate jumps from 0 % at *n* = 6 to 100 % at *n* = 7, the step at which the smallest attainable *p* first falls below 0.05. Seven is therefore an upper bound on what that rig needs, not a measurement of it: at *n* = 5 and *n* = 6 the mean margin over the null is already +20.1 and +22.3 points with not one subset significant. Paderborn's nine is a measurement, because its floor becomes reachable at *n* = 7 and the detection rate then climbs over the next two unit counts rather than saturating at once.

The thresholds are therefore nine units on Paderborn and at most seven on XJTU-SY, and they should not be averaged into one number: the first is a measurement and the second a bound. With eleven units, both datasets have enough for the strongest channel on each rig: two units more than needed on Paderborn, and at least four more on XJTU-SY, where the requirement itself is an upper bound. The negative controls behave as expected. If a channel carries no class information, a valid test at α = 0.05 should give *p* < 0.05 in at most 5 % of subsets. From seven units upward, where *p* < 0.05 can be reached, Paderborn's load-zone channel reaches it in 5 of 102 subsets and XJTU-SY's non-geometry channel in 3 of 102, that is 4.9 % and 2.9 %. Because the subsets share bearings, these rates are only rough. On average both channels stay below their own null's 95th percentile at every size. Below seven units neither can reach significance, by construction. XJTU-SY's load-zone channel, which does clear its null on that rig, needs eight.

Two limitations apply to these thresholds. The subsets at a given size are drawn from the same eleven bearings and are far from independent (at *n* = 9 each bearing appears in as many as 28 of the 30 subsets), so the intervals above are optimistic, and the threshold is itself estimated from eleven units. Separately, the number of units needed to detect an effect is not the number at which accuracy stops improving. In exploratory work not reported here, accuracy stopped improving at about six units; the two numbers should not be confused.

### 5.5 A deep baseline under its own null

WDCNN [46] on decimated raw vibration, evaluated bearing-held-out with a fixed training budget, no early stopping and paired seeds, reaches 48.9 % (XJTU-SY) and 33.2 % (Paderborn), against 94.1 % and 85.5 % for the forest on geometry features with identical folds, and sits inside its own null on both rigs (*p* = 0.788, 0.723; Fig. 3).

The failure is not due to undertraining. Quadrupling the budget leaves the same units wrong. Within each fold it reaches 97.5 % (XJTU-SY) and 99.4 % (Paderborn) on a held-out sample of its own training units (at least 90 % in every one of 22 folds), while scoring 48.9 % and 33.2 % on the unit it did not see.

Its calls seem to follow the bearing and its operating condition rather than the fault. Hendriks et al. [12] showed that a network can learn bearing-specific features, and their remedy was splits with independent bearings. We use that remedy, since every number here is bearing-held-out, and the behaviour can still be measured: the same network at the same budget distinguishes individual bearings at 99.5 % (chance 9.1 %) and 95.8 % (chance 16.7 %). Shuffling every sample in time, which destroys the waveform but keeps the amplitudes, does not reduce race accuracy (61.4 % against 48.9 % on XJTU-SY, 32.3 % against 33.2 % on Paderborn). On XJTU-SY its held-out calls match a label-only rule ("the majority race among other units at the same speed and load") on 11 of 11 units. Across all 165 labellings its accuracy is also correlated with operating-condition purity, the extent to which a labelling lines up with the operating conditions (Spearman +0.26, *p* = 9 × 10⁻⁴), while the forest's is not (−0.06, *p* = 0.44; Fig. 4). This correlation is weak, and purity takes only five values on these eleven units; the 11-of-11 match is the stronger evidence.

Two findings extend beyond this one architecture, as the second network shows. First, the null depends on the model and its input, not only on the dataset. On the identical 462 Paderborn labellings the 95th percentile runs from 69.1 % for the small spectrum-CNN through 69.5 % and 71.4 % for the forests to 76.4 % for WDCNN on the raw waveform, a spread of 7.3 points; on XJTU-SY's 165 it runs 71.1–76.3 %. WDCNN's null is the widest on both rigs, plausibly because higher capacity on the least-processed input finds more ways to fit an arbitrary grouping of eleven units. The ordering does not simply separate deep from classical models, however: the narrowest null in the study also belongs to a network, the spectrum-CNN on Paderborn. A threshold borrowed from another model can therefore be wrong by up to seven points, in either direction. Second, in the network's Paderborn null one labelling other than the true race labels scores 95.0 %, so a bearing-held-out accuracy of that size, from a standard deep model on eleven units, is not by itself evidence of diagnosis.

Finally, at this study size the choice of input mattered more than the choice of architecture. Given the envelope spectrum on a shaft-order axis instead of the raw waveform, both a small CNN (70.9 %, *p* = 0.0368) and a forest on the same 576 bins (78.6 %, *p* = 0.0195) clear their nulls on Paderborn. On XJTU-SY both fail their own exact nulls, at 60.4 % (*p* = 0.376) and 58.6 % (*p* = 0.697), and subtracting each unit's own healthy-period spectrum from its inputs moves them only to 62.4 % and 57.9 %. The failure on that rig is therefore measured against each model's own null rather than inferred from other models' thresholds. Only the physics-based geometry features clear the null on both rigs.

### 5.6 Remaining life: an identity and a constant baseline

On run-to-failure data, RUL prediction is total-lifetime prediction. On XJTU-SY a unit with a lifetime of *T* minutes has *T* snapshots, at minutes $t = 0, 1, \dots, T - 1$, and its remaining life at minute *t* is

$$ \mathrm{RUL}(t) = (T - 1) - t \tag{3} $$

which is the standard definition of remaining life as end of life minus current time [15]. Because *t* is known exactly, an estimate $\widehat{\mathrm{RUL}}(t)$ implies the lifetime estimate $\widehat{T} = \widehat{\mathrm{RUL}}(t) + t + 1$, and $\widehat{T} - T = \widehat{\mathrm{RUL}}(t) - \mathrm{RUL}(t)$: the two errors are the same number. Across 73 728 estimator-snapshots the largest discrepancy between the two errors is 9.1 × 10⁻¹³ minutes, two units in the last place of float64 at the longest lifetime. This is a property of the benchmark, not of a model, and it means a reported RUL error should be benchmarked against estimators that predict lifetime without reading a sensor.

Against those, even our best state-based estimator has a higher post-alarm median per-unit MAE than the constant (Table 2). The MAE column is each unit's mean absolute error over its post-alarm snapshots, with the median taken over units. The second column divides each snapshot's absolute error by the life actually remaining at that snapshot, then takes the median over each unit's snapshots and the median over units; each unit's final snapshot, where the remaining life is zero, is left out. We call it the scale-free measure: short-lived and long-lived bearings count on the same scale, and predicting zero always scores 1.00 on it.

| Estimator | MAE (min) | Scale-free measure |
|---|---|---|
| **Zero (no life left)** | **51** | 1.00 |
| Two-channel state, fitted post-alarm | 57 | **0.79** |
| Two-channel state | 78 | 1.26 |
| Learned clock (elapsed time + test speed) | 89 | 1.74 |
| Fleet-median lifetime at this condition | 83 | 1.00 |

Predicting *zero* gives the lowest post-alarm MAE of the five estimators. Its advantage comes from the end of life, where the life remaining, and so the constant's error, is small. Fig. 5 splits each unit's life into fifths and uses all snapshots, before and after the alarm. In the final fifth, the period in which a maintenance decision is most likely to be taken, the two-channel state's error is 7.1 times the constant's (113 min against 16); counting post-alarm snapshots only, it is 4.6 times (74 against 16). Of the five estimators in Table 2, only the post-alarm-fitted model does better than the constant on the scale-free measure (0.79 against 1.00), on 9 of 15 units; by MAE it is better than the constant on only 8 of 15.

We do not conclude that remaining life is unpredictable. We conclude that post-alarm MAE on this benchmark cannot support a remaining-life claim, because a constant that reads no sensor achieves the lowest MAE, and that what can be supported is a bound. For one unit, let *S* be its set of post-alarm snapshots. The *safety* of an estimate is the share of *S* on which it does not overstate the remaining life,

$$ \text{safety} = \frac{1}{|S|} \sum_{t \in S} \mathbf{1}\big[\widehat{\mathrm{RUL}}(t) \le \mathrm{RUL}(t)\big] \tag{4} $$

and its *give-up* is the median of how far it falls below the remaining life, as a fraction of that life,

$$ \text{give-up} = \operatorname*{median}_{t \in S,\ \mathrm{RUL}(t) > 0} \frac{\mathrm{RUL}(t) - \widehat{\mathrm{RUL}}(t)}{\mathrm{RUL}(t)} \tag{5} $$

which leaves out the final snapshot, where the remaining life is zero. A lower bound taken from the 10 % quantile of the forest's per-tree predictions has a median safety over units of 91 %, against 19 % for the point estimate, and a median give-up of 63 %: it understates the remaining life by about two thirds. Trained on permuted lifetimes, the same bound falls to 17.5 % median safety, so its safety comes from the lifetime information it was trained on rather than from caution alone.

Whether the bound is safe depends on the bearing, so its 91 % is not a rate that every unit can expect. Its safety is correlated with the ratio of a bearing's post-alarm window to the median post-alarm window of its training fleet (ρ = +0.757 over 1 055 fleet–bearing pairs): the share of pairs reaching 90 % safety rises from 2.7 % below a ratio of 0.5 to 92.9 % above a ratio of 2, crossing near 1. Roughly, the bound is safe for a bearing whose post-alarm window is at least as long as the fleet's median. Two of fifteen bearings never reach 90 % safety in any fleet, and none is 90 % safe in every fleet. Fleet size does not fix this: the share of bearings reaching 90 % safety moves only from 44.9 % to 53.3 % as the fleet grows from 4 to 14 bearings (Spearman ρ = +0.32, *p* = 0.33 over the eleven fleet sizes). Per-fleet safety correlates +0.05 with bearing count but −0.21 with snapshot count, which we attribute to three long-lived bearings supplying most of the snapshots; at a fixed fleet of twelve bearings the correlation with snapshot count is −0.44. Adding snapshots from long-lived bearings makes the bound less safe.

### 5.7 Calibration error does not detect an uninformative key

We built and scored a reliability layer that attaches to each race call the rate at which such calls are wrong, read from a table keyed by the called fault element and the phase of post-alarm life. Permuting the phase labels within each unit, which destroys the key's content, moved calibration error by 0.002 in the favourable direction while discrimination fell from an AUROC of 0.703 to 0.466, with a matched random key at 0.371; six keyed schemes lay within a 0.008-wide band of calibration error while their AUROC ranged over 0.33.

A statement can therefore be well calibrated and still carry no information, and calibration should be reported together with discrimination and resolution.

The same analysis also illustrates the per-unit problem. A rate of 9.4 % wrong early calls fits none of the bearings in the dataset: eight units are right through 100 % of their early window and three are wrong through 100 % of theirs. Read per unit it is 3 of 11 (Clopper–Pearson 90 % CI [7.9 %, 56.4 %]); an 80 % between-unit interval is [0 %, 100 %]: it covers all 11 held-out units and carries no information.

### 5.8 Rejecting naturally grown out-of-scope damage

§2 notes that Xu et al. [47] report normalised F1 scores of up to 1.00 for open-set recognition. In their protocol each known class is a single bearing, and its test windows are cut from the same bearing as its training windows. The measurement below, made with whole bearings held out, is why we do not claim a reject option for our pipeline.

XJTU-SY provides a natural test: of its fifteen bearings, eleven failed at an inner or outer race, the two classes the attribution stage knows, and four failed in other ways: two at the cage, one with a compound fault and one with combined inner- and outer-race damage. Those four are naturally grown out-of-scope damage, which seeded-fault benchmarks cannot offer. We score four standard rejection rules on them, leave-one-bearing-out, on post-alarm snapshots expressed as change from each unit's own healthy baseline, over four feature sets (Table 3). Every rule is oriented so that a high score means *less like the training classes*, and the comparator is chance: a rule that reads nothing scores 0.500.

| Rule | 9 envelope orders | + RMS, kurtosis | + vertical channel | everything measured |
|---|---|---|---|---|
| Model confidence | 0.613 | 0.526 | **0.660** | 0.561 |
| 10-nearest-neighbour distance | 0.523 | 0.458 | 0.539 | 0.496 |
| One-class forest | 0.480 | 0.457 | 0.509 | 0.523 |
| Mahalanobis to class centre | 0.463 | 0.453 | 0.487 | 0.495 |

Sixteen rule-and-feature combinations span 0.453 to 0.660, and eight of the sixteen are below chance. At a practical threshold, one that rejects 5 % of in-scope snapshots, model confidence on the nine envelope orders rejects 2.3 % of out-of-scope snapshots and the other three rules reject none; the best combination, model confidence with the vertical channel added, rejects 24.9 %. Attribution accuracy is unaffected (94.1 %), so the rules are not trading accuracy for caution; they simply separate the out-of-scope units poorly.

More features do not help: adding the second accelerometer moves the two distance rules by 0.016 and 0.023, and adding the detection channel's RMS and kurtosis makes every rule worse. The likely reason is that, on these features, the out-of-scope units do not look different from the trained classes. The two cage units show this most clearly. One raises BPFO by +15.9 dB, where the outer-race group raises it by +15.4 dB, so on the features that define that class it looks like an outer-race fault. The other shows no clear rise in any feature, and a snapshot near the origin of this change-from-baseline space lies inside the training data, not far from it. Moreover, the FTF lift of the two cage units, +0.9 and +2.3 dB, lies inside the −2.6 to +4.0 dB spread of the eleven race units: the one feature whose physics is specific to the cage does not separate the two units that failed at the cage.

Supervision does not solve the problem either. A supervised in-scope-versus-out-of-scope classifier, trained leave-one-unit-out over all fifteen units, classifies 12 of 15 units correctly, against 11 of 15 for a model that always answers "in scope". The one-unit margin comes from the combined inner-and-outer unit, the case where "out of scope" is least defensible, since an outer-race call there is half right; neither cage unit nor the compound unit is detected. Even with labels, these features carry little information about out-of-scope damage.

The general finding is that novelty relative to healthy data is measurable here, but novelty relative to known damage is not. Holding the channel, the rules and the retention calibration fixed on Paderborn and changing only the reference population, distance from the *healthy* bearings flags five of eleven damaged units on most of their runs, with the per-unit rate spanning the full 0–100 %; distance from the *known damage classes*, represented by the eight artificial-damage bearings, flags not one of the eleven on more than 2 % of its runs. Once a unit has alarmed it is far from healthy by construction, because that distance is what triggered the alarm. A reject option needs a different distance, the distance from the known damage classes, and in a bearing-geometry feature space that distance carries little information, plausibly because damaged units resemble one another whatever the fault. A check on the other rig agrees: KA22, a real Paderborn outer-race bearing whose fault energy sits at order 5.5 rather than at BPFO, ranks 7th of 11 at the 37th novelty percentile, less novel than the average damaged bearing, and no threshold flags it without also flagging the others.

What the attribution stage can offer instead is narrower but useful: the same model confidence separates a wrong call from a right one at an AUROC of 0.724, against 0.613 for separating out-of-scope from in-scope units. (Pooling every snapshot instead of 80 per unit gives 0.752; we quote the per-unit figure because three bearings supply 71 % of the snapshots, and §4.1 explains why they should not dominate the estimate.) The pipeline can therefore flag its likely errors within its scope, but it cannot detect that a unit lies outside that scope.

---

## 6. Five rules

Each rule corresponds to a case in which one of our own numbers would otherwise have been reported misleadingly.

1. **Report a permuted-label null *and* a deliberately incapable control.** Neither replaces the other (§5.3).
2. **Compute the null for the model you are using, on the input you are giving it.** The 95th percentile moved 7.3 points across the five model-and-input combinations we scored on identical labellings, and the highest and the lowest both belong to networks (§5.5).
3. **Put a degenerate baseline beside any remaining-life error reported in physical units.** On run-to-failure data a constant has a lower post-alarm mean absolute error than every trained estimator we built (§5.6). Prognostic metrics have been formalised [15]; this rule gives the reference point against which they are read.
4. **Report a learned model's identity-recognition rate beside its accuracy.** At these study sizes a model can use unit identity in place of fault class (§4.6, §5.5).
5. **Report calibration with discrimination and resolution, never alone** (§5.7).

Two further points concern how a result is read. A population statistic does not give a guarantee for a single unit: a median safety of 91 % says little about a given bearing when no bearing is 90 % safe in every fleet (§5.6). And a result pooled over units should be reported against both bearing count and snapshot count, because the two can point in opposite directions (§5.6).

---

## 7. Limitations

1. **Study size.** The main evaluations use 11–15 bearings: the eleven race units on each rig for attribution, and the fifteen XJTU-SY units for remaining life. Some analyses use more Paderborn bearings: the choice of the Paderborn configuration used the eleven real-damage units together with the eight artificial-damage bearings (§4.2), §5.5 uses the six healthy bearings, and §5.8 uses both the six healthy and the eight artificial bearings. The study-size threshold is itself measured at this size, and its subsets reuse units, so it is an estimate with wide uncertainty that we have not quantified.

   An obvious objection follows: our own curve says nine units are needed, we have eleven, and so our conclusions may be as marginal as the results we evaluate. We accept this and separate two kinds of claim. Detecting an effect needs power. For the attribution results we keep as positive evidence, we rely on exact *p*-values at or near the test's floor: 0.0061 on XJTU-SY, which is the floor (1/165), and 0.0043 on Paderborn (2/462, against a floor of 1/462 = 0.0022), or 0.011 (5/462) when the Paderborn configuration choice is repeated under every labelling (§4.2). We do not rank two such effects against each other (§5.2). The six results inside their null are reported as *no demonstrable signal*, never as no effect, because at this size a failure to reject the null is weak evidence of absence. We make no population-level effect-size claim. The null band is a complete enumeration of what these eleven units give under a fixed protocol; the RUL ≡ lifetime identity is arithmetic, checked to 9.1 × 10⁻¹³ min; and the identity-recognition rates and the degenerate-baseline comparison are measurements on this benchmark. Adding units would not make these numbers more precise. It would change what is measured, because the band depends on the number of units. Measuring a benchmark therefore does not need the statistical power that detecting an effect in a population needs, and this is why the study can be done at the small sizes it examines.

2. **Two architectures, one training regime.** We ran a standard wide-first-kernel 1D CNN on raw waveforms and a smaller convolutional network on order spectra, both from scratch, at a fixed budget, without pretraining, augmentation or transfer learning. Two architectures are enough to show that the null moves with the model and its input, and that the movement does not follow the deep/classical divide; they are not enough to characterise deep learning. The claims are about these two models under this protocol. A pretrained or augmented model may behave differently, and we did not test it.
3. **Two laboratory rigs.** Both are accelerated test benches, not machines in service. We have no field data, and no data from a motor with an electrical fault.
4. **RUL on one dataset.** Every remaining-life result uses XJTU-SY. FEMTO/PRONOSTIA [34] and IMS are untested. Because the bound's safety depends on how a bearing's post-alarm window compares with its fleet's (§5.6), adding short-lived units would probably change the remaining-life results.
5. **Saturated tests.** The two XJTU-SY results that clear their null both sit at the 1/165 floor, so the exact test cannot rank them. The floor also limits multiplicity correction: a Bonferroni threshold for Table 1's eleven tests is 0.0045, below anything this design can return on that rig (§4.2).
6. **Limited feature coverage.** We tested geometry, shaft-synchronous, resonance/spectral-shape and order-spectrum inputs, all built by us. Other feature families may behave differently.
7. **Selection on the evaluated units.** The features, and the Paderborn operating condition, were chosen in exploratory work on the units the nulls are computed on. Repeating the last Paderborn choice under every labelling raises its *p* from 0.0043 to 0.011 (§4.2); earlier exploratory steps are not corrected for.

---

## 8. Conclusion

Bearing diagnosis is often evaluated on datasets so small that meaningless labellings can reach high accuracy, and remaining-life prediction on run-to-failure data where elapsed time alone carries much of the information. We measured both effects, and they change how reported numbers should be read.

On eleven bearings, meaningless labellings reach 95th-percentile bearing-held-out accuracies of 69.1–76.4 %, and where the threshold falls in that range depends on the rig, the model and its input. For the geometry features, the true labelling reaches *p* < 0.05 in at least 80 % of subsets from nine units on one rig and from seven on the other, where seven is an upper bound set by the design's floor rather than by the effect. The eleven units available are enough for the strongest channel on each rig, by two units on Paderborn and by at least four on XJTU-SY. On run-to-failure data an RUL error is a lifetime error. After the alarm, a constant has a lower median per-unit mean absolute error than every trained estimator we built, and most of its advantage comes from the final fifth of life, where little life remains to misjudge.

These results do not show that the task is impossible. Our geometry features clear their null on both rigs (*p* ≤ 0.011, including when the Paderborn configuration choice is repeated under every labelling), and they are two of the three attribution results, out of eleven, that survive the Benjamini–Hochberg correction (*q* ≤ 0.040). A lower bound on remaining life also passes its control with permuted lifetimes. What the results do show is that a reported accuracy or error should be compared with an appropriate reference point before it is taken as evidence. We provide the tests and thresholds for that comparison and apply them to all 31 of our own claims: six of our eleven attribution results fall inside their null, and two more clear it only before the multiplicity correction.

---

## Figures

Exported at journal resolution to `paper/figures/` — vector PDF and 600 dpi RGB TIFF (LZW), flattened onto white — by the notebook that produces each one, so a figure cannot drift from its source. A greyscale proof of every figure is in `paper/figures/greyscale-proofs/`; all five were checked for legibility without colour.

| Fig. | Content | File | Source |
|---|---|---|---|
| 1 | The eleven attribution numbers against their own null's 95th percentile, and the `LZ_frdev` speed reading against the geometry null it clears | `fig1_attribution_vs_null` | `notebooks/29_methods_and_registry.ipynb` |
| 2 | Study-size curves, both rigs, with negative controls and the region where *p* < 0.05 is unreachable | `fig2_study_size_curves` | `notebooks/29_methods_and_registry.ipynb` |
| 3 | True labelling inside the deep baseline's null, both rigs | `fig3_deep_null_both_rigs` | `notebooks/28_deep_baseline_under_the_null.ipynb` |
| 4 | Accuracy of every labelling against its operating-condition purity, network against forest | `fig4_operating_point_alignment` | `notebooks/28_deep_baseline_under_the_null.ipynb` |
| 5 | Remaining-life estimators against the degenerate baseline, per fifth of life | `fig5_remaining_life_vs_constant` | `notebooks/27_xjtu_clock_table.ipynb` |

## Data and code availability

Both datasets are public [1, 2]. The Paderborn data are available from the KAt-DataCenter of the Chair of Design and Drive Technology, Paderborn University (https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter), under a CC BY-NC 4.0 licence; the XJTU-SY data from https://biaowang.tech/xjtu-sy-bearing-datasets/. The code, notebooks and results of this paper are available at https://github.com/Mahmudulmadu/when-chance-is-hard-to-beat. All results are produced by build scripts in `src/` and read by numbered notebooks. `results/registry.csv` lists all 31 claims with their comparators; each is recomputed from the cached output of the build scripts rather than copied by hand, so every reported number can be traced to the code that produced it.

## Declarations

**Funding.** This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors. The work was carried out independently, outside the author's employment, and received no institutional or corporate support.

**Declaration of competing interest.** The author declares that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

**Declaration of generative AI and AI-assisted technologies in the manuscript preparation process.** During the preparation of this work, the author used Claude (Anthropic) for language editing, drafting assistance, programming assistance, and discussion of methodological and interpretive questions. The author independently directed the research and made the final research decisions, and all AI-assisted outputs, code, analyses, interpretations, and reported results were reviewed and verified by the author. The author takes full responsibility for the accuracy and integrity of the published work.

## References

1. B. Wang, Y. Lei, N. Li, N. Li. *A hybrid prognostics approach for estimating remaining useful life of rolling element bearings.* IEEE Transactions on Reliability 69(1):401-412, 2020. DOI 10.1109/TR.2018.2882682.
2. C. Lessmeier, J. K. Kimotho, D. Zimmer, W. Sextro. *Condition monitoring of bearing damage in electromechanical drive systems by using motor current signals of electric motors: a benchmark data set for data-driven classification.* PHM Society European Conference 3(1), 2016. DOI 10.36001/phme.2016.v3i1.1577.
3. Z. Zhao, T. Li, J. Wu, C. Sun, S. Wang, R. Yan, X. Chen. *Deep learning algorithms for rotating machinery intelligent diagnosis: an open source benchmark study.* ISA Transactions 107:224-255, 2020. DOI 10.1016/j.isatra.2020.08.010.
4. F. Jia, Y. Lei, J. Lin, X. Zhou, N. Lu. *Deep neural networks: a promising tool for fault characteristic mining and intelligent diagnosis of rotating machinery with massive data.* Mechanical Systems and Signal Processing 72-73:303-315, 2016. DOI 10.1016/j.ymssp.2015.10.025.
5. L. Wen, X. Li, L. Gao, Y. Zhang. *A new convolutional neural network-based data-driven fault diagnosis method.* IEEE Transactions on Industrial Electronics 65(7):5990-5998, 2018. DOI 10.1109/TIE.2017.2774777.
6. E. Ramasso, A. Saxena. *Review and analysis of algorithmic approaches developed for prognostics on CMAPSS dataset.* Annual Conference of the PHM Society 6(1), 2014. DOI 10.36001/phmconf.2014.v6i1.2512.
7. X. Li, Q. Ding, J.-Q. Sun. *Remaining useful life estimation in prognostics using deep convolution neural networks.* Reliability Engineering & System Safety 172:1-11, 2018. DOI 10.1016/j.ress.2017.11.021.
8. J. Zhu, N. Chen, W. Peng. *Estimation of bearing remaining useful life based on multiscale convolutional neural network.* IEEE Transactions on Industrial Electronics 66(4):3208-3216, 2019. DOI 10.1109/TIE.2018.2844856.
9. J. P. Vieira, V. A. Bauler, R. K. Rosa, D. Silva. *Towards a more realistic evaluation of machine learning models for bearing fault diagnosis.* Mechanical Systems and Signal Processing 258:114640, 2026. DOI 10.1016/j.ymssp.2026.114640. (Preprint arXiv:2509.22267, 2025.)
10. L. Wheat, M. V. Mohrenschildt, S. Habibi, D. Al-Ani. *Impact of data leakage in vibration signals used for bearing fault diagnosis.* IEEE Access 12:169879-169895, 2024. DOI 10.1109/ACCESS.2024.3497716.
11. S. Kapoor, A. Narayanan. *Leakage and the reproducibility crisis in machine-learning-based science.* Patterns 4(9):100804, 2023. DOI 10.1016/j.patter.2023.100804.
12. J. Hendriks, P. Dumond, D. Knox. *Towards better benchmarking using the CWRU bearing fault dataset.* Mechanical Systems and Signal Processing 169:108732, 2022. DOI 10.1016/j.ymssp.2021.108732.
13. O. Matania, R. Cohen, E. Bechhoefer, J. Bortman. *Test-training leakage in evaluation of machine learning algorithms for condition-based maintenance.* PHM Society European Conference 8(1):13, 2024. DOI 10.36001/phme.2024.v8i1.4125.
14. P. Knap, U. Jachymczyk, K. Lalik. *Leakage-safe, reproducible benchmarking for vibration-based fault diagnosis.* PHM Society European Conference 9(1):1-8, 2026. DOI 10.36001/phme.2026.v9i1.4924.
15. A. Saxena, J. Celaya, B. Saha, S. Saha, K. Goebel. *Metrics for offline evaluation of prognostic performance.* International Journal of Prognostics and Health Management 1(1), 2010. DOI 10.36001/ijphm.2010.v1i1.1336.
16. R. K. Rosa, D. Braga, D. Silva. *Benchmarking deep learning models for bearing fault diagnosis using the CWRU dataset: a multi-label approach.* arXiv:2407.14625, 2024.
17. W. A. Smith, R. B. Randall. *Rolling element bearing diagnostics using the Case Western Reserve University data: a benchmark study.* Mechanical Systems and Signal Processing 64-65:100-131, 2015. DOI 10.1016/j.ymssp.2015.04.021.
18. J. Zhang, G. Shen. *Protocol and implementation sensitivity in raw-index-audited few-shot bearing fault diagnosis benchmarking: evidence from CWRU and HUSTbearing.* Machines 14(8):891, 2026. DOI 10.3390/machines14080891.
19. R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, F. A. Wichmann. *Shortcut learning in deep neural networks.* Nature Machine Intelligence 2:665-673, 2020. DOI 10.1038/s42256-020-00257-z.
20. W. Ye, L. Jiang, E. Xie, G. Zheng, Y. Ma, X. Cao, et al. *The Clever Hans mirage: a comprehensive survey on spurious correlations in machine learning.* arXiv:2402.12715, 2024 (rev. 2025).
21. R. B. Randall, J. Antoni. *Rolling element bearing diagnostics - a tutorial.* Mechanical Systems and Signal Processing 25(2):485-520, 2011. DOI 10.1016/j.ymssp.2010.07.017.
22. J. Antoni. *The spectral kurtosis: a useful tool for characterising non-stationary signals.* Mechanical Systems and Signal Processing 20(2):282-307, 2006. DOI 10.1016/j.ymssp.2004.09.001.
23. J. Antoni. *Fast computation of the kurtogram for the detection of transient faults.* Mechanical Systems and Signal Processing 21(1):108-124, 2007. DOI 10.1016/j.ymssp.2005.12.002.
24. Y. Lei, B. Yang, X. Jiang, F. Jia, N. Li, A. K. Nandi. *Applications of machine learning to machine fault diagnosis: a review and roadmap.* Mechanical Systems and Signal Processing 138:106587, 2020. DOI 10.1016/j.ymssp.2019.106587.
25. C. Li, S. Zhang, Y. Qin, E. Estupinan. *A systematic review of deep transfer learning for machinery fault diagnosis.* Neurocomputing 407:121-135, 2020. DOI 10.1016/j.neucom.2020.04.045.
26. H. Shao, H. Jiang, H. Zhao, F. Wang. *A novel deep autoencoder feature learning method for rotating machinery fault diagnosis.* Mechanical Systems and Signal Processing 95:187-204, 2017. DOI 10.1016/j.ymssp.2017.03.034.
27. M. Zhao, S. Zhong, X. Fu, B. Tang, M. Pecht. *Deep residual shrinkage networks for fault diagnosis.* IEEE Transactions on Industrial Informatics 16(7):4681-4690, 2020. DOI 10.1109/TII.2019.2943898.
28. X. Li, W. Zhang, Q. Ding. *Cross-domain fault diagnosis of rolling element bearings using deep generative neural networks.* IEEE Transactions on Industrial Electronics 66(7):5525-5534, 2019. DOI 10.1109/TIE.2018.2868023.
29. E. Combrisson, K. Jerbi. *Exceeding chance level by chance: the caveat of theoretical chance levels in brain signal classification and statistical assessment of decoding accuracy.* Journal of Neuroscience Methods 250:126-136, 2015. DOI 10.1016/j.jneumeth.2015.01.010.
30. G. Varoquaux. *Cross-validation failure: small sample sizes lead to large error bars.* NeuroImage 180:68-77, 2018. DOI 10.1016/j.neuroimage.2017.06.061.
31. K. S. Button, J. P. A. Ioannidis, C. Mokrysz, B. A. Nosek, J. Flint, E. S. J. Robinson, M. R. Munafo. *Power failure: why small sample size undermines the reliability of neuroscience.* Nature Reviews Neuroscience 14(5):365-376, 2013. DOI 10.1038/nrn3475.
32. M. Ojala, G. C. Garriga. *Permutation tests for studying classifier performance.* Journal of Machine Learning Research 11(62):1833-1863, 2010.
33. Y. Lei, N. Li, L. Guo, N. Li, T. Yan, J. Lin. *Machinery health prognostics: a systematic review from data acquisition to RUL prediction.* Mechanical Systems and Signal Processing 104:799-834, 2018. DOI 10.1016/j.ymssp.2017.11.016.
34. P. Nectoux, R. Gouriveau, K. Medjaher, E. Ramasso, B. Chebel-Morello, N. Zerhouni, C. Varnier. *PRONOSTIA: an experimental platform for bearings accelerated degradation tests.* IEEE International Conference on Prognostics and Health Management (PHM'12), Denver, 2012. HAL hal-00719503.
35. L. Guo, N. Li, F. Jia, Y. Lei, J. Lin. *A recurrent neural network based health indicator for remaining useful life prediction of bearings.* Neurocomputing 240:98-109, 2017. DOI 10.1016/j.neucom.2017.02.045.
36. E. Zio. *Prognostics and health management (PHM): where are we and where do we (need to) go in theory and practice.* Reliability Engineering & System Safety 218:108119, 2022. DOI 10.1016/j.ress.2021.108119.
37. T. P. K. Nguyen, K. Medjaher. *A new dynamic predictive maintenance framework using deep learning for failure prognostics.* Reliability Engineering & System Safety 188:251-262, 2019. DOI 10.1016/j.ress.2019.03.018.
38. A. Rivas, G. K. Delipei, I. Davis, S. Bhongale, J. Yang, J. Hou. *A component diagnostic and prognostic framework for pump bearings based on deep learning with data augmentation.* Reliability Engineering & System Safety 247:110121, 2024. DOI 10.1016/j.ress.2024.110121.
39. Y.-H. Lin, P.-C. Yan, E. Zio. *Recent advances in uncertainty analysis for prognostics and remaining useful life prediction: a review.* Reliability Engineering & System Safety 269:112110, 2026. DOI 10.1016/j.ress.2025.112110.
40. A. N. Angelopoulos, S. Bates. *Conformal prediction: a gentle introduction.* Foundations and Trends in Machine Learning 16(4):494-591, 2023. DOI 10.1561/2200000101.
41. Z. Xu, E. Zio. *Uncertainty-aware prediction of remaining useful life in complex systems.* Annual Conference of the PHM Society 16(1), 2024. DOI 10.36001/phmconf.2024.v16i1.4178.
42. S. Wang, Y. Vidal, F. Pozo. *Uncertainty-aware bearing remaining useful life prediction based on conformal prediction.* PHM Society European Conference 9(1):1-10, 2026. DOI 10.36001/phme.2026.v9i1.4902.
43. S. Yang, J. Wang, Y. Wang. *Empirical calibration and conditional-reliability diagnostics for bearing RUL prediction under operating-regime shift.* arXiv:2607.08273, 2026.
44. S. Kumar, G. Cirrincione, R. R. Kumar. *Bearing remaining useful life estimation using proximal policy optimization (PPO): validation on the XJTU-SY run-to-failure dataset.* Machines 14(6):672, 2026. DOI 10.3390/machines14060672.
45. K. Hendrickx, W. Meert, Y. Mollet, J. Gyselinck, B. Cornelis, K. Gryllias, J. Davis. *A general anomaly detection framework for fleet-based condition monitoring of machines.* Mechanical Systems and Signal Processing 139:106585, 2020. DOI 10.1016/j.ymssp.2019.106585.
46. W. Zhang, G. Peng, C. Li, Y. Chen, Z. Zhang. *A new deep learning model for fault diagnosis with good anti-noise and domain adaptation ability on raw vibration signals.* Sensors 17(2):425, 2017. DOI 10.3390/s17020425.
47. J. Xu, Y. Wang, R. Xu, H. Wang, X. Zhou. *Research on open-set recognition methods for rolling bearing fault diagnosis.* Sensors 25(10):3019, 2025. DOI 10.3390/s25103019.
