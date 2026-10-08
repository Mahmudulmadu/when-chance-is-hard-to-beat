# When chance is hard to beat

Code, notebooks and results for the paper

> **When chance is hard to beat: exact permutation nulls and degenerate baselines for bearing fault diagnosis and remaining useful life**
> Md. Mahmudul Hasan, Independent Researcher

The manuscript is in [`paper/manuscript.md`](paper/manuscript.md).

## What the paper shows

Bearing fault diagnosis and remaining-useful-life (RUL) prediction are evaluated largely on a few public datasets with at most a few dozen bearings each. The paper measures two reference points for interpreting results on those datasets:

- **Chance at the actual study size.** An exact, bearing-level label-permutation test, enumerating every class-preserving labelling (165 on XJTU-SY, 462 on Paderborn), shows that on eleven bearings the 95th percentile of accuracy over labellings with no physical meaning is **69.1-76.4 %**, and that this band depends on the model and its input.
- **A degenerate baseline for errors in minutes.** On run-to-failure data an RUL error and a total-lifetime error are the same quantity, and after the alarm a constant "no life left" prediction has a lower mean absolute error than every trained estimator.

The same standard is applied to the paper's own results: all 31 claims are checked against their comparators in [`results/registry.csv`](results/registry.csv), and six of the eleven attribution results fall inside their own null. The Paderborn feature set and operating condition were chosen on the same bearings the null is computed on; `src/build_selection_null.py` repeats that choice under every labelling, which moves the Paderborn geometry result from p = 0.0043 to p = 0.011 (paper, Section 4.2).

## Repository layout

```
paper/
  manuscript.md          the paper
  figures/               the five figures, vector PDF
results/
  registry.csv           every claim in the paper beside its comparator; all recomputed from cache
src/
  download_*.py          fetch the two public datasets
  build_*.py             feature builders and every analysis the paper reports
  verify_references.py   checks each reference against Crossref / arXiv
notebooks/
  11-29                  the measurement notebooks behind the paper, in the order the work was done
```

## Reproducing the results

**Environment.** Python 3.13. Install the requirements, and PyTorch from the CPU index, which is what the reported results were produced on:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

**Data.** Both datasets are public and free: the Paderborn University bearing dataset [Lessmeier et al. 2016] and XJTU-SY [Wang et al. 2020]. `src/download_paderborn.py` and `src/download_xjtu.py` fetch them into `data/raw/`.

**Caches and analyses.** Every script in `src/` states in its docstring which notebook it serves and which files it writes to `data/processed/`. Run the notebooks in numerical order, running any `src/build_*.py` a notebook reads from before that notebook. The permutation nulls are the expensive part: the exact deep-network nulls take several hours on a 10-core CPU.

**Determinism.** In our CPU environment PyTorch did not give identical results at different thread counts, so every network result compared against a null was produced at `--threads 5`; use the same setting to reproduce them exactly. The scripts check that any quantity computed twice agrees.

**The registry.** `python src/build_registry.py` recomputes all 31 claims from the caches and writes `results/registry.csv`. It reads the output of `src/build_selection_null.py`, so run that first (about 20 minutes).

## Not included

- **The datasets and derived caches**, which are public and rebuilt by the scripts above.
- **Earlier exploratory notebooks (01-10).** Dataset loading and a first artificial-to-real transfer study, which led to this work but carry no result it reports. One comparison cell at the end of notebook 11 reads a cache they produce; no number in the paper depends on it.
- **Follow-up work.** A later notebook on document retrieval over the rigs' documentation, and the scripts behind it, belong to a separate paper; they are referred to in one or two notebooks here and are not needed for any result in this one.

## Licence

MIT - see [`LICENSE`](LICENSE).
