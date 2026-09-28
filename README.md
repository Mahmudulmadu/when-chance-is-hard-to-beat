# When chance is hard to beat

Code, notebooks and results for the paper

> **When chance is hard to beat: exact permutation nulls and degenerate baselines for bearing fault diagnosis and remaining useful life**
> Md. Mahmudul Hasan, Independent Researcher

The manuscript is in [`paper/manuscript.md`](paper/manuscript.md).

## What the paper shows

Bearing fault diagnosis and remaining-useful-life (RUL) prediction are evaluated almost entirely on a few public datasets with ten to twenty bearings each. The paper measures two reference points that results on those datasets are rarely compared against:

- **Chance at the actual study size.** An exact, bearing-level label-permutation test, enumerating every class-preserving labelling (165 on XJTU-SY, 462 on Paderborn), shows that a labelling with no physical meaning reaches a 95th-percentile accuracy of **69.1-76.4 %** on eleven bearings, and that this band depends on the model and its input.
- **A degenerate baseline for errors in minutes.** On run-to-failure data an RUL error and a total-lifetime error are the same quantity, and a constant "no life left" beats every trained estimator after the alarm.

The same standard is applied to the paper's own results: all 31 claims are priced against their comparators in [`results/registry.csv`](results/registry.csv), and six of the eleven attribution results fall inside their own null.

## Repository layout

```
paper/
  manuscript.md          the paper
  figures/               the five figures, vector PDF
results/
  registry.csv           every claim in the paper beside its comparator; all recomputed from cache
src/
  download_*.py          fetch the public datasets
  build_*.py             feature builders and every analysis the paper reports
  verify_references.py   checks each reference against Crossref / arXiv
notebooks/
  01-29                  the measurement notebooks, in the order the work was done
```

## Reproducing the results

**Environment.** Python 3.13. Install the requirements, and PyTorch from the CPU index, which is what the reported results were produced on:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

**Data.** Both datasets are public and free: the Paderborn University bearing dataset [Lessmeier et al. 2016] and XJTU-SY [Wang et al. 2020]. `src/download_paderborn.py` and `src/download_xjtu.py` fetch them into `data/raw/`.

**Caches and analyses.** Every script in `src/` states in its docstring which notebook it serves and which files it writes to `data/processed/`. Run the notebooks in numerical order, running any `src/build_*.py` a notebook reads from before that notebook. The permutation nulls are the expensive part: the exact deep-network nulls take several hours on a 10-core CPU.

**Determinism.** CPU PyTorch is not bit-stable across thread counts, so every network result compared against a null was produced at `--threads 5`; use the same setting to reproduce them exactly. The scripts assert that any quantity computed twice agrees.

**The registry.** `python src/build_registry.py` recomputes all 31 claims from the caches and writes `results/registry.csv`.

## Not included

- **The datasets and derived caches**, which are public and rebuilt by the scripts above.
- **Follow-up work.** A later notebook on document retrieval over the rigs' documentation, and the scripts behind it, belong to a separate paper; they are referred to in one or two notebooks here and are not needed for any result in this one.

## Licence

MIT - see [`LICENSE`](LICENSE).
