# When chance is hard to beat

Code, notebooks and results for the paper

> When chance is hard to beat: exact permutation nulls and degenerate baselines for bearing fault diagnosis and remaining useful life
>
> Md. Mahmudul Hasan, Independent Researcher, Dhaka, Bangladesh
>
> ORCID: [0009-0001-0778-064X](https://orcid.org/0009-0001-0778-064X)

The paper is [`paper/when-chance-is-hard-to-beat.pdf`](paper/when-chance-is-hard-to-beat.pdf). The same text is in Markdown in [`paper/manuscript.md`](paper/manuscript.md).

## Summary

Bearing fault diagnosis and remaining-useful-life (RUL) prediction are mostly evaluated on a few public datasets with at most a few dozen bearings each. The paper measures two reference points for reading results on such datasets.

- Chance level at the real study size. We enumerate every class-preserving labelling of the bearings (165 on XJTU-SY, 462 on Paderborn) and score each one with the same bearing-held-out protocol. With eleven bearings, the 95th percentile of accuracy under labellings with no physical meaning is 69.1–76.4 %, depending on the model and its input.
- A baseline for errors in minutes. On run-to-failure data, an RUL error and a total-lifetime error are the same number. After the alarm, a constant "no life left" prediction has a lower mean absolute error than every trained estimator we built.

We apply the same tests to our own results. All 31 claims in the paper are listed with their comparators in [`results/registry.csv`](results/registry.csv), and six of the eleven attribution results fall inside their own null. The Paderborn feature set and operating condition were chosen on the same bearings the null is computed on. `src/build_selection_null.py` repeats that choice under every labelling, which moves the Paderborn geometry result from p = 0.0043 to p = 0.011 (Section 4.2 of the paper).

## Repository layout

```
paper/
  when-chance-is-hard-to-beat.pdf   the paper
  manuscript.md                     the same text in Markdown
  figures/                          the five figures (vector PDF)
results/
  registry.csv                      every claim in the paper with its comparator, recomputed from the caches
src/
  download_*.py                     download the two public datasets
  build_*.py                        feature builders and the analyses the paper reports
  verify_references.py              checks each reference in paper/manuscript.md against Crossref or arXiv
notebooks/
  11-29                             the notebooks behind the paper, in the order the work was done
```

## Reproducing the results

### Environment

Python 3.13. Install the requirements. The extra index gives the CPU build of PyTorch, which is the build the reported results were produced with:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### Data

Both datasets are public: the Paderborn University bearing dataset (Lessmeier et al., 2016) and XJTU-SY (Wang et al., 2020). `src/download_paderborn.py` and `src/download_xjtu.py` download them into `data/raw/`.

### Caches and analyses

The docstring of each `build_*.py` script says which notebook uses it and which files it writes to `data/processed/`. Run the notebooks in numerical order, and run any `build_*.py` script a notebook reads from before that notebook. The permutation nulls are the slow part: the exact nulls of the deep network took several hours on a 10-core laptop CPU.

### Determinism

On our CPU, PyTorch did not give identical results at different thread counts. Every network result that is compared with a null was produced with `--threads 5`, so pass `--threads 5` to `src/build_cnn_null.py` (its default is 8) to reproduce them exactly. `src/build_cnn_summary.py` stops if a quantity that is scored twice disagrees.

### Registry

`python src/build_registry.py` recomputes all 31 claims from the caches and writes `results/registry.csv`. It reads the output of `src/build_selection_null.py`, so run that first (about 20 minutes on 8 cores).

## Not included

- The datasets and the derived caches. Both datasets are public, and the scripts above rebuild the caches.
- Exploratory notebooks 01–10. They cover dataset loading and a first artificial-to-real transfer study. They led to this work, but no result in the paper comes from them. The last code cell of notebook 11 makes a comparison with a cache they write; no number in the paper depends on it.
- Follow-up work. Notebook 30, on document retrieval over the rigs' documentation, and its scripts belong to a separate paper. Notebook 28 mentions it in one correction note; nothing in this paper depends on it.

## Licence

MIT, see [`LICENSE`](LICENSE).
