# ADP-SRC: sparse representation classification of anti-diabetic peptides

Reference implementation and reproducibility package for **ADP-SRC (PCA160-SRC)**, a sparse representation classifier evaluated on the historical 472-peptide AntiDMPpred benchmark.

The repository contains:

- the exact benchmark FASTA and CD-HIT50 grouping file;
- leakage-safe feature extraction, model selection, and repeated cross-validation code;
- the proposed PCA160-SRC model and conventional/SRC comparators;
- saved fold, repeat, aggregate, and out-of-fold results;
- the fitted deployment model and a FASTA prediction command;
- manuscript LaTeX, bibliography, tables, figures, and compiled PDF;
- automated integrity checks and tests.

> **Important label caveat:** the 236 benchmark controls are AVPdb-derived antiviral peptides, not peptides experimentally verified to be inactive against every antidiabetic target. Model outputs therefore describe membership in the historical benchmark class and are **not calibrated probabilities of biological activity**.

## Method at a glance

1. Encode each 5â€“50-residue peptide using 520 AAC, DPC, type-I PseAAC, and CKSAAGP descriptors.
2. Fit standardization and PCA using training data only.
3. Retain 160 principal components and L2-normalize every embedding.
4. Use all training peptides as labeled dictionary atoms.
5. Estimate sparse coefficients with L1-regularized coding.
6. Predict the class with the smaller class-specific reconstruction residual.

An outer five-fold training partition contains 377 or 378 atoms in 160 dimensions, yielding an average atom-to-dimension ratio of 2.36. The all-data deployment dictionary contains 472 atoms (ratio 2.95).

## Main results

Values are mean Â± sample standard deviation across five partition seeds, each using stratified five-fold cross-validation.

| Analysis | Model | Accuracy | MCC | F1 | ROC AUC |
|---|---|---:|---:|---:|---:|
| Direct protocol | **PCA160-SRC** | **0.791 Â± 0.009** | **0.582 Â± 0.017** | 0.795 Â± 0.008 | **0.881 Â± 0.007** |
| Direct protocol | RBF-SVM | 0.775 Â± 0.008 | 0.550 Â± 0.017 | 0.773 Â± 0.009 | 0.844 Â± 0.003 |
| Direct protocol | AntiDMPpred-RF reimplementation | 0.730 Â± 0.012 | 0.461 Â± 0.024 | 0.739 Â± 0.013 | 0.806 Â± 0.005 |
| Common-width SRC variants | AWMKPCA160-L1-SRC | 0.784 Â± 0.013 | 0.572 Â± 0.027 | **0.796 Â± 0.014** | 0.864 Â± 0.010 |
| Common-width SRC variants | CKSAAP-KPCA160-L1-SRC | 0.784 Â± 0.005 | 0.570 Â± 0.010 | 0.779 Â± 0.005 | 0.881 Â± 0.006 |
| CD-HIT50 robustness | **PCA160-SRC** | **0.714 Â± 0.016** | **0.429 Â± 0.031** | **0.716 Â± 0.016** | **0.791 Â± 0.012** |
| CD-HIT50 robustness | AntiDMPpred-RF reimplementation | 0.682 Â± 0.011 | 0.364 Â± 0.022 | 0.691 Â± 0.010 | 0.765 Â± 0.006 |

AntiDMPpred values are regenerated, same-fold results from this codebase; they are not copied from the published paper.

## Repository structure

```text
adp_src/          Python implementation
data/             Benchmark FASTA and CD-HIT50 groups
final_model/      Serialized inference models and metadata
results/          Fold-, repeat-, aggregate-, and OOF-level outputs
scripts/          Reproduction, verification, inference, and paper scripts
tests/            Unit and release-integrity tests
paper_revised/    Canonical LaTeX manuscript, figures, tables, and PDF
config.json       Frozen experiment and model-selection record
```

## Installation

### Conda (recommended)

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd ADP-SRC
conda env create -f environment.yml
conda activate adp-src-deeplse
python -m pip install -e .
```

### Python virtual environment

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,neural]"
```

PyTorch is optional because the proposed PCA160-SRC model and inference command do not require it. The `neural` extra enables DeepLSE and autoencoder experiments. For a platform-specific CPU or CUDA build, install PyTorch using its official instructions first and then run `python -m pip install -e ".[test]"`.

## Verify the release

Run the tests and frozen-result audit from the repository root:

```bash
python -m pytest -q
python scripts/verify_revised_release.py
```

If PyTorch is not installed, the DeepLSE unit test is reported as skipped; all PCA-SRC, kernel-SRC, data, result, and release checks still run. The Conda environment includes CPU-only PyTorch and runs the complete suite.

Successful verification ends with:

```text
ADP_PCA160_SRC_COMMON_WIDTH_160_VERIFIED
```

The audit checks the dataset SHA-256 hash, expected repeated-CV outputs, model-selection record, headline metrics, common-width comparison, CD-HIT50 results, dictionary dimensions, saved model, and manuscript assets.

## Predict new peptides

Prepare a FASTA containing canonical amino acids and sequences 5â€“50 residues long:

```text
>candidate_1
VLGPQ
>candidate_2
KLPVPQ
```

Then run:

```bash
python scripts/predict_pca_src.py \
  --model final_model/pca160_src/pca160_src_model.npz \
  --input candidates.fasta \
  --output candidate_predictions.csv
```

The output reports the predicted historical class, residual-derived score, class-specific residuals, and number of nonzero dictionary atoms. Treat predictions as candidate-ranking evidence requiring experimental validation.

## Reproduce the study

### 1. Conventional classifiers and PCA-dimension ablation

```bash
bash scripts/run_revised_classical_study.sh
```

This regenerates:

- the direct repeated five-by-five conventional comparison;
- the 16â€“240 component PCA-SRC ablation;
- the PCA160-SRC CD-HIT50 grouped robustness analysis.

### 2. Common-width representation study

```bash
bash scripts/run_revised_representation_study.sh
```

All variants use a 160-dimensional output representation: PCA-SRC, DeepLSE-SRC, autoencoder-SRC, CKSAAP-KPCA with matching pursuit or L1 coding, and alignment-weighted multikernel KPCA-SRC. Neural training can take substantially longer than the classical experiment and can use a CUDA device when available.

### 3. Fit the deployment model

```bash
python scripts/finalize_pca_src.py \
  --dimension 160 \
  --output-dir final_model/pca160_src
```

The serialized model is fitted on all 472 benchmark peptides for future inference. Reported paper performance comes exclusively from held-out outer-fold predictions, not from this all-data model.

### 4. Rebuild tables, figures, and manuscript

```bash
python scripts/build_revised_paper_outputs.py
bash scripts/build_revised_manuscript.sh
```

The LaTeX build requires `pdflatex`, `bibtex`, and `pdfinfo`. The resulting PDF is written to `paper_revised/build/manuscript.pdf`.

## Experimental design

- **Primary protocol:** `StratifiedKFold(n_splits=5, shuffle=True)` repeated with seeds 6, 17, 29, 41, and 53.
- **Robustness protocol:** CD-HIT50 groups with repeated `StratifiedGroupKFold` using the same seeds.
- **Nested fitting:** scaling, PCA/representation learning, sparse-penalty selection, and dictionary construction are restricted to training data.
- **Summary:** metrics are first calculated from the complete out-of-fold prediction vector for each seed and then reported as mean Â± sample standard deviation over five seeds.
- **No SMOTE:** the historical benchmark is exactly balanced (236/236).

## PCA-dimension selection

The tested grid is `{16, 32, 64, 96, 128, 160, 192, 224, 240}`. PCA240 has the literal highest mean F1 (0.7960), but PCA160 was selected as the stability-aware operating point because it is within 0.005 F1 of the maximum, ties the highest mean accuracy, has the highest mean MCC and AUC, and has lower F1 variability than PCA240. The selection rule is frozen in `config.json`.

## Reproducibility notes

- Benchmark SHA-256: `9371de76e93319d8f12cf6ec6ce1ad3b565476cf2cd1cfdcef4422574b132edb`
- Cross-validation seeds: `6, 17, 29, 41, 53`
- Final PCA-SRC L1 penalty: `lambda = 0.001`
- L1 candidates: `0.001, 0.005, 0.01, 0.02`
- Saved OOF predictions are available for the principal direct comparisons.
- PyTorch models request deterministic algorithms; small numerical differences may still occur across hardware, BLAS libraries, or PyTorch releases.

## Scientific limitations

- The benchmark is small and lacks an independent prospective test set.
- Its controls are source-defined antiviral peptides, not universal assay-confirmed negatives.
- Repeated CV measures sensitivity to fold assignment; its standard deviation is not a confidence interval over independent datasets.
- PCA-dimension selection is exploratory and uses the development benchmark.
- CD-HIT50 is a secondary homology-aware stress test, not the primary historical comparison.
- Predictions require biochemical and biological validation.

## Manuscript and citation

The canonical manuscript source is `paper_revised/manuscript.tex`; bibliographic metadata are in `paper_revised/references.bib`, and source checks are summarized in `paper_revised/citation_audit.csv`.

If you use this repository, cite the accompanying manuscript and the archived software release. Citation metadata are provided in `CITATION.cff`.

## Contributing

Bug reports and reproducibility issues are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## License

Released under the [MIT License](LICENSE).
