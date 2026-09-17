# PCA160-SRC model card

## Intended use

Research benchmarking and peptide prioritization for canonical 5--50-residue sequences under the historical AntiDMPpred label definition. This model is not a clinical tool and does not establish antidiabetic activity experimentally.

## Representation and dictionary

- Input: 520 AAC, DPC, type-I PseAAC, and CKSAAGP descriptors.
- Transform: training-fitted standardization, 160-component PCA, and row L2 normalization.
- Dictionary: every labeled training peptide is one atom.
- Deployment dictionary: 472 atoms in 160 dimensions; 236 atoms per historical class.
- Deployment overcompleteness ratio: 2.95.
- Sparse solver: L1 Lasso-LARS, alpha 0.001.
- Decision: the class with the smaller reconstruction residual.
- SMOTE/class weighting: not used because the benchmark is exactly balanced.

The deployment representation and dictionary are fitted to all 472 peptides only after evaluation. All reported scores come from held-out out-of-fold predictions.

## Performance

Repeated five-by-five direct stratified CV:

- Accuracy: 0.791 +/- 0.009
- MCC: 0.582 +/- 0.017
- F1: 0.795 +/- 0.008
- AUC: 0.881 +/- 0.007

Repeated CD-HIT50 grouped CV:

- Accuracy: 0.714 +/- 0.016
- MCC: 0.429 +/- 0.031
- F1: 0.716 +/- 0.016
- AUC: 0.791 +/- 0.012

In the common-width comparison, all six SRC representations produced 160 coordinates. PCA160-SRC achieved the highest accuracy and MCC. CKSAAP-KPCA160-L1-SRC matched its rounded AUC, while AWMKPCA160-L1-SRC had a marginally higher F1 but lower specificity and AUC.

## Dimension selection

Nine dimensions from 16 to 240 were evaluated. F1 plateaued from approximately 128 dimensions onward. PCA240 had the highest mean F1 by 0.0013, but almost twice the F1 standard deviation of PCA160. PCA160 was selected as the balanced operating point because it tied the highest accuracy, achieved the highest MCC and AUC, and had the lowest F1 variability in the plateau.

## Limitations

- Historical controls are AVPdb-derived antiviral peptides, not target-specific assay-confirmed inactive peptides.
- No independent prospective cohort was available.
- Dimension selection is exploratory and was conducted on the development benchmark.
- Output is a historical class score, not a calibrated biological probability.
- Homology-aware scores are lower than direct stratified-CV scores.
