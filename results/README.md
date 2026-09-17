# Saved results

Each experiment directory can contain:

- `fold_metrics.csv`: metrics for each outer fold;
- `repeat_metrics.csv`: metrics from the complete out-of-fold vector for each seed;
- `aggregate_metrics.csv`: mean and sample standard deviation across seeds;
- `oof_predictions.csv`: held-out predictions with fold and seed identifiers;
- `inner_selection.csv`: training-only hyperparameter selections;
- `metadata.json`: protocol and implementation metadata;
- representation-specific audit files such as dictionary structure or kernel weights.

The principal release directories are:

- `revised_direct_comparators_5x5/`
- `revised_src_variants_160_5x5/`
- `revised_pca_dimension_ablation_extended_5x5/`
- `revised_pca160_cdhit50_5x5/`
- `stability_cluster_5x5/` (AntiDMPpred-RF grouped comparator)

Run `python scripts/verify_revised_release.py` to check the frozen numerical values, model configuration, and result completeness.
