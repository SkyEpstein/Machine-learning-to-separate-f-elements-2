# Manuscript figures and tables

A curated, publication-ready set for the writeup. The full figure set is in `figures/` and every number is in `docs/RESULTS_TABLE.md`. Per project rule, our model's confidence-filtered figure is shown alongside each headline, since the calibrated confidence layer is the differentiator (Dr. Zhang's model has none).

## Headline results (lead the paper with these)

| Quantity | Overall | Our model + confidence (top 10%) | Regime |
|---|---|---|---|
| Track A logD (new molecule) | R2 0.466, RMSE 1.148 | R2 0.912, RMSE 0.493 | molecule-grouped CV |
| Track B logD (known molecule) | R2 0.61, RMSE 0.98 | R2 0.874, RMSE 0.430 | condition-key grouped CV |
| Free energy delta G (per-pair) | R2 0.46 +/- 0.01, RMSE 6.3 kJ/mol | RMSE 3.6 kJ/mol (R2 0.776) | molecule-grouped CV |
| 90% interval coverage | 0.902 (target 0.90) | n/a | molecule-disjoint conformal |
| Separation factor (known extractant) | signed R2 0.356, direction 0.726 | direction 0.789, Spearman 0.776, norm RMSE 0.64 | condition-key, matched conditions |
| Separation factor (new extractant) | signed R2 0.188, direction 0.656 | confidence not net-useful | molecule-grouped CV |
| Separation, Zhang XGBoost (known) | signed R2 0.369, direction 0.715 | none (his model has no confidence layer) | condition-key, matched conditions |

## Main figures (narrative order)

1. `figures/pred_vs_actual_a.png` - Track A, new-extractant screening: predicted vs actual logD under molecule-grouped cross-validation. Overall R2 0.466 (RMSE 1.148); on the most confident 10 percent, R2 0.912 (RMSE 0.493).
2. `figures/pred_vs_actual_b.png` - Track B, condition optimization for a known extractant: predicted vs actual logD. Honest condition-key R2 0.61 (RMSE 0.98); most confident 10 percent R2 0.874 (RMSE 0.430).
3. `figures/confidence_curve_a.png` - Accuracy concentrates with confidence (Track A): error falls and R2 rises as predictions are restricted to the most confident fraction. The calibrated confidence layer is the model's deployable contribution.
4. `figures/conformal_calibration.png` - Calibrated 90 percent prediction intervals: empirical coverage 0.902 on a molecule-disjoint split, so the uncertainty is trustworthy rather than nominal.
5. `figures/dg.png` - Free energy of extraction (delta G), per-pair framing: R2 0.46 plus or minus 0.01 (RMSE 6.3 kJ/mol); most confident 10 percent RMSE 3.6 kJ/mol. RandomForest chosen by a seven-model bake-off.
6. `figures/sep_factor_eval.png` - Separation factor between two f-elements (the objective), differencing the logD model: direction is predicted moderately (0.73 known, 0.66 new) and magnitude poorly. The direct delta model failed and is not used.
7. `figures/sep_factor_confidence.png` - Separation prediction improves with confidence only for a known extractant (direction 0.73 to 0.79; normalized error beats a random-ranking benchmark, adversarially verified). For a new extractant confidence is not net-useful.
8. `figures/sep_factor_zhang.png` - Our model vs Dr. Zhang's XGBoost on the separation factor (same data): they tie at full coverage, but only our model carries a confidence layer, so only ours has the confidence-filtered operating point (direction 0.789).
9. `figures/zhang_comparison_selective.png` - Comparison with Dr. Zhang on logD (same data): about equal in accuracy; our addition is the calibrated, confidence-aware prediction, not higher raw accuracy.
10. `figures/active_analysis_trends.png` - Experiment prioritization: ranking candidates by prediction concentrates strong extractants near the top (81 percent of top picks in the strongest tertile, 3.6x random).
11. `figures/by_metal_pair_separation.png` - Per-pair separation accuracy is highly uneven across f-element pairs: large-gap pairs predict well, adjacent rare earths do not.

## Supplementary figures

The remaining figures in `figures/` support the above (per-metal and per-pair detail, the UCB and active-learning panels, ensemble weights, coverage illustration, and the Zhang head-to-head splits):

- `figures/pred_vs_actual stragglers`
- `figures/confidence_curve_b.png`
- `figures/coverage_illustration.png`
- `figures/ensemble_weights.png`
- `figures/by_metal_accuracy.png`
- `figures/by_metal_confidence_vs_error.png`
- `figures/by_pair_confidence_vs_error.png`
- `figures/confidence_per_metal.png`
- `figures/active_analysis.png`
- `figures/active_learning_ucb.png`
- `figures/ucb_graphs.png`
- `figures/picks_trends.png`
- `figures/our_data_vs_zhang_data.png`
- `figures/zhang_his_split_headtohead.png`
- `figures/sep_factor_confidence.png (detail)`

Combined main set: `figures/manuscript_figures.pdf` (11 figures).

