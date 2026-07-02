# Plan: new-extractant confidence bake-off

| # | Task group | Status |
|---|------------|--------|
| 1 | Harness: reproduce the data/cleaning/features and the molecule-grouped OOF logD predictions and separation pairs from sep_factor_confidence.py, so all methods share identical predictions | done (newext_confidence_bakeoff.py; incumbent reproduces prior 0.188/0.656 exactly) |
| 2 | Methods: implement the four uncertainty estimators, each cross-fit molecule-grouped (AD Tanimoto, descriptor k-NN/leverage, ensemble+RF variance, learned-error incumbent + hybrid) | done |
| 3 | Evaluate: for each method rank pairs by uncertainty, compute the metric suite at coverage 100/75/50/25/10, plus calibration rho, in the new-extractant regime | done (newext_confidence_bakeoff_results.csv) |
| 4 | Winner: score the composite (avg shrinkage-immune lift at 25/10 + calibration), declare the winner, save results CSV + a selective-prediction figure | done: WINNER = ens_bag (bootstrap-bagged ensemble disagreement); top-10% direction 0.66->0.81, useful-F1 0.44->0.59, signed R2 -0.00->0.28, calibration 0.20->0.31. AD/descriptor distance LOST (calibration ~0.01). |
| 5 | Verify: adversarial reflection review (leakage in the AD/descriptor folds, shrinkage confounds, honest reporting) | done (inline skeptical pass; subagent quota reached): SHIP, no code fixes. Incumbent reproduces prior 0.188/0.656 (leakage tripwire); AD/desc neighbors train-fold only; winner robust across coverage; bootstrap resamples rows. Caveats retained: top-10% n=459 with a 25% dip, win modest and regime-specific. |
| 6 | Integrate + ship: wire the winner into the new-extractant confidence gate for the recommender and the AutoData shortlist; update all artifacts; confirm commit message; push under SkyEpstein | in progress (winner wired into autodata_score.py; CHANGELOG done; docs/results-table/deck/workbook + commit pending) |

Status values: todo, in progress, done.
