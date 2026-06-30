# Changelog

A reverse-chronological log of major milestones and decisions. Numbers are the audited, honest values.

## 2026-06-30 Audit, corrections, and documentation
- Ran a full adversarial audit (two passes plus an independent reproduction). Verdict: fundamentally honest, with genuine molecule-grouped cross-validation, no feature-target leakage, and calibrated intervals.
- Corrected optimistic framing. Track B logD honest R-squared about 0.61 (the random-row 0.725 is an upper bound that includes replicate memorization; about 0.44 on new molecules). Free energy (delta G) about 0.46 plus or minus 0.01 (0.473 was a selected maximum). The Zhang comparison was reframed to the same-data result.
- Restored the real source over eleven stub scripts so the repo is reproducible.
- Added METHODS_AND_RESULTS.md (the full end-to-end explanation), the project constitution, the Audit corrections workbook tab, the active-analysis and pick trends, and the mentor-writeup outline.

## 2026-06-28 Free-energy model and active analysis
- Added the free-energy (delta G) model and the seven-model bake-off; RandomForest won and the NNLS stack only tied it for that target.
- Added UCB active analysis, experiment triage, sequential active learning, the pseudo-labeling negative result, and the calibrated-interval coverage check.

## Earlier logD tracks, confidence, and Zhang
- Built the two logD tracks (A new-molecule, B known-molecule), the NNLS-stacked ensemble, the confidence layer, the per-metal analysis, the classifiers, and the comparison with Dr. Zhang.
