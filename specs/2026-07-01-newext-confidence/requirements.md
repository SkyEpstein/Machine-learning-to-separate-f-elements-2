# Requirements: a confidence algorithm that works for NEW extractants

## Why
The deployed confidence layer (a learned-error model predicting the OOF logD residual) sharpens
separation prediction for KNOWN extractants (top-10% signed R2 0.36 -> 0.59, RMSE 1.29 -> 0.37,
direction 73% -> 79%) but does NOT work for NEW extractants: at top-10% coverage signed R2 goes
0.19 -> -0.00 and direction 66% -> 59% (the RMSE drop is just variance shrinkage). The AutoData
candidates are by definition new extractants, so we cannot trust "keep the most confident" for
them. The learned-error model does not measure novelty/extrapolation, which is what drives error
on unseen chemistry. We need a confidence estimator selected by a bake-off that genuinely ranks
new-extractant separation predictions by reliability.

## Scope
- In: a bake-off of uncertainty estimators evaluated in the molecule-grouped (new-extractant)
  regime on the separation task, using the SAME OOF logD predictions so only the confidence
  ranking differs. Pick a winner by a fixed composite criterion, wire it into the recommender
  and the AutoData shortlist confidence gate.
- Out: changing the logD point predictions; the known-extractant confidence layer (unchanged,
  it already works); wet-lab validation.

## Key decisions (resolved with Skyler, 2026-07-01)
- Winner metric: COMPOSITE. Best average lift in the shrinkage-immune metrics (direction accuracy,
  Spearman, useful-F1) at top-25% and top-10% coverage, plus calibration (Spearman of predicted
  uncertainty vs actual |error|), in the molecule-grouped new-extractant regime.
- Methods to bake off (all four): (1) applicability domain by ECFP Tanimoto novelty distance
  (1 - max Tanimoto to training extractants); (2) descriptor-space distance (k-NN / leverage in
  standardized extractant-descriptor space); (3) ensemble disagreement (bootstrap-bagged logD
  variance and Random Forest tree variance); (4) the incumbent learned-error model plus a HYBRID
  that adds the AD + descriptor distances as features to the error model.

## Context and constraints (honest)
- Report both R2 and RMSE, and lead with the shrinkage-immune ranking metrics (direction, Spearman,
  useful-F1), since selective-prediction R2/RMSE partly reflect variance shrinkage.
- Everything is molecule-grouped so there is no extractant leakage; every uncertainty is cross-fit
  on the same disjoint folds. If no method beats the incumbent in the new-extractant regime, we say
  so honestly rather than manufacture an improvement.
