# Methods and results

This document explains the project end to end. It covers the data and features, the two logD tracks and the model sweep, the free-energy (delta G) model, the confidence and conformal layers, the per-metal behavior, the sign and separation analyses, the active analysis and the candidate-experiment screen, the classifiers, the comparison with Dr. Zhang, the approaches that were tried and rejected, the adversarial audit, and how to reproduce everything. All numbers are the audited, honest values; where an earlier figure was optimistic the corrected one is used and the reason is stated.

## Contents
1. Overview and data
2. The two logD tracks
3. logD model sweep and the ensemble-versus-single reconciliation
4. Free energy of extraction (delta G)
5. Confidence
6. Per-metal and per-pair accuracy and confidence
7. Sign correctness and separation
8. Active analysis
9. UCB candidate screen
10. Classification
11. Comparison with Dr. Zhang's model
12. Rejected approaches
13. Audit
14. Reproducibility

## Overview and data

This project uses machine learning, written with LLM assistance, to predict how well a given extractant separates two f-elements in a liquid-liquid extraction. The quantity being predicted is logD, the base-10 logarithm of the distribution coefficient D, which measures how strongly a metal partitions into the organic phase. Two neighboring f-elements are separated well when their logD values differ, so a model that predicts logD across extractants, metals, and conditions is in effect a model of separation potential. A companion framing predicts the Gibbs free energy of extraction, delta G = -2.303 R T logD (in kJ/mol, with R the gas constant and T the temperature in kelvin), so a favorable extraction gives a negative, spontaneous delta G; that framing is meant to capture the thermodynamics of an extractant and a metal independent of the particular acid and temperature.

### Dataset

The raw dataset is 8,075 rows spanning 295 distinct extractant molecules and 28 f-element metals (the lanthanides plus Am, Cm, Cf, Np, Pu, Pa, Th, and U). It is split into two molecule-disjoint files, `Training_Data_V27.csv` for training and validation and `Testing_Data_V39.csv` as a held-out test set, so that no extractant molecule appears in both. Because the table is wide but shallow in distinct chemistry (only 295 molecules), interpolating new conditions for a molecule already seen is much easier than extrapolating to a genuinely new molecule. That asymmetry is why results are reported on two separate tracks rather than a single number.

### Feature groups

Each row is described by four kinds of input, and each group was included because it drives part of the extraction chemistry rather than to pad the feature count.

**Extractant structure.** The extractant enters as a SMILES string. From it the code computes a Morgan (circular) fingerprint encoding which substructures are present, a block of RDKit molecular descriptors covering size, shape, lipophilicity, and ring and functional-group counts, and a few simple ligand descriptors: the counts of oxygen, nitrogen, sulfur, and phosphorus donor atoms plus the calculated logP. These describe how the extractant can bind a metal, since the donor atoms and the surrounding structure set both the strength and the selectivity of that binding.

**Metal descriptors.** Each metal is represented by numeric properties rather than only a name: atomic number, ionic radius, oxidation state, metallic radius, the first three ionization energies, Pauling electronegativity, density, melting and boiling points, and the metal concentration. The ionic radius and charge drive the lanthanide-contraction trend that makes neighboring rare earths so chemically alike, and giving the model these numbers lets it relate one metal to another along that trend, which a one-hot label cannot do.

**Acid and process conditions.** The acid type and concentration, the temperature, the extractant concentration, and the diluent volume fractions are included because logD is an equilibrium quantity. It shifts with acid concentration, with extractant concentration, and with pH, so the conditions are as important as the molecule itself. These condition knobs are kept for the logD models and deliberately dropped for the free-energy framing, where the target is meant to be a structure-and-metal property.

**Diluent (solvent) descriptors.** The two diluent components are described by molar mass, logP, boiling and melting points, density, water solubility, and dipole moment. The diluent sets the organic-phase environment and affects how the extractant aggregates and how well it pulls the metal into the organic phase.

### Cleaning and the label-noise floor

The labels carry measurement noise and some outright contradictions, so the data is quality-controlled before any model is fit. Exact duplicate rows are removed, and any replicate group (the same molecule, metal, and conditions) whose logD spans more than 2 log units is dropped as internally contradictory. This leaves 7,066 rows. Replicate scatter in the surviving data implies an irreducible label-noise floor of roughly 0.45 log units (the raw, uncleaned data sits nearer RMSE 0.77). Because cleaning removes the most contradictory measurements, every score reported downstream is computed on this cleaned set and should be read as an upper bound relative to the raw measurements, not as performance on arbitrary new lab data. No model can be expected to predict logD more precisely than this noise floor allows.

## The two logD tracks

LogD is predicted in two separate tracks because the same number means different things depending on whether the extractant molecule has been seen before. Conflating them is the single largest source of inflated headline metrics in this problem, so the split is deliberate and the evaluation regime is reported alongside every number.

**Why the tracks are separated.** The cleaned dataset contains many rows that share the same molecule, metal, acid, and condition setpoint and differ only as experimental replicates. Any fingerprint or ligand descriptor is constant within a molecule, so under a plain random-row split a model can memorize a molecule from rows that land in the training fold and recall it for the sibling rows in the test fold. That is legitimate when the use case is optimizing conditions for an extractant already in hand, but it is not generalization to a new molecule. The two tracks pin down which question is being answered:

- **Track A (new-molecule screen):** can the model rank an extractant it has never seen? Evaluated with molecule-grouped cross-validation, so every molecule appears in only one fold and there is no replicate leakage. Features are reaction conditions plus metal-ion descriptors only (no fingerprint of the candidate molecule, since that is what would be unknown for a true new candidate). Single LightGBM. Honest performance is **R2 about 0.466, RMSE about 1.148**.

- **Track B (known-molecule condition optimization):** given an extractant that is already characterized, can the model interpolate logD across new conditions (acid concentration, temperature, extractant loading, metal)? Features are conditions plus ECFP fingerprint plus ligand descriptors. NNLS-stacked trees. The honest operating number here is condition-key grouped cross-validation: hold out whole condition setpoints for molecules the model has otherwise seen. That gives **R2 about 0.61, RMSE about 0.98**.

**The 0.725 caveat, stated explicitly.** The random-row Track B figure of about 0.725 is an upper bound, not a generalization estimate. It is inflated by roughly 0.07 R2 because fingerprints are constant within a molecule and replicate rows leak across the split, so part of what looks like skill is memorization. The honest condition-interpolation number is about 0.61. And when the same rich Track B features (conditions + ECFP + ligand) are evaluated under molecule-grouped CV, the kind of split that reflects a genuinely new molecule, performance falls to about **0.44**, essentially back to Track A. In other words the fingerprint and ligand descriptors buy almost nothing on unseen molecules; their apparent value in Track B is the known-molecule advantage, not new-molecule prediction.

The audit script `logd_audit.py` reproduces this collapse from scratch with one consistent LightGBM across both tracks and three seeds, deliberately separating the four regimes so the gap is visible side by side. Its single-model reproduction lands close to the stacked production numbers and preserves the ordering: Track A new-molecule R2 0.481 +/- 0.008 (RMSE 1.131), Track B random-row upper bound 0.68 (RMSE 0.888), Track B condition-key grouped 0.607 (RMSE 0.984), and Track B rich features under molecule-grouped CV 0.443 (RMSE 1.172). The within-condition-key label-noise floor it measures is about 0.445 log units, so the residual RMSE in both tracks is already within a small multiple of irreducible measurement noise and the achievable ceiling is limited.

All numbers above are on the cleaned set (7066 rows after dropping exact duplicates and replicate groups whose logD range exceeds 2 log units), and should be read as an upper bound on a held-out deployment.

Files: `logd_audit.py`, `track_ab_results.csv`, and the confidence err-model tuning in `confidence_tune.py` (which carries the same Track A grouped / Track B random regime split forward into the uncertainty model).

## logD model sweep and the ensemble-versus-single reconciliation

The two logD tracks and the per-pair free-energy (delta G) target are all scored with the same protocol: a roster of base learners is trained out of fold under the cross-validation appropriate to the question, each base model is reported with both R2 and RMSE, and the out-of-fold predictions are then combined two ways, equal weight and a non-negative least squares (NNLS) stack whose weights are themselves fit and scored under a second cross-validation so the stacked number is honest rather than in-sample. The relevant scripts are `scripts/logd_ensemble.py` (the logD sweep), `scripts/ensemble_final.py` (the deployable NNLS-stacked logD pipeline with confidence and conformal layers), and `scripts/deploy_final.py` (the final per-track deployment built from the saved out-of-fold arrays).

### What the two logD tracks ask

`logd_ensemble.py` runs both logD tracks over a base roster of LightGBM, ExtraTrees, RandomForest, HistGradientBoosting, Ridge, and, when the libraries are present, XGBoost and CatBoost.

- Track A, new-molecule screening: features are reaction conditions plus metal-ion descriptors only, with no fingerprint of the candidate molecule, evaluated under molecule-grouped cross-validation so no molecule appears in more than one fold.
- Track B, known-molecule condition optimization: features add the ECFP fingerprint and ligand descriptors, evaluated under random-row cross-validation.

### logD base-model table

A caveat on sourcing has to be stated plainly. `logd_ensemble.py` writes its per-model results to `logd_ensemble_results.csv`, but that file is not committed to the repository (only the delta G counterpart, `results/dg_ensemble_results.csv`, is present). The per-model logD R2 and RMSE rows therefore cannot be quoted from an audited CSV the way the delta G rows below can. What is verifiable for the logD tracks is the stack-over-best gain and the deployed numbers, taken from the `ensemble_final.py` docstring and the committed prediction files:

| logD quantity | source | R2 | RMSE (log units) |
|---|---|---|---|
| Track A NNLS stack over best single | `ensemble_final.py` docstring | +0.018 (to about 0.495 at the oracle ceiling) | |
| Track B NNLS stack over best single | `ensemble_final.py` docstring | +0.013 (to about 0.721 at the oracle ceiling) | |
| Track A deployed (single LightGBM) | `results/deploy_A_screening_predictions.csv` | 0.466 | 1.148 |
| Track B deployed (NNLS stack) | `results/deploy_B_condition_predictions.csv` | 0.725 | 0.823 |

The Track B deployed 0.725 is the random-row figure and is an upper bound, not a generalization estimate; the honest condition-key-grouped Track B number is about 0.61 (RMSE about 0.98), per `results/track_ab_results.csv` and `docs/METHODS_AND_RESULTS.md`. The point for this section is the comparison between the stack and the single model, which is computed on the same split for both, so the relative gain is meaningful even though the absolute Track B level carries the replicate-memorization caveat.

### delta G base-model table (the mirror)

The per-pair delta G sweep is fully audited and committed, so its base-model table can be quoted directly from `results/dg_ensemble_results.csv` and `results/dg_ensemble_weights.csv`. All rows are out-of-fold under molecule-grouped cross-validation on the per-pair target (2273 distinct extractant-metal-acid-solvent systems):

| model | R2 | RMSE (kJ/mol) | NNLS weight |
|---|---|---|---|
| RandomForest | 0.473 | 6.31 | 0.556 |
| XGBoost | 0.444 | 6.48 | 0.00 |
| CatBoost | 0.441 | 6.50 | 0.278 |
| HistGB | 0.426 | 6.58 | 0.00 |
| LightGBM | 0.424 | 6.60 | 0.00 |
| ExtraTrees | 0.419 | 6.62 | 0.187 |
| Ridge | 0.026 | 8.58 | 0.00 |
| equal-weight | 0.459 | 6.40 | |
| NNLS stack (in-sample) | 0.479 | 6.28 | |
| NNLS stack (CV) | 0.474 | 6.31 | |

RandomForest is the best single model. The tree ensembles cluster between R2 0.42 and 0.47; Ridge is essentially flat, which says the structure-to-free-energy relationship is nonlinear. The honest cross-validated NNLS stack reaches R2 0.474 at RMSE 6.31, which only ties the single RandomForest; the in-sample stack reads 0.479 but is optimistic for the usual reason.

### Reconciling the apparent contradiction

The same NNLS-stacking procedure helps on the logD tracks and does not help on the per-pair delta G target. There is no contradiction once the data behind each is accounted for.

For the logD tracks, the stack of a diverse roster (lgb, xgb, hgb, et, cat, ridge, mlp) beats the best single model by about +0.018 R2 on Track A and about +0.013 R2 on Track B (`ensemble_final.py` docstring). These tracks run on the full multi-thousand-row logD table, where the members are genuinely diverse and each contributes some uncorrelated signal, so NNLS finds a positive-weight blend that improves on any one member. This gain is real and is what the deployed Track B model uses.

For the per-pair delta G target, the data is first collapsed to 2273 system-level means, a much smaller and wider table. On that table the tree ensembles are tightly clustered and largely redundant, RandomForest already dominates, and NNLS loads most of its weight back onto RandomForest (0.556), with only ExtraTrees and CatBoost contributing and every booster plus Ridge zeroed out. With no member adding independent signal beyond what RandomForest already captures, the honestly cross-validated stack lands at 0.474, statistically a tie with RandomForest's 0.473. Stacking is therefore not claimed as a gain there. The difference is the amount and diversity of data each target sees, not a difference in method.

### What is deployed for each

- Track A, new-molecule screening: single LightGBM. The stack trades a little confidence ranking for a little accuracy here, and the confidence bakeoff locked the single model in (`deploy_final.py`, `results/deploy_A_screening_predictions.csv`).
- Track B, known-molecule condition optimization: NNLS stack. Best accuracy and confidence as good as the single model (`deploy_final.py`, `results/deploy_B_condition_predictions.csv`).
- Per-pair delta G: single RandomForest. The NNLS stack only ties it, so the simpler single model is reported (`results/dg_ensemble_results.csv`, `docs/METHODS_AND_RESULTS.md`).

## Free energy of extraction (delta G)

The second target is the Gibbs free energy of extraction rather than logD itself. It is derived from each logD measurement by the standard relation

```
delta G = -2.303 * R * T * logD     (kJ/mol, R = 8.314e-3 kJ/mol/K, T in K)
```

so a favorable extraction (logD > 0) maps to a negative, spontaneous delta G. Treating delta G as the prediction target reframes the problem in thermodynamic terms: instead of asking what distribution ratio a given experiment will produce, the model estimates the free-energy change of moving the metal from the aqueous to the organic phase for a given extractant and metal.

### Conditions are dropped

The reaction-condition knobs are removed from the feature set on purpose. The columns dropped are extractant concentration, temperature, acid concentration, metal concentration, and the two volume fractions. The reasoning is that delta G as defined here is a property of the extractant structure and the metal, and the condition variables are the experimental dials that move logD around a given system rather than descriptors of the chemistry being predicted. The remaining features are the extractant fingerprint and RDKit descriptors, the metal descriptors, a one-hot of the acid type, and the diluent descriptors. The learned embedding columns are also excluded. This makes the delta G model a structure-and-metal predictor, distinct from the logD models that keep the conditions in.

A consequence of dropping conditions is worth stating plainly. Because temperature enters the delta G conversion but is not a feature, and because the same extractant-metal system was often measured under several conditions, identical feature rows can carry different delta G values. That within-system spread is irreducible noise the structure features cannot explain, and it sets a ceiling on the per-row framing below.

### Per-row versus per-pair

Two framings of the target are compared.

- **Per-row.** One delta G per measurement, conditions dropped. Because repeated measurements of the same system produce different delta G values that the structure features cannot distinguish, this framing carries the within-system scatter directly into the residual. On the cleaned set (7066 rows) the per-row model reaches R2 0.282 with RMSE 7.55 kJ/mol.
- **Per-pair.** One mean delta G per (extractant, metal, acid, solvent A, solvent B) system, condition-independent by construction. Averaging over the repeated measurements of a system removes the within-system noise that the structure features were never going to capture, leaving a cleaner target. This collapses the data to 2273 distinct systems and reaches R2 0.473 with RMSE 6.31 kJ/mol.

The per-pair framing is the reported one. It matches what the features can actually see (structure and metal, not conditions), so its accuracy reflects the chemistry rather than the leftover scatter from repeated experiments.

Both framings use molecule-grouped five-fold cross-validation: a given extractant never appears in both the training and the held-out fold, so the reported numbers describe generalization to genuinely new molecules rather than memorization of structures already seen.

### Model choice from the sweep

RandomForest was not assumed; it was selected from a seven-model sweep on the per-pair target, all under the same molecule-grouped cross-validation. The base models and their out-of-fold scores were:

| Model | R2 | RMSE (kJ/mol) |
|---|---|---|
| RandomForest | 0.473 | 6.31 |
| XGBoost | 0.444 | 6.48 |
| CatBoost | 0.441 | 6.50 |
| HistGB | 0.426 | 6.58 |
| LightGBM | 0.424 | 6.60 |
| ExtraTrees | 0.419 | 6.62 |
| Ridge | 0.026 | 8.58 |

RandomForest was the best single model. The tree ensembles cluster tightly between R2 0.42 and 0.47, while the linear Ridge baseline is essentially flat at 0.026, which says the structure-to-free-energy relationship is nonlinear and that the choice of tree family matters less than the framing.

A combined predictor was also tested. The out-of-fold predictions were stacked two ways: equal weight, and non-negative least squares (NNLS), which self-prunes weak members. To keep the stacked number honest, the NNLS weights were fit and scored under a second cross-validation over the out-of-fold matrix rather than fit and tested on the same rows. The honest cross-validated NNLS stack reaches R2 0.474 (RMSE 6.31 kJ/mol), which only ties the single RandomForest (the in-sample stack reads 0.479 but is optimistic for the usual reason). The NNLS weights load mostly on RandomForest (0.556), with ExtraTrees (0.187) and CatBoost (0.278) contributing and the boosting models and Ridge dropped to zero. Since the stack does not beat the best single model, the reported delta G model is the single RandomForest.

### Honest accuracy

The headline free-energy result is **R2 about 0.46, RMSE about 6.3 kJ/mol** on the per-pair target under molecule-grouped cross-validation. The 0.473 in `dg_results.csv` is the selected maximum of a roughly 55-way model and hyperparameter search; across shuffled splits the mean is 0.461 +/- 0.009, so 0.46 is the number to quote and 0.473 is the favorable end of the run-to-run spread rather than a stable point estimate.

The same script also fits a separate LightGBM error model out of fold and reports accuracy on the most-confident subsets (top 25 percent R2 0.683, top 10 percent R2 0.776 at RMSE 3.62 kJ/mol). These are selective-prediction operating points, not the model's overall accuracy; they describe how much sharper the predictions are when restricted to the cases the error model flags as reliable.


## Confidence

Every prediction is shipped with a confidence estimate. The point of the layer is operational: a chemist deciding which extractant or condition to test next needs to know not just the predicted logD or free energy of extraction but how far that number can be trusted. The confidence layer has three parts: an out-of-fold error model that ranks predictions, normalized split-conformal intervals that put a calibrated band around each one, and a selective-prediction view that reports how accuracy improves when only the most confident predictions are kept. A separate sweep checks that the error model is actually the best available confidence signal rather than an arbitrary choice.

All of it is computed under molecule-grouped cross-validation. The predictor and the error model are both fit on training molecules and applied to held-out molecules, and the conformal quantile is calibrated on one set of molecules and measured on a disjoint set. No molecule appears on both sides of any split, so the confidence numbers describe behavior on genuinely new chemistry rather than memorized replicates.

### Error model

The base predictor leaves a residual on each held-out point. Rather than treat that residual as unknowable, a second model is trained to predict its magnitude. Concretely, the base model (the per-pair RandomForest for free energy, LightGBM for the Track A logD model) produces out-of-fold predictions; the absolute residuals from those predictions become the regression target for a LightGBM error model, which is itself fit out-of-fold on the same molecule-grouped folds (`dg_coverage.py`, `dg_confidence_sweep.py`). The error model uses the same features as the predictor, so it learns where in feature space the predictor tends to be wrong. Its output is a per-prediction expected absolute error, which is what ranks predictions from most to least confident.

This is a learned confidence signal, not a probabilistic guarantee on its own. Its value is established empirically below: it both ranks predictions well (low predicted error tracks low true error) and, once passed through conformal calibration, produces intervals that cover at their nominal rate.

### Normalized split-conformal intervals and calibration

To turn the error model's ranking into honest intervals, split-conformal calibration is applied with the error model as the normalizer. For each held-out point the conformal score is the absolute residual divided by the error model's predicted error, `s = |y - yhat| / err`. The calibration set's empirical quantile of these scores, `q`, sets a single multiplier; the interval half-width for any point is then `q * err` for that point. Because `err` varies point to point, the intervals are wide where the model expects to be wrong and narrow where it expects to be right, instead of one fixed band for everyone. The calibration quantile is taken on one half of the molecules and coverage is measured on the disjoint other half, averaged over thirty random molecule splits (`dg_coverage.py`).

On the per-pair free energy model the intervals are well calibrated (`dg_coverage_results.csv`):

| Target | Empirical coverage | Mean width |
|--------|--------------------|------------|
| 90% | 0.902 | 20.9 kJ/mol |
| 80% | 0.802 | (same construction) |

A nominal 90 percent interval contains the true free energy 90.2 percent of the time and a nominal 80 percent interval covers 80.2 percent, both on molecule-disjoint evaluation. The agreement is close enough that the intervals can be used as stated. The mean 90 percent width is about 21 kJ/mol, which sets the practical resolution of a single free energy prediction and is the band that downstream active analysis carries into its selection decisions.

### Selective prediction

Confidence is most useful as a triage rule: trust and skip the predictions the model is sure about, and spend experiments on the ones it is not. Ranking held-out predictions by the error model and keeping only the most confident fraction (the top X percent by predicted error) gives the selective-prediction curve. Accuracy rises sharply as the low-confidence tail is dropped. On the free energy model, R2 goes from 0.473 over all pairs to 0.689 on the most confident 25 percent and 0.759 on the most confident 10 percent, with the top-10 percent RMSE falling from 6.3 to 3.71 kJ/mol (`dg_confidence_sweep_results.csv`). The headline numbers for the logD tracks, top-X percent R2 of about 0.912, 0.874, and 0.776, are the analogous selective-prediction operating points on those models.

Two honesty caveats apply. First, these are operating points, not a single accuracy figure: a top-X percent R2 describes a chosen subset and only means something paired with the fraction retained, so the cleaner way to read the curve is the RMSE drop on a fixed budget. Second, the Track B 0.940 figure was computed on a split that lets fingerprint-derived features memorize replicates of the same molecule, so it was inflated; recomputed on the molecule-disjoint condition-key split the honest Track B top-10 percent is R2 0.874 (RMSE 0.430), with all-rows R2 0.611. The free energy selective-prediction numbers above are already on molecule-grouped CV and do not carry that caveat.

### Confidence-model sweep

A confidence layer is only worth trusting if the chosen signal beats the alternatives. The sweep in `dg_confidence_sweep.py` fixes the predictor at the best model (RandomForest for free energy) and varies only how confidence is estimated: a LightGBM error model, a RandomForest error model, an ExtraTrees error model, and RandomForest's own across-tree prediction spread, the model's native uncertainty. Each is scored by the selective-prediction R2 it concentrates and by the Spearman correlation between its signal and the true absolute error.

The learned error models clearly beat the native tree spread. The LightGBM and RandomForest error models reach top-25 percent R2 of 0.689 and 0.696 and top-10 percent R2 of 0.759 and 0.748, while RandomForest's own tree-spread reaches only 0.613 and 0.659 (`dg_confidence_sweep_results.csv`). In other words, training a dedicated model to predict where the predictor will err ranks predictions better than reading off the ensemble's internal disagreement. The LightGBM and RandomForest error models are close to each other; LightGBM is kept as the default error model for consistency with the calibrated intervals above and the rest of the pipeline. A companion sweep (Part A of the same script) confirms the confidence filter helps every candidate predictor, not just RandomForest, so the gains come from the confidence layer itself rather than from one lucky predictor.

## Per-metal and per-pair accuracy and confidence

This section reports how accuracy and confidence vary across individual metals and across metal pairs, produced by `scripts/metal_confidence.py`. The script loads the saved out-of-fold predictions from the known-molecule random-CV stack (`conf_oof_B.npz`, Track B), re-derives the metal and condition columns by re-running the same cleaning so rows line up, and fits a regularized LightGBM confidence model on the prediction residuals using 5-fold out-of-fold estimates. For each metal with at least 20 rows it reports n, R2, RMSE, the median predicted confidence error, and the R2/RMSE on the top 25 percent most confident predictions. For each metal pair sharing the same extractant and solution conditions but differing in metal, it reports the separation R2/RMSE (predicted dlogD against actual dlogD) and the mean pair confidence error.

These numbers are all on the Track B known-molecule (random) split. They are therefore known-molecule optimistic: every metal here was seen during training, so per-metal accuracy reflects interpolation within familiar chemistry rather than performance on novel extractants. The honest held-out picture is weaker.

### Per-metal accuracy and confidence

Values below are read directly from `results/metal_confidence_by_metal.csv`, sorted by RMSE. Lower median confidence error means the model expects to be more accurate on that metal.

| Metal | n | R2 | RMSE | Median conf. err | Top-25% R2 | Top-25% RMSE |
|---|---|---|---|---|---|---|
| Cm(III) | 44 | 0.784 | 0.436 | 0.318 | 0.870 | 0.284 |
| Tb(III) | 238 | 0.504 | 0.561 | 0.364 | 0.353 | 0.448 |
| Ce(III) | 333 | 0.741 | 0.638 | 0.449 | 0.929 | 0.344 |
| Dy(III) | 349 | 0.720 | 0.648 | 0.406 | 0.917 | 0.380 |
| Am(III) | 1121 | 0.801 | 0.656 | 0.445 | 0.950 | 0.382 |
| Pu(IV) | 138 | 0.644 | 0.709 | 0.531 | 0.754 | 0.634 |
| Ho(III) | 211 | 0.794 | 0.737 | 0.534 | 0.885 | 0.577 |
| Pr(III) | 298 | 0.725 | 0.754 | 0.518 | 0.903 | 0.450 |
| Th(IV) | 443 | 0.750 | 0.757 | 0.395 | 0.936 | 0.306 |
| Eu(III) | 986 | 0.712 | 0.775 | 0.493 | 0.889 | 0.409 |
| Nd(III) | 343 | 0.560 | 0.782 | 0.458 | 0.900 | 0.341 |
| Pu(VI) | 26 | 0.568 | 0.793 | 0.662 | n/a | n/a |
| Lu(III) | 236 | 0.701 | 0.837 | 0.464 | 0.861 | 0.501 |
| Gd(III) | 297 | 0.769 | 0.840 | 0.570 | 0.887 | 0.594 |
| La(III) | 331 | 0.781 | 0.844 | 0.547 | 0.804 | 0.500 |
| Sm(III) | 273 | 0.529 | 0.882 | 0.599 | 0.803 | 0.541 |
| Yb(III) | 210 | 0.534 | 0.927 | 0.634 | 0.782 | 0.488 |
| Np(IV) | 68 | 0.398 | 0.996 | 0.668 | -0.339 | 0.311 |
| Np(V) | 147 | 0.053 | 1.027 | 0.819 | 0.406 | 0.788 |
| Tm(III) | 196 | 0.574 | 1.037 | 0.719 | 0.688 | 0.558 |
| U(VI) | 491 | 0.314 | 1.062 | 0.644 | 0.585 | 0.576 |
| Er(III) | 254 | 0.391 | 1.392 | 0.667 | 0.787 | 0.501 |

The model is most reliable on Am(III), the largest single-metal group at 1121 rows, with R2 0.801 and RMSE 0.656, rising to R2 0.950 and RMSE 0.382 on its most confident quartile. Cm(III) has the lowest RMSE overall at 0.436 (R2 0.784) and the lowest median confidence error at 0.318, though on only 44 rows. Ce(III), Dy(III), Ho(III), Th(IV), Gd(III) and La(III) also predict well, with R2 in the 0.72 to 0.79 range and confidence that tracks accuracy: their most confident quartile improves to R2 around 0.9. The confidence signal works for these metals, the predictions it flags as trustworthy are in fact more accurate.

The story is honest about where the model fails. Np(V) is the weakest case, with R2 0.053 and RMSE 1.027, and it carries the highest median confidence error in the table at 0.819, so the model does at least know it is unreliable there. U(VI), despite 491 rows, manages only R2 0.314 and RMSE 1.062. Er(III) has the worst RMSE in the table at 1.392 (R2 0.391). Np(IV) reaches only R2 0.398, and its most confident quartile actually gets worse (R2 -0.339), meaning the confidence ranking provides no useful signal for that metal. Yb(III), Sm(III), Tm(III) and Pu(VI) round out the weaker group. For these metals the predictions should not be trusted, and in the Np(IV) case the confidence estimate cannot be trusted to triage them either.

### Per-metal-pair separation

Selectivity is what matters for extraction, so `metal_confidence_by_pair.csv` reports how well the model predicts the difference in log D between two metals held under matched extractant and solution conditions. The full file lists 70 distinct pairs; the higher-count pairs are summarized below (n, separation R2/RMSE, mean pair confidence error).

| Metal pair | n | Sep. R2 | Sep. RMSE | Mean conf. err |
|---|---|---|---|---|
| Dy(III)/Tb(III) | 806 | 0.316 | 0.699 | 0.440 |
| Dy(III)/Eu(III) | 312 | -0.011 | 0.626 | 0.459 |
| Eu(III)/Tb(III) | 268 | 0.472 | 0.465 | 0.367 |
| Dy(III)/Er(III) | 249 | 0.574 | 1.040 | 0.817 |
| Dy(III)/La(III) | 189 | 0.573 | 0.511 | 0.507 |
| Dy(III)/Yb(III) | 176 | 0.810 | 1.000 | 0.592 |
| Nd(III)/Tb(III) | 144 | 0.331 | 0.713 | 0.559 |
| Am(III)/Eu(III) | 128 | 0.378 | 1.192 | 0.586 |
| La(III)/Tb(III) | 120 | 0.477 | 0.686 | 0.470 |
| Er(III)/Tb(III) | 116 | -0.050 | 2.529 | 0.965 |
| Dy(III)/Nd(III) | 113 | 0.527 | 0.665 | 0.579 |

Separation is harder than absolute prediction because two error-prone predictions are differenced. The largest pair, Dy(III)/Tb(III) at 806 matched cases, reaches only R2 0.316 (RMSE 0.699). Some pairs separate well: Dy(III)/Yb(III) reaches R2 0.810, and Eu(III)/Tb(III), Dy(III)/Er(III) and Dy(III)/La(III) sit in the 0.47 to 0.57 range. Others provide no separation signal at all. Dy(III)/Eu(III) is essentially zero (R2 -0.011), and Er(III)/Tb(III) is negative (R2 -0.050) with a large RMSE of 2.529 and the highest mean confidence error among the high-count pairs (0.965), consistent with Er(III)'s poor single-metal accuracy carrying through into its pairings. Beyond the high-count rows, the file contains many small-n pairs with strongly negative separation R2 (for example Er(III)/Nd(III) at n=55 with R2 -0.025 and RMSE 4.131, and several n in the teens to twenties with R2 below -1), which should be read with caution given their counts. The pattern is consistent: pairs involving Er(III), Np(V) and the actinide oxidation states are where both accuracy and confidence break down.

### Dedicated f-element separation-factor evaluation, with confidence

A focused evaluation (`scripts/sep_factor_eval.py`, `scripts/sep_factor_confidence.py`) measures the separation factor, the difference in logD between two f-elements at the same extractant and conditions, across all 4,598 f-element pairs (28 elements including actinides), obtained by differencing the logD model (a direct delta model was tested in `sep_factor_model.py` and lost outright, signed R2 -1.92, so differencing is used). It is reported in two regimes and, per the project's main contribution, always with the confidence-filtered operating point.

Full coverage, known extractant: signed R2 0.356, direction accuracy 0.726 (which f-element is extracted more), Spearman 0.601, magnitude R2 0.166. New extractant: signed R2 0.188, direction 0.656, Spearman 0.462, magnitude R2 -0.199. The model predicts the order moderately and the magnitude poorly.

With our confidence layer (known extractant, most confident 10 percent): direction accuracy 0.789, Spearman 0.776, and the normalized error RMSE/spread falls to 0.64, which beats a random-ranking shrinkage benchmark of 0.80. This gain was adversarially verified: within each true-gap bin the confident pairs are more accurate than the unconfident at the same gap size (gap-balanced direction +0.12, paired bootstrap +0.065, p=0.001), so it is neither a gap-selection artifact nor variance shrinkage. For a new extractant confidence is not net-useful (direction declines to 0.586; the RMSE drop is pure shrinkage), because the per-row confidence is too weakly calibrated on unseen ligands (rho 0.20 versus 0.39). The confidence-filtered figures are the headline for this task; the full-coverage numbers are never reported alone.

Dr. Zhang's own model on the same task (`scripts/sep_factor_zhang.py`, his XGBoost on his ECFP-centric features, differenced) ties ours at full coverage (known signed R2 0.369 versus our 0.356, direction 0.715 versus 0.726; new signed R2 0.227 versus 0.188, marginally ahead from the fingerprint our descriptor-only setup drops). Both hit the same wall, so the ceiling is the data, not the model. The distinction is that his model has no confidence layer, so it cannot produce the confidence-filtered operating point (direction 0.789) that makes ours deployable. Figures `figures/sep_factor_eval.png`, `figures/sep_factor_confidence.png`, `figures/sep_factor_zhang.png`.

### Figures

- `figures/by_metal_accuracy.png`: per-metal R2 and RMSE.
- `figures/confidence_per_metal.png`: per-metal confidence (median predicted error).
- `figures/by_metal_confidence_vs_error.png`: per-metal predicted confidence against actual error, showing where the confidence estimate is well-calibrated and where it is not.
- `figures/by_metal_pair_separation.png`: per-pair separation accuracy.
- `figures/by_pair_confidence_vs_error.png`: per-pair predicted confidence against actual separation error.

All figures and numbers above were verified against `results/metal_confidence_by_metal.csv`, `results/metal_confidence_by_pair.csv` and `scripts/metal_confidence.py` in `REE_logD_submission/`, which are byte-identical to the canonical copies in the working folder.

## Sign correctness and separation

Two analyses speak most directly to the separation objective: whether the model puts the predicted distribution coefficient on the correct side of zero (sign correctness), and how well it reproduces the gap in logD between two metals held at the same extractant and conditions (separation magnitude). The first is produced by `sign_separation.py`, the second by both `sign_separation.py` (Track B) and `track_a_separation.py` (Track A). Results are read from `results/sign_separation_results.csv` and `results/track_a_separation_results.csv`. Both scripts operate on the deployable prediction files `deploy_A_screening_predictions.csv` (Track A, a new extractant molecule held out) and `deploy_B_condition_predictions.csv` (Track B, a known molecule at new conditions).

### Sign correctness

Sign correctness asks whether the predicted logD has the same sign as the measured logD, that is, whether the model gets the extract-or-not decision right. This is computed as the fraction of rows where `sign(predicted) == sign(actual)`, reported overall and after keeping only the most confident rows ranked by the model's own predicted-error estimate.

On Track A, the new-molecule case, the predicted sign matches the actual sign for 0.743 of rows overall. Restricting to the most confident 25 percent raises this to 0.855, and the most confident 10 percent reach 0.903. On Track B, the known-molecule case, sign correctness is higher across the board: 0.836 overall, 0.913 on the most confident 25 percent, and 0.919 on the most confident 10 percent. The pattern is consistent in both tracks. Filtering on the model's confidence signal removes the rows it is least sure about and leaves a subset where the extract-or-not call is correct roughly nine times in ten.

### Separation magnitude

The separation analysis pairs two different metals measured at identical extractant, acid concentration, and temperature, then compares the predicted difference in logD against the actual difference. Pairs are formed within groups that share the same SMILES, acid concentration (rounded to two decimals), and temperature (rounded to whole kelvin); same-metal pairs are excluded.

For Track B, where direction matters because the known molecule is being applied at new conditions, the signed difference between the two metals is predicted with an R2 of 0.582. The selectivity direction, meaning which of the two metals has the higher logD, is called correctly for 0.763 of pairs. The predict-under fraction is 0.633.

For Track A, where the relevant quantity is the size of the separation rather than its direction (whichever metal is left in solution can be taken up with a separate known extractant), the analysis works with absolute differences over 44,894 metal pairs at identical conditions. The predict-under fraction is 0.567 over all pairs, and it moves from 0.544 at the most confident 50 percent (n=22,585) to 0.541 at the most confident 25 percent (n=11,280) to 0.486 at the most confident 10 percent (n=4,601). For context, `track_a_separation.py` also reports the mean absolute actual and predicted differences so the direction of any systematic under-prediction can be read off directly.

### The predict-under criterion

The predict-under criterion is the fraction of pairs where the predicted separation magnitude is at most the actual separation magnitude, `|predicted difference| <= |actual difference|`. It is a conservativeness check rather than an accuracy check. When it holds, the model has predicted a gap no larger than the one that really exists, so the true separation is at least as large as what the model promised. For a screening or design decision this is the safe direction to err: a pair flagged as separable by a conservative predicted gap will, on average, separate at least that well in practice. The fraction is reported overall and by confidence so the trade-off between coverage and conservativeness is visible. On Track A it declines from 0.567 over all pairs to 0.486 at the most confident 10 percent, meaning the model's highest-confidence pairs are slightly more likely to over-state the gap than its typical pair.

## Active analysis

The models are not only scored, they are put to work. The active-analysis stage asks the question a lab actually cares about: given a trained model and a confidence estimate for every prediction, can the model point at which extractant-and-condition systems to run next, and which to skip? This section covers the one-shot screening loop, the comparison of acquisition rules (greedy, UCB, uncertainty, random), experiment triage, sequential active learning over rounds, the pseudo-labeling negative result, and the trends in what gets chosen. The work runs on the free energy of extraction (delta G = -2.303 R T logD, kJ/mol), because delta G is the quantity that should drive a screen, with the per-pair RandomForest under molecule-grouped cross-validation so that nothing the model is praised for has leaked from a molecule it already saw. A favorable extractant is the most negative delta G. The supporting code is in `dg_ucb.py`, `active_learning_ucb.py`, `active_analysis_trends.py`, and `picks_trends.py`; figures are `active_analysis.png`, `ucb_graphs.png`, `active_analysis_trends.png`, and `picks_trends.png`.

### Greedy versus UCB versus random, one-shot

The simplest screen ranks the pool once by the model and takes the top slice. Three rules are compared on the same out-of-fold delta G predictions (`dg_ucb.py`): greedy, which ranks by the plain prediction; UCB, which ranks by the optimistic bound (prediction minus the learned uncertainty, so points the model is unsure about are pulled toward the favorable end); and random. Performance is the mean actual delta G of the picked set and the recall of the pool's genuinely strongest systems.

Greedy wins clearly. At the top 10 percent its picks average -8.54 kJ/mol against a population mean near zero, and it recovers 43 percent of the true best decile; at the top 5 percent it averages -10.47 kJ/mol with 34 percent recall. UCB does worse, not better: -4.06 kJ/mol and 31 percent recall at the top 10 percent. Random is essentially flat (around 0 kJ/mol, 10 percent recall, i.e. chance). The optimistic UCB bound, which is the right move when the goal is to explore and improve the model, is the wrong move for a one-shot screen whose only goal is to pick winners now: it spends the budget on uncertain candidates rather than confident strong ones. For one-shot selection, trust the prediction. This is shown in the delta G selection panel of `ucb_graphs.png`. The same ordering holds on the raw logD models (`active_analysis.png`, from `ucb_analysis_results.csv`): greedy beats UCB beats random on both Track A (new-molecule) and Track B (known-molecule) screens.

### Are the picks genuinely good, or just mid?

A high mean could still hide a model that avoids disasters without finding standouts, so `active_analysis_trends.py` looks past the mean (`active_analysis_trends.png`). Three results matter. First, composition: of greedy's top-10 percent picks, 81.5 percent fall in the genuinely strong (best) tertile by actual delta G, 11 percent are mid, and 7.5 percent are weak. UCB's top-10 percent is 66 percent good and 18.5 percent weak; random is 37 percent good and 32 percent weak. Greedy is choosing real winners, not safe middles. Second, precision: precision@10 percent (share of picks that land in the true best decile) is 0.43 for greedy, 0.40 for UCB, and 0.12 for random. At a tighter top 2 percent greedy precision rises to 0.73. Third, depth: greedy's top-10 percent picks capture about 62 percent of the cumulative depth an oracle ranking would achieve at the same budget, against 43 percent for UCB and 2 percent for random. The picks regress toward the middle only slowly as the budget grows, which is the expected and honest behavior of a finite, noisy model rather than a sign of failure.

### Per-metal behavior and what gets chosen

Ranking by prediction within each metal surfaces that metal's strong extractants for nearly every metal with enough data: greedy's within-metal top-20 percent beats the metal's own mean delta G for almost all metals with at least 20 systems (`active_analysis_by_metal.csv`, lower-right panel of `active_analysis_trends.png`). The exceptions are metals whose entire accessible chemistry is weak in this dataset, where even the metal's own best systems are only modestly favorable, so there is no strong tail to find.

Across the whole pool the picks concentrate on a chemically coherent set (`picks_trends.py`, `picks_trends.png`). By metal, americium is the most over-represented at 2.85 times its base rate, followed by Nd(III) at 2.50, Cm(III) at 2.23, Pu(VI) at 2.15, and Er(III) at 2.09; the picks lean toward americium and the trivalent lanthanides and actinides (`picks_metal_enrichment.csv`). By extractant structure the effects are real but modest in size (the largest standardized mean difference is Cohen's d about -0.54): the chosen extractants tend to be slightly less aromatic and lower in a few aromatic-surface descriptors (lower dipole moment, fewer aryl-methyl and pyridine fragments, fewer aromatic rings and benzene rings, `picks_feature_trends.csv`). These are the model's learned associations with strong extraction, offered as hypotheses to check against known chemistry rather than as established structure-activity claims; the small effect sizes argue for caution.

### Experiment triage

If a confident prediction can be trusted, an experiment can be saved. The triage test (`dg_ucb.py`) auto-accepts the most confident predictions, the ones with the lowest predicted absolute error, and measures how good the accepted set actually is. Accepting the most confident 25 percent gives an RMSE of 4.13 kJ/mol on that accepted set, with 62 percent of accepted predictions landing within 3 kJ/mol of truth; accepting the most confident 50 percent gives 4.64 kJ/mol and 54 percent within 3 kJ/mol. Accepting more lowers the bar, as expected. The operating rule is to trust and skip the confident predictions and spend the experimental budget on the uncertain ones. A related check inside `active_analysis_trends.py` splits greedy's top-10 percent picks into a confident half and an uncertain half: both halves are mostly strong, with the confident half modestly cleaner, so confidence is a useful but not decisive secondary filter on top of the prediction.

### Sequential active learning over rounds

The one-shot screen is the wrong place to expect UCB or uncertainty sampling to shine, because their job is to improve the model, not to harvest known winners. `active_learning_ucb.py` runs the proper sequential experiment: start from 300 labeled systems, and each round pick the next batch of 200 by an acquisition rule, reveal their logD, retrain a 5-model bagged LightGBM, and repeat for 10 rounds on the new-molecule (conditions plus metal) features. Two outcomes are tracked, model quality (R2 on a fixed 1000-system held-out test set) and discovery (share of the pool's true top-10 percent found), and five strategies are compared on the same seed set: random, greedy, UCB, pure uncertainty (the spread across the bagged models), and confidence (the same learned err model used in the deployed pipeline, fit on within-labeled out-of-fold residuals each round and used to acquire the points it flags as most uncertain). Curves are in `ucb_graphs.png` and `active_learning_ucb.png`.

The two objectives pull in opposite directions and no single strategy wins both. For model quality, uncertainty sampling is best, reaching test R2 0.561 by the final round against 0.535 for random, 0.465 for confidence, 0.460 for greedy, and 0.437 for UCB; querying where the model is unsure improves it the fastest. For discovery, greedy is best, finding 72.8 percent of the strong tail, with UCB close behind at 71.3 percent, while uncertainty (38.3 percent), confidence (43.6 percent), and random (34.7 percent) lag because they spend queries on informative rather than strong systems. UCB ties or loses to greedy on both axes here, so it is not the recommended rule for either objective in this dataset. The practical reading: to build a better model, sample the uncertain; to find strong extractants now, take the greedy picks.

### Pseudo-labeling: a negative result

Pseudo-labeling, treating the model's most confident predictions on unlabeled systems as if they were real measurements and adding them to training, was tested as a way to get model improvement without running experiments. It added nothing. Folding confident predictions back in did not improve held-out performance, because confident predictions are by construction the ones the model already gets right, so they carry no new information, while the uncertain systems that would actually teach the model are exactly the ones pseudo-labeling is least able to label. The lesson is that there is no shortcut around the wet lab: real new measurements are required to move the model, which is why uncertainty-driven sequential acquisition is the right tool for improvement and pseudo-labeling is not.

### Summary

Greedy ranking by predicted delta G is the right rule for a one-shot screen: its top-10 percent picks are 81.5 percent genuinely strong, hit the true best decile at precision 0.43, capture about 62 percent of oracle depth, and recover 43 percent of the strongest systems, with picks that are chemically coherent (americium and the trivalent lanthanides and actinides, slightly less aromatic extractants). UCB underperforms greedy for selection. Confidence supports triage, trust and skip the confident predictions and test the uncertain ones. For improving the model rather than harvesting winners, uncertainty sampling over rounds is fastest (R2 0.56 versus 0.44 for greedy), and pseudo-labeling adds nothing, so genuine experiments remain necessary. All numbers here are out-of-fold under molecule-grouped cross-validation on the cleaned 7066-system set and should be read as an upper bound given the roughly 0.45 log-unit label-noise floor.

## UCB candidate screen

This section was specified around a script named `ucb_90_zone_screen.py` writing to `ucb_90_zone_outputs/`. Neither exists in the repository. A full search under `/Users/skyepstein/Downloads/ML Model Folder 8 - CC Environment/` and its `REE_logD_submission/` subtree returns no file matching `ucb_90_zone`, no directory matching `*zone*`, and no script that maps predictions into three D bands (D < 0.5, 0.5 to 10, > 10) or aggregates them to one recommended experiment per molecule with a ranked list. The qhat value of about 0.826 referenced in the request also does not appear as a conformal quantile anywhere. The only 0.826 in the repository is an unrelated per-pair confidence number for Nd(III)/Pr(III) in `results/metal_confidence_by_pair.csv`. The three-zone screen as described could not be verified and is not documented here as if it existed.

What the repository does contain, audited against the source and the output CSVs, is the following application-facing work.

### Calibrated 90 percent intervals (conformal coverage)

`scripts/dg_coverage.py` builds the prediction intervals that make the candidate ranking trustworthy. It fits a per-pair RandomForest on the free energy target (delta G) with molecule-grouped cross-validation for honest out-of-fold predictions, then a LightGBM error model that predicts each point's absolute residual. It applies normalized split conformal: the conformal quantile is calibrated on one set of molecules and coverage is measured on a disjoint set, averaged over 30 random molecule splits. The verified output in `results/dg_coverage_results.csv` is:

- 90 percent interval: empirical coverage 0.902 against a target of 0.90, mean width 20.9 kJ/mol.
- 80 percent interval: empirical coverage 0.802 against a target of 0.80.

The 90 percent coverage of 0.902 is the figure that backs the interval-based screening. The qhat is computed inside this script as the 0.90 quantile of normalized residuals on the calibration molecules and is not written to a standalone value file.

### UCB candidate ranking (upper bound of the 90 percent interval)

`scripts/ucb_analysis.py` is the candidate-experiment screen that is present. It ranks untested extractant-and-condition combinations by an upper confidence bound, defined as the top of the 90 percent conformal interval (hi90 = prediction + uncertainty), and compares that against greedy ranking by the point prediction and against random. It evaluates two metrics, the mean actual logD of the selected set and the recall of the genuinely top group, at the top 5 percent and top 10 percent selection sizes. Verified results from `results/ucb_analysis_results.csv`:

| Track | Top % | Method | Mean actual logD | Recall of true best |
| --- | --- | --- | --- | --- |
| A (new molecule) | 5 | UCB (hi90) | 0.84 | 0.18 |
| A (new molecule) | 5 | greedy (prediction) | 2.23 | 0.45 |
| A (new molecule) | 5 | random | 0.04 | 0.05 |
| A (new molecule) | 10 | UCB (hi90) | 0.82 | 0.26 |
| A (new molecule) | 10 | greedy (prediction) | 1.86 | 0.54 |
| A (new molecule) | 10 | random | 0.06 | 0.11 |
| B (known molecule) | 5 | UCB (hi90) | 1.81 | 0.38 |
| B (known molecule) | 5 | greedy (prediction) | 2.59 | 0.57 |
| B (known molecule) | 5 | random | -0.01 | 0.06 |
| B (known molecule) | 10 | UCB (hi90) | 1.70 | 0.50 |
| B (known molecule) | 10 | greedy (prediction) | 2.26 | 0.67 |
| B (known molecule) | 10 | random | -0.03 | 0.09 |

In this one-shot screening setting the greedy point prediction outperforms the UCB rule on both tracks and at both selection sizes. UCB still beats random by a wide margin, but optimism under uncertainty does not help when the whole pool is scored at once. The companion `scripts/dg_ucb.py` notes that UCB is meant to pay off in sequential active learning rather than one-shot screening.

### Experiment triage

`results/experiment_triage_results.csv` reports auto-accepting the most confident predictions so the experiment budget goes to the uncertain ones. At the 25 percent auto-accept level the verified numbers are:

- Track A (new molecule): accepted-set RMSE 0.642 logD, 70 percent of accepted predictions within 0.5 logD (recorded as 0.699), 1767 experiments saved out of 5298 needed.
- Track B (known molecule): accepted-set RMSE 0.424 logD, 85 percent within 0.5 logD (recorded as 0.847), 1767 experiments saved out of 5298 needed.

Higher auto-accept fractions trade accuracy for more saved experiments. Full table from the CSV:

| Track | Auto-accept % | Experiments saved | Experiments needed | Accepted RMSE | Within 0.5 |
| --- | --- | --- | --- | --- | --- |
| A (new molecule) | 25 | 1767 | 5298 | 0.642 | 0.699 |
| A (new molecule) | 50 | 3535 | 3530 | 0.812 | 0.588 |
| A (new molecule) | 75 | 5299 | 1766 | 0.943 | 0.508 |
| B (known molecule) | 25 | 1767 | 5298 | 0.424 | 0.847 |
| B (known molecule) | 50 | 3535 | 3530 | 0.535 | 0.746 |
| B (known molecule) | 75 | 5299 | 1766 | 0.631 | 0.672 |

### Pseudo-labeling

`results/dg_pseudo_results.csv` tests whether adding the model's own confident predictions as training labels helps, with a true-label oracle as the ceiling. Test metrics are R2 and RMSE in kJ/mol on the delta G target:

| Setting | Added confident % | Test R2 | Test RMSE (kJ/mol) |
| --- | --- | --- | --- |
| baseline (seed only) | 0 | 0.296 | 7.07 |
| pseudo (predicted labels) | 50 | 0.296 | 7.06 |
| oracle (true labels) | 50 | 0.355 | 6.76 |
| pseudo (predicted labels) | 100 | 0.310 | 6.99 |
| oracle (true labels) | 100 | 0.392 | 6.56 |

Pseudo-labeling gives essentially nothing over the baseline R2 of 0.296: 0.296 at the 50 percent add level and 0.310 at 100 percent. The oracle that adds the same points with their true labels reaches 0.355 and 0.392, which shows the headroom comes from real measurements, not from the model relabeling its own pool.

### Source files

- `/Users/skyepstein/Downloads/ML Model Folder 8 - CC Environment/REE_logD_submission/scripts/dg_coverage.py`
- `/Users/skyepstein/Downloads/ML Model Folder 8 - CC Environment/REE_logD_submission/scripts/ucb_analysis.py`
- `/Users/skyepstein/Downloads/ML Model Folder 8 - CC Environment/REE_logD_submission/scripts/dg_ucb.py`
- `/Users/skyepstein/Downloads/ML Model Folder 8 - CC Environment/REE_logD_submission/results/dg_coverage_results.csv`
- `/Users/skyepstein/Downloads/ML Model Folder 8 - CC Environment/REE_logD_submission/results/ucb_analysis_results.csv`
- `/Users/skyepstein/Downloads/ML Model Folder 8 - CC Environment/REE_logD_submission/results/experiment_triage_results.csv`
- `/Users/skyepstein/Downloads/ML Model Folder 8 - CC Environment/REE_logD_submission/results/dg_pseudo_results.csv`

Note for the writer: the `ucb_90_zone_screen.py` script, its `ucb_90_zone_outputs/` directory, the three-band D < 0.5 / 0.5 to 10 / > 10 mapping, the per-molecule aggregated recommended-experiment list, and the qhat value of about 0.826 were not found in the repository and should not be presented as existing artifacts. The triage and pseudo-labeling numbers above are verified, as is the 0.902 conformal coverage at the 90 percent target.

## Classification

Dr. Zhang's published work frames rare-earth extraction as a classification problem: bin the distribution coefficient D and predict which bin a ligand-condition pair falls into. This repository reproduces that framing and adds the piece his model does not report, a per-prediction confidence score, so that a user can rank predictions and act on the ones the model is sure about. The classifiers are built in `scripts/classifier_confidence.py`, with a companion `scripts/xgb_confidence.py` that bolts the same confidence layer onto a reproduction of his own XGBoost model.

### Tasks, classes, and splits

Two targets are defined on the same cleaned data used elsewhere in the project (7,065 rows after deduplication and the within-group logD range filter, pooled from `Training_Data_V27.csv` and `Testing_Data_V39.csv`):

- 3-class: logD binned at D = 0.5 and D = 10, that is at log10(0.5) and 1.0, giving low / medium / high. These are Zhang's inferred cut points.
- Binary: will the ligand extract at all, logD > 0.

Each target is run on two tracks:

- Track A, new-molecule. Features are conditions plus metal descriptors only, scored with molecule-grouped 5-fold cross-validation so that no ligand appears in both train and test. This is the honest, generalization-to-unseen-ligands setting and the direct analog of Zhang's held-out-by-molecule test.
- Track B, known-molecule. Features add ECFP fingerprints and per-ligand atom counts, scored with random 5-fold cross-validation. Here the same molecules can appear in train and test under different conditions, so these numbers describe interpolation within known chemistry, not extrapolation to new ligands.

Confidence is the model's own probability for the class it picked: the maximum softmax probability for the 3-class task, and distance from 0.5 for the binary task. Selective accuracy is then reported on the top 50, 25, and 10 percent of predictions ranked by that confidence.

### Headline results (LightGBM classifiers)

From `results/classifier_confidence_results.csv` and the run log:

| Track | Task | Accuracy | Macro-F1 / ROC AUC | Baseline | Top 25% conf. | Top 10% conf. |
|---|---|---|---|---|---|---|
| A new-molecule (grouped, honest) | 3-class | 0.625 | macro-F1 0.619 | majority 0.38 | 0.854 | 0.912 |
| A new-molecule (grouped, honest) | binary logD>0 | 0.742 | ROC AUC 0.817 | base rate 0.53 | 0.896 | 0.962 |
| B known-molecule | 3-class | 0.753 | macro-F1 0.752 | majority 0.38 | 0.956 | 0.983 |
| B known-molecule | binary logD>0 | 0.832 | ROC AUC 0.910 | base rate 0.53 | 0.973 | 0.982 |

The honest, generalization numbers are Track A: 0.625 three-class accuracy against a 0.38 majority baseline, and 0.742 binary accuracy against a 0.53 base rate. The Track B numbers (0.753 and 0.832) are higher because they include known molecules under new conditions, so they should be read as interpolation rather than new-ligand prediction.

The confidence layer is what makes these usable. On the new-molecule three-class task, accuracy climbs from 0.625 over all 7,065 predictions to 0.854 on the most confident 25 percent and 0.912 on the most confident 10 percent. The same pattern holds across every track and task: the model's stated confidence is a reliable guide to where it is right, which lets a screening campaign trust the top-ranked calls rather than the flat average.

### Zhang's XGBoost plus our confidence

`scripts/xgb_confidence.py` reproduces Zhang's model directly, an XGBoost multi:softprob three-class classifier on fingerprints plus conditions plus metal and ligand features, scored with molecule-grouped 5-fold CV to match his held-out-by-molecule setup, and adds the same confidence ranking. From `results/xgb_confidence_results.csv` and the run log:

- 3-class new-molecule: accuracy 0.605, macro-F1 0.598, against the 0.38 majority baseline. With the confidence layer, top 25 percent reaches 0.847 and top 10 percent reaches 0.914.
- Binary logD>0 new-molecule (run log only, not written to the CSV): accuracy 0.731, ROC AUC 0.798, base rate 0.53, rising to 0.900 on the top 25 percent and 0.936 on the top 10 percent.

Zhang reports a single 0.72 three-class accuracy on one 494-row molecule-held-out test, with no confidence ranking. The 0.605 here is the same model architecture re-scored across all five molecule-grouped folds on the 7,065-row pooled set, so it is a more conservative average over the full data rather than one favorable split. The contribution is not a higher flat number; it is the confidence score his model lacks, which turns one undifferentiated accuracy figure into a ranked list where the top decile is correct more than nine times in ten.

A note on comparability: the LightGBM Track A three-class number (0.625) and this XGBoost reproduction (0.605) are both honest molecule-grouped numbers on the same cleaned data, so they are directly comparable. The Track B figures and Zhang's published 0.72 are measured on different splits and should not be compared head-to-head with the grouped numbers.


## Comparison with Dr. Zhang's model

A prior model from Dr. Zhang predicts a three-class label (logD binned at D = 0.5 and D = 10) from a Morgan fingerprint plus an RDKit descriptor block, the extraction conditions, and numeric metal descriptors, trained with XGBoost. His reported headline is a 0.72 three-class accuracy on a 494-row held-out test set spanning 15 molecules. To make a like-for-like comparison rather than comparing across different data and different splits, we ran both models on his committed featurized data (trainVal plus test, 8075 rows total, his exact features and his exact `Class_index` target) and crossed the model choice with the split choice in a 2x2.

### Same data, model crossed with split

The script (`scripts/zhang_2x2.py`, results in `results/zhang_2x2_results.csv`) fits each model under each split and reports three-class accuracy.

| | Our split (molecule-grouped 5-fold CV) | His split (single 494-row holdout) |
| --- | --- | --- |
| Our model (LightGBM) | 0.657 | 0.692 |
| His model (XGBoost) | 0.648 | 0.680 |

Reading down a column, the two models are within about 0.01 accuracy of each other under either split: 0.657 vs 0.648 under our cross-validation, 0.692 vs 0.680 on his holdout. Reading across a row, moving from the grouped cross-validation to his single holdout moves accuracy by about 0.03 to 0.04 for the same model. The gap between cells is driven by the choice of split far more than by the choice of model.

### Honest verdict

His reported 0.72 reproduces at 0.68 when his own XGBoost is rerun on his test set under our pipeline, so the headline 0.72 is a property of that one favorable 494-row, 15-molecule split rather than of the model. Both models land near 0.65 under a genuinely molecule-grouped split and near 0.69 on his easier holdout. The two approaches are about equal in predictive accuracy on this task; the split, not the model, drives his reported number. We do not claim to beat his model and the comparison does not support either side claiming a meaningful accuracy edge.

For context, our regression head on the same features under the same molecule-grouped cross-validation reaches R2 = 0.411 and RMSE = 1.221 log units on continuous logD, consistent with the new-molecule (Track A) regime where held-out molecules make the problem genuinely hard.

### What our method adds

The accuracy parity is the point: on the shared task the models tie, so the contribution is not a higher number but a more useful and more honest output.

- Calibrated uncertainty. Each prediction carries an out-of-fold absolute-error estimate that ranks confidence, plus normalized split-conformal intervals that are calibrated under a molecule-disjoint split (90 percent intervals cover 90.2 percent, 80 percent cover 80.2 percent). This supports selective prediction and triage rather than a single point label.
- Continuous logD instead of a three-class bin. We predict logD directly, which preserves the quantitative information that a three-bin label discards and is what downstream design and screening actually need.
- Separations. The continuous logD prediction feeds the free-energy and separation-factor analysis (delta G = -2.303 R T logD), letting the pipeline reason about how two metals separate rather than only classifying a single extraction outcome.


## Rejected approaches

Several modeling choices were tested against held-out data and dropped because they did not improve, or actively hurt, prediction on molecules the model had not seen during training. The decisions below come from two audited experiments: the log D feature-subset sweep in `scripts/baseline_v39.py` (results in `baseline_v39_results.csv`) and the delta G model sweep in `scripts/dg_sweep2.py` (results in `results/dg_sweep2_results.csv`).

### Extra feature blocks that hurt new-molecule generalization

The feature sweep ran the same five-model ensemble (XGBoost, LightGBM, HistGradientBoosting, ExtraTrees, RandomForest) on three feature sets, scored on the chemistry-disjoint V39 test set (no test molecule appears in training). Reported as external R2 and RMSE on that set:

- ECFP fingerprints plus reaction conditions only: R2 0.604, RMSE 1.063 (825 features)
- Add RDKit descriptors: R2 0.534, RMSE 1.154 (993 features)
- Add RDKit descriptors and the 768 learned embeddings: R2 0.500, RMSE 1.195 (1761 features)

The lean ECFP-plus-conditions set is the best of the three, and every added block makes the held-out fit worse. The `embedding_` columns are 768 learned molecular embeddings (a per-molecule vector produced by a neural encoder rather than a hand-defined chemical descriptor). They were excluded because adding them lowers external R2 from 0.604 to 0.500. The RDKit physicochemical descriptors were dropped for the same reason. Both blocks add hundreds of columns that let the trees fit training chemistry more tightly without carrying that accuracy over to unseen molecules, so the final log D model uses ECFP plus conditions only.

### Graph neural network (Chemprop D-MPNN)

A directed message-passing graph network was trained on the molecular graphs directly as a side experiment. On the same chemistry-disjoint V39 test set it reached only R2 0.2273, RMSE 1.486 (recorded as the reference line in `baseline_v39.py`). That is far below the 0.604 from ECFP plus conditions, so the standalone graph network was not adopted as the predictor. The descriptor-and-tree route, with explicit reaction conditions and metal properties, carries the signal that the graph-only model misses.

### TabPFN

TabPFN is a small-data specialist, and this regime is small, so it was tested rather than dismissed. It was rejected on two grounds. On log D, inside the stack it earned an NNLS weight of 0.13 but added only +0.0011 R2 over the trees-only stack, with an error correlation of 0.82 to the trees, meaning it brings almost no diversity and does not clear the keep threshold. On delta G, it could not run above roughly 1000 context rows on CPU, which makes a single global TabPFN weak on this dataset (about R2 0.36 with the context capped at 1000 rows). It was dropped from both targets.

### Delta G model alternatives

For the delta G target the sweep compared several regressors under grouped (by molecule) five-fold cross-validation, scored as R2 and RMSE in kJ/mol:

- RandomForest, descriptor-only features: R2 0.472, RMSE 6.32
- RandomForest, tuned with max_features sqrt: R2 0.348, RMSE 7.02
- SVR with RBF kernel on scaled descriptors: R2 0.17, RMSE 7.92
- kNN on scaled descriptors: R2 0.105, RMSE 8.22

The kNN and SVR-rbf models were far behind and rejected. The "tuned" RandomForest with the sqrt feature heuristic (0.348) scored worse than the default RandomForest (0.472), so that tuning was rejected and the default settings kept. The NNLS stack that included TabPFN landed at 0.471, essentially tied with the plain descriptor-only RandomForest at 0.472, which is the bar the other approaches failed to beat.


## Audit

The results in this repository were put through two adversarial review passes and a direct reproduction before being published, on the assumption that a skeptical reviewer would try to break every number. This section records what was checked, what survived, and every correction that was made. The short verdict is that the work is fundamentally honest: the cross-validation genuinely holds molecules out, no feature encodes the target, the uncertainty intervals are calibrated, and the headline numbers reproduce from the code in the repository. The corrections were about optimistic framing and about repository hygiene, not about hidden leakage or fabricated scores.

### What was checked

**Cross-validation and grouping.** Every reported new-molecule number was confirmed to come from molecule-grouped cross-validation, in which the unique extractants are shuffled and split into folds so that no molecule can appear in both training and testing. This was verified by reading the fold-assignment code directly rather than trusting the summary tables: the grouping keys on the canonical SMILES, and the fold map is built from the unique molecule list, so all rows of a molecule move together. The earlier single split that read R2 = 0.60 was traced to a molecule that sat on both sides of the split, and that inflation does not survive grouping.

**Target-aware cleaning.** The data cleaning was checked for the specific failure of cleaning to the model, meaning removing rows because the model misses them and then reporting the improved score as if it were real. The two cleaning steps that feed the published numbers, dropping exact duplicate rows and dropping replicate groups whose logD range exceeds 2 log units, are both independent of any model: they depend only on agreement between repeated measurements of the same molecule, metal, and conditions. The audit script does additionally flag the rows the model misses worst (the top 2 percent of out-of-fold residuals, 162 rows), but those are written out for manual vetting only and are explicitly not removed before scoring. Removing only the label-discordant groups, on an independent criterion, moves the new-molecule R2 only from 0.450 to about 0.477 while pulling the within-group noise floor down from 0.77 to about 0.45, which is consistent with bad labels rather than a model artifact.

**Confidence and conformal honesty.** The confidence layer was checked to confirm it is a genuine out-of-fold predictor and not a post hoc fit to the test residuals. The error model predicts each prediction's absolute error out-of-fold, and the rows are ranked by that predicted error. The split-conformal intervals were checked for calibration against their stated targets on a molecule-disjoint split: the 90 percent intervals cover 90.2 percent and the 80 percent intervals cover 80.2 percent, so the coverage claim holds. The selective-prediction operating points (the top-X-percent-by-confidence accuracies) were checked to make sure they were not being read as overall accuracy.

**Selection overfitting.** The free-energy result was checked for the classic max-of-many-runs inflation. The 0.473 that had been quoted was found to be the single best of a roughly 55-way model and hyperparameter search on one fixed split. The honest statement is the mean over repeated shuffled molecule-grouped splits, which is R2 about 0.461 +/- 0.009 (RMSE 6.3 kJ/mol), and that is now what is reported. RandomForest was confirmed to be the genuine best of seven base learners on this small wide table, and an NNLS stack scored with its own cross-validation only ties it, so stacking is not claimed as a gain there.

**Evaluation framing.** Each headline was checked to make sure it answers the question it appears to answer. The Track B random-row score of 0.725 was identified as an upper bound rather than a generalization number, because the fingerprint is constant within a molecule and a random-row split lets sibling rows of the same molecule sit in both folds, so the model can memorize about 0.07 R2 of that figure. On a leakage-free condition-key grouped split the honest Track B number is about 0.61 (RMSE about 0.98), and on genuinely new molecules the same rich features collapse to about 0.44, close to Track A. The selective-prediction figures (Track A 0.912, Track B 0.940) were reframed as operating points at 10 percent coverage rather than overall accuracy, with the RMSE drop led as the honest summary; the Track B 0.940 in particular sat on a leaky split and was recomputed on the condition-key split to an honest 0.874 (RMSE 0.430).

**Feature leakage.** The feature set was scanned for any column that encodes the target. None was found. The features are extractant structure (fingerprint, RDKit descriptors, ligand donor-atom counts, logP), metal descriptors (atomic number, ionic radius, oxidation state, ionization energies, and so on), acid and process conditions, and diluent descriptors. None of these is computed from logD, and the free-energy framing that derives delta G from logD keeps that derived quantity strictly on the label side, never as an input.

**Data integrity.** The raw data was validated directly. All 295 unique SMILES parse under RDKit (zero invalid), the dataset holds 8074 usable rows across 295 molecules and 28 f-element metals, logD spans roughly -12.5 to 5.0, there are 18 exact duplicate rows and 3 rows with absolute logD above 10. Replicate discordance was quantified: 1088 repeated-condition groups cover 47 percent of the rows, the median within-group logD range is 0.58, and 12 percent of rows sit in groups whose range exceeds 2 log units, which is the population the cleaning removes. This is what sets the label-noise floor near 0.45 log units on the cleaned data and is why all reported scores are framed as an upper bound relative to the raw measurements.

**Comparison to Zhang (SAFE-MolGen).** The comparison was checked for a stacked-deck setup. Because both datasets have exactly 8075 rows, the two almost certainly use the same integrated f-element dataset, so the comparison is fair on the data. Zhang's XGBoost classifier was reproduced directly under the same molecule-grouped cross-validation and scored about 0.605 over all new molecules, below his reported 0.72 and about equal to the classifiers built here. The conclusion is that his headline 0.72 reflects his single 494-row molecule-held-out holdout and its more imbalanced classes rather than a stronger model; on the same data and split the two approaches are about equal (ours 0.657 CV and 0.692 holdout against his 0.648 and 0.680), and his external 0.72 reproduces at 0.68.

**Reproducibility.** The repository was checked to confirm that the scripts shipped actually produce the reported numbers. During this pass it was found that some of the runnable scripts had been reduced to stubs, and the real working code was restored over them so that a reviewer who runs the scripts in order reproduces the tables. Re-running the reproduction script returns Track A at R2 0.481, RMSE 1.131 and the honest condition-key Track B at R2 0.607, RMSE 0.984, matching the published numbers within run-to-run noise.

### Verdict

The work is fundamentally honest. The cross-validation genuinely holds molecules out with zero molecule leakage, the cleaning criterion is independent of the model, no feature encodes the target, the conformal intervals are calibrated on a molecule-disjoint split, and the headline numbers reproduce from the code in the repository.

### Corrections made

The audit changed the framing and the repository, not the underlying modeling. The specific corrections were:

- Track B: stopped presenting the random-row 0.725 as the result and led with the leakage-free condition-key number of about 0.61, labeling 0.725 as an upper bound inflated about 0.07 by replicate memorization and noting that on new molecules the same features fall to about 0.44.
- Free energy: replaced the selected-best 0.473 point estimate with the repeated-split mean of about 0.461 +/- 0.009, and stated plainly that the NNLS stack only ties RandomForest rather than improving on it.
- Top-X-percent figures: reframed the 0.912, 0.940, and 0.776 numbers as selective-prediction operating points at low coverage rather than overall accuracy, led with the RMSE drop, and recomputed the Track B 0.940 on the leakage-free condition-key split to an honest 0.874 (RMSE 0.430).
- Zhang: corrected the comparison to state that the split, not the model, drives his 0.72 headline, supported by the direct reproduction at about 0.605 under matched grouping.
- Reproducibility: restored the real analysis code over stub scripts so the repository regenerates its own tables.


## Reproducibility

All numbers in this document come from scripts in this repository run against the two shipped CSVs. This section gives the layout, the steps to reproduce, and a map from each headline number and figure to the script that produces it.

### Repository layout

```
REE_logD_submission/
├── data/
│   └── data.zip            # Training_Data_V27.csv + Testing_Data_V39.csv, compressed
├── scripts/                # all runnable Python (one concern per file)
├── results/                # generated CSVs (metrics tables, per-row predictions)
├── figures/                # generated PNGs and a combined all_figures.pdf
├── docs/                   # METHODS_AND_RESULTS.md, results workbook, slides, archive/
├── requirements.txt
└── README.md
```

The raw CSVs are shipped only inside `data/data.zip`; the unzipped copies and the row-level prediction files are git-ignored because they carry measured logD values. The `results/` and `figures/` directories hold the committed outputs so a reviewer can read the numbers without rerunning anything.

### Environment

Python 3 with the packages pinned in `requirements.txt`: numpy, pandas, scipy, scikit-learn, lightgbm, xgboost, catboost, rdkit, and openpyxl. TabPFN is pinned to the v2 line (`tabpfn<3`) on purpose, because the v3 client sends data to a remote API; the local v2 model is the one used. chemprop is left commented out and is only needed for the graph-network side experiment, which is not part of any headline number.

### Reproduce

Run from the repository root. The scripts read `Training_Data_V27.csv` and `Testing_Data_V39.csv` from the current working directory, so the unzip has to land them next to where the scripts are invoked.

```bash
pip install -r requirements.txt
unzip data/data.zip            # -> Training_Data_V27.csv, Testing_Data_V39.csv
```

Two ordering constraints matter. First, `confidence_tune.py` writes the out-of-fold caches `conf_oof_A.npz` and `conf_oof_B.npz`, and `deploy_final.py` reads them, so run the tuning script before the deploy script. Second, the Zhang head-to-head scripts (`zhang_2x2.py`, `zhang_his_split.py`, `zhang_data_model.py`) read pre-featurized files from `/tmp` (`/tmp/z_trainVal_dataset.csv`, `/tmp/z_test_dataset.csv`, `/tmp/zhang_combined.csv`), which are Dr. Zhang's committed split carrying his fingerprint and descriptor block. Those files are an external input, not contents of `data.zip`; the Zhang comparison cannot be reproduced from this repo alone without them.

A representative end-to-end run:

```bash
# honest reproduction of the two logD tracks and the noise floor
python3 scripts/logd_audit.py          # Track A / Track B audit numbers

# free energy (delta G) model and hyperparameter search
python3 scripts/dg_model.py            # per-row and per-pair delta G
python3 scripts/dg_hpo.py              # RF vs XGB vs LGB on per-pair delta G

# confidence and the deployable models
python3 scripts/confidence_tune.py     # writes conf_oof_A.npz, conf_oof_B.npz
python3 scripts/deploy_final.py        # reads the OOF caches; final intervals

# active analysis
python3 scripts/active_analysis_trends.py
python3 scripts/picks_trends.py

# figures, workbook, slides
python3 scripts/make_figures.py
python3 scripts/build_workbook2.py
python3 scripts/build_slides.py
```

The `How to run` block in `README.md` lists an older, shorter sequence centered on the confidence and Zhang scripts; the commands above add the audit, delta G, and active-analysis scripts that produce the corrected headline numbers in this document.

### Which script produces which number and figure

logD tracks (cleaned set, replicate logD range ≤ 2):

- `logd_audit.py` reproduces both logD tracks from scratch with one consistent LightGBM and writes `results/logd_audit_results.csv` and `results/track_ab_results.csv`. These hold Track A (new-molecule, molecule-grouped, R2 0.481 / RMSE 1.131 in the audit table; the single-model deliverable figure is R2 0.466 / RMSE 1.148), the Track B random-row upper bound (R2 0.68 / RMSE 0.888), the honest condition-key-grouped Track B (R2 0.607 / RMSE 0.984), the molecule-grouped collapse of Track B's rich features to about Track A (R2 0.443 / RMSE 1.172), and the label-noise floor (within-condition-key logD std about 0.445).
- `track_b_clean.py` and `track_b_molecule_effect.py` are the supporting Track B analyses: the label-quality cleaning test and the explicit per-molecule effect (target encoding versus the oracle ceiling).
- `baseline_v39.py` is the descriptor-and-tree feature-set baseline on the V27/V39 split under molecule-grouped CV (ALL vs ECFP+RDKit+conditions vs ECFP+conditions).

Free energy (delta G):

- `dg_model.py` produces the per-row and per-pair delta G results in `results/dg_results.csv` (per-row R2 0.282 / RMSE 7.55; per-pair R2 0.473 / RMSE 6.31 over 2273 systems) and the `figures/dg.png` panel. The 0.473 is the selected maximum of the model search; the shuffled-split mean of about 0.461 ± 0.009 and the headline R2 of about 0.46 ± 0.01 come from the repeated grouped splits.
- `dg_hpo.py` is the honest hyperparameter search over RandomForest, XGBoost, and LightGBM on the per-pair target (all under the same molecule-grouped 5-fold CV), writing `results/dg_hpo_results.csv`; it is the evidence that a tuned model does not beat the default RandomForest.
- `dg_ensemble.py` is the seven-learner ensemble sweep showing the NNLS stack only ties RandomForest. `dg_tabpfn.py`, `dg_sweep2.py`, `dg_confidence_sweep.py`, and `dg_coverage.py` are the supporting delta G confidence and coverage runs.

Confidence and deployment:

- `confidence_tune.py` runs the err-model bakeoff on the honest metric (RMSE at top-k and Spearman, not the variance-confounded top-k R2) and caches the OOF predictions.
- `deploy_final.py` builds the final deployable per track from those caches: the LightGBM error model that ranks confidence, the confidence-filtered R2 and RMSE, and the normalized split-conformal intervals (90 percent covers 90.2, 80 covers 80.2, on a molecule-disjoint split). It writes the `results/deploy_*_predictions.csv` files. The Track A top-10 percent operating point (R2 0.912, RMSE 0.493) is reported here; the Track B selective figure, recomputed on the condition-key split, is R2 0.874 (RMSE 0.430); the old 0.940 was leaky.
- `metal_confidence.py`, `classifier_confidence.py`, and `xgb_confidence.py` produce the per-metal and per-pair confidence tables and the classifier confidence results in `results/`. `tabpfn_in_stack.py` and `ensemble_final.py` are the TabPFN-in-stack and NNLS-stack experiments.

Active analysis:

- `active_analysis_trends.py` writes `results/active_analysis_honest.csv` (greedy 81 percent strongest-tertile share, precision@10 percent 0.427, depth captured 0.62; UCB ties or loses; random near zero) plus the by-metal and composition tables, and `figures/active_analysis_trends.png`.
- `picks_trends.py` writes `results/picks_metal_enrichment.csv` (Am(III) over-represented 2.85x) and `results/picks_feature_trends.csv` plus `figures/picks_trends.png`.
- `active_learning_ucb.py`, `ucb_analysis.py`, `ucb_graphs.py`, `dg_ucb.py`, `dg_pseudo.py`, and `experiment_triage.py` are the round-by-round active-learning, UCB, pseudo-labeling, and triage analyses (uncertainty sampling improves the model fastest, R2 0.56 vs 0.44; pseudo-labeling adds nothing).

Zhang comparison (requires the external `/tmp` featurized files):

- `zhang_his_split.py` evaluates our model on his exact features, split, and 494-row holdout, writing `results/zhang_his_split_results.csv`.
- `zhang_2x2.py` fills the model-by-split 2x2 (our and his model on our and his split) to show the split drives the gap, writing `results/zhang_2x2_results.csv`.
- `zhang_data_model.py` trains our model on his combined data, writing `results/zhang_data_results.csv`.

Sign and magnitude checks: `sign_separation.py` and `track_a_separation.py` write the sign-correctness and separation-magnitude tables in `results/`.

Figures, workbook, and slides:

- `make_figures.py` reads the saved result CSVs and writes one PNG per figure into `figures/` plus the combined `figures/all_figures.pdf` (predicted-vs-actual, confidence curves, conformal calibration, per-metal accuracy, Zhang comparison panels). It does not retrain anything, so it depends on the result CSVs already being present.
- `build_workbook2.py` assembles `docs/REE_Results_Organized.xlsx` from the result CSVs (including `dg_results.csv`); `build_slides.py` and `dg_slides.py` build the slide decks under `docs/`.

### Caveats for a re-runner

The numbers are reproducible on the cleaned set and are an upper bound relative to the raw measurements; the irreducible label-noise floor is about 0.445 log units. The committed `results/` and `figures/` are the reference outputs, so a re-run should match them within seeding noise. Two things will not reproduce out of the box: the Zhang scripts need his `/tmp` featurized files, and the `.gitignore` currently carries an unresolved merge-conflict block (the `<<<<<<<` / `=======` / `>>>>>>>` markers around the Office lock-file rules) that should be resolved before the repo is treated as clean.

