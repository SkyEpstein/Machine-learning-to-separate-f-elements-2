# Project constitution and spec sheet
### Machine learning for f-element separation

## 1. Mission
Build an honest, reproducible machine-learning system that predicts how well an extractant separates two f-elements, and, more importantly, guides which experiments to run next so that selective extractants are found with far fewer experiments than brute-force testing. The code and modeling are LLM-assisted. The scientific manuscript is written by Skyler Epstein.

## 2. Scope
**In scope:** prediction of logD and the free energy of extraction (delta G); calibrated per-prediction uncertainty; active analysis (ranking which experiments to run, and triage of which predictions to trust); the closed-loop discovery cycle; the comparison with Dr. Zhang's published model.

**Out of scope:** wet-lab synthesis and measurement (the experiment step of the loop is done by chemists); the manuscript prose; production deployment infrastructure.

## 3. Core principles (non-negotiable)
Reinforced by a full adversarial audit; every result must satisfy these.

1. **Evaluation honesty.** A "new molecule" claim uses molecule-grouped cross-validation, with no molecule (directly or through replicate or fingerprint siblings) in both train and test. Every number states its evaluation regime.
2. **No overclaiming.** Report the honest number. Any figure that is an upper bound (for example the random-row Track B 0.725) is labeled as such. A selected maximum is reported with its mean and spread, not as a point fact.
3. **Both metrics.** Always report R-squared and RMSE together, never one alone.
4. **Calibrated uncertainty is the product.** Intervals are validated by an honest coverage check before they are quoted. Confidence-filtered "top X percent" numbers are selective-prediction operating points, not overall accuracy, and lead with the RMSE.
5. **Reproducibility.** Every headline number is backed by committed code and a results file. No hardcoded literals without a script that regenerates them, and scripts in the repository must actually run.
6. **Fair comparison.** External comparisons use the same data, split, and metric, with no cherry-picking; figures not reproduced are labeled as external.
7. **Data transparency.** Cleaning rules and the label-noise floor are stated, and all numbers are understood as upper bounds on the cleaned data.
8. **Plain, honest writing.** Plain technical prose, no em dashes, no marketing or AI-sounding language.
9. **Audit before publishing.** Adversarially check for leakage, overfitting, and optimistic framing before a result is presented or pushed.
10. **Evidence-based selection.** Every model, feature set, confidence estimator, and acquisition rule is chosen by a documented bake-off under the same honest evaluation, never arbitrarily. Rejected approaches are recorded alongside the winners.

## 4. Method selection by bake-off
New methods are never adopted on intuition. Every modeling choice is decided by a bake-off: all candidate models, feature sets, confidence estimators, and acquisition rules are run under the same honest cross-validation, and the winner is chosen by the metrics (R-squared and RMSE together), with the full comparison saved. Nothing in the deployed system is arbitrarily selected, and the approaches that lost are documented next to the ones that won, so any choice can be traced to the evidence behind it.

Comparisons already on record:
- **Models:** a seven-way sweep (RandomForest, ExtraTrees, HistGB, LightGBM, XGBoost, CatBoost, Ridge) plus equal-weight and NNLS-stacked ensembles on the same molecule-grouped folds. RandomForest won the delta G target; the NNLS stack won the logD tracks; Ridge was near zero, confirming the signal is nonlinear.
- **Confidence:** a sweep of confidence rankers (LightGBM err model, RandomForest err model, ExtraTrees err model, and RandomForest tree-spread). The learned LightGBM err model won; the native tree-spread was weakest.
- **Features:** a feature-set bake-off found ECFP plus conditions best. RDKit descriptors and the 768 learned embeddings hurt new-molecule generalization, and a graph neural network (about R-squared 0.23 alone) and TabPFN underperformed.
- **Tuning:** a hyperparameter search of dozens of configurations did not beat the defaults, so the defaults are kept.
- **Acquisition:** greedy, UCB, uncertainty sampling, and a random baseline are compared head to head for each goal (discovery versus model improvement).

Every deployed choice is the winner of one of these comparisons, and the losers are kept in the record (the rejected-approaches section of `METHODS_AND_RESULTS.md`, the Decision Log, and the saved sweep result files), so no part of the system looks arbitrary.

## 5. Targets and tracks
- **logD:** log distribution coefficient, the primary measured quantity.
- **delta G = -2.303 R T logD (kJ/mol):** the free-energy framing, a structure-and-metal property with reaction conditions dropped.
- **Track A (new-molecule screen):** conditions and metal features, molecule-grouped CV, single LightGBM. Honest R-squared 0.466, RMSE 1.148.
- **Track B (known-molecule condition optimization):** conditions, ECFP, and ligand features, NNLS-stacked trees. Honest condition-interpolation R-squared about 0.61; the random-row 0.725 is an upper bound; new-molecule about 0.44.
- **delta G per-pair:** RandomForest, molecule-grouped CV, R-squared about 0.46 plus or minus 0.01.
- **Deployed:** Track A single LightGBM, Track B NNLS stack, delta G single RandomForest, each with a calibrated confidence layer.

## 6. Active analysis
- **Acquisition rules:** greedy (rank by prediction), UCB (prediction plus uncertainty), uncertainty sampling, and a random baseline.
- **Triage:** trust and skip confident predictions; send uncertain ones to the lab.
- **The loop:** predict, prioritize, run experiments, feed results back, retrain, repeat.
- **Purpose:** find selective extractants for a target f-element pair with minimal experiments; each round expands the roughly 295-molecule dataset, which is the binding constraint.

## 7. Known limits
- The dataset has about 295 distinct extractants, which caps new-molecule accuracy near R-squared 0.46. The base model is moderate, so the system is a prioritizer with calibrated uncertainty, not an oracle.
- The lever for improvement is more distinct extractants, obtained by running the loop, not a different model: a full model sweep and hyperparameter search did not beat the defaults.

## 8. Success criteria
Success is not a high raw R-squared. It is calibrated, honest uncertainty; an active-analysis screen that reliably concentrates good extractants in its top picks and tells the user which predictions to trust; and a fully reproducible, audited, honestly documented repository.

## 9. Roles and logistics
- **Skyler Epstein:** direction, the manuscript, running experiments.
- **Assistant:** coding, modeling, analysis, documentation.
- **Repository:** public, github.com/SkyEpstein/Machine-learning-to-separate-f-elements-2, pushed under SkyEpstein; training data intentionally public.
- **Governance artifacts:** `docs/METHODS_AND_RESULTS.md`, `docs/TECH_STACK.md`, `docs/ROADMAP.md`, `CHANGELOG.md`, the `specs/` feature specs, the `spec-feature` skill, and the Decision Log and Audit corrections tab in the workbook.

## 10. Amendment
Reviewed and amended by Skyler. Material modeling decisions are recorded in the Decision Log; audit corrections are recorded in the Audit corrections tab.
