# Predicting logD for rare-earth solvent extraction

This document describes the data, models, confidence method, and results. Metrics reported are R² (higher is better) and RMSE in log units (lower is better).

## Problem and data
- Target: logD across ~17.5 log units.  
- Dataset: 8,075 raw rows spanning 295 distinct extractant molecules and 28 f-element metals (lanthanides plus Am, Cm, Cf, Np, Pu, Pa, Th, U).  
- Each row: SMILES string, metal, acid type & concentration, temperature, extractant concentration, diluent, measured logD.

Because the dataset contains relatively few distinct molecules (295), interpolation for known molecules is easier than extrapolating to new molecules; this motivates the two-track evaluation below.

## Two prediction tasks
- **Track A (new-molecule screening):** molecule-grouped cross-validation (no molecule in both training and test). Measures generalization to genuinely new extractants.
- **Track B (condition optimization for known molecules):** random-row cross-validation (same molecule may appear in train and test at different conditions). Measures interpolation across conditions.

Keeping the tasks separate avoids leakage that inflates performance (an earlier single split reported R² = 0.60 but did not hold up when leakage was removed).

## Data cleaning
- Remove exact duplicate rows.
- Remove any replicate group (same molecule, metal, conditions) whose logD spans > 2 log units (contradictory labels).
- Cleaning reduces rows from 8,075 to 7,065.
- Replicate scatter sets a noise floor near RMSE ≈ 0.77 on raw data.

## Features
- Track A: conditions and metal descriptors only (molecular structure did not help for genuinely new molecules).
- Track B: conditions, ECFP fingerprint, and a few simple ligand descriptors. RDKit descriptor blocks and pretrained embeddings were tested but did not outperform ECFP+conditions.

## Models
Tested models: LightGBM, XGBoost, CatBoost, HistGradientBoosting, ExtraTrees, ridge regression, a small neural network, a Chemprop graph network, and TabPFN.

- Track A: single tuned LightGBM chosen (better residual predictability for confidence).
- Track B: NNLS stack of tree models. NNLS sets non-helpful model weights to zero; final weights: ExtraTrees 0.43, LightGBM 0.40, XGBoost 0.14, CatBoost 0.03.
- Graph network: R² ≈ 0.23 on new molecules — added nothing in blends.
- Global TabPFN (CPU-limited to 1000 context rows): R² ≈ 0.36 by itself.

## Confidence and uncertainty
- A second model predicts absolute error for each prediction using conditions and the prediction itself.
- Rows ranked by predicted error; most confident predictions are far more accurate.
- Uncertainty intervals from normalized split-conformal: interval width scales with predicted error and is calibrated to target coverage (90% and 80% targets), matching measured coverage.

## Results

### Track A (new molecules, LightGBM, molecule-grouped CV)
- All rows: R² = 0.466, RMSE = 1.148  
- Most confident 50%: R² = 0.719, RMSE = 0.812  
- Most confident 25%: R² = 0.825, RMSE = 0.642  
- Most confident 10%: R² = 0.912, RMSE = 0.493  
- Conformal intervals: measured coverage ≈ 0.89 (target 90%), median width = 3.61 log units

### Track B (known molecules, NNLS tree stack, random-row CV)
- All rows: R² = 0.725, RMSE = 0.823  
- Most confident 50%: R² = 0.871, RMSE = 0.535  
- Most confident 25%: R² = 0.911, RMSE = 0.424  
- Most confident 10%: R² = 0.940, RMSE = 0.341  
- Conformal coverage: 0.90 at 90% target, median width = 2.34 log units

Track B improves on an earlier best known-molecule test RMSE near 0.96.

## Confidence by metal and metal pair
- Strongest metals (Track B): Am(III) R² = 0.801, RMSE = 0.656 (n=1121); Cm(III) R² = 0.784, RMSE = 0.436.  
- Weakest: Er(III) R² = 0.391, RMSE = 1.392; Np(V) R² = 0.053, RMSE = 1.027; U(VI) R² = 0.314, RMSE = 1.062.  
- Because predicted error tracks observed RMSE, the model flags metals it predicts poorly.  
- Separation (difference in logD between two metals at same conditions): overall R² = 0.599, RMSE = 1.025. Pairs far apart in logD (e.g. Dy/Yb) predicted well; adjacent pairs (e.g. Er/Tb) are hardest (R² near 0, RMSE ≈ 2.53), and confidence is highest for those rows.

## TabPFN experiment
- Tried local TabPFN experts per cluster (to stay within context limit), then NNLS to weight them.
- With 6 regions and 3 folds, TabPFN got NNLS weight 0.13 but raised R² by only +0.0011; error-correlation with tree stack = 0.82. Not diverse enough, so dropped.

## Comparison to Dr. Zhang's model (SAFE-MolGen)
- Dr. Zhang: LLM-generated candidate extractants + supervised XGBoost classifier fit on 8,075 rows with 1,860 features (Morgan fingerprints + conditions). Reported 72% accuracy on a 3-class task from a fixed 494-row molecule-held-out test.
- This project and Zhang likely use the same integrated dataset (ACSEPT, DGA, IDEaL, ORNL).
- When our regression is binned to his 3-class thresholds (0.5 and 10) on Track A new molecules: accuracy = 0.623, macro-F1 = 0.619 (rising with confidence to 0.847 on the most confident 10%).
- Reproducing his XGBoost classifier under molecule-grouped CV scored 0.605 (below his reported 0.72), suggesting his single 494-row holdout and class imbalance explain some of the difference.
- Adding the confidence layer to his XGBoost lifts the confident tenth to 0.914 and the quarter to 0.847; the binary extract screen ROC AUC = 0.798.

## Classifiers with confidence
- Native 3-class and binary classifiers were trained (with confidence = predicted probability of chosen class).
- Track A (new molecules):
  - 3-class: 0.625 overall; 0.766/0.854/0.912 on most confident 50%/25%/10%
  - Binary (logD > 0): 0.742 accuracy, ROC AUC = 0.817; rises to 0.962 on most confident 10%
- Track B (known molecules):
  - 3-class: 0.753 overall; 0.906/0.956/0.983 on most confident 50%/25%/10%
  - Binary: 0.832 accuracy; ROC AUC = 0.910

## Limitations
- New-molecule accuracy limited by only 295 distinct extractants — more distinct molecules needed to raise Track A performance.
- Track B bounded by label noise.
- Comparison to Dr. Zhang close but not exact due to inferred class cut points and possible cleaning differences.

## How to run
Place `Training_Data_V27.csv` and `Testing_Data_V39.csv` in the working directory, then:
```bash
python3 confidence_tune.py
python3 deploy_final.py
python3 metal_confidence.py
python3 classifier_confidence.py
python3 xgb_confidence.py
python3 tabpfn_in_stack.py
python3 build_workbook2.py
```
