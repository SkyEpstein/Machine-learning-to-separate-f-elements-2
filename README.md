# Machine learning for f-element separation

This project uses LLM-assisted coding to predict how well a certain extractant will work to separate two f-elements. Inputs include molecular descriptors, SMILES, molecular fingerprints, and environmental factors (e.g. pH and temperature). The target is logD (the log of the distribution coefficient, D). Every prediction is produced with a confidence score and an uncertainty interval so unreliable predictions can be set aside. Results are reported as both R² (higher is better) and RMSE (lower is better, in log units).

## Summary of the two tracks
- **Track A (screening new molecules)**  
  - Evaluated with molecule-grouped cross-validation (no molecule is in both training and test folds).  
  - Single tuned LightGBM model.  
  - Overall: R² = 0.466, RMSE = 1.148.  
  - Most confident 10%: R² = 0.912, RMSE = 0.493.

- **Track B (optimizing conditions for known molecules)**  
  - Evaluated with random-row cross-validation (same molecule may appear in training and test at different conditions).  
  - NNLS-stacked tree ensemble (ExtraTrees, LightGBM, XGBoost, CatBoost).  
  - Overall: R² = 0.725, RMSE = 0.823.  
  - Most confident 10%: R² = 0.940, RMSE = 0.341.

> Note: The two tracks are kept separate because allowing the same molecule in both training and testing inflates performance estimates.

## Scripts
- `confidence_tune.py`        — cross-validated predictions and the confidence comparison  
- `deploy_final.py`           — builds the two final models and writes their predictions  
- `ensemble_final.py`         — the full Track B stack with confidence  
- `metal_confidence.py`       — accuracy and confidence by metal and by metal pair  
- `classifier_confidence.py`  — 3-class and binary classifiers with confidence  
- `zhang_data_model.py`       — our model trained on Dr. Zhang's data  
- `zhang_his_split.py`        — our model on his exact 494-row test set  
- `zhang_2x2.py`              — our and his model crossed with our and his split  
- `xgb_confidence.py`         — his XGBoost classifier with our confidence added  
- `tabpfn_in_stack.py`        — whether a local TabPFN expert earns a place in the stack  
- `make_figures.py`           — builds the figures  
- `build_workbook2.py`        — builds the results spreadsheet  
- `build_slides.py`           — builds the slide deck

## Outputs
- `REE_Results_Organized.xlsx` — results spreadsheet  
- `REE_Results_Slides.pptx`    — slide deck  
- `figures/`                   — figures plus `all_figures.pdf`  
- `METHODS_AND_RESULTS.txt`    — full write-up (also provided in this repo)  
- `*_results.csv`              — per-metal, per-pair, classifier, and Zhang result tables

## How to run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Unpack the data:
   ```bash
   unzip data.zip
   ```
3. Run the scripts in roughly this order (adjust as needed):
   ```bash
   python3 confidence_tune.py
   python3 deploy_final.py
   python3 metal_confidence.py
   python3 classifier_confidence.py
   python3 xgb_confidence.py
   python3 tabpfn_in_stack.py
   python3 build_workbook2.py
   python3 build_slides.py
   ```

> The Zhang comparison scripts require Dr. Zhang's data files from his repository.

## Data
Unzip `data.zip` to produce:
- `Training_Data_V27.csv` (training and validation)
- `Testing_Data_V39.csv` (held-out test)

Per-row prediction files are not included in the repo because they are model outputs.

## How the confidence works
A second model predicts the absolute error for each prediction (from the conditions and the prediction itself). Rows are ranked by predicted error (lower = more confident), which makes the most confident predictions much more accurate. Uncertainty intervals are generated via normalized split-conformal: interval widths scale with the predicted error and are calibrated to nominal coverage levels (90% and 80%). Measured coverage matched the targets (≈0.89–0.90 for the 90% target).

## Contact / Notes
See `METHODS_AND_RESULTS.txt` for the long-form methods and results write-up.
