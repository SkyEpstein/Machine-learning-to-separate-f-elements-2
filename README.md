# Machine learning for f-element separation

This project uses LLM-assisted coding to predict how well a certain extractant will work to separate two f-elements. Inputs include molecular descriptors, SMILES, molecular fingerprints, and environmental factors (e.g. pH and temperature) with the targeted output being logD (The log of the distribution coefficient, D).

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
Scripts live in the `scripts/` directory. Example invocation:

```bash
python3 scripts/confidence_tune.py
python3 scripts/deploy_final.py
```

## Outputs
- Results and figure artifacts are in `results/` and `figures/` respectively.
- The long-form methods and results writeup is at `docs/METHODS_AND_RESULTS.md`.

## How to run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Unpack the data:
   ```bash
   unzip data/data.zip
   ```
3. Run the scripts in roughly this order (adjust as needed):
   ```bash
   python3 scripts/confidence_tune.py
   python3 scripts/deploy_final.py
   python3 scripts/metal_confidence.py
   python3 scripts/classifier_confidence.py
   python3 scripts/xgb_confidence.py
   python3 scripts/tabpfn_in_stack.py
   python3 scripts/build_workbook2.py
   python3 scripts/build_slides.py
   ```

## Data
Unzip `data/data.zip` to produce:
- `Training_Data_V27.csv` (training and validation)
- `Testing_Data_V39.csv` (held-out test)

## Contact / Notes
See `docs/METHODS_AND_RESULTS.md` for the long-form methods and results write-up.
