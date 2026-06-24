# Machine learning for f-element separation

This project uses LLM-assisted coding to predict how well a certain extractant will work to separate two f-elements. Inputs include molecular descriptors, SMILES, molecular fingerprints, and environmental factors (e.g. pH and temperature), with the target output being logD (The log of the distribution coefficient, D).

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

## Project Structure

```
.
├── data/                           # Data files
│   └── data.zip                    # Raw training and test data (unzip to get CSVs)
├── scripts/                        # All runnable Python scripts
│   ├── confidence_tune.py
│   ├── deploy_final.py
│   ├── metal_confidence.py
│   ├── classifier_confidence.py
│   ├── xgb_confidence.py
│   ├── tabpfn_in_stack.py
│   ├── ensemble_final.py
│   ├── zhang_2x2.py
│   ├── zhang_data_model.py
│   ├── zhang_his_split.py
│   ├── make_figures.py
│   ├── build_workbook2.py
│   └── build_slides.py
├── results/                        # Generated result CSV files
│   ├── classifier_confidence_results.csv
│   ├── metal_confidence_by_metal.csv
│   ├── metal_confidence_by_pair.csv
│   ├── xgb_confidence_results.csv
│   ├── zhang_2x2_results.csv
│   ├── zhang_data_results.csv
│   └── zhang_his_split_results.csv
├── figures/                        # Generated visualizations
├── docs/                           # Documentation and outputs
│   ├── METHODS_AND_RESULTS.md      # Long-form methods and results writeup
│   ├── REE_Results_Organized.xlsx  # Results spreadsheet
│   └── REE_Results_Slides_Final.pptx  # Presentation slides
├── requirements.txt
└── README.md
```

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

## Documentation

See `docs/METHODS_AND_RESULTS.md` for the long-form methods and results write-up.
