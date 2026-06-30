# Tech stack

## Language and core libraries
- Python 3
- numpy and pandas for data handling
- scikit-learn: RandomForest, ExtraTrees, HistGradientBoosting, Ridge, GroupKFold, metrics
- LightGBM, XGBoost, CatBoost: gradient-boosted tree models
- scipy: non-negative least squares (NNLS) for ensemble stacking, and statistics

## Chemistry and features
- RDKit: Morgan (ECFP) fingerprints and molecular descriptors computed from SMILES
- Curated numeric tables of metal properties and diluent properties
- A 768-dimensional learned molecular embedding exists in the data but is excluded, because it hurts new-molecule generalization (see the rejected-approaches section of METHODS_AND_RESULTS.md)

## Uncertainty
- A learned error model (LightGBM) that predicts each prediction's absolute error, used to rank confidence
- Normalized split-conformal prediction intervals, validated by an honest coverage check

## Outputs and reporting
- matplotlib for figures, saved as opaque RGB PNGs for PowerPoint compatibility
- openpyxl for the results workbook (docs/REE_Results_Organized.xlsx)
- python-pptx for the slide deck (docs/REE_Results_Slides_Final.pptx)

## Data
- CSV files Training_Data_V27.csv and Testing_Data_V39.csv (molecule-disjoint), shipped zipped as data/data.zip

## Tooling and deployment
- git and GitHub, public repository, pushed under the SkyEpstein account
- There is no production service. The deliverables are the model scripts, the prediction tables, the workbook, the deck, and the documentation
- Optional deep-learning extras (a Chemprop graph network, TabPFN) were tried and dropped; they are documented but not part of the deployed pipeline
