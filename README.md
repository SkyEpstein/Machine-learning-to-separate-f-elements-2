# Machine learning for f-element separation

This project uses LLM-assisted coding to predict how well a certain extractant will work to separate two f-elements. Inputs include molecular descriptors, SMILES, molecular fingerprints, and environmental factors (e.g. pH and temperature), with the target output being logD (The log of the distribution coefficient, D).

## Dataset features and why they are used

The model sees four kinds of input, and each one was included because it drives part of the extraction chemistry.

**Extractant structure.** The extractant is given as a SMILES string, from which the code computes a Morgan fingerprint (which substructures are present), a block of RDKit molecular descriptors (size, shape, lipophilicity, and ring and functional-group counts), and a few simple ligand descriptors (the counts of oxygen, nitrogen, sulfur, and phosphorus donor atoms, plus the calculated logP). These describe how the extractant can bind a metal, since the donor atoms and the surrounding structure set the strength and the selectivity of that binding.

**Metal descriptors.** Each metal is described by numeric properties rather than just its name: atomic number, ionic radius, oxidation state, metallic radius, the first three ionization energies, Pauling electronegativity, density, melting and boiling points, and the metal concentration. These matter because the ionic radius and the charge drive the lanthanide-contraction trend that makes neighboring rare earths so chemically similar, and giving the model these numbers lets it relate one metal to another along that trend, which a one-hot label cannot do.

**Acid and process conditions.** The acid type and concentration, the temperature, the extractant concentration, and the diluent volume fractions are included because logD is an equilibrium quantity: it shifts with acid concentration and with extractant concentration and with pH, so the conditions are as important as the molecule itself.

**Diluent (solvent) descriptors.** The two diluent components are described by molar mass, logP, boiling and melting points, density, water solubility, and dipole moment, because the diluent sets the organic-phase environment and affects how the extractant aggregates and how well it pulls the metal into the organic phase.

For new molecules (Track A) the structure does not generalize from only about 300 distinct extractants, so that model leans on the conditions and the metal descriptors. For known molecules (Track B) the fingerprint and the ligand descriptors are added and do help, because the molecule has already been seen at other conditions.

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

## ECM: predicting the free energy of extraction

A second framing predicts the Gibbs free energy of extraction (delta G) instead of logD, using only the extractant structure and the metal, with the reaction conditions dropped. The free energy is computed from logD by the standard relation delta G = -2.303 R T logD (in kJ per mole, with R the gas constant and T the temperature), so a favorable extraction gives a negative, spontaneous delta G. The point of this version is that the free energy is meant to be a property of the extractant and the metal rather than of the particular acid concentration or temperature, so the conditions are left out and the model predicts the thermodynamics from structure alone.

This was built two ways. The per-row version keeps every measurement as its own delta G; because identical structures then repeat with different free energies that come from the dropped conditions, the structure cannot explain that spread and the accuracy is limited (R2 = 0.282, RMSE = 7.55 kJ/mol). The per-pair version averages delta G to a single value for each extractant, metal, acid, and diluent system, which is condition-independent by construction, and it is the better target (R2 = 0.473, RMSE = 6.31 kJ/mol over 2,273 systems). Both are scored with molecule-grouped cross-validation, so a new extractant never appears in both training and testing. The model is a RandomForest, chosen by an ensemble sweep over seven base learners: the bagged-tree models beat the gradient boosters on this small, wide table, and an NNLS stack scored with its own cross-validation only ties RandomForest, so stacking is not used here.

The confidence layer carries over and is strong here: for the per-pair model the most confident quarter of predictions reach R2 = 0.683 and the most confident tenth reach R2 = 0.776 (RMSE = 3.62 kJ/mol). So the free-energy model is moderate on its own for a brand-new extractant but becomes reliable on the fraction it is confident about, which is the part that matters for screening.

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
