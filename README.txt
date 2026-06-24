# Machine learning for f-element separation

This project uses LLM-assisted coding to predict how well a certain extractant will
work to separate two f-elements. Inputted values include molecular descriptors,
SMILES, molecular fingerprints, and environmental factors (e.g. pH and temperature),
with the targeted output being logD (the log of the distribution coefficient, D).
Every prediction also comes with a confidence score and an uncertainty interval, so
the unreliable ones can be set aside. Results are given as both R² (higher is
better) and RMSE (lower is better, in log units).

There are two models. Track A screens a new extractant that has not been tested. It
uses molecule-grouped cross-validation, so no molecule is in both the training and
testing sets, and it reaches R² 0.466 and RMSE 1.148 overall, or R² 0.912 and RMSE
0.493 on its most confident 10%. Track B tunes the conditions of an extractant that
is already in the data. It uses random-row cross-validation and reaches R² 0.725 and
RMSE 0.823 overall, or R² 0.940 and RMSE 0.341 on its most confident 10%. The two are
kept apart because a molecule in both sets inflates the score, and an earlier split
that mixed them read 0.60 and did not hold up.

## Scripts:
  confidence_tune.py        cross-validated predictions and the confidence comparison.
  deploy_final.py           builds the two final models and writes their predictions.
  ensemble_final.py         the full Track B stack with confidence.
  metal_confidence.py       accuracy and confidence by metal and by metal pair.
  classifier_confidence.py  3-class and yes/no classifiers with confidence.
  zhang_data_model.py       our model trained on Dr. Zhang's data.
  zhang_his_split.py        our model on his exact 494-row test set.
  zhang_2x2.py              our and his model crossed with our and his split.
  xgb_confidence.py         his XGBoost classifier with our confidence added.
  tabpfn_in_stack.py        whether a local TabPFN expert earns a place in the stack.
  make_figures.py           builds the figures.
  build_workbook2.py        builds the results spreadsheet.
  build_slides.py           builds the slide deck.

## Outputs:
  REE_Results_Organized.xlsx   the results spreadsheet.
  REE_Results_Slides.pptx      the slide deck.
  figures/                     the figures, plus all_figures.pdf.
  METHODS_AND_RESULTS.txt      the full write-up.
  the *_results.csv files      per-metal, per-pair, classifier, and Zhang result tables.

## To run:
  pip install -r requirements.txt
  unzip data.zip
Then run the scripts in the order listed above. The Zhang comparison scripts also
need his data files from his repository.

## Data:
The dataset is shipped compressed as data.zip. Unzip it first with "unzip data.zip",
which produces Training_Data_V27.csv (training and validation) and
Testing_Data_V39.csv (held-out test). The per-row prediction files are left out
because they are just model outputs.

## How the confidence works:
A second model is trained to guess how far off each prediction will be, and the
predictions are ranked by that guess, so the most confident ones are much more
accurate than the rest. The intervals are scaled by the same guess and tuned to hit
their target, and they came out to 0.89 to 0.90 against a 90% target.
