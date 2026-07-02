# Validation: AutoData candidate-extractant generator

## Success criteria
- [ ] The loop produces a pool of N valid, novel (not in the 295), diverse candidate extractants for a target f-element pair
- [ ] Each candidate carries a predicted separation and a calibrated uncertainty from our honestly-evaluated model
- [ ] The recommender ranks the pool into EXPLORE (optimistic upper-bound) and CONFIDENT (lower-bound) shortlists, always showing the confidence
- [ ] Pool quality is reported: validity rate, novelty rate, diversity, and the predicted-separation distribution (any model metric used is reported with both R-squared and RMSE)
- [ ] Honest framing throughout: predicted is not truth, candidates are for lab validation, and separation magnitude is low-trust for new extractants

## Reflection review
- [ ] a separate adversarial pass confirms no overclaiming (generated candidates are not presented as validated) and that the predicted-separation model is the honestly-evaluated one

## Merge checklist
- [ ] reproducible script(s) and a results file under results/
- [ ] all artifacts updated (methods, results table, deck, workbook, changelog)
- [ ] commit message confirmed with Skyler and pushed under SkyEpstein
