# Validation: new-extractant confidence bake-off

## Success criteria
- [ ] All four methods are evaluated on the SAME molecule-grouped OOF logD predictions and the SAME separation pairs (only the confidence ranking differs)
- [ ] Every uncertainty is cross-fit on disjoint molecule groups (no extractant leakage into its own confidence estimate)
- [ ] The metric suite (signed R2, RMSE, magnitude R2, direction acc, Spearman, useful-F1) and calibration rho are reported at coverage 100/75/50/25/10 for the new-extractant regime, both metrics shown
- [ ] A winner is declared by the pre-registered composite (avg lift in direction/Spearman/useful-F1 at 25/10 + calibration), not cherry-picked
- [ ] If the winner beats the incumbent, the lift is shown with the honest note that selective R2/RMSE partly reflect variance shrinkage; if nothing beats the incumbent, that is stated plainly
- [ ] Results saved to results/ and a selective-prediction figure produced

## Reflection review
- [ ] a separate adversarial pass confirms: no leakage in the AD/descriptor neighbor computation (train-fold only), the composite was applied as pre-registered, and no overclaiming from variance shrinkage

## Merge checklist
- [ ] reproducible script + results file
- [ ] winner wired into the new-extractant confidence gate (recommender + AutoData shortlist)
- [ ] all artifacts updated (methods, results table, deck, workbook, changelog)
- [ ] commit message confirmed with Skyler and pushed under SkyEpstein
