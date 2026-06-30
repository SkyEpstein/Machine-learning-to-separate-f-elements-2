# Roadmap

## Done
- Data assembly and cleaning, and the four feature groups (extractant structure, metal descriptors, acid and conditions, diluent)
- Track A, the new-molecule logD screen, and Track B, known-molecule condition optimization
- The free-energy (delta G) model and the seven-model bake-off (RandomForest chosen)
- The confidence layer and the calibrated split-conformal intervals
- Active analysis: greedy, UCB, uncertainty, and random acquisition; experiment triage; sequential active learning; the pseudo-labeling negative result; and the pick trends
- Per-metal and per-pair analysis; the sign-correctness and separation-magnitude analyses; classifiers with confidence
- The comparison with Dr. Zhang's model
- A full adversarial audit and the honest-number corrections
- Documentation: README, METHODS_AND_RESULTS.md, the workbook, the deck, and the project constitution

## In progress
- Scrub the remaining old literals in a couple of workbook sheets (already superseded by the Audit corrections tab)
- Recompute the Track B most-confident-10 percent logD curve on the leakage-free condition-key split

## Planned
- Run the closed loop: rank candidate experiments, run the top ones in the lab, feed results back, retrain
- Operationalize the UCB 90-percent-zone candidate screen into a standing recommended-experiment list with intervals
- Grow the dataset past the roughly 295 distinct extractants, which is the binding constraint on new-molecule accuracy
- Target specific f-element pairs of interest for separation (to be specified by Skyler)
