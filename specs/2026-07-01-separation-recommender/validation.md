# Validation: separation recommender

Phase 7. Results from `scripts/sep_recommend.py` on current data (3 demo pairs). The spec is the source of truth; limitations are stated, not hidden.

## Success criteria
- [x] AC1: the engine runs for a target pair and writes a ranked shortlist CSV (`results/sep_recommend_shortlists.csv`), with jointly-optimized conditions per candidate.
- [x] AC2: ranking validated honestly on current data (OOF-predicted vs measured separation at matched conditions), trust level stated.
- [x] AC3: every pair shows the confidence-forward list alongside the explore list; the output is labeled a triage/discovery tool, not a precise optimizer.

## What it produces
Two shortlists per pair: EXPLORE (uncertain but promising, predicted |sep| >= 1, ranked by the optimistic upper bound UCB = |sep| + uncertainty) and CONFIDENT best-bets (ranked by the lower bound |sep| - uncertainty). Candidates are the current extractants; a molGen SMILES pool plugs into the same interface.

## Honest results
- Encouraging sanity signal: the CONFIDENT best-bets for Am(III)/Eu(III) surface soft sulfur-donor ligands (a dithioamide at rank 1), which is the correct real-world chemistry for that separation. The model recovered known selectivity chemistry unprompted.
- Trust is low, as the separation eval already established. The only demo pair with enough matched systems to validate is Am(III)/Eu(III) (18 systems): direction accuracy 0.72 but |sep| Spearman 0.07, so the magnitude ranking is essentially uninformative. Dy/Nd (5 systems) and Eu/Gd (6 systems) have too few matched systems to validate.
- A confidently wrong case exists: one Am/Eu confident pick has predicted |sep| 2.6 but measured 0.01. Confidence bounds the average error, not every case.

## Verdict and honest framing
The engine is a usable design tool for triage and discovery: trust the direction it predicts and the chemistry it surfaces, not the separation magnitude. The reliable regime remains conditions-for-a-known-extractant; recommending a new extractant is genuine exploration, better than random but low-trust per candidate. This matches the adversarially-verified finding that new-extractant separation magnitude is not predictable. The design tool is honest about this in its output.

## Reflection review notes
- Feature assembly is exercised by the OOF validation (direction 0.72 for Am/Eu matches the separation eval's 0.66 to 0.73 range), so the assembled feature rows produce sensible predictions.
- The pure-UCB degeneracy (ranking the model's blind spots) was caught and fixed by the |sep| >= 1 promise filter plus the confident list.
- Open follow-up when molGen lands: re-run the validation on a held-out set of truly novel scaffolds, since current validation is on known extractants.
