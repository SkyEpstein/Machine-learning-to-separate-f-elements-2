# Plan: separation recommender

Date: 2026-07-01  |  Spec: ./spec.md

## Approach summary
A query engine `sep_recommend.py`. Given a target pair (metal_i, metal_j), it scores every candidate extractant across a bounded conditions grid using the differenced logD model, keeps each candidate's best conditions, and ranks candidates by predicted separation with a confidence companion. Separation is `logD_i - logD_j`; a good separator has a large absolute value (one metal into the organic phase, the other left behind).

## Design
1. Featurizer and lookups: build a metal -> descriptor lookup and an extractant -> descriptor lookup from the data (both are constant per entity). Fix the diluent to the most common one and expose it as a parameter. Assemble a feature row for any (extractant, metal, conditions) triple, using the same feature columns as the separation eval (extractant descriptors + conditions + metal descriptors + acid one-hot).
2. Model: one logD regressor (new-extractant regime features) trained on all data for deployment, plus a cross-fitted error model for confidence. For the honest validation, also compute molecule-grouped out-of-fold logD and error, so ranking a current extractant never uses its own rows.
3. Conditions grid: acid type from the observed set, acid concentration, temperature, and extractant concentration each over a small grid bounded to observed ranges (no extrapolation beyond the data).
4. Recommend(pair, candidates, grid): for each candidate and grid point, predict logD_i and logD_j, compute predicted separation and pair uncertainty; keep the conditions maximizing predicted separation; rank candidates. Report predicted separation, sign, confidence, best conditions, and the measured separation where the pair was actually run on that extractant.
5. Pluggable candidates: the candidate set is a list of (id, extractant-descriptor-vector). Now it is the current extractants; a molGen SMILES pool is featurized the same way and drops in.

## Method-selection note
Separation is by differencing, not a direct delta model: the direct model was already bake-off tested and lost badly (signed R2 -1.92 vs 0.188). No new bake-off needed; reuse the established winner.

## Honesty
Rank by predicted separation but present as triage. Always show the confidence companion and lead with it. State the trust level from the separation eval (new-extractant Spearman ~0.46, direction ~0.66) so no one reads the shortlist as precise. Validate the ranking on current data (does it enrich for high measured separation) and report that enrichment honestly, including where it fails (adjacent rare earths).

## Risks
- Conditions extrapolation: mitigated by bounding the grid to observed ranges.
- Over-trust of the shortlist: mitigated by the confidence companion and the explicit triage framing.
- Feature assembly for arbitrary combos: validated by reproducing a known measured system's prediction before trusting the grid.
