# Spec: separation recommender (best conditions + extractant for a metal pair)

Date: 2026-07-01  |  Status: draft (awaiting plan confirmation)

## Problem and why it matters
Given a target f-element pair to separate, recommend the extractant and the reaction conditions that maximize the separation factor, as a ranked, confidence-flagged shortlist for wet-lab prioritization. This turns the predictive model into a design tool: the actual goal of the project is choosing what to run next, not scoring what was already run.

## Scope
- In scope: a query engine that takes a target metal pair, a candidate extractant set, and a conditions grid; jointly optimizes conditions per candidate; and ranks candidates by predicted separation with a confidence companion.
- Out of scope (for now): the generative molGen/LLM model that will produce novel candidates (that is a separate project); wet-lab validation.

## Decisions (from the clarify step)
- Candidate source: the eventual source is molGen (LLM-based generation), but for now the engine runs on the current data (the ~295 measured extractants). The candidate source is a pluggable input so a molGen SMILES pool drops in later with no change to the engine.
- Conditions: jointly optimized per candidate (grid search over acid, concentration, temperature, extractant concentration), reporting the best conditions for each.

## Functional requirements
- FR1: given (metal_i, metal_j), for each candidate extractant, grid-search the conditions and return the conditions that maximize the predicted separation factor.
- FR2: rank candidates by predicted separation, always paired with the confidence companion (pair uncertainty and a confidence-filtered view), per the always-show-confidence rule.
- FR3: emit a ranked shortlist (extractant, best conditions, predicted separation and sign, confidence, and the measured separation where it exists).
- FR4: candidate source is pluggable: anything featurizable from SMILES (current extractants now, molGen candidates later).

## Acceptance criteria
- AC1: the engine runs for a specified pair and writes a ranked shortlist CSV.
- AC2: the ranking is validated honestly on current data (enrichment / Spearman of predicted vs measured separation), with the trust level stated explicitly.
- AC3: every ranking shows the confidence companion, and the output states plainly that the new-extractant regime is triage, not a precise optimizer.

## Evaluation contract and leakage map
- Model: separation by differencing the logD model in the new-extractant regime (molecule-grouped: conditions + metal descriptors + extractant descriptors), with confidence from a cross-fitted error model. For validating the ranking of current extractants, use molecule-grouped out-of-fold predictions so a candidate's own measured rows never rank it. For deployment on a truly new (molGen) candidate, train on all data and predict.
- Metric: does ranking by predicted separation enrich for high measured separation (precision@k and Spearman) for a target pair. The trust level is set by the already-measured separation quality: new-extractant Spearman ~0.46, direction ~0.66; known-extractant Spearman ~0.60, direction ~0.73.
- Limit: magnitude is poorly predicted and the new-extractant regime is the weakest, so the deliverable is a triage shortlist, not a guaranteed separation factor. Conditions grid is bounded to observed data ranges to avoid wild extrapolation.

## Clarifications
- Q (2026-07-01): where do candidate new extractants come from? -> A: molGen (LLM-based) eventually; for now run on current data, pluggable so molGen drops in later.
- Q (2026-07-01): how are conditions handled? -> A: jointly optimize per candidate.
