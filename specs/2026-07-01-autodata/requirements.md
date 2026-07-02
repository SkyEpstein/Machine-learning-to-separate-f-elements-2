# Requirements: AutoData candidate-extractant generator

## Why
The binding constraint on the models is the roughly 295 distinct extractants. AutoData (arxiv 2606.25996) is an agentic generate-evaluate-refine loop for high-quality synthetic data. Here it generates high-quality CANDIDATE EXTRACTANTS to expand chemical space and feed the separation recommender and the closed experimental loop.

## Scope
- In: an agentic loop that proposes candidate extractant molecules (SMILES) for a target f-element pair, scores them, analyzes failures, and refines its strategy, producing a ranked candidate pool that plugs into sep_recommend.py.
- Out: training the model on synthetic predicted labels (the pseudo-labeling experiment proved this does not help); wet-lab synthesis (the real-label step of the loop).

## Key decisions (resolved with Skyler, 2026-07-01)
- Generation engine: HYBRID. LLM agents propose SMILES, RDKit validates and repairs them, and mutation of known extractants fills gaps.
- What it generates: candidate extractant molecules (SMILES); conditions come from the recommender's grid.
- Scoring: PREDICTED SEPARATION plus CALIBRATED UNCERTAINTY for the target pair is the ranking score; chemical validity and novelty versus the 295 are prerequisite gates. Synthesizability and diversity are deferred to a later version.
- v1 scope: the FULL meta-optimizing loop (generate, evaluate, analyze failures, refine the generation strategy, repeat).
- Target pair: Am(III)/Eu(III) first, then adjacent lanthanides (Nd/Dy) later.

## Context and constraints (honest)
- Predicted labels are NOT ground truth. Candidates carry a predicted separation and a calibrated uncertainty and are flagged for lab validation. New-extractant separation magnitude is unreliable (only direction and the confident subset are usable), so the generator's value is surfacing promising, novel, valid, synthesizable candidates for experiments, not "designing the best extractant."
- Constitution applies: honest evaluation, both R-squared and RMSE, bake-off for method choices, verify with an adversarial review, update every artifact.
