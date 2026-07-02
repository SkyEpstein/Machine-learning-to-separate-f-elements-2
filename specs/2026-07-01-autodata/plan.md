# Plan: AutoData candidate-extractant generator

| # | Task group | Status |
|---|------------|--------|
| 1 | Generation: agentic loop proposes candidate SMILES for a target pair (seeded from known extractants and target-pair chemistry) | done (Workflow autodata-extractant-loop: 4 rounds x 4 parallel LLM generators, soft-donor Am/Eu seed; 142 candidates generated) |
| 2 | Evaluation: score each candidate (RDKit validity, novelty vs the 295, predicted separation plus uncertainty) | done (autodata_score.py; sanity-checked: soft-donor dithioamide -> highest predicted Am/Eu sep) |
| 3 | Failure analysis and refinement: analyze rejected candidates and refine the generation strategy (the meta-optimization step) | done (per-round evaluator agent analyzed failures - flagged invalid O-P alkoxy forms and rewrote the strategy toward the dithiophosphinic core each round) |
| 4 | Integration: rank the accepted pool into EXPLORE (optimistic UCB) and CONFIDENT (low-uncertainty lower-bound) shortlists via the parity-featurized scorer, with the 295 known extractants as a same-model baseline | done (autodata_pool.py -> candidate_pool.csv, shortlist_explore.csv, shortlist_confident.csv, figures/autodata_pool.png) |
| 5 | Honest validation: report pool validity, novelty, diversity, predicted-separation distribution and the honest caveats; adversarial reflection review | done (Workflow autodata-reflection-review): verdict SHIP-WITH-FIXES; applied the majors (in-sample-baseline relabel, parse-validity vs dedup split, dropped "calibrated", novelty scoped as new-to-dataset). |
| 6 | Ship: update all artifacts (methods, results table, deck, workbook, changelog), confirm commit message, push under SkyEpstein | in progress (CHANGELOG done; docs/results-table/deck/workbook + commit pending) |

Status values: todo, in progress, done.
