---
name: experiment-campaign
description: Run a reproducible experiment campaign for the f-element-extraction ML project — a model/feature/hyperparameter bake-off or a repeated-split evaluation — using parallel subagents, a provenance manifest per run, an aggregated master table with mean±std, and a separate adversarial audit. Use when a task means running many independent training runs (a bake-off over base learners, a hyperparameter sweep, repeated shuffled molecule-grouped splits, or a per-pair/per-metal evaluation) and you want the results parallelized, reproducible, and audited before they are believed. Complements spec-feature: spec-feature governs one change end-to-end; this governs the experiment sweep inside step 2.
---

# Experiment campaign workflow

Use this when the honest way to answer a question is "run N independent experiments and compare," not "run one script." That is most of this project: the ~55-way model+HP search, the 7-learner ensemble bake-off, the repeated shuffled molecule-grouped splits behind every mean±std, the per-pair and per-metal tables. This skill makes those runs (1) parallel, (2) reproducible, and (3) audited.

It sits inside `spec-feature` step 2 (Implement). Keep the constitution in force: molecule-grouped CV for any new-molecule claim, condition-key grouping for known-molecule claims, no replicate or fingerprint leakage, R² and RMSE reported together, and method choice by bake-off — never arbitrarily.

## 1. Frame the campaign

Write a one-paragraph campaign spec before launching anything:
- **The grid.** Enumerate the runs explicitly — every (model, feature set, split-seed) cell. A bake-off over 7 learners × 5 seeds is 35 runs; name them.
- **The fixed evaluation.** One split protocol for the whole grid (e.g. molecule-grouped 5-fold, seeds 0–4). Every run uses the identical protocol so the numbers are comparable. Decide it once, here.
- **The metric of record.** R² and RMSE together, plus whatever the objective needs (direction accuracy, Spearman, useful-separation F1 for separation factor). Pick the one number that decides the winner.

## 2. Parallelize with subagents

Run the grid concurrently instead of serially. Launch one subagent (Task tool) per run, or per small batch of runs when each is cheap:
- Give each subagent a **self-contained task**: which model, which feature set, which seed, which data files, where to write its result. It should not depend on any other run.
- Each subagent trains, evaluates under the fixed protocol, and writes **two files** to `results/campaign_<name>/`: a metrics row (CSV or JSON) and a run manifest (see §3). It returns its metric row to the parent.
- Batch sensibly: dozens of fast tree fits → group several per subagent; a few slow fits (Chemprop, TabPFN) → one per subagent. Do not exceed what the machine's cores/RAM support at once.
- Keep raw prediction files (they contain measured logD) out of git per `.gitignore`; commit the metric rows and manifests, not the row-level predictions.

## 3. Provenance manifest (per run)

Every run writes a manifest so any number is reproducible months later. Minimum fields:

```
run_id, timestamp
git_sha            # git rev-parse HEAD
git_dirty          # true if uncommitted changes (a dirty run is not reproducible — flag it)
script             # the exact script/module executed
model, feature_set, hyperparams
split_protocol, seed
data_files         # name + sha256 of each CSV actually read
env_hash           # sha256 of `pip freeze` (or the conda export)
metric_of_record, r2, rmse, n
```

A run whose `git_dirty` is true or whose `data_files` checksums don't match the committed data is a red flag — do not fold it into a headline number until it is clean and re-run.

## 4. Aggregate into a master table

Collect the metric rows into one table under `results/campaign_<name>/master.csv`:
- One row per grid cell, plus a **mean ± std across seeds** for each (model, feature set). The std is the point — a single fixed-split maximum (like the 0.473 dG number that was really the max of a 55-way search) overstates; report the repeated-split mean and its spread instead.
- Rank by the metric of record. Note the winner **and** whether its lead exceeds the seed-to-seed std — if it doesn't, say the bake-off is a tie and prefer the simpler/faster model (this is how RandomForest was chosen over the NNLS stack for dG).

## 5. Adversarial audit (fresh reviewer)

Before believing the winner, launch a **separate** subagent with a clean context whose only job is to attack the result. It sees the master table, the manifests, and the training/eval code — not the reasoning that produced them. It checks:
- **Leakage.** Does any split let a replicate row or a molecule appear in both train and test? Is the fingerprint constant within a molecule under a random-row split (the ~0.07 R² replicate-memorization inflation)? Are known-molecule and new-molecule regimes labeled correctly?
- **Overfitting to the search.** Is the reported number a max over the grid rather than a mean? Was the "winning" model chosen on the same split used to report it?
- **Overclaiming.** Is a selective-prediction operating point (top-10% confidence) being read as overall accuracy? Is RMSE reported alongside R² so a shrinking-variance R² gain isn't mistaken for skill?
- **Reproducibility.** Do the manifests' checksums and git SHAs let someone re-run and land on the same numbers?

The audit returns a pass/fail per check with the specific offending run. If a check fails, fix the campaign — do not paper over it. Default to skeptical: a result that survives this pass is one you can put in the methods doc.

## 6. Record

- Write the campaign result (winner, mean±std, audit verdict) into `CHANGELOG.md`; if a headline number moved, update `docs/METHODS_AND_RESULTS.md`, `docs/RESULTS_TABLE.md`, and the workbook.
- Confirm the active git account is SkyEpstein (remote: github.com/SkyEpstein/Machine-learning-to-separate-f-elements-2), ask for the commit message, commit, and push.

## Templates

Copy these into `results/campaign_<name>/` when you start a campaign:
- `run_manifest.json` — the §3 fields for one run.
- `campaign_spec.md` — the §1 grid, protocol, and metric of record.
- `aggregate_campaign.py` — reads all `run_*.json` manifests in a campaign dir and writes `master.csv` with the mean±std rollup (§4).

Run the aggregator after the grid finishes:
```bash
python3 .claude/skills/experiment-campaign/scripts/aggregate_campaign.py results/campaign_<name>
```

The point of the skill is that a headline number in this project should be a *mean over repeated honest splits, produced by a committed script from checksummed data, that survived a skeptical second read* — and that producing it should be parallel, not an overnight serial loop.
