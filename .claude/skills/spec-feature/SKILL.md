---
name: spec-feature
description: Plan-implement-verify loop for a new feature in this f-element-extraction ML project. Use when starting any new analysis, model, or substantive change. Creates a dated spec, confirms it with the user, implements task groups, verifies with an adversarial reflection review, updates the changelog, and commits under SkyEpstein after confirming the message.
---

# Spec-driven feature workflow

Follow this loop for any substantive new work. It encodes the project constitution (docs/PROJECT_CONSTITUTION.md): ask before building, honest leakage-free evaluation, report R-squared and RMSE together, choose by bake-off rather than arbitrarily, and audit before publishing.

## 1. Specify
- Create `specs/YYYY-MM-DD-short-name/` by copying the three files in `specs/_template/`.
- Fill `requirements.md` (why, scope, key decisions, context), `plan.md` (numbered task groups), and `validation.md` (success criteria and a merge checklist).
- Ask Skyler to confirm the requirements and the plan, and exactly how he wants it implemented, before writing code. Adjust the spec to his answers.

## 2. Implement
- Work the task groups in `plan.md` in order, updating each task's status as it completes.
- Evaluate honestly: molecule-grouped cross-validation for any new-molecule claim, no replicate or fingerprint leakage, and report R-squared and RMSE together. If choosing a model, feature set, or method, run a bake-off and record it; do not pick arbitrarily.

## 3. Verify (reflection review)
- Run the code and confirm the numbers are reproduced from a committed script and saved to a results file.
- Do an adversarial reflection review (a separate skeptical pass or a verification workflow) that checks for data leakage, overfitting, and overclaiming. Default to skeptical.
- Tick `validation.md`. If a check fails, fix the spec and the work; do not paper over it.

## 4. Record and commit
- Update `CHANGELOG.md`, and if a headline number changed, `METHODS_AND_RESULTS.md` and the workbook.
- Confirm the active gh account is SkyEpstein.
- Ask Skyler what the commit message should say (offer a suggested message plus a free-text option), commit with his wording, and push under SkyEpstein.

The spec is the source of truth: when something changes, update the spec rather than applying a quick fix.
