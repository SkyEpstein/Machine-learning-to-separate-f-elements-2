#!/usr/bin/env python3
"""
make_master_table.py — one consolidated, manuscript-ready scoreboard of every
headline result, each row carrying the metric, RMSE where applicable, n, the exact
evaluation regime, a one-line caveat, and the source result file. All numbers are the
audited honest values. Writes results/master_results_table.csv and a markdown table
docs/RESULTS_TABLE.md.
"""
import pandas as pd
COLS = ["Quantity", "Model", "Metric", "RMSE", "n", "Evaluation", "Caveat", "Source"]
R = [
    ["Track A logD (new molecule)", "LightGBM", "R2 0.466", "1.148", "7066", "molecule-grouped CV", "Honest new-extractant screening", "track_ab_results.csv"],
    ["Track B logD (condition interpolation)", "NNLS stack", "R2 0.61", "0.98", "7066", "condition-key grouped CV", "Honest known-molecule number", "track_ab_results.csv"],
    ["Track B logD (random-row)", "NNLS stack", "R2 0.725", "0.823", "7066", "random-row CV", "UPPER BOUND; ~0.07 is replicate memorization", "track_ab_results.csv"],
    ["Track B logD (new molecule)", "LightGBM", "R2 0.44", "1.17", "7066", "molecule-grouped CV", "Rich features collapse to ~Track A on new molecules", "track_ab_results.csv"],
    ["Track B confidence, top 10%", "NNLS stack + err", "R2 0.874", "0.430", "707", "condition-key, top 10% by confidence", "Selective-prediction operating point, not overall accuracy", "track_b_confidence_honest.csv"],
    ["Free energy delta G, per-pair", "RandomForest", "R2 0.46 +/- 0.01", "6.31 kJ/mol", "2273", "molecule-grouped CV", "0.473 was a selected maximum of a ~55-way search", "dg_results.csv"],
    ["Free energy delta G, top 10%", "RandomForest + err", "R2 0.776", "3.62 kJ/mol", "228", "molecule-grouped, top 10% by confidence", "Selective-prediction operating point", "dg_results.csv"],
    ["delta G 90% interval coverage", "split conformal", "0.902 (target 0.90)", "width 20.9 kJ/mol", "2273", "molecule-disjoint calibration and eval", "Validates the procedure; width is wide", "dg_coverage_results.csv"],
    ["Classifier, 3-class (new molecule)", "LightGBM", "acc 0.625", "", "7066", "molecule-grouped CV", "vs 0.385 majority baseline; top 10% conf 0.912", "classifier_confidence_results.csv"],
    ["Classifier, binary logD>0 (new molecule)", "LightGBM", "acc 0.742 (AUC 0.817)", "", "7066", "molecule-grouped CV", "vs 0.53 base rate; top 10% conf 0.962", "classifier_confidence_results.csv"],
    ["Sign correctness, Track A", "LightGBM", "0.743 (0.903 top 10%)", "", "7066", "molecule-grouped CV", "Predicted logD sign matches actual (extract-or-not)", "sign_separation_results.csv"],
    ["Separation, Track B (differencing)", "logD model differenced", "signed R2 0.582", "", "", "matched conditions", "Direction correct 76.3%; predict-under 63.3%", "sign_separation_results.csv"],
    ["Separation, direct delta-logD model", "LightGBM (pairs)", "signed R2 -1.92", "2.76", "4598", "molecule-grouped CV", "Loses to differencing (0.188); too few extractants with pairs", "sep_factor_results.csv"],
    ["Separation factor, f-elements (known extractant)", "logD diff, cond-key", "signed R2 0.356", "1.294", "4598", "condition-key CV, matched conditions", "Direction 72.6%, Spearman 0.60, magnitude R2 0.166; deployed stack with fingerprints reaches ~0.58", "sep_factor_eval_results.csv"],
    ["Separation factor, f-elements (new extractant)", "logD diff, mol-grouped", "signed R2 0.188", "1.453", "4598", "molecule-grouped CV", "Direction 65.6%, Spearman 0.46, magnitude R2 -0.199; order is predicted, size is not", "sep_factor_eval_results.csv"],
    ["Comparison with Dr. Zhang (same data)", "ours vs his XGBoost", "ours 0.657 CV / 0.692 holdout", "", "", "common split + his 494-row holdout", "His 0.648 / 0.680; about equal, the split drives his 0.72 headline", "zhang_2x2_results.csv"],
]
df = pd.DataFrame(R, columns=COLS)
df.to_csv("master_results_table.csv", index=False)
md = ["# Master results table", "",
      "Every headline result with its honest number, evaluation regime, caveat, and source file. All evaluations are leakage-free (the regime is stated); numbers are on the label-QC-cleaned data (an upper bound), and the label-noise floor is about 0.45 log units / 6 kJ/mol.", "",
      "| " + " | ".join(COLS) + " |", "|" + "|".join(["---"] * len(COLS)) + "|"]
for _, r in df.iterrows():
    md.append("| " + " | ".join(str(x) for x in r) + " |")
open("RESULTS_TABLE.md", "w").write("\n".join(md) + "\n")
print(f"wrote master_results_table.csv ({len(df)} rows) and RESULTS_TABLE.md")
print(df[["Quantity", "Metric", "Evaluation"]].to_string(index=False))
