#!/usr/bin/env python3
"""
ucb_analysis.py — use the confidence interval for active analysis (UCB). To decide
which untested extractant-and-condition combinations are worth running, rank them
by an upper confidence bound: the top of the 90 percent conformal interval (hi90),
which equals the prediction plus its uncertainty. This is optimism under
uncertainty, and it matters here because the model under-predicts the extremes, so
a purely greedy ranking by the point prediction overlooks high-uncertainty
candidates that may actually be excellent.

For each track and each selection size we compare three ways to pick the top
candidates (UCB by hi90, greedy by the point prediction, and random) on two things:
the mean actual logD of the picked set (how good they really are) and the recall of
the true best set (how much of the genuinely top group the selection captures).
Uses the deployable prediction files.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
rng = np.random.RandomState(0)
rows = []
print("=== UCB / active analysis (select promising high-logD combos with the confidence interval) ===")
print("UCB = upper end of the 90% conformal interval (hi90) = prediction + uncertainty.\n")
for f, lab in [("deploy_A_screening_predictions.csv", "Track A (new molecule)"), ("deploy_B_condition_predictions.csv", "Track B (known molecule)")]:
    d = pd.read_csv(f); a = d.Actual_LogD.values; pred = d.Pred_LogD.values; ucb = d.hi90.values; n = len(d)
    print(f"{lab}: n={n}, mean actual logD over all = {a.mean():.2f}")
    for frac in [0.05, 0.10]:
        k = int(n * frac); truebest = set(np.argsort(-a)[:k].tolist())
        for name, score in [("UCB (hi90)", ucb), ("greedy (prediction)", pred), ("random", rng.rand(n))]:
            sel = np.argsort(-score)[:k]
            ma = float(a[sel].mean()); rec = len(set(sel.tolist()) & truebest) / k
            print(f"   top {int(frac*100):>2d}% by {name:<20s}: mean actual logD = {ma:5.2f},  recall of true-best = {rec:.2f}")
            rows.append({'track': lab, 'select_top_pct': int(frac * 100), 'method': name, 'mean_actual_logD': round(ma, 2), 'recall_true_best': round(rec, 2)})
        print()
pd.DataFrame(rows).to_csv("ucb_analysis_results.csv", index=False)
print("saved ucb_analysis_results.csv")
