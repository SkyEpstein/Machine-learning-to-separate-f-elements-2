#!/usr/bin/env python3
"""
experiment_triage.py — the practical use of confidence: you do not need to run
experiments on the high-confidence predictions, you trust them; experiments go to
the uncertain ones. This quantifies the trade: if you auto-accept the most confident
fraction of predictions (no experiment) and only test the rest, how accurate is the
accepted set and how many experiments are saved. Accuracy is shown as RMSE and as
the share of accepted predictions within 0.5 log units of the true value (a
practical "good enough to trust" tolerance). Uses the deployable prediction files.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
rows = []
for f, lab in [("deploy_A_screening_predictions.csv", "Track A (new molecule)"), ("deploy_B_condition_predictions.csv", "Track B (known molecule)")]:
    d = pd.read_csv(f); p = d.Pred_LogD.values; a = d.Actual_LogD.values; err = d.confidence_pred_err.values; n = len(d)
    res = np.abs(p - a)
    print(f"\n=== {lab}: auto-accept the most confident predictions (no experiment), test the rest ===")
    print(f"  {'auto-accept':>11} {'experiments still needed':>24} {'accepted RMSE':>14} {'accepted within 0.5':>20}")
    for acc in [0.25, 0.50, 0.75]:
        m = err <= np.percentile(err, acc * 100)
        within = float((res[m] <= 0.5).mean())
        print(f"  {f'top {int(acc*100)}%':>11} {f'{n-int(m.sum())} of {n}':>24} {rmse(a[m], p[m]):>14.3f} {within:>19.0%}")
        rows.append({'track': lab, 'auto_accept_pct': int(acc * 100), 'experiments_saved': int(m.sum()),
                     'experiments_needed': n - int(m.sum()), 'accepted_RMSE': round(rmse(a[m], p[m]), 3), 'accepted_within_0p5': round(within, 3)})
pd.DataFrame(rows).to_csv("experiment_triage_results.csv", index=False)
print("\nThe accepted (confident) predictions are trusted as-is; the experiment budget goes to the uncertain remainder, where uncertainty sampling improves the model fastest.")
print("saved experiment_triage_results.csv")
