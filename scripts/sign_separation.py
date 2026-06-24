#!/usr/bin/env python3
"""
sign_separation.py — two checks the project asked for.
(1) Sign correctness: does the predicted logD have the same sign as the actual
    logD (does the model get extract-or-not, logD > 0 vs < 0, right), overall and
    by confidence.
(2) Separation: for two metals at the same conditions, compare predicted logD
    difference against actual difference: how well the magnitude is predicted, how
    often the selectivity direction (which metal is higher) is right, and how often
    the predicted separation magnitude exceeds the actual one.
Reads the deployable prediction files written by deploy_final.py.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from itertools import combinations
from sklearn.metrics import r2_score, mean_squared_error
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
rows = []

print("=== SIGN CORRECTNESS (predicted logD sign vs actual sign) ===")
for f, lab in [("deploy_A_screening_predictions.csv", "Track A (new molecule)"), ("deploy_B_condition_predictions.csv", "Track B (known molecule)")]:
    d = pd.read_csv(f); p = d.Pred_LogD.values; a = d.Actual_LogD.values; err = d.confidence_pred_err.values
    same = np.sign(p) == np.sign(a)
    print(f"\n{lab}: sign matches {same.mean():.3f} of {len(d)} rows  (actual logD>0: {(a>0).mean():.2f}, predicted>0: {(p>0).mean():.2f})")
    for q in [50, 25, 10]:
        m = err <= np.percentile(err, q); print(f"   most confident {q}%: sign matches {same[m].mean():.3f}")
    rows.append({'analysis': 'sign match', 'track': lab, 'all': round(same.mean(), 3),
                 'top25_conf': round(same[err <= np.percentile(err, 25)].mean(), 3),
                 'top10_conf': round(same[err <= np.percentile(err, 10)].mean(), 3)})

print("\n=== SEPARATION (predicted vs actual logD difference between two metals, same conditions) ===")
d = pd.read_csv("deploy_B_condition_predictions.csv")
key = (d.SMILES.astype(str) + '|' + d.Acid_conc_M.round(2).astype(str) + '|' + d.Temperature_K.round(0).astype(str)).values
metal = d.Metal.astype(str).values; P = d.Pred_LogD.values; A = d.Actual_LogD.values
pe, ae = [], []
order = np.argsort(key); ks = key[order]; bounds = np.where(ks[1:] != ks[:-1])[0] + 1
for grp in np.split(order, bounds):
    if len(grp) < 2: continue
    for i, j in combinations(grp, 2):
        if metal[i] == metal[j]: continue
        pe.append(P[i] - P[j]); ae.append(A[i] - A[j])
pe = np.array(pe); ae = np.array(ae)
print(f"matched metal pairs at identical conditions: {len(pe)}")
print(f"  signed separation:     R2={r2_score(ae, pe):.3f}  RMSE={rmse(ae, pe):.3f}")
print(f"  magnitude |pred| vs |actual|: R2={r2_score(np.abs(ae), np.abs(pe)):.3f}  RMSE={rmse(np.abs(ae), np.abs(pe)):.3f}")
print(f"  selectivity direction correct (sign of the difference matches): {(np.sign(pe) == np.sign(ae)).mean():.3f}")
print(f"  predicted separation magnitude exceeds actual (|pred| > |actual|): {(np.abs(pe) > np.abs(ae)).mean():.3f}")
rows.append({'analysis': 'separation', 'track': 'Track B pairs', 'all': f"signed R2 {r2_score(ae,pe):.3f}",
             'top25_conf': f"dir correct {(np.sign(pe)==np.sign(ae)).mean():.3f}", 'top10_conf': f"|pred|>|actual| {(np.abs(pe)>np.abs(ae)).mean():.3f}"})
pd.DataFrame(rows).to_csv("sign_separation_results.csv", index=False)
print("\nsaved sign_separation_results.csv")
