#!/usr/bin/env python3
"""
track_a_separation.py — separation test for Track A (new molecules, the screening
case). For two metals at the same extractant and conditions, compare the predicted
logD difference against the actual one. Direction does not matter here (whichever
metal is left in solution can be taken with a known extractant); what matters is the
separation magnitude. Reports how well |pred diff| tracks |actual diff|, and the
fraction of pairs where |pred diff| <= |actual diff| (the predict-under criterion:
the model's predicted separation is conservative, so the real gap is at least as
large), overall and by confidence.
Uses the Track A deployable predictions (molecule-grouped, new molecule).
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from itertools import combinations
from sklearn.metrics import r2_score, mean_squared_error
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
d = pd.read_csv("deploy_A_screening_predictions.csv")
key = (d.SMILES.astype(str) + '|' + d.Acid_conc_M.round(2).astype(str) + '|' + d.Temperature_K.round(0).astype(str)).values
metal = d.Metal.astype(str).values; P = d.Pred_LogD.values; A = d.Actual_LogD.values; E = d.confidence_pred_err.values
dp, da, pconf = [], [], []
order = np.argsort(key); ks = key[order]; bounds = np.where(ks[1:] != ks[:-1])[0] + 1
for grp in np.split(order, bounds):
    if len(grp) < 2: continue
    for i, j in combinations(grp, 2):
        if metal[i] == metal[j]: continue
        dp.append(P[i] - P[j]); da.append(A[i] - A[j]); pconf.append(max(E[i], E[j]))
dp = np.array(dp); da = np.array(da); pconf = np.array(pconf)
adp, ada = np.abs(dp), np.abs(da)
print(f"=== TRACK A separation (new molecules): {len(dp)} metal pairs at identical conditions ===")
print(f"  magnitude |pred diff| vs |actual diff|: R2={r2_score(ada, adp):.3f}  RMSE={rmse(ada, adp):.3f}")
print(f"  fraction with |pred diff| <= |actual diff| (predict-under criterion):")
rows = []
for q in [100, 50, 25, 10]:
    m = np.ones(len(dp), bool) if q == 100 else (pconf <= np.percentile(pconf, q))
    frac = (adp[m] <= ada[m]).mean()
    print(f"     {'all pairs' if q==100 else f'most confident {q}%':<18s}: {frac:.3f}   (n={int(m.sum())})")
    rows.append({'subset': 'all' if q == 100 else f'top{q}_conf', 'n': int(m.sum()), 'frac_pred_le_actual': round(float(frac), 3)})
print(f"  for context, mean |actual diff| = {ada.mean():.2f}, mean |pred diff| = {adp.mean():.2f} (model under-predicts if pred mean is smaller)")
pd.DataFrame(rows).to_csv("track_a_separation_results.csv", index=False)
print("saved track_a_separation_results.csv")
