#!/usr/bin/env python3
"""
dg_ucb.py — UCB active analysis on the delta G (free energy) model, since delta G is
the predictor meant to drive the screening. The per-pair RandomForest is run with
molecule-grouped cross-validation to get honest out-of-fold predictions and a
learned uncertainty (the err model), and a normalized 90 percent interval. A
favorable extraction is the most negative delta G, so the optimistic bound is
prediction minus uncertainty.

Two parts:
  UCB selection : to find the strongest extractants (most negative delta G), rank by
    the optimistic bound (UCB), by the plain prediction (greedy), and at random;
    report the mean actual delta G of the picked set and the recall of the true best.
  Triage        : auto-accept the most confident predictions (no experiment) and see
    how accurate the accepted set is, so the experiment budget goes to the uncertain.
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
RkJ = 8.314e-3; SMI = 'SMILES_canonical'; rng = np.random.RandomState(0)
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
tr = pd.read_csv("Training_Data_V27.csv", low_memory=False); te = pd.read_csv("Testing_Data_V39.csv", low_memory=False)
df = pd.concat([tr, te], ignore_index=True)
df = df[df['Log_D'].notna() & df[SMI].notna()].reset_index(drop=True)
kdf = df[[SMI, 'Metal', 'Acid_type']].astype(str).copy()
for c, r in [('Acid_conc_M', 2), ('Temperature_K', 0), ('Extractant_conc_M', 3), ('Metal_conc_mM', 4)]:
    if c in df: kdf[c] = df[c].round(r).astype(str)
grng = pd.Series(df['Log_D'].values).groupby(kdf.agg('|'.join, axis=1)).transform(lambda v: v.max() - v.min()).values
df = df[(~df.duplicated().values) & (grng <= 2.0)].reset_index(drop=True)
df['dG'] = -2.302585 * RkJ * df['Temperature_K'].astype(float) * df['Log_D'].astype(float)
DROP = ['Extractant_conc_M', 'Temperature_K', 'Acid_conc_M', 'Metal_conc_mM', 'Volume_fraction_A', 'Volume_fraction_B']
num = df.select_dtypes(np.number).columns.tolist()
feat = [c for c in num if c not in DROP + ['Log_D', 'dG'] and not c.startswith('embedding_')]
acid = pd.get_dummies(df['Acid_type'].astype(str), prefix='acid').astype(float)
Xfull = pd.concat([df[feat], acid], axis=1)
def san(M):
    M = M.astype(np.float64).copy(); b = ~np.isfinite(M) | (np.abs(M) > 1e30)
    md = np.nan_to_num(np.nanmedian(np.where(b, np.nan, M), axis=0)); ix = np.where(b); M[ix] = np.take(md, ix[1])
    for j in np.where(np.nanmax(np.abs(M), axis=0) > 1e7)[0]: M[:, j] = np.sign(M[:, j]) * np.log1p(np.abs(M[:, j]))
    return M
df['_pk'] = df[[SMI, 'Metal', 'Acid_type', 'Solvent_A', 'Solvent_B']].astype(str).agg('|'.join, axis=1)
fi = df.drop_duplicates('_pk').index; pkf = df.loc[fi, '_pk']
y = df.groupby('_pk')['dG'].mean().reindex(pkf).values
X = san(Xfull.loc[fi].values); groups = df.loc[fi, SMI].values
gkf = GroupKFold(5); pred = np.zeros(len(y)); err = np.zeros(len(y))
for ti, vi in gkf.split(X, y, groups):
    m = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=0).fit(X[ti], y[ti]); pred[vi] = m.predict(X[vi])
resid = np.abs(y - pred)
for ti, vi in gkf.split(X, y, groups):
    em = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, random_state=0, n_jobs=-1, verbosity=-1).fit(X[ti], resid[ti]); err[vi] = em.predict(X[vi])
err = np.clip(err, 1e-6, None)
q = np.quantile(resid / err, 0.90); lo = pred - q * err  # optimistic (most favorable) 90% bound
print(f"delta G model (per-pair RandomForest): n={len(y)}, R2={r2_score(y,pred):.3f}, RMSE={rmse(y,pred):.2f} kJ/mol, 90% interval half-width q={q:.2f}")
print("\n=== UCB selection: find the strongest extractants (most negative delta G) ===")
rows = []
for frac in [0.05, 0.10]:
    k = int(len(y) * frac); truebest = set(np.argsort(y)[:k].tolist())  # lowest (most favorable) actual dG
    for name, score in [("UCB (pred - unc)", lo), ("greedy (pred)", pred), ("random", rng.rand(len(y)))]:
        sel = np.argsort(score)[:k]  # most favorable = smallest score
        ma = float(y[sel].mean()); rec = len(set(sel.tolist()) & truebest) / k
        print(f"  top {int(frac*100):>2d}% by {name:<16s}: mean actual dG={ma:6.2f} kJ/mol, recall of true-best={rec:.2f}")
        rows.append({'analysis': 'UCB selection', 'select_top_pct': int(frac * 100), 'method': name, 'mean_actual_dG': round(ma, 2), 'recall_true_best': round(rec, 2)})
print("\n=== Triage: auto-accept the most confident delta G predictions (no experiment) ===")
for acc in [0.25, 0.50]:
    mk = err <= np.quantile(err, acc); within = float((np.abs(pred - y)[mk] <= 3.0).mean())
    print(f"  accept top {int(acc*100)}% conf: {len(y)-int(mk.sum())} of {len(y)} still need testing, accepted RMSE={rmse(y[mk],pred[mk]):.2f} kJ/mol, within 3 kJ/mol={within:.0%}")
    rows.append({'analysis': 'triage', 'select_top_pct': int(acc * 100), 'method': 'accept confident', 'mean_actual_dG': round(rmse(y[mk], pred[mk]), 2), 'recall_true_best': round(within, 2)})
pd.DataFrame(rows).to_csv("dg_ucb_results.csv", index=False)
print("\nsaved dg_ucb_results.csv")
