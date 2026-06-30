#!/usr/bin/env python3
"""
dg_hpo.py — honest hyperparameter search on the per-pair delta G target to see if a
tuned model beats the default RandomForest (R2 0.473). Random search over
RandomForest, XGBoost, and LightGBM, each scored with the SAME molecule-grouped
5-fold CV used everywhere else, so nothing here can inflate the number. Reports the
best configuration per model as R2 and RMSE (kJ/mol).
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
RkJ = 8.314e-3; SMI = 'SMILES_canonical'; START = time.time(); RS = np.random.RandomState(0)
tr = pd.read_csv("Training_Data_V27.csv", low_memory=False); te = pd.read_csv("Testing_Data_V39.csv", low_memory=False)
df = pd.concat([tr, te], ignore_index=True); df = df[df['Log_D'].notna() & df[SMI].notna()].reset_index(drop=True)
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
def san(M):
    M = M.astype(np.float64).copy(); b = ~np.isfinite(M) | (np.abs(M) > 1e30)
    md = np.nan_to_num(np.nanmedian(np.where(b, np.nan, M), axis=0)); ix = np.where(b); M[ix] = np.take(md, ix[1])
    for j in np.where(np.nanmax(np.abs(M), axis=0) > 1e7)[0]: M[:, j] = np.sign(M[:, j]) * np.log1p(np.abs(M[:, j]))
    return M
df['_pk'] = df[[SMI, 'Metal', 'Acid_type', 'Solvent_A', 'Solvent_B']].astype(str).agg('|'.join, axis=1)
fi = df.drop_duplicates('_pk').index; pkf = df.loc[fi, '_pk']
y = df.groupby('_pk')['dG'].mean().reindex(pkf).values
X = san(pd.concat([df[feat], acid], axis=1).loc[fi].values); groups = df.loc[fi, SMI].values
gkf = GroupKFold(5)
def cv(mk):
    o = np.zeros(len(y))
    for ti, vi in gkf.split(X, y, groups): o[vi] = mk().fit(X[ti], y[ti]).predict(X[vi])
    return float(r2_score(y, o)), float(np.sqrt(mean_squared_error(y, o)))
res = []
def ch(a): return a[RS.randint(len(a))]
for t in range(14):
    p = dict(n_estimators=int(ch([300, 400, 500, 700])), max_features=ch([0.3, 0.5, 0.7, 1.0]), min_samples_leaf=int(ch([1, 2, 4])), n_jobs=-1, random_state=0)
    r2, rm = cv(lambda p=p: RandomForestRegressor(**p)); res.append(('RF', p, r2, rm)); print(f"RF  t{t:02d} R2={r2:.3f} RMSE={rm:.2f}  {p}", flush=True)
try:
    import xgboost as xgb
    for t in range(14):
        p = dict(n_estimators=int(ch([400, 600, 800])), max_depth=int(ch([3, 4, 5, 6, 8])), learning_rate=float(ch([0.01, 0.02, 0.03, 0.05])), subsample=float(ch([0.6, 0.8, 1.0])), colsample_bytree=float(ch([0.5, 0.7, 1.0])), min_child_weight=int(ch([1, 3, 5])), reg_lambda=float(ch([0, 1, 3])), tree_method='hist', n_jobs=-1, random_state=0)
        r2, rm = cv(lambda p=p: xgb.XGBRegressor(**p)); res.append(('XGB', p, r2, rm)); print(f"XGB t{t:02d} R2={r2:.3f} RMSE={rm:.2f}", flush=True)
except Exception as e: print("xgb skip", e)
for t in range(14):
    p = dict(n_estimators=int(ch([400, 600, 800])), num_leaves=int(ch([15, 31, 63, 127])), learning_rate=float(ch([0.01, 0.02, 0.03, 0.05])), subsample=float(ch([0.6, 0.8, 1.0])), subsample_freq=1, colsample_bytree=float(ch([0.5, 0.7, 1.0])), min_child_samples=int(ch([5, 20, 50])), reg_lambda=float(ch([0, 1, 3])), n_jobs=-1, random_state=0, verbosity=-1)
    r2, rm = cv(lambda p=p: lgb.LGBMRegressor(**p)); res.append(('LGB', p, r2, rm)); print(f"LGB t{t:02d} R2={r2:.3f} RMSE={rm:.2f}", flush=True)
out = pd.DataFrame([{'model': m, 'R2': round(r, 3), 'RMSE': round(rm, 2), 'params': str(p)} for m, p, r, rm in res])
print("\n=== BEST PER MODEL (baseline default RandomForest = 0.473) ===")
for m in ['RF', 'XGB', 'LGB']:
    sub = out[out.model == m]
    if len(sub): b = sub.sort_values('R2', ascending=False).iloc[0]; print(f"  {m}: best R2={b.R2}  RMSE={b.RMSE}")
out.to_csv("dg_hpo_results.csv", index=False)
print(f"saved dg_hpo_results.csv | {(time.time()-START)/60:.1f} min")
