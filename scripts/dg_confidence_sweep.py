#!/usr/bin/env python3
"""
dg_confidence_sweep.py — two confidence sweeps on the per-pair delta G model.

Part A (how every model does with confidence): for each prediction model, apply the
same confidence filter (a shared LightGBM err model that ranks predictions by their
expected error) and report the top-X% R2, so we see whether confidence filtering
helps the other models as much as it helps RandomForest.

Part B (confidence-model sweep): fix the predictor at RandomForest (the best) and
sweep the way confidence is estimated, an err model that is LightGBM, RandomForest,
or ExtraTrees, plus RandomForest's own tree spread. Each is judged by the top-X% R2
it concentrates and by the Spearman correlation between its signal and the true
absolute error. Molecule-grouped CV; R2 and RMSE in kJ/mol.
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
import lightgbm as lgb
RkJ = 8.314e-3; SMI = 'SMILES_canonical'; START = time.time()
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
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
print(f"per-pair systems {len(y)}, features {X.shape[1]}\n")
def oof_pred(mk):
    o = np.zeros(len(y))
    for ti, vi in gkf.split(X, y, groups): o[vi] = mk().fit(X[ti], y[ti]).predict(X[vi])
    return o
def err_oof(mk, resid):
    e = np.zeros(len(y))
    for ti, vi in gkf.split(X, y, groups): e[vi] = mk().fit(X[ti], resid[ti]).predict(X[vi])
    return e
def curve(pred, sig):
    return [(q, r2_score(y[(np.ones(len(y), bool) if q == 100 else sig <= np.percentile(sig, q))], pred[(np.ones(len(y), bool) if q == 100 else sig <= np.percentile(sig, q))]),
             rmse(y[(np.ones(len(y), bool) if q == 100 else sig <= np.percentile(sig, q))], pred[(np.ones(len(y), bool) if q == 100 else sig <= np.percentile(sig, q))])) for q in [100, 50, 25, 10]]
LGBE = lambda: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, random_state=0, n_jobs=-1, verbosity=-1)
PRED = {'RandomForest': lambda: RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=0),
        'ExtraTrees': lambda: ExtraTreesRegressor(n_estimators=400, n_jobs=-1, random_state=0),
        'HistGB': lambda: HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=0),
        'LightGBM': lambda: lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63, subsample=0.8, colsample_bytree=0.7, subsample_freq=1, random_state=0, n_jobs=-1, verbosity=-1)}
try:
    import xgboost as xgb; PRED['XGBoost'] = lambda: xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.7, tree_method='hist', random_state=0, n_jobs=-1)
except Exception: pass
try:
    import catboost as cb; PRED['CatBoost'] = lambda: cb.CatBoostRegressor(n_estimators=600, learning_rate=0.03, depth=6, random_state=0, verbose=0)
except Exception: pass
rows = []; oofs = {}
print("PART A: confidence curve per prediction model (shared LightGBM err ranker)")
print(f"  {'model':<14}{'all R2':>8}{'top50':>8}{'top25':>8}{'top10':>8}{'RMSE top10':>12}")
for name, mk in PRED.items():
    o = oof_pred(mk); oofs[name] = o; c = curve(o, err_oof(LGBE, np.abs(y - o)))
    print(f"  {name:<14}{c[0][1]:>8.3f}{c[1][1]:>8.3f}{c[2][1]:>8.3f}{c[3][1]:>8.3f}{c[3][2]:>12.2f}")
    rows.append({'part': 'A pred-model', 'name': name, 'all_R2': round(c[0][1], 3), 'top25_R2': round(c[2][1], 3), 'top10_R2': round(c[3][1], 3), 'top10_RMSE': round(c[3][2], 2)})
print("\nPART B: confidence-model sweep on RandomForest predictions")
rf = oofs['RandomForest']; resid = np.abs(y - rf)
tstd = np.zeros(len(y))
for ti, vi in gkf.split(X, y, groups):
    m = RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=0).fit(X[ti], y[ti]); tstd[vi] = np.stack([t.predict(X[vi]) for t in m.estimators_]).std(0)
RANK = {'err-LightGBM': err_oof(LGBE, resid),
        'err-RandomForest': err_oof(lambda: RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0), resid),
        'err-ExtraTrees': err_oof(lambda: ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=0), resid),
        'RF tree-spread (native)': tstd}
print(f"  {'confidence ranker':<24}{'top25 R2':>10}{'top10 R2':>10}{'Spearman':>10}")
for name, sig in RANK.items():
    c = curve(rf, sig); sp = float(spearmanr(sig, resid).correlation)
    print(f"  {name:<24}{c[2][1]:>10.3f}{c[3][1]:>10.3f}{sp:>10.3f}")
    rows.append({'part': 'B conf-ranker (RF)', 'name': name, 'all_R2': round(r2_score(y, rf), 3), 'top25_R2': round(c[2][1], 3), 'top10_R2': round(c[3][1], 3), 'top10_RMSE': round(c[3][2], 2)})
pd.DataFrame(rows).to_csv("dg_confidence_sweep_results.csv", index=False)
print(f"\nsaved dg_confidence_sweep_results.csv | {(time.time()-START)/60:.1f} min")
