#!/usr/bin/env python3
"""
dg_ensemble.py — ensemble sweep for the per-pair delta G (free energy) target. Each
diverse base model is evaluated with molecule-grouped cross-validation, then their
out-of-fold predictions are combined two ways: equal weight, and non-negative least
squares (NNLS), which self-prunes weak members. The NNLS stack is also scored with
a second cross-validation over the out-of-fold matrix, so its number is honest and
not just the weights fit and tested on the same rows. Everything is reported as R2
and RMSE (kJ/mol) to find the best model for the free energy of extraction of a new
extractant.
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.optimize import nnls
import lightgbm as lgb
RkJ = 8.314e-3; START = time.time(); SMI = 'SMILES_canonical'
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
acid = pd.get_dummies(df['Acid_type'].astype(str), prefix='acid')
Xfull = pd.concat([df[feat], acid.astype(float)], axis=1)
def san(M):
    M = M.astype(np.float64).copy(); b = ~np.isfinite(M) | (np.abs(M) > 1e30)
    md = np.nan_to_num(np.nanmedian(np.where(b, np.nan, M), axis=0)); ix = np.where(b); M[ix] = np.take(md, ix[1])
    for j in np.where(np.nanmax(np.abs(M), axis=0) > 1e7)[0]: M[:, j] = np.sign(M[:, j]) * np.log1p(np.abs(M[:, j]))
    return M
df['_pk'] = df[[SMI, 'Metal', 'Acid_type', 'Solvent_A', 'Solvent_B']].astype(str).agg('|'.join, axis=1)
fi = df.drop_duplicates('_pk').index; pkf = df.loc[fi, '_pk']
y = df.groupby('_pk')['dG'].mean().reindex(pkf).values
X = san(Xfull.loc[fi].values); groups = df.loc[fi, SMI].values
print(f"per-pair systems {len(y)}, features {X.shape[1]}")
def models():
    m = {'LightGBM': lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=63, subsample=0.8, colsample_bytree=0.7, subsample_freq=1, random_state=0, n_jobs=-1, verbosity=-1),
         'ExtraTrees': ExtraTreesRegressor(n_estimators=500, n_jobs=-1, random_state=0),
         'RandomForest': RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=0),
         'HistGB': HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=0),
         'Ridge': make_pipeline(StandardScaler(), Ridge(alpha=10.0))}
    try:
        import xgboost as xgb; m['XGBoost'] = xgb.XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.7, tree_method='hist', random_state=0, n_jobs=-1)
    except Exception as e: print('  (no xgboost:', e, ')')
    try:
        import catboost as cb; m['CatBoost'] = cb.CatBoostRegressor(n_estimators=800, learning_rate=0.03, depth=6, random_state=0, verbose=0)
    except Exception as e: print('  (no catboost:', e, ')')
    return m
names = list(models().keys()); gkf = GroupKFold(5)
oof = {n: np.zeros(len(y)) for n in names}
for ti, vi in gkf.split(X, y, groups):
    for n, model in models().items():
        model.fit(X[ti], y[ti]); oof[n][vi] = model.predict(X[vi])
print("\nBASE MODELS (per-pair dG, molecule-grouped CV):")
rows = []
for n in names:
    R2 = r2_score(y, oof[n]); RM = rmse(y, oof[n]); rows.append({'model': n, 'R2': round(R2, 3), 'RMSE_kJmol': round(RM, 2)})
    print(f"  {n:<14} R2={R2:.3f}  RMSE={RM:.2f}")
P = np.column_stack([oof[n] for n in names])
eq = P.mean(1)
w, _ = nnls(P, y); pred_in = P @ w
stack_cv = np.zeros(len(y))
for ti, vi in gkf.split(P, y, groups):
    wf, _ = nnls(P[ti], y[ti]); stack_cv[vi] = P[vi] @ wf
print(f"\n  equal-weight          R2={r2_score(y,eq):.3f}  RMSE={rmse(y,eq):.2f}")
print(f"  NNLS stack (in-sample) R2={r2_score(y,pred_in):.3f}  RMSE={rmse(y,pred_in):.2f}")
print(f"  NNLS stack (CV, honest) R2={r2_score(y,stack_cv):.3f}  RMSE={rmse(y,stack_cv):.2f}")
print("  NNLS weights:", {n: round(float(wi), 3) for n, wi in zip(names, w) if wi > 1e-4})
best_single = max(names, key=lambda n: r2_score(y, oof[n]))
print(f"  best single = {best_single} (R2={r2_score(y,oof[best_single]):.3f}); stack-minus-best = {r2_score(y,stack_cv)-r2_score(y,oof[best_single]):+.3f}")
rows += [{'model': 'equal-weight', 'R2': round(r2_score(y, eq), 3), 'RMSE_kJmol': round(rmse(y, eq), 2)},
         {'model': 'NNLS stack (in-sample)', 'R2': round(r2_score(y, pred_in), 3), 'RMSE_kJmol': round(rmse(y, pred_in), 2)},
         {'model': 'NNLS stack (CV)', 'R2': round(r2_score(y, stack_cv), 3), 'RMSE_kJmol': round(rmse(y, stack_cv), 2)}]
pd.DataFrame(rows).to_csv("dg_ensemble_results.csv", index=False)
pd.DataFrame([{'model': n, 'nnls_weight': round(float(wi), 4)} for n, wi in zip(names, w)]).to_csv("dg_ensemble_weights.csv", index=False)
print(f"\nsaved dg_ensemble_results.csv, dg_ensemble_weights.csv | {(time.time()-START)/60:.1f} min")
