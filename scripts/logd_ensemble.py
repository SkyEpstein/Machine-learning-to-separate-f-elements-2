#!/usr/bin/env python3
"""
logd_ensemble.py — ensemble sweep for the two logD tracks, now with RandomForest in
the field, to test whether the ECM finding (bagged trees beat the boosters) carries
over to the larger logD data or whether more rows favor the boosters. Track A is the
new-molecule screen (conditions and metal only, molecule-grouped CV). Track B is the
known-molecule condition model (conditions, ECFP, and ligand descriptors, random-row
CV). Each base model is scored by R2 and RMSE, then combined equal-weight and by an
NNLS stack scored with its own cross-validation so the stack number is honest.
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.optimize import nnls
import lightgbm as lgb
START = time.time(); SMI = 'SMILES_canonical'
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
tr = pd.read_csv("Training_Data_V27.csv", low_memory=False); te = pd.read_csv("Testing_Data_V39.csv", low_memory=False)
df = pd.concat([tr, te], ignore_index=True)
df = df[df['Log_D'].notna() & df[SMI].notna()].reset_index(drop=True)
kdf = df[[SMI, 'Metal', 'Acid_type']].astype(str).copy()
for c, r in [('Acid_conc_M', 2), ('Temperature_K', 0), ('Extractant_conc_M', 3), ('Metal_conc_mM', 4)]:
    if c in df: kdf[c] = df[c].round(r).astype(str)
grng = pd.Series(df['Log_D'].values).groupby(kdf.agg('|'.join, axis=1)).transform(lambda v: v.max() - v.min()).values
df = df[(~df.duplicated().values) & (grng <= 2.0)].reset_index(drop=True)
y = df['Log_D'].values.astype(float); groups = df[SMI].values
COND = ['Extractant_conc_M', 'Molar_mass(g/mol) A', 'Log_P A', 'Boiling_point(K) A', 'Melting_point(K) A', 'Density(g/mL) A', 'Solubility_in_water(g/L) A', 'Molar_mass(g/mol) B', 'Log_P B', 'Boiling_point(K) B', 'Melting_point(K) B', 'Density(g/mL) B', 'Solubility_in_water(g/L) B', 'Volume_fraction_A', 'Volume_fraction_B', 'Atomic_number', 'Melting_point_K', 'Boiling_point_K', 'Density_g/cm3', 'First_IE_kJ/mol', 'Second_IE_kJ/mol', 'Third_IE_kJ/mol', 'Matallic_radius_nm', 'Pauling_EN', 'Ionic_radius_nm', 'Oxidation_state', 'Metal_conc_mM', 'Dipole_moment_D', 'Acid_conc_M', 'Temperature_K']
LIG = [c for c in ['MolLogP', 'NumHDonors', 'NumHAcceptors', 'NOCount', 'NHOHCount', 'TPSA', 'NumRotatableBonds', 'RingCount', 'FractionCSP3', 'NumAromaticRings'] if c in df]
fp = [c for c in df.columns if c.startswith('fp_')]
acid = pd.get_dummies(df['Acid_type'].astype(str), prefix='acid').astype(float)
def san(M):
    M = M.astype(np.float64).copy(); b = ~np.isfinite(M) | (np.abs(M) > 1e30)
    md = np.nan_to_num(np.nanmedian(np.where(b, np.nan, M), axis=0)); ix = np.where(b); M[ix] = np.take(md, ix[1])
    for j in np.where(np.nanmax(np.abs(M), axis=0) > 1e7)[0]: M[:, j] = np.sign(M[:, j]) * np.log1p(np.abs(M[:, j]))
    return M
def models():
    m = {'LightGBM': lgb.LGBMRegressor(n_estimators=700, learning_rate=0.03, num_leaves=63, subsample=0.8, colsample_bytree=0.7, subsample_freq=1, random_state=0, n_jobs=-1, verbosity=-1),
         'ExtraTrees': ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=0),
         'RandomForest': RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0),
         'HistGB': HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=0),
         'Ridge': make_pipeline(StandardScaler(), Ridge(alpha=10.0))}
    try:
        import xgboost as xgb; m['XGBoost'] = xgb.XGBRegressor(n_estimators=700, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.7, tree_method='hist', random_state=0, n_jobs=-1)
    except Exception: pass
    try:
        import catboost as cb; m['CatBoost'] = cb.CatBoostRegressor(n_estimators=600, learning_rate=0.03, depth=6, random_state=0, verbose=0)
    except Exception: pass
    return m
allrows = []
def sweep(X, splits, label):
    names = list(models().keys()); oof = {n: np.zeros(len(y)) for n in names}
    for ti, vi in splits:
        for n, model in models().items():
            model.fit(X[ti], y[ti]); oof[n][vi] = model.predict(X[vi])
    print(f"\n=== {label} ===")
    for n in names:
        R2 = r2_score(y, oof[n]); RM = rmse(y, oof[n]); allrows.append({'track': label, 'model': n, 'R2': round(R2, 3), 'RMSE': round(RM, 3)})
        print(f"  {n:<14} R2={R2:.3f}  RMSE={RM:.3f}")
    P = np.column_stack([oof[n] for n in names]); eq = P.mean(1)
    w, _ = nnls(P, y); stack_cv = np.zeros(len(y))
    sp = list(splits)
    for ti, vi in (GroupKFold(5).split(P, y, groups) if 'A' in label else KFold(5, shuffle=True, random_state=1).split(P)):
        wf, _ = nnls(P[ti], y[ti]); stack_cv[vi] = P[vi] @ wf
    best = max(names, key=lambda n: r2_score(y, oof[n]))
    print(f"  equal-weight   R2={r2_score(y,eq):.3f}  RMSE={rmse(y,eq):.3f}")
    print(f"  NNLS stack(CV) R2={r2_score(y,stack_cv):.3f}  RMSE={rmse(y,stack_cv):.3f}")
    print(f"  best single = {best} (R2={r2_score(y,oof[best]):.3f}); stack-minus-best = {r2_score(y,stack_cv)-r2_score(y,oof[best]):+.3f}")
    print(f"  NNLS weights: {{{', '.join(f'{n}:{wi:.2f}' for n,wi in zip(names,w) if wi>1e-4)}}}")
    allrows.append({'track': label, 'model': 'equal-weight', 'R2': round(r2_score(y, eq), 3), 'RMSE': round(rmse(y, eq), 3)})
    allrows.append({'track': label, 'model': 'NNLS stack (CV)', 'R2': round(r2_score(y, stack_cv), 3), 'RMSE': round(rmse(y, stack_cv), 3)})
XA = san(pd.concat([df[COND], acid], axis=1).values)
sweep(XA, list(GroupKFold(5).split(XA, y, groups)), "Track A (new molecule)")
XB = san(pd.concat([df[COND + LIG], acid, df[fp]], axis=1).values)
sweep(XB, list(KFold(5, shuffle=True, random_state=0).split(XB)), "Track B (known molecule)")
pd.DataFrame(allrows).to_csv("logd_ensemble_results.csv", index=False)
print(f"\nsaved logd_ensemble_results.csv | {(time.time()-START)/60:.1f} min")
