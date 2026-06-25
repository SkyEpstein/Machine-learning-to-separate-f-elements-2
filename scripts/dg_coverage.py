#!/usr/bin/env python3
"""
dg_coverage.py — are the delta G prediction intervals calibrated? That is what makes
active analysis trustworthy: a 90 percent interval must actually contain the truth
about 90 percent of the time. Uses the per-pair RandomForest with the LightGBM err
model, normalized split conformal. Predictions and per-point error are computed with
molecule-grouped CV (honest), then the conformal quantile is calibrated on one set of
molecules and the coverage is measured on a disjoint set of molecules, averaged over
many such splits. Reports empirical coverage at the 90 and 80 percent targets and the
mean interval width in kJ/mol.
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
RkJ = 8.314e-3; SMI = 'SMILES_canonical'; START = time.time(); rng = np.random.RandomState(0)
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
gkf = GroupKFold(5); pred = np.zeros(len(y)); err = np.zeros(len(y))
for ti, vi in gkf.split(X, y, groups):
    pred[vi] = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=0).fit(X[ti], y[ti]).predict(X[vi])
resid = np.abs(y - pred)
for ti, vi in gkf.split(X, y, groups):
    err[vi] = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, random_state=0, n_jobs=-1, verbosity=-1).fit(X[ti], resid[ti]).predict(X[vi])
err = np.clip(err, 1e-6, None); s = resid / err
um = np.array(sorted(set(groups)))
cov90, cov80, w90 = [], [], []
for rep in range(30):
    rng.shuffle(um); cal_m = set(um[:len(um) // 2].tolist())
    cal = np.array([g in cal_m for g in groups]); ev = ~cal
    q90 = np.quantile(s[cal], 0.90); q80 = np.quantile(s[cal], 0.80)
    cov90.append(float((resid[ev] <= q90 * err[ev]).mean())); cov80.append(float((resid[ev] <= q80 * err[ev]).mean()))
    w90.append(float((2 * q90 * err[ev]).mean()))
print(f"per-pair delta G, n={len(y)}, model R2={1 - (resid**2).sum()/((y-y.mean())**2).sum():.3f}")
print(f"  90% interval: empirical coverage = {np.mean(cov90):.3f} (target 0.90), mean width = {np.mean(w90):.1f} kJ/mol")
print(f"  80% interval: empirical coverage = {np.mean(cov80):.3f} (target 0.80)")
pd.DataFrame([{'target': 0.90, 'empirical_coverage': round(float(np.mean(cov90)), 3), 'mean_width_kJmol': round(float(np.mean(w90)), 1)},
             {'target': 0.80, 'empirical_coverage': round(float(np.mean(cov80)), 3), 'mean_width_kJmol': ''}]).to_csv("dg_coverage_results.csv", index=False)
print(f"saved dg_coverage_results.csv | {(time.time()-START)/60:.1f} min")
