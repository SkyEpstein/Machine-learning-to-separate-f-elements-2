#!/usr/bin/env python3
"""
track_b_confidence_honest.py — recompute the Track B confidence curve on a
condition-key-grouped split, the honest replacement for the leaky random-row 0.940.
The group key is molecule|metal|acid|rounded conditions, so replicate siblings of the
same system cannot land in both folds. Single LightGBM (the dominant stack member),
err model ranks confidence. Reports all / top-50 / top-25 / top-10 percent R2 and
RMSE for logD, against the old random-row figures.
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
SMI = 'SMILES_canonical'; START = time.time()
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
tr = pd.read_csv("Training_Data_V27.csv", low_memory=False); te = pd.read_csv("Testing_Data_V39.csv", low_memory=False)
df = pd.concat([tr, te], ignore_index=True); df = df[df['Log_D'].notna() & df[SMI].notna()].reset_index(drop=True)
ck = df[[SMI, 'Metal', 'Acid_type']].astype(str).copy()
for c, r in [('Acid_conc_M', 2), ('Temperature_K', 0), ('Extractant_conc_M', 3), ('Metal_conc_mM', 4)]:
    if c in df: ck[c] = df[c].round(r).astype(str)
condkey = ck.agg('|'.join, axis=1)
grng = pd.Series(df['Log_D'].values).groupby(condkey).transform(lambda v: v.max() - v.min()).values
keep = (~df.duplicated().values) & (grng <= 2.0)
df = df[keep].reset_index(drop=True); condkey = condkey[keep].reset_index(drop=True).values
y = df['Log_D'].values.astype(float)
COND = ['Extractant_conc_M', 'Molar_mass(g/mol) A', 'Log_P A', 'Boiling_point(K) A', 'Melting_point(K) A', 'Density(g/mL) A', 'Solubility_in_water(g/L) A', 'Molar_mass(g/mol) B', 'Log_P B', 'Boiling_point(K) B', 'Melting_point(K) B', 'Density(g/mL) B', 'Solubility_in_water(g/L) B', 'Volume_fraction_A', 'Volume_fraction_B', 'Atomic_number', 'Melting_point_K', 'Boiling_point_K', 'Density_g/cm3', 'First_IE_kJ/mol', 'Second_IE_kJ/mol', 'Third_IE_kJ/mol', 'Matallic_radius_nm', 'Pauling_EN', 'Ionic_radius_nm', 'Oxidation_state', 'Metal_conc_mM', 'Dipole_moment_D', 'Acid_conc_M', 'Temperature_K']
LIG = [c for c in ['MolLogP', 'NumHDonors', 'NumHAcceptors', 'NOCount', 'NHOHCount', 'TPSA', 'NumRotatableBonds', 'RingCount', 'FractionCSP3', 'NumAromaticRings'] if c in df]
fp = [c for c in df.columns if c.startswith('fp_')]
acid = pd.get_dummies(df['Acid_type'].astype(str), prefix='acid').astype(float)
def san(M):
    M = M.astype(np.float64).copy(); b = ~np.isfinite(M) | (np.abs(M) > 1e30)
    md = np.nan_to_num(np.nanmedian(np.where(b, np.nan, M), axis=0)); ix = np.where(b); M[ix] = np.take(md, ix[1])
    for j in np.where(np.nanmax(np.abs(M), axis=0) > 1e7)[0]: M[:, j] = np.sign(M[:, j]) * np.log1p(np.abs(M[:, j]))
    return M
X = san(pd.concat([df[COND + LIG], acid, df[fp]], axis=1).values)
def gsplits(garr, k, seed):
    um = np.array(sorted(set(garr))); np.random.RandomState(seed).shuffle(um)
    fm = {}; [fm.update({m: i for m in ch}) for i, ch in enumerate(np.array_split(um, k))]
    f = np.array([fm[g] for g in garr]); return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]
def lgbm(n): return lgb.LGBMRegressor(n_estimators=n, learning_rate=0.03, num_leaves=63, subsample=0.8, colsample_bytree=0.7, subsample_freq=1, random_state=0, n_jobs=-1, verbosity=-1)
splits = gsplits(condkey, 5, 0)
oof = np.zeros(len(y))
for ti, vi in splits: oof[vi] = lgbm(700).fit(X[ti], y[ti]).predict(X[vi])
err = np.zeros(len(y)); resid = np.abs(y - oof)
for ti, vi in splits: err[vi] = lgbm(400).fit(X[ti], resid[ti]).predict(X[vi])
print(f"Track B logD, CONDITION-KEY grouped CV (honest), n={len(y)}:")
rows = []
for q in [100, 50, 25, 10]:
    m = np.ones(len(y), bool) if q == 100 else err <= np.percentile(err, q)
    R2, RM = r2_score(y[m], oof[m]), rmse(y[m], oof[m])
    print(f"  {'all' if q==100 else f'top {q}%':<8} R2={R2:.3f}  RMSE={RM:.3f}  (n={int(m.sum())})")
    rows.append({'subset': 'all' if q == 100 else f'top{q}pct', 'R2': round(R2, 3), 'RMSE': round(RM, 3), 'n': int(m.sum())})
print("  (old random-row figures were R2 0.725 all / 0.940 top-10%, both leaky)")
pd.DataFrame(rows).to_csv("track_b_confidence_honest.csv", index=False)
print(f"saved track_b_confidence_honest.csv | {(time.time()-START)/60:.1f} min")
