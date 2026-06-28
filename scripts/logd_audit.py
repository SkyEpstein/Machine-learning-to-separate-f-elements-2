#!/usr/bin/env python3
"""
logd_audit.py — empirical audit of the old logD models. Reproduces Track A (new
molecule: conditions + metal, molecule-grouped CV) and Track B (known molecule:
conditions + ECFP + ligand descriptors, random-row CV) from scratch with a single
consistent LightGBM, and answers the load-bearing questions a code read cannot:
  - Is Track A stable across seeds, or a lucky split?
  - Is Track B's high number genuinely the known-molecule (condition-optimization)
    regime? To check, run Track B's SAME rich features under MOLECULE-GROUPED CV; if
    the number collapses toward Track A, then 0.725 is the known-molecule advantage,
    not new-molecule generalization (the honest framing). If it stays high under
    grouping, that points to real leakage.
  - How many rows share a molecule+metal+acid+condition key (the near-replicates that
    random CV scatters across train and test)?
Reports R2 and RMSE (mean +/- std over seeds).
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
SMI = 'SMILES_canonical'; START = time.time()
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
tr = pd.read_csv("Training_Data_V27.csv", low_memory=False); te = pd.read_csv("Testing_Data_V39.csv", low_memory=False)
df = pd.concat([tr, te], ignore_index=True); df = df[df['Log_D'].notna() & df[SMI].notna()].reset_index(drop=True)
kcols = df[[SMI, 'Metal', 'Acid_type']].astype(str).copy()
for c, r in [('Acid_conc_M', 2), ('Temperature_K', 0), ('Extractant_conc_M', 3), ('Metal_conc_mM', 4)]:
    if c in df: kcols[c] = df[c].round(r).astype(str)
key = kcols.agg('|'.join, axis=1)
grng = pd.Series(df['Log_D'].values).groupby(key).transform(lambda v: v.max() - v.min()).values
df = df[(~df.duplicated().values) & (grng <= 2.0)].reset_index(drop=True)
y = df['Log_D'].values.astype(float); groups = df[SMI].astype(str).values
COND = ['Extractant_conc_M', 'Molar_mass(g/mol) A', 'Log_P A', 'Boiling_point(K) A', 'Melting_point(K) A', 'Density(g/mL) A', 'Solubility_in_water(g/L) A', 'Molar_mass(g/mol) B', 'Log_P B', 'Boiling_point(K) B', 'Melting_point(K) B', 'Density(g/mL) B', 'Solubility_in_water(g/L) B', 'Volume_fraction_A', 'Volume_fraction_B', 'Atomic_number', 'Melting_point_K', 'Boiling_point_K', 'Density_g/cm3', 'First_IE_kJ/mol', 'Second_IE_kJ/mol', 'Third_IE_kJ/mol', 'Matallic_radius_nm', 'Pauling_EN', 'Ionic_radius_nm', 'Oxidation_state', 'Metal_conc_mM', 'Dipole_moment_D', 'Acid_conc_M', 'Temperature_K']
LIG = [c for c in ['MolLogP', 'NumHDonors', 'NumHAcceptors', 'NOCount', 'NHOHCount', 'TPSA', 'NumRotatableBonds', 'RingCount', 'FractionCSP3', 'NumAromaticRings'] if c in df]
fp = [c for c in df.columns if c.startswith('fp_')]
acid = pd.get_dummies(df['Acid_type'].astype(str), prefix='acid').astype(float)
def san(M):
    M = M.astype(np.float64).copy(); b = ~np.isfinite(M) | (np.abs(M) > 1e30)
    md = np.nan_to_num(np.nanmedian(np.where(b, np.nan, M), axis=0)); ix = np.where(b); M[ix] = np.take(md, ix[1])
    for j in np.where(np.nanmax(np.abs(M), axis=0) > 1e7)[0]: M[:, j] = np.sign(M[:, j]) * np.log1p(np.abs(M[:, j]))
    return M
XA = san(pd.concat([df[COND], acid], axis=1).values)
XB = san(pd.concat([df[COND + LIG], acid, df[fp]], axis=1).values)
def gsplits(garr, k, seed):
    um = np.array(sorted(set(garr))); np.random.RandomState(seed).shuffle(um)
    fm = {}; [fm.update({m: i for m in ch}) for i, ch in enumerate(np.array_split(um, k))]
    f = np.array([fm[g] for g in garr])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]
def ev(X, splits):
    o = np.zeros(len(y))
    for ti, vi in splits:
        o[vi] = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=63, subsample=0.8, colsample_bytree=0.7, subsample_freq=1, random_state=0, n_jobs=-1, verbosity=-1).fit(X[ti], y[ti]).predict(X[vi])
    return r2_score(y, o), rmse(y, o)
ck = df[[SMI, 'Metal', 'Acid_type']].astype(str).copy()
for c, r in [('Acid_conc_M', 2), ('Temperature_K', 0), ('Extractant_conc_M', 3), ('Metal_conc_mM', 4)]:
    if c in df: ck[c] = df[c].round(r).astype(str)
condkey = ck.agg('|'.join, axis=1).values
print(f"rows {len(df)}, unique molecules {len(set(groups))}, condition-keys {len(set(condkey))}")
def agg3(res): return round(float(np.mean([r for r, _ in res])), 3), round(float(np.std([r for r, _ in res])), 3), round(float(np.mean([m for _, m in res])), 3)
a = agg3([ev(XA, gsplits(groups, 5, s)) for s in range(3)])
b = agg3([ev(XB, list(KFold(5, shuffle=True, random_state=s).split(XB))) for s in range(3)])
c = agg3([ev(XB, gsplits(condkey, 5, s)) for s in range(3)])
g = agg3([ev(XB, gsplits(groups, 5, s)) for s in range(3)])
nfloor = float(np.sqrt(pd.Series(y).groupby(condkey).var().dropna().mean()))
print(f"Track A new-molecule (cond+metal, molecule-grouped):         R2 {a[0]} +/- {a[1]}, RMSE {a[2]}")
print(f"Track B random-row (UPPER BOUND, replicate memorization):    R2 {b[0]} +/- {b[1]}, RMSE {b[2]}")
print(f"Track B condition-key grouped (HONEST condition-interp):     R2 {c[0]} +/- {c[1]}, RMSE {c[2]}")
print(f"Track B molecule-grouped (new molecule):                     R2 {g[0]} +/- {g[1]}, RMSE {g[2]}")
print(f"label-noise floor (within condition-key Log_D std):          {nfloor:.3f} log units")
pd.DataFrame([
    {'track': 'Track A', 'regime': 'new molecule (molecule-grouped)', 'features': 'conditions+metal', 'R2': a[0], 'R2_std': a[1], 'RMSE': a[2], 'note': 'honest new-extractant screening (single-LightGBM reproduction)'},
    {'track': 'Track B', 'regime': 'known molecule, random-row (UPPER BOUND)', 'features': 'conditions+ECFP+ligand', 'R2': b[0], 'R2_std': b[1], 'RMSE': b[2], 'note': 'includes replicate memorization, not generalization'},
    {'track': 'Track B', 'regime': 'condition interpolation (condition-key grouped, HONEST)', 'features': 'conditions+ECFP+ligand', 'R2': c[0], 'R2_std': c[1], 'RMSE': c[2], 'note': 'interpolate new conditions for an already-seen molecule'},
    {'track': 'Track B', 'regime': 'new molecule (molecule-grouped)', 'features': 'conditions+ECFP+ligand', 'R2': g[0], 'R2_std': g[1], 'RMSE': g[2], 'note': 'rich features collapse to about Track A on new molecules'},
    {'track': 'noise floor', 'regime': 'within condition-key Log_D std', 'features': '', 'R2': '', 'R2_std': '', 'RMSE': round(nfloor, 3), 'note': 'irreducible label noise'},
]).to_csv('track_ab_results.csv', index=False)
print(f"saved track_ab_results.csv | {(time.time()-START)/60:.1f} min")
