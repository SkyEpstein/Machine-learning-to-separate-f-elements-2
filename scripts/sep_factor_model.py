#!/usr/bin/env python3
"""
sep_factor_model.py — direct separation-factor (delta-logD) regressor vs differencing.
For two metals i, j measured at the same extractant and conditions, the separation
target is delta = logD_i - logD_j (the log separation factor).
  DIRECT     : a model trained on pair rows, features = extractant descriptors +
               shared conditions + both metals' descriptors, target = delta.
  DIFFERENCE : a per-row logD model (extractant + conditions + single-metal
               descriptors), out-of-fold predictions subtracted for each pair.
Both use molecule-grouped CV (group = extractant SMILES), so neither sees the held-out
extractant. The question is whether predicting delta directly beats differencing via
shared-error cancellation. Reports signed-delta R2/RMSE, magnitude R2, and direction
accuracy for each, plus a per-pair trust map.
"""
import os, time, warnings, itertools; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
SMI = 'SMILES_canonical'; START = time.time()
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
tr = pd.read_csv("Training_Data_V27.csv", low_memory=False); te = pd.read_csv("Testing_Data_V39.csv", low_memory=False)
df = pd.concat([tr, te], ignore_index=True); df = df[df['Log_D'].notna() & df[SMI].notna()].reset_index(drop=True)
kc = df[[SMI, 'Metal', 'Acid_type']].astype(str).copy()
for c, r in [('Acid_conc_M', 2), ('Temperature_K', 0), ('Extractant_conc_M', 3), ('Metal_conc_mM', 4)]:
    if c in df: kc[c] = df[c].round(r).astype(str)
grng = pd.Series(df['Log_D'].values).groupby(kc.agg('|'.join, axis=1)).transform(lambda v: v.max() - v.min()).values
df = df[(~df.duplicated().values) & (grng <= 2.0)].reset_index(drop=True)
METAL = ['Atomic_number', 'Melting_point_K', 'Boiling_point_K', 'Density_g/cm3', 'First_IE_kJ/mol', 'Second_IE_kJ/mol', 'Third_IE_kJ/mol', 'Matallic_radius_nm', 'Pauling_EN', 'Ionic_radius_nm', 'Oxidation_state', 'Metal_conc_mM']
COND = ['Extractant_conc_M', 'Acid_conc_M', 'Temperature_K', 'Dipole_moment_D', 'Volume_fraction_A', 'Volume_fraction_B', 'Molar_mass(g/mol) A', 'Log_P A', 'Boiling_point(K) A', 'Melting_point(K) A', 'Density(g/mL) A', 'Solubility_in_water(g/L) A', 'Molar_mass(g/mol) B', 'Log_P B', 'Boiling_point(K) B', 'Melting_point(K) B', 'Density(g/mL) B', 'Solubility_in_water(g/L) B']
num = df.select_dtypes(np.number).columns.tolist()
EXT = [c for c in num if c not in METAL + COND + ['Log_D'] and not c.startswith('fp_') and not c.startswith('embedding_')]
acid = pd.get_dummies(df['Acid_type'].astype(str), prefix='acid').astype(float)
def san(M):
    M = M.astype(np.float64).copy(); b = ~np.isfinite(M) | (np.abs(M) > 1e30)
    md = np.nan_to_num(np.nanmedian(np.where(b, np.nan, M), axis=0)); ix = np.where(b); M[ix] = np.take(md, ix[1])
    for j in np.where(np.nanmax(np.abs(M), axis=0) > 1e7)[0]: M[:, j] = np.sign(M[:, j]) * np.log1p(np.abs(M[:, j]))
    return M
Xshared = san(pd.concat([df[EXT + COND], acid], axis=1).values)  # per-row, shared within a pair
Xmetal = san(df[METAL].values)                                   # per-row metal block
logD = df['Log_D'].values.astype(float); smiles = df[SMI].astype(str).values
def lgbm(): return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=63, subsample=0.8, colsample_bytree=0.7, subsample_freq=1, random_state=0, n_jobs=-1, verbosity=-1)
gkf = GroupKFold(5)
# row-level logD model (differencing baseline predictor)
Xrow = np.hstack([Xshared, Xmetal]); oof = np.zeros(len(df))
for ti, vi in gkf.split(Xrow, logD, smiles):
    oof[vi] = lgbm().fit(Xrow[ti], logD[ti]).predict(Xrow[vi])
print(f"row-level logD model: R2={r2_score(logD,oof):.3f} (this is the per-metal predictor)")
# build all metal pairs within (extractant, acid, conditions) groups
gk = df[[SMI, 'Acid_type']].astype(str).agg('|'.join, axis=1)
for c, r in [('Acid_conc_M', 2), ('Temperature_K', 0), ('Extractant_conc_M', 3)]:
    gk = gk + '|' + df[c].round(r).astype(str)
gkey = gk.values
ii, jj = [], []
g2idx = {}
for idx, g in enumerate(gkey): g2idx.setdefault(g, []).append(idx)
for g, idxs in g2idx.items():
    if len(idxs) < 2: continue
    for a, b in itertools.combinations(idxs, 2):
        if df['Metal'].iloc[a] != df['Metal'].iloc[b]: ii.append(a); jj.append(b)
ii, jj = np.array(ii), np.array(jj)
Xpair = np.hstack([Xshared[ii], Xmetal[ii], Xmetal[jj]])         # shared + metal_i + metal_j
target = logD[ii] - logD[jj]; grp = smiles[ii]
diff = oof[ii] - oof[jj]                                          # differencing baseline
print(f"pairs: {len(ii)} across {len(set(grp))} extractants")
pred = np.zeros(len(ii))
for ti, vi in gkf.split(Xpair, target, grp):
    pred[vi] = lgbm().fit(Xpair[ti], target[ti]).predict(Xpair[vi])
def report(name, p):
    return {'method': name, 'signed_R2': round(r2_score(target, p), 3), 'signed_RMSE': round(rmse(target, p), 3),
            'magnitude_R2': round(r2_score(np.abs(target), np.abs(p)), 3), 'direction_acc': round(float((np.sign(p) == np.sign(target)).mean()), 3)}
rows = [report('direct delta model', pred), report('differencing baseline', diff)]
print("\n=== separation factor (delta logD), molecule-grouped CV ===")
for r in rows: print(f"  {r['method']:<22} signed R2={r['signed_R2']:.3f} RMSE={r['signed_RMSE']:.3f} | |mag| R2={r['magnitude_R2']:.3f} | direction {r['direction_acc']:.3f}")
pd.DataFrame(rows).to_csv("sep_factor_results.csv", index=False)
# per-pair trust map (unordered metal pair), direct model
pl = [tuple(sorted((df['Metal'].iloc[a], df['Metal'].iloc[b]))) for a, b in zip(ii, jj)]
pdf = pd.DataFrame({'pair': ['/'.join(p) for p in pl], 'target': target, 'pred': pred})
tm = pdf.groupby('pair').apply(lambda d: pd.Series({'n': len(d), 'R2': r2_score(d.target, d.pred) if len(d) >= 30 else np.nan})).dropna().sort_values('R2', ascending=False)
tm.to_csv("sep_factor_by_pair.csv")
print(f"\nper-pair trust map (n>=30): {len(tm)} pairs; best {tm.index[0]} R2={tm.R2.iloc[0]:.2f}, worst {tm.index[-1]} R2={tm.R2.iloc[-1]:.2f}")
print(f"saved sep_factor_results.csv, sep_factor_by_pair.csv | {(time.time()-START)/60:.1f} min")
