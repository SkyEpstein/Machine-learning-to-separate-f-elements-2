#!/usr/bin/env python3
"""TabPFN on per-pair delta G (descriptor feature set), with the CPU sample guard
overridden so it can use all ~1800 training rows per fold. Molecule-grouped CV,
R2 and RMSE (kJ/mol). Bar to beat: RandomForest at R2 0.473."""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
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
feat_desc = [c for c in num if c not in DROP + ['Log_D', 'dG'] and not c.startswith('embedding_') and not c.startswith('fp_')]
acid = pd.get_dummies(df['Acid_type'].astype(str), prefix='acid').astype(float)
def san(M):
    M = M.astype(np.float64).copy(); b = ~np.isfinite(M) | (np.abs(M) > 1e30)
    md = np.nan_to_num(np.nanmedian(np.where(b, np.nan, M), axis=0)); ix = np.where(b); M[ix] = np.take(md, ix[1])
    for j in np.where(np.nanmax(np.abs(M), axis=0) > 1e7)[0]: M[:, j] = np.sign(M[:, j]) * np.log1p(np.abs(M[:, j]))
    return M
df['_pk'] = df[[SMI, 'Metal', 'Acid_type', 'Solvent_A', 'Solvent_B']].astype(str).agg('|'.join, axis=1)
fi = df.drop_duplicates('_pk').index; pkf = df.loc[fi, '_pk']
y = df.groupby('_pk')['dG'].mean().reindex(pkf).values
X = san(pd.concat([df[feat_desc], acid], axis=1).loc[fi].values); groups = df.loc[fi, SMI].values
print(f"per-pair systems {len(y)}, features {X.shape[1]}")
from tabpfn import TabPFNRegressor
def mk():
    try: return TabPFNRegressor(device='cpu', ignore_pretraining_limits=True)
    except TypeError: return TabPFNRegressor(device='cpu')
o = np.zeros(len(y)); gkf = GroupKFold(5)
for k, (ti, vi) in enumerate(gkf.split(X, y, groups)):
    o[vi] = mk().fit(X[ti], y[ti]).predict(X[vi]); print(f"  fold {k+1}/5 done ({(time.time()-START)/60:.1f} min)")
print(f"\nTabPFN (desc): R2={r2_score(y,o):.3f}  RMSE={rmse(y,o):.2f} kJ/mol   (bar: RandomForest 0.473)")
pd.DataFrame([{'model': 'TabPFN (desc)', 'R2': round(r2_score(y, o), 3), 'RMSE_kJmol': round(rmse(y, o), 2)}]).to_csv("dg_tabpfn_results.csv", index=False)
print(f"saved dg_tabpfn_results.csv | {(time.time()-START)/60:.1f} min")
