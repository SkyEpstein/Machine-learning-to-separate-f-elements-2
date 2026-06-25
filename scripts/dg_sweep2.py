#!/usr/bin/env python3
"""
dg_sweep2.py — expanded model sweep for per-pair delta G, beyond the first seven.
Adds TabPFN (a small-data specialist, which this regime is), a descriptor-only
feature set (the ~95 dense RDKit / metal / diluent features, dropping the ~900
sparse fingerprint bits that distance and kernel methods choke on), k-NN, an RBF
SVR, a tuned RandomForest, and a richer NNLS stack that includes TabPFN. Everything
uses molecule-grouped CV and is reported as R2 and RMSE (kJ/mol). The bar to beat is
the RandomForest at R2 0.473.
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.optimize import nnls
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
feat_all = [c for c in num if c not in DROP + ['Log_D', 'dG'] and not c.startswith('embedding_')]
feat_desc = [c for c in feat_all if not c.startswith('fp_')]
acid = pd.get_dummies(df['Acid_type'].astype(str), prefix='acid').astype(float)
def san(M):
    M = M.astype(np.float64).copy(); b = ~np.isfinite(M) | (np.abs(M) > 1e30)
    md = np.nan_to_num(np.nanmedian(np.where(b, np.nan, M), axis=0)); ix = np.where(b); M[ix] = np.take(md, ix[1])
    for j in np.where(np.nanmax(np.abs(M), axis=0) > 1e7)[0]: M[:, j] = np.sign(M[:, j]) * np.log1p(np.abs(M[:, j]))
    return M
df['_pk'] = df[[SMI, 'Metal', 'Acid_type', 'Solvent_A', 'Solvent_B']].astype(str).agg('|'.join, axis=1)
fi = df.drop_duplicates('_pk').index; pkf = df.loc[fi, '_pk']
y = df.groupby('_pk')['dG'].mean().reindex(pkf).values
Xall = san(pd.concat([df[feat_all], acid], axis=1).loc[fi].values)
Xdesc = san(pd.concat([df[feat_desc], acid], axis=1).loc[fi].values)
groups = df.loc[fi, SMI].values
print(f"per-pair systems {len(y)}, unique molecules {len(set(groups))}, full feats {Xall.shape[1]}, descriptor feats {Xdesc.shape[1]}\n")
SPECS = {
    'RF (full 995)': (lambda: RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=0), 'all'),
    'ExtraTrees (full)': (lambda: ExtraTreesRegressor(n_estimators=500, n_jobs=-1, random_state=0), 'all'),
    'RF tuned (full, sqrt)': (lambda: RandomForestRegressor(n_estimators=800, max_features='sqrt', min_samples_leaf=2, n_jobs=-1, random_state=0), 'all'),
    'RF (desc only)': (lambda: RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=0), 'desc'),
    'kNN (desc, scaled)': (lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10, weights='distance')), 'desc'),
    'SVR rbf (desc, scaled)': (lambda: make_pipeline(StandardScaler(), SVR(C=10.0, gamma='scale')), 'desc'),
}
gkf = GroupKFold(5); oof = {}
for name, (mk, fs) in SPECS.items():
    X = Xall if fs == 'all' else Xdesc; o = np.zeros(len(y))
    for ti, vi in gkf.split(X, y, groups):
        o[vi] = mk().fit(X[ti], y[ti]).predict(X[vi])
    oof[name] = o; print(f"  {name:<24} R2={r2_score(y,o):.3f}  RMSE={rmse(y,o):.2f}")
try:
    from tabpfn import TabPFNRegressor
    o = np.zeros(len(y))
    for ti, vi in gkf.split(Xdesc, y, groups):
        o[vi] = TabPFNRegressor(device='cpu').fit(Xdesc[ti], y[ti]).predict(Xdesc[vi])
    oof['TabPFN (desc)'] = o; print(f"  {'TabPFN (desc)':<24} R2={r2_score(y,o):.3f}  RMSE={rmse(y,o):.2f}")
except Exception as e:
    print("  TabPFN skipped:", str(e)[:90])
members = [n for n in ['RF (full 995)', 'ExtraTrees (full)', 'SVR rbf (desc, scaled)', 'TabPFN (desc)'] if n in oof]
P = np.column_stack([oof[n] for n in members]); stack = np.zeros(len(y))
for ti, vi in gkf.split(P, y, groups):
    w, _ = nnls(P[ti], y[ti]); stack[vi] = P[vi] @ w
wf, _ = nnls(P, y)
print(f"\n  NNLS stack {members}\n     R2={r2_score(y,stack):.3f}  RMSE={rmse(y,stack):.2f}  weights={dict(zip(members,[round(float(x),2) for x in wf]))}")
print("\n  (bar to beat: RandomForest full = R2 0.473)")
rows = [{'model': n, 'R2': round(r2_score(y, o), 3), 'RMSE_kJmol': round(rmse(y, o), 2)} for n, o in oof.items()]
rows.append({'model': 'NNLS stack (RF+ET+SVR+TabPFN)', 'R2': round(r2_score(y, stack), 3), 'RMSE_kJmol': round(rmse(y, stack), 2)})
pd.DataFrame(rows).to_csv("dg_sweep2_results.csv", index=False)
print(f"\nsaved dg_sweep2_results.csv | {(time.time()-START)/60:.1f} min")
