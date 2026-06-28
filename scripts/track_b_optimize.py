#!/usr/bin/env python3
"""
track_b_optimize.py — push the KNOWN-MOLECULE condition-optimization model toward
the ~0.77 ceiling. Task split = random-row 5-fold CV (predict new rows/conditions
for extractants already in the DB). Structure HELPS here, so test richer feature
sets (incl. the 768 embeddings, which hurt new-molecule but may help known) and a
strong ensemble; then confidence-rank the winner. Baseline (single lgb, cond+struct)
≈ 0.63; noise-floor ceiling ≈ 0.77.
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
import lightgbm as lgb, xgboost as xgb
from rdkit import Chem
from rdkit.Chem import Descriptors
try:
    from catboost import CatBoostRegressor; CAT = True
except Exception: CAT = False
SEED = 42; START = time.time()
TARGET, SMI = "Log_D", "SMILES_canonical"
TEXT = {"Solvent_A", "Solvent_B", "Metal", "Acid_type", "SMILES_canonical"}
COND = ['Extractant_conc_M','Molar_mass(g/mol) A','Log_P A','Boiling_point(K) A','Melting_point(K) A','Density(g/mL) A','Solubility_in_water(g/L) A','Molar_mass(g/mol) B','Log_P B','Boiling_point(K) B','Melting_point(K) B','Density(g/mL) B','Solubility_in_water(g/L) B','Volume_fraction_A','Volume_fraction_B','Atomic_number','Melting_point_K','Boiling_point_K','Density_g/cm3','First_IE_kJ/mol','Second_IE_kJ/mol','Third_IE_kJ/mol','Matallic_radius_nm','Pauling_EN','Ionic_radius_nm','Oxidation_state','Metal_conc_mM','Dipole_moment_D','Acid_conc_M','Temperature_K']
tr = pd.read_csv("Training_Data_V27.csv", low_memory=False); te = pd.read_csv("Testing_Data_V39.csv", low_memory=False)
df = pd.concat([tr, te], ignore_index=True)
num = sorted(set(tr.select_dtypes(np.number).columns) & set(te.select_dtypes(np.number).columns))
allf = [c for c in num if c != TARGET and c not in TEXT]
df = df[df[allf + [TARGET]].notna().all(axis=1) & df[SMI].notna()].reset_index(drop=True)
y = df[TARGET].values.astype(float)
def lig(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return [0, 0, 0, 0, 0.0]
    sy = [a.GetSymbol() for a in m.GetAtoms()]; return [sy.count('O'), sy.count('N'), sy.count('S'), sy.count('P'), Descriptors.MolLogP(m)]
Lg = np.array([lig(s) for s in df[SMI].astype(str)], float)
fp = [c for c in allf if c.lower().startswith('fp_')]; emb = [c for c in allf if c.startswith('embedding_')]
rdk = [c for c in allf if c not in fp and c not in emb and c not in COND]
def san(M):
    M = M.astype(np.float64).copy(); b = ~np.isfinite(M) | (np.abs(M) > 1e30); md = np.nan_to_num(np.nanmedian(np.where(b, np.nan, M), axis=0)); ix = np.where(b); M[ix] = np.take(md, ix[1])
    for j in np.where(np.nanmax(np.abs(M), axis=0) > 1e7)[0]: M[:, j] = np.sign(M[:, j]) * np.log1p(np.abs(M[:, j]))
    return M
C = df[[c for c in COND if c in df]].values; F = df[fp].values; R = df[rdk].values; E = df[emb].values
SETS = {
    'cond+metal': san(C),
    'cond+ECFP+ligand': san(np.hstack([C, F, Lg])),
    'cond+ECFP+ligand+RDKit': san(np.hstack([C, F, Lg, R])),
    'ALL (+embeddings)': san(np.hstack([C, F, Lg, R, E])),
}
def rfolds(rep):
    f = np.zeros(len(y), int)
    for i, (_, va) in enumerate(KFold(5, shuffle=True, random_state=SEED + rep).split(y)): f[va] = i
    return f
def lgbm(): return lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=63, min_child_samples=10, subsample=0.85, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1)
def cv(X, model_fn, reps=2):
    r = []
    for rep in range(reps):
        fold = rfolds(rep); oof = np.zeros(len(y))
        for fdi in range(5):
            trr = np.where(fold != fdi)[0]; va = np.where(fold == fdi)[0]; oof[va] = model_fn(X[trr], y[trr], X[va])
        r.append(r2_score(y, oof))
    return np.mean(r), np.std(r), oof
def lgb_pred(Xt, yt, Xv): return lgbm().fit(Xt, yt).predict(Xv)
print("=== PHASE 1: feature-set scan (single lgb, random-row CV) — current 0.63, ceiling ~0.77 ===", flush=True)
best = None
for name, X in SETS.items():
    m, s, _ = cv(X, lgb_pred); print(f"  {name:<26s} R²={m:.4f}±{s:.3f}  ({X.shape[1]} feat)", flush=True)
    if best is None or m > best[1]: best = (name, m, X)
print(f"\nbest feature set: {best[0]} (lgb R²={best[1]:.4f})", flush=True)

print("\n=== PHASE 2: strong ensemble on best feature set ===", flush=True)
def build(name, X, y_, Xi, yi, Xv, yv):
    if name == 'xgb': m = xgb.XGBRegressor(n_estimators=3000, learning_rate=0.02, max_depth=7, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=-1, early_stopping_rounds=80, verbosity=0); m.fit(Xi, yi, eval_set=[(Xv, yv)], verbose=False); return m
    if name == 'lgb': m = lgbm(); m.fit(Xi, yi, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(80, verbose=False)]); return m
    if name == 'hgb': return HistGradientBoostingRegressor(max_iter=1500, learning_rate=0.03, max_depth=10, min_samples_leaf=8, early_stopping=True, n_iter_no_change=60, random_state=SEED).fit(X, y_)
    if name == 'et': return ExtraTreesRegressor(n_estimators=600, min_samples_leaf=2, max_features=0.6, random_state=SEED, n_jobs=-1).fit(X, y_)
    if name == 'rf': return RandomForestRegressor(n_estimators=600, min_samples_leaf=2, max_features=0.5, random_state=SEED, n_jobs=-1).fit(X, y_)
    if name == 'cat' and CAT: m = CatBoostRegressor(iterations=4000, learning_rate=0.02, depth=8, random_state=SEED, early_stopping_rounds=80, verbose=False, allow_writing_files=False); m.fit(Xi, yi, eval_set=(Xv, yv), verbose=False); return m
ROST = ['xgb', 'lgb', 'hgb', 'et', 'rf'] + (['cat'] if CAT else [])
Xb = best[2]; r2s = []; oof_blend = None
for rep in range(2):
    fold = rfolds(rep); per = {n: np.zeros(len(y)) for n in ROST}
    for fdi in range(5):
        trr = np.where(fold != fdi)[0]; va = np.where(fold == fdi)[0]
        Xi, Xv, yi, yv = train_test_split(Xb[trr], y[trr], test_size=0.1, random_state=SEED)
        for n in ROST:
            try: per[n][va] = np.nan_to_num(build(n, Xb[trr], y[trr], Xi, yi, Xv, yv).predict(Xb[va]))
            except Exception: pass
    w = np.array([max(0, r2_score(y, per[n])) ** 2 for n in ROST]); w /= w.sum()
    blend = sum(wi * per[n] for wi, n in zip(w, ROST)); r2s.append(r2_score(y, blend))
    if rep == 0: oof_blend = blend
print(f"  strong blend ({'+'.join(ROST)}) on {best[0]}: R²={np.mean(r2s):.4f}±{np.std(r2s):.3f}", flush=True)
print(f"  (single lgb was {best[1]:.4f}; current Track-B 0.63; ceiling ~0.77)", flush=True)
# confidence curve on the blend
resid = np.abs(y - oof_blend); fold = rfolds(0); conf = np.zeros(len(y))
Cf = np.column_stack([san(C), oof_blend])
for fdi in range(5):
    trr = np.where(fold != fdi)[0]; va = np.where(fold == fdi)[0]; conf[va] = lgbm().fit(Cf[trr], resid[trr]).predict(Cf[va])
print("\n  confidence-filtered (blend + err_lgb):", flush=True)
rows = []
for p in [100, 50, 25, 10, 5]:
    m = np.ones(len(y), bool) if p == 100 else (conf <= np.percentile(conf, p))
    r2 = r2_score(y[m], oof_blend[m]); rm = np.sqrt(mean_squared_error(y[m], oof_blend[m]))
    print(f"    top {p:>3d}%  R²={r2:.4f}  RMSE={rm:.4f}", flush=True); rows.append({'pct': p, 'r2': r2, 'rmse': rm})
pd.DataFrame(rows).to_csv("track_b_curve.csv", index=False)
print(f"\nsaved track_b_curve.csv | {(time.time()-START)/60:.1f} min", flush=True)
