#!/usr/bin/env python3
"""
confidence_tune.py — settle the confidence question honestly and pick the best
err-model recipe. Generates OOF ONCE for (a) the single tuned LightGBM ("before")
and (b) the NNLS stack, then bakes off err-model recipes on the HONEST metric
(RMSE@top-k + Spearman), not the variance-confounded top-k R². Saves OOF to
conf_oof_<A|B>.npz so further tuning is instant.

Recipes tested on the stack: plain [cond, pred] / rich [+members+disagreement+
novelty] / lean [+disagreement+novelty only], each with a strong or regularized
err-LightGBM. The single model gets plain+strong (its final_deliverable recipe).
Track A grouped CV (cond+metal); Track B random CV (cond+ECFP+ligand); cleaned.
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from scipy.stats import spearmanr
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
kdf = df[[SMI, 'Metal'] + [c for c in ['Acid_type'] if c in df]].copy()
for c, r in [('Acid_conc_M', 2), ('Temperature_K', 0), ('Extractant_conc_M', 3), ('Metal_conc_mM', 4)]:
    if c in df: kdf[c] = df[c].round(r)
grng = pd.Series(df[TARGET].values).groupby(kdf.astype(str).agg('|'.join, axis=1)).transform(lambda v: v.max() - v.min()).values
df = df[(~df.duplicated().values) & (grng <= 2.0)].reset_index(drop=True)
y = df[TARGET].values.astype(float); smi = df[SMI].astype(str).values
def lig(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return [0, 0, 0, 0, 0.0]
    sy = [a.GetSymbol() for a in m.GetAtoms()]; return [sy.count('O'), sy.count('N'), sy.count('S'), sy.count('P'), Descriptors.MolLogP(m)]
Lg = np.array([lig(s) for s in smi], float); fp = [c for c in allf if c.lower().startswith('fp_')]
def san(M):
    M = M.astype(np.float64).copy(); b = ~np.isfinite(M) | (np.abs(M) > 1e30); md = np.nan_to_num(np.nanmedian(np.where(b, np.nan, M), axis=0)); ix = np.where(b); M[ix] = np.take(md, ix[1])
    for j in np.where(np.nanmax(np.abs(M), axis=0) > 1e7)[0]: M[:, j] = np.sign(M[:, j]) * np.log1p(np.abs(M[:, j]))
    return M
Xcond = san(df[[c for c in COND if c in df]].values)
Xfull = san(np.hstack([df[[c for c in COND if c in df]].values, df[fp].values, Lg]))
def MODELS(X):
    m = {'lgb': lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=63, min_child_samples=12, random_state=SEED, n_jobs=-1, verbosity=-1),
         'xgb': xgb.XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=7, subsample=0.9, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=0),
         'hgb': HistGradientBoostingRegressor(max_iter=1200, learning_rate=0.03, max_depth=10, random_state=SEED),
         'et': ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.6, random_state=SEED, n_jobs=-1),
         'ridge': Ridge(alpha=5.0), 'mlp': MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=400, early_stopping=True, random_state=SEED)}
    if CAT: m['cat'] = CatBoostRegressor(iterations=2500, learning_rate=0.03, depth=8, random_state=SEED, verbose=False, allow_writing_files=False)
    return m
SCALED = {'ridge', 'mlp'}
STRONG = lambda: lgb.LGBMRegressor(n_estimators=1800, learning_rate=0.03, num_leaves=63, min_child_samples=12, subsample=0.85, colsample_bytree=0.8, reg_lambda=1.5, random_state=SEED, n_jobs=-1, verbosity=-1)
REG = lambda: lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, random_state=SEED, n_jobs=-1, verbosity=-1)
def folds(grouped, rep):
    if grouped:
        uq = np.unique(smi).copy(); np.random.RandomState(SEED + rep).shuffle(uq); fo = {m: i % 5 for i, m in enumerate(uq)}; return np.array([fo[s] for s in smi])
    f = np.zeros(len(y), int)
    for i, (_, va) in enumerate(KFold(5, shuffle=True, random_state=SEED + rep).split(y)): f[va] = i
    return f
def err_oof(feat, resid, fold, model_fn):
    e = np.zeros(len(y))
    for f in range(5):
        trr = np.where(fold != f)[0]; va = np.where(fold == f)[0]
        e[va] = model_fn().fit(feat[trr], resid[trr]).predict(feat[va])
    return np.clip(e, 0.05, None)
def show(tag, y, pred, err):
    sp = spearmanr(err, np.abs(y - pred)).correlation; row = f"  {tag:<26s} Spearman={sp:.3f} | "
    for p in [50, 25, 10]:
        m = err <= np.percentile(err, p); row += f"top{p}: R²={r2_score(y[m],pred[m]):.3f} RMSE={np.sqrt(mean_squared_error(y[m],pred[m])):.3f}  "
    print(row, flush=True)

for X, grouped, label in [(Xcond, True, "TRACK A (cond+metal, grouped)"), (Xfull, False, "TRACK B (cond+ECFP+ligand, random)")]:
    fold = folds(grouped, 0); sc = StandardScaler().fit(X); Xs = sc.transform(X); names = list(MODELS(X).keys())
    oof = {n: np.zeros(len(y)) for n in names}; single = np.zeros(len(y))
    for f in range(5):
        trr = np.where(fold != f)[0]; va = np.where(fold == f)[0]
        for n, mdl in MODELS(X).items():
            XX = Xs if n in SCALED else X
            try: oof[n][va] = np.nan_to_num(mdl.fit(XX[trr], y[trr]).predict(XX[va]))
            except Exception: pass
        single[va] = STRONG().fit(X[trr], y[trr]).predict(X[va])
    P = np.column_stack([oof[n] for n in names]); lr = LinearRegression(positive=True, fit_intercept=False).fit(P, y)
    w = np.clip(lr.coef_, 0, None); w /= (w.sum() + 1e-12); stack = P @ w
    mstd = P.std(1, keepdims=True); mrng = (P.max(1) - P.min(1)).reshape(-1, 1); nov = np.zeros(len(y))
    for f in range(5):
        trr = np.where(fold != f)[0]; va = np.where(fold == f)[0]
        scn = StandardScaler().fit(X[trr]); Zva = scn.transform(X[va])
        pc = PCA(n_components=min(15, X.shape[1]), svd_solver='randomized', random_state=SEED).fit(scn.transform(X[trr]))
        nov[va] = ((Zva - pc.inverse_transform(pc.transform(Zva))) ** 2).sum(1)
    np.savez(f"conf_oof_{'A' if grouped else 'B'}.npz", y=y, single=single, stack=stack, P=P, Xcond=Xcond, mstd=mstd, mrng=mrng, nov=nov, fold=fold, names=names)
    print(f"\n=== {label} ===  single R²={r2_score(y,single):.4f}  stack R²={r2_score(y,stack):.4f}", flush=True)
    rs, rk = np.abs(y - single), np.abs(y - stack); nv = nov.reshape(-1, 1)
    show("SINGLE plain+strong", y, single, err_oof(np.column_stack([Xcond, single]), rs, fold, STRONG))
    show("STACK plain+strong",  y, stack,  err_oof(np.column_stack([Xcond, stack]), rk, fold, STRONG))
    show("STACK plain+reg",     y, stack,  err_oof(np.column_stack([Xcond, stack]), rk, fold, REG))
    show("STACK rich+reg",      y, stack,  err_oof(np.column_stack([Xcond, stack, P, mstd, mrng, nv]), rk, fold, REG))
    show("STACK lean+strong",   y, stack,  err_oof(np.column_stack([Xcond, stack, mstd, nv]), rk, fold, STRONG))
print(f"\ntotal {(time.time()-START)/60:.1f} min", flush=True)
