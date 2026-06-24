#!/usr/bin/env python3
"""
ensemble_final.py — the deployable, UPGRADED to the NNLS-stacked ensemble that
ensemble_fair.py showed beats the best single model (Track A +0.018 -> ~0.495 at
the oracle ceiling; Track B +0.013 -> ~0.721). Same cleaning + err_lgb confidence
+ normalized split-conformal as final_deliverable.py; the ONLY change is the point
prediction is now a non-negative least-squares (NNLS) stack of a diverse roster
(lgb/xgb/hgb/et/cat/ridge/mlp) instead of a single LightGBM. NNLS auto-zeros the
weak/redundant members, so the stack is self-pruning. Prints the learned weights
(what's actually deployed) and per-model OOF R² for the record.

Track A NEW-molecule screening: conditions+metal, molecule-grouped CV.
Track B KNOWN-molecule condition-opt: conditions+ECFP+ligand, random-row CV.
Saves ensemble_<A|B>_predictions.csv (pred, confidence, lo90/hi90, lo80/hi80).
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.decomposition import PCA
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
# ---- cleaning (same as final_deliverable) ----
kdf = df[[SMI, 'Metal'] + [c for c in ['Acid_type'] if c in df]].copy()
for c, r in [('Acid_conc_M', 2), ('Temperature_K', 0), ('Extractant_conc_M', 3), ('Metal_conc_mM', 4)]:
    if c in df: kdf[c] = df[c].round(r)
gid = kdf.astype(str).agg('|'.join, axis=1)
grng = pd.Series(df[TARGET].values).groupby(gid).transform(lambda v: v.max() - v.min()).values
clean = (~df.duplicated().values) & (grng <= 2.0)
print(f"cleaning: kept {clean.sum()}/{len(df)} rows (dropped {(~clean).sum()}: exact-dups + discordant range>2)", flush=True)
df = df[clean].reset_index(drop=True)
y = df[TARGET].values.astype(float); smi = df[SMI].astype(str).values
def lig(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return [0, 0, 0, 0, 0.0]
    sy = [a.GetSymbol() for a in m.GetAtoms()]; return [sy.count('O'), sy.count('N'), sy.count('S'), sy.count('P'), Descriptors.MolLogP(m)]
Lg = np.array([lig(s) for s in smi], float)
fp = [c for c in allf if c.lower().startswith('fp_')]
def san(M):
    M = M.astype(np.float64).copy(); b = ~np.isfinite(M) | (np.abs(M) > 1e30); md = np.nan_to_num(np.nanmedian(np.where(b, np.nan, M), axis=0)); ix = np.where(b); M[ix] = np.take(md, ix[1])
    for j in np.where(np.nanmax(np.abs(M), axis=0) > 1e7)[0]: M[:, j] = np.sign(M[:, j]) * np.log1p(np.abs(M[:, j]))
    return M
Xcond = san(df[[c for c in COND if c in df]].values)
Xfull = san(np.hstack([df[[c for c in COND if c in df]].values, df[fp].values, Lg]))
def MODELS(X):  # diverse roster; NNLS self-prunes the weak/redundant ones
    m = {'lgb': lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=63, min_child_samples=12, random_state=SEED, n_jobs=-1, verbosity=-1),
         'xgb': xgb.XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=7, subsample=0.9, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=0),
         'hgb': HistGradientBoostingRegressor(max_iter=1200, learning_rate=0.03, max_depth=10, random_state=SEED),
         'et': ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.6, random_state=SEED, n_jobs=-1),
         'ridge': Ridge(alpha=5.0), 'mlp': MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=400, early_stopping=True, random_state=SEED)}
    if CAT: m['cat'] = CatBoostRegressor(iterations=2500, learning_rate=0.03, depth=8, random_state=SEED, verbose=False, allow_writing_files=False)
    return m
SCALED = {'ridge', 'mlp'}
errmk = lambda: lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, random_state=SEED, n_jobs=-1, verbosity=-1)
def folds(grouped, rep):
    if grouped:
        uq = np.unique(smi).copy(); np.random.RandomState(SEED + rep).shuffle(uq); fo = {m: i % 5 for i, m in enumerate(uq)}; return np.array([fo[s] for s in smi])
    f = np.zeros(len(y), int)
    for i, (_, va) in enumerate(KFold(5, shuffle=True, random_state=SEED + rep).split(y)): f[va] = i
    return f

def run(X, grouped, label, tag):
    names = list(MODELS(X).keys()); fold = folds(grouped, 0)
    sc = StandardScaler().fit(X); Xs = sc.transform(X)
    oof = {n: np.zeros(len(y)) for n in names}
    for f in range(5):
        trr = np.where(fold != f)[0]; va = np.where(fold == f)[0]
        for n, mdl in MODELS(X).items():
            XX = Xs if n in SCALED else X
            try: oof[n][va] = np.nan_to_num(mdl.fit(XX[trr], y[trr]).predict(XX[va]))
            except Exception: pass
    P = np.column_stack([oof[n] for n in names])
    lr = LinearRegression(positive=True, fit_intercept=False).fit(P, y); w = np.clip(lr.coef_, 0, None); w /= (w.sum() + 1e-12)
    pred = P @ w
    # per-model + stack
    r2s = {n: r2_score(y, oof[n]) for n in names}
    best = max(r2s, key=lambda k: r2s[k])
    r2 = r2_score(y, pred); rmse = np.sqrt(mean_squared_error(y, pred))
    print(f"\n=== {label} ===  (cleaned data)", flush=True)
    for n in sorted(r2s, key=lambda k: -r2s[k]): print(f"    {n:<6s} R²={r2s[n]:.4f}  weight={w[names.index(n)]:.2f}", flush=True)
    print(f"  NNLS STACK R²={r2:.4f} RMSE={rmse:.4f}   (best single {best} {r2s[best]:.4f}; Δ={r2-r2s[best]:+.4f})", flush=True)
    # ---- confidence: RICH err_lgb (conditions + stack pred + MEMBER preds + DISAGREEMENT + PCA-novelty) ----
    Pmem = np.column_stack([oof[n] for n in names])              # per-member OOF preds (ensemble's native confidence signal)
    mstd = Pmem.std(axis=1, keepdims=True); mrng = (Pmem.max(1) - Pmem.min(1)).reshape(-1, 1)
    res = np.abs(y - pred); err = np.zeros(len(y)); nov = np.zeros(len(y))
    for f in range(5):                                           # CV-safe PCA-novelty: recon error vs train-fold subspace
        trr = np.where(fold != f)[0]; va = np.where(fold == f)[0]
        scn = StandardScaler().fit(X[trr]); Zva = scn.transform(X[va])
        pc = PCA(n_components=min(15, X.shape[1]), svd_solver='randomized', random_state=SEED).fit(scn.transform(X[trr]))
        nov[va] = ((Zva - pc.inverse_transform(pc.transform(Zva))) ** 2).sum(1)
    Cf = np.column_stack([Xcond, pred, Pmem, mstd, mrng, nov.reshape(-1, 1)])
    for f in range(5):
        trr = np.where(fold != f)[0]; va = np.where(fold == f)[0]
        err[va] = errmk().fit(Cf[trr], res[trr]).predict(Cf[va])
    err = np.clip(err, 0.05, None)
    print(f"  confidence err_lgb (rich: +members +disagreement +novelty)  Spearman(err,|resid|)={spearmanr(err, res).correlation:.3f}", flush=True)
    print("  confidence-filtered:", flush=True)
    for p in [100, 50, 25, 10]:
        m = np.ones(len(y), bool) if p == 100 else (err <= np.percentile(err, p))
        print(f"    top {p:>3d}%  R²={r2_score(y[m],pred[m]):.3f}  RMSE={np.sqrt(mean_squared_error(y[m],pred[m])):.3f}", flush=True)
    # normalized split-conformal: calibrate q on 3 folds, measure coverage/width on 2 (4 reps)
    print("  normalized conformal (split-calibrated, honest coverage):", flush=True)
    cov90, cov80, w90, w90c = [], [], [], []
    for rep in range(4):
        f2 = folds(grouped, 100 + rep); cal = np.where(f2 < 3)[0]; tst = np.where(f2 >= 3)[0]
        s = np.abs(y[cal] - pred[cal]) / err[cal]
        for alpha, cov, wl in [(0.10, cov90, w90), (0.20, cov80, None)]:
            qn = min(1.0, np.ceil((len(cal) + 1) * (1 - alpha)) / len(cal)); q = np.quantile(s, qn)
            lo = pred[tst] - q * err[tst]; hi = pred[tst] + q * err[tst]
            cov.append(np.mean((y[tst] >= lo) & (y[tst] <= hi)))
            if wl is not None:
                wl.append(np.median(hi - lo)); conf25 = err[tst] <= np.percentile(err[tst], 25); w90c.append(np.median((hi - lo)[conf25]))
    print(f"    90% target -> empirical coverage {np.mean(cov90):.2f}, median width {np.mean(w90):.2f} log units (top-25% conf width {np.mean(w90c):.2f})", flush=True)
    print(f"    80% target -> empirical coverage {np.mean(cov80):.2f}", flush=True)
    s = np.abs(y - pred) / err; q90 = np.quantile(s, min(1.0, np.ceil((len(y) + 1) * 0.9) / len(y))); q80 = np.quantile(s, min(1.0, np.ceil((len(y) + 1) * 0.8) / len(y)))
    out = pd.DataFrame({'SMILES': smi, 'Metal': df['Metal'].values if 'Metal' in df else '', 'Acid_conc_M': df['Acid_conc_M'].values, 'Temperature_K': df['Temperature_K'].values,
                        'Actual_LogD': y, 'Pred_LogD': pred, 'confidence_pred_err': err,
                        'lo90': pred - q90 * err, 'hi90': pred + q90 * err, 'lo80': pred - q80 * err, 'hi80': pred + q80 * err})
    out.to_csv(f"ensemble_{tag}_predictions.csv", index=False)
    print(f"  saved ensemble_{tag}_predictions.csv  | deployed weights: " + ", ".join(f"{n}={w[names.index(n)]:.2f}" for n in names if w[names.index(n)] > 0.005), flush=True)

run(Xcond, True, "TRACK A — new-molecule screening (conditions+metal)", "A_screening")
run(Xfull, False, "TRACK B — known-molecule condition-opt (conditions+ECFP+ligand)", "B_condition")
print(f"\ntotal {(time.time()-START)/60:.1f} min", flush=True)
