#!/usr/bin/env python3
"""
sep_factor_zhang.py — how well does Dr. Zhang's own model predict the separation factor
(the difference in logD between two f-elements)? Runs the same two new tests we ran on our
model, but with HIS model and feature style: an XGBoost regressor (n_estimators=700,
max_depth=6, lr=0.05, subsample=0.9, colsample_bytree=0.6, reg_lambda=1.5) on his
ECFP-centric feature set (Morgan fingerprint + RDKit descriptors + conditions + metal +
acid). Separation is obtained by differencing his per-row logD predictions, in two regimes
(known extractant via condition-key folds, new extractant via molecule-grouped folds), and
scored with the full suite plus the confidence sweep. His model's numbers are printed next
to ours.

Honest note: Dr. Zhang's combined dataset is the same underlying data as ours (our audit
established same rows, molecules, and metals), so this compares his MODEL to ours on the
same measurements, not on independent data.
"""
import os, time, warnings, itertools; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, re
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, f1_score
from scipy.stats import spearmanr
import xgboost as xgb
SMI = 'SMILES_canonical'; SEED = 42; START = time.time()
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
FELEM = set("La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split())
def sym(m):
    x = re.match(r"\s*([A-Z][a-z]?)", str(m)); return x.group(1) if x else ""
tr = pd.read_csv("Training_Data_V27.csv", low_memory=False); te = pd.read_csv("Testing_Data_V39.csv", low_memory=False)
df = pd.concat([tr, te], ignore_index=True); df = df[df['Log_D'].notna() & df[SMI].notna()].reset_index(drop=True)
kc = df[[SMI, 'Metal', 'Acid_type']].astype(str).copy()
for c, r in [('Acid_conc_M', 2), ('Temperature_K', 0), ('Extractant_conc_M', 3), ('Metal_conc_mM', 4)]:
    if c in df: kc[c] = df[c].round(r).astype(str)
ckey = kc.agg('|'.join, axis=1)
grng = pd.Series(df['Log_D'].values).groupby(ckey).transform(lambda v: v.max() - v.min()).values
keep = (~df.duplicated().values) & (grng <= 2.0)
df = df[keep].reset_index(drop=True); ckey = ckey[keep].reset_index(drop=True)
METAL = ['Atomic_number', 'Melting_point_K', 'Boiling_point_K', 'Density_g/cm3', 'First_IE_kJ/mol', 'Second_IE_kJ/mol', 'Third_IE_kJ/mol', 'Matallic_radius_nm', 'Pauling_EN', 'Ionic_radius_nm', 'Oxidation_state', 'Metal_conc_mM']
COND = ['Extractant_conc_M', 'Acid_conc_M', 'Temperature_K', 'Dipole_moment_D', 'Volume_fraction_A', 'Volume_fraction_B', 'Molar_mass(g/mol) A', 'Log_P A', 'Boiling_point(K) A', 'Melting_point(K) A', 'Density(g/mL) A', 'Solubility_in_water(g/L) A', 'Molar_mass(g/mol) B', 'Log_P B', 'Boiling_point(K) B', 'Melting_point(K) B', 'Density(g/mL) B', 'Solubility_in_water(g/L) B']
num = df.select_dtypes(np.number).columns.tolist()
FP = [c for c in num if c.startswith('fp_')]
EXT = [c for c in num if c not in METAL + COND + ['Log_D'] and not c.startswith('fp_') and not c.startswith('embedding_')]
acid = pd.get_dummies(df['Acid_type'].astype(str), prefix='acid').astype(float)
def san(M):
    M = M.astype(np.float64).copy(); b = ~np.isfinite(M) | (np.abs(M) > 1e30)
    md = np.nan_to_num(np.nanmedian(np.where(b, np.nan, M), axis=0)); ix = np.where(b); M[ix] = np.take(md, ix[1])
    for j in np.where(np.nanmax(np.abs(M), axis=0) > 1e7)[0]: M[:, j] = np.sign(M[:, j]) * np.log1p(np.abs(M[:, j]))
    return M
X = san(pd.concat([df[FP + EXT + COND + METAL], acid], axis=1).values)  # his ECFP-centric features
logD = df['Log_D'].values.astype(float); smiles = df[SMI].astype(str).values
print(f"his feature set: {X.shape[1]} cols ({len(FP)} ECFP + {len(EXT)} descriptors + conditions + metal + acid)")
def zmodel(): return xgb.XGBRegressor(n_estimators=700, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.6, reg_lambda=1.5, random_state=SEED, n_jobs=-1, verbosity=0)
def oof_logD_and_err(groups):
    o = np.zeros(len(df))
    for ti, vi in GroupKFold(5).split(X, logD, groups): o[vi] = zmodel().fit(X[ti], logD[ti]).predict(X[vi])
    resid = np.abs(logD - o); e = np.zeros(len(df))
    for ti, vi in GroupKFold(5).split(X, resid, groups): e[vi] = zmodel().fit(X[ti], resid[ti]).predict(X[vi])
    return o, np.clip(e, 1e-6, None)
REG = {'known extractant': ckey.values, 'new extractant': smiles}
OO = {g: oof_logD_and_err(grp) for g, grp in REG.items()}
for g in REG: print(f"{g:>16}: his-model logD OOF R2={r2_score(logD,OO[g][0]):.3f}")
gk = df[[SMI, 'Acid_type']].astype(str).agg('|'.join, axis=1)
for c, r in [('Acid_conc_M', 2), ('Temperature_K', 0), ('Extractant_conc_M', 3)]: gk = gk + '|' + df[c].round(r).astype(str)
gkey = gk.values; metal = df['Metal'].astype(str).values; isf = np.array([sym(m) in FELEM for m in metal])
g2idx = {}
for idx, g in enumerate(gkey):
    if isf[idx]: g2idx.setdefault(g, []).append(idx)
ii, jj = [], []
for g, idxs in g2idx.items():
    for a, b in itertools.combinations(idxs, 2):
        if metal[a] != metal[b]: ii.append(a); jj.append(b)
ii, jj = np.array(ii), np.array(jj); target = logD[ii] - logD[jj]
COV = [1.0, 0.5, 0.1]; rows = []
for g in REG:
    o, e = OO[g]; pred = o[ii] - o[jj]; unc = np.sqrt(e[ii]**2 + e[jj]**2); order = np.argsort(unc)
    for cov in COV:
        k = max(40, int(len(ii) * cov)); sel = order[:k]; t, p = target[sel], pred[sel]
        ya, yp = (np.abs(t) >= 1.0), (np.abs(p) >= 1.0)
        rows.append({'model': 'Zhang XGBoost', 'regime': g, 'coverage': f"{int(cov*100)}%", 'n': k,
                     'signed_R2': round(r2_score(t, p), 3), 'signed_RMSE': round(rmse(t, p), 3),
                     'norm_RMSE': round(rmse(t, p) / t.std(), 3), 'magnitude_R2': round(r2_score(np.abs(t), np.abs(p)), 3),
                     'direction_acc': round(float((np.sign(p) == np.sign(t)).mean()), 3),
                     'spearman': round(float(spearmanr(t, p).correlation), 3), 'useful_F1': round(float(f1_score(ya, yp, zero_division=0)), 3)})
res = pd.DataFrame(rows); res.to_csv("sep_factor_zhang_results.csv", index=False)
pd.set_option('display.width', 220)
print("\n=== Zhang's XGBoost model: separation factor between two f-elements (differenced) ===")
print(res.to_string(index=False))
print("\n--- our model, same task (for comparison) ---")
print(" known 100%: dir 0.726 signed_R2 0.356 spearman 0.601 | 10%: dir 0.789 spearman 0.776 norm_RMSE 0.640")
print("   new 100%: dir 0.656 signed_R2 0.188 spearman 0.462 | 10%: dir 0.586 norm_RMSE 1.002")
print(f"\nsaved sep_factor_zhang_results.csv | {(time.time()-START)/60:.1f} min")
