#!/usr/bin/env python3
"""
sep_factor_confidence.py — does separation-factor prediction improve with confidence?
Extends the f-element separation evaluation with a confidence dimension. In each regime
(known/new extractant) we cross-fit a learned-error model (LightGBM predicting the
absolute OOF logD residual from features, on the same disjoint groups, so no leakage).
A pair's uncertainty is sqrt(err_i^2 + err_j^2), the combined uncertainty of the two
endpoints of the difference. We rank pairs from most to least confident and recompute the
full metric suite at coverage 100/75/50/25/10 percent.

We report RMSE alongside R2 because part of any R2 rise at low coverage is the narrowing
variance of the retained subset; direction accuracy and the useful-separation F1 are not
subject to that shrinkage, so they are the clean test of whether confidence helps. We also
report the calibration: the Spearman correlation between a pair's predicted uncertainty and
its actual error |pred delta - actual delta|.
"""
import os, time, warnings, itertools; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, re
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, f1_score
from scipy.stats import spearmanr
import lightgbm as lgb
SMI = 'SMILES_canonical'; START = time.time()
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
EXT = [c for c in num if c not in METAL + COND + ['Log_D'] and not c.startswith('fp_') and not c.startswith('embedding_')]
acid = pd.get_dummies(df['Acid_type'].astype(str), prefix='acid').astype(float)
def san(M):
    M = M.astype(np.float64).copy(); b = ~np.isfinite(M) | (np.abs(M) > 1e30)
    md = np.nan_to_num(np.nanmedian(np.where(b, np.nan, M), axis=0)); ix = np.where(b); M[ix] = np.take(md, ix[1])
    for j in np.where(np.nanmax(np.abs(M), axis=0) > 1e7)[0]: M[:, j] = np.sign(M[:, j]) * np.log1p(np.abs(M[:, j]))
    return M
X = san(np.hstack([pd.concat([df[EXT + COND], acid], axis=1).values, san(df[METAL].values)]))
logD = df['Log_D'].values.astype(float); smiles = df[SMI].astype(str).values
def lgbm(): return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=63, subsample=0.8, colsample_bytree=0.7, subsample_freq=1, random_state=0, n_jobs=-1, verbosity=-1)
def oof_logD_and_err(groups):
    o = np.zeros(len(df))
    for ti, vi in GroupKFold(5).split(X, logD, groups): o[vi] = lgbm().fit(X[ti], logD[ti]).predict(X[vi])
    resid = np.abs(logD - o); e = np.zeros(len(df))
    for ti, vi in GroupKFold(5).split(X, resid, groups): e[vi] = lgbm().fit(X[ti], resid[ti]).predict(X[vi])
    return o, np.clip(e, 1e-6, None)
REG = {'known extractant': ckey.values, 'new extractant': smiles}
OO = {g: oof_logD_and_err(grp) for g, grp in REG.items()}
for g in REG: print(f"{g:>16}: logD OOF R2={r2_score(logD,OO[g][0]):.3f}")
# f-element pairs at matched conditions
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
COV = [1.0, 0.75, 0.5, 0.25, 0.1]; rows = []
for g in REG:
    o, e = OO[g]; pred = o[ii] - o[jj]; unc = np.sqrt(e[ii]**2 + e[jj]**2)
    order = np.argsort(unc); cal = spearmanr(unc, np.abs(pred - target)).correlation
    for cov in COV:
        k = max(40, int(len(ii) * cov)); sel = order[:k]
        t, p = target[sel], pred[sel]
        ya, yp = (np.abs(t) >= 1.0), (np.abs(p) >= 1.0)
        rows.append({'regime': g, 'coverage': f"{int(cov*100)}%", 'n': k,
                     'mean_abs_delta': round(float(np.abs(t).mean()), 3), 'target_std': round(float(t.std()), 3),
                     'signed_R2': round(r2_score(t, p), 3), 'signed_RMSE': round(rmse(t, p), 3),
                     'magnitude_R2': round(r2_score(np.abs(t), np.abs(p)), 3),
                     'direction_acc': round(float((np.sign(p) == np.sign(t)).mean()), 3),
                     'spearman': round(float(spearmanr(t, p).correlation), 3),
                     'useful_F1': round(float(f1_score(ya, yp, zero_division=0)), 3),
                     'unc_calibration_rho': round(float(cal), 3)})
pl = ['/'.join(sorted((metal[a], metal[b]))) for a, b in zip(ii, jj)]
ok, ek = OO['known extractant']; on, en = OO['new extractant']
pd.DataFrame({'pair': pl, 'target': target, 'pred_known': ok[ii] - ok[jj], 'unc_known': np.sqrt(ek[ii]**2 + ek[jj]**2),
              'pred_new': on[ii] - on[jj], 'unc_new': np.sqrt(en[ii]**2 + en[jj]**2)}).to_csv("sep_factor_confidence_pairs.csv", index=False)
res = pd.DataFrame(rows); res.to_csv("sep_factor_confidence_results.csv", index=False)
pd.set_option('display.width', 200)
print("\n=== separation factor vs confidence coverage (most-confident X% of f-element pairs) ===")
print(res.to_string(index=False))
print(f"\nsaved sep_factor_confidence_results.csv | {(time.time()-START)/60:.1f} min")
