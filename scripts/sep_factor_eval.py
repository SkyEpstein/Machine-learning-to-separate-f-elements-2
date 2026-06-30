#!/usr/bin/env python3
"""
sep_factor_eval.py — evaluate the separation factor between two f-elements, i.e. how
well the model predicts |logD_i - logD_j| for two metals at the same extractant and
conditions. This is the project's actual objective. Separation is obtained by
differencing the logD model (the direct delta model was tested and lost). We report it
in two regimes and with a full metric suite, because |delta|-with-R2 alone understates a
model that orders pairs correctly:
  KNOWN extractant  : logD out-of-fold predictions from condition-key grouped CV (the
                      ligand is seen at other conditions; the exact row is held out).
  NEW extractant    : logD out-of-fold predictions from molecule-grouped CV (ligand unseen).
Metrics per regime: signed-delta R2/RMSE, magnitude |delta| R2/RMSE, direction accuracy
(which metal wins), Spearman on the signed delta, and threshold accuracy/F1 for flagging
a useful separation (|delta logD| >= 1, separation factor >= 10). Plus a per-pair table.
Restricted to f-element / f-element pairs (lanthanides and actinides).
"""
import os, time, warnings, itertools; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, re
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, f1_score
from scipy.stats import spearmanr
import lightgbm as lgb
SMI = 'SMILES_canonical'; START = time.time()
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
LAN = "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split()
ACT = "Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split()
FELEM = set(LAN + ACT)
def sym(m):
    x = re.match(r"\s*([A-Z][a-z]?)", str(m)); return x.group(1) if x else ""
tr = pd.read_csv("Training_Data_V27.csv", low_memory=False); te = pd.read_csv("Testing_Data_V39.csv", low_memory=False)
df = pd.concat([tr, te], ignore_index=True); df = df[df['Log_D'].notna() & df[SMI].notna()].reset_index(drop=True)
kc = df[[SMI, 'Metal', 'Acid_type']].astype(str).copy()
for c, r in [('Acid_conc_M', 2), ('Temperature_K', 0), ('Extractant_conc_M', 3), ('Metal_conc_mM', 4)]:
    if c in df: kc[c] = df[c].round(r).astype(str)
ckey_full = kc.agg('|'.join, axis=1)
grng = pd.Series(df['Log_D'].values).groupby(ckey_full).transform(lambda v: v.max() - v.min()).values
keep = (~df.duplicated().values) & (grng <= 2.0)
df = df[keep].reset_index(drop=True); ckey_full = ckey_full[keep].reset_index(drop=True)
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
def oof_for(groups):
    o = np.zeros(len(df))
    for ti, vi in GroupKFold(5).split(X, logD, groups): o[vi] = lgbm().fit(X[ti], logD[ti]).predict(X[vi])
    return o
oof = {'known extractant': oof_for(ckey_full.values), 'new extractant': oof_for(smiles)}
print(f"logD OOF R2  known={r2_score(logD,oof['known extractant']):.3f}  new={r2_score(logD,oof['new extractant']):.3f}")
# build all f-element / f-element pairs at matched (extractant, acid, conditions)
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
ii, jj = np.array(ii), np.array(jj)
target = logD[ii] - logD[jj]
print(f"f-element pairs: {len(ii)} across {len(set(smiles[ii]))} extractants, metals {sorted(set(metal[isf]))}")
def suite(regime):
    p = oof[regime][ii] - oof[regime][jj]
    out = {'regime': regime, 'signed_R2': round(r2_score(target, p), 3), 'signed_RMSE': round(rmse(target, p), 3),
           'magnitude_R2': round(r2_score(np.abs(target), np.abs(p)), 3), 'magnitude_RMSE': round(rmse(np.abs(target), np.abs(p)), 3),
           'direction_acc': round(float((np.sign(p) == np.sign(target)).mean()), 3),
           'spearman_signed': round(float(spearmanr(target, p).correlation), 3)}
    for thr in (0.5, 1.0):
        ya, yp = (np.abs(target) >= thr), (np.abs(p) >= thr)
        out[f'useful>={thr}_acc'] = round(float((ya == yp).mean()), 3)
        out[f'useful>={thr}_F1'] = round(float(f1_score(ya, yp, zero_division=0)), 3)
    return out
rows = [suite('known extractant'), suite('new extractant')]
res = pd.DataFrame(rows); res.to_csv("sep_factor_eval_results.csv", index=False)
print("\n=== separation factor between two f-elements (differencing the logD model) ===")
print(res.to_string(index=False))
# per-pair table (both regimes), unordered metal pair, n>=30
pl = ['/'.join(sorted((metal[a], metal[b]))) for a, b in zip(ii, jj)]
base = pd.DataFrame({'pair': pl, 'target': target, 'pk': oof['known extractant'][ii] - oof['known extractant'][jj], 'pn': oof['new extractant'][ii] - oof['new extractant'][jj]})
def bypair(d):
    return pd.Series({'n': len(d), 'mean_abs_delta': round(float(np.abs(d.target).mean()), 2),
                      'dir_acc_known': round(float((np.sign(d.pk) == np.sign(d.target)).mean()), 3),
                      'dir_acc_new': round(float((np.sign(d.pn) == np.sign(d.target)).mean()), 3),
                      'signed_R2_known': round(r2_score(d.target, d.pk), 2) if len(d) >= 30 else np.nan})
tm = base.groupby('pair').apply(bypair).reset_index()
tm = tm[tm.n >= 30].sort_values('dir_acc_known', ascending=False)
tm.to_csv("sep_factor_eval_by_pair.csv", index=False)
print(f"\nper-pair (n>=30): {len(tm)} f-element pairs")
print(tm.head(12).to_string(index=False))
print(f"\nsaved sep_factor_eval_results.csv, sep_factor_eval_by_pair.csv | {(time.time()-START)/60:.1f} min")
