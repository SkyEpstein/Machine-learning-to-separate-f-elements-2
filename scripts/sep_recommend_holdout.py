#!/usr/bin/env python3
"""
sep_recommend_holdout.py — the honest prospective test of the separation recommender:
leave extractants out, then vet them as if they were new. Every extractant is held out
once (molecule-grouped 5-fold), so its logD is predicted by a model that never saw it.
For each target f-element pair we then rank the held-out extractants by predicted
separation and ask whether that ranking recovers the ones that actually separate best.

This answers the deployment question directly: if we use this to vet a brand-new extractant
before any lab work, how well does the ranking hold up? Reports, per pair and pooled,
Spearman and direction of predicted vs measured separation across held-out extractants,
precision@top-25% and its enrichment over chance, and the same restricted to the half the
model is most confident about (the confidence companion).
"""
import os, time, warnings, itertools; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, re
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
import lightgbm as lgb
SMI = 'SMILES_canonical'; START = time.time()
FELEM = set("La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split())
def sym(m):
    x = re.match(r"\s*([A-Z][a-z]?)", str(m)); return x.group(1) if x else ""
tr = pd.read_csv("Training_Data_V27.csv", low_memory=False); te = pd.read_csv("Testing_Data_V39.csv", low_memory=False)
df = pd.concat([tr, te], ignore_index=True); df = df[df['Log_D'].notna() & df[SMI].notna()].reset_index(drop=True)
kc = df[[SMI, 'Metal', 'Acid_type']].astype(str).copy()
for c, r in [('Acid_conc_M', 2), ('Temperature_K', 0), ('Extractant_conc_M', 3), ('Metal_conc_mM', 4)]: kc[c] = df[c].round(r).astype(str)
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
X = san(np.hstack([pd.concat([df[EXT + COND], acid], axis=1).values, san(df[METAL].values)]))
logD = df['Log_D'].values.astype(float); smiles = df[SMI].astype(str).values
def lgbm(): return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63, subsample=0.8, colsample_bytree=0.7, subsample_freq=1, random_state=0, n_jobs=-1, verbosity=-1)
# leave extractants out: molecule-grouped OOF logD + cross-fitted error (each extractant vetted as novel)
oof = np.zeros(len(df))
for ti, vi in GroupKFold(5).split(X, logD, smiles): oof[vi] = lgbm().fit(X[ti], logD[ti]).predict(X[vi])
resid = np.abs(logD - oof); err = np.zeros(len(df))
for ti, vi in GroupKFold(5).split(X, resid, smiles): err[vi] = lgbm().fit(X[ti], resid[ti]).predict(X[vi])
err = np.clip(err, 1e-6, None)
print(f"held-out (molecule-grouped) logD OOF R2={1 - np.var(logD - oof)/np.var(logD):.3f} | {(time.time()-START)/60:.1f} min")
# build f-element pairs at matched conditions
gk = df[[SMI, 'Acid_type']].astype(str).agg('|'.join, axis=1)
for c, r in [('Acid_conc_M', 2), ('Temperature_K', 0), ('Extractant_conc_M', 3)]: gk = gk + '|' + df[c].round(r).astype(str)
gkey = gk.values; metal = df['Metal'].astype(str).values; isf = np.array([sym(m) in FELEM for m in metal])
g2idx = {}
for idx, g in enumerate(gkey):
    if isf[idx]: g2idx.setdefault(g, []).append(idx)
recs = []
for g, idxs in g2idx.items():
    for a, b in itertools.combinations(idxs, 2):
        if metal[a] != metal[b]:
            hi, lo = (a, b) if metal[a] < metal[b] else (b, a)  # order by metal name for a consistent pair label
            recs.append((f'{metal[hi]}/{metal[lo]}', smiles[hi], logD[hi]-logD[lo], oof[hi]-oof[lo], np.sqrt(err[hi]**2+err[lo]**2)))
P = pd.DataFrame(recs, columns=['pair', 'ext', 'meas', 'pred', 'unc'])
# aggregate to one value per (pair, extractant): mean over its matched systems = its separation for that pair
E = P.groupby(['pair', 'ext']).agg(meas=('meas', 'mean'), pred=('pred', 'mean'), unc=('unc', 'mean'), n=('meas', 'size')).reset_index()
def prec_at(m, p, frac=0.25):
    k = max(2, int(np.ceil(frac * len(m))))
    top_p = set(np.argsort(-np.abs(p))[:k]); top_m = set(np.argsort(-np.abs(m))[:k])
    return len(top_p & top_m) / k
rows = []
for pair, d in E.groupby('pair'):
    if len(d) < 8: continue
    m, p, u = d['meas'].values, d['pred'].values, d['unc'].values
    conf = u <= np.median(u)  # most-confident half of held-out extractants
    rows.append({'pair': pair, 'n_ext': len(d),
                 'dir_acc': round(float((np.sign(p) == np.sign(m)).mean()), 3),
                 'spearman_absep': round(float(spearmanr(np.abs(m), np.abs(p)).correlation), 3),
                 'prec@25%': round(prec_at(m, p), 3), 'enrichment': round(prec_at(m, p) / 0.25, 2),
                 'dir_acc_conf50': round(float((np.sign(p[conf]) == np.sign(m[conf])).mean()), 3),
                 'spearman_conf50': round(float(spearmanr(np.abs(m[conf]), np.abs(p[conf])).correlation), 3)})
R = pd.DataFrame(rows).sort_values('n_ext', ascending=False)
# pooled across pairs (standardize |sep| within pair so pairs are comparable, then pool)
E2 = E.groupby('pair').filter(lambda d: len(d) >= 8).copy()
E2['zpred'] = E2.groupby('pair')['pred'].transform(lambda s: (np.abs(s) - np.abs(s).mean()) / (np.abs(s).std() + 1e-9))
E2['zmeas'] = E2.groupby('pair')['meas'].transform(lambda s: (np.abs(s) - np.abs(s).mean()) / (np.abs(s).std() + 1e-9))
pooled_rho = spearmanr(E2['zmeas'], E2['zpred']).correlation
pooled_dir = float((np.sign(E2['pred']) == np.sign(E2['meas'])).mean())
cf = E2['unc'] <= E2.groupby('pair')['unc'].transform('median')
pooled_rho_c = spearmanr(E2.loc[cf, 'zmeas'], E2.loc[cf, 'zpred']).correlation
pooled_dir_c = float((np.sign(E2.loc[cf, 'pred']) == np.sign(E2.loc[cf, 'meas'])).mean())
R.to_csv("sep_recommend_holdout_results.csv", index=False)
pd.set_option('display.width', 200)
print("\n=== held-out-extractant vetting: rank NOVEL extractants by predicted separation (per pair, n_ext>=8) ===")
print(R.to_string(index=False))
print(f"\nPOOLED across {E2['pair'].nunique()} pairs / {len(E2)} held-out (pair,extractant) cases:")
print(f"  all:        direction {pooled_dir:.3f}, |sep| Spearman {pooled_rho:.3f}")
print(f"  confident half: direction {pooled_dir_c:.3f}, |sep| Spearman {pooled_rho_c:.3f}")
print(f"\nsaved sep_recommend_holdout_results.csv | {(time.time()-START)/60:.1f} min")
