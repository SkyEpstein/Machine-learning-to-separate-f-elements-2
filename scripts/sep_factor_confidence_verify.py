#!/usr/bin/env python3
"""
sep_factor_confidence_verify.py — committed, reproducible verification that confidence
genuinely improves separation prediction for a known extractant (and not for a new one),
beyond the gap-selection confound and variance shrinkage. Recomputes the OOF logD and the
cross-fitted error model from raw data (same pipeline as sep_factor_confidence.py), builds
the f-element pairs, then runs three controls and writes sep_factor_confidence_verify_results.csv:
  1. within-true-gap-bin direction advantage: split each true-|delta| quartile into a
     confident and an unconfident half (by pair uncertainty) and compare direction accuracy;
     report per-bin and the gap-balanced mean (controls the gap-selection confound).
  2. paired bootstrap of [direction acc on most-confident 10%] minus [direction acc overall],
     resampled on the same draw to honor nesting; report mean, 95% CI, and p.
  3. random-ranking shrinkage benchmark: normalized RMSE (RMSE / target spread) of the
     most-confident 10% vs the mean over random 10% subsets (controls variance shrinkage).
"""
import os, time, warnings, itertools; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, re
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
SMI = 'SMILES_canonical'; START = time.time(); rng = np.random.default_rng(0)
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
def oof(groups):
    o = np.zeros(len(df))
    for ti, vi in GroupKFold(5).split(X, logD, groups): o[vi] = lgbm().fit(X[ti], logD[ti]).predict(X[vi])
    resid = np.abs(logD - o); e = np.zeros(len(df))
    for ti, vi in GroupKFold(5).split(X, resid, groups): e[vi] = lgbm().fit(X[ti], resid[ti]).predict(X[vi])
    return o, np.clip(e, 1e-6, None)
REG = {'known extractant': ckey.values, 'new extractant': smiles}
OO = {g: oof(grp) for g, grp in REG.items()}
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
ii, jj = np.array(ii), np.array(jj); target = logD[ii] - logD[jj]; absd = np.abs(target)
rows = []
for g in REG:
    o, e = OO[g]; pred = o[ii] - o[jj]; unc = np.sqrt(e[ii]**2 + e[jj]**2)
    correct = (np.sign(pred) == np.sign(target)).astype(float)
    # 1. within-gap-bin confident-vs-unconfident direction advantage (quartiles of true |delta|)
    q = np.quantile(absd, [0.25, 0.5, 0.75]); binid = np.digitize(absd, q)
    perbin = []
    for b in range(4):
        m = binid == b
        if m.sum() < 8: perbin.append(np.nan); continue
        umed = np.median(unc[m]); conf = m & (unc <= umed); unco = m & (unc > umed)
        perbin.append(correct[conf].mean() - correct[unco].mean())
    gap_balanced = float(np.nanmean(perbin))
    # 2. paired bootstrap of dir_acc(top-10%) - dir_acc(all)
    k = max(40, int(0.1 * len(ii))); diffs = []
    for _ in range(5000):
        s = rng.integers(0, len(ii), len(ii))
        cs, ps = correct[s], unc[s]
        thr = np.partition(ps, k)[k]; top = ps <= thr
        diffs.append(cs[top].mean() - cs.mean())
    diffs = np.array(diffs); pval = float((diffs <= 0).mean())
    # 3. random-ranking shrinkage benchmark on normalized RMSE at 10%
    order = np.argsort(unc)[:k]; t10 = target[order]; p10 = pred[order]
    norm_conf = rmse(t10, p10) / t10.std()
    randnorm = []
    for _ in range(200):
        s = rng.permutation(len(ii))[:k]; randnorm.append(rmse(target[s], pred[s]) / target[s].std())
    rows.append({'regime': g,
                 'gap_balanced_dir_advantage': round(gap_balanced, 3),
                 'per_bin_dir_advantage': ';'.join(f'{x:.3f}' for x in perbin),
                 'bootstrap_top10_minus_all_dir': round(float(diffs.mean()), 3),
                 'bootstrap_CI95_low': round(float(np.quantile(diffs, 0.025)), 3),
                 'bootstrap_CI95_high': round(float(np.quantile(diffs, 0.975)), 3),
                 'bootstrap_p': round(pval, 4),
                 'normRMSE_conf_top10': round(norm_conf, 3),
                 'normRMSE_random_top10': round(float(np.mean(randnorm)), 3),
                 'confidence_beats_random': bool(norm_conf < np.mean(randnorm))})
res = pd.DataFrame(rows); res.to_csv("sep_factor_confidence_verify_results.csv", index=False)
pd.set_option('display.width', 220)
print(res.to_string(index=False))
print(f"\nsaved sep_factor_confidence_verify_results.csv | {(time.time()-START)/60:.1f} min")
