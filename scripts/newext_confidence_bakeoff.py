#!/usr/bin/env python3
"""
newext_confidence_bakeoff.py — bake-off of uncertainty estimators for the NEW-EXTRACTANT
regime on the separation task. All methods share the SAME molecule-grouped OOF logD
predictions and the SAME separation pairs; only the confidence RANKING of pairs differs, so
the comparison is clean. Every uncertainty is cross-fit on disjoint molecule groups (each
extractant lives entirely in one fold), so no extractant informs its own confidence.

Methods:
  err     : incumbent learned-error model, LightGBM(X -> |OOF logD residual|)
  ad_tan  : applicability domain, 1 - max ECFP4 Tanimoto to training-fold extractants
  desc_knn: mean distance to k=5 nearest training-fold extractants in standardized descriptor space
  ens_bag : bootstrap-bagged logD ensemble, std across 12 models
  rf_var  : Random Forest tree-variance (std across 300 trees)
  hybrid  : learned-error model with the AD + descriptor distances added as features

Winner = pre-registered COMPOSITE: average lift in the shrinkage-immune metrics (direction
accuracy + Spearman + useful-F1) at top-25% and top-10% coverage, plus calibration rho
(Spearman of predicted uncertainty vs actual |error|). We report both R2 and RMSE but do not
let selective R2/RMSE (which partly reflect variance shrinkage) drive the choice.
"""
import os, time, warnings, itertools; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, re
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, f1_score
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
import lightgbm as lgb
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
SMI = 'SMILES_canonical'; START = time.time()
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
FELEM = set("La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split())
def sym(m):
    x = re.match(r"\s*([A-Z][a-z]?)", str(m)); return x.group(1) if x else ""

# ---- data, cleaning, features (identical to sep_factor_confidence.py) ----
tr = pd.read_csv("Training_Data_V27.csv", low_memory=False); te = pd.read_csv("Testing_Data_V39.csv", low_memory=False)
df = pd.concat([tr, te], ignore_index=True); df = df[df['Log_D'].notna() & df[SMI].notna()].reset_index(drop=True)
kc = df[[SMI, 'Metal', 'Acid_type']].astype(str).copy()
for c, r in [('Acid_conc_M', 2), ('Temperature_K', 0), ('Extractant_conc_M', 3), ('Metal_conc_mM', 4)]:
    if c in df: kc[c] = df[c].round(r).astype(str)
ckey = kc.agg('|'.join, axis=1)
grng = pd.Series(df['Log_D'].values).groupby(ckey).transform(lambda v: v.max() - v.min()).values
keep = (~df.duplicated().values) & (grng <= 2.0)
df = df[keep].reset_index(drop=True)
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
Xext = san(df[EXT].values)                       # extractant-only descriptors, for descriptor distance
logD = df['Log_D'].values.astype(float); smiles = df[SMI].astype(str).values
def lgbm(seed=0, n=500): return lgb.LGBMRegressor(n_estimators=n, learning_rate=0.03, num_leaves=63, subsample=0.8, colsample_bytree=0.7, subsample_freq=1, random_state=seed, n_jobs=-1, verbosity=-1)

# ---- ECFP fingerprints per row (via unique smiles) ----
uniq = {s: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in set(smiles) if Chem.MolFromSmiles(s) is not None}
fps = [uniq.get(s) for s in smiles]

# ---- one fixed molecule-grouped split, shared by everything ----
folds = list(GroupKFold(5).split(X, logD, smiles))
o = np.zeros(len(df))                              # OOF logD (new-extractant regime)
for ti, vi in folds: o[vi] = lgbm().fit(X[ti], logD[ti]).predict(X[vi])
resid = np.abs(logD - o)

# ---- per-extractant applicability-domain distances, cross-fit ----
u_ad = np.zeros(len(df)); u_desc = np.zeros(len(df))
for ti, vi in folds:
    tr_s = list({smiles[i] for i in ti}); tr_fp = [uniq[s] for s in tr_s if s in uniq]
    # descriptor centroids per unique train extractant
    tr_ext = np.vstack([Xext[np.where(smiles == s)[0][0]] for s in tr_s])
    mu, sd = tr_ext.mean(0), tr_ext.std(0) + 1e-9
    tr_extz = (tr_ext - mu) / sd
    for i in vi:
        fp = fps[i]
        u_ad[i] = 1.0 - (max(DataStructs.BulkTanimotoSimilarity(fp, tr_fp)) if (fp is not None and tr_fp) else 0.0)
        z = (Xext[i] - mu) / sd
        d = np.sqrt(((tr_extz - z) ** 2).sum(1)); u_desc[i] = np.sort(d)[:5].mean()

# ---- ensemble disagreement (bootstrap bag) and RF tree variance, cross-fit ----
u_ens = np.zeros(len(df)); u_rf = np.zeros(len(df))
rng = np.random.RandomState(0)
for ti, vi in folds:
    preds = []
    for b in range(12):
        bi = ti[rng.randint(0, len(ti), len(ti))]
        preds.append(lgbm(seed=b, n=300).fit(X[bi], logD[bi]).predict(X[vi]))
    u_ens[vi] = np.std(np.column_stack(preds), axis=1)
    rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0).fit(X[ti], logD[ti])
    u_rf[vi] = np.std(np.column_stack([t.predict(X[vi]) for t in rf.estimators_]), axis=1)

# ---- learned-error incumbent + hybrid (error model + AD/desc features), cross-fit ----
def cross_error(feat):
    e = np.zeros(len(df))
    for ti, vi in folds: e[vi] = lgbm().fit(feat[ti], resid[ti]).predict(feat[vi])
    return np.clip(e, 1e-6, None)
u_err = cross_error(X)
u_hybrid = cross_error(np.hstack([X, u_ad[:, None], u_desc[:, None]]))

# ---- separation pairs (identical construction) ----
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
ii, jj = np.array(ii), np.array(jj); target = logD[ii] - logD[jj]; pred = o[ii] - o[jj]

METHODS = {  # per-endpoint arrays; combined per pair below
    'err (incumbent)': ('rowwise', u_err), 'hybrid': ('rowwise', u_hybrid),
    'ens_bag': ('rowwise', u_ens), 'rf_var': ('rowwise', u_rf),
    'ad_tanimoto': ('perext', u_ad), 'desc_knn': ('perext', u_desc),
}
def pair_unc(kind, e): return e[ii] if kind == 'perext' else np.sqrt(e[ii] ** 2 + e[jj] ** 2)

COV = [1.0, 0.75, 0.5, 0.25, 0.1]; rows = []
for name, (kind, e) in METHODS.items():
    unc = pair_unc(kind, e); order = np.argsort(unc); cal = spearmanr(unc, np.abs(pred - target)).correlation
    for cov in COV:
        k = max(40, int(len(ii) * cov)); sel = order[:k]; t, p = target[sel], pred[sel]
        ya, yp = (np.abs(t) >= 1.0), (np.abs(p) >= 1.0)
        rows.append({'method': name, 'coverage': f"{int(cov*100)}%", 'n': k,
                     'signed_R2': round(r2_score(t, p), 3), 'signed_RMSE': round(rmse(t, p), 3),
                     'direction_acc': round(float((np.sign(p) == np.sign(t)).mean()), 3),
                     'spearman': round(float(spearmanr(t, p).correlation), 3),
                     'useful_F1': round(float(f1_score(ya, yp, zero_division=0)), 3),
                     'calibration_rho': round(float(cal), 3)})
res = pd.DataFrame(rows)

# ---- composite winner: avg lift in direction+spearman+useful_F1 at 25/10 vs 100, + calibration ----
comp = []
for name in METHODS:
    d = res[res.method == name].set_index('coverage')
    base = d.loc['100%']; lift = 0.0
    for cov in ['25%', '10%']:
        lift += (d.loc[cov, 'direction_acc'] - base['direction_acc']) + (d.loc[cov, 'spearman'] - base['spearman']) + (d.loc[cov, 'useful_F1'] - base['useful_F1'])
    comp.append({'method': name, 'rank_lift': round(lift / 2, 4), 'calibration_rho': d.loc['100%', 'calibration_rho']})
comp = pd.DataFrame(comp)
z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
comp['composite'] = (z(comp.rank_lift) + z(comp.calibration_rho)) / 2
comp = comp.sort_values('composite', ascending=False).reset_index(drop=True)
winner = comp.iloc[0]['method']

res.to_csv("newext_confidence_bakeoff_results.csv", index=False)
comp.to_csv("newext_confidence_bakeoff_composite.csv", index=False)
pd.set_option('display.width', 220)
print("=== new-extractant regime: metric suite by method and coverage ===")
print(res.to_string(index=False))
print("\n=== composite ranking (winner metric: avg shrinkage-immune lift @25/10 + calibration) ===")
print(comp.to_string(index=False))
print(f"\nWINNER: {winner}")
print(f"\nsanity: incumbent 'err' at 100% should match the prior new-extractant row (signed_R2 0.188, direction 0.656)")

# ---- figure: direction accuracy vs coverage per method + calibration bars ----
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
covx = [100, 75, 50, 25, 10]
for name in METHODS:
    d = res[res.method == name].set_index('coverage')
    ax[0].plot(covx, [d.loc[f'{c}%', 'direction_acc'] for c in covx], marker='o', label=name, lw=2 if name == winner else 1, alpha=1 if name == winner else .7)
ax[0].invert_xaxis(); ax[0].set_xlabel('coverage (%, most-confident kept)'); ax[0].set_ylabel('direction accuracy')
ax[0].set_title('New-extractant separation: does confidence sharpen direction?\n(rising = good confidence signal)'); ax[0].legend(fontsize=7)
cb = comp.sort_values('calibration_rho')
ax[1].barh(range(len(cb)), cb.calibration_rho, color=['#d1495b' if m == winner else '#4c72b0' for m in cb.method])
ax[1].set_yticks(range(len(cb))); ax[1].set_yticklabels(cb.method, fontsize=8)
ax[1].set_xlabel('calibration rho (uncertainty vs actual |error|)'); ax[1].set_title('Calibration in the new-extractant regime')
plt.tight_layout()
out = "figures/newext_confidence_bakeoff.png"
fig.savefig(out, dpi=140, facecolor='white'); Image.open(out).convert('RGB').save(out)
print(f"\nsaved {out} | {(time.time()-START)/60:.1f} min")
