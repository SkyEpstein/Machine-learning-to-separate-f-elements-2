#!/usr/bin/env python3
"""
tabpfn_in_stack.py — the DECISIVE TabPFN test: does a local-TabPFN expert earn
NNLS weight INSIDE the tree-stack? Generates OOF for the tree members
(lgb/xgb/et/cat) and for local TabPFN (one per <=1000-row k-means region, its
sweet spot), then NNLS-stacks (a) trees-only vs (b) trees + TabPFN.

Decision rule (the user's "drop it if it's not the best"): if TabPFN's NNLS
weight ~0 and ΔR² ~0, it does NOT belong in the final stack; if it earns weight
and lifts R², it goes in. Track B (known-molecule, random CV, cleaned) — the only
track with headroom (Track A is already at its oracle ceiling). Also reports
error-correlation(tree-stack, TabPFN): the lower it is, the more diversity TabPFN
brings, which is its only path to earning weight. device cpu (GPU=mps to try MPS).
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
os.environ.pop("TABPFN_TOKEN", None)
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
import lightgbm as lgb, xgboost as xgb
from rdkit import Chem
from rdkit.Chem import Descriptors
try:
    from catboost import CatBoostRegressor; CAT = True
except Exception: CAT = False
from tabpfn import TabPFNRegressor
SEED = 42; START = time.time(); DEV = os.environ.get("GPU", "cpu"); KREG = int(os.environ.get("KREG", "10")); CAP = int(os.environ.get("CAP", "1000")); NFOLD = int(os.environ.get("NFOLD", "5"))
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
Xtree = san(np.hstack([df[[c for c in COND if c in df]].values, df[fp].values, Lg]))          # tree members: full cond+ECFP+ligand
pcaf = PCA(50, random_state=SEED).fit_transform(StandardScaler().fit_transform(san(df[fp].values)))
Xt = san(np.hstack([df[[c for c in COND if c in df]].values, pcaf, Lg]))                        # TabPFN: cond+PCA(ECFP,50)+ligand
region = KMeans(KREG, random_state=SEED, n_init=10).fit_predict(StandardScaler().fit_transform(Xt))
print(f"cleaned {len(y)} rows | regions={KREG} sizes={np.bincount(region).tolist()} | TabPFN device={DEV}", flush=True)
def trees():
    m = {'lgb': lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=63, min_child_samples=12, random_state=SEED, n_jobs=-1, verbosity=-1),
         'xgb': xgb.XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=7, subsample=0.9, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=0),
         'et': ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.6, random_state=SEED, n_jobs=-1)}
    if CAT: m['cat'] = CatBoostRegressor(iterations=2500, learning_rate=0.03, depth=8, random_state=SEED, verbose=False, allow_writing_files=False)
    return m
tnames = list(trees().keys())
fold = np.zeros(len(y), int)
for i, (_, va) in enumerate(KFold(NFOLD, shuffle=True, random_state=SEED).split(y)): fold[va] = i
oof = {n: np.zeros(len(y)) for n in tnames}; tab = np.zeros(len(y)); rng = np.random.RandomState(SEED)
for f in range(NFOLD):
    t0 = time.time(); trr = np.where(fold != f)[0]; va = np.where(fold == f)[0]
    for n, mdl in trees().items(): oof[n][va] = mdl.fit(Xtree[trr], y[trr]).predict(Xtree[va])
    g = oof['lgb'][va]  # fallback for tiny regions = LightGBM pred
    for rg in range(KREG):
        rtr = trr[region[trr] == rg]; rva = va[region[va] == rg]
        if len(rva) == 0: continue
        pos = np.searchsorted(va, rva)
        if len(rtr) < 50: tab[rva] = g[pos]; continue
        if len(rtr) > CAP: rtr = rng.choice(rtr, CAP, replace=False)
        try:
            m = TabPFNRegressor(device=DEV, ignore_pretraining_limits=True); m.fit(Xt[rtr], y[rtr]); tab[rva] = m.predict(Xt[rva])
        except Exception as e:
            tab[rva] = g[pos]
            if rg == 0 and f == 0: print(f"  [tabpfn fallback] {repr(e)[:150]}", flush=True)
    print(f"  fold {f+1}/{NFOLD} ({(time.time()-t0):.0f}s)", flush=True)
# per-member R²
print("\n=== Track B (known-molecule, random CV, cleaned) — per-member OOF R² ===", flush=True)
for n in tnames: print(f"  {n:<12s} R²={r2_score(y, oof[n]):.4f}", flush=True)
print(f"  {'tabpfn_local':<12s} R²={r2_score(y, tab):.4f}", flush=True)
def nnls(cols):
    P = np.column_stack(cols); lr = LinearRegression(positive=True, fit_intercept=False).fit(P, y)
    w = np.clip(lr.coef_, 0, None); w /= (w.sum() + 1e-12); return P @ w, w
tcols = [oof[n] for n in tnames]
pred_t, wt = nnls(tcols); pred_tp, wtp = nnls(tcols + [tab])
# diversity: error-correlation between the tree-stack and TabPFN
ec = np.corrcoef(y - pred_t, y - tab)[0, 1]
print(f"\n  error-correlation(tree-stack, TabPFN) = {ec:.2f}   (LOW => more diversity => more room to earn weight)", flush=True)
print(f"\n  NNLS trees-only        R²={r2_score(y, pred_t):.4f}   weights " + ", ".join(f"{n}={wt[i]:.2f}" for i, n in enumerate(tnames)), flush=True)
print(f"  NNLS trees + TabPFN    R²={r2_score(y, pred_tp):.4f}   weights " + ", ".join(f"{n}={wtp[i]:.2f}" for i, n in enumerate(tnames)) + f", tabpfn={wtp[-1]:.2f}", flush=True)
print(f"\n  VERDICT: TabPFN weight in stack = {wtp[-1]:.2f}; ΔR² (with TabPFN − trees-only) = {r2_score(y, pred_tp) - r2_score(y, pred_t):+.4f}", flush=True)
print(f"  -> {'INCLUDE TabPFN' if (wtp[-1] > 0.03 and r2_score(y, pred_tp) - r2_score(y, pred_t) > 0.002) else 'DROP TabPFN (does not earn its place)'}", flush=True)
print(f"\ntotal {(time.time()-START)/60:.1f} min", flush=True)
