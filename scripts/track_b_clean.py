#!/usr/bin/env python3
"""
track_b_clean.py — the audit showed ~17% of rows are discordant replicates that
inflate the noise floor (0.77 all -> 0.37 clean, ceiling ~0.94). Test whether
removing those unreliable LABELS lifts the models — especially Track B
(known-molecule), which was capped by that floor. Discordance is a LABEL-quality
criterion (replicate scatter), independent of the model -> honest cleaning.
"""
import os, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import Descriptors
SEED = 42
TARGET, SMI = "Log_D", "SMILES_canonical"
TEXT = {"Solvent_A", "Solvent_B", "Metal", "Acid_type", "SMILES_canonical"}
COND = ['Extractant_conc_M','Molar_mass(g/mol) A','Log_P A','Boiling_point(K) A','Melting_point(K) A','Density(g/mL) A','Solubility_in_water(g/L) A','Molar_mass(g/mol) B','Log_P B','Boiling_point(K) B','Melting_point(K) B','Density(g/mL) B','Solubility_in_water(g/L) B','Volume_fraction_A','Volume_fraction_B','Atomic_number','Melting_point_K','Boiling_point_K','Density_g/cm3','First_IE_kJ/mol','Second_IE_kJ/mol','Third_IE_kJ/mol','Matallic_radius_nm','Pauling_EN','Ionic_radius_nm','Oxidation_state','Metal_conc_mM','Dipole_moment_D','Acid_conc_M','Temperature_K']
tr = pd.read_csv("Training_Data_V27.csv", low_memory=False); te = pd.read_csv("Testing_Data_V39.csv", low_memory=False)
df = pd.concat([tr, te], ignore_index=True)
num = sorted(set(tr.select_dtypes(np.number).columns) & set(te.select_dtypes(np.number).columns))
allf = [c for c in num if c != TARGET and c not in TEXT]
df = df[df[allf + [TARGET]].notna().all(axis=1) & df[SMI].notna()].reset_index(drop=True)
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
Xfull = san(np.hstack([df[[c for c in COND if c in df]].values, df[fp].values, Lg]))
# discordance groups
kdf = df[[SMI, 'Metal'] + [c for c in ['Acid_type'] if c in df]].copy()
for c, r in [('Acid_conc_M', 2), ('Temperature_K', 0), ('Extractant_conc_M', 3), ('Metal_conc_mM', 4)]:
    if c in df: kdf[c] = df[c].round(r)
gid = kdf.astype(str).agg('|'.join, axis=1).values
grng = pd.Series(y).groupby(gid).transform(lambda v: v.max() - v.min()).values
def mk(): return lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=63, min_child_samples=10, subsample=0.85, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1)
def cv(mask, grouped, reps=2):
    idxall = np.where(mask)[0]; r = []
    for rep in range(reps):
        if grouped:
            uq = np.unique(smi[idxall]).copy(); np.random.RandomState(SEED + rep).shuffle(uq); fo = {m: i % 5 for i, m in enumerate(uq)}
            fold = np.array([fo[smi[i]] for i in idxall])
        else:
            fold = np.zeros(len(idxall), int)
            for i, (_, va) in enumerate(KFold(5, shuffle=True, random_state=SEED + rep).split(idxall)): fold[va] = i
        oof = np.zeros(len(idxall))
        for f in range(5):
            trp = np.where(fold != f)[0]; vap = np.where(fold == f)[0]
            oof[vap] = mk().fit(Xfull[idxall[trp]], y[idxall[trp]]).predict(Xfull[idxall[vap]])
        r.append(r2_score(y[idxall], oof))
    return np.mean(r), np.std(r)
print("Effect of removing discordant-replicate rows (cond+ECFP+ligand):\n", flush=True)
print(f"{'data':<26s} {'rows':>6s} {'KNOWN(random) R²':>18s} {'NEW(grouped) R²':>17s}", flush=True)
for thr, lab in [(np.inf, 'all data'), (3.0, 'drop range>3'), (2.0, 'drop range>2'), (1.5, 'drop range>1.5')]:
    mask = grng <= thr if np.isfinite(thr) else np.ones(len(y), bool)
    k = cv(mask, False); n = cv(mask, True)
    print(f"{lab:<26s} {int(mask.sum()):>6d} {k[0]:>12.4f}±{k[1]:.3f} {n[0]:>11.4f}±{n[1]:.3f}", flush=True)
print("\n(known-molecule clean ceiling ~0.94 from noise floor; was capped at ~0.77 by bad data)", flush=True)
