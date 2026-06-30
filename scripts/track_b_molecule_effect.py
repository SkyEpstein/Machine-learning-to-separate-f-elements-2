#!/usr/bin/env python3
"""
track_b_molecule_effect.py — push Track B (known-molecule) toward ~0.77 with the
lever feature/ensemble tuning can't provide: an EXPLICIT per-molecule effect.
Since the molecule is in training (known-molecule task), use its own measured rows
to calibrate it (smoothed target encoding of the conditions-model residual) —
much more direct than letting ECFP infer molecule identity.

random-row 5-fold CV (the known-molecule task), honest target encoding (molecule
effect from TRAIN rows only). Compare:
  M1 cond+ECFP+ligand            (current best, ~0.65)
  M2 cond+ECFP+ligand + mol_TE   (add explicit molecule effect)
  M3 cond+metal base + mol_TE    (two-stage additive)
  ORACLE base + molecule mean residual from ALL rows (leaky upper bound, ~ceiling)
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import Descriptors
SEED = 42; START = time.time()
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
Xcond = san(df[[c for c in COND if c in df]].values)
Xstruct = san(np.hstack([df[[c for c in COND if c in df]].values, df[fp].values, Lg]))
def mk(): return lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=63, min_child_samples=10, subsample=0.85, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1)
def rfolds(rep):
    f = np.zeros(len(y), int)
    for i, (_, va) in enumerate(KFold(5, shuffle=True, random_state=SEED + rep).split(y)): f[va] = i
    return f
K_SMOOTH = 5
def mol_te(resid, mol, tr_idx, te_idx):
    """smoothed per-molecule mean residual from TRAIN rows only."""
    g = pd.Series(resid[tr_idx]).groupby(mol[tr_idx]); s = g.sum(); c = g.count()
    te_val = {m: s[m] / (c[m] + K_SMOOTH) for m in s.index}
    return np.array([te_val.get(mol[i], 0.0) for i in te_idx])

res = {'M1 cond+ECFP+ligand': [], 'M2 +molecule_TE': [], 'M3 cond+metal+molecule_TE': [], 'ORACLE (leaky)': []}
oof_store = {}
for rep in range(2):
    fold = rfolds(rep)
    m1 = np.zeros(len(y)); m2 = np.zeros(len(y)); m3 = np.zeros(len(y)); orc = np.zeros(len(y))
    for fdi in range(5):
        trr = np.where(fold != fdi)[0]; va = np.where(fold == fdi)[0]
        # conditions+metal base (for residual + TE + M3)
        base = mk().fit(Xcond[trr], y[trr]); base_tr = base.predict(Xcond[trr]); base_va = base.predict(Xcond[va])
        resid_tr_full = np.zeros(len(y)); resid_tr_full[trr] = y[trr] - base_tr
        te_va = mol_te(resid_tr_full, smi, trr, va)
        # M1: structure model
        m1[va] = mk().fit(Xstruct[trr], y[trr]).predict(Xstruct[va])
        # M2: structure + molecule_TE feature
        te_tr = mol_te(resid_tr_full, smi, trr, trr)
        X2tr = np.column_stack([Xstruct[trr], te_tr]); X2va = np.column_stack([Xstruct[va], te_va])
        m2[va] = mk().fit(X2tr, y[trr]).predict(X2va)
        # M3: base + molecule_TE (additive)
        m3[va] = base_va + te_va
        # ORACLE: base + molecule mean residual from ALL rows (leaky)
        allres = pd.Series(y - np.concatenate([base.predict(Xcond)])).groupby(smi).transform('mean').values  # approx
    res['M1 cond+ECFP+ligand'].append(r2_score(y, m1)); res['M2 +molecule_TE'].append(r2_score(y, m2))
    res['M3 cond+metal+molecule_TE'].append(r2_score(y, m3))
    # honest-ish oracle over full (each molecule mean residual vs a full-data base)
    bfull = mk().fit(Xcond, y).predict(Xcond); orc = bfull + pd.Series(y - bfull).groupby(smi).transform('mean').values
    res['ORACLE (leaky)'].append(r2_score(y, orc))
    if rep == 0: oof_store = dict(m2=m2)
print("=== TRACK B + per-molecule effect (random-row CV; current 0.65, ceiling ~0.77) ===", flush=True)
for k, v in res.items():
    print(f"  {k:<28s} R²={np.mean(v):.4f}±{np.std(v):.3f}", flush=True)
# confidence on M2
m2 = oof_store['m2']; resid = np.abs(y - m2); fold = rfolds(0); conf = np.zeros(len(y)); Cf = np.column_stack([Xcond, m2])
for fdi in range(5):
    trr = np.where(fold != fdi)[0]; va = np.where(fold == fdi)[0]; conf[va] = mk().fit(Cf[trr], resid[trr]).predict(Cf[va])
print("\n  confidence-filtered (M2 + err_lgb):", flush=True)
for p in [100, 50, 25, 10, 5]:
    m = np.ones(len(y), bool) if p == 100 else (conf <= np.percentile(conf, p))
    print(f"    top {p:>3d}%  R²={r2_score(y[m],m2[m]):.4f}  RMSE={np.sqrt(mean_squared_error(y[m],m2[m])):.4f}", flush=True)
print(f"\ndone | {(time.time()-START)/60:.1f} min", flush=True)
