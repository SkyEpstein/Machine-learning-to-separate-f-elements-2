#!/usr/bin/env python3
"""
metal_confidence.py — how prediction accuracy and confidence vary by metal and by
metal pair (the separation use case). Loads the saved OOF (conf_oof_B.npz, the
known-molecule random-CV stack) and re-derives the Metal and condition columns by
re-running the same cleaning, so rows line up. Computes the err_lgb confidence with
the chosen recipe, then:
  (1) per-metal: n, R^2, RMSE, median confidence, and top-25% confident R^2/RMSE
  (2) per-metal-pair separation: among rows sharing the same extractant and solution
      conditions but a different metal, predicted dlogD vs actual dlogD, with R^2,
      RMSE, and mean pair confidence.
Both metrics (R^2 and RMSE) are always reported together. Saves two CSVs for the
results workbook.
"""
import os, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from itertools import combinations
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import Descriptors
SEED = 42
TARGET, SMI = "Log_D", "SMILES_canonical"
TEXT = {"Solvent_A", "Solvent_B", "Metal", "Acid_type", "SMILES_canonical"}
COND = ['Extractant_conc_M','Molar_mass(g/mol) A','Log_P A','Boiling_point(K) A','Melting_point(K) A','Density(g/mL) A','Solubility_in_water(g/L) A','Molar_mass(g/mol) B','Log_P B','Boiling_point(K) B','Melting_point(K) B','Density(g/mL) B','Solubility_in_water(g/L) B','Volume_fraction_A','Volume_fraction_B','Atomic_number','Melting_point_K','Boiling_point_K','Density_g/cm3','First_IE_kJ/mol','Second_IE_kJ/mol','Third_IE_kJ/mol','Matallic_radius_nm','Pauling_EN','Ionic_radius_nm','Oxidation_state','Metal_conc_mM','Dipole_moment_D','Acid_conc_M','Temperature_K']

def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))

def load_clean():
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
    return df

d = np.load("conf_oof_B.npz", allow_pickle=True)
y, pred, Xcond, fold = d['y'], d['stack'], d['Xcond'], d['fold']
df = load_clean()
assert len(df) == len(y), f"row mismatch {len(df)} vs {len(y)}"
metal = df['Metal'].astype(str).values

# confidence: plain + regularized err_lgb (best stack recipe from confidence_tune), 5-fold OOF
REG = lambda: lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, random_state=SEED, n_jobs=-1, verbosity=-1)
Cf = np.column_stack([Xcond, pred]); res = np.abs(y - pred); err = np.zeros(len(y))
for f in range(5):
    trr = np.where(fold != f)[0]; va = np.where(fold == f)[0]
    err[va] = REG().fit(Cf[trr], res[trr]).predict(Cf[va])
err = np.clip(err, 0.05, None)
print(f"loaded {len(y)} rows | overall stack R^2={r2_score(y,pred):.4f} RMSE={rmse(y,pred):.4f}", flush=True)

# (1) per-metal
rows = []
for m in sorted(pd.unique(metal)):
    ix = np.where(metal == m)[0]
    if len(ix) < 20: continue
    e = err[ix]; top = ix[e <= np.percentile(e, 25)]
    rows.append({'Metal': m, 'n': len(ix),
                 'R2': round(r2_score(y[ix], pred[ix]), 3), 'RMSE': round(rmse(y[ix], pred[ix]), 3),
                 'median_confidence_err': round(float(np.median(e)), 3),
                 'top25_R2': round(r2_score(y[top], pred[top]), 3) if len(top) > 8 else np.nan,
                 'top25_RMSE': round(rmse(y[top], pred[top]), 3) if len(top) > 8 else np.nan})
by_metal = pd.DataFrame(rows).sort_values('RMSE')
by_metal.to_csv("metal_confidence_by_metal.csv", index=False)
print("\n(1) PER-METAL (sorted by RMSE, both metrics shown):", flush=True)
print(by_metal.to_string(index=False), flush=True)

# (2) per-metal-pair separation: match rows on extractant + solution conditions, differ in metal
key = (df[SMI].astype(str) + '|' + df['Acid_conc_M'].round(2).astype(str) + '|' + df['Temperature_K'].round(0).astype(str)
       + '|' + df['Extractant_conc_M'].round(3).astype(str)).values
pred_sep, true_sep, pair_conf, pair_name = [], [], [], []
order = np.argsort(key)
ksort = key[order]
bounds = np.where(ksort[1:] != ksort[:-1])[0] + 1
for grp in np.split(order, bounds):
    if len(grp) < 2: continue
    for i, j in combinations(grp, 2):
        if metal[i] == metal[j]: continue
        a, b = (i, j) if metal[i] < metal[j] else (j, i)
        pred_sep.append(pred[a] - pred[b]); true_sep.append(y[a] - y[b])
        pair_conf.append(max(err[a], err[b])); pair_name.append(f"{metal[a]}/{metal[b]}")
sep = pd.DataFrame({'pair': pair_name, 'pred_sep': pred_sep, 'true_sep': true_sep, 'conf_err': pair_conf})
print(f"\n(2) SEPARATION (metal pairs): {len(sep)} matched pairs across {sep['pair'].nunique()} distinct metal pairs", flush=True)
if len(sep) > 20:
    print(f"  overall separation prediction: R^2={r2_score(sep.true_sep, sep.pred_sep):.3f}  RMSE={rmse(sep.true_sep, sep.pred_sep):.3f}", flush=True)
prows = []
for p, g in sep.groupby('pair'):
    if len(g) < 15: continue
    prows.append({'metal_pair': p, 'n': len(g),
                  'sep_R2': round(r2_score(g.true_sep, g.pred_sep), 3), 'sep_RMSE': round(rmse(g.true_sep, g.pred_sep), 3),
                  'mean_conf_err': round(float(g.conf_err.mean()), 3)})
by_pair = pd.DataFrame(prows).sort_values('n', ascending=False)
by_pair.to_csv("metal_confidence_by_pair.csv", index=False)
print("  top metal pairs by count (both metrics shown):", flush=True)
print(by_pair.head(20).to_string(index=False), flush=True)
print("\nsaved metal_confidence_by_metal.csv and metal_confidence_by_pair.csv", flush=True)
