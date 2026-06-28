#!/usr/bin/env python3
"""
baseline_v39.py — Honest descriptor+tree baseline on the chemistry-disjoint
V27/V39 split with MOLECULE-GROUPED CV. This is the real number-to-beat on the
583-row V39 test set (v55's 0.60 is on the different V37 split — not comparable).

Also answers "which features matter": v55's 0.60 used ECFP+conditions only (no
RDKit, no embeddings), so we run the SAME ensemble on three feature sets:
  ALL      : embeddings(768) + ECFP(795) + RDKit(~165) + conditions/metal(~30)
  NO_EMB   : ECFP + RDKit + conditions         (drop the 768 learned embeddings)
  FP_COND  : ECFP + conditions                 (the V26-like set, on V39 molecules)

Models: XGB, LGB, HistGBM, ExtraTrees, RandomForest. Trees are scale-invariant
so no scaling. Weights = squared MOLECULE-GROUPED validation R² (no molecule
straddles train/val -> honest weights, unlike random-row CV which over-credits
flexible models at ~30 rows/molecule). Final fit on all train rows.
"""
import os, time, warnings
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingRegressor, ExtraTreesRegressor,
                              RandomForestRegressor)
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

TRAIN  = "Training_Data_V27.csv"
TEST   = "Testing_Data_V39.csv"
TARGET = "Log_D"
SMI    = "SMILES_canonical"
TEXT   = {"Solvent_A", "Solvent_B", "Metal", "Acid_type", "SMILES_canonical"}
SEED   = 42
np.random.seed(SEED)
START  = time.time()

# Condition / solvent / metal-property numeric columns (the V26 "other" set).
COND_COLS = ['Extractant_conc_M', 'Molar_mass(g/mol) A', 'Log_P A', 'Boiling_point(K) A',
    'Melting_point(K) A', 'Density(g/mL) A', 'Solubility_in_water(g/L) A',
    'Molar_mass(g/mol) B', 'Log_P B', 'Boiling_point(K) B', 'Melting_point(K) B',
    'Density(g/mL) B', 'Solubility_in_water(g/L) B', 'Volume_fraction_A', 'Volume_fraction_B',
    'Atomic_number', 'Melting_point_K', 'Boiling_point_K', 'Density_g/cm3', 'First_IE_kJ/mol',
    'Second_IE_kJ/mol', 'Third_IE_kJ/mol', 'Matallic_radius_nm', 'Pauling_EN',
    'Ionic_radius_nm', 'Oxidation_state', 'Metal_conc_mM', 'Dipole_moment_D',
    'Acid_conc_M', 'Temperature_K']

# ----------------------------------------------------------------- load + clean
tr = pd.read_csv(TRAIN, low_memory=False)
te = pd.read_csv(TEST,  low_memory=False)
num = sorted(set(tr.select_dtypes(np.number).columns) & set(te.select_dtypes(np.number).columns))
allfeat = [c for c in num if c != TARGET and c not in TEXT]
m_tr = tr[allfeat + [TARGET]].notna().all(axis=1) & tr[SMI].notna()
m_te = te[allfeat + [TARGET]].notna().all(axis=1) & te[SMI].notna()
tr = tr[m_tr].reset_index(drop=True)
te = te[m_te].reset_index(drop=True)
smi_tr = tr[SMI].astype(str).values
smi_te = te[SMI].astype(str).values
y_tr = tr[TARGET].values.astype(np.float64)
y_te = te[TARGET].values.astype(np.float64)

# feature groups
emb  = [c for c in allfeat if c.startswith("embedding_")]
fp   = [c for c in allfeat if c.lower().startswith("fp_")]
cond = [c for c in allfeat if c in COND_COLS]
rdkit = [c for c in allfeat if c not in emb and c not in fp and c not in cond]
SETS = {
    "ALL":     emb + fp + rdkit + cond,
    "NO_EMB":  fp + rdkit + cond,
    "FP_COND": fp + cond,
}
print(f"feature groups: embeddings={len(emb)} ECFP={len(fp)} RDKit={len(rdkit)} cond/metal={len(cond)}")

# ------------------------------------------------- sanitize once (cf. ML_v58)
Xtr_full = tr[allfeat].values.astype(np.float64)
Xte_full = te[allfeat].values.astype(np.float64)
def _bad(M): return ~np.isfinite(M) | (np.abs(M) > 1e30)
_tmp = Xtr_full.copy(); _tmp[_bad(Xtr_full)] = np.nan
_med = np.nanmedian(_tmp, axis=0); _med = np.where(np.isfinite(_med), _med, 0.0)
def _imp(M):
    M = M.copy(); idx = np.where(_bad(M)); M[idx] = np.take(_med, idx[1]); return M
Xtr_full, Xte_full = _imp(Xtr_full), _imp(Xte_full)
_logc = np.where(np.nanmax(np.abs(Xtr_full), axis=0) > 1e7)[0]
for j in _logc:
    Xtr_full[:, j] = np.sign(Xtr_full[:, j]) * np.log1p(np.abs(Xtr_full[:, j]))
    Xte_full[:, j] = np.sign(Xte_full[:, j]) * np.log1p(np.abs(Xte_full[:, j]))
col_idx = {c: i for i, c in enumerate(allfeat)}

# --------------------------------------------- molecule-grouped val (shared)
rng = np.random.RandomState(SEED)
uniq = np.unique(smi_tr).copy(); rng.shuffle(uniq)
k = max(1, int(round(0.15 * len(uniq))))
valset = set(uniq[:k])
vmask = np.array([s in valset for s in smi_tr])
fit_i, val_i = np.where(~vmask)[0], np.where(vmask)[0]
print(f"train molecules={len(uniq)}  grouped-val molecules={k}  "
      f"(fit rows={len(fit_i)}, val rows={len(val_i)})  | external rows={len(y_te)}")

def make_models():
    return {
        'xgb': xgb.XGBRegressor(n_estimators=3000, learning_rate=0.02, max_depth=6,
            min_child_weight=3, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1,
            reg_lambda=1.0, random_state=SEED, n_jobs=-1, early_stopping_rounds=100, verbosity=0),
        'lgb': lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.02, num_leaves=63,
            min_child_samples=10, subsample=0.85, colsample_bytree=0.85,
            random_state=SEED, n_jobs=-1, verbosity=-1),
        'hgb': HistGradientBoostingRegressor(max_iter=1500, learning_rate=0.03, max_depth=8,
            min_samples_leaf=10, l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.12, n_iter_no_change=80, random_state=SEED),
        'et': ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.6,
            random_state=SEED, n_jobs=-1),
        'rf': RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.5,
            random_state=SEED, n_jobs=-1),
    }

def fit_one(name, m, Xf, yf, Xv, yv):
    if name == 'xgb':
        m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
    elif name == 'lgb':
        m.fit(Xf, yf, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        m.fit(Xf, yf)
    return m

results = {}
for sname, cols in SETS.items():
    t0 = time.time()
    ci = [col_idx[c] for c in cols]
    Xtr = Xtr_full[:, ci]; Xte = Xte_full[:, ci]
    Xf, yf = Xtr[fit_i], y_tr[fit_i]
    Xv, yv = Xtr[val_i], y_tr[val_i]
    # (1) grouped-val R² per model -> weights
    val_r2, ext_pred = {}, {}
    for name, m in make_models().items():
        fit_one(name, m, Xf, yf, Xv, yv)
        val_r2[name] = max(0.0, r2_score(yv, m.predict(Xv)))
    # (2) refit on all train rows, predict external
    for name, m in make_models().items():
        fit_one(name, m, Xtr, y_tr, Xv, yv)   # Xv only used by xgb/lgb early-stop
        ext_pred[name] = m.predict(Xte)
    # blend by squared grouped-val R²
    w = np.array([val_r2[n] ** 2 for n in ext_pred]); w = w / (w.sum() + 1e-12)
    blend = np.zeros(len(y_te))
    for wi, n in zip(w, ext_pred): blend += wi * ext_pred[n]
    br2 = r2_score(y_te, blend); brmse = np.sqrt(mean_squared_error(y_te, blend))
    per_model = {n: r2_score(y_te, ext_pred[n]) for n in ext_pred}
    results[sname] = dict(n_feat=len(cols), val_r2=val_r2, ext_r2=per_model,
                          blend_r2=br2, blend_rmse=brmse)
    print(f"\n[{sname}] {len(cols)} feats ({(time.time()-t0)/60:.1f} min)")
    print(f"   per-model ext R²: " + "  ".join(f"{n}={per_model[n]:.3f}" for n in per_model))
    print(f"   weighted blend  : R²={br2:.4f}  RMSE={brmse:.4f}")

print("\n" + "=" * 70)
print("BASELINE SUMMARY — external V39 (chemistry-disjoint, n=%d)" % len(y_te))
print("=" * 70)
print(f"{'feature set':<10s} {'#feat':>6s} {'blend R²':>9s} {'blend RMSE':>11s}")
for s, r in results.items():
    print(f"{s:<10s} {r['n_feat']:>6d} {r['blend_r2']:>9.4f} {r['blend_rmse']:>11.4f}")
print("-" * 70)
print(f"{'GNN-alone':<10s} {'-':>6s} {0.2273:>9.4f} {1.4856:>11.4f}   (real D-MPNN, for reference)")
print("=" * 70)

rows = []
for s, r in results.items():
    rows.append({'feature_set': s, 'n_feat': r['n_feat'],
                 'blend_r2': r['blend_r2'], 'blend_rmse': r['blend_rmse'],
                 **{f'ext_r2_{n}': r['ext_r2'][n] for n in r['ext_r2']}})
pd.DataFrame(rows).to_csv('baseline_v39_results.csv', index=False)
print(f"saved baseline_v39_results.csv   |   total {(time.time()-START)/60:.1f} min")
