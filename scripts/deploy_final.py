#!/usr/bin/env python3
"""
deploy_final.py — the final deployable, built directly from the saved OOF
(conf_oof_A.npz, conf_oof_B.npz) so the reported numbers match the confidence
bakeoff exactly. Per-track choice locked by that bakeoff:

  Track A (new-molecule screening): SINGLE LightGBM, err recipe plain+strong.
      Best confidence ranking (the stack trades confidence for a bit of accuracy here).
  Track B (known-molecule condition optimization): NNLS STACK, err recipe plain+reg.
      Best accuracy, and confidence as good as the single model.

For each track: err_lgb confidence (5-fold OOF), confidence-filtered R^2 AND RMSE,
normalized split-conformal coverage and width, and a saved predictions CSV. Both
metrics are always reported together.
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
SEED = 42; START = time.time()
TARGET, SMI = "Log_D", "SMILES_canonical"
TEXT = {"Solvent_A", "Solvent_B", "Metal", "Acid_type", "SMILES_canonical"}
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
STRONG = lambda: lgb.LGBMRegressor(n_estimators=1800, learning_rate=0.03, num_leaves=63, min_child_samples=12, subsample=0.85, colsample_bytree=0.8, reg_lambda=1.5, random_state=SEED, n_jobs=-1, verbosity=-1)
REG = lambda: lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, random_state=SEED, n_jobs=-1, verbosity=-1)

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
    return df[(~df.duplicated().values) & (grng <= 2.0)].reset_index(drop=True)

df = load_clean(); smi_all = df[SMI].astype(str).values

def folds(grouped, rep):
    if grouped:
        uq = np.unique(smi_all).copy(); np.random.RandomState(SEED + rep).shuffle(uq); fo = {m: i % 5 for i, m in enumerate(uq)}
        return np.array([fo[s] for s in smi_all])
    f = np.zeros(len(df), int)
    for i, (_, va) in enumerate(KFold(5, shuffle=True, random_state=SEED + rep).split(f)): f[va] = i
    return f

def run(npz, predkey, errmk, grouped, label, tag):
    d = np.load(npz, allow_pickle=True)
    y, pred, Xcond, fold = d['y'], d[predkey], d['Xcond'], d['fold']
    assert len(df) == len(y), f"row mismatch for {tag}"
    Cf = np.column_stack([Xcond, pred]); res = np.abs(y - pred); err = np.zeros(len(y))
    for f in range(5):
        trr = np.where(fold != f)[0]; va = np.where(fold == f)[0]
        err[va] = errmk().fit(Cf[trr], res[trr]).predict(Cf[va])
    err = np.clip(err, 0.05, None)
    print(f"\n=== {label} ===  R^2={r2_score(y,pred):.4f}  RMSE={rmse(y,pred):.4f}  (cleaned data, n={len(y)})", flush=True)
    print("  confidence-filtered (R^2 and RMSE):", flush=True)
    for p in [100, 50, 25, 10]:
        m = np.ones(len(y), bool) if p == 100 else (err <= np.percentile(err, p))
        print(f"    top {p:>3d}%   R^2={r2_score(y[m],pred[m]):.3f}   RMSE={rmse(y[m],pred[m]):.3f}", flush=True)
    cov90, cov80, w90 = [], [], []
    for rep in range(4):
        f2 = folds(grouped, 100 + rep); cal = np.where(f2 < 3)[0]; tst = np.where(f2 >= 3)[0]
        s = np.abs(y[cal] - pred[cal]) / err[cal]
        for alpha, cov in [(0.10, cov90), (0.20, cov80)]:
            qn = min(1.0, np.ceil((len(cal) + 1) * (1 - alpha)) / len(cal)); q = np.quantile(s, qn)
            lo = pred[tst] - q * err[tst]; hi = pred[tst] + q * err[tst]
            cov.append(np.mean((y[tst] >= lo) & (y[tst] <= hi)))
            if alpha == 0.10: w90.append(np.median(hi - lo))
    print(f"  conformal: 90% target -> coverage {np.mean(cov90):.2f}, median width {np.mean(w90):.2f} log units; 80% -> coverage {np.mean(cov80):.2f}", flush=True)
    s = np.abs(y - pred) / err; q90 = np.quantile(s, min(1.0, np.ceil((len(y) + 1) * 0.9) / len(y))); q80 = np.quantile(s, min(1.0, np.ceil((len(y) + 1) * 0.8) / len(y)))
    out = pd.DataFrame({'SMILES': smi_all, 'Metal': df['Metal'].astype(str).values, 'Acid_conc_M': df['Acid_conc_M'].values,
                        'Temperature_K': df['Temperature_K'].values, 'Actual_LogD': y, 'Pred_LogD': np.round(pred, 4),
                        'confidence_pred_err': np.round(err, 4), 'lo90': np.round(pred - q90 * err, 4), 'hi90': np.round(pred + q90 * err, 4),
                        'lo80': np.round(pred - q80 * err, 4), 'hi80': np.round(pred + q80 * err, 4)})
    out.to_csv(f"deploy_{tag}_predictions.csv", index=False)
    print(f"  saved deploy_{tag}_predictions.csv", flush=True)

run("conf_oof_A.npz", "single", STRONG, True, "Track A, new-molecule screening (single LightGBM)", "A_screening")
run("conf_oof_B.npz", "stack", REG, False, "Track B, known-molecule condition optimization (NNLS stack)", "B_condition")
print(f"\ntotal {(time.time()-START)/60:.1f} min", flush=True)
