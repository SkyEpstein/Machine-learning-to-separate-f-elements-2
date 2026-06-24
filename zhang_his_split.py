#!/usr/bin/env python3
"""
zhang_his_split.py — the purest head-to-head: our model on Dr. Zhang's exact
features, his exact train/test split, and his exact 494-row held-out test (15
molecules), scored against his reported 0.72.

His committed split files already carry the full feature set (Morgan CircularFP, a
large RDKit descriptor block, the conditions, and the numeric metal descriptors)
plus the targets Class_index (3 classes at D = 0.5 and 10), Distribution_ratio, and
Log_D. We train our models on his trainVal (7581 rows) and evaluate on his test
(494 rows): a 3-class classifier with our confidence (max class probability) and
selective accuracy, and a LightGBM regressor with our err_lgb confidence. Since the
test molecules are held out, this is the new-molecule (screening) setting.
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import lightgbm as lgb
SEED = 42; START = time.time()
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
tr = pd.read_csv("/tmp/z_trainVal_dataset.csv"); te = pd.read_csv("/tmp/z_test_dataset.csv")
DROP = ['Class_index', 'SMILES', 'SMILES_class', 'Distribution_ratio', 'Log_D']
feat = [c for c in tr.columns if c not in DROP]
def san(M):
    M = M.astype(np.float64).copy(); b = ~np.isfinite(M) | (np.abs(M) > 1e30)
    md = np.nan_to_num(np.nanmedian(np.where(b, np.nan, M), axis=0)); ix = np.where(b); M[ix] = np.take(md, ix[1]); return M
Xtr = san(tr[feat].values); Xte = san(te[feat].values)
ytr = tr['Log_D'].values.astype(float); yte = te['Log_D'].values.astype(float)
ctr = tr['Class_index'].values.astype(int); cte = te['Class_index'].values.astype(int)
print(f"his split: trainVal {Xtr.shape}, test {Xte.shape} | test molecules {te['SMILES'].nunique()} | features {len(feat)}", flush=True)
print(f"test class balance {np.bincount(cte).tolist()} (majority baseline {np.bincount(cte).max()/len(cte):.2f})", flush=True)

# our 3-class classifier on his exact split
clf = lgb.LGBMClassifier(objective='multiclass', num_class=3, n_estimators=900, learning_rate=0.04, num_leaves=63, min_child_samples=15, subsample=0.85, colsample_bytree=0.7, reg_lambda=1.5, random_state=SEED, n_jobs=-1, verbosity=-1)
clf.fit(Xtr, ctr); proba = clf.predict_proba(Xte); pc = proba.argmax(1); conf = proba.max(1)
acc = accuracy_score(cte, pc)
print(f"\n=== OUR CLASSIFIER on his exact 494-row test ===", flush=True)
print(f"  3-class accuracy = {acc:.3f}   macro-F1 = {f1_score(cte,pc,average='macro'):.3f}   per-class F1 = {np.round(f1_score(cte,pc,average=None),2).tolist()}", flush=True)
print(f"  Dr. Zhang on the same test: 0.72 (XGBoost, no confidence ranking)", flush=True)
print("  selective accuracy (confidence = max class probability):", flush=True)
for p in [100, 50, 25, 10]:
    m = np.ones(len(cte), bool) if p == 100 else conf >= np.percentile(conf, 100 - p)
    print(f"      top {p:>3d}% conf: accuracy = {accuracy_score(cte[m],pc[m]):.3f}  (n={int(m.sum())})", flush=True)

# our regression on his exact split (held-out molecules = screening setting)
LGB = lambda: lgb.LGBMRegressor(n_estimators=1800, learning_rate=0.03, num_leaves=63, min_child_samples=12, subsample=0.85, colsample_bytree=0.8, reg_lambda=1.5, random_state=SEED, n_jobs=-1, verbosity=-1)
reg = LGB().fit(Xtr, ytr); pte = reg.predict(Xte)
# err_lgb confidence: train on trainVal OOF residuals, apply to test
oof = np.zeros(len(ytr)); fold = np.zeros(len(ytr), int)
for i, (_, va) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(ytr)): fold[va] = i
for f in range(5):
    trr = np.where(fold != f)[0]; va = np.where(fold == f)[0]; oof[va] = LGB().fit(Xtr[trr], ytr[trr]).predict(Xtr[va])
ECf = lambda: lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8, reg_lambda=2.0, random_state=SEED, n_jobs=-1, verbosity=-1)
errm = ECf().fit(np.column_stack([Xtr, oof]), np.abs(ytr - oof)); errte = np.clip(errm.predict(np.column_stack([Xte, pte])), 0.05, None)
print(f"\n=== OUR REGRESSION on his exact 494-row test (held-out molecules) ===", flush=True)
print(f"  R2 = {r2_score(yte,pte):.3f}   RMSE = {rmse(yte,pte):.3f}   Spearman(err,|res|) = {spearmanr(errte, np.abs(yte-pte)).correlation:.3f}", flush=True)
for p in [100, 50, 25, 10]:
    m = np.ones(len(yte), bool) if p == 100 else errte <= np.percentile(errte, p)
    print(f"      top {p:>3d}% conf: R2 = {r2_score(yte[m],pte[m]):.3f}  RMSE = {rmse(yte[m],pte[m]):.3f}", flush=True)
pd.DataFrame([
    {'metric': '3-class accuracy (his test)', 'ours': round(acc, 3), 'zhang': 0.72},
    {'metric': '3-class acc, top 25% conf', 'ours': round(accuracy_score(cte[conf >= np.percentile(conf, 75)], pc[conf >= np.percentile(conf, 75)]), 3), 'zhang': np.nan},
    {'metric': '3-class acc, top 10% conf', 'ours': round(accuracy_score(cte[conf >= np.percentile(conf, 90)], pc[conf >= np.percentile(conf, 90)]), 3), 'zhang': np.nan},
    {'metric': 'regression R2 (his test)', 'ours': round(r2_score(yte, pte), 3), 'zhang': np.nan},
    {'metric': 'regression RMSE (his test)', 'ours': round(rmse(yte, pte), 3), 'zhang': np.nan},
]).to_csv("zhang_his_split_results.csv", index=False)
print(f"\nsaved zhang_his_split_results.csv | total {(time.time()-START)/60:.1f} min", flush=True)
