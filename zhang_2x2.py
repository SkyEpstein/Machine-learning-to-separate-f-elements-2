#!/usr/bin/env python3
"""
zhang_2x2.py — model crossed with split, on the same featurized data.

Uses Dr. Zhang's committed featurized rows (trainVal + test = 8075 rows, his Morgan
fingerprint + descriptors + conditions + metal descriptors, target Class_index, the
3 classes at D = 0.5 and 10). Fills a 2 by 2:

                 our split (molecule-grouped 5-fold CV)   his split (single 494-row holdout)
  our model (LightGBM)              a                                  b
  his model (XGBoost)               c                                  d

So "our model on our split" (a) and "his model on his split" (d, his reported 0.72)
sit in one table, with the off-diagonal cells showing that the split, not the model,
drives most of the gap. Also reports our regression under our split for the headline.
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import lightgbm as lgb, xgboost as xgb
SEED = 42; START = time.time()
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
tr = pd.read_csv("/tmp/z_trainVal_dataset.csv"); te = pd.read_csv("/tmp/z_test_dataset.csv")
DROP = ['Class_index', 'SMILES', 'SMILES_class', 'Distribution_ratio', 'Log_D']
feat = [c for c in tr.columns if c not in DROP]
def san(M):
    M = M.astype(np.float64).copy(); b = ~np.isfinite(M) | (np.abs(M) > 1e30)
    md = np.nan_to_num(np.nanmedian(np.where(b, np.nan, M), axis=0)); ix = np.where(b); M[ix] = np.take(md, ix[1]); return M
alldf = pd.concat([tr, te], ignore_index=True)
X = san(alldf[feat].values); c = alldf['Class_index'].values.astype(int); ylog = alldf['Log_D'].values.astype(float)
smi = alldf['SMILES'].astype(str).values
Xtr = san(tr[feat].values); ctr = tr['Class_index'].values.astype(int)
Xte = san(te[feat].values); cte = te['Class_index'].values.astype(int)
LGBc = lambda: lgb.LGBMClassifier(objective='multiclass', num_class=3, n_estimators=900, learning_rate=0.04, num_leaves=63, min_child_samples=15, subsample=0.85, colsample_bytree=0.7, reg_lambda=1.5, random_state=SEED, n_jobs=-1, verbosity=-1)
XGBc = lambda: xgb.XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=700, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.6, reg_lambda=1.5, random_state=SEED, n_jobs=-1, verbosity=0, eval_metric='mlogloss')
# our split: molecule-grouped 5-fold CV
uq = np.unique(smi).copy(); np.random.RandomState(SEED).shuffle(uq); fo = {m: i % 5 for i, m in enumerate(uq)}; fold = np.array([fo[s] for s in smi])
def cv_acc(mk):
    pred = np.zeros(len(c), int)
    for f in range(5):
        trr = np.where(fold != f)[0]; va = np.where(fold == f)[0]; pred[va] = mk().fit(X[trr], c[trr]).predict(X[va])
    return accuracy_score(c, pred)
def holdout_acc(mk):
    return accuracy_score(cte, mk().fit(Xtr, ctr).predict(Xte))
print("computing 2x2 (model x split) on shared featurized data...", flush=True)
a = cv_acc(LGBc); print(f"  a our model (LightGBM), our split (CV)   = {a:.3f}", flush=True)
c_ = cv_acc(XGBc); print(f"  c his model (XGBoost),  our split (CV)   = {c_:.3f}", flush=True)
b = holdout_acc(LGBc); print(f"  b our model (LightGBM), his split (494)  = {b:.3f}", flush=True)
d = holdout_acc(XGBc); print(f"  d his model (XGBoost),  his split (494)  = {d:.3f}  (his reported 0.72)", flush=True)
# our regression under our split (Track A analog, held-out molecules)
predr = np.zeros(len(ylog))
for f in range(5):
    trr = np.where(fold != f)[0]; va = np.where(fold == f)[0]
    predr[va] = lgb.LGBMRegressor(n_estimators=1800, learning_rate=0.03, num_leaves=63, min_child_samples=12, subsample=0.85, colsample_bytree=0.8, reg_lambda=1.5, random_state=SEED, n_jobs=-1, verbosity=-1).fit(X[trr], ylog[trr]).predict(X[va])
print(f"\n  our regression, our split (molecule-grouped CV): R2={r2_score(ylog,predr):.3f} RMSE={rmse(ylog,predr):.3f}", flush=True)
pd.DataFrame([
    {'model': 'our model (LightGBM)', 'our_split_CV': round(a, 3), 'his_split_holdout': round(b, 3)},
    {'model': 'his model (XGBoost)', 'our_split_CV': round(c_, 3), 'his_split_holdout': round(d, 3)},
]).to_csv("zhang_2x2_results.csv", index=False)
print("\nsaved zhang_2x2_results.csv | total %.1f min" % ((time.time()-START)/60), flush=True)
