#!/usr/bin/env python3
"""
xgb_confidence.py — take Dr. Zhang's ML and add our confidence layer.

His ML: an XGBoost 3-class classifier on fingerprints + conditions, split by
molecule. This reproduces that (XGBoost, multi:softprob, num_class=3, ECFP +
conditions + metal + ligand) and adds a confidence score (the max class
probability) plus selective accuracy at the top 50/25/10 percent by confidence,
which his single flat number does not provide.

Setup matches his held-out-by-molecule test: molecule-grouped 5-fold CV (the
new-molecule case). 3 classes at distribution coefficient D = 0.5 and 10. A binary
"will it extract" (logD > 0) classifier is included too. Reference: Zhang 0.72.
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb
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
X = san(np.hstack([df[[c for c in COND if c in df]].values, df[fp].values, Lg]))   # fingerprints + conditions, like his 1860
print(f"cleaned {len(y)} rows | features {X.shape[1]} (conditions + ECFP + ligand)", flush=True)
cuts = np.array([np.log10(0.5), 1.0]); y3 = np.digitize(y, cuts); yb = (y > 0).astype(int)
# molecule-grouped folds = his held-out-by-molecule setup
uq = np.unique(smi).copy(); np.random.RandomState(SEED).shuffle(uq); fo = {m: i % 5 for i, m in enumerate(uq)}
fold = np.array([fo[s] for s in smi])
def xgb3(): return xgb.XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=700, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.6, reg_lambda=1.5, random_state=SEED, n_jobs=-1, verbosity=0, eval_metric='mlogloss')
def xgbb(): return xgb.XGBClassifier(objective='binary:logistic', n_estimators=700, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.6, reg_lambda=1.5, random_state=SEED, n_jobs=-1, verbosity=0, eval_metric='logloss')
proba3 = np.zeros((len(y), 3)); probab = np.zeros(len(y))
for f in range(5):
    trr = np.where(fold != f)[0]; va = np.where(fold == f)[0]
    proba3[va] = xgb3().fit(X[trr], y3[trr]).predict_proba(X[va])
    probab[va] = xgbb().fit(X[trr], yb[trr]).predict_proba(X[va])[:, 1]
    print(f"  fold {f+1}/5 done", flush=True)
p3 = proba3.argmax(1); conf3 = proba3.max(1); pb = (probab > 0.5).astype(int); confb = np.abs(probab - 0.5)
print("\n=== Zhang's ML (XGBoost) + our confidence | new-molecule, molecule-grouped CV ===", flush=True)
print(f"  3-class: accuracy={accuracy_score(y3,p3):.3f}  macro-F1={f1_score(y3,p3,average='macro'):.3f}  per-class F1={np.round(f1_score(y3,p3,average=None),2).tolist()}  (majority baseline {np.bincount(y3).max()/len(y3):.2f})", flush=True)
print("  3-class selective accuracy (confidence = max class probability):", flush=True)
for p in [100, 50, 25, 10]:
    m = np.ones(len(y), bool) if p == 100 else (conf3 >= np.percentile(conf3, 100 - p))
    print(f"    top {p:>3d}% confidence: accuracy={accuracy_score(y3[m],p3[m]):.3f}  (n={int(m.sum())})", flush=True)
print(f"  binary logD>0: accuracy={accuracy_score(yb,pb):.3f}  ROC AUC={roc_auc_score(yb,probab):.3f}  (base rate {max(yb.mean(),1-yb.mean()):.2f})", flush=True)
for p in [100, 50, 25, 10]:
    m = np.ones(len(y), bool) if p == 100 else (confb >= np.percentile(confb, 100 - p))
    print(f"    binary top {p:>3d}% confidence: accuracy={accuracy_score(yb[m],pb[m]):.3f}", flush=True)
pd.DataFrame([{'model': 'Zhang XGBoost + our confidence', 'task': '3-class', 'accuracy_all': round(accuracy_score(y3,p3),3), 'macro_F1': round(f1_score(y3,p3,average='macro'),3),
               'top25_acc': round(accuracy_score(y3[conf3>=np.percentile(conf3,75)],p3[conf3>=np.percentile(conf3,75)]),3),
               'top10_acc': round(accuracy_score(y3[conf3>=np.percentile(conf3,90)],p3[conf3>=np.percentile(conf3,90)]),3)}]).to_csv("xgb_confidence_results.csv", index=False)
print("\nReference: Zhang XGBoost 3-class accuracy 0.72 on one 494-row molecule-held-out test (no confidence ranking). saved xgb_confidence_results.csv", flush=True)
print(f"total {(time.time()-START)/60:.1f} min", flush=True)
