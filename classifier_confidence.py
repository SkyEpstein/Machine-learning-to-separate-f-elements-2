#!/usr/bin/env python3
"""
classifier_confidence.py — build native classifiers with a confidence score, to
match Dr. Zhang's task directly and add the confidence layer his model lacks.

Two targets:
  3-class at distribution coefficient D = 0.5 and 10 (his inferred cut points): low / medium / high.
  binary: will it extract at all (logD > 0).
Two tracks:
  Track A new-molecule (conditions + metal, molecule-grouped CV) -> the Zhang analog.
  Track B known-molecule (conditions + ECFP + ligand, random CV).
Confidence is the predicted probability of the chosen class (max softmax). We report
overall accuracy and macro-F1, per-class F1, ROC AUC for the binary case, and
selective accuracy at the top 50/25/10 percent by confidence. Both the headline
metric and the confidence-ranked subsets are shown.
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
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
Xcond = san(df[[c for c in COND if c in df]].values)
Xfull = san(np.hstack([df[[c for c in COND if c in df]].values, df[fp].values, Lg]))
cuts = np.array([np.log10(0.5), 1.0]); y3 = np.digitize(y, cuts); yb = (y > 0).astype(int)
def folds(grouped):
    if grouped:
        uq = np.unique(smi).copy(); np.random.RandomState(SEED).shuffle(uq); fo = {m: i % 5 for i, m in enumerate(uq)}; return np.array([fo[s] for s in smi])
    f = np.zeros(len(y), int)
    for i, (_, va) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(y)): f[va] = i
    return f
def clf(multi):
    return lgb.LGBMClassifier(objective='multiclass' if multi else 'binary', num_class=3 if multi else None,
                              n_estimators=900, learning_rate=0.04, num_leaves=63, min_child_samples=15,
                              subsample=0.85, colsample_bytree=0.7, reg_lambda=1.5, random_state=SEED, n_jobs=-1, verbosity=-1)
def selective(acc_true, pred, conf, header):
    print(header, flush=True)
    for p in [100, 50, 25, 10]:
        m = np.ones(len(conf), bool) if p == 100 else (conf >= np.percentile(conf, 100 - p))
        print(f"    top {p:>3d}% confidence: accuracy={accuracy_score(acc_true[m], pred[m]):.3f}  (n={int(m.sum())})", flush=True)

rows = []
for X, grouped, label in [(Xcond, True, "Track A new-molecule (Zhang analog)"), (Xfull, False, "Track B known-molecule")]:
    fold = folds(grouped)
    proba3 = np.zeros((len(y), 3)); probab = np.zeros(len(y))
    for f in range(5):
        trr = np.where(fold != f)[0]; va = np.where(fold == f)[0]
        proba3[va] = clf(True).fit(X[trr], y3[trr]).predict_proba(X[va])
        probab[va] = clf(False).fit(X[trr], yb[trr]).predict_proba(X[va])[:, 1]
    p3 = proba3.argmax(1); conf3 = proba3.max(1); pb = (probab > 0.5).astype(int); confb = np.abs(probab - 0.5)
    acc = accuracy_score(y3, p3); mf1 = f1_score(y3, p3, average='macro'); perf1 = f1_score(y3, p3, average=None)
    print(f"\n=== {label} ===", flush=True)
    print(f"  3-class: accuracy={acc:.3f}  macro-F1={mf1:.3f}  per-class F1={np.round(perf1,2).tolist()}  (majority baseline {np.bincount(y3).max()/len(y3):.2f})", flush=True)
    selective(y3, p3, conf3, "  3-class selective accuracy (confidence = max class probability):")
    print(f"  binary logD>0: accuracy={accuracy_score(yb,pb):.3f}  ROC AUC={roc_auc_score(yb,probab):.3f}  (base rate {max(yb.mean(),1-yb.mean()):.2f})", flush=True)
    selective(yb, pb, confb, "  binary selective accuracy (confidence = distance from 0.5):")
    rows.append({'track': label, 'task': '3-class', 'accuracy': round(acc,3), 'macro_F1': round(mf1,3),
                 'top25_acc': round(accuracy_score(y3[conf3>=np.percentile(conf3,75)], p3[conf3>=np.percentile(conf3,75)]),3),
                 'top10_acc': round(accuracy_score(y3[conf3>=np.percentile(conf3,90)], p3[conf3>=np.percentile(conf3,90)]),3)})
    rows.append({'track': label, 'task': 'binary logD>0', 'accuracy': round(accuracy_score(yb,pb),3), 'macro_F1': round(roc_auc_score(yb,probab),3),
                 'top25_acc': round(accuracy_score(yb[confb>=np.percentile(confb,75)], pb[confb>=np.percentile(confb,75)]),3),
                 'top10_acc': round(accuracy_score(yb[confb>=np.percentile(confb,90)], pb[confb>=np.percentile(confb,90)]),3)})
pd.DataFrame(rows).to_csv("classifier_confidence_results.csv", index=False)
print("\nZhang XGBoost 3-class: accuracy 0.72, macro-F1 0.67 (no confidence ranking). saved classifier_confidence_results.csv", flush=True)
print(f"total {(time.time()-START)/60:.1f} min", flush=True)
