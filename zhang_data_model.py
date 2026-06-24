#!/usr/bin/env python3
"""
zhang_data_model.py — train our best model on Dr. Zhang's own integrated dataset
and compare to his result and to our data.

His data is the Ln integrated set plus the An integrated set from his repo, combined
to 8075 rows (the same count, molecules, and metals as ours, which is itself
evidence the two datasets are the same). His clean file carries raw categorical
columns (Metal, Acid_type, Solvent names) rather than the numeric descriptors our
copy had, so here the categoricals are one-hot encoded and the ECFP fingerprint is
computed from the SMILES.

Runs: Track A regression (single LightGBM, conditions + metal + acid + solvent,
molecule-grouped CV), Track B regression (NNLS stack of lgb/xgb/et/cat, + ECFP +
ligand, random CV), and a 3-class classifier at D = 0.5 and 10 (molecule-grouped)
with our confidence and selective accuracy. Reports R2 and RMSE with confidence
curves and the 3-class accuracy against his 0.72.
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from scipy.stats import spearmanr
import lightgbm as lgb, xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
try:
    from catboost import CatBoostRegressor; CAT = True
except Exception: CAT = False
SEED = 42; START = time.time()
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
df = pd.read_csv("/tmp/zhang_combined.csv", low_memory=False)
df = df[df['SMILES'].notna() & df['Log_D'].notna()].reset_index(drop=True)
# our cleaning: drop exact duplicates and discordant replicate groups (range > 2 log units)
NUMC = ['Extractant_conc_M','Volume_fraction_A','Volume_fraction_B','Metal_conc_mM','Acid_conc_M','Temperature_K']
key = df['SMILES'].astype(str) + '|' + df['Metal'].astype(str) + '|' + df['Acid_type'].astype(str)
for c, r in [('Acid_conc_M',2),('Temperature_K',0),('Extractant_conc_M',3),('Metal_conc_mM',4)]:
    key = key + '|' + df[c].round(r).astype(str)
grng = pd.Series(df['Log_D'].values).groupby(key.values).transform(lambda v: v.max()-v.min()).values
df = df[(~df.duplicated().values) & (grng <= 2.0)].reset_index(drop=True)
y = df['Log_D'].values.astype(float); smi = df['SMILES'].astype(str).values
print(f"Zhang data after our cleaning: {len(y)} rows, {df['SMILES'].nunique()} molecules, {df['Metal'].nunique()} metals", flush=True)
# features
def san(M):
    M = M.astype(np.float64).copy(); b = ~np.isfinite(M) | (np.abs(M) > 1e30)
    md = np.nan_to_num(np.nanmedian(np.where(b, np.nan, M), axis=0)); ix = np.where(b); M[ix] = np.take(md, ix[1]); return M
num = san(df[NUMC].values)
cat = pd.get_dummies(df[['Metal','Acid_type','Solvent_A','Solvent_B']].astype(str), dummy_na=True).values.astype(float)
def ecfp(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return np.zeros(1024)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024), dtype=float)
def lig(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return [0,0,0,0,0.0]
    sy=[a.GetSymbol() for a in m.GetAtoms()]; return [sy.count('O'),sy.count('N'),sy.count('S'),sy.count('P'),Descriptors.MolLogP(m)]
FP = np.array([ecfp(s) for s in smi]); LG = np.array([lig(s) for s in smi], float)
Xcond = np.hstack([num, cat])                      # conditions + metal + acid + solvent
Xfull = np.hstack([Xcond, FP, LG])                 # + structure
cuts = np.array([np.log10(0.5), 1.0]); y3 = np.digitize(y, cuts)
def gfold():
    uq=np.unique(smi).copy(); np.random.RandomState(SEED).shuffle(uq); fo={m:i%5 for i,m in enumerate(uq)}; return np.array([fo[s] for s in smi])
def rfold():
    f=np.zeros(len(y),int)
    for i,(_,va) in enumerate(KFold(5,shuffle=True,random_state=SEED).split(y)): f[va]=i
    return f
LGB = lambda: lgb.LGBMRegressor(n_estimators=1800, learning_rate=0.03, num_leaves=63, min_child_samples=12, subsample=0.85, colsample_bytree=0.8, reg_lambda=1.5, random_state=SEED, n_jobs=-1, verbosity=-1)
REG = lambda: lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, random_state=SEED, n_jobs=-1, verbosity=-1)
def members():
    m={'lgb':lgb.LGBMRegressor(n_estimators=1500,learning_rate=0.03,num_leaves=63,min_child_samples=12,random_state=SEED,n_jobs=-1,verbosity=-1),
       'xgb':xgb.XGBRegressor(n_estimators=1200,learning_rate=0.03,max_depth=7,subsample=0.9,colsample_bytree=0.8,random_state=SEED,n_jobs=-1,verbosity=0),
       'et':ExtraTreesRegressor(n_estimators=400,min_samples_leaf=2,max_features=0.6,random_state=SEED,n_jobs=-1)}
    if CAT: m['cat']=CatBoostRegressor(iterations=1500,learning_rate=0.03,depth=8,random_state=SEED,verbose=False,allow_writing_files=False)
    return m
def oof_single(X, fold, mk):
    p=np.zeros(len(y))
    for f in range(5):
        tr=np.where(fold!=f)[0]; va=np.where(fold==f)[0]; p[va]=mk().fit(X[tr],y[tr]).predict(X[va])
    return p
def oof_stack(X, fold):
    names=list(members().keys()); oof={n:np.zeros(len(y)) for n in names}
    for f in range(5):
        tr=np.where(fold!=f)[0]; va=np.where(fold==f)[0]
        for n,md in members().items(): oof[n][va]=md.fit(X[tr],y[tr]).predict(X[va])
    P=np.column_stack([oof[n] for n in names]); w=np.clip(LinearRegression(positive=True,fit_intercept=False).fit(P,y).coef_,0,None); w/=w.sum()+1e-12
    return P@w, dict(zip(names, np.round(w,2)))
def conf_curve(tag, pred, fold, errmk):
    res=np.abs(y-pred); Cf=np.column_stack([Xcond,pred]); err=np.zeros(len(y))
    for f in range(5):
        tr=np.where(fold!=f)[0]; va=np.where(fold==f)[0]; err[va]=errmk().fit(Cf[tr],res[tr]).predict(Cf[va])
    err=np.clip(err,0.05,None)
    print(f"  {tag}: R2={r2_score(y,pred):.3f} RMSE={rmse(y,pred):.3f} | Spearman(err,|res|)={spearmanr(err,res).correlation:.3f}", flush=True)
    for p in [50,25,10]:
        m=err<=np.percentile(err,p); print(f"      top {p:>2d}% conf: R2={r2_score(y[m],pred[m]):.3f} RMSE={rmse(y[m],pred[m]):.3f}", flush=True)
    return err

print("\n=== OUR MODEL ON ZHANG'S DATA ===", flush=True)
gf, rf = gfold(), rfold()
predA = oof_single(Xcond, gf, LGB); conf_curve("Track A new-molecule (single LightGBM, grouped CV)", predA, gf, LGB)
predB, wB = oof_stack(Xfull, rf); print(f"  Track B stack weights: {wB}", flush=True); conf_curve("Track B known-molecule (NNLS stack, random CV)", predB, rf, REG)

# 3-class classifier head-to-head (molecule-grouped, his task), with confidence
proba=np.zeros((len(y),3))
for f in range(5):
    tr=np.where(gf!=f)[0]; va=np.where(gf==f)[0]
    proba[va]=lgb.LGBMClassifier(objective='multiclass',num_class=3,n_estimators=900,learning_rate=0.04,num_leaves=63,min_child_samples=15,subsample=0.85,colsample_bytree=0.7,reg_lambda=1.5,random_state=SEED,n_jobs=-1,verbosity=-1).fit(Xfull[tr],y3[tr]).predict_proba(Xfull[va])
pc=proba.argmax(1); cf=proba.max(1)
print(f"\n  3-class classifier on Zhang data (new-molecule, grouped CV): accuracy={accuracy_score(y3,pc):.3f} macro-F1={f1_score(y3,pc,average='macro'):.3f} (baseline {np.bincount(y3).max()/len(y3):.2f})", flush=True)
for p in [100,50,25,10]:
    m=np.ones(len(y),bool) if p==100 else cf>=np.percentile(cf,100-p)
    print(f"      top {p:>3d}% conf: accuracy={accuracy_score(y3[m],pc[m]):.3f}", flush=True)
print("  Reference: Dr. Zhang reported 0.72 on this data (one 494-row holdout, no confidence ranking).", flush=True)
pd.DataFrame([
    {'task':'Track A regression (new molecule)','R2_all':round(r2_score(y,predA),3),'RMSE_all':round(rmse(y,predA),3)},
    {'task':'Track B regression (known molecule)','R2_all':round(r2_score(y,predB),3),'RMSE_all':round(rmse(y,predB),3)},
    {'task':'3-class (new molecule) accuracy_all','R2_all':round(accuracy_score(y3,pc),3),'RMSE_all':np.nan},
]).to_csv("zhang_data_results.csv", index=False)
print(f"\nsaved zhang_data_results.csv | total {(time.time()-START)/60:.1f} min", flush=True)
