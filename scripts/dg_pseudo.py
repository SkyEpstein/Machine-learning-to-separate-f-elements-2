#!/usr/bin/env python3
"""
dg_pseudo.py — pseudo-labeling on the delta G model: the idea that the confident
predictions (narrow 90 percent interval) can be added as training rows without
running experiments. This tests whether that actually helps, honestly.

Molecules are split into a held-out test set and a training pool. The pool is split
into a small labeled seed and an unlabeled remainder (labels hidden). A RandomForest
is trained on the seed; the most confident unlabeled systems (lowest spread across
the forest's trees, the narrow-interval ones) are selected. Three models are then
compared on the same test set:
  baseline : trained on the labeled seed only.
  pseudo   : seed plus the confident unlabeled rows added with their PREDICTED delta
             G (no experiment) - the user's idea.
  oracle   : seed plus those same rows added with their TRUE delta G (i.e. if the
             experiments were actually run) - the upper bound.
The gap from pseudo to oracle is the value the real experiments would have added.
"""
import os, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
RkJ = 8.314e-3; SMI = 'SMILES_canonical'; rng = np.random.RandomState(0)
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
def fit_pred(Xtr, ytr, Xte):
    rf = RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=0).fit(Xtr, ytr); return rf
tr = pd.read_csv("Training_Data_V27.csv", low_memory=False); te = pd.read_csv("Testing_Data_V39.csv", low_memory=False)
df = pd.concat([tr, te], ignore_index=True)
df = df[df['Log_D'].notna() & df[SMI].notna()].reset_index(drop=True)
kdf = df[[SMI, 'Metal', 'Acid_type']].astype(str).copy()
for c, r in [('Acid_conc_M', 2), ('Temperature_K', 0), ('Extractant_conc_M', 3), ('Metal_conc_mM', 4)]:
    if c in df: kdf[c] = df[c].round(r).astype(str)
grng = pd.Series(df['Log_D'].values).groupby(kdf.agg('|'.join, axis=1)).transform(lambda v: v.max() - v.min()).values
df = df[(~df.duplicated().values) & (grng <= 2.0)].reset_index(drop=True)
df['dG'] = -2.302585 * RkJ * df['Temperature_K'].astype(float) * df['Log_D'].astype(float)
DROP = ['Extractant_conc_M', 'Temperature_K', 'Acid_conc_M', 'Metal_conc_mM', 'Volume_fraction_A', 'Volume_fraction_B']
num = df.select_dtypes(np.number).columns.tolist()
feat = [c for c in num if c not in DROP + ['Log_D', 'dG'] and not c.startswith('embedding_')]
acid = pd.get_dummies(df['Acid_type'].astype(str), prefix='acid').astype(float)
Xfull = pd.concat([df[feat], acid], axis=1)
def san(M):
    M = M.astype(np.float64).copy(); b = ~np.isfinite(M) | (np.abs(M) > 1e30)
    md = np.nan_to_num(np.nanmedian(np.where(b, np.nan, M), axis=0)); ix = np.where(b); M[ix] = np.take(md, ix[1])
    for j in np.where(np.nanmax(np.abs(M), axis=0) > 1e7)[0]: M[:, j] = np.sign(M[:, j]) * np.log1p(np.abs(M[:, j]))
    return M
df['_pk'] = df[[SMI, 'Metal', 'Acid_type', 'Solvent_A', 'Solvent_B']].astype(str).agg('|'.join, axis=1)
fi = df.drop_duplicates('_pk').index; pkf = df.loc[fi, '_pk']
y = df.groupby('_pk')['dG'].mean().reindex(pkf).values
X = san(Xfull.loc[fi].values); mol = df.loc[fi, SMI].values
# molecule-grouped split: test molecules, then labeled seed vs unlabeled pool
um = np.array(sorted(set(mol))); rng.shuffle(um)
test_mol = set(um[:int(0.25 * len(um))].tolist())
test = np.array([m in test_mol for m in mol]); pool = ~test
pidx = np.where(pool)[0]; rng.shuffle(pidx)
lab = pidx[:int(0.35 * len(pidx))]; unl = pidx[int(0.35 * len(pidx)):]
Xte, yte = X[test], y[test]
print(f"per-pair systems {len(y)} | test {test.sum()}, labeled seed {len(lab)}, unlabeled pool {len(unl)}")
base = fit_pred(X[lab], y[lab], Xte); base_p = base.predict(Xte)
# confidence on the unlabeled = low spread across the forest's trees (narrow interval)
tree_pred = np.stack([t.predict(X[unl]) for t in base.estimators_]); std = tree_pred.std(0); mu = tree_pred.mean(0)
rows = []
print(f"\n{'setting':<34}{'test R2':>9}{'test RMSE':>11}")
print(f"  {'baseline (labeled seed only)':<32}{r2_score(yte,base_p):>9.3f}{rmse(yte,base_p):>11.2f}")
rows.append({'setting': 'baseline (seed only)', 'add_confident_pct': 0, 'test_R2': round(r2_score(yte, base_p), 3), 'test_RMSE_kJmol': round(rmse(yte, base_p), 2)})
for conf_frac in [0.5, 1.0]:
    take = np.argsort(std)[:int(conf_frac * len(unl))]; sel = unl[take]
    Xp = np.vstack([X[lab], X[sel]])
    yp_pseudo = np.concatenate([y[lab], mu[take]])      # predicted labels (no experiment)
    yp_oracle = np.concatenate([y[lab], y[sel]])        # true labels (experiments run)
    pp = fit_pred(Xp, yp_pseudo, Xte).predict(Xte); po = fit_pred(Xp, yp_oracle, Xte).predict(Xte)
    tag = f"{int(conf_frac*100)}% most confident"
    print(f"  {('pseudo, +'+tag):<32}{r2_score(yte,pp):>9.3f}{rmse(yte,pp):>11.2f}")
    print(f"  {('oracle, +'+tag):<32}{r2_score(yte,po):>9.3f}{rmse(yte,po):>11.2f}")
    rows.append({'setting': 'pseudo (predicted labels)', 'add_confident_pct': int(conf_frac * 100), 'test_R2': round(r2_score(yte, pp), 3), 'test_RMSE_kJmol': round(rmse(yte, pp), 2)})
    rows.append({'setting': 'oracle (true labels)', 'add_confident_pct': int(conf_frac * 100), 'test_R2': round(r2_score(yte, po), 3), 'test_RMSE_kJmol': round(rmse(yte, po), 2)})
pd.DataFrame(rows).to_csv("dg_pseudo_results.csv", index=False)
print("\nsaved dg_pseudo_results.csv")
