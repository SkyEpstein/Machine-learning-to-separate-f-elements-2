#!/usr/bin/env python3
"""
ecm.py — ECM model. Predict the Gibbs free energy of extraction (delta G) instead
of logD, from the extractant structure and the metal alone, with the reaction
conditions dropped. Per the mentor's framing, delta G is the thermodynamic target
and does not need the reaction conditions.

delta G is computed from logD by the standard relation
    delta G = -2.303 * R * T * logD      (kJ/mol, R = 8.314e-3 kJ/mol/K, T in K)
so a favorable extraction (logD > 0) gives a negative, spontaneous delta G.

Two framings are compared, which is the per-row vs condition-independent choice:
  per-row  : one delta G per measurement, conditions dropped. Same-structure rows
             repeat with different delta G, so the within-system spread is
             irreducible noise the structure features cannot explain.
  per-pair : one mean delta G per (extractant, metal, acid, diluent) system,
             condition-independent by construction and the cleaner target.
Both use molecule-grouped cross-validation (a new extractant never appears in both
train and test) and report R2 and RMSE plus a confidence curve. Features are the
extractant fingerprint and RDKit descriptors, the metal descriptors, the acid type,
and the diluent descriptors; all concentration, temperature, and volume-fraction
knobs are dropped.
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
RkJ = 8.314e-3; START = time.time(); SMI = 'SMILES_canonical'
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
tr = pd.read_csv("Training_Data_V27.csv", low_memory=False); te = pd.read_csv("Testing_Data_V39.csv", low_memory=False)
df = pd.concat([tr, te], ignore_index=True)
df = df[df['Log_D'].notna() & df[SMI].notna()].reset_index(drop=True)
# clean: drop exact duplicates and replicate groups disagreeing by > 2 log units
kdf = df[[SMI, 'Metal', 'Acid_type']].astype(str).copy()
for c, r in [('Acid_conc_M', 2), ('Temperature_K', 0), ('Extractant_conc_M', 3), ('Metal_conc_mM', 4)]:
    if c in df: kdf[c] = df[c].round(r).astype(str)
grng = pd.Series(df['Log_D'].values).groupby(kdf.agg('|'.join, axis=1)).transform(lambda v: v.max() - v.min()).values
df = df[(~df.duplicated().values) & (grng <= 2.0)].reset_index(drop=True)
# target: Gibbs free energy of extraction (kJ/mol)
df['dG'] = -2.302585 * RkJ * df['Temperature_K'].astype(float) * df['Log_D'].astype(float)
# features: drop the reaction-condition knobs, the targets/ids, and the embeddings
DROP_COND = ['Extractant_conc_M', 'Temperature_K', 'Acid_conc_M', 'Metal_conc_mM', 'Volume_fraction_A', 'Volume_fraction_B']
numcols = df.select_dtypes(np.number).columns.tolist()
feat = [c for c in numcols if c not in DROP_COND + ['Log_D', 'dG'] and not c.startswith('embedding_')]
acid = pd.get_dummies(df['Acid_type'].astype(str), prefix='acid')
Xfull = pd.concat([df[feat], acid.astype(float)], axis=1)
print(f"rows {len(df)} | features {Xfull.shape[1]} (dropped conditions: {', '.join(DROP_COND)})")
def san(M):
    M = M.astype(np.float64).copy(); b = ~np.isfinite(M) | (np.abs(M) > 1e30)
    md = np.nan_to_num(np.nanmedian(np.where(b, np.nan, M), axis=0)); ix = np.where(b); M[ix] = np.take(md, ix[1])
    for j in np.where(np.nanmax(np.abs(M), axis=0) > 1e7)[0]: M[:, j] = np.sign(M[:, j]) * np.log1p(np.abs(M[:, j]))
    return M
def cv_eval(X, y, grp, label):
    gkf = GroupKFold(5); oof = np.zeros(len(y))
    for ti, vi in gkf.split(X, y, grp):
        m = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=0).fit(X[ti], y[ti]); oof[vi] = m.predict(X[vi])  # RandomForest: best model from the ensemble sweep (ecm_ensemble.py)
    err = np.zeros(len(y)); resid = np.abs(y - oof)
    for ti, vi in gkf.split(X, y, grp):
        em = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, random_state=0, n_jobs=-1, verbosity=-1).fit(X[ti], resid[ti]); err[vi] = em.predict(X[vi])
    conf = []
    for q in [100, 50, 25, 10]:
        mm = np.ones(len(y), bool) if q == 100 else err <= np.percentile(err, q)
        conf.append((q, r2_score(y[mm], oof[mm]), rmse(y[mm], oof[mm]), int(mm.sum())))
    print(f"\n{label}: n={len(y)}, dG {y.min():.1f}..{y.max():.1f} kJ/mol (mean {y.mean():.1f})")
    print(f"   all          R2={conf[0][1]:.3f}  RMSE={conf[0][2]:.2f} kJ/mol")
    for q, r, rm, n in conf[1:]:
        print(f"   top {q:>2d}% conf  R2={r:.3f}  RMSE={rm:.2f} kJ/mol  (n={n})")
    return conf, oof
# framing 1: per row
Xr = san(Xfull.values); yr = df['dG'].values; groups = df[SMI].values
confr, oofr = cv_eval(Xr, yr, groups, "PER-ROW dG (framing 1, conditions dropped)")
# framing 3: per (extractant, metal, acid, diluent) system, mean dG
df['_pk'] = df[[SMI, 'Metal', 'Acid_type', 'Solvent_A', 'Solvent_B']].astype(str).agg('|'.join, axis=1)
first_idx = df.drop_duplicates('_pk').index
pk_first = df.loc[first_idx, '_pk']
yp = df.groupby('_pk')['dG'].mean().reindex(pk_first).values
Xp = san(Xfull.loc[first_idx].values); grp_p = df.loc[first_idx, SMI].values
confp, oofp = cv_eval(Xp, yp, grp_p, "PER-PAIR mean dG (framing 3, condition-independent)")
# record results
rows = []
for lab, conf in [("per-row (framing 1)", confr), ("per-pair (framing 3)", confp)]:
    rows.append({'framing': lab, 'n': conf[0][3], 'R2_all': round(conf[0][1], 3), 'RMSE_all_kJmol': round(conf[0][2], 2),
                 'R2_top25': round(conf[2][1], 3), 'R2_top10': round(conf[3][1], 3), 'RMSE_top10_kJmol': round(conf[3][2], 2)})
pd.DataFrame(rows).to_csv("ecm_results.csv", index=False)
out = df.loc[first_idx, [SMI, 'Metal', 'Acid_type']].copy(); out['dG_actual_kJmol'] = np.round(yp, 2); out['dG_pred_kJmol'] = np.round(oofp, 2)
out.to_csv("ecm_perpair_predictions.csv", index=False)
# figure
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].scatter(yp, oofp, s=6, alpha=0.3, color='#1F4E79', edgecolors='none')
lim = [min(yp.min(), oofp.min()), max(yp.max(), oofp.max())]; ax[0].plot(lim, lim, color='#B5402E', lw=1)
ax[0].set_xlabel('actual dG (kJ/mol)'); ax[0].set_ylabel('predicted dG (kJ/mol)'); ax[0].set_title(f'Per-pair dG, new extractants (R2={confp[0][1]:.2f})')
labels = ['per-row\nall', 'per-row\ntop10%', 'per-pair\nall', 'per-pair\ntop10%']; vals = [confr[0][1], confr[3][1], confp[0][1], confp[3][1]]
ax[1].bar(labels, vals, color=['#999999', '#999999', '#2E8B57', '#2E8B57']); ax[1].set_ylabel('R-squared'); ax[1].set_title('ECM accuracy by framing and confidence'); ax[1].axhline(0, color='k', lw=0.6)
fig.tight_layout(); fig.savefig('figures/ecm.png', bbox_inches='tight', facecolor='white'); plt.close(fig)
_im = Image.open('figures/ecm.png').convert('RGBA'); _bg = Image.new('RGB', _im.size, (255, 255, 255)); _bg.paste(_im, mask=_im.split()[3]); _bg.save('figures/ecm.png')
print(f"\nsaved ecm_results.csv, ecm_perpair_predictions.csv, figures/ecm.png | {(time.time()-START)/60:.1f} min")
