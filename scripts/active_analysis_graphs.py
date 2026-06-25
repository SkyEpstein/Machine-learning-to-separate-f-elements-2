#!/usr/bin/env python3
"""The active-analysis graphs for the delta G model: (1) calibration, target vs
empirical coverage of the conformal intervals; (2) selective prediction, R2 and RMSE
as you keep only the most confident fraction; (3) predicted vs actual with the most
confident quarter highlighted. RandomForest + LightGBM err, molecule-grouped CV."""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
RkJ = 8.314e-3; SMI = 'SMILES_canonical'; rng = np.random.RandomState(0)
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
tr = pd.read_csv("Training_Data_V27.csv", low_memory=False); te = pd.read_csv("Testing_Data_V39.csv", low_memory=False)
df = pd.concat([tr, te], ignore_index=True); df = df[df['Log_D'].notna() & df[SMI].notna()].reset_index(drop=True)
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
def san(M):
    M = M.astype(np.float64).copy(); b = ~np.isfinite(M) | (np.abs(M) > 1e30)
    md = np.nan_to_num(np.nanmedian(np.where(b, np.nan, M), axis=0)); ix = np.where(b); M[ix] = np.take(md, ix[1])
    for j in np.where(np.nanmax(np.abs(M), axis=0) > 1e7)[0]: M[:, j] = np.sign(M[:, j]) * np.log1p(np.abs(M[:, j]))
    return M
df['_pk'] = df[[SMI, 'Metal', 'Acid_type', 'Solvent_A', 'Solvent_B']].astype(str).agg('|'.join, axis=1)
fi = df.drop_duplicates('_pk').index; pkf = df.loc[fi, '_pk']
y = df.groupby('_pk')['dG'].mean().reindex(pkf).values
X = san(pd.concat([df[feat], acid], axis=1).loc[fi].values); groups = df.loc[fi, SMI].values
gkf = GroupKFold(5); pred = np.zeros(len(y)); err = np.zeros(len(y))
for ti, vi in gkf.split(X, y, groups):
    pred[vi] = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=0).fit(X[ti], y[ti]).predict(X[vi])
resid = np.abs(y - pred)
for ti, vi in gkf.split(X, y, groups):
    err[vi] = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, random_state=0, n_jobs=-1, verbosity=-1).fit(X[ti], resid[ti]).predict(X[vi])
err = np.clip(err, 1e-6, None); s = resid / err; um = np.array(sorted(set(groups)))
# calibration across targets
targets = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]; emp = []
for a in targets:
    cs = []
    for _ in range(30):
        rng.shuffle(um); cal = np.array([g in set(um[:len(um) // 2].tolist()) for g in groups]); ev = ~cal
        q = np.quantile(s[cal], a); cs.append(float((resid[ev] <= q * err[ev]).mean()))
    emp.append(np.mean(cs))
# selective prediction
order = np.argsort(err); fracs = np.arange(0.1, 1.001, 0.1)
r2s = [r2_score(y[order[:int(f * len(y))]], pred[order[:int(f * len(y))]]) for f in fracs]
rms = [rmse(y[order[:int(f * len(y))]], pred[order[:int(f * len(y))]]) for f in fracs]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.7))
ax[0].plot([0.5, 1], [0.5, 1], '--', color='#999999', lw=1, label='perfect')
ax[0].plot(targets, emp, marker='o', color='#1F4E79', label='delta G intervals')
ax[0].set_xlabel('target coverage'); ax[0].set_ylabel('empirical coverage'); ax[0].set_title('Interval calibration'); ax[0].legend(fontsize=9)
ax1b = ax[1].twinx()
ax[1].plot(fracs * 100, r2s, marker='o', color='#1F4E79'); ax1b.plot(fracs * 100, rms, marker='s', color='#B5402E')
ax[1].set_xlabel('% of predictions kept (most confident first)'); ax[1].set_ylabel('R-squared', color='#1F4E79'); ax1b.set_ylabel('RMSE (kJ/mol)', color='#B5402E')
ax[1].set_title('Accuracy vs how much you keep'); ax[1].invert_xaxis()
c25 = err <= np.percentile(err, 25)
ax[2].scatter(y[~c25], pred[~c25], s=7, alpha=0.25, color='#999999', edgecolors='none', label='rest')
ax[2].scatter(y[c25], pred[c25], s=9, alpha=0.6, color='#1F4E79', edgecolors='none', label='most confident 25%')
lim = [min(y.min(), pred.min()), max(y.max(), pred.max())]; ax[2].plot(lim, lim, color='#B5402E', lw=1)
ax[2].set_xlabel('actual delta G (kJ/mol)'); ax[2].set_ylabel('predicted delta G (kJ/mol)'); ax[2].set_title('Predicted vs actual'); ax[2].legend(fontsize=9)
fig.tight_layout(); fig.savefig('figures/active_analysis.png', bbox_inches='tight', facecolor='white', dpi=130); plt.close(fig)
_im = Image.open('figures/active_analysis.png').convert('RGBA'); _bg = Image.new('RGB', _im.size, (255, 255, 255)); _bg.paste(_im, mask=_im.split()[3]); _bg.save('figures/active_analysis.png')
print("saved figures/active_analysis.png  | calibration:", [f"{t}->{e:.2f}" for t, e in zip(targets, emp)])
