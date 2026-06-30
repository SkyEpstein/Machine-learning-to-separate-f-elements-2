#!/usr/bin/env python3
"""
picks_trends.py — what do the extractants that active analysis chooses have in
common? Take greedy's top-20% picks on the per-pair delta G model (most negative
predicted delta G = strongest) and compare them to the rest on: which metals are
enriched, which interpretable extractant-structure descriptors differ most
(standardized mean difference / Cohen's d), and how concentrated the picks are in a
few molecules. These are the model's learned associations with strong extraction, to
be sanity-checked against known chemistry.
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
RkJ = 8.314e-3; SMI = 'SMILES_canonical'
tr = pd.read_csv("Training_Data_V27.csv", low_memory=False); te = pd.read_csv("Testing_Data_V39.csv", low_memory=False)
df = pd.concat([tr, te], ignore_index=True); df = df[df['Log_D'].notna() & df[SMI].notna()].reset_index(drop=True)
kdf = df[[SMI, 'Metal', 'Acid_type']].astype(str).copy()
for c, r in [('Acid_conc_M', 2), ('Temperature_K', 0), ('Extractant_conc_M', 3), ('Metal_conc_mM', 4)]:
    if c in df: kdf[c] = df[c].round(r).astype(str)
grng = pd.Series(df['Log_D'].values).groupby(kdf.agg('|'.join, axis=1)).transform(lambda v: v.max() - v.min()).values
df = df[(~df.duplicated().values) & (grng <= 2.0)].reset_index(drop=True)
df['dG'] = -2.302585 * RkJ * df['Temperature_K'].astype(float) * df['Log_D'].astype(float)
DROP = ['Extractant_conc_M', 'Temperature_K', 'Acid_conc_M', 'Metal_conc_mM', 'Volume_fraction_A', 'Volume_fraction_B']
METAL = ['Atomic_number', 'Melting_point_K', 'Boiling_point_K', 'Density_g/cm3', 'First_IE_kJ/mol', 'Second_IE_kJ/mol', 'Third_IE_kJ/mol', 'Matallic_radius_nm', 'Pauling_EN', 'Ionic_radius_nm', 'Oxidation_state']
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
fi = df.drop_duplicates('_pk').index
y = df.groupby('_pk')['dG'].mean().reindex(df.loc[fi, '_pk']).values
X = san(Xfull.loc[fi].values); groups = df.loc[fi, SMI].astype(str).values
metal = df.loc[fi, 'Metal'].astype(str).values
raw = Xfull.loc[fi].reset_index(drop=True)              # raw (un-sanitized) named features for interpretation
gkf = GroupKFold(5); pred = np.zeros(len(y))
for ti, vi in gkf.split(X, y, groups):
    pred[vi] = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=0).fit(X[ti], y[ti]).predict(X[vi])
k = int(0.20 * len(y)); picks = np.argsort(pred)[:k]; rest = np.argsort(pred)[k:]
pmask = np.zeros(len(y), bool); pmask[picks] = True
print(f"per-pair systems {len(y)}, top-20% picks = {k}; picks mean actual dG {y[picks].mean():.2f} vs rest {y[rest].mean():.2f} kJ/mol")
print(f"unique extractant molecules: picks {len(set(groups[picks]))} of {len(set(groups))} total; picks span {len(set(metal[picks]))} of {len(set(metal))} metals")
# metal enrichment
mc = pd.Series(metal[picks]).value_counts(normalize=True); ma = pd.Series(metal).value_counts(normalize=True)
enr = (mc / ma).dropna().sort_values(ascending=False)
print("\nMETAL enrichment in picks (share-in-picks / share-overall, >1 = over-represented):")
for mt, e in list(enr.items())[:8]: print(f"  {mt:<10} {e:.2f}x   ({mc.get(mt,0)*100:.0f}% of picks)")
# extractant structure trends (interpretable RDKit descriptors)
ext_cols = [c for c in raw.columns if c not in METAL and not c.endswith(' A') and not c.endswith(' B') and not c.startswith('fp_') and not c.startswith('acid_')]
rows = []
for c in ext_cols:
    v = raw[c].values.astype(float); a, b = v[pmask], v[~pmask]
    sd = np.sqrt((a.var() + b.var()) / 2)
    if sd > 1e-9 and np.isfinite(sd): rows.append((c, (a.mean() - b.mean()) / sd, a.mean(), b.mean()))
td = pd.DataFrame(rows, columns=['feature', 'cohens_d', 'picks_mean', 'rest_mean']).reindex(pd.DataFrame(rows, columns=['feature', 'cohens_d', 'picks_mean', 'rest_mean']).cohens_d.abs().sort_values(ascending=False).index)
print("\nEXTRACTANT-structure features most different in picks (Cohen's d, + = higher in picks):")
for _, r in td.head(14).iterrows(): print(f"  {r.feature:<22} d={r.cohens_d:+.2f}  picks {r.picks_mean:.2f} vs rest {r.rest_mean:.2f}")
enr.to_csv('picks_metal_enrichment.csv'); td.to_csv('picks_feature_trends.csv', index=False)
# figure
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
top = enr.head(10)[::-1]; ax[0].barh(range(len(top)), top.values, color='#1F4E79'); ax[0].set_yticks(range(len(top))); ax[0].set_yticklabels(top.index, fontsize=8)
ax[0].axvline(1, color='k', lw=0.7); ax[0].set_xlabel('enrichment (x over base rate)'); ax[0].set_title('Metals over-represented in the picks')
t12 = td.head(12)[::-1]; cols = ['#B5402E' if d < 0 else '#2E8B57' for d in t12.cohens_d]
ax[1].barh(range(len(t12)), t12.cohens_d.values, color=cols); ax[1].set_yticks(range(len(t12))); ax[1].set_yticklabels(t12.feature, fontsize=8)
ax[1].axvline(0, color='k', lw=0.7); ax[1].set_xlabel("Cohen's d (green = higher in picks, red = lower)"); ax[1].set_title('Extractant structure: what the picks share')
fig.tight_layout(); fig.savefig('figures/picks_trends.png', bbox_inches='tight', facecolor='white', dpi=120); plt.close(fig)
_im = Image.open('figures/picks_trends.png').convert('RGBA'); _bg = Image.new('RGB', _im.size, (255, 255, 255)); _bg.paste(_im, mask=_im.split()[3]); _bg.save('figures/picks_trends.png')
print("\nsaved picks_metal_enrichment.csv, picks_feature_trends.csv, figures/picks_trends.png")
