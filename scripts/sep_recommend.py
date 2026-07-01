#!/usr/bin/env python3
"""
sep_recommend.py — separation recommender. Given a target f-element pair, recommend the
extractant and the conditions that best separate them, as a ranked shortlist for wet-lab
prioritization. Separation is logD_i - logD_j from the differenced logD model (the direct
delta model already lost the bake-off). For each candidate extractant we grid-search the
conditions (acid, concentration, temperature, extractant concentration, bounded to observed
ranges), keep the best, and rank by the OPTIMISTIC upper confidence bound
UCB = |predicted separation| + pair-uncertainty, so uncertain-but-promising candidates
surface for testing. Every row carries the confidence (pair uncertainty) and both bounds.

Candidates are pluggable: now the ~295 measured extractants; a molGen SMILES pool drops in
the same way (featurize to the same descriptor columns). HONEST FRAMING: the new-extractant
regime is the model's weakest (separation Spearman ~0.46, direction ~0.66), so this is a
triage shortlist, not a precise optimizer. Per-pair validation is printed so the trust level
is explicit.
"""
import os, time, warnings, itertools; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
import lightgbm as lgb
SMI = 'SMILES_canonical'; START = time.time()
tr = pd.read_csv("Training_Data_V27.csv", low_memory=False); te = pd.read_csv("Testing_Data_V39.csv", low_memory=False)
df = pd.concat([tr, te], ignore_index=True); df = df[df['Log_D'].notna() & df[SMI].notna()].reset_index(drop=True)
kc = df[[SMI, 'Metal', 'Acid_type']].astype(str).copy()
for c, r in [('Acid_conc_M', 2), ('Temperature_K', 0), ('Extractant_conc_M', 3), ('Metal_conc_mM', 4)]: kc[c] = df[c].round(r).astype(str)
grng = pd.Series(df['Log_D'].values).groupby(kc.agg('|'.join, axis=1)).transform(lambda v: v.max() - v.min()).values
df = df[(~df.duplicated().values) & (grng <= 2.0)].reset_index(drop=True)
METAL = ['Atomic_number', 'Melting_point_K', 'Boiling_point_K', 'Density_g/cm3', 'First_IE_kJ/mol', 'Second_IE_kJ/mol', 'Third_IE_kJ/mol', 'Matallic_radius_nm', 'Pauling_EN', 'Ionic_radius_nm', 'Oxidation_state', 'Metal_conc_mM']
COND = ['Extractant_conc_M', 'Acid_conc_M', 'Temperature_K', 'Dipole_moment_D', 'Volume_fraction_A', 'Volume_fraction_B', 'Molar_mass(g/mol) A', 'Log_P A', 'Boiling_point(K) A', 'Melting_point(K) A', 'Density(g/mL) A', 'Solubility_in_water(g/L) A', 'Molar_mass(g/mol) B', 'Log_P B', 'Boiling_point(K) B', 'Melting_point(K) B', 'Density(g/mL) B', 'Solubility_in_water(g/L) B']
COND_TUNE = ['Acid_conc_M', 'Temperature_K', 'Extractant_conc_M']
COND_FIX = [c for c in COND if c not in COND_TUNE]
num = df.select_dtypes(np.number).columns.tolist()
EXT = [c for c in num if c not in METAL + COND + ['Log_D'] and not c.startswith('fp_') and not c.startswith('embedding_')]
acid = pd.get_dummies(df['Acid_type'].astype(str), prefix='acid').astype(float); ACIDCOLS = list(acid.columns)
FEAT = EXT + COND + METAL + ACIDCOLS
def fit_san(M):
    M = M.astype(float); fin = np.where(np.isfinite(M) & (np.abs(M) < 1e30), M, np.nan)
    med = np.nan_to_num(np.nanmedian(fin, axis=0)); logc = np.where(np.nanmax(np.abs(fin), axis=0) > 1e7)[0]
    return {'med': med, 'logc': logc}
def apply_san(p, M):
    M = M.astype(float).copy(); b = ~np.isfinite(M) | (np.abs(M) > 1e30); ix = np.where(b); M[ix] = np.take(p['med'], ix[1])
    for j in p['logc']: M[:, j] = np.sign(M[:, j]) * np.log1p(np.abs(M[:, j]))
    return M
rawX = pd.concat([df[EXT + COND + METAL], acid], axis=1)[FEAT]
P = fit_san(rawX.values); X = apply_san(P, rawX.values)
logD = df['Log_D'].values.astype(float); smiles = df[SMI].astype(str).values
def lgbm(): return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63, subsample=0.8, colsample_bytree=0.7, subsample_freq=1, random_state=0, n_jobs=-1, verbosity=-1)
# molecule-grouped OOF for honest validation + residuals
oof = np.zeros(len(df))
for ti, vi in GroupKFold(5).split(X, logD, smiles): oof[vi] = lgbm().fit(X[ti], logD[ti]).predict(X[vi])
resid = np.abs(logD - oof)
# deployable full-data models: logD + error (confidence)
M_logD = lgbm().fit(X, logD); M_err = lgbm().fit(X, resid)
print(f"logD model OOF R2={1 - np.var(logD - oof) / np.var(logD):.3f}; deployable models trained | {(time.time()-START)/60:.1f} min")
# lookups
metal_desc = df.groupby('Metal')[METAL].median()
ext_fix = df.groupby(SMI).agg({**{c: 'median' for c in EXT + COND_FIX}}); ext_fix = ext_fix[EXT + COND_FIX]
ext_acid = df.groupby(SMI)['Acid_type'].agg(lambda s: s.astype(str).mode().iat[0])  # typical acid per extractant
# conditions grid, bounded to observed ranges
def qgrid(col, n): return list(np.round(np.quantile(df[col].dropna(), np.linspace(0.1, 0.9, n)), 3))
GRID_ACIDCONC = qgrid('Acid_conc_M', 5); GRID_EXTCONC = qgrid('Extractant_conc_M', 4)
GRID_TEMP = sorted(df['Temperature_K'].round(0).value_counts().head(2).index.tolist())
GRID_ACIDTYPE = df['Acid_type'].astype(str).value_counts().head(3).index.tolist()
grid = pd.DataFrame([{'Acid_conc_M': a, 'Temperature_K': t, 'Extractant_conc_M': e, 'Acid_type': ac}
                     for a in GRID_ACIDCONC for t in GRID_TEMP for e in GRID_EXTCONC for ac in GRID_ACIDTYPE])
print(f"conditions grid: {len(grid)} points ({len(GRID_ACIDCONC)} acid_conc x {len(GRID_TEMP)} temp x {len(GRID_EXTCONC)} ext_conc x {len(GRID_ACIDTYPE)} acid)")

def score_rows(cands, g, metal):
    """feature matrix for cands x grid at a fixed metal, columns = FEAT."""
    nC, nG = len(cands), len(g)
    base = pd.DataFrame(index=range(nC * nG))
    ci = np.repeat(np.arange(nC), nG); gi = np.tile(np.arange(nG), nC)
    for c in EXT + COND_FIX: base[c] = ext_fix.loc[cands, c].values[ci]
    for c in COND_TUNE: base[c] = g[c].values[gi]
    for c in METAL: base[c] = metal_desc.loc[metal, c]
    for c in ACIDCOLS: base[c] = 0.0
    at = g['Acid_type'].values[gi]
    for k, acol in enumerate(ACIDCOLS):
        nm = acol.replace('acid_', ''); base.loc[at == nm, acol] = 1.0
    return apply_san(P, base[FEAT].values), ci, gi

def recommend(mi, mj, topn=15):
    cands = list(ext_fix.index)
    Xi, ci, gi = score_rows(cands, grid, mi); Xj, _, _ = score_rows(cands, grid, mj)
    li, lj = M_logD.predict(Xi), M_logD.predict(Xj)
    ei, ej = M_err.predict(Xi), M_err.predict(Xj)
    sep = li - lj; unc = np.sqrt(np.clip(ei, 1e-6, None) ** 2 + np.clip(ej, 1e-6, None) ** 2)
    ucb = np.abs(sep) + unc  # optimistic upper bound (explore)
    R = pd.DataFrame({'ci': ci, 'gi': gi, 'sep': sep, 'absep': np.abs(sep), 'unc': unc, 'ucb': ucb})
    best = R.loc[R.groupby('ci')['ucb'].idxmax()].reset_index(drop=True)  # best conditions per candidate by UCB
    best['SMILES'] = [cands[i] for i in best['ci']]
    for c in COND_TUNE + ['Acid_type']: best[c] = grid[c].values[best['gi'].values]
    best['lcb'] = np.clip(best['absep'] - best['unc'], 0, None)
    # measured separation for reference (max |dlogD| actually observed for this pair on that extractant)
    sub = df[df['Metal'].isin([mi, mj])]
    meas = {}
    for smi_, gsub in sub.groupby(SMI):
        key = gsub[['Acid_type', 'Acid_conc_M', 'Temperature_K', 'Extractant_conc_M']].round(3).astype(str).agg('|'.join, axis=1)
        best_m = 0.0; ok = False
        for _, idx in gsub.assign(k=key.values).groupby('k').groups.items():
            g2 = gsub.loc[idx]
            if set(g2['Metal']) >= {mi, mj}:
                d = abs(g2[g2.Metal == mi]['Log_D'].mean() - g2[g2.Metal == mj]['Log_D'].mean()); best_m = max(best_m, d); ok = True
        if ok: meas[smi_] = round(best_m, 2)
    best['measured_absep'] = best['SMILES'].map(meas)
    cols = ['SMILES', 'Acid_type', 'Acid_conc_M', 'Temperature_K', 'Extractant_conc_M', 'sep', 'absep', 'unc', 'ucb', 'lcb', 'measured_absep']
    # EXPLORE: uncertain-but-PROMISING (predicted separation factor >= 10) ranked by optimistic UCB
    explore = best[best['absep'] >= 1.0].sort_values('ucb', ascending=False).reset_index(drop=True)
    explore.insert(0, 'rank', np.arange(1, len(explore) + 1))
    # CONFIDENT best-bets: high predicted separation the model is SURE about, ranked by lower bound
    conf = best.sort_values('lcb', ascending=False).reset_index(drop=True)
    conf.insert(0, 'rank', np.arange(1, len(conf) + 1))
    return explore[['rank'] + cols], conf[['rank'] + cols]

def validate(mi, mj):
    """honest ranking quality on current data: OOF-predicted vs measured separation at matched conditions."""
    sub = df[df['Metal'].isin([mi, mj])].copy(); sub['_p'] = oof[sub.index.values]
    sub['k'] = sub[[SMI, 'Acid_type', 'Acid_conc_M', 'Temperature_K', 'Extractant_conc_M']].round(3).astype(str).agg('|'.join, axis=1)
    rows = []
    for k, g in sub.groupby('k'):
        if set(g['Metal']) >= {mi, mj}:
            mm = g[g.Metal == mi]; nn = g[g.Metal == mj]
            rows.append((mm['Log_D'].mean() - nn['Log_D'].mean(), mm['_p'].mean() - nn['_p'].mean(), k.split('|')[0]))
    if len(rows) < 8: return f"  validation: only {len(rows)} matched systems, too few"
    a = np.array([r[0] for r in rows]); p = np.array([r[1] for r in rows])
    rho = spearmanr(np.abs(a), np.abs(p)).correlation; dacc = float((np.sign(a) == np.sign(p)).mean())
    return f"  validation ({len(rows)} matched systems): direction {dacc:.2f}, |sep| Spearman {rho:.2f} -> {'usable triage' if rho>0.2 else 'weak, low trust'}"

def shortsmi(dfx, n=48):
    d = dfx.copy(); d['SMILES'] = d['SMILES'].str.slice(0, n) + np.where(d['SMILES'].str.len() > n, '...', ''); return d
PAIRS = [('Dy(III)', 'Nd(III)'), ('Eu(III)', 'Gd(III)'), ('Am(III)', 'Eu(III)')]
allout = []
for mi, mj in PAIRS:
    if mi not in metal_desc.index or mj not in metal_desc.index:
        print(f"\n### {mi} vs {mj}: one metal absent, skipped"); continue
    explore, conf = recommend(mi, mj)
    for d, kind in [(explore, 'explore'), (conf, 'confident')]:
        d = d.copy(); d.insert(1, 'pair', f'{mi}/{mj}'); d.insert(2, 'list', kind); allout.append(d)
    print(f"\n### {mi} vs {mj}")
    print(validate(mi, mj))
    print("  EXPLORE (uncertain but promising, |pred sep|>=1, ranked by optimistic UCB):")
    print(shortsmi(explore).head(5).to_string(index=False))
    print("  CONFIDENT best-bets (ranked by lower bound |sep|-uncertainty):")
    print(shortsmi(conf).head(5).to_string(index=False))
pd.concat(allout, ignore_index=True).to_csv("sep_recommend_shortlists.csv", index=False)
print(f"\nsaved sep_recommend_shortlists.csv | {(time.time()-START)/60:.1f} min")
