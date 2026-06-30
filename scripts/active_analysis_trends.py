#!/usr/bin/env python3
"""
active_analysis_trends.py — is active analysis choosing genuinely good extractants or
just mid ones? On the per-pair delta G model (RandomForest, molecule-grouped CV), a
favorable extractant is the most negative delta G. For each acquisition rule (greedy
by prediction, UCB = prediction minus uncertainty, random) we look past the mean at:
  1. precision@K   : of the top-K picked, what share are truly in the best decile.
  2. quality trend : cumulative mean actual delta G of the top-K vs the optimal pick
                     and the population mean, so we see if picks stay strong or
                     regress toward the middle as the budget grows.
  3. composition   : of the top-10% picked, the split into good / mid / weak by actual
                     tertile, the direct good-vs-mid answer.
  4. per metal     : within each metal, does ranking by prediction surface that metal's
                     strong extractants (top-20% mean vs that metal's mean).
Saves result CSVs and a 4-panel figure.
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
RkJ = 8.314e-3; SMI = 'SMILES_canonical'; rng = np.random.RandomState(0)
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
metal = df.loc[fi, 'Metal'].astype(str).values
gkf = GroupKFold(5); pred = np.zeros(len(y)); err = np.zeros(len(y))
for ti, vi in gkf.split(X, y, groups):
    pred[vi] = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=0).fit(X[ti], y[ti]).predict(X[vi])
resid = np.abs(y - pred)
for ti, vi in gkf.split(X, y, groups):
    err[vi] = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, random_state=0, n_jobs=-1, verbosity=-1).fit(X[ti], resid[ti]).predict(X[vi])
n = len(y); pop_mean = y.mean()
best_decile = set(np.argsort(y)[:int(0.10 * n)].tolist())     # truly best 10% (most negative)
t1, t2 = np.percentile(y, 100 / 3), np.percentile(y, 200 / 3)  # tertile cuts for good/mid/weak
def tier(v): return 'good' if v <= t1 else ('mid' if v <= t2 else 'weak')
scores = {'greedy': pred, 'UCB': pred - 1.0 * err, 'random': rng.rand(n)}
order = {m: np.argsort(s) for m, s in scores.items()}         # most favorable first (smallest score)
opt = np.argsort(y)                                            # optimal order by actual
Ks = np.arange(0.02, 0.301, 0.02)
prec, cmean = {m: [] for m in scores}, {m: [] for m in scores}; optmean = []
for f in Ks:
    k = max(1, int(f * n)); optmean.append(float(y[opt[:k]].mean()))
    for m in scores:
        sel = order[m][:k]
        prec[m].append(len(set(sel.tolist()) & best_decile) / k)
        cmean[m].append(float(y[sel].mean()))
# composition of the top-10% picks
k10 = int(0.10 * n); comp = {}
for m in scores:
    sel = order[m][:k10]; ts = [tier(y[i]) for i in sel]
    comp[m] = {t: ts.count(t) / k10 for t in ['good', 'mid', 'weak']}
# per-metal greedy (within metal, top 20% by pred): mean actual vs metal mean
rows_metal = []
for mt in sorted(set(metal)):
    idx = np.where(metal == mt)[0]
    if len(idx) < 5: continue
    k = max(1, int(0.2 * len(idx))); sub = idx[np.argsort(pred[idx])[:k]]
    rows_metal.append({'metal': mt, 'n': len(idx), 'metal_mean_dG': round(float(y[idx].mean()), 2),
                       'greedy_top20_mean_dG': round(float(y[sub].mean()), 2), 'metal_best_mean_dG': round(float(np.sort(y[idx])[:k].mean()), 2)})
md = pd.DataFrame(rows_metal)
# --- honest framing metrics (added after adversarial verification) ---
bd_cut = float(np.sort(y)[int(0.1 * n)])                 # best-decile cut = genuinely strong
opt10 = float(y[opt[:k10]].mean())
strong = {m: float(np.mean(y[order[m][:k10]] <= bd_cut)) for m in scores}   # strong-tail hit rate at top-10%
depth = {m: round(cmean[m][4] / opt10, 2) for m in scores}                  # fraction of optimal cumulative depth at top-10%
g10 = order['greedy'][:k10]; emed = float(np.median(err))
gconf = g10[err[g10] <= emed]; gunc = g10[err[g10] > emed]
gate = {'good_conf': round(float(np.mean([tier(y[i]) == 'good' for i in gconf])) if len(gconf) else 0, 2),
        'good_unc': round(float(np.mean([tier(y[i]) == 'good' for i in gunc])) if len(gunc) else 0, 2),
        'strong_conf': round(float(np.mean(y[gconf] <= bd_cut)) if len(gconf) else 0, 2),
        'strong_unc': round(float(np.mean(y[gunc] <= bd_cut)) if len(gunc) else 0, 2)}
big = md[md.n >= 20]; wins = int((big.greedy_top20_mean_dG < big.metal_mean_dG).sum()); tot = len(big)
fails = big[big.greedy_top20_mean_dG >= big.metal_mean_dG]['metal'].tolist()
weak_ceiling = int((md.metal_best_mean_dG > -10).sum())
print("\n--- honest framing ---")
print(f"  strong-tail cut (best decile) = {bd_cut:.2f} kJ/mol; population mean = {pop_mean:.2f}")
print("  strong-tail hit rate (top-10% picks clearing the cut): " + ", ".join(f"{m} {strong[m]:.2f}" for m in scores))
print("  fraction of optimal depth captured at top-10%: " + ", ".join(f"{m} {depth[m]}" for m in scores))
print(f"  greedy beats metal mean for {wins}/{tot} metals (n>=20); loses: {fails}; metals whose own best is still weak (> -10): {weak_ceiling}")
print(f"  greedy top-10%, confident half vs uncertain half: good {gate['good_conf']} vs {gate['good_unc']}, strong {gate['strong_conf']} vs {gate['strong_unc']}")
pd.DataFrame([{'method': m, 'good_tertile_share': round(comp[m]['good'], 2), 'strong_tail_share': round(strong[m], 2),
               'precision_at10': round(prec[m][4], 3), 'depth_captured': depth[m]} for m in scores]).to_csv('active_analysis_honest.csv', index=False)
# save
pd.DataFrame({'K_pct': (Ks * 100).round(0), 'optimal_mean_dG': np.round(optmean, 2),
              **{f'{m}_precision_at_K': np.round(prec[m], 3) for m in scores},
              **{f'{m}_mean_dG': np.round(cmean[m], 2) for m in scores}}).to_csv('active_analysis_trends.csv', index=False)
pd.DataFrame([{'method': m, **{t: round(comp[m][t], 3) for t in ['good', 'mid', 'weak']}} for m in scores]).to_csv('active_analysis_composition.csv', index=False)
md.to_csv('active_analysis_by_metal.csv', index=False)
print(f"n={n}, pop mean dG={pop_mean:.2f}, best-decile cut={np.sort(y)[int(0.1*n)]:.2f}, tertiles {t1:.2f}/{t2:.2f}")
print("composition of top-10% picks (good/mid/weak):")
for m in scores: print(f"  {m:<8} good={comp[m]['good']:.2f} mid={comp[m]['mid']:.2f} weak={comp[m]['weak']:.2f}")
print("precision@10% (share truly in best decile):", {m: round(prec[m][4], 3) for m in scores})
col = {'greedy': '#B5402E', 'UCB': '#1F4E79', 'random': '#999999'}
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
for m in scores: ax[0, 0].plot(Ks * 100, prec[m], marker='o', ms=3, color=col[m], label=m)
ax[0, 0].set_xlabel('% selected (K)'); ax[0, 0].set_ylabel('precision (share in true best 10%)'); ax[0, 0].set_title('Precision@K: are the picks truly top-tier?'); ax[0, 0].legend(fontsize=9)
for m in scores: ax[0, 1].plot(Ks * 100, cmean[m], marker='o', ms=3, color=col[m], label=m)
ax[0, 1].plot(Ks * 100, optmean, '--', color='#2E8B57', label='optimal'); ax[0, 1].axhline(pop_mean, color='k', lw=0.7, ls=':', label='population mean')
ax[0, 1].set_xlabel('% selected (K)'); ax[0, 1].set_ylabel('mean actual delta G (kJ/mol)'); ax[0, 1].set_title('Quality of picks vs optimal (lower = stronger)'); ax[0, 1].legend(fontsize=8)
x = np.arange(3); w = 0.25
for i, t in enumerate(['good', 'mid', 'weak']):
    ax[1, 0].bar(x + i * w, [comp[m][t] for m in scores], w, label=t, color=['#2E8B57', '#C9A227', '#999999'][i])
ax[1, 0].set_xticks(x + w); ax[1, 0].set_xticklabels(list(scores)); ax[1, 0].set_ylabel('share of top-10% picks'); ax[1, 0].set_title('Composition of the top-10% picks'); ax[1, 0].legend(fontsize=9)
mm = md.sort_values('metal_mean_dG')
xm = np.arange(len(mm)); ax[1, 1].bar(xm - 0.2, mm['metal_mean_dG'], 0.4, color='#999999', label='metal mean')
ax[1, 1].bar(xm + 0.2, mm['greedy_top20_mean_dG'], 0.4, color='#B5402E', label='greedy top-20% mean')
ax[1, 1].set_xticks(xm); ax[1, 1].set_xticklabels(mm['metal'], rotation=90, fontsize=7); ax[1, 1].set_ylabel('mean actual delta G (kJ/mol)'); ax[1, 1].set_title('Per metal: does it find each metal\'s strong extractants?'); ax[1, 1].legend(fontsize=8)
fig.tight_layout(); fig.savefig('figures/active_analysis_trends.png', bbox_inches='tight', facecolor='white', dpi=120); plt.close(fig)
_im = Image.open('figures/active_analysis_trends.png').convert('RGBA'); _bg = Image.new('RGB', _im.size, (255, 255, 255)); _bg.paste(_im, mask=_im.split()[3]); _bg.save('figures/active_analysis_trends.png')
print("saved active_analysis_trends.csv, active_analysis_composition.csv, active_analysis_by_metal.csv, figures/active_analysis_trends.png")
