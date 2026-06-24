#!/usr/bin/env python3
"""
make_figures.py — plots for everything in the results workbook. Reads the saved
result files (deploy predictions, metal and pair tables, classifier results, and
the Zhang-data results if present) and writes one PNG per figure into figures/
plus a combined all_figures.pdf. Both R2 and RMSE are shown wherever they apply.
"""
import os, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import r2_score, mean_squared_error
plt.rcParams.update({'figure.dpi': 130, 'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3, 'axes.axisbelow': True})
os.makedirs("figures", exist_ok=True); pdf = PdfPages("figures/all_figures.pdf"); saved = []
def finish(fig, name):
    fig.tight_layout(); fig.savefig(f"figures/{name}.png", bbox_inches='tight'); pdf.savefig(fig); plt.close(fig); saved.append(name)
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
import re
NAMES = {'La': 'Lanthanum', 'Ce': 'Cerium', 'Pr': 'Praseodymium', 'Nd': 'Neodymium', 'Pm': 'Promethium', 'Sm': 'Samarium', 'Eu': 'Europium', 'Gd': 'Gadolinium', 'Tb': 'Terbium', 'Dy': 'Dysprosium', 'Ho': 'Holmium', 'Er': 'Erbium', 'Tm': 'Thulium', 'Yb': 'Ytterbium', 'Lu': 'Lutetium', 'Th': 'Thorium', 'Pa': 'Protactinium', 'U': 'Uranium', 'Np': 'Neptunium', 'Pu': 'Plutonium', 'Am': 'Americium', 'Cm': 'Curium', 'Cf': 'Californium', 'Sc': 'Scandium', 'Y': 'Yttrium'}
def fullname(lbl):
    m = re.match(r'([A-Z][a-z]?)(\(.*\))?$', str(lbl).strip())
    return (NAMES.get(m.group(1), m.group(1)) + (m.group(2) or '')) if m else str(lbl)
def fullpair(lbl): return '/'.join(fullname(x) for x in str(lbl).split('/'))
def conf_curve(df, label):
    y = df['Actual_LogD'].values; p = df['Pred_LogD'].values; e = df['confidence_pred_err'].values
    fr = np.linspace(0.1, 1.0, 19); r2s = []; rms = []
    for f in fr:
        m = e <= np.quantile(e, f); r2s.append(r2_score(y[m], p[m])); rms.append(rmse(y[m], p[m]))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(fr*100, r2s, 'o-', color='#1F4E79'); ax[0].set_xlabel('most confident percent of predictions kept'); ax[0].set_ylabel('R-squared'); ax[0].set_title(f'{label}: R-squared vs confidence'); ax[0].invert_xaxis()
    ax[1].plot(fr*100, rms, 'o-', color='#B5402E'); ax[1].set_xlabel('most confident percent of predictions kept'); ax[1].set_ylabel('RMSE (log units)'); ax[1].set_title(f'{label}: RMSE vs confidence'); ax[1].invert_xaxis()
    finish(fig, f'confidence_curve_{label.split()[1].lower()}')
def scatter(df, label):
    y = df['Actual_LogD'].values; p = df['Pred_LogD'].values; e = df['confidence_pred_err'].values
    fig, ax = plt.subplots(figsize=(5.6, 5))
    sc = ax.scatter(y, p, c=e, cmap='viridis_r', s=8, alpha=0.6); lim = [min(y.min(), p.min()), max(y.max(), p.max())]
    ax.plot(lim, lim, 'k--', lw=1); ax.set_xlabel('actual logD'); ax.set_ylabel('predicted logD')
    ax.set_title(f'{label}: predicted vs actual\nR2={r2_score(y,p):.3f}  RMSE={rmse(y,p):.3f}'); fig.colorbar(sc, label='predicted error (lower = more confident)')
    finish(fig, f'pred_vs_actual_{label.split()[1].lower()}')

# 1-2 confidence curves, 3-4 scatters
for fn, lab in [("deploy_A_screening_predictions.csv", "Track A"), ("deploy_B_condition_predictions.csv", "Track B")]:
    if os.path.exists(fn):
        d = pd.read_csv(fn); conf_curve(d, lab); scatter(d, lab)

# per-metal accuracy, confidence, and calibration (full element names)
if os.path.exists("metal_confidence_by_metal.csv"):
    bm = pd.read_csv("metal_confidence_by_metal.csv").sort_values('RMSE'); bm['name'] = bm['Metal'].map(fullname)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    ax[0].barh(bm['name'], bm['R2'], color='#1F4E79'); ax[0].set_xlabel('R-squared'); ax[0].set_title('R-squared by metal (Track B)')
    ax[1].barh(bm['name'], bm['RMSE'], color='#B5402E'); ax[1].set_xlabel('RMSE (log units)'); ax[1].set_title('RMSE by metal (Track B)')
    finish(fig, 'by_metal_accuracy')
    bc = bm.sort_values('median_confidence_err', ascending=False)
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.barh(bc['name'], bc['median_confidence_err'], color='#1C7293'); ax.set_xlabel('median predicted error  (shorter bar = more confident)'); ax.set_title('Confidence per metal (Track B)')
    finish(fig, 'confidence_per_metal')
    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    ax.scatter(bm['median_confidence_err'], bm['RMSE'], color='#1F4E79')
    for _, r in bm.iterrows(): ax.annotate(r['name'], (r['median_confidence_err'], r['RMSE']), fontsize=6.5, alpha=0.7)
    ax.set_xlabel('median predicted error (the confidence signal)'); ax.set_ylabel('actual RMSE'); ax.set_title('Confidence tracks real error, by metal')
    finish(fig, 'by_metal_confidence_vs_error')

# per-pair separation and confidence (full element names)
if os.path.exists("metal_confidence_by_pair.csv"):
    bp = pd.read_csv("metal_confidence_by_pair.csv").sort_values('n', ascending=False).head(12)[::-1]; bp['name'] = bp['metal_pair'].map(fullpair)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].barh(bp['name'], bp['sep_R2'], color='#1F4E79'); ax[0].set_xlabel('separation R-squared'); ax[0].set_title('Separation R-squared by metal pair')
    ax[1].barh(bp['name'], bp['sep_RMSE'], color='#B5402E'); ax[1].set_xlabel('separation RMSE'); ax[1].set_title('Separation RMSE by metal pair')
    finish(fig, 'by_metal_pair_separation')
    bp2 = pd.read_csv("metal_confidence_by_pair.csv").sort_values('n', ascending=False).head(10); bp2['name'] = bp2['metal_pair'].map(fullpair)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.scatter(bp2['mean_conf_err'], bp2['sep_RMSE'], color='#1F4E79')
    for _, r in bp2.iterrows(): ax.annotate(r['name'], (r['mean_conf_err'], r['sep_RMSE']), fontsize=7, alpha=0.75)
    ax.set_xlabel('mean predicted error for the pair (shorter = more confident)'); ax.set_ylabel('actual separation RMSE'); ax.set_title('Confidence tracks difficulty, by metal pair')
    finish(fig, 'by_pair_confidence_vs_error')

# 8 ensemble weights
wA = {'cat':0.37,'et':0.24,'xgb':0.16,'lgb':0.11,'ridge':0.10,'hgb':0.02,'mlp':0.0}
wB = {'et':0.43,'lgb':0.40,'xgb':0.14,'cat':0.03,'hgb':0.0,'mlp':0.0,'ridge':0.0}
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(list(wA), list(wA.values()), color='#1F4E79'); ax[0].set_title('Track A stack weights (NNLS)'); ax[0].set_ylabel('weight')
ax[1].bar(list(wB), list(wB.values()), color='#1F4E79'); ax[1].set_title('Track B stack weights (NNLS)')
finish(fig, 'ensemble_weights')

# 9 conformal calibration from deploy files
rows = []
for fn, lab in [("deploy_A_screening_predictions.csv", "Track A"), ("deploy_B_condition_predictions.csv", "Track B")]:
    if os.path.exists(fn):
        d = pd.read_csv(fn); y = d['Actual_LogD']
        c90 = float(((y >= d['lo90']) & (y <= d['hi90'])).mean()); c80 = float(((y >= d['lo80']) & (y <= d['hi80'])).mean())
        rows.append((lab, c90, c80, float((d['hi90']-d['lo90']).median())))
if rows:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4)); labs=[r[0] for r in rows]; x=np.arange(len(labs))
    ax[0].bar(x-0.2,[r[1] for r in rows],0.4,label='90% interval',color='#1F4E79'); ax[0].bar(x+0.2,[r[2] for r in rows],0.4,label='80% interval',color='#7FA6CC')
    ax[0].axhline(0.9,ls='--',c='#1F4E79',lw=1); ax[0].axhline(0.8,ls='--',c='#7FA6CC',lw=1); ax[0].set_xticks(x); ax[0].set_xticklabels(labs); ax[0].set_ylabel('empirical coverage'); ax[0].set_title('Conformal coverage vs target'); ax[0].legend(); ax[0].set_ylim(0,1)
    ax[1].bar(labs,[r[3] for r in rows],color='#B5402E'); ax[1].set_ylabel('median 90% interval width (log units)'); ax[1].set_title('Interval width')
    finish(fig, 'conformal_calibration')

# 10 Zhang comparison: selective 3-class accuracy vs his flat 0.72
if os.path.exists("classifier_confidence_results.csv"):
    cc = pd.read_csv("classifier_confidence_results.csv"); c3 = cc[cc['task']=='3-class']
    fig, ax = plt.subplots(figsize=(8, 5)); xs=['all','top 25%','top 10%']
    for _, r in c3.iterrows():
        ax.plot(xs, [r['accuracy'], r['top25_acc'], r['top10_acc']], 'o-', label=r['track'])
    ax.axhline(0.72, ls='--', color='gray', label='Zhang 0.72 (flat, no ranking)')
    ax.set_ylabel('3-class accuracy'); ax.set_title('Our classifier with confidence vs Zhang (3-class)'); ax.legend(); ax.set_ylim(0.5,1.0)
    finish(fig, 'zhang_comparison_selective')

# 12 our-data vs zhang-data (if present)
if os.path.exists("zhang_data_results.csv"):
    zd = pd.read_csv("zhang_data_results.csv")
    our = {'Track A regression (new molecule)':(0.466,1.148),'Track B regression (known molecule)':(0.725,0.823),'3-class (new molecule) accuracy_all':(0.625,np.nan)}
    fig, ax = plt.subplots(figsize=(9,4.5)); tasks=list(zd['task']); x=np.arange(len(tasks))
    ax.bar(x-0.2,[our.get(t,(np.nan,))[0] for t in tasks],0.4,label='our data',color='#1F4E79')
    ax.bar(x+0.2,zd['R2_all'],0.4,label="Zhang's data",color='#5B9BD5')
    ax.set_xticks(x); ax.set_xticklabels([t.split(' (')[0] for t in tasks], rotation=15, ha='right'); ax.set_ylabel('R-squared or accuracy'); ax.set_title('Our model: our data vs Dr. Zhang data'); ax.legend()
    finish(fig, 'our_data_vs_zhang_data')

# 13 his exact split head-to-head
if os.path.exists("zhang_his_split_results.csv"):
    hs = pd.read_csv("zhang_his_split_results.csv").set_index('metric')['ours']
    fig, ax = plt.subplots(figsize=(7.5, 5)); xs = ['all\n(n=494)', 'top 25%\nconfidence', 'top 10%\nconfidence']
    vals = [hs['3-class accuracy (his test)'], hs['3-class acc, top 25% conf'], hs['3-class acc, top 10% conf']]
    bars = ax.bar(xs, vals, color=['#1F4E79', '#2E75B6', '#9DC3E6'])
    ax.axhline(0.72, ls='--', color='#B5402E', lw=1.5, label='Dr. Zhang 0.72 (flat, no ranking)')
    for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.3f}', ha='center', fontweight='bold')
    ax.set_ylim(0.5, 1.06); ax.set_ylabel('3-class accuracy on his exact 494-row test')
    ax.set_title('Our model on Dr. Zhang\'s exact split and test\nsame data, same features, same 15 held-out molecules'); ax.legend(loc='lower right')
    finish(fig, 'zhang_his_split_headtohead')

pdf.close()
print("saved", len(saved), "figures to figures/ and figures/all_figures.pdf:")
for s in saved: print("  -", s)
