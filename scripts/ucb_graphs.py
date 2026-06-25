#!/usr/bin/env python3
"""Render the UCB graphs from the saved results: the sequential active-learning
curves (model quality and discovery over rounds, all acquisition strategies) and the
delta G UCB one-shot selection (mean actual delta G of the picked set)."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
al = pd.read_csv("active_learning_results.csv"); du = pd.read_csv("dg_ucb_results.csv")
col = {'random': '#999999', 'greedy': '#B5402E', 'ucb': '#1F4E79', 'uncertainty': '#2E8B57', 'confidence': '#8E44AD'}
fig, ax = plt.subplots(1, 3, figsize=(16, 4.7))
for _, r in al.iterrows():
    ys = [float(x) for x in r['R2_by_round'].split(';')]; ax[0].plot(range(1, len(ys) + 1), ys, marker='o', ms=3, color=col.get(r['strategy'], '#000'), label=r['strategy'])
ax[0].set_xlabel('round'); ax[0].set_ylabel('test R-squared'); ax[0].set_title('Active learning: model quality over rounds'); ax[0].legend(fontsize=8)
for _, r in al.iterrows():
    ys = [float(x) for x in r['discovery_by_round'].split(';')]; ax[1].plot(range(1, len(ys) + 1), ys, marker='o', ms=3, color=col.get(r['strategy'], '#000'), label=r['strategy'])
ax[1].set_xlabel('round'); ax[1].set_ylabel('share of true top-10% found'); ax[1].set_title('Active learning: discovery over rounds'); ax[1].legend(fontsize=8)
sel = du[du['analysis'] == 'UCB selection']; methods = ['greedy (pred)', 'UCB (pred - unc)', 'random']
mcol = {'greedy (pred)': '#B5402E', 'UCB (pred - unc)': '#1F4E79', 'random': '#999999'}
x = np.arange(2); w = 0.26
for i, m in enumerate(methods):
    vals = [sel[(sel.select_top_pct == p) & (sel.method == m)]['mean_actual_dG'].values[0] for p in [5, 10]]
    ax[2].bar(x + i * w, vals, w, color=mcol[m], label=m)
ax[2].set_xticks(x + w); ax[2].set_xticklabels(['top 5%', 'top 10%']); ax[2].set_ylabel('mean actual delta G (kJ/mol)')
ax[2].set_title('Delta G UCB selection (more negative = stronger)'); ax[2].legend(fontsize=8); ax[2].axhline(0, color='k', lw=0.6)
fig.tight_layout(); fig.savefig('figures/ucb_graphs.png', bbox_inches='tight', facecolor='white', dpi=130); plt.close(fig)
_im = Image.open('figures/ucb_graphs.png').convert('RGBA'); _bg = Image.new('RGB', _im.size, (255, 255, 255)); _bg.paste(_im, mask=_im.split()[3]); _bg.save('figures/ucb_graphs.png')
print("saved figures/ucb_graphs.png")
