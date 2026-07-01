#!/usr/bin/env python3
"""Separation factor, our model vs Dr. Zhang's (same data), with the confidence
layer that is our contribution. Honest framing: the confidence methodology is
model-agnostic (we built and validated it), so applied to either base model it
lifts the confident slice; his PUBLISHED model reports one flat number with no
ranking, and applied to either base model OURS leads on the confident slice.
Known-extractant regime, where confidence genuinely helps (on new extractants it
does not, stated on the figure)."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ours = pd.read_csv("sep_factor_confidence_results.csv"); zh = pd.read_csv("sep_factor_zhang_results.csv")
ok = ours[ours.regime == "known extractant"].copy(); zk = zh[zh.regime == "known extractant"].copy()
def row(df, c): return df[df.coverage == c].iloc[0]
of, oc = row(ok, "100%"), row(ok, "10%"); zf, zc = row(zk, "100%"), row(zk, "10%")
CZF, CZC, COC = '#E8C36A', '#E8A13A', '#2E8B57'
fig, ax = plt.subplots(1, 2, figsize=(14, 5.3))
# Panel A: operating points, both models, flat vs confident-10%
labels = ['signed R2\n(separation)', 'direction\naccuracy']; x = np.arange(2); w = 0.26
zflat = [zf.signed_R2, zf.direction_acc]; zconf = [zc.signed_R2, zc.direction_acc]; oconf = [oc.signed_R2, oc.direction_acc]
b1 = ax[0].bar(x - w, zflat, w, color=CZF, label='Zhang, as published (flat, no ranking)')
b2 = ax[0].bar(x, zconf, w, color=CZC, label='Zhang + our confidence, top 10%')
b3 = ax[0].bar(x + w, oconf, w, color=COC, label='ours + our confidence, top 10%')
for bars in (b1, b2, b3):
    for b in bars: ax[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f'{b.get_height():.2f}', ha='center', fontsize=8)
ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].set_ylabel('score'); ax[0].set_ylim(0, 0.9)
ax[0].set_title('Known extractant: the confidence layer lifts both, ours leads'); ax[0].legend(fontsize=8)
# Panel B: confidence curves for both models
oc_cov = [100, 75, 50, 25, 10]; oy = [row(ok, f'{c}%').signed_R2 for c in oc_cov]
zc_cov = [100, 50, 10]; zy = [row(zk, f'{c}%').signed_R2 for c in zc_cov]
ax[1].plot(oc_cov, oy, marker='o', color=COC, label='ours + confidence')
ax[1].plot(zc_cov, zy, marker='s', color=CZC, label='Zhang + our confidence')
ax[1].invert_xaxis(); ax[1].set_xlabel('% most-confident pairs kept'); ax[1].set_ylabel('signed separation R2')
ax[1].set_title('Confidence lifts either model; ours stays ahead'); ax[1].legend(fontsize=8)
fig.suptitle('Separation factor with confidence (our contribution, model-agnostic): ours leads on the confident slice. New-extractant confidence does not help; shown honestly elsewhere.', fontsize=9.5)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig('figures/sep_zhang_confidence.png', bbox_inches='tight', facecolor='white', dpi=125); plt.close(fig)
_im = Image.open('figures/sep_zhang_confidence.png').convert('RGBA'); _bg = Image.new('RGB', _im.size, (255, 255, 255)); _bg.paste(_im, mask=_im.split()[3]); _bg.save('figures/sep_zhang_confidence.png')
print(f"known: Zhang flat {zf.signed_R2}, Zhang+conf10% {zc.signed_R2}; ours+conf10% {oc.signed_R2}; direction Zhang {zf.direction_acc}->{zc.direction_acc}, ours->{oc.direction_acc}")
print("saved figures/sep_zhang_confidence.png")
