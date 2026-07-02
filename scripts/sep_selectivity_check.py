#!/usr/bin/env python3
"""
sep_selectivity_check.py — the honest test of whether the model rewards genuine Am/Eu
SELECTIVITY (differential affinity for one metal over the other) or is just tracking
general extraction strength (extract-everything, which separates nothing).

signed_sep = predicted logD(Am) - predicted logD(Eu)   (+ favours Am, the useful direction)
level      = mean predicted logD across the pair         (general extraction strength)

If pred_sep were just strength, signed_sep would rise with level and never change sign.
Genuine selectivity shows up as: (a) low/no correlation with level, (b) sign flips across
extractants, and (c) the correct chemistry - soft S-donors (Cyanex-301 family) discriminate
Am from Eu more than hard O-donors. We check all three on the KNOWN 295 (where the model has
data: 150 extractants measured with BOTH Am and Eu) and on the NOVEL generated pool (out of
training, where the model extrapolates).
"""
import os, re, pickle, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from PIL import Image
import autodata_score, __main__ as _m
_m.SepScorer = autodata_score.SepScorer
DIR = os.environ.get('REE_ROOT', '.')

def donor(s):
    if 'P(=S)' in s or 'C(=S)' in s or 'P(S)' in s: return 'soft S-donor'
    if re.search(r'n[0-9]', s) or 'nc' in s or 'nn' in s: return 'soft N-heterocycle'
    if 'P(=O)' in s or 'OP(=O)' in s: return 'hard O/P-donor'
    return 'other'

sc = pickle.load(open(os.path.join(DIR, 'autodata_scorer.pkl'), 'rb'))
kb = sc.score(sorted(sc.known), 'Am(III)', 'Eu(III)'); kb = kb[kb.valid == True].copy()
p = pd.read_csv(os.path.join(DIR, 'results/autodata/candidate_pool.csv'))
for d in (kb, p):
    d['signed'] = d.pred_logD_A - d.pred_logD_B
    d['level'] = (d.pred_logD_A + d.pred_logD_B) / 2
    d['donor'] = d.smiles.map(donor)

print('SELECTIVITY (signed_sep = logD_Am - logD_Eu) vs GENERAL STRENGTH (level)')
print(f"  KNOWN 295 : spearman(pred_sep, level) = {spearmanr(kb.pred_sep, kb.level).correlation:+.2f} | favours Am {100*(kb.signed>0).mean():.0f}% | signed sd {kb.signed.std():.2f} range [{kb.signed.min():.2f},{kb.signed.max():.2f}]")
print(f"  NOVEL pool: spearman(pred_sep, level) = {spearmanr(p.pred_sep, p.level).correlation:+.2f} | favours Am {100*(p.signed>0).mean():.0f}% | signed sd {p.signed.std():.2f} range [{p.signed.min():.2f},{p.signed.max():.2f}]")
print('\nMECHANISM (median signed_sep by donor class, KNOWN 295) - soft S-donors should be highest:')
print(kb.groupby('donor').agg(n=('smiles', 'size'), median_signed=('signed', 'median')).round(3).to_string())

order = ['hard O/P-donor', 'soft N-heterocycle', 'other', 'soft S-donor']
col = {'hard O/P-donor': '#4c72b0', 'soft N-heterocycle': '#55a868', 'other': '#b0b0b0', 'soft S-donor': '#d1495b'}
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
# Panel A: is separation just strength? scatter signed_sep vs level, known vs novel
ax[0].axhline(0, color='k', lw=.7, ls='--')
ax[0].scatter(kb.level, kb.signed, s=20, c='#4c72b0', alpha=.55, label=f'known 295 (r={spearmanr(kb.pred_sep,kb.level).correlation:+.2f})')
ax[0].scatter(p.level, p.signed, s=26, c='#d1495b', alpha=.7, label=f'novel pool (r={spearmanr(p.pred_sep,p.level).correlation:+.2f})')
ax[0].set_xlabel('general extraction strength  (mean predicted logD)')
ax[0].set_ylabel('Am/Eu selectivity  (signed: +favours Am)')
ax[0].set_title('Is separation just strength?\nknown: flat (selectivity != strength) | novel: tilts up')
ax[0].legend(fontsize=8, loc='lower right')
# Panel B: mechanism - selectivity by donor class (known)
groups = [kb[kb.donor == d].signed.values for d in order]
bp = ax[1].boxplot(groups, labels=[d.replace(' ', '\n') for d in order], patch_artist=True, showfliers=False, widths=.6)
for patch, d in zip(bp['boxes'], order): patch.set_facecolor(col[d]); patch.set_alpha(.75)
ax[1].axhline(0, color='k', lw=.7, ls='--')
ax[1].set_ylabel('Am/Eu selectivity  (signed logD gap)')
ax[1].set_title('Does the model capture the right chemistry?\nsoft S-donors (Cyanex-301 family) are most Am-selective')
plt.tight_layout()
out = os.path.join(DIR, 'figures/autodata_selectivity.png')
fig.savefig(out, dpi=140, facecolor='white'); Image.open(out).convert('RGB').save(out)
print('\nsaved', out)
