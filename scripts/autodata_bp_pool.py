#!/usr/bin/env python3
"""
autodata_bp_pool.py — aggregate the SEPARATION-target AutoData pool into TWO confidence-gated
shortlists, as Skyler chose:
  EXPLORE  = each molecule's best pair among pairs with >=5 measured extractants (higher SF, thin data)
  TRUSTED  = each molecule's best pair among pairs with >=20 measured extractants (real data support)
Each is gated to the most-confident top 25% and top 10% by the bake-off-winning uncertainty
(bootstrap-bagged ensemble disagreement; LOWER = more confident), then ranked by predicted SF.

Scores come from results/autodata_bp/rescored_all.csv, never from any agent's text.

HONEST FRAMING: novel extractants are the model's weak regime. The winning confidence reaches
top-10% separation direction ~0.81 / signed R2 ~0.28; predicted SF magnitudes are low-trust. These
are the most-trustworthy-available triage lists for the wet lab, not certifications.
"""
import os, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
DIR = os.environ.get('REE_ROOT', '.')
d = pd.read_csv(os.path.join(DIR, 'results/autodata_bp/rescored_all.csv'))
tot = len(d)
d = d[(d.valid == True) & (d.novel == True)].copy()
d = d.drop_duplicates('smiles', keep='first').reset_index(drop=True)
n = len(d)
print(f"best-pair pool: {tot} scored rows -> {n} unique valid+novel candidates")

def shortlists(kind):
    SF, unc, pair, sup = f'{kind}_SF', f'{kind}_unc', f'{kind}_pair', f'{kind}_support'
    sub = d.dropna(subset=[SF, unc]).copy()
    out = {}
    for frac, tag in [(0.25, 'top25'), (0.10, 'top10')]:
        k = max(1, int(round(frac * len(sub)))); conf = sub.nsmallest(k, unc)
        out[tag] = conf.sort_values(SF, ascending=False)[['smiles', pair, SF, unc, sup, 'cand_unc']].reset_index(drop=True)
    return out, SF, unc, pair, sup

for kind in ['explore', 'trusted']:
    sl, SF, unc, pair, sup = shortlists(kind)
    floor = '>=5' if kind == 'explore' else '>=20'
    print(f"\n================  {kind.upper()} shortlist (best pair over support {floor})  ================")
    print(f"--- top-25% most confident (n={len(sl['top25'])}), ranked by predicted SF ---")
    print(sl['top25'].head(12).to_string(index=False))
    print(f"--- top-10% most confident (n={len(sl['top10'])}), ranked by predicted SF ---")
    print(sl['top10'].to_string(index=False))
    sl['top25'].to_csv(os.path.join(DIR, f'results/autodata_bp/shortlist_{kind}_top25.csv'), index=False)
    sl['top10'].to_csv(os.path.join(DIR, f'results/autodata_bp/shortlist_{kind}_top10.csv'), index=False)
d.to_csv(os.path.join(DIR, 'results/autodata_bp/bestpair_pool_ranked.csv'), index=False)

# figure: EXPLORE vs TRUSTED, SF vs uncertainty, top-10% highlighted
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for a, kind, title in [(ax[0], 'explore', 'EXPLORE (best pair, support >=5)'), (ax[1], 'trusted', 'TRUSTED (best pair, support >=20)')]:
    SF, unc = f'{kind}_SF', f'{kind}_unc'
    sub = d.dropna(subset=[SF, unc]); k10 = sub.nsmallest(max(1, int(round(0.10 * len(sub)))), unc); k25 = sub.nsmallest(max(1, int(round(0.25 * len(sub)))), unc)
    a.scatter(sub[unc], sub[SF], s=20, c='#c7ccd1', label='pool')
    a.scatter(k25[unc], k25[SF], s=28, c='#4c72b0', label='top-25%')
    a.scatter(k10[unc], k10[SF], s=46, c='#d1495b', label='top-10%')
    a.set_yscale('log'); a.set_xlabel('best-pair uncertainty (bootstrap; lower = more confident)'); a.set_ylabel('predicted separation factor')
    a.set_title(title); a.legend(fontsize=8)
plt.tight_layout()
out = os.path.join(DIR, 'figures/autodata_bestpair_confident.png')
fig.savefig(out, dpi=140, facecolor='white'); Image.open(out).convert('RGB').save(out)
print('\nsaved', out)
