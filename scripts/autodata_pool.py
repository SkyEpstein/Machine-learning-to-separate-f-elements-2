#!/usr/bin/env python3
"""
autodata_pool.py — aggregate the AutoData loop's per-round scored candidates into one
ranked pool and the two shortlists the separation recommender exposes:
  EXPLORE   = optimistic upper bound (pred_sep + uncertainty); surfaces uncertain-but-promising
  CONFIDENT = lower bound (pred_sep - uncertainty) among the low-uncertainty half; the safe bets

Real scores are read from results/autodata/scored_round_*.csv (written by autodata_score.py),
never from any agent's text. The 295 known extractants are scored for the same pair with the
SAME model as an IN-SAMPLE reference (optimistic: the model was trained on those 295, so their
distribution is generous to the known set); it is not an even-footing baseline. Even so the
novel pool does not reach the best known extractants, which is the honest takeaway.

HONEST FRAMING: predicted separation for a NEW extractant is the model's weakest regime
(new-extractant signed R2 ~0.19, magnitude unreliable, direction ~0.66). This RANKS candidates
for the wet lab; it does not certify them. Every row carries the confidence (uncertainty), and
the whole pool is a triage list for experiments, not a set of designed winners.
"""
import os, glob, warnings, pickle; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from PIL import Image

DIR = os.environ.get('REE_ROOT', '.')
files = sorted(glob.glob(os.path.join(DIR, 'results/autodata/scored_round_*.csv')))
assert files, "no scored_round_*.csv found; run the AutoData loop first"
raw = pd.concat([pd.read_csv(f).assign(round=int(''.join(c for c in os.path.basename(f) if c.isdigit())))
                 for f in files], ignore_index=True)
gen_total = len(raw)
val = raw[raw['valid'] == True].copy()
val = val.sort_values('pred_sep', ascending=False).drop_duplicates('smiles', keep='first')
novel = val[val['novel'] == True].dropna(subset=['pred_sep', 'uncertainty']).copy()
novel['ucb'] = novel['pred_sep'] + novel['uncertainty']
novel['lcb'] = (novel['pred_sep'] - novel['uncertainty']).clip(lower=0)

def scaf(s):
    m = Chem.MolFromSmiles(s)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None
novel['scaffold'] = novel['smiles'].map(scaf)

EXPLORE = novel.sort_values('ucb', ascending=False).head(15)
umed = novel['uncertainty'].median()
CONFIDENT = novel[novel['uncertainty'] <= umed].sort_values('lcb', ascending=False).head(15)

# known-extractant baseline for the same pair, same model
# the pickle was saved from __main__ of autodata_score.py, so register the class on __main__ before loading
import autodata_score, __main__ as _m
_m.SepScorer = autodata_score.SepScorer
scorer = pickle.load(open(os.path.join(DIR, 'autodata_scorer.pkl'), 'rb'))
kb = scorer.score(sorted(scorer.known), 'Am(III)', 'Eu(III)')
kb = kb[kb['valid'] == True]

n_parse_valid = int((raw['valid'] == True).sum())
print(f"generated (scored rows across {len(files)} rounds): {gen_total}")
print(f"parse-valid: {n_parse_valid}/{gen_total} ({n_parse_valid/gen_total:.0%})  |  unique valid: {len(val)} (uniqueness {len(val)/max(n_parse_valid,1):.0%})  |  novel vs our 295 (dataset novelty, NOT new-to-chemistry): {len(novel)} ({len(novel)/max(len(val),1):.0%} of unique valid)")
print(f"unique Murcko scaffolds in novel pool: {novel['scaffold'].nunique()}")
print(f"novel pool pred_sep : median {novel['pred_sep'].median():.3f}  p90 {novel['pred_sep'].quantile(.9):.3f}  max {novel['pred_sep'].max():.3f}")
print(f"known-295 pred_sep* : median {kb['pred_sep'].median():.3f}  p90 {kb['pred_sep'].quantile(.9):.3f}  max {kb['pred_sep'].max():.3f}   (*IN-SAMPLE reference, optimistic: the model was trained on these 295)")
print(f"honest read: novel does NOT beat the best known (novel p90 {novel['pred_sep'].quantile(.9):.3f} < known {kb['pred_sep'].quantile(.9):.3f}); medians ~equal and within noise")
print("\n=== EXPLORE (optimistic upper bound = pred_sep + uncertainty) ===")
print(EXPLORE[['smiles', 'pred_sep', 'uncertainty', 'ucb']].to_string(index=False))
print("\n=== CONFIDENT (low-uncertainty half, lower bound = pred_sep - uncertainty) ===")
print(CONFIDENT[['smiles', 'pred_sep', 'uncertainty', 'lcb']].to_string(index=False))

novel.to_csv(os.path.join(DIR, 'results/autodata/candidate_pool.csv'), index=False)
EXPLORE.to_csv(os.path.join(DIR, 'results/autodata/shortlist_explore.csv'), index=False)
CONFIDENT.to_csv(os.path.join(DIR, 'results/autodata/shortlist_confident.csv'), index=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].scatter(novel['uncertainty'], novel['pred_sep'], s=22, c='#9aa7b4', label='novel pool')
ax[0].scatter(EXPLORE['uncertainty'], EXPLORE['pred_sep'], s=42, c='#d1495b', label='EXPLORE (top UCB)')
ax[0].scatter(CONFIDENT['uncertainty'], CONFIDENT['pred_sep'], s=42, c='#2e7d32', marker='^', label='CONFIDENT')
ax[0].set_xlabel('uncertainty (model disagreement)'); ax[0].set_ylabel('predicted Am/Eu separation')
ax[0].set_title('AutoData candidate pool: separation vs confidence'); ax[0].legend(fontsize=8)
hi = max(novel['pred_sep'].max(), kb['pred_sep'].max())
bins = np.linspace(0, hi, 24)
ax[1].hist(kb['pred_sep'], bins=bins, alpha=.6, color='#4c72b0', label=f'known 295 (n={len(kb)})', density=True)
ax[1].hist(novel['pred_sep'], bins=bins, alpha=.6, color='#d1495b', label=f'AutoData novel (n={len(novel)})', density=True)
ax[1].set_xlabel('predicted Am/Eu separation'); ax[1].set_ylabel('density')
ax[1].set_title('Novel candidates vs known extractants (same model; known = in-sample)'); ax[1].legend(fontsize=8)
plt.tight_layout()
out = os.path.join(DIR, 'figures/autodata_pool.png')
fig.savefig(out, dpi=140, facecolor='white')
Image.open(out).convert('RGB').save(out)
print('\nsaved', out)
