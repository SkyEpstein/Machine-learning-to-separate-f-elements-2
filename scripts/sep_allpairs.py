#!/usr/bin/env python3
"""
sep_allpairs.py — generalize the candidate ranking from Am/Eu to EVERY f-element pair,
ranked by STRENGTH-ADJUSTED selectivity, not general extraction strength.

For each candidate extractant we predict logD across all 28 f-element metals at reference
conditions, subtract the candidate's OWN mean logD (removing the "extracts everything
strongly" component), and measure how much it still DIFFERENTIATES the metals. A candidate
that pulls every f-element equally hard scores ~0 no matter how strong it is; only genuine
selectivity survives. We report:
  - diff_III  : strength-adjusted spread across the trivalent (III) f-elements (the hard,
                industrially relevant selectivity); the primary ranking key.
  - best_pair : for each candidate, the data-supported pair it separates best, with the gap
                (= log10 separation factor), the pair uncertainty, and the pair's data support.
  - a pair -> best novel candidate table for every well-supported pair.

HONEST FRAMING: predicted separation for a NEW extractant is the model's weakest regime
(new-extractant direction ~66%, signed R2 ~0.19 and RMSE ~0.9 log units on the same-data
grouped-CV separation eval, not measured on this scorer). The pairwise selectivity signal is
genuine but ~1/3 extractant-specific and ~2/3 a common metal gradient (PC1 = 66% of centered
variance), and adjacent-lanthanide magnitudes are low-trust. Uncertainty is a RELATIVE 5-bag
seed-disagreement signal (not a calibrated interval). This is a wet-lab triage ranking, not a
certification, and the known-295 reference below is scored IN-SAMPLE (optimistic for them).
"""
import os, re, pickle, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from itertools import combinations
from PIL import Image
import autodata_score, __main__ as _m
_m.SepScorer = autodata_score.SepScorer

DIR = os.environ.get('REE_ROOT', '.')
sc = pickle.load(open(os.path.join(DIR, 'autodata_scorer.pkl'), 'rb'))
FELEM = set('La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Ac Th Pa U Np Pu Am Cm Bk Cf Es'.split())
def sym(m):
    x = re.match(r'\s*([A-Z][a-z]?)', str(m)); return x.group(1) if x else ''
fmetals = sorted([m for m in sc.metalrow if sym(m) in FELEM])
iii_idx = [i for i, m in enumerate(fmetals) if '(III)' in m]

# data support per pair (shared extractants measured with both metals)
tr = pd.read_csv(os.path.join(DIR, 'Training_Data_V27.csv'), low_memory=False)
te = pd.read_csv(os.path.join(DIR, 'Testing_Data_V39.csv'), low_memory=False)
raw = pd.concat([tr, te], ignore_index=True); raw = raw[raw.Log_D.notna()]; SMI = 'SMILES_canonical'
ext_by_metal = {m: set(raw[raw.Metal.astype(str) == m][SMI].dropna().astype(str)) for m in fmetals}
support = {frozenset((a, b)): len(ext_by_metal[a] & ext_by_metal[b]) for a, b in combinations(fmetals, 2)}

def profile_matrix(smiles):
    """return kept smiles and (n x 28) predicted-logD mean and bag-std matrices."""
    keep, MU, SD = [], [], []
    for s in smiles:
        d = autodata_score.descriptors(s)
        if d is None: continue
        X = np.vstack([np.hstack([d, sc.metalrow[m], sc.ref_cond]) for m in fmetals])
        P = np.column_stack([mm.predict(X) for mm in sc.models])  # (28, 5 bags)
        keep.append(s); MU.append(P.mean(1)); SD.append(P.std(1))
    return keep, np.array(MU), np.array(SD)

def candidate_table(keep, MU, SD):
    C = MU - MU.mean(1, keepdims=True)               # strength-adjusted profiles
    diff_iii = C[:, iii_idx].std(1)
    sup_pairs = [(a, b) for a, b in combinations(fmetals, 2) if support[frozenset((a, b))] >= 5]
    recs = []
    for k, s in enumerate(keep):
        best = None
        for a, b in sup_pairs:
            ia, ib = fmetals.index(a), fmetals.index(b)
            gap = abs(MU[k, ia] - MU[k, ib])
            if best is None or gap > best[2]:
                best = (a, b, gap, np.hypot(SD[k, ia], SD[k, ib]), support[frozenset((a, b))])
        recs.append(dict(smiles=s, diff_III=round(float(diff_iii[k]), 3), cand_unc=round(float(SD[k].mean()), 3),
                         best_pair=f'{best[0]}/{best[1]}', best_gap=round(float(best[2]), 3),
                         best_SF=round(float(10 ** best[2]), 1), best_unc=round(float(best[3]), 3), best_support=int(best[4])))
    return pd.DataFrame(recs)

pool = pd.read_csv(os.path.join(DIR, 'results/autodata/candidate_pool.csv'))
nk, MUn, SDn = profile_matrix(pool.smiles.tolist())
kk, MUk, SDk = profile_matrix(sorted(sc.known))
nov = candidate_table(nk, MUn, SDn)
kn = candidate_table(kk, MUk, SDk)

print(f"f-element metals scored: {len(fmetals)} ({len(iii_idx)} trivalent) | pairs with >=5 data support: {sum(v>=5 for v in support.values())}/{len(support)}")
print(f"\nSTRENGTH-ADJUSTED differentiation across the (III) series (higher = separates f-elements more, strength removed):")
print(f"  novel pool : median {nov.diff_III.median():.3f}  p90 {nov.diff_III.quantile(.9):.3f}  max {nov.diff_III.max():.3f}")
print(f"  known 295* : median {kn.diff_III.median():.3f}  p90 {kn.diff_III.quantile(.9):.3f}  max {kn.diff_III.max():.3f}   (*in-sample, optimistic)")

umed = nov.cand_unc.median()
conf = nov[nov.cand_unc <= umed]
top = nov.sort_values('diff_III', ascending=False).head(12)
print("\n=== NOVEL candidates ranked by strength-adjusted selectivity (top 12) ===")
print(top[['smiles', 'diff_III', 'cand_unc', 'best_pair', 'best_SF', 'best_unc', 'best_support']].to_string(index=False))
print("\n=== CONFIDENT novel shortlist (uncertainty <= median), top 10 by strength-adjusted selectivity ===")
print(conf.sort_values('diff_III', ascending=False).head(10)[['smiles', 'diff_III', 'best_pair', 'best_SF', 'best_unc', 'best_support']].to_string(index=False))

# per-pair best novel candidate, well-supported pairs only
rows = []
for a, b in combinations(fmetals, 2):
    sup = support[frozenset((a, b))]
    if sup < 20: continue
    ia, ib = fmetals.index(a), fmetals.index(b)
    gaps = np.abs(MUn[:, ia] - MUn[:, ib]); k = int(gaps.argmax())
    rows.append(dict(pair=f'{a}/{b}', best_candidate=nk[k], gap=round(float(gaps[k]), 3), SF=round(float(10 ** gaps[k]), 1),
                     unc=round(float(np.hypot(SDn[k, ia], SDn[k, ib])), 3), support=sup))
pairtab = pd.DataFrame(rows).sort_values('gap', ascending=False)
print(f"\n=== per-pair best NOVEL candidate (well-supported pairs, >=20), top 12 of {len(pairtab)} ===")
print(pairtab.head(12).to_string(index=False))

nov.to_csv(os.path.join(DIR, 'results/autodata/allpairs_novel_ranked.csv'), index=False)
pairtab.to_csv(os.path.join(DIR, 'results/autodata/allpairs_best_by_pair.csv'), index=False)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
bins = np.linspace(0, max(nov.diff_III.max(), kn.diff_III.max()), 22)
ax[0].hist(kn.diff_III, bins=bins, alpha=.6, color='#4c72b0', density=True, label=f'known 295* in-sample (n={len(kn)})')
ax[0].hist(nov.diff_III, bins=bins, alpha=.6, color='#d1495b', density=True, label=f'AutoData novel (n={len(nov)})')
ax[0].set_xlabel('strength-adjusted (III)-series differentiation'); ax[0].set_ylabel('density')
ax[0].set_title('Selectivity, strength removed:\nnovel candidates vs known extractants'); ax[0].legend(fontsize=8)
pt = pairtab.head(12).iloc[::-1]
ax[1].barh(range(len(pt)), pt.SF, color='#2e7d32', alpha=.8)
ax[1].set_yticks(range(len(pt))); ax[1].set_yticklabels(pt.pair, fontsize=8)
ax[1].set_xlabel('best novel candidate: predicted separation factor (x)')
ax[1].set_title('Best novel candidate per well-supported pair\n(predicted; low-trust magnitude, for lab triage)')
plt.tight_layout()
out = os.path.join(DIR, 'figures/autodata_allpairs.png')
fig.savefig(out, dpi=140, facecolor='white'); Image.open(out).convert('RGB').save(out)
print('\nsaved', out)
