#!/usr/bin/env python3
"""
autodata_bestpair.py — the SEPARATION-as-target evaluator for the AutoData loop.

Instead of scoring a single fixed pair, each candidate is scored by the BEST single
f-element pair it separates: best_gap = max over data-supported pairs of
|predicted logD(i) - predicted logD(j)| at reference conditions, i.e. the largest predicted
log10 separation factor the molecule achieves for any real, data-supported pair. This makes
SEPARATION the explicit search objective. Confidence travels with every candidate so the loop
and the final shortlist can be gated to the most trustworthy predictions (top 25% / top 10%).

Columns: smiles, valid, novel, best_pair, best_gap, best_SF(=10**best_gap), best_unc (pair
5-bag disagreement), cand_unc (mean bag disagreement across all f-metals = overall confidence),
best_support (# extractants measured with BOTH metals of the best pair).

HONEST: predicted separation for a NEW extractant is the model's weakest regime; best_gap is a
RANKING signal for wet-lab triage, magnitude is low-trust, and uncertainty is a relative 5-bag
seed-disagreement signal, not a calibrated interval. Pairs with low best_support are low-trust.
"""
import os, re, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from itertools import combinations
import autodata_score
from autodata_score import descriptors, canon, METAL, COND, SMI

FELEM = set('La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Ac Th Pa U Np Pu Am Cm Bk Cf Es'.split())
def _sym(m):
    x = re.match(r'\s*([A-Z][a-z]?)', str(m)); return x.group(1) if x else ''

class BestPairScorer:
    """wraps a trained SepScorer (logD model) and scores candidates by best-pair separation."""
    def __init__(self, base, min_support=5):
        self.base = base
        self.fmetals = sorted([m for m in base.metalrow if _sym(m) in FELEM])
        tr = pd.read_csv("Training_Data_V27.csv", low_memory=False)
        te = pd.read_csv("Testing_Data_V39.csv", low_memory=False)
        raw = pd.concat([tr, te], ignore_index=True); raw = raw[raw['Log_D'].notna()]
        ebm = {m: set(raw[raw['Metal'].astype(str) == m][SMI].dropna().astype(str)) for m in self.fmetals}
        self.support = {frozenset((a, b)): len(ebm[a] & ebm[b]) for a, b in combinations(self.fmetals, 2)}
        allpairs = list(combinations(self.fmetals, 2))
        # two support floors: EXPLORE (>=5, higher SF but thin data) and TRUSTED (>=20, real data)
        self.pairs5 = [(a, b) for a, b in allpairs if self.support[frozenset((a, b))] >= 5]
        self.pairs20 = [(a, b) for a, b in allpairs if self.support[frozenset((a, b))] >= 20]

    def _profile(self, d):
        X = np.vstack([np.hstack([d, self.base.metalrow[m], self.base.ref_cond]) for m in self.fmetals])
        P = np.column_stack([mm.predict(X) for mm in self.base.models])
        return P.mean(1), P.std(1)

    def _best(self, mu, sd, idx, pairs):
        best = None
        for a, b in pairs:
            gap = abs(mu[idx[a]] - mu[idx[b]])
            if best is None or gap > best[2]:
                best = (a, b, gap, np.hypot(sd[idx[a]], sd[idx[b]]), self.support[frozenset((a, b))])
        return best

    def score(self, smiles_list):
        rows = []
        for smi in smiles_list:
            d = descriptors(smi); c = canon(smi)
            if d is None:
                rows.append({'smiles': smi, 'valid': False, 'novel': False, 'cand_unc': np.nan,
                             'explore_pair': None, 'explore_SF': np.nan, 'explore_unc': np.nan, 'explore_support': 0,
                             'trusted_pair': None, 'trusted_SF': np.nan, 'trusted_unc': np.nan, 'trusted_support': 0}); continue
            mu, sd = self._profile(d); idx = {m: k for k, m in enumerate(self.fmetals)}
            b5 = self._best(mu, sd, idx, self.pairs5); b20 = self._best(mu, sd, idx, self.pairs20)
            rows.append({'smiles': c, 'valid': True, 'novel': bool(c not in self.base.known), 'cand_unc': round(float(sd.mean()), 3),
                         'explore_pair': f'{b5[0]}/{b5[1]}', 'explore_SF': round(float(10 ** b5[2]), 1), 'explore_unc': round(float(b5[3]), 3), 'explore_support': int(b5[4]),
                         'trusted_pair': f'{b20[0]}/{b20[1]}', 'trusted_SF': round(float(10 ** b20[2]), 1), 'trusted_unc': round(float(b20[3]), 3), 'trusted_support': int(b20[4])})
        return pd.DataFrame(rows)

if __name__ == "__main__":
    import argparse, pickle
    ap = argparse.ArgumentParser()
    ap.add_argument('--score'); ap.add_argument('--pickle', default='autodata_scorer.pkl'); ap.add_argument('--out', default='bestpair.csv')
    a, _ = ap.parse_known_args()
    import __main__ as _m; _m.SepScorer = autodata_score.SepScorer
    base = pickle.load(open(a.pickle, 'rb')) if os.path.exists(a.pickle) else autodata_score.SepScorer()
    bp = BestPairScorer(base)
    if a.score:
        smis = [l.strip() for l in open(a.score) if l.strip()]
        df = bp.score(smis); df.to_csv(a.out, index=False); print(df.to_string(index=False))
    else:
        df = bp.score(["CC1CCC(P(=S)(S)C2CCC(C)CC2)CC1", "CCCCOP(=O)(OCCCC)OCCCC", "CCN(CC)C(=S)c1ccccc1C(=S)N(CC)CC", "not_a_smiles"])
        print(df.to_string(index=False))
