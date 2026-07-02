#!/usr/bin/env python3
"""
autodata_score.py — the evaluator for the AutoData candidate-extractant loop. Trains
a logD model on the known extractants (RDKit descriptors computed the SAME way for
training molecules and candidates, plus metal and condition features), then scores
any candidate SMILES for the separation of a target f-element pair, with a bootstrap-bagged
ensemble-disagreement uncertainty, and gates on chemical validity and novelty.

The uncertainty is the std across a bootstrap-bagged ensemble; it WON the new-extractant
confidence bake-off (see newext_confidence_bakeoff.py), beating the learned-error model and
the applicability-domain distances. It is a relative RANKING signal, not a calibrated interval.

HONEST CAVEAT: predicting a NEW extractant's separation is the weak regime (grouped-CV
separation signed R2 ~0.19 / RMSE ~1.45 at full coverage; magnitude unreliable). The score is
a RANKING signal for triage, not a truth; the uncertainty flags which candidates to trust
(top-10% direction ~0.81), and every candidate still needs a real lab measurement. This is a
generator for the closed loop, not a labeler.
"""
import os, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
import lightgbm as lgb
SMI = 'SMILES_canonical'
DNAMES = [d[0] for d in Descriptors._descList]
CALC = MolecularDescriptorCalculator(DNAMES)
METAL = ['Atomic_number', 'Melting_point_K', 'Boiling_point_K', 'Density_g/cm3', 'First_IE_kJ/mol', 'Second_IE_kJ/mol', 'Third_IE_kJ/mol', 'Matallic_radius_nm', 'Pauling_EN', 'Ionic_radius_nm', 'Oxidation_state']
COND = ['Acid_conc_M', 'Temperature_K', 'Extractant_conc_M', 'Metal_conc_mM']
def descriptors(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None: return None
    v = np.array(CALC.CalcDescriptors(m), dtype=np.float64)
    return v if np.isfinite(v).all() else np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
def canon(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return Chem.MolToSmiles(m) if m is not None else None
class SepScorer:
    def __init__(self, train_csv="Training_Data_V27.csv", test_csv="Testing_Data_V39.csv", n_bag=12):
        tr = pd.read_csv(train_csv, low_memory=False); te = pd.read_csv(test_csv, low_memory=False)
        df = pd.concat([tr, te], ignore_index=True)
        df = df[df['Log_D'].notna() & df[SMI].notna() & df[METAL + COND].notna().all(axis=1)].reset_index(drop=True)
        # descriptors per unique extractant (computed from SMILES, same as candidates)
        uniq = sorted(set(df[SMI].astype(str)))
        dmap = {s: descriptors(s) for s in uniq}
        keep = df[SMI].astype(str).map(lambda s: dmap.get(s) is not None)
        df = df[keep].reset_index(drop=True)
        D = np.vstack([dmap[s] for s in df[SMI].astype(str)])
        self.known = set(c for c in (canon(s) for s in uniq) if c)
        X = np.hstack([D, df[METAL].values.astype(float), df[COND].values.astype(float)])
        y = df['Log_D'].values.astype(float)
        # BOOTSTRAP-bagged ensemble: each model is fit on a bootstrap resample of the rows.
        # The bag disagreement (std across models) is our uncertainty. This estimator WON the
        # new-extractant confidence bake-off (newext_confidence_bakeoff.py): at top-10% coverage
        # it lifts separation direction accuracy 0.66 -> 0.81 and calibration 0.20 -> 0.31, beating
        # the learned-error model and the applicability-domain distances. It replaces the earlier
        # 5-bag seed variance, which did not widen for out-of-distribution chemistry.
        rng = np.random.RandomState(0)
        self.models = [lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=63, subsample=0.8, colsample_bytree=0.7, subsample_freq=1, random_state=s, n_jobs=-1, verbosity=-1).fit(X[bi], y[bi])
                       for s in range(n_bag) for bi in [rng.randint(0, len(X), len(X))]]
        self.ref_cond = df[COND].median().values.astype(float)
        self.metalrow = {m: df[df['Metal'].astype(str) == m][METAL].iloc[0].values.astype(float) for m in df['Metal'].astype(str).unique()}
        self.n_desc = D.shape[1]
        print(f"trained on {len(df)} rows / {len(uniq)} extractants; {self.n_desc} descriptors; metals: {len(self.metalrow)}")
    def _predict(self, X):
        P = np.column_stack([m.predict(X) for m in self.models]); return P.mean(1), P.std(1)
    def score(self, smiles_list, mA='Am(III)', mB='Eu(III)'):
        rows = []
        for smi in smiles_list:
            d = descriptors(smi); c = canon(smi)
            if d is None:
                rows.append({'smiles': smi, 'valid': False, 'novel': False, 'pred_logD_A': np.nan, 'pred_logD_B': np.nan, 'pred_sep': np.nan, 'uncertainty': np.nan}); continue
            novel = c not in self.known
            XA = np.hstack([d, self.metalrow[mA], self.ref_cond])[None, :]
            XB = np.hstack([d, self.metalrow[mB], self.ref_cond])[None, :]
            mA_, sA = self._predict(XA); mB_, sB = self._predict(XB)
            rows.append({'smiles': c, 'valid': True, 'novel': bool(novel), 'pred_logD_A': round(float(mA_[0]), 3), 'pred_logD_B': round(float(mB_[0]), 3),
                         'pred_sep': round(abs(float(mA_[0] - mB_[0])), 3), 'uncertainty': round(float(np.sqrt(sA[0]**2 + sB[0]**2)), 3)})
        return pd.DataFrame(rows)
if __name__ == "__main__":
    import argparse, pickle
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-pickle'); ap.add_argument('--score'); ap.add_argument('--pickle')
    ap.add_argument('--out', default='scored.csv'); ap.add_argument('--mA', default='Am(III)'); ap.add_argument('--mB', default='Eu(III)')
    a, _ = ap.parse_known_args()
    if a.build_pickle:
        pickle.dump(SepScorer(), open(a.build_pickle, 'wb')); print('pickled ->', a.build_pickle)
    elif a.score:
        s = pickle.load(open(a.pickle, 'rb')) if (a.pickle and os.path.exists(a.pickle)) else SepScorer()
        smis = [l.strip() for l in open(a.score) if l.strip()]
        df = s.score(smis, a.mA, a.mB); df.to_csv(a.out, index=False); print(df.to_string(index=False))
    else:
        s = SepScorer()
        print(s.score(["CCCCC(CC)COP(=O)(O)OCC(CC)CCCC", "CCCCOP(=O)(OCCCC)OCCCC", "CCN(CC)C(=S)c1ccccc1C(=S)N(CC)CC", "not_a_smiles"]).to_string(index=False))
