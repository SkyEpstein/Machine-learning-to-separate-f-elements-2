#!/usr/bin/env python3
"""
active_learning_ucb.py — UCB done properly, as sequential active learning. Start
from a small set of "run" experiments, and each round pick the next batch to run by
an acquisition rule, reveal their logD, retrain, and repeat. This is where UCB
(prediction + uncertainty) is supposed to help, unlike the one-shot screen.

Strategies compared each round (same seed set, fair): random, greedy (pick highest
predicted logD), UCB (predicted + 1.0 * uncertainty), and pure uncertainty.
Uncertainty is the spread across a 5-model bagged LightGBM. Two outcomes are
tracked over rounds: model quality (R^2 on a fixed held-out test set) and discovery
(how much of the pool's true top-10% strongest extractants has been found).
Conditions + metal features (the new-molecule screening setting).
"""
import os, time, warnings; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import r2_score
import lightgbm as lgb
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SEED = 42; START = time.time(); ROUNDS = 10; BATCH = 200; SEED_N = 300; TEST_N = 1000; KBAG = 5
TARGET, SMI = "Log_D", "SMILES_canonical"
TEXT = {"Solvent_A", "Solvent_B", "Metal", "Acid_type", "SMILES_canonical"}
COND = ['Extractant_conc_M','Molar_mass(g/mol) A','Log_P A','Boiling_point(K) A','Melting_point(K) A','Density(g/mL) A','Solubility_in_water(g/L) A','Molar_mass(g/mol) B','Log_P B','Boiling_point(K) B','Melting_point(K) B','Density(g/mL) B','Solubility_in_water(g/L) B','Volume_fraction_A','Volume_fraction_B','Atomic_number','Melting_point_K','Boiling_point_K','Density_g/cm3','First_IE_kJ/mol','Second_IE_kJ/mol','Third_IE_kJ/mol','Matallic_radius_nm','Pauling_EN','Ionic_radius_nm','Oxidation_state','Metal_conc_mM','Dipole_moment_D','Acid_conc_M','Temperature_K']
tr = pd.read_csv("Training_Data_V27.csv", low_memory=False); te = pd.read_csv("Testing_Data_V39.csv", low_memory=False)
df = pd.concat([tr, te], ignore_index=True)
num = sorted(set(tr.select_dtypes(np.number).columns) & set(te.select_dtypes(np.number).columns))
allf = [c for c in num if c != TARGET and c not in TEXT]
df = df[df[allf + [TARGET]].notna().all(axis=1) & df[SMI].notna()].reset_index(drop=True)
kdf = df[[SMI, 'Metal'] + [c for c in ['Acid_type'] if c in df]].copy()
for c, r in [('Acid_conc_M', 2), ('Temperature_K', 0), ('Extractant_conc_M', 3), ('Metal_conc_mM', 4)]:
    if c in df: kdf[c] = df[c].round(r)
grng = pd.Series(df[TARGET].values).groupby(kdf.astype(str).agg('|'.join, axis=1)).transform(lambda v: v.max() - v.min()).values
df = df[(~df.duplicated().values) & (grng <= 2.0)].reset_index(drop=True)
y = df[TARGET].values.astype(float)
def san(M):
    M = M.astype(np.float64).copy(); b = ~np.isfinite(M) | (np.abs(M) > 1e30); md = np.nan_to_num(np.nanmedian(np.where(b, np.nan, M), axis=0)); ix = np.where(b); M[ix] = np.take(md, ix[1])
    for j in np.where(np.nanmax(np.abs(M), axis=0) > 1e7)[0]: M[:, j] = np.sign(M[:, j]) * np.log1p(np.abs(M[:, j]))
    return M
X = san(df[[c for c in COND if c in df]].values)
rng = np.random.RandomState(SEED); idx = rng.permutation(len(y))
test = idx[:TEST_N]; pool = idx[TEST_N:]; seed0 = pool[:SEED_N]
truebest = set(pool[np.argsort(-y[pool])[:int(0.10 * len(pool))]].tolist())
def bag(Xtr, ytr):
    return [lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.7, colsample_bytree=0.7, subsample_freq=1, random_state=s, n_jobs=-1, verbosity=-1).fit(Xtr, ytr) for s in range(KBAG)]
def run(strategy):
    lab = list(seed0); unlab = [i for i in pool if i not in set(seed0)]; r2s, recs = [], []
    arng = np.random.RandomState(0)
    for _ in range(ROUNDS):
        models = bag(X[lab], y[lab])
        tp = np.mean([m.predict(X[test]) for m in models], 0); r2s.append(r2_score(y[test], tp))
        recs.append(len(set(lab) & truebest) / len(truebest))
        if not unlab: break
        ua = np.array(unlab); preds = np.array([m.predict(X[ua]) for m in models]); mu = preds.mean(0); sd = preds.std(0)
        score = {'random': arng.rand(len(ua)), 'greedy': mu, 'ucb': mu + 1.0 * sd, 'uncertainty': sd}[strategy]
        pick = ua[np.argsort(-score)[:BATCH]]
        lab += pick.tolist(); pks = set(pick.tolist()); unlab = [i for i in unlab if i not in pks]
    return r2s, recs
res = {s: run(s) for s in ['random', 'greedy', 'ucb', 'uncertainty']}
print(f"sequential active learning | pool {len(pool)}, seed {SEED_N}, +{BATCH}/round x {ROUNDS}, test {TEST_N}")
print(f"\n{'strategy':<12} {'final test R2':>13} {'final discovery':>16}  (discovery = share of the pool's true top-10% found)")
rows = []
for s, (r2s, recs) in res.items():
    print(f"  {s:<10} {r2s[-1]:>13.3f} {recs[-1]:>16.3f}")
    rows.append({'strategy': s, 'final_test_R2': round(r2s[-1], 3), 'final_discovery': round(recs[-1], 3), 'R2_by_round': ';'.join(f'{v:.3f}' for v in r2s), 'discovery_by_round': ';'.join(f'{v:.3f}' for v in recs)})
pd.DataFrame(rows).to_csv("active_learning_results.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
xr = range(1, ROUNDS + 1); col = {'random': '#999999', 'greedy': '#B5402E', 'ucb': '#1F4E79', 'uncertainty': '#2E8B57'}
for s, (r2s, recs) in res.items():
    ax[0].plot(xr, r2s, marker='o', ms=3, label=s, color=col[s]); ax[1].plot(xr, recs, marker='o', ms=3, label=s, color=col[s])
ax[0].set_xlabel('round'); ax[0].set_ylabel('test R-squared'); ax[0].set_title('Model quality over rounds'); ax[0].legend(fontsize=8)
ax[1].set_xlabel('round'); ax[1].set_ylabel('share of true top-10% found'); ax[1].set_title('Discovery of strong extractants'); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig('figures/active_learning_ucb.png', bbox_inches='tight', facecolor='white'); plt.close(fig)
_im = Image.open('figures/active_learning_ucb.png').convert('RGBA'); _bg = Image.new('RGB', _im.size, (255, 255, 255)); _bg.paste(_im, mask=_im.split()[3]); _bg.save('figures/active_learning_ucb.png')
print(f"\nsaved active_learning_results.csv and figures/active_learning_ucb.png | total {(time.time()-START)/60:.1f} min")
