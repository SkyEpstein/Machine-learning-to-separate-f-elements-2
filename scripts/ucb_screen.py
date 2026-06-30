#!/usr/bin/env python3
"""
ucb_screen.py — turn the model into a ranked recommended-experiment list. Reads the
committed Track A deployable predictions (logD prediction plus the calibrated 90
percent interval), ranks candidates by the upper confidence bound (the optimistic
top of the interval, hi90), maps the prediction to the three distribution-coefficient
zones (low D<0.5, medium 0.5-10, high D>10), and emits the best experiment per
extractant molecule. Self-contained: depends only on results/deploy_A_screening_predictions.csv.

Note: the deployable file contains already-measured systems, so this run is the
screening machinery demonstrated on known rows. For a prospective campaign, point it
at an unmeasured candidate pool (same columns) and drop rows that already have an
Actual_LogD; the ranking logic is identical.
"""
import numpy as np, pandas as pd
SRC = "deploy_A_screening_predictions.csv"
d = pd.read_csv(SRC)
def zone(l):
    return "high (D>10)" if l > 1.0 else ("medium (0.5<=D<=10)" if l >= -0.301 else "low (D<0.5)")
d["predicted_zone"] = d["Pred_LogD"].apply(zone)
d["UCB_logD"] = d["hi90"]                       # optimistic upper bound = UCB for strong extraction
already_measured = d["Actual_LogD"].notna().sum() if "Actual_LogD" in d else 0
# best (highest-UCB) experiment per extractant molecule
best = d.sort_values("UCB_logD", ascending=False).drop_duplicates("SMILES").reset_index(drop=True)
best.insert(0, "rank", np.arange(1, len(best) + 1))
cols = [c for c in ["rank", "SMILES", "Metal", "Acid_conc_M", "Temperature_K", "Pred_LogD", "lo90", "hi90", "UCB_logD", "confidence_pred_err", "predicted_zone"] if c in best.columns]
best[cols].to_csv("ucb_recommended_experiments.csv", index=False)
print(f"screened {len(d)} candidate rows ({already_measured} already measured) across {d['SMILES'].nunique()} extractants")
print(f"zone mix of all candidates: {dict(d['predicted_zone'].value_counts())}")
print("\ntop 10 recommended experiments (best per molecule, ranked by UCB = upper 90% logD bound):")
print(best[cols].head(10).to_string(index=False))
print("\nsaved ucb_recommended_experiments.csv")
