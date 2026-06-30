#!/usr/bin/env python3
"""sep_factor_eval_fig.py — two-panel figure for the f-element separation-factor evaluation.
Left: the metric suite, known vs new extractant. Right: per-pair direction accuracy against
the true separation size, showing the order is predicted well even when the gap is tiny."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
res = pd.read_csv("sep_factor_eval_results.csv").set_index("regime")
pair = pd.read_csv("sep_factor_eval_by_pair.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
mets = ["signed_R2", "magnitude_R2", "direction_acc", "spearman_signed", "useful>=1.0_F1"]
lab = ["signed Δ\nR²", "|Δ| magnitude\nR²", "direction\naccuracy", "Spearman\n(signed Δ)", "useful sep.\nF1 (|Δ|≥1)"]
x = np.arange(len(mets)); w = 0.38
ax1.bar(x - w/2, [res.loc["known extractant", m] for m in mets], w, label="known extractant", color="#2c7fb8")
ax1.bar(x + w/2, [res.loc["new extractant", m] for m in mets], w, label="new extractant", color="#c7e9b4")
ax1.axhline(0, color="k", lw=0.8); ax1.set_xticks(x); ax1.set_xticklabels(lab, fontsize=9)
ax1.set_title("Separation factor between two f-elements\n(differencing the logD model)", fontsize=11)
ax1.set_ylabel("score"); ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)
ax2.scatter(pair.mean_abs_delta, pair.dir_acc_known, s=40, color="#2c7fb8", label="known extractant", zorder=3)
ax2.scatter(pair.mean_abs_delta, pair.dir_acc_new, s=40, color="#7fbf7b", marker="s", label="new extractant", zorder=3)
ax2.axhline(0.5, color="grey", ls="--", lw=1, label="chance (0.5)")
for _, r in pair.iterrows():
    if r.mean_abs_delta < 0.85 or r.dir_acc_known > 0.95:
        ax2.annotate(r["pair"].replace("(III)", "").replace("(IV)", ""), (r.mean_abs_delta, r.dir_acc_known), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
ax2.set_xlabel("true mean |Δ logD| of the pair  (separation size)"); ax2.set_ylabel("direction accuracy (which metal wins)")
ax2.set_title("Order is predicted well even for tiny gaps;\nsize (and new extractants) are harder", fontsize=11)
ax2.legend(fontsize=9); ax2.grid(alpha=0.3); ax2.set_ylim(0.35, 1.03)
plt.tight_layout(); plt.savefig("sep_factor_eval.png", dpi=150, bbox_inches="tight")
print("saved sep_factor_eval.png")
