#!/usr/bin/env python3
"""sep_factor_zhang_fig.py — Dr. Zhang's XGBoost model vs our model on the separation factor
(differenced logD), at full coverage, in both regimes. Shows they are about equal."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
# 100%-coverage separation metrics, [ours, his]
DATA = {
    "known extractant": {"signed R²": [0.356, 0.369], "direction acc": [0.726, 0.715], "Spearman": [0.601, 0.604]},
    "new extractant":   {"signed R²": [0.188, 0.227], "direction acc": [0.656, 0.672], "Spearman": [0.462, 0.507]},
}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
for ax, (regime, mets) in zip(axes, DATA.items()):
    labels = list(mets.keys()); x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w/2, [mets[m][0] for m in labels], w, label="our model (LightGBM + descriptors)", color="#2c7fb8")
    ax.bar(x + w/2, [mets[m][1] for m in labels], w, label="Zhang XGBoost (+ ECFP)", color="#fec44f")
    for i, m in enumerate(labels):
        ax.text(i - w/2, mets[m][0] + 0.01, f"{mets[m][0]:.2f}", ha="center", fontsize=8)
        ax.text(i + w/2, mets[m][1] + 0.01, f"{mets[m][1]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_title(regime, fontsize=11); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 0.85)
axes[0].set_ylabel("score"); axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle("Separation factor between two f-elements: our model vs Dr. Zhang's (same data)", fontsize=12)
plt.tight_layout(); plt.savefig("sep_factor_zhang.png", dpi=150, bbox_inches="tight")
print("saved sep_factor_zhang.png")
