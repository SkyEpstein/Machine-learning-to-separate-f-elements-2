#!/usr/bin/env python3
"""sep_factor_confidence_fig.py — does separation prediction improve with confidence?
Two shrinkage-safe views vs confidence coverage, for both regimes. Left: direction accuracy
(not subject to variance shrinkage). Right: normalized error RMSE/target_std (divides out the
narrowing spread of the retained subset, so a drop here is genuine skill, not shrinkage)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd
d = pd.read_csv("sep_factor_confidence_results.csv")
d["covpct"] = d["coverage"].str.rstrip("%").astype(int)
d["norm_rmse"] = d["signed_RMSE"] / d["target_std"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
col = {"known extractant": "#2c7fb8", "new extractant": "#d95f0e"}
for g, sub in d.groupby("regime"):
    sub = sub.sort_values("covpct", ascending=False)
    ax1.plot(sub["covpct"], sub["direction_acc"], "o-", color=col[g], label=g, lw=2)
    ax2.plot(sub["covpct"], sub["norm_rmse"], "o-", color=col[g], label=g, lw=2)
ax1.axhline(0.5, color="grey", ls="--", lw=1, label="chance")
ax1.set_xlabel("confidence coverage (% most-confident pairs kept)"); ax1.set_ylabel("direction accuracy (which f-element wins)")
ax1.set_title("Direction accuracy vs confidence\n(known improves, new slips)", fontsize=11)
ax1.invert_xaxis(); ax1.grid(alpha=0.3); ax1.legend(fontsize=9); ax1.set_ylim(0.5, 0.85)
ax2.axhline(1.0, color="grey", ls=":", lw=1, label="no skill (RMSE = spread)")
ax2.set_xlabel("confidence coverage (% most-confident pairs kept)"); ax2.set_ylabel("normalized error  RMSE / target spread")
ax2.set_title("Shrinkage-corrected error vs confidence\n(known drops genuinely, new is flat)", fontsize=11)
ax2.invert_xaxis(); ax2.grid(alpha=0.3); ax2.legend(fontsize=9)
plt.tight_layout(); plt.savefig("sep_factor_confidence.png", dpi=150, bbox_inches="tight")
print("saved sep_factor_confidence.png")
