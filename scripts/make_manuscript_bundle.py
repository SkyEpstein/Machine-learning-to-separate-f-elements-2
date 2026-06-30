#!/usr/bin/env python3
"""
make_manuscript_bundle.py — assemble a publication-ready figure and table bundle for the
manuscript. Writes docs/MANUSCRIPT_FIGURES.md (ordered main figures with paper-ready
captions, a supplementary list, and a headline results table that leads with our model's
confidence companion) and figures/manuscript_figures.pdf (the curated main set in
narrative order, one figure per page with its caption).
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import os
FIG = "figures"
MAIN = [
    ("pred_vs_actual_a.png", "Figure 1. Track A, new-extractant screening: predicted vs actual logD under molecule-grouped cross-validation. Overall R2 0.466 (RMSE 1.148); on the most confident 10 percent, R2 0.912 (RMSE 0.493)."),
    ("pred_vs_actual_b.png", "Figure 2. Track B, condition optimization for a known extractant: predicted vs actual logD. Honest condition-key R2 0.61 (RMSE 0.98); most confident 10 percent R2 0.874 (RMSE 0.430)."),
    ("confidence_curve_a.png", "Figure 3. Accuracy concentrates with confidence (Track A): error falls and R2 rises as predictions are restricted to the most confident fraction. The calibrated confidence layer is the model's deployable contribution."),
    ("conformal_calibration.png", "Figure 4. Calibrated 90 percent prediction intervals: empirical coverage 0.902 on a molecule-disjoint split, so the uncertainty is trustworthy rather than nominal."),
    ("dg.png", "Figure 5. Free energy of extraction (delta G), per-pair framing: R2 0.46 plus or minus 0.01 (RMSE 6.3 kJ/mol); most confident 10 percent RMSE 3.6 kJ/mol. RandomForest chosen by a seven-model bake-off."),
    ("sep_factor_eval.png", "Figure 6. Separation factor between two f-elements (the objective), differencing the logD model: direction is predicted moderately (0.73 known, 0.66 new) and magnitude poorly. The direct delta model failed and is not used."),
    ("sep_factor_confidence.png", "Figure 7. Separation prediction improves with confidence only for a known extractant (direction 0.73 to 0.79; normalized error beats a random-ranking benchmark, adversarially verified). For a new extractant confidence is not net-useful."),
    ("sep_factor_zhang.png", "Figure 8. Our model vs Dr. Zhang's XGBoost on the separation factor (same data): they tie at full coverage, but only our model carries a confidence layer, so only ours has the confidence-filtered operating point (direction 0.789)."),
    ("zhang_comparison_selective.png", "Figure 9. Comparison with Dr. Zhang on logD (same data): about equal in accuracy; our addition is the calibrated, confidence-aware prediction, not higher raw accuracy."),
    ("active_analysis_trends.png", "Figure 10. Experiment prioritization: ranking candidates by prediction concentrates strong extractants near the top (81 percent of top picks in the strongest tertile, 3.6x random)."),
    ("by_metal_pair_separation.png", "Figure 11. Per-pair separation accuracy is highly uneven across f-element pairs: large-gap pairs predict well, adjacent rare earths do not."),
]
SUPP = ["pred_vs_actual stragglers", "confidence_curve_b.png", "coverage_illustration.png", "ensemble_weights.png",
        "by_metal_accuracy.png", "by_metal_confidence_vs_error.png", "by_pair_confidence_vs_error.png", "confidence_per_metal.png",
        "active_analysis.png", "active_learning_ucb.png", "ucb_graphs.png", "picks_trends.png",
        "our_data_vs_zhang_data.png", "zhang_his_split_headtohead.png", "sep_factor_confidence.png (detail)"]
present = [(f, c) for f, c in MAIN if os.path.exists(f"{FIG}/{f}")]
missing = [f for f, c in MAIN if not os.path.exists(f"{FIG}/{f}")]
with PdfPages(f"{FIG}/manuscript_figures.pdf") as pdf:
    for fname, cap in present:
        fig = plt.figure(figsize=(8.5, 8.5))
        ax = fig.add_axes([0.04, 0.16, 0.92, 0.80]); ax.imshow(mpimg.imread(f"{FIG}/{fname}")); ax.axis("off")
        fig.text(0.06, 0.10, cap, fontsize=9, wrap=True, va="top", ha="left")
        pdf.savefig(fig); plt.close(fig)
md = ["# Manuscript figures and tables", "",
      "A curated, publication-ready set for the writeup. The full figure set is in `figures/` and every number is in `docs/RESULTS_TABLE.md`. Per project rule, our model's confidence-filtered figure is shown alongside each headline, since the calibrated confidence layer is the differentiator (Dr. Zhang's model has none).", "",
      "## Headline results (lead the paper with these)", "",
      "| Quantity | Overall | Our model + confidence (top 10%) | Regime |",
      "|---|---|---|---|",
      "| Track A logD (new molecule) | R2 0.466, RMSE 1.148 | R2 0.912, RMSE 0.493 | molecule-grouped CV |",
      "| Track B logD (known molecule) | R2 0.61, RMSE 0.98 | R2 0.874, RMSE 0.430 | condition-key grouped CV |",
      "| Free energy delta G (per-pair) | R2 0.46 +/- 0.01, RMSE 6.3 kJ/mol | RMSE 3.6 kJ/mol (R2 0.776) | molecule-grouped CV |",
      "| 90% interval coverage | 0.902 (target 0.90) | n/a | molecule-disjoint conformal |",
      "| Separation factor (known extractant) | signed R2 0.356, direction 0.726 | direction 0.789, Spearman 0.776, norm RMSE 0.64 | condition-key, matched conditions |",
      "| Separation factor (new extractant) | signed R2 0.188, direction 0.656 | confidence not net-useful | molecule-grouped CV |",
      "| Separation, Zhang XGBoost (known) | signed R2 0.369, direction 0.715 | none (his model has no confidence layer) | condition-key, matched conditions |",
      "", "## Main figures (narrative order)", ""]
for i, (fname, cap) in enumerate(present, 1):
    md.append(f"{i}. `figures/{fname}` - {cap.split('. ', 1)[1] if '. ' in cap else cap}")
md += ["", "## Supplementary figures", "", "The remaining figures in `figures/` support the above (per-metal and per-pair detail, the UCB and active-learning panels, ensemble weights, coverage illustration, and the Zhang head-to-head splits):", ""]
for f in SUPP: md.append(f"- `figures/{f}`")
md += ["", f"Combined main set: `figures/manuscript_figures.pdf` ({len(present)} figures).", ""]
open("MANUSCRIPT_FIGURES.md", "w").write("\n".join(md) + "\n")
print(f"wrote MANUSCRIPT_FIGURES.md and figures/manuscript_figures.pdf ({len(present)} main figures)")
if missing: print("MISSING (skipped):", missing)
