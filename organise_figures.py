"""
Organise Figures for the Paper
==============================

Project
-------
Synthetic Data Generation and Predictive Utility Evaluation
for Primary Biliary Cirrhosis (PBC) — Mayo Clinic Dataset

Purpose of this module
-----------------------
The generative pipeline, notebook 03, and the auxiliary analysis scripts each
save figures to output/figures/ under their own working names (fig0_..., fig9_...,
tsne_..., etc.). This script copies the twenty figures the paper actually uses
into paper/figures/ under a single, ordered naming scheme (fig01 ... fig20) that
matches the order the figures appear in the manuscript.

Run this after regenerating the figures so the paper always references a clean,
consecutively numbered set:

    python organise_figures.py
"""

import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, "output", "figures")
DST_DIR  = os.path.join(BASE_DIR, "paper", "figures")

os.makedirs(DST_DIR, exist_ok=True)

# (paper figure number, source filename, destination filename)
FIGURE_MAP = [
    ( 1, "eda_correlation_matrix.png",    "fig01_correlation_matrix.png"),
    ( 2, "fig9_pipeline_flowchart.png",   "fig02_pipeline_flowchart.png"),
    ( 3, "fig0_training_losses.png",      "fig03_training_losses.png"),
    ( 4, "fig2_iqr_filtering.png",        "fig04_iqr_filtering.png"),
    ( 5, "fig5_consensus_distribution.png","fig05_consensus_distribution.png"),
    ( 6, "fig1_fid_comparison.png",       "fig06_fid_comparison.png"),
    ( 7, "fig3_distribution_comparison.png","fig07_distribution_comparison.png"),
    ( 8, "fig4_correlation_heatmap.png",  "fig08_correlation_heatmap.png"),
    ( 9, "tsne_real_vs_synthetic.png",    "fig09_tsne_projection.png"),
    (10, "privacy_risk_analysis.png",     "fig10_privacy_risk_analysis.png"),
    (11, "roc_curves.png",                "fig11_roc_curves.png"),
    (12, "pr_curves.png",                 "fig12_pr_curves.png"),
    (13, "auc_detailed_comparison.png",   "fig13_auc_comparison.png"),
    (14, "fig6_performance_heatmap.png",  "fig14_performance_heatmap.png"),
    (15, "fig7_model_comparison.png",     "fig15_model_comparison.png"),
    (16, "fig8_all_metrics.png",          "fig16_all_metrics.png"),
    (17, "feature_importance.png",        "fig17_feature_importance.png"),
    (18, "power_analysis.png",            "fig18_power_analysis.png"),
    (19, "subgroup_analysis.png",         "fig19_subgroup_analysis.png"),
    (20, "privacy_enhancement.png",       "fig20_privacy_enhancement.png"),
]


def main():
    copied, missing = 0, []
    for num, src_name, dst_name in FIGURE_MAP:
        src = os.path.join(SRC_DIR, src_name)
        dst = os.path.join(DST_DIR, dst_name)
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            print(f"  Fig {num:>2}: {src_name:<34} ->  {dst_name}")
            copied += 1
        else:
            missing.append((num, src_name))

    print(f"\n  Copied {copied}/{len(FIGURE_MAP)} figures to {DST_DIR}")
    if missing:
        print("  MISSING source figures (regenerate these):")
        for num, src_name in missing:
            print(f"    Fig {num}: {src_name}")


if __name__ == "__main__":
    main()
