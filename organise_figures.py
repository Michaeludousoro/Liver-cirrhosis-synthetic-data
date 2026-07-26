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
tsne_..., etc.). This script copies them into paper/figures/ under an ordered
naming scheme that matches the order they appear in the manuscript.

Twenty figures proved too many for an eleven-page manuscript, so the set is split
in two. Ten figures that carry the argument stay in the main paper as
fig01 ... fig10. Ten that are either secondary detail or duplicate information
already given in the tables move to supplementary material as figS01 ... figS10,
where the manuscript cites them as Fig. S1 ... Fig. S10.

Run this after regenerating the figures so the paper always references a clean,
consecutively numbered set:

    python organise_figures.py
    python optimise_figures.py
"""

import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, "output", "figures")
DST_DIR  = os.path.join(BASE_DIR, "paper", "figures")
SUP_DIR  = os.path.join(BASE_DIR, "paper", "figures_supplementary")

os.makedirs(DST_DIR, exist_ok=True)
os.makedirs(SUP_DIR, exist_ok=True)

# Main manuscript: the ten figures that carry the argument.
# (paper figure number, source filename, destination filename)
FIGURE_MAP = [
    ( 1, "fig9_pipeline_flowchart.png",   "fig01_pipeline_flowchart.png"),
    ( 2, "fig2_iqr_filtering.png",        "fig02_iqr_filtering.png"),
    ( 3, "fig5_consensus_distribution.png","fig03_consensus_distribution.png"),
    ( 4, "fig1_fid_comparison.png",       "fig04_fid_comparison.png"),
    ( 5, "privacy_risk_analysis.png",     "fig05_privacy_risk_analysis.png"),
    ( 6, "roc_curves.png",                "fig06_roc_curves.png"),
    ( 7, "feature_importance.png",        "fig07_feature_importance.png"),
    ( 8, "power_analysis.png",            "fig08_power_analysis.png"),
    ( 9, "subgroup_analysis.png",         "fig09_subgroup_analysis.png"),
    (10, "privacy_enhancement.png",       "fig10_privacy_enhancement.png"),
]

# Supplementary material: secondary detail, or views whose numbers already
# appear in Tables I to VI of the main paper.
SUPPLEMENTARY_MAP = [
    ( 1, "eda_correlation_matrix.png",    "figS01_correlation_matrix.png"),
    ( 2, "fig0_training_losses.png",      "figS02_training_losses.png"),
    ( 3, "fig3_distribution_comparison.png","figS03_distribution_comparison.png"),
    ( 4, "fig4_correlation_heatmap.png",  "figS04_correlation_heatmap.png"),
    ( 5, "tsne_real_vs_synthetic.png",    "figS05_tsne_projection.png"),
    ( 6, "pr_curves.png",                 "figS06_pr_curves.png"),
    ( 7, "auc_detailed_comparison.png",   "figS07_auc_comparison.png"),
    ( 8, "fig6_performance_heatmap.png",  "figS08_performance_heatmap.png"),
    ( 9, "fig7_model_comparison.png",     "figS09_model_comparison.png"),
    (10, "fig8_all_metrics.png",          "figS10_all_metrics.png"),
]


def copy_set(mapping, dst_dir, label, prefix):
    """Copy one figure set. Returns (copied count, list of missing sources)."""
    copied, missing = 0, []
    for num, src_name, dst_name in mapping:
        src = os.path.join(SRC_DIR, src_name)
        dst = os.path.join(dst_dir, dst_name)
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            print(f"  {prefix}{num:>2}: {src_name:<34} ->  {dst_name}")
            copied += 1
        else:
            missing.append((num, src_name))
    print(f"\n  {label}: copied {copied}/{len(mapping)} to {dst_dir}")
    return copied, missing


def main():
    print("MAIN MANUSCRIPT FIGURES\n")
    _, missing_main = copy_set(FIGURE_MAP, DST_DIR, "Main paper", "Fig ")

    print("\nSUPPLEMENTARY FIGURES\n")
    _, missing_sup = copy_set(SUPPLEMENTARY_MAP, SUP_DIR, "Supplementary", "Fig S")

    missing = missing_main + missing_sup
    if missing:
        print("\n  MISSING source figures (regenerate these):")
        for num, src_name in missing:
            print(f"    {src_name}")


if __name__ == "__main__":
    main()
