"""
Consensus Threshold Sensitivity Analysis
=========================================

This script tests five different threshold values for the consensus voting
mechanism and records, for each threshold:

    - How many synthetic records pass
    - What percentage came from each model (GAN, cGAN, VAE)
    - The class balance of the accepted records (percent deceased)
    - The FID score of the accepted records versus real training patients

The results are printed as a table and saved as a figure and a CSV.
No model retraining is needed. The script loads the filtered synthetic
datasets that were already saved by the main pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Print-safe sizing: matplotlib points are relative to the figure canvas, so a
# wide canvas placed at \\textwidth by LaTeX shrinks every label. Sizes here are
# about twice their intended printed size so they land near 8 pt on the page.
import matplotlib.pyplot as _plt
_S = 2.0
_plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5*_S, "axes.titlesize": 9.5*_S,
    "axes.labelsize": 8.5*_S, "xtick.labelsize": 7.5*_S, "ytick.labelsize": 7.5*_S,
    "legend.fontsize": 7.5*_S, "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.linewidth": 0.8*_S, "lines.linewidth": 1.4*_S, "xtick.major.width": 0.8*_S,
    "ytick.major.width": 0.8*_S, "patch.linewidth": 0.6*_S, "savefig.dpi": 300,
})

from sklearn.preprocessing import StandardScaler

# Allow Python to find the src package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.fid_calculator import compute_fid

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "output", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "output", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "output", "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data():
    """Load the filtered synthetic datasets and real training data."""
    train_df       = pd.read_csv(os.path.join(DATA_DIR, "train_real.csv"))
    filtered_gan   = pd.read_csv(os.path.join(DATA_DIR, "filtered_gan.csv"))
    filtered_ctgan = pd.read_csv(os.path.join(DATA_DIR, "filtered_ctgan.csv"))
    filtered_tvae  = pd.read_csv(os.path.join(DATA_DIR, "filtered_tvae.csv"))

    print(f"Loaded training data:    {len(train_df)} real patients")
    print(f"Loaded filtered GAN:     {len(filtered_gan)} records")
    print(f"Loaded filtered cGAN:   {len(filtered_ctgan)} records")
    print(f"Loaded filtered VAE:    {len(filtered_tvae)} records")
    print()

    return train_df, filtered_gan, filtered_ctgan, filtered_tvae


def run_consensus_at_threshold(filtered_gan, filtered_ctgan, filtered_tvae,
                                tolerance):
    """
    Run consensus voting at a single threshold value.

    A synthetic record is accepted if both of the other two models have at
    least one record within the given tolerance in standardised feature space.

    Returns the accepted records as a DataFrame and a dictionary showing
    how many records came from each model.
    """
    methods = {
        "GAN":   filtered_gan.values.astype(float),
        "cGAN": filtered_ctgan.values.astype(float),
        "VAE":  filtered_tvae.values.astype(float),
    }
    col_names    = filtered_gan.columns.tolist()
    method_names = list(methods.keys())

    all_data = np.vstack(list(methods.values()))
    scaler   = StandardScaler()
    scaler.fit(all_data)
    scaled_methods = {name: scaler.transform(arr) for name, arr in methods.items()}

    accepted_rows = []
    source_labels = []

    for i, candidate_name in enumerate(method_names):
        other_names = [method_names[j] for j in range(3) if j != i]
        candidate   = scaled_methods[candidate_name]

        for row in candidate:
            votes = 0
            for other_name in other_names:
                distances = np.linalg.norm(scaled_methods[other_name] - row, axis=1)
                if distances.min() < tolerance:
                    votes += 1
            if votes >= 2:
                accepted_rows.append(row)
                source_labels.append(candidate_name)

    if len(accepted_rows) == 0:
        return pd.DataFrame(columns=col_names), {n: 0 for n in method_names}

    original_arr = scaler.inverse_transform(np.array(accepted_rows))
    consensus_df = pd.DataFrame(original_arr, columns=col_names)
    consensus_df = consensus_df.drop_duplicates().reset_index(drop=True)

    source_counts = {name: source_labels.count(name) for name in method_names}

    return consensus_df, source_counts


def run_sensitivity_analysis(train_df, filtered_gan, filtered_ctgan, filtered_tvae):
    """
    Test five threshold values and collect results for each.

    For each threshold, the function records how many records pass, the source
    breakdown, the class balance, and the FID score versus real training data.
    """
    thresholds = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    rows = []

    print("Running consensus at each threshold value. This may take a few minutes.")
    print()

    for thr in thresholds:
        print(f"  Threshold {thr} ...", end=" ", flush=True)

        consensus_df, source_counts = run_consensus_at_threshold(
            filtered_gan, filtered_ctgan, filtered_tvae, thr
        )

        n_total = len(consensus_df)

        if n_total == 0:
            print(f"no records passed.")
            rows.append({
                "Threshold":      thr,
                "Records passed": 0,
                "GAN percent":    0.0,
                "cGAN percent":  0.0,
                "VAE percent":   0.0,
                "Deceased percent": None,
                "FID score":      None,
            })
            continue

        pre_dedup_total = sum(source_counts.values())
        gan_pct   = round(source_counts["GAN"]   / pre_dedup_total * 100, 1)
        ctgan_pct = round(source_counts["cGAN"] / pre_dedup_total * 100, 1)
        tvae_pct  = round(source_counts["VAE"]  / pre_dedup_total * 100, 1)

        if "Status" in consensus_df.columns:
            deceased_pct = round(consensus_df["Status"].mean() * 100, 1)
        else:
            deceased_pct = None

        fid = compute_fid(train_df, consensus_df)

        print(f"{n_total} records accepted. FID = {fid:.4f}")

        rows.append({
            "Threshold":        thr,
            "Records passed":   n_total,
            "GAN percent":      gan_pct,
            "cGAN percent":    ctgan_pct,
            "VAE percent":     tvae_pct,
            "Deceased percent": deceased_pct,
            "FID score":        round(fid, 4),
        })

    return pd.DataFrame(rows)


def print_results_table(results_df):
    """Print the results as a readable table with clear column labels."""
    print()
    print("=" * 80)
    print("CONSENSUS THRESHOLD SENSITIVITY ANALYSIS RESULTS")
    print("=" * 80)
    print()
    print(f"{'Threshold':<12} {'Records':<10} {'GAN %':<9} {'cGAN %':<10} "
          f"{'VAE %':<9} {'Deceased %':<13} {'FID score':<10}")
    print("-" * 75)
    for _, row in results_df.iterrows():
        thr    = row["Threshold"]
        n      = int(row["Records passed"]) if not pd.isna(row["Records passed"]) else 0
        gan    = row["GAN percent"]
        ctgan  = row["cGAN percent"]
        tvae   = row["VAE percent"]
        dec    = f"{row['Deceased percent']:.1f}" if row["Deceased percent"] is not None else "N/A"
        fid    = f"{row['FID score']:.4f}" if row["FID score"] is not None else "N/A"
        marker = " <-- original threshold" if thr == 0.5 else ""
        print(f"{thr:<12} {n:<10} {gan:<9} {ctgan:<10} {tvae:<9} {dec:<13} {fid:<10}{marker}")
    print()

    # Key interpretation
    valid = results_df[results_df["FID score"].notna()].copy()
    if len(valid) > 0:
        best_fid_row = valid.loc[valid["FID score"].idxmin()]
        print(f"Best FID score is at threshold {best_fid_row['Threshold']} "
              f"with FID = {best_fid_row['FID score']:.4f} "
              f"and {int(best_fid_row['Records passed'])} records accepted.")
        print()

    print("What this means:")
    print()
    print("  Threshold controls how strict the agreement requirement is.")
    print("  A lower threshold means only records that all three models")
    print("  agree very closely on are accepted, so fewer records pass")
    print("  but they are more conservative. A higher threshold accepts")
    print("  more records but allows more disagreement between models.")
    print()
    print("  The FID score tells you how similar the accepted records are")
    print("  to real patients. Lower FID means better quality.")
    print()
    print("  Use this table to justify the chosen threshold in the article.")
    print()


def plot_sensitivity(results_df):
    """
    Generate a two-panel sensitivity curve figure.

    Left panel shows how many records pass at each threshold.
    Right panel shows the FID score at each threshold.
    The original threshold of 0.5 is marked with a vertical dashed line.
    """
    valid = results_df[results_df["FID score"].notna()].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Left panel: number of records accepted
    ax1 = axes[0]
    ax1.plot(valid["Threshold"], valid["Records passed"],
             color="#1e3a5f", linewidth=2, marker="o", markersize=7)
    ax1.axvline(x=0.5, color="#dc2626", linestyle="--", linewidth=1.2,
                label="Original threshold (0.5)")
    ax1.set_xlabel("Consensus threshold", fontsize=11*_S)
    ax1.set_ylabel("Number of records accepted", fontsize=11*_S)
    ax1.set_title("Records accepted at each threshold", fontsize=11*_S)
    ax1.legend(fontsize=9*_S)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(valid["Threshold"])

    # Right panel: FID score
    ax2 = axes[1]
    ax2.plot(valid["Threshold"], valid["FID score"],
             color="#059669", linewidth=2, marker="s", markersize=7)
    ax2.axvline(x=0.5, color="#dc2626", linestyle="--", linewidth=1.2,
                label="Original threshold (0.5)")
    ax2.set_xlabel("Consensus threshold", fontsize=11*_S)
    ax2.set_ylabel("FID score (lower is better)", fontsize=11*_S)
    ax2.set_title("FID score of accepted records at each threshold", fontsize=11*_S)
    ax2.legend(fontsize=9*_S)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(valid["Threshold"])

    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "threshold_sensitivity.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Sensitivity figure saved to {out_path}")


def main():
    train_df, filtered_gan, filtered_ctgan, filtered_tvae = load_data()

    results_df = run_sensitivity_analysis(
        train_df, filtered_gan, filtered_ctgan, filtered_tvae
    )

    print_results_table(results_df)

    out_csv = os.path.join(RESULTS_DIR, "threshold_sensitivity.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"Results table saved to {out_csv}")
    print()

    plot_sensitivity(results_df)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
