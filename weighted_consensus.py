"""
Quality-Weighted Consensus Voting
===================================

This script compares four consensus approaches:

    1. Original flat voting at tolerance 5.0 (what the paper currently does)
    2. Weighted voting at tolerance 5.0 with threshold 0.5 (same behaviour as flat)
    3. Weighted voting at tolerance 5.0 with threshold 0.6 (requires CTGAN to confirm)
    4. Weighted voting at tolerance 5.0 with threshold 0.6 plus TVAE downsampled
       to 210 records to match the other pool sizes

For each approach, the script reports:
    - How many records are accepted
    - What percentage came from GAN, CTGAN, and TVAE
    - The class balance of accepted records (percent deceased)
    - The FID score versus real training patients

Weights are derived from inverse FID scores:
    GAN   FID 0.0972  ->  weight 36.5 percent
    CTGAN FID 0.0751  ->  weight 47.3 percent
    TVAE  FID 0.2194  ->  weight 16.2 percent

At weighted threshold 0.6, GAN + TVAE (0.527) is below the cutoff, so
CTGAN must always be one of the two confirming models. This gives the best
quality model a gating role in the consensus.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.fid_calculator import compute_fid

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# FID-derived quality weights (lower FID = higher weight)
WEIGHTS = {"GAN": 0.3653, "CTGAN": 0.4728, "TVAE": 0.1618}

FID_SCORES = {"GAN": 0.0972, "CTGAN": 0.0751, "TVAE": 0.2194}


def run_flat_consensus(filtered_gan, filtered_ctgan, filtered_tvae,
                       tolerance=5.0):
    """
    Original flat voting: a record passes if both other models have a
    record within the tolerance distance in standardised feature space.
    """
    methods = {
        "GAN":   filtered_gan.values.astype(float),
        "CTGAN": filtered_ctgan.values.astype(float),
        "TVAE":  filtered_tvae.values.astype(float),
    }
    col_names    = filtered_gan.columns.tolist()
    method_names = list(methods.keys())

    all_data = np.vstack(list(methods.values()))
    scaler   = StandardScaler()
    scaler.fit(all_data)
    scaled   = {n: scaler.transform(v) for n, v in methods.items()}

    accepted_rows = []
    source_labels = []

    for i, cname in enumerate(method_names):
        others = [method_names[j] for j in range(3) if j != i]
        for row in scaled[cname]:
            votes = sum(
                1 for oname in others
                if np.linalg.norm(scaled[oname] - row, axis=1).min() < tolerance
            )
            if votes >= 2:
                accepted_rows.append(row)
                source_labels.append(cname)

    if not accepted_rows:
        return pd.DataFrame(columns=col_names), {}

    original_arr = scaler.inverse_transform(np.array(accepted_rows))
    df = pd.DataFrame(original_arr, columns=col_names).drop_duplicates().reset_index(drop=True)
    source_counts = {n: source_labels.count(n) for n in method_names}
    return df, source_counts


def run_weighted_consensus(filtered_gan, filtered_ctgan, filtered_tvae,
                            tolerance=5.0, weighted_threshold=0.5):
    """
    Quality-weighted voting: each confirming model contributes its
    quality weight (derived from inverse FID). A record passes if the
    sum of confirming weights meets the weighted threshold.

    At threshold 0.5, any two models confirming is sufficient
    (minimum two-model weight sum is GAN + TVAE = 0.527).

    At threshold 0.6, CTGAN must be one of the confirming models
    (GAN + TVAE = 0.527 < 0.6 fails).
    """
    methods = {
        "GAN":   filtered_gan.values.astype(float),
        "CTGAN": filtered_ctgan.values.astype(float),
        "TVAE":  filtered_tvae.values.astype(float),
    }
    col_names    = filtered_gan.columns.tolist()
    method_names = list(methods.keys())

    all_data = np.vstack(list(methods.values()))
    scaler   = StandardScaler()
    scaler.fit(all_data)
    scaled   = {n: scaler.transform(v) for n, v in methods.items()}

    accepted_rows = []
    source_labels = []

    for i, cname in enumerate(method_names):
        others = [method_names[j] for j in range(3) if j != i]
        for row in scaled[cname]:
            weight_sum = sum(
                WEIGHTS[oname]
                for oname in others
                if np.linalg.norm(scaled[oname] - row, axis=1).min() < tolerance
            )
            if weight_sum >= weighted_threshold:
                accepted_rows.append(row)
                source_labels.append(cname)

    if not accepted_rows:
        return pd.DataFrame(columns=col_names), {}

    original_arr = scaler.inverse_transform(np.array(accepted_rows))
    df = pd.DataFrame(original_arr, columns=col_names).drop_duplicates().reset_index(drop=True)
    source_counts = {n: source_labels.count(n) for n in method_names}
    return df, source_counts


def summarise(label, consensus_df, source_counts, train_df):
    """Compute and return a summary row for one consensus approach."""
    n = len(consensus_df)
    if n == 0:
        return {"Approach": label, "Records": 0,
                "GAN%": 0, "CTGAN%": 0, "TVAE%": 0,
                "Deceased%": None, "FID": None}

    total_pre = sum(source_counts.values())
    gan_pct   = round(source_counts.get("GAN",   0) / total_pre * 100, 1)
    ctgan_pct = round(source_counts.get("CTGAN", 0) / total_pre * 100, 1)
    tvae_pct  = round(source_counts.get("TVAE",  0) / total_pre * 100, 1)
    dec_pct   = round(consensus_df["Status"].mean() * 100, 1) if "Status" in consensus_df.columns else None
    fid       = round(compute_fid(train_df, consensus_df), 4)

    return {"Approach": label, "Records": n,
            "GAN%": gan_pct, "CTGAN%": ctgan_pct, "TVAE%": tvae_pct,
            "Deceased%": dec_pct, "FID": fid}


def main():
    train_df       = pd.read_csv(os.path.join(DATA_DIR, "train_real.csv"))
    filtered_gan   = pd.read_csv(os.path.join(DATA_DIR, "filtered_gan.csv"))
    filtered_ctgan = pd.read_csv(os.path.join(DATA_DIR, "filtered_ctgan.csv"))
    filtered_tvae  = pd.read_csv(os.path.join(DATA_DIR, "filtered_tvae.csv"))

    print(f"Loaded: GAN {len(filtered_gan)} | CTGAN {len(filtered_ctgan)} | TVAE {len(filtered_tvae)}")
    print(f"Quality weights: GAN {WEIGHTS['GAN']*100:.1f}%  CTGAN {WEIGHTS['CTGAN']*100:.1f}%  TVAE {WEIGHTS['TVAE']*100:.1f}%")
    print()

    rows = []

    # Approach 1: original flat consensus at 5.0
    print("Running Approach 1: flat voting at tolerance 5.0 ...")
    df1, sc1 = run_flat_consensus(filtered_gan, filtered_ctgan, filtered_tvae, 5.0)
    rows.append(summarise("1. Flat vote (tolerance 5.0, current paper)", df1, sc1, train_df))

    # Approach 2: weighted at 5.0, threshold 0.5 (same as flat in practice)
    print("Running Approach 2: weighted vote at 5.0, threshold 0.5 ...")
    df2, sc2 = run_weighted_consensus(filtered_gan, filtered_ctgan, filtered_tvae, 5.0, 0.5)
    rows.append(summarise("2. Weighted vote (tol 5.0, wt-thr 0.5)", df2, sc2, train_df))

    # Approach 3: weighted at 5.0, threshold 0.6 (requires CTGAN to confirm)
    print("Running Approach 3: weighted vote at 5.0, threshold 0.6 (CTGAN must confirm) ...")
    df3, sc3 = run_weighted_consensus(filtered_gan, filtered_ctgan, filtered_tvae, 5.0, 0.6)
    rows.append(summarise("3. Weighted vote (tol 5.0, wt-thr 0.6)", df3, sc3, train_df))

    # Approach 4: weighted at 5.0, threshold 0.6, TVAE downsampled to 210
    print("Running Approach 4: weighted vote at 5.0, threshold 0.6, TVAE downsampled to 210 ...")
    tvae_down = filtered_tvae.sample(n=210, random_state=42).reset_index(drop=True)
    df4, sc4 = run_weighted_consensus(filtered_gan, filtered_ctgan, tvae_down, 5.0, 0.6)
    rows.append(summarise("4. Weighted vote (tol 5.0, wt-thr 0.6, TVAE=210)", df4, sc4, train_df))

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(RESULTS_DIR, "weighted_consensus_comparison.csv"), index=False)

    # Print table
    print()
    print("=" * 85)
    print("QUALITY-WEIGHTED CONSENSUS COMPARISON")
    print("=" * 85)
    print()
    hdr = f"{'Approach':<45} {'Records':<9} {'GAN%':<7} {'CTGAN%':<9} {'TVAE%':<8} {'Dead%':<7} {'FID':<8}"
    print(hdr)
    print("-" * 85)
    for _, r in results_df.iterrows():
        dec = f"{r['Deceased%']:.1f}" if r["Deceased%"] is not None else "N/A"
        fid = f"{r['FID']:.4f}" if r["FID"] is not None else "N/A"
        print(f"{r['Approach']:<45} {int(r['Records']):<9} {r['GAN%']:<7} {r['CTGAN%']:<9} {r['TVAE%']:<8} {dec:<7} {fid:<8}")

    print()
    print("Key findings:")
    print()

    r1 = results_df.iloc[0]
    r3 = results_df.iloc[2]
    r4 = results_df.iloc[3]

    print(f"  In the current paper (Approach 1), TVAE contributes {r1['TVAE%']}% of the")
    print(f"  consensus records despite having the worst FID score ({FID_SCORES['TVAE']}).")
    print()
    print(f"  With weighted voting at threshold 0.6 (Approach 3), TVAE drops to")
    print(f"  {r3['TVAE%']}% and CTGAN rises to {r3['CTGAN%']}% because CTGAN must always")
    print(f"  be one of the two confirming models. FID improves from {r1['FID']} to {r3['FID']}.")
    print()
    if r4["Records"] > 0:
        print(f"  Adding TVAE downsampling (Approach 4) further reduces TVAE to {r4['TVAE%']}%")
        print(f"  with {int(r4['Records'])} records and FID {r4['FID']}.")
    print()

    # Plot
    plot_comparison(results_df)
    print(f"Figure saved to {os.path.join(FIGURES_DIR, 'weighted_consensus_comparison.png')}")
    print(f"Results saved to {os.path.join(RESULTS_DIR, 'weighted_consensus_comparison.csv')}")


def plot_comparison(results_df):
    valid = results_df.copy()
    labels = [r.split("(")[0].strip() for r in valid["Approach"]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Quality-Weighted Consensus Comparison\n"
        "Flat voting vs. quality-weighted voting (CTGAN as gating model)",
        fontsize=12, fontweight="bold"
    )

    x = np.arange(len(labels))
    width = 0.25

    ax1 = axes[0]
    bars_gan   = ax1.bar(x - width, valid["GAN%"],   width, label="GAN",   color="#ef4444")
    bars_ctgan = ax1.bar(x,         valid["CTGAN%"], width, label="CTGAN", color="#3b82f6")
    bars_tvae  = ax1.bar(x + width, valid["TVAE%"],  width, label="TVAE",  color="#10b981")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("Percentage of accepted records", fontsize=10)
    ax1.set_title("Source breakdown by approach", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim(0, 80)

    ax2 = axes[1]
    ax2.bar(x, valid["Records"], color="#6366f1", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("Records accepted after deduplication", fontsize=10)
    ax2.set_title("Number of records accepted", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")

    ax3 = axes[2]
    fid_vals = [r if r is not None else 0 for r in valid["FID"]]
    bars = ax3.bar(x, fid_vals, color="#f59e0b", alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax3.set_ylabel("FID score (lower is better)", fontsize=10)
    ax3.set_title("Data quality (FID score)", fontsize=11)
    ax3.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, fid_vals):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "weighted_consensus_comparison.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()


if __name__ == "__main__":
    main()
