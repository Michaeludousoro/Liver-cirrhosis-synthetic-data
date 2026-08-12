"""
TVAE Privacy Enhancement Analysis
====================================

The filtered TVAE pool has a near-duplicate rate of 38.2%, meaning that
38.2% of its 500 generated records fall within the 5th percentile of
real-to-real nearest-neighbour distances in standardised feature space.
This matches the method used in the original notebook (Section 12).
The threshold is data-driven rather than fixed, and uses all 18 feature
columns (continuous, binary, and ordinal) for a comprehensive measure.

This script addresses that limitation in two ways:

    1. Output perturbation analysis.
       Calibrated Gaussian noise is added to the continuous features of
       TVAE-generated records after generation. This moves records away from
       their nearest real patients, reducing the near-duplicate rate.
       The privacy-utility tradeoff is measured by reporting near-duplicate
       rate and FID at each noise level.

    2. Formal differential privacy (DP-SGD) epsilon estimation.
       The moments accountant formula for the Gaussian mechanism is used to
       estimate what (epsilon, delta)-DP guarantee a DP-SGD retrained TVAE
       would achieve, across a range of noise multiplier values. This provides
       the theoretical complement to the empirical output perturbation.

Results are saved as a figure and a CSV. No model retraining is required.
"""

import os, sys
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
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.fid_calculator import compute_fid
from src.data_loader import CONTINUOUS_COLS, ALL_FEATURE_COLS

BASE    = os.path.dirname(os.path.abspath(__file__))
DATA    = os.path.join(BASE, "output", "data")
RESULTS = os.path.join(BASE, "output", "results")
FIGURES = os.path.join(BASE, "output", "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)


def compute_risk_threshold(real_df, feat_cols):
    """
    Compute the 5th percentile of real-to-real nearest-neighbour distances.

    This matches the original notebook method. Any synthetic record closer
    to a real patient than this threshold is flagged as a near-duplicate.
    Using the 5th percentile means 5% of real patients are already 'at risk'
    relative to each other, giving a meaningful baseline for comparison.
    """
    cols = [c for c in feat_cols if c in real_df.columns]
    sc = StandardScaler().fit(real_df[cols].values.astype(float))
    real_s = sc.transform(real_df[cols].values.astype(float))

    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(real_s)
    dists, _ = nn.kneighbors(real_s)
    real_real_min = dists[:, 1]
    return np.percentile(real_real_min, 5), sc, cols


def near_duplicate_rate(real_df, synthetic_df, threshold, scaler, feat_cols):
    """
    Fraction of synthetic records within threshold standardised distance
    of any real training patient.

    Uses the same scaler fitted on real data so the feature space is
    consistent across all calls.
    """
    syn_cols = [c for c in feat_cols if c in synthetic_df.columns]
    real_s = scaler.transform(real_df[syn_cols].values.astype(float))
    syn_s  = scaler.transform(synthetic_df[syn_cols].values.astype(float))

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(real_s)
    dists, _ = nn.kneighbors(syn_s)
    min_dists = dists[:, 0]
    near_dup = (min_dists < threshold).sum()
    return near_dup / len(syn_s), int(near_dup)


def add_output_perturbation(synthetic_df, sigma, train_df, continuous_cols, rng):
    """
    Add zero-mean Gaussian noise to continuous features of synthetic records.

    Noise is scaled relative to each feature's standard deviation in the
    real training data so that sigma=1.0 adds noise equal to one standard
    deviation of that feature.
    """
    df = synthetic_df.copy()
    for col in continuous_cols:
        if col in df.columns and col in train_df.columns:
            feature_std = train_df[col].std()
            noise = rng.normal(0, sigma * feature_std, size=len(df))
            df[col] = df[col] + noise
            df[col] = df[col].clip(lower=0)
    return df


def compute_dp_epsilon(n_train, n_epochs, batch_size, noise_multiplier,
                       delta=1e-5, alpha_orders=None):
    """
    Estimate (epsilon, delta)-DP using the Gaussian mechanism moments accountant.

    This uses the Renyi DP composition rule for the Gaussian mechanism with
    Poisson subsampling, following the approach of Mironov et al. (2017) and
    Wang et al. (2019).

    The result is an estimate; exact computation requires the full moments
    accountant implementation. This approximation is used for presentation
    purposes to indicate the order of magnitude of the privacy budget.
    """
    if alpha_orders is None:
        alpha_orders = list(range(2, 65))

    q     = batch_size / n_train            # subsampling rate
    steps = int(n_epochs * n_train / batch_size)  # total gradient steps

    best_eps = float('inf')
    for alpha in alpha_orders:
        # RDP guarantee for one step of Gaussian mechanism with subsampling
        # Approximation using the dominant term of the Renyi divergence
        # D_alpha(M(S) || M(S')) ≤ (q^2 * alpha) / (2 * sigma^2)
        rdp_per_step = (q ** 2 * alpha) / (2 * noise_multiplier ** 2)

        # Composition: T steps
        rdp_total = steps * rdp_per_step

        # Convert RDP to (epsilon, delta)-DP
        # epsilon = rdp_total + log(1/delta) / (alpha - 1)
        eps = rdp_total + np.log(1 / delta) / (alpha - 1)
        if eps < best_eps:
            best_eps = eps

    return best_eps


def run_output_perturbation(train_df, filtered_tvae):
    """
    Test noise levels and report near-duplicate rate and FID for each.

    Uses the same threshold and scaler as the original notebook so the
    baseline (sigma=0.0) reproduces the 38.2% figure from the paper.
    """
    rng         = np.random.default_rng(42)
    cont_cols   = [c for c in CONTINUOUS_COLS if c in filtered_tvae.columns]
    sigma_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    rows = []

    threshold, scaler, feat_cols = compute_risk_threshold(train_df, ALL_FEATURE_COLS)
    print(f"Risk threshold (5th pct real-to-real distance): {threshold:.4f}")
    print()
    print("Output perturbation analysis (noise added to TVAE continuous features):")

    for sigma in sigma_values:
        if sigma == 0.0:
            perturbed = filtered_tvae.copy()
        else:
            perturbed = add_output_perturbation(filtered_tvae, sigma, train_df, cont_cols, rng)

        nd_rate, nd_count = near_duplicate_rate(train_df, perturbed, threshold, scaler, feat_cols)
        fid = compute_fid(train_df, perturbed)

        label = "No noise (original)" if sigma == 0.0 else f"sigma = {sigma}"
        print(f"  {label:<22}: near-dup rate = {nd_rate*100:.1f}%  ({nd_count}/{len(perturbed)})  FID = {fid:.4f}")
        rows.append({"sigma": sigma, "near_dup_rate_pct": round(nd_rate*100, 1),
                     "near_dup_count": nd_count, "FID": round(fid, 4)})

    return pd.DataFrame(rows)


def run_dp_estimation(n_train=193, n_epochs=300, batch_size=32, delta=1e-5):
    """
    Estimate epsilon across a range of noise multiplier values for DP-TVAE.
    """
    sigmas  = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
    rows = []

    print()
    print(f"DP-SGD epsilon estimates (n={n_train}, epochs={n_epochs}, batch={batch_size}, delta={delta}):")
    print(f"  {'Noise multiplier':>18}  {'Epsilon':>10}  {'Interpretation'}")
    print(f"  {'-'*18}  {'-'*10}  {'-'*30}")

    for sigma in sigmas:
        eps = compute_dp_epsilon(n_train, n_epochs, batch_size, sigma, delta)
        if eps > 100:
            interp = "Essentially no DP guarantee"
        elif eps > 10:
            interp = "Very weak privacy guarantee"
        elif eps > 3:
            interp = "Weak guarantee (epsilon > 3)"
        elif eps > 1:
            interp = "Moderate guarantee (1 < eps <= 3)"
        else:
            interp = "Strong guarantee (epsilon <= 1)"
        print(f"  {sigma:>18.1f}  {eps:>10.2f}  {interp}")
        rows.append({"noise_multiplier": sigma, "epsilon": round(eps, 4),
                     "interpretation": interp})

    return pd.DataFrame(rows)


def plot_results(perturb_df, dp_df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    ax1 = axes[0]
    ax1.plot(perturb_df["sigma"], perturb_df["near_dup_rate_pct"],
             color="#ef4444", linewidth=2.5, marker="o", markersize=8)
    ax1.axhline(5.0, color="black", linestyle="--", linewidth=1.2,
                label="5% target")
    ax1.set_xlabel("Noise $\\sigma$ (feature SD)", fontsize=11*_S)
    ax1.set_ylabel("Near-duplicate rate (%)", fontsize=11*_S)
    ax1.set_title("Privacy: near-duplicate rate\nvs output noise level", fontsize=11*_S)
    ax1.legend(fontsize=9*_S)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(perturb_df["sigma"], perturb_df["FID"],
             color="#3b82f6", linewidth=2.5, marker="s", markersize=8)
    ax2.axhline(0.0862, color="#059669", linestyle="--", linewidth=1.2,
                label="Consensus FID 0.0862")
    ax2.set_xlabel("Noise $\\sigma$ (feature SD)", fontsize=11*_S)
    ax2.set_ylabel("FID score (lower is better)", fontsize=11*_S)
    ax2.set_title("Data quality: FID score\nvs output noise level", fontsize=11*_S)
    ax2.legend(fontsize=9*_S)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    valid_dp = dp_df[dp_df["epsilon"] <= 100]
    ax3.plot(valid_dp["noise_multiplier"], valid_dp["epsilon"],
             color="#7c3aed", linewidth=2.5, marker="^", markersize=8)
    ax3.axhline(3.0, color="#f59e0b", linestyle="--", linewidth=1.2,
                label=r"$\varepsilon=3$ (weak)")
    ax3.axhline(1.0, color="#059669", linestyle="--", linewidth=1.2,
                label=r"$\varepsilon=1$ (strong)")
    ax3.set_xlabel("DP-SGD noise multiplier", fontsize=11*_S)
    ax3.set_ylabel("Privacy budget $\\varepsilon$", fontsize=11*_S)
    ax3.set_title("DP-SGD: estimated epsilon\nvs noise multiplier", fontsize=11*_S)
    ax3.legend(fontsize=9*_S)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 30)

    plt.tight_layout()
    out = os.path.join(FIGURES, "privacy_enhancement.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"\nFigure saved: {out}")


def main():
    train_df      = pd.read_csv(os.path.join(DATA, "train_real.csv"))
    filtered_tvae = pd.read_csv(os.path.join(DATA, "filtered_tvae.csv"))
    print(f"Loaded: train {len(train_df)} | TVAE {len(filtered_tvae)}")
    print()

    perturb_df = run_output_perturbation(train_df, filtered_tvae)
    dp_df      = run_dp_estimation()

    perturb_df.to_csv(os.path.join(RESULTS, "output_perturbation.csv"), index=False)
    dp_df.to_csv(os.path.join(RESULTS, "dp_epsilon_estimates.csv"), index=False)

    plot_results(perturb_df, dp_df)

    print()
    opt = perturb_df[perturb_df["near_dup_rate_pct"] <= 5.0]
    if not opt.empty:
        best = opt.iloc[0]
        print(f"Optimal output perturbation point:")
        print(f"  sigma = {best['sigma']}  ->  near-dup rate = {best['near_dup_rate_pct']}%"
              f"  FID = {best['FID']}")
    print()
    print("Results saved.")


if __name__ == "__main__":
    main()
