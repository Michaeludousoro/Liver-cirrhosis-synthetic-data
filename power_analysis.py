"""
Post-Hoc Power Analysis for McNemar's Exact Test
==================================================

Purpose
-------
Fig. 8 of the paper reports the statistical power of McNemar's exact test
on the held-out test set, and how large a test set would be needed to
reliably detect a real accuracy improvement. No script producing this
figure existed in the repository (it was an orphaned artifact) — this is
that missing source.

Method
------
McNemar's exact test reduces to a two-sided exact binomial test on the
n = n10 + n01 discordant pairs against p = 0.5. Given a true probability
p_alt that a discordant pair favours the augmented model, the power of the
exact test at a given n is:

    power(n, p_alt) = P(reject H0 | X ~ Binomial(n, p_alt))

where the rejection region is the set of counts k for which the two-sided
exact binomial test against p=0.5 yields p < alpha. This is computed
directly (not via a normal approximation), matching how mcnemar_scenarios()
in src/statistical_analysis.py performs the test itself.

A true overall accuracy improvement of `delta` on a test set of size
n_test, with n_discordant pairs observed, corresponds to a net advantage
of delta * n_test discordant pairs, giving:

    p_alt = 0.5 + (delta * n_test) / (2 * n_discordant)

This lets us ask two questions with the same underlying formula: given the
n_discordant actually observed in this study's McNemar results, what power
did the test have to detect a plausible accuracy improvement (left panel)?
And, fixing a discordant-pair rate, how large would the test set need to be
for 80% power (right panel)?

This script reads output/results/mcnemar_tests.csv (produced by
src.statistical_analysis.mcnemar_scenarios) to use the actually observed
discordant-pair counts rather than an assumed rate, wherever that file is
available; the right-hand "required sample size" sweep uses the discordant
rate implied by that same data as its fixed assumption.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binom, binomtest

_S = 2.0
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5*_S, "axes.titlesize": 9.5*_S,
    "axes.labelsize": 8.5*_S, "xtick.labelsize": 7.5*_S, "ytick.labelsize": 7.5*_S,
    "legend.fontsize": 7.5*_S, "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.linewidth": 0.8*_S, "lines.linewidth": 1.4*_S, "xtick.major.width": 0.8*_S,
    "ytick.major.width": 0.8*_S, "patch.linewidth": 0.6*_S, "savefig.dpi": 300,
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "output", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "output", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

ALPHA = 0.05


def mcnemar_power(n_discordant, p_alt, alpha=ALPHA):
    """Exact power of McNemar's test (as an exact binomial test) at n_discordant
    discordant pairs against a true discordant-pair proportion p_alt."""
    if n_discordant == 0:
        return 0.0
    reject_ks = [
        k for k in range(n_discordant + 1)
        if binomtest(min(k, n_discordant - k), n_discordant, 0.5,
                      alternative="two-sided").pvalue < alpha
    ]
    return float(sum(binom.pmf(k, n_discordant, p_alt) for k in reject_ks))


def p_alt_from_delta(delta, n_test, n_discordant):
    """Discordant-pair proportion implied by a true overall accuracy
    improvement of `delta` on a test set of size n_test."""
    if n_discordant == 0:
        return 0.5
    net = delta * n_test
    return float(min(0.5 + net / (2 * n_discordant), 0.999))


def required_n_for_power(delta, discordant_rate, target_power=0.80,
                          n_grid=range(20, 1001, 5)):
    """Smallest test-set size (within n_grid) reaching target_power to
    detect a true accuracy improvement of `delta`, holding the discordant
    pair rate fixed."""
    for n_test in n_grid:
        n_disc = max(round(discordant_rate * n_test), 1)
        p_alt = p_alt_from_delta(delta, n_test, n_disc)
        if mcnemar_power(n_disc, p_alt) >= target_power:
            return n_test
    return None


def load_observed_discordant_rate(n_test):
    """Read the actually observed McNemar discordant-pair counts, if
    available, and return their rate relative to n_test. Falls back to a
    stated 15% assumption (documented as such) if no results file exists
    yet, so this script is runnable standalone."""
    path = os.path.join(RESULTS_DIR, "mcnemar_tests.csv")
    if not os.path.exists(path):
        print(f"  {path} not found; assuming a 15% discordant-pair rate.")
        return 0.15, None

    mc_df = pd.read_csv(path)
    mc_df["n_discordant"] = mc_df["n10"] + mc_df["n01"]
    mean_n_disc = mc_df["n_discordant"].mean()
    rate = mean_n_disc / n_test
    print(f"  Observed mean discordant pairs across {len(mc_df)} "
          f"comparisons: {mean_n_disc:.1f} ({rate*100:.1f}% of n_test={n_test})")
    return rate, mean_n_disc


def run_power_analysis(n_test):
    discordant_rate, observed_mean_n_disc = load_observed_discordant_rate(n_test)
    n_discordant = (round(observed_mean_n_disc) if observed_mean_n_disc is not None
                    else round(discordant_rate * n_test))

    deltas_curve = [0.05, 0.10, 0.15]
    n_disc_grid = list(range(2, max(n_test, n_discordant + 20)))

    power_curves = {}
    for delta in deltas_curve:
        powers = []
        for n_disc in n_disc_grid:
            p_alt = p_alt_from_delta(delta, n_test, n_disc)
            powers.append(mcnemar_power(n_disc, p_alt))
        power_curves[delta] = powers

    power_at_observed = {
        delta: mcnemar_power(n_discordant, p_alt_from_delta(delta, n_test, n_discordant))
        for delta in deltas_curve
    }

    delta_grid = np.round(np.arange(0.02, 0.21, 0.01), 2)
    required_n = {
        delta: required_n_for_power(delta, discordant_rate) for delta in delta_grid
    }

    summary_rows = [{
        "n_test": n_test,
        "n_discordant_observed": n_discordant,
        "discordant_rate": round(discordant_rate, 4),
        "delta": delta,
        "power_at_observed_n": round(power_at_observed[delta], 4),
        "required_n_for_80pct_power": required_n[delta],
    } for delta in deltas_curve]
    for delta in delta_grid:
        if delta not in deltas_curve:
            summary_rows.append({
                "n_test": n_test, "n_discordant_observed": n_discordant,
                "discordant_rate": round(discordant_rate, 4), "delta": delta,
                "power_at_observed_n": round(
                    mcnemar_power(n_discordant, p_alt_from_delta(delta, n_test, n_discordant)), 4),
                "required_n_for_80pct_power": required_n[delta],
            })

    return (pd.DataFrame(summary_rows).sort_values("delta").reset_index(drop=True),
            n_disc_grid, power_curves, n_discordant, delta_grid, required_n)


def plot_power_analysis(n_test, n_disc_grid, power_curves, n_discordant,
                         delta_grid, required_n):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax1 = axes[0]
    colors = {"0.05": "#1e3a5f", "0.1": "#059669", "0.15": "#dc2626"}
    for delta, powers in power_curves.items():
        ax1.plot(n_disc_grid, powers, linewidth=2,
                 label=f"{int(delta*100)} pp accuracy gain",
                 color=colors.get(str(delta), None))
    ax1.axvline(x=n_discordant, color="gray", linestyle="--", linewidth=1.2,
                label=f"Observed ({n_discordant} pairs)")
    ax1.axhline(y=0.80, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
    ax1.set_xlabel("Number of discordant pairs", fontsize=11*_S)
    ax1.set_ylabel("Statistical power", fontsize=11*_S)
    ax1.set_title(f"McNemar test power ($n$={n_test})", fontsize=10.5*_S)
    ax1.legend(fontsize=7.5*_S, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    ax2 = axes[1]
    valid_deltas = [d for d in delta_grid if required_n[d] is not None]
    valid_n = [required_n[d] for d in valid_deltas]
    ax2.plot(valid_deltas, valid_n, linewidth=2, marker="o", markersize=5,
             color="#7c3aed")
    ax2.set_xlabel("True accuracy improvement", fontsize=11*_S)
    ax2.set_ylabel("Test-set size needed for 80% power", fontsize=11*_S)
    ax2.set_title("Sample size needed to detect an effect", fontsize=10.5*_S)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(w_pad=4.0)
    out_path = os.path.join(FIGURES_DIR, "power_analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Power analysis figure saved to {out_path}")


def main(n_test=83):
    print(f"Running McNemar power analysis for n_test = {n_test} ...")
    (summary_df, n_disc_grid, power_curves,
     n_discordant, delta_grid, required_n) = run_power_analysis(n_test)

    out_csv = os.path.join(RESULTS_DIR, "power_analysis.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")
    print()
    print(summary_df[summary_df["delta"].isin([0.05, 0.10, 0.15])].to_string(index=False))

    plot_power_analysis(n_test, n_disc_grid, power_curves, n_discordant,
                        delta_grid, required_n)


if __name__ == "__main__":
    main()
