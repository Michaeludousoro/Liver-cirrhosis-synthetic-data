"""
Master Pipeline Runner
======================

Project
-------
Synthetic Data Generation and Predictive Utility Evaluation
for Primary Biliary Cirrhosis (PBC) — Mayo Clinic Dataset

Project Objectives
------------------
Primary Biliary Cirrhosis (PBC) is a rare autoimmune liver disease with a
small patient population. The Mayo Clinic trial enrolled only 312 randomised
patients, and the complete-case dataset (after removing records with missing
values) contains approximately 276 patients. This is a very small sample for
training machine learning models reliably.

Our goal is to investigate whether synthetic data augmentation can improve
the performance of predictive models trained on this limited real data. We
compare three generative approaches: a standard Vanilla GAN, a Conditional
Tabular GAN (CTGAN), and a Tabular Variational Autoencoder (TVAE).

We contribute three methodological innovations:

    1. IQR-based outlier filtering to remove physiologically implausible
       synthetic patient records before they enter any downstream analysis.

    2. Multi-method consensus voting to identify a high-confidence subset of
       synthetic records that all three generative models agree are realistic.

    3. Comprehensive evaluation that goes beyond distributional fidelity metrics
       (FID scores) to assess whether augmentation actually improves the
       downstream classification of patient outcomes.

Pipeline steps
--------------
    Step 1: Load the raw cirrhosis CSV file, remove all patients with any
            missing value, encode categorical features, and split into a 70
            percent training set and a 30 percent held-out test set.

    Step 2: Train three generative models on the scaled training data and
            generate 500 synthetic patient records from each model.

    Step 3: Apply IQR-based filtering to each synthetic dataset to remove
            records whose feature values fall outside the range observed
            in real training patients.

    Step 4: Run consensus voting to identify synthetic records that are
            corroborated by at least two of the three generative methods.

    Step 5: Compute tabular FID scores to quantify how closely each synthetic
            dataset resembles the real training data distribution.

    Step 6: Train Random Forest, Gradient Boosting, and Logistic Regression
            classifiers under three scenarios and evaluate them on the
            held-out real test set.

    Step 7: Run statistical tests (Shapiro-Wilk, Kolmogorov-Smirnov, Jensen-
            Shannon divergence, Cohen's d, chi-square) to formally assess
            distributional differences between real and synthetic data.

    Step 8: Generate ten publication-ready figures at 300 DPI.

    Step 9: Write a plain-text summary report collecting all key results.

How to run
----------
    Full run with paper-quality training (500 epochs for GAN, 300 for CTGAN and TVAE)

        python master_runner.py

    Quick smoke test to verify the pipeline runs without errors (50 epochs)

        python master_runner.py --quick

    Custom number of generated samples

        python master_runner.py --n_syn 300

Output
------
    output/data/      all CSV files for real, synthetic, filtered, and consensus data
    output/figures/   ten PNG figures at 300 DPI
    output/results/   numeric result tables as CSV files and a plain-text report
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

# Allow Python to find the src package when this script is run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import (
    load_complete_data, split_data, fit_scaler,
    save_data, save_scaler, post_process_synthetic,
    ALL_FEATURE_COLS, TARGET_COL, CONTINUOUS_COLS,
)
from src.synthetic_generator import VanillaGAN, CTGAN, TVAE, generate_synthetic
from src.iqr_filter          import filter_all, filter_summary_df
from src.consensus_voting    import run_consensus, consensus_summary_df
from src.fid_calculator      import compute_all_fids
from src.predictive_modeling import run_all_scenarios, pivot_results
from src.statistical_analysis import run_all_statistical_tests
from src.visualizations      import (
    plot_training_losses, plot_fid_comparison, plot_iqr_filtering,
    plot_distribution_comparison, plot_correlation_heatmap,
    plot_consensus_distribution, plot_performance_heatmap,
    plot_model_comparison, plot_all_metrics, plot_pipeline_flowchart,
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUT_DATA    = os.path.join(BASE_DIR, "output", "data")
OUT_RESULTS = os.path.join(BASE_DIR, "output", "results")
RAW_CSV     = os.path.join(BASE_DIR, "..", "data", "raw", "cirrhosis.csv")

os.makedirs(OUT_DATA,    exist_ok=True)
os.makedirs(OUT_RESULTS, exist_ok=True)


def section_header(title):
    """Print a clearly readable section header to the terminal."""
    print(f"\n{'=' * 68}")
    print(f"  {title}")
    print(f"{'=' * 68}")


def elapsed_time(start):
    """Return a human-readable elapsed time string."""
    seconds = time.time() - start
    return f"{seconds:.1f} seconds"


def main(args):
    """
    Orchestrate the full pipeline from data loading through report generation.

    Every step is self-contained and calls a dedicated module from the src
    directory. Results from each step are passed explicitly to subsequent steps
    rather than relying on global state. This makes it straightforward to
    re-run individual steps in isolation during development.
    """
    start_time = time.time()


    section_header("Step 1 of 9: Data Loading and Preprocessing")

    full_df = load_complete_data(RAW_CSV)
    print(f"  Patients available after complete-case filter: {len(full_df)}")
    print(f"  Class distribution in full dataset: {full_df[TARGET_COL].value_counts().to_dict()}")
    print(f"  Label meaning: 0 = survived or transplant, 1 = deceased")

    train_df, test_df = split_data(full_df, test_size=0.30, random_state=42)
    print(f"  Training set: {len(train_df)} patients")
    print(f"  Test set (held out, never used in training): {len(test_df)} patients")

    scaler    = fit_scaler(train_df)
    col_order = ALL_FEATURE_COLS + [TARGET_COL]

    save_data(full_df,  "complete_data.csv")
    save_data(train_df, "train_real.csv")
    save_data(test_df,  "test_real.csv")
    save_scaler(scaler)

    # Normalise the training data to [0, 1] for the generative models
    X_train_scaled = scaler.transform(train_df[col_order])
    target_idx     = col_order.index(TARGET_COL)


    section_header("Step 2 of 9: Synthetic Data Generation")

    print(f"\n  Training Vanilla GAN for {args.epochs_gan} epochs ...")
    gan = VanillaGAN(
        latent_dim=100,
        epochs=args.epochs_gan,
        batch_size=32,
        lr=2e-4,
        print_every=args.print_every
    )
    gan.fit(X_train_scaled, verbose=True)
    synthetic_gan = generate_synthetic(gan, args.n_syn, scaler, col_order,
                                        post_process_synthetic)
    save_data(synthetic_gan, "synthetic_gan.csv")
    print(f"  Vanilla GAN generated {len(synthetic_gan)} synthetic records")

    print(f"\n  Training CTGAN for {args.epochs_ctgan} epochs ...")
    ctgan = CTGAN(
        latent_dim=100,
        epochs=args.epochs_ctgan,
        batch_size=32,
        lr=2e-4,
        n_classes=2,
        target_col_idx=target_idx,
        print_every=args.print_every
    )
    ctgan.fit(X_train_scaled, verbose=True)
    synthetic_ctgan = generate_synthetic(ctgan, args.n_syn, scaler, col_order,
                                          post_process_synthetic)
    save_data(synthetic_ctgan, "synthetic_ctgan.csv")
    print(f"  CTGAN generated {len(synthetic_ctgan)} synthetic records")

    print(f"\n  Training TVAE for {args.epochs_tvae} epochs ...")
    tvae = TVAE(
        latent_dim=32,
        epochs=args.epochs_tvae,
        batch_size=32,
        lr=1e-3,
        beta=1.0,
        print_every=args.print_every
    )
    tvae.fit(X_train_scaled, verbose=True)
    synthetic_tvae = generate_synthetic(tvae, args.n_syn, scaler, col_order,
                                         post_process_synthetic)
    save_data(synthetic_tvae, "synthetic_tvae.csv")
    print(f"  TVAE generated {len(synthetic_tvae)} synthetic records")


    section_header("Step 3 of 9: IQR-Based Outlier Filtering")

    raw_synthetics = {
        "GAN":   synthetic_gan,
        "CTGAN": synthetic_ctgan,
        "TVAE":  synthetic_tvae,
    }
    iqr_result = filter_all(raw_synthetics, train_df)

    filtered_gan   = iqr_result["filtered"]["GAN"]
    filtered_ctgan = iqr_result["filtered"]["CTGAN"]
    filtered_tvae  = iqr_result["filtered"]["TVAE"]

    save_data(filtered_gan,   "filtered_gan.csv")
    save_data(filtered_ctgan, "filtered_ctgan.csv")
    save_data(filtered_tvae,  "filtered_tvae.csv")

    iqr_summary = filter_summary_df(iqr_result["retention"])
    iqr_summary.to_csv(os.path.join(OUT_RESULTS, "iqr_filter_summary.csv"), index=False)
    print("\n  IQR filter summary:")
    print(iqr_summary.to_string(index=False))


    section_header("Step 4 of 9: Consensus Voting")

    # Start with the standard tolerance and increase it if no consensus is found.
    # This can happen when models are under-trained (e.g. in quick mode).
    consensus_df, source_counts = run_consensus(
        filtered_gan, filtered_ctgan, filtered_tvae,
        tolerance=0.5, min_votes=2, verbose=True
    )

    for fallback_tolerance in [1.0, 2.0, 5.0]:
        if len(consensus_df) > 0:
            break
        print(f"  No consensus found at default tolerance. Retrying at tolerance {fallback_tolerance} ...")
        consensus_df, source_counts = run_consensus(
            filtered_gan, filtered_ctgan, filtered_tvae,
            tolerance=fallback_tolerance, min_votes=2, verbose=True
        )

    if len(consensus_df) == 0:
        print("  No consensus found even at tolerance 5.0.")
        print("  This typically means the models were trained for too few epochs.")
        print("  Substituting filtered CTGAN data as a fallback for Scenario C.")
        consensus_df  = filtered_ctgan.copy()
        source_counts = {"CTGAN": len(filtered_ctgan), "GAN": 0, "TVAE": 0}

    save_data(consensus_df, "consensus.csv")
    consensus_summary = consensus_summary_df(source_counts)
    consensus_summary.to_csv(
        os.path.join(OUT_RESULTS, "consensus_summary.csv"), index=False
    )
    print("\n  Consensus summary:")
    print(consensus_summary.to_string(index=False))


    section_header("Step 5 of 9: FID Score Computation")

    fid_df = compute_all_fids(
        train_df, filtered_gan, filtered_ctgan, filtered_tvae, consensus_df
    )
    fid_df.to_csv(os.path.join(OUT_RESULTS, "fid_scores.csv"), index=False)
    print("\n  FID scores (lower is better):")
    print(fid_df.to_string(index=False))


    section_header("Step 6 of 9: Predictive Utility Evaluation")

    performance_df = run_all_scenarios(
        train_df, test_df, filtered_ctgan, consensus_df
    )
    performance_df.to_csv(
        os.path.join(OUT_RESULTS, "model_performance.csv"), index=False
    )
    print("\n  F1 score summary by scenario and classifier:")
    print(pivot_results(performance_df, "F1").to_string())


    section_header("Step 7 of 9: Statistical Analysis")

    synthetic_dict = {
        "GAN (filtered)":   filtered_gan,
        "CTGAN (filtered)": filtered_ctgan,
        "TVAE (filtered)":  filtered_tvae,
        "Consensus":        consensus_df,
    }
    statistical_results = run_all_statistical_tests(train_df, synthetic_dict)

    for test_name, result_df in statistical_results.items():
        if len(result_df) > 0:
            result_df.to_csv(
                os.path.join(OUT_RESULTS, f"stats_{test_name}.csv"), index=False
            )
    print(f"  All statistical test results saved to {OUT_RESULTS}")


    section_header("Step 8 of 9: Generating Publication Figures")

    print("  Generating Figure 0: training loss curves ...")
    plot_training_losses(gan, ctgan, tvae)

    print("  Generating Figure 1: FID comparison ...")
    plot_fid_comparison(fid_df)

    print("  Generating Figure 2: IQR filtering effect ...")
    plot_iqr_filtering(
        train_df, raw_synthetics,
        {"GAN": filtered_gan, "CTGAN": filtered_ctgan, "TVAE": filtered_tvae}
    )

    print("  Generating Figure 3: feature distribution comparison ...")
    plot_distribution_comparison(train_df, {
        "GAN":       filtered_gan,
        "CTGAN":     filtered_ctgan,
        "TVAE":      filtered_tvae,
        "Consensus": consensus_df,
    })

    print("  Generating Figure 4: correlation heatmap ...")
    plot_correlation_heatmap(train_df, consensus_df)

    print("  Generating Figure 5: consensus source distribution ...")
    plot_consensus_distribution(source_counts)

    print("  Generating Figure 6: performance heatmap ...")
    plot_performance_heatmap(performance_df, metric="F1")

    print("  Generating Figure 7: model comparison bar chart ...")
    plot_model_comparison(performance_df)

    print("  Generating Figure 8: all metrics line plot ...")
    plot_all_metrics(performance_df)

    print("  Generating Figure 9: pipeline flowchart ...")
    plot_pipeline_flowchart()

    print(f"\n  All figures saved to {os.path.join(BASE_DIR, 'output', 'figures')}")


    section_header("Step 9 of 9: Writing Summary Report")

    write_summary_report(
        full_df, train_df, test_df, raw_synthetics, iqr_result,
        consensus_df, source_counts, fid_df, performance_df, statistical_results
    )

    section_header(f"Pipeline complete in {elapsed_time(start_time)}")
    print(f"  All outputs are in: {os.path.join(BASE_DIR, 'output')}")
    print(f"  Data files:         output/data/")
    print(f"  Figures (300 DPI):  output/figures/")
    print(f"  Result tables:      output/results/")


def write_summary_report(full_df, train_df, test_df, raw_synthetics,
                          iqr_result, consensus_df, source_counts,
                          fid_df, performance_df, statistical_results):
    """
    Write a plain-text summary report that brings together all key results.

    This file is intended to give a quick overview of what the pipeline
    produced, serving as a convenient starting point when interpreting the
    outputs or writing the paper results section.

    Parameters
    ----------
    All parameters are the results collected during the pipeline steps above.
    """
    lines = [
        "=" * 68,
        "SYNTHETIC DATA PIPELINE  SUMMARY REPORT",
        "Primary Biliary Cirrhosis  Mayo Clinic PBC Dataset",
        "=" * 68,
        "",
        "DATA",
        "",
        f"  Complete patient records used   : {len(full_df)}",
        f"  Training set size               : {len(train_df)}",
        f"  Test set size (held out)        : {len(test_df)}",
        f"  Number of features              : {len(full_df.columns) - 1}",
        f"  Target variable                 : Status (0 = alive, 1 = deceased)",
        f"  Training set class balance      : {train_df[TARGET_COL].value_counts().to_dict()}",
        "",
        "SYNTHETIC GENERATION",
        "",
    ]

    for name, df in raw_synthetics.items():
        filtered = iqr_result["filtered"][name]
        retained = iqr_result["retention"][name]
        lines.append(
            f"  {name:<8}  generated {len(df)} records  "
            f"  after IQR filter {len(filtered)} records kept  "
            f"  ({retained:.1f} percent retained)"
        )

    lines += [
        "",
        "CONSENSUS VOTING",
        "",
        f"  Consensus records accepted : {len(consensus_df)}",
    ]
    for method, count in source_counts.items():
        lines.append(f"    Records from {method:<8}: {count}")

    lines += [
        "",
        "FID SCORES (lower is better)",
        "",
    ]
    for _, row in fid_df.iterrows():
        lines.append(
            f"  {row['Method']:<25}  FID = {row['FID']:.4f}  "
            f"  (n = {row['n_samples']})"
        )

    lines += [
        "",
        "PREDICTIVE UTILITY (evaluated on held-out real test set)",
        "",
    ]
    for _, row in performance_df.iterrows():
        lines.append(
            f"  {row['Scenario'][:38]:<38}  {row['Classifier']:<22}  "
            f"Accuracy {row['Accuracy']:.4f}  F1 {row['F1']:.4f}  "
            f"AUC {row['AUC']}"
        )

    lines += [
        "",
        "STATISTICAL TESTS",
        "",
        f"  Shapiro-Wilk tests run  : {len(statistical_results.get('shapiro_wilk', []))}",
        f"  KS tests run            : {len(statistical_results.get('ks_tests', []))}",
        f"  Chi-square tests run    : {len(statistical_results.get('chi_square', []))}",
        "",
        "OUTPUT LOCATIONS",
        "",
        "  output/data/      all CSV data files",
        "  output/figures/   all publication figures at 300 DPI",
        "  output/results/   numeric result tables and this report",
        "",
        "=" * 68,
    ]

    report_text = "\n".join(lines)
    print(report_text)

    report_path = os.path.join(OUT_RESULTS, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\n  Report written to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PBC Liver Cirrhosis Synthetic Data Pipeline"
    )
    parser.add_argument(
        "--n_syn", type=int, default=500,
        help="Number of synthetic records to generate per method (default 500)"
    )
    parser.add_argument(
        "--epochs_gan", type=int, default=500,
        help="Training epochs for the Vanilla GAN (default 500)"
    )
    parser.add_argument(
        "--epochs_ctgan", type=int, default=300,
        help="Training epochs for CTGAN (default 300)"
    )
    parser.add_argument(
        "--epochs_tvae", type=int, default=300,
        help="Training epochs for TVAE (default 300)"
    )
    parser.add_argument(
        "--print_every", type=int, default=100,
        help="Print training progress every this many epochs (default 100)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick smoke test: 50 epochs per model, 200 samples (for verifying the pipeline)"
    )

    args = parser.parse_args()

    if args.quick:
        args.epochs_gan   = 50
        args.epochs_ctgan = 50
        args.epochs_tvae  = 50
        args.n_syn        = 200
        args.print_every  = 10
        print("Quick mode enabled: 50 training epochs and 200 samples per method")

    main(args)
