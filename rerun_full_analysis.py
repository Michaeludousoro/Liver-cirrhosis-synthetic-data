"""
Full Corrected-Pipeline Rerun
===============================

Purpose
-------
Regenerates every predictive-utility result in the paper under the leak-
free feature set (N_Days excluded from classifier input), plus the two new
framings the reviewer asked for (fixed-horizon landmark classification and
full time-to-event survival modelling) and the six-generator expansion of
Scenario E (previously cGAN-only).

What is NOT retrained here
    Vanilla GAN, cGAN, VAE, the masked-loss VAE, and consensus voting are
    unaffected by the leakage fix (they still generate N_Days as a
    feature; the fix is entirely downstream, in which columns a
    classifier is allowed to read). Their existing output/data/*.csv
    files were verified to be byte-identical to a fresh seed=42 run
    (train_real.csv/test_real.csv equality-checked directly), so they are
    reused rather than retrained. Only CTGANProper needed fresh training,
    since it was previously orphaned (see run_ctgan_proper.py).

Outputs (all written to output/results/)
    model_performance.csv, smote_results.csv, cv_results.csv,
    bootstrap_ci.csv, mcnemar_tests.csv
        Overwritten in place with the leak-fixed (N_Days-excluded)
        numbers. These are the corrected version of the original Table
        III/IV/V/VI/VII, not a new framing.
    landmark_model_performance.csv, landmark_smote_results.csv
        Scenarios A-D under the 3-year landmark framing.
    scenario_e_six_generators.csv
        Scenario E (train-on-synthetic, test-on-real) for all six
        generators, both classification framings, merged with each
        generator's FID.
    survival_scenarios.csv
        Scenarios A-E under the Cox/RSF time-to-event framing.
    km_logrank_six_generators.csv
        Kaplan-Meier / log-rank comparison of each generator's synthetic
        (N_Days, Status) distribution against the real training cohort.
"""

import os
import numpy as np
import pandas as pd

from src.data_loader import (load_complete_data, split_data,
                             CLASSIFICATION_FEATURE_COLS, TARGET_COL,
                             landmark_label)
from src.predictive_modeling import (run_all_scenarios, run_smote_scenario,
                                     run_landmark_scenarios, run_scenario_e_multi,
                                     apply_landmark_frame)
from src.statistical_analysis import (cross_validate_scenarios,
                                      bootstrap_ci_scenarios, mcnemar_scenarios)
from src.survival_analysis import run_survival_scenarios, km_logrank_comparison
from src.fid_calculator import compute_fid

SEED = 42
HORIZON_DAYS = 3 * 365.25
DATA_DIR = "output/data"
RESULTS_DIR = "output/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_pools():
    train_real = pd.read_csv(f"{DATA_DIR}/train_real.csv")
    test_real = pd.read_csv(f"{DATA_DIR}/test_real.csv")

    pools = {
        "Vanilla GAN": pd.read_csv(f"{DATA_DIR}/filtered_gan.csv"),
        "cGAN": pd.read_csv(f"{DATA_DIR}/filtered_ctgan.csv"),
        "VAE": pd.read_csv(f"{DATA_DIR}/filtered_tvae.csv"),
        "Masked-loss VAE": pd.read_csv(f"{DATA_DIR}/filtered_masked_vae.csv"),
        "Consensus": pd.read_csv(f"{DATA_DIR}/consensus_equalised.csv"),
    }

    ctgan_proper_path = f"{DATA_DIR}/filtered_ctgan_proper.csv"
    if os.path.exists(ctgan_proper_path):
        pools["CTGAN"] = pd.read_csv(ctgan_proper_path)
    else:
        print("  WARNING: filtered_ctgan_proper.csv not found yet — "
              "CTGAN excluded from the six-generator comparison this run.")

    return train_real, test_real, pools


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    np.random.seed(SEED)
    train_real, test_real, pools = load_pools()
    filtered_ctgan_cgan = pools["cGAN"]  # Scenario B/E's original single-generator choice
    consensus_df = pools["Consensus"]

    # -----------------------------------------------------------------
    section("1. Leak-fixed classification (Scenarios A-E), N_Days excluded")
    # -----------------------------------------------------------------
    perf_df = run_all_scenarios(train_real, test_real, filtered_ctgan_cgan, consensus_df)
    smote_df = run_smote_scenario(train_real, test_real)
    perf_df.to_csv(f"{RESULTS_DIR}/model_performance.csv", index=False)
    smote_df.to_csv(f"{RESULTS_DIR}/smote_results.csv", index=False)
    print(perf_df[["Scenario", "Classifier", "n_train", "AUC"]].to_string(index=False))
    print(smote_df[["Scenario", "Classifier", "n_train", "AUC"]].to_string(index=False))

    # -----------------------------------------------------------------
    section("2. Cross-validation, bootstrap CI, McNemar (leak-fixed)")
    # -----------------------------------------------------------------
    cv_df = cross_validate_scenarios(train_real, {
        "A: Baseline (real data only)": None,
        "B: Real data plus filtered cGAN": filtered_ctgan_cgan,
        "C: Real data plus consensus synthetic": consensus_df,
        "D: Real data plus SMOTE": "smote",
    })
    cv_df.to_csv(f"{RESULTS_DIR}/cv_results.csv", index=False)
    print(cv_df.to_string(index=False))

    def combine(*dfs):
        cols = CLASSIFICATION_FEATURE_COLS + [TARGET_COL]
        return pd.concat(list(dfs), ignore_index=True).dropna(subset=cols)

    scenario_train_dfs = {
        "A: Baseline": train_real,
        "B: Real + cGAN": combine(train_real, filtered_ctgan_cgan),
        "C: Real + Consensus": combine(train_real, consensus_df),
    }
    boot_df = bootstrap_ci_scenarios(scenario_train_dfs, test_real, n_boot=1000)
    boot_df.to_csv(f"{RESULTS_DIR}/bootstrap_ci.csv", index=False)
    print(boot_df.to_string(index=False))

    mc_df = mcnemar_scenarios(scenario_train_dfs, test_real, baseline_name="A: Baseline")
    mc_df.to_csv(f"{RESULTS_DIR}/mcnemar_tests.csv", index=False)
    print(mc_df.to_string(index=False))

    # -----------------------------------------------------------------
    section(f"3. Landmark classification (horizon = 3 years = {HORIZON_DAYS:.0f} days)")
    # -----------------------------------------------------------------
    lm_perf_df = run_landmark_scenarios(train_real, test_real, filtered_ctgan_cgan,
                                        consensus_df, horizon_days=HORIZON_DAYS)
    lm_perf_df.to_csv(f"{RESULTS_DIR}/landmark_model_performance.csv", index=False)
    print(lm_perf_df[["Scenario", "Classifier", "n_train", "AUC"]].to_string(index=False))

    train_lm = apply_landmark_frame(train_real, HORIZON_DAYS)
    test_lm = apply_landmark_frame(test_real, HORIZON_DAYS)
    lm_smote_df = run_smote_scenario(train_lm, test_lm, target_col="landmark_event")
    lm_smote_df.to_csv(f"{RESULTS_DIR}/landmark_smote_results.csv", index=False)
    print(lm_smote_df[["Scenario", "Classifier", "n_train", "AUC"]].to_string(index=False))

    print(f"\n  Landmark cohort: train usable {len(train_lm)}/{len(train_real)}, "
          f"test usable {len(test_lm)}/{len(test_real)}, "
          f"event rate train={train_lm['landmark_event'].mean():.3f} "
          f"test={test_lm['landmark_event'].mean():.3f}")

    # -----------------------------------------------------------------
    section("4. Scenario E across six generators (classification + landmark)")
    # -----------------------------------------------------------------
    e_multi_df = run_scenario_e_multi(train_real, test_real, pools)
    e_multi_df["Framing"] = "Classification (Status, N_Days excluded)"

    landmark_pools = {name: apply_landmark_frame(df, HORIZON_DAYS) for name, df in pools.items()}
    e_multi_lm_df = run_scenario_e_multi(train_lm, test_lm, landmark_pools,
                                          target_col="landmark_event")
    e_multi_lm_df["Framing"] = "Landmark (3-year horizon)"

    fid_rows = []
    for name, df in pools.items():
        fid = compute_fid(train_real, df)
        fid_rows.append({"Generator": name, "FID": round(fid, 4)})
    fid_df = pd.DataFrame(fid_rows)
    print(fid_df.to_string(index=False))

    combined_e = pd.concat([e_multi_df, e_multi_lm_df], ignore_index=True)
    combined_e = combined_e.merge(fid_df, on="Generator", how="left")
    combined_e.to_csv(f"{RESULTS_DIR}/scenario_e_six_generators.csv", index=False)
    fid_df.to_csv(f"{RESULTS_DIR}/fid_scores_six_generators.csv", index=False)
    print(combined_e[["Generator", "Framing", "Classifier", "AUC", "FID"]].to_string(index=False))

    # -----------------------------------------------------------------
    section("5. Survival modelling (CoxPH / RSF), Scenarios A-E")
    # -----------------------------------------------------------------
    surv_df = run_survival_scenarios(train_real, test_real, filtered_ctgan_cgan, consensus_df)
    surv_df.to_csv(f"{RESULTS_DIR}/survival_scenarios.csv", index=False)
    print(surv_df.to_string(index=False))

    # -----------------------------------------------------------------
    section("6. Kaplan-Meier / log-rank comparison across six generators")
    # -----------------------------------------------------------------
    km_df, _fitters = km_logrank_comparison(train_real, pools)
    km_df = km_df.merge(fid_df, on="Generator", how="left")
    km_df.to_csv(f"{RESULTS_DIR}/km_logrank_six_generators.csv", index=False)
    print(km_df.to_string(index=False))

    print("\nDone. All outputs written to output/results/.")


if __name__ == "__main__":
    main()
