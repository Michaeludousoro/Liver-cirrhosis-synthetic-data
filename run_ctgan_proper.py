"""
CTGANProper Driver
===================

Project
-------
Synthetic Data Generation and Predictive Utility Evaluation
for Primary Biliary Cirrhosis (PBC) — Mayo Clinic Dataset

Purpose
-------
src/ctgan_proper.py implements the full CTGAN algorithm of Xu et al.
(mode-specific normalisation, training-by-sampling, PacGAN critic with the
WGAN gradient penalty), but nothing in the tracked pipeline calls it — it
was an orphaned module. This script is that missing driver: it trains
CTGANProper on the real training cohort, generates 500 synthetic records,
applies the same IQR plausibility filter used for every other generator,
and computes its FID, all under the fixed seed=42 regime the rest of the
pipeline uses. It exists so this generator can take its place as one of
the six generators in the Scenario E (train-on-synthetic, test-on-real)
comparison, alongside Vanilla GAN, cGAN, VAE, the masked-loss VAE, and
consensus.

Epoch budget
    CTGANProper trains on a table of 193 rows with a fixed batch size,
    so each "epoch" here is a single gradient step (steps = max(n //
    batch_size, 1) = 1). Prior exploration on this cohort found FID had
    not converged even at 18,000 such steps (FID 0.427 at 4,800 steps,
    0.134 at 12,000, 0.109 at 18,000). This run uses 4,000 epochs (close
    to the first of those three checkpoints), chosen for a reliably
    bounded wall-clock time on this machine rather than to reach
    convergence. The FID actually achieved is reported plainly rather
    than assumed to match the fully-converged figure.
"""

import time
import numpy as np
import pandas as pd

from src.data_loader import (load_complete_data, split_data,
                             ALL_FEATURE_COLS, TARGET_COL,
                             BINARY_COLS, ORDINAL_COLS, OUT_DATA)
from src.ctgan_proper import CTGANProper
from src.iqr_filter import compute_iqr_bounds, apply_iqr_filter
from src.fid_calculator import compute_fid

SEED = 42
EPOCHS = 4000
N_GENERATE = 500

COL_ORDER = ALL_FEATURE_COLS + [TARGET_COL]
DISCRETE_COLS = [COL_ORDER.index(c) for c in
                 BINARY_COLS + list(ORDINAL_COLS.keys()) + [TARGET_COL]]


def main():
    np.random.seed(SEED)

    df = load_complete_data()
    train_real, test_real = split_data(df, random_state=SEED)

    X_raw = train_real[COL_ORDER].values.astype("float64")

    print(f"Training CTGANProper for {EPOCHS} epochs "
          f"(1 gradient step/epoch on n={len(train_real)} rows)...")
    t0 = time.time()
    model = CTGANProper(epochs=EPOCHS, seed=SEED, print_every=1000)
    model.fit(X_raw, DISCRETE_COLS, verbose=True)
    print(f"Training took {time.time() - t0:.1f}s")

    synthetic = model.generate(N_GENERATE)
    synth_df = pd.DataFrame(synthetic, columns=COL_ORDER)

    # Match the post-processing every other generator receives: round
    # binary/ordinal columns to valid integers, clip continuous columns
    # at zero, clip Status to {0,1}.
    for col in BINARY_COLS + [TARGET_COL]:
        synth_df[col] = synth_df[col].round().clip(0, 1).astype(int)
    for col, (lo, hi) in ORDINAL_COLS.items():
        synth_df[col] = synth_df[col].round().clip(lo, hi).astype(int)
    for col in ALL_FEATURE_COLS:
        if col not in BINARY_COLS and col not in ORDINAL_COLS:
            synth_df[col] = synth_df[col].clip(lower=0)

    synth_df.to_csv(f"{OUT_DATA}/synthetic_ctgan_proper.csv", index=False)
    print(f"Saved {len(synth_df)} raw CTGANProper records")

    bounds = compute_iqr_bounds(train_real)
    filtered_df, retention = apply_iqr_filter(synth_df, bounds)
    filtered_df.to_csv(f"{OUT_DATA}/filtered_ctgan_proper.csv", index=False)
    print(f"IQR filter retained {len(filtered_df)}/{len(synth_df)} "
          f"({retention:.1f}%)")

    fid = compute_fid(train_real, filtered_df)
    print(f"CTGANProper FID (filtered, n={len(filtered_df)}): {fid:.4f}")

    summary = pd.DataFrame([{
        "Method": "CTGANProper (filtered)",
        "FID": round(fid, 4),
        "n_samples": len(filtered_df),
        "epochs": EPOCHS,
        "retention_pct": round(retention, 1),
    }])
    summary_path = "output/results/ctgan_proper_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
