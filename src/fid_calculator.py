"""
Fréchet Inception Distance Calculator
======================================

Project
-------
Synthetic Data Generation and Predictive Utility Evaluation
for Primary Biliary Cirrhosis (PBC) — Mayo Clinic Dataset

Purpose of this module
-----------------------
We need a principled way to measure how similar the statistical distribution
of synthetic patient data is to the real patient data. The Fréchet Inception
Distance (FID) is the standard metric for this purpose. Originally developed
for image generation, it has been adapted here for tabular clinical data.

A lower FID score means the synthetic distribution is closer to the real one.
A score of zero would mean the two distributions are identical. In practice
scores well below 1.0 indicate good fidelity for this dataset.

How FID is computed
    FID compares the two distributions by looking at their multivariate
    Gaussian approximations. For each distribution we compute the mean vector
    and the covariance matrix of the feature values. The FID is then:

        FID = squared norm of (mean_real minus mean_synthetic)
            + trace of (covariance_real + covariance_synthetic
                        minus 2 times the square root of the product
                        of the two covariance matrices)

    The second term captures differences in the spread and correlations of
    the features, not just differences in their central tendency.

    We compute FID on the 11 continuous clinical features (bilirubin, albumin,
    etc.) after normalising both datasets with a Min-Max scaler fitted on the
    real data. Normalisation ensures that features measured in very different
    units contribute equally to the distance.

Numerical stability
    The matrix square root of the covariance product is computed using scipy's
    implementation of the Schur decomposition. Small regularisation terms are
    added to the diagonal of each covariance matrix to ensure they are
    positive semi-definite, which is necessary for the matrix square root
    to be well defined.
"""

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from sklearn.preprocessing import MinMaxScaler

from .data_loader import CONTINUOUS_COLS


def compute_fid(real_df, synthetic_df, numeric_cols=None):
    """
    Compute the Fréchet Inception Distance between real and synthetic data.

    Parameters
    ----------
    real_df      : pandas DataFrame of real patient records
    synthetic_df : pandas DataFrame of synthetic patient records
    numeric_cols : list of continuous feature columns to use for the comparison
                   if None, all 11 standard continuous clinical features are used

    Returns
    -------
    fid_score : float (lower is better; 0 means the distributions are identical)
                returns NaN if either dataset has fewer than 2 rows
    """
    if numeric_cols is None:
        numeric_cols = CONTINUOUS_COLS

    # Keep only the columns that are present in both DataFrames
    cols = [c for c in numeric_cols
            if c in real_df.columns and c in synthetic_df.columns]

    real_arr  = real_df[cols].values.astype(float)
    synth_arr = synthetic_df[cols].values.astype(float)

    if len(real_arr) < 2 or len(synth_arr) < 2:
        return np.nan

    # Normalise both datasets using the real data statistics.
    # We fit only on the real data because the scaler represents what is
    # realistic in the patient population.
    scaler  = MinMaxScaler()
    real_s  = scaler.fit_transform(real_arr)
    synth_s = scaler.transform(synth_arr)

    mean_real  = real_s.mean(axis=0)
    mean_synth = synth_s.mean(axis=0)

    # Add a small value to the diagonal of each covariance matrix to ensure
    # positive semi-definiteness and numerical stability
    eps      = 1e-6
    n_cols   = len(cols)
    cov_real  = np.cov(real_s,  rowvar=False) + eps * np.eye(n_cols)
    cov_synth = np.cov(synth_s, rowvar=False) + eps * np.eye(n_cols)

    # Squared Euclidean distance between the two mean vectors
    mean_diff_sq = np.sum((mean_real - mean_synth) ** 2)

    # Matrix square root of the product of the two covariance matrices
    cov_product   = cov_real @ cov_synth
    sqrt_cov, _   = sqrtm(cov_product, disp=False)

    # Discard the imaginary part if it is negligibly small (a numerical artefact)
    if np.iscomplexobj(sqrt_cov):
        sqrt_cov = sqrt_cov.real

    fid = mean_diff_sq + np.trace(cov_real + cov_synth - 2 * sqrt_cov)
    return float(fid)


def compute_all_fids(real_df, filtered_gan, filtered_ctgan,
                     filtered_tvae, consensus_df, numeric_cols=None):
    """
    Compute the FID score for each synthetic dataset and return a summary table.

    Parameters
    ----------
    real_df        : real training patient records
    filtered_gan   : IQR-filtered GAN synthetic records
    filtered_ctgan : IQR-filtered cGAN synthetic records
    filtered_tvae  : IQR-filtered VAE synthetic records
    consensus_df   : consensus synthetic records
    numeric_cols   : continuous features to use (default: all 11 standard ones)

    Returns
    -------
    pandas DataFrame with columns Method, FID, and n_samples
    """
    datasets = {
        "GAN (filtered)":   filtered_gan,
        "cGAN (filtered)": filtered_ctgan,
        "VAE (filtered)":  filtered_tvae,
        "Consensus":        consensus_df,
    }

    rows = []
    for name, df in datasets.items():
        score = compute_fid(real_df, df, numeric_cols)
        rows.append({
            "Method":   name,
            "FID":      round(score, 4),
            "n_samples": len(df)
        })
        print(f"  FID for {name:<25}: {score:.4f}  (n = {len(df)})")

    return pd.DataFrame(rows)
