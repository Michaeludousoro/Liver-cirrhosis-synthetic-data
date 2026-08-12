"""
Consensus Voting
================

Project
-------
Synthetic Data Generation and Predictive Utility Evaluation
for Primary Biliary Cirrhosis (PBC) — Mayo Clinic Dataset

Purpose of this module
-----------------------
After generating and filtering synthetic data from three different generative
models, we want to identify the subset of synthetic records that multiple
models agree are representative. A record that appears in the realistic region
of GAN output, cGAN output, and VAE output simultaneously is more likely to
reflect a genuine pattern in the data than a record that only one model produces.

This is the concept behind consensus voting: we retain only those synthetic
records that are corroborated by at least two of the three generative methods.

How the algorithm works
    We process each method's filtered output in turn, calling it the "candidate
    method" for that round. For each candidate record we measure how close it is
    to the nearest record from each of the other two methods. We work in
    standardised feature space so that differences in measurement scale do not
    give undue weight to features with large numerical values.

    Specifically:
        1. Combine all three filtered datasets and fit a standard scaler on
           the union so every feature has mean zero and unit variance.
        2. For a candidate record from method A, compute the minimum Euclidean
           distance to any record from method B, and separately the minimum
           Euclidean distance to any record from method C.
        3. If the distance to method B is less than the tolerance threshold,
           method B casts a vote for this record. Similarly for method C.
        4. The candidate record is accepted into the consensus set if it
           receives at least min_votes votes (default 2, meaning both B and C
           must endorse it).

    This is repeated for all records from all three methods acting as the
    candidate method in turn, so every filtered synthetic record has the
    opportunity to be included.

    After collecting consensus records, we remove any exact duplicates that
    arose because the same region of feature space was covered by multiple
    methods.

Expected output size
    With properly trained models and the default tolerance of 0.5, we typically
    obtain 50 to 200 consensus records. A very small consensus set suggests the
    models have not converged to the same region of feature space, while a
    very large one (close to the full filtered output) suggests the tolerance
    is too permissive.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def run_consensus(filtered_gan, filtered_ctgan, filtered_tvae,
                  tolerance=0.5, min_votes=2, verbose=True):
    """
    Build a consensus set of synthetic records supported by multiple models.

    Parameters
    ----------
    filtered_gan    : IQR-filtered synthetic records from the Vanilla GAN
    filtered_ctgan  : IQR-filtered synthetic records from cGAN
    filtered_tvae   : IQR-filtered synthetic records from VAE
    tolerance       : maximum Euclidean distance in standardised space for a
                      neighbouring record to cast a vote
    min_votes       : number of corroborating methods required (default 2)
    verbose         : if True, print progress and summary statistics

    Returns
    -------
    consensus_df   : pandas DataFrame of consensus synthetic records
    source_counts  : dictionary reporting how many consensus records came
                     from each method (before deduplication)
    """
    methods = {
        "GAN":   filtered_gan.values.astype(float),
        "cGAN": filtered_ctgan.values.astype(float),
        "VAE":  filtered_tvae.values.astype(float),
    }
    col_names    = filtered_gan.columns.tolist()
    method_names = list(methods.keys())

    # Fit a standard scaler on the union of all three filtered datasets.
    # Standardising means features with larger numerical ranges (like
    # Cholesterol in the hundreds) do not dominate the distance calculation
    # over features with smaller ranges (like Albumin near 3 to 4).
    all_data = np.vstack(list(methods.values()))
    scaler   = StandardScaler()
    scaler.fit(all_data)

    scaled_methods = {name: scaler.transform(arr) for name, arr in methods.items()}

    accepted_rows  = []
    source_labels  = []

    for i, candidate_name in enumerate(method_names):
        other_names = [method_names[j] for j in range(3) if j != i]
        candidate   = scaled_methods[candidate_name]

        if verbose:
            print(f"  Consensus: evaluating {len(candidate)} records from "
                  f"{candidate_name} against {other_names}")

        for row in candidate:
            votes = 0
            for other_name in other_names:
                other_data  = scaled_methods[other_name]
                distances   = np.linalg.norm(other_data - row, axis=1)
                if distances.min() < tolerance:
                    votes += 1

            if votes >= min_votes:
                accepted_rows.append(row)
                source_labels.append(candidate_name)

    if len(accepted_rows) == 0:
        if verbose:
            print(f"  Consensus: no records passed at tolerance {tolerance}.")
        return pd.DataFrame(columns=col_names), {}

    # Inverse-transform back to original feature units
    accepted_arr  = np.array(accepted_rows)
    original_arr  = scaler.inverse_transform(accepted_arr)
    consensus_df  = pd.DataFrame(original_arr, columns=col_names)

    # Remove exact duplicate rows that occur when multiple methods occupy
    # the same region of feature space
    consensus_df = consensus_df.drop_duplicates().reset_index(drop=True)

    source_counts = {name: source_labels.count(name) for name in method_names}

    if verbose:
        total = len(consensus_df)
        counts_str = "  ".join(f"{k} contributed {v}" for k, v in source_counts.items())
        print(f"  Consensus: {total} records accepted.  {counts_str}")

    return consensus_df, source_counts


def consensus_summary_df(source_counts):
    """
    Build a readable summary table of how many records each method contributed.

    Parameters
    ----------
    source_counts : dictionary from run_consensus

    Returns
    -------
    pandas DataFrame with columns Source, Count, and Percentage
    """
    if not source_counts:
        return pd.DataFrame(columns=["Source", "Count", "Percentage"])

    rows  = [{"Source": name, "Count": count}
             for name, count in source_counts.items()]
    df    = pd.DataFrame(rows)
    total = df["Count"].sum()
    df["Percentage"] = (df["Count"] / max(total, 1) * 100).round(1)
    return df
