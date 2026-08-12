"""
IQR-Based Outlier Filter
========================

Project
-------
Synthetic Data Generation and Predictive Utility Evaluation
for Primary Biliary Cirrhosis (PBC) — Mayo Clinic Dataset

Purpose of this module
-----------------------
Generative models sometimes produce synthetic patient records whose feature
values fall far outside the range that appears in real clinical data. A
generated bilirubin value of 80 mg/dL, for example, would be physiologically
implausible in this patient population. These extreme outliers can harm
downstream classifiers by pulling decision boundaries away from where real
patients actually cluster.

This module implements an IQR-based filter that removes any synthetic record
containing at least one feature value that would be considered a statistical
outlier relative to the real training data.

How the filter works
    For each continuous feature we compute the first quartile (Q1) and third
    quartile (Q3) from the real training patients. The interquartile range
    (IQR) is Q3 minus Q1. We define the acceptable bounds as:

        lower bound = Q1 minus 1.5 times IQR
        upper bound = Q3 plus 1.5 times IQR

    This is the same rule used in standard box-plot analysis and is well
    established in statistics for identifying outliers. A synthetic record
    is kept only if every one of its continuous feature values falls within
    its respective bounds. A single out-of-range value disqualifies the
    entire record.

Why only continuous features?
    Binary and ordinal features (such as Sex, Ascites, or Stage) have already
    been snapped to their valid integer ranges by the post-processing step
    in data_loader, so there is nothing further to filter there.

Expected retention rates
    For a well-trained generative model we expect to retain roughly 50 to 80
    percent of generated records. Lower retention (such as 20 percent for an
    under-trained GAN) indicates the model has not yet learned the realistic
    range of clinical values and would benefit from more training epochs.
"""

import pandas as pd
import numpy as np

from .data_loader import CONTINUOUS_COLS


def compute_iqr_bounds(real_df, cols=None):
    """
    Compute the IQR-based acceptable range for each continuous feature.

    The bounds are computed from the real training data only. We never
    include synthetic data in this calculation because the bounds represent
    what is realistic in the actual patient population.

    Parameters
    ----------
    real_df : pandas DataFrame containing the real training patients
    cols    : list of column names to compute bounds for
              if None, defaults to all 11 continuous clinical features

    Returns
    -------
    bounds : dictionary mapping each column name to a (lower, upper) tuple
    """
    if cols is None:
        cols = CONTINUOUS_COLS

    bounds = {}
    for col in cols:
        if col not in real_df.columns:
            continue
        q1  = real_df[col].quantile(0.25)
        q3  = real_df[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    return bounds


def apply_iqr_filter(synthetic_df, bounds):
    """
    Remove any synthetic record that contains an out-of-range feature value.

    We build a boolean mask that starts as True for every row. For each
    feature in the bounds dictionary we set the mask to False for any row
    where that feature falls outside its acceptable range. The final mask
    selects only rows that passed every check.

    Parameters
    ----------
    synthetic_df : pandas DataFrame of synthetic patient records
    bounds       : dictionary from compute_iqr_bounds

    Returns
    -------
    filtered_df    : pandas DataFrame containing only the records that passed
    retention_pct  : float representing the percentage of records retained
    """
    mask = pd.Series(True, index=synthetic_df.index)

    for col, (lower, upper) in bounds.items():
        if col in synthetic_df.columns:
            mask = mask & (synthetic_df[col] >= lower) & (synthetic_df[col] <= upper)

    filtered_df   = synthetic_df[mask].reset_index(drop=True)
    retention_pct = 100.0 * len(filtered_df) / max(len(synthetic_df), 1)

    return filtered_df, retention_pct


def filter_all(synthetic_dict, real_df):
    """
    Apply the IQR filter to all three synthetic datasets in one call.

    This function computes the IQR bounds once from the real data and then
    applies the same bounds to each of the three synthetic datasets. Reporting
    the retention percentage for each method gives a useful signal about how
    well each generative model has learned the realistic range of clinical values.

    Parameters
    ----------
    synthetic_dict : dictionary with keys 'GAN', 'cGAN', and 'TVAE',
                     each mapping to a pandas DataFrame of synthetic records
    real_df        : the real training DataFrame used to compute the bounds

    Returns
    -------
    result : dictionary with three entries:
        'filtered'   : dictionary of filtered DataFrames (same keys as input)
        'retention'  : dictionary of retention percentages (same keys)
        'bounds'     : the computed IQR bounds dictionary
    """
    bounds    = compute_iqr_bounds(real_df)
    filtered  = {}
    retention = {}

    for name, df in synthetic_dict.items():
        filtered_df, pct = apply_iqr_filter(df, bounds)
        filtered[name]   = filtered_df
        retention[name]  = pct
        print(f"  IQR filter {name:>6}: started with {len(df)} records, "
              f"kept {len(filtered_df)} ({pct:.1f}% retained)")

    return {"filtered": filtered, "retention": retention, "bounds": bounds}


def filter_summary_df(retention):
    """
    Build a simple summary table of IQR filter retention rates.

    Parameters
    ----------
    retention : dictionary mapping method names to retention percentages

    Returns
    -------
    pandas DataFrame with columns Method and Retention_Percent
    """
    rows = [{"Method": name.upper(), "Retention_Percent": round(pct, 2)}
            for name, pct in retention.items()]
    return pd.DataFrame(rows)
