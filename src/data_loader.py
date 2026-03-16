"""
Data Loader
===========

Project
-------
Synthetic Data Generation and Predictive Utility Evaluation
for Primary Biliary Cirrhosis (PBC) — Mayo Clinic Dataset

Purpose of this module
-----------------------
This module is responsible for loading the raw cirrhosis CSV file, removing
every patient record that contains even one missing value (complete-case
analysis), encoding all categorical variables into numbers, and splitting
the data into a training set and a held-out test set.

Why complete-case analysis?
    The original dataset has 418 patients. Rows 313 to 418 belong to an
    observational cohort and are missing most clinical measurements.
    Earlier rows also have scattered missing values in laboratory results.
    Rather than imputing unknown values, we keep only the 276 rows where
    every measurement is present. This avoids any imputation-related bias
    entering our synthetic generation or model evaluation.

Target variable
    Status is encoded as binary: patients who were deceased (D) receive the label 1
    and patients who were censored or received a liver transplant (C or CL)
    receive the label 0.

Author note
    All preprocessing decisions in this module are intentional and match the
    methodology described in the paper. Do not change the encoding maps
    without updating the paper as well.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle


# File path constants
# BASE_DIR is the project root (liver_cirrhosis_article/).
# The raw CSV lives inside the project under data/raw/.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "cirrhosis.csv")
OUT_DATA = os.path.join(BASE_DIR, "output", "data")


# Feature type registry
# These lists define how each column is treated during post-processing.
# All values below are AFTER encoding (everything is numeric at that point).

# Continuous laboratory and clinical measurements
CONTINUOUS_COLS = [
    "N_Days", "Age", "Bilirubin", "Cholesterol", "Albumin",
    "Copper", "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin"
]

# Binary clinical indicators encoded as 0 or 1
BINARY_COLS = ["Sex", "Drug", "Ascites", "Hepatomegaly", "Spiders"]

# Ordinal variables with specific integer ranges
# Edema ranges from 0 (none) to 2 (confirmed edema)
# Stage ranges from 1 (least severe) to 4 (most severe)
ORDINAL_COLS = {
    "Edema": (0, 2),
    "Stage": (1, 4),
}

# The outcome variable we train classifiers to predict
TARGET_COL = "Status"

# All feature columns in the order they appear after preprocessing
ALL_FEATURE_COLS = CONTINUOUS_COLS + BINARY_COLS + list(ORDINAL_COLS.keys())


def load_complete_data(raw_path=RAW_PATH):
    """
    Load the raw CSV, keep only complete rows, and encode all columns.

    This function performs three tasks in sequence.

    First, it reads the raw file and drops the patient ID column because
    an arbitrary identifier carries no clinical meaning and should not
    influence any model.

    Second, it removes every row that has a missing value in any column.
    We treat this as a strict requirement because our generative models
    need to learn from clean, fully observed examples.

    Third, it converts every categorical column to an integer so that
    all downstream code works with purely numeric arrays.

    Encoding decisions
        Sex           : Female = 0, Male = 1
        Drug          : Placebo = 0, D-penicillamine = 1
        Ascites       : No = 0, Yes = 1
        Hepatomegaly  : No = 0, Yes = 1
        Spiders       : No = 0, Yes = 1
        Edema         : No edema = 0, Slight or suspected = 1, Confirmed = 2
        Status        : Censored or transplant = 0, Died = 1
        Stage         : Already numeric, kept as 1 through 4

    Parameters
    ----------
    raw_path : str
        Path to cirrhosis.csv

    Returns
    -------
    df : pandas DataFrame with shape (n_complete, 19)
         Columns are ALL_FEATURE_COLS followed by TARGET_COL
    """
    df = pd.read_csv(raw_path, na_values=["NA", "N/A", ""])

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    n_raw = len(df)
    df = df.dropna().copy()
    n_complete = len(df)
    print(f"Data loader: {n_raw} total patients in raw file, {n_complete} patients kept after complete-case filter")

    # Encode the target variable first so we can use it for stratified splitting
    df[TARGET_COL] = df[TARGET_COL].map({"D": 1, "C": 0, "CL": 0}).astype(int)

    # Encode binary clinical features
    df["Sex"]  = df["Sex"].map({"F": 0, "M": 1}).astype(int)
    df["Drug"] = df["Drug"].map({"Placebo": 0, "D-penicillamine": 1}).astype(int)
    df["Ascites"]      = df["Ascites"].map({"N": 0, "Y": 1}).astype(int)
    df["Hepatomegaly"] = df["Hepatomegaly"].map({"N": 0, "Y": 1}).astype(int)
    df["Spiders"]      = df["Spiders"].map({"N": 0, "Y": 1}).astype(int)

    # Encode ordinal features
    df["Edema"] = df["Edema"].map({"N": 0, "S": 1, "Y": 2}).astype(int)
    df["Stage"] = df["Stage"].astype(int)

    # Arrange columns so features come first and the target comes last
    col_order = ALL_FEATURE_COLS + [TARGET_COL]
    df = df[col_order].reset_index(drop=True)

    # Safety check: confirm no missing values remain
    assert df.isnull().sum().sum() == 0, "Unexpected missing values remain after complete-case filter"

    return df


def split_data(df, test_size=0.30, random_state=42):
    """
    Divide the complete dataset into a training set and a held-out test set.

    We use stratified splitting so that the proportion of patients who were deceased
    is the same in both partitions. The test set is treated as completely
    off-limits during all synthetic generation and model training steps.
    It is only used at the very end to evaluate prediction performance.

    Parameters
    ----------
    df           : the output of load_complete_data
    test_size    : fraction to hold out for testing (default 30 percent)
    random_state : seed for reproducibility (fixed at 42 throughout the paper)

    Returns
    -------
    train_df : pandas DataFrame with 70 percent of complete cases
    test_df  : pandas DataFrame with 30 percent of complete cases
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COL]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def fit_scaler(train_df):
    """
    Fit a Min-Max scaler on the training data.

    The scaler maps every column to the range [0, 1] based on the minimum
    and maximum values observed in the training set. We fit on training data
    only to prevent information about the test set from leaking into the
    training process.

    This scaler is later used to normalise the training data before feeding
    it into the generative models, and to inverse-transform generated samples
    back into the original measurement units.

    Parameters
    ----------
    train_df : pandas DataFrame (output of split_data)

    Returns
    -------
    scaler : fitted MinMaxScaler
    """
    scaler = MinMaxScaler()
    scaler.fit(train_df)
    return scaler


def save_data(df, filename, out_dir=OUT_DATA):
    """
    Write a DataFrame to a CSV file in the output data directory.

    Parameters
    ----------
    df       : pandas DataFrame to save
    filename : output filename, for example 'complete_data.csv'
    out_dir  : directory where the file will be written
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    print(f"Saved {filename} to {path}")


def save_scaler(scaler, filename="scaler.pkl", out_dir=OUT_DATA):
    """Serialise the fitted scaler to disk so it can be reloaded later."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {path}")


def load_scaler(filename="scaler.pkl", out_dir=OUT_DATA):
    """Load a previously saved scaler from disk."""
    path = os.path.join(out_dir, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


def post_process_synthetic(df_scaled, scaler, col_order):
    """
    Convert scaled synthetic samples back into the original measurement units.

    After a generative model produces values in the [0, 1] range, this
    function performs two steps. First it applies the inverse of the Min-Max
    scaler to restore the natural scale of every feature. Second it rounds
    binary and ordinal columns to their valid integer values, because a
    generated value of 0.73 for a binary feature should become 1, and a
    generated value of 3.2 for Stage should become 3.

    Continuous features are clipped at zero because no clinical measurement
    in this dataset can be negative.

    Parameters
    ----------
    df_scaled  : pandas DataFrame of scaled values in [0, 1]
    scaler     : the fitted MinMaxScaler from fit_scaler
    col_order  : list of column names in the same order as the scaler expects

    Returns
    -------
    df_out : pandas DataFrame with values in original measurement units
    """
    arr   = scaler.inverse_transform(df_scaled.values)
    df_out = pd.DataFrame(arr, columns=col_order)

    # Round binary columns to the nearest integer and clip to 0 or 1
    for col in BINARY_COLS + [TARGET_COL]:
        if col in df_out.columns:
            df_out[col] = df_out[col].round().clip(0, 1).astype(int)

    # Round ordinal columns and clip to their valid range
    for col, (lo, hi) in ORDINAL_COLS.items():
        if col in df_out.columns:
            df_out[col] = df_out[col].round().clip(lo, hi).astype(int)

    # Ensure continuous measurements are non-negative
    for col in CONTINUOUS_COLS:
        if col in df_out.columns:
            df_out[col] = df_out[col].clip(lower=0)

    return df_out
