"""
Predictive Utility Evaluation
==============================

Project
-------
Synthetic Data Generation and Predictive Utility Evaluation
for Primary Biliary Cirrhosis (PBC) — Mayo Clinic Dataset

Purpose of this module
-----------------------
A synthetic dataset is only truly useful if models trained with it perform
at least as well as models trained on real data alone. Distributional fidelity
metrics like FID tell us whether the synthetic data looks statistically similar
to the real data, but they do not tell us whether the synthetic data is useful
for the task we actually care about, which is predicting patient outcomes.

This module evaluates predictive utility by training classifiers under the
real-data and synthetic-only scenarios (A, B, C, E) and measuring their
performance on a held-out real test set. The SMOTE scenario (D) is computed
separately in notebook 03.

Training scenarios
    Scenario A: Baseline
        The classifier is trained only on the real training patients.
        This is the reference point against which we compare augmented training.

    Scenario B: Real plus filtered CTGAN
        The real training patients are combined with the IQR-filtered CTGAN
        synthetic records before training. We use CTGAN specifically here
        because it achieved the best FID score among the three individual methods.

    Scenario C: Real plus consensus synthetic
        The real training patients are combined with the consensus synthetic
        records. The consensus set is expected to be the highest quality subset
        because it was endorsed by all three generative models.

    Scenario D: Real plus SMOTE (classical baseline)
        Computed separately in notebook 03: the real training patients are
        augmented with SMOTE oversampling. This provides a classical
        interpolation baseline against the deep generative scenarios.

    Scenario E: Synthetic data only (CTGAN filtered)
        The classifier is trained exclusively on IQR-filtered CTGAN synthetic
        records with no real training patients included. This scenario acts as
        the synthetic-only baseline and answers the question: how well does a
        model generalise to real patients when it has never seen any real
        training data? It is the direct counterpart to Scenario A.

Test set rule
    The test set is always and only composed of real patients. It is never
    contaminated with synthetic data. This is essential for a fair comparison:
    we want to know how well the model generalises to real future patients, not
    how well it performs on synthetic ones.

Classifiers evaluated
    Random Forest: an ensemble of decision trees that is robust to feature
    scale and naturally handles mixed feature types.

    Gradient Boosting: a sequential ensemble that often achieves the highest
    accuracy on tabular data by correcting the errors of previous trees.

    Logistic Regression: a linear model that provides an interpretable baseline
    and is useful for assessing whether the prediction problem is linearly
    separable.

Performance metrics
    Accuracy: proportion of test patients correctly classified.
    F1 Score: harmonic mean of precision and recall, weighted by class support.
    Precision: of the patients predicted to be deceased, how many actually were.
    Recall: of the patients who were actually deceased, how many were correctly predicted.
    AUC: area under the ROC curve, measuring discrimination across all thresholds.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble       import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model   import LogisticRegression
from sklearn.metrics        import (accuracy_score, f1_score,
                                    precision_score, recall_score,
                                    roc_auc_score)
from sklearn.preprocessing  import StandardScaler

from .data_loader import TARGET_COL, ALL_FEATURE_COLS


def _build_classifiers():
    """
    Instantiate the three classifiers with fixed hyperparameters.

    The random state is fixed at 42 throughout to ensure all results are
    reproducible. The hyperparameters chosen here are standard configurations
    known to work well on small to medium tabular datasets without tuning.
    """
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, solver="lbfgs"
        ),
    }


def _evaluate_one_classifier(clf, X_train, y_train, X_test, y_test):
    """
    Train a single classifier and evaluate it on the test set.

    Parameters
    ----------
    clf             : a scikit-learn classifier instance
    X_train, y_train: training features and labels
    X_test, y_test  : test features and labels (always real patients)

    Returns
    -------
    metrics : dictionary with Accuracy, F1, Precision, Recall, and AUC
    """
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    try:
        y_prob = clf.predict_proba(X_test)
        if y_prob.shape[1] == 2:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = np.nan

    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "F1":        round(f1_score(y_test, y_pred, average="weighted",
                                    zero_division=0), 4),
        "Precision": round(precision_score(y_test, y_pred, average="weighted",
                                           zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, average="weighted",
                                        zero_division=0), 4),
        "AUC":       round(auc, 4) if not np.isnan(auc) else np.nan,
    }


def run_all_scenarios(train_real, test_real, filtered_ctgan, consensus_df):
    """
    Run all three training scenarios for all three classifiers.

    For each scenario we standardise the training features using a scaler
    fitted on that scenario's training data, and apply the same transformation
    to the test features. This prevents information about the test set from
    influencing the feature scaling.

    Parameters
    ----------
    train_real     : real training patients (70 percent split)
    test_real      : real test patients (30 percent, never used for training)
    filtered_ctgan : IQR-filtered CTGAN synthetic records for Scenario B
    consensus_df   : consensus synthetic records for Scenario C

    Returns
    -------
    results_df : pandas DataFrame with one row per (scenario, classifier) pair,
                 containing n_train, Accuracy, F1, Precision, Recall, and AUC
    """
    # Only use feature columns that are actually present in the data
    feat_cols = [c for c in ALL_FEATURE_COLS if c in train_real.columns]

    X_test = test_real[feat_cols].values
    y_test = np.round(test_real[TARGET_COL].values).astype(int)

    def combine_and_clean(*dataframes):
        """Concatenate DataFrames and remove any rows that still have missing values."""
        combined = pd.concat(list(dataframes), ignore_index=True)
        return combined.dropna(subset=feat_cols + [TARGET_COL])

    scenarios = {
        "A: Baseline (real data only)":          train_real,
        "B: Real data plus filtered CTGAN":      combine_and_clean(train_real, filtered_ctgan),
        "C: Real data plus consensus synthetic":  combine_and_clean(train_real, consensus_df),
        "E: Synthetic data only (CTGAN filtered)": filtered_ctgan,
    }

    all_records = []

    for scenario_name, train_df in scenarios.items():
        X_train = train_df[feat_cols].values
        # Round then cast to int so floating-point near-zero values from
        # generative models (e.g. 2.8e-17 instead of 0) are treated as
        # binary class labels, not continuous regression targets.
        y_train = np.round(train_df[TARGET_COL].values).astype(int)

        # Fit the scaler on this scenario's training data only
        scaler    = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        print(f"\n  Scenario: {scenario_name}")
        print(f"  Training set size: {len(train_df)}  |  Test set size: {len(test_real)}")

        for clf_name, clf in _build_classifiers().items():
            metrics = _evaluate_one_classifier(
                clf, X_train_s, y_train, X_test_s, y_test
            )
            row = {
                "Scenario":   scenario_name,
                "Classifier": clf_name,
                "n_train":    len(train_df),
                **metrics
            }
            all_records.append(row)
            print(f"    {clf_name:<22}  Accuracy {metrics['Accuracy']:.4f}  "
                  f"F1 {metrics['F1']:.4f}  AUC {metrics['AUC']}")

    return pd.DataFrame(all_records)


def pivot_results(results_df, metric="F1"):
    """
    Rearrange the results table into a Scenario by Classifier grid.

    This format is convenient for inserting into a paper table or for
    generating a heatmap visualisation.

    Parameters
    ----------
    results_df : output of run_all_scenarios
    metric     : the performance metric to show in the grid cells

    Returns
    -------
    pivot table as a pandas DataFrame
    """
    return results_df.pivot(
        index="Scenario", columns="Classifier", values=metric
    ).round(4)
