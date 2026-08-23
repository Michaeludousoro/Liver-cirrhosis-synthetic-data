"""
Survival Analysis (Time-to-Event Framing)
==========================================

Project
-------
Synthetic Data Generation and Predictive Utility Evaluation
for Primary Biliary Cirrhosis (PBC) — Mayo Clinic Dataset

Purpose of this module
-----------------------
The main predictive-modeling framing (src/predictive_modeling.py) binarises
Status as censored/transplant = 0 vs deceased = 1, which discards censoring:
a patient censored at 400 days and one censored at 4,500 days are coded
identically. This module instead treats N_Days as the follow-up duration
and Status as the event indicator, and evaluates a Cox proportional-hazards
model (and, where scikit-survival is available, a Random Survival Forest)
with Harrell's concordance index (C-index) rather than accuracy/AUC.

N_Days is excluded from the covariate set here for the same reason it is
excluded from CLASSIFICATION_FEATURE_COLS: it is the duration itself, not a
baseline covariate, so it cannot also appear as a predictor.

Augmentation scenarios A-E mirror the classification framing: baseline,
real+cGAN, real+consensus, real+SMOTE-analog, synthetic-only. Real and
synthetic records are pooled into one training frame and a single Cox model
(or RSF) is fit on the pool, exactly as the classification scenarios pool
real and synthetic rows before fitting a classifier.

SMOTE-analog for censored data
    Standard SMOTE (imblearn) has no notion of a censored duration target:
    it linearly interpolates feature vectors between a minority-class
    anchor and one of its same-class nearest neighbours, which is
    well-defined for a categorical label but not for a duration that may
    itself be censored. survival_smote_analog() below applies the same
    interpolation idea directly to (covariates, duration) jointly for the
    event=1 (deceased) minority class: for each anchor, a same-class
    nearest neighbour is found in standardised covariate space, and a
    synthetic record is produced by interpolating both the covariates and
    N_Days by the same random weight, keeping event=1. This is a deviation
    from the reviewer's literal "SMOTE" wording, applied because no
    off-the-shelf implementation supports censored targets, and it is
    documented here and in the paper rather than silently substituted.
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

from .data_loader import TARGET_COL, CLASSIFICATION_FEATURE_COLS, DURATION_COL

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_censored
    _HAS_SKSURV = True
except ImportError:
    _HAS_SKSURV = False
    concordance_index_censored = None


def _cox_frame(df, feat_cols, duration_col, event_col):
    """Select and clean the columns CoxPHFitter needs, dropping any
    zero-duration rows (Cox partial likelihood requires positive time)."""
    cols = feat_cols + [duration_col, event_col]
    out = df[cols].copy()
    out[duration_col] = out[duration_col].clip(lower=1e-6)
    out[event_col] = np.round(out[event_col]).astype(int)
    return out


def fit_cox(train_df, feat_cols=None, duration_col=DURATION_COL,
            event_col=TARGET_COL, penalizer=0.1):
    """
    Fit a Cox proportional-hazards model.

    A small ridge penalizer (0.1) is used throughout because several
    covariates are correlated (e.g. Bilirubin with Copper and SGOT, per
    the correlation analysis in the paper), and Cox regression on a
    training set of under 200 patients with 17 covariates is otherwise
    prone to unstable coefficient estimates.
    """
    if feat_cols is None:
        feat_cols = CLASSIFICATION_FEATURE_COLS
    cox_df = _cox_frame(train_df, feat_cols, duration_col, event_col)
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(cox_df, duration_col=duration_col, event_col=event_col)
    return cph


def concordance_on_test(cph, test_df, feat_cols=None, duration_col=DURATION_COL,
                         event_col=TARGET_COL):
    """Harrell's C-index of a fitted CoxPHFitter on held-out real patients."""
    if feat_cols is None:
        feat_cols = CLASSIFICATION_FEATURE_COLS
    cox_df = _cox_frame(test_df, feat_cols, duration_col, event_col)
    return cph.score(cox_df, scoring_method="concordance_index")


def fit_rsf(train_df, feat_cols=None, duration_col=DURATION_COL,
            event_col=TARGET_COL, random_state=42):
    """
    Fit a Random Survival Forest via scikit-survival, if installed.

    Returns None if scikit-survival is unavailable, so callers can fall
    back to Cox-only rather than fail; this is reported explicitly in the
    paper wherever it happens rather than silently omitted.
    """
    if not _HAS_SKSURV:
        return None
    if feat_cols is None:
        feat_cols = CLASSIFICATION_FEATURE_COLS
    X = train_df[feat_cols].values
    y = Surv.from_dataframe(event_col, duration_col,
                             train_df.assign(**{event_col: np.round(train_df[event_col]).astype(bool)}))
    rsf = RandomSurvivalForest(
        n_estimators=200, min_samples_split=6, min_samples_leaf=3,
        max_features="sqrt", n_jobs=-1, random_state=random_state
    )
    rsf.fit(X, y)
    return rsf


def concordance_rsf_on_test(rsf, test_df, feat_cols=None, duration_col=DURATION_COL,
                             event_col=TARGET_COL):
    """C-index of a fitted RandomSurvivalForest on held-out real patients."""
    if rsf is None:
        return np.nan
    if feat_cols is None:
        feat_cols = CLASSIFICATION_FEATURE_COLS
    X_test = test_df[feat_cols].values
    risk_scores = rsf.predict(X_test)
    return concordance_index_censored(
        np.round(test_df[event_col]).astype(bool).values,
        test_df[duration_col].values,
        risk_scores
    )[0]


def survival_smote_analog(train_df, feat_cols=None, duration_col=DURATION_COL,
                           event_col=TARGET_COL, k_neighbors=5, random_state=42):
    """
    SMOTE-style oversampling of the event=1 (deceased) minority class,
    extended to interpolate the censored duration jointly with covariates.
    See the module docstring for why this deviates from off-the-shelf
    SMOTE. Balances event=1 records up to the event=0 count, matching
    imblearn SMOTE's default sampling_strategy.

    Returns
    -------
    augmented_df : train_df with synthetic minority-class rows appended
    """
    if feat_cols is None:
        feat_cols = CLASSIFICATION_FEATURE_COLS
    rng = np.random.RandomState(random_state)

    minority = train_df[np.round(train_df[event_col]).astype(int) == 1].reset_index(drop=True)
    majority = train_df[np.round(train_df[event_col]).astype(int) == 0]
    n_to_generate = len(majority) - len(minority)
    if n_to_generate <= 0 or len(minority) < 2:
        return train_df.copy()

    scaler = StandardScaler()
    X_min_s = scaler.fit_transform(minority[feat_cols].values)

    k = min(k_neighbors, len(minority) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min_s)
    _, neighbor_idx = nn.kneighbors(X_min_s)

    synthetic_rows = []
    for _ in range(n_to_generate):
        anchor_i = rng.randint(len(minority))
        neighbor_i = neighbor_idx[anchor_i, rng.randint(1, k + 1)]
        lam = rng.uniform(0, 1)

        anchor = minority.iloc[anchor_i]
        neighbor = minority.iloc[neighbor_i]

        row = {}
        for c in feat_cols:
            row[c] = anchor[c] + lam * (neighbor[c] - anchor[c])
        row[duration_col] = anchor[duration_col] + lam * (neighbor[duration_col] - anchor[duration_col])
        row[event_col] = 1
        synthetic_rows.append(row)

    synth_df = pd.DataFrame(synthetic_rows)
    return pd.concat([train_df, synth_df], ignore_index=True)


def run_survival_scenarios(train_real, test_real, filtered_ctgan, consensus_df,
                            feat_cols=None, duration_col=DURATION_COL,
                            event_col=TARGET_COL, include_rsf=True):
    """
    Scenarios A'-E' under the survival framing: baseline, real+cGAN,
    real+consensus, real+SMOTE-analog, synthetic-only. Evaluated by
    Harrell's C-index on the real held-out test set (CoxPH always; RSF
    also, if scikit-survival is installed and include_rsf is True).

    Returns
    -------
    results_df : one row per scenario, with columns Scenario, n_train,
                 C_index_Cox, and C_index_RSF (NaN if RSF unavailable)
    """
    if feat_cols is None:
        feat_cols = CLASSIFICATION_FEATURE_COLS

    def combine(*frames):
        return pd.concat(list(frames), ignore_index=True)

    smote_analog_df = survival_smote_analog(
        train_real, feat_cols=feat_cols, duration_col=duration_col, event_col=event_col
    )

    scenarios = {
        "A: Baseline (real data only)":            train_real,
        "B: Real data plus filtered cGAN":         combine(train_real, filtered_ctgan),
        "C: Real data plus consensus synthetic":   combine(train_real, consensus_df),
        "D: Real data plus SMOTE-analog":          smote_analog_df,
        "E: Synthetic data only (cGAN filtered)":  filtered_ctgan,
    }

    records = []
    for name, train_df in scenarios.items():
        cph = fit_cox(train_df, feat_cols, duration_col, event_col)
        c_cox = concordance_on_test(cph, test_real, feat_cols, duration_col, event_col)

        c_rsf = np.nan
        if include_rsf and _HAS_SKSURV:
            rsf = fit_rsf(train_df, feat_cols, duration_col, event_col)
            c_rsf = concordance_rsf_on_test(rsf, test_real, feat_cols, duration_col, event_col)

        rsf_str = f"{c_rsf:.4f}" if not np.isnan(c_rsf) else "n/a"
        print(f"  Survival scenario: {name}  n_train={len(train_df)}  "
              f"C-index (Cox)={c_cox:.4f}  C-index (RSF)={rsf_str}")
        records.append({
            "Scenario": name, "n_train": len(train_df),
            "C_index_Cox": round(c_cox, 4),
            "C_index_RSF": round(c_rsf, 4) if not np.isnan(c_rsf) else np.nan,
        })

    return pd.DataFrame(records)


def km_logrank_comparison(real_df, synthetic_pools, duration_col=DURATION_COL,
                           event_col=TARGET_COL):
    """
    Compare the real training cohort's (N_Days, Status) distribution
    against each generator's synthetic pool via Kaplan-Meier curves and a
    log-rank test. This directly evaluates whether a generator reproduces
    the joint (time, event) structure, which the classification-only FID
    metric does not assess (FID summarises the full 18-dimensional
    Euclidean geometry, not the survival-curve shape specifically).

    Parameters
    ----------
    real_df          : real training patients
    synthetic_pools  : dict {generator_name: synthetic_df}

    Returns
    -------
    results_df   : one row per generator, with the log-rank test statistic
                   and p-value (low p = synthetic survival curve differs
                   significantly from the real one)
    km_fitters   : dict {generator_name: (real_kmf, synth_kmf)} of fitted
                   KaplanMeierFitter objects, for plotting
    """
    real_duration = real_df[duration_col].values
    real_event = np.round(real_df[event_col]).astype(int).values

    real_kmf = KaplanMeierFitter()
    real_kmf.fit(real_duration, event_observed=real_event, label="Real")

    records = []
    fitters = {}
    for gen_name, synth_df in synthetic_pools.items():
        synth_duration = synth_df[duration_col].clip(lower=1e-6).values
        synth_event = np.round(synth_df[event_col]).astype(int).values

        synth_kmf = KaplanMeierFitter()
        synth_kmf.fit(synth_duration, event_observed=synth_event, label=gen_name)
        fitters[gen_name] = (real_kmf, synth_kmf)

        result = logrank_test(
            real_duration, synth_duration,
            event_observed_A=real_event, event_observed_B=synth_event
        )
        records.append({
            "Generator": gen_name,
            "logrank_statistic": round(result.test_statistic, 4),
            "p_value": round(result.p_value, 4),
        })

    return pd.DataFrame(records), fitters
