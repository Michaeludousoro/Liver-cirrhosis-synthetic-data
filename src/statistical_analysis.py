"""
Statistical Analysis
====================

Project
-------
Synthetic Data Generation and Predictive Utility Evaluation
for Primary Biliary Cirrhosis (PBC) — Mayo Clinic Dataset

Purpose of this module
-----------------------
Reporting FID scores and classification accuracy alone is not enough for
a rigorous scientific paper. We also need to formally test whether the
observed differences between synthetic and real data distributions are
statistically meaningful, and we need to quantify how large those differences
actually are in practice.

This module runs five statistical tests on the real and synthetic feature
distributions. Each test answers a different scientific question about the
quality and fidelity of the generated data.

Tests carried out and what each one tells us
    Shapiro-Wilk normality test
        We test whether each continuous feature in the real training data
        follows a normal distribution. This matters because it determines
        whether we should use parametric or non-parametric comparisons
        elsewhere. A p-value below 0.05 means the feature is not normally
        distributed, and non-parametric tests like KS are therefore more
        appropriate.

    Kolmogorov-Smirnov two-sample test
        For each continuous feature and each synthetic method, we compare
        the full empirical distribution of real values against the synthetic
        values. This test is distribution-free and is sensitive to differences
        anywhere in the distribution, not just in the mean or variance. A
        p-value below 0.05 suggests the synthetic feature distribution differs
        significantly from the real one.

    Jensen-Shannon Divergence
        A symmetric measure of how different two probability distributions are.
        We compute it by discretising each continuous feature into histogram
        bins and comparing the bin proportions. The squared JSD falls between
        zero (identical distributions) and one (completely non-overlapping
        distributions). This gives us a smooth, bounded complement to the KS
        test result.

    Cohen's d effect size
        Measures the standardised mean difference between the real and
        synthetic distributions for each continuous feature. We use this
        because statistical significance alone does not tell us whether a
        difference is practically important. Values near zero indicate
        negligible differences; values above 0.8 indicate large differences
        in the means relative to the spread.

    Chi-square goodness-of-fit test
        For binary and ordinal features (sex, drug assignment, disease stage,
        edema, ascites, hepatomegaly, spiders, and patient status), we check
        whether the proportions in the synthetic data match those in the real
        data. A significant result means the synthetic data assigns different
        frequencies to the categories than the real patient population does.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import binomtest

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

from .data_loader import (CONTINUOUS_COLS, BINARY_COLS, ORDINAL_COLS,
                          TARGET_COL, CLASSIFICATION_FEATURE_COLS)
from .predictive_modeling import _build_classifiers


def shapiro_wilk_normality(real_df, cols=None):
    """
    Test whether each continuous feature in the real data is normally distributed.

    We apply this test to the real data only, not the synthetic data, because
    the purpose is to characterise the real patient population and to inform
    the choice of subsequent tests. If a feature is not normally distributed,
    non-parametric tests such as KS are more appropriate than t-tests.

    Parameters
    ----------
    real_df : real training DataFrame
    cols    : columns to test (default: all 11 continuous clinical features)

    Returns
    -------
    pandas DataFrame with one row per feature containing the W statistic,
    p-value, and a plain-language conclusion about normality
    """
    if cols is None:
        cols = CONTINUOUS_COLS

    rows = []
    for col in cols:
        if col not in real_df.columns:
            continue

        # Shapiro-Wilk works on the actual observed values with missing rows removed
        w_stat, p_val = stats.shapiro(real_df[col].dropna())

        rows.append({
            "Feature":     col,
            "W_statistic": round(w_stat, 4),
            "p_value":     round(p_val, 6),
            # A p-value above 0.05 means we cannot reject the assumption of normality
            "Normal":      "Yes" if p_val > 0.05 else "No",
        })

    return pd.DataFrame(rows)


def kolmogorov_smirnov_tests(real_df, synthetic_dict, cols=None):
    """
    Compare the distribution of each continuous feature between real and synthetic data.

    The KS two-sample test compares full empirical distributions rather than
    just comparing means or standard deviations. A significant result for a
    given feature and method means that the synthetic method did not reproduce
    the real distribution of that feature well.

    Parameters
    ----------
    real_df        : real training DataFrame
    synthetic_dict : dictionary mapping method names to synthetic DataFrames,
                     for example {'GAN (filtered)': df, 'cGAN (filtered)': df}
    cols           : continuous features to test (default: all 11 standard ones)

    Returns
    -------
    pandas DataFrame with one row per (method, feature) combination
    """
    if cols is None:
        cols = CONTINUOUS_COLS

    rows = []
    for method_name, synth_df in synthetic_dict.items():
        for col in cols:
            if col not in real_df.columns or col not in synth_df.columns:
                continue

            ks_stat, p_val = stats.ks_2samp(
                real_df[col].dropna().values,
                synth_df[col].dropna().values
            )

            rows.append({
                "Method":                  method_name,
                "Feature":                 col,
                "KS_stat":                 round(ks_stat, 4),
                "p_value":                 round(p_val, 6),
                # A significant difference here suggests the synthetic method
                # failed to reproduce this feature's distribution faithfully
                "Significant_difference":  "Yes" if p_val < 0.05 else "No",
            })

    return pd.DataFrame(rows)


def _build_histogram(values, bin_edges):
    """
    Build a normalised histogram over a fixed set of bin edges.

    We add a tiny constant before normalising to avoid zero-probability bins,
    which would cause numerical problems when computing the Jensen-Shannon
    divergence. The resulting histogram sums to 1 and represents a
    discrete probability distribution over the bin intervals.
    """
    counts, _ = np.histogram(values, bins=bin_edges)

    # Add a small floor value so no bin has exactly zero probability
    counts = counts.astype(float) + 1e-10

    return counts / counts.sum()


def jensen_shannon_divergences(real_df, synthetic_dict, cols=None, n_bins=30):
    """
    Compute the Jensen-Shannon divergence for each continuous feature and method.

    We discretise each feature into a fixed set of bins that spans the combined
    range of real and synthetic values. Using the same bin edges for both
    distributions ensures the comparison is fair and meaningful. The squared JSD
    is bounded between zero (identical distributions) and one (distributions
    with no overlapping mass).

    Parameters
    ----------
    real_df        : real training DataFrame
    synthetic_dict : dictionary of synthetic DataFrames
    cols           : continuous features (default: all 11 standard ones)
    n_bins         : number of histogram bins per feature

    Returns
    -------
    pandas DataFrame with columns Method, Feature, and JSD
    """
    if cols is None:
        cols = CONTINUOUS_COLS

    rows = []
    for method_name, synth_df in synthetic_dict.items():
        for col in cols:
            if col not in real_df.columns or col not in synth_df.columns:
                continue

            real_vals  = real_df[col].dropna().values
            synth_vals = synth_df[col].dropna().values

            # We span both datasets so neither one is artificially clipped
            combined = np.concatenate([real_vals, synth_vals])
            lo, hi   = combined.min(), combined.max()

            # Skip features with no variation — divergence is undefined there
            if lo == hi:
                continue

            bin_edges = np.linspace(lo, hi, n_bins + 1)
            p = _build_histogram(real_vals,  bin_edges)
            q = _build_histogram(synth_vals, bin_edges)

            # scipy returns the square root of JSD, so we square it to get
            # a value in [0, 1] that is directly interpretable as a proportion
            jsd = jensenshannon(p, q) ** 2

            rows.append({
                "Method":  method_name,
                "Feature": col,
                "JSD":     round(jsd, 6),
            })

    return pd.DataFrame(rows)


def cohens_d_effect_sizes(real_df, synthetic_dict, cols=None):
    """
    Compute Cohen's d between real and synthetic feature distributions.

    Cohen's d is the standardised mean difference: the difference in means
    between the real and synthetic distributions divided by the pooled
    standard deviation. This tells us not just whether a difference is
    statistically significant but how large it is in practical terms.

    Interpretation guide
        Negligible: absolute value less than 0.2
        Small: between 0.2 and 0.5
        Medium: between 0.5 and 0.8
        Large: above 0.8

    Parameters
    ----------
    real_df        : real training DataFrame
    synthetic_dict : dictionary of synthetic DataFrames
    cols           : continuous features (default: all 11 standard ones)

    Returns
    -------
    pandas DataFrame with columns Method, Feature, Cohen_d, and Effect_size
    """
    if cols is None:
        cols = CONTINUOUS_COLS

    rows = []
    for method_name, synth_df in synthetic_dict.items():
        for col in cols:
            if col not in real_df.columns or col not in synth_df.columns:
                continue

            r = real_df[col].dropna().values
            s = synth_df[col].dropna().values

            # The pooled standard deviation weights each group by its sample size
            pooled_std = np.sqrt(
                ((len(r) - 1) * r.std(ddof=1) ** 2 +
                 (len(s) - 1) * s.std(ddof=1) ** 2) /
                max(len(r) + len(s) - 2, 1)
            )

            # A small epsilon in the denominator prevents division by zero
            # when both distributions have identical values
            d = (r.mean() - s.mean()) / (pooled_std + 1e-10)

            # Assign a plain-language label based on the absolute magnitude
            if   abs(d) < 0.2: effect = "negligible"
            elif abs(d) < 0.5: effect = "small"
            elif abs(d) < 0.8: effect = "medium"
            else:               effect = "large"

            rows.append({
                "Method":      method_name,
                "Feature":     col,
                "Cohen_d":     round(d, 4),
                "Effect_size": effect,
            })

    return pd.DataFrame(rows)


def chi_square_tests(real_df, synthetic_dict, cat_cols=None):
    """
    Test whether the synthetic categorical features match the real proportions.

    For each categorical feature we compare how often each category appears
    in the real data against how often it appears in the synthetic data. A
    significant result means the synthetic data is generating a different
    mix of categories than what we observe in the real patient population.

    This test covers binary features (sex, drug assignment, ascites, etc.),
    ordinal features (edema stage, disease stage), and the target outcome.

    Parameters
    ----------
    real_df        : real training DataFrame
    synthetic_dict : dictionary of synthetic DataFrames
    cat_cols       : categorical columns to test
                     default includes all binary features, ordinal features,
                     and the patient status target variable

    Returns
    -------
    pandas DataFrame with one row per (method, feature) combination
    """
    if cat_cols is None:
        cat_cols = BINARY_COLS + list(ORDINAL_COLS.keys()) + [TARGET_COL]

    rows = []
    for method_name, synth_df in synthetic_dict.items():
        for col in cat_cols:
            if col not in real_df.columns or col not in synth_df.columns:
                continue

            real_counts  = real_df[col].value_counts().sort_index()
            synth_counts = synth_df[col].value_counts().sort_index()

            # Use the union of categories so that categories present in only
            # one dataset are still represented in the comparison
            all_categories = real_counts.index.union(synth_counts.index)
            observed = np.array([real_counts.get(v, 0) for v in all_categories], float)
            expected = np.array([synth_counts.get(v, 0) for v in all_categories], float)

            # Skip this feature if either dataset has no records
            if observed.sum() == 0 or expected.sum() == 0:
                continue

            # Scale the expected counts so they have the same total as the
            # observed counts — chi-square requires this for a valid comparison
            expected_scaled = expected / expected.sum() * observed.sum()

            # Add a small floor to prevent zero-expected cells, which would
            # make the chi-square statistic undefined
            expected_scaled = np.maximum(expected_scaled, 0.5)

            try:
                chi2, p_val = stats.chisquare(observed, f_exp=expected_scaled)
                rows.append({
                    "Method":                  method_name,
                    "Feature":                 col,
                    "Chi2":                    round(chi2, 4),
                    "p_value":                 round(p_val, 6),
                    "Significant_difference":  "Yes" if p_val < 0.05 else "No",
                })
            except Exception:
                # Skip categories that cause numerical issues (e.g. only one bin)
                pass

    return pd.DataFrame(rows)


def run_all_statistical_tests(real_df, synthetic_dict):
    """
    Run the full battery of statistical tests and return all results.

    This is the single entry point that the master runner calls. It runs all
    five tests in sequence and packages the results into a dictionary so that
    each test's output can be saved independently as a CSV and referenced
    in the paper.

    Parameters
    ----------
    real_df        : real training DataFrame
    synthetic_dict : dictionary mapping method names to synthetic DataFrames

    Returns
    -------
    results : dictionary with one key per test, each containing a DataFrame
    """
    print("\n  Running Shapiro-Wilk normality tests on real data features ...")
    normality = shapiro_wilk_normality(real_df)

    print("  Running Kolmogorov-Smirnov distribution comparison tests ...")
    ks = kolmogorov_smirnov_tests(real_df, synthetic_dict)

    print("  Computing Jensen-Shannon divergences ...")
    jsd = jensen_shannon_divergences(real_df, synthetic_dict)

    print("  Computing Cohen's d effect sizes ...")
    cohens = cohens_d_effect_sizes(real_df, synthetic_dict)

    print("  Running chi-square tests on categorical features ...")
    chi = chi_square_tests(real_df, synthetic_dict)

    return {
        "shapiro_wilk":  normality,
        "ks_tests":      ks,
        "js_divergence": jsd,
        "cohens_d":      cohens,
        "chi_square":    chi,
    }


# ---------------------------------------------------------------------
# Predictive-utility inferential tests
#
# These three functions (cross_validate_scenarios, bootstrap_ci_scenarios,
# mcnemar_scenarios) were previously implemented only as inline, copy-
# pasted cells in 03_predictive_modeling_and_statistical_evaluation.ipynb
# (cells 38, 40, 42), each redefining its own classifier dict and its own
# feat_cols logic. They are extracted here, generalized over an explicit
# scenario dict, so the same tested implementation can be reused for the
# leak-fixed classification framing, the landmark framing, and any other
# scenario set (e.g. six generators) without copy-pasting notebook cells
# again. All three default feat_cols to CLASSIFICATION_FEATURE_COLS
# (N_Days excluded) rather than ALL_FEATURE_COLS.
# ---------------------------------------------------------------------

def cross_validate_scenarios(train_real, scenario_augment, feat_cols=None,
                              target_col=TARGET_COL, n_splits=5, random_state=42):
    """
    5-fold stratified cross-validation, folds drawn from real training
    patients only. Synthetic augmentation (or SMOTE) is applied to the
    TRAIN side of each fold; every validation fold contains real patients
    exclusively, so no synthetic record ever leaks into validation.

    Parameters
    ----------
    train_real       : real training patients (folds are drawn from this)
    scenario_augment : dict {scenario_name: augment}, where augment is
                        None (baseline, no augmentation), a DataFrame
                        (synthetic pool concatenated to the train fold), or
                        the string "smote" (SMOTE applied inside the fold)
    feat_cols        : defaults to CLASSIFICATION_FEATURE_COLS
    target_col       : defaults to TARGET_COL; pass "landmark_event" for
                        the landmark framing

    Returns
    -------
    cv_df : one row per (scenario, classifier), with Mean AUC, Std AUC,
            and a naive Normal-approximation 95% CI (Mean +/- 1.96*Std)
    """
    if feat_cols is None:
        feat_cols = CLASSIFICATION_FEATURE_COLS
    feat_cols = [c for c in feat_cols if c in train_real.columns]

    def combine_clean(*dfs):
        combined = pd.concat(list(dfs), ignore_index=True)
        return combined.dropna(subset=feat_cols + [target_col])

    X_real = train_real[feat_cols].values
    y_real = np.round(train_real[target_col].values).astype(int)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    records = []

    for scenario_name, augment in scenario_augment.items():
        for clf_name, base_clf in _build_classifiers().items():
            fold_aucs = []
            for tr_idx, va_idx in splitter.split(X_real, y_real):
                X_va = X_real[va_idx]
                y_va = y_real[va_idx]
                real_tr = train_real.iloc[tr_idx]
                scaler = StandardScaler()

                if augment is None:
                    X_tr_s = scaler.fit_transform(real_tr[feat_cols].values)
                    y_tr = np.round(real_tr[target_col].values).astype(int)
                elif isinstance(augment, str) and augment == "smote":
                    X_tr_real = scaler.fit_transform(real_tr[feat_cols].values)
                    y_tr_real = np.round(real_tr[target_col].values).astype(int)
                    X_tr_s, y_tr = SMOTE(random_state=random_state).fit_resample(
                        X_tr_real, y_tr_real)
                else:
                    tr_df = combine_clean(real_tr, augment)
                    X_tr_s = scaler.fit_transform(tr_df[feat_cols].values)
                    y_tr = np.round(tr_df[target_col].values).astype(int)

                X_va_s = scaler.transform(X_va)
                clf = clone(base_clf)
                clf.fit(X_tr_s, y_tr)
                fold_aucs.append(roc_auc_score(y_va, clf.predict_proba(X_va_s)[:, 1]))

            mean_auc, std_auc = np.mean(fold_aucs), np.std(fold_aucs)
            records.append({
                "Scenario": scenario_name, "Classifier": clf_name,
                "Mean AUC": round(mean_auc, 4), "Std AUC": round(std_auc, 4),
                "CI Lower": round(mean_auc - 1.96 * std_auc, 4),
                "CI Upper": round(mean_auc + 1.96 * std_auc, 4),
            })

    return pd.DataFrame(records)


def bootstrap_ci_scenarios(scenario_train_dfs, test_real, feat_cols=None,
                            target_col=TARGET_COL, n_boot=1000, random_state=42):
    """
    95% bootstrap confidence intervals for test-set AUC, one classifier
    trained per (scenario, classifier) pair and evaluated on n_boot
    resamples of the held-out real test set.

    Parameters
    ----------
    scenario_train_dfs : dict {scenario_name: train_df}, already combined
                          (e.g. real+cGAN concatenated) — one classifier
                          fit per scenario, not per bootstrap resample
    test_real           : real test patients
    feat_cols, target_col : as elsewhere

    Returns
    -------
    boot_df : one row per (scenario, classifier), with AUC, CI Lower, CI Upper
    """
    if feat_cols is None:
        feat_cols = CLASSIFICATION_FEATURE_COLS
    feat_cols = [c for c in feat_cols if c in test_real.columns]

    X_test = test_real[feat_cols].values
    y_test = np.round(test_real[target_col].values).astype(int)

    trained_probs = {}
    for scenario_name, train_df in scenario_train_dfs.items():
        X_train = train_df[feat_cols].values
        y_train = np.round(train_df[target_col].values).astype(int)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        trained_probs[scenario_name] = {}
        for clf_name, clf in _build_classifiers().items():
            clf.fit(X_train_s, y_train)
            trained_probs[scenario_name][clf_name] = clf.predict_proba(X_test_s)[:, 1]

    rng = np.random.RandomState(random_state)
    n_te = len(y_test)
    records = []

    for scenario_name, probs_by_clf in trained_probs.items():
        for clf_name, y_prob in probs_by_clf.items():
            boot_aucs = []
            for _ in range(n_boot):
                idx = rng.randint(0, n_te, size=n_te)
                y_b, yp_b = y_test[idx], y_prob[idx]
                if len(np.unique(y_b)) < 2:
                    continue
                boot_aucs.append(roc_auc_score(y_b, yp_b))

            point = roc_auc_score(y_test, y_prob)
            records.append({
                "Scenario": scenario_name, "Classifier": clf_name,
                "AUC": round(point, 4),
                "CI Lower": round(np.percentile(boot_aucs, 2.5), 4),
                "CI Upper": round(np.percentile(boot_aucs, 97.5), 4),
            })

    return pd.DataFrame(records)


def mcnemar_scenarios(scenario_train_dfs, test_real, baseline_name,
                       feat_cols=None, target_col=TARGET_COL):
    """
    McNemar's exact test (via the exact binomial test on discordant pairs)
    comparing every non-baseline scenario against the baseline scenario,
    for each classifier.

    Parameters
    ----------
    scenario_train_dfs : dict {scenario_name: train_df}, must include
                          baseline_name as one of the keys
    test_real            : real test patients
    baseline_name        : key of the baseline scenario in scenario_train_dfs
    feat_cols, target_col : as elsewhere

    Returns
    -------
    mcnemar_df : one row per (classifier, compared scenario), with n10, n01
                 (discordant pair counts) and the two-sided exact p-value
    """
    if feat_cols is None:
        feat_cols = CLASSIFICATION_FEATURE_COLS
    feat_cols = [c for c in feat_cols if c in test_real.columns]

    X_test = test_real[feat_cols].values
    y_test = np.round(test_real[target_col].values).astype(int)

    def get_preds(train_df):
        X_train = train_df[feat_cols].values
        y_train = np.round(train_df[target_col].values).astype(int)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        out = {}
        for clf_name, clf in _build_classifiers().items():
            clf.fit(X_train_s, y_train)
            out[clf_name] = clf.predict(X_test_s)
        return out

    preds_by_scenario = {name: get_preds(df) for name, df in scenario_train_dfs.items()}
    preds_baseline = preds_by_scenario[baseline_name]

    records = []
    for clf_name in preds_baseline:
        correct_baseline = (preds_baseline[clf_name] == y_test)
        for scenario_name, preds in preds_by_scenario.items():
            if scenario_name == baseline_name:
                continue
            correct_x = (preds[clf_name] == y_test)
            n10 = int((correct_baseline & ~correct_x).sum())
            n01 = int((~correct_baseline & correct_x).sum())
            n = n10 + n01

            p_val = 1.0 if n == 0 else binomtest(
                min(n10, n01), n=n, p=0.5, alternative="two-sided").pvalue

            records.append({
                "Classifier": clf_name,
                "Scenario_baseline": baseline_name,
                "Scenario_compare": scenario_name,
                "n10": n10, "n01": n01,
                "p_value": round(p_val, 4),
                "Significant": "Yes" if p_val < 0.05 else "No",
            })

    return pd.DataFrame(records)
