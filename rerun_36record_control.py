"""
36-Same-Protocol-Record Control (Leak-Free Rerun)
====================================================

Purpose
-------
The paper's Threats to Validity section reports two controls testing
whether the masked-loss VAE's failure to improve prediction (Section
IV) is an artefact of pooling two different missingness protocols. Both
controls were originally run before the N_Days leakage fix. This script
reruns them on the leak-free 17-feature input (Section III):

  Control 1: add the 36 randomised-arm records (missing only 1-2 fields)
             directly to the 193-patient training set after median
             imputation, and measure the AUC change.
  Control 2: train a masked-loss VAE on 193 + 36 = 229 patients (as
             opposed to the full 335-patient pool used in the main
             Section IV robustness check), generate synthetic records,
             IQR-filter them, and augment with those instead of the raw
             imputed records, to see whether passing real records
             through a generator loses information relative to using
             them directly.
"""

import numpy as np
import pandas as pd

from masked_vae_augmentation import (assemble, train_masked_vae, generate,
                                     COL_ORDER, EPOCHS, N_GENERATE, RES_DIR)
from src.data_loader import (TARGET_COL, CLASSIFICATION_FEATURE_COLS,
                             BINARY_COLS, ORDINAL_COLS)
from src.seeding import set_global_seeds
from src.predictive_modeling import _build_classifiers, _evaluate_one_classifier
from sklearn.preprocessing import StandardScaler

SEED = 42


def evaluate_scenario(train_df, test_df, label, rows):
    feat_cols = CLASSIFICATION_FEATURE_COLS
    X_tr = train_df[feat_cols].values
    y_tr = np.round(train_df[TARGET_COL].values).astype(int)
    X_te = test_df[feat_cols].values
    y_te = np.round(test_df[TARGET_COL].values).astype(int)

    scaler = StandardScaler()
    X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)

    for name, clf in _build_classifiers().items():
        metrics = _evaluate_one_classifier(clf, X_tr_s, y_tr, X_te_s, y_te)
        rows.append({"Scenario": label, "Classifier": name,
                     "n_train": len(train_df), **metrics})


def main():
    set_global_seeds(SEED)
    train_df, test_df, partial_df = assemble()

    n_observed = partial_df[COL_ORDER].notna().sum(axis=1)
    same_protocol = partial_df[n_observed >= len(COL_ORDER) - 2].reset_index(drop=True)
    observational = partial_df[n_observed < len(COL_ORDER) - 2].reset_index(drop=True)
    print(f"same-protocol (missing 1-2 fields): {len(same_protocol)}")
    print(f"observational (missing 9-10 fields): {len(observational)}")
    assert len(same_protocol) == 36, f"expected 36 same-protocol records, got {len(same_protocol)}"

    rows = []
    evaluate_scenario(train_df, test_df, "A: Baseline (193 real)", rows)

    # Control 1: 36 same-protocol records, median-imputed, added directly.
    med = train_df[COL_ORDER].median()
    imputed_36 = same_protocol[COL_ORDER].fillna(med)
    for c in BINARY_COLS + list(ORDINAL_COLS.keys()) + [TARGET_COL]:
        imputed_36[c] = imputed_36[c].round().astype(int)
    scen_h = pd.concat([train_df, imputed_36], ignore_index=True)
    evaluate_scenario(scen_h, test_df, "H: Real + 36 imputed same-protocol", rows)

    # Control 2: train a masked VAE on 193 + 36 = 229 patients only, generate,
    # IQR-filter, and augment with the synthetic pool instead of the raw records.
    pool_229 = pd.concat([train_df, same_protocol], ignore_index=True)
    print(f"\nTraining masked VAE on {len(pool_229)} patients (193 + 36 same-protocol)...")
    vae, scaler = train_masked_vae(pool_229)
    synth, filtered, retention = generate(vae, scaler, train_df, n=N_GENERATE)
    print(f"generated {len(synth)}, retained {len(filtered)} after IQR filter ({retention:.1f}%)")
    scen_i = pd.concat([train_df, filtered[COL_ORDER]], ignore_index=True)
    evaluate_scenario(scen_i, test_df, "I: Real + synthetic-from-36-pool", rows)

    res = pd.DataFrame(rows)
    print("\n" + "=" * 74)
    print("  36-RECORD CONTROL RESULTS (leak-free, N_Days excluded)")
    print("=" * 74)
    print(res.to_string(index=False))

    print("\nAUC change against baseline:")
    base = res[res.Scenario.str.startswith("A")].set_index("Classifier").AUC
    for scen in ["H: Real + 36 imputed same-protocol", "I: Real + synthetic-from-36-pool"]:
        sub = res[res.Scenario == scen].set_index("Classifier").AUC
        deltas = ", ".join(f"{c.split()[0]} {sub[c]-base[c]:+.4f}" for c in base.index)
        print(f"  {scen:<38} {deltas}")

    out = f"{RES_DIR}/same_protocol_control.csv"
    res.to_csv(out, index=False)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
