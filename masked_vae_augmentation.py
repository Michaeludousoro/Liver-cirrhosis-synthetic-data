"""
Masked-VAE Augmentation Using the Discarded Partial Records
===========================================================

Project
-------
Synthetic Data Generation and Predictive Utility Evaluation
for Primary Biliary Cirrhosis (PBC) — Mayo Clinic Dataset

Motivation
----------
The main study applies complete-case analysis, which discards 142 of the 418
raw records and trains every generative model on the 193 complete training
patients. Section IV of the paper argues from the data processing inequality
that a generator restricted to those 193 records cannot introduce information
they do not already contain, which places a ceiling on what augmentation can
achieve.

The 142 discarded records sit outside that ceiling. Every one of them retains
the survival outcome together with N_Days, Bilirubin and Albumin, which the
feature-importance analysis ranks first, second and fourth. They are real
observations the generator never sees.

This script tests whether learning from them changes the null result. It trains
a variational autoencoder whose reconstruction loss is masked to the observed
entries, so gradients flow only through measured dimensions and partial records
can be used without imputation. The generator therefore learns from 335
patients (193 complete training records plus 142 partial ones) instead of 193.

Three scenarios are compared on the same 83 held-out real patients used
throughout the paper:

    A  193 real patients only                        (baseline, unchanged)
    F  193 real + masked-VAE synthetic records       (the proposal)
    G  193 real + 142 partial records, median-imputed (control)

Scenario G is the control that matters. If F beats A but G also beats A, the
gain came from having more data rather than from the generator, and the
generative step is doing no work.

The 83 test patients are never used for scaling, training or generation.

Usage
-----
    python master_runner.py            # first, for the baseline artefacts
    python masked_vae_augmentation.py
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.seeding import set_global_seeds
from src.data_loader import (load_complete_data, split_data,
                             ALL_FEATURE_COLS, TARGET_COL, CONTINUOUS_COLS,
                             BINARY_COLS, ORDINAL_COLS, post_process_synthetic)
from src.iqr_filter import compute_iqr_bounds, apply_iqr_filter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "cirrhosis.csv")
RES_DIR  = os.path.join(BASE_DIR, "output", "results")

COL_ORDER = ALL_FEATURE_COLS + [TARGET_COL]
N_GENERATE = 500
LATENT_DIM = 8
EPOCHS = 400
# KL weight. At beta = 1.0 the KL term overwhelms the masked reconstruction
# loss and the latent space collapses to the data mean (all generated records
# near-identical, 0 per cent deceased). A sweep over beta in {1, 0.1, 0.01,
# 0.001} and latent dimension in {8, 32} selected the setting below on training
# FID alone; the held-out test patients played no part in the choice.
BETA = 0.01
BATCH = 32
SEED = 42


# ---------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------
def load_all_encoded():
    """Load all 418 raw records with the paper's encodings, keeping NaN in place."""
    df = pd.read_csv(RAW_PATH)
    df = df.drop(columns=["ID"])
    df[TARGET_COL] = df[TARGET_COL].map({"D": 1, "C": 0, "CL": 0})
    df["Sex"]  = df["Sex"].map({"F": 0, "M": 1})
    df["Drug"] = df["Drug"].map({"Placebo": 0, "D-penicillamine": 1})
    for c in ["Ascites", "Hepatomegaly", "Spiders"]:
        df[c] = df[c].map({"N": 0, "Y": 1})
    df["Edema"] = df["Edema"].map({"N": 0, "S": 1, "Y": 2})
    return df[COL_ORDER]


def assemble():
    """Return the 193 training records, the 83 test records, and the 142 partial ones."""
    complete = load_complete_data(RAW_PATH)
    train_df, test_df = split_data(complete)

    all_df = load_all_encoded()
    complete_mask = all_df[COL_ORDER].notna().all(axis=1)
    partial_df = all_df[~complete_mask].reset_index(drop=True)

    return train_df[COL_ORDER], test_df[COL_ORDER], partial_df


# ---------------------------------------------------------------
# Masked VAE
# ---------------------------------------------------------------
class MaskedVAE(keras.Model):
    """
    A VAE that trains on records with missing entries.

    The encoder is told which entries are observed by receiving the mask
    alongside the (zero-filled) data. The reconstruction loss is averaged over
    observed entries only, so a missing measurement contributes no gradient
    rather than being treated as a zero to be reproduced.
    """

    def __init__(self, n_features, latent_dim=LATENT_DIM, beta=BETA):
        super().__init__()
        self.n_features = n_features
        self.beta = beta

        self.encoder = keras.Sequential([
            layers.Input(shape=(n_features * 2,)),   # data concatenated with mask
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
        ])
        self.to_mu     = layers.Dense(latent_dim)
        self.to_logvar = layers.Dense(latent_dim)

        self.decoder = keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(n_features, activation="sigmoid"),
        ])

    def call(self, inputs, training=False):
        x, mask = inputs
        h = self.encoder(tf.concat([x * mask, mask], axis=1))
        mu, logvar = self.to_mu(h), self.to_logvar(h)
        eps = tf.random.normal(tf.shape(mu))
        z = mu + eps * tf.exp(0.5 * logvar)
        return self.decoder(z), mu, logvar

    def train_step(self, data):
        x, mask = data
        with tf.GradientTape() as tape:
            x_hat, mu, logvar = self((x, mask), training=True)
            se = tf.square(x - x_hat) * mask

            # Average over the observed entries of each record, then rescale by
            # the feature count so the term has the same magnitude as a sum over
            # all features. Without this rescaling the reconstruction loss is
            # roughly n_features times smaller than the per-sample KL term, the
            # KL dominates, and the latent space collapses to the data mean.
            per_sample = tf.reduce_sum(se, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-8)
            recon = tf.reduce_mean(per_sample) * self.n_features

            kl = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1))
            loss = recon + self.beta * kl
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss, "recon": recon, "kl": kl}

    def sample(self, n):
        z = tf.random.normal((n, self.to_mu.units))
        return self.decoder(z).numpy()


def train_masked_vae(pool_df):
    """Scale the pool on observed entries only, then fit the masked VAE."""
    values = pool_df[COL_ORDER].values.astype("float32")
    mask = (~np.isnan(values)).astype("float32")

    # Fit the scaler column by column on observed entries only. Fitting on a
    # two-row [min; max] array gives MinMaxScaler exactly the per-column range
    # of the observed data, which a global fill value would destroy (N_Days is
    # in thousands while Bilirubin is in single digits). The pool contains no
    # test patients, so this introduces no leakage.
    col_min = np.nanmin(values, axis=0)
    col_max = np.nanmax(values, axis=0)
    scaler = MinMaxScaler()
    scaler.fit(np.vstack([col_min, col_max]))

    scaled = (values - col_min) / np.where(col_max - col_min == 0, 1.0, col_max - col_min)
    scaled = np.clip(np.nan_to_num(scaled, nan=0.0), 0.0, 1.0).astype("float32") * mask

    vae = MaskedVAE(n_features=len(COL_ORDER))
    vae.compile(optimizer=keras.optimizers.Adam(1e-3))
    ds = tf.data.Dataset.from_tensor_slices((scaled, mask)).shuffle(512, seed=SEED).batch(BATCH)
    vae.fit(ds, epochs=EPOCHS, verbose=0)
    return vae, scaler


def generate(vae, scaler, real_train_df, n=N_GENERATE):
    """Sample from the VAE, restore clinical units, snap categoricals, IQR filter."""
    raw = vae.sample(n)
    df_scaled = pd.DataFrame(raw, columns=COL_ORDER)
    synth = post_process_synthetic(df_scaled, scaler, COL_ORDER)
    synth[TARGET_COL] = synth[TARGET_COL].round().clip(0, 1).astype(int)

    bounds = compute_iqr_bounds(real_train_df)
    filtered, retention = apply_iqr_filter(synth, bounds)
    return synth, filtered, retention


# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------
def classifiers():
    return {
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=SEED),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs"),
    }


def evaluate(train_df, test_df, label, rows):
    X_tr = train_df[ALL_FEATURE_COLS].values
    y_tr = np.round(train_df[TARGET_COL].values).astype(int)
    X_te = test_df[ALL_FEATURE_COLS].values
    y_te = np.round(test_df[TARGET_COL].values).astype(int)

    sc = StandardScaler()
    X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)

    for name, clf in classifiers().items():
        clf.fit(X_tr_s, y_tr)
        pred = clf.predict(X_te_s)
        prob = clf.predict_proba(X_te_s)[:, 1]
        rows.append({
            "Scenario": label, "Classifier": name, "n_train": len(train_df),
            "Accuracy": round(accuracy_score(y_te, pred), 4),
            "F1":       round(f1_score(y_te, pred, average="weighted", zero_division=0), 4),
            "AUC":      round(roc_auc_score(y_te, prob), 4),
        })


def main():
    set_global_seeds(SEED)
    train_df, test_df, partial_df = assemble()

    print(f"  complete training records : {len(train_df)}")
    print(f"  held-out test records     : {len(test_df)}")
    print(f"  discarded partial records : {len(partial_df)}")
    obs = partial_df[COL_ORDER].notna().sum(axis=1)
    print(f"  observed fields in partial: min {obs.min()}, max {obs.max()}, mean {obs.mean():.1f} of {len(COL_ORDER)}")

    pool = pd.concat([train_df, partial_df], ignore_index=True)
    print(f"\n  masked-VAE training pool  : {len(pool)} patients")

    print("  training masked VAE ...")
    vae, scaler = train_masked_vae(pool)
    synth, filtered, retention = generate(vae, scaler, train_df)
    print(f"  generated {len(synth)}, retained {len(filtered)} after IQR filter "
          f"({retention:.1f}%)")

    # Persist the generated pool so its distributional quality and disclosure
    # risk can be compared against the generators in the main paper.
    filtered[COL_ORDER].to_csv(
        os.path.join(BASE_DIR, "output", "data", "filtered_masked_vae.csv"), index=False)

    rows = []
    evaluate(train_df, test_df, "A: Baseline (193 real)", rows)

    scen_f = pd.concat([train_df, filtered[COL_ORDER]], ignore_index=True)
    evaluate(scen_f, test_df, "F: Real + masked-VAE synthetic", rows)

    med = train_df[COL_ORDER].median()
    imputed = partial_df[COL_ORDER].fillna(med)
    for c in BINARY_COLS + list(ORDINAL_COLS.keys()) + [TARGET_COL]:
        imputed[c] = imputed[c].round().astype(int)
    scen_g = pd.concat([train_df, imputed], ignore_index=True)
    evaluate(scen_g, test_df, "G: Real + 142 imputed partial", rows)

    res = pd.DataFrame(rows)
    print("\n" + "=" * 74)
    print("  TEST-SET RESULTS (83 real held-out patients)")
    print("=" * 74)
    print(res.to_string(index=False))

    print("\n  AUC change against baseline:")
    base = res[res.Scenario.str.startswith("A")].set_index("Classifier").AUC
    for scen in ["F: Real + masked-VAE synthetic", "G: Real + 142 imputed partial"]:
        sub = res[res.Scenario == scen].set_index("Classifier").AUC
        deltas = ", ".join(f"{c.split()[0]} {sub[c]-base[c]:+.4f}" for c in base.index)
        print(f"    {scen:<34} {deltas}")

    out = os.path.join(RES_DIR, "masked_vae_augmentation.csv")
    res.to_csv(out, index=False)
    print(f"\n  Written to {out}")


if __name__ == "__main__":
    main()
