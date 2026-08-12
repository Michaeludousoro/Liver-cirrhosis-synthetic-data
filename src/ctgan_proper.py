"""
Conditional Tabular GAN (CTGAN) as specified by Xu et al.
=========================================================

Project
-------
Synthetic Data Generation and Predictive Utility Evaluation
for Primary Biliary Cirrhosis (PBC) — Mayo Clinic Dataset

Why this module exists
----------------------
The generator previously labelled "CTGAN" in this project was a conditional GAN:
it concatenated a one-hot outcome label to the noise vector and to the
discriminator input. That is the idea of Mirza and Osindero (2014) applied to
tabular data, not the architecture of Xu et al. (NeurIPS 2019), and describing
it as CTGAN invites a reviewer to compare it against a method it does not
implement.

This module implements the published algorithm. The three components that
distinguish CTGAN from a plain conditional GAN are all present:

1. Mode-specific normalisation. Each continuous column is fitted with a
   variational Gaussian mixture. A value is then represented by which mode it
   belongs to (one-hot) plus its normalised offset within that mode (scalar).
   This is what lets CTGAN reproduce multimodal and heavily skewed columns such
   as Bilirubin and Alkaline Phosphatase, which a single sigmoid output on
   min-max scaled data cannot.

2. A conditional vector with training-by-sampling. Rather than always
   conditioning on the outcome, training picks a discrete column at random and
   then a category from it with probability proportional to the logarithm of its
   frequency. Real samples are drawn to match. Log-frequency sampling is what
   gives rare categories enough exposure without distorting the marginal.

3. A PacGAN critic trained with the WGAN gradient penalty. The critic sees
   samples in packs of `pac`, which removes the incentive for the generator to
   collapse onto a single mode, and the gradient penalty replaces the original
   GAN loss that made the previous implementation unstable.

Input convention
----------------
Unlike the other generators in this project, CTGAN is given data in ORIGINAL
clinical units, not min-max scaled. Mode-specific normalisation performs its own
scaling, and applying a min-max transform first would destroy the mode structure
it is designed to find.

Reference
---------
L. Xu, M. Skoularidou, A. Cuesta-Infante and K. Veeramachaneni, "Modeling
Tabular Data using Conditional GAN," NeurIPS 2019.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.mixture import BayesianGaussianMixture


# ---------------------------------------------------------------
# 1. Mode-specific normalisation
# ---------------------------------------------------------------
class DataTransformer:
    """
    Encode a mixed continuous/discrete table into CTGAN's representation.

    A continuous column becomes 1 + m columns: a scalar offset in [-1, 1] and a
    one-hot indicator over the m retained mixture modes. A discrete column
    becomes a one-hot over its categories.
    """

    def __init__(self, max_modes=10, weight_threshold=0.005, n_std=4.0):
        self.max_modes = max_modes
        self.weight_threshold = weight_threshold
        self.n_std = n_std
        self.info = []          # per-column metadata
        self.output_dim = 0
        self.cont_idx = []
        self.disc_idx = []

    def fit(self, X, discrete_columns):
        """X is (n, d) in original units; discrete_columns is a list of column indices."""
        self.info, self.output_dim = [], 0
        n_cols = X.shape[1]
        for c in range(n_cols):
            col = X[:, c]
            if c in discrete_columns:
                cats = np.unique(col)
                self.info.append({"type": "discrete", "categories": cats,
                                  "dim": len(cats), "start": self.output_dim})
                self.output_dim += len(cats)
                self.disc_idx.append(c)
            else:
                gm = BayesianGaussianMixture(
                    n_components=self.max_modes,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=1e-3,
                    max_iter=200, n_init=1, random_state=42,
                )
                gm.fit(col.reshape(-1, 1))
                keep = gm.weights_ > self.weight_threshold
                n_modes = max(int(keep.sum()), 1)
                self.info.append({
                    "type": "continuous", "gm": gm, "keep": keep,
                    "n_modes": n_modes, "dim": 1 + n_modes, "start": self.output_dim,
                })
                self.output_dim += 1 + n_modes
                self.cont_idx.append(c)
        return self

    def transform(self, X):
        out = []
        for c, meta in enumerate(self.info):
            col = X[:, c]
            if meta["type"] == "discrete":
                oh = np.zeros((len(col), meta["dim"]), dtype="float32")
                for j, cat in enumerate(meta["categories"]):
                    oh[col == cat, j] = 1.0
                out.append(oh)
            else:
                gm, keep = meta["gm"], meta["keep"]
                means = gm.means_.reshape(-1)[keep]
                stds = np.sqrt(gm.covariances_.reshape(-1))[keep]
                probs = gm.predict_proba(col.reshape(-1, 1))[:, keep].astype("float64")
                probs = np.clip(probs, 1e-12, None)
                probs /= probs.sum(axis=1, keepdims=True)

                # Sample the mode rather than taking the argmax, as the paper
                # does. Inverse-CDF sampling avoids the float tolerance that
                # np.random.choice enforces on the probability vector.
                rng = np.random.RandomState(42)
                cdf = np.cumsum(probs, axis=1)
                u = rng.random_sample((len(col), 1))
                modes = (u > cdf).sum(axis=1)
                modes = np.clip(modes, 0, len(means) - 1)
                alpha = (col - means[modes]) / (self.n_std * stds[modes] + 1e-8)
                alpha = np.clip(alpha, -1.0, 1.0).astype("float32")

                oh = np.zeros((len(col), len(means)), dtype="float32")
                oh[np.arange(len(col)), modes] = 1.0
                out.append(np.concatenate([alpha.reshape(-1, 1), oh], axis=1))
        return np.concatenate(out, axis=1).astype("float32")

    def inverse_transform(self, Z):
        cols = []
        for meta in self.info:
            s, d = meta["start"], meta["dim"]
            block = Z[:, s:s + d]
            if meta["type"] == "discrete":
                idx = block.argmax(axis=1)
                cols.append(meta["categories"][idx].astype("float64"))
            else:
                gm, keep = meta["gm"], meta["keep"]
                means = gm.means_.reshape(-1)[keep]
                stds = np.sqrt(gm.covariances_.reshape(-1))[keep]
                alpha = np.clip(block[:, 0], -1.0, 1.0)
                modes = block[:, 1:].argmax(axis=1)
                cols.append(alpha * self.n_std * stds[modes] + means[modes])
        return np.stack(cols, axis=1)

    def discrete_spans(self):
        """(start, dim) of each discrete one-hot block in the encoded space."""
        return [(m["start"], m["dim"]) for m in self.info if m["type"] == "discrete"]

    def mode_spans(self):
        """(start_of_onehot, n_modes) for each continuous column's mode indicator."""
        return [(m["start"] + 1, m["n_modes"]) for m in self.info if m["type"] == "continuous"]


# ---------------------------------------------------------------
# 2. Conditional vector with training-by-sampling
# ---------------------------------------------------------------
class ConditionalSampler:
    """
    Build CTGAN's conditional vector and supply matching real rows.

    A discrete column is chosen uniformly, then a category within it is chosen
    with probability proportional to log(count). Real training rows are then
    drawn from those that actually carry that category, which is the
    training-by-sampling step.
    """

    def __init__(self, encoded, spans, seed=42):
        self.spans = spans
        self.cond_dim = sum(d for _, d in spans)
        self.rng = np.random.RandomState(seed)

        self.category_probs, self.true_probs, self.row_index = [], [], []
        for start, dim in spans:
            block = encoded[:, start:start + dim]
            counts = block.sum(axis=0)

            # Training uses log-frequency so that rare categories are seen often
            # enough for the generator to learn them.
            logf = np.log(counts + 1.0)
            self.category_probs.append(logf / logf.sum())

            # Generation uses the true frequency, otherwise the log-frequency
            # over-sampling of rare categories would distort the marginals of
            # the synthetic set (Xu et al., Section 4.3).
            self.true_probs.append(counts / counts.sum())

            self.row_index.append([np.where(block[:, j] > 0.5)[0] for j in range(dim)])

    def sample(self, batch_size, training=True):
        """Draw a conditional vector. training=True uses log-frequency sampling."""
        probs = self.category_probs if training else self.true_probs
        cond = np.zeros((batch_size, self.cond_dim), dtype="float32")
        col_choice = self.rng.randint(0, len(self.spans), batch_size)
        cat_choice = np.zeros(batch_size, dtype=int)
        offsets = np.cumsum([0] + [d for _, d in self.spans])

        for i, c in enumerate(col_choice):
            j = self.rng.choice(len(probs[c]), p=probs[c])
            cat_choice[i] = j
            cond[i, offsets[c] + j] = 1.0
        return cond, col_choice, cat_choice

    def matching_rows(self, col_choice, cat_choice):
        out = np.empty(len(col_choice), dtype=int)
        for i, (c, j) in enumerate(zip(col_choice, cat_choice)):
            pool = self.row_index[c][j]
            out[i] = self.rng.choice(pool) if len(pool) else self.rng.randint(0, 1)
        return out


# ---------------------------------------------------------------
# 3. Networks
# ---------------------------------------------------------------
class ResidualBlock(layers.Layer):
    """Linear, BatchNorm, ReLU, then concatenate the input (Xu et al. Fig. 2)."""

    def __init__(self, units):
        super().__init__()
        self.fc = layers.Dense(units)
        self.bn = layers.BatchNormalization()

    def call(self, x, training=False):
        h = tf.nn.relu(self.bn(self.fc(x), training=training))
        return tf.concat([h, x], axis=1)


def gumbel_softmax(logits, tau=0.2, seed=None):
    """Differentiable sampling from a categorical, used for one-hot outputs."""
    u = tf.random.uniform(tf.shape(logits), minval=1e-9, maxval=1.0, seed=seed)
    g = -tf.math.log(-tf.math.log(u))
    return tf.nn.softmax((logits + g) / tau, axis=-1)


class Generator(keras.Model):
    def __init__(self, out_dim, disc_spans, mode_spans, hidden=(256, 256)):
        super().__init__()
        self.blocks = [ResidualBlock(h) for h in hidden]
        self.out = layers.Dense(out_dim)
        self.disc_spans = disc_spans
        self.mode_spans = mode_spans
        self.out_dim = out_dim

    def call(self, z_cond, training=False):
        h = z_cond
        for b in self.blocks:
            h = b(h, training=training)
        raw = self.out(h)

        # tanh on the scalar offsets, gumbel-softmax on every one-hot block.
        pieces = tf.unstack(raw, axis=1)
        activated = [tf.tanh(p) for p in pieces]
        for start, dim in list(self.disc_spans) + list(self.mode_spans):
            block = gumbel_softmax(raw[:, start:start + dim])
            for k in range(dim):
                activated[start + k] = block[:, k]
        return tf.stack(activated, axis=1)


class Critic(keras.Model):
    """PacGAN critic: scores `pac` records jointly to discourage mode collapse."""

    def __init__(self, pac=10, hidden=(256, 256)):
        super().__init__()
        self.pac = pac
        self.net = keras.Sequential()
        for h in hidden:
            self.net.add(layers.Dense(h))
            self.net.add(layers.LeakyReLU(0.2))
            self.net.add(layers.Dropout(0.5))
        self.net.add(layers.Dense(1))

    def call(self, x, training=False):
        packed = tf.reshape(x, (-1, x.shape[1] * self.pac))
        return self.net(packed, training=training)


# ---------------------------------------------------------------
# 4. The model
# ---------------------------------------------------------------
class CTGANProper:
    """
    CTGAN trained with WGAN-GP, a PacGAN critic and training-by-sampling.

    Call fit() with data in ORIGINAL units plus the indices of discrete columns.
    """

    def __init__(self, latent_dim=128, epochs=300, batch_size=100, lr=2e-4,
                 pac=10, gp_lambda=10.0, discriminator_steps=1, seed=42,
                 print_every=50):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size - (batch_size % pac)   # must divide by pac
        self.lr = lr
        self.pac = pac
        self.gp_lambda = gp_lambda
        self.d_steps = discriminator_steps
        self.seed = seed
        self.print_every = print_every
        self.g_losses, self.d_losses = [], []

    def _gradient_penalty(self, real, fake):
        eps = tf.random.uniform((tf.shape(real)[0] // self.pac, 1))
        eps = tf.repeat(eps, self.pac, axis=0)
        mixed = eps * real + (1 - eps) * fake
        with tf.GradientTape() as t:
            t.watch(mixed)
            score = self.C(mixed, training=True)
        grad = t.gradient(score, mixed)
        grad = tf.reshape(grad, (-1, grad.shape[1] * self.pac))
        norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1) + 1e-12)
        return tf.reduce_mean(tf.square(norm - 1.0))

    def _cond_loss(self, fake, cond, col_choice, cat_choice):
        """Cross-entropy pushing the generated record to carry the requested category."""
        offsets = np.cumsum([0] + [d for _, d in self.transformer.discrete_spans()])
        total, n = 0.0, 0
        for c, (start, dim) in enumerate(self.transformer.discrete_spans()):
            sel = np.where(col_choice == c)[0]
            if len(sel) == 0:
                continue
            logits = tf.gather(fake[:, start:start + dim], sel)
            labels = tf.constant(cat_choice[sel], dtype=tf.int32)
            total += tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
            n += len(sel)
        return total / max(n, 1)

    def fit(self, X_raw, discrete_columns, verbose=True):
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        self.transformer = DataTransformer().fit(X_raw, discrete_columns)
        encoded = self.transformer.transform(X_raw)
        self.sampler = ConditionalSampler(
            encoded, self.transformer.discrete_spans(), seed=self.seed)

        d_out = self.transformer.output_dim
        c_dim = self.sampler.cond_dim
        self.G = Generator(d_out, self.transformer.discrete_spans(),
                           self.transformer.mode_spans())
        self.C = Critic(pac=self.pac)

        g_opt = keras.optimizers.Adam(self.lr, beta_1=0.5, beta_2=0.9)
        c_opt = keras.optimizers.Adam(self.lr, beta_1=0.5, beta_2=0.9)

        n = len(encoded)
        steps = max(n // self.batch_size, 1)

        for epoch in range(self.epochs):
            gl = dl = 0.0
            for _ in range(steps):
                # ---- critic ----
                for _ in range(self.d_steps):
                    cond, col_c, cat_c = self.sampler.sample(self.batch_size)
                    rows = self.sampler.matching_rows(col_c, cat_c)
                    real = tf.constant(encoded[rows])
                    z = tf.random.normal((self.batch_size, self.latent_dim))
                    with tf.GradientTape() as tape:
                        fake = self.G(tf.concat([z, cond], axis=1), training=True)
                        real_in = tf.concat([real, cond], axis=1)
                        fake_in = tf.concat([fake, cond], axis=1)
                        loss_c = (tf.reduce_mean(self.C(fake_in, training=True))
                                  - tf.reduce_mean(self.C(real_in, training=True))
                                  + self.gp_lambda * self._gradient_penalty(real_in, fake_in))
                    c_opt.apply_gradients(
                        zip(tape.gradient(loss_c, self.C.trainable_variables),
                            self.C.trainable_variables))

                # ---- generator ----
                cond, col_c, cat_c = self.sampler.sample(self.batch_size)
                z = tf.random.normal((self.batch_size, self.latent_dim))
                with tf.GradientTape() as tape:
                    fake = self.G(tf.concat([z, cond], axis=1), training=True)
                    loss_g = (-tf.reduce_mean(self.C(tf.concat([fake, cond], axis=1), training=True))
                              + self._cond_loss(fake, cond, col_c, cat_c))
                g_opt.apply_gradients(
                    zip(tape.gradient(loss_g, self.G.trainable_variables),
                        self.G.trainable_variables))

                gl += float(loss_g); dl += float(loss_c)

            self.g_losses.append(gl / steps)
            self.d_losses.append(dl / steps)
            if verbose and (epoch + 1) % self.print_every == 0:
                print(f"    epoch {epoch+1:>4}/{self.epochs}  "
                      f"critic {dl/steps:+.4f}  generator {gl/steps:+.4f}")
        return self

    def generate(self, n_samples):
        """Return n_samples rows in ORIGINAL units."""
        out, made = [], 0
        while made < n_samples:
            b = min(self.batch_size, n_samples - made)
            b = max(b, self.pac)
            cond, _, _ = self.sampler.sample(b, training=False)
            z = tf.random.normal((b, self.latent_dim))
            fake = self.G(tf.concat([z, cond], axis=1), training=False).numpy()
            out.append(fake)
            made += b
        return self.transformer.inverse_transform(np.concatenate(out)[:n_samples])
