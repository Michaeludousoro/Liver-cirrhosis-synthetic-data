"""
Global Seeding for Reproducibility
==================================

Project
-------
Synthetic Data Generation and Predictive Utility Evaluation
for Primary Biliary Cirrhosis (PBC) — Mayo Clinic Dataset

Purpose of this module
-----------------------
The generative models are trained with TensorFlow, which draws random numbers
for weight initialisation, latent-noise sampling, mini-batch shuffling, and
dropout. Without a fixed seed, every run produces different synthetic data and
therefore different retention counts, FID scores, and downstream results.

Calling set_global_seeds() once at the start of a pipeline run fixes the Python,
NumPy, and TensorFlow random number generators and requests deterministic
TensorFlow ops, so the whole pipeline reproduces the same numbers on re-run.

Note
----
For strict determinism the environment variables TF_DETERMINISTIC_OPS and
PYTHONHASHSEED should ideally be set before the Python process starts. This
module sets them defensively, and also pins TensorFlow to single-threaded
execution, which removes the remaining source of floating-point nondeterminism
from parallel reductions on CPU.
"""

import os
import random

import numpy as np


def set_global_seeds(seed: int = 42, single_thread: bool = True):
    """
    Fix all random number generators used across the pipeline.

    Parameters
    ----------
    seed          : the integer seed applied to Python, NumPy, and TensorFlow
    single_thread : if True, pin TensorFlow to one intra-op and one inter-op
                    thread. This makes CPU reductions deterministic at a small
                    cost in speed, which is negligible for the small networks
                    and dataset used here.
    """
    os.environ["PYTHONHASHSEED"]      = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    if single_thread:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

    return seed
