"""Checkpoint host conversion that submits transfers for the complete tree."""


def parallel_host_snapshot(tree):
    """Preserve tree structure while converting array leaves to host NumPy.

    Whole-tree device_get lets JAX submit all leaves' asynchronous host copies
    before waiting for individual array conversions. Static metadata remains
    unchanged. Imports are lazy so importing this module never selects or
    initializes an accelerator backend. Already-host NumPy arrays may be
    reused, matching the original checkpoint snapshot's np.asarray behavior.
    """
    import jax
    import numpy as np

    host_tree = jax.device_get(tree)
    return jax.tree.map(lambda leaf: np.asarray(leaf) if hasattr(leaf, "shape") else leaf, host_tree)
