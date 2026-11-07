"""A single complete M1 epoch with deterministic minimal batch padding.

No examples are dropped. First visit every example in a seeded permutation,
then repeat only the prefix needed to fill the final global batch. The same
seed, dataset order, and NumPy permutation implementation define the schedule;
persist these along with the manifest and start_step for resumable training.
"""

from itertools import islice
from numbers import Integral

import numpy as np


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def epoch_plan(n, batch=256, checkpoints=5):
    """Return JSON-safe epoch counts and zero-based stored checkpoint labels.

    ``total_steps`` is ceil(n/batch); ``padding_examples`` is the minimum
    number of repeats to make all global batches full. Checkpoint labels are
    floor(total_steps*i/k)-1, i=1..k, where k=min(checkpoints,total_steps),
    so short runs never schedule duplicate labels and the final step is saved.
    """
    n = _integer(n, "n", 1)
    batch = _integer(batch, "batch", 1)
    checkpoints = _integer(checkpoints, "checkpoints", 1)
    total_steps = (n + batch - 1) // batch
    count = min(checkpoints, total_steps)
    return {
        "samples": n,
        "batch_size": batch,
        "total_steps": total_steps,
        "padding_examples": total_steps * batch - n,
        "checkpoint_steps": [total_steps * i // count - 1 for i in range(1, count + 1)],
    }


class EpochSampler:
    """Torch-independent map-style sampler for exactly one deterministic epoch.

    Each iteration uses ``default_rng(seed).permutation(n)`` and appends a
    cyclic prefix only for the minimum required padding (also when n<batch).
    Reiterating deliberately reproduces the same epoch, not a new shuffle.

    ``start_step`` counts completed GLOBAL batches. Resuming yields the exact
    sequence suffix at start_step*batch; no preceding examples are decoded or
    yielded. Reconstructing the permutation needs O(n) CPU indices but no
    tensors/GPU state or dataset access. start_step==total_steps is empty.
    """

    def __init__(self, n, batch, seed, start_step=0):
        self.n = _integer(n, "n", 1)
        self.batch = _integer(batch, "batch", 1)
        self.seed = _integer(seed, "seed", 0)
        self.start_step = _integer(start_step, "start_step", 0)
        self.total_steps = (self.n + self.batch - 1) // self.batch
        if self.start_step > self.total_steps:
            raise ValueError("start_step exceeds the complete epoch")

    def __len__(self):
        return (self.total_steps - self.start_step) * self.batch

    def __iter__(self):
        if self.start_step == self.total_steps:
            return
        permutation = np.random.default_rng(self.seed).permutation(self.n)
        offset = self.start_step * self.batch
        for index in permutation[min(offset, self.n):]:
            yield int(index)
        total_examples = self.total_steps * self.batch
        for position in range(max(offset, self.n), total_examples):
            yield int(permutation[(position - self.n) % self.n])


def contiguous_rank_samples(sampler, rank, world_size):
    """Split each global batch into consecutive, equal rank-local batches."""
    if not 0 <= rank < world_size or sampler.batch % world_size:
        raise ValueError("Expected an even batch split and a valid rank")
    rank_size = sampler.batch // world_size
    start = rank * rank_size
    indices = iter(sampler)
    for _ in range(len(sampler) // sampler.batch):
        batch = list(islice(indices, sampler.batch))
        yield from batch[start:start + rank_size]
