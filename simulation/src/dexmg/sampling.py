"""Deterministic frame permutations with direct optimizer-step resumption."""

from numbers import Integral

import numpy as np


class ResumableIndexSampler:
    """Yield frame indices for optimizer steps in ``[start_step, stop_step)``.

    Each epoch independently permutes the full dataset with
    ``default_rng(SeedSequence([seed, epoch]))``. Only complete batches are
    consumed; a new epoch reshuffles all frames, including the previous tail.
    Resumption derives the epoch and batch offset from the global step and
    never reads prior examples or constructs prior epochs' permutations.

    Iteration is repeatable, uses O(size) index memory, and does not touch
    NumPy's global RNG. Use with an explicit batch size, drop_last=True, and
    a finite loader batch budget of ``stop_step - start_step``. This class
    deliberately depends on neither Torch nor an accelerator runtime.
    """

    def __init__(self, size, batch_size, seed, start_step, stop_step):
        for name, value in (
            ("size", size), ("batch_size", batch_size), ("seed", seed),
            ("start_step", start_step), ("stop_step", stop_step),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer, got {value!r}")
            setattr(self, name, int(value))
        if self.batch_size <= 0 or self.size < self.batch_size:
            raise ValueError("size must contain at least one full positive batch")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.start_step < 0 or self.stop_step < self.start_step:
            raise ValueError("step range must satisfy 0 <= start_step <= stop_step")
        self.batches_per_epoch = self.size // self.batch_size

    def __len__(self):
        return (self.stop_step - self.start_step) * self.batch_size

    def __iter__(self):
        step = self.start_step
        while step < self.stop_step:
            epoch, batch_offset = divmod(step, self.batches_per_epoch)
            permutation = np.random.default_rng(np.random.SeedSequence([self.seed, epoch])).permutation(self.size)
            batches = min(self.batches_per_epoch - batch_offset, self.stop_step - step)
            begin = batch_offset * self.batch_size
            end = begin + batches * self.batch_size
            yield from (int(index) for index in permutation[begin:end])
            step += batches
