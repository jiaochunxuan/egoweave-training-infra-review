"""Pi0.5 tokenizer guard used by M4 pretraining.

The upstream tokenizer silently truncates an overlong state-conditioned prompt.
M4 treats that as a data/configuration error so the same token budget is used
for preflight and training.
"""

from __future__ import annotations

import numpy as np


def build_strict_tokenizer(max_length: int):
    """Construct the guard lazily so importing this package stays CPU-only."""
    from openpi.models.tokenizer import PaligemmaTokenizer

    class StrictPaligemmaTokenizer(PaligemmaTokenizer):
        def tokenize(self, prompt, state=None):
            cleaned = str(prompt).strip().replace("_", " ").replace("\n", " ")
            if state is not None:
                state = np.asarray(state)
                if not np.isfinite(state).all():
                    raise ValueError("non-finite state passed to tokenizer")
                values = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
                text = f"Task: {cleaned}, State: {' '.join(map(str, values))};\nAction: "
                tokens = self._tokenizer.encode(text, add_bos=True)
            else:
                tokens = self._tokenizer.encode(cleaned, add_bos=True)
                tokens += self._tokenizer.encode("\n")
            if len(tokens) > max_length:
                raise ValueError(
                    f"complete input requires {len(tokens)} tokens; max_token_len={max_length}"
                )
            result = np.zeros(max_length, dtype=np.int64)
            result[: len(tokens)] = tokens
            return result, np.arange(max_length) < len(tokens)

    return StrictPaligemmaTokenizer
