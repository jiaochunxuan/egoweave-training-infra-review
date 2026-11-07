"""Public M4 structured-pretraining adapters.

The package contains the representation, masking, normalization, and manifest
contracts used before robot-native fine-tuning.  It deliberately does not ship
human-motion data, checkpoints, or a simulator.
"""

from .recipe import M4_RECIPE
from .representation import REPRESENTATION_VERSION, encode_many_masked, pack_pose

__all__ = [
    "M4_RECIPE",
    "REPRESENTATION_VERSION",
    "encode_many_masked",
    "pack_pose",
]
