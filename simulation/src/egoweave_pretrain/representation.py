"""The public 153-D structured representation used by final M4.

The input packet stores world-frame camera/head and hand geometry.  The output
is intentionally small and explicit: a 156-D current state and a 153-D future
action chunk.  Missing camera/hand labels are represented by finite zero
placeholders plus masks; they are never silently imputed as valid geometry.
"""

from numbers import Integral

import numpy as np


REPRESENTATION_VERSION = "m4_153d_camera_head_progress_local21_v1"
_SCALE_EPS = 1e-6


def _real_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values")
    return array


def _float32_checked(value, name):
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must remain finite in float32")
    return result


def _transforms(value, name, shape):
    array = _real_array(value, name)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    array = array.astype(np.float64, copy=False)
    if not np.allclose(array[..., 3, :], [0, 0, 0, 1], atol=1e-5, rtol=0):
        raise ValueError(f"{name} contains invalid homogeneous rows")
    rotation = array[..., :3, :3]
    gram = np.swapaxes(rotation, -1, -2) @ rotation
    if (not np.allclose(gram, np.eye(3), atol=1e-3, rtol=0)
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-3, rtol=0)):
        raise ValueError(f"{name} contains non-SO(3) rotations")
    return array


def pack_pose(camera, left_hand, right_hand, landmarks):
    """Pack validated geometry into an owned float32 ``[T,162]`` array.

    ``landmarks`` is world XYZ with shape ``[T,2,21,3]``.  Local hand points
    are stored in their wrist frames; columns are camera R/P, wrist P/R, and
    the two local21 arrays.  This is the only place that turns raw transforms
    into the cached packet representation.
    """
    camera = _real_array(camera, "camera")
    if camera.ndim != 3 or camera.shape[1:] != (4, 4) or len(camera) == 0:
        raise ValueError("camera must have nonempty shape [T,4,4]")
    count = len(camera)
    camera = _transforms(camera, "camera", (count, 4, 4))
    left = _transforms(left_hand, "left_hand", (count, 4, 4))
    right = _transforms(right_hand, "right_hand", (count, 4, 4))
    landmarks = _real_array(landmarks, "landmarks")
    if landmarks.shape != (count, 2, 21, 3) or not np.isfinite(landmarks).all():
        raise ValueError("landmarks must be finite world XYZ [T,2,21,3]")
    left_local = (landmarks[:, 0] - left[:, None, :3, 3]) @ left[:, :3, :3]
    right_local = (landmarks[:, 1] - right[:, None, :3, 3]) @ right[:, :3, :3]
    pose = np.concatenate([
        camera[:, :3, :3].reshape(count, 9), camera[:, :3, 3],
        left[:, :3, 3], right[:, :3, 3],
        left[:, :3, :3].reshape(count, 9), right[:, :3, :3].reshape(count, 9),
        left_local.reshape(count, 63), right_local.reshape(count, 63),
    ], axis=-1)
    return _float32_checked(pose, "packed pose")


def _positive_horizon(horizon):
    if isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, Integral):
        raise ValueError("horizon must be a positive integer")
    if horizon < 1:
        raise ValueError("horizon must be a positive integer")
    return int(horizon)


def _rot6(rotation):
    return np.concatenate([rotation[..., :, 0], rotation[..., :, 1]], axis=-1)


def _pose_array(pose):
    pose = _real_array(pose, "pose")
    if pose.ndim != 2 or pose.shape[1] != 162:
        raise ValueError("pose must have shape [T,162]")
    if not np.isfinite(pose).all():
        raise ValueError("pose contains non-finite values")
    return pose.astype(np.float32, copy=False)


def _local_scale(local):
    local = np.asarray(local, dtype=np.float32)
    centered = local - local[..., :1, :]
    distance = np.linalg.norm(centered, axis=-1)
    middle = distance[..., 9]
    valid = distance > _SCALE_EPS
    sums = np.sum(np.where(valid, distance, 0), axis=-1, dtype=np.float32)
    counts = np.sum(valid, axis=-1)
    fallback = np.ones_like(sums, dtype=np.float32)
    np.divide(sums, counts, out=fallback, where=counts > 0)
    fallback = np.maximum(fallback, _SCALE_EPS)
    return np.where(middle > _SCALE_EPS, middle, fallback).astype(np.float32)


def _frame_features(rows):
    leading = rows.shape[:-1]
    head_rotation = rows[..., :9].reshape(leading + (3, 3))
    head_inverse = np.swapaxes(head_rotation, -1, -2)
    head_position = rows[..., 9:12]
    hands = []
    for position_start, rotation_start, local_start in ((12, 18, 36), (15, 27, 99)):
        position = rows[..., position_start:position_start + 3]
        rotation = rows[..., rotation_start:rotation_start + 9].reshape(leading + (3, 3))
        wrist_position = ((position - head_position)[..., None, :] @ head_rotation)[..., 0, :]
        wrist = np.concatenate([wrist_position, _rot6(head_inverse @ rotation)], axis=-1)
        local = rows[..., local_start:local_start + 63].reshape(leading + (21, 3))
        normalized = (local - local[..., :1, :]) / _local_scale(local)[..., None, None]
        hands.append(np.concatenate([wrist, normalized.reshape(leading + (63,))], axis=-1))
    return np.concatenate([head_position, _rot6(head_rotation), *hands], axis=-1)


def encode_many(pose, anchors, horizon=32):
    """Encode valid anchors into ``states [N,156]`` and ``actions [N,H,153]``."""
    horizon = _positive_horizon(horizon)
    pose = _pose_array(pose)
    anchors = np.asarray(anchors)
    if anchors.ndim != 1 or anchors.dtype.kind not in "iu":
        raise ValueError("anchors must be a 1-D integer array")
    if anchors.size and (horizon >= len(pose) or np.any(anchors < 0)
                         or np.any(anchors >= len(pose) - horizon)):
        raise ValueError("Action chunk crosses episode boundary")
    if not len(anchors):
        return (np.empty((0, 156), dtype=np.float32),
                np.empty((0, horizon, 153), dtype=np.float32))
    anchors = anchors.astype(np.intp, copy=False)
    count = len(anchors)
    current = pose[anchors]
    future = pose[anchors[:, None] + np.arange(1, horizon + 1)]
    state_geometry = _frame_features(current)
    progress = anchors.astype(np.float64) / max(len(pose) - 1, 1)
    angle = 2 * np.pi * progress
    progress_features = np.stack([progress, np.sin(angle), np.cos(angle)], axis=-1).astype(np.float32)
    states = np.concatenate([state_geometry, progress_features], axis=-1)
    actions = _frame_features(future)
    actions[..., :3] = future[..., 9:12] - current[:, None, 9:12]
    current_head = current[:, :9].reshape(count, 3, 3)
    future_head = future[..., :9].reshape(count, horizon, 3, 3)
    actions[..., 3:9] = _rot6(np.swapaxes(current_head, -1, -2)[:, None] @ future_head)
    return (_float32_checked(states, "states"), _float32_checked(actions, "actions"))


def _channel_masks(camera_valid, hand_valid, anchors, horizon):
    """Build per-channel masks without turning missing labels into supervision."""
    camera_valid = np.asarray(camera_valid, dtype=bool)
    hand_valid = np.asarray(hand_valid, dtype=bool)
    anchors = np.asarray(anchors, dtype=np.intp)
    current_camera = camera_valid[anchors]
    current_hands = hand_valid[anchors]
    future_camera = camera_valid[anchors[:, None] + np.arange(1, horizon + 1)]
    future_hands = hand_valid[anchors[:, None] + np.arange(1, horizon + 1)]
    state = np.zeros((len(anchors), 156), dtype=bool)
    state[:, :9] = current_camera[:, None]
    state[:, 153:] = True
    for side, start in enumerate((9, 81)):
        state[:, start:start + 72] = current_camera[:, None] & current_hands[:, side, None]
    action = np.zeros((len(anchors), horizon, 153), dtype=bool)
    action[:, :, :9] = (current_camera[:, None] & future_camera)[:, :, None]
    for side, start in enumerate((9, 81)):
        hand = current_camera[:, None] & future_camera & future_hands[:, :, side]
        action[:, :, start:start + 72] = hand[:, :, None]
    return state, action


def encode_many_masked(packet, anchors, horizon=32):
    """Encode a packet and return geometry plus ``state_valid``/``reliability``."""
    horizon = _positive_horizon(horizon)
    anchors = np.asarray(anchors)
    if anchors.ndim != 1 or anchors.dtype.kind not in "iu":
        raise ValueError("anchors must be a 1-D integer array")
    pose = _pose_array(packet["pose153"])
    camera = np.asarray(packet["camera_valid"], dtype=bool)
    heads = np.asarray(packet.get("head_valid", camera), dtype=bool)
    hands = np.asarray(packet["hand_valid"], dtype=bool)
    if len(camera) != len(pose) or heads.shape != camera.shape or hands.shape != (len(pose), 2):
        raise ValueError("packet validity arrays do not match pose153")
    if not np.array_equal(heads, camera):
        # The representation currently uses the camera transform as its head
        # proxy; accepting a stricter head mask is useful for partial packets.
        camera = camera & heads
    states, actions = encode_many(pose, anchors, horizon)
    state_valid, action_valid = _channel_masks(camera, hands, anchors, horizon)
    states = np.where(state_valid, states, 0).astype(np.float32)
    actions = np.where(action_valid, actions, 0).astype(np.float32)
    return states, actions, state_valid, action_valid.astype(np.float32)


def encode(pose, anchor, horizon=32):
    if isinstance(anchor, (bool, np.bool_)) or not isinstance(anchor, Integral):
        raise ValueError("anchor must be an integer")
    states, actions = encode_many(pose, np.asarray([anchor]), horizon=horizon)
    return states[0], actions[0]
