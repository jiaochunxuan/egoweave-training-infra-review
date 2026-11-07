"""JAX implementation of the M4 local21 reconstruction objective.

This module is independent of the OpenPI model tree.  The public model overlay
can call :func:`local21_reconstruction_loss` from its flow-matching loss, which
keeps the geometry term auditable without publishing a second full OpenPI
checkout.
"""

from __future__ import annotations


def weighted_flow_matching_loss(error, action_loss_weights, reliability=None):
    """Reduce a flow-matching error with the final M4 channel weights.

    ``error`` is ``[B,H,153]``.  When a reliability tensor is supplied it is
    multiplied before reduction and each example is normalized by its valid
    weight, so unavailable hand channels do not become artificial zero labels.
    """
    import jax.numpy as jnp

    error = jnp.asarray(error)
    weights = jnp.asarray(action_loss_weights, dtype=error.dtype)
    if error.shape[-1] != weights.shape[0]:
        raise ValueError("action_loss_weights width does not match error")
    squared = jnp.square(error) * weights
    if reliability is None:
        return jnp.sum(squared, axis=-1) / jnp.maximum(jnp.sum(weights), 1e-6)
    reliability = jnp.asarray(reliability, dtype=error.dtype)
    if reliability.shape != error.shape:
        raise ValueError("reliability shape does not match flow error")
    weighted = reliability * weights
    return jnp.sum(squared * reliability, axis=-1) / jnp.maximum(jnp.sum(weighted, axis=-1), 1e-6)


def local21_reconstruction_loss(
    state,
    predicted_actions,
    target_actions,
    state_mean,
    state_std,
    action_mean,
    action_std,
    *,
    left_scale=0.09564778953790665,
    right_scale=0.09564778953790665,
    left_weight=1.0,
    right_weight=1.0,
    hand_valid=None,
):
    """Return per-example/per-horizon local21 world-space MSE.

    Inputs are normalized M4 tensors with shapes ``[B,156]`` and
    ``[B,H,153]``.  ``hand_valid`` may be ``[B,H,2]``; when supplied, missing
    sides are excluded from the geometry average rather than contributing zero
    targets.  The unmasked default matches the complete-window training path.
    """
    import jax.numpy as jnp

    state = jnp.asarray(state)
    predicted_actions = jnp.asarray(predicted_actions)
    target_actions = jnp.asarray(target_actions)
    raw_state = state * (jnp.asarray(state_std) + 1e-6) + jnp.asarray(state_mean)
    raw_pred = predicted_actions * (jnp.asarray(action_std) + 1e-6) + jnp.asarray(action_mean)
    raw_target = target_actions * (jnp.asarray(action_std) + 1e-6) + jnp.asarray(action_mean)
    raw_state = raw_state[:, : raw_pred.shape[-1]]
    current_head_pos = raw_state[:, 0:3]
    current_head_rot = _rotation_6d_to_matrix(raw_state[:, 3:9])

    def side(position_slice, rotation_slice, local_slice, scale):
        current_wrist_pos = raw_state[:, position_slice]
        current_wrist_rot = _rotation_6d_to_matrix(raw_state[:, rotation_slice])
        pred_head_pos = current_head_pos[:, None, :] + raw_pred[:, :, 0:3]
        target_head_pos = current_head_pos[:, None, :] + raw_target[:, :, 0:3]
        pred_head_rel = _rotation_6d_to_matrix(raw_pred[:, :, 3:9])
        target_head_rel = _rotation_6d_to_matrix(raw_target[:, :, 3:9])
        pred_head_rot = jnp.einsum("bij,bhjk->bhik", current_head_rot, pred_head_rel)
        target_head_rot = jnp.einsum("bij,bhjk->bhik", current_head_rot, target_head_rel)
        pred_pos = pred_head_pos + jnp.einsum("bhij,bhj->bhi", pred_head_rot, raw_pred[:, :, position_slice])
        target_pos = target_head_pos + jnp.einsum("bhij,bhj->bhi", target_head_rot, raw_target[:, :, position_slice])
        pred_rot = jnp.einsum("bhij,bhjk->bhik", pred_head_rot,
                              _rotation_6d_to_matrix(raw_pred[:, :, rotation_slice]))
        target_rot = jnp.einsum("bhij,bhjk->bhik", target_head_rot,
                                _rotation_6d_to_matrix(raw_target[:, :, rotation_slice]))
        pred_local = raw_pred[:, :, local_slice].reshape(raw_pred.shape[0], raw_pred.shape[1], 21, 3) * scale
        target_local = raw_target[:, :, local_slice].reshape(raw_target.shape[0], raw_target.shape[1], 21, 3) * scale
        pred_world = pred_pos[:, :, None, :] + jnp.einsum("bhij,bhnj->bhni", pred_rot, pred_local)
        target_world = target_pos[:, :, None, :] + jnp.einsum("bhij,bhnj->bhni", target_rot, target_local)
        return jnp.mean(jnp.square(pred_world - target_world), axis=(-1, -2))

    left = side(slice(9, 12), slice(12, 18), slice(27, 90), left_scale)
    right = side(slice(18, 21), slice(21, 27), slice(90, 153), right_scale)
    if hand_valid is None:
        left_valid = right_valid = jnp.ones_like(left)
    else:
        valid = jnp.asarray(hand_valid, dtype=left.dtype)
        left_valid, right_valid = valid[..., 0], valid[..., 1]
    numerator = left_weight * left * left_valid + right_weight * right * right_valid
    denominator = left_weight * left_valid + right_weight * right_valid
    return numerator / jnp.maximum(denominator, 1e-6)


def _rotation_6d_to_matrix(rot_6d):
    import jax.numpy as jnp

    first = rot_6d[..., 0:3]
    second = rot_6d[..., 3:6]
    b1 = first / jnp.maximum(jnp.linalg.norm(first, axis=-1, keepdims=True), 1e-6)
    second = second - jnp.sum(second * b1, axis=-1, keepdims=True) * b1
    b2 = second / jnp.maximum(jnp.linalg.norm(second, axis=-1, keepdims=True), 1e-6)
    return jnp.stack([b1, b2, jnp.cross(b1, b2, axis=-1)], axis=-1)
