"""Compare the Torch port with the frozen JAX objective on identical inputs."""
import numpy as np
import jax
import jax.numpy as jnp
import torch

from openpi.models.pi0 import Pi0, reduce_flow_matching_error
from targets import SCALE, action_weights, geometric_loss, head_ablation_flow
from torch_objective import losses


class FrozenGeometry:
    _rotation_6d_to_matrix = staticmethod(Pi0._rotation_6d_to_matrix)
    _quest_local21_side_reconstruction_loss = Pi0._quest_local21_side_reconstruction_loss
    quest_local21_recon_wrist_pose_relative_to_head = True
    quest_local21_recon_left_weight = quest_local21_recon_right_weight = 1.0


def main():
    rng = np.random.default_rng(42)
    state = rng.standard_normal((3, 156), dtype=np.float32)
    target, noise, velocity = rng.standard_normal((3, 3, 32, 153), dtype=np.float32)
    time = np.array([.2, .45, .9], np.float32)[:, None, None]
    mask = np.ones_like(target)
    mask[0, :8, :9] = 0
    mask[1, 4:9, 90:] = 0
    mask[2, :8] = 0
    means = rng.standard_normal((2, 153), dtype=np.float32) / 3
    stds = rng.uniform(.1, 1.2, means.shape).astype(np.float32)
    sm = np.pad(means[0], (0, 3))
    ss = np.pad(stds[0], (0, 3), constant_values=1)
    am, ast = means[1], stds[1]
    reference = FrozenGeometry()
    reference.quest_local21_recon_state_mean = sm
    reference.quest_local21_recon_state_std = ss
    reference.quest_local21_recon_action_mean = am
    reference.quest_local21_recon_action_std = ast
    tensors = tuple(map(torch.from_numpy, (state, target, noise, time, mask, sm, ss, am, ast)))
    for condition in ('full', 'no_explicit_head', 'current_head', 'no_scale'):
        reference.quest_local21_recon_left_scale = reference.quest_local21_recon_right_scale = (
            1.0 if condition == 'no_scale' else SCALE
        )

        def jax_loss(v):
            error = noise - target - v
            if condition == 'no_explicit_head':
                flow = head_ablation_flow(error, mask, reduce_flow_matching_error, jnp)
            else:
                flow = reduce_flow_matching_error(error, reliability=mask, action_loss_weights=action_weights(condition))
            pred = time * noise + (1 - time) * target - time * v
            if condition == 'current_head':
                geom = geometric_loss(state * (ss + 1e-6) + sm, pred * (ast + 1e-6) + am,
                                      target * (ast + 1e-6) + am, mask, condition, jnp)
            else:
                geom = Pi0._quest_local21_reconstruction_loss(reference, state, pred, target, mask)
            return (flow + 100 * geom).mean(), (flow, geom)

        (expected, (flow, geom)), grad = jax.value_and_grad(jax_loss, has_aux=True)(jnp.asarray(velocity))
        v = torch.tensor(velocity, requires_grad=True)
        tf, tg = losses(v, *tensors, condition)
        actual = (tf + 100 * tg).mean()
        actual.backward()
        np.testing.assert_allclose(tf.detach(), flow, rtol=2e-5, atol=1e-7)
        np.testing.assert_allclose(tg.detach(), geom, rtol=2e-5, atol=1e-7)
        np.testing.assert_allclose(actual.detach(), expected, rtol=2e-5, atol=1e-7)
        np.testing.assert_allclose(v.grad, grad, rtol=1e-3, atol=1e-6)
        print(condition, 'loss and velocity-gradient match', float(actual.detach()))


if __name__ == '__main__':
    main()
