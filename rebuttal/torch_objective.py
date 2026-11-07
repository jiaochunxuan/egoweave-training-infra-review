"""PyTorch adapter for the matched structured-motion objective."""
import torch

from targets import geometric_loss


class TorchGeometry:
    """Array operations shared with the reference target decoder."""

    sqrt = staticmethod(torch.sqrt)
    maximum = staticmethod(torch.clamp_min)
    sum = staticmethod(torch.sum)
    stack = staticmethod(torch.stack)
    einsum = staticmethod(torch.einsum)
    mean = staticmethod(torch.mean)
    square = staticmethod(torch.square)
    all = staticmethod(torch.all)
    where = staticmethod(torch.where)

    @staticmethod
    def cross(left, right):
        return torch.linalg.cross(left, right, dim=-1)


def losses(velocity, state, target, noise, time, reliability, state_mean, state_std,
           action_mean, action_std, condition):
    """Return unweighted [batch, horizon] flow and geometric loss maps."""
    velocity = velocity.float()
    error = noise - target - velocity
    weights = error.new_tensor([3.] * 3 + [1.5] * 6 + ([20.] * 3 + [4.] * 6) * 2 + [.3] * 126)
    full_weight = reliability * weights
    active_weight = full_weight
    if condition == 'no_explicit_head':
        active_weight = full_weight.clone()
        active_weight[..., :9] = 0
    flow = (error.square() * active_weight).sum(-1) / full_weight.sum(-1).clamp_min(1e-6)
    pred = time * noise + (1 - time) * target - time * velocity
    raw_state = state * (state_std + 1e-6) + state_mean
    raw_pred = pred * (action_std + 1e-6) + action_mean
    raw_target = target * (action_std + 1e-6) + action_mean
    geom = geometric_loss(raw_state, raw_pred, raw_target, reliability, condition, TorchGeometry)
    return flow, geom
