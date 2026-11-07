from pathlib import Path
import jax
import numpy as np
from openpi.models.pi0_config import Pi0Config
cfg=Pi0Config(pi05=True,action_dim=153,state_dim=156,action_horizon=32,max_token_len=768,image_keys=('base_0_rgb',),preserve_image_token_layout=True)
@jax.jit
def interface(rng):
    model=cfg.create(rng)
    return model.action_in_proj.kernel.value.T,model.action_out_proj.kernel.value.T,model.action_out_proj.bias.value
_, init_rng=jax.random.split(jax.random.key(42))
_, model_rng=jax.random.split(init_rng)
a,b,c=interface(model_rng)
out=Path(__file__).parent/'base32'/'interface-seed42.npz'
np.savez(out,action_in=np.asarray(a),action_out=np.asarray(b),action_bias=np.asarray(c))
print('MATCHED_INTERFACE',out,a.shape,b.shape,c.shape,flush=True)
