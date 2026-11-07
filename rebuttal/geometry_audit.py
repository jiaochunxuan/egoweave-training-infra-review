"""Execute actual frozen geometry methods without loading the billion-param model."""
import ast,json,sys
import numpy as np
from protocol import ROOT,OUT,sha,write
from targets import encode,decode,geometric_loss,action_weights,head_ablation_flow

def main():
    import os
    os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
    import jax,jax.numpy as jnp
    source=ROOT/'experiments/m3_m4_partial_epoch1_20260915/openpi_snapshot/src/openpi/models/pi0.py'
    tree=ast.parse(source.read_text());wanted={'_quest_local21_reconstruction_loss','_quest_local21_side_reconstruction_loss','_rotation_6d_to_matrix'}
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Pi0')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    if len(methods)!=3:raise ValueError('Frozen geometry API changed')
    for f in methods:
        f.returns=None
        for arg in [*f.args.posonlyargs,*f.args.args,*f.args.kwonlyargs]:arg.annotation=None
    module=ast.Module(body=[ast.ClassDef(name='ActualFrozenGeometry',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
    scope={'jnp':jnp};exec(compile(ast.fix_missing_locations(module),str(source),'exec'),scope)
    reducer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reduce_flow_matching_error')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[reducer],type_ignores=[])),str(source),'exec'),scope)
    reduce=scope['reduce_flow_matching_error'];error=jnp.ones((2,32,153));reliability=jnp.ones_like(error)
    full_grad=jax.grad(lambda x:reduce(x,reliability=reliability,action_loss_weights=action_weights('full')).sum())(error)
    ablated_grad=jax.grad(lambda x:head_ablation_flow(x,reliability,reduce,jnp).sum())(error)
    np.testing.assert_array_equal(ablated_grad[...,:9],0)
    np.testing.assert_allclose(ablated_grad[...,9:],full_grad[...,9:],atol=1e-7)
    model=scope['ActualFrozenGeometry']()
    for k,v in dict(quest_local21_recon_state_mean=np.zeros(156),quest_local21_recon_state_std=np.ones(156)-1e-6,
        quest_local21_recon_action_mean=np.zeros(153),quest_local21_recon_action_std=np.ones(153)-1e-6,
        quest_local21_recon_left_scale=.09564778953790665,quest_local21_recon_right_scale=.09564778953790665,
        quest_local21_recon_left_weight=1.,quest_local21_recon_right_weight=1.,quest_local21_recon_wrist_pose_relative_to_head=True).items():setattr(model,k,v)
    sys.path.insert(0,str(ROOT/'experiments/egodex_official48_epoch1_20260914'))
    sys.path.insert(0,str(ROOT/'experiments/egodex_official153_epoch1_20260914'))
    from m2.representation import encode_many
    from tests.test_targets import packet
    p=packet();s,a,_,mask=encode(p,np.array([0,3]),'full');ss,aa=encode_many(p['pose153'],np.array([0,3]),32)
    np.testing.assert_allclose(s,ss,atol=1e-6);np.testing.assert_allclose(a,aa,atol=1e-6)
    pred=a.copy();pred[...,:3]+=[.02,-.01,.03];pred[...,3:9]+=[.03,0,.01,0,.02,-.01]
    s,pr,tg,rel=map(jnp.asarray,[s,pred,a,mask.astype(np.float32)])
    actual=model._quest_local21_reconstruction_loss(s,pr,tg,rel)
    expected=geometric_loss(s,pr,tg,rel,'full',jnp)
    np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=1e-8)
    grad=jax.grad(lambda q:model._quest_local21_reconstruction_loss(s,q,tg,rel).mean())(pr)
    assert np.isfinite(grad).all() and np.linalg.norm(grad[...,:9])>0
    # Current-head coordinate geometry must not propagate through future head predictions.
    cs,ca,_,cm=encode(p,np.array([0,3]),'current_head');cp=ca.copy();cp[...,9:12]+=.02
    cg=jax.grad(lambda q:geometric_loss(jnp.asarray(cs),q,jnp.asarray(ca),jnp.asarray(cm),'current_head',jnp).mean())(jnp.asarray(cp))
    np.testing.assert_array_equal(cg[...,:9],0)
    zero=jax.grad(lambda q:geometric_loss(jnp.zeros((1,156)),q,jnp.zeros((1,32,153)),jnp.zeros((1,32,153)),'full',jnp).mean())(jnp.zeros((1,32,153)))
    assert np.isfinite(zero).all()
    report=dict(passed=True,frozen_model_sha256=sha(source),actual_future_head_rotation_present=True,
        full_encoder_matches_frozen=True,full_geometry_matches_frozen=True,head_geometry_gradient_norm=float(np.linalg.norm(grad[...,:9])),
        current_head_future_head_gradient_zero=True,fully_missing_new_decoder_gradient_finite=True,
        explicit_head_flow_gradient_zero=True,nonhead_flow_gradient_scale_unchanged=True,
        scope='Actual numerical geometry methods and encoder; no full model training, hardware rollout or paper equation correction claimed.')
    write(OUT/'validation/geometry_audit.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
