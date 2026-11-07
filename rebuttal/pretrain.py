"""Fresh pi0.5 initialization and matched structured-supervision ablations."""
import argparse,dataclasses,json,os,sys,time
from pathlib import Path
from protocol import ROOT,HERE,OUT,sha,write,configure_runtime,safe_name,load_module
from targets import CONDITIONS,SCALE,action_weights,geometric_loss,head_ablation_flow
from human_data import PacketDataset,MaskAfterNormalize,validate_view_pair

def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=CONDITIONS,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'data/m4_hybrid_smoothed_20260920/manifest.json')
    p.add_argument('--norm-dir',type=Path,required=True);p.add_argument('--selection',type=Path)
    p.add_argument('--paired-local',type=Path);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--tag',default='',help='Distinct annotation/cohort study identity, e.g. official_matched')
    p.add_argument('--workers',type=int,default=24);p.add_argument('--mode',choices=['describe','preflight','train'],default='describe')
    a=p.parse_args()
    if a.tag:safe_name(a.tag)
    run=safe_name(f'{a.condition}'+('_'+a.tag if a.tag else '')+f'_seed{a.seed}');out=OUT/'human'/run
    if a.seed<0:raise ValueError('Seed must be nonnegative')
    if a.mode=='describe':print(json.dumps(dict(condition=a.condition,seed=a.seed,manifest=str(a.manifest),norm_dir=str(a.norm_dir),fsdp=8,batch=256,horizon=32,output=str(out)),indent=2));return
    configure_runtime(a.mode=='preflight')
    if a.condition=='global_axis':
        if a.paired_local is None or a.selection is None:raise ValueError('Global control needs paired local manifest and matched selection')
        validate_view_pair(a.manifest,a.paired_local)
    snapshot=ROOT/'experiments/m3_m4_partial_epoch1_20260915/openpi_snapshot'
    m1=ROOT/'experiments/egodex_official48_epoch1_20260914';m2=ROOT/'experiments/egodex_official153_epoch1_20260914'
    sys.path[:0]=list(map(str,[m1,m2,ROOT/'code/lerobot',snapshot,snapshot/'src',snapshot/'packages/openpi-client/src']))
    os.chdir(m1)
    import numpy as np
    import jax,jax.numpy as jnp
    from flax import nnx
    import flax.traverse_util as traverse
    from openpi.models import pi0_config,pi0,model as model_lib
    from openpi.training import config as configs,data_loader as loaders,optimizer
    from openpi import transforms
    from openpi.shared import normalize
    from m1.protocol import epoch_plan,EpochSampler
    from m2.train import merge_base_leaves
    sys.path.append(str(ROOT/'experiments/m4_smoothed_retrain_20260920'))
    import strict_tokenizer as strict
    ds=PacketDataset(a.manifest,a.condition,a.selection)
    if a.mode=='train' and ds.manifest.get('smoke',False):raise ValueError('Sample cache cannot be used as a formal experiment')
    if ds.selection and not ds.selection['complete']:raise ValueError('Matched audit is incomplete')
    provenance=json.loads((a.norm_dir/'provenance.json').read_text())
    expected=dict(condition=a.condition,manifest_sha256=sha(a.manifest),selection_sha256=sha(a.selection) if a.selection else None,norm_sha256=sha(a.norm_dir/'norm_stats.json'))
    if any(provenance.get(k)!=v for k,v in expected.items()):raise ValueError('Normalization/condition/window provenance mismatch')
    if provenance['windows']!=len(ds):raise ValueError('Normalization window count mismatch')
    norms=normalize.load(a.norm_dir);plan=epoch_plan(len(ds),256,5);steps=plan['total_steps']
    scale=1. if a.condition=='no_scale' else SCALE
    model_cfg=pi0_config.Pi0Config(pi05=True,action_dim=153,state_dim=156,action_horizon=32,max_token_len=768,
        paligemma_variant='gemma_2b',action_expert_variant='gemma_300m',image_keys=('base_0_rgb',),preserve_image_token_layout=True,
        action_loss_weights=tuple(map(float,action_weights(a.condition))),flow_loss_type='mse',quest_local21_recon_loss_weight=0. if a.condition=='no_geometry' else 100.,
        quest_local21_recon_left_weight=1.,quest_local21_recon_right_weight=1.,quest_local21_recon_left_scale=scale,quest_local21_recon_right_scale=scale,
        quest_local21_recon_wrist_pose_relative_to_head=True,quest_local21_recon_wrist_position_absolute=False,quest_local21_recon_wrist_rotation_absolute=False,
        quest_local21_recon_state_mean=tuple(map(float,norms['state'].mean)),quest_local21_recon_state_std=tuple(map(float,norms['state'].std)),
        quest_local21_recon_action_mean=tuple(map(float,norms['actions'].mean)),quest_local21_recon_action_std=tuple(map(float,norms['actions'].std)))
    if a.condition=='current_head':
        def reconstruction(self,state,pred,target,reliability=None):
            raw_state=state*(jnp.asarray(self.quest_local21_recon_state_std)+1e-6)+jnp.asarray(self.quest_local21_recon_state_mean)
            std=jnp.asarray(self.quest_local21_recon_action_std)+1e-6;mean=jnp.asarray(self.quest_local21_recon_action_mean)
            rel=jnp.ones_like(target) if reliability is None else reliability
            return geometric_loss(raw_state,pred*std+mean,target*std+mean,rel,'current_head',jnp)
        pi0.Pi0._quest_local21_reconstruction_loss=reconstruction
    if a.condition=='no_explicit_head':
        original_reduce=pi0.reduce_flow_matching_error
        def reduce_without_head(error,*,loss_type='mse',reliability=None,action_loss_weights=None,**kwargs):
            if loss_type!='mse':raise ValueError('Explicit-head ablation is defined for the fixed MSE objective')
            return head_ablation_flow(error,reliability,original_reduce,jnp)
        pi0.reduce_flow_matching_error=reduce_without_head
    mt=configs.ModelTransformFactory()(model_cfg)
    mt=dataclasses.replace(mt,inputs=[dataclasses.replace(t,tokenizer=strict.StrictPaligemmaTokenizer(768)) if isinstance(t,transforms.TokenizePrompt) else t for t in mt.inputs])
    mt=dataclasses.replace(mt,inputs=[MaskAfterNormalize(),*mt.inputs[:-1],transforms.PadStatesAndActions(153,model_state_dim=156)])
    data_cfg=configs.DataConfig(repo_id='egoweave_rebuttal_cached',asset_id=a.condition,norm_stats=norms,use_quantile_norm=False,model_transforms=mt)
    transformed=loaders.transform_dataset(ds,data_cfg);examples=[]
    for i in [0,len(ds)//2,len(ds)-1]:
        x=transformed[i]
        if x['state'].shape!=(156,) or x['actions'].shape!=(32,153) or not np.isfinite(x['actions']).all():raise ValueError('Transformed data contract')
        if not np.all(x['actions'][x['reliability']==0]==0):raise ValueError('Unavailable labels leaked after normalization')
        examples.append(dict(index=i,valid_fraction=float(x['reliability'].mean())))
    contract=dict(**expected,seed=a.seed,sampler_seed=a.seed,plan=plan,batch=256,fsdp=8,output=str(out),examples=examples,
        objective='explicit_head_loss_removed_but_geometry_retains_head_gradient' if a.condition=='no_explicit_head' else a.condition,
        source_code_sha256={x.name:sha(x) for x in HERE.glob('*.py')},frozen_geometry_sha256=sha(snapshot/'src/openpi/models/pi0.py'))
    write(out/(a.mode+'.contract.json'),contract)
    if a.mode=='preflight':print('PREFLIGHT_PASS',json.dumps(contract),flush=True);return
    if jax.device_count()!=8 or jax.process_count()!=1:raise RuntimeError('Exactly eight GPUs in one process required')
    @dataclasses.dataclass(frozen=True)
    class Factory:
        def create(self,assets_dirs,model_config):return data_cfg
    @dataclasses.dataclass(frozen=True)
    class Initializer:
        def load(self,target):
            source=model_lib.restore_params(str(ROOT/'openpi_cache/checkpoints/pi05_base/params'),restore_type=np.ndarray)
            merged,audit=merge_base_leaves(traverse.flatten_dict(source,sep='/'),traverse.flatten_dict(target,sep='/'))
            if len(audit['random_interface'])!=3:raise ValueError('Unexpected reset leaves')
            write(out/'initialization.json',audit);return traverse.unflatten_dict(merged,sep='/')
    cfg=configs.TrainConfig(name='structured_motion',exp_name=run,project_name='egoweave_rebuttal',model=model_cfg,data=Factory(),weight_loader=Initializer(),
        assets_base_dir=str(out/'assets'),checkpoint_base_dir=str(out/'checkpoints'),batch_size=256,num_workers=a.workers,num_train_steps=steps,seed=a.seed,fsdp_devices=8,
        log_interval=20,save_interval=max(1,steps//5),checkpoints_per_run=5,keep_period=1,overwrite=False,resume=False,ema_decay=None,wandb_enabled=False,freeze_filter=nnx.Nothing,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=20,peak_lr=2.5e-5,decay_steps=steps,decay_lr=2.5e-5),optimizer=optimizer.AdamW())
    if cfg.checkpoint_dir.exists():raise FileExistsError(cfg.checkpoint_dir)
    def create_loader(config,*,sharding=None,shuffle=False,**kwargs):
        if not shuffle:raise ValueError('Training must request shuffled windows')
        sampler=EpochSampler(len(ds),batch=256,seed=a.seed)
        runtime=loaders.TorchDataLoader(transformed,256,sharding=sharding,sampler=sampler,shuffle=False,num_batches=steps,num_workers=a.workers,seed=a.seed)
        return loaders.DataLoaderImpl(data_cfg,runtime)
    loaders.create_data_loader=create_loader
    base=load_module('rebuttal_openpi_human_train',snapshot/'scripts/train.py')
    base.compute_checkpoint_steps=lambda **kwargs:set(plan['checkpoint_steps'])
    def metric(step,info):
        row=dict(step=int(step),loss=float(info['loss']),grad_norm=float(info['grad_norm']),time=time.time())
        if not np.isfinite([row['loss'],row['grad_norm']]).all():raise FloatingPointError(row)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    base.host_metrics_callback=metric;base.main(cfg)
    final=Path(cfg.checkpoint_dir)/str(steps-1)
    if not (final/'params/_METADATA').is_file():raise RuntimeError('Final parameters missing')
    write(out/'complete.json',dict(checkpoint=str(final),updates=steps,contract=contract))

if __name__=='__main__':main()
