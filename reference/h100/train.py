"""M4 with M2 coordinate definitions, partial-hand masks, fresh Pi0.5 init."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
import time
from aligned_data import ROOT,STUDY,CACHE,M1,M2,LEGACY,sha,write_json,validate_manifest,contract as coordinate_contract

SNAPSHOT=LEGACY/'openpi_snapshot'
OUTPUT=ROOT/'outputs/quest_vla/m4_m2aligned_20260917'


def bootstrap():
    for path in (M1,M2,ROOT/'code/lerobot',SNAPSHOT,SNAPSHOT/'src',SNAPSHOT/'packages/openpi-client/src',STUDY):
        sys.path.insert(0,str(path))
    # Legacy config registry imports config.yaml relative to the process cwd.
    os.chdir(M1)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--max-token-len',type=int,default=768)
    p.add_argument('--preflight-only',action='store_true'); p.add_argument('--smoke',action='store_true')
    args=p.parse_args(); method='m4'; bootstrap()
    import numpy as np
    import jax
    import flax.nnx as nnx
    import flax.traverse_util as traverse
    import openpi.transforms as transforms
    from openpi.models import pi0_config,model as model_lib
    from openpi.training import config as configs,data_loader as loaders,optimizer
    from openpi.shared import normalize
    from scripts import train as base_train
    from m1.protocol import EpochSampler,epoch_plan
    from m2.train import merge_base_leaves as merge153
    from dataset import PartialDataset,MaskAfterNormalize
    from strict_tokenizer import StrictPaligemmaTokenizer
    manifest_path=CACHE/'manifest.json'; manifest=validate_manifest(manifest_path)
    if manifest['smoke'] or (manifest['total_episodes'],manifest['total_frames'])!=(43702,10800000):
        raise ValueError('Full aligned M4 cache required')
    normdir=CACHE/method/'norm_stats'; prov=json.loads((normdir/'provenance.json').read_text())
    if prov['manifest_sha256']!=sha(manifest_path) or prov['norm_sha256']!=sha(normdir/'norm_stats.json'):
        raise ValueError('Masked norm provenance mismatch')
    ds=PartialDataset(manifest_path,method); norms=normalize.load(normdir)
    plan=epoch_plan(len(ds),batch=256,checkpoints=5)
    steps=3 if args.smoke else plan['total_steps']
    mode='smoke' if args.smoke else 'formal'
    logs=STUDY/'logs'/method; logs.mkdir(parents=True,exist_ok=True)
    sd,ad=156,153
    weights=[3.]*3+[1.5]*6+([20.]*3+[4.]*6)*2+[.3]*126
    extra=dict(action_loss_weights=tuple(weights),flow_loss_type='mse',
        quest_local21_recon_loss_weight=100.,quest_local21_recon_left_weight=1.,quest_local21_recon_right_weight=1.,
        quest_local21_recon_left_scale=.09564778953790665,quest_local21_recon_right_scale=.09564778953790665,
        quest_local21_recon_wrist_pose_relative_to_head=True,
        quest_local21_recon_wrist_position_absolute=False,quest_local21_recon_wrist_rotation_absolute=False,
        quest_local21_recon_state_mean=tuple(map(float,norms['state'].mean)),
        quest_local21_recon_state_std=tuple(map(float,norms['state'].std)),
        quest_local21_recon_action_mean=tuple(map(float,norms['actions'].mean)),
        quest_local21_recon_action_std=tuple(map(float,norms['actions'].std)))
    model_cfg=pi0_config.Pi0Config(pi05=True,action_dim=ad,state_dim=sd,action_horizon=32,
        max_token_len=args.max_token_len,paligemma_variant='gemma_2b',action_expert_variant='gemma_300m',
        image_keys=('base_0_rgb',),preserve_image_token_layout=True,**extra)
    mt=configs.ModelTransformFactory()(model_cfg)
    mt=dataclasses.replace(mt,inputs=[dataclasses.replace(t,tokenizer=StrictPaligemmaTokenizer(args.max_token_len))
        if isinstance(t,transforms.TokenizePrompt) else t for t in mt.inputs])
    mt=dataclasses.replace(mt,inputs=[MaskAfterNormalize(),*mt.inputs[:-1],transforms.PadStatesAndActions(ad,model_state_dim=sd)])
    data_cfg=configs.DataConfig(repo_id='fastgarland_m2aligned_100h',asset_id=method,
        norm_stats=norms,use_quantile_norm=False,model_transforms=mt)
    transformed=loaders.transform_dataset(ds,data_cfg)
    indices={0,len(ds)//2,len(ds)-1}
    # Include actual single-current-hand windows in preflight, not just full-valid samples.
    for number,e in enumerate(manifest['episodes']):
        if e['anchors_one_current_hand']:
            _,packet,_,_=ds.episode(number)
            loc=np.where(packet['hand_valid'][packet['anchors']].sum(axis=1)==1)[0]
            indices.add(int((ds.ends[number-1] if number else 0)+loc[0])); break
    checks=[]
    for i in sorted(indices):
        sample=transformed[i]
        if (sample['state'].shape!=(sd,) or sample['actions'].shape!=(32,ad) or
            sample['reliability'].shape!=(32,ad) or not sample['reliability'].any() or
            not np.isfinite(sample['state']).all() or not np.isfinite(sample['actions']).all()):
            raise ValueError('Transformed partial-hand contract failed')
        if not np.all(sample['actions'][sample['reliability']==0]==0):
            raise ValueError('Invalid target was not zeroed after normalization')
        checks.append(dict(index=i,valid_action_fraction=float(sample['reliability'].mean())))
    name='m4_m2aligned'; run='m4_m2aligned_epoch1'+('_smoke' if args.smoke else '')
    cp_root=OUTPUT/method/'checkpoints'/name/run
    sources=[STUDY/n for n in ('train.py','dataset.py','aligned_data.py','prepare.py','strict_tokenizer.py')]+[
        SNAPSHOT/'src/openpi/models/pi0.py',SNAPSHOT/'src/openpi/models/pi0_config.py',
        SNAPSHOT/'scripts/train.py',SNAPSHOT/'src/openpi/training/data_loader.py']
    contract=dict(**plan,method=method,mode=mode,steps=steps,fsdp_devices=8,per_gpu_batch=32,
        action_dim=ad,state_dim=sd,horizon=32,action_stride=1,sample_stride=2,num_workers=24,
        seed=42,learning_rate=2.5e-5,warmup_steps=20,head_source='official_camera_head_worn_view_proxy',
        coordinate_contract=coordinate_contract(),max_token_len=args.max_token_len,token_overflow='error',
        loss='availability_masked_flow_mse',reconstruction_weight=100.,
        snapshot=str(SNAPSHOT),manifest=str(manifest_path),manifest_sha256=sha(manifest_path),
        norm_dir=str(normdir),norm_sha256=sha(normdir/'norm_stats.json'),
        checkpoint_dir=str(cp_root),pretrained_params=str(ROOT/'openpi_cache/checkpoints/pi05_base/params'),
        tested_file_hashes={str(p):sha(p) for p in sources},preflight_examples=checks,
        input_missingness='current-hand-only prompt annotation; normalized zero placeholder',
        future_validity='loss-only sidecar, never prefix conditioning')
    write_json(logs/(mode+'.contract.json'),contract)
    if args.preflight_only:
        print('PREFLIGHT_PASS',json.dumps(contract),flush=True); return
    if jax.device_count()!=8 or jax.process_count()!=1: raise RuntimeError('Exactly one eight-GPU host required')
    if cp_root.exists(): raise RuntimeError('Existing run: refusing implicit overwrite/resume')

    @dataclasses.dataclass(frozen=True)
    class Factory:
        def create(self,assets_dirs,model_config): return data_cfg

    @dataclasses.dataclass(frozen=True)
    class Loader:
        def load(self,target):
            source=model_lib.restore_params(contract['pretrained_params'],restore_type=np.ndarray)
            merged,audit=merge153(traverse.flatten_dict(source,sep='/'),traverse.flatten_dict(target,sep='/'))
            if len(audit['random_interface'])!=3: raise ValueError('Unexpected reset leaves')
            write_json(logs/(mode+'.weight_init.json'),audit)
            return traverse.unflatten_dict(merged,sep='/')

    cfg=configs.TrainConfig(name=name,exp_name=run,project_name='m4_m2aligned_epoch1',model=model_cfg,
        data=Factory(),weight_loader=Loader(),assets_base_dir=str(OUTPUT/method/'assets'),
        checkpoint_base_dir=str(OUTPUT/method/'checkpoints'),batch_size=256,num_workers=24,
        num_train_steps=steps,seed=42,fsdp_devices=8,log_interval=1 if args.smoke else 20,
        save_interval=max(1,plan['total_steps']//5),checkpoints_per_run=5,keep_period=1,
        overwrite=False,resume=False,ema_decay=None,wandb_enabled=False,freeze_filter=nnx.Nothing,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=20,peak_lr=2.5e-5,
            decay_steps=plan['total_steps'],decay_lr=2.5e-5),optimizer=optimizer.AdamW())
    def create_loader(config,*,sharding=None,shuffle=False,**kwargs):
        if not shuffle: raise ValueError('One-epoch sampler needs shuffled trainer request')
        sampler=EpochSampler(len(ds),batch=256,seed=42)
        runtime=loaders.TorchDataLoader(transformed,256,sharding=sharding,sampler=sampler,
            shuffle=False,num_batches=steps,num_workers=24,seed=42)
        return loaders.DataLoaderImpl(data_cfg,runtime)
    loaders.create_data_loader=create_loader
    base_train.compute_checkpoint_steps=lambda **kwargs:set() if args.smoke else set(plan['checkpoint_steps'])
    previous=jax.config.update
    def cache_update(key,value):
        return previous(key,str(STUDY/'jax_cache'/method) if key=='jax_compilation_cache_dir' else value)
    jax.config.update=cache_update
    def record(step,info):
        row=dict(step=int(step),loss=float(info['loss']),grad_norm=float(info['grad_norm']),
                 time=time.time(),total_steps=steps,mode=mode)
        if not np.isfinite([row['loss'],row['grad_norm']]).all(): raise FloatingPointError('Nonfinite metrics')
        with (logs/(mode+'.metrics.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
        write_json(logs/(mode+'.latest.json'),row)
    base_train.host_metrics_callback=record
    base_train.main(cfg)
    write_json(logs/(mode+'.completed.json'),dict(time=time.time(),contract=contract))


if __name__=='__main__': main()
