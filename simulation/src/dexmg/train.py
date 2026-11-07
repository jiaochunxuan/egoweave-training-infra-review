"""Native DexMimicGen adapter for an isolated, verified OpenPI training snapshot.

No accelerator import occurs until bootstrap(). No old training tree is modified.
"""
from dataclasses import dataclass,asdict
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import time

ROOT=Path(os.environ.get('EGO_WEAVE_ROOT','.')).expanduser()
STUDY=ROOT/'experiments/dexmimicgen_ab'


def bootstrap():
    snapshot=Path(os.environ.get('DEXMG_OPENPI_ROOT',str(STUDY/'openpi_snapshot'))).resolve()
    if not (snapshot/'scripts/train.py').is_file():raise FileNotFoundError(f'Missing frozen snapshot: {snapshot}')
    for path in (snapshot,snapshot/'src',snapshot/'packages/openpi-client/src',Path(os.environ.get('LEROBOT_ROOT', ROOT/'external/lerobot'))):
        if str(path) not in sys.path:sys.path.insert(0,str(path))
    # The frozen config registry opens its bundled config.yaml relative to cwd.
    # This only satisfies registry import; NativeDataFactory never uses DexJoCo data.
    os.chdir(snapshot)
    os.environ.setdefault('WANDB_MODE','disabled')
    os.environ.setdefault('OPENPI_DATA_HOME',str(ROOT/'openpi_cache'))
    return snapshot


@dataclass(frozen=True)
class NativeDataFactory:
    spec: dict

    def create(self,assets_dirs,model_config):
        from openpi.training.config import DataConfig
        from openpi import transforms as t
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.shared import normalize
        stats=normalize.load(Path(self.spec['norm_stats_path']).parent)
        if set(stats)!={'state','actions'}:raise ValueError('Expected separate native state/action norms')
        for key,dim in (('state',self.spec['state_dim']),('actions',self.spec['action_dim'])):
            if stats[key].mean.shape!=(dim,):raise ValueError(f'Wrong {key} normalization dimension')
        return DataConfig(repo_id='dexmimicgen_native',asset_id=self.spec['asset_id'],
            norm_stats=stats,use_quantile_norm=True,
            model_transforms=t.Group(inputs=[t.ResizeImages(224,224),
                t.TokenizePrompt(PaligemmaTokenizer(model_config.max_token_len),discrete_state_input=True),
                t.PadStatesAndActions(model_config.action_dim,model_state_dim=model_config.state_dim)]))


@dataclass(frozen=True)
class AuditedNativeTransfer:
    source_path: str
    base_path: str
    audit_dir: str

    def load(self,target):
        import numpy as np
        from flax.traverse_util import flatten_dict,unflatten_dict
        from openpi.models.model import restore_params
        from dexmg.transfer import transfer_flat
        base=flatten_dict(restore_params(self.base_path,restore_type=np.ndarray),sep='/')
        source=base if self.source_path==self.base_path else flatten_dict(restore_params(self.source_path,restore_type=np.ndarray),sep='/')
        result,audit=transfer_flat(flatten_dict(target,sep='/'),source,base)
        expected={'action_in_proj/kernel','action_in_proj/bias','action_out_proj/kernel','action_out_proj/bias'}
        if not expected.issubset(result):raise ValueError('Missing expected action interface leaves')
        audit.update(source_path=self.source_path,base_path=self.base_path)
        directory=Path(self.audit_dir)
        with (directory/'transfer_receipt.json').open('x') as file:json.dump(audit,file,indent=2)
        np.savez(directory/'initial_action_interface.npz',**{k:result[k] for k in sorted(expected)})
        return unflatten_dict(result,sep='/')


def build_config(spec):
    bootstrap()
    from openpi.models.pi0_config import Pi0Config
    from openpi.training.config import TrainConfig
    from openpi.training.optimizer import AdamW,CosineDecaySchedule
    return TrainConfig(name=spec['name'],project_name=spec['project_name'],exp_name=spec['exp_name'],
        model=Pi0Config(pi05=True,action_dim=spec['action_dim'],state_dim=spec['state_dim'],
            action_horizon=spec['horizon'],max_token_len=spec['max_token_len'],discrete_state_input=True),
        data=NativeDataFactory(spec),
        weight_loader=AuditedNativeTransfer(spec['checkpoint_path'],spec['base_checkpoint_path'],spec['run_dir']),
        assets_base_dir=spec['assets_dir'],checkpoint_base_dir=spec['checkpoint_base_dir'],
        seed=spec['seed'],batch_size=spec['batch_size'],num_workers=spec['num_workers'],
        num_train_steps=spec['num_train_steps'],fsdp_devices=spec['fsdp_devices'],
        save_interval=spec['save_interval'],keep_period=spec['keep_period'],log_interval=1 if spec['mode']=='smoke' else 20,
        overwrite=False,resume=spec['resume'],wandb_enabled=False,
        lr_schedule=CosineDecaySchedule(warmup_steps=spec['warmup_steps'],peak_lr=spec['learning_rate'],
            decay_steps=spec['lr_decay_steps'],decay_lr=spec['decay_lr']),
        optimizer=AdamW(clip_gradient_norm=spec['clip_gradient_norm']),ema_decay=spec['ema_decay'],
        policy_metadata={'dataset':'DexMimicGen','group':spec['group'],'manifest_path':spec['manifest_path'],
            'native_action_dim':spec['action_dim'],'state_dim':spec['state_dim'],'adapter':'dexmg.contracts'})


def preflight(spec):
    import numpy as np
    from dexmg.guard import validate_assets,sha256_file
    from dexmg.dataset import HDF5Dataset
    cfg=build_config(spec)
    from openpi.training import data_loader
    from dexmg.specs import config_fingerprint
    receipt=validate_assets(spec)
    for path in (spec['checkpoint_path'],spec['base_checkpoint_path']):
        p=Path(path)
        if not p.is_dir():raise FileNotFoundError(p)
        if p.parent.name.isdigit():
            meta=json.loads((p.parent/'_CHECKPOINT_METADATA').read_text())
            if meta.get('commit_timestamp_nsecs',0)<=0:raise ValueError(f'Uncommitted source checkpoint {p}')
    ds=HDF5Dataset(spec['manifest_path'],spec['horizon'])
    if (ds.action_dim,ds.state_dim)!=(spec['action_dim'],spec['state_dim']):raise ValueError('Model/dataset dimensions differ')
    if len(ds)<spec['batch_size']:raise ValueError('Dataset smaller than one complete batch')
    data=cfg.data.create(cfg.assets_dirs,cfg.model)
    transformed=data_loader.transform_dataset(ds,data)
    indices={0,len(ds)-1}
    offset=0
    for source in ds.manifest['sources']:
        indices.add(offset)
        offset+=sum(e['length'] for e in source['episodes'])
    for index in sorted(indices):
        sample=transformed[index]
        if sample['actions'].shape!=(spec['horizon'],spec['action_dim']):raise ValueError('Wrong transformed action shape')
        if sample['state'].shape!=(spec['state_dim'],):raise ValueError('Wrong transformed state shape')
        if not all(v.shape==(224,224,3) for v in sample['image'].values()):raise ValueError('Wrong image size')
        if sample['tokenized_prompt'].shape!=(spec['max_token_len'],):raise ValueError('Wrong token length')
        if not np.isfinite(sample['actions']).all() or not np.isfinite(sample['state']).all():raise ValueError('Nonfinite normalized data')
    if hasattr(ds,'close'):ds.close()
    snapshot=bootstrap()
    trainer_source=(snapshot/'scripts/train.py').read_text()
    if 'shuffle=True' not in trainer_source:raise ValueError('Frozen trainer does not explicitly request shuffle=True')
    code_hashes={str(p.relative_to(snapshot)):sha256_file(p) for p in sorted(snapshot.rglob('*.py'))}
    adapter_hashes={p.name:sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    return cfg,dict(spec=spec,config_fingerprint=config_fingerprint({**spec,'resume':False}),
        model=asdict(cfg.model),optimizer=asdict(cfg.optimizer),lr=asdict(cfg.lr_schedule),
        normalization=receipt,shuffle=True,frames=len(ds),checked_indices=sorted(indices),
        trainer_sha256=sha256_file(snapshot/'scripts/train.py'),snapshot=str(snapshot),
        frozen_code_sha256=code_hashes,adapter_code_sha256=adapter_hashes,
        sampler='numpy_SeedSequence_seed_epoch_full_permutation_drop_last_v1',
        checkpoint=str(cfg.checkpoint_dir),time=time.strftime('%Y-%m-%d %H:%M:%S'),pid=os.getpid())


def load_trainer(snapshot):
    module_spec=importlib.util.spec_from_file_location('dexmg_frozen_train',snapshot/'scripts/train.py')
    module=importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name]=module
    module_spec.loader.exec_module(module)
    return module


def validate_smoke_metrics(metrics,expected_steps):
    import math
    if len(metrics)!=expected_steps or {m['step'] for m in metrics}!=set(range(1,expected_steps+1)):
        raise ValueError('Smoke did not complete every requested update')
    if any(not math.isfinite(m[k]) for m in metrics for k in ('loss','grad_norm','param_norm','raw_action_update_max')):
        raise ValueError('Nonfinite smoke metrics')
    if min(m['loss'] for m in metrics)<=0 or max(m['raw_action_update_max'] for m in metrics)<=0:
        raise ValueError('Smoke requires positive loss and a raw optimizer parameter update, not only EMA drift')


def run(spec):
    """Call only after the launcher has checked and locked physical GPUs."""
    from dexmg.guard import latest_full_checkpoint
    if spec['checkpoint_mode']!='synchronous_host_snapshot':raise ValueError('Unsupported checkpoint mode')
    os.environ['OPENPI_SYNC_CHECKPOINTS']='1'
    cfg,report=preflight(spec)
    directory=Path(spec['run_dir'])
    checkpoint_dir=Path(cfg.checkpoint_dir)
    start_step=0
    if spec['resume']:
        previous=json.loads((directory/'protocol.json').read_text())
        if previous['config_fingerprint']!=report['config_fingerprint'] or previous['normalization']!=report['normalization']:
            raise ValueError('Resume protocol, dataset, or normalization differs; refusing a silent new run')
        if any(previous[key]!=report[key] for key in ('frozen_code_sha256','adapter_code_sha256','sampler')):
            raise ValueError('Resume code or sampler implementation changed')
        checkpoint=latest_full_checkpoint(checkpoint_dir)
        report['resumed_from']=str(checkpoint)
        start_step=int(checkpoint.name)+1
        if int(checkpoint.name)+1>=spec['num_train_steps']:raise ValueError('Run already reached its specified final step')
    else:
        if checkpoint_dir.exists():raise FileExistsError(checkpoint_dir)
        directory.mkdir(parents=True,exist_ok=False)
        with (directory/'protocol.json').open('x') as file:json.dump(report,file,indent=2)
    print('PREFLIGHT',json.dumps(report),flush=True)
    import jax
    import numpy as np
    from openpi.training import data_loader
    from dexmg.dataset import HDF5Dataset
    if jax.process_count()!=1 or jax.device_count()!=spec['fsdp_devices']:
        raise ValueError('This recipe requires one process and exactly the selected local GPU count')
    original_loader=data_loader.create_data_loader
    def shuffled_loader(*args,**kwargs):
        if kwargs.get('shuffle') is not True:raise ValueError('Training without shuffle=True is forbidden')
        from dexmg.sampling import ResumableIndexSampler
        data=cfg.data.create(cfg.assets_dirs,cfg.model)
        dataset=HDF5Dataset(spec['manifest_path'],spec['horizon'])
        sampler=ResumableIndexSampler(len(dataset),spec['batch_size'],spec['seed'],start_step,spec['num_train_steps'])
        dataset=data_loader.transform_dataset(dataset,data)
        runtime=data_loader.TorchDataLoader(dataset,local_batch_size=spec['batch_size'],
            sharding=kwargs.get('sharding'),sampler=sampler,shuffle=False,
            num_batches=spec['num_train_steps']-start_step,num_workers=spec['num_workers'],seed=spec['seed'],framework='jax')
        return data_loader.DataLoaderImpl(data,runtime)
    data_loader.create_data_loader=shuffled_loader
    trainer=load_trainer(bootstrap())
    from dexmg.snapshot import parallel_host_snapshot
    original_snapshot=trainer._checkpoints.host_checkpoint_snapshot
    trainer._checkpoints.host_checkpoint_snapshot=parallel_host_snapshot
    original_step=trainer.train_step
    metrics=[]
    metric_file=(directory/'metrics.jsonl').open('a',buffering=1)
    def collect(step,loss,grad_norm,param_norm,raw_action_update_max):
        row=dict(step=int(step),loss=float(loss),grad_norm=float(grad_norm),param_norm=float(param_norm),
                 raw_action_update_max=float(raw_action_update_max),timestamp=time.time())
        if not np.isfinite([row['loss'],row['grad_norm'],row['param_norm']]).all():raise ValueError(f'Nonfinite training metrics: {row}')
        metric_file.write(json.dumps(row)+'\n')
        if spec['mode']=='smoke':metrics.append(row)
        if spec['mode']=='smoke' or row['step']%20==0:print('TRAIN_METRIC',json.dumps(row),flush=True)
    def checked_step(*args,**kwargs):
        import jax.numpy as jnp
        old_state=args[2] if len(args)>2 else kwargs['state']
        before=old_state.params['action_out_proj']['kernel'].value
        state,info=original_step(*args,**kwargs)
        update=jnp.max(jnp.abs(state.params['action_out_proj']['kernel'].value-before))
        jax.debug.callback(collect,state.step,info['loss'],info['grad_norm'],info['param_norm'],update)
        return state,info
    trainer.train_step=checked_step
    logging.basicConfig(level=logging.INFO)
    try:
        trainer.main(cfg)
        jax.effects_barrier()
    finally:
        metric_file.close()
        data_loader.create_data_loader=original_loader
        trainer._checkpoints.host_checkpoint_snapshot=original_snapshot
    final=latest_full_checkpoint(checkpoint_dir)
    if final.name!=str(spec['num_train_steps']-1):raise ValueError('Final checkpoint label differs from requested budget')
    result=dict(final=str(final),complete=True,mode=spec['mode'],gpu_count=spec['fsdp_devices'],
                batch_size=spec['batch_size'],updates=spec['num_train_steps'],metrics=metrics)
    if spec['mode']=='smoke':
        validate_smoke_metrics(metrics,spec['num_train_steps'])
        # Reload persisted inference parameters and check the learned interface, not merely exit code.
        from openpi.models.model import restore_params
        from flax.traverse_util import flatten_dict
        saved=flatten_dict(restore_params(str(final/'params'),restore_type=np.ndarray),sep='/')
        with np.load(directory/'initial_action_interface.npz') as initial:
            changes={k:float(np.max(np.abs(saved[k].astype(np.float32)-initial[k].astype(np.float32)))) for k in initial.files}
        if not all(np.isfinite(x) for x in changes.values()) or max(changes.values())<=0:
            raise ValueError(f'No persisted action parameter change: {changes}')
        result['persisted_action_max_abs_change']=changes
        result['inference_params_reload']=True
    with (directory/'complete.json').open('x') as file:json.dump(result,file,indent=2)
    print('TRAIN_COMPLETE',json.dumps(result),flush=True)
    return result
