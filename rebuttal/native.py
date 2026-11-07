"""Isolated seeded native finetuning adapters for the four reported suites."""
import argparse,dataclasses,json,os,sys,time
from pathlib import Path
import numpy as np
from protocol import ROOT,OUT,HERE,BUDGETS,native_spec,configure_runtime,load_module,write,sha,EpochSampler,safe_name


def build_dexjoco(s):
    refdir=ROOT/'experiments/egoquest_egodex48_20260911';sys.path.insert(0,str(refdir))
    ft=load_module('rebuttal_dexjoco_reference',refdir/'finetune_entry.py')
    cpu=os.environ.get('JAX_PLATFORMS')=='cpu';gpus=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3')
    os.environ.update(ft.runtime_env('single'));os.environ['CUDA_VISIBLE_DEVICES']=gpus;configure_runtime(cpu)
    trainer,old,_=ft.build_config('single')
    spec=dataclasses.replace(old,checkpoint_path=Path(s['source'])/'params',output_root=Path(s['output']),name=s['run'],num_train_steps=s['updates'],action_head_seed=s['seed'])
    cfg=trainer.build_train_config(spec,overwrite=False,resume=False)
    recipe=load_module('rebuttal_dexjoco_quantiles',ROOT/'experiments/m4_smoothed_off_dexjoco_20260925/recipe.py')
    normdir=Path(s['output'])/'native_norm'/old.norm_stats_asset_id
    normdir.mkdir(parents=True,exist_ok=True)
    raw=ft.validated_norm_bytes(ft.arm_spec('single').norm_source,ft.arm_spec('single').norm_sha256,23,22)
    normfile=normdir/'norm_stats.json'
    if normfile.exists() and normfile.read_bytes()!=raw:raise ValueError('Existing output normalization changed')
    normfile.write_bytes(raw)
    cfg=dataclasses.replace(cfg,seed=s['seed'],data=recipe.QuantileFactory(cfg.data,normdir),checkpoints_per_run=1)
    os.environ['OPENPI_EXPLICIT_CHECKPOINT_STEPS']=str(s['eval_checkpoint'])
    from scripts import train
    train.compute_checkpoint_steps=lambda **kwargs:{s['eval_checkpoint']}
    def probe():
        data=cfg.data.create(cfg.assets_dirs,cfg.model)
        # Install the exact five-task training reader, then inspect the actual loader.
        oldpatch=trainer.install_dexjoco_dataset_patch(spec)
        try:
            from openpi.training import data_loader
            loader=data_loader.create_data_loader(cfg,shuffle=False,num_batches=1)
            obs,actions=next(iter(loader));return dict(batch_state=list(obs.state.shape),batch_actions=list(actions.shape),tasks=list(spec.task_names),norm_sha256=sha(normfile))
        finally:trainer.uninstall_dexjoco_dataset_patch(oldpatch)
    def run():
        patch=trainer.install_dexjoco_dataset_patch(spec)
        try:train.main(cfg)
        finally:trainer.uninstall_dexjoco_dataset_patch(patch)
    return cfg,probe,run


def build_dmg(s):
    code=ROOT/'robot_finetune/dexmimicgen';sys.path.insert(0,str(code))
    from dexmg import specs,train,sampling
    group='A' if s['suite']=='dmg_a' else 'B'
    spec=specs.make_spec(group,'base',s['updates']);out=Path(s['output'])
    spec.update(seed=s['seed'],batch_size=128,checkpoint_path=str(Path(s['source'])/'params'),run_id=s['run'],exp_name=s['run'],
        run_dir=str(out/'run'),checkpoint_base_dir=str(out/'checkpoints'),checkpoint_dir=str(out/'checkpoints'/spec['name']/s['run']),
        save_interval=5000,keep_period=5000,epoch_sampler='seeded_single_permutation_prefix_no_tail_reached',sampler_seed=s['seed'])
    original_config=train.build_config;original_preflight=train.preflight;original_loader=train.load_trainer
    def config(sp):return dataclasses.replace(original_config(sp),checkpoints_per_run=1)
    def preflight(sp):
        cfg,report=original_preflight(sp);report['sampler']=sp['epoch_sampler'];report['sampler_seed']=sp['seed'];return cfg,report
    def load_trainer(snapshot):
        module=original_loader(snapshot);module.compute_checkpoint_steps=lambda **kwargs:{s['eval_checkpoint']};return module
    class PrefixSampler:
        def __init__(self,size,batch,seed,start_step,stop_step):
            if start_step!=0:raise ValueError('No implicit resume in rebuttal runs')
            if stop_step*batch>size:raise ValueError('Protocol would cross first permutation; define another cohort explicitly')
            self.inner=EpochSampler(size,batch,seed,stop_step)
        def __len__(self):return len(self.inner)
        def __iter__(self):return iter(self.inner)
    sampling.ResumableIndexSampler=PrefixSampler
    train.build_config=config;train.preflight=preflight;train.load_trainer=load_trainer
    cfg=config(spec)
    def probe():return preflight(spec)[1]
    def run():return train.run(spec)
    return cfg,probe,run


class RobotwinInitializer:
    def __init__(self,source,report):self.source=source;self.report=report
    def load(self,target):
        from openpi.models.model import restore_params
        from flax.traverse_util import flatten_dict,unflatten_dict
        off=ROOT/'experiments/robotwin_m4_smoothed_off_20260925';sys.path.insert(0,str(off))
        ref=load_module('rebuttal_robotwin_transfer',off/'train_run.py')
        source=restore_params(str(Path(self.source)/'params'),restore_type=np.ndarray)
        base=restore_params(str(ROOT/'openpi_cache/checkpoints/pi05_base/params'),restore_type=np.ndarray)
        merged,audit=ref.merge_weights(flatten_dict(target,sep='/'),flatten_dict(source,sep='/'),flatten_dict(base,sep='/'))
        write(self.report,audit);return unflatten_dict(merged,sep='/')


def build_robotwin(s):
    refdir=ROOT/'experiments/robotwin_matched5_20260912';sys.path.insert(0,str(refdir))
    os.chdir(refdir/'openpi_snapshot');sys.path.insert(0,str(ROOT/'code/lerobot'))
    ref=load_module('rebuttal_robotwin_reference',refdir/'run_train.py')
    spec=dataclasses.replace(ref.specification('base'),name=s['run'],variant='base',checkpoint_path=Path(s['source'])/'params',output_root=Path(s['output']),num_train_steps=5000,save_interval=2500,keep_period=1)
    cfg=ref.tr.build_train_config(spec,overwrite=False,resume=False)
    cfg=dataclasses.replace(cfg,seed=s['seed'],checkpoints_per_run=2,weight_loader=RobotwinInitializer(s['source'],Path(s['output'])/'initialization.json'))
    ref.tr._create_robotwin_dataset=ref.dataset_factory
    def probe():
        data=cfg.data.create(cfg.assets_dirs,cfg.model);ds=ref.CachedDataset(refdir/'cache',50)
        transformed=ref.tr._data_loader.transform_dataset(ds,data)
        samples=[transformed[i] for i in [0,len(ds)//2,len(ds)-1]]
        if any(x['state'].shape!=(32,) or x['actions'].shape!=(50,32) or not np.isfinite(x['actions']).all() for x in samples):raise ValueError('RoboTwin sample contract')
        return dict(frames=len(ds),tasks=list(ref.TASKS),state=[32],actions=[50,32],norm_sha256=sha(ROOT/'openpi_cache/checkpoints/pi05_base/assets/trossen/norm_stats.json'))
    def run():
        ref.tr.patch_robotwin_loader(spec.task_filters)
        try:
            trainer=ref.tr._load_openpi_train_module();trainer.compute_checkpoint_steps=lambda **kwargs:{2500,4999};trainer.main(cfg)
        finally:ref.tr.unpatch_robotwin_loader()
    return cfg,probe,run


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=BUDGETS,required=True);p.add_argument('--method',required=True)
    p.add_argument('--seed',type=int,required=True);sources=p.add_mutually_exclusive_group()
    sources.add_argument('--source',type=Path);sources.add_argument('--human-run')
    p.add_argument('--gpus',default='0,1,2,3')
    p.add_argument('--mode',choices=['describe','preflight','train'],default='describe');a=p.parse_args()
    source=a.source
    if a.human_run:
        complete=OUT/'human'/safe_name(a.human_run)/'complete.json'
        trained=json.loads(complete.read_text());source=Path(trained['checkpoint'])
    s=native_spec(a.suite,a.method,a.seed,source)
    if a.mode=='describe':print(json.dumps(s,indent=2));return
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpus;configure_runtime(a.mode=='preflight')
    if not (Path(s['source'])/'params/_METADATA').is_file():raise FileNotFoundError('Source checkpoint not migrated or trained: '+s['source'])
    builder=build_dexjoco if a.suite=='dexjoco' else build_robotwin if a.suite=='robotwin' else build_dmg
    cfg,probe,run=builder(s)
    if cfg.seed!=a.seed or cfg.batch_size!=s['batch'] or cfg.num_train_steps!=s['updates'] or cfg.fsdp_devices!=4:raise ValueError('Native config drift')
    if cfg.model.quest_local21_recon_loss_weight!=0:raise ValueError('Human geometric objective must be disabled in native finetuning')
    report=dict(protocol=s,checkpoint_dir=str(cfg.checkpoint_dir),config_seed=cfg.seed,model=dataclasses.asdict(cfg.model),source_code_sha256={f.name:sha(f) for f in HERE.glob('*.py')})
    if a.mode=='preflight':
        report['data_probe']=probe();write(Path(s['output'])/'preflight.json',report);print('PREFLIGHT_PASS',json.dumps(report,default=str));return
    import jax
    if jax.device_count()!=4 or jax.process_count()!=1:raise RuntimeError('Exactly four selected GPUs required')
    if Path(cfg.checkpoint_dir).exists():raise FileExistsError(cfg.checkpoint_dir)
    out=Path(s['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'train.contract.json').open('x') as f:json.dump(report,f,indent=2,default=str)
    if a.suite=='dexjoco':
        recipe=sys.modules['rebuttal_dexjoco_quantiles'];sidecar=recipe.contract_path(cfg.checkpoint_dir)
        if sidecar.exists():raise FileExistsError(sidecar)
        norm=out/'native_norm/dexjoco4_singlehand_v1/norm_stats.json'
        write(sidecar,dict(protocol=recipe.PROTOCOL,normalization='q01_q99',arm='single',checkpoint_dir=str(cfg.checkpoint_dir),
            norm_asset='dexjoco4_singlehand_v1',norm_sha256=sha(norm),source=s['source'],seed=s['seed'],saved_steps=[10000]))
    run()
    final=Path(cfg.checkpoint_dir)/str(s['eval_checkpoint'])
    if not (final/'params/_METADATA').is_file():raise ValueError('Expected evaluation checkpoint missing')
    write(out/'complete.json',dict(checkpoint=str(final),protocol=s,time=time.time()))

if __name__=='__main__':main()
