"""Explicit seeds, budgets and isolated identities; no accelerator imports."""
from pathlib import Path
import os,json,hashlib,importlib.util,sys,re
import numpy as np

ROOT=Path(os.environ.get('EGOWEAVE_EVOPOLICY_ROOT','/path/to/evopolicy'))
HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/rebuttal_20261107'
SOURCES={
 'direct':'openpi_cache/checkpoints/pi05_base',
 'egodex_style':'outputs/quest_vla/m1_quantile_20260918/checkpoints/m1_official_48d/m1_official_egodex48_epoch1_quantile/18399',
 'official_structured':'outputs/quest_vla/m2_m4_quantile_20260917/m2/checkpoints/m2_official_our153/m2_official_egodex_our153_epoch1/18399',
 'reconstructed_wrist_fingertip':'outputs/quest_vla/m3_quantile_20260918/m3/checkpoints/m3_partial/m3_partial_epoch1/18327',
 'egoweave':'outputs/quest_vla/m4_smoothed_pretrain_20260920/m4/checkpoints/m4_m2aligned/m4_m2aligned_epoch1/18399',
 'no_geometry':'outputs/quest_vla/m4_smoothed_off_pretrain_20260925/m4/checkpoints/m4_m2aligned/m4_m2aligned_epoch1/18399'}
BUDGETS={'dexjoco':(10001,10000,32),'dmg_a':(5001,5000,128),'dmg_b':(5001,5000,128),'robotwin':(5000,4999,64)}

def safe_name(s):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*',s):raise ValueError('Unsafe experiment identity')
    return s

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n');tmp.replace(p)

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def native_spec(suite,method,seed,source=None):
    safe_name(method)
    if isinstance(seed,bool) or int(seed)!=seed or seed<0:raise ValueError('Invalid seed')
    updates,label,batch=BUDGETS[suite];run=f'{suite}_{method}_seed{seed}'
    if source is None:
        if method not in SOURCES:raise ValueError('New ablations require --source checkpoint')
        source=ROOT/SOURCES[method]
    return dict(suite=suite,method=method,seed=int(seed),sampler_seed=int(seed),updates=updates,
        eval_checkpoint=label,batch=batch,fsdp=4,horizon=50,learning_rate=5e-5,warmup_steps=100,
        source=str(source),run=run,output=str(OUT/'native'/run),overwrite=False,resume=False,
        human_geometric_loss=0,training_task_cohort='legacy5_including_fold_glasses' if suite=='dexjoco' else 'frozen_native_manifest',
        reported_tasks=['hammer_nail','pick_bucket','pinch_tongs','water_plant'] if suite=='dexjoco' else None)

class EpochSampler:
    """One permutation per epoch; final padding is explicit and seed-dependent."""
    def __init__(self,size,batch,seed,steps):
        if min(size,batch,steps)<=0 or seed<0:raise ValueError('Invalid sampler budget')
        self.size,self.batch,self.seed,self.steps=map(int,(size,batch,seed,steps))
    def __len__(self):return self.steps*self.batch
    def __iter__(self):
        remaining=len(self);epoch=0
        while remaining:
            ids=np.random.default_rng(np.random.SeedSequence([self.seed,epoch])).permutation(self.size)
            padded=((self.size+self.batch-1)//self.batch)*self.batch
            ids=np.resize(ids,padded);take=min(remaining,padded)
            yield from map(int,ids[:take]);remaining-=take;epoch+=1

def real_steps(windows):
    steps=2*(int(windows)//32)
    if steps<=0:raise ValueError('Insufficient real demonstration windows')
    return steps

def configure_runtime(cpu=False):
    os.environ.setdefault('WANDB_MODE','disabled');os.environ.setdefault('OPENPI_DATA_HOME',str(ROOT/'openpi_cache'))
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    flags=os.environ.get('XLA_FLAGS','')
    if '--xla_gpu_enable_triton_gemm=' not in flags:os.environ['XLA_FLAGS']=(flags+' --xla_gpu_enable_triton_gemm=false').strip()
    os.environ.setdefault('OPENPI_JAX_COMPILATION_CACHE_DIR',str(OUT/'cache/jax'))
    if cpu:os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='')
