#!/usr/bin/env python
"""Explicit launch entrypoint. No queue, kill, GPU sharing or automatic production run."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

RELEASE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RELEASE_ROOT / "src"))


def backend_selection(probe=False):
    # Metric callbacks execute on CPU even while the model is trained on CUDA.
    return 'cpu' if probe else 'cuda,cpu'


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',choices=['A','B'],required=True)
    p.add_argument('--variant',choices=['base','ego_our','egodexstyle'],required=True)
    p.add_argument('--steps',type=int,required=True)
    p.add_argument('--gpus',help='Exactly four idle physical IDs; smoke uses two')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--probe',action='store_true',help='CPU data/config checks only')
    p.add_argument('--resume',action='store_true',help='Resume same spec from committed full checkpoint')
    args=p.parse_args(argv)
    if not args.probe and not args.gpus:p.error('--gpus is required for training')
    if args.smoke and not 3<=args.steps<=10:p.error('Smoke budget must be 3..10 updates')
    if args.smoke and args.resume:p.error('Smoke resume not supported; use a distinct bounded budget for another smoke')
    return args


@contextlib.contextmanager
def gpu_locks(spec,indices):
    import fcntl
    # The release does not coordinate a cluster.  Keep the lock names
    # portable and machine-agnostic; callers should place ``study_root`` on a
    # local filesystem when running independent hosts.
    lock_root=Path(spec['study_root'])/'gpu_locks'
    lock_root.mkdir(parents=True,exist_ok=True)
    handles=[]
    try:
        for index in sorted(indices):
            file=(lock_root/f'gpu{index}.lock').open('a+')
            handles.append(file)
            fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:
        for file in handles:file.close()


def main(argv=None):
    args=parse_args(argv)
    from dexmg.specs import make_spec
    spec=make_spec(args.group,args.variant,args.steps,smoke=args.smoke)
    spec['resume']=args.resume
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    os.environ.setdefault('OMP_NUM_THREADS','1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    os.environ.setdefault('MKL_NUM_THREADS','1')
    os.environ.setdefault('PYTHONUNBUFFERED','1')
    if args.probe:
        os.environ['JAX_PLATFORMS']=backend_selection(probe=True)
        os.environ['CUDA_VISIBLE_DEVICES']=''
        from dexmg.train import preflight
        _,report=preflight(spec)
        print(json.dumps(report,indent=2))
        return
    from dexmg.guard import query_gpus,validate_gpus
    indices=validate_gpus(args.gpus,*query_gpus(),spec['fsdp_devices'])
    with gpu_locks(spec,indices):
        validate_gpus(args.gpus,*query_gpus(),spec['fsdp_devices'])
        os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(i) for i in indices)
        os.environ['JAX_PLATFORMS']=backend_selection()
        os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','0.90')
        # This host has no need to change allocator, drivers, dependencies or existing jobs.
        from dexmg.train import run
        run(spec)


if __name__=='__main__':main()
