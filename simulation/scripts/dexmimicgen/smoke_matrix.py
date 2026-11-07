#!/usr/bin/env python
"""Six sequential bounded GPU smokes; stop immediately on a failed configuration."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def smoke_commands(gpus,steps):
    if not 3<=steps<=10:raise ValueError('Smoke only supports 3..10 steps')
    entry=Path(__file__).with_name('run_train.py')
    return [(f'{group}_{variant}',[sys.executable,'-u',str(entry),'--group',group,'--variant',variant,
             '--steps',str(steps),'--smoke','--gpus',gpus])
            for group in ('A','B') for variant in ('base','ego_our','egodexstyle')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',required=True);p.add_argument('--steps',type=int,default=3)
    p.add_argument('--report-dir',type=Path,required=True);a=p.parse_args()
    a.report_dir.mkdir(parents=True,exist_ok=False)
    reports=[]
    for name,command in smoke_commands(a.gpus,a.steps):
        start=time.time();print('SMOKE_START',name,flush=True)
        with (a.report_dir/f'{name}.log').open('x') as log:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        reports.append(dict(name=name,returncode=completed.returncode,elapsed_seconds=time.time()-start,command=command))
        with (a.report_dir/f'{name}.result.json').open('x') as file:json.dump(reports[-1],file,indent=2)
        print('SMOKE_EXIT',json.dumps(reports[-1]),flush=True)
        if completed.returncode:raise SystemExit(completed.returncode)
    with (a.report_dir/'COMPLETE.json').open('x') as file:json.dump(reports,file,indent=2)
    print('ALL_SIX_SMOKES_COMPLETE',flush=True)


if __name__=='__main__':main()
