"""Task-balanced rollout summaries, training-seed spread and paired intervals."""
from collections import defaultdict
import argparse,json
from pathlib import Path
import numpy as np
from protocol import BUDGETS,write

TASKS={'dexjoco':{'hammer_nail','pick_bucket','pinch_tongs','water_plant'},'dmg_a':{'Box Cleanup','Drawer Cleanup','Lift Tray'},
       'dmg_b':{'Coffee','Pouring','Can Sort'},'robotwin':{'stamp_seal','turn_switch','move_can_pot','handover_block','pick_dual_bottles'}}

def summarize(rows):
    groups=defaultdict(list);seen=set()
    for r in rows:
        if r['checkpoint_step']!=BUDGETS[r['suite']][1]:raise ValueError('Wrong evaluation checkpoint')
        if not isinstance(r['success'],bool):raise ValueError('Success must be a boolean attempt outcome')
        key=tuple(r[k] for k in ['method','train_seed','suite','task','group','rollout_seed'])
        if key in seen:raise ValueError('Duplicate rollout')
        seen.add(key);groups[key[:-1]].append(int(r['success']))
    per_task=defaultdict(list);attempts={}
    for key,values in groups.items():
        per_task[key[:-1]].append(float(np.mean(values)));attempts[str(key)]=len(values)
    per_seed=defaultdict(dict)
    for (method,seed,suite,task),values in per_task.items():
        if suite=='dmg_b' and task=='Can Sort' and len(values)!=2:raise ValueError('Can Sort needs both color groups')
        per_seed[(method,suite,seed)][task]=float(np.mean(values))
    output=defaultdict(list)
    for (method,suite,seed),tasks in per_seed.items():
        if set(tasks)!=TASKS[suite]:raise ValueError('Incomplete task list for '+str((method,suite,seed)))
        output[(method,suite)].append(dict(seed=seed,success=float(np.mean(list(tasks.values()))),tasks=tasks))
    return [dict(method=method,suite=suite,mean_success=float(np.mean([x['success'] for x in seeds])),
                 training_seed_std=float(np.std([x['success'] for x in seeds],ddof=1)) if len(seeds)>1 else None,
                 independent_finetuning_seeds=len(seeds),per_seed=seeds,
                 variation_scope='finetuning_seeds_only; human_pretraining_checkpoint_is_shared') for (method,suite),seeds in sorted(output.items())]

def paired_interval(rows,left,right,seed=0,repeats=2000):
    """Conditional on a fixed trained checkpoint; not a pretraining-seed interval."""
    samples={}
    for method in [left,right]:
        samples[method]={tuple(r[k] for k in ['train_seed','suite','task','group','rollout_seed']):int(r['success']) for r in rows if r['method']==method}
    if not samples[left] or samples[left].keys()!=samples[right].keys():raise ValueError('Paired methods must have identical rollout identities')
    groups=defaultdict(list)
    for k,v in samples[left].items():groups[k[:-1]].append(v-samples[right][k])
    rng=np.random.default_rng(seed);values=[]
    for _ in range(repeats):
        task=defaultdict(list)
        for (train_seed,suite,name,group),data in groups.items():task[(train_seed,suite,name)].append(float(rng.choice(data,len(data),replace=True).mean()))
        suites=defaultdict(list)
        for (train_seed,suite,name),data in task.items():suites[(train_seed,suite)].append(float(np.mean(data)))
        values.append(float(np.mean([np.mean(x) for x in suites.values()])))
    return dict(left=left,right=right,lower=float(np.quantile(values,.025)),upper=float(np.quantile(values,.975)),
        scope='paired rollout resampling conditional on the supplied checkpoints, averaged equally over supplied seed/suite cells')

def main():
    p=argparse.ArgumentParser();p.add_argument('rollouts',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--pair',nargs=2);a=p.parse_args()
    rows=[json.loads(l) for l in a.rollouts.read_text().splitlines() if l.strip()];report=dict(summaries=summarize(rows))
    if a.pair:report['paired_interval']=paired_interval(rows,*a.pair)
    write(a.out,report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
