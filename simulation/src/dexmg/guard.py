"""Fail-closed guards, independent of accelerator libraries."""
import hashlib
import json
import math
from pathlib import Path
import subprocess


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def validate_gpus(selection, rows, processes, expected_count):
    try: indices=[int(x.strip()) for x in selection.split(',')]
    except (ValueError,AttributeError) as exc:raise ValueError('GPU IDs must be comma-separated integers') from exc
    if len(indices)!=expected_count or len(set(indices))!=expected_count:
        raise ValueError(f'Expected exactly {expected_count} distinct GPU IDs')
    by_id={int(r['index']):r for r in rows}
    busy={r['gpu_uuid'] for r in processes}
    for index in indices:
        if index not in by_id:raise ValueError(f'Unknown GPU {index}')
        row=by_id[index]
        if row['uuid'] in busy or row['memory_used']>512 or row['utilization']>5:
            raise ValueError(f'GPU {index} is occupied; no existing process will be stopped')
        if row['memory_total']-row['memory_used']<60000:
            raise ValueError(f'GPU {index} has insufficient free memory for this tested recipe')
    return indices


def query_gpus():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
                                    '--format=csv,noheader,nounits'],text=True)
    rows=[]
    for line in output.splitlines():
        index,uuid,used,total,util=[x.strip() for x in line.split(',')]
        rows.append(dict(index=int(index),uuid=uuid,memory_used=int(used),memory_total=int(total),utilization=int(util)))
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    processes=[]
    for line in output.splitlines():
        if line.strip():
            uuid,pid=[x.strip() for x in line.split(',')]
            processes.append(dict(gpu_uuid=uuid,pid=int(pid)))
    return rows,processes


def validate_assets(spec):
    manifest_path=Path(spec['manifest_path']);norm_path=Path(spec['norm_stats_path'])
    manifest=json.loads(manifest_path.read_text())
    receipt=json.loads(norm_path.with_name('provenance.json').read_text())
    smoke=spec['mode']=='smoke'
    for data in (manifest,receipt):
        if data.get('group')!=spec['group'] or data.get('smoke') is not smoke:
            raise ValueError('Dataset/norm group or smoke/formal mode mismatch')
    if manifest.get('split')!='train':raise ValueError('Only a train manifest may supply normalization')
    if receipt.get('manifest_sha256')!=sha256_file(manifest_path):raise ValueError('Norm manifest fingerprint mismatch')
    if receipt.get('norm_sha256')!=sha256_file(norm_path):raise ValueError('Normalization fingerprint mismatch')
    diagnostics=receipt.get('normalization',{})
    if diagnostics.get('policy')!='q01_q99_collapsed_range_minmax_v1':raise ValueError('Unaudited normalization version')
    maximum=diagnostics.get('max_abs_normalized',{}).get('actions',float('nan'))
    if not math.isfinite(maximum) or maximum>100:raise ValueError('Exploding normalized actions; inspect source ranges before training')
    return receipt


def latest_full_checkpoint(root):
    candidates=sorted((p for p in Path(root).iterdir() if p.is_dir() and p.name.isdigit()),key=lambda p:int(p.name)) if Path(root).exists() else []
    if not candidates:raise ValueError('Resume requires a committed full checkpoint; no silent restart')
    path=candidates[-1]
    meta=path/'_CHECKPOINT_METADATA'
    if not meta.is_file() or json.loads(meta.read_text()).get('commit_timestamp_nsecs',0)<=0:
        raise ValueError(f'Latest checkpoint is incomplete: {path}')
    if not all((path/k).is_dir() for k in ('train_state','params','assets')):
        raise ValueError(f'Inference-only/incomplete checkpoint cannot resume training: {path}')
    return path
