"""Prepare condition-specific normalization from cached poses; never source videos."""
import argparse,json
from pathlib import Path
import numpy as np
from protocol import ROOT,sha,write
from human_data import PacketDataset,packet_for
from targets import encode,rescale_reference_norms

DEFAULT=ROOT/'data/m4_hybrid_smoothed_20260920/manifest.json'
BASE_NORM=DEFAULT.parent/'m4/norm_stats/norm_stats.json'

def moments(x,m):
    x=x.reshape(-1,x.shape[-1]).astype(np.float64);m=m.reshape(x.shape)
    x=np.where(m,x,0)
    if not np.isfinite(x).all():raise ValueError('Nonfinite supervised targets')
    return np.stack([m.sum(axis=0),x.sum(axis=0),(x*x).sum(axis=0)])

def finish(z):
    n,s,ss=z
    if np.any(n<=0):raise ValueError('A motion channel has no observations')
    mean=s/n;std=np.sqrt(np.maximum(0,ss/n-mean*mean));std=np.where(std<1e-6,1,std)
    return dict(mean=mean.tolist(),std=std.tolist(),q01=None,q99=None)

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=DEFAULT)
    p.add_argument('--condition',required=True,choices=['full','no_explicit_head','current_head','no_scale','no_geometry','global_axis'])
    p.add_argument('--selection',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--reference-norm',type=Path)
    a=p.parse_args()
    if (a.out/'norm_stats.json').exists():raise FileExistsError(a.out)
    ds=PacketDataset(a.manifest,a.condition,a.selection)
    if ds.selection and not ds.selection.get('complete'):raise ValueError('Incomplete matched-window audit')
    reference=a.reference_norm
    if reference is None and a.manifest.resolve()==DEFAULT.resolve() and a.selection is None:reference=BASE_NORM
    can_reuse=reference is not None and a.condition in ['full','no_explicit_head','no_geometry','no_scale']
    if can_reuse:
        provenance=json.loads(reference.with_name('provenance.json').read_text())
        expected=provenance.get('manifest_sha256')
        if expected!=sha(a.manifest):raise ValueError('Reference norms belong to different annotations')
        if provenance.get('selection_sha256')!=(sha(a.selection) if a.selection else None):raise ValueError('Reference norms use different windows')
        data=json.loads(reference.read_text());stats=data.get('norm_stats',data)
        stats=rescale_reference_norms(stats,a.condition);mode='fixed_reference_unit_conversion' if a.condition=='no_scale' else 'exact_reference'
    else:
        if a.condition=='no_scale':raise ValueError('Prepare full norms first, then pass --reference-norm; no per-frame scale fitting')
        sz=np.zeros((3,156));az=np.zeros((3,153))
        for i,e in enumerate(ds.entries):
            packet=packet_for(e);anchors=ds.chosen[i]
            if anchors is None:anchors=packet['anchors']
            other=None
            if ds.selection:
                f=Path(ds.selection['mask_dir'])/ds.selection['mask_files'][e['key']]
                with np.load(f) as z:
                    other=(np.unpackbits(z['state_bits'],axis=-1,count=156,bitorder='little').astype(bool),np.unpackbits(z['action_bits'],axis=-1,count=32*153,bitorder='little').reshape(-1,32,153).astype(bool))
            for start in range(0,len(anchors),64):
                s,actions,sm,am=encode(packet,anchors[start:start+64],a.condition)
                if other is not None:sm&=other[0][start:start+64];am&=other[1][start:start+64]
                sz+=moments(s,sm);az+=moments(actions,am)
            if i%100==0:print('NORM_EPISODE',i,len(ds.entries),flush=True)
        stats=dict(state=finish(sz),actions=finish(az));mode='availability_masked_streaming_moments'
    write(a.out/'norm_stats.json',dict(norm_stats=stats))
    write(a.out/'provenance.json',dict(condition=a.condition,manifest_sha256=sha(a.manifest),selection_sha256=sha(a.selection) if a.selection else None,
        windows=len(ds),mode=mode,reference_norm_sha256=sha(reference) if can_reuse else None,norm_sha256=sha(a.out/'norm_stats.json')))
    print('NORMS_READY',a.out,len(ds))

if __name__=='__main__':main()
