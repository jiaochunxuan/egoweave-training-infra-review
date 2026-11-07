"""Matched episode/anchor identities and common supervised channels, CPU only."""
import argparse,json
from pathlib import Path
import numpy as np
from human_data import packet_for
from targets import masks
from protocol import sha,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--limit',type=int);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    left=json.loads(a.left.read_text());right=json.loads(a.right.read_text());lookup={e['key']:e for e in right['episodes']}
    selection={};files={};hashes={};counts=np.zeros((3,153),np.int64);totals=[0,0,0]
    maskdir=(a.out/'masks').resolve();maskdir.mkdir(parents=True)
    episodes=left['episodes'] if a.limit is None else left['episodes'][:a.limit]
    for i,e in enumerate(episodes):
        f=lookup.get(e['key'])
        if f is None:continue
        if e['source_mp4_sha256']!=f['source_mp4_sha256'] or e['frames']!=f['frames']:raise ValueError('Mismatched source recording: '+e['key'])
        l,r=packet_for(e),packet_for(f);la,ra=l['anchors'],r['anchors'];anchors=np.intersect1d(la,ra)
        common_state=[];common_action=[]
        for packet,ids,row in [(l,la,0),(r,ra,1)]:
            totals[row]+=len(ids)
            for start in range(0,len(ids),128):counts[row]+=masks(packet,ids[start:start+128])[1].sum(axis=(0,1))
        if not len(anchors):continue
        for start in range(0,len(anchors),128):
            ls,lm=masks(l,anchors[start:start+128]);rs,rm=masks(r,anchors[start:start+128]);sm=ls&rs;am=lm&rm
            common_state.append(sm);common_action.append(am);counts[2]+=am.sum(axis=(0,1))
        totals[2]+=len(anchors);selection[e['key']]=anchors.tolist();name=f'{i:05d}.npz';files[e['key']]=name
        np.savez_compressed(maskdir/name,anchors=anchors,state_bits=np.packbits(np.concatenate(common_state),axis=-1,bitorder='little'),action_bits=np.packbits(np.concatenate(common_action).reshape(len(anchors),-1),axis=-1,bitorder='little'))
        hashes[e['key']]=sha(maskdir/name)
        if i%100==0:print('AUDIT_EPISODE',i,len(episodes),flush=True)
    write(a.out/'selection.json',dict(complete=a.limit is None,manifest_sha256s=[sha(a.left),sha(a.right)],anchors=selection,mask_dir=str(maskdir),mask_files=files,mask_sha256s=hashes))
    write(a.out/'summary.json',dict(complete=a.limit is None,episodes_scanned=len(episodes),manifest_total_windows=[left['total_windows'],right['total_windows']],windows_in_scanned_common_episodes=totals,
        supervised_action_channels_by_dimension=counts.tolist(),note='Official channels mean finite cached annotations, not independently calibrated tracking confidence. Matched selection enforces identical windows and common per-channel masks.'))
    print('AUDIT_READY',totals,'complete',a.limit is None)

if __name__=='__main__':main()
