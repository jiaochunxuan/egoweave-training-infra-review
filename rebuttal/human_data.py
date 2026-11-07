"""Cached-only structured data reader; original RGB and annotation hashes retained."""
from pathlib import Path
from collections import OrderedDict
import json,os,io
import numpy as np
from protocol import sha
from targets import encode


def packet_for(entry):
    if 'packet_path' in entry:
        path=Path(entry['packet_path'])
        if sha(path)!=entry['packet_sha256']:raise ValueError('Packet hash mismatch')
        with np.load(path,allow_pickle=False) as f:
            return {k:f[k] for k in ['pose153','anchors','camera_valid','head_valid','hand_valid']}
    import h5py
    with h5py.File(entry['cache_path'],'r') as f:
        p=f['pose'][:];anchors=f['anchors'][:]
    if p.shape[1]!=162:raise ValueError('Structured training needs the official structured pose cache')
    hv=np.isfinite(p[:,:12]).all(axis=1)
    hands=np.stack([np.isfinite(p[:,np.r_[ps:ps+3,rs:rs+9,ls:ls+63]]).all(axis=1) for ps,rs,ls in [(12,18,36),(15,27,99)]],axis=-1)
    return dict(pose153=p,anchors=anchors,camera_valid=hv,head_valid=hv,hand_valid=hands)


def validate_view_pair(global_manifest,local_manifest):
    g=json.loads(Path(global_manifest).read_text());l=json.loads(Path(local_manifest).read_text())
    gc=g.get('reconstruction_control',{});lc=l.get('reconstruction_control',{})
    if gc.get('view')!='global_axis' or lc.get('view')!='hand_centered':raise ValueError('Missing actual global/local reconstruction provenance')
    for k in ['recordings_sha256','roi_sha256','calibration_sha256','estimator_sha256','fusion_recipe_sha256','smoothing_recipe_sha256']:
        if not gc.get(k) or gc[k]!=lc.get(k):raise ValueError('Unmatched reconstruction control: '+k)
    for k in ['coordinate_contract','image_manifest_sha256','image_config_sha256']:
        if not g.get(k) or g[k]!=l.get(k):raise ValueError('Unmatched target frame or policy RGB: '+k)
    if sha(global_manifest)==sha(local_manifest):raise ValueError('Same annotations cannot be a view comparison')


class PacketDataset:
    def __init__(self,manifest,condition='full',selection=None,max_open=8):
        self.path=str(manifest);self.manifest=json.loads(Path(manifest).read_text());self.condition=condition
        self.selection=None if selection is None else json.loads(Path(selection).read_text())
        if self.selection and sha(manifest) not in self.selection['manifest_sha256s']:raise ValueError('Selection does not belong to manifest')
        self.entries=[];self.chosen=[]
        for e in self.manifest['episodes']:
            if self.selection:
                ids=self.selection['anchors'].get(e['key'],[])
                if not ids:continue
                self.chosen.append(np.array(ids,dtype=np.int64));self.entries.append(e)
            else:self.chosen.append(None);self.entries.append(e)
        counts=[len(a) if a is not None else e['windows'] for e,a in zip(self.entries,self.chosen)]
        if not counts or sum(counts)<=0:raise ValueError('No training windows')
        self.ends=np.cumsum(counts);self.max_open=max_open;self.handles=OrderedDict();self.pid=os.getpid()
    def __len__(self):return int(self.ends[-1])
    def close(self):
        for h,*_ in self.handles.values():h.close()
        self.handles.clear()
    def __getstate__(self):return {**self.__dict__,'handles':OrderedDict(),'pid':None}
    def episode(self,i):
        import h5py
        if os.getpid()!=self.pid:self.close();self.pid=os.getpid()
        if i in self.handles:self.handles.move_to_end(i);return self.handles[i]
        e=self.entries[i];p=packet_for(e);anchors=self.chosen[i]
        if anchors is None:anchors=p['anchors']
        if not np.isin(anchors,p['anchors']).all():raise ValueError('Selection contains unavailable anchors')
        img=e.get('image_cache_path',e.get('m1_cache_path'));h=h5py.File(img,'r')
        header=json.loads(h.attrs['header_json'])
        if header['key']!=e['key'] or not header['committed'] or header['source']['mp4_sha256']!=e['source_mp4_sha256']:
            h.close();raise ValueError('RGB identity mismatch')
        ia=h['anchors'][:];loc=np.searchsorted(ia,anchors)
        if np.any(loc>=len(ia)) or not np.array_equal(ia[loc],anchors):h.close();raise ValueError('RGB window missing')
        other=None
        if self.selection:
            f=Path(self.selection['mask_dir'])/self.selection['mask_files'][e['key']]
            if sha(f)!=self.selection['mask_sha256s'][e['key']]:raise ValueError('Matched supervision mask changed')
            with np.load(f,allow_pickle=False) as z:
                if not np.array_equal(z['anchors'],anchors):raise ValueError('Selection mask anchor mismatch')
                sm=np.unpackbits(z['state_bits'],axis=-1,count=156,bitorder='little').astype(bool)
                am=np.unpackbits(z['action_bits'],axis=-1,count=32*153,bitorder='little').reshape(-1,32,153).astype(bool)
                other=(sm,am)
        while len(self.handles)>=self.max_open:self.handles.popitem(last=False)[1][0].close()
        value=(h,p,h['jpeg_offsets'][:],loc,anchors,other);self.handles[i]=value;return value
    def __getitem__(self,index):
        from PIL import Image
        if not 0<=index<len(self):raise IndexError(index)
        i=int(np.searchsorted(self.ends,index,side='right'));offset=int(index-(self.ends[i-1] if i else 0))
        h,p,offs,loc,anchors,other=self.episode(i);anchor=anchors[offset]
        s,a,sm,am=encode(p,np.array([anchor]),self.condition)
        if other is not None:
            sm&=other[0][offset:offset+1];am&=other[1][offset:offset+1];s=np.where(sm,s,0);a=np.where(am,a,0)
        row=int(loc[offset]);raw=h['jpeg_bytes'][int(offs[row]):int(offs[row+1])].tobytes()
        with Image.open(io.BytesIO(raw)) as im:rgb=np.asarray(im.convert('RGB')).copy()
        current_hands=p['hand_valid'][anchor]
        if other is not None:current_hands=np.array([sm[0,9:18].all() and sm[0,27:90].all(),sm[0,18:27].all() and sm[0,90:153].all()])
        prompt=self.entries[i]['prompt'];missing=[k for k,v in zip(['left','right'],current_hands) if not v]
        if missing:prompt+=' Current hand tracking unavailable: '+', '.join(missing)+'.'
        return dict(state=s[0],actions=a[0],state_valid=sm[0],reliability=am[0].astype(np.float32),prompt=prompt,image={'base_0_rgb':rgb},image_mask={'base_0_rgb':True})


class MaskAfterNormalize:
    def __call__(self,data):
        data=dict(data);data['state']=np.where(data.pop('state_valid'),data['state'],0).astype(np.float32)
        data['actions']=np.where(data['reliability']>0,data['actions'],0).astype(np.float32);return data
