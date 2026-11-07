"""Compare the new reader with the actual frozen smoothed-data reader."""
import sys,json
import numpy as np
from protocol import ROOT,OUT,write
from human_data import PacketDataset

def main():
    src=ROOT/'experiments/m4_m2aligned_20260917';sys.path.insert(0,str(src))
    import aligned_data
    manifest=ROOT/'data/m4_hybrid_smoothed_20260920/manifest.json'
    aligned_data.CACHE=manifest.parent
    from dataset import PartialDataset
    original=PartialDataset(manifest,'m4');new=PacketDataset(manifest)
    indices={0,len(new)//2,len(new)-1}
    for i,e in enumerate(new.entries):
        if e['anchors_one_current_hand']:
            _,p,_,_,anchors,_=new.episode(i);offset=int(np.where(p['hand_valid'][anchors].sum(axis=1)==1)[0][0])
            indices.add(int(new.ends[i-1] if i else 0)+offset);break
    for i in sorted(indices):
        a,b=original[i],new[i]
        for key in ['state','actions','state_valid','reliability']:np.testing.assert_allclose(a[key],b[key],atol=1e-6)
        np.testing.assert_array_equal(a['image']['base_0_rgb'],b['image']['base_0_rgb']);assert a['prompt']==b['prompt']
    report=dict(passed=True,windows=len(new),tested_indices=sorted(indices),same_rgb=True,same_targets=True,same_masks=True,same_current_only_prompt=True)
    write(OUT/'validation/cached_reader.json',report);print(json.dumps(report,indent=2));original.close();new.close()

if __name__=='__main__':main()
