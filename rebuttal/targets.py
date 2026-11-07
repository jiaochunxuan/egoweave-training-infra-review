"""Motion targets and differentiable world composition for controlled ablations."""
import numpy as np

SCALE=.09564778953790665
CONDITIONS=('full','no_explicit_head','current_head','no_scale','no_geometry','global_axis')


def rot6(r,xp=np):
    return xp.concatenate([r[..., :,0],r[..., :,1]],axis=-1)


def matrix(v,xp):
    a,b=v[...,:3],v[...,3:6]
    a=a/xp.sqrt(xp.maximum(xp.sum(a*a,axis=-1,keepdims=True),1e-12))
    b=b-xp.sum(a*b,axis=-1,keepdims=True)*a
    b=b/xp.sqrt(xp.maximum(xp.sum(b*b,axis=-1,keepdims=True),1e-12))
    return xp.stack([a,b,xp.cross(a,b)],axis=-1)


def local_scale(local):
    d=np.linalg.norm(local-local[...,:1,:],axis=-1)
    valid=d>1e-6;count=valid.sum(axis=-1)
    fallback=np.divide(np.where(valid,d,0).sum(axis=-1),count,out=np.ones_like(d[...,0]),where=count>0)
    return np.where(d[...,9]>1e-6,d[...,9],np.maximum(fallback,1e-6)).astype(np.float32)


def encode(packet,anchors,condition='full',horizon=32):
    if condition not in CONDITIONS:raise ValueError(condition)
    p=np.asarray(packet['pose153'],dtype=np.float32);anchors=np.asarray(anchors)
    if p.ndim!=2 or p.shape[1]!=162 or anchors.ndim!=1 or anchors.dtype.kind not in 'iu':raise ValueError('Invalid poses or anchors')
    if horizon<=0 or np.any(anchors<0) or np.any(anchors+horizon>=len(p)):raise ValueError('Window crosses episode boundary')
    ix=anchors[:,None]+np.arange(1,horizon+1);now=p[anchors];future=p[ix]
    def features(rows,reference=None):
        shape=rows.shape[:-1];hr=rows[...,:9].reshape(*shape,3,3);hp=rows[...,9:12]
        rr,rp=(hr,hp) if reference is None else reference
        parts=[hp,rot6(hr)];local=[]
        for ps,rs,ls in [(12,18,36),(15,27,99)]:
            wr=rows[...,rs:rs+9].reshape(*shape,3,3)
            wp=np.einsum('...ji,...j->...i',rr,rows[...,ps:ps+3]-rp)
            relative=rr.swapaxes(-1,-2)@wr
            parts.append(np.concatenate([wp,rot6(relative)],axis=-1))
            loc=rows[...,ls:ls+63].reshape(*shape,21,3);loc=loc-loc[...,:1,:]
            if condition!='no_scale':loc=loc/local_scale(loc)[...,None,None]
            local.append(loc.reshape(*shape,63))
        return np.concatenate([*parts,*local],axis=-1)
    h0=now[:,:9].reshape(-1,3,3);hp0=now[:,9:12]
    progress=anchors.astype(np.float64)/max(len(p)-1,1)
    s=np.concatenate([features(now),np.stack([progress,np.sin(2*np.pi*progress),np.cos(2*np.pi*progress)],axis=-1).astype(np.float32)],axis=-1)
    reference=(h0[:,None],hp0[:,None]) if condition=='current_head' else None
    a=features(future,reference);a[...,:3]=future[...,9:12]-hp0[:,None]
    a[...,3:9]=rot6(h0.swapaxes(-1,-2)[:,None]@future[...,:9].reshape(len(anchors),horizon,3,3))
    sm,am=masks(packet,anchors,condition,horizon)
    return np.where(sm,s,0).astype(np.float32),np.where(am,a,0).astype(np.float32),sm,am


def masks(packet,anchors,condition='full',horizon=32):
    anchors=np.asarray(anchors,dtype=np.int64);ix=anchors[:,None]+np.arange(1,horizon+1)
    hv=np.asarray(packet['head_valid'],bool);hands=np.asarray(packet['hand_valid'],bool)
    sm=np.zeros((len(anchors),156),bool);am=np.zeros((len(anchors),horizon,153),bool)
    sm[:,:9]=hv[anchors,None];sm[:,153:]=True
    am[...,:9]=(hv[anchors,None]&hv[ix])[...,None]
    for side,w,l in [(0,9,27),(1,18,90)]:
        sm[:,w:w+9]=(hv[anchors]&hands[anchors,side])[:,None];sm[:,l:l+63]=hands[anchors,side,None]
        # Freeze label support across ablations, even when the new reference
        # would mathematically allow additional wrist supervision.
        reference_ok=hv[ix]
        am[...,w:w+9]=(reference_ok&hands[ix,side])[...,None];am[...,l:l+63]=hands[ix,side,None]
    return sm,am


def action_weights(condition):
    if condition not in CONDITIONS:raise ValueError(condition)
    w=np.array([3.]*3+[1.5]*6+([20.]*3+[4.]*6)*2+[.3]*126,dtype=np.float32)
    if condition=='no_explicit_head':w[:9]=0
    return w


def head_ablation_flow(error,reliability,reducer,xp):
    """Remove head numerators while preserving the full objective denominator."""
    reduced=reducer(error,loss_type='mse',reliability=reliability,action_loss_weights=action_weights('no_explicit_head'))
    rel=xp.ones_like(error) if reliability is None else reliability
    kept=xp.sum(rel*xp.asarray(action_weights('no_explicit_head')),axis=-1)
    full=xp.sum(rel*xp.asarray(action_weights('full')),axis=-1)
    return reduced*kept/xp.maximum(full,1e-6)


def decode(state,actions,condition='full',xp=np):
    """Return [B,H,2,21,3] world landmarks at the objective's stated hand scale."""
    h0=matrix(state[:,3:9],xp);p0=state[:,:3]
    if condition=='current_head':hr=h0[:,None];hp=p0[:,None]
    else:hr=h0[:,None]@matrix(actions[...,3:9],xp);hp=p0[:,None]+actions[...,:3]
    result=[]
    for w,l in [(9,27),(18,90)]:
        wp=hp+xp.einsum('bhij,bhj->bhi',hr,actions[...,w:w+3])
        wr=hr@matrix(actions[...,w+3:w+9],xp)
        loc=actions[...,l:l+63].reshape(*actions.shape[:-1],21,3)
        if condition!='no_scale':loc=loc*SCALE
        result.append(wp[...,None,:]+xp.einsum('bhij,bhnj->bhni',wr,loc))
    return xp.stack(result,axis=2)


def geometric_loss(state,pred,target,reliability,condition,xp):
    pred_xyz=decode(state,pred,condition,xp);target_xyz=decode(state,target,condition,xp)
    side_loss=xp.mean(xp.square(pred_xyz-target_xyz),axis=(-1,-2))
    valid=[]
    for w,l in [(9,27),(18,90)]:
        good=xp.all(reliability[...,w:w+9]>0,axis=-1)&xp.all(reliability[...,l:l+63]>0,axis=-1)
        if condition!='current_head':good=good&xp.all(reliability[...,:9]>0,axis=-1)
        valid.append(good)
    valid=xp.stack(valid,axis=-1);denom=xp.sum(valid,axis=-1)
    loss=xp.sum(xp.where(valid,side_loss,0),axis=-1)/xp.maximum(denom,1)
    available=denom>0
    return loss*available.shape[-1]/xp.maximum(xp.sum(available,axis=-1,keepdims=True),1)


def rescale_reference_norms(stats,condition):
    """Physical local targets use one fixed unit conversion, not per-frame scales."""
    import copy
    out=copy.deepcopy(stats)
    if condition=='no_scale':
        for key in ['state','actions']:
            for field in ['mean','std']:
                a=np.array(out[key][field],float);a[27:153]*=SCALE;out[key][field]=a.tolist()
    return out
