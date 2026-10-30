"""Exact Viterbi with learned directed switch rewards; no changes to old decoder."""
import numpy as np
from .decoder import emission_costs
from .validation import period_blocks


def directed_costs(event_probability,base_switch,strength,config):
    q=np.asarray(event_probability,dtype=np.float64)
    if q.ndim!=2 or q.shape[1]!=3 or not np.isfinite(q).all() or (q<0).any() or (q>1).any() or not np.allclose(q.sum(axis=1),1,atol=1e-5):
        raise ValueError('Expected normalized no-event/to-home/to-away probabilities')
    if not np.isfinite(base_switch) or base_switch<0 or not np.isfinite(strength) or strength<0:raise ValueError('Invalid reward scale')
    threshold=config['reward_threshold']
    if not 0<threshold<1 or config['minimum_round_trip_cost']<0:raise ValueError('Invalid event reward settings')
    if strength==0:return np.full(len(q),base_switch),np.full(len(q),base_switch)
    p=np.clip(q[:,1:],1e-6,1-1e-6)
    reward=np.maximum(0,np.log(p)-np.log1p(-p)-np.log(threshold/(1-threshold)))
    costs=np.maximum(config['minimum_directed_switch_cost'],base_switch-strength*reward)
    correction=np.maximum(0,(config['minimum_round_trip_cost']-costs.sum(axis=1))/2)
    costs+=correction[:,None]
    return costs[:,0],costs[:,1]


def decode_events(home,dead,period,event_probability,profile,strength,config,epsilon=1e-6):
    if profile['mode']!='fixed':raise ValueError('Event decoder requires frozen fixed base costs')
    emission=emission_costs(home,dead,period,epsilon)
    if event_probability.shape!=(len(period),3):raise ValueError('Event probability/frame mismatch')
    to_home,to_away=directed_costs(event_probability,profile['switch'],strength,config)
    stop,restart=float(profile['stop']),float(profile['restart'])
    if not np.isfinite([stop,restart]).all() or min(stop,restart)<0:raise ValueError('Invalid stop/restart costs')
    eh,ea,ed=(emission[:,j].tolist() for j in range(3));ch=to_home.tolist();ca=to_away.tolist()
    labels=np.full(len(period),'X',dtype='<U1');state_labels=np.array(['H','A','D']);total=0.
    for a,b in period_blocks(period):
        back=np.zeros((b-a,3),dtype=np.uint8);sh,sa,sd=eh[a],ea[a],ed[a]
        for t in range(a+1,b):
            bh,ph=sh,0
            if sa+ch[t]<bh:bh,ph=sa+ch[t],1
            if sd+restart<bh:bh,ph=sd+restart,2
            ba,pa=sa,1
            if sh+ca[t]<ba:ba,pa=sh+ca[t],0
            if sd+restart<ba:ba,pa=sd+restart,2
            bd,pd=sd,2
            if sh+stop<bd:bd,pd=sh+stop,0
            if sa+stop<bd:bd,pd=sa+stop,1
            back[t-a]=ph,pa,pd;sh,sa,sd=bh+eh[t],ba+ea[t],bd+ed[t]
        state=int(np.argmin([sh,sa,sd]));total+=min(sh,sa,sd)
        for t in range(b-1,a-1,-1):
            labels[t]=state_labels[state]
            if t>a:state=int(back[t-a,state])
    if (labels[period>0]=='X').any():raise ValueError('Event decoder left playing-period unknowns')
    return {'labels':labels,'path_cost':float(total)}
