"""Exact three-state offline Viterbi decoding; references never enter this API."""
import numpy as np

from .validation import period_blocks


def emission_costs(home_probability,dead_probability,period,epsilon=1e-6):
    n=len(period)
    if home_probability.shape!=(n,) or dead_probability.shape!=(n,) or not np.isin(period,[0,1,2]).all():
        raise ValueError('Invalid probability/period alignment')
    if not 0<epsilon<.5:raise ValueError('Invalid probability clipping')
    active=period>0
    for p in (home_probability,dead_probability):
        if not np.isfinite(p[active]).all() or ((p[active]<0)|(p[active]>1)).any():
            raise ValueError('Invalid playing-period probability')
    h=np.clip(np.where(active,home_probability,.5),epsilon,1-epsilon).astype(np.float64)
    d=np.clip(np.where(active,dead_probability,.5),epsilon,1-epsilon).astype(np.float64)
    probabilities=np.column_stack([(1-d)*h,(1-d)*(1-h),d])
    return -np.log(probabilities)


def transition_costs(home,dead,profile):
    stop,restart,switch=profile['stop'],profile['restart'],profile['switch']
    if not all(np.isfinite(c) and c>=0 for c in (stop,restart,switch)):
        raise ValueError('Transition costs must be finite and nonnegative')
    if profile['mode']=='fixed':
        stop_scale=restart_scale=to_home_scale=to_away_scale=1.
    elif profile['mode']=='evidence':
        stop_scale,restart_scale=1.5-dead,.5+dead
        to_home_scale,to_away_scale=1.5-home,.5+home
    else:raise ValueError('Unknown decoder transition mode')
    return np.array([[0,switch*to_away_scale,stop*stop_scale],
                     [switch*to_home_scale,0,stop*stop_scale],
                     [restart*restart_scale,restart*restart_scale,0]],dtype=float)


def decode(home_probability,dead_probability,period,profile,epsilon=1e-6):
    costs=emission_costs(home_probability,dead_probability,period,epsilon)
    # Validate settings once; use scalar arithmetic in the native-frame loop.
    transition_costs(.5,.5,profile)
    stop,restart,switch=float(profile['stop']),float(profile['restart']),float(profile['switch'])
    adaptive=profile['mode']=='evidence'
    h=home_probability.tolist();d=dead_probability.tolist()
    eh,ea,ed=(costs[:,j].tolist() for j in range(3))
    labels=np.full(len(period),'X',dtype='<U1')
    total=0.
    state_labels=np.array(['H','A','D'])
    for a,b in period_blocks(period):
        back=np.zeros((b-a,3),dtype=np.uint8)
        sh,sa,sd=eh[a],ea[a],ed[a]
        for t in range(a+1,b):
            if adaptive:
                st=stop*(1.5-d[t]);rs=restart*(.5+d[t])
                to_h=switch*(1.5-h[t]);to_a=switch*(.5+h[t])
            else:
                st,rs,to_h,to_a=stop,restart,switch,switch
            # Stay wins exact ties, then remaining source order is deterministic.
            bh,ph=sh,0
            if sa+to_h<bh:bh,ph=sa+to_h,1
            if sd+rs<bh:bh,ph=sd+rs,2
            ba,pa=sa,1
            if sh+to_a<ba:ba,pa=sh+to_a,0
            if sd+rs<ba:ba,pa=sd+rs,2
            bd,pd=sd,2
            if sh+st<bd:bd,pd=sh+st,0
            if sa+st<bd:bd,pd=sa+st,1
            back[t-a]=ph,pa,pd
            sh,sa,sd=bh+eh[t],ba+ea[t],bd+ed[t]
        state=int(np.argmin([sh,sa,sd]));total+=min(sh,sa,sd)
        for t in range(b-1,a-1,-1):
            labels[t]=state_labels[state]
            if t>a:state=int(back[t-a,state])
    if (labels[period>0]=='X').any() or (labels[period==0]!='X').any():
        raise ValueError('Decoded state violates published periods')
    return {'labels':labels,'path_cost':float(total)}


def candidates(config):
    result=[{'id':'frame_map','mode':'fixed','stop':0,'restart':0,'switch':0}]
    for mode in config['modes']:
        for stop,restart in config['stop_restart_costs']:
            for switch in config['owner_switch_costs']:
                result.append({'id':f'{mode}_s{stop}_r{restart}_o{switch}',
                               'mode':mode,'stop':stop,'restart':restart,'switch':switch})
    return result
