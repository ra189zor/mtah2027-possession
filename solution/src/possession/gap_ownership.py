"""Offline ownership features from observed control on both sides of a gap."""
import numpy as np

from .validation import period_blocks


def anchor_features(basic, period, config):
    names=basic['names'].tolist()
    n=len(period)

    def col(name):
        j=names.index(name)
        return np.where(basic['valid'][:,j],basic['values'][:,j],np.nan)

    home,away=col('home_ball_distance_m'),col('away_ball_distance_m')
    margin=away-home
    raw=np.full(n,-1,dtype=np.int8)
    near=np.minimum(home,away)
    raw[(near<=config['control_distance_m']) & (margin>=config['owner_margin_m'])]=1
    raw[(near<=config['control_distance_m']) & (margin<=-config['owner_margin_m'])]=0
    anchors=np.full(n,np.nan,dtype=np.float32)
    minimum=max(1,int(round(config['anchor_confirm_s']*25)))
    for a,b in period_blocks(period):
        edges=np.r_[a,np.flatnonzero(raw[a+1:b]!=raw[a:b-1])+a+1,b]
        for low,high in zip(edges[:-1],edges[1:]):
            if raw[low]>=0 and high-low>=minimum:
                anchors[low:high]=raw[low]
    previous=np.full(n,-1,dtype=np.int64)
    following=previous.copy()
    for a,b in period_blocks(period):
        ids=np.arange(a,b)
        known=np.isfinite(anchors[a:b])
        previous[a:b]=np.maximum.accumulate(np.where(known,ids,-1))
        next_ids=np.minimum.accumulate(np.where(known,ids,n)[::-1])[::-1]
        following[a:b]=np.where(next_ids<n,next_ids,-1)
    out={}
    for prefix,index in (('previous',previous),('next',following)):
        found=index>=0
        safe=np.maximum(index,0)
        out['owner_'+prefix+'_home']=np.where(found,anchors[safe],np.nan)
        age=(np.arange(n)-index if prefix=='previous' else index-np.arange(n))/25
        out['owner_'+prefix+'_distance_s']=np.where(found,age,np.nan)
        for source in ('ball_x_m','ball_y_m','ball_speed_mps'):
            out['owner_'+prefix+'_'+source]=np.where(found,col(source)[safe],np.nan)
    past,future=out['owner_previous_home'],out['owner_next_home']
    age,until=out['owner_previous_distance_s'],out['owner_next_distance_s']
    both=np.isfinite(past)&np.isfinite(future)
    total=age+until
    fraction=np.divide(age,total,out=np.full(n,.5),where=total>0)
    out['owner_anchor_gap_s']=np.where(both,total,np.nan)
    out['owner_gap_fraction_offline']=np.where(both,fraction,np.nan)
    out['owner_anchor_agreement']=np.where(both,past==future,np.nan)
    out['owner_anchor_count_sides']=np.isfinite(past).astype(np.float32)+np.isfinite(future)
    bridge=np.full(n,.5,dtype=np.float32)
    geometry=np.isfinite(margin)
    bridge[geometry]=1/(1+np.exp(-np.clip(margin[geometry]/config['geometry_logit_scale_m'],-20,20)))
    bridge[np.isfinite(past)]=past[np.isfinite(past)]
    bridge[np.isfinite(future)&~np.isfinite(past)]=future[np.isfinite(future)&~np.isfinite(past)]
    bridge[both]=(past[both]*(1-fraction[both])+future[both]*fraction[both])
    confidence=config['anchor_probability']
    bridge=(1-confidence)+(2*confidence-1)*bridge
    bridge[period==0]=np.nan
    out['owner_bridge_home_probability']=bridge
    for name,values in out.items():
        values=np.array(values,dtype=np.float32,copy=True)
        values[period==0]=np.nan
        out[name]=values
    return out


def extract_owner_features(basic,dead_features,dead_names,period,config):
    if len(dead_features)!=len(period) or dead_features.shape[1]!=len(dead_names):
        raise ValueError('Owner/dead feature alignment mismatch')
    columns=anchor_features(basic,period,config)
    names=basic['names'].tolist()
    for suffix in ('ball_distance_m','nearest_player_speed_mps',
                   'ball_player_speed_difference_mps','nearest_closing_speed_mps'):
        for team in ('home','away'):
            source=team+'_'+suffix
            j=names.index(source)
            columns['owner_'+source]=np.where(basic['valid'][:,j],basic['values'][:,j],np.nan)
    values=np.column_stack([dead_features,*columns.values()]).astype(np.float32)
    return {'values':values,'names':np.r_[dead_names,list(columns)],
            'bridge_probability':columns['owner_bridge_home_probability'],
            'anchor_sides':columns['owner_anchor_count_sides']}


def compose_ownership(home_probability,dead_probability,period,confirmation_s,switch_confidence,dead_threshold=.5):
    n=len(period)
    if home_probability.shape!=(n,) or dead_probability.shape!=(n,):
        raise ValueError('Owner probability alignment mismatch')
    active=period>0
    if not np.isin(period,[0,1,2]).all(): raise ValueError('Unexpected periods')
    if not 0<=confirmation_s or not .5<=switch_confidence<=1 or not 0<dead_threshold<1:
        raise ValueError('Invalid ownership settings')
    for p in (home_probability,dead_probability):
        if not np.isfinite(p[active]).all() or ((p[active]<0)|(p[active]>1)).any():
            raise ValueError('Invalid playing-period probability')
    labels=np.full(n,'X',dtype='<U1')
    confidence=np.maximum(home_probability,1-home_probability)
    dead=active&(dead_probability>=dead_threshold)
    required=max(1,int(round(confirmation_s*25)))
    for a,b in period_blocks(period):
        owner,pending,run=None,None,0
        for i in range(a,b):
            if dead[i]:
                labels[i]='D'; owner,pending,run=None,None,0
                continue
            proposed='H' if home_probability[i]>=.5 else 'A'
            if owner is None:
                owner=proposed
            if confidence[i]>=switch_confidence and proposed!=owner:
                if pending==proposed: run+=1
                else: pending,run=proposed,1
                if run>=required:
                    owner=proposed
                    labels[i-run+1:i+1]=owner
                    pending,run=None,0
            else:
                pending,run=None,0
            labels[i]=owner
    if (labels[active]=='X').any() or not np.array_equal(labels=='D',dead):
        raise ValueError('Owner recovery changed the frozen dead mask or left unknowns')
    return {'labels':labels,'low_confidence':active&~dead&(confidence<switch_confidence)}
