"""Label-free offline event evidence, and separate supervised switch targets."""
import numpy as np
from .validation import period_blocks
from .dead_model import training_rows


def physical_columns(matrix,names):
    names=list(names)
    chosen=names[:51]+names[275:]
    required=('ball_speed_mps','owner_home_ball_distance_m','owner_away_ball_distance_m')
    if matrix.shape[1]!=298 or any(name not in chosen for name in required):raise ValueError('Unexpected switch feature schema')
    indices=[names.index(name) for name in chosen]
    return np.asarray(matrix[:,indices],dtype=np.float32),chosen


def shifted(values,period,lag):
    out=np.full(len(values),np.nan,dtype=np.float32)
    for a,b in period_blocks(period):
        if lag>=0:
            if a+lag<b:out[a+lag:b]=values[a:b-lag]
        else:
            step=-lag
            if a+step<b:out[a:b-step]=values[a+step:b]
    return out


def extract_event_features(physical,names,home,dead,period,config):
    n=len(period)
    if physical.shape!=(n,len(names)) or home.shape!=(n,) or dead.shape!=(n,):raise ValueError('Switch evidence alignment')
    active=period>0
    if not np.isin(period,[0,1,2]).all():raise ValueError('Invalid switch periods')
    for p in (home,dead):
        if not np.isfinite(p[active]).all() or ((p[active]<0)|(p[active]>1)).any():raise ValueError('Invalid upstream probability')
    col=lambda name:physical[:,list(names).index(name)]
    margin=col('owner_away_ball_distance_m')-col('owner_home_ball_distance_m')
    h=np.where(active,home,np.nan).astype(np.float32);d=np.where(active,dead,np.nan).astype(np.float32)
    columns=[h,d,np.abs(h-.5),margin];extra=['home_probability','dead_probability','owner_confidence','distance_advantage']
    for lag in config['lags_frames']:
        for name,value in [('home',h),('dead',d),('distance_advantage',margin)]:
            past=shifted(value,period,lag);future=shifted(value,period,-lag)
            columns.extend([past,future,future-past])
            extra.extend([f'{name}_past_{lag}',f'{name}_future_{lag}',f'{name}_change_{lag}'])
    result=np.column_stack([physical,*columns]).astype(np.float32)
    result[~active]=np.nan
    if np.isinf(result).any():raise ValueError('Infinite event feature')
    return result,list(names)+extra


def event_targets(labels,period,radius):
    if labels.shape!=period.shape or not isinstance(radius,int) or radius<0:raise ValueError('Invalid event targets')
    result=np.where((period>0)&(labels!='X'),0,-1).astype(np.int8)
    alive_period=np.where(np.isin(labels,['H','A']),period,0)
    centers=[]
    for a,b in period_blocks(alive_period):
        changes=np.flatnonzero(labels[a+1:b]!=labels[a:b-1])+a+1
        if not len(changes):continue
        ids=np.arange(a,b);right=np.searchsorted(changes,ids)
        before=changes[np.maximum(0,right-1)];after=changes[np.minimum(len(changes)-1,right)]
        nearest=np.where(np.abs(ids-before)<=np.abs(after-ids),before,after)
        near=np.abs(ids-nearest)<=radius
        result[ids[near]]=np.where(labels[nearest[near]]=='H',1,2)
        centers.extend((int(t),a,b) for t in changes)
    return result,centers


def event_training_rows(labels,period,seed,config):
    targets,centers=event_targets(labels,period,config['event_radius_frames'])
    rows=training_rows(labels,period,config['training_stride_frames'],seed).tolist()
    for t,a,b in centers:
        rows.extend(t+offset for offset in config['positive_sample_offsets_frames'] if a<=t+offset<b)
    rows=np.unique(rows).astype(np.int64);rows=rows[targets[rows]>=0]
    full=np.bincount(targets[targets>=0],minlength=3);sample=np.bincount(targets[rows],minlength=3)
    if (sample==0).any():raise ValueError('Missing switch training class')
    weights=(full/sample)[targets[rows]].astype(float);weights/=weights.sum()
    if not np.allclose(np.bincount(targets[rows],weights=weights,minlength=3),full/full.sum()):
        raise ValueError('Switch sampling changed class proportions')
    return rows,targets[rows],weights,{'events':len(centers),'class_frame_counts':full.tolist(),
                                      'class_sample_counts':sample.tolist(),'sampled_rows':len(rows)}
