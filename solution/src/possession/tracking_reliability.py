"""Experimental input-only quality and long-gap context. Does not edit observations."""
import numpy as np
from .validation import period_blocks
from .switch_events import shifted


def quality_features(inputs):
    ball=inputs['ball_xy'];players=inputs['players_xy'];teams=inputs['team_of_slot'];period=inputs['period']
    n=len(period);visible=np.isfinite(ball).all(1);cols={}
    cols['visible']=visible.astype(float)
    for team in (0,1):
        distance=np.linalg.norm(players[:,teams==team]-ball[:,None],axis=2)
        distance=np.where(np.isfinite(distance),distance,np.inf)
        near=distance.min(1);near[~np.isfinite(near)]=np.nan
        cols[f'near_{team}']=near
        cols[f'players_near_{team}']=(distance<4).sum(1).astype(float)
    cols['margin']=cols['near_1']-cols['near_0']
    for lag in (1,5,25):
        past=np.column_stack([shifted(ball[:,j],period,lag) for j in (0,1)])
        future=np.column_stack([shifted(ball[:,j],period,-lag) for j in (0,1)])
        cols[f'speed_past_{lag}']=np.linalg.norm(ball-past,axis=1)*25/lag
        cols[f'speed_future_{lag}']=np.linalg.norm(future-ball,axis=1)*25/lag
        cols[f'bridge_error_{lag}']=np.linalg.norm(ball-(past+future)/2,axis=1)
    pitch=inputs['pitch_m']
    cols['outside_distance']=np.maximum.reduce([-ball[:,0],ball[:,0]-pitch[0],-ball[:,1],ball[:,1]-pitch[1],np.zeros(n)])
    previous=np.full(n,-1,dtype=int);following=previous.copy()
    for a,b in period_blocks(period):
        ids=np.arange(a,b);known=visible[a:b]
        previous[a:b]=np.maximum.accumulate(np.where(known,ids,-1))
        after=np.minimum.accumulate(np.where(known,ids,n)[::-1])[::-1]
        following[a:b]=np.where(after<n,after,-1)
    ids=np.arange(n)
    for label,index in [('previous',previous),('following',following)]:
        valid=index>=0;safe=np.maximum(index,0)
        cols[label+'_distance_s']=np.where(valid,np.abs(ids-index)/25,np.nan)
        cols[label+'_margin']=np.where(valid,cols['margin'][safe],np.nan)
        for axis in (0,1):
            # Current players' distance to the last/next observed ball location.
            for team in (0,1):
                xy=players[:,teams==team,axis];count=np.isfinite(xy).sum(1)
                centre=np.divide(np.nansum(xy,axis=1),count,out=np.full(n,np.nan),where=count>0)
                cols[f'{label}_team_{team}_axis_{axis}']=np.where(valid,centre-ball[safe,axis],np.nan)
    duration=cols['previous_distance_s']+cols['following_distance_s']
    cols['gap_duration_s']=duration
    cols['gap_fraction']=np.divide(cols['previous_distance_s'],duration,out=np.zeros(n),where=duration>0)
    matrix=np.column_stack(list(cols.values())).astype(np.float32);matrix[period<=0]=np.nan
    if np.isinf(matrix).any():raise ValueError('Infinite quality feature')
    return matrix,list(cols)


def inject_training_ghosts(inputs,config,seed):
    out={k:v.copy() for k,v in inputs.items()};mask=np.zeros(len(inputs['period']),bool)
    rng=np.random.default_rng(seed)
    for a,b in period_blocks(inputs['period']):
        starts=np.flatnonzero(rng.random(b-a)<1/(25*config['training_ghost_interval_s']))+a
        for start in starts:
            end=min(b,start+int(25*rng.uniform(*config['training_ghost_duration_s'])))
            angle=rng.uniform(0,2*np.pi);offset=rng.uniform(*config['training_ghost_offset_m'])*np.array([np.cos(angle),np.sin(angle)])
            source=out['ball_xy'][start:end].copy()
            if rng.random()<.5 and np.isfinite(source[0]).all():source[:]=source[0]
            visible=np.isfinite(source).all(1)
            out['ball_xy'][start:end][visible]=source[visible]+offset
            mask[start:end]|=visible
    return out,mask


def combined_features(event,quality,bad_probability):
    if len(event)!=len(quality) or bad_probability.shape!=(len(event),):raise ValueError('Feature alignment')
    return np.column_stack([event,quality,bad_probability]).astype(np.float32)


def compose_candidates(home,events,gap_home,new_events,missing,bad,blend=.5):
    if not 0<=blend<=1:raise ValueError('Invalid blend')
    damp=1-blend*bad
    reliable_home=.5+(home-.5)*damp
    reliable_events=events.copy();reliable_events[:,1:]*=damp[:,None];reliable_events[:,0]=1-reliable_events[:,1:].sum(1)
    gap_home=np.where(missing,(1-blend)*home+blend*gap_home,home)
    boundary=(1-blend)*events+blend*new_events
    combined_home=np.where(missing,gap_home,reliable_home)
    combined_events=boundary.copy();combined_events[:,1:]*=damp[:,None];combined_events[:,0]=1-combined_events[:,1:].sum(1)
    return {'event4':(home,events),'quality':(reliable_home,reliable_events),
            'gap':(gap_home,events),'boundary':(home,boundary),'combined':(combined_home,combined_events)}
