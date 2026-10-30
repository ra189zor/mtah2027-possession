"""Input-only contextual stress and explicitly evaluation-only label sensitivity."""
import numpy as np

from .validation import check_inputs, corrupt, period_blocks, summarize_inputs


def contextual_corrupt(inputs, base_regime, config, seed):
    check_inputs(inputs)
    out=corrupt(inputs,base_regime,seed)
    rng=np.random.default_rng(seed+701927)
    n=len(inputs['period']);fps=int(inputs['fps']);pitch=inputs['pitch_m']
    masks={k:np.zeros(n,dtype=bool) for k in ('risk','extra_dropout','ghost','wrong_team')}
    ball=inputs['ball_xy'];players=inputs['players_xy'];teams=inputs['team_of_slot']
    for a,b in period_blocks(inputs['period']):
        speed=np.full(b-a,np.nan)
        speed[1:]=np.linalg.norm(np.diff(ball[a:b],axis=0),axis=1)*fps
        distances=np.linalg.norm(players[a:b]-ball[a:b,None,:],axis=2)
        distances[~np.isfinite(distances)]=np.inf
        unsupported=distances.min(axis=1)>config['unsupported_ball_distance_m']
        box=((ball[a:b,0]<16.5)|(ball[a:b,0]>pitch[0]-16.5)) & (np.abs(ball[a:b,1]-pitch[1]/2)<20.16)
        crowded=(distances<config['crowding_radius_m']).sum(axis=1)>=config['crowding_players']
        risk=((speed>config['fast_ball_mps']) & unsupported) | (box & crowded)
        masks['risk'][a:b]=risk
        # Independent burst starts; cumulative coverage may overlap. Risk uses original inputs only.
        chance=(1+(config['risk_multiplier']-1)*risk)/(fps*config['extra_gap_event_mean_s'])
        starts=np.flatnonzero(rng.random(b-a)<chance)+a
        for t in starts:
            length=max(1,int(round(fps*rng.lognormal(np.log(config['extra_gap_median_s']),config['extra_gap_sigma']))))
            stop=min(b,t+length);out['ball_xy'][t:stop]=np.nan;masks['extra_dropout'][t:stop]=True
        for t in np.flatnonzero(rng.random(b-a)<1/(fps*config['ghost_event_mean_s']))+a:
            length=max(1,int(round(fps*rng.lognormal(np.log(config['ghost_median_s']),config['ghost_sigma']))))
            stop=min(b,t+length);theta=rng.uniform(0,2*np.pi)
            delta=rng.uniform(*config['ghost_offset_m'])*np.array([np.cos(theta),np.sin(theta)])
            source=ball[t:stop].copy();missing=~np.isfinite(source).all(axis=1)
            source[missing]=rng.uniform([0,0],pitch,size=(missing.sum(),2))
            # A moving offset track can persist through a real dropout; do not clip at boundaries.
            out['ball_xy'][t:stop]=source+delta;masks['ghost'][t:stop]=True
        for t in np.flatnonzero(rng.random(b-a)<1/(fps*config['wrong_team_event_mean_s']))+a:
            length=max(1,int(round(fps*rng.lognormal(np.log(config['wrong_team_median_s']),config['wrong_team_sigma']))))
            stop=min(b,t+length)
            observed=np.isfinite(out['players_xy'][t]).all(axis=1)
            home=np.flatnonzero(observed&(teams==0));away=np.flatnonzero(observed&(teams==1))
            if not len(home) or not len(away):continue
            h,g=int(rng.choice(home)),int(rng.choice(away))
            out['players_xy'][t:stop,[h,g]]=out['players_xy'][t:stop,[g,h]]
            masks['wrong_team'][t:stop]=True
    active=inputs['period']>0
    audit={**summarize_inputs(out),**{k+'_fraction':float(v[active].mean()) for k,v in masks.items()}}
    audit['extra_gap_start_risk_multiplier']=config['risk_multiplier']
    return out,masks,audit


def absorb_short_opposition(labels,period,seconds=3.):
    """Alternate evaluation references; preserve all original dead/ungraded frames."""
    if labels.shape!=period.shape or seconds<=0:raise ValueError('Invalid label sensitivity inputs')
    out=labels.copy();absorbed=0
    for a,b in period_blocks(period):
        edges=np.r_[a,np.flatnonzero(out[a+1:b]!=out[a:b-1])+a+1,b]
        runs=[[int(low),int(high),out[low]] for low,high in zip(edges[:-1],edges[1:])]
        while True:
            eligible=[i for i in range(1,len(runs)-1) if runs[i][2] in ('H','A') and
                runs[i-1][2]==runs[i+1][2] and runs[i-1][2] in ('H','A') and
                runs[i-1][2]!=runs[i][2] and runs[i][1]-runs[i][0]<seconds*25]
            if not eligible:break
            i=min(eligible,key=lambda j:(runs[j][1]-runs[j][0],runs[j][0]))
            low,high,_=runs[i];out[low:high]=runs[i-1][2];absorbed+=1
            runs[i-1:i+2]=[[runs[i-1][0],runs[i+1][1],runs[i-1][2]]]
    if not np.array_equal(out[np.isin(labels,['D','X'])],labels[np.isin(labels,['D','X'])]):
        raise ValueError('Sensitivity changed dead/ungraded reference')
    return out,{'absorbed_spells':absorbed,'changed_frames':int((out!=labels).sum())}


def excluded_pairs(matches):
    if len(matches)!=len(set(matches)) or len(matches)<3:raise ValueError('Expected distinct match IDs')
    return [(a,b) for i,a in enumerate(matches) for b in matches[i+1:]]


def nested_training_matches(matches,excluded):
    if not excluded or not set(excluded)<=set(matches):raise ValueError('Unknown/empty exclusion')
    result=[m for m in matches if m not in excluded]
    if not result or set(result)&set(excluded):raise ValueError('Invalid training fold')
    return result
