"""Short observed control evidence; no reference labels or target counts accepted."""
import numpy as np
from .validation import period_blocks


def control_log_odds(matrix,names,period,config):
    if len(matrix)!=len(period) or matrix.shape[1]!=len(names):raise ValueError('Control feature alignment mismatch')
    if not np.isin(period,[0,1,2]).all():raise ValueError('Invalid periods')
    names=list(names)
    col=lambda name:np.asarray(matrix[:,names.index(name)],dtype=np.float64)
    hd=col('owner_home_ball_distance_m');ad=col('owner_away_ball_distance_m')
    speed=col('ball_speed_mps')
    hs=col('owner_home_ball_player_speed_difference_mps');ass=col('owner_away_ball_player_speed_difference_mps')
    valid=np.isfinite(hd)&np.isfinite(ad)&np.isfinite(speed)&(speed<=config['maximum_ball_speed_mps'])&(speed>=0)&(period>0)
    home=valid&(hd>=0)&(hd<=config['control_distance_m'])&(ad-hd>=config['opponent_margin_m'])&np.isfinite(hs)&(np.abs(hs)<=config['maximum_speed_difference_mps'])
    away=valid&(ad>=0)&(ad<=config['control_distance_m'])&(hd-ad>=config['opponent_margin_m'])&np.isfinite(ass)&(np.abs(ass)<=config['maximum_speed_difference_mps'])
    raw=home.astype(np.int8)-away.astype(np.int8);result=np.zeros(len(period),dtype=np.float32)
    minimum=config['confirmation_frames'];probability=config['control_probability']
    if not isinstance(minimum,int) or minimum<1 or not .5<probability<1:raise ValueError('Invalid control settings')
    magnitude=float(np.log(probability/(1-probability)))
    for a,b in period_blocks(period):
        edges=np.r_[a,np.flatnonzero(raw[a+1:b]!=raw[a:b-1])+a+1,b]
        for low,high in zip(edges[:-1],edges[1:]):
            if raw[low] and high-low>=minimum:result[low:high]=raw[low]*magnitude
    return result


def combine_ownership(home_probability,control,strength,period):
    if home_probability.shape!=control.shape or control.shape!=period.shape:raise ValueError('Control/probability mismatch')
    if not np.isfinite(strength) or strength<0:raise ValueError('Invalid control strength')
    active=period>0
    if not np.isfinite(control).all():raise ValueError('Nonfinite control evidence')
    if not np.isfinite(home_probability[active]).all() or ((home_probability[active]<0)|(home_probability[active]>1)).any():
        raise ValueError('Invalid learned home probability')
    if strength==0:return home_probability.copy()
    p=np.clip(np.where(active,home_probability,.5),1e-6,1-1e-6).astype(np.float64)
    logits=np.log(p)-np.log1p(-p)+strength*control
    result=(1/(1+np.exp(-np.clip(logits,-40,40)))).astype(np.float32)
    # Preserve the original probabilities exactly wherever the residual is absent.
    result[control==0]=home_probability[control==0]
    result[~active]=home_probability[~active]
    return result


def profiles(config):
    return [{'id':f'control{strength}_switch{switch}','strength':strength,'switch':switch}
            for strength in config['control_strengths'] for switch in config['switch_costs']]
