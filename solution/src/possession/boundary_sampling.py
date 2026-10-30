"""Supervised training sampling only; reference arrays never enter model inference."""
import numpy as np
from .dead_model import training_rows
from .validation import period_blocks


def owner_training_rows(labels,period,seed,config):
    if labels.shape!=period.shape:raise ValueError('Sampling labels/period alignment')
    alive=np.where(np.isin(labels,['H','A']),labels,'X')
    sampled=training_rows(alive,period,config['sampling_stride_frames'],seed).tolist()
    spell_ids=np.full(len(labels),-1,dtype=np.int64);spells=0;original=set(sampled);original_spells=0
    for a,b in period_blocks(period):
        edges=np.r_[a,np.flatnonzero(labels[a+1:b]!=labels[a:b-1])+a+1,b]
        for low,high in zip(edges[:-1],edges[1:]):
            if labels[low] not in ('H','A'):continue
            spell_ids[low:high]=spells;spells+=1
            original_spells+=any(i in original for i in range(int(low),int(high)))
            for offset in config['inside_boundary_offsets_frames']:
                if offset<0:raise ValueError('Negative boundary offset')
                if low+offset<high:sampled.append(int(low+offset))
                if high-1-offset>=low:sampled.append(int(high-1-offset))
            if config['include_spell_midpoint']:sampled.append(int((low+high-1)//2))
    rows=np.unique(sampled).astype(np.int64)
    if not len(rows) or (spell_ids[rows]<0).any():raise ValueError('Invalid owner samples')
    counts=np.bincount(spell_ids[rows],minlength=spells)
    if (counts==0).any():raise ValueError('An alive spell is absent from training samples')
    weights=1./counts[spell_ids[rows]]
    if not np.allclose(np.bincount(spell_ids[rows],weights=weights,minlength=spells),1.):
        raise ValueError('Sampled spell weights are unequal')
    weights/=weights.sum()
    return rows,weights,{'provider_spells':spells,'original_sampled_spells':int(original_spells),
                        'represented_spells':int((counts>0).sum()),'sampled_rows':len(rows)}
