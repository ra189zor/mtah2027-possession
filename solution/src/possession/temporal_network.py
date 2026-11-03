"""Small offline TCN and preprocessing; inference accepts features, never labels."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .validation import period_blocks


def fit_normalizer(matrices):
    count = total = squared = None
    for matrix in matrices:
        x = np.asarray(matrix, dtype=np.float64)
        if x.ndim != 2 or np.isinf(x).any():
            raise ValueError('Normalization expects finite-or-missing feature matrices')
        known = np.isfinite(x)
        z = np.where(known, x, 0.)
        if count is None:
            count = np.zeros(x.shape[1], dtype=np.int64)
            total = np.zeros(x.shape[1], dtype=np.float64)
            squared = total.copy()
        if x.shape[1] != len(count): raise ValueError('Feature schema changed')
        count += known.sum(0); total += z.sum(0); squared += (z*z).sum(0)
    if count is None: raise ValueError('No normalization training inputs')
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    variance = np.maximum(0, np.divide(squared, count, out=np.zeros_like(total), where=count > 0)-mean*mean)
    scale = np.sqrt(variance)
    scale[(count < 2) | (scale < 1e-3)] = 1.
    return {'mean': mean.astype(np.float32), 'scale': scale.astype(np.float32), 'count': count}


def normalize(matrix, normalizer, clip=8.):
    x = np.asarray(matrix, dtype=np.float32)
    mean, scale = np.asarray(normalizer['mean'], dtype=np.float32), np.asarray(normalizer['scale'], dtype=np.float32)
    if x.shape[-1] != len(mean) or len(scale) != len(mean) or not np.isfinite(scale).all() or (scale <= 0).any() or np.isinf(x).any():
        raise ValueError('Invalid normalization schema or values')
    known = np.isfinite(x)
    values = np.where(known, (x-mean)/scale, 0.)
    return np.concatenate([np.clip(values, -clip, clip), known.astype(np.float32)], axis=-1).astype(np.float32)


def sampled_indices(period, stride):
    if not isinstance(stride, int) or stride < 1: raise ValueError('Invalid sample stride')
    blocks = period_blocks(period)
    return np.concatenate([np.arange(a, b, stride) for a, b in blocks]) if blocks else np.array([], dtype=int)


def upsample_probabilities(sampled, indices, period):
    if sampled.shape != (len(indices), 2) or not np.isfinite(sampled).all() or ((sampled < 0) | (sampled > 1)).any():
        raise ValueError('Invalid sampled home/dead probabilities')
    result = np.full((len(period), 2), np.nan, dtype=np.float32)
    for a, b in period_blocks(period):
        mask = (indices >= a) & (indices < b)
        ids = indices[mask]
        if not len(ids) or ids[0] != a or (np.diff(ids) <= 0).any(): raise ValueError('Missing/invalid half anchors')
        for j in range(2): result[a:b, j] = np.interp(np.arange(a, b), ids, sampled[mask, j])
    return result


class TimeLayerNorm(nn.Module):
    """Channel normalization at each time step; never normalize across time."""
    def __init__(self, channels):
        super().__init__(); self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.layers = nn.Sequential(TimeLayerNorm(channels), nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation), nn.Dropout(dropout),
            TimeLayerNorm(channels), nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation), nn.Dropout(dropout))

    def forward(self, x): return x+self.layers(x)


class PossessionTCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['kernel_size'] != 3: raise ValueError('This TCN fixes three-tap kernels')
        channels = config['hidden_channels']
        self.radius = 2*sum(config['dilations'])
        self.project = nn.Conv1d(config['input_features']*2, channels, 1)
        self.blocks = nn.Sequential(*[ResidualBlock(channels, d, config['dropout']) for d in config['dilations']])
        self.final_norm = TimeLayerNorm(channels)
        self.state = nn.Conv1d(channels, 2, 1)  # conditional home, dead
        self.boundary = nn.Conv1d(channels, 4, 1)  # auxiliary none / enter H / enter A / enter D

    def forward(self, x):
        z = self.final_norm(self.blocks(F.gelu(self.project(x))))
        return self.state(z), self.boundary(z)


def boundary_targets(labels, period, radius):
    """Supervised training helper, separate from feature extraction and inference."""
    if labels.shape != period.shape or radius < 0: raise ValueError('Invalid boundary targets')
    targets = np.where((period > 0) & (labels >= 0), 0, -1).astype(np.int64)
    for a, b in period_blocks(period):
        changes = np.flatnonzero((labels[a+1:b] != labels[a:b-1]) & (labels[a+1:b] >= 0) & (labels[a:b-1] >= 0))+a+1
        if not len(changes): continue
        ids = np.arange(a, b); right = np.searchsorted(changes, ids)
        before, after = changes[np.maximum(0, right-1)], changes[np.minimum(len(changes)-1, right)]
        nearest = np.where(np.abs(ids-before) <= np.abs(after-ids), before, after)
        near = (np.abs(ids-nearest) <= radius) & (labels[ids] >= 0)
        targets[ids[near]] = labels[nearest[near]]+1
    return targets


def losses(state, boundary, labels, owner_weights, boundary_labels, config):
    valid = labels >= 0
    alive = valid & (labels != 2)
    dead_loss = (F.binary_cross_entropy_with_logits(state[:, 1], (labels == 2).float(), reduction='none')*valid).sum()/valid.sum().clamp_min(1)
    hw = owner_weights*alive
    home_loss = (F.binary_cross_entropy_with_logits(state[:, 0], (labels == 0).float(), reduction='none')*hw).sum()/hw.sum().clamp_min(1)
    weights = state.new_tensor(config['boundary_class_weights'])
    auxiliary = F.cross_entropy(boundary, boundary_labels, weight=weights, ignore_index=-1, reduction='none')
    aux_mask = boundary_labels >= 0
    denominator = (weights[boundary_labels.clamp_min(0)]*aux_mask).sum().clamp_min(1)
    boundary_loss = auxiliary.sum()/denominator
    total = dead_loss+home_loss+config['boundary_loss_weight']*boundary_loss
    return total, {'dead': dead_loss, 'home': home_loss, 'boundary': boundary_loss}


def infer_sampled(model, matrix, sampled_period, normalizer, config):
    """Chunk with an exact receptive-field halo and independent half padding."""
    result = np.full((len(matrix), 2), np.nan, dtype=np.float32)
    radius, core = model.radius, config['inference_core_steps']
    model.eval()
    with torch.inference_mode():
        for a, b in period_blocks(sampled_period):
            for start in range(a, b, core):
                end = min(b, start+core)
                low, high = max(a, start-radius), min(b, end+radius)
                x = normalize(matrix[low:high], normalizer, config['normalization_clip'])
                # Explicit zero/missing input padding matches training windows at half edges.
                left, right = low-(start-radius), (end+radius)-high
                x = np.pad(x, ((left, right), (0, 0)))
                tensor = torch.from_numpy(x.T.copy()).unsqueeze(0)
                logits, _ = model(tensor)
                result[start:end] = torch.sigmoid(logits[0, :, radius:radius+end-start]).T.numpy()
    return result


def combine_probabilities(home, dead, tcn_home, tcn_dead, period, blend):
    if not 0 <= blend <= 1: raise ValueError('Invalid blend')
    active = period > 0
    for p in (home, dead, tcn_home, tcn_dead):
        if p.shape != period.shape or not np.isfinite(p[active]).all() or ((p[active] < 0) | (p[active] > 1)).any():
            raise ValueError('Invalid candidate probability')
    old = np.column_stack([(1-dead)*home, (1-dead)*(1-home), dead])
    new = np.column_stack([(1-tcn_dead)*tcn_home, (1-tcn_dead)*(1-tcn_home), tcn_dead])
    joint = (1-blend)*old+blend*new
    mixed_home = np.divide(joint[:, 0], joint[:, :2].sum(1), out=np.full(len(period), .5), where=joint[:, :2].sum(1) > 0)
    return {'current': (home, dead), 'tcn': (tcn_home, tcn_dead),
            'tcn_blend': (mixed_home, joint[:, 2]),
            'tcn_owner_blend': ((1-blend)*home+blend*tcn_home, dead)}
