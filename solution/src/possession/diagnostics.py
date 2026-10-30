"""External-label oracle interventions for diagnosis only, never inference."""
import numpy as np

from .validation import period_blocks


def extend_reference_team(labels, period):
    """Carry the last true alive team across dead stretches, backfill a leading stretch.

    Team identity during a true dead ball is undefined. This explicit convention
    lets a false-alive prediction remain H/A rather than silently repairing it to D.
    No information crosses a half boundary.
    """
    team = np.full(len(labels), 'X', dtype='<U1')
    for start, stop in period_blocks(period):
        local = labels[start:stop]
        alive = np.flatnonzero(np.isin(local, ['H', 'A']))
        if not len(alive):
            raise ValueError('Oracle team undefined: period contains no reference alive team')
        previous = np.maximum.accumulate(np.where(np.isin(local, ['H', 'A']), np.arange(len(local)), -1))
        previous[previous < 0] = alive[0]
        team[start:stop] = local[previous]
    return team


def oracle_predictions(labels, prediction, owner_only, period):
    if not (labels.shape == prediction.shape == owner_only.shape == period.shape):
        raise ValueError('Oracle alignment mismatch')
    if not all(np.isin(x, ['H', 'A', 'D', 'X']).all() for x in (labels, prediction, owner_only)):
        raise ValueError('Invalid oracle labels')
    if (owner_only == 'D').any():
        raise ValueError('Owner-only branch must not contain dead decisions')
    graded = (labels != 'X') & (period > 0)
    reference_team = extend_reference_team(labels, period)
    true_dead = np.where(labels == 'D', 'D', owner_only)
    true_team = np.where(prediction == 'D', 'D', reference_team)
    # Sensitivity checks retain the production branch's original abstentions.
    strict_dead = np.where(labels == 'D', 'D', np.where(prediction == 'D', 'X', prediction))
    strict_team = np.where(prediction == 'X', 'X', true_team)
    both = np.where(labels == 'D', 'D', reference_team)
    result = {'true_dead_predicted_team': true_dead,
              'true_team_predicted_dead': true_team,
              'both_true': both,
              'true_dead_existing_output': strict_dead,
              'true_team_keep_abstentions': strict_team}
    for name, values in result.items():
        result[name] = np.where(graded, values, 'X').astype('<U1')
    if not np.array_equal(result['both_true'][graded], labels[graded]):
        raise ValueError('Both-true oracle did not reconstruct reference labels')
    return result
