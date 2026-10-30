"""Weak automatic phase evidence, never hard possession or restart labels."""
import numpy as np

from .validation import period_blocks

ATTACK = frozenset({'BUILD UP', 'PROGRESSION', 'CREATING CHANCES', 'BALL IN FINAL THIRD',
    'BALL IN THE BOX', 'PLAYERS IN THE BOX', 'LONG BALL', 'ATTACKING TRANSITION'})
DEFENSE = frozenset({'DEFENDING IN ATTACKING THIRD', 'DEFENDING IN MIDDLE THIRD',
    'DEFENDING IN DEFENSIVE THIRD', 'DEFENSIVE TRANSITION'})
COUNTS = ('coverage', 'home_evidence', 'away_evidence', 'restart_window',
          'restart_home', 'restart_away', 'restart_unknown_team', 'goal_window', 'unknown_team')


def build_tag_features(events, period, team_mapping):
    n = len(period)
    output = {'tag_available': np.where(period > 0, float(events is not None), np.nan).astype(np.float32)}
    if events is None:
        output.update({'tag_' + name: np.full(n, np.nan, dtype=np.float32)
            for name in (*COUNTS, 'home_evidence_fraction', 'conflict')})
        return output
    deltas = {name: np.zeros(n + 1, dtype=np.float32) for name in COUNTS}
    mapping = {key.strip().casefold(): value for key, value in team_mapping.items()}
    for event in events:
        code = (event['code'] or '').strip().upper()
        tags = event.get('tags', {})
        team = mapping.get((tags.get('Team') or '').strip().casefold())
        half = {'1st half': 1, '2nd half': 2}.get((tags.get('Half') or '').strip().casefold())
        for a, b in period_blocks(period):
            if half is not None and period[a] != half:
                continue
            low, high = max(a, int(event['start_frame'])), min(b, int(event['end_frame']))
            if low >= high:
                continue
            names = ['coverage']
            if team is None:
                names.append('unknown_team')
            if code in ATTACK | DEFENSE and team in ('H', 'A'):
                owner = team if code in ATTACK else ('A' if team == 'H' else 'H')
                names.append('home_evidence' if owner == 'H' else 'away_evidence')
            if code == 'SET PIECES':
                names.extend(['restart_window', {'H': 'restart_home', 'A': 'restart_away'}.get(team, 'restart_unknown_team')])
            if code == 'GOALS':
                names.append('goal_window')
            for name in names:
                deltas[name][low] += 1
                deltas[name][high] -= 1
    counts = {name: np.cumsum(values[:-1]) for name, values in deltas.items()}
    counts['conflict'] = ((counts['home_evidence'] > 0) & (counts['away_evidence'] > 0)).astype(np.float32)
    total = counts['home_evidence'] + counts['away_evidence']
    counts['home_evidence_fraction'] = np.divide(counts['home_evidence'], total,
        out=np.full(n, np.nan, dtype=np.float32), where=total > 0)
    # Keep identical channel order whether the source has tags or not.
    for name in (*COUNTS, 'home_evidence_fraction', 'conflict'):
        output['tag_' + name] = np.where(period > 0, counts[name], np.nan)
    return output
