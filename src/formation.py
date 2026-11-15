"""Lay a squad out on the pitch by formation, so every player gets his own slot and nobody overlaps.

The data has no tactical line-ups, so the formation is inferred: each template below is filled with the players
who have the most league minutes, using an optimal assignment (Hungarian algorithm) where a player can fill his own
role fully and a neighbouring role at a discount. The template that seats the most weighted minutes wins, with a small handicap on back threes (back fours are far more common;
on 2017-2024 squads this gives about one club in five a back three, Inter and Union Berlin among them). The app also lets the user override it.
Players who do not make the XI go on the bench strip under the pitch."""
import numpy as np
from scipy.optimize import linear_sum_assignment

# Pitch coordinates for a 360 x 380 SVG, attacking upwards. Slot roles: CB, FB, WB (wing-back), DM, CM, AM, W, ST.
TEMPLATES = {
    '4-3-3': [('FB', 52, 272), ('CB', 135, 298), ('CB', 225, 298), ('FB', 308, 272), ('DM', 180, 238), ('CM', 105, 186), ('CM', 255, 186), ('W', 62, 100), ('ST', 180, 72), ('W', 298, 100)],
    '4-2-3-1': [('FB', 52, 276), ('CB', 135, 300), ('CB', 225, 300), ('FB', 308, 276), ('DM', 128, 226), ('DM', 232, 226), ('W', 58, 146), ('AM', 180, 150), ('W', 302, 146), ('ST', 180, 70)],
    '4-4-2': [('FB', 52, 278), ('CB', 135, 300), ('CB', 225, 300), ('FB', 308, 278), ('W', 52, 182), ('CM', 135, 204), ('CM', 225, 204), ('W', 308, 182), ('ST', 132, 82), ('ST', 228, 82)],
    '4-1-4-1': [('FB', 52, 278), ('CB', 135, 300), ('CB', 225, 300), ('FB', 308, 278), ('DM', 180, 244), ('W', 52, 162), ('CM', 130, 178), ('CM', 230, 178), ('W', 308, 162), ('ST', 180, 72)],
    '3-5-2': [('CB', 96, 294), ('CB', 180, 300), ('CB', 264, 294), ('WB', 40, 196), ('DM', 180, 240), ('CM', 112, 178), ('CM', 248, 178), ('WB', 320, 196), ('ST', 132, 80), ('ST', 228, 80)],
    '3-4-3': [('CB', 96, 294), ('CB', 180, 300), ('CB', 264, 294), ('WB', 42, 206), ('CM', 135, 214), ('CM', 225, 214), ('WB', 318, 206), ('W', 70, 108), ('ST', 180, 76), ('W', 290, 108)],
}
GK_SLOT = (180, 352)
PRIOR_3 = 0.96   # back-four shapes are far more common in the five leagues, so a back three has to seat clearly more minutes to win
FIT = {   # how well a player of fine position (key of the inner dict) fills a slot role
    'CB': {'CB': 1.0, 'DM': 0.5, 'FB': 0.4},
    'FB': {'FB': 1.0, 'W': 0.45, 'CB': 0.5},
    'WB': {'FB': 1.0, 'W': 0.8},
    'DM': {'DM': 1.0, 'CM': 0.8, 'CB': 0.3},
    'CM': {'CM': 1.0, 'DM': 0.8, 'AM': 0.75},
    'AM': {'AM': 1.0, 'CM': 0.7, 'W': 0.65, 'ST': 0.5},
    'W': {'W': 1.0, 'AM': 0.7, 'ST': 0.55, 'FB': 0.35},
    'ST': {'ST': 1.0, 'W': 0.6, 'AM': 0.5},
}


def _fill(template, pos, mins):
    """Optimal assignment of players to the template's slots. Returns (total weighted minutes, {slot index: player index})."""
    W = np.array([[FIT[role].get(p, 0.0) * m for p, m in zip(pos, mins)] for role, _, _ in template]) if len(pos) else np.zeros((len(template), 0))
    if W.size == 0: return 0.0, {}
    r, c = linear_sum_assignment(-W); seat = {int(i): int(j) for i, j in zip(r, c) if W[i, j] > 0}
    return float(sum(W[i, j] for i, j in seat.items())), seat


def layout(pos, minutes, formation='auto'):
    """pos, minutes: one entry per player. Returns (formation name, [(slot role, x, y, player index or None)], [bench player indices])."""
    pos = list(pos); mins = np.nan_to_num(np.asarray(minutes, float))
    if formation not in TEMPLATES:
        scores = {name: _fill(t, pos, mins)[0] * (PRIOR_3 if name.startswith('3') else 1.0) for name, t in TEMPLATES.items()}; formation = max(scores, key=scores.get)   # dict order breaks ties: 4-3-3 first
    t = TEMPLATES[formation]; _, seat = _fill(t, pos, mins)
    slots = [(role, x, y, seat.get(i)) for i, (role, x, y) in enumerate(t)]; used = set(seat.values())
    bench = sorted((j for j in range(len(pos)) if j not in used), key=lambda j: -mins[j])
    return formation, slots, bench
