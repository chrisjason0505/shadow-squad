"""Football language for the explainer, and hindsight for past shortlists.

Two jobs:
1. Turn the engine's numbers into what a scout would say: per-90 metrics grouped into football dimensions (goal threat,
   chance creation, ball progression, defensive work, aerial), z-scores into "top 10% for wingers", ranking features
   into reasons ("priced in the same bracket", "at the age players step up"), and the club's style indices into words.
2. Hindsight: for a player on a past shortlist, where he is now and what Transfermarkt says he is worth.
   Sources, best first: lake/player_now.parquet (written by src/build_player_now.py from the Transfermarkt dump),
   then the 2025-26 live squads and shortlists, then later signings in the replacement events, then the last season
   in the feature lake. Values come from the Transfermarkt valuation history in lake/value_history.parquet."""
import os
import numpy as np, pandas as pd

POS = {'CB': 'centre-back', 'FB': 'full-back', 'DM': 'defensive midfielder', 'CM': 'central midfielder', 'AM': 'attacking midfielder', 'W': 'winger', 'ST': 'striker'}
POS_PL = {k: v + 's' for k, v in POS.items()}

# Backtest engine (FBref per 90)
MET_BT = {'gls': 'goals', 'npxg': 'non-penalty xG', 'sh': 'shots', 'ast': 'assists', 'kp': 'key passes', 'sca': 'shot-creating actions', 'ppa': 'passes into the box',
          'prgp': 'progressive passes', 'prgc': 'progressive carries', 'take_on': 'dribbles completed', 'att_pen': 'touches in the box', 'tklw': 'tackles won',
          'int': 'interceptions', 'clr': 'clearances', 'blocks': 'blocks', 'aer_won': 'aerial duels won (% of duels)'}
DIM_BT = {'Goal threat': ['gls', 'npxg', 'sh', 'att_pen'], 'Chance creation': ['ast', 'kp', 'sca', 'ppa'], 'Ball progression': ['prgp', 'prgc', 'take_on'],
          'Defensive work': ['tklw', 'int', 'clr', 'blocks'], 'In the air': ['aer_won']}
# Live engine (Understat per 90)
MET_LIVE = {'goals': 'goals', 'npg': 'non-penalty goals', 'xG': 'xG', 'npxG': 'non-penalty xG', 'assists': 'assists', 'xA': 'expected assists (xA)', 'shots': 'shots',
            'key_passes': 'key passes', 'xGChain': 'involvement in attacking moves (xGChain)', 'xGBuildup': 'build-up play (xGBuildup)'}
DIM_LIVE = {'Goal threat': ['goals', 'npg', 'xG', 'npxG', 'shots'], 'Chance creation': ['assists', 'xA', 'key_passes'], 'Build-up and link play': ['xGChain', 'xGBuildup']}
DIM_NOUN = {'Goal threat': 'goal threat', 'Chance creation': 'chance creation', 'Ball progression': 'ball progression', 'Defensive work': 'defensive work', 'In the air': 'aerial presence',
            'Build-up and link play': 'build-up and link play'}

# Ranking features as a scout would put them: (when it pushes him up, when it holds him back). {S} = the starter, {C} = the buying club.
WHY = {'sim_translated': ("plays like {S} once his numbers are adjusted to {C}'s style", "does not play like {S}, even adjusted to {C}'s style"),
       'traj_sim': ("his last seasons followed the same curve as {S}'s at that age", "his career has not developed the way {S}'s did"),
       'pipeline_rank': ('he comes from a club that regularly sells players up the ladder', 'his club is not a usual stepping stone to bigger clubs'),
       'club_taste': ('{C} has a history of buying this type of player', '{C} rarely buys this type of player'),
       'hazard': ('he is at the career stage where players make the step up', 'players at his career stage rarely make the step up'),
       'log_value': ('his market value fits a signing at this level', 'his market value is off for a signing at this level'),
       'value_gap': ('he is priced in the same bracket as {S}', 'he is priced far from {S}'),
       'age_num': ('his age fits the profile clubs buy for this role', 'his age counts against him'),
       'composite_perf_index': ('his all-round output is high for his position', 'his all-round output is modest for his position'),
       'pattern_match': ('he ticks the boxes past signings at this position ticked', 'he lacks the traits past signings at this position shared')}


def tier(z):
    """A z-score of log per-90 within the position, in scout's words (the scale is close to normal)."""
    if not np.isfinite(z): return 'no data'
    return ('top 10%' if z >= 1.28 else 'top 25%' if z >= 0.67 else 'above average' if z >= 0.25 else 'around average' if z > -0.25 else 'below average' if z > -0.67
            else 'bottom 25%' if z > -1.28 else 'bottom 10%')


def dims(zmap, live):
    """Mean z per football dimension. zmap: metric -> z."""
    D = DIM_LIVE if live else DIM_BT
    return {d: float(np.nanmean([zmap.get(m, np.nan) for m in ms])) for d, ms in D.items() if any(np.isfinite(zmap.get(m, np.nan)) for m in ms)}


def role(zmap, pos, live):
    """'a winger whose game is built on ball progression and chance creation'."""
    skip = set() if pos in ('CB', 'ST') else {'In the air'}      # one-number dimensions only define a role where they are the job
    d = sorted(((k, v) for k, v in dims(zmap, live).items() if k not in skip), key=lambda kv: -kv[1]); good = [k for k, v in d if v >= 0.25][:2]
    base = POS.get(pos, pos)
    if not good: return f'a {base} without a standout area in the numbers'
    return f"a {base} whose game is built on {DIM_NOUN[good[0]]}" + (f" and {DIM_NOUN[good[1]]}" if len(good) > 1 else '')


def like_for_like(zs, zc, live, pos):
    """What the candidate keeps, where he is a step down, and what he adds, metric by metric (z of starter, z of candidate)."""
    names = MET_LIVE if live else MET_BT; pl = POS_PL.get(pos, 'players in his position'); keeps, down, adds = [], [], []
    for m, name in names.items():
        a, b = zs.get(m, np.nan), zc.get(m, np.nan)
        if not (np.isfinite(a) and np.isfinite(b)): continue
        if a >= 0.67 and b >= 0.67: keeps.append((a + b, f'{name}: both {tier(min(a, b))} or better among {pl}'))
        elif a >= 0.67 and b < a - 0.8: down.append((a - b, f'{name}: {tier(a)} for the starter, {tier(b)} for him'))
        elif b >= 0.67 and a < 0.2: adds.append((b - a, f'{name}: {tier(b)} among {pl}, where the starter was {tier(a)}'))
    srt = lambda L: [t for _, t in sorted(L, key=lambda x: -x[0])]
    return srt(keeps), srt(down), srt(adds)


def club_style(x_dom, x_press):
    """The engine's two style indices (standardised across club-seasons) in words."""
    if not (np.isfinite(x_dom) and np.isfinite(x_press)): return None
    a = 'a possession-dominant side' if x_dom >= 0.36 else 'a side that often plays without the ball' if x_dom <= -0.4 else 'a side with an even share of the ball'
    b = 'that presses high' if x_press >= 0.12 else 'that tends to sit off and defend deeper' if x_press <= -0.13 else 'with a middling press'
    return f'{a} {b}'


def why(feats_shap, S, C, k=4):
    """Football reasons from SHAP: (pros, cons). feats_shap: list of (feature key, shap)."""
    f = lambda s: s.format(S=S, C=C)
    pros = [f(WHY[k_][0]) for k_, v in sorted(feats_shap, key=lambda x: -x[1]) if v > 0.01 and k_ in WHY][:k]
    cons = [f(WHY[k_][1]) for k_, v in sorted(feats_shap, key=lambda x: x[1]) if v < -0.01 and k_ in WHY][:2]
    return pros, cons


def eur(v):
    return '-' if v is None or not np.isfinite(v) else (f'EUR {v / 1e6:.1f}m' if v < 1e7 else f'EUR {v / 1e6:.0f}m')


class Hindsight:
    def __init__(self, F, SQ, SL, EV, VH, root):
        self.F = F; self.VH = VH
        p = os.path.join(root, 'lake', 'player_now.parquet'); self.PN = pd.read_parquet(p).set_index('player_id') if os.path.exists(p) else None
        self.live_club = {}
        for D in (SL, SQ):          # squads last, so a player's own club beats the club whose list he is on (they are the same: lists hold players at their current club)
            if D is not None:
                for pid, c in zip(D['player_id'], D['club']): self.live_club[int(pid)] = c
        self.ev = EV.dropna(subset=['N']).assign(N=lambda d: d['N'].astype(int), N_date=lambda d: pd.to_datetime(d['N_date'], errors='coerce'))
        self.last_val = VH.sort_values('date').groupby('player_id').tail(1).set_index('player_id') if VH is not None else None

    def value_now(self, pid):
        if self.PN is not None and pid in self.PN.index and pd.notna(self.PN.at[pid, 'market_value_in_eur']):
            return float(self.PN.at[pid, 'market_value_in_eur']), pd.to_datetime(self.PN.at[pid, 'as_of'])
        if self.last_val is not None and pid in self.last_val.index: return float(self.last_val.at[pid, 'value']), self.last_val.at[pid, 'date']
        return np.nan, None

    def club_now(self, pid):
        """(club, as-of text) from the freshest source that knows the player."""
        if self.PN is not None and pid in self.PN.index and pd.notna(self.PN.at[pid, 'current_club_name']):
            return self.PN.at[pid, 'current_club_name'], f"Transfermarkt, {pd.to_datetime(self.PN.at[pid, 'as_of']):%b %Y}"
        opts = []
        if pid in self.live_club: opts.append((pd.Timestamp('2026-07-01'), self.live_club[pid], 'summer 2026'))
        e = self.ev[self.ev['N'] == pid]
        if len(e): r = e.sort_values('N_date').iloc[-1]; opts.append((r['N_date'] if pd.notna(r['N_date']) else pd.Timestamp(f"{int(r['t']) + 1}-07-01"), r['club'], f"signed {r['N_date']:%b %Y}" if pd.notna(r['N_date']) else 'signed'))
        f = self.F[self.F['player_id'] == pid]
        if len(f): r = f.sort_values('yr').iloc[-1]; opts.append((pd.Timestamp(f"{int(r['yr']) + 1}-05-31"), r['squad'], f"{int(r['yr'])}-{str(int(r['yr']) + 1)[2:]} season"))
        if not opts: return None, None
        d, c, src = max(opts, key=lambda o: o[0]); return c, src

    def path(self, pid, t):
        """Clubs after season t, in order, with the first season at each and whether it is a top club."""
        f = self.F[(self.F['player_id'] == pid) & (self.F['yr'] > t)].sort_values('yr'); out = []
        for r in f.itertuples():
            if not out or out[-1][0] != r.squad: out.append((r.squad, int(r.yr), bool(r.elite_club)))
        return out

    def row(self, pid, t, club_then, value_then, buyer=None):
        """One hindsight record for a player on the season-t list."""
        pid = int(pid); club, src = self.club_now(pid); v, vd = self.value_now(pid); moves = [m for m in self.path(pid, t) if m[0] != club_then]
        signed_by_buyer = buyer is not None and len(self.ev[(self.ev['N'] == pid) & (self.ev['club'] == buyer) & (self.ev['t'] >= t)]) > 0
        if signed_by_buyer: moves = [m for m in moves if m[0] != buyer]
        up = [m for m in moves if m[2]]; ss = lambda m: f"{m[1]}-{str(m[1] + 1)[2:]}"
        story = ('Signed by ' + buyer + '. ' if signed_by_buyer else '') + (f"Stepped up to {up[0][0]} ({ss(up[0])})." if up else (f"Moved to {moves[0][0]} ({ss(moves[0])})." if moves else ''))
        last_yr = int(self.F['yr'].max()); f = self.F[self.F['player_id'] == pid]
        gone = club is not None and src and src.endswith('season') and len(f) and int(f['yr'].max()) < last_yr and pid not in self.live_club and (self.PN is None or pid not in self.PN.index)
        if gone:      # no trace in the five leagues after that season
            y = int(f['yr'].max()); story = (story + ' ' if story else '') + f"No trace in the five-league data after {y}-{str(y + 1)[2:]} (likely moved abroad or down)."; club, src = f"outside the five leagues as far as the data knows (last: {club})", None
        elif not story and club == club_then: story = f'Stayed at {club_then}.'
        chg = (v / value_then - 1) if (np.isfinite(v) and value_then and np.isfinite(value_then) and value_then > 0) else np.nan
        return {'player_id': pid, 'club_then': club_then, 'value_then': value_then, 'club_now': club, 'club_now_source': src, 'value_now': v,
                'value_now_date': vd, 'value_change': chg, 'story': story.strip(), 'stepped_up': bool(up), 'signed_by_buyer': signed_by_buyer}

    def line(self, h, name):
        """One sentence for the dossier and the cards."""
        val = f"worth {eur(h['value_now'])} on Transfermarkt ({h['value_now_date']:%b %Y})" if np.isfinite(h['value_now']) and h['value_now_date'] is not None else 'no recent Transfermarkt value'
        chg = f", {h['value_change']:+.0%} since that summer" if np.isfinite(h['value_change']) else ''
        now = (f"Now at {h['club_now']} ({h['club_now_source']})" if h['club_now_source'] else f"Now {h['club_now']}") if h['club_now'] else 'Current club unknown'
        return f"{name}: then at {h['club_then']}, {eur(h['value_then'])}. {h['story']} {now}, {val}{chg}.".replace('  ', ' ')


# ---- department matrix: the starter against his replacements, metric by metric ----
DEPT_BT = {'Goal threat': ['gls', 'npxg', 'sh', 'att_pen'], 'Creation': ['ast', 'kp', 'sca', 'ppa'], 'Progression': ['prgp', 'prgc', 'take_on'],
           'Defending': ['tklw', 'int', 'clr', 'blocks'], 'Aerial': ['aer_won']}
DEPT_LIVE = {'Goal threat': ['goals', 'npg', 'xG', 'npxG', 'shots'], 'Creation': ['assists', 'xA', 'key_passes'], 'Build-up': ['xGChain', 'xGBuildup']}
DEPT_ORDER = {'CB': ['Defending', 'Aerial', 'Progression', 'Creation', 'Goal threat'], 'FB': ['Progression', 'Creation', 'Defending', 'Goal threat', 'Aerial'],
              'DM': ['Defending', 'Progression', 'Creation', 'Aerial', 'Goal threat'], 'CM': ['Progression', 'Creation', 'Defending', 'Goal threat', 'Aerial'],
              'AM': ['Creation', 'Goal threat', 'Progression', 'Defending', 'Aerial'], 'W': ['Creation', 'Goal threat', 'Progression', 'Defending', 'Aerial'],
              'ST': ['Goal threat', 'Creation', 'Aerial', 'Progression', 'Defending']}


def season_pct(F, min_minutes=900):
    """Percentile (0-100) of every raw per-90 number within its fine position and season, among players with min_minutes+ league minutes."""
    P = F[['player_id', 'yr', 'fine_pos', 'min_num'] + [m + '_p90raw' for m in MET_BT]].copy(); ok = P['min_num'] >= min_minutes
    for m in MET_BT:
        c = m + '_p90raw'; P[m] = np.nan
        P.loc[ok, m] = P[ok].groupby(['yr', 'fine_pos'])[c].rank(pct=True) * 100
        P[m + '_raw'] = P[c]
    return P[['player_id', 'yr', 'fine_pos'] + list(MET_BT) + [m + '_raw' for m in MET_BT]].set_index(['player_id', 'yr'])


def matrix_bt(PCT, pids, t, peak=False):
    """{pid: {metric: (percentile, per-90, season)}} for season t, or the best season of t-2..t per metric when peak."""
    out = {}
    for pid in pids:
        rows = [(y, PCT.loc[(pid, y)]) for y in ((t - 2, t - 1, t) if peak else (t,)) if (pid, y) in PCT.index]; d = {}
        for m in MET_BT:
            best = max(((r[m], r[m + '_raw'], y) for y, r in rows if np.isfinite(r[m])), default=(np.nan, np.nan, None), key=lambda x: x[0])
            d[m] = best
        out[pid] = d
    return out


def matrix_live(rows, zcols):
    """Live lists hold z-scores only: turn them into approximate percentiles with the normal curve."""
    from math import erf, sqrt
    cdf = lambda z: 50 * (1 + erf(z / sqrt(2))) if np.isfinite(z) else np.nan
    return {int(r['player_id']): {c[2:]: (cdf(float(r[c])), np.nan, None) for c in zcols} for _, r in rows.iterrows()}
