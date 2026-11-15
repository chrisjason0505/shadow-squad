"""Rate every action against the way the player's team plays.

Raw per-90 numbers punish players in dominant sides: a centre-back at a team with 60% of the ball faces fewer attacks,
so he makes fewer tackles and interceptions, and a creator in a side that sits deep gets fewer touches to create from.
This module fits, for each position and metric, how much of a per-90 number is explained by team style
(possession and pressing, from lake/team_style.parquet), and then:

1. Possession-adjusted per-90 (the PAdj idea): defensive actions are scaled up for players whose team had the ball,
   because they had fewer chances to defend, using 2 / (1 + exp(-0.1 * (possession - 50))); actions on the ball are
   scaled the other way at half strength (the square root), because touches grow with possession but not one for one.
   Shooting and aerial win rate are left alone. Percentiles of those are the "adjusted" ratings in the app.
2. Style surprise: how far above or below what his team's style predicts he is, in standard deviations. Large positive
   values are the outliers: the creator who keeps taking risks in a low block, the defender who wins the ball far more
   than a side with the ball needs.
3. Team defence: for defenders and holding midfielders, how little the team conceded (xG against per match) while he
   played most minutes. It lets a quiet but positionally sound centre-back at the best defence in the league be read
   as sound, not passive.
4. A 0-100 rating per player-season, raw and style-adjusted, from department averages weighted by what matters for the position.

The fit is descriptive (it is used for display and explanation, not by the ranking models), pooled over 2017-2024."""
import numpy as np, pandas as pd
import football as FB

STYLE_X = ['x_poss', 'x_press']            # possession (points above 50, per 10) and pressing (minus the league-season z of PPDA)
WEIGHTS = [0.38, 0.26, 0.18, 0.11, 0.07]   # department weights in the order of FB.DEPT_ORDER for the position
TEAM_D = {'CB': 0.25, 'FB': 0.15, 'DM': 0.15}  # share of the rating given to team defence, for starters only
MIN_STARTER = 1800
DEF_M = {'tklw', 'int', 'clr', 'blocks'}; BALL_M = {'ast', 'kp', 'sca', 'ppa', 'prgp', 'prgc', 'take_on'}


def padj(poss):
    """The possession adjustment factor: 1 at 50%, about 1.4 at 58%, about 0.6 at 42%."""
    return 2 / (1 + np.exp(-0.1 * (np.asarray(poss, float) - 50)))


def _design(D):
    return np.c_[np.ones(len(D)), D[STYLE_X].to_numpy(float)]


class StyleRatings:
    def __init__(self, F, T, min_minutes=900):
        T = T.copy(); T['x_poss'] = (T['possession'] - 50) / 10; T['x_press'] = -T['z_ppda']
        keep = ['squad', 'yr', 'possession', 'ppda', 'xga_pm', 'deep_allowed_pm', 'z_xga_pm', 'z_possession', 'z_ppda', 'rank_xga', 'n_teams', 'x_poss', 'x_press', 'tackles_pm', 'interceptions_pm', 'pts_pm', 'z_long_balls_pm', 'z_xg_pm']
        F = F.drop(columns=[c for c in ('possession',) if c in F]).copy()
        P = F[['player_id', 'player', 'yr', 'squad', 'comp', 'fine_pos', 'min_num'] + [m + '_p90raw' for m in FB.MET_BT]].merge(T[[c for c in keep if c in T]], on=['squad', 'yr'], how='left')
        P[STYLE_X] = P[STYLE_X].fillna(0.0); ok = P['min_num'] >= min_minutes; self.coef = {}
        for m in FB.MET_BT:
            c = m + '_p90raw'; x = P[c].astype(float); eps = max(float(x[ok & (x > 0)].median() or 0) * 0.1, 1e-3) if (ok & (x > 0)).any() else 1e-3
            y = np.log(x.clip(lower=0) + eps); P[m + '_adj'] = np.nan; P[m + '_surp'] = np.nan
            for pos, g in P[ok].groupby('fine_pos'):
                gi = g.index[y[g.index].notna()]
                if len(gi) < 40: continue
                A = _design(P.loc[gi]); b, *_ = np.linalg.lstsq(A, y[gi], rcond=None); res = y[gi] - A @ b; sd = float(res.std()) or 1.0
                self.coef[(pos, m)] = b
                allp = P.index[(P['fine_pos'] == pos) & y.notna()]; Ap = _design(P.loc[allp])
                P.loc[allp, m + '_surp'] = (y[allp] - Ap @ b) / sd
            f = padj(P['possession'].fillna(50)); P[m + '_adj'] = x * (f if m in DEF_M else 1 / np.sqrt(f) if m in BALL_M else 1.0)
            P[m + '_pct_raw'] = np.nan; P[m + '_pct_adj'] = np.nan
            P.loc[ok, m + '_pct_raw'] = P[ok].groupby(['yr', 'fine_pos'])[c].rank(pct=True) * 100
            P.loc[ok, m + '_pct_adj'] = P[ok].groupby(['yr', 'fine_pos'])[m + '_adj'].rank(pct=True) * 100
        P['team_def_pct'] = 100 * (1 - (P['rank_xga'] - 1) / (P['n_teams'] - 1).clip(lower=1))
        self.P = P.set_index(['player_id', 'yr']); self.ok = ok.to_numpy()
        self._rate()

    # ---- ratings ----
    def _dept(self, r, kind):
        return {d: np.nanmean([r[m + '_pct_' + kind] for m in ms]) if any(np.isfinite(r[m + '_pct_' + kind]) for m in ms) else np.nan for d, ms in FB.DEPT_BT.items()}

    def _rate(self):
        P = self.P; raw, adj = [], []
        for (pid, yr), r in P.iterrows():
            order = FB.DEPT_ORDER.get(r['fine_pos'], list(FB.DEPT_BT)); w = dict(zip(order, WEIGHTS))
            out = []
            for kind in ('raw', 'adj'):
                d = self._dept(r, kind); num = sum(w[k] * v for k, v in d.items() if np.isfinite(v)); den = sum(w[k] for k, v in d.items() if np.isfinite(v))
                val = num / den if den else np.nan
                td = TEAM_D.get(r['fine_pos'], 0.0)
                if kind == 'adj' and td and r['min_num'] >= MIN_STARTER and np.isfinite(r['team_def_pct']) and np.isfinite(val): val = (1 - td) * val + td * r['team_def_pct']
                out.append(val)
            raw.append(out[0]); adj.append(out[1])
        P['rating_raw'] = raw; P['rating_adj'] = adj

    def get(self, pid, yr):
        k = (int(pid), int(yr)); return self.P.loc[k] if k in self.P.index else None

    def dept_pct(self, pid, yr, kind='adj'):
        r = self.get(pid, yr); return {} if r is None else self._dept(r, kind)

    # ---- the football reading: style and outliers ----
    @staticmethod
    def team_words(r):
        """'held 58% of the ball, pressed hard (PPDA 9.6) and conceded 0.84 xG a game, the best defence in the league'"""
        if r is None or not np.isfinite(r.get('possession', np.nan)): return None
        p = r['possession']; press = r['z_ppda']
        ball = 'dominated the ball' if p >= 57 else 'had more of the ball than most' if p >= 52 else 'shared the ball' if p > 48 else 'played without the ball' if p > 43 else 'sat deep and conceded possession'
        pr = 'pressed high and hard' if press <= -1 else 'pressed more than most' if press <= -0.4 else 'held a mid-block' if press < 0.4 else 'dropped off and defended deep'
        rk, n = r['rank_xga'], r['n_teams']; d = f"conceded {r['xga_pm']:.2f} xG a game" + (f", the best defence in the league" if rk == 1 else f", the {FB_ord(int(rk))} best in the league" if rk <= 5 else f", {FB_ord(int(rk))} of {int(n)}" if np.isfinite(rk) else '')
        return f"{ball} ({p:.0f}% possession), {pr} (PPDA {r['ppda']:.1f}) and {d}"

    def outliers(self, pid, yr, k=3):
        """Style-based outliers, strongest first: [(tag, sentence)]. Each archetype pairs what the team's style asks for
        (or makes hard) with what the player still does. Ball-winning stands in for pressing work: the bundled FBref
        tables have tackles won and interceptions but not the pressures count, which FBref dropped."""
        r = self.get(pid, yr)
        if r is None: return []
        pos = r['fine_pos']; club = r['squad']; P = FB.POS.get(pos, pos); out = []
        poss = r.get('possession', np.nan); press = r.get('x_press', np.nan); zlong = r.get('z_long_balls_pm', np.nan); zxg = r.get('z_xg_pm', np.nan)
        pct = lambda ms: np.nanmean([r[m + '_pct_raw'] for m in ms]) if any(np.isfinite(r[m + '_pct_raw']) for m in ms) else np.nan
        sup = lambda ms: np.nanmean([r[m + '_surp'] for m in ms]) if any(np.isfinite(r[m + '_surp']) for m in ms) else np.nan
        ok = lambda *v: all(np.isfinite(x) for x in v)
        win, drib, threat, create, prog = pct(['tklw', 'int']), pct(['take_on', 'prgc']), pct(['npxg', 'gls', 'sh']), pct(['kp', 'sca', 'ppa', 'ast']), pct(['prgp', 'prgc'])
        PL = FB.POS_PL.get(pos, 'players'); ppda = f"PPDA {r['ppda']:.1f}" if ok(r.get('ppda', np.nan)) else 'a high press'
        # 1. two-way player in a high-octane pressing side
        if pos in ('W', 'AM', 'CM', 'ST', 'FB') and ok(press, win, drib) and press >= 0.7 and win >= 65 and drib >= 70:
            out.append(((win + drib) / 50, 'Two-way engine in a pressing side', f"{club} pressed as hard as anyone ({ppda}), which drains attackers, yet he is in the top {100 - win:.0f}% of {PL} for winning the ball back and the top {100 - drib:.0f}% for dribbles and carries. Most players in a system like that give up one side of the game; he keeps both."))
        if pos in ('ST', 'W', 'AM') and ok(press, win, threat) and press >= 0.7 and win >= 70 and threat >= 70:
            out.append(((win + threat) / 50, 'Presses and still scores', f"Leads the press in a high-intensity side ({ppda}), top {100 - win:.0f}% of {PL} for ball-winning, and still top {100 - threat:.0f}% for goal threat."))
        # 2. risk-taker in a deep side
        for d, ms, word in (('Creation', ['kp', 'sca', 'ppa', 'ast'], 'chance creation'), ('Progression', ['prgp', 'prgc', 'take_on'], 'ball progression')):
            sv = sup(ms)
            if ok(sv, poss) and sv >= 1.2 and poss <= 46:
                out.append((sv + 1.5, 'Risk-taker in a deep side', f"{club} had only {poss:.0f}% of the ball, yet his {word} is {sv:+.1f} standard deviations above what that style predicts. He keeps trying the difficult pass instead of the safe one; the team's attacking play runs through him."))
                break
        # 3. direct dribbler in a patient possession side
        sv = sup(['take_on', 'prgc'])
        if pos in ('W', 'AM', 'CM', 'FB', 'ST') and ok(sv, poss) and poss >= 56 and sv >= 1.0:
            out.append((sv + 1.2, 'Direct runner in a patient side', f"{club} built slowly with {poss:.0f}% of the ball, a style that recycles possession rather than running with it, yet he beats men and carries forward {sv:+.1f} sd more than that predicts. He is the one who breaks the pattern."))
        # 4. creative hub in a direct, long-ball side
        sv = sup(['kp', 'prgp', 'ppa'])
        if ok(sv, zlong) and zlong >= 1.0 and sv >= 1.2:
            out.append((sv + 1.0, 'Playmaker in a direct side', f"{club} went long more than most, which bypasses midfield, yet his creative and progressive passing is {sv:+.1f} sd above what that style predicts. The ball goes through him when they do play."))
        # 5. carries a low-xG attack
        best = max((v for v in (threat, create) if np.isfinite(v)), default=np.nan)
        if pos in ('ST', 'W', 'AM', 'CM') and ok(zxg, best) and zxg <= -0.7 and best >= 80:
            out.append((best / 40, 'Carries a blunt attack', f"{club} created little as a team (xG per game well below the league average), yet he is in the top {100 - best:.0f}% of {PL} for " + ('goal threat' if best == threat else 'chance creation') + '. Take him out and there is not much left.'))
        # 6. ball-playing defender in a deep side
        if pos in ('CB', 'FB') and ok(prog, poss) and poss <= 47 and prog >= 75:
            out.append((prog / 40, 'Ball-player in a deep side', f"A {P} who is top {100 - prog:.0f}% for progressive passes and carries in a side with only {poss:.0f}% of the ball, where defenders are usually told to clear it."))
        # 7. front-foot defender in a side that has the ball
        sv = sup(['tklw', 'int', 'blocks'])
        if pos in ('CB', 'FB', 'DM', 'CM') and ok(sv, poss) and sv >= 1.2 and poss >= 55:
            out.append((sv + 1.0, 'Front-foot defender', f"{club} had {poss:.0f}% of the ball, so there was little defending to do, yet he made far more defensive actions than that predicts ({sv:+.1f} sd). He hunts the ball rather than waiting for it."))
        # 8. positionally sound defender in a tight defence
        raw = self._dept(r, 'raw'); adj = self._dept(r, 'adj')
        if pos in ('CB', 'FB', 'DM') and r['min_num'] >= MIN_STARTER and ok(raw.get('Defending', np.nan), r['z_xga_pm']) and raw['Defending'] < 40 and r['z_xga_pm'] <= -1.0:
            out.append((3.0 - r['z_xga_pm'] / 2, 'Positionally sound', f"Few tackles and interceptions on paper ({FB_ord(int(round(raw['Defending'])))} percentile raw), but {club} conceded only {r['xga_pm']:.2f} xG a game with him playing {int(r['min_num']):,} minutes" + (", the best defence in the league" if r['rank_xga'] == 1 else f", the {FB_ord(int(r['rank_xga']))} best in the league") +
                        f". A defender in a side like that is rarely exposed: low numbers point to positioning and reading the game, not a weak defender. Adjusted for possession his defending is {FB_ord(int(round(adj['Defending'])))} percentile."))
        # 9. keeps it simple (the one negative read)
        sv = sup(['kp', 'prgp', 'ppa'])
        if pos in ('CB', 'FB', 'DM', 'CM') and ok(sv, poss) and sv <= -1.3 and poss >= 55:
            out.append((-sv, 'Keeps it simple', f"His creative and progressive passing is well below what a side with {poss:.0f}% of the ball predicts ({sv:+.1f} sd): a safe, low-risk passer in a team that gives him the ball."))
        return [(t, s_) for _, t, s_ in sorted(out, key=lambda x: -x[0])[:k]]


def FB_ord(n):
    return f"{n}" + ("th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))
