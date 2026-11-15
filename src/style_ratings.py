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
    def __init__(self, F, T, min_minutes=900, DX=None):
        T = T.copy(); T['x_poss'] = (T['possession'] - 50) / 10; T['x_press'] = -T['z_ppda']
        keep = ['squad', 'yr', 'possession', 'ppda', 'xga_pm', 'deep_allowed_pm', 'z_xga_pm', 'z_possession', 'z_ppda', 'rank_xga', 'n_teams', 'league', 'x_poss', 'x_press', 'tackles_pm', 'interceptions_pm', 'pts_pm', 'z_long_balls_pm', 'z_xg_pm']
        F = F.drop(columns=[c for c in ('possession',) if c in F]).copy()
        P = F[['player_id', 'player', 'yr', 'squad', 'comp', 'fine_pos', 'min_num'] + [m + '_p90raw' for m in FB.MET_BT]].merge(T[[c for c in keep if c in T]], on=['squad', 'yr'], how='left')
        if {'long_balls_pm', 'passes_pm'} <= set(T.columns):     # directness: share of passes that are long, against the league that season
            d = T[['squad', 'yr', 'league']].assign(direct=T['long_balls_pm'] / T['passes_pm'])
            d['z_direct'] = d.groupby(['league', 'yr'])['direct'].transform(lambda x: (x - x.mean()) / x.std(ddof=0) if x.notna().sum() > 3 else np.nan)
            P = P.merge(d[['squad', 'yr', 'direct', 'z_direct']], on=['squad', 'yr'], how='left')
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
        for m in ('prgp', 'kp', 'npxg', 'sca', 'aer_won'):
            tot = P[m + '_p90raw'] * P['min_num'] / 90
            if m == 'aer_won': continue
            P['share_' + m] = tot / tot.groupby([P['squad'], P['yr']]).transform('sum')
            P['rank_' + m] = tot.groupby([P['squad'], P['yr']]).rank(ascending=False, method='min')
        if DX is not None:     # where he wins the ball (src/build_defence.py): the pressing proxy, and aerial volume
            ex = DX[DX['src'] == 'fbref'][['player_id', 'yr', 'press_hi_p90', 'press_hi_p90_pct', 'recov_p90', 'recov_p90_pct', 'aer_w_p90', 'aer_w_p90_pct']].drop_duplicates(['player_id', 'yr'])
            P = P.merge(ex, on=['player_id', 'yr'], how='left')
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

    # ---- the football reading: categorised style reads with the numbers behind them ----
    CATS = ('Changes the team\'s dynamic', 'Focal point', 'Defensive intelligence', 'Role note')

    def reads(self, pid, yr, k=4):
        """Style reads, strongest first: [{'cat', 'tag', 'text', 'ev': [(label, value)], 'score'}].
        Changes the team's dynamic: does what his team's style does not ask for, or makes hard.
        Focal point: the team leans on him (his share of the regulars' output, or the job the style hands him).
        Defensive intelligence: defends well in ways tackle counts miss (position, a high line, a tight defence).
        Role note: a neutral read of how he plays in the system.
        Pressing is read from where he wins the ball (tackles in midfield and the final third, ball recoveries), because FBref no longer publishes pressures.
        Team shares are among the squad's regulars (900+ league minutes)."""
        r = self.get(pid, yr)
        if r is None: return []
        pos = r['fine_pos']; club = r['squad']; P = FB.POS.get(pos, pos); PL = FB.POS_PL.get(pos, 'players'); out = []
        poss = r.get('possession', np.nan); press = r.get('x_press', np.nan); zlong = r.get('z_direct', np.nan); zxg = r.get('z_xg_pm', np.nan); zx = r.get('z_xga_pm', np.nan)
        ok = lambda *v: all(v_ is not None and np.isfinite(v_) for v_ in v)
        pct = lambda ms: np.nanmean([r[m + '_pct_raw'] for m in ms]) if any(np.isfinite(r[m + '_pct_raw']) for m in ms) else np.nan
        sup = lambda ms: np.nanmean([r[m + '_surp'] for m in ms]) if any(np.isfinite(r[m + '_surp']) for m in ms) else np.nan
        top = lambda p: f"top {max(1, round(100 - p))}%"
        per = lambda m: f"{r[m + '_p90raw']:.2f} per 90 ({FB_ord(int(round(r[m + '_pct_raw'])))} pct)" if ok(r[m + '_p90raw'], r[m + '_pct_raw']) else 'n/a'
        T = [('Team possession', f"{poss:.0f}%")] if ok(poss) else []
        if ok(r.get('ppda', np.nan)): T.append(('Team PPDA (lower = harder press)', f"{r['ppda']:.1f}"))
        def add(cat, tag, score, text, ev): out.append({'cat': cat, 'tag': tag, 'score': float(score), 'text': text, 'ev': ev})
        hi = [r.get(c, np.nan) for c in ('press_hi_p90_pct', 'recov_p90_pct')]; hi = [v for v in hi if v is not None and np.isfinite(v)]
        win = float(np.mean(hi)) if hi else pct(['tklw', 'int'])
        def pr_ev():
            if ok(r.get('press_hi_p90', np.nan), r.get('press_hi_p90_pct', np.nan)): return [('Tackles in midfield and final third', f"{r['press_hi_p90']:.2f} per 90 ({FB_ord(int(round(r['press_hi_p90_pct'])))} pct)"), ('Ball recoveries', f"{r['recov_p90']:.2f} per 90 ({FB_ord(int(round(r['recov_p90_pct'])))} pct)" if ok(r.get('recov_p90', np.nan), r.get('recov_p90_pct', np.nan)) else 'n/a')]
            return [('Tackles won', per('tklw')), ('Interceptions', per('int'))]
        _, drib, threat, create, prog, aer = None, pct(['take_on', 'prgc']), pct(['npxg', 'gls', 'sh']), pct(['kp', 'sca', 'ppa', 'ast']), pct(['prgp', 'prgc']), pct(['aer_won'])
        sh = lambda m: r.get('share_' + m, np.nan); rk = lambda m: r.get('rank_' + m, np.nan)
        def share_ev(m, label):
            return (f'Share of team {label}', f"{sh(m):.0%}, {FB_ord(int(rk(m)))} in the squad") if ok(sh(m), rk(m)) else (f'Team {label}', 'n/a')
        D = "Changes the team's dynamic"; FP = 'Focal point'; DI = 'Defensive intelligence'

        # ---------- changes the team's dynamic ----------
        if pos in ('W', 'AM', 'CM', 'ST', 'FB') and ok(press, win, drib) and press >= 0.7 and win >= 65 and drib >= 70:
            add(D, 'Two-way engine in a pressing side', 2 + (win + drib) / 100, f"{club} pressed as hard as anyone, which drains attackers, yet he is {top(win)} of {PL} for winning the ball back and {top(drib)} for dribbles and carries. Most players in that system give up one side of the game; he keeps both.",
                T[1:] + pr_ev() + [('Dribbles completed', per('take_on')), ('Progressive carries', per('prgc'))])
        if pos in ('ST', 'W', 'AM') and ok(press, win, threat) and press >= 0.7 and win >= 70 and threat >= 70:
            add(D, 'Presses and still scores', 2 + (win + threat) / 100, f"Leads the press in a high-intensity side ({top(win)} of {PL} for ball-winning) and is still {top(threat)} for goal threat.", T[1:] + pr_ev() + [('Non-penalty xG', per('npxg')), ('Goals', per('gls'))])
        for ms, word in ((['kp', 'sca', 'ppa', 'ast'], 'chance creation'), (['prgp', 'prgc', 'take_on'], 'ball progression')):
            sv = sup(ms)
            if ok(sv, poss) and sv >= 1.2 and poss <= 46:
                add(D, 'Risk-taker in a deep side', 1.5 + sv, f"{club} had only {poss:.0f}% of the ball, yet his {word} is far above what that style predicts. He keeps trying the difficult pass instead of the safe one, and the team's attacking play runs through him.",
                    T[:1] + [(f'{word.capitalize()} vs what the style predicts', f"{sv:+.1f} sd")] + [(FB.MET_BT[m].capitalize(), per(m)) for m in ms[:3]]); break
        sv = sup(['take_on', 'prgc'])
        if pos in ('W', 'AM', 'CM', 'FB', 'ST') and ok(sv, poss) and poss >= 56 and sv >= 1.0:
            add(D, 'Direct runner in a patient side', 1.2 + sv, f"{club} built slowly with {poss:.0f}% of the ball, a style that recycles possession rather than running with it. He beats men and carries forward far more than that predicts, so he is the one who breaks the pattern.",
                T[:1] + [('Dribbles and carries vs style', f"{sv:+.1f} sd"), ('Dribbles completed', per('take_on')), ('Progressive carries', per('prgc'))])
        sv = sup(['kp', 'prgp', 'ppa'])
        if ok(sv, zlong) and zlong >= 1.0 and sv >= 1.2:
            add(D, 'Playmaker in a direct side', 1.0 + sv, f"{club} went long more than most, which bypasses midfield, yet his creative and progressive passing is far above what that style predicts. When they do play through the thirds, it goes through him.",
                [('Long balls as a share of passes', f"{r['direct']:.0%} ({zlong:+.1f} sd vs league)"), ('Passing vs style', f"{sv:+.1f} sd"), ('Key passes', per('kp')), ('Progressive passes', per('prgp'))])
        if pos in ('CB', 'FB') and ok(prog, poss) and poss <= 47 and prog >= 75:
            add(D, 'Ball-player in a deep side', 1 + prog / 50, f"A {P} who is {top(prog)} for progressive passes and carries in a side with only {poss:.0f}% of the ball, where defenders are usually told to clear it.", T[:1] + [('Progressive passes', per('prgp')), ('Progressive carries', per('prgc'))])
        sv = sup(['tklw', 'int', 'blocks'])
        if pos in ('CB', 'FB', 'DM', 'CM') and ok(sv, poss) and sv >= 1.2 and poss >= 55:
            add(D, 'Front-foot defender', 1.0 + sv, f"{club} had {poss:.0f}% of the ball, so there was little defending to do, yet he made far more defensive actions than that predicts. He steps in and hunts the ball rather than waiting for it.", T[:1] + [('Defending vs style', f"{sv:+.1f} sd"), ('Tackles won', per('tklw')), ('Interceptions', per('int'))])

        # ---------- focal point ----------
        if pos in ('CB', 'FB') and ok(sh('prgp'), rk('prgp')) and rk('prgp') <= 2 and sh('prgp') >= 0.14 and r['min_num'] >= MIN_STARTER:
            add(FP, 'Starts the attacks', 2 + 5 * sh('prgp'), f"The {P} {club} build through: {'more' if rk('prgp') == 1 else 'the second most'} progressive passes of anyone in the squad. Taking responsibility on the ball from {'centre-back' if pos == 'CB' else 'full-back' if pos == 'FB' else 'the base of midfield'} is a job most teams give to a midfielder.",
                [share_ev('prgp', 'progressive passes'), ('Progressive passes', per('prgp')), ('Passes into the box', per('ppa'))] + T[:1])
        if pos == 'ST' and ok(aer) and aer >= 75 and ok(zlong) and zlong >= 0.3:
            add(FP, 'Target man', 1.5 + aer / 100 + zlong / 2, f"{club} went long more than most, and he is the outlet: {top(aer)} of strikers for aerial duels won. The direct game only works if the ball sticks up front.",
                [('Aerial duels won', f"{r['aer_won_p90raw']:.0f}% ({FB_ord(int(round(aer)))} pct)"), ('Long balls as a share of passes', f"{r['direct']:.0%} ({zlong:+.1f} sd vs league)"), share_ev('npxg', 'non-penalty xG')])
        elif pos == 'ST' and ok(aer) and aer >= 90:
            add(FP, 'Wins everything in the air', 1.2 + aer / 100, f"{top(aer)} of strikers for aerial duels won: a reference point for crosses and clearances alike.", [('Aerial duels won', f"{r['aer_won_p90raw']:.0f}% ({FB_ord(int(round(aer)))} pct)")])
        if pos in ('ST', 'W', 'AM') and ok(sh('npxg'), rk('npxg')) and rk('npxg') == 1 and sh('npxg') >= 0.30:
            add(FP, 'The goal threat', 1.5 + 3 * sh('npxg'), f"{sh('npxg'):.0%} of the non-penalty xG among {club}'s regulars came through him: take him out and the chances dry up.", [share_ev('npxg', 'non-penalty xG'), ('Non-penalty xG', per('npxg')), ('Shots', per('sh'))])
        if pos in ('AM', 'W', 'CM', 'FB') and ok(sh('kp'), rk('kp')) and rk('kp') == 1 and sh('kp') >= 0.22:
            add(FP, 'Creative hub', 1.4 + 3 * sh('kp'), f"The main creator: {sh('kp'):.0%} of the key passes among {club}'s regulars.", [share_ev('kp', 'key passes'), ('Key passes', per('kp')), ('Shot-creating actions', per('sca'))])
        best = max((v for v in (threat, create) if np.isfinite(v)), default=np.nan)
        if pos in ('ST', 'W', 'AM', 'CM') and ok(zxg, best) and zxg <= -0.7 and best >= 80:
            add(FP, 'Carries a blunt attack', 1 + best / 50, f"{club} created little as a team, yet he is {top(best)} of {PL} for " + ('goal threat' if best == threat else 'chance creation') + '. Take him out and there is not much left.',
                [('Team xG per game (vs league)', f"{zxg:+.1f} sd"), ('Non-penalty xG', per('npxg')), ('Key passes', per('kp'))])

        # ---------- defensive intelligence ----------
        raw = self._dept(r, 'raw'); adj = self._dept(r, 'adj'); starter = r['min_num'] >= MIN_STARTER
        xga_ev = ('Team xG against', f"{r['xga_pm']:.2f} a game, {FB_ord(int(r['rank_xga']))} of {int(r['n_teams'])}") if ok(r.get('xga_pm', np.nan), r.get('rank_xga', np.nan)) else ('Team xG against', 'n/a')
        high_line = ok(poss, press) and poss >= 54 and press >= 0.5
        if pos in ('CB', 'FB') and starter and high_line and ok(zx, r.get('rank_xga', np.nan)) and zx <= -0.9 and r['rank_xga'] <= 4:
            add(DI, 'Holds a high line', 2.6 - zx / 2, f"{club} pressed high with {poss:.0f}% of the ball, which leaves space in behind, and still conceded very little with him playing {int(r['min_num']):,} minutes. Defending that much grass takes recovery pace, timing and the reading of when to step up.",
                T + [xga_ev, ('Interceptions', per('int')), ('Aerial duels won', f"{r['aer_won_p90raw']:.0f}%" if ok(r['aer_won_p90raw']) else 'n/a')])
        if pos in ('CB', 'FB', 'DM') and starter and ok(raw.get('Defending', np.nan), zx) and raw['Defending'] < 40 and zx <= -1.0:
            add(DI, 'Positionally sound', 2.5 - zx / 2, f"Few tackles and interceptions on paper, but {club} conceded only {r['xga_pm']:.2f} xG a game with him playing {int(r['min_num']):,} minutes. A defender in a side like that is rarely exposed: low numbers point to positioning and reading the game, not a weak defender.",
                [xga_ev, ('Defending (raw)', f"{FB_ord(int(round(raw['Defending'])))} pct"), ('Defending (possession-adjusted)', f"{FB_ord(int(round(adj['Defending'])))} pct")] + T[:1])
        if pos in ('CB', 'FB', 'DM') and starter and ok(zx, raw.get('Defending', np.nan)) and zx <= -0.8 and raw['Defending'] >= 70:
            add(DI, 'Anchor of a tight defence', 1.8 - zx / 2, f"Busy and effective: {top(raw['Defending'])} of {PL} for defensive actions in one of the tightest defences in the league.", [xga_ev, ('Tackles won', per('tklw')), ('Interceptions', per('int')), ('Blocks', per('blocks'))])

        # ---------- role note ----------
        sv = sup(['kp', 'prgp', 'ppa'])
        if pos in ('CB', 'FB', 'DM', 'CM') and ok(sv, poss) and sv <= -1.3 and poss >= 55:
            add('Role note', 'Keeps it simple', 0.5 - sv / 3, f"His creative and progressive passing is well below what a side with {poss:.0f}% of the ball predicts: a safe, low-risk passer in a team that gives him the ball.", T[:1] + [('Passing vs style', f"{sv:+.1f} sd")])
        out.sort(key=lambda x: -x['score']); seen = set(); res = []
        for x in out:
            if x['tag'] in seen: continue
            seen.add(x['tag']); res.append(x)
        return res[:k]

    def outliers(self, pid, yr, k=3):
        """[(tag, sentence)] for compact places (tags, dossier lines)."""
        return [(x['tag'], x['text'] + ' (' + '; '.join(f'{a}: {b}' for a, b in x['ev']) + ')') for x in self.reads(pid, yr, k)]


def FB_ord(n):
    return f"{n}" + ("th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))
