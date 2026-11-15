"""Shadow Squad: a scouting dossier. Run: streamlit run app/shadow_squad.py"""
import io, json, os, sys, urllib.parse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np, pandas as pd, streamlit as st, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import engine as E, ga as GA, formation as FM, scout_rag as RAG, football as FB
from common import MID

PITCH, TURF, CHALK, LINE, GOLD, MUTE = '#0f2a22', '#173b30', '#ece7da', '#2c5a4a', '#e0a43a', '#9db3a7'
st.set_page_config(page_title='Shadow Squad', layout='wide')
st.html(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap');
html, body, [class*="st-"], .stMarkdown p {{ font-family: 'Barlow', sans-serif; }}
[data-testid="stIconMaterial"], [data-testid="stExpanderToggleIcon"] {{ font-family: 'Material Symbols Rounded' !important; }}
.block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
h1, h2, h3, .disp {{ font-family: 'Barlow Condensed', sans-serif; letter-spacing: .01em; }}
.mast {{ display:flex; align-items:baseline; gap:18px; border-bottom:2px solid {GOLD}; padding-bottom:10px; margin-bottom:6px; }}
.mast .name {{ font:700 44px/1 'Barlow Condensed'; color:{CHALK}; }}
.mast .sub {{ color:{MUTE}; font-size:15px; }}
.badge {{ margin-left:auto; font:600 15px 'Barlow Condensed'; padding:5px 12px; border-radius:3px; }}
.badge.live {{ background:{GOLD}; color:{PITCH}; }} .badge.back {{ border:1px solid {MUTE}; color:{MUTE}; }}
.card {{ background:{TURF}; border-left:3px solid {LINE}; padding:14px 18px 13px; margin:0 0 10px; display:grid; grid-template-columns:64px 1fr auto; gap:4px 16px; }}
.card.first {{ border-left-color:{GOLD}; }}
.card .rk {{ font:700 40px/1 'Barlow Condensed'; color:{GOLD}; grid-row:1 / span 2; }}
.card .rk small {{ display:block; font:500 13px 'Barlow'; color:{MUTE}; margin-top:4px; }}
.card .who {{ font:600 25px/1.1 'Barlow Condensed'; color:{CHALK}; }}
.card .meta {{ color:{MUTE}; font-size:14px; }}
.card .val {{ font:600 24px/1 'Barlow Condensed'; color:{CHALK}; text-align:right; }}
.card .val small {{ display:block; font:400 12.5px 'Barlow'; color:{MUTE}; margin-top:3px; }}
.card .why {{ grid-column:2 / span 2; font-size:14.5px; line-height:1.5; color:{CHALK}; margin-top:4px; }}
.card .pat {{ grid-column:2 / span 2; font-size:13px; color:{MUTE}; }}
.ini {{ display:inline-block; width:26px; height:26px; line-height:26px; text-align:center; border-radius:50%; background:{LINE}; color:{CHALK}; font:600 12px 'Barlow'; margin-right:8px; vertical-align:2px; }}
.lg {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }}
.note {{ color:{MUTE}; font-size:14px; line-height:1.55; max-width:70ch; }}
.big {{ font:700 34px/1 'Barlow Condensed'; color:{CHALK}; }} .big small {{ display:block; font:400 13px 'Barlow'; color:{MUTE}; margin-top:4px; }}
.st-key-pitch a, .st-key-pitch a:hover {{ text-decoration:none !important; }} a.slot {{ cursor:pointer; }} a.slot:focus circle {{ stroke:{CHALK}; stroke-width:3; }}
@media (max-width: 800px) {{ .card {{ grid-template-columns:48px 1fr; }} .card .val {{ grid-column:2; text-align:left; }} }}
</style>""")
LGCOL = {'Premier League': '#8f6bd6', 'La Liga': '#e0733a', 'Bundesliga': '#d64545', 'Serie A': '#3f8fd1', 'Ligue 1': '#3fb58a'}
L = E.L; qp = st.query_params
WORD = {'sim_translated': 'his numbers, translated into this club, match the starter', 'traj_sim': 'his last seasons developed like the starter\'s', 'pipeline_rank': 'his club is a regular seller to bigger clubs', 'club_taste': 'the club has bought this type of player before',
        'hazard': 'he is at the stage where players get bought', 'log_value': 'the market already rates him', 'value_gap': 'he is priced close to the starter', 'age_num': 'his age fits', 'composite_perf_index': 'his overall output is high for the position', 'pattern_match': 'his traits match what past signings at this position had'}


@st.cache_resource
def load():
    lake = E.Lake(); models = E.load_models(); ev = pd.read_parquet(L('replacement_events')); rd = lambda n: pd.read_parquet(L(n)) if os.path.exists(L(n)) else None
    return lake, models, ev, rd('live_squads.parquet'), rd('live_shortlists.parquet'), rd('value_history.parquet')
lake, models, EV, SQ, SL, VH = load(); F = lake.F
def pick(label, options, key, fmt=str, col=st):
    options = list(options); d = qp.get(key); i = next((n for n, o in enumerate(options) if str(o) == d), 0); v = col.selectbox(label, options, index=i, format_func=fmt, key='w_' + key); qp[key] = str(v); return v
ini = lambda n: ''.join(w[0] for w in str(n).replace('-', ' ').split()[:2]).upper()
eur = lambda v: '-' if pd.isna(v) else (f'EUR {v / 1e6:.1f}m' if v < 1e7 else f'EUR {v / 1e6:.0f}m')

# ---------------- masthead and controls ----------------
seasons = (['live'] if SQ is not None else []) + sorted(F.loc[F['elite_club'], 'yr'].unique().tolist(), reverse=True)
LEAGUE_ORDER = ['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1']
c1, c2, c3, c4 = st.columns([1.15, 1.05, 1.35, 2.2]); season = pick('Season', seasons, 'season', lambda s: 'Now (after 2025-26)' if s == 'live' else f'{s}-{str(int(s) + 1)[2:]} (backtest)', c1); live = season == 'live'
if live: pool_sq = SQ.rename(columns={'name': 'player', 'age_today': 'age'}).assign(comp=lambda d: d['league'], left=np.nan)
else:
    t = int(season); pool_sq = F[(F['yr'] == t) & (F['min_num'] >= MID['ELITE_MIN_MIN'])].rename(columns={'age_num': 'age', 'min_num': 'minutes'}).copy()
    gone = set(EV[(EV['t'] == t)]['S']); pool_sq['left'] = pool_sq['player_id'].isin(gone); pool_sq['risk'] = pool_sq['left'].astype(float)
leagues = [l for l in LEAGUE_ORDER if l in set(pool_sq['comp'])] + sorted(set(pool_sq['comp']) - set(LEAGUE_ORDER))
if not qp.get('league') and qp.get('club'):      # old links (and src/screens.py) carry only the club: open its league
    hit = pool_sq.loc[pool_sq['club' if live else 'squad'] == qp.get('club'), 'comp']
    if len(hit): qp['league'] = hit.iat[0]
league = pick('League', leagues, 'league', col=c2); lsq = pool_sq[pool_sq['comp'] == league]; clubcol = 'club' if live else 'squad'
elite = set(lsq.loc[lsq['elite_club'], clubcol]) if 'elite_club' in lsq else set(lsq[clubcol])
clubs = sorted(lsq[clubcol].unique(), key=lambda c: (c not in elite, c)) if not live else sorted(lsq[clubcol].unique())
club = pick('Club', clubs, 'club', col=c3, fmt=lambda c: c if live or c in elite else f'{c} \u00b7 other club'); squad = lsq[lsq[clubcol] == club].copy()
view = c4.segmented_control('View', ['Squad', 'Shadow list', 'Compare', 'Matrix', 'Budget rebuild'], default=qp.get('view', 'Squad') if qp.get('view') in ('Squad', 'Shadow list', 'Compare', 'Matrix', 'Budget rebuild') else 'Squad', key='w_view') or 'Squad'; qp['view'] = view
st.html(f"""<div class="mast"><span class="name">{club}</span><span class="sub">{league} \u00b7 {'who could leave, and who could come in' if live else 'what the list would have said that summer'}</span>
<span class="badge {'live' if live else 'back'}">{'Live: data to July 2026' if live else f'Backtest {season}-{str(int(season) + 1)[2:]}'}</span></div>""")
if live: st.html(f'<div class="note">Live shortlists are precomputed for the top clubs only ({SQ["club"].nunique()} across the five leagues, {len(clubs)} in {league}). Pick a past season to see every club in the league.</div>')
elif club not in elite: st.html('<div class="note">The ranking models learned from signings by top clubs. For this club, read the list as players who fit the profile and price of the starter, not as a forecast of what the club will do.</div>')
squad = squad.sort_values('minutes', ascending=False).head(14); sid = qp.get('starter'); sid = int(sid) if sid and sid.isdigit() and int(sid) in set(squad['player_id']) else int(squad.sort_values('risk', ascending=False)['player_id'].iat[0])
srow = squad[squad['player_id'] == sid].iloc[0]


@st.cache_data(show_spinner=False)
def shortlist(live, club, season, sid):
    """One table for both modes: rank, interval, reasons, patterns, value, and the vectors the radar needs."""
    if live:
        q = SQ[(SQ['club'] == club) & (SQ['player_id'] == sid)]['qid'].iat[0]; X = SL[SL['qid'] == q].sort_values('rank').copy(); X['player'] = X['name']; X['squad'] = X['club']; X['comp'] = X['league']; X['age'] = X['age_today']; X['why'] = X['reasons']; X['patterns'] = ''; X['n_pool'] = np.nan; return X
    r = F[(F['player_id'] == sid) & (F['yr'] == int(season))].iloc[0]; X = E.rank(lake, models, int(r['club_id']), int(season), int(sid), r['fine_pos'], top=10 ** 6); key = X['model_used'].iat[0]; m = models[int(key) if key != 'all' else 'all']
    mr = np.stack([pd.Series(mm.predict(X[E.FEATS])).rank(ascending=False, method='min').to_numpy() for mm in m.ms], 1); X['rank'] = np.arange(1, len(X) + 1); X['n_pool'] = len(X); X['score_pool_median'] = float(X['score'].median())
    for f in E.FEATS: X['pct_' + f] = X[f].rank(pct=True) * 100
    X['rank_lo'] = np.quantile(mr, .1, axis=1).round().astype(int); X['rank_hi'] = np.quantile(mr, .9, axis=1).round().astype(int); X = X.head(15).copy()
    import viz; ex = viz.explain(m, X[E.FEATS]); order = np.argsort(-ex.values, axis=1); why = []
    for i in range(len(X)):
        pos = [WORD[E.FEATS[j]] for j in order[i, :3] if ex.values[i, j] > 0]; neg = E.FEATS[order[i, -1]]; why.append((('Ranked here mainly because ' + ', and '.join(pos[:2]) + ('; also ' + pos[2] if len(pos) > 2 else '') + '.') if pos else 'No single score stands out.') + (f" Held back because {WORD[neg].replace('his ', 'his ').replace('matches', 'does not match').replace('match ', 'do not match ').replace(' is high', ' is not high').replace('fits', 'does not fit').replace('already rates', 'does not yet rate').replace(' is priced close', ' is not priced close').replace('is a regular', 'is not a regular').replace('has bought', 'has not bought').replace('is at the stage', 'is not yet at the stage').replace('developed like', 'did not develop like')}." if ex.values[i, order[i, -1]] < -0.05 else ''))
    for j, f in enumerate(E.FEATS): X['shap_' + f] = ex.values[:, j]
    X['why'] = why; X['patterns'] = [f"Matches {int(a)} of the {int(b)} maximal trait profiles past signings at this position had: {c}" if pd.notna(a) and a > 0 else '' for a, b, c in zip(X.get('n_maximal_matched', np.nan), X.get('n_maximal_total', np.nan), X.get('maximal_matched', ''))]
    X['age'] = X['age_num']; X['ref_value'] = X['S_value']; return X
X = shortlist(live, club, season, sid)


def pitch(squad, sel, formation='auto'):
    """The squad in its formation: one slot per player, inferred from positions and minutes (src/formation.py). Players outside the XI sit on the bench strip."""
    sq = squad.reset_index(drop=True); name, slots, bench = FM.layout(sq['fine_pos'], sq['minutes'], formation); out = []
    def href(pid): return f"?season={season}&league={urllib.parse.quote(league)}&club={urllib.parse.quote(club)}&starter={pid}&view=Shadow%20list"
    def token(r, x, y, rad):
        k = float(np.clip(np.nan_to_num(r.risk), 0, 1)); s = r.player_id == sel; nm = str(r.player).split()[-1][:12]; rr = rad + 3 if s else rad
        return (f'<a class="slot" href="{href(r.player_id)}" target="_self"><title>{r.player} ({r.fine_pos}): departure risk {k:.0%}</title>'
                f'<circle cx="{x:.0f}" cy="{y}" r="{rr}" fill="{GOLD}" fill-opacity="{0.12 + 0.88 * k:.2f}" stroke="{CHALK if s else GOLD}" stroke-width="{2.5 if s else 1.2}"/>'
                f'<text x="{x:.0f}" y="{y + 4}" text-anchor="middle" font-family="Barlow Condensed" font-weight="600" font-size="{12 if rad >= 14 else 10.5}" fill="{PITCH if k > .5 else CHALK}">{ini(r.player)}</text>'
                f'<text x="{x:.0f}" y="{y + rr + 12}" text-anchor="middle" font-family="Barlow" font-size="9.5" fill="{CHALK}">{nm}</text></a>')
    for role, x, y, j in slots:
        if j is None: out.append(f'<circle cx="{x}" cy="{y}" r="14" fill="none" stroke="{MUTE}" stroke-dasharray="3 3" stroke-width="1"/><text x="{x}" y="{y + 4}" text-anchor="middle" font-family="Barlow" font-size="9" fill="{MUTE}">{role}</text>')
        else: out.append(token(sq.iloc[j], x, y, 14))
    gx, gy = FM.GK_SLOT; out.append(f'<circle cx="{gx}" cy="{gy}" r="12" fill="none" stroke="{LINE}" stroke-width="1.2"/><text x="{gx}" y="{gy + 4}" text-anchor="middle" font-family="Barlow" font-size="9" fill="{MUTE}">GK</text>')
    rows = [bench[i:i + 6] for i in range(0, len(bench), 6)]; H = 380 + (34 + 50 * len(rows) if rows else 0)
    if rows: out.append(f'<text x="18" y="400" font-family="Barlow" font-size="10" fill="{MUTE}">Also in the squad</text>')
    for ri, row in enumerate(rows):
        xs = np.linspace(40, 320, 6)[:len(row)] if len(row) > 1 else [40]
        out += [token(sq.iloc[j], x, 426 + 50 * ri, 11) for x, j in zip(xs, row)]
    ln = f'stroke="{LINE}" stroke-width="1.4" fill="none"'
    svg = (f'<svg viewBox="0 0 360 {H}" role="img" aria-label="{club} in a {name}, shaded by departure risk" style="width:100%;max-width:520px;background:{TURF};border-radius:4px">'
           f'<rect x="14" y="14" width="332" height="352" {ln}/><line x1="14" y1="190" x2="346" y2="190" {ln}/><circle cx="180" cy="190" r="34" {ln}/><rect x="100" y="14" width="160" height="52" {ln}/><rect x="100" y="314" width="160" height="52" {ln}/>'
           f'<rect x="146" y="14" width="68" height="20" {ln}/><rect x="146" y="346" width="68" height="20" {ln}/>'
           f'<text x="340" y="30" text-anchor="end" font-family="Barlow Condensed" font-weight="600" font-size="15" fill="{GOLD}">{name}</text>' + ''.join(out) + '</svg>')
    return svg, name


def radar(labels, series):
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False); a = np.append(ang, ang[0]); fig, ax = plt.subplots(figsize=(4.8, 4.8), subplot_kw={'polar': True}); fig.patch.set_facecolor(PITCH); ax.set_facecolor(TURF)
    for name, v, col, ls, lw in series:
        v = np.clip(np.nan_to_num(np.asarray(v, float)), -2.5, 2.5); ax.plot(a, np.append(v, v[0]), color=col, lw=lw, ls=ls, label=name)
        if ls == '-' and col == CHALK: ax.fill(a, np.append(v, v[0]), color=CHALK, alpha=.07)
    ax.set_xticks(ang, labels, fontsize=7.5, color=MUTE); ax.set_ylim(-2.6, 2.6); ax.set_yticks([-2, 0, 2], ['', '', ''], fontsize=6); ax.grid(color=LINE, lw=.7); ax.spines['polar'].set_color(LINE)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), fontsize=8, frameon=False, labelcolor=CHALK); fig.tight_layout(); return fig


def spark(pid, w=260, h=54):
    if VH is None: return ''
    v = VH[VH['player_id'] == pid].sort_values('date')
    if len(v) < 3: return ''
    x = np.linspace(4, w - 4, len(v)); y = h - 6 - (v['value'] / v['value'].max()).to_numpy() * (h - 14); pts = ' '.join(f'{a:.1f},{b:.1f}' for a, b in zip(x, y))
    return f'<svg viewBox="0 0 {w} {h}" style="width:100%;max-width:{w}px" role="img" aria-label="Market value history"><polyline points="{pts}" fill="none" stroke="{GOLD}" stroke-width="2"/><circle cx="{x[-1]:.1f}" cy="{y[-1]:.1f}" r="3" fill="{GOLD}"/></svg><div class="note">Market value, {v["date"].iat[0].year} to {v["date"].iat[-1].year}; peak {eur(v["value"].max())}</div>'


import viz
NICE = viz.NICE
@st.cache_resource
def retriever(): return RAG.Retriever()
@st.cache_data
def engine_record():
    """Backtest headline numbers for the 'how far to trust this' note, read from the evaluation table the notebooks wrote."""
    p = os.path.join(E.ROOT, 'outputs', 'master_results.csv')
    if not os.path.exists(p): return None
    m = pd.read_csv(p).set_index('method'); g = lambda k: m.loc[k] if k in m.index else None
    try: n = json.load(open(os.path.join(E.ROOT, 'outputs', 'metrics.json')))['phase5']['main']['test_queries']
    except Exception: n = None
    return {'engine': g('Shadow Squad (trained on all signings)'), 'random': g('Random order of the pool'), 'market': g('Highest market value'), 'n': n}


def secret(k):
    """A value from the Streamlit secrets, falling back to the environment (st.secrets raises when there is no secrets file)."""
    try:
        if k in st.secrets: return st.secrets[k]
    except Exception: pass
    return os.environ.get(k)


def evidence(r, cid):
    """Everything the explainer knows about one candidate: feature rows (label, value, pool percentile, SHAP), per-metric z-scores and their share of the similarity."""
    if live:
        ZL = [c for c in SL.columns if c.startswith('z_')]; s0 = SQ[SQ['player_id'] == sid].iloc[0]; a = np.nan_to_num(s0[ZL].to_numpy(float)); b = np.nan_to_num(r[ZL].to_numpy(float))
        M = pd.DataFrame({'metric (per 90)': [c[2:] for c in ZL], 'starter z': a, 'candidate z': b})
        feats = pd.DataFrame({'feature': ['profile similarity (sim_profile)', 'market value', 'age', 'league minutes', 'composite output percentile'],
                              'value': [r['sim_profile'], r['value'], r['age'], r['minutes'], r['composite']]})
    else:
        t = int(season); fp = [r['fine_pos']]; xi, si = lake.idx[(cid, t)], lake.idx[(sid, t)]; z = lambda v: lake.zraw(np.asarray(v, float)[None, :], fp, t)[0]
        a = np.nan_to_num(z(lake.RAWV[si])); b = np.nan_to_num(z([r['tr_' + m] for m in E.MET])); raw = np.nan_to_num(z(lake.RAWV[xi]))
        M = pd.DataFrame({'metric (per 90)': E.MET, 'starter z': a, 'candidate raw z': raw, 'candidate translated z': b})
        feats = pd.DataFrame({'feature': [NICE[f] for f in E.FEATS], 'key': E.FEATS, 'value': [r[f] for f in E.FEATS], 'pool percentile': [r['pct_' + f] for f in E.FEATS], 'SHAP': [r['shap_' + f] for f in E.FEATS]})
        feats['share of |SHAP|'] = feats['SHAP'].abs() / max(feats['SHAP'].abs().sum(), 1e-12); feats = feats.sort_values('SHAP', key=np.abs, ascending=False).reset_index(drop=True)
    den = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12; M['gap (cand - starter)'] = b - a; M['share of similarity'] = a * b / den
    return feats, M.sort_values('share of similarity', ascending=False).reset_index(drop=True), float((a * b).sum() / den)


@st.cache_resource
def hindsight(): return FB.Hindsight(lake.F, SQ, SL, EV, VH, E.ROOT)


def hs_rows(Xh):
    """Where the players on a past list are now (backtest only)."""
    H = hindsight(); t = int(season)
    return {int(p): H.row(p, t, c, v, buyer=club) for p, c, v in zip(Xh['player_id'], Xh['squad'], Xh['value'])}


def hs_short(h):
    if h is None: return ''
    v = f"{eur(h['value_now'])}" + (f" ({h['value_change']:+.0%})" if np.isfinite(h['value_change']) else '') if np.isfinite(h['value_now']) else 'no recent value'
    return f"Now: {h['club_now'] or 'unknown'} · {v}" + (f" · {h['story']}" if h['story'] else '')


def scout(r, cid, feats, M):
    """The football reading of one candidate: roles, like-for-like, reasons, club style, comparison with No. 1, hindsight."""
    zc_col = 'candidate z' if live else 'candidate translated z'; met = M['metric (per 90)'].tolist()
    zs = dict(zip(met, M['starter z'])); zc = dict(zip(met, M[zc_col])); pos = srow['fine_pos']
    out = {'zs': zs, 'zc': zc, 'dims_s': FB.dims(zs, live), 'dims_c': FB.dims(zc, live), 'role_s': FB.role(zs, pos, live), 'role_c': FB.role(zc, pos, live)}
    out['keeps'], out['down'], out['adds'] = FB.like_for_like(zs, zc, live, pos)
    out['style'] = None; out['vs1'] = None; out['hs'] = None; out['top3'] = []
    if live:
        out['pros'] = [s.strip() for s in str(r['why']).split(';') if s.strip()]; out['cons'] = []
    else:
        sh = [(f, r['shap_' + f]) for f in E.FEATS]; out['pros'], out['cons'] = FB.why(sh, srow['player'], club)
        st_ = lake.STY[(lake.STY['club_id'] == srow['club_id']) & (lake.STY['yr'] == int(season))]
        if len(st_): out['style'] = FB.club_style(float(st_['x_dom'].iat[0]), float(st_['x_press'].iat[0]))
        if int(r['rank']) > 1:
            r1 = X.iloc[0]; gap = sorted(((f, r1['shap_' + f] - r['shap_' + f]) for f in E.FEATS), key=lambda x: -x[1])
            out['vs1'] = (r1['player'], [FB.WHY[f][0].format(S=srow['player'], C=club) for f, g in gap if g > 0.02][:2])
        H = hs_rows(X.head(3) if int(r['rank']) <= 3 else pd.concat([X.head(3), X[X['player_id'] == cid]]))
        out['hs'] = H.get(int(cid)); out['top3'] = [(x['player'], int(x['rank']), H[int(x['player_id'])]) for _, x in X.head(3).iterrows()]
    return out


def dossier(r, feats, M, sim, sc):
    """Everything the explainer may use, football reading first, technical detail last."""
    H = hindsight(); S = srow['player']; pos = FB.POS.get(srow['fine_pos'], srow['fine_pos']); names = FB.MET_LIVE if live else FB.MET_BT
    real = EV[(EV['t'] == int(season)) & (EV['S'] == sid)] if not live else EV.iloc[:0]
    L_ = [f"Mode: {'live shortlist for the next window, data to July 2026' if live else f'backtest: the list as it would have looked in the {season}-{str(int(season) + 1)[2:]} summer, using only data available then'}",
          f"Buying club: {club} ({league})." + (f" That season {club} were {sc['style']} (the engine's style indices)." if sc['style'] else ''),
          f"Player leaving: {S}, {pos}, age {srow['age']:.0f}, {int(srow['minutes']):,} league minutes, valued at {eur(srow.get('value', np.nan))}. Profile: {sc['role_s']}.",
          f"Candidate: {r['player']}, {r['squad']} ({r['comp']}), age {r['age']:.0f}, valued at {eur(r['value'])}. Profile: {sc['role_c']}.",
          f"Place on the shadow list: No. {int(r['rank'])}" + (f" of {int(r['n_pool'])} {FB.POS_PL.get(srow['fine_pos'], 'players')} considered" if pd.notna(r.get('n_pool', np.nan)) else ' of the 15 listed') + f"; across the ten models of the ensemble he ranks between {int(r['rank_lo'])} and {int(r['rank_hi'])}."]
    if sc['vs1']: L_.append(f"No. 1 on the list is {sc['vs1'][0]}; the model puts him ahead mainly because " + (', and '.join(sc['vs1'][1]) if sc['vs1'][1] else 'of small margins across several factors') + '.')
    L_.append('Why the model rates him (strongest first): ' + ('; '.join(sc['pros']) or 'no single reason stands out') + '.')
    if sc['cons']: L_.append('What counts against him: ' + '; '.join(sc['cons']) + '.')
    L_.append(f"Football dimensions, percentile tier among {FB.POS_PL.get(srow['fine_pos'], 'players')} in the five leagues (starter vs candidate" + ('' if live else ", the candidate's numbers adjusted to the buying club's style") + '):')
    for d in sc['dims_s']: L_.append(f"- {d}: {S} {FB.tier(sc['dims_s'][d])}, {r['player']} {FB.tier(sc['dims_c'].get(d, np.nan))}")
    for lab, items in (('Keeps what the club loses', sc['keeps']), ("Where he is a step down", sc['down']), ('What he adds', sc['adds'])):
        if items: L_.append(f"{lab}: " + '; '.join(items[:5]) + '.')
    if len(real): L_.append('What the club really did: signed ' + '; '.join(f'{x.N_name} from {x.N_from}' for x in real.itertuples()) + '.' + (' This candidate was the real signing.' if cid in set(real['N']) else ''))
    if sc['hs']: L_.append('Hindsight for this candidate: ' + H.line(sc['hs'], r['player']))
    if sc['top3']:
        L_.append('Hindsight for the top 3 of this list:')
        for n, k, h in sc['top3']: L_.append(f'- No. {k} ' + H.line(h, n))
    L_.append('Technical detail (use only if asked about the model or the statistics):')
    if not live:
        for x in feats.itertuples(): L_.append(f"- {x.feature} [{x.key}]: value {x.value:.3f}, pool percentile {x._4:.0f}, SHAP {x.SHAP:+.4f}")
    L_.append(f"- Profile similarity to the starter (cosine of per-90 z-scores): {sim:.3f}. Per-90 z-scores, starter / candidate: " + ', '.join(f"{names.get(m, m)} {sc['zs'][m]:+.2f} / {sc['zc'][m]:+.2f}" for m in sc['zs']))
    rec = engine_record()
    if rec and rec['engine'] is not None and not live:
        e, rd, mk = rec['engine'], rec['random'], rec['market']; L_.append(f"- Backtest accuracy on unseen seasons: the real signing was in the top 10 in {e['recall@10']:.0%} of cases (random order {rd['recall@10']:.0%}, highest market value {mk['recall@10']:.0%}); median rank {e['median_rank']:.0f}.")
    return '\n'.join(L_)


def shap_bars(feats):
    f = feats.sort_values('SHAP'); fig, ax = plt.subplots(figsize=(6.2, 3.6)); fig.patch.set_facecolor(PITCH); ax.set_facecolor(PITCH)
    ax.barh(f['feature'], f['SHAP'], color=[GOLD if v > 0 else MUTE for v in f['SHAP']], height=.62); ax.axvline(0, color=LINE, lw=1)
    for y, v in enumerate(f['SHAP']): ax.text(v + (0.004 if v >= 0 else -0.004), y, f'{v:+.3f}', va='center', ha='left' if v >= 0 else 'right', fontsize=7.5, color=CHALK)
    ax.tick_params(colors=CHALK, labelsize=8); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xlabel('SHAP contribution to his score (gold pushes him up)', color=MUTE, fontsize=8)
    lim = max(f['SHAP'].abs().max() * 1.35, 0.02); ax.set_xlim(-lim, lim); fig.tight_layout(); return fig


def dim_chart(sc, cname):
    """Starter and candidate on each football dimension, on a bottom-10% to top-10% scale for the position."""
    D = list(sc['dims_s']); fig, ax = plt.subplots(figsize=(6.4, 0.62 * len(D) + 1.1)); fig.patch.set_facecolor(PITCH); ax.set_facecolor(PITCH)
    for x0, x1, c in ((-2.4, -1.28, '#14322a'), (-1.28, -0.67, '#173b30'), (-0.67, 0.67, '#1b4537'), (0.67, 1.28, '#173b30'), (1.28, 2.4, '#14322a')): ax.axvspan(x0, x1, color=c, lw=0)
    for y, d in enumerate(D[::-1]):
        a, b = np.clip(sc['dims_s'][d], -2.3, 2.3), np.clip(sc['dims_c'].get(d, np.nan), -2.3, 2.3)
        ax.plot([a, b], [y, y], color=LINE, lw=3, zorder=1); ax.scatter([a], [y], s=90, color=CHALK, zorder=2, label=srow['player'] if y == 0 else None); ax.scatter([b], [y], s=90, color=GOLD, zorder=3, label=cname if y == 0 else None)
    ax.set_yticks(range(len(D)), D[::-1], color=CHALK, fontsize=9); ax.set_xlim(-2.4, 2.4); ax.set_xticks([-1.8, -0.97, 0, 0.97, 1.8], ['bottom 10%', 'bottom 25%', 'average', 'top 25%', 'top 10%'], color=MUTE, fontsize=8)
    ax.tick_params(length=0); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, labelcolor=CHALK, fontsize=8.5); fig.tight_layout(); return fig


def li(items, empty):
    return '<ul style="margin:6px 0 0 18px;padding:0">' + ''.join(f'<li class="note" style="margin:3px 0">{x}</li>' for x in items) + '</ul>' if items else f'<div class="note" style="margin-top:6px">{empty}</div>'


def explain_block(r, cid):
    feats, M, sim = evidence(r, cid); sc = scout(r, cid, feats, M); dos = dossier(r, feats, M, sim, sc); H = hindsight()
    tab1, tab2 = st.tabs(['Ask why', "Scout's report (for football nerds)"])
    with tab1:
        key = secret('ANTHROPIC_API_KEY'); hk = f'qa_{season}_{club}_{sid}_{cid}'; hist = st.session_state.setdefault(hk, [])
        drivers = ['live engine', 'sim_profile', 'value_gap'] if live else list(feats['key'].head(4))
        default_q = 'Why is he the right replacement, and where is he a risk?'
        if not live: st.caption('Try: "Why is he the best replacement?", "What would the team lose?", "Where are the top 3 now?"')
        with st.form(f'f_{hk}', clear_on_submit=True, border=False):
            q = st.text_input('Ask about this pick', placeholder=default_q, key=f'q_{hk}'); go = st.form_submit_button('Explain')
        if go: q = q.strip() or default_q
        for qq, aa, src in hist:
            with st.chat_message('user'): st.markdown(qq)
            with st.chat_message('assistant'): st.markdown(aa)
        if go:
            passages = retriever().search(RAG.retrieval_query(q, drivers + ['football', 'replacement']), k=4, exclude={'The ten ranking features (backtest engine)'} if live else {'Live engine'})
            with st.chat_message('user'): st.markdown(q)
            with st.chat_message('assistant'):
                ans = None
                if key:
                    try: ans = st.write_stream(RAG.stream_answer(q, dos, passages, key, [(a, b) for a, b, _ in hist], model=secret('CLAUDE_MODEL')))
                    except Exception as e: st.warning(f'The language model could not be reached ({type(e).__name__}); showing the built-in explanation instead.')
                if not ans:
                    ans = RAG.offline_answer({'name': r['player'], 'rank': int(r['rank']), 'lo': int(r['rank_lo']), 'hi': int(r['rank_hi']), 'starter': srow['player'], 'club': club,
                                              'role_s': sc['role_s'], 'role_c': sc['role_c'], 'pros': sc['pros'], 'cons': sc['cons'], 'keeps': sc['keeps'], 'down': sc['down'], 'adds': sc['adds'],
                                              'price': (eur(r['value']), eur(srow.get('value', np.nan))), 'age': (float(r['age']), float(srow['age'])), 'vs1': sc['vs1'], 'style': sc['style'],
                                              'dims': [(d, FB.tier(sc['dims_s'][d]), FB.tier(sc['dims_c'].get(d, np.nan))) for d in sc['dims_s']],
                                              'hs': H.line(sc['hs'], r['player']) if sc['hs'] else None, 'top3': [f'No. {k} ' + H.line(h, n) for n, k, h in sc['top3']], 'question': q})
                    st.markdown(ans)
                with st.expander('Notes this answer drew on'):
                    for p_ in passages: st.markdown(f"**{p_['title']}** (match {p_['score']:.2f})\n\n{p_['text']}")
            hist.append((q, ans, passages))
        if not key: st.caption('Free-form answers need an Anthropic API key in the app secrets (ANTHROPIC_API_KEY). Without one you get the built-in scout report, built from the same numbers.')
    with tab2:
        verdict = (f"Top of the list to replace {srow['player']}." if int(r['rank']) == 1 else f"No. {int(r['rank'])} on the list to replace {srow['player']}" + (f"; No. 1 is {sc['vs1'][0]}." if sc['vs1'] else '.'))
        st.html(f'<div class="big">{r["player"]}<small>{sc["role_c"][0].upper() + sc["role_c"][1:]}. {verdict}</small></div>'
                f'<div class="note" style="margin-top:8px">Replacing {sc["role_s"]}: {srow["player"]}, {eur(srow.get("value", np.nan))}, age {srow["age"]:.0f}. He costs {eur(r["value"])} at age {r["age"]:.0f}.'
                + (f' {club} were {sc["style"]} that season, and his numbers below are adjusted to that.' if sc['style'] else '') + '</div>')
        c_a, c_b = st.columns([1.15, 1])
        with c_a:
            fig = dim_chart(sc, r['player']); buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=150, facecolor=PITCH); plt.close(fig); st.image(buf, alt=f'{srow["player"]} and {r["player"]} on each football dimension')
            st.html(f'<div class="note">Each dot is the average of the per-90 numbers in that part of the game, ranked against every {FB.POS.get(srow["fine_pos"], "player")} in the five leagues.</div>')
        with c_b:
            st.html(f'<div class="disp" style="font:600 19px Barlow Condensed;color:{GOLD}">Why him</div>' + li([p[0].upper() + p[1:] for p in sc['pros']], 'No single reason stands out.')
                    + (f'<div class="disp" style="font:600 19px Barlow Condensed;color:{MUTE};margin-top:10px">Question marks</div>' + li([c[0].upper() + c[1:] for c in sc['cons']], '') if sc['cons'] else '')
                    + (f'<div class="note" style="margin-top:10px">Why {sc["vs1"][0]} is No. 1 instead: ' + (', and '.join(sc['vs1'][1]) if sc['vs1'][1] else 'small margins across several factors') + '.</div>' if sc['vs1'] else ''))
        k1, k2, k3 = st.columns(3)
        k1.html(f'<div class="disp" style="font:600 19px Barlow Condensed;color:{CHALK}">Keeps what you lose</div>' + li(sc['keeps'][:5], 'None of the starter\'s strengths is matched like for like.'))
        k2.html(f'<div class="disp" style="font:600 19px Barlow Condensed;color:{CHALK}">Step down</div>' + li(sc['down'][:5], 'No clear drop-off in any area.'))
        k3.html(f'<div class="disp" style="font:600 19px Barlow Condensed;color:{CHALK}">Adds</div>' + li(sc['adds'][:5], 'Nothing new the starter did not bring.'))
        if sc['top3']:
            st.html(f'<div class="disp" style="font:600 19px Barlow Condensed;color:{GOLD};margin-top:16px">Hindsight: where the top 3 are now</div>' + li([f'No. {k} ' + H.line(h, n) for n, k, h in sc['top3']], '')
                    + ('' if int(r['rank']) <= 3 or not sc['hs'] else f'<div class="note" style="margin-top:6px">This candidate: {H.line(sc["hs"], r["player"])}</div>')
                    + '<div class="note" style="margin-top:6px;font-size:12.5px">Clubs are the latest the data knows (the source is in brackets); values are the latest Transfermarkt valuation in the data. Run src/build_player_now.py after refreshing the Transfermarkt dump for fresher ones.</div>')
        with st.expander('The numbers behind it (for stats nerds)'):
            k1, k2, k3, k4 = st.columns(4)
            k1.html(f'<div class="big">{int(r["rank"])}<small>rank; {int(r["rank_lo"])} to {int(r["rank_hi"])} across the ten models</small></div>')
            k2.html(f'<div class="big">{r["score"]:.2f}<small>model score' + (f'; pool median {r["score_pool_median"]:.2f}' if 'score_pool_median' in r else '') + '</small></div>')
            k3.html(f'<div class="big">{sim:.2f}<small>profile similarity to the starter (cosine, -1 to 1)</small></div>')
            k4.html(f'<div class="big">{int(r["n_pool"]) if pd.notna(r.get("n_pool", np.nan)) else 15}<small>{"candidates scored for this query" if not live else "candidates on the live list"}</small></div>')
            if not live:
                fig = shap_bars(feats); buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=150, facecolor=PITCH); plt.close(fig); st.image(buf, alt='SHAP contribution of each feature to his score')
                st.dataframe(feats.drop(columns='key'), hide_index=True, width='stretch', column_config={'value': st.column_config.NumberColumn(format='%.3f'), 'pool percentile': st.column_config.ProgressColumn(min_value=0, max_value=100, format='%.0f'), 'SHAP': st.column_config.NumberColumn(format='%+.4f'), 'share of |SHAP|': st.column_config.NumberColumn(format='percent')})
            Mf = M.copy(); Mf.insert(1, 'in football terms', [(FB.MET_LIVE if live else FB.MET_BT).get(m, m) for m in M['metric (per 90)']])
            st.dataframe(Mf, hide_index=True, width='stretch', column_config={c: st.column_config.NumberColumn(format='%+.2f') for c in M.columns[1:]})
            rec = engine_record()
            if rec and rec['engine'] is not None and not live:
                e, rd, mk = rec['engine'], rec['random'], rec['market']
                st.html(f'<div class="note" style="margin-top:10px"><b style="color:{CHALK}">How far to trust a rank.</b> On {rec["n"] or "the"} replacement events in unseen seasons (2020 to 2024) the real signing was in this engine\'s top 10 in {e["recall@10"]:.0%} of events (95% interval {e["recall@10_lo"]:.0%} to {e["recall@10_hi"]:.0%}) and its median rank was {e["median_rank"]:.0f}. Random order: {rd["recall@10"]:.0%} and {rd["median_rank"]:.0f}. Highest market value: {mk["recall@10"]:.0%} and {mk["median_rank"]:.0f}.</div>')
            st.code(dos, language=None)


@st.cache_resource
def pct_table(): return FB.season_pct(lake.F)


ordn = lambda p: f"{p:.0f}" + ("th" if 10 <= round(p) % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(round(p) % 10, "th"))
UP, DN, DOT = "\u25b2", "\u25bc", "\u00b7"


def heat(p):
    """Cell colour for a percentile: dark turf at the bottom, gold at the top."""
    if not np.isfinite(p): return 'background:#14322a;color:#5f7a6e'
    a = p / 100; return f'background:rgba(224,164,58,{0.06 + 0.82 * a ** 1.6:.2f});color:{PITCH if p >= 75 else CHALK}'


def matrix_html(cols, data, pos, live, show_raw):
    """Departments x metrics against the starter and his replacements. cols: [(pid, header html)]; data: {pid: {metric: (pct, per90, season)}}."""
    depts = FB.DEPT_LIVE if live else FB.DEPT_BT; order = [d for d in FB.DEPT_ORDER.get(pos, list(depts)) if d in depts] + [d for d in depts if d not in FB.DEPT_ORDER.get(pos, [])]
    names = FB.MET_LIVE if live else FB.MET_BT; s0 = cols[0][0]; th = 'padding:8px 10px;font:600 14px Barlow Condensed;color:' + CHALK
    h = ['<div style="overflow-x:auto"><table style="border-collapse:separate;border-spacing:2px;width:100%;font-family:Barlow;font-size:13px">',
         f'<tr><th style="{th};text-align:left">Department / metric (per 90)</th>' + ''.join(f'<th style="{th};text-align:center;min-width:92px;{"border-bottom:2px solid " + CHALK if i == 0 else ""}">{c}</th>' for i, (_, c) in enumerate(cols)) + '</tr>']
    for d in order:
        ms = depts[d]; avg = {pid: np.nanmean([data[pid][m][0] for m in ms]) if any(np.isfinite(data[pid][m][0]) for m in ms) else np.nan for pid, _ in cols}
        h.append(f'<tr><td style="padding:9px 10px 5px;font:600 16px Barlow Condensed;color:{GOLD}">{d}</td>' + ''.join(f'<td style="{heat(avg[pid])};text-align:center;font:600 15px Barlow Condensed;border-radius:3px">{"" if not np.isfinite(avg[pid]) else f"{avg[pid]:.0f}"}' + (f' <span style="font-size:11px">{UP if avg[pid] - avg[s0] >= 10 else DN if avg[pid] - avg[s0] <= -10 else ""}</span>' if pid != s0 and np.isfinite(avg[pid]) and np.isfinite(avg[s0]) else '') + '</td>' for pid, _ in cols) + '</tr>')
        for m in ms:
            cells = []
            for pid, _ in cols:
                pc, raw, yr = data[pid][m]; txt = '-' if not np.isfinite(pc) else ((f'{raw:.0f}%' if m == 'aer_won' else f'{raw:.2f}') if show_raw and np.isfinite(raw) else f'{pc:.0f}')
                sub = (f'<div style="font-size:10px;opacity:.75">{ordn(pc)} pct' + (f' {DOT} {yr}-{str(yr + 1)[2:]}' if yr is not None and show_raw else '') + '</div>') if show_raw and np.isfinite(pc) else ''
                cells.append(f'<td style="{heat(pc)};text-align:center;padding:5px 4px;border-radius:2px">{txt}{sub}</td>')
            h.append(f'<tr><td style="padding:4px 10px 4px 22px;color:{MUTE}">{names.get(m, m)}</td>' + ''.join(cells) + '</tr>')
    return ''.join(h) + '</table></div>'


if view == 'Squad':
    a, b = st.columns([1.05, 1])
    fo = a.selectbox('Formation', ['auto'] + list(FM.TEMPLATES), format_func=lambda f: 'Auto (from positions and minutes)' if f == 'auto' else f, key='w_form')
    with a.container(key='pitch'): svg, fname = pitch(squad, sid, fo); st.markdown(svg, unsafe_allow_html=True); st.html(f'<div class="note">{"The fuller the circle, the higher the estimated chance the player is gone by next season." if live else "Filled circles are starters who played under 450 league minutes for the club the next season."} Select a player to open his shadow list.</div>')
    with b:
        st.html(f'<div class="big">{srow["player"]}<small>{srow["fine_pos"]}, age {srow["age"]:.0f}, {int(srow["minutes"]):,} league minutes, {eur(srow.get("value", np.nan))}</small></div>')
        if live: st.html(f'<div class="note" style="margin-top:14px">Departure risk <b style="color:{GOLD}">{srow["risk"]:.0%}</b>. Contract until {srow["contract_until"] if pd.notna(srow["contract_until"]) else "unknown"} (shown for context; the risk model does not use it, because past contract dates are not in the data).</div>')
        else:
            real = EV[(EV['t'] == int(season)) & (EV['S'] == sid)]
            st.html('<div class="note" style="margin-top:14px">' + ('He left or lost his place. ' + ('The club signed ' + '; '.join(f"{r.N_name} from {r.N_from}" for r in real.itertuples()) + '.' if len(real) else '') if srow['left'] else 'He stayed and played the following season.') + '</div>')
        HS3 = {} if live else hs_rows(X.head(3))
        st.html('<div class="note" style="margin-top:18px">First three names on his shadow list' + ('' if live else ', and where they are now') + '</div>' + ''.join(f'<div class="card {"first" if i == 0 else ""}"><div class="rk">{int(r["rank"])}</div><div class="who">{r["player"]}</div><div class="val">{eur(r["value"])}<small>{"value now" if live else "value then"}</small></div><div class="meta"><span class="lg" style="background:{LGCOL.get(r["comp"], MUTE)}"></span>{r["squad"]}, age {r["age"]:.0f}</div>' + (f'<div class="pat">{hs_short(HS3.get(int(r["player_id"])))}</div>' if HS3 else '') + '</div>' for i, (_, r) in enumerate(X.head(3).iterrows())))
elif view == 'Shadow list':
    o1, o2 = st.columns([2.2, 1]); sort = o2.segmented_control('Sort by', ['Fit', 'Value', 'Value for money'], default='Fit', key='w_sort') or 'Fit'; Y = X.copy(); Y['vfm'] = (len(Y) + 1 - Y['rank']) / np.log10(Y['value'].clip(lower=1e5))
    Y = Y.sort_values({'Fit': 'rank', 'Value': 'value', 'Value for money': 'vfm'}[sort], ascending=sort != 'Value for money')
    o1.html(f'<div class="note">Replacing <b style="color:{CHALK}">{srow["player"]}</b> ({srow["fine_pos"]}, {eur(srow.get("value", np.nan))}). Rank is shown with the range it takes across the ten models in the ensemble.' + ('' if live else (' What happened: ' + '; '.join(f"{r.N_name} from {r.N_from}" for r in EV[(EV["t"] == int(season)) & (EV["S"] == sid)].itertuples()) + '.' if len(EV[(EV["t"] == int(season)) & (EV["S"] == sid)]) else '')) + '</div>')
    with st.container(key='cards'):
        st.html(''.join(f'<div class="card {"first" if int(r["rank"]) == 1 else ""}"><div class="rk">{int(r["rank"])}<small>{int(r["rank_lo"])} to {int(r["rank_hi"])}</small></div><div class="who"><span class="ini">{ini(r["player"])}</span>{r["player"]}</div>'
                        f'<div class="val">{eur(r["value"])}<small>starter {eur(r["ref_value"])}</small></div><div class="meta"><span class="lg" style="background:{LGCOL.get(r["comp"], MUTE)}"></span>{r["squad"]}, {r["comp"]}, age {r["age"]:.0f}</div>'
                        f'<div class="why">{r["why"]}</div>' + (f'<div class="pat">{r["patterns"]}</div>' if r["patterns"] else '') + '</div>' for _, r in Y.head(10).iterrows()))
    if not live:
        HSa = hs_rows(X.head(10)); st.html(f'<div class="disp" style="font:600 22px Barlow Condensed;color:{GOLD};margin-top:18px">Hindsight: where the top 10 are now</div>')
        st.dataframe(pd.DataFrame([{'rank': int(x['rank']), 'player': x['player'], 'club then': h['club_then'], 'value then': h['value_then'] / 1e6, 'club now': h['club_now'], 'as of': h['club_now_source'],
                                    'value now': h['value_now'] / 1e6, 'change': h['value_change'], 'what happened': h['story']} for _, x in X.head(10).iterrows() for h in [HSa[int(x['player_id'])]]]),
                     hide_index=True, width='stretch', column_config={'value then': st.column_config.NumberColumn('value then (EUR m)', format='%.1f'), 'value now': st.column_config.NumberColumn('value now (EUR m)', format='%.1f'), 'change': st.column_config.NumberColumn(format='percent')})
        st.html('<div class="note">Club now is the latest the data knows (source in "as of"); value now is the latest Transfermarkt valuation in the data, up to June 2026. Run src/build_player_now.py after refreshing the Transfermarkt dump for fresher numbers.</div>')
elif view == 'Compare':
    cid = pick('Candidate', X['player_id'].tolist(), 'cand', lambda i: f"{int(X.loc[X['player_id'] == i, 'rank'].iat[0])}. {X.loc[X['player_id'] == i, 'player'].iat[0]}"); r = X[X['player_id'] == cid].iloc[0]; a, b = st.columns([1, 1.05])
    with a.container(key='radar'):
        if live:
            ZL = [c for c in SL.columns if c.startswith('z_')]; s0 = SQ[SQ['player_id'] == sid].iloc[0]; fig = radar([c[2:] for c in ZL], [(f'{srow["player"]} (starter)', s0[ZL], CHALK, '-', 2.2), (r['player'], r[ZL], GOLD, '-', 2.2)])
        else:
            t = int(season); fp = [r['fine_pos']]; xi, si = lake.idx[(cid, t)], lake.idx[(sid, t)]; z = lambda v: lake.zraw(np.asarray(v, float)[None, :], fp, t)[0]
            fig = radar(E.MET, [(f'{srow["player"]} at the club', z(lake.RAWV[si]), CHALK, '-', 2.2), (f'{r["player"]}, raw', z(lake.RAWV[xi]), MUTE, '--', 1.3), (f'{r["player"]}, translated into the club', z([r['tr_' + m] for m in E.MET]), GOLD, '-', 2.2)])
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=150, facecolor=PITCH); plt.close(fig); st.image(buf, alt=f'Radar comparing {srow["player"]} and {r["player"]}'); st.html('<div class="note">Per-90 numbers in standard deviations for the position; further out is more.</div>')
    with b.container(key='why'):
        st.html(f'<div class="big">{r["player"]}<small>{r["squad"]}, age {r["age"]:.0f}. Rank {int(r["rank"])} ({int(r["rank_lo"])} to {int(r["rank_hi"])})</small></div><div class="why" style="margin:16px 0 6px;font-size:16px;line-height:1.55;max-width:60ch">{r["why"]}</div>' + (f'<div class="note">{r["patterns"]}</div>' if r['patterns'] else ''))
        m1, m2 = st.columns(2); m1.html(f'<div class="big">{eur(r["value"])}<small>his market value</small></div>'); m2.html(f'<div class="big">{eur(r["ref_value"])}<small>the starter\'s</small></div>'); st.markdown('<div style="margin-top:18px">' + spark(int(cid)) + '</div>', unsafe_allow_html=True)
    st.divider(); explain_block(r, cid)
elif view == 'Matrix':
    m1, m2, m3 = st.columns([1.2, 1.4, 1.4]); n = m1.slider('Replacements to compare', 3, 10, 5, key='w_mn')
    mode = 'This season' if live else (m2.segmented_control('Numbers', ['That season', 'Peak (best of last 3 seasons)'], default='That season', key='w_mpeak') or 'That season')
    show = m3.segmented_control('Show', ['Percentile', 'Per 90 + percentile'], default='Per 90 + percentile' if not live else 'Percentile', key='w_mraw') or 'Percentile'
    Xs = X.head(n); peak = mode.startswith('Peak')
    if live:
        ZL = [c for c in SL.columns if c.startswith('z_')]; s0 = SQ[SQ['player_id'] == sid]
        data = FB.matrix_live(pd.concat([s0[['player_id'] + ZL], Xs[['player_id'] + ZL]]), ZL); show_raw = False
    else:
        data = FB.matrix_bt(pct_table(), [sid] + Xs['player_id'].tolist(), int(season), peak=peak); show_raw = show != 'Percentile'
    cols = [(sid, f'{srow["player"]}<div style="font:400 11px Barlow;color:{MUTE}">leaving \u00b7 {eur(srow.get("value", np.nan))}</div>')] + [(int(x['player_id']), f'{int(x["rank"])}. {x["player"]}<div style="font:400 11px Barlow;color:{MUTE}">{x["squad"]} \u00b7 {eur(x["value"])}</div>') for _, x in Xs.iterrows()]
    st.html(f'<div class="note" style="margin-bottom:8px">{srow["player"]} against the first {n} names on his shadow list, department by department, ordered by what matters most for a {FB.POS.get(srow["fine_pos"], "player")}. Each cell is the percentile among {FB.POS_PL.get(srow["fine_pos"], "players")} in the five leagues that season (900+ minutes)' + ('' if live else ', with the per-90 number above it') + '. Department rows average their metrics; \u25b2 / \u25bc mark a replacement 10+ points above or below the starter.' + (' Peak takes each player\'s best season of the last three for each metric, so it shows his ceiling rather than his latest form.' if peak else '') + (' Live numbers are Understat per-90s; percentiles are estimated from the standardised scores.' if live else ' These are the numbers the players actually posted at their clubs, before the style adjustment the engine applies.') + '</div>')
    st.html(matrix_html(cols, data, srow['fine_pos'], live, show_raw))
else:
    risky = squad.sort_values('risk', ascending=False); default = risky['player_id'].head(3).tolist() if live else (risky[risky['risk'] >= 1]['player_id'].head(4).tolist() if (risky['risk'] >= 1).sum() >= 2 else risky['player_id'].head(2).tolist())
    chosen = st.multiselect('Starters to replace', squad['player_id'].tolist(), default=default, format_func=lambda i: f"{squad.loc[squad['player_id'] == i, 'player'].iat[0]} ({squad.loc[squad['player_id'] == i, 'fine_pos'].iat[0]})", key='w_slots')
    if len(chosen) < 2: st.html('<div class="note">Pick at least two starters. One position needs no optimiser: take the first name on its shadow list.</div>')
    else:
        lists = [shortlist(live, club, season, int(i)).dropna(subset=['value']).head(16) for i in chosen]; K = min(len(l) for l in lists); lists = [l.head(K) for l in lists]; q = np.stack([(K - np.arange(K)) / K for _ in lists]); pr = np.stack([l['value'].to_numpy() / 1e6 for l in lists]); pid = np.stack([l['player_id'].to_numpy() for l in lists])
        worth = float(np.nansum([squad.loc[squad['player_id'] == i, 'value'].iat[0] if 'value' in squad else np.nan for i in chosen]) / 1e6) or float(pr.mean() * len(chosen)); lo = float(np.sort(pr, 1)[:, 0].sum()); hi = float(pr.max(1).sum())
        bud = st.slider('Budget, EUR million', min_value=int(np.ceil(lo)), max_value=int(np.ceil(hi)) + 1, value=int(np.clip(worth, np.ceil(lo), hi)), key='w_bud'); bits = int(np.ceil(np.log2(K)))
        ipick, iopt, ok = GA.squad_ilp(q, pr, pid, bud); best, bf, hist = GA.run(GA.squad_fitness(q, pr, pid, bud, bits), bits * len(lists), pop=60, gens=40, seed=42); gidx = np.minimum(GA.decode(best, len(lists), bits), K - 1); ar = np.arange(len(lists)); gq = q[ar, gidx].sum(); gcost = pr[ar, gidx].sum()
        a, b, c = st.columns(3); a.html(f'<div class="big">{gq / iopt:.0%}<small>of the best possible fit, reached by the genetic algorithm</small></div>'); b.html(f'<div class="big">EUR {gcost:.0f}m<small>spent of EUR {bud}m (optimum spends EUR {pr[ar, ipick].sum():.0f}m)</small></div>'); c.html(f'<div class="big">{int(np.argmax([h[0] for h in hist]))}<small>generations until the best squad appeared (of 40)</small></div>')
        st.dataframe(pd.DataFrame({'replaces': [squad.loc[squad['player_id'] == i, 'player'].iat[0] for i in chosen], 'genetic algorithm': [l['player'].iat[k] for l, k in zip(lists, gidx)], 'GA price (EUR m)': pr[ar, gidx].round(1), 'exact optimum (ILP)': [l['player'].iat[k] for l, k in zip(lists, ipick)], 'ILP price (EUR m)': pr[ar, ipick].round(1)}), hide_index=True, alt='Replacement squad chosen by the genetic algorithm and by integer programming')
        st.html('<div class="note">Each position has its best-ranked candidates. The genetic algorithm (60 squads, 40 generations, roulette selection, single-point crossover, mutation) searches for the set with the best summed fit inside the budget, never the same player twice. Integer programming gives the true optimum for comparison.</div>')
