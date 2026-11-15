"""How each team plays: one row per club-season with possession, pressing, territory and how much it concedes.

Sources (both are Kaggle downloads, see fetch_data.sh):
  data/understat/understats/<league>/season.csv   mexwell/understat-database: per-match team xG, xGA, PPDA and deep completions, 2014-15 to 2024-25, all five leagues
  data/fotmob/<league>/<yyyy_yyyy>/teams/*.json   (or data/matches/...) FotMob season team tables (possession %, tackles, interceptions, ball wins high up, long balls, clearances); no Bundesliga
Writes lake/team_style.parquet, keyed on the feature lake's (squad, yr) so the app can join it to players. Small: commit it.

    .venv/bin/python src/build_team_style.py                      # reads data/, writes lake/team_style.parquet
    python3 src/build_team_style.py --data DIR --raw team_raw.csv  # only flatten the raw files (no lake needed)
"""
import argparse, glob, json, os, re, unicodedata
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
US_LEAGUE = {'EPL': 'Premier League', 'La_Liga': 'La Liga', 'Bundesliga': 'Bundesliga', 'Serie_A': 'Serie A', 'Ligue_1': 'Ligue 1'}
FM_LEAGUE = {'Premier_League': 'Premier League', 'LaLiga': 'La Liga', 'Bundesliga': 'Bundesliga', 'Serie_A': 'Serie A', 'Ligue_1': 'Ligue 1'}
# FotMob stat file -> (column, per match?)  'per match' tables already hold per-match values; season totals are divided by matches played
FM_STATS = {'possession_percentage_team': ('possession', False), 'total_tackle_team': ('tackles_pm', False), 'interception_team': ('interceptions_pm', False),
            'poss_won_att_3rd_team': ('wins_final_third_pm', False), 'accurate_long_balls_team': ('long_balls_pm', False), 'accurate_pass_team': ('passes_pm', False),
            'effective_clearance_team': ('clearances_pm', False), 'accurate_cross_team': ('crosses_pm', False), 'clean_sheet_team': ('clean_sheets', False),
            'expected_goals_conceded_team': ('fm_xga_pm', True), 'touches_in_opp_box_team': ('box_touches_pm', True)}


def understat(data):
    out = []
    for path in glob.glob(f'{data}/understat/**/season.csv', recursive=True):
        lg = US_LEAGUE.get(os.path.basename(os.path.dirname(path)))
        if not lg: continue
        d = pd.read_csv(path)
        d = d[pd.to_numeric(d['year'], errors='coerce').notna()].astype({'year': int})
        g = d.groupby(['title', 'year']).agg(matches=('xG', 'size'), xg=('xG', 'sum'), xga=('xGA', 'sum'), npxg=('npxG', 'sum'), npxga=('npxGA', 'sum'), deep=('deep', 'sum'),
                                             deep_allowed=('deep_allowed', 'sum'), scored=('scored', 'sum'), conceded=('missed', 'sum'), pts=('pts', 'sum'),
                                             ppda_att=('ppda.att', 'sum'), ppda_def=('ppda.def', 'sum'), oppda_att=('ppda_allowed.att', 'sum'), oppda_def=('ppda_allowed.def', 'sum')).reset_index()
        m = g['matches']
        out.append(pd.DataFrame({'league': lg, 'yr': g['year'], 'team_us': g['title'], 'matches': m, 'xg_pm': g['xg'] / m, 'xga_pm': g['xga'] / m, 'npxga_pm': g['npxga'] / m,
                                 'goals_pm': g['scored'] / m, 'conceded_pm': g['conceded'] / m, 'pts_pm': g['pts'] / m, 'deep_pm': g['deep'] / m, 'deep_allowed_pm': g['deep_allowed'] / m,
                                 'ppda': g['ppda_att'] / g['ppda_def'].replace(0, np.nan), 'oppda': g['oppda_att'] / g['oppda_def'].replace(0, np.nan)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def fotmob(data):
    rows = {}
    for path in glob.glob(f'{data}/*/*/teams/*.json'):
        stat = os.path.basename(path)[:-5]; season_dir = os.path.basename(os.path.dirname(os.path.dirname(path))); lg = FM_LEAGUE.get(os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path)))))
        if stat not in FM_STATS or not lg or not re.fullmatch(r'\d{4}_\d{4}', season_dir): continue
        col, total = FM_STATS[stat]; yr = int(season_dir[:4])
        try: lst = json.load(open(path, encoding='utf-8'))['TopLists'][0]['StatList']
        except Exception: continue
        for s in lst:
            r = rows.setdefault((lg, yr, s['ParticipantName']), {'league': lg, 'yr': yr, 'team_fm': s['ParticipantName']})
            v = s.get('StatValue'); n = s.get('MatchesPlayed') or np.nan
            r[col] = v / n if total and v is not None else v
    return pd.DataFrame(list(rows.values()))


def flatten(data):
    """Raw team-season table from both sources, still in each source's own team names."""
    fm_dir = next((d for d in (f'{data}/fotmob', f'{data}/matches') if os.path.isdir(d)), None)
    return understat(data), (fotmob(fm_dir) if fm_dir else pd.DataFrame())


# ---- matching source names to the feature lake's FBref squad names ----
ALIAS = {'manchester united': 'manchester utd', 'newcastle united': 'newcastle utd', 'wolverhampton wanderers': 'wolves', 'nottingham forest': "nott'ham forest",
         'sheffield united': 'sheffield utd', 'west bromwich albion': 'west brom', 'brighton & hove albion': 'brighton', 'brighton and hove albion': 'brighton',
         'tottenham hotspur': 'tottenham', 'west ham united': 'west ham', 'leeds united': 'leeds united', 'luton town': 'luton town', 'ipswich town': 'ipswich town',
         'afc bournemouth': 'bournemouth', 'atletico madrid': 'atletico madrid', 'athletic club': 'athletic club', 'real betis': 'betis', 'deportivo alaves': 'alaves',
         'celta vigo': 'celta vigo', 'rayo vallecano': 'rayo vallecano', 'sd huesca': 'huesca', 'cadiz': 'cadiz', 'real valladolid': 'valladolid', 'sporting gijon': 'sporting gijon',
         'bayer leverkusen': 'leverkusen', 'borussia m.gladbach': 'gladbach', "borussia monchengladbach": 'gladbach', 'borussia dortmund': 'dortmund', 'eintracht frankfurt': 'eint frankfurt',
         'rasenballsport leipzig': 'rb leipzig', 'fc cologne': 'koln', '1. fc koln': 'koln', 'fortuna duesseldorf': 'dusseldorf', 'fortuna dusseldorf': 'dusseldorf', 'hertha berlin': 'hertha bsc',
         'mainz 05': 'mainz 05', 'fsv mainz 05': 'mainz 05', 'schalke 04': 'schalke 04', 'arminia bielefeld': 'arminia', 'greuther fuerth': 'greuther furth', 'paderborn': 'paderborn 07',
         'vfb stuttgart': 'stuttgart', 'werder bremen': 'werder bremen', 'hoffenheim': 'hoffenheim', 'augsburg': 'augsburg', 'freiburg': 'freiburg', 'wolfsburg': 'wolfsburg', 'bochum': 'bochum',
         'darmstadt': 'darmstadt 98', 'heidenheim': 'heidenheim', 'holstein kiel': 'holstein kiel', 'st. pauli': 'st. pauli', 'union berlin': 'union berlin', 'hamburger sv': 'hamburger sv',
         'ac milan': 'milan', 'inter milan': 'inter', 'internazionale': 'inter', 'parma calcio 1913': 'parma', 'spal 2013': 'spal', 'hellas verona': 'hellas verona', 'verona': 'hellas verona',
         'paris saint germain': 'paris s-g', 'paris saint-germain': 'paris s-g', 'saint-etienne': 'saint-etienne', 'st etienne': 'saint-etienne', 'olympique marseille': 'marseille',
         'olympique lyonnais': 'lyon', 'stade rennais': 'rennes', 'stade brestois': 'brest', 'clermont foot': 'clermont foot', 'nimes': 'nimes', 'as monaco': 'monaco', 'ogc nice': 'nice',
         'rc lens': 'lens', 'losc lille': 'lille', 'stade de reims': 'reims', 'rc strasbourg': 'strasbourg', 'fc lorient': 'lorient', 'fc nantes': 'nantes', 'le havre ac': 'le havre',
         'gfc ajaccio': 'ajaccio', 'ac ajaccio': 'ajaccio', 'sc bastia': 'bastia', 'amiens sc': 'amiens', 'dijon fco': 'dijon', 'toulouse fc': 'toulouse', 'angers sco': 'angers', 'troyes': 'troyes',
         'montpellier hsc': 'montpellier', 'girondins bordeaux': 'bordeaux', 'bordeaux': 'bordeaux', 'metz': 'metz', 'auxerre': 'auxerre', 'guingamp': 'guingamp', 'caen': 'caen'}


def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower().strip()
    return ALIAS.get(s, s)


def key(s):
    """Loose key: drop club prefixes/suffixes and punctuation."""
    s = re.sub(r"\b(fc|afc|cf|sc|ac|as|ss|ssc|us|cd|ud|rc|sd|ogc|losc|vfl|vfb|tsg|sv|1\.|club|calcio|de|hsc|1913|1907|1899|1846|04|05)\b", ' ', norm(s))
    return re.sub(r'[^a-z]', '', s)


def match_names(src, lake_names):
    """src: list of source names in one league-season; lake_names: FBref squads in that league-season. Returns {src: lake}."""
    by_norm = {norm(n): n for n in lake_names}; by_key = {key(n): n for n in lake_names}; out = {}
    for s in src:
        n, k = norm(s), key(s)
        hit = by_norm.get(n) or by_key.get(k) or next((v for kk, v in by_key.items() if kk and k and (kk in k or k in kk)), None)
        if hit: out[s] = hit
    return out


def attach(us, fm, F):
    """Join both sources onto the lake's (comp, yr, squad) list and add league-season z-scores."""
    base = F[['comp', 'yr', 'squad', 'club_id']].drop_duplicates(['comp', 'yr', 'squad']).rename(columns={'comp': 'league'})
    parts = []
    for (lg, yr), g in base.groupby(['league', 'yr']):
        names = g['squad'].tolist(); row = g.copy()
        for df, col in ((us, 'team_us'), (fm, 'team_fm')):
            if df is None or not len(df): continue
            s = df[(df['league'] == lg) & (df['yr'] == yr)]; mp = match_names(s[col].tolist(), names)
            s = s.assign(squad=s[col].map(mp)).dropna(subset=['squad']).drop(columns=['league', 'yr'])
            row = row.merge(s.drop_duplicates('squad'), on='squad', how='left')
        parts.append(row)
    T = pd.concat(parts, ignore_index=True)
    # FotMob has no Bundesliga: estimate possession there from Understat's territory and pressing numbers (fit on the other four leagues)
    if 'possession' in T:
        cols = ['deep_pm', 'deep_allowed_pm', 'ppda', 'oppda', 'xg_pm', 'xga_pm']; ok = T[cols].notna().all(axis=1)
        fit = ok & T['possession'].notna(); A = np.c_[np.ones(fit.sum()), np.log(T.loc[fit, cols].clip(lower=0.05))]
        b = np.linalg.lstsq(A, T.loc[fit, 'possession'], rcond=None)[0]; miss = ok & T['possession'].isna()
        T['possession_estimated'] = miss
        T.loc[miss, 'possession'] = (np.c_[np.ones(miss.sum()), np.log(T.loc[miss, cols].clip(lower=0.05))] @ b).clip(30, 70)
        r = T.loc[fit, 'possession'] - A @ b; print(f'possession estimate for leagues without FotMob: residual sd {r.std():.1f} points')
    # z-scores inside each league-season, so "dominant" means dominant for that league that year
    for c in ('possession', 'ppda', 'deep_pm', 'deep_allowed_pm', 'xga_pm', 'xg_pm', 'tackles_pm', 'interceptions_pm', 'wins_final_third_pm', 'long_balls_pm', 'pts_pm'):
        if c in T: T['z_' + c] = T.groupby(['league', 'yr'])[c].transform(lambda x: (x - x.mean()) / x.std(ddof=0) if x.notna().sum() > 3 else np.nan)
    T['rank_xga'] = T.groupby(['league', 'yr'])['xga_pm'].rank(method='min') if 'xga_pm' in T else np.nan
    T['n_teams'] = T.groupby(['league', 'yr'])['squad'].transform('size')
    return T


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default=f'{ROOT}/data'); ap.add_argument('--raw', help='write the flattened source tables to this CSV prefix and stop')
    ap.add_argument('--us-csv'); ap.add_argument('--fm-csv'); a = ap.parse_args()
    if a.us_csv: us, fm = pd.read_csv(a.us_csv), (pd.read_csv(a.fm_csv) if a.fm_csv else None)
    else: us, fm = flatten(a.data)
    if a.raw:
        us.to_csv(a.raw + '_understat.csv', index=False); fm.to_csv(a.raw + '_fotmob.csv', index=False); print(f'{len(us)} Understat and {len(fm)} FotMob team-seasons -> {a.raw}_*.csv'); raise SystemExit
    F = pd.read_parquet(f'{ROOT}/lake/features', columns=['comp', 'yr', 'squad', 'club_id'])
    T = attach(us, fm, F); T.to_parquet(f'{ROOT}/lake/team_style.parquet', index=False)
    print(f"{len(T):,} club-seasons; Understat matched {T['xga_pm'].notna().mean():.0%}, FotMob possession {T['possession'].notna().mean() if 'possession' in T else 0:.0%} -> lake/team_style.parquet")
