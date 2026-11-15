"""Defending, pressing and aerial numbers the main lake does not carry, for the Matrix and the style reads.

Sources (Kaggle, see fetch_data.sh):
  data/emrey/Top5_League_Players_2017to2024_dataset.csv   FBref 2017-18 to 2024-25: tackles by third, tackles won, dribblers
                                                          tackled, interceptions, blocks, clearances, ball recoveries, aerial duels
  data/chuong/PERFECT_scouting_data_2026.csv              2025-26 (FBref basics + Sofascore): possession won in the final third,
                                                          ball recoveries, tackles won, interceptions, aerial and ground duels
Pressing: FBref dropped its pressures count, so the pressing numbers here are where a player wins the ball: tackles in the
middle and attacking thirds, ball recoveries, and (2025-26) possession won in the final third.
Writes lake/defence_ext.parquet keyed on (player_id, yr), matched to the feature lake and the live squads by name and club.

    .venv/bin/python src/build_defence.py
    python3 src/build_defence.py --data DIR --raw out.csv     # only flatten the sources (no lake needed)
"""
import argparse, os, re, unicodedata
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LG = {'ENG-Premier League': 'Premier League', 'ESP-La Liga': 'La Liga', 'GER-Bundesliga': 'Bundesliga', 'ITA-Serie A': 'Serie A', 'FRA-Ligue 1': 'Ligue 1'}


def num(s):
    return pd.to_numeric(s.astype(str).str.replace(',', '.', regex=False), errors='coerce')


def flatten(data):
    out = []
    p = f'{data}/emrey/Top5_League_Players_2017to2024_dataset.csv'
    if os.path.exists(p):
        e = pd.read_csv(p, sep=';', low_memory=False); e = e[e['season'].astype(str).str.fullmatch(r'\d{4}')]
        cols = {'Tackles_Att 3rd': 'tkl_att3', 'Tackles_Mid 3rd': 'tkl_mid3', 'Tackles_Def 3rd': 'tkl_def3', 'Tackles_TklW': 'tklw', 'Challenges_Tkl': 'chal_tkl', 'Challenges_Att': 'chal_att',
                'Int_': 'int', 'Blocks_Blocks': 'blocks', 'Clr_': 'clr', 'Performance_Recov': 'recov', 'Aerial Duels_Won': 'aer_w', 'Aerial Duels_Lost': 'aer_l'}
        d = pd.DataFrame({'src': 'fbref', 'league': e['league'].map(LG), 'yr': 2000 + e['season'].astype(int) // 100, 'team': e['team'], 'player': e['player'], 'minutes': num(e['Playing Time_Min'])})
        for c, n in cols.items(): d[n] = num(e[c]) if c in e else np.nan
        out.append(d)
    p = f'{data}/chuong/PERFECT_scouting_data_2026.csv'
    if os.path.exists(p):
        c = pd.read_csv(p, low_memory=False)
        g = lambda k: num(c[k]) if k in c else np.nan
        d = pd.DataFrame({'src': 'sofascore', 'league': c['league'].map(LG), 'yr': 2000 + num(c['season']).astype('Int64') // 100, 'team': c['team'], 'player': c['player'], 'minutes': g('Playing Time_Min'),
                          'tklw': g('tacklesWon'), 'int': g('interceptions'), 'blocks': g('outfielderBlocks'), 'recov': g('ballRecovery'), 'won_att3': g('possessionWonAttThird'),
                          'aer_w': g('aerialDuelsWon'), 'aer_l': g('aerialLost'), 'ground_w': g('groundDuelsWon'), 'duel_pct': g('totalDuelsWonPercentage')})
        out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def norm(s):
    return re.sub(r'[^a-z]', '', unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower())


def per90(D):
    """Per-90 numbers, rates and per-season percentiles within position (900+ minutes)."""
    n90 = D['minutes'] / 90
    for c in ('tkl_att3', 'tkl_mid3', 'tkl_def3', 'tklw', 'int', 'blocks', 'clr', 'recov', 'won_att3', 'aer_w', 'ground_w'):
        if c in D: D[c + '_p90'] = D[c] / n90
    D['press_hi_p90'] = D[['tkl_att3', 'tkl_mid3']].sum(axis=1, min_count=1) / n90 if 'tkl_att3' in D else np.nan
    D['aer_pct'] = 100 * D['aer_w'] / (D['aer_w'] + D['aer_l']).replace(0, np.nan)
    D['chal_pct'] = 100 * D['chal_tkl'] / D['chal_att'].replace(0, np.nan) if 'chal_tkl' in D else np.nan
    D.loc[(D['aer_w'] + D['aer_l']) < 10, 'aer_pct'] = np.nan; D.loc[D.get('chal_att', pd.Series(0, index=D.index)) < 8, 'chal_pct'] = np.nan
    ok = D['minutes'] >= 900
    for c in [c for c in D.columns if c.endswith('_p90')] + ['aer_pct', 'chal_pct']:
        D[c + '_pct'] = np.nan
        D.loc[ok, c + '_pct'] = D[ok].groupby(['yr', 'fine_pos'])[c].rank(pct=True) * 100
    return D


def attach(raw, F, SQ=None):
    """Match the flattened rows to player ids: the feature lake by (name, club, season), the live squads by (name, club)."""
    raw = raw.copy(); raw['k'] = raw['player'].map(norm); raw['kt'] = raw['team'].map(norm)
    lk = F[['player_id', 'player', 'squad', 'yr', 'fine_pos']].assign(k=lambda d: d['player'].map(norm), kt=lambda d: d['squad'].map(norm))
    a = raw.merge(lk[['player_id', 'yr', 'k', 'kt', 'fine_pos']], on=['k', 'kt', 'yr'], how='left')
    # players who moved mid-table or whose club name differs: fall back to name + season when that is unique
    miss = a['player_id'].isna(); u = lk.drop_duplicates(['k', 'yr'], keep=False)[['k', 'yr', 'player_id', 'fine_pos']]
    b = a[miss].drop(columns=['player_id', 'fine_pos']).merge(u, on=['k', 'yr'], how='left'); a = pd.concat([a[~miss], b], ignore_index=True)
    # seasons after the lake (2025-26): the live squads and shortlists by name, then a name that maps to one lake player
    ids = [lk[['player_id', 'player', 'fine_pos']].rename(columns={'player': 'name'})] + [d[['player_id', 'name', 'fine_pos']] for d in (SQ,) if d is not None and 'fine_pos' in d]
    idn = pd.concat(ids).assign(k=lambda d: d['name'].map(norm)).drop_duplicates(['k', 'player_id'])
    idn = idn[~idn.duplicated('k', keep=False)]
    m = a['player_id'].isna(); c = a[m].drop(columns=['player_id', 'fine_pos']).merge(idn[['k', 'player_id', 'fine_pos']], on='k', how='left'); a = pd.concat([a[~m], c], ignore_index=True)
    known = lk.drop_duplicates('player_id').set_index('player_id')['fine_pos']
    a['fine_pos'] = a['fine_pos'].fillna(a['player_id'].map(known))
    a = a.dropna(subset=['player_id', 'fine_pos']).astype({'player_id': int})
    a = a.sort_values('minutes', ascending=False).drop_duplicates(['player_id', 'yr', 'src'])
    return per90(a.drop(columns=['k', 'kt']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default=f'{ROOT}/data'); ap.add_argument('--raw'); ap.add_argument('--raw-csv'); x = ap.parse_args()
    raw = pd.read_csv(x.raw_csv) if x.raw_csv else flatten(x.data)
    if x.raw: raw.to_csv(x.raw, index=False); print(f'{len(raw):,} rows -> {x.raw}'); raise SystemExit
    F = pd.read_parquet(f'{ROOT}/lake/features', columns=['player_id', 'player', 'squad', 'yr', 'fine_pos'])
    lv = [pd.read_parquet(f'{ROOT}/lake/{t}') for t in ('live_squads.parquet', 'live_shortlists.parquet') if os.path.exists(f'{ROOT}/lake/{t}')]
    SQ = pd.concat([d[[c for c in ('player_id', 'name', 'fine_pos') if c in d]] for d in lv]) if lv else None
    D = attach(raw, F, SQ); D.to_parquet(f'{ROOT}/lake/defence_ext.parquet', index=False)
    print(f"{len(D):,} player-seasons matched of {len(raw):,} rows ({D['yr'].min()}-{D['yr'].max()}) -> lake/defence_ext.parquet")
