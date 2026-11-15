"""Refresh 'where is he now': each player's current club and latest market value, from the Transfermarkt dump.
Writes lake/player_now.parquet (small; commit it so the hosted app has it). The app falls back to the bundled tables without it.

    ./fetch_data.sh                          # or re-download only data/transfermarkt for fresher values
    .venv/bin/python src/build_player_now.py
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT

TM = f'{ROOT}/data/transfermarkt'
pl = pd.read_csv(f'{TM}/players.csv', usecols=lambda c: c in {'player_id', 'name', 'current_club_id', 'current_club_name', 'market_value_in_eur', 'highest_market_value_in_eur', 'last_season', 'contract_expiration_date'})
if 'current_club_name' not in pl or pl['current_club_name'].isna().all():      # older dumps carry only the id
    pl = pl.drop(columns='current_club_name', errors='ignore').merge(pd.read_csv(f'{TM}/clubs.csv', usecols=['club_id', 'name']).rename(columns={'club_id': 'current_club_id', 'name': 'current_club_name'}), on='current_club_id', how='left')
va = pd.read_csv(f'{TM}/player_valuations.csv', usecols=['player_id', 'date']); as_of = pd.to_datetime(va['date']).max()
known = set(pd.read_parquet(f'{ROOT}/lake/features', columns=['player_id'])['player_id'])
for t in ('live_squads.parquet', 'live_shortlists.parquet'):
    if os.path.exists(f'{ROOT}/lake/{t}'): known |= set(pd.read_parquet(f'{ROOT}/lake/{t}', columns=['player_id'])['player_id'])
out = pl[pl['player_id'].isin(known)].assign(as_of=as_of)
out.to_parquet(f'{ROOT}/lake/player_now.parquet', index=False)
print(f'{len(out):,} players, values as of {as_of:%Y-%m-%d} -> lake/player_now.parquet')
