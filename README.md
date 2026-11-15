# Shadow Squad

Player replacement scouting for Europe's top five leagues: given a club and a departing starter (or a position), rank the players who could replace him, and check the list against what clubs really did.

End-semester project, Analysis and Systems of Big Data, IIITDM Kancheepuram.

## Try the dashboard (no data download needed)

The small tables the app reads are in the repository.

```
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app/shadow_squad.py
```

Pick a season ("Now" for the live shortlists, or a past season to see what the list would have said), then a league, then a club, and click a player on the pitch.

- **Clubs covered.** Past seasons list every club in the five leagues (about 96 to 98 a season); the top clubs are listed first. "Now" lists the 32 top clubs that have precomputed live shortlists (`lake/live_squads.parquet`). The app calls no external football API: everything is read from the Parquet tables in `lake/`.
- **Pitch.** The starters are laid out in a formation inferred from their positions and minutes (`src/formation.py`: 4-3-3, 4-2-3-1, 4-4-2, 4-1-4-1, 3-5-2 or 3-4-3), one slot per player; the rest sit on a bench strip. The formation can be overridden above the pitch.
- **Matrix.** The departing starter against the first 3 to 10 names on his shadow list, department by department (ordered by what matters for the position), each cell a percentile among players in that position with the per-90 number. "Peak" swaps in each player's best season of the last three.
- **Hindsight.** For past seasons the Squad cards and the Shadow list show where the players on the list are now and their latest Transfermarkt value (data up to June 2026). To refresh: re-download `data/transfermarkt`, run `.venv/bin/python src/build_player_now.py`, and commit `lake/player_now.parquet`.
- **Why this player?** Compare view, under the radar. "Ask why" answers questions about the pick with Claude, grounded in a dossier of that candidate's numbers (SHAP values, pool percentiles, per-90 z-scores, rank interval) and the method notes in `app/knowledge/`, retrieved with TF-IDF (`src/scout_rag.py`). "Scout's report" puts the same case in football terms (role, what he keeps, where he is a step down, fit with the club's style, where the top 3 are now), with the raw numbers in an expander.

### Secrets (optional)

Free-form answers need an Anthropic API key. Without it the app still runs and shows a built-in explanation built from the same numbers.

```
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then fill in ANTHROPIC_API_KEY; do not commit this file
```

On Streamlit Community Cloud, paste the same lines into App settings > Secrets instead. `CLAUDE_MODEL` is optional (default `claude-opus-5-5`). The explainer turns on server-side refusal fallbacks, so a question a safety filter declines is retried on another model.

## Reproduce everything

Needs Python 3.12, Java 17 or newer (Spark 4), and a Kaggle API token.

```
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/playwright install chromium      # only for the dashboard screenshots
./fetch_data.sh                            # seven Kaggle datasets into data/ (about 1 GB)
.venv/bin/python src/run_baseline.py       # runs midsem_frozen.ipynb unchanged; writes outputs/baseline_metrics.json and lake/midsem/
./run_nb.sh endsem_01_lake                 # Parquet lake, entity resolution, walk-forward style model
./run_nb.sh endsem_02_labels               # transfer labels, survival rows, replacement and signing events
./run_nb.sh endsem_03_mining               # Spark mining: LSH, PrefixSpan, bisecting k-means, PageRank, ALS, SVM/MLP, streaming
./run_nb.sh endsem_03a_postmid             # closed / maximal / longest frequent itemsets, distributed DBSCAN, pattern-match score
./run_nb.sh endsem_04_engine               # style translation check, candidate scores, learning-to-rank
./run_nb.sh endsem_04b_beyond              # hedonic pricing, causal forest, red-team, genetic algorithm vs ILP
./run_nb.sh endsem_05_eval                 # baselines, fee tiers, ablation, survival, conformal
./run_nb.sh endsem_05b_live                # live engine, summer-2026 test, drift check, current shortlists
./run_nb.sh endsem_06_explain              # SHAP, case studies, dashboard screenshots
.venv/bin/python src/build_report.py       # report/endsem.pdf (and the combined PDF if the mid-sem report PDF is in the folder)
```

The whole run takes about 40 minutes on a laptop. Reruns give identical numbers.

## Layout

| Path | What it is |
|---|---|
| `endsem_*.py` / `.ipynb` | each notebook and the script it is generated from (jupytext) |
| `src/` | shared code: engine, live engine, itemset algorithms, distributed DBSCAN, genetic algorithm, report builder |
| `app/shadow_squad.py` | the Streamlit dashboard |
| `app/knowledge/` | method notes the "why this player" explainer retrieves from |
| `.streamlit/` | dark pitch theme, and an example secrets file |
| `lake/` | only the tables the dashboard needs; the rest is rebuilt by the notebooks |
| `outputs/metrics.json` | every number in the report, written by the notebook that computed it |
| `outputs/figs/`, `outputs/screens/` | figures and dashboard screenshots used in the report |
| `midsem_frozen.ipynb` | the mid-semester notebook, unchanged |

Every model that ranks a season is fitted on earlier seasons only. `data/`, most of `lake/`, the lecture slides and the report PDFs are not in the repository.
