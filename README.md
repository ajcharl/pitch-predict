# MatchIQ ⚽

Football match outcome prediction — home win, draw or away win — for Europe's big five
leagues, from a Random Forest trained on **43,708 real matches** played between 2000 and 2025.

**Premier League · La Liga · Bundesliga · Serie A · Ligue 1**

It opens on **this week's real fixtures**, with a call on every one: probabilities for all
three outcomes, both teams' current form, and the factors that drove it. Click any match for
the full breakdown.

---

## What it does

| Page | What you get |
|---|---|
| **Fixtures** *(landing page)* | This week's big-five matches, each with the model's call, both sides' last-5 form, Elo, and a trust score. Click a row for the full breakdown |
| **Match detail** | Probability bars, the verdict, both teams' form cards, a "why this prediction" feature breakdown, and head-to-head |
| **Custom matchup** | Predict any two teams that aren't scheduled — including across leagues (Arsenal v Bayern) |
| **Track record** | Browse past matchdays and see the model's out-of-sample call next to what actually happened |
| **Leagues** | All five leagues and their current squads, ranked by Elo, linking through to team profiles |
| **Head to Head** | Full meeting record between two clubs, with charts |
| **Team profile** | Season stats, points-per-game trend, and a plain-English read on the team's strengths |
| **Model Info** | Accuracy, per-league breakdown, confusion matrix, feature importance, and how it all works |

---

## How the model works

### The one rule that matters: never look at the future

The raw data says *"Arsenal 2–1 Chelsea, 1 April 2024"*. That's the **answer**, not a clue.
To predict a match you have to describe the two teams **as they were the moment before
kick-off**.

Getting this wrong is called **data leakage**, and it is the single most common way a
football model ends up lying to you. If your "average goals in the last 5 games" quietly
includes the match you're predicting, the model partly *sees the result*, reports a
spectacular accuracy, and then falls apart on real fixtures.

MatchIQ makes leakage structurally impossible. `model/features.py` walks every match in
strict date order and uses a read-then-write loop:

```python
for each match (oldest -> newest):
    features = state.read_features(home, away)   # describe teams using ONLY the past
    rows.append(features)
    state.update(match)                          # NOW fold in this match's result
```

Because `read` always happens before `update`, a match can never influence its own
features. The same `TeamState` object then builds the feature vector at prediction
time — so the model is served by exactly the code that trained it.

### The features

33 numbers per match, in five groups:

**Elo rating (the heavy lifter).** Elo is a chess-style rating: every club has a number
(~1500 is average), it rises when you win and falls when you lose, and beating a strong
opponent moves it more than beating a weak one. It compresses years of results into a
single value. The **difference** between the two ratings matters more than either alone —
1900 vs 1500 is a mismatch, 1900 vs 1880 is a coin flip. Because ClubElo rates every
European club on one scale, this also works across leagues — Arsenal v Bayern is a fair
comparison even though they never meet domestically.

**Form, last 5 matches.** Win rate, points per game, goals scored, goals conceded, clean
sheets — for both sides. Five games is the football standard: long enough to smooth out one
fluky result, short enough to reflect current shape rather than the whole season.

**Venue-specific form.** Home advantage is the strongest single effect in football (~45% of
all matches are home wins) but it isn't equal for every club. Some are fortresses at home
and timid away, which a blended form number hides. So the home team's last 5 *home* games
and the away team's last 5 *away* games are tracked separately.

**Head to head.** Win rate, draw rate and average goals in recent meetings — plus a count of
how many meetings we actually have, so the model can learn to discount a fixture the two
sides have barely played.

**Season context.** League position, goal difference, points per game, and how far into the
season the match is (early-season form is noisier than late-season form).

Plus explicit **difference** features (`form_diff`, `elo_diff`, `attack_vs_defence`). A
Random Forest splits on one feature at a time, so handing it the subtraction directly turns
a comparison that would need many nested splits into one cheap one.

### Why a Random Forest

400 decision trees, each trained on a random slice of the data and features, voting
together. It handles mixed scales with no normalisation, finds interactions on its own
(*"home form matters more when the Elo gap is small"*), resists overfitting once you set a
sensible minimum leaf size — and it gives both usable probabilities and a readable
importance ranking, so the app can explain *why*. Deep learning would be overkill here:
with 33 tabular features and 40k rows, a forest with good features wins.

**What it actually leans on:**

| Feature | Importance |
|---|---|
| `elo_diff` | 18.5% |
| `home_elo` | 8.2% |
| `goal_diff_diff` | 7.9% |
| `away_elo` | 7.3% |
| `position_diff` | 4.7% |

---

## Accuracy, and what it means

Trained on seasons **2000–2022** (40,205 matches), tested on seasons **2023–2025**
(3,503 matches). The split is **by time, not random** — because we're predicting the
future, the test set has to sit strictly after the training set. A random split would let
the model train on April and be tested on January of the same season, and it would score far
better in testing than in life.

```
Model accuracy:              52.2%
Baseline (always home win):  42.6%
Beats baseline by:           +9.6 percentage points
Log loss:                    0.982  (a blind 1/3 guess scores 1.099)
```

**Per league** (test period):

| League | Model | Baseline | Edge |
|---|---|---|---|
| La Liga | 54.1% | 44.2% | +9.9 |
| Premier League | 53.8% | 43.4% | +10.4 |
| Serie A | 52.1% | 40.8% | +11.3 |
| Ligue 1 | 50.4% | 43.0% | +7.4 |
| Bundesliga | 49.8% | 41.2% | +8.6 |

### Is 52% good? Yes.

A blind three-way guess is 33%. Always backing the home team gets 42.6%. Commercial models
with far richer data (injuries, lineups, xG, transfer news) top out around 52–56%. Football
is **irreducibly unpredictable** — a deflection, a red card, or a keeper's one good day
decides matches no model can foresee. The gap between 42.6% and 52.2% is the part that
*is* predictable, and that's what MatchIQ captures.

An earlier build also covered 11 second-tier and smaller leagues and scored 50.4% overall.
Narrowing to the big five *raised* accuracy to 52.2% and nearly doubled the edge over the
baseline: those leagues have more squad turnover, less reliable Elo, and more randomness,
and they were dragging the average down.

### One honest caveat: the model rarely predicts draws

| Outcome | Recall | Precision |
|---|---|---|
| Home Win | 85.1% | 51.4% |
| Draw | 0.0% | — |
| Away Win | 50.4% | 54.2% |

A draw is almost never the single *most likely* result — it typically sits at 25–28% while
one side leads. So the model's headline pick is nearly always a win, even though the draw
probability it reports is meaningful and well-calibrated. **Read the three probabilities,
not just the verdict.** This is a genuine property of three-way football prediction, not a
bug; forcing more draw predictions with class weights would raise draw recall while making
the probabilities less honest and overall accuracy worse.

### The trust score is real, not decoration

Every prediction carries a 1–10 trust badge, rescaled from the top probability. Measured on
the 3,503-match backtest, accuracy rises monotonically with it:

```
 1/10 -> 34.5%     4/10 -> 46.3%     7/10 -> 59.4%     10/10 -> 75.8%
```

A low-trust row is a genuinely hard call, not a broken one.

---

## Predicting upcoming fixtures

The Kaggle dataset ends on 2025-06-01, so out of the box the model describes every club as it
looked at the end of the 2024-25 season. `model/refresh.py` closes that gap from three free,
key-less feeds:

```bash
.venv/Scripts/python.exe model/refresh.py
```

| | Source | What it gives |
|---|---|---|
| Results | `football-data.co.uk/mmz4281/…` | Matches played since the cutoff, in the same schema as the Kaggle data |
| Elo | `api.clubelo.com` | Current strength ratings |
| Fixtures | `football-data.co.uk/fixtures.csv` | The coming week's matches |

Then restart the API. The fixtures page picks them up automatically.

Three deliberate design choices:

**It refreshes state, not the model.** The forest still trains on 2000–2022 and is validated on
2023–2025, so the accuracy figures above stay meaningful. New results change what we know about
the teams, not what the model has learned about football.

**It is idempotent.** `state.pkl` (from training) is treated as read-only and refresh always
rebuilds from it into `state_live.pkl`. Running it twice does not replay the same matches twice
and double-count the standings. The API prefers `state_live.pkl` when it exists.

**It refuses to guess at Elo.** ClubElo's daily snapshot carries ~594 clubs worldwide, so some
sides simply are not in it. Names are matched exactly within the same country, plus a
hand-checked alias table (`Ath Madrid` → `Atletico`, `Sp Lisbon` → `Sporting`, `PSV Eindhoven` →
`PSV`). Fuzzy matching alone would cheerfully map `Ath Madrid` onto `Real Madrid` and silently
corrupt the model's most important feature. Anything unmatched keeps its last known rating, and
those rows are badged **stale elo** in the UI.

Incoming team names are also re-spelled to match the state's existing names — the new season files
write `Nott'm Forest` where the training data settled on `Nottm Forest`, and replaying them blind
would create a second, empty Forest with no history.

### A caveat about early season

At the time of writing it is matchday 2. Form and season-context features are nearly empty, so
the model leans almost entirely on Elo, and its confidence reflects that — most calls land at
2–5/10 rather than the 8–10/10 you see mid-season. That is the model being honest, not broken.
Accuracy early in a campaign is genuinely lower than the 52% headline.

---

## Running it locally

**Requirements:** Python 3.10+ and Node 18+.

### 1. Backend

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -r backend/requirements.txt
```

(on macOS/Linux: `.venv/bin/pip install -r backend/requirements.txt`)

Download the dataset into `model/data/`:

```bash
curl -L -o model/data/raw.zip "https://www.kaggle.com/api/v1/datasets/download/adamgbor/club-football-match-data-2000-2025" && unzip -o model/data/raw.zip -d model/data && rm model/data/raw.zip
```

Train the model (~50 seconds; writes `model.pkl`, `state.pkl`, `metrics.json`, `backtest.csv`):

```bash
.venv/Scripts/python.exe model/train.py
```

Start the API:

```bash
.venv/Scripts/python.exe -m uvicorn backend.main:app --reload --port 8000
```

Interactive API docs: <http://localhost:8000/docs>

### 2. Frontend

```bash
npm install --prefix frontend && npm run dev --prefix frontend
```

Open <http://localhost:5173>. Vite proxies `/api` to the backend, so there's no CORS setup
in development.

> **Two gotchas.** Use `localhost`, not `127.0.0.1`, for the UI — Vite binds to IPv6, so
> `http://127.0.0.1:5173` won't connect. And keep the ports at 8000/5173 unless you change
> them together: the Vite proxy targets `127.0.0.1:8000` and the backend's CORS allowlist
> expects the UI on 5173.

---

## API

| Endpoint | Returns |
|---|---|
| `GET /api/health` | Service and model status |
| `GET /api/freshness` | How current the model's view of the world is |
| `GET /api/leagues` | The five leagues |
| `GET /api/teams?league=E0` | Current squad, strongest first |
| `GET /api/predict?home=Arsenal&away=Chelsea` | Full prediction with probabilities, form and features |
| `GET /api/fixtures?league=E0` | Upcoming fixtures, each with a prediction |
| `GET /api/fixtures/dates` | Dates that have upcoming fixtures |
| `GET /api/team/{name}/form` | Recent form, season stats, match history |
| `GET /api/head-to-head?team1=X&team2=Y` | H2H record and meetings |
| `GET /api/matches?date=2025-05-25` | Past matchday with predictions vs actual results |
| `GET /api/model/accuracy` | Full model metrics |

League codes: `E0` Premier League, `SP1` La Liga, `D1` Bundesliga, `I1` Serie A, `F1` Ligue 1.

Unknown team names return a 404 **with spelling suggestions** — `Barcelonaa` suggests
`Barcelona`. Names are matched loosely (case, accents and punctuation are ignored), so
`man utd` and `Real Madrid ` both resolve.

---

## Project structure

```
pitch-predict/
├── model/
│   ├── data/           # Matches.csv, EloRatings.csv, backtest.csv, live/
│   ├── explore.py      # dataset exploration + leakage checks
│   ├── features.py     # feature engineering (the important file)
│   ├── train.py        # training and evaluation
│   ├── predict.py      # prediction + team/H2H/fixture lookups
│   ├── refresh.py      # pull recent results, live Elo and upcoming fixtures
│   ├── model.pkl       # trained forest
│   ├── state.pkl       # cached team state from training (read-only)
│   ├── state_live.pkl  # refreshed state, written by refresh.py
│   └── metrics.json    # evaluation results
├── backend/
│   ├── main.py         # FastAPI app, CORS, error handling
│   └── routes/         # predictions.py, stats.py
└── frontend/
    └── src/
        ├── pages/      # FixturesPage (/), MatchPage, CustomMatchPage, ...
        ├── components/ # PredictionDetail, TeamCard, Sidebar, ui primitives
        ├── theme.js    # chart palette, mirrors the CSS tokens
        └── styles.css  # the design system
```

Generated artefacts (`.venv`, `node_modules`, the CSVs, `*.pkl`, `metrics.json`) are
gitignored — they're all reproducible from the steps above.

---

## Tech stack

**Model** — Python 3.12, pandas, scikit-learn (`RandomForestClassifier`), joblib
**Backend** — FastAPI + uvicorn. Model and team state load once at startup, not per request
**Frontend** — React 18, Vite, Recharts, React Router. Plain JavaScript, no TypeScript
**Live data** — football-data.co.uk (results, fixtures) and ClubElo (ratings), no API keys
**Storage** — none. Everything runs from CSVs and the pickled model

---

## Notes on the data

**Source:** [Club Football Match Data 2000–2025](https://www.kaggle.com/datasets/adamgbor/club-football-match-data-2000-2025)
by Adam Gábor, built on [Football-Data.co.uk](https://www.football-data.co.uk/) results and
[ClubElo](http://clubelo.com/) ratings.

**Elo columns are pre-match.** Verified in `model/explore.py` by matching every
`Matches.HomeElo` against ClubElo's dated snapshot for that day — 100% agreement to within
0.01. Had they been post-match values, they would have leaked the result.

**Leagues covered.** Europe's big five: Premier League (`E0`), La Liga (`SP1`), Bundesliga
(`D1`), Serie A (`I1`) and Ligue 1 (`F1`). All five have 100% Elo coverage in the dataset.
To add another, put its football-data.co.uk division code in `TRAIN_DIVISIONS` and
`DIVISION_NAMES` in `model/features.py`, add its country to `DIVISION_COUNTRY` in
`model/refresh.py`, then retrain.

**Cross-league matchups still work.** The five leagues never meet in this data, but Elo is
calibrated across all of them, so the Custom matchup page will happily rate Arsenal v Bayern.
Those predictions lean on Elo plus each side's domestic form, and honestly report "no recent
meetings" for head-to-head.

**Clubs with no top-flight history cannot be predicted.** Covering only the top divisions means
a side promoted from a second tier arrives with no record at all — Elversberg, newly up to the
Bundesliga, is skipped rather than guessed at, and the fixtures table says so in the row. Clubs
that have previously spent time in their top flight (Leeds, Hull, Coventry) keep that history
and are predicted normally.

**Team names are canonicalised on load.** The source spells a few clubs two ways —
`Nottm Forest` for the bulk of its matches and `Nott'm Forest` for the most recent ones. Left
alone, each spelling becomes a separate team whose form, Elo and H2H all fragment, so the club
you actually want to predict for looks newly promoted with no history.
`canonicalise_team_names()` merges variants that share a normalised key, keeping the most
common spelling.

---

## Screenshots

Add PNGs to `docs/screenshots/` to populate this section:

| | |
|---|---|
| ![Fixtures](docs/screenshots/fixtures.png) | ![Match detail](docs/screenshots/match.png) |
| ![Model info](docs/screenshots/model.png) | ![Team profile](docs/screenshots/team.png) |

---

## Disclaimer

MatchIQ estimates probabilities from historical form. Football is genuinely unpredictable and
these are not certainties. This is a data-science project, not betting advice.
