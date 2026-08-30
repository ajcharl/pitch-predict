"""
Step 5 - Turning the trained model into an answer about a real fixture.

Training asked "given these 33 numbers, what happened?" thousands of times.
Prediction runs the same machinery once, forwards: look up how both teams are
playing *right now*, build the identical 33 numbers in the identical order,
and ask the forest to vote.

The important guarantee here is TRAIN/SERVE CONSISTENCY. We do not
re-implement feature building for prediction -- we reuse the exact
`TeamState.read_features` that produced the training rows, restored from the
cached state that train.py saved after walking all 130k historical matches.
If the two ever drifted apart, the model would be fed subtly different numbers
than it learned on and would quietly get worse; sharing the code makes that
impossible.
"""

import json
import sys
from difflib import get_close_matches
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import (DIVISION_NAMES, FEATURE_COLUMNS, FEATURE_DESCRIPTIONS,
                      MODEL_DIR, TRAIN_DIVISIONS, normalize_team_name)

MODEL_PATH = MODEL_DIR / "model.pkl"
STATE_PATH = MODEL_DIR / "state.pkl"
# Written by model/refresh.py. When present it holds the same team state
# advanced with results published since the dataset cutoff, plus current Elo,
# so predictions describe teams as they are now rather than in May 2025.
LIVE_STATE_PATH = MODEL_DIR / "state_live.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
BACKTEST_PATH = MODEL_DIR / "data" / "backtest.csv"
FIXTURES_PATH = MODEL_DIR / "data" / "live" / "fixtures.csv"

OUTCOME_NAMES = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
OUTCOME_KEYS = {"H": "home_win", "D": "draw", "A": "away_win"}

# Every league the app serves. All are featured -- there is no long tail to
# hide behind a "more" list any more.
FEATURED_LEAGUES = list(TRAIN_DIVISIONS)

# A team needs some history before its form features mean anything.
MIN_MATCHES_FOR_PREDICTION = 5


class TeamNotFound(Exception):
    """Raised when a team name cannot be matched, with near-miss suggestions."""

    def __init__(self, name, suggestions):
        self.name = name
        self.suggestions = suggestions
        msg = "Unknown team: '{}'.".format(name)
        if suggestions:
            msg += " Did you mean: {}?".format(", ".join(suggestions))
        super().__init__(msg)


class InsufficientHistory(Exception):
    """Raised when a team has too few matches on record to predict from."""


# Shared with features.py so lookup and training agree on team identity.
_norm = normalize_team_name


class Predictor:
    """Loads the model + cached team state once, then answers many questions.

    The FastAPI app builds exactly one of these at startup (see backend/main.py)
    so that no request pays the loading cost.
    """

    def __init__(self, model_path=MODEL_PATH, state_path=STATE_PATH, metrics_path=METRICS_PATH):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        # Column ORDER is part of the model's contract -- take it from the
        # bundle rather than trusting the import to still match.
        self.feature_columns = bundle["feature_columns"]
        self.fill_values = bundle["fill_values"]
        self.classes = list(self.model.classes_)
        self.trained_at = bundle.get("trained_at")

        # Prefer the refreshed state when refresh.py has produced one.
        live = Path(LIVE_STATE_PATH)
        self.state_is_live = live.exists()
        self.state = joblib.load(live if self.state_is_live else state_path)
        self.state_refreshed_at = getattr(self.state, "refreshed_at", None)
        self.elo_refreshed_at = getattr(self.state, "elo_refreshed_at", None)
        self.elo_stale_teams = set(getattr(self.state, "elo_stale_teams", []) or [])

        self.metrics = {}
        if Path(metrics_path).exists():
            with open(metrics_path, encoding="utf-8") as fh:
                self.metrics = json.load(fh)

        # Lookup index: normalised name -> canonical name, built once.
        self._index = {}
        for team in self.state.matches_played:
            self._index.setdefault(_norm(team), team)

        self._backtest = None
        self._fixtures = None

    # ------------------------------------------------------------------
    # team lookup
    # ------------------------------------------------------------------
    def resolve_team(self, name):
        """Map user input to a canonical team name, tolerantly.

        Users type "man utd", "Real Madrid ", "Atletico". We normalise, then
        fall back to fuzzy matching so a near-miss produces a helpful error
        with suggestions instead of a bare 404.
        """
        if name is None or not str(name).strip():
            raise TeamNotFound(name, [])
        key = _norm(name)
        if key in self._index:
            return self._index[key]
        close = get_close_matches(key, list(self._index), n=5, cutoff=0.7)
        raise TeamNotFound(name, [self._index[c] for c in close])

    def current_season(self, league=None):
        """The most recent season on record, per league.

        Computed per league rather than globally because the big five do not
        kick off together -- when this was written the Bundesliga had not
        started its 2026-27 season while the other four had, so a single global
        "current season" emptied the Bundesliga list entirely.
        """
        seasons = [
            season for team, season in self.state.last_season.items()
            if season is not None
            and (league is None or self.state.last_division.get(team) == league)
        ]
        return max(seasons) if seasons else 0

    def teams_in_league(self, league=None, active_since=None):
        """Teams currently in a league, strongest first.

        Defaults to the current season, so a league lists the clubs actually
        in it rather than every side that has passed through since 2023 --
        otherwise the Premier League dropdown offers 27 teams.
        """
        if active_since is None:
            active_since = self.current_season(league)
        out = []
        for team, division in self.state.last_division.items():
            if league and division != league:
                continue
            if self.state.last_season.get(team, -1) < active_since:
                continue
            out.append(self._team_brief(team))
        out.sort(key=lambda t: (-(t["elo"] or 0), t["name"]))
        return out

    def _team_brief(self, team):
        div = self.state.last_division.get(team)
        elo = self.state.elo.get(team)
        return {
            "name": team,
            "league": div,
            "league_name": DIVISION_NAMES.get(div, div),
            "elo": round(float(elo), 1) if elo is not None else None,
            "matches_played": self.state.matches_played.get(team, 0),
        }

    def leagues(self):
        """The five leagues the model serves."""
        return [{"code": code, "name": DIVISION_NAMES.get(code, code),
                 "featured": True, "synthetic": False,
                 "teams": len(self.teams_in_league(code))}
                for code in TRAIN_DIVISIONS]

    # ------------------------------------------------------------------
    # the prediction itself
    # ------------------------------------------------------------------
    def predict(self, home, away, league=None):
        """Predict one fixture. Returns probabilities plus supporting stats."""
        home = self.resolve_team(home)
        away = self.resolve_team(away)
        if home == away:
            raise ValueError("A team cannot play itself.")

        for team in (home, away):
            played = self.state.matches_played.get(team, 0)
            if played < MIN_MATCHES_FOR_PREDICTION:
                raise InsufficientHistory(
                    "Not enough history for '{}' ({} matches on record, need {}).".format(
                        team, played, MIN_MATCHES_FOR_PREDICTION))

        # Each team's season context comes from its OWN league table, so a
        # cross-league tie still gets each side's real domestic position.
        home_div = self.state.last_division.get(home)
        away_div = self.state.last_division.get(away)
        home_season = self.state.last_season.get(home)
        away_season = self.state.last_season.get(away)

        # Build the feature vector with the SAME function that built training
        # rows -- this is the train/serve consistency guarantee.
        raw = self.state.read_features(
            home, away, home_div, home_season,
            away_division=away_div, away_season=away_season,
        )

        # Impute exactly as training did, using the medians saved in the bundle.
        row = {}
        for col in self.feature_columns:
            val = raw.get(col, np.nan)
            row[col] = self.fill_values[col] if pd.isna(val) else float(val)
        X = pd.DataFrame([row], columns=self.feature_columns)

        proba = self.model.predict_proba(X)[0]
        probs = {OUTCOME_KEYS[c]: float(p) for c, p in zip(self.classes, proba)}
        best_idx = int(np.argmax(proba))
        best_class = self.classes[best_idx]
        confidence = float(proba[best_idx])

        cross_league = (home_div != away_div)

        return {
            "home_team": home,
            "away_team": away,
            "league": league or home_div,
            "league_name": ("{} v {}".format(DIVISION_NAMES.get(home_div, home_div),
                                             DIVISION_NAMES.get(away_div, away_div))
                            if cross_league else DIVISION_NAMES.get(home_div, home_div)),
            "cross_league": cross_league,
            "predictions": {
                "home_win": round(probs.get("home_win", 0.0), 4),
                "draw": round(probs.get("draw", 0.0), 4),
                "away_win": round(probs.get("away_win", 0.0), 4),
            },
            "predicted_outcome": OUTCOME_NAMES[best_class],
            "predicted_code": best_class,
            "confidence": round(confidence, 4),
            # A 1-10 "trust" score for the UI badge. Confidence for a 3-way
            # market runs from ~0.33 (a coin flip between three) to ~0.80, so we
            # rescale that band onto 1-10 rather than using the raw probability.
            "confidence_score": self._confidence_score(confidence),
            "features_used": {k: _clean(v) for k, v in raw.items()},
            "feature_contributions": self._top_contributions(raw),
            "home_stats": self.team_form(home),
            "away_stats": self.team_form(away),
            "head_to_head": self.head_to_head(home, away, limit=5),
            "model_accuracy": self.metrics.get("accuracy"),
        }

    # Confidence band used for the 1-10 trust badge. For a 3-way market the
    # top probability runs from ~0.33 (a coin flip between three) up to ~0.95,
    # but 95% of predictions fall below 0.76 -- so mapping the full theoretical
    # range would squash almost everything into the low scores. Rescaling
    # 0.34-0.70 instead spreads predictions across all ten steps.
    #
    # The badge is meaningful, not decorative: measured on the 10,712-match
    # backtest, accuracy rises monotonically with it --
    #   1/10 -> 34.5%   4/10 -> 46.3%   7/10 -> 59.4%   10/10 -> 75.8%
    CONFIDENCE_FLOOR = 0.34
    CONFIDENCE_CEIL = 0.70

    @classmethod
    def _confidence_score(cls, confidence):
        span = cls.CONFIDENCE_CEIL - cls.CONFIDENCE_FLOOR
        score = (confidence - cls.CONFIDENCE_FLOOR) / span * 9 + 1
        return int(round(min(10, max(1, score))))

    def _top_contributions(self, raw, top_n=6):
        """The features that mattered most, for the "why this prediction" panel.

        These are the FOREST'S GLOBAL importances (how much each feature helped
        across all training matches), paired with this fixture's actual values.
        That is an honest "here is what the model looks at and what it saw",
        not a per-match causal attribution -- which a Random Forest cannot give
        without a much heavier method like SHAP.
        """
        ranked = self.metrics.get("feature_importance") or []
        out = []
        for item in ranked[:top_n]:
            name = item["feature"]
            out.append({
                "feature": name,
                "label": FEATURE_DESCRIPTIONS.get(name, name),
                "importance": round(float(item["importance"]), 4),
                "value": _clean(raw.get(name)),
            })
        return out

    # ------------------------------------------------------------------
    # supporting views
    # ------------------------------------------------------------------
    def team_form(self, team):
        """Recent form and season standing for one team."""
        team = self.resolve_team(team)
        snap = self.state.team_snapshot(team)
        div = self.state.last_division.get(team)
        season = self.state.last_season.get(team)
        season_snap = self.state.season_snapshot(team, div, season)

        log = list(self.state.match_log.get(team, []))
        recent = []
        for m in reversed(log[-20:]):
            is_home = m["home_team"] == team
            gf = m["home_goals"] if is_home else m["away_goals"]
            ga = m["away_goals"] if is_home else m["home_goals"]
            recent.append({
                "date": m["date"],
                "opponent": m["away_team"] if is_home else m["home_team"],
                "venue": "H" if is_home else "A",
                "goals_for": gf, "goals_against": ga,
                "result": "W" if gf > ga else ("D" if gf == ga else "L"),
                "score": "{}-{}".format(m["home_goals"], m["away_goals"]),
            })

        # Season totals, derived from the match log for the current season.
        wins = draws = losses = scored = conceded = clean = 0
        for m in log:
            if m["season"] != season:
                continue
            is_home = m["home_team"] == team
            gf = m["home_goals"] if is_home else m["away_goals"]
            ga = m["away_goals"] if is_home else m["home_goals"]
            scored += gf
            conceded += ga
            clean += 1 if ga == 0 else 0
            if gf > ga:
                wins += 1
            elif gf == ga:
                draws += 1
            else:
                losses += 1

        return {
            "team": team,
            "league": div,
            "league_name": DIVISION_NAMES.get(div, div),
            "season": int(season) if season is not None else None,
            "last_5": snap["last_5"][::-1],   # newest first, as the UI reads it
            "goals_scored_avg": snap["goals_scored_avg"],
            "goals_conceded_avg": snap["goals_conceded_avg"],
            "points_per_game": snap["points_per_game"],
            "clean_sheets_last_5": snap["clean_sheets_last_5"],
            "elo": snap["elo"],
            "league_position": season_snap["league_position"],
            "matches_played": snap["matches_played"],
            "season_stats": {
                "played": wins + draws + losses,
                "wins": wins, "draws": draws, "losses": losses,
                "goals_scored": scored, "goals_conceded": conceded,
                "goal_difference": scored - conceded,
                "clean_sheets": clean,
                "points": season_snap["points"],
            },
            "recent_matches": recent,
        }

    def head_to_head(self, team1, team2, limit=20):
        """Head-to-head record and recent meetings between two teams."""
        team1 = self.resolve_team(team1)
        team2 = self.resolve_team(team2)
        key = (team1, team2) if team1 <= team2 else (team2, team1)
        log = list(self.state.h2h_log.get(key, []))

        t1_wins = t2_wins = draws = goals1 = goals2 = 0
        meetings = []
        for m in reversed(log[-limit:]):
            h, a = m["home_team"], m["away_team"]
            hg, ag = m["home_goals"], m["away_goals"]
            if hg == ag:
                winner = None
            else:
                winner = h if hg > ag else a
            meetings.append({
                "date": m["date"], "home_team": h, "away_team": a,
                "home_goals": hg, "away_goals": ag,
                "score": "{}-{}".format(hg, ag),
                "winner": winner,
                "league": m["division"],
                "season": m["season"],
            })
        for m in log:
            h, hg, ag = m["home_team"], m["home_goals"], m["away_goals"]
            g1 = hg if h == team1 else ag
            g2 = ag if h == team1 else hg
            goals1 += g1
            goals2 += g2
            if g1 > g2:
                t1_wins += 1
            elif g1 < g2:
                t2_wins += 1
            else:
                draws += 1

        played = len(log)
        return {
            "team1": team1, "team2": team2,
            "played": played,
            "team1_wins": t1_wins, "team2_wins": t2_wins, "draws": draws,
            "team1_goals": goals1, "team2_goals": goals2,
            "avg_goals": round((goals1 + goals2) / played, 2) if played else None,
            "meetings": meetings,
            # Cross-league ties genuinely have no shared domestic history.
            "note": None if played else "No recent meetings on record between these teams.",
        }

    # ------------------------------------------------------------------
    # browsable matchdays (backtest: predictions vs what actually happened)
    # ------------------------------------------------------------------
    def _backtest_df(self):
        """Lazily load the per-match backtest written by train.py."""
        if self._backtest is None:
            if Path(BACKTEST_PATH).exists():
                self._backtest = pd.read_csv(BACKTEST_PATH)
            else:
                self._backtest = pd.DataFrame(columns=[
                    "date", "division", "home_team", "away_team", "predicted",
                    "actual", "prob_H", "prob_D", "prob_A", "correct"])
        return self._backtest

    def available_dates(self, league=None, limit=400):
        """Dates that have matches, newest first, for the date picker."""
        df = self._backtest_df()
        if league:
            df = df[df.division == league]
        counts = df.groupby("date").size().sort_index(ascending=False)
        dates = [{"date": d, "matches": int(n)} for d, n in counts.head(limit).items()]
        return {
            "dates": dates,
            "latest": dates[0]["date"] if dates else None,
            "range": [str(df.date.min()), str(df.date.max())] if len(df) else None,
        }

    def matches_on(self, date=None, league=None):
        """Every backtested match on a date, with prediction and actual result."""
        df = self._backtest_df()
        if league:
            df = df[df.division == league]
        if not len(df):
            return {"date": date, "league": league, "matches": [], "summary": None}
        # Default to the most recent date we have, so the UI opens on real data.
        date = date or str(df.date.max())
        day = df[df.date == date]

        out = []
        for row in day.itertuples(index=False):
            probs = {"home_win": float(row.prob_H), "draw": float(row.prob_D),
                     "away_win": float(row.prob_A)}
            conf = max(probs.values())
            out.append({
                "date": row.date,
                "league": row.division,
                "league_name": DIVISION_NAMES.get(row.division, row.division),
                "home_team": row.home_team,
                "away_team": row.away_team,
                "predictions": {k: round(v, 4) for k, v in probs.items()},
                "predicted_outcome": OUTCOME_NAMES[row.predicted],
                "predicted_code": row.predicted,
                "actual_outcome": OUTCOME_NAMES[row.actual],
                "actual_code": row.actual,
                "correct": bool(row.correct),
                "confidence": round(conf, 4),
                "confidence_score": self._confidence_score(conf),
                "home_elo": _clean(row.home_elo),
                "away_elo": _clean(row.away_elo),
            })
        out.sort(key=lambda m: (m["league_name"], m["home_team"]))
        hits = sum(1 for m in out if m["correct"])
        return {
            "date": date,
            "league": league,
            "matches": out,
            "summary": {
                "played": len(out),
                "correct": hits,
                "accuracy": round(hits / len(out), 4) if out else None,
            },
        }

    # ------------------------------------------------------------------
    # upcoming fixtures
    # ------------------------------------------------------------------
    def _fixtures_df(self):
        """Lazily load the fixture list written by model/refresh.py."""
        if self._fixtures is None:
            if Path(FIXTURES_PATH).exists():
                df = pd.read_csv(FIXTURES_PATH, parse_dates=["MatchDate"])
                self._fixtures = df.sort_values(["MatchDate", "MatchTime"],
                                                kind="mergesort")
            else:
                self._fixtures = pd.DataFrame(columns=[
                    "Division", "MatchDate", "MatchTime", "HomeTeam", "AwayTeam"])
        return self._fixtures

    def fixture_dates(self, league=None):
        """Dates that have upcoming fixtures, for the date picker."""
        df = self._fixtures_df()
        if league:
            df = df[df.Division == league]
        if not len(df):
            return {"dates": [], "latest": None, "range": None}
        counts = df.groupby(df.MatchDate.dt.strftime("%Y-%m-%d")).size().sort_index()
        dates = [{"date": d, "matches": int(n)} for d, n in counts.items()]
        return {
            "dates": dates,
            "latest": dates[0]["date"],           # soonest, for upcoming matches
            "range": [dates[0]["date"], dates[-1]["date"]],
        }

    def fixtures(self, date=None, league=None):
        """Upcoming fixtures with a prediction for each.

        Unlike /api/matches (which replays the backtest) these have not been
        played yet, so there is no actual result to compare against.
        """
        df = self._fixtures_df()
        if league:
            df = df[df.Division == league]
        if not len(df):
            return {"date": date, "league": league, "fixtures": [],
                    "note": "No fixtures loaded. Run `python model/refresh.py`."}
        if date:
            df = df[df.MatchDate.dt.strftime("%Y-%m-%d") == date]

        # Predicting a full slate one call at a time is slow: each predict()
        # rebuilds form, head-to-head and contribution views the table never
        # shows. Instead we build every feature row first and run ONE batched
        # predict_proba -- the forest vectorises over rows, so 148 fixtures cost
        # about the same as one.
        out, rows, idx = [], [], []
        for row in df.itertuples(index=False):
            item = {
                "date": row.MatchDate.strftime("%Y-%m-%d"),
                "time": None if pd.isna(row.MatchTime) else str(row.MatchTime),
                "league": row.Division,
                "league_name": DIVISION_NAMES.get(row.Division, row.Division),
                "home_team": row.HomeTeam,
                "away_team": row.AwayTeam,
            }
            # A fixture can involve a club we have no history for (promoted from
            # a division we do not model). Report that rather than failing the
            # whole list.
            try:
                home = self.resolve_team(row.HomeTeam)
                away = self.resolve_team(row.AwayTeam)
                if home == away:
                    raise ValueError("A team cannot play itself.")
                for team in (home, away):
                    played = self.state.matches_played.get(team, 0)
                    if played < MIN_MATCHES_FOR_PREDICTION:
                        raise InsufficientHistory(
                            "Not enough history for '{}' ({} matches on record).".format(
                                team, played))
                raw = self.state.read_features(
                    home, away,
                    self.state.last_division.get(home), self.state.last_season.get(home),
                    away_division=self.state.last_division.get(away),
                    away_season=self.state.last_season.get(away),
                )
                rows.append({c: (self.fill_values[c] if pd.isna(raw.get(c, np.nan))
                                 else float(raw.get(c)))
                             for c in self.feature_columns})
                item.update({
                    "home_team": home, "away_team": away,
                    "home_elo": _clean(raw.get("home_elo")),
                    "away_elo": _clean(raw.get("away_elo")),
                    "home_last_5": list(self.state.results[home])[::-1],
                    "away_last_5": list(self.state.results[away])[::-1],
                    "elo_stale": (home in self.elo_stale_teams
                                  or away in self.elo_stale_teams),
                    "predictable": True,
                })
                idx.append(len(out))
            except (TeamNotFound, InsufficientHistory, ValueError) as exc:
                item.update({"predictable": False, "reason": str(exc)})
            out.append(item)

        if rows:
            X = pd.DataFrame(rows, columns=self.feature_columns)
            proba = self.model.predict_proba(X)
            for pos, probs in zip(idx, proba):
                by_class = {OUTCOME_KEYS[c]: float(p) for c, p in zip(self.classes, probs)}
                best = int(np.argmax(probs))
                conf = float(probs[best])
                out[pos].update({
                    "predictions": {k: round(by_class.get(k, 0.0), 4)
                                    for k in ("home_win", "draw", "away_win")},
                    "predicted_outcome": OUTCOME_NAMES[self.classes[best]],
                    "predicted_code": self.classes[best],
                    "confidence": round(conf, 4),
                    "confidence_score": self._confidence_score(conf),
                })

        predictable = [f for f in out if f.get("predictable")]
        return {
            "date": date,
            "league": league,
            "fixtures": out,
            "summary": {
                "total": len(out),
                "predictable": len(predictable),
                "skipped": len(out) - len(predictable),
            },
            "data_freshness": self.freshness(),
        }

    def freshness(self):
        """How current the underlying state is -- surfaced in the UI."""
        last = None
        if self.state.last_match_date:
            last = max(self.state.last_match_date.values())
            last = last.strftime("%Y-%m-%d") if hasattr(last, "strftime") else str(last)
        return {
            "live_state": self.state_is_live,
            "refreshed_at": self.state_refreshed_at,
            "elo_refreshed_at": self.elo_refreshed_at,
            "last_result": last,
            "teams_with_stale_elo": len(self.elo_stale_teams),
        }

    def model_info(self):
        """Everything the Model Info page needs."""
        info = dict(self.metrics)
        info["trained_at"] = self.trained_at
        info["feature_descriptions"] = FEATURE_DESCRIPTIONS
        # A 3-way coin flip scores ln(3)=1.0986; our log loss should beat it.
        info["uniform_log_loss"] = round(float(np.log(3)), 4)
        info["data_freshness"] = self.freshness()
        return info


def _clean(v):
    """JSON-safe: NaN/inf are not valid JSON, so they become null."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if np.isnan(f) or np.isinf(f):
        return None
    return round(f, 4)


_default = None


def get_predictor():
    """Process-wide singleton, so the model loads once rather than per request."""
    global _default
    if _default is None:
        _default = Predictor()
    return _default


def predict_match(home_team, away_team, league=None):
    """Convenience wrapper matching the spec in Step 5."""
    return get_predictor().predict(home_team, away_team, league=league)


if __name__ == "__main__":
    p = get_predictor()
    print("Model trained at:", p.trained_at)
    print("Teams indexed   :", len(p._index))

    for home, away in [("Real Madrid", "Barcelona"),
                       ("Man City", "Liverpool"),
                       ("Arsenal", "Bayern Munich")]:      # cross-league matchup
        try:
            r = p.predict(home, away)
        except (TeamNotFound, InsufficientHistory) as exc:
            print("\n{} vs {} -> {}".format(home, away, exc))
            continue
        print("\n" + "=" * 60)
        print("{} vs {}   [{}]".format(r["home_team"], r["away_team"], r["league_name"]))
        print("  home_win {home_win:.1%}   draw {draw:.1%}   away_win {away_win:.1%}".format(
            **r["predictions"]))
        print("  -> {} (confidence {:.1%}, score {}/10)".format(
            r["predicted_outcome"], r["confidence"], r["confidence_score"]))
        print("  home last5: {}  elo {}  pos {}".format(
            r["home_stats"]["last_5"], r["home_stats"]["elo"],
            r["home_stats"]["league_position"]))
        print("  away last5: {}  elo {}  pos {}".format(
            r["away_stats"]["last_5"], r["away_stats"]["elo"],
            r["away_stats"]["league_position"]))
        print("  h2h: played {} ({} - {} - {})".format(
            r["head_to_head"]["played"], r["head_to_head"]["team1_wins"],
            r["head_to_head"]["draws"], r["head_to_head"]["team2_wins"]))

    # Edge cases
    print("\n" + "=" * 60)
    print("EDGE CASES")
    for h, a in [("Barcelonaa", "Real Madrid"), ("Arsenal", "Arsenal")]:
        try:
            p.predict(h, a)
            print("  {} vs {} -> no error raised".format(h, a))
        except Exception as exc:
            print("  {} vs {} -> {}: {}".format(h, a, type(exc).__name__, exc))
