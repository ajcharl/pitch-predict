"""
Step 3 - Feature engineering.

THE MOST IMPORTANT FILE IN THE PROJECT.

A machine learning model can only be as good as the numbers you feed it.
The raw dataset gives us "Arsenal 2 - 1 Chelsea on 2024-04-01". That is the
ANSWER, not a clue. To predict a match we need to describe the two teams as
they looked *going into* that match: recent form, goals, home/away splits,
head-to-head history, league position, Elo rating.

--------------------------------------------------------------------------
THE GOLDEN RULE: NO DATA LEAKAGE
--------------------------------------------------------------------------
"Leakage" is when information from the future sneaks into a feature. An
example of a leak: computing "average goals scored in the last 5 games" but
including the current match in that average. The model would then partly
*see* the result it is meant to predict, and would report a fantastic
accuracy that collapses the moment you use it on a real upcoming fixture.

We make leakage structurally impossible by processing matches in strict date
order and using a READ-THEN-WRITE loop:

    for each match (oldest -> newest):
        features = state.read_features(home, away)  # describe teams using ONLY the past
        rows.append(features)
        state.update(match)                         # NOW fold in this match's result

Because `read` happens before `update`, a match can never influence its own
features. This is the same reasoning behind the time-based train/test split
used in train.py.

--------------------------------------------------------------------------
BONUS: TRAIN/SERVE CONSISTENCY
--------------------------------------------------------------------------
The same `TeamState` object builds the training table AND the feature vector
for a live prediction. So the features the model is asked about at prediction
time are produced by exactly the same code that produced its training data --
a classic source of silent bugs, eliminated by construction.
"""

import unicodedata
from collections import deque, defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

# How many recent matches count as "current form". 5 is the football standard
# ("the last five"): long enough to smooth out one fluky result, short enough
# to still reflect a team's current shape rather than their whole season.
# Project paths, anchored to this file so every script and the API can run
# from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
DATA_DIR = MODEL_DIR / "data"
MATCHES_CSV = DATA_DIR / "Matches.csv"

FORM_WINDOW = 5

# The five leagues this app covers. All have 100% Elo coverage in the dataset,
# which keeps our single strongest feature dense. Elo is calibrated ACROSS
# leagues, so a cross-league matchup is still a fair comparison.
TRAIN_DIVISIONS = [
    "E0",   # England  - Premier League
    "SP1",  # Spain    - La Liga
    "D1",   # Germany  - Bundesliga
    "I1",   # Italy    - Serie A
    "F1",   # France   - Ligue 1
]

# Human-readable names, used by the API and UI.
DIVISION_NAMES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

# The exact feature columns, in a fixed order. The model is trained on this
# order and predict.py rebuilds vectors in this order -- never reorder these
# without retraining.
FEATURE_COLUMNS = [
    # --- Elo: a single number summarising team strength --------------------
    "home_elo", "away_elo", "elo_diff",
    # --- Overall form over the last 5 matches (any venue) ------------------
    "home_win_rate_5", "away_win_rate_5",
    "home_ppg_5", "away_ppg_5",
    "home_gf_avg_5", "away_gf_avg_5",
    "home_ga_avg_5", "away_ga_avg_5",
    "home_clean_sheets_5", "away_clean_sheets_5",
    # --- Venue-specific form -----------------------------------------------
    "home_home_win_rate_5", "away_away_win_rate_5",
    "home_home_gf_avg_5", "away_away_ga_avg_5",
    # --- Head to head -------------------------------------------------------
    "h2h_home_win_rate", "h2h_draw_rate", "h2h_avg_goals", "h2h_played",
    # --- Season context -----------------------------------------------------
    "home_position", "away_position", "position_diff",
    "home_goal_diff", "away_goal_diff", "goal_diff_diff",
    "home_ppg_season", "away_ppg_season", "matchday",
    # --- Explicit difference features ---------------------------------------
    # Random Forests split on one feature at a time, so they can only express
    # "home_ppg_5 - away_ppg_5 > 0.4" through many nested splits. Handing the
    # difference over directly turns that comparison into a single cheap split.
    "form_diff", "attack_vs_defence", "defence_vs_attack",
]

# Plain-English descriptions, surfaced in the UI's "why this prediction" panel.
FEATURE_DESCRIPTIONS = {
    "home_elo": "Home team's Elo rating (overall strength)",
    "away_elo": "Away team's Elo rating (overall strength)",
    "elo_diff": "Elo gap between the two teams",
    "home_win_rate_5": "Home team's win rate in its last 5 matches",
    "away_win_rate_5": "Away team's win rate in its last 5 matches",
    "home_ppg_5": "Home team's points per game, last 5",
    "away_ppg_5": "Away team's points per game, last 5",
    "home_gf_avg_5": "Home team's goals scored per game, last 5",
    "away_gf_avg_5": "Away team's goals scored per game, last 5",
    "home_ga_avg_5": "Home team's goals conceded per game, last 5",
    "away_ga_avg_5": "Away team's goals conceded per game, last 5",
    "home_clean_sheets_5": "Home team's clean sheets in its last 5",
    "away_clean_sheets_5": "Away team's clean sheets in its last 5",
    "home_home_win_rate_5": "Home team's win rate in its last 5 home matches",
    "away_away_win_rate_5": "Away team's win rate in its last 5 away matches",
    "home_home_gf_avg_5": "Home team's goals scored per home match",
    "away_away_ga_avg_5": "Away team's goals conceded per away match",
    "h2h_home_win_rate": "Home team's win rate in recent meetings",
    "h2h_draw_rate": "Draw rate in recent meetings",
    "h2h_avg_goals": "Average goals in recent meetings",
    "h2h_played": "How many recent meetings we have on record",
    "home_position": "Home team's current league position",
    "away_position": "Away team's current league position",
    "position_diff": "League positions apart",
    "home_goal_diff": "Home team's goal difference this season",
    "away_goal_diff": "Away team's goal difference this season",
    "goal_diff_diff": "Season goal-difference gap",
    "home_ppg_season": "Home team's points per game this season",
    "away_ppg_season": "Away team's points per game this season",
    "matchday": "How far into the season the match is",
    "form_diff": "Form gap (home points per game minus away)",
    "attack_vs_defence": "Home attack vs away defence",
    "defence_vs_attack": "Away attack vs home defence",
}


def normalize_team_name(name):
    """Reduce a team name to a comparison key: no accents, case or punctuation.

    "Nott'm Forest", "Nottm Forest" and "Nottm Forest " all reduce to
    "nottmforest".
    """
    text = "".join(c for c in unicodedata.normalize("NFKD", str(name))
                   if not unicodedata.combining(c))
    return "".join(ch for ch in text.lower() if ch.isalnum())


def canonicalise_team_names(df, verbose=False):
    """Merge spelling variants of the same club into one canonical name.

    The source data spells a handful of clubs two ways -- "Nottm Forest" for
    1,089 matches and "Nott'm Forest" for the 21 most recent ones, plus a few
    names with a stray trailing space. Left alone, each variant becomes a
    SEPARATE team: its form deques, head-to-head record and Elo all fragment,
    so the club we actually want to predict for looks like a newly promoted
    side with no history.

    We fix it by grouping names that share a normalised key and keeping the
    most frequently used spelling as canonical. No hardcoded list, so new
    variants in refreshed data are handled automatically.
    """
    counts = defaultdict(int)
    for col in ("HomeTeam", "AwayTeam"):
        for team, n in df[col].value_counts().items():
            counts[team] += n

    groups = defaultdict(list)
    for team in counts:
        groups[normalize_team_name(team)].append(team)

    mapping = {}
    for variants in groups.values():
        if len(variants) < 2:
            continue
        # The spelling used most often wins.
        canonical = max(variants, key=lambda t: counts[t])
        for variant in variants:
            if variant != canonical:
                mapping[variant] = canonical
                if verbose:
                    print("      merged '{}' ({}) -> '{}' ({})".format(
                        variant, counts[variant], canonical, counts[canonical]))

    if mapping:
        df["HomeTeam"] = df.HomeTeam.replace(mapping)
        df["AwayTeam"] = df.AwayTeam.replace(mapping)
    return df, mapping


def season_of(date):
    """European seasons straddle the new year (Aug 2024 -> May 2025).

    We label that whole span '2024'. Anything from July onwards belongs to the
    season starting that calendar year; January-June belongs to the season that
    started the previous year.
    """
    return date.year if date.month >= 7 else date.year - 1


def _safe_mean(values):
    """Mean of a deque, or NaN when we have no history yet.

    NaN rather than 0 is deliberate: 0 would be a lie meaning "this team scores
    no goals", whereas NaN honestly means "we don't know yet" and gets filled
    with the column median at training time.
    """
    return float(np.mean(values)) if len(values) else np.nan


def _diff(x, y):
    return (x - y) if not (pd.isna(x) or pd.isna(y)) else np.nan


def _round(v, nd=2):
    return None if v is None or pd.isna(v) else round(float(v), nd)


def _int(v):
    return None if v is None or pd.isna(v) else int(v)


def _deque_factory(window):
    """Picklable replacement for `lambda: deque(maxlen=window)`.

    defaultdict stores its default_factory, and a lambda cannot be pickled --
    which would stop us caching a fully-built TeamState for fast API startup.
    functools.partial pickles fine.
    """
    return partial(deque, maxlen=window)


class TeamState:
    """Rolling memory of every team, as of "right now" in the walk through history.

    Everything held here describes the PAST only. Call `read_features` to
    describe an upcoming match, then `update` once its result is known.
    """

    def __init__(self, window=FORM_WINDOW):
        self.window = window

        # --- Form: last `window` matches, any venue ------------------------
        # Each deque holds one entry per recent match, newest appended last.
        self.results = defaultdict(_deque_factory(window))        # 'W'/'D'/'L'
        self.points = defaultdict(_deque_factory(window))         # 3/1/0
        self.goals_for = defaultdict(_deque_factory(window))
        self.goals_against = defaultdict(_deque_factory(window))
        self.clean_sheets = defaultdict(_deque_factory(window))   # 1/0

        # --- Venue-specific form -------------------------------------------
        # Some teams are fortresses at home and timid away; one blended form
        # number hides that, so we track home and away games separately.
        self.home_results = defaultdict(_deque_factory(window))
        self.home_goals_for = defaultdict(_deque_factory(window))
        self.away_results = defaultdict(_deque_factory(window))
        self.away_goals_against = defaultdict(_deque_factory(window))

        # --- Head to head ----------------------------------------------------
        # Keyed by an unordered pair so Arsenal-Chelsea and Chelsea-Arsenal
        # share one history. We store each outcome from the perspective of the
        # alphabetically-first team, then flip it at read time when needed.
        self.h2h = defaultdict(_deque_factory(window))  # (a_won, drew, total_goals)
        # Full meeting log, for the head-to-head page in the UI.
        self.h2h_log = defaultdict(list)

        # --- Season standings -------------------------------------------------
        # (division, season) -> {team: {"pts", "gd", "played"}}
        self.standings = defaultdict(dict)

        # --- Latest known Elo, plus bookkeeping --------------------------------
        self.elo = {}
        self.last_division = {}
        self.last_season = {}
        self.matches_played = defaultdict(int)
        self.last_match_date = {}
        # Recent match log per team, for the team profile page.
        self.match_log = defaultdict(list)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pair_key(a, b):
        return (a, b) if a <= b else (b, a)

    def _table(self, division, season):
        return self.standings[(division, season)]

    def _position(self, division, season, team):
        """League position = rank by points, then goal difference.

        Computed from the standings as they stand BEFORE the current match, so
        it reflects only completed fixtures.
        """
        table = self._table(division, season)
        if team not in table:
            return np.nan
        ordered = sorted(table.items(), key=lambda kv: (-kv[1]["pts"], -kv[1]["gd"]))
        for rank, (name, _) in enumerate(ordered, start=1):
            if name == team:
                return float(rank)
        return np.nan

    # ------------------------------------------------------------------
    # reading (never mutates)
    # ------------------------------------------------------------------
    def read_features(self, home, away, division, season, home_elo=None, away_elo=None,
                      away_division=None, away_season=None):
        """Describe an upcoming match as a dict of features.

        `away_division`/`away_season` exist for CROSS-LEAGUE fixtures (a
        cross-league matchup, say). In a normal league match both teams share
        one league table, so these default to the home team's and nothing
        changes -- which is exactly the case during training.
        """
        f = {}

        # ---- Elo ---------------------------------------------------------
        # Elo is a chess-style rating: every team has a number (~1500 is
        # average), it rises when you win and falls when you lose, and beating
        # a strong opponent moves it more than beating a weak one. It is the
        # single most informative feature we have because it compresses years
        # of results into one calibrated number -- and because ClubElo rates
        # every European club on the same scale, it also works across leagues.
        he = home_elo if home_elo is not None and not pd.isna(home_elo) else self.elo.get(home, np.nan)
        ae = away_elo if away_elo is not None and not pd.isna(away_elo) else self.elo.get(away, np.nan)
        f["home_elo"] = he
        f["away_elo"] = ae
        # The DIFFERENCE matters more than the absolute values: 1900 vs 1500 is
        # a mismatch, 1900 vs 1880 is a coin flip, yet both share home_elo=1900.
        f["elo_diff"] = _diff(he, ae)

        # ---- Overall form over the last 5 ---------------------------------
        for side, team in (("home", home), ("away", away)):
            res = self.results[team]
            f[side + "_win_rate_5"] = _safe_mean([r == "W" for r in res])
            f[side + "_ppg_5"] = _safe_mean(self.points[team])
            f[side + "_gf_avg_5"] = _safe_mean(self.goals_for[team])
            f[side + "_ga_avg_5"] = _safe_mean(self.goals_against[team])
            # A raw count (0-5) rather than a rate: "4 clean sheets in 5" is the
            # natural way to express defensive solidity.
            cs = self.clean_sheets[team]
            f[side + "_clean_sheets_5"] = float(sum(cs)) if len(cs) else np.nan

        # ---- Venue-specific form -------------------------------------------
        # Home advantage is the strongest single effect in football (~45% of all
        # matches are home wins), but it is not equal for every team.
        f["home_home_win_rate_5"] = _safe_mean([r == "W" for r in self.home_results[home]])
        f["away_away_win_rate_5"] = _safe_mean([r == "W" for r in self.away_results[away]])
        f["home_home_gf_avg_5"] = _safe_mean(self.home_goals_for[home])
        f["away_away_ga_avg_5"] = _safe_mean(self.away_goals_against[away])

        # ---- Head to head ---------------------------------------------------
        # Some fixtures have a character that transcends form (a team that
        # "always" beats another). With only 5 meetings this is a weak, noisy
        # signal, so we also expose h2h_played and let the model learn to
        # discount it when two sides have barely met.
        a, b = self._pair_key(home, away)
        hist = self.h2h[(a, b)]
        if len(hist):
            home_is_a = (home == a)
            home_wins = [(aw if home_is_a else (not aw and not dr)) for aw, dr, _ in hist]
            f["h2h_home_win_rate"] = float(np.mean(home_wins))
            f["h2h_draw_rate"] = float(np.mean([dr for _, dr, _ in hist]))
            f["h2h_avg_goals"] = float(np.mean([g for _, _, g in hist]))
        else:
            f["h2h_home_win_rate"] = np.nan
            f["h2h_draw_rate"] = np.nan
            f["h2h_avg_goals"] = np.nan
        f["h2h_played"] = float(len(hist))

        # ---- Season context --------------------------------------------------
        # Where the teams sit in the table right now, and how far into the
        # season we are (early-season form is noisier than late-season form).
        a_div = away_division if away_division is not None else division
        a_seas = away_season if away_season is not None else season
        hs = self._table(division, season).get(home, {"pts": 0, "gd": 0, "played": 0})
        aw_ = self._table(a_div, a_seas).get(away, {"pts": 0, "gd": 0, "played": 0})
        f["home_position"] = self._position(division, season, home)
        f["away_position"] = self._position(a_div, a_seas, away)
        f["position_diff"] = _diff(f["home_position"], f["away_position"])
        f["home_goal_diff"] = float(hs["gd"])
        f["away_goal_diff"] = float(aw_["gd"])
        f["goal_diff_diff"] = float(hs["gd"] - aw_["gd"])
        f["home_ppg_season"] = float(hs["pts"] / hs["played"]) if hs["played"] else np.nan
        f["away_ppg_season"] = float(aw_["pts"] / aw_["played"]) if aw_["played"] else np.nan
        f["matchday"] = float(max(hs["played"], aw_["played"]) + 1)

        # ---- Explicit comparisons ---------------------------------------------
        f["form_diff"] = _diff(f["home_ppg_5"], f["away_ppg_5"])
        # Home's attack against Away's defence, and vice versa. A team scoring
        # 2.4/game facing a defence conceding 1.8/game is a very different
        # prospect from the same attack facing a defence conceding 0.4.
        f["attack_vs_defence"] = _diff(f["home_gf_avg_5"], f["away_ga_avg_5"])
        f["defence_vs_attack"] = _diff(f["away_gf_avg_5"], f["home_ga_avg_5"])
        return f

    # ------------------------------------------------------------------
    # human-readable snapshots for the UI (not model features)
    # ------------------------------------------------------------------
    def team_snapshot(self, team):
        return {
            "last_5": list(self.results[team]),
            "goals_scored_avg": _round(_safe_mean(self.goals_for[team])),
            "goals_conceded_avg": _round(_safe_mean(self.goals_against[team])),
            "points_per_game": _round(_safe_mean(self.points[team])),
            "clean_sheets_last_5": int(sum(self.clean_sheets[team])),
            "elo": _round(self.elo.get(team, np.nan)),
            "matches_played": self.matches_played[team],
            "league": self.last_division.get(team),
        }

    def season_snapshot(self, team, division=None, season=None):
        division = division or self.last_division.get(team)
        season = season if season is not None else self.last_season.get(team)
        if division is None or season is None:
            return {"league_position": None, "played": 0, "points": 0, "goal_diff": 0}
        rec = self._table(division, season).get(team)
        if not rec:
            return {"league_position": None, "played": 0, "points": 0, "goal_diff": 0}
        return {
            "league_position": _int(self._position(division, season, team)),
            "played": rec["played"],
            "points": rec["pts"],
            "goal_diff": rec["gd"],
        }

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------
    def update(self, home, away, home_goals, away_goals, division, season, date=None,
               home_elo=None, away_elo=None, keep_log=False):
        """Fold a finished match into the rolling state. Call AFTER read_features."""
        if home_goals > away_goals:
            hr, ar, hp, ap = "W", "L", 3, 0
        elif home_goals < away_goals:
            hr, ar, hp, ap = "L", "W", 0, 3
        else:
            hr, ar, hp, ap = "D", "D", 1, 1

        for team, res, pts, gf, ga in ((home, hr, hp, home_goals, away_goals),
                                       (away, ar, ap, away_goals, home_goals)):
            self.results[team].append(res)
            self.points[team].append(pts)
            self.goals_for[team].append(gf)
            self.goals_against[team].append(ga)
            self.clean_sheets[team].append(1 if ga == 0 else 0)
            self.matches_played[team] += 1
            self.last_division[team] = division
            self.last_season[team] = season
            if date is not None:
                self.last_match_date[team] = date

        # Venue-specific deques
        self.home_results[home].append(hr)
        self.home_goals_for[home].append(home_goals)
        self.away_results[away].append(ar)
        self.away_goals_against[away].append(home_goals)

        # Head to head, stored from the alphabetically-first team's perspective
        a, b = self._pair_key(home, away)
        a_won = (home == a and hr == "W") or (away == a and ar == "W")
        drew = hr == "D"
        self.h2h[(a, b)].append((a_won, drew, home_goals + away_goals))

        # Season table
        table = self._table(division, season)
        for team, pts, gf, ga in ((home, hp, home_goals, away_goals),
                                  (away, ap, away_goals, home_goals)):
            rec = table.setdefault(team, {"pts": 0, "gd": 0, "played": 0})
            rec["pts"] += pts
            rec["gd"] += gf - ga
            rec["played"] += 1

        # Elo. The dataset's HomeElo/AwayElo are PRE-match ratings (verified in
        # explore.py against ClubElo's dated snapshots), so the most recent one
        # we have seen is our best estimate of the team's current strength.
        if home_elo is not None and not pd.isna(home_elo):
            self.elo[home] = float(home_elo)
        if away_elo is not None and not pd.isna(away_elo):
            self.elo[away] = float(away_elo)

        # Optional match logs, used by the API for form/H2H history views.
        if keep_log:
            entry = {
                "date": str(date.date()) if hasattr(date, "date") else str(date),
                "home_team": home, "away_team": away,
                "home_goals": int(home_goals), "away_goals": int(away_goals),
                "division": division, "season": int(season),
            }
            self.h2h_log[(a, b)].append(entry)
            self.match_log[home].append(entry)
            self.match_log[away].append(entry)


def load_matches(path=None, divisions=None, verbose=False):
    """Load the raw dataset and keep only rows we can actually learn from."""
    df = pd.read_csv(path or MATCHES_CSV, parse_dates=["MatchDate"], low_memory=False)
    divisions = divisions if divisions is not None else TRAIN_DIVISIONS
    df = df[df.Division.isin(divisions)]
    # A match with no score is useless: it can be neither a feature nor a label.
    df = df.dropna(subset=["FTHome", "FTAway", "FTResult", "HomeTeam", "AwayTeam"])
    # Merge spelling variants BEFORE anything reads team identity, so a club's
    # history stays in one place.
    df, _merged = canonicalise_team_names(df, verbose=verbose)
    # mergesort is stable, so same-day matches keep their original file order.
    df = df.sort_values("MatchDate", kind="mergesort").reset_index(drop=True)
    df["Season"] = df.MatchDate.map(season_of)
    return df


def build_feature_table(df, state=None, keep_log=False, progress_every=25000):
    """Walk every match oldest -> newest, emitting one feature row per match.

    This is the read-then-write loop described at the top of the file, and it
    is the reason no future information can reach a feature.
    """
    state = state if state is not None else TeamState()
    rows = []
    cols = ["HomeTeam", "AwayTeam", "Division", "Season", "FTHome", "FTAway",
            "FTResult", "MatchDate", "HomeElo", "AwayElo"]
    for i, (home, away, div, season, hg, ag, res, date, he, ae) in enumerate(
            df[cols].itertuples(index=False, name=None)):
        # 1. READ -- describe the match using only what has happened so far
        feats = state.read_features(home, away, div, season, home_elo=he, away_elo=ae)
        feats["MatchDate"] = date
        feats["Division"] = div
        feats["Season"] = season
        feats["HomeTeam"] = home
        feats["AwayTeam"] = away
        feats["FTResult"] = res
        rows.append(feats)
        # 2. WRITE -- only now does this match become part of history
        state.update(home, away, hg, ag, div, season, date=date,
                     home_elo=he, away_elo=ae, keep_log=keep_log)
        if progress_every and i and i % progress_every == 0:
            print("  ...{:,} matches processed".format(i))

    return pd.DataFrame(rows), state


if __name__ == "__main__":
    print("Loading raw matches...")
    matches = load_matches()
    print("{:,} matches across {} divisions".format(len(matches), matches.Division.nunique()))
    print("Building features (single pass, oldest -> newest)...")
    table, _state = build_feature_table(matches)
    out = DATA_DIR / "features.csv"
    table.to_csv(out, index=False)
    print("wrote", out, table.shape)
    print(table[FEATURE_COLUMNS].describe().T.to_string())
