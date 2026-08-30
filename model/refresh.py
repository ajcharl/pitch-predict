"""
Bring the model's view of the world up to date, and fetch upcoming fixtures.

WHY THIS EXISTS
---------------
The Kaggle dataset ends on 2025-06-01. `state.pkl` therefore describes every
club as it looked at the end of the 2024-25 season: Liverpool's "last 5" is
from May 2025, and clubs promoted since are still filed under their old
division. Predicting this weekend's fixtures from that is predicting with a
15-month-old newspaper.

This script closes the gap using three free feeds, all of which need no key:

  1. RESULTS   football-data.co.uk season files -- the same upstream source the
               Kaggle set is built from, with identical columns.
  2. ELO       api.clubelo.com daily snapshot, for current team strength.
  3. FIXTURES  football-data.co.uk/fixtures.csv -- the coming week's matches.

WHAT IT DOES AND DOES NOT CHANGE
--------------------------------
It refreshes the STATE (form, standings, head-to-head, Elo) that predictions
are read from. It does NOT retrain the model. That is deliberate: the forest
was trained on seasons 2000-2022 and honestly validated on 2023-2025, and
those numbers stay meaningful only if we leave it alone. New matches change
what we know about the teams, not what the model has learned about football.

A NOTE ON ELO COVERAGE
----------------------
ClubElo's daily snapshot carries roughly 594 clubs worldwide, so some sides in
our smaller leagues simply are not in it. Rather than guess -- fuzzy matching
happily maps "Ath Madrid" onto "Real Madrid", which would be a disaster -- we
match exactly on a normalised name within the same country, plus a curated
alias table for the well-known abbreviations football-data.co.uk uses. Anything
still unmatched KEEPS its last known rating, and the script reports exactly how
many, so the staleness is visible rather than hidden.
"""

import io
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import (DATA_DIR, MODEL_DIR, TRAIN_DIVISIONS, canonicalise_team_names,
                      normalize_team_name, season_of)

LIVE_DIR = DATA_DIR / "live"
# The training state is treated as READ-ONLY and never mutated: refresh always
# rebuilds from it. That makes this script idempotent -- running it twice does
# not replay the same matches twice and double-count them into the standings.
BASE_STATE_PATH = MODEL_DIR / "state.pkl"
LIVE_STATE_PATH = MODEL_DIR / "state_live.pkl"
FIXTURES_PATH = LIVE_DIR / "fixtures.csv"
NEW_RESULTS_PATH = LIVE_DIR / "new_results.csv"
CLUBELO_PATH = LIVE_DIR / "clubelo.csv"

RESULTS_URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
CLUBELO_URL = "http://api.clubelo.com/{date}"

# football-data season codes to pull, in order. "2526" = the 2025-26 season.
# Everything from here on is newer than the Kaggle dataset's 2025-06-01 cutoff.
SEASON_CODES = ["2526", "2627"]

# Our division codes -> ClubElo's country codes, for safe same-country matching.
DIVISION_COUNTRY = {
    "E0": "ENG", "SP1": "ESP", "D1": "GER", "I1": "ITA", "F1": "FRA",
}

# Curated aliases: football-data.co.uk's abbreviation -> ClubElo's name.
# Only for clubs a normalised exact match misses. Hand-checked against the
# ClubElo country lists; deliberately conservative.
CLUBELO_ALIASES = {
    # England
    "Nottm Forest": "Forest",
    # Spain
    "Ath Madrid": "Atletico", "Ath Bilbao": "Bilbao", "Espanol": "Espanyol",
    "Vallecano": "Rayo Vallecano", "La Coruna": "Depor", "Sp Gijon": "Gijon",
    "Real Sociedad": "Sociedad",
    # Germany
    "M'gladbach": "Gladbach", "MGladbach": "Gladbach", "Ein Frankfurt": "Frankfurt",
    "FC Koln": "Koeln", "Kaiserslautern": "Lautern",
    "Greuther Furth": "Fuerth", "Nurnberg": "Nuernberg", "Hansa Rostock": "Hansa",
    # France
    "St Etienne": "Saint-Etienne",
}


def _get(url, timeout=30):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp


def _read_csv(resp):
    """Parse a football-data.co.uk CSV response.

    Their files start with a UTF-8 byte-order mark, which otherwise ends up
    glued to the first column name ("﻿Div" instead of "Div"). Reading the
    raw bytes with encoding="utf-8-sig" lets pandas strip it properly, rather
    than relying on a literal BOM character surviving in this source file.
    """
    df = pd.read_csv(io.BytesIO(resp.content), encoding="utf-8-sig",
                     encoding_errors="ignore", on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# 1. Results
# ---------------------------------------------------------------------------
def fetch_new_results(divisions=None, season_codes=None, since="2025-06-01", verbose=True):
    """Download recent season results and normalise them to our schema.

    football-data.co.uk publishes one CSV per league per season, with columns
    (Div, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, ...). We rename those to
    match the Kaggle dataset so the rest of the pipeline cannot tell the
    difference.
    """
    divisions = divisions or TRAIN_DIVISIONS
    season_codes = season_codes or SEASON_CODES
    frames = []
    for code in season_codes:
        for div in divisions:
            url = RESULTS_URL.format(season=code, div=div)
            try:
                resp = _get(url)
            except Exception as exc:
                if verbose:
                    print("      {} {}: unavailable ({})".format(code, div, exc))
                continue
            try:
                raw = _read_csv(resp)
            except Exception:
                continue
            need = {"Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
            if not need.issubset(raw.columns):
                continue
            # Only finished matches carry a score.
            raw = raw.dropna(subset=["FTHG", "FTAG", "FTR", "HomeTeam", "AwayTeam"])
            if not len(raw):
                continue
            out = pd.DataFrame({
                "Division": raw.Div,
                # football-data uses dd/mm/yy or dd/mm/yyyy.
                "MatchDate": pd.to_datetime(raw.Date, dayfirst=True, format="mixed",
                                            errors="coerce"),
                "MatchTime": raw.get("Time"),
                "HomeTeam": raw.HomeTeam.astype(str).str.strip(),
                "AwayTeam": raw.AwayTeam.astype(str).str.strip(),
                "FTHome": pd.to_numeric(raw.FTHG, errors="coerce"),
                "FTAway": pd.to_numeric(raw.FTAG, errors="coerce"),
                "FTResult": raw.FTR,
            })
            out = out.dropna(subset=["MatchDate", "FTHome", "FTAway"])
            frames.append(out)
            if verbose:
                print("      {} {:<4} {:>4} matches".format(code, div, len(out)))

    if not frames:
        return pd.DataFrame(columns=["Division", "MatchDate", "HomeTeam", "AwayTeam",
                                     "FTHome", "FTAway", "FTResult"])

    df = pd.concat(frames, ignore_index=True)
    # Only matches AFTER the Kaggle cutoff, so nothing is replayed twice.
    df = df[df.MatchDate > pd.Timestamp(since)]
    # Same spelling-variant merge the training data gets.
    df, _ = canonicalise_team_names(df)
    df = df.sort_values("MatchDate", kind="mergesort").reset_index(drop=True)
    df["Season"] = df.MatchDate.map(season_of)
    return df


def align_names_to_state(df, state, verbose=True):
    """Re-spell incoming team names to match the ones already in the state.

    `canonicalise_team_names` only unifies spellings WITHIN one dataframe. The
    new season files spell Forest "Nott'm Forest", while the training data
    settled on "Nottm Forest" -- so replaying them blind would create a second,
    empty Forest with no history and no Elo, which is exactly the bug the
    canonicaliser exists to prevent.

    So we map each incoming name onto the state's existing spelling whenever
    they share a normalised key. Genuinely new clubs (promoted from a division
    we do not model) keep their own name and simply start with no history.
    """
    known = {}
    for team in state.matches_played:
        known.setdefault(normalize_team_name(team), team)

    mapping, brand_new = {}, set()
    for col in ("HomeTeam", "AwayTeam"):
        for name in df[col].unique():
            canonical = known.get(normalize_team_name(name))
            if canonical is None:
                brand_new.add(name)
            elif canonical != name:
                mapping[name] = canonical

    if mapping:
        df["HomeTeam"] = df.HomeTeam.replace(mapping)
        df["AwayTeam"] = df.AwayTeam.replace(mapping)
    if verbose:
        if mapping:
            preview = ", ".join("{!r}->{!r}".format(k, v) for k, v in list(mapping.items())[:5])
            print("      re-spelled {} names to match existing teams ({})".format(
                len(mapping), preview))
        if brand_new:
            print("      {} clubs new to the model: {}".format(
                len(brand_new), ", ".join(sorted(brand_new)[:8])))
    return df, mapping, brand_new


# ---------------------------------------------------------------------------
# 2. Elo
# ---------------------------------------------------------------------------
def fetch_clubelo(date=None, verbose=True):
    """Download ClubElo's rating snapshot for a date (default: today)."""
    date = date or datetime.now().strftime("%Y-%m-%d")
    resp = _get(CLUBELO_URL.format(date=date))
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    CLUBELO_PATH.write_text(resp.text, encoding="utf-8")
    elo = _read_csv(resp)
    if verbose:
        print("      {} clubs rated as of {}".format(len(elo), date))
    return elo


def build_elo_mapping(state, elo, verbose=True):
    """Map our team names onto ClubElo's, conservatively.

    Exact normalised match within the same country first, then the curated
    alias table. Anything left over is reported, not guessed -- a wrong match
    would silently corrupt the model's most important feature.
    """
    by_country = {}
    for row in elo.itertuples(index=False):
        by_country.setdefault((normalize_team_name(row.Club), row.Country), float(row.Elo))

    # Aliases resolved against the whole snapshot (country-independent), since
    # they are hand-checked.
    by_name = {}
    for row in elo.itertuples(index=False):
        by_name.setdefault(normalize_team_name(row.Club), float(row.Elo))

    mapping, unmatched = {}, []
    for team, div in state.last_division.items():
        if state.last_season.get(team, -1) < 2023:
            continue
        country = DIVISION_COUNTRY.get(div)
        key = (normalize_team_name(team), country)
        if key in by_country:
            mapping[team] = by_country[key]
            continue
        alias = CLUBELO_ALIASES.get(team)
        if alias and normalize_team_name(alias) in by_name:
            mapping[team] = by_name[normalize_team_name(alias)]
            continue
        unmatched.append(team)

    if verbose:
        total = len(mapping) + len(unmatched)
        print("      matched {}/{} active teams to a live rating".format(len(mapping), total))
        print("      {} keep their last known Elo (not in ClubElo's top-594 snapshot)"
              .format(len(unmatched)))
    return mapping, unmatched


# ---------------------------------------------------------------------------
# 3. Fixtures
# ---------------------------------------------------------------------------
def fetch_fixtures(divisions=None, state=None, verbose=True):
    """Download upcoming fixtures and keep the ones in leagues we model."""
    divisions = set(divisions or TRAIN_DIVISIONS)
    resp = _get(FIXTURES_URL)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    raw = _read_csv(resp)
    fx = pd.DataFrame({
        "Division": raw.Div,
        "MatchDate": pd.to_datetime(raw.Date, dayfirst=True, format="mixed", errors="coerce"),
        "MatchTime": raw.get("Time"),
        "HomeTeam": raw.HomeTeam.astype(str).str.strip(),
        "AwayTeam": raw.AwayTeam.astype(str).str.strip(),
    }).dropna(subset=["MatchDate", "HomeTeam", "AwayTeam"])
    fx = fx[fx.Division.isin(divisions)]
    fx, _ = canonicalise_team_names(fx)
    fx = fx.sort_values(["MatchDate", "MatchTime"], kind="mergesort").reset_index(drop=True)
    if state is not None:
        fx, _m, _f = align_names_to_state(fx, state, verbose=False)
    fx.to_csv(FIXTURES_PATH, index=False)
    if verbose:
        print("      {} fixtures in modelled leagues, {} -> {}".format(
            len(fx), fx.MatchDate.min().date(), fx.MatchDate.max().date()))
    return fx


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def refresh(verbose=True):
    t0 = time.time()
    LIVE_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Downloading results since the dataset cutoff (2025-06-01)...")
    new = fetch_new_results(verbose=verbose)
    print("      {:,} new matches across {} divisions".format(
        len(new), new.Division.nunique() if len(new) else 0))

    print("[2/4] Replaying them through the team state...")
    state = joblib.load(BASE_STATE_PATH)   # always from the pristine base
    before = dict(state.last_division)
    if len(new):
        new, _remap, _fresh = align_names_to_state(new, state)
        cols = ["HomeTeam", "AwayTeam", "Division", "Season", "FTHome", "FTAway", "MatchDate"]
        for home, away, div, season, hg, ag, date in new[cols].itertuples(index=False, name=None):
            # Same update path training used, so form, standings, H2H and the
            # match logs all stay consistent with how the model was fed.
            state.update(home, away, hg, ag, div, season, date=date, keep_log=True)
        new.to_csv(NEW_RESULTS_PATH, index=False)
        moved = sum(1 for t, d in state.last_division.items() if before.get(t) != d)
        print("      state advanced to {}".format(max(state.last_match_date.values()).date()))
        print("      {} teams changed division (promotions/relegations)".format(moved))

    print("[3/4] Refreshing Elo from ClubElo...")
    try:
        elo = fetch_clubelo(verbose=verbose)
        mapping, unmatched = build_elo_mapping(state, elo, verbose=verbose)
        changes = []
        for team, value in mapping.items():
            old = state.elo.get(team)
            if old is not None:
                changes.append(value - old)
            state.elo[team] = value
        if changes:
            print("      median Elo shift since May 2025: {:+.0f} points".format(
                float(pd.Series(changes).abs().median())))
        state.elo_stale_teams = sorted(unmatched)
        state.elo_refreshed_at = datetime.now().strftime("%Y-%m-%d")
    except Exception as exc:
        print("      WARNING: ClubElo refresh failed ({}); keeping existing ratings".format(exc))
        state.elo_stale_teams = []
        state.elo_refreshed_at = None

    print("[4/4] Downloading upcoming fixtures...")
    try:
        fetch_fixtures(state=state, verbose=verbose)
    except Exception as exc:
        print("      WARNING: fixture download failed ({})".format(exc))

    state.refreshed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    joblib.dump(state, LIVE_STATE_PATH, compress=3)
    print("\nSaved refreshed state -> {}".format(LIVE_STATE_PATH))
    print("Done in {:.0f}s. Restart the API to pick it up.".format(time.time() - t0))
    return state


if __name__ == "__main__":
    refresh()
