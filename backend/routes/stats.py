"""Stats endpoints: team form, head-to-head, browsable matchdays, model metrics."""

from fastapi import APIRouter, Path, Query

router = APIRouter(tags=["stats"])

# Replaced by main.py at import time (see routes/predictions.py for why).
get_predictor = None


@router.get("/team/{team_name}/form")
def team_form(team_name: str = Path(..., description="Team name")):
    """Recent form, season stats and match history for one team."""
    return get_predictor().team_form(team_name)


@router.get("/head-to-head")
def head_to_head(
    team1: str = Query(..., description="First team"),
    team2: str = Query(..., description="Second team"),
    limit: int = Query(20, ge=1, le=50, description="How many recent meetings"),
):
    """Head-to-head record and recent meetings between two teams."""
    return get_predictor().head_to_head(team1, team2, limit=limit)


@router.get("/matches")
def matches(
    date: str = Query(None, description="Match date, YYYY-MM-DD"),
    league: str = Query(None, description="Optional league code filter"),
):
    """Matches on a given date, each with the model's prediction and the result.

    These come from the backtest over the test period (seasons 2023-2025) --
    matches the model never trained on -- so the prediction shown next to each
    result is a genuine out-of-sample call.
    """
    return get_predictor().matches_on(date, league=league)


@router.get("/matches/dates")
def match_dates(league: str = Query(None, description="Optional league code filter")):
    """Which dates have matches available, for the date picker."""
    return get_predictor().available_dates(league=league)


@router.get("/fixtures")
def fixtures(
    date: str = Query(None, description="Fixture date, YYYY-MM-DD"),
    league: str = Query(None, description="Optional league code filter"),
):
    """Upcoming fixtures, each with a prediction.

    These have not been played yet, so there is no result to compare against.
    Populated by `python model/refresh.py`.
    """
    return get_predictor().fixtures(date, league=league)


@router.get("/fixtures/dates")
def fixture_dates(league: str = Query(None, description="Optional league code filter")):
    """Dates that have upcoming fixtures, for the date picker."""
    return get_predictor().fixture_dates(league=league)


@router.get("/freshness")
def freshness():
    """How current the model's view of the world is."""
    return get_predictor().freshness()


@router.get("/model/accuracy")
def model_accuracy():
    """Model performance: overall and per-class accuracy, per-league breakdown,
    confusion matrix, feature importance and the baseline comparison."""
    return get_predictor().model_info()
