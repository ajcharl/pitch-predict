"""Prediction endpoints: leagues, team lists, and the prediction itself."""

from fastapi import APIRouter, Query

router = APIRouter(tags=["predictions"])

# Replaced by main.py at import time with the real accessor (this keeps the
# route modules free of a circular import back into main).
get_predictor = None


@router.get("/leagues")
def list_leagues():
    """The five leagues the model serves."""
    return {"leagues": get_predictor().leagues()}


@router.get("/teams")
def list_teams(
    league: str = Query(None, description="League code: E0, SP1, D1, I1 or F1"),
    active_since: int = Query(None, description="Only teams active since this season "
                              "(defaults to the current season)"),
):
    """Teams available for a league, strongest first.

    Omit `league` to get every team the model knows about.
    """
    p = get_predictor()
    teams = p.teams_in_league(league, active_since=active_since)
    return {"league": league, "count": len(teams), "teams": teams}


@router.get("/predict")
def predict(
    home: str = Query(..., description="Home team name"),
    away: str = Query(..., description="Away team name"),
    league: str = Query(None, description="Optional league code for labelling"),
):
    """Predict a single fixture.

    Returns probabilities for all three outcomes, the most likely result, a
    confidence score, the feature values behind it, and both teams' current
    form. Unknown team names produce a 404 with spelling suggestions.
    """
    return get_predictor().predict(home, away, league=league)
