"""
Step 6 - The FastAPI application.

Loading the model and replaying 130k matches of history takes a couple of
seconds. Doing that per request would make the API unusable, so it happens
ONCE during the startup lifespan and every request then shares the same
in-memory Predictor.
"""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Make the sibling `model/` package importable without an install step.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "model"))
# Also add backend/ itself so `from routes import ...` works whether you run
# `uvicorn main:app` from backend/ or `uvicorn backend.main:app` from the root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict import (InsufficientHistory, Predictor,  # noqa: E402
                     TeamNotFound)
from routes import predictions, stats  # noqa: E402

# Populated on startup and read by the route modules via `get_predictor`.
_state = {"predictor": None, "loaded_at": None, "load_seconds": None, "error": None}


def get_predictor():
    """Dependency used by the routes. Raises if the model failed to load."""
    if _state["predictor"] is None:
        raise RuntimeError(
            "Model is not loaded. Run `python model/train.py` to build "
            "model/model.pkl and model/state.pkl, then restart the API."
        )
    return _state["predictor"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    try:
        _state["predictor"] = Predictor()
        _state["load_seconds"] = round(time.time() - t0, 2)
        _state["loaded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        print("[matchiq] model loaded in {}s".format(_state["load_seconds"]))
    except Exception as exc:  # keep the API up so /api/health can explain why
        _state["error"] = str(exc)
        print("[matchiq] FAILED to load model: {}".format(exc))
    yield
    _state["predictor"] = None


app = FastAPI(
    title="MatchIQ API",
    description="Football match outcome predictions from a Random Forest "
                "trained on 130k historical matches.",
    version="1.0.0",
    lifespan=lifespan,
)

# The Vite dev server proxies /api, but allow direct browser calls too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:4173", "http://127.0.0.1:4173"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Wire the dependency into the route modules, then mount them.
predictions.get_predictor = get_predictor
stats.get_predictor = get_predictor
app.include_router(predictions.router, prefix="/api")
app.include_router(stats.router, prefix="/api")


# ---------------------------------------------------------------------------
# Error handling: turn our domain exceptions into clear JSON, so the frontend
# can show a useful message instead of a generic 500.
# ---------------------------------------------------------------------------
@app.exception_handler(TeamNotFound)
async def team_not_found_handler(request: Request, exc: TeamNotFound):
    return JSONResponse(
        status_code=404,
        content={"error": "team_not_found", "message": str(exc),
                 "team": str(exc.name), "suggestions": exc.suggestions},
    )


@app.exception_handler(InsufficientHistory)
async def insufficient_history_handler(request: Request, exc: InsufficientHistory):
    return JSONResponse(
        status_code=422,
        content={"error": "insufficient_history", "message": str(exc)},
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"error": "bad_request", "message": str(exc)},
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    return JSONResponse(
        status_code=503,
        content={"error": "model_unavailable", "message": str(exc)},
    )


@app.get("/api/health")
def health():
    """Health check: is the API up, and did the model load?"""
    ok = _state["predictor"] is not None
    body = {
        "status": "ok" if ok else "degraded",
        "model_loaded": ok,
        "loaded_at": _state["loaded_at"],
        "load_seconds": _state["load_seconds"],
    }
    if ok:
        p = _state["predictor"]
        body["teams"] = len(p._index)
        body["trained_at"] = p.trained_at
        body["accuracy"] = p.metrics.get("accuracy")
    else:
        body["message"] = _state["error"] or "model not loaded"
    return JSONResponse(status_code=200 if ok else 503, content=body)


@app.get("/")
def root():
    return {
        "name": "MatchIQ API",
        "docs": "/docs",
        "endpoints": [
            "/api/health", "/api/leagues", "/api/teams?league=E0",
            "/api/predict?home=Arsenal&away=Chelsea",
            "/api/team/{team_name}/form",
            "/api/head-to-head?team1=Arsenal&team2=Chelsea",
            "/api/matches?date=2025-05-25",
            "/api/model/accuracy",
        ],
    }
