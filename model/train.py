"""
Step 4 - Train and evaluate the model.

WHY A TIME-BASED SPLIT (and not the usual random one)
-----------------------------------------------------
The normal way to test a model is to shuffle your rows and hold out 20% at
random. That would be WRONG here. Our job is to predict matches that have not
happened yet, so the test set has to sit strictly in the future relative to
the training set. If we shuffled, the model would train on April 2024 games
and be tested on January 2024 games -- it would already "know" how the season
turned out, and would score far better in testing than in real life.

So: train on seasons 2000-2022, test on seasons 2023-2025. Nothing in the
test period influences the model in any way.

WHY A RANDOM FOREST
-------------------
A Random Forest is a few hundred decision trees, each trained on a random
subsample of the data and of the features, whose votes get averaged. It is a
good first choice here because it handles mixed feature scales without any
normalisation, captures interactions between features on its own (e.g. "good
home form matters more when the Elo gap is small"), is hard to overfit badly
once you set a sensible minimum leaf size, and -- importantly for this app --
gives calibrated-ish class probabilities and a readable feature-importance
ranking, so we can explain WHY it predicted something.

WHAT COUNTS AS A GOOD SCORE
---------------------------
Three-way football prediction is genuinely hard. Roughly 45% of matches are
home wins, so "always predict home win" is the baseline to beat. Serious
models land around 52-56%. Anything dramatically higher on this feature set
means a leak, not a breakthrough.
"""

import json
import sys
import time
from pathlib import Path

# Allow `python model/train.py` and `python train.py` alike.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, log_loss)

from features import (DIVISION_NAMES, FEATURE_COLUMNS, MODEL_DIR,
                      build_feature_table, load_matches)

# Seasons 2000-2022 train, 2023 onwards test (season 2023 = Aug 2023-May 2024).
TEST_SEASON_START = 2023

# Human-readable class labels. The dataset encodes the result as H/D/A.
LABELS = ["H", "D", "A"]
LABEL_NAMES = {"H": "Home Win", "D": "Draw", "A": "Away Win"}

MODEL_PATH = MODEL_DIR / "model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
STATE_PATH = MODEL_DIR / "state.pkl"
BACKTEST_PATH = MODEL_DIR / "data" / "backtest.csv"

# How much match history to keep in the cached state for the UI's history views.
MATCH_LOG_KEEP = 40
H2H_LOG_KEEP = 20


def main():
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Build the feature table (single leak-free pass over history)
    # ------------------------------------------------------------------
    print("[1/7] Loading raw matches...")
    matches = load_matches(verbose=True)
    print("      {:,} matches, {} divisions, {} -> {}".format(
        len(matches), matches.Division.nunique(),
        matches.MatchDate.min().date(), matches.MatchDate.max().date()))

    print("[2/7] Building features (oldest -> newest, no future data)...")
    # keep_log=True also records recent match history, which the API serves on
    # the team-form and head-to-head pages.
    data, state = build_feature_table(matches, keep_log=True)

    # ------------------------------------------------------------------
    # 2. Time-based split
    # ------------------------------------------------------------------
    print("[3/7] Splitting by season (train <= {}, test >= {})...".format(
        TEST_SEASON_START - 1, TEST_SEASON_START))
    train = data[data.Season < TEST_SEASON_START]
    test = data[data.Season >= TEST_SEASON_START]
    print("      train: {:,} matches (seasons {}-{})".format(
        len(train), int(train.Season.min()), int(train.Season.max())))
    print("      test:  {:,} matches (seasons {}-{})".format(
        len(test), int(test.Season.min()), int(test.Season.max())))

    X_train_raw = train[FEATURE_COLUMNS]
    X_test_raw = test[FEATURE_COLUMNS]
    y_train = train.FTResult.values
    y_test = test.FTResult.values

    # ------------------------------------------------------------------
    # 3. Missing values
    # ------------------------------------------------------------------
    # A newly promoted team has no head-to-head record; a team's first ever
    # match has no form. We fill those with the column MEDIAN (typical value),
    # except for the counting features where 0 is the honest answer.
    #
    # Crucially the medians are computed on the TRAINING SET ONLY. Using the
    # full dataset's medians would let test-set information seep into the
    # training pipeline -- a subtle but real form of leakage.
    print("[4/7] Imputing missing values (medians from TRAIN only)...")
    zero_fill = {"h2h_played", "home_goal_diff", "away_goal_diff", "goal_diff_diff"}
    fill_values = {}
    for col in FEATURE_COLUMNS:
        fill_values[col] = 0.0 if col in zero_fill else float(X_train_raw[col].median())
    X_train = X_train_raw.fillna(fill_values)
    X_test = X_test_raw.fillna(fill_values)
    missing_pct = (X_train_raw.isna().mean() * 100).round(2)
    top_missing = missing_pct[missing_pct > 0].sort_values(ascending=False)
    print("      columns with gaps: {}".format(
        ", ".join("{} {}%".format(k, v) for k, v in top_missing.head(6).items()) or "none"))

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    print("[5/7] Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=400,       # more trees = steadier probabilities (diminishing returns after ~300)
        min_samples_leaf=25,    # the main overfitting guard: no leaf may describe fewer than 25 matches
        max_features="sqrt",    # each split considers sqrt(n_features); decorrelates the trees
        class_weight=None,      # keep the natural 45/27/28 balance so probabilities stay honest
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    print("[6/7] Evaluating on unseen future seasons...")
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, pred)
    train_accuracy = accuracy_score(y_train, model.predict(X_train))

    # Baseline: bet on the home team every single time.
    baseline_pred = np.full_like(y_test, "H", dtype=object)
    baseline = accuracy_score(y_test, baseline_pred)

    cm = confusion_matrix(y_test, pred, labels=LABELS)
    report = classification_report(y_test, pred, labels=LABELS,
                                   target_names=[LABEL_NAMES[c] for c in LABELS],
                                   output_dict=True, zero_division=0)

    print("\n" + "=" * 68)
    print("EVALUATION  (test = seasons {}+, {:,} matches never seen in training)".format(
        TEST_SEASON_START, len(test)))
    print("=" * 68)
    print("Overall accuracy      : {:.2%}".format(accuracy))
    print("Baseline (always home): {:.2%}".format(baseline))
    print("Improvement           : {:+.2f} percentage points".format((accuracy - baseline) * 100))
    print("Training accuracy     : {:.2%}  (gap to test = overfitting check)".format(train_accuracy))
    print("Log loss              : {:.4f}  (lower = better-calibrated probabilities)".format(
        log_loss(y_test, proba, labels=list(model.classes_))))

    print("\nPER-CLASS ACCURACY (recall: of all real X, how many did we catch?)")
    per_class = {}
    for i, cls in enumerate(LABELS):
        mask = y_test == cls
        acc_c = float((pred[mask] == cls).mean()) if mask.sum() else float("nan")
        prec = report[LABEL_NAMES[cls]]["precision"]
        per_class[cls] = {"name": LABEL_NAMES[cls], "recall": acc_c,
                          "precision": prec, "support": int(mask.sum())}
        print("  {:<9} recall {:>6.2%}   precision {:>6.2%}   ({:,} matches)".format(
            LABEL_NAMES[cls], acc_c, prec, int(mask.sum())))

    print("\nCONFUSION MATRIX (rows = actual, cols = predicted)")
    print("            " + "".join("{:>12}".format(LABEL_NAMES[c]) for c in LABELS))
    for i, cls in enumerate(LABELS):
        print("{:<12}".format(LABEL_NAMES[cls]) + "".join("{:>12,}".format(v) for v in cm[i]))

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, pred, labels=LABELS,
                                target_names=[LABEL_NAMES[c] for c in LABELS],
                                zero_division=0))

    # Per-league accuracy, shown on the Model Info page.
    print("PER-LEAGUE ACCURACY (test period)")
    per_league = {}
    test_div = test.Division.values
    for div in sorted(set(test_div)):
        m = test_div == div
        if m.sum() < 50:
            continue
        acc_d = float((pred[m] == y_test[m]).mean())
        base_d = float((y_test[m] == "H").mean())
        per_league[div] = {"name": DIVISION_NAMES.get(div, div), "accuracy": acc_d,
                           "baseline": base_d, "matches": int(m.sum())}
        print("  {:<24} {:>7.2%}  (baseline {:>6.2%}, {:,} matches)".format(
            DIVISION_NAMES.get(div, div), acc_d, base_d, int(m.sum())))

    # Feature importance: how much each feature reduced impurity across the
    # forest. It says which features the model LEANED ON, not what causes what.
    importances = sorted(zip(FEATURE_COLUMNS, model.feature_importances_),
                         key=lambda kv: -kv[1])
    print("\nFEATURE IMPORTANCE (top 15)")
    for name, imp in importances[:15]:
        bar = "#" * int(round(imp * 200))
        print("  {:<24} {:.4f}  {}".format(name, imp, bar))

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    print("\n[7/7] Saving model, metrics and cached team state...")
    # compress=3 takes the forest from ~155MB to ~35MB on disk. It costs a
    # couple of seconds at load, which we pay once at API startup.
    joblib.dump({
        "model": model,
        "feature_columns": FEATURE_COLUMNS,   # so predict.py builds vectors in the same order
        "fill_values": fill_values,           # so predict.py imputes exactly as training did
        "classes": list(model.classes_),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, MODEL_PATH, compress=3)

    metrics = {
        "accuracy": accuracy,
        "baseline_accuracy": baseline,
        "improvement": accuracy - baseline,
        "train_accuracy": train_accuracy,
        "log_loss": float(log_loss(y_test, proba, labels=list(model.classes_))),
        "test_matches": int(len(test)),
        "train_matches": int(len(train)),
        "train_seasons": [int(train.Season.min()), int(train.Season.max())],
        "test_seasons": [int(test.Season.min()), int(test.Season.max())],
        "labels": LABELS,
        "label_names": LABEL_NAMES,
        "per_class": per_class,
        "per_league": per_league,
        "confusion_matrix": cm.tolist(),
        "feature_importance": [{"feature": n, "importance": float(i)} for n, i in importances],
        "class_distribution": {c: float((y_test == c).mean()) for c in LABELS},
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # Save a per-match backtest over the TEST period: the model's probabilities
    # for matches it never trained on, next to what actually happened. The UI's
    # date browser reads this, so users can page through real matchdays and see
    # the prediction and the result side by side.
    backtest = pd.DataFrame({
        "date": test.MatchDate.dt.strftime("%Y-%m-%d").values,
        "division": test.Division.values,
        "home_team": test.HomeTeam.values,
        "away_team": test.AwayTeam.values,
        "predicted": pred,
        "actual": y_test,
    })
    for i, cls in enumerate(model.classes_):
        backtest["prob_" + cls] = proba[:, i].round(4)
    backtest["correct"] = (backtest.predicted == backtest.actual)
    backtest["home_elo"] = test.home_elo.round(1).values
    backtest["away_elo"] = test.away_elo.round(1).values
    backtest.to_csv(BACKTEST_PATH, index=False)

    # Trim the history logs before caching, so the pickle stays small.
    for team in list(state.match_log):
        state.match_log[team] = state.match_log[team][-MATCH_LOG_KEEP:]
    for pair in list(state.h2h_log):
        state.h2h_log[pair] = state.h2h_log[pair][-H2H_LOG_KEEP:]
    joblib.dump(state, STATE_PATH, compress=3)

    print("      -> {}".format(MODEL_PATH))
    print("      -> {}".format(METRICS_PATH))
    print("      -> {}".format(STATE_PATH))
    print("      -> {}".format(BACKTEST_PATH))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    print("Model accuracy: {:.1%}".format(accuracy))
    print("Baseline (always home win): {:.1%}".format(baseline))
    print("Model beats baseline by: {:.1f} percentage points".format((accuracy - baseline) * 100))
    print("Top 5 features: {}".format(", ".join(n for n, _ in importances[:5])))
    print("Done in {:.0f}s".format(time.time() - t0))

    if accuracy < 0.45:
        print("\nWARNING: accuracy below 45% -- check feature engineering before continuing.")


if __name__ == "__main__":
    main()
