"""
Step 2 - Data exploration.

Goal: understand the raw Kaggle dataset before we build any features, and
critically: verify that the pre-computed Elo / Form columns are PRE-match
values. If they were post-match, using them as features would leak the
result we are trying to predict and the model's accuracy would be a lie.
"""
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

RAW = "model/data/Matches.csv"
ELO = "model/data/EloRatings.csv"

# Division codes used by football-data.co.uk (the dataset's upstream source)
LEAGUE_NAMES = {"E0": "Premier League", "SP1": "La Liga", "I1": "Serie A",
                "D1": "Bundesliga", "F1": "Ligue 1"}


def main():
    df = pd.read_csv(RAW, parse_dates=["MatchDate"], low_memory=False)
    print("=" * 70)
    print("SHAPE:", df.shape[0], "rows x", df.shape[1], "columns")
    print("DATE RANGE:", df.MatchDate.min().date(), "->", df.MatchDate.max().date())
    print("DIVISIONS:", df.Division.nunique())

    print("\n" + "=" * 70)
    print("COLUMNS WE CARE ABOUT (missing % across whole dataset)")
    core = ["Division", "MatchDate", "HomeTeam", "AwayTeam", "FTHome", "FTAway",
            "FTResult", "HomeElo", "AwayElo", "Form5Home", "Form5Away",
            "HomeShots", "AwayShots", "OddHome", "OddDraw", "OddAway"]
    miss = (df[core].isna().mean() * 100).round(2)
    print(miss.to_string())

    print("\n" + "=" * 70)
    print("TARGET DISTRIBUTION (FTResult) - full dataset")
    print(df.FTResult.value_counts(normalize=True).mul(100).round(2).to_string())

    for code in ["E0", "SP1"]:
        sub = df[df.Division == code]
        print(f"\n--- {LEAGUE_NAMES[code]} ({code}): {len(sub)} matches ---")
        print(sub.FTResult.value_counts(normalize=True).mul(100).round(2).to_string())
        print(sub[["MatchDate", "HomeTeam", "AwayTeam", "FTHome", "FTAway",
                   "FTResult", "HomeElo", "AwayElo"]].tail(3).to_string(index=False))

    # ------------------------------------------------------------------
    # LEAKAGE CHECK 1: is HomeElo/AwayElo a PRE-match rating?
    # EloRatings.csv holds a dated snapshot of every club's rating. A snapshot
    # dated D is the rating going INTO day D. If Matches.HomeElo equals that
    # snapshot, the match file's Elo is pre-match and safe to use.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("LEAKAGE CHECK 1 - is Elo pre-match?")
    elo = pd.read_csv(ELO, parse_dates=["date"])
    elo.columns = [c.strip().strip('"') for c in elo.columns]
    snap = elo.rename(columns={"date": "MatchDate", "club": "HomeTeam", "elo": "snap_elo"})
    merged = df.merge(snap[["MatchDate", "HomeTeam", "snap_elo"]],
                      on=["MatchDate", "HomeTeam"], how="inner")
    print(f"matched {len(merged)} rows against dated Elo snapshots")
    if len(merged):
        diff = (merged.HomeElo - merged.snap_elo).abs()
        print(f"  mean |Matches.HomeElo - snapshot_elo| = {diff.mean():.4f}")
        print(f"  share within 0.01 = {(diff < 0.01).mean():.1%}")
        print("  -> ~0 difference means Matches Elo IS the pre-match snapshot (SAFE)")

    # ------------------------------------------------------------------
    # LEAKAGE CHECK 2: does the current match's result show up in its own
    # Form5 value? Form5 should be points from the PREVIOUS 5 games. If it
    # already contains this match, home wins would have systematically
    # higher Form5 in a way that perfectly separates the classes.
    # A blunt but effective test: correlation of Form5Home with this match's
    # home points. A leak looks like corr > 0.5; honest form is ~0.1.
    # ------------------------------------------------------------------
    print("\nLEAKAGE CHECK 2 - does Form5 contain the current result?")
    d = df.dropna(subset=["FTResult", "Form5Home"]).copy()
    home_pts = d.FTResult.map({"H": 3, "D": 1, "A": 0})
    print(f"  corr(Form5Home, this match home points) = {d.Form5Home.corr(home_pts):.3f}")
    print("  -> a value near 0.1-0.2 is honest form; >0.5 would mean leakage")

    # Sanity: Elo difference should predict outcome but not perfectly
    d2 = df.dropna(subset=["HomeElo", "AwayElo", "FTResult"])
    print(f"\n  corr(elo_diff, home points) = "
          f"{(d2.HomeElo - d2.AwayElo).corr(d2.FTResult.map({'H': 3, 'D': 1, 'A': 0})):.3f}")

    print("\n" + "=" * 70)
    print("LEAGUE COVERAGE")
    print("divisions present:", sorted(df.Division.unique()))
    print("-> domestic leagues only; this app models the big five (E0/SP1/D1/I1/F1)")


if __name__ == "__main__":
    main()
