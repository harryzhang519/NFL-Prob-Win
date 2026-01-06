import nflreadpy as nfl
import polars as pl
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

SEASONS = list(range(2015, 2025))
X_COLS = ["epa_diff", "success_diff", "elo_diff", "is_home"]

def _build_weekly_stats(pbp: pl.DataFrame) -> pl.DataFrame:
    weekly_stats = (
        pbp
        .filter(pl.col("play_type").is_in(["run", "pass"]))
        .group_by(["season", "week", "posteam"])
        .agg([
            pl.mean("epa").alias("epa_per_play"),
            pl.mean("success").alias("success_rate"),
        ])
        .rename({"posteam": "team"})
        .sort(["team", "season", "week"])
        .with_columns([
            pl.col("epa_per_play")
              .shift(1)
              .rolling_mean(window_size=8)
              .over(["team", "season"])
              .alias("epa_roll"),
            pl.col("success_rate")
              .shift(1)
              .rolling_mean(window_size=8)
              .over(["team", "season"])
              .alias("success_roll"),
        ])
        .drop_nulls(["epa_roll", "success_roll"])
    )
    return weekly_stats

def _build_elo_df(schedules: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    ELO_START = 1500
    K = 20

    elo: dict[str, float] = {}
    elo_rows = []

    for row in schedules.sort(["season", "week"]).iter_rows(named=True):
        h, a = row["home_team"], row["away_team"]
        eh = elo.setdefault(h, ELO_START)
        ea = elo.setdefault(a, ELO_START)

        # pre-game Elo for that matchup week
        elo_rows.extend([
            {"season": row["season"], "week": row["week"], "team": h, "elo": eh},
            {"season": row["season"], "week": row["week"], "team": a, "elo": ea},
        ])

        exp_h = 1 / (1 + 10 ** ((ea - eh) / 400))
        act_h = 1 if row["home_score"] > row["away_score"] else 0

        elo[h] = eh + K * (act_h - exp_h)
        elo[a] = ea + K * ((1 - act_h) - (1 - exp_h))

    return pl.DataFrame(elo_rows), elo  # elo dict ends as "latest Elo"

def build_model_latest():
    schedules = nfl.load_schedules(SEASONS)
    pbp = nfl.load_pbp(SEASONS)

    weekly_stats = _build_weekly_stats(pbp)

    games = schedules.select([
        "season", "week",
        "home_team", "away_team",
        "home_score", "away_score",
    ])

    home = games.with_columns([
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opp"),
        (pl.col("home_score") > pl.col("away_score")).cast(int).alias("win"),
        pl.lit(1).alias("is_home"),
    ]).select(["season", "week", "team", "opp", "win", "is_home"])

    away = games.with_columns([
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opp"),
        (pl.col("away_score") > pl.col("home_score")).cast(int).alias("win"),
        pl.lit(0).alias("is_home"),
    ]).select(["season", "week", "team", "opp", "win", "is_home"])

    game_level = pl.concat([home, away])

    df = game_level.join(weekly_stats, on=["season", "week", "team"], how="inner")

    opp_stats = weekly_stats.rename({
        c: f"opp_{c}" for c in weekly_stats.columns if c not in ["season", "week", "team"]
    })

    df = df.join(
        opp_stats,
        left_on=["season", "week", "opp"],
        right_on=["season", "week", "team"],
        how="inner",
    )

    df = df.with_columns([
        (pl.col("epa_roll") - pl.col("opp_epa_roll")).alias("epa_diff"),
        (pl.col("success_roll") - pl.col("opp_success_roll")).alias("success_diff"),
    ])

    elo_df, elo_latest_dict = _build_elo_df(schedules)

    df = df.join(elo_df, on=["season", "week", "team"], how="inner")

    opp_elo = elo_df.rename({"elo": "opp_elo"})
    df = df.join(
        opp_elo,
        left_on=["season", "week", "opp"],
        right_on=["season", "week", "team"],
        how="inner",
    )

    df = df.with_columns((pl.col("elo") - pl.col("opp_elo")).alias("elo_diff"))

    # Train
    pdf = df.select(X_COLS + ["win"]).to_pandas()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pdf[X_COLS])
    model = LogisticRegression(max_iter=2000)
    model.fit(X_scaled, pdf["win"])

    # Build "latest rolling stats per team" (most recent row in weekly_stats)
    latest_team_stats = (
        weekly_stats
        .sort(["team", "season", "week"])
        .group_by("team")
        .agg([
            pl.last("epa_roll").alias("epa_roll"),
            pl.last("success_roll").alias("success_roll"),
        ])
    )

    # Latest Elo per team (from dict after processing all games)
    latest_elo = pl.DataFrame({
        "team": list(elo_latest_dict.keys()),
        "elo": list(elo_latest_dict.values()),
    })

    # Merge to one lookup table
    latest_features = (
        latest_team_stats
        .join(latest_elo, on="team", how="inner")
        .to_pandas()
        .set_index("team")
    )

    teams = sorted(latest_features.index.to_list())
    return model, scaler, latest_features, teams

# Build once at import
MODEL, SCALER, LATEST, TEAMS = build_model_latest()

def win_prob_latest(home_team: str, away_team: str) -> dict:
    if home_team not in LATEST.index or away_team not in LATEST.index:
        raise ValueError("Unknown team code. Use /api/teams to see valid codes.")

    h = LATEST.loc[home_team]
    a = LATEST.loc[away_team]

    epa_diff = float(h["epa_roll"] - a["epa_roll"])
    success_diff = float(h["success_roll"] - a["success_roll"])
    elo_diff = float(h["elo"] - a["elo"])
    is_home = 1.0

    x = [[epa_diff, success_diff, elo_diff, is_home]]
    x_scaled = SCALER.transform(x)

    p_home = float(MODEL.predict_proba(x_scaled)[0, 1])
    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_prob": p_home,
        "away_win_prob": 1.0 - p_home,
    }

import joblib

def save_artifacts():
    joblib.dump(MODEL, "model.joblib")
    joblib.dump(SCALER, "scaler.joblib")
    LATEST.to_csv("latest_features.csv")

save_artifacts()
