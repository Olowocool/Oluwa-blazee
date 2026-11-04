#!/usr/bin/env python3
"""Stage C: point-in-time player skill + rotation/RAPM impact features.

Consumes the Stage A player-history profiles and the existing V4 Phase 2
pregame rotation/injury/RAPM outputs. It deliberately ASOF-joins each projected
rotation player to their latest profile strictly before the target game date.
It never uses Stage B date-cohort z-scores, target-game participation, or
target-game minutes as model inputs.

Research only. V5 and production are not modified.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

PROFILE = Path("outputs/research/player_intelligence_profiles/player_game_profiles.parquet")
PHASE2 = Path("outputs/research/v4_phase2")
MINUTES = PHASE2 / "v4_phase2_projected_minutes.csv"
GAMES = PHASE2 / "v4_phase2_game_features.csv"
OUT = Path("outputs/research/player_intelligence_stage_c")
OUT.mkdir(parents=True, exist_ok=True)

# Profile-derived features are descriptive skill/role summaries. RAPM and
# projected-minute deltas are the causal-ish lineup impact proxy.
PROFILE_FEATURES = [
    "expected_pts", "expected_ast", "expected_tov", "expected_oreb",
    "expected_dreb", "expected_stl", "expected_blk",
    "minute_weighted_ts", "minute_weighted_usage",
    "minute_weighted_three_pa_share", "minute_weighted_three_pct",
    "minute_weighted_free_throw_rate", "profiled_minutes_share",
]
RAPM_FEATURES = [
    "home_off_delta", "home_def_delta", "away_off_delta", "away_def_delta",
]
MODEL_FEATURES = [
    "home_profile_expected_pts", "away_profile_expected_pts",
    "home_profile_expected_ast", "away_profile_expected_ast",
    "home_profile_expected_tov", "away_profile_expected_tov",
    "home_profile_expected_oreb", "away_profile_expected_oreb",
    "home_profile_expected_dreb", "away_profile_expected_dreb",
    "home_profile_expected_stl", "away_profile_expected_stl",
    "home_profile_expected_blk", "away_profile_expected_blk",
    "home_profile_minute_weighted_ts", "away_profile_minute_weighted_ts",
    "home_profile_minute_weighted_usage", "away_profile_minute_weighted_usage",
    "home_profile_minute_weighted_three_pa_share", "away_profile_minute_weighted_three_pa_share",
    "home_profile_minute_weighted_three_pct", "away_profile_minute_weighted_three_pct",
    "home_profile_minute_weighted_free_throw_rate", "away_profile_minute_weighted_free_throw_rate",
    "home_profile_profiled_minutes_share", "away_profile_profiled_minutes_share",
]


def sql_literal(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def build_player_features() -> tuple[pd.DataFrame, dict]:
    for p in (PROFILE, MINUTES, GAMES):
        if not p.exists():
            raise FileNotFoundError(f"Required Stage C input missing: {p}")
    con = duckdb.connect()
    con.execute(f"CREATE VIEW profiles AS SELECT * FROM read_parquet({sql_literal(PROFILE)})")
    con.execute(f"""
        CREATE VIEW candidates AS
        SELECT * REPLACE (CAST(game_id AS VARCHAR) AS game_id),
               CAST(game_date AS DATE) AS target_date,
               CAST(expected_minutes AS DOUBLE) AS exp_min
        FROM read_csv_auto({sql_literal(MINUTES)}, header=true, all_varchar=true)
    """)
    # Critical PIT join: select the latest player-game profile dated strictly
    # before the target game. Candidate rows come from projected rotations,
    # not the target game's actual player logs.
    con.execute("""
        CREATE TABLE candidate_profiles AS
        SELECT c.season, c.game_id, c.target_date AS game_date, c.side,
               c.player_id, c.exp_min, c.status, c.availability_probability,
               p.game_date AS profile_source_date,
               p.profile_through_date, p.prior_game_days,
               p.l20_min, p.l20_pts, p.l20_ast, p.l20_tov,
               p.l20_oreb, p.l20_reb, p.l20_stl, p.l20_blk,
               p.l20_fga, p.l20_fg3a, p.l20_fg3m, p.l20_fta,
               p.l20_ts_pct, p.l20_usg_pct
        FROM candidates c
        ASOF LEFT JOIN profiles p
          ON c.player_id = p.player_id
         AND c.target_date > CAST(p.game_date AS DATE)
    """)
    # The ASOF key is strictly earlier than target_date. Also assert the
    # profile's own source-date and history-through-date are both pregame.
    bad = con.execute("""
        SELECT COUNT(*) FROM candidate_profiles
        WHERE (profile_source_date IS NOT NULL AND profile_source_date >= game_date)
           OR (profile_through_date IS NOT NULL AND profile_through_date >= game_date)
    """).fetchone()[0]
    if bad:
        raise RuntimeError(f"Point-in-time profile leakage rows: {bad}")

    con.execute("""
        CREATE TABLE player_components AS
        SELECT *,
            CASE WHEN l20_min >= 60 THEN 36.0*l20_pts/l20_min END AS pts_per36,
            CASE WHEN l20_min >= 60 THEN 36.0*l20_ast/l20_min END AS ast_per36,
            CASE WHEN l20_min >= 60 THEN 36.0*l20_tov/l20_min END AS tov_per36,
            CASE WHEN l20_min >= 60 THEN 36.0*l20_oreb/l20_min END AS oreb_per36,
            CASE WHEN l20_min >= 60 THEN 36.0*(l20_reb-l20_oreb)/l20_min END AS dreb_per36,
            CASE WHEN l20_min >= 60 THEN 36.0*l20_stl/l20_min END AS stl_per36,
            CASE WHEN l20_min >= 60 THEN 36.0*l20_blk/l20_min END AS blk_per36,
            CASE WHEN l20_fga >= 20 THEN l20_ts_pct END AS ts_eligible,
            CASE WHEN l20_fga >= 20 THEN l20_usg_pct END AS usage_eligible,
            CASE WHEN l20_fga >= 20 THEN l20_fg3a/l20_fga END AS three_pa_share,
            CASE WHEN l20_fg3a >= 10 THEN l20_fg3m/l20_fg3a END AS three_pct,
            CASE WHEN l20_fga >= 20 THEN l20_fta/l20_fga END AS free_throw_rate,
            CASE WHEN l20_min >= 60 AND prior_game_days >= 5 THEN 1 ELSE 0 END AS profile_eligible
        FROM candidate_profiles
    """)
    con.execute("""
        CREATE TABLE team_profiles AS
        SELECT season, game_id, game_date, side,
            SUM(CASE WHEN profile_eligible=1 THEN exp_min*pts_per36/36.0 ELSE 0 END) AS expected_pts,
            SUM(CASE WHEN profile_eligible=1 THEN exp_min*ast_per36/36.0 ELSE 0 END) AS expected_ast,
            SUM(CASE WHEN profile_eligible=1 THEN exp_min*tov_per36/36.0 ELSE 0 END) AS expected_tov,
            SUM(CASE WHEN profile_eligible=1 THEN exp_min*oreb_per36/36.0 ELSE 0 END) AS expected_oreb,
            SUM(CASE WHEN profile_eligible=1 THEN exp_min*dreb_per36/36.0 ELSE 0 END) AS expected_dreb,
            SUM(CASE WHEN profile_eligible=1 THEN exp_min*stl_per36/36.0 ELSE 0 END) AS expected_stl,
            SUM(CASE WHEN profile_eligible=1 THEN exp_min*blk_per36/36.0 ELSE 0 END) AS expected_blk,
            SUM(CASE WHEN ts_eligible IS NOT NULL THEN exp_min*ts_eligible ELSE 0 END)
              / NULLIF(SUM(CASE WHEN ts_eligible IS NOT NULL THEN exp_min ELSE 0 END),0) AS minute_weighted_ts,
            SUM(CASE WHEN usage_eligible IS NOT NULL THEN exp_min*usage_eligible ELSE 0 END)
              / NULLIF(SUM(CASE WHEN usage_eligible IS NOT NULL THEN exp_min ELSE 0 END),0) AS minute_weighted_usage,
            SUM(CASE WHEN three_pa_share IS NOT NULL THEN exp_min*three_pa_share ELSE 0 END)
              / NULLIF(SUM(CASE WHEN three_pa_share IS NOT NULL THEN exp_min ELSE 0 END),0) AS minute_weighted_three_pa_share,
            SUM(CASE WHEN three_pct IS NOT NULL THEN exp_min*three_pct ELSE 0 END)
              / NULLIF(SUM(CASE WHEN three_pct IS NOT NULL THEN exp_min ELSE 0 END),0) AS minute_weighted_three_pct,
            SUM(CASE WHEN free_throw_rate IS NOT NULL THEN exp_min*free_throw_rate ELSE 0 END)
              / NULLIF(SUM(CASE WHEN free_throw_rate IS NOT NULL THEN exp_min ELSE 0 END),0) AS minute_weighted_free_throw_rate,
            SUM(CASE WHEN profile_eligible=1 THEN exp_min ELSE 0 END)/240.0 AS profiled_minutes_share,
            COUNT(*) AS projected_rotation_players,
            COUNT(*) FILTER (WHERE profile_eligible=1) AS profiled_rotation_players,
            SUM(exp_min) AS projected_minutes
        FROM player_components
        GROUP BY season, game_id, game_date, side
    """)
    team = con.execute("SELECT * FROM team_profiles").fetchdf()
    con.execute(f"""
        COPY candidate_profiles TO {sql_literal(OUT / "stage_c_candidate_profiles.parquet")}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.execute(f"""
        COPY player_components TO {sql_literal(OUT / "stage_c_player_components.parquet")}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    candidate_stats = con.execute("""
        SELECT COUNT(*) AS candidate_rows,
               COUNT(DISTINCT game_id) AS games,
               COUNT(*) FILTER (WHERE profile_source_date IS NOT NULL) AS rows_with_prior_profile,
               COUNT(*) FILTER (WHERE profile_eligible=1) AS eligible_profile_rows,
               COUNT(*) FILTER (WHERE profile_source_date >= game_date) AS source_date_leakage,
               COUNT(*) FILTER (WHERE profile_through_date >= game_date) AS history_date_leakage,
               MIN(game_date) AS first_game_date, MAX(game_date) AS last_game_date
        FROM player_components
    """).fetchdf().to_dict("records")[0]
    con.close()
    return team, candidate_stats


def add_wide_features(team: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "expected_pts","expected_ast","expected_tov","expected_oreb","expected_dreb",
        "expected_stl","expected_blk","minute_weighted_ts","minute_weighted_usage",
        "minute_weighted_three_pa_share","minute_weighted_three_pct",
        "minute_weighted_free_throw_rate","profiled_minutes_share",
    ]
    pieces = []
    for side in ("home","away"):
        x = team[team.side.eq(side)].copy()
        x = x.drop(columns=["side","season","game_date"])
        x = x.rename(columns={m:f"{side}_profile_{m}" for m in metrics})
        pieces.append(x)
    wide = pieces[0].merge(pieces[1],on="game_id",how="inner",validate="one_to_one")
    out = base.merge(wide,on="game_id",how="inner",validate="one_to_one")
    return out


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], alpha: float) -> np.ndarray:
    x_train = train[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0).to_numpy(float)
    x_test = test[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0).to_numpy(float)
    y_train = (train.actual_total-train.v4_phase1_total).to_numpy(float)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(x_train,y_train)
    return test.v4_phase1_total.to_numpy(float)+model.predict(x_test)


def metrics(name: str, season: int, y: np.ndarray, pred: np.ndarray) -> dict:
    err=pred-y
    return {
        "model":name,"season":season,"games":int(len(y)),
        "mae":float(np.mean(np.abs(err))),
        "rmse":float(np.sqrt(np.mean(err**2))),
        "bias_prediction_minus_actual":float(np.mean(err)),
        "within_5_points_pct":float(np.mean(np.abs(err)<=5)*100),
        "within_10_points_pct":float(np.mean(np.abs(err)<=10)*100),
    }


def main() -> None:
    team, candidate_stats = build_player_features()
    # Preserve NBA game identifiers as strings end-to-end. Numeric inference can
    # drop leading zeroes and produce false join failures.
    base = pd.read_csv(GAMES, dtype={"game_id": "string"})
    team["game_id"] = team["game_id"].astype("string")
    base["game_id"] = base["game_id"].astype("string")
    base["game_date"] = pd.to_datetime(base.game_date).dt.normalize()
    features = add_wide_features(team,base)
    if features.empty:
        raise RuntimeError("No matched player-profile/rotation games produced")
    if features.duplicated(["game_id"]).any():
        raise RuntimeError("Duplicate game features")
    if features[MODEL_FEATURES].isna().all(axis=None):
        raise RuntimeError("All player profile model features are null")
    # Profile coverage is explicit: missing profile minutes are not silently
    # treated as measured zero skill; coverage columns let the model learn it.
    predictions=[]; metric_rows=[]
    for season in (2025,2026):
        train=features[(features.season<season)&(features.season>=2024)].dropna(subset=["actual_total","v4_phase1_total"])
        test=features[features.season.eq(season)].dropna(subset=["actual_total","v4_phase1_total"]).copy()
        if len(train)<300 or len(test)<300:
            raise RuntimeError(f"Insufficient chronological rows for {season}: train={len(train)}, test={len(test)}")
        y=test.actual_total.to_numpy(float)
        # All models use the same games and target; the baseline has no fitted features.
        baseline=test.v4_phase1_total.to_numpy(float)
        rapm=fit_predict(train,test,RAPM_FEATURES,alpha=10.0)
        profile=fit_predict(train,test,MODEL_FEATURES,alpha=25.0)
        combined=fit_predict(train,test,RAPM_FEATURES+MODEL_FEATURES,alpha=25.0)
        metric_rows.extend([
            metrics("V4 Phase 1 baseline",season,y,baseline),
            metrics("V4 Phase 1 + RAPM/injury rotation",season,y,rapm),
            metrics("V4 Phase 1 + player profile skills",season,y,profile),
            metrics("V4 Phase 1 + RAPM/injury + player profiles",season,y,combined),
        ])
        test["stage_c_profile_prediction"]=profile
        test["stage_c_rapm_prediction"]=rapm
        test["stage_c_combined_prediction"]=combined
        predictions.append(test)
    pred=pd.concat(predictions,ignore_index=True)
    pred.to_csv(OUT/"stage_c_forward_predictions.csv",index=False)
    metric_df=pd.DataFrame(metric_rows)
    metric_df.to_csv(OUT/"stage_c_metrics.csv",index=False)

    # Paired date-cluster bootstrap for combined model vs frozen V4 Phase 1
    # baseline; resample dates, not individual games.
    y=pred.actual_total.to_numpy(float)
    base=pred.v4_phase1_total.to_numpy(float)
    challenger=pred.stage_c_combined_prediction.to_numpy(float)
    delta=np.abs(base-y)-np.abs(challenger-y)  # positive favours challenger
    date_groups=pred.groupby("game_date").indices
    dates=list(date_groups)
    rng=np.random.default_rng(20261104)
    boot=[]
    for _ in range(1000):
        sampled=rng.integers(0,len(dates),size=len(dates))
        ix=np.concatenate([np.asarray(date_groups[dates[i]],dtype=int) for i in sampled])
        boot.append(float(delta[ix].mean()))
    ci=np.quantile(boot,[0.025,0.975]).tolist()
    summary={
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "stage":"C: point-in-time player skill profiles + projected rotation + RAPM/injury",
        "inputs":[str(PROFILE),str(MINUTES),str(GAMES)],
        "profile_join":"ASOF latest player profile where profile game_date < target game_date; candidates come from projected rotation",
        "profile_leakage_checks":{
            "source_date_leakage_rows":int(candidate_stats["source_date_leakage"]),
            "history_date_leakage_rows":int(candidate_stats["history_date_leakage"]),
        },
        "candidate_profile_coverage":candidate_stats,
        "test_seasons_end_year":[2025,2026],
        "model_features":{
            "RAPM":RAPM_FEATURES,
            "player_profile":MODEL_FEATURES,
            "ridge_alpha_RAPM":10.0,
            "ridge_alpha_profile_and_combined":25.0,
            "standardization":"fit on earlier-season training data only",
        },
        "forward_metrics":metric_rows,
        "combined_vs_V4_baseline_date_cluster_bootstrap":{
            "mean_MAE_improvement_points":float(delta.mean()),
            "95pct_CI":ci,
            "dates":len(dates),
            "bootstrap_replicates":1000,
        },
        "production_modified":False,
        "V5_modified":False,
        "profitability_claim":False,
        "limitations":[
            "Player profile rates are descriptive box-score skills, not causal impact.",
            "RAPM and projected minutes are the player-impact/availability layer; skill profiles add role and production context.",
            "No target-game player logs, target-game minutes, or Stage B date-cohort z-scores enter the feature join.",
            "Injury status reports are the existing V4 Phase 2 snapshots; only complete pregame report dates are included.",
            "The historical evaluation uses completed 2024-25 and 2025-26 seasons and is not a live betting ROI test.",
            "Defensive steals/blocks are activity statistics, not direct opponent shot suppression measures.",
        ],
    }
    (OUT/"stage_c_manifest.json").write_text(json.dumps(summary,indent=2,default=str),encoding="utf-8")
    print(json.dumps(summary,indent=2,default=str))


if __name__=="__main__":
    main()
