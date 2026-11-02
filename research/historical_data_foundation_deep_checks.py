"""Deeper reconciliation for Objective 1: historical NBA data foundation."""
import json
from pathlib import Path
import pandas as pd
import numpy as np

OUT=Path("outputs/research/historical_data_foundation")
OUT.mkdir(parents=True,exist_ok=True)

def norm_team(s):
    return s.astype("string").str.lower().str.replace(r"[^a-z0-9]","",regex=True)

def keyed(df,date,home,away):
    d=pd.to_datetime(df[date],errors="coerce").dt.strftime("%Y-%m-%d")
    return d+"|"+norm_team(df[home])+"|"+norm_team(df[away])

def compare(a,b,label):
    ka=keyed(a,"game_date" if "game_date" in a else "date",
             "home_team" if "home_team" in a else "home_team_name",
             "away_team" if "away_team" in a else "away_team_name")
    kb=keyed(b,"game_date" if "game_date" in b else "date",
             "home_team" if "home_team" in b else "home_team_name",
             "away_team" if "away_team" in b else "away_team_name")
    sa=set(ka.dropna()); sb=set(kb.dropna())
    return {"comparison":label,"rows_a":len(a),"rows_b":len(b),
      "unique_keys_a":len(sa),"unique_keys_b":len(sb),
      "matched_keys":len(sa&sb),"a_only":len(sa-sb),"b_only":len(sb-sa),
      "duplicate_key_rows_a":int(ka.duplicated(keep=False).sum()),
      "duplicate_key_rows_b":int(kb.duplicated(keep=False).sum())}

def main():
    game=pd.read_csv("data/game.csv",low_memory=False)
    scores=pd.read_csv("data/historical_nba_scores.csv",low_memory=False)
    nba=pd.read_csv("data/nba_games.csv",low_memory=False)
    train=pd.read_parquet("outputs/training_dataset.parquet")
    market=pd.read_csv("research/data/dunkapi_recovered_2025-11-26_to_2025-12-23.csv",low_memory=False)
    patches=[pd.read_csv(p,low_memory=False) for p in [
      "research/data/score_history_patch_2026-10-21_to_2026-10-29.csv",
      "research/data/score_history_patch_2026-10-30.csv"]]
    patch=pd.concat(patches,ignore_index=True)
    score_plus=pd.concat([scores,patch],ignore_index=True)
    score_key=keyed(score_plus,"game_date","home_team","away_team")
    patch_key=keyed(patch,"game_date","home_team","away_team")
    score_key_counts=score_key.value_counts(dropna=True)
    patch_key_counts=patch_key.value_counts(dropna=True)
    combined_patch={
      "base_score_rows":len(scores),"patch_rows":len(patch),"combined_rows":len(score_plus),
      "base_date_max":str(pd.to_datetime(scores.game_date).max().date()),
      "patch_date_min":str(pd.to_datetime(patch.game_date).min().date()),
      "patch_date_max":str(pd.to_datetime(patch.game_date).max().date()),
      "combined_date_max":str(pd.to_datetime(score_plus.game_date).max().date()),
      "duplicate_keys_in_base_plus_patch":int(score_key.duplicated(keep=False).sum()),
      "duplicate_keys_within_patch":int(patch_key.duplicated(keep=False).sum()),
      "unique_combined_games":int(score_key.nunique(dropna=True)),
    }
    # Reconcile game IDs: duplicate groups and whether duplicates disagree on key fields.
    dup=game[game.game_id.duplicated(keep=False)].copy()
    dup_groups=[]
    for gid,g in dup.groupby("game_id",dropna=False):
      dup_groups.append({"game_id":str(gid),"rows":len(g),
       "distinct_dates":int(g.game_date.nunique(dropna=True)),
       "distinct_home_away":int((g.team_name_home.astype(str)+"|"+g.team_name_away.astype(str)).nunique()),
       "distinct_scores":int((g.pts_home.astype(str)+"|"+g.pts_away.astype(str)).nunique())})
    duplicate_game_id_summary={
      "duplicate_id_rows":len(dup),"duplicate_id_groups":int(dup.game_id.nunique()),
      "groups_with_multiple_dates":sum(x["distinct_dates"]>1 for x in dup_groups),
      "groups_with_multiple_matchups":sum(x["distinct_home_away"]>1 for x in dup_groups),
      "groups_with_multiple_score_pairs":sum(x["distinct_scores"]>1 for x in dup_groups),
      "sample_groups":dup_groups[:15],
    }
    # Box-score availability by season and possession formula.
    poss_cols=["fga_home","fta_home","oreb_home","tov_home","fga_away","fta_away","oreb_away","tov_away"]
    for c in poss_cols: game[c]=pd.to_numeric(game[c],errors="coerce")
    game["both_team_possessions_available"]=game[poss_cols].notna().all(axis=1)
    game["home_poss_est"]=game.fga_home+.44*game.fta_home-game.oreb_home+game.tov_home
    game["away_poss_est"]=game.fga_away+.44*game.fta_away-game.oreb_away+game.tov_away
    game["both_poss_est_valid"]=game.both_team_possessions_available & game.home_poss_est.between(50,150) & game.away_poss_est.between(50,150)
    season_col="season_id" if "season_id" in game else "season"
    season=game.groupby(season_col,dropna=False).agg(
      game_rows=("game_id","size"),unique_game_ids=("game_id","nunique"),
      possession_complete_rows=("both_team_possessions_available","sum"),
      plausible_both_possessions=("both_poss_est_valid","sum"),
      first_date=("game_date","min"),last_date=("game_date","max")
    ).reset_index()
    season["possession_coverage_pct"]=(100*season.possession_complete_rows/season.game_rows).round(2)
    season.to_csv(OUT/"game_csv_season_coverage.csv",index=False)
    # Critical stat missingness.
    critical=["fga_home","fta_home","oreb_home","tov_home","fga_away","fta_away","oreb_away","tov_away",
      "pts_home","pts_away","fgm_home","fgm_away","fg3m_home","fg3m_away","ast_home","ast_away","reb_home","reb_away"]
    missing=[]
    for c in critical:
      if c in game:
       missing.append({"column":c,"missing_rows":int(game[c].isna().sum()),
         "coverage_pct":round(float(game[c].notna().mean()*100),2)})
    missing_df=pd.DataFrame(missing).sort_values(["coverage_pct","column"])
    missing_df.to_csv(OUT/"game_csv_critical_field_coverage.csv",index=False)
    # Cross-source joins and score disagreement on shared game keys.
    comparisons=[
      compare(scores,game.rename(columns={"game_date":"date","team_name_home":"home_team_name","team_name_away":"away_team_name"}),"historical score file vs game.csv"),
      compare(scores,nba,"historical score file vs nba_games.csv"),
      compare(nba,train,"nba_games.csv vs training_dataset.parquet"),
      compare(market,scores,"DunkAPI recovered archive vs historical scores"),
    ]
    def score_map(df,date,h,a,hscore,ascore):
      x=df.copy(); x["_key"]=keyed(x,date,h,a)
      x["_home"]=pd.to_numeric(x[hscore],errors="coerce"); x["_away"]=pd.to_numeric(x[ascore],errors="coerce")
      return x.dropna(subset=["_key"]).drop_duplicates("_key").set_index("_key")[["_home","_away"]]
    # Compare historical score pairs on matched score keys.
    score_source=score_map(scores,"game_date","home_team","away_team","home_score","away_score")
    game_source=score_map(game.rename(columns={"team_name_home":"home_team_name","team_name_away":"away_team_name"}),"game_date","home_team_name","away_team_name","pts_home","pts_away")
    common=score_source.index.intersection(game_source.index)
    score_mismatches=int(((score_source.loc[common].values!=game_source.loc[common].values).any(axis=1)).sum())
    # Training data's date/team columns and score fields.
    tr=train.rename(columns={"date":"game_date","home_team_name":"home_team","away_team_name":"away_team"})
    tr_map=score_map(tr,"game_date","home_team","away_team","home_score","away_score")
    nba_map=score_map(nba.rename(columns={"date":"game_date","home_team_name":"home_team","away_team_name":"away_team"}),"game_date","home_team","away_team","home_score","away_score")
    tr_common=tr_map.index.intersection(nba_map.index)
    train_nba_score_mismatch=int(((tr_map.loc[tr_common].values!=nba_map.loc[tr_common].values).any(axis=1)).sum())
    # Market archive rows can be checked against score history.
    market_map=score_map(market,"game_date","home_team","away_team","actual_home_points","actual_away_points")
    market_common=market_map.index.intersection(score_source.index)
    market_score_mismatch=int(((market_map.loc[market_common].values!=score_source.loc[market_common].values).any(axis=1)).sum())
    result={
      "score_history_plus_patch":combined_patch,
      "game_csv_duplicate_game_ids":duplicate_game_id_summary,
      "game_csv_possession_coverage":{
        "rows":len(game),"both_team_formula_inputs":int(game.both_team_possessions_available.sum()),
        "both_team_plausible_estimates":int(game.both_poss_est_valid.sum()),
        "all_history_coverage_pct":round(float(game.both_team_possessions_available.mean()*100),2),
        "by_season_csv":"game_csv_season_coverage.csv",
      },
      "cross_source_comparisons":comparisons,
      "score_consistency":{
        "historical_scores_vs_game_csv_shared_keys":len(common),
        "score_pair_mismatches":score_mismatches,
        "training_dataset_vs_nba_games_shared_keys":len(tr_common),
        "training_score_pair_mismatches":train_nba_score_mismatch,
        "market_archive_vs_historical_scores_shared_keys":len(market_common),
        "market_score_pair_mismatches":market_score_mismatch,
      },
      "critical_field_coverage_csv":"game_csv_critical_field_coverage.csv",
    }
    (OUT/"historical_data_foundation_deep_checks.json").write_text(json.dumps(result,indent=2,default=str))
    lines=["# Historical data foundation — reconciliation and coverage","",
      "## Score-history extension",json.dumps(combined_patch,indent=2),"",
      "## Duplicate game IDs in game.csv",json.dumps(duplicate_game_id_summary,indent=2),"",
      "## Possession availability",json.dumps(result["game_csv_possession_coverage"],indent=2),"",
      "## Cross-source matchability",json.dumps(comparisons,indent=2),"",
      "## Score consistency",json.dumps(result["score_consistency"],indent=2),"",
      "## Critical field coverage","See game_csv_critical_field_coverage.csv.",
      "## Possession coverage by season","See game_csv_season_coverage.csv.",""]
    (OUT/"historical_data_foundation_deep_checks.md").write_text("\n".join(lines))
    print(json.dumps(result,indent=2,default=str))

if __name__=="__main__": main()
