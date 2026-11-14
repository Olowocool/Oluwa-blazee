#!/usr/bin/env python3
"""ORBIT-10 Stage 4: two preregistered interaction screens; holdout sealed."""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("outputs/research/orbit10_stage4")
BASELINE = Path("outputs/research/orbit10_stage2/orbit10_baseline_predictions.csv")
STAGE2 = Path("research/orbit10_stage2_baseline.py")
SEED = 20261114
BOOTSTRAPS = 2000
MIN_TRAIN_CELL = 30
MIN_VALID_CELL = 15
P1_HOME_Z = 0.657915
P1_AWAY_Z = 0.732879
MIN_VENUE_GAMES = 5
ROLLING = 25

def load_spine():
    spec = importlib.util.spec_from_file_location("orbit10_stage2", STAGE2)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    legacy, _, _ = mod.load_legacy()
    modern, _, _ = mod.load_modern()
    spine = pd.concat([legacy, modern], ignore_index=True)
    spine = spine[spine.season_end_year.between(2001, 2026)].copy()
    spine = spine.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    if spine.game_id.duplicated().any():
        raise RuntimeError("Stage 4 source spine contains duplicate game IDs.")
    return spine

def build_features(spine):
    histories = {}
    rows = []
    for day, games in spine.groupby("game_date", sort=True):
        venue_stats = {}
        pace_values = {}
        for team, hist in histories.items():
            home = [x for x in hist if x["is_home"]][-ROLLING:]
            road = [x for x in hist if not x["is_home"]][-ROLLING:]
            if len(home) >= MIN_VENUE_GAMES:
                venue_stats[(team, "home")] = float(np.mean([x["pts"] for x in home]))
            if len(road) >= MIN_VENUE_GAMES and sum(x["opp_poss"] for x in road) > 0:
                venue_stats[(team, "road")] = float(100 * sum(x["opp_pts"] for x in road) / sum(x["opp_poss"] for x in road))
            if len(hist) >= MIN_VENUE_GAMES:
                pace_values[team] = float(np.mean([x["pace"] for x in hist[-ROLLING:]]))
        home_vals = np.array([v for (t, k), v in venue_stats.items() if k == "home"], dtype=float)
        road_vals = np.array([v for (t, k), v in venue_stats.items() if k == "road"], dtype=float)
        hm = float(home_vals.mean()) if len(home_vals) else np.nan
        hs = float(home_vals.std(ddof=0)) if len(home_vals) > 1 else np.nan
        rm = float(road_vals.mean()) if len(road_vals) else np.nan
        rs = float(road_vals.std(ddof=0)) if len(road_vals) > 1 else np.nan
        ranked = sorted(pace_values.items(), key=lambda z: z[1])
        bottom = {t for t, _ in ranked[:3]} if len(ranked) >= 20 else set()
        top = {t for t, _ in ranked[-3:]} if len(ranked) >= 20 else set()
        for g in games.to_dict("records"):
            ht, at = str(g["home_team_id"]), str(g["away_team_id"])
            hppg = venue_stats.get((ht, "home"), np.nan)
            adef = venue_stats.get((at, "road"), np.nan)
            zh = (hppg - hm) / hs if np.isfinite(hppg) and np.isfinite(hm) and hs > 0 else np.nan
            za = (adef - rm) / rs if np.isfinite(adef) and np.isfinite(rm) and rs > 0 else np.nan
            hh, ah = histories.get(ht, []), histories.get(at, [])
            hr = int((day - hh[-1]["date"]).days - 1) if hh else None
            ar = int((day - ah[-1]["date"]).days - 1) if ah else None
            p5 = ((hr is not None and ar is not None) and
                  ((hr <= 1 and ar >= 3) or (ar <= 1 and hr >= 3)))
            p7 = ((ht in top and at in bottom) or (at in top and ht in bottom))
            rows.append({
                "game_id": g["game_id"], "game_date": day,
                "home_ppg_z": zh, "away_road_def_z": za,
                "home_rest_days": hr, "away_rest_days": ar,
                "p5_rest_disadvantage": bool(p5),
                "p7_top3_vs_bottom3_pace": bool(p7),
                "ranked_teams_for_pace": len(ranked),
            })
        # Same-day results enter history only after all same-day features are made.
        for g in games.to_dict("records"):
            ht, at = str(g["home_team_id"]), str(g["away_team_id"])
            hp, ap = float(g["poss_home"]), float(g["poss_away"])
            pace = (hp + ap) / 2.0
            histories.setdefault(ht, []).append({
                "date": day, "pts": float(g["pts_home"]), "opp_pts": float(g["pts_away"]),
                "opp_poss": ap, "pace": pace, "is_home": True})
            histories.setdefault(at, []).append({
                "date": day, "pts": float(g["pts_away"]), "opp_pts": float(g["pts_home"]),
                "opp_poss": hp, "pace": pace, "is_home": False})
    return pd.DataFrame(rows)

def metrics(y, pred):
    err = np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)
    return {"games": int(len(err)), "mae": round(float(np.mean(np.abs(err))), 4),
            "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
            "bias": round(float(np.mean(err)), 4)}

def cell_arrays(frame, partner):
    cells = {}
    for p1 in (0, 1):
        for p2 in (0, 1):
            vals = frame.loc[(frame.p1.astype(int) == p1) &
                             (frame[partner].astype(int) == p2), "residual"].to_numpy(float)
            cells[(p1, p2)] = vals
    return cells

def contrast(cells):
    return float(cells[(1,1)].mean() - cells[(1,0)].mean()
                 - cells[(0,1)].mean() + cells[(0,0)].mean())

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not BASELINE.is_file():
        raise FileNotFoundError("Frozen Stage 2 baseline predictions are missing.")
    baseline = pd.read_csv(BASELINE, parse_dates=["game_date"])
    # Filter before feature merge: no holdout records enter the Stage 4 analysis frame.
    baseline = baseline[baseline.split.isin(["train", "validation"])].copy()
    spine = load_spine()
    features = build_features(spine)
    df0 = baseline.merge(features, on=["game_id", "game_date"], how="left", validate="one_to_one")
    required = ["home_ppg_z", "away_road_def_z", "p5_rest_disadvantage", "p7_top3_vs_bottom3_pace"]
    coverage = {}
    for split in ("train", "validation"):
        part = df0[df0.split == split]
        eligible = part[required].notna().all(axis=1)
        coverage[split] = {"baseline_games": int(len(part)), "eligible_games": int(eligible.sum()),
                           "excluded_missing_features": int((~eligible).sum()),
                           "coverage_rate": round(float(eligible.mean()), 6) if len(part) else None}
    df = df0[df0[required].notna().all(axis=1)].copy()
    if df.empty:
        raise RuntimeError("No complete-case train/validation rows for Stage 4.")
    df["p1"] = ((df.home_ppg_z >= P1_HOME_Z) & (df.away_road_def_z >= P1_AWAY_Z)).astype(int)
    df["residual"] = df.predicted_total - df.actual_total
    train = df[df.split == "train"]
    p1_yes = train.loc[train.p1 == 1, "residual"].to_numpy(float)
    p1_no = train.loc[train.p1 == 0, "residual"].to_numpy(float)
    if min(len(p1_yes), len(p1_no)) < 100:
        raise RuntimeError(f"Pattern 1 train group minimum not met: {len(p1_yes)} / {len(p1_no)}.")
    p1_effect = float(p1_yes.mean() - p1_no.mean())
    df["p1_only_prediction"] = df.predicted_total - df.p1 * p1_effect
    rng = np.random.default_rng(SEED)
    results = []
    prediction_frames = []
    for partner, label, mechanism in [
        ("p5_rest_disadvantage", "Pattern 1 × Pattern 5: rest disadvantage",
         "Rest imbalance may modify the normalized home scoring/road defence matchup."),
        ("p7_top3_vs_bottom3_pace", "Pattern 1 × Pattern 7: pace collision",
         "A top-three versus bottom-three pace matchup may modify the normalized scoring/defence matchup.")
    ]:
        item = {"interaction": label, "partner_feature": partner, "mechanism": mechanism,
                "train_cell_counts": {}, "validation_cell_counts": {}}
        train_cells = cell_arrays(train, partner)
        valid = df[df.split == "validation"]
        valid_cells = cell_arrays(valid, partner)
        for split_name, cells, target in [
            ("train", train_cells, item["train_cell_counts"]),
            ("validation", valid_cells, item["validation_cell_counts"])
        ]:
            for key, values in cells.items():
                target[f"p1_{key[0]}_partner_{key[1]}"] = int(len(values))
        train_min = min(len(v) for v in train_cells.values())
        valid_min = min(len(v) for v in valid_cells.values())
        item["minimum_train_cell_n"] = int(train_min)
        item["minimum_validation_cell_n"] = int(valid_min)
        item["status"] = "INSUFFICIENT EVIDENCE"
        item["reason"] = "At least one 2×2 cell is below the preregistered minimum; no interaction coefficient or adjusted score is reported."
        if train_min < MIN_TRAIN_CELL or valid_min < MIN_VALID_CELL:
            results.append(item)
            continue
        gamma = contrast(train_cells)
        boot = np.empty(BOOTSTRAPS, dtype=float)
        for b in range(BOOTSTRAPS):
            sampled = {k: rng.choice(v, size=len(v), replace=True) for k, v in train_cells.items()}
            boot[b] = contrast(sampled)
        ci = [round(float(np.quantile(boot, .025)), 4), round(float(np.quantile(boot, .975)), 4)]
        valid_gamma = contrast(valid_cells)
        adjusted = df.loc[df.split == "validation", ["actual_total", "p1_only_prediction", partner]].copy()
        adjusted["interaction_prediction"] = adjusted.p1_only_prediction - (
            adjusted[partner].astype(int) * df.loc[df.split == "validation", "p1"].to_numpy() * gamma)
        base_metrics = metrics(adjusted.actual_total, adjusted.p1_only_prediction)
        interaction_metrics = metrics(adjusted.actual_total, adjusted.interaction_prediction)
        item.update({
            "status": "TESTED",
            "train_interaction_contrast": round(gamma, 4),
            "train_bootstrap_95ci": ci,
            "validation_interaction_contrast": round(valid_gamma, 4),
            "validation_p1_only_metrics": base_metrics,
            "validation_interaction_adjusted_metrics": interaction_metrics,
            "validation_mae_change_adjusted_minus_p1_only": round(interaction_metrics["mae"] - base_metrics["mae"], 4),
            "validation_rmse_change_adjusted_minus_p1_only": round(interaction_metrics["rmse"] - base_metrics["rmse"], 4),
        })
        criteria = {
            "all_train_cells_at_least_30": train_min >= MIN_TRAIN_CELL,
            "train_bootstrap_interval_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
            "all_validation_cells_at_least_15": valid_min >= MIN_VALID_CELL,
            "validation_interaction_same_sign": bool(np.sign(valid_gamma) == np.sign(gamma)),
            "validation_mae_improves": interaction_metrics["mae"] < base_metrics["mae"],
            "validation_rmse_improves": interaction_metrics["rmse"] < base_metrics["rmse"],
        }
        item["decision_criteria"] = criteria
        item["passes_all"] = all(criteria.values())
        item["decision"] = "SCREENING PASS — confirmatory study only" if item["passes_all"] else "DO NOT RETAIN — screening criteria not met"
        item["reason"] = item["decision"]
        val_rows = df[df.split == "validation"].copy()
        val_rows["interaction"] = partner
        val_rows["p1_only_prediction"] = val_rows["p1_only_prediction"]
        val_rows["interaction_prediction"] = val_rows["p1_only_prediction"] - (
            val_rows[partner].astype(int) * val_rows["p1"] * gamma)
        prediction_frames.append(val_rows[["game_id", "game_date", "split", "actual_total", "predicted_total",
                                           "p1", partner, "p1_only_prediction", "interaction_prediction", "interaction"]])
        results.append(item)
    summary = {
        "stage": "ORBIT-10 Stage 4 — preregistered interaction screen",
        "source_spine_games": int(len(spine)),
        "analysis_rows": int(len(df)),
        "coverage": coverage,
        "p1_frozen_thresholds": {"home_ppg_z": P1_HOME_Z, "away_road_def_z": P1_AWAY_Z},
        "p1_train_only_residual_effect": round(p1_effect, 6),
        "p1_train_qualifying": int(len(p1_yes)),
        "p1_train_control": int(len(p1_no)),
        "interaction_results": results,
        "protocol": {"bootstrap_replicates": BOOTSTRAPS, "seed": SEED,
                     "minimum_train_cell": MIN_TRAIN_CELL, "minimum_validation_cell": MIN_VALID_CELL,
                     "holdout": "SEALED — no holdout rows merged, scored, or used",
                     "same_day": "All games are featured before same-day results enter histories."},
        "interpretation": "A screening pass permits only a later confirmatory study; it is not causal evidence, profitability proof, combined-model approval, or deployment approval.",
    }
    (OUT / "orbit10_stage4_interaction_summary.json").write_text(json.dumps(summary, indent=2))
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(OUT / "orbit10_stage4_interaction_predictions.csv", index=False)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
