#!/usr/bin/env python3
"""Synthetic contract test for full individual projected rotations."""
from __future__ import annotations

import importlib.util
from datetime import timedelta
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).with_name("v4_phase2_rotation.py")
spec = importlib.util.spec_from_file_location("v4_phase2_rotation_under_test", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

GAME_ID = "0099999999"
TEAM_ID = "1610610001"
GAME_DATE = pd.Timestamp("2025-01-20")
PLAYER_COUNT = 22

team_hist = {}
player_hist = {}
first_seen = {}
actual_lookup = {}
rapm = {}
for i in range(PLAYER_COUNT):
    pid = str(100000 + i)
    rows = []
    for j in range(10):
        game_date = GAME_DATE - timedelta(days=2 * (10 - j))
        row = (game_date, f"hist-{i}-{j}", TEAM_ID, float(8 + (i % 7) * 4))
        rows.append(row)
    team_hist[(TEAM_ID, pid)] = rows
    player_hist[pid] = rows
    first_seen[(TEAM_ID, pid)] = rows[0][0]
    actual_lookup[(GAME_ID, pid)] = float(10 + (i % 5))
    rapm[pid] = {"o_rapm": (i - 10) * 0.1, "d_rapm": (10 - i) * 0.05}

reports = pd.DataFrame([
    {"team_id": TEAM_ID, "player_id": "100000", "status": "OUT"},
    {"team_id": TEAM_ID, "player_id": "100001", "status": "QUESTIONABLE"},
])
features, minutes = module.rotation_for_team(
    game_id=GAME_ID,
    team_id=TEAM_ID,
    game_date=GAME_DATE,
    report_rows=reports,
    opposing_reported_ids=set(),
    team_hist=team_hist,
    player_hist=player_hist,
    first_seen=first_seen,
    actual_lookup=actual_lookup,
    rapm=rapm,
    team_actual_totals={(GAME_ID, TEAM_ID): 261.0},
)

assert features is not None and minutes is not None
assert len(minutes) == PLAYER_COUNT, (len(minutes), PLAYER_COUNT)
assert set(minutes.player_id.astype(str)) == {str(100000 + i) for i in range(PLAYER_COUNT)}
assert not minutes.player_id.astype(str).eq("__ROTATION_TAIL__").any()
assert not minutes.is_rotation_tail.any()
assert minutes.rotation_rank.tolist() == list(range(1, PLAYER_COUNT + 1))
assert minutes.player_id.is_unique
assert abs(float(minutes.baseline_minutes.sum()) - 240.0) < 1e-8
assert abs(float(minutes.expected_minutes.sum()) - 240.0) < 1e-8
assert abs(float(minutes.actual_minutes_240.sum()) - 240.0) < 1e-8
assert minutes.loc[minutes.player_id.eq("100000"), "status"].iloc[0] == "OUT"
assert minutes.loc[minutes.player_id.eq("100000"), "availability_probability"].iloc[0] < 0.1
assert features["reported_out_minutes"] > 0
assert features["questionable_minutes"] > 0

print({
    "status": "PASS",
    "individual_player_rows": len(minutes),
    "rotation_ranks": [int(minutes.rotation_rank.min()), int(minutes.rotation_rank.max())],
    "aggregate_tail_rows": int(minutes.is_rotation_tail.sum()),
    "baseline_minutes": float(minutes.baseline_minutes.sum()),
    "expected_minutes": float(minutes.expected_minutes.sum()),
    "actual_minutes_240": float(minutes.actual_minutes_240.sum()),
    "out_player_probability": float(minutes.loc[minutes.player_id.eq("100000"), "availability_probability"].iloc[0]),
})
