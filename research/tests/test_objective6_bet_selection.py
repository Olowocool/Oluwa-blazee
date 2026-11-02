import pandas as pd
import pytest
from research.objective6_bet_selection import development_gate, select_holdout_bets, wilson_lower_bound

def fixture():
    rows=[]
    for day in range(20):
        date=pd.Timestamp("2026-01-01")+pd.Timedelta(days=day)
        for game in range(10):
            over=(game%2==0); p=0.72 if over else 0.28
            rows.append({"game_date":date,"commence_time_utc":date.tz_localize("UTC"),
                "home_team":"Home","away_team":"Away","entry_total":220.5,
                "entry_over_odds_decimal":1.91,"entry_under_odds_decimal":1.91,
                "actual_total":221.0 if over else 220.0,
                "evaluation_split":"development" if day<16 else "reserved_holdout",
                "residual_model_status":"walk_forward_scored" if day<16 else "frozen_holdout_scored",
                "residual_model_over_probability":p,"residual_model_under_probability":1-p,
                "residual_model_push_probability":0.0,"over_probability_no_vig":0.5})
    return pd.DataFrame(rows)

def test_wilson_lower_bound_below_point_estimate():
    assert 0<wilson_lower_bound(0.60,200)<0.60

def test_wilson_rejects_invalid_inputs():
    with pytest.raises(ValueError): wilson_lower_bound(1.1,100)
    with pytest.raises(ValueError): wilson_lower_bound(0.6,0)

def test_development_gate_uses_development_only():
    gate=development_gate(fixture(),{"status":"completed","readiness":{"development_walk_forward_predictions":160}})
    assert gate["development_probability_games"]==160
    assert gate["development_probability_dates"]==16
    assert gate["passed"]

def test_failed_gate_means_no_bet():
    frame=fixture()
    gate=development_gate(frame,{"status":"insufficient_calibration_sample","readiness":{}})
    decisions=select_holdout_bets(frame,gate,0)
    assert len(decisions)==40
    assert set(decisions.decision)=={"No Bet"}
    assert decisions.stake_units.sum()==0

def test_holdout_outcomes_do_not_change_decisions():
    frame=fixture()
    gate=development_gate(frame,{"status":"completed","readiness":{"development_walk_forward_predictions":160}})
    first=select_holdout_bets(frame,gate,160)
    changed=frame.copy()
    changed.loc[changed.evaluation_split.eq("reserved_holdout"),"actual_total"]+=100
    second=select_holdout_bets(changed,gate,160)
    cols=["decision","stake_units","wilson_lower_probability","conservative_ev_per_unit"]
    pd.testing.assert_frame_equal(first[cols],second[cols])

def test_daily_exposure_and_per_game_caps():
    frame=fixture()
    gate=development_gate(frame,{"status":"completed","readiness":{"development_walk_forward_predictions":160}})
    decisions=select_holdout_bets(frame,gate,160,bankroll_units=100)
    assert (decisions.groupby("game_date").stake_units.sum()<=2.0+1e-9).all()
    assert (decisions.groupby("game_date").stake_units.max()<=0.5+1e-9).all()
