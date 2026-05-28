import random


def generate_market_intelligence(df):
    df = df.copy()

    for idx, row in df.iterrows():

        current_odds = float(row.get("odds", 2.0))

        opening_shift = random.uniform(-0.15, 0.15)

        opening_odds = round(
            current_odds + opening_shift,
            2
        )

        closing_shift = random.uniform(-0.10, 0.10)

        closing_odds = round(
            current_odds + closing_shift,
            2
        )

        line_movement_pct = round(
            ((closing_odds - opening_odds) / opening_odds) * 100,
            4
        )

        sharp_support_pct = round(
            random.uniform(0.35, 0.85),
            4
        )

        steam_move = 1 if abs(line_movement_pct) > 5 else 0

        reverse_line_movement = (
            1
            if (
                sharp_support_pct > 0.70
                and line_movement_pct < 0
            )
            else 0
        )

        df.loc[idx, "opening_odds"] = opening_odds
        df.loc[idx, "closing_odds"] = closing_odds
        df.loc[idx, "line_movement_diff"] = line_movement_pct
        df.loc[idx, "sharp_support_pct"] = sharp_support_pct
        df.loc[idx, "steam_move"] = steam_move
        df.loc[idx, "reverse_line_movement"] = reverse_line_movement

    return df
