import pandas as pd


def calculate_totals_clv(saved_total, closing_total):

    try:
        saved_total = float(saved_total)
        closing_total = float(closing_total)

        return round(
            saved_total - closing_total,
            2
        )

    except Exception:
        return None
