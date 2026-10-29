"""Canonical NBA team-name handling shared by the totals feature engines."""

import re
from functools import lru_cache

# Canonical display names keyed by a compact, case-insensitive team label.
# Unknown labels are preserved, so historical teams are not silently renamed.
_CANONICAL_NAMES = {
    "atlantahawks": "Atlanta Hawks",
    "bostonceltics": "Boston Celtics",
    "brooklynnets": "Brooklyn Nets",
    "charlottehornets": "Charlotte Hornets",
    "chicagobulls": "Chicago Bulls",
    "clevelandcavaliers": "Cleveland Cavaliers",
    "dallasmavericks": "Dallas Mavericks",
    "denvernuggets": "Denver Nuggets",
    "detroitpistons": "Detroit Pistons",
    "goldenstatewarriors": "Golden State Warriors",
    "gswarriors": "Golden State Warriors",
    "houstonrockets": "Houston Rockets",
    "indianapacers": "Indiana Pacers",
    "laclippers": "Los Angeles Clippers",
    "losangelesclippers": "Los Angeles Clippers",
    "lalakers": "Los Angeles Lakers",
    "losangeleslakers": "Los Angeles Lakers",
    "memphisgrizzlies": "Memphis Grizzlies",
    "miamiheat": "Miami Heat",
    "milwaukeebucks": "Milwaukee Bucks",
    "minnesotatimberwolves": "Minnesota Timberwolves",
    "neworleanspelicans": "New Orleans Pelicans",
    "nyknicks": "New York Knicks",
    "newyorkknicks": "New York Knicks",
    "okcthunder": "Oklahoma City Thunder",
    "oklahomacitythunder": "Oklahoma City Thunder",
    "orlandomagic": "Orlando Magic",
    "philadelphiasixers": "Philadelphia 76ers",
    "philadelphia76ers": "Philadelphia 76ers",
    "phoenixsuns": "Phoenix Suns",
    "portlandblazers": "Portland Trail Blazers",
    "portlandtrailblazers": "Portland Trail Blazers",
    "sacramentokings": "Sacramento Kings",
    "sanantoniospurs": "San Antonio Spurs",
    "torontoraptors": "Toronto Raptors",
    "utahjazz": "Utah Jazz",
    "washingtonwizards": "Washington Wizards",
}


@lru_cache(maxsize=512)
def _normalize_cached(raw_name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", raw_name.casefold())
    return _CANONICAL_NAMES.get(key, raw_name)


def normalize_team_name(name) -> str:
    """Return a canonical display name for known NBA aliases."""
    return _normalize_cached(str(name or "").strip())


def normalize_history_team_columns(history_df):
    """Return a copy whose home/away team labels use canonical names.

    A dataframe attribute marks the canonicalized form to avoid repeatedly
    scanning large historical datasets in chronological backtests.
    """
    if history_df is None:
        return history_df

    df = history_df.copy()
    if df.empty or not {"home_team", "away_team"}.issubset(df.columns):
        return df

    if df.attrs.get("_team_names_normalized") is True:
        return df

    df["home_team"] = df["home_team"].map(normalize_team_name)
    df["away_team"] = df["away_team"].map(normalize_team_name)
    df.attrs["_team_names_normalized"] = True
    return df
