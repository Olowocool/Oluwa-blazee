# Standalone NBA Total Points Excel Model

This folder is isolated from the deployed application and all existing prediction engines.

## Current build
- Retrieves the selected date's fixture list from ESPN's public scoreboard endpoint.
- Retrieves current roster entries for teams on that slate.
- Creates a workbook with Dashboard, Schedule, Team Inputs, Player Inputs, Game Engine, Settings and Source Log sheets.
- The workbook contains transparent formulas for possessions and team scoring.
- It does **not** invent injury status, projected minutes, team efficiency or player impact. Missing team inputs are explicitly flagged.

## Build
Run `python excel_totals_model/build_workbook.py` with Python 3.12 and `requests` + `openpyxl` installed. Set `GAME_DATE=YYYY-MM-DD` to choose a date. The GitHub Actions workflow can also be run manually and uploads the workbook as an artifact.

## Important status
This is the first deliverable scaffold, not a validated forecasting model. Automatic fixture/roster refresh is implemented; automatic point-in-time team ratings, injury reports and projected player minutes are not yet wired. The dashboard must not be treated as a complete prediction until those inputs are sourced and populated.
