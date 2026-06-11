# injury_data_collector.py

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup


def collect_injury_data():

    try:

        url = "https://www.espn.com/nba/injuries"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64)"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        tables = soup.find_all("table")

        injuries = []

        for table in tables:

            rows = table.find_all("tr")

            for row in rows:

                cols = row.find_all("td")

                if len(cols) < 4:
                    continue

                try:

                    player_name = cols[0].get_text(
                        strip=True
                    )

                    status = cols[2].get_text(
                        strip=True
                    )

                    details = cols[3].get_text(
                        strip=True
                    )

                    injuries.append({
                        "player_name": player_name,
                        "status": status,
                        "details": details,
                    })

                except Exception:
                    continue

        injury_df = pd.DataFrame(injuries)

        os.makedirs(
            "data",
            exist_ok=True
        )

        injury_df.to_csv(
            "data/injury_report.csv",
            index=False
        )

        injury_df.to_csv(
            "injury_report.csv",
            index=False
        )

        return {
            "status": "success",
            "rows": len(injury_df),
            "file": "data/injury_report.csv"
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    print(
        collect_injury_data()
    )
