#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prospective_player_data import build_pregame_player_state
from player_state_snapshot import write_player_state_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/player_state_snapshot/latest.json.gz"))
    args = parser.parse_args()
    state = build_pregame_player_state(args.target_date)
    manifest = write_player_state_snapshot(state, args.target_date, args.output)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
