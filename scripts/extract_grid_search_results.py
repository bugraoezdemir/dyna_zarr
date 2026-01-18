#!/usr/bin/env python3
"""Extract successful trials from grid_search_results.json and write CSV.

Usage: python3 scripts/extract_grid_search_results.py
Creates: grid_search_ok_sorted.csv in repository root.
"""
import json
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "grid_search_results.json"
OUT = ROOT / "grid_search_ok_sorted.csv"


def main():
    data = json.loads(SRC.read_text())

    # collect all config keys to build CSV header
    config_keys = set()
    rows = []
    for entry in data:
        result = entry.get("result", {})
        if result.get("status") != "ok":
            continue
        config = entry.get("config", {})
        config_keys.update(config.keys())
        elapsed = result.get("elapsed")
        trial_time = entry.get("trial_time")
        rows.append({"elapsed": elapsed, "trial_time": trial_time, **config})

    if not rows:
        print("No successful trials found.")
        return

    # order config columns consistently
    config_keys = sorted(config_keys)

    # sort rows by elapsed (ascending)
    rows.sort(key=lambda r: float(r.get("elapsed", float("inf"))))

    # write CSV header
    header = ["elapsed", "trial_time"] + config_keys

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            # ensure all keys present
            row = {k: r.get(k, "") for k in header}
            w.writerow(row)

    print(f"Wrote {len(rows)} successful trials to {OUT}")


if __name__ == "__main__":
    main()
