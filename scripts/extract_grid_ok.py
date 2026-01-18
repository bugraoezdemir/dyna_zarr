#!/usr/bin/env python3
import json
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / 'grid_search_results.json'
OUT_CSV = ROOT / 'grid_search_ok_sorted.csv'

with JSON_PATH.open() as f:
    data = json.load(f)

ok_entries = []
config_keys = set()
for i, entry in enumerate(data):
    res = entry.get('result', {})
    if res.get('status') == 'ok':
        cfg = entry.get('config', {})
        config_keys.update(cfg.keys())
        row = {
            'index': i,
            'elapsed': res.get('elapsed'),
            'trial_time': entry.get('trial_time')
        }
        # flatten config
        for k, v in cfg.items():
            row[f'cfg_{k}'] = v
        # include any other result fields except status/elapsed
        for k, v in res.items():
            if k in ('status','elapsed'):
                continue
            row[f'result_{k}'] = v
        ok_entries.append(row)

# sort by elapsed (None goes to end)
ok_entries.sort(key=lambda r: (r['elapsed'] is None, r['elapsed']))

# build header
cfg_keys_sorted = sorted(list(config_keys))
headers = ['index','elapsed','trial_time'] + [f'cfg_{k}' for k in cfg_keys_sorted]
# collect extra result keys
extra_result_keys = set()
for r in ok_entries:
    for k in r.keys():
        if k.startswith('result_'):
            extra_result_keys.add(k)
headers += sorted(extra_result_keys)

with OUT_CSV.open('w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    writer.writeheader()
    for r in ok_entries:
        # ensure all header keys exist
        out = {h: r.get(h, '') for h in headers}
        writer.writerow(out)

print(f'Wrote {len(ok_entries)} successful trials to {OUT_CSV}')
