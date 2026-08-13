#!/usr/bin/env python3
"""Aggregate StreamVLN result.json / DualVLN progress.json and compare to paper."""
import argparse, json, math, os, sys
from pathlib import Path

PAPER = {
    "StreamVLN": {
        "r2r": {"NE": 4.90, "OS": 63.6, "SR": 56.4, "SPL": 50.2},
        "rxr": {"NE": 5.65, "SR": 54.4, "SPL": 45.4, "nDTW": 63.7},
        "source": "StreamVLN README v1_3 checkpoint",
    },
    "InternVLA-N1-DualVLN": {
        "r2r": {"NE": 4.05, "OS": 70.7, "SR": 64.3, "SPL": 58.5},
        "rxr": {"NE": 4.58, "nDTW": 70.0, "SR": 61.4, "SPL": 51.8},
        "source": "NAV-RES-002 monocular table / DualVLN paper",
    },
}

def load_rows(path: Path):
    rows = []
    text = path.read_text().strip()
    if not text:
        return rows
    # JSONL or single JSON list
    if text.startswith('['):
        rows = json.loads(text)
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        # StreamVLN appends a final aggregate dict without per-episode fields
        if "success" not in obj and "suc" not in obj and "sr" not in obj:
            continue
        rows.append(obj)
    return rows

def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float('nan')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('result_file')
    ap.add_argument('--method', default='StreamVLN')
    ap.add_argument('--split', default='r2r')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    path = Path(args.result_file)
    rows = load_rows(path)
    n = len(rows)
    # field aliases
    def g(r, *keys):
        for k in keys:
            if k in r and r[k] is not None:
                return float(r[k])
        return None
    sucs = [g(r, 'success', 'suc', 'sr') for r in rows]
    spls = [g(r, 'spl') for r in rows]
    oss = [g(r, 'os', 'oracle_success') for r in rows]
    nes = [g(r, 'ne', 'navigation_error') for r in rows]
    ndtws = [g(r, 'ndtw', 'nDTW') for r in rows]
    metrics = {
        'n': n,
        'SR': 100.0 * mean(sucs) if n else None,
        'SPL': 100.0 * mean(spls) if n else None,
        'OS': 100.0 * mean(oss) if n else None,
        'NE': mean(nes) if n else None,
    }
    if any(x is not None for x in ndtws):
        metrics['nDTW'] = 100.0 * mean([x for x in ndtws if x is not None])
    paper = PAPER.get(args.method, {}).get(args.split, {})
    cmp = {}
    for k, pv in paper.items():
        if k in metrics and metrics[k] is not None:
            cmp[k] = {'ours': round(metrics[k], 2), 'paper': pv, 'delta': round(metrics[k] - pv, 2)}
    out = {
        'method': args.method,
        'split': args.split,
        'source_file': str(path),
        'metrics_pct_or_raw': {k: (None if v is None else round(v, 2)) for k, v in metrics.items()},
        'paper': paper,
        'comparison': cmp,
        'note': 'Partial N-episode sample; NOT full val_unseen. Deltas are indicative only.',
        'paper_source': PAPER.get(args.method, {}).get('source'),
    }
    s = json.dumps(out, indent=2, ensure_ascii=False)
    print(s)
    if args.out:
        Path(args.out).write_text(s + '\n')

if __name__ == '__main__':
    main()
