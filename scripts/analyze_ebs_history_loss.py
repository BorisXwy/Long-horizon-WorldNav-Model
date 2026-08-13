#!/usr/bin/env python3
"""Compare ebs16 (accum=16) vs ebs1 (no accumulation) Stage One 1.0 runs,
and break down loss by history_chunks (1/2/3).

Usage:
    python analyze_ebs_history_loss.py
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

LOG_ROOT = Path("/mnt/pool1/sharehome/xiewenyuan/academic/3d_wm_vln/NAV/log")

RUNS = {
    "A_ebs1 (DL3DV 10000, no accum)": "stage-one-v1-dl3dv-a-10000",
    "B_ebs1 (DL3DV 10000, no accum)": "stage-one-v1-dl3dv-b-10000",
    "A_ebs16 (DL3DV 1000, accum=16)": "stage-one-v1-from-scratch-a-ebs16-1000",
    "B_ebs16 (DL3DV 1000, accum=16)": "stage-one-v1-from-scratch-b-ebs16-1000",
}


def parse(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" not in o:
                continue
            rows.append(o)
    return rows


def fmt(x, n=4):
    if x is None:
        return "  -  "
    return f"{x:.{n}f}"


def summarize(rows, label, step_lo=None, step_hi=None):
    sel = [r for r in rows
           if (step_lo is None or r["step"] >= step_lo)
           and (step_hi is None or r["step"] < step_hi)]
    if not sel:
        print(f"  {label}: no data")
        return
    losses = [r["loss"] for r in sel]
    grads = [r["register_grad_norm"] for r in sel]
    by_h = defaultdict(list)
    for r in sel:
        by_h[r.get("history_chunks")].append(r["loss"])
    print(f"  {label}: n={len(sel)} steps [{sel[0]['step']}-{sel[-1]['step']}]")
    print(f"    loss  mean={fmt(st.mean(losses))} median={fmt(st.median(losses))} "
          f"min={fmt(min(losses))} max={fmt(max(losses))} std={fmt(st.pstdev(losses))}")
    print(f"    rgrad mean={fmt(st.mean(grads),3)} median={fmt(st.median(grads),3)}")
    for h in sorted([k for k in by_h if k is not None]):
        ls = by_h[h]
        print(f"    hist={h}: n={len(ls):4d} mean={fmt(st.mean(ls))} "
              f"median={fmt(st.median(ls))} std={fmt(st.pstdev(ls))}")


def main():
    data = {name: parse(LOG_ROOT / run / "train.log") for name, run in RUNS.items()}
    print("=" * 78)
    print("EFFECTIVE BATCH SIZE COMPARISON (ebs1 no-accum vs ebs16 accum=16)")
    print("Both DL3DV from scratch, same seed 20260730, history_lengths [1,2,3]")
    print("=" * 78)

    # 1) Same number of optimizer updates: first 1000 effective steps
    print("\n[1] First 1000 OPTIMIZER steps (ebs1: 1000 samples; ebs16: 16000 samples)")
    for name in RUNS:
        rows = data[name]
        print(f"\n  --- {name} ---")
        summarize(rows, "step [1,1000)", 1, 1000)

    # 2) Same number of samples seen: ebs1 first 16000 steps vs ebs16 1000 steps
    print("\n\n[2] Same SAMPLE count: ebs1 first 16000 steps vs ebs16 1000 effective steps")
    for name in RUNS:
        rows = data[name]
        print(f"\n  --- {name} ---")
        if "ebs16" in name:
            summarize(rows, "step [1,1000) [ebs16]", 1, 1000)
        else:
            summarize(rows, "step [1,16000) [ebs1]", 1, 16000)

    # 3) Late-window comparison: ebs1 steps [9000,10000) vs ebs16 last avail
    print("\n\n[3] Late window: ebs1 steps [9000,10000) vs ebs16 all avail")
    for name in RUNS:
        rows = data[name]
        print(f"\n  --- {name} ---")
        if "ebs16" in name:
            summarize(rows, "all avail")
        else:
            summarize(rows, "step [9000,10000)", 9000, 10000)

    # 4) History-length loss trajectory in 200-step buckets (ebs1 only, has enough data)
    print("\n\n[4] ebs1 loss by 200-step bucket x history (A)")
    rows = data["A_ebs1 (DL3DV 10000, no accum)"]
    for lo in range(1, 10000, 1000):
        sel = [r for r in rows if lo <= r["step"] < lo + 1000]
        if not sel:
            continue
        by_h = defaultdict(list)
        for r in sel:
            by_h[r.get("history_chunks")].append(r["loss"])
        line = f"  [{lo:5d},{lo+1000:5d}) n={len(sel):4d} "
        for h in [1, 2, 3]:
            ls = by_h.get(h, [])
            if ls:
                line += f"h{h}:m={fmt(st.mean(ls))}(med{fmt(st.median(ls))}) "
            else:
                line += f"h{h}:  -    "
        print(line)

    print("\n[5] ebs16 loss by 100-step bucket x history (A)")
    rows = data["A_ebs16 (DL3DV 1000, accum=16)"]
    for lo in range(1, 1000, 100):
        sel = [r for r in rows if lo <= r["step"] < lo + 100]
        if not sel:
            continue
        by_h = defaultdict(list)
        for r in sel:
            by_h[r.get("history_chunks")].append(r["loss"])
        line = f"  [{lo:4d},{lo+100:4d}) n={len(sel):3d} "
        for h in [1, 2, 3]:
            ls = by_h.get(h, [])
            if ls:
                line += f"h{h}:m={fmt(st.mean(ls))}(med{fmt(st.median(ls))}) "
            else:
                line += f"h{h}:  -    "
        print(line)

    print("\n[6] ebs16 loss by 100-step bucket x history (B)")
    rows = data["B_ebs16 (DL3DV 1000, accum=16)"]
    for lo in range(1, 1000, 100):
        sel = [r for r in rows if lo <= r["step"] < lo + 100]
        if not sel:
            continue
        by_h = defaultdict(list)
        for r in sel:
            by_h[r.get("history_chunks")].append(r["loss"])
        line = f"  [{lo:4d},{lo+100:4d}) n={len(sel):3d} "
        for h in [1, 2, 3]:
            ls = by_h.get(h, [])
            if ls:
                line += f"h{h}:m={fmt(st.mean(ls))}(med{fmt(st.median(ls))}) "
            else:
                line += f"h{h}:  -    "
        print(line)


if __name__ == "__main__":
    main()
