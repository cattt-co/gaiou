#!/usr/bin/env python
"""Test GA-NMS (orientation-aware suppression) vs ProbIoU-NMS vs polygon RIoU-NMS.

(A) Controlled pair: two overlapping squares, vary the relative angle. Show ProbIoU
    overlap is flat in angle (blind), while true RIoU drops and GA-sim follows.
(B) Packed grid of distinct square objects at varied orientations: run NMS with each
    similarity and count survivors. RIoU is ground truth; ProbIoU over-suppresses
    (merges differently-oriented neighbours); GA recovers toward the truth.
(C) Speed of each pairwise similarity.
"""

import time
import math

import torch

from aod_yolov8.obb_metric import batch_probiou, batch_riou, batch_ga
from ultralytics.utils.nms import TorchNMS


def controlled_pair():
    print("=== (A) two overlapping 40x40 squares, center offset 22px, vary angle ===")
    print(f"  {'dtheta':>7} {'ProbIoU':>8} {'RIoU(true)':>11} {'GA-sim':>8}")
    base = torch.tensor([[0., 0, 40, 40, 0]])
    for d in [0, 15, 30, 45, 60, 75, 90]:
        other = torch.tensor([[22., 0, 40, 40, d * math.pi / 180]])
        p = batch_probiou(base, other)[0, 0].item()
        r = batch_riou(base, other)[0, 0].item()
        g = batch_ga(base, other)[0, 0].item()
        print(f"  {d:>6}d {p:8.3f} {r:11.3f} {g:8.3f}")


def packed_grid(thr=0.3):
    # Tightly packed row of distinct square objects, alternating angle 0/45 deg.
    # Adjacent boxes are real, separate objects whose TRUE overlap (~0.25) is below
    # thr (keep both), but ProbIoU's orientation-blind overlap (~0.40) is above thr
    # (wrongly suppress). 8 boxes spaced 22px (< side 40).
    print(f"\n=== (B) row of 8 distinct squares, alternating 0/45 deg, NMS@{thr} ===")
    torch.manual_seed(0)
    boxes, scores = [], []
    for k in range(8):
        ang = (45 if k % 2 else 0) * math.pi / 180
        boxes.append([60 + k * 22, 100, 40, 40, ang]); scores.append(0.9 - 0.02 * k)
    boxes = torch.tensor(boxes); scores = torch.tensor(scores)
    n = len(boxes)
    for name, fn in [("ProbIoU-NMS", batch_probiou), ("GA-NMS (ours)", batch_ga), ("RIoU-NMS (truth)", batch_riou)]:
        keep = TorchNMS.fast_nms(boxes.clone(), scores.clone(), thr, iou_func=fn)
        print(f"  {name:18s} survivors: {len(keep)} / {n}")
    print("  (all 8 are distinct -> RIoU=truth; ProbIoU over-suppresses, GA should match RIoU)")

    # sanity: true duplicates (same place, same angle) must still be suppressed by all
    dup = torch.tensor([[200., 200, 40, 40, 0.3]] * 4)
    ds = torch.tensor([0.9, 0.8, 0.7, 0.6])
    print("  true-duplicate check (4 identical boxes -> should collapse to 1):", end=" ")
    print({n2: len(TorchNMS.fast_nms(dup.clone(), ds.clone(), thr, iou_func=f))
           for n2, f in [("Prob", batch_probiou), ("GA", batch_ga), ("RIoU", batch_riou)]})


def speed():
    print("\n=== (C) pairwise similarity speed ===", flush=True)
    torch.manual_seed(0)

    def bench(fn, it=20, wu=3):
        for _ in range(wu):
            fn()
        t = time.perf_counter()
        for _ in range(it):
            fn()
        return (time.perf_counter() - t) / it * 1000

    for N in [256, 512]:
        b = torch.rand(N, 5); b[:, 2:4] = b[:, 2:4] * 30 + 20; b[:, 4] = (b[:, 4] - .5) * 3.14
        tp = bench(lambda: batch_probiou(b, b))
        tg = bench(lambda: batch_ga(b, b))
        tr = bench(lambda: batch_riou(b, b), it=3, wu=1)  # polygon: few iters, it is slow
        print(f"  N={N:4d} (NxN): ProbIoU {tp:7.2f} ms | GA {tg:7.2f} ms ({tg/tp:.2f}x) | "
              f"RIoU {tr:9.1f} ms ({tr/tg:.0f}x slower than GA)", flush=True)


if __name__ == "__main__":
    controlled_pair()
    packed_grid()
    speed()
