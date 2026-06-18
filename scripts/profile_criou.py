#!/usr/bin/env python
"""Profile the polygon CRIoU pipeline and benchmark faster alternatives.

Reports, for random OBB pairs on GPU:
  (1) Stage breakdown of cal_ciou: corners / intersection (split into
      edge-intersect, corner-in-box, build-vertices, vertex-sort, shoelace) /
      smallest enclosing box / aspect term.
  (2) smallest *rotated* enclosing box vs cheap *axis-aligned* enclosing box
      (lever: replace it in the DIoU term) — time and value agreement.
  (3) If mmcv is available, vendored cal_iou vs mmcv diff_iou_rotated_2d (fused,
      Sutherland-Hodgman CUDA op) — forward and forward+backward.

Usage: python scripts/profile_criou.py [--device cuda] [--sizes 256 1024 4096]
"""

import argparse
import time

import math

import torch

from ultralytics.utils.metrics import probiou
from aod_yolov8.obb_loss import kfiou_per_pair
from rotated_iou.oriented_iou_loss import box2corners_th, cal_iou, cal_diou, cal_ciou, enclosing_box
from rotated_iou.box_intersection_2d import (
    box_intersection_th, box_in_box_th, build_vertices, sort_indices, calculate_area,
)
from rotated_iou.min_enclosing_box import smallest_bounding_box


def make_pairs(n, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    b1 = torch.zeros(1, n, 5)
    b1[..., 0:2] = torch.rand(1, n, 2, generator=g) * 1024
    b1[..., 2:4] = 10 + torch.rand(1, n, 2, generator=g) * 90
    b1[..., 4] = (torch.rand(1, n, generator=g) - 0.5) * 3.14159
    b2 = b1.clone()
    b2[..., 0:2] += torch.randn(1, n, 2, generator=g) * 8     # nearby -> real overlap
    b2[..., 2:4] *= 1 + torch.randn(1, n, 2, generator=g) * 0.1
    b2[..., 4] += torch.randn(1, n, generator=g) * 0.1
    return b1.to(device), b2.to(device)


def bench(fn, iters=50, warmup=10, cuda=True):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms/call


def aabb_c2(c1, c2):
    """Axis-aligned enclosing-box diagonal^2 of the 8 corners (cheap DIoU term)."""
    allc = torch.cat([c1, c2], dim=2)  # (B,N,8,2)
    xmn = allc[..., 0].min(2)[0]; xmx = allc[..., 0].max(2)[0]
    ymn = allc[..., 1].min(2)[0]; ymx = allc[..., 1].max(2)[0]
    return (xmx - xmn) ** 2 + (ymx - ymn) ** 2


def fast_ciou(b1, b2):
    """Proposed CRIoU using the axis-aligned (not smallest-rotated) enclosing box."""
    iou, c1, c2c, _u = cal_iou(b1, b2)
    c2 = aabb_c2(c1, c2c)
    d2 = (b1[..., 0] - b2[..., 0]) ** 2 + (b1[..., 1] - b2[..., 1]) ** 2
    v = (4 / math.pi ** 2) * (torch.atan(b2[..., 2] / b2[..., 3]) - torch.atan(b1[..., 2] / b1[..., 3])) ** 2
    alpha = v / ((1.0 - iou) + v + 1e-7)
    return 1.0 - iou + d2 / c2 + alpha * v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 1024, 4096])
    args = ap.parse_args()
    dev = args.device
    cuda = dev.startswith("cuda")
    print(f"device={dev}  torch={torch.__version__}")
    try:
        from mmcv.ops import diff_iou_rotated_2d
        has_mmcv = True
        print("mmcv: available (diff_iou_rotated_2d)")
    except Exception as e:
        has_mmcv = False
        print(f"mmcv: NOT available ({type(e).__name__})")

    for n in args.sizes:
        print(f"\n================  N = {n} pairs  ================")
        b1, b2 = make_pairs(n, dev)
        c1, c2 = box2corners_th(b1), box2corners_th(b2)

        # ---- (1) stage breakdown ----
        t_corners = bench(lambda: (box2corners_th(b1), box2corners_th(b2)), cuda=cuda)

        def inter_substeps():
            inters, mi = box_intersection_th(c1, c2)
            a, bb = box_in_box_th(c1, c2)
            v, m = build_vertices(c1, c2, a, bb, inters, mi)
            si = sort_indices(v, m)
            return calculate_area(si, v)
        t_inter_total = bench(inter_substeps, cuda=cuda)

        t_edge = bench(lambda: box_intersection_th(c1, c2), cuda=cuda)
        inters, mi = box_intersection_th(c1, c2)
        t_cinb = bench(lambda: box_in_box_th(c1, c2), cuda=cuda)
        a, bb = box_in_box_th(c1, c2)
        t_build = bench(lambda: build_vertices(c1, c2, a, bb, inters, mi), cuda=cuda)
        v, m = build_vertices(c1, c2, a, bb, inters, mi)
        t_sort = bench(lambda: sort_indices(v, m), cuda=cuda)
        si = sort_indices(v, m)
        t_shoe = bench(lambda: calculate_area(si, v), cuda=cuda)

        t_enclose = bench(lambda: enclosing_box(c1, c2, "smallest"), cuda=cuda)
        t_probiou = bench(lambda: probiou(b1[0], b2[0]), cuda=cuda)  # probiou wants (N,5)
        t_kfiou = bench(lambda: kfiou_per_pair(b1, b2), cuda=cuda)
        t_riou = bench(lambda: cal_iou(b1, b2), cuda=cuda)
        t_ciou = bench(lambda: cal_ciou(b1, b2), cuda=cuda)
        t_fast = bench(lambda: fast_ciou(b1, b2), cuda=cuda)

        print(f"  corners (x2)            {t_corners:8.3f} ms")
        print(f"  intersection TOTAL      {t_inter_total:8.3f} ms")
        print(f"    - edge intersect      {t_edge:8.3f} ms")
        print(f"    - corner-in-box       {t_cinb:8.3f} ms")
        print(f"    - build vertices      {t_build:8.3f} ms")
        print(f"    - VERTEX SORT         {t_sort:8.3f} ms   <-- removed by Sutherland-Hodgman")
        print(f"    - shoelace area       {t_shoe:8.3f} ms")
        print(f"  smallest enclosing box  {t_enclose:8.3f} ms   <-- DIoU term")
        print(f"  --- loss forward cost (ms) and slowdown vs ProbIoU ---")
        print(f"  ProbIoU  (Gaussian)     {t_probiou:8.3f} ms   (1.0x baseline)")
        print(f"  KFIoU    (Gaussian)     {t_kfiou:8.3f} ms   ({t_kfiou / t_probiou:5.1f}x)")
        print(f"  RIoU     (polygon)      {t_riou:8.3f} ms   ({t_riou / t_probiou:5.1f}x)")
        print(f"  CRIoU    (polygon+rot)  {t_ciou:8.3f} ms   ({t_ciou / t_probiou:5.1f}x)")
        print(f"  fast-CRIoU (polygon+AA) {t_fast:8.3f} ms   ({t_fast / t_probiou:5.1f}x)  proposed")

        # ---- (2) enclosing box: rotated vs axis-aligned ----
        t_aabb = bench(lambda: aabb_c2(c1, c2), cuda=cuda)
        w, h = enclosing_box(c1, c2, "smallest")
        c2_rot = (w * w + h * h)
        c2_ax = aabb_c2(c1, c2)
        ratio = (c2_ax / c2_rot.clamp(min=1e-6)).mean().item()
        spd = t_enclose / max(t_aabb, 1e-9)
        print(f"  [enclosing box]  rotated {t_enclose:7.3f} ms  vs  axis-aligned {t_aabb:7.3f} ms"
              f"  ({spd:.1f}x faster, c2 ratio ax/rot = {ratio:.3f})")

        # ---- (3) vendored vs mmcv fused op ----
        if has_mmcv:
            bb1 = b1.clone().requires_grad_(True)
            bb2 = b2.clone()
            t_vend_f = bench(lambda: cal_iou(b1, b2), cuda=cuda)
            t_mmcv_f = bench(lambda: diff_iou_rotated_2d(b1, b2), cuda=cuda)

            def vend_fb():
                bb1.grad = None
                cal_iou(bb1, bb2)[0].sum().backward()

            def mmcv_fb():
                bb1.grad = None
                diff_iou_rotated_2d(bb1, bb2).sum().backward()
            t_vend_fb = bench(vend_fb, cuda=cuda)
            t_mmcv_fb = bench(mmcv_fb, cuda=cuda)
            # agreement
            iou_v = cal_iou(b1, b2)[0]
            iou_m = diff_iou_rotated_2d(b1, b2).squeeze(0) if diff_iou_rotated_2d(b1, b2).dim() == 2 else diff_iou_rotated_2d(b1, b2)
            mad = (iou_v.flatten() - iou_m.flatten()).abs().mean().item()
            print(f"  [mmcv vs vendored]  fwd: {t_vend_f:7.3f} -> {t_mmcv_f:7.3f} ms"
                  f"  ({t_vend_f / max(t_mmcv_f,1e-9):.1f}x) | fwd+bwd: {t_vend_fb:7.3f} -> {t_mmcv_fb:7.3f} ms"
                  f"  ({t_vend_fb / max(t_mmcv_fb,1e-9):.1f}x) | mean|dIoU|={mad:.4f}")


if __name__ == "__main__":
    main()
