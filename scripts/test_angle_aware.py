#!/usr/bin/env python
"""Test a fast, aspect-adaptive angle term that restores orientation sensitivity
for square objects, where Gaussian IoU (ProbIoU/KFIoU) is blind.

For a square GT box, sweeps the predicted angle and compares the angle response of:
  - polygon RIoU            (ground truth, correct but slow)
  - ProbIoU                 (Gaussian, expected flat -> the bug)
  - proposed Gaussian+angle (ProbIoU overlap + cheap explicit angle term)

Reports angle sensitivity, gradient at a small misalignment, and shape correlation
to the polygon ground truth. Repeats for an elongated box as a control.
"""

import math

import torch

from ultralytics.utils.metrics import probiou
from rotated_iou.oriented_iou_loss import cal_iou


def angle_term(b1, b2):
    """Aspect-adaptive orientation penalty. 0 when aligned, smooth, cheap."""
    dtheta = b1[..., 4] - b2[..., 4]
    w, h = b2[..., 2], b2[..., 3]
    s = 1.0 - torch.min(w, h) / torch.max(w, h)          # 0 square -> 1 elongated
    t2 = (1.0 - torch.cos(2 * dtheta)) / 2               # 180-periodic
    t4 = (1.0 - torch.cos(4 * dtheta)) / 2               # 90-periodic
    return s * t2 + (1.0 - s) * t4


def polygon_iou(b1, b2):
    return cal_iou(b1.unsqueeze(0), b2.unsqueeze(0))[0].squeeze(0)


def sweep(gt, label, period_deg):
    print(f"\n===== {label}  (GT wxh = {gt[2]:.0f}x{gt[3]:.0f}) =====")
    angles = torch.arange(0, period_deg + 1, 1.0) * math.pi / 180.0
    n = len(angles)
    gtb = torch.tensor(gt, dtype=torch.float32).repeat(n, 1)
    pred = gtb.clone()
    pred[:, 4] = gtb[:, 4] + angles

    poly = polygon_iou(pred, gtb)                 # (n,)
    pro = probiou(pred, gtb).squeeze(-1)          # (n,)
    at = angle_term(pred, gtb)
    proposed_iou = pro * (1.0 - at)               # proxy IoU for shape comparison

    def rng(x):
        return (x.max() - x.min()).item()

    print(f"  {'angle':>6} {'polyIoU':>8} {'ProbIoU':>8} {'proposed':>9}")
    for d in [0, 15, 30, 45, 60, 75, 90]:
        if d > period_deg:
            continue
        i = d
        print(f"  {d:>5}d {poly[i]:8.3f} {pro[i]:8.3f} {proposed_iou[i]:9.3f}")

    print(f"  angle sensitivity (max-min over sweep):")
    print(f"    polygon  {rng(poly):.3f}   <- true orientation signal")
    print(f"    ProbIoU  {rng(pro):.3f}   <- ~0 means blind to angle")
    print(f"    proposed {rng(proposed_iou):.3f}")

    # gradient at 5 degrees misalignment
    d5 = torch.tensor(5.0 * math.pi / 180.0)
    for name, fn in [
        ("ProbIoU ", lambda p, g: 1.0 - probiou(p, g)),
        ("polygon ", lambda p, g: 1.0 - polygon_iou(p, g)),
        ("proposed", lambda p, g: (1.0 - probiou(p, g)) + angle_term(p, g)),
    ]:
        g = torch.tensor([gt], dtype=torch.float32)
        p = g.clone()
        p[:, 4] = p[:, 4] + d5
        p.requires_grad_(True)
        loss = fn(p, g).sum()
        loss.backward()
        gth = p.grad[0, 4].item()
        print(f"    dLoss/dtheta @5deg  {name}: {gth:+.4f}"
              + ("   (no signal)" if abs(gth) < 1e-4 else ""))

    # shape correlation of proposed vs polygon
    pv = proposed_iou - proposed_iou.mean()
    gv = poly - poly.mean()
    corr = (pv * gv).sum() / (pv.norm() * gv.norm() + 1e-9)
    print(f"  shape correlation proposed vs polygon: {corr.item():+.3f}")


def main():
    torch.set_grad_enabled(True)
    # x, y, w, h, theta
    sweep([0, 0, 40, 40, 0.0], "SQUARE object", 90)
    sweep([0, 0, 40, 30, 0.0], "NEAR-SQUARE object (intermediate, s=0.25)", 180)
    sweep([0, 0, 60, 20, 0.0], "ELONGATED object (control)", 180)


if __name__ == "__main__":
    main()
