#!/usr/bin/env python
"""Calibrate GA-NMS angle weight lambda to avoid both over- and under-suppression.

OVER  : a row of distinct squares at alternating 0/45 deg. ProbIoU-NMS (lambda=0)
        wrongly merges them; we want all kept (matches polygon RIoU).
UNDER : ONE object detected several times with angle noise (0..30 deg, same centre).
        Full GA (lambda=1) over-discounts and leaves duplicates; we want one kept.
A good intermediate lambda fixes both. We also show GA-sim tracks RIoU vs lambda.
"""

import os
import math

import torch

from aod_yolov8.obb_metric import batch_probiou, batch_riou, batch_ga
from ultralytics.utils.nms import TorchNMS

THR = 0.3


def nms_count(boxes, scores, lam):
    os.environ["AOD_GA_NMS_LAMBDA"] = str(lam)
    return len(TorchNMS.fast_nms(boxes.clone(), scores.clone(), THR, iou_func=batch_ga))


def over_scene():
    boxes = torch.tensor([[60 + k * 22, 100, 40, 40, (45 if k % 2 else 0) * math.pi / 180] for k in range(8)])
    scores = torch.tensor([0.9 - 0.02 * k for k in range(8)])
    return boxes, scores


def under_scene():  # one object, 2 detections, worst-case 45-deg angle noise, same centre
    # true overlap (RIoU) ~0.707 -> a duplicate that must be suppressed; full GA (lam=1)
    # zeroes the similarity at 45 deg and would leave both.
    boxes = torch.tensor([[200., 200, 40, 40, 0.0], [200., 200, 40, 40, 45 * math.pi / 180]])
    scores = torch.tensor([0.9, 0.8])
    return boxes, scores


def main():
    ob, os_ = over_scene()
    un, us = under_scene()
    truth_over = len(TorchNMS.fast_nms(ob.clone(), os_.clone(), THR, iou_func=batch_riou))
    truth_under = len(TorchNMS.fast_nms(un.clone(), us.clone(), THR, iou_func=batch_riou))
    prob_over = len(TorchNMS.fast_nms(ob.clone(), os_.clone(), THR, iou_func=batch_probiou))
    prob_under = len(TorchNMS.fast_nms(un.clone(), us.clone(), THR, iou_func=batch_probiou))

    print(f"NMS@{THR}.   OVER: 8 distinct objects (want 8).   UNDER: 1 object x4 detections (want 1).")
    print(f"  {'method':<16}{'OVER surv':>11}{'UNDER surv':>12}")
    print(f"  {'RIoU (truth)':<16}{truth_over:>11}{truth_under:>12}")
    print(f"  {'ProbIoU (lam0)':<16}{prob_over:>11}{prob_under:>12}   <- over-suppresses")
    for lam in [0.25, 0.5, 0.75, 1.0]:
        o = nms_count(ob, os_, lam)
        u = nms_count(un, us, lam)
        tag = "" if (o == truth_over and u == truth_under) else "   <- fails one mode"
        print(f"  GA lam={lam:<10}{o:>11}{u:>12}{tag}")

    # how well GA-sim tracks the true (RIoU) overlap on the controlled pair, vs lambda
    print("\n  GA-sim vs RIoU on two squares (offset 22px) -- mean |GA - RIoU| over 0..90 deg:")
    base = torch.tensor([[0., 0, 40, 40, 0]])
    angs = torch.arange(0, 91, 5) * math.pi / 180
    riou = torch.tensor([batch_riou(base, torch.tensor([[22., 0, 40, 40, a]]))[0, 0] for a in angs])
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        os.environ["AOD_GA_NMS_LAMBDA"] = str(lam)
        ga = torch.tensor([batch_ga(base, torch.tensor([[22., 0, 40, 40, a]]))[0, 0] for a in angs])
        print(f"    lam={lam:<5} mean|GA-RIoU|={ (ga - riou).abs().mean():.3f}")


if __name__ == "__main__":
    main()
