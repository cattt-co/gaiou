"""Selectable pairwise OBB IoU used for *evaluation* (AP matching and NMS).

Table 4.2 of the thesis reports every loss evaluated under two metrics: ProbIoU
and RIoU. Ultralytics computes both the validation TP/FP matching and rotated NMS
with ``batch_probiou``. This module provides a drop-in ``batch_iou(obb1, obb2)``
that returns an (N, M) IoU matrix using either metric:

    AOD_EVAL_IOU = probiou | riou   (default: probiou)

RIoU here is the exact polygon overlap from the rotated_iou repo (the same metric
used in the loss), so the evaluation is consistent with what CRIoU/RIoU optimize.
Because the AP matching threshold in the thesis is 0.45, an accurate (not
Gaussian-approximate) IoU matters for the square-object DOTA-RBD comparison.
"""

import os

import numpy as np
import torch

from ultralytics.utils.metrics import batch_probiou

# rotated_iou is imported lazily inside batch_riou to avoid a circular import
# (its CUDA-op fallback imports aod_yolov8._sort_cpu).


def _selected() -> str:
    return os.environ.get("AOD_EVAL_IOU", "probiou").strip().lower()


def batch_riou(obb1: torch.Tensor, obb2: torch.Tensor) -> torch.Tensor:
    """Pairwise Rotated IoU. obb1 (N,5), obb2 (M,5) xywhr -> (N, M)."""
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2
    from rotated_iou.oriented_iou_loss import cal_iou  # lazy: avoids import cycle

    n, m = obb1.shape[0], obb2.shape[0]
    if n == 0 or m == 0:
        return obb1.new_zeros((n, m))
    # Broadcast to all N*M pairs, run the polygon IoU, reshape to (N, M).
    b1 = obb1.unsqueeze(1).expand(n, m, 5).reshape(1, n * m, 5).float()
    b2 = obb2.unsqueeze(0).expand(n, m, 5).reshape(1, n * m, 5).float()
    iou, _, _, _ = cal_iou(b1, b2)
    return iou.view(n, m).to(obb1.dtype if obb1.is_floating_point() else torch.float32)


def batch_ga(obb1, obb2):
    """Orientation-aware NMS similarity (this work), Gaussian-speed.

    Plain ProbIoU is orientation-blind on squares, so its NMS over-suppresses two
    adjacent square objects that differ only in orientation. GA-sim discounts the
    Gaussian overlap by the aspect-adaptive angle disagreement, so differently
    oriented neighbours are kept. Closed-form (~ProbIoU cost), no polygon.

        GA-sim(i, j) = ProbIoU(i, j) * (1 - lambda_nms * angle_term(theta_i - theta_j))

    ``lambda_nms`` (env ``AOD_GA_NMS_LAMBDA``, default 0.5) controls how much the
    angle disagreement discounts the overlap. lambda=0 recovers plain ProbIoU
    (over-suppresses differently-oriented neighbours); lambda=1 fully zeroes the
    similarity at the symmetry angle (under-suppresses true duplicates that have
    angle noise). An intermediate value tracks the true polygon overlap and avoids
    both failure modes.
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2
    lam = float(os.environ.get("AOD_GA_NMS_LAMBDA", "0.5"))
    p = batch_probiou(obb1, obb2)  # (N, M)
    dth = obb1[..., 4].unsqueeze(-1) - obb2[..., 4].unsqueeze(0)  # (N, M)
    w1, h1 = obb1[..., 2], obb1[..., 3]
    s = (1.0 - torch.min(w1, h1) / torch.max(w1, h1).clamp(min=1e-9)).unsqueeze(-1)  # (N,1)
    t2 = (1.0 - torch.cos(2.0 * dth)) / 2.0
    t4 = (1.0 - torch.cos(4.0 * dth)) / 2.0
    a = s * t2 + (1.0 - s) * t4
    return p * (1.0 - lam * a)


def batch_iou(obb1, obb2):
    """Dispatch to the metric chosen by ``AOD_EVAL_IOU`` (probiou default)."""
    if _selected() == "riou":
        return batch_riou(obb1, obb2)
    return batch_probiou(obb1, obb2)


def batch_nms_iou(obb1, obb2):
    """IoU used by rotated NMS, chosen by ``AOD_NMS_IOU`` (falls back to ``AOD_EVAL_IOU``)."""
    kind = os.environ.get("AOD_NMS_IOU", os.environ.get("AOD_EVAL_IOU", "probiou")).strip().lower()
    if kind == "riou":
        return batch_riou(obb1, obb2)
    if kind == "ga":
        return batch_ga(obb1, obb2)
    return batch_probiou(obb1, obb2)
