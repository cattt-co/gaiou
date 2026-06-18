"""Pluggable oriented-bounding-box regression loss for YOLOv8-OBB.

This is the heart of the AOD-YOLOv8 thesis: it lets the OBB regression term of
Ultralytics' ``RotatedBboxLoss`` be swapped between four rotation-aware losses so
they can be compared on the same detector / dataset / schedule:

    AOD_OBB_LOSS = probiou | riou | criou | kfiou   (default: probiou)

* probiou  -> Ultralytics default (Gaussian Bhattacharyya, L = 1 - ProbIoU).
* riou     -> Rotated IoU, polygon overlap (Zhou et al. 2019), L = 1 - RIoU.
* criou    -> Complete Rotated IoU (this thesis, eq. 3.18):
                  L = 1 - RIoU + rho^2(b_p, b_g)/c^2 + alpha * v
              i.e. RIoU plus the DIoU center-distance penalty and the CIoU
              aspect-ratio penalty. Implemented as ``cal_ciou`` in
              ``rotated_iou/oriented_iou_loss.py``.
* kfiou    -> Kalman-filter IoU (Yang et al. 2022), eq. 3.8-3.12:
                  L_obb = SmoothL1(centers) + exp(1 - KFIoU)

All four return the per-anchor regression term; ``RotatedBboxLoss.forward``
applies the Task-Aligned weighting (``weight``) and normalization
(``target_scores_sum``) exactly as for the default ProbIoU path.

Boxes are in ``xywhr`` (radians) — the same format the rotated_iou repo's
``box2corners_th`` expects, so OBBs are fed directly with no conversion.
"""

import math
import os

import torch
import torch.nn.functional as F

from ultralytics.utils.metrics import _get_covariance_matrix, probiou

# rotated_iou (vendored) is imported lazily inside obb_regression_loss. Its CUDA-op
# fallback imports aod_yolov8._sort_cpu, so a top-level import here would create a
# circular import when rotated_iou is imported before this package.


def _selected() -> str:
    return os.environ.get("AOD_OBB_LOSS", "probiou").strip().lower()


def kfiou_per_pair(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """KFIoU regression term per box pair.

    Args:
        pred, target: (B, N, 5) xywhr.
    Returns:
        (B, N) tensor: L_obb = SmoothL1(centers) + exp(1 - KFIoU).

    KFIoU models each OBB as a Gaussian with covariance Sigma = [[a, c], [c, b]].
    The Kalman "fused" (intersection) covariance has the closed form
    Sigma_i = Sigma1 (Sigma1 + Sigma2)^-1 Sigma2, so
    det(Sigma_i) = det(Sigma1) det(Sigma2) / det(Sigma1 + Sigma2).
    Box area proxy V_B(Sigma) = 4 * sqrt(det Sigma)  (thesis eq. 3.9), giving
    KFIoU = V_i / (V_1 + V_2 - V_i)                   (thesis eq. 3.10).
    """
    # Covariance components, shape (B*N, 1) each -> reshape back to (B, N).
    shape = pred.shape[:-1]
    a1, b1, c1 = (t.view(shape) for t in _get_covariance_matrix(pred.reshape(-1, 5)))
    a2, b2, c2 = (t.view(shape) for t in _get_covariance_matrix(target.reshape(-1, 5)))

    det1 = (a1 * b1 - c1.pow(2)).clamp(min=0)
    det2 = (a2 * b2 - c2.pow(2)).clamp(min=0)
    aS, bS, cS = a1 + a2, b1 + b2, c1 + c2
    detS = (aS * bS - cS.pow(2)).clamp(min=eps)
    det_i = det1 * det2 / detS

    v1 = 4 * det1.sqrt()
    v2 = 4 * det2.sqrt()
    vi = 4 * det_i.clamp(min=0).sqrt()
    kfiou = vi / (v1 + v2 - vi + eps)

    l_kf = torch.exp(1.0 - kfiou)  # thesis eq. 3.12
    # Center distance term, L_c = SmoothL1(c1, c2)  (thesis eq. 3.11).
    l_c = F.smooth_l1_loss(pred[..., :2], target[..., :2], reduction="none").sum(-1)
    return l_c + l_kf


def angle_term(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Aspect-adaptive explicit orientation penalty (this work).

    The Gaussian (covariance) representation is rotation-invariant for square boxes
    (w == h makes the covariance isotropic), so ProbIoU/KFIoU give zero angle
    gradient there. This term operates on theta directly with the correct
    periodicity: a square's polygon IoU is 90-deg periodic, an elongated box's is
    180-deg periodic, so we blend cos(4*dtheta) and cos(2*dtheta) by the aspect
    asymmetry s = 1 - min(w,h)/max(w,h)  (0 for square, ->1 for elongated).

    Args:
        pred, target: (..., 5) xywhr. Returns (...,) in [0, 1], 0 when aligned.
    """
    dtheta = pred[..., 4] - target[..., 4]
    w, h = target[..., 2], target[..., 3]
    s = 1.0 - torch.min(w, h) / torch.max(w, h).clamp(min=1e-9)
    t2 = (1.0 - torch.cos(2.0 * dtheta)) / 2.0   # 180-deg periodic (elongated)
    t4 = (1.0 - torch.cos(4.0 * dtheta)) / 2.0   # 90-deg periodic (square)
    return s * t2 + (1.0 - s) * t4


def fast_ciou(pred: torch.Tensor, target: torch.Tensor):
    """CRIoU with the axis-aligned (not smallest-rotated) enclosing box (this work).

    CRIoU's dominant extra cost over RIoU is the smallest *rotated* enclosing box
    used for the DIoU c^2 term, which scales poorly. Horizontal CIoU/DIoU already
    use the axis-aligned enclosing box; doing the same here removes that cost and
    makes CRIoU about as cheap as RIoU. pred/target: (B, N, 5). Returns (B, N).
    """
    from rotated_iou.oriented_iou_loss import cal_iou  # lazy: avoids import cycle

    iou, corners1, corners2, _u = cal_iou(pred, target)
    allc = torch.cat([corners1, corners2], dim=2)  # (B,N,8,2)
    xmn = allc[..., 0].min(2)[0]; xmx = allc[..., 0].max(2)[0]
    ymn = allc[..., 1].min(2)[0]; ymx = allc[..., 1].max(2)[0]
    c2 = (xmx - xmn) ** 2 + (ymx - ymn) ** 2
    d2 = (pred[..., 0] - target[..., 0]) ** 2 + (pred[..., 1] - target[..., 1]) ** 2
    w_gt, h_gt = target[..., 2], target[..., 3]
    w_pr, h_pr = pred[..., 2], pred[..., 3]
    v = (4 / (math.pi ** 2)) * (torch.atan(w_gt / h_gt) - torch.atan(w_pr / h_pr)) ** 2
    alpha = v / ((1.0 - iou) + v + 1e-7)
    # The axis-aligned enclosing box is ~1.38x larger than the smallest rotated one,
    # so d2/c2 is diluted ~28%. AOD_FASTCRIOU_DSCALE rescales the distance term to
    # compensate (set ~1.38 to recover CRIoU's distance penalty magnitude).
    dscale = float(os.environ.get("AOD_FASTCRIOU_DSCALE", "1.0"))
    return 1.0 - iou + dscale * d2 / c2.clamp(min=1e-9) + alpha * v


def obb_regression_loss(
    pred_bboxes_fg: torch.Tensor,
    target_bboxes_fg: torch.Tensor,
    weight: torch.Tensor,
    target_scores_sum: torch.Tensor,
) -> torch.Tensor:
    """Weighted, normalized OBB regression loss, selected by ``AOD_OBB_LOSS``.

    Args:
        pred_bboxes_fg:   (M, 5) xywhr, foreground predictions.
        target_bboxes_fg: (M, 5) xywhr, matched targets.
        weight:           (M, 1) Task-Aligned per-anchor weight.
        target_scores_sum: scalar normalizer.
    """
    kind = _selected()

    if kind == "probiou":
        iou = probiou(pred_bboxes_fg, target_bboxes_fg)
        return ((1.0 - iou) * weight).sum() / target_scores_sum

    if kind == "gaiou":
        # Gaussian overlap (ProbIoU) + cheap explicit angle term (this work).
        lam = float(os.environ.get("AOD_GAIOU_LAMBDA", "1.0"))
        iou = probiou(pred_bboxes_fg, target_bboxes_fg).squeeze(-1)  # (M,)
        per = (1.0 - iou) + lam * angle_term(pred_bboxes_fg, target_bboxes_fg)
        return (per.unsqueeze(-1).to(weight.dtype) * weight).sum() / target_scores_sum

    from rotated_iou.oriented_iou_loss import cal_iou, cal_ciou  # lazy: avoids import cycle

    # rotated_iou functions operate on (B, N, 5); add a batch dim and use float32
    # (the polygon kernels are numerically sensitive in half precision).
    p = pred_bboxes_fg.unsqueeze(0).float()
    t = target_bboxes_fg.unsqueeze(0).float()

    if kind == "riou":
        iou, _, _, _ = cal_iou(p, t)
        per = 1.0 - iou  # (1, M)
    elif kind == "criou":
        per, _iou = cal_ciou(p, t)  # (1, M) = 1 - RIoU + d^2/c^2 + alpha*v
    elif kind == "kfiou":
        per = kfiou_per_pair(p, t)  # (1, M)
    elif kind == "fast_criou":
        per = fast_ciou(p, t)  # (1, M) = CRIoU with axis-aligned enclosing box
    else:
        raise ValueError(
            f"Unknown AOD_OBB_LOSS={kind!r}; expected one of "
            f"probiou|riou|criou|kfiou|fast_criou|gaiou"
        )

    per = per.squeeze(0).unsqueeze(-1).to(weight.dtype)  # (M, 1)
    return (per * weight).sum() / target_scores_sum
