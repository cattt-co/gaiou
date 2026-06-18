"""AOD-YOLOv8: arbitrary-oriented object detection with selectable OBB losses.

Public surface:
    obb_regression_loss  -- pluggable OBB regression term (probiou/riou/criou/kfiou)
    batch_iou            -- pluggable pairwise OBB IoU for evaluation (probiou/riou)

Selection is via environment variables so a single patched Ultralytics install can
run the whole comparison study without code edits between runs:
    AOD_OBB_LOSS = probiou | riou | criou | kfiou   (training loss)
    AOD_EVAL_IOU = probiou | riou                    (evaluation metric)
    AOD_NMS_IOU  = probiou | riou                    (rotated NMS, defaults to AOD_EVAL_IOU)
"""

from .obb_loss import obb_regression_loss, kfiou_per_pair, angle_term, fast_ciou
from .obb_metric import batch_iou, batch_riou

__all__ = [
    "obb_regression_loss", "kfiou_per_pair", "angle_term", "fast_ciou",
    "batch_iou", "batch_riou",
]
