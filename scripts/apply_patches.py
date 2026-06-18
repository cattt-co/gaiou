#!/usr/bin/env python
"""Apply the AOD-YOLOv8 patches to the installed Ultralytics package (idempotent).

Re-run after any `pip install -U ultralytics`. Mirrors PATCHES.md. Verifies each
target string exists before editing; skips edits already applied. Also writes the
`aod.pth` file so the patched Ultralytics can import `aod_yolov8` / `rotated_iou`.

Usage:  python scripts/apply_patches.py
"""

import site
import sys
from pathlib import Path

import ultralytics

UL = Path(ultralytics.__file__).parent
REPO = Path(__file__).resolve().parent.parent

EDITS = [
    # (file, marker_if_already_applied, old, new)
    (
        UL / "utils" / "loss.py",
        "from aod_yolov8.obb_loss import obb_regression_loss",
        '        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])\n'
        '        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum',
        '        # AOD-YOLOv8: pluggable OBB regression loss via AOD_OBB_LOSS.\n'
        '        from aod_yolov8.obb_loss import obb_regression_loss\n'
        '        loss_iou = obb_regression_loss(\n'
        '            pred_bboxes[fg_mask], target_bboxes[fg_mask], weight, target_scores_sum\n'
        '        )',
    ),
    (
        UL / "models" / "yolo" / "obb" / "val.py",
        "from aod_yolov8.obb_metric import batch_iou",
        '        iou = batch_probiou(batch["bboxes"], preds["bboxes"])',
        '        from aod_yolov8.obb_metric import batch_iou  # AOD: selectable eval IoU\n'
        '        iou = batch_iou(batch["bboxes"], preds["bboxes"])',
    ),
    (
        UL / "models" / "yolo" / "obb" / "val.py",
        "fast_nms(b, scores, 0.3, iou_func=batch_nms_iou)",
        '                i = TorchNMS.fast_nms(b, scores, 0.3, iou_func=batch_probiou)',
        '                from aod_yolov8.obb_metric import batch_nms_iou  # AOD: selectable NMS IoU\n'
        '                i = TorchNMS.fast_nms(b, scores, 0.3, iou_func=batch_nms_iou)',
    ),
    (
        UL / "utils" / "nms.py",
        "fast_nms(boxes, scores, iou_thres, iou_func=batch_nms_iou)",
        '            i = TorchNMS.fast_nms(boxes, scores, iou_thres, iou_func=batch_probiou)',
        '            from aod_yolov8.obb_metric import batch_nms_iou  # AOD: selectable NMS IoU\n'
        '            i = TorchNMS.fast_nms(boxes, scores, iou_thres, iou_func=batch_nms_iou)',
    ),
]


def main() -> int:
    ok = True
    for path, marker, old, new in EDITS:
        text = path.read_text()
        if marker in text:
            print(f"[ok] already patched: {path.relative_to(UL.parent)}")
            continue
        if old not in text:
            print(f"[FAIL] TARGET NOT FOUND in {path.relative_to(UL.parent)} — Ultralytics "
                  f"version differs; patch manually per PATCHES.md")
            ok = False
            continue
        path.write_text(text.replace(old, new, 1))
        print(f"[ok] patched: {path.relative_to(UL.parent)}")

    # make the repo importable from the venv
    sp = Path(site.getsitepackages()[0]) / "aod.pth"
    sp.write_text(str(REPO) + "\n")
    print(f"[ok] wrote {sp} -> {REPO}")

    # sanity import
    try:
        import aod_yolov8  # noqa: F401
        print("[ok] aod_yolov8 importable")
    except Exception as e:
        print(f"[FAIL] import aod_yolov8 failed: {e}")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
