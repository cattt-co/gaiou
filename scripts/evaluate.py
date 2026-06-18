#!/usr/bin/env python
"""Evaluate a YOLOv8-OBB model with a selectable IoU metric.

Reproduces Table 4.1 / 4.2: every model is scored under a chosen evaluation IoU,
so a CRIoU-trained model can be measured with either the default ProbIoU metric or
the polygon-exact RIoU metric (the latter being the fairer metric for square
objects in DOTA-RBD).

    --iou-metric probiou | riou     -> AP-matching IoU (AOD_EVAL_IOU)
    --nms-metric probiou | riou     -> rotated NMS IoU (AOD_NMS_IOU; default = iou-metric)

The TP threshold is 0.45 as in the thesis (section 3.4.1). mAP@50 and mAP@50-95
are reported by Ultralytics' OBBValidator.

Example:
    python scripts/evaluate.py --weights runs/aod/aod_criou/weights/best.pt \
        --data configs/dota_rbd.yaml --iou-metric riou
"""

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="trained .pt checkpoint")
    ap.add_argument("--data", required=True, help="dataset yaml")
    ap.add_argument("--iou-metric", choices=["probiou", "riou"], default="riou",
                    help="IoU used for AP matching (thesis Table 4.2 metric column)")
    ap.add_argument("--nms-metric", choices=["probiou", "riou"], default=None,
                    help="IoU used in rotated NMS (default: same as --iou-metric)")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--iou", type=float, default=0.45, help="TP IoU threshold (thesis: 0.45)")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()

    # patched Ultralytics reads these in obb/val.py and nms.py
    os.environ["AOD_EVAL_IOU"] = args.iou_metric
    os.environ["AOD_NMS_IOU"] = args.nms_metric or args.iou_metric
    print(f">>> AOD_EVAL_IOU = {os.environ['AOD_EVAL_IOU']} | AOD_NMS_IOU = {os.environ['AOD_NMS_IOU']}")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        iou=args.iou,
        split=args.split,
    )
    print(f"\nmAP@50    = {metrics.box.map50:.4f}")
    print(f"mAP@50-95 = {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
