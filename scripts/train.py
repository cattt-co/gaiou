#!/usr/bin/env python
"""Train YOLOv8-OBB with a selectable rotation-aware regression loss.

This is the AOD-YOLOv8 training entry point. The `--loss` flag selects which OBB
regression loss the (patched) Ultralytics `RotatedBboxLoss` uses, by setting the
`AOD_OBB_LOSS` environment variable that the patch reads:

    probiou  -- Ultralytics default (Gaussian Bhattacharyya)         [baseline]
    riou     -- Rotated IoU polygon overlap                          [baseline]
    criou    -- Complete Rotated IoU (this thesis, eq. 3.18)         [proposed]
    kfiou    -- Kalman-filter IoU (Gaussian)                         [baseline]

Multi-task loss weights follow the thesis (eq. 3.19) and Ultralytics defaults:
    box=7.5, cls=0.5, dfl=1.5.

Environment in the thesis: PyTorch 2.4.0, Python 3.11.8, 2x A40 (48GB), CUDA 12.4.
Pass `--device 0,1` to use both GPUs.

Example:
    python scripts/train.py --loss criou --data configs/dota_rbd.yaml \
        --model yolov8n-obb.pt --imgsz 1024 --epochs 100 --batch 16 --device 0,1
"""

import argparse
import os

LOSSES = ["probiou", "riou", "criou", "kfiou", "fast_criou", "gaiou"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loss", choices=LOSSES, default="criou", help="OBB regression loss")
    ap.add_argument("--data", required=True, help="dataset yaml (e.g. configs/dota_rbd.yaml or DOTAv1.yaml)")
    ap.add_argument("--model", default="yolov8n-obb.pt",
                    help="model: pretrained .pt or arch yaml (yolov8{n,s,m,l,x}-obb)")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="e.g. '0', '0,1', or 'cpu'")
    ap.add_argument("--seed", type=int, default=0, help="random seed (vary for variance runs)")
    ap.add_argument("--name", default=None, help="run name (default: aod_<loss>)")
    ap.add_argument("--project", default="runs/aod", help="results dir")
    ap.add_argument("--amp", action="store_true",
                    help="enable AMP (default off: the MPS AMP check hangs, and fp16 hurts the polygon losses)")
    # multi-task loss gains (thesis eq. 3.19 defaults)
    ap.add_argument("--box", type=float, default=7.5)
    ap.add_argument("--cls", type=float, default=0.5)
    ap.add_argument("--dfl", type=float, default=1.5)
    args = ap.parse_args()

    # The patched Ultralytics RotatedBboxLoss reads this at loss-construction time.
    os.environ["AOD_OBB_LOSS"] = args.loss
    print(f">>> AOD_OBB_LOSS = {args.loss}")

    from pathlib import Path
    from ultralytics import YOLO  # import after env var is set

    # Force an absolute project dir; otherwise Ultralytics resolves it against its
    # own settings `runs_dir`, which may live outside this repo.
    project = str(Path(args.project).expanduser().resolve())

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        amp=args.amp,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        project=project,
        name=args.name or f"aod_{args.loss}",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
