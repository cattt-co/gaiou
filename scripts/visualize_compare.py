#!/usr/bin/env python
"""Run each loss's model on sample images and draw OBBs side-by-side for comparison.

For each chosen image, produces one panel:  [ground truth | probiou | riou | criou | kfiou]
with predicted oriented boxes drawn and the per-image detection count labelled.

Example:
    python scripts/visualize_compare.py --data datasets/dota_rbd_local.yaml \
        --runs runs/aod --out viz --num 6
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np

LOSSES = ["probiou", "riou", "criou", "kfiou"]
NAMES = {0: "baseball diamond", 1: "roundabout"}
COLORS = {0: (0, 200, 0), 1: (0, 140, 255)}  # BGR: green=BD, orange=RA


def draw_obb_poly(img, poly, color, label=None):
    poly = poly.reshape(-1, 2).astype(np.int32)
    cv2.polylines(img, [poly], isClosed=True, color=color, thickness=3)
    if label:
        x, y = poly[0]
        cv2.putText(img, label, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def load_gt(label_file, w, h):
    """YOLO-OBB label -> list of (cls, 4x2 px polygon)."""
    out = []
    if not label_file.exists():
        return out
    for line in label_file.read_text().splitlines():
        p = line.split()
        if len(p) < 9:
            continue
        cls = int(float(p[0]))
        pts = np.array([float(v) for v in p[1:9]], dtype=np.float32).reshape(4, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        out.append((cls, pts))
    return out


def panel(img, title):
    h, w = img.shape[:2]
    bar = np.full((40, w, 3), 30, np.uint8)
    cv2.putText(bar, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return np.vstack([bar, img])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset yaml (for val image dir)")
    ap.add_argument("--runs", default="runs/aod", help="dir with aod_<loss>/weights/best.pt")
    ap.add_argument("--out", default="viz", help="output dir for comparison images")
    ap.add_argument("--num", type=int, default=6, help="how many sample images")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--nms-metric", choices=["probiou", "riou"], default="probiou")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.environ["AOD_NMS_IOU"] = args.nms_metric
    import yaml
    from ultralytics import YOLO

    cfg = yaml.safe_load(Path(args.data).read_text())
    root = Path(cfg["path"])
    val_dir = root / cfg.get("val", "images/val")
    imgs = sorted(val_dir.glob("*.jpg"))
    random.Random(args.seed).shuffle(imgs)
    imgs = imgs[: args.num]

    # load available models
    models = {}
    for L in LOSSES:
        w = Path(args.runs) / f"aod_{L}" / "weights" / "best.pt"
        if w.exists():
            models[L] = YOLO(str(w))
        else:
            print(f"  ! missing weights for {L}: {w}")
    if not models:
        raise SystemExit("No trained models found. Run scripts/run_all_losses.sh first.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for im_path in imgs:
        img0 = cv2.imread(str(im_path))
        h, w = img0.shape[:2]
        lbl = root / "labels" / "val" / (im_path.stem + ".txt")

        # Crop around the ground-truth objects (+padding) so large aerial scenes
        # still show the square objects clearly. Falls back to the full image.
        gts = load_gt(lbl, w, h)
        if gts:
            allpts = np.concatenate([p for _, p in gts], axis=0)
            x0, y0 = allpts.min(0)
            x1, y1 = allpts.max(0)
            pad = 0.6 * max(x1 - x0, y1 - y0) + 40
            cx0, cy0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
            cx1, cy1 = min(w, int(x1 + pad)), min(h, int(y1 + pad))
            img0 = img0[cy0:cy1, cx0:cx1]
            h, w = img0.shape[:2]
            crop_off = np.array([cx0, cy0], dtype=np.float32)
        else:
            crop_off = np.array([0, 0], dtype=np.float32)

        # ground-truth panel (shift full-image coords into the crop)
        gt = img0.copy()
        for cls, pts in gts:
            draw_obb_poly(gt, pts - crop_off, COLORS[cls])
        cols = [panel(gt, "ground truth")]

        # one panel per loss
        for L in LOSSES:
            if L not in models:
                continue
            vis = img0.copy()
            r = models[L].predict(str(im_path), imgsz=args.imgsz, conf=args.conf,
                                   device=args.device, verbose=False)[0]
            n = 0
            if r.obb is not None and len(r.obb):
                polys = r.obb.xyxyxyxy.cpu().numpy()  # (N,4,2)
                clss = r.obb.cls.cpu().numpy().astype(int)
                confs = r.obb.conf.cpu().numpy()
                n = len(polys)
                for poly, c, cf in zip(polys, clss, confs):
                    draw_obb_poly(vis, poly - crop_off, COLORS.get(int(c), (255, 0, 0)), f"{cf:.2f}")
            cols.append(panel(vis, f"{L}  ({n} det)"))

        # resize all panels to common height and concat horizontally
        target_h = min(c.shape[0] for c in cols)
        cols = [cv2.resize(c, (int(c.shape[1] * target_h / c.shape[0]), target_h)) for c in cols]
        grid = np.hstack(cols)
        dst = out / f"{im_path.stem}_compare.jpg"
        cv2.imwrite(str(dst), grid)
        print(f"wrote {dst}")

    print(f"\nDone. Comparison images in {out}/")


if __name__ == "__main__":
    main()
