#!/usr/bin/env python
"""Build DOTA-RBD: the square-object subset of DOTAv1 (thesis section 4.1.1).

DOTA-RBD keeps only images containing **roundabout** or **baseball diamond** and
relabels them as a 2-class problem. These two categories are (near-)isometric /
square objects, which is exactly where Gaussian-based losses (ProbIoU, KFIoU)
struggle to recover orientation and where RIoU/CRIoU are expected to win
(Table 4.2). The thesis subset has 389 images (281 train / 108 val); 175
baseball-diamond and 231 roundabout instances.

Input  (YOLO-OBB format, e.g. Ultralytics' DOTAv1 download):
    <dotav1>/images/{train,val}/*.jpg
    <dotav1>/labels/{train,val}/*.txt   # lines: cls x1 y1 x2 y2 x3 y3 x4 y4 (normalized)

Output (same layout, filtered + remapped to 2 classes):
    <out>/images/{train,val}/...
    <out>/labels/{train,val}/...

DOTAv1 class ids used as the source filter:
    3 = baseball diamond   ->  0
    12 = roundabout        ->  1
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path

# DOTAv1 source class id -> DOTA-RBD class id (and name)
SRC_TO_DST = {3: 0, 12: 1}
DST_NAMES = {0: "baseball diamond", 1: "roundabout"}


def convert_split(src_root: Path, out_root: Path, split: str, link: bool) -> Counter:
    """Filter one split; return per-class instance counts kept."""
    src_im = src_root / "images" / split
    src_lb = src_root / "labels" / split
    out_im = out_root / "images" / split
    out_lb = out_root / "labels" / split
    out_im.mkdir(parents=True, exist_ok=True)
    out_lb.mkdir(parents=True, exist_ok=True)
    assert src_lb.exists(), f"missing labels dir: {src_lb}"

    counts: Counter = Counter()
    n_imgs = 0
    for lb_file in sorted(src_lb.glob("*.txt")):
        kept_lines = []
        for line in lb_file.read_text().splitlines():
            parts = line.split()
            if len(parts) < 9:
                continue
            cls = int(float(parts[0]))
            if cls in SRC_TO_DST:
                dst = SRC_TO_DST[cls]
                kept_lines.append(f"{dst} " + " ".join(parts[1:]))
                counts[dst] += 1
        if not kept_lines:
            continue  # drop images with no roundabout/baseball-diamond instance

        # find the matching image (any common extension)
        img = next((p for ext in (".jpg", ".png", ".jpeg", ".tif", ".bmp")
                    if (p := src_im / (lb_file.stem + ext)).exists()), None)
        if img is None:
            print(f"  ! no image for {lb_file.name}, skipping")
            continue

        (out_lb / lb_file.name).write_text("\n".join(kept_lines) + "\n")
        dst_img = out_im / img.name
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        if link:
            dst_img.symlink_to(img.resolve())
        else:
            shutil.copy2(img, dst_img)
        n_imgs += 1

    print(f"[{split}] images kept: {n_imgs} | "
          + ", ".join(f"{DST_NAMES[k]}={counts[k]}" for k in sorted(counts)))
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="DOTAv1 root (YOLO-OBB format) with images/ and labels/")
    ap.add_argument("--out", required=True, help="output DOTA-RBD root")
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    args = ap.parse_args()

    src_root, out_root = Path(args.src).expanduser(), Path(args.out).expanduser()
    total: Counter = Counter()
    for split in ("train", "val"):
        if (src_root / "labels" / split).exists():
            total += convert_split(src_root, out_root, split, link=not args.copy)
        else:
            print(f"[{split}] not found in source, skipping")

    print("\nDOTA-RBD totals: " + ", ".join(f"{DST_NAMES[k]}={total[k]}" for k in sorted(total)))
    print(f"Wrote dataset to {out_root}")
    print("Reference dataset yaml: configs/dota_rbd.yaml")


if __name__ == "__main__":
    main()
