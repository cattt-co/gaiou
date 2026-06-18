#!/usr/bin/env python
"""Multi-scale tiling for DOTA / DOTA-RBD (thesis section 4.1.1.1 "Rescaling").

DOTA images are huge (800x800 up to 4000x4000), so they are rescaled and cropped
into fixed tiles before training. The thesis rescales each image to 0.5, 1.0 and
1.5x and slices it into overlapping 1024px windows. This is exactly Ultralytics'
``split_trainval`` with ``rates=(0.5, 1.0, 1.5)``.

Data augmentation (section 4.1.1.2): blurring and CLAHE contrast enhancement are
applied **online** during training by Ultralytics' built-in Albumentations
transforms (Blur, MedianBlur, CLAHE, ToGray) — install ``albumentations`` and they
are enabled automatically. No offline augmentation step is needed here.

Input layout (YOLO-OBB):  <src>/images/{train,val}, <src>/labels/{train,val}
Output layout (tiled):    <out>/images/{train,val}, <out>/labels/{train,val}
"""

import argparse
from pathlib import Path

from ultralytics.data.split_dota import split_trainval


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="dataset root (DOTAv1 or DOTA-RBD), YOLO-OBB format")
    ap.add_argument("--out", required=True, help="output root for tiled dataset")
    ap.add_argument("--crop-size", type=int, default=1024, help="base tile size (default 1024)")
    ap.add_argument("--gap", type=int, default=200, help="overlap between tiles (default 200)")
    ap.add_argument("--rates", type=float, nargs="+", default=[0.5, 1.0, 1.5],
                    help="rescale factors before tiling (thesis: 0.5 1.0 1.5)")
    args = ap.parse_args()

    print(f"Tiling {args.src} -> {args.out}")
    print(f"  crop_size={args.crop_size} gap={args.gap} rates={tuple(args.rates)}")
    split_trainval(
        data_root=str(Path(args.src).expanduser()),
        save_dir=str(Path(args.out).expanduser()),
        crop_size=args.crop_size,
        gap=args.gap,
        rates=tuple(args.rates),
    )
    print("Done. Point your dataset yaml `path:` at the output root.")


if __name__ == "__main__":
    main()
