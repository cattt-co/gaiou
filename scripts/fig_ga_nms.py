#!/usr/bin/env python
"""Figure: GA-NMS vs ProbIoU-NMS on a row of distinct, differently-oriented squares.

Draws three panels (ProbIoU-NMS, GA-NMS, RIoU-NMS). Survivors are solid; suppressed
boxes are dashed/faded. Shows ProbIoU-NMS wrongly merging distinct square objects
while GA-NMS keeps them, matching the exact polygon NMS. Saves PDF+PNG.
"""

import os
import math

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from aod_yolov8.obb_metric import batch_probiou, batch_riou, batch_ga
from ultralytics.utils.nms import TorchNMS

THR = 0.3
os.environ["AOD_GA_NMS_LAMBDA"] = "0.5"


def corners(box):
    x, y, w, h, t = box
    c, s = math.cos(t), math.sin(t)
    dx = [-w / 2, w / 2, w / 2, -w / 2]
    dy = [-h / 2, -h / 2, h / 2, h / 2]
    return np.array([[x + X * c - Y * s, y + X * s + Y * c] for X, Y in zip(dx, dy)])


def main():
    boxes = torch.tensor([[60 + k * 22, 100, 40, 40, (45 if k % 2 else 0) * math.pi / 180] for k in range(8)])
    scores = torch.tensor([0.9 - 0.02 * k for k in range(8)])

    methods = [("ProbIoU-NMS", batch_probiou), ("GA-NMS (ours)", batch_ga), ("RIoU-NMS (exact)", batch_riou)]
    fig, axes = plt.subplots(3, 1, figsize=(5.0, 3.4), constrained_layout=True)
    for ax, (name, fn) in zip(axes, methods):
        keep = set(TorchNMS.fast_nms(boxes.clone(), scores.clone(), THR, iou_func=fn).tolist())
        for k in range(8):
            cs = corners(boxes[k].tolist())
            if k in keep:
                ax.add_patch(Polygon(cs, closed=True, fill=True, facecolor="#2ca02c",
                                     edgecolor="#1a661a", alpha=0.55, lw=1.3))
            else:
                ax.add_patch(Polygon(cs, closed=True, fill=False, edgecolor="#999999",
                                     ls="--", lw=1.0, alpha=0.85))
        ax.set_title(f"{name}: {len(keep)}/8 survive", fontsize=10, loc="left", pad=2)
        ax.set_xlim(36, 214); ax.set_ylim(70, 130); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#cccccc")
    for ext in ("pdf", "png"):
        fig.savefig(f"paper/fig_ganms.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.02)
    print("wrote paper/fig_ganms.pdf and .png")


if __name__ == "__main__":
    main()
