#!/usr/bin/env python
"""Demonstrate OBB tracking by panning a window across a large aerial image.

Ultralytics OBB models support tracking (ByteTrack/BoT-SORT, with oriented-box
association). DOTA is still images, so we synthesize camera motion by sliding a
crop window across one big image; objects translate consistently frame-to-frame
and the tracker assigns persistent IDs. Output is an annotated mp4.
"""

import argparse

import cv2
import numpy as np
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="/tmp/obb_track.mp4")
    ap.add_argument("--model", default="yolov8n-obb.pt")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--frames", type=int, default=72)
    ap.add_argument("--pan", type=int, default=360, help="total pan distance in px (small steps -> stable tracks)")
    ap.add_argument("--origin", type=int, nargs=2, default=None, help="top-left start y x")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--save-frames", default=None, help="dir to also dump sample frames")
    args = ap.parse_args()

    img = cv2.imread(args.src)
    H, W = img.shape[:2]
    win = min(args.window, H, W)
    # gentle diagonal pan: small per-frame steps so objects shift only a few px
    # and the tracker keeps stable IDs (large steps fragment dense small objects).
    pan = min(args.pan, H - win, W - win)
    oy, ox = args.origin if args.origin else ((H - win - pan) // 2, (W - win - pan) // 2)
    oy, ox = max(0, oy), max(0, ox)
    ys = (oy + np.linspace(0, pan, args.frames)).astype(int)
    xs = (ox + np.linspace(0, pan, args.frames)).astype(int)
    print(f"image {W}x{H}, window {win}, pan {pan}px over {args.frames} frames "
          f"(~{pan/args.frames:.1f}px/frame)")

    model = YOLO(args.model)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (win, win))
    ids_seen = set()
    saved = 0
    for i, (y, x) in enumerate(zip(ys, xs)):
        crop = img[y:y + win, x:x + win].copy()
        r = model.track(crop, persist=True, tracker=args.tracker, conf=args.conf,
                        imgsz=win, device=args.device, verbose=False)[0]
        n = 0
        if r.obb is not None and len(r.obb):
            n = len(r.obb)
            if r.obb.id is not None:
                ids_seen.update(r.obb.id.int().tolist())
        vis = r.plot()  # draws OBBs + track IDs
        cv2.rectangle(vis, (0, 0), (win, 40), (0, 0, 0), -1)
        cv2.putText(vis, f"frame {i+1}/{args.frames}  dets {n}  unique tracks {len(ids_seen)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        vw.write(vis)
        if args.save_frames and i % (args.frames // 4) == 0:
            cv2.imwrite(f"{args.save_frames}/track_{i:03d}.jpg", vis)
            saved += 1
    vw.release()
    print(f"wrote {args.out}  ({args.frames} frames, {len(ids_seen)} unique track IDs)")


if __name__ == "__main__":
    main()
