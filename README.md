# GAIoU: Recovering Square-Object Orientation at Gaussian Speed

Code for *"Recovering Square-Object Orientation at Gaussian Speed: Efficient
Orientation-Aware IoU Losses for Arbitrary-Oriented Object Detection"*
(Kaung Htet San, Aung Ye Kyaw — Cattt Lab).

Gaussian IoU losses (ProbIoU, KFIoU, GWD) are fast but **structurally blind to the
orientation of square objects**: the covariance of an OBB is
`Σ = R diag(w²/12, h²/12) Rᵀ`, so when `w = h` the rotation `R` cancels and the
angle gradient is identically zero. Polygon losses (RIoU, CRIoU) fix this but cost
2.7–7×. This repo provides two cheap alternatives plus an NMS analysis.

| Loss | idea | cost |
|------|------|------|
| **GAIoU** | Gaussian overlap + a cheap, aspect-adaptive explicit angle term — restores the square-orientation gradient the covariance destroys | ~1.2× ProbIoU |
| **Fast-CRIoU** | CRIoU with the axis-aligned (not smallest-rotated) enclosing box + one compensating constant `γ≈1.38` | ~2.7× (RIoU tier) |
| **GA-NMS** | orientation-aware suppression (analysis; neutral on standard aerial data) | ~1.5× ProbIoU |

This work builds on the authors' prior **CRIoU**
([JCSSE 2024](https://ieeexplore.ieee.org/document/10613659),
[code](https://github.com/kaunghtetsan275/aod-yolov8)); CRIoU is included here as a
baseline. The rotated-IoU core is vendored from
[lilanxiao/Rotated_IoU](https://github.com/lilanxiao/Rotated_IoU) (MIT).

## Results (DOTA-RBD, RIoU metric @ TP IoU 0.45, YOLOv8n-OBB, 3 seeds)

| Loss | mAP₅₀ | mAP₅₀₋₉₅ | speed |
|------|------:|---------:|------:|
| **GAIoU** | **0.741 ± 0.002** | 0.377 ± 0.006 | 1.2× |
| **Fast-CRIoU (γ=1.38)** | 0.740 ± 0.005 | **0.390 ± 0.002** | 2.7× |
| CRIoU (prior work) | 0.737 ± 0.006 | 0.377 ± 0.008 | 4–7× |

GAIoU matches CRIoU at Gaussian speed (lowest variance); Fast-CRIoU gives a
seed-significant mAP₅₀₋₉₅ gain at lower cost. Both transfer to the NMS-free YOLO26
backbone.

## How loss / metric selection works

The losses plug into a (patched) Ultralytics YOLOv8/26-OBB via environment variables:

| Env var | Values | Effect |
|---------|--------|--------|
| `AOD_OBB_LOSS` | `probiou`·`riou`·`criou`·`kfiou`·`fast_criou`·`gaiou` | training regression loss |
| `AOD_EVAL_IOU` | `probiou`·`riou` | AP-matching IoU |
| `AOD_NMS_IOU` | `probiou`·`riou`·`ga` | rotated-NMS IoU |
| `AOD_FASTCRIOU_DSCALE` | float (default `1.0`; use `1.38`) | Fast-CRIoU distance rescale γ |
| `AOD_GA_NMS_LAMBDA` | float (default `0.5`) | GA-NMS angle weight |

Defaults reproduce stock Ultralytics behaviour. See `aod_yolov8/obb_loss.py`,
`aod_yolov8/obb_metric.py`, and `PATCHES.md`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/apply_patches.py     # patch installed Ultralytics + register import path
```

## Train & evaluate

```bash
# build the square-object subset + multi-scale tiles, then:
python scripts/train.py --loss gaiou --data configs/dota_rbd.yaml \
    --model yolov8n-obb.pt --imgsz 1024 --epochs 100 --batch 8 --amp --device 0
python scripts/evaluate.py --weights runs/aod/aod_gaiou/weights/best.pt \
    --data configs/dota_rbd.yaml --iou-metric riou --iou 0.45
```
Swap `--model yolo26n-obb.pt` for the NMS-free backbone.

## Analysis scripts
- `scripts/test_angle_aware.py` — ProbIoU's flat square-angle response vs. GAIoU.
- `scripts/profile_criou.py` — per-stage speed breakdown (isolates the enclosing box).
- `scripts/test_ga_nms.py`, `scripts/test_ga_nms_lambda.py` — GA-NMS over/under-suppression and λ calibration.

## Citation
Paper (Zenodo preprint, [10.5281/zenodo.20754295](https://doi.org/10.5281/zenodo.20754295)):

```bibtex
@misc{san2026gaiou,
  title        = {Recovering Square-Object Orientation at Gaussian Speed:
                  Efficient Orientation-Aware IoU Losses for Arbitrary-Oriented Object Detection},
  author       = {San, Kaung Htet and Kyaw, Aung Ye},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20754295},
  url          = {https://doi.org/10.5281/zenodo.20754295},
  note         = {Preprint}
}
```

## License
AGPL-3.0 (see [LICENSE](LICENSE)), consistent with the Ultralytics base it extends.
The vendored `rotated_iou/` retains its original MIT license.
