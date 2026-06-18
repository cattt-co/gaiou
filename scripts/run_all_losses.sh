#!/usr/bin/env bash
# Fine-tune YOLOv8n-OBB on DOTA-RBD with each of the four thesis losses.
# Starts from the DOTAv1-pretrained yolov8n-obb.pt so short runs already detect well.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DATA=datasets/dota_rbd_local.yaml
EPOCHS=${EPOCHS:-12}
IMGSZ=${IMGSZ:-896}
BATCH=${BATCH:-8}
DEVICE=${DEVICE:-mps}

for LOSS in probiou riou criou kfiou; do
  echo "=================================================================="
  echo ">>> Fine-tuning with $LOSS  ($(date +%H:%M:%S))"
  echo "=================================================================="
  $PY scripts/train.py --loss "$LOSS" --data "$DATA" --model yolov8n-obb.pt \
      --imgsz "$IMGSZ" --epochs "$EPOCHS" --batch "$BATCH" --device "$DEVICE" \
      --project runs/aod --name "aod_$LOSS" 2>&1 \
    | grep -E "AOD_OBB|Epoch|/$EPOCHS|mAP50|all |baseball|roundabout|Error|Traceback" \
    | tail -40
  echo ">>> $LOSS done ($(date +%H:%M:%S)); weights: runs/aod/aod_$LOSS/weights/best.pt"
done
echo "ALL FOUR TRAININGS COMPLETE"
