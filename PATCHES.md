# Ultralytics in-place patches

The comparison study needs four edits to the installed Ultralytics package. They
all delegate to `aod_yolov8/` and default to stock behaviour, so they are safe.
Re-apply after any `pip install -U ultralytics`. Verified against **ultralytics
8.4.60**; line numbers drift between versions — match on the surrounding code.

Make the repo importable first (once per venv):
```bash
echo "$(pwd)" > "$(python -c 'import site;print(site.getsitepackages()[0])')/aod.pth"
```

### 1. Training loss — `ultralytics/utils/loss.py`, `RotatedBboxLoss.forward`
```diff
         weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
-        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
-        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
+        # AOD-YOLOv8: pluggable OBB regression loss (probiou/riou/criou/kfiou) via
+        # AOD_OBB_LOSS env var. Defaults to the original ProbIoU behaviour.
+        from aod_yolov8.obb_loss import obb_regression_loss
+        loss_iou = obb_regression_loss(
+            pred_bboxes[fg_mask], target_bboxes[fg_mask], weight, target_scores_sum
+        )
```

### 2. Eval AP-matching IoU — `ultralytics/models/yolo/obb/val.py`, `_process_batch`
```diff
-        iou = batch_probiou(batch["bboxes"], preds["bboxes"])
+        # AOD-YOLOv8: AP-matching IoU selectable via AOD_EVAL_IOU (probiou/riou).
+        from aod_yolov8.obb_metric import batch_iou
+        iou = batch_iou(batch["bboxes"], preds["bboxes"])
```

### 3. Merge-NMS IoU — `ultralytics/models/yolo/obb/val.py` (DOTA merge path)
```diff
-                i = TorchNMS.fast_nms(b, scores, 0.3, iou_func=batch_probiou)
+                from aod_yolov8.obb_metric import batch_nms_iou  # AOD: selectable NMS IoU
+                i = TorchNMS.fast_nms(b, scores, 0.3, iou_func=batch_nms_iou)
```

### 4. Inference rotated-NMS IoU — `ultralytics/utils/nms.py`
```diff
             boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
-            i = TorchNMS.fast_nms(boxes, scores, iou_thres, iou_func=batch_probiou)
+            from aod_yolov8.obb_metric import batch_nms_iou  # AOD: selectable NMS IoU
+            i = TorchNMS.fast_nms(boxes, scores, iou_thres, iou_func=batch_nms_iou)
```

(Two further `batch_probiou` NMS call-sites exist in `engine/exporter.py` and
`trackers/utils/matching.py`; the thesis doesn't touch export or tracking, so
they are left unpatched.)
