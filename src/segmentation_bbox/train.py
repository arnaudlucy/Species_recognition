import os
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO

# =============================================================================
# train.py
# K-fold cross-validation training script for YOLO11 segmentation models.
# Trains one model per fold, saves per-class metrics as CSV, and prints
# mean mAP, mAP@0.5:0.95 and recall across all folds.
#
# Prerequisites:
#   - Pre-split fold directories at OUTPUT_DIR/fold_<k>/
#   - Each fold directory must contain a data_yaml.yaml file
#   - Base YOLO11 segmentation weights at BASE_MODEL_PATH
#
# Outputs (per fold):
#   - fold_<k>/train_results/yolo_fold_<k>/ : training artefacts
#   - fold_<k>/model.pt                     : saved weights
#   - fold_<k>/per_class_metrics_fold<k>.csv: per-class precision/recall/mAP
#
# Run:
#   python src/segmentation_bbox/train.py
# =============================================================================

# =============================================================================
# CONFIGURATION — edit only this block
# =============================================================================
BASE_MODEL_PATH = os.path.join('Classifiers', 'yolo11l-seg.pt')
OUTPUT_DIR      = os.path.join('data', 'segmentation', 'k_cross')
K_FOLDS         = 15
EPOCHS          = 100
IMG_SIZE        = 640
BATCH_TRAIN     = 64
BATCH_VAL       = 32
RANDOM_SEED     = 42
# =============================================================================

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

fold_map50, fold_map50_95, fold_recalls = [], [], []

for fold in range(K_FOLDS):
    print('\n[INFO] Fold {}/{} starting...'.format(fold + 1, K_FOLDS))

    fold_dir  = os.path.join(OUTPUT_DIR, 'fold_{}'.format(fold))
    data_yaml = os.path.join(fold_dir, 'data_yaml.yaml')

    model = YOLO(BASE_MODEL_PATH)

    model.train(
        data          = data_yaml,
        imgsz         = IMG_SIZE,
        epochs        = EPOCHS,
        device        = 0,
        batch         = BATCH_TRAIN,
        weight_decay  = 0.001,
        mixup         = 0.2,
        fliplr        = 0.5,
        close_mosaic  = 20,
        lr0           = 0.0051,
        lrf           = 0.001,
        iou           = 0.6,
        auto_augment  = 'randaugment',
        warmup_epochs = 5,
        momentum      = 0.85,
        dropout       = 0.1,
        box           = 5,
        dfl           = 1.0,
        patience      = 30,
        project       = os.path.join(fold_dir, 'train_results'),
        name          = 'yolo_fold_{}'.format(fold),
        exist_ok      = True,
        seed          = RANDOM_SEED,
    )

    print('[OK] Fold {} training complete.'.format(fold))
    model.save(os.path.join(fold_dir, 'model.pt'))

    # Validation
    val_results = model.val(data=data_yaml, batch=BATCH_VAL)

    # Collect per-class metrics
    results = []
    names = val_results.names
    for i in range(len(names)):
        cls_name = names[i]

        try:
            box_p     = val_results.box.p[i]
            box_r     = val_results.box.r[i]
            box_map50 = val_results.box.ap50[i]
            box_map   = val_results.box.ap[i]
        except IndexError:
            box_p = box_r = box_map50 = box_map = None

        try:
            seg        = val_results.seg if hasattr(val_results, 'seg') else None
            mask_p     = seg.p[i]    if seg else None
            mask_r     = seg.r[i]    if seg else None
            mask_map50 = seg.ap50[i] if seg else None
            mask_map   = seg.ap[i]   if seg else None
        except (IndexError, AttributeError):
            mask_p = mask_r = mask_map50 = mask_map = None

        results.append({
            'class':          cls_name,
            'box_precision':  box_p,
            'box_recall':     box_r,
            'box_mAP50':      box_map50,
            'box_mAP50-95':   box_map,
            'mask_precision': mask_p,
            'mask_recall':    mask_r,
            'mask_mAP50':     mask_map50,
            'mask_mAP50-95':  mask_map,
        })

    csv_path = os.path.join(fold_dir, 'per_class_metrics_fold{}.csv'.format(fold))
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print('[OK] Per-class metrics saved: {}'.format(csv_path))

    fold_map50.append(val_results.box.map50)
    fold_map50_95.append(val_results.box.map)
    fold_recalls.append(val_results.box.mr)

    # Predictions on train and val images
    for split in ('train', 'val'):
        src_dir  = os.path.join(fold_dir, 'images', split)
        pred_dir = os.path.join(fold_dir, 'preds', split)
        os.makedirs(pred_dir, exist_ok=True)
        print('[INFO] Running predictions on {} images (fold {})...'.format(split, fold))
        model.predict(
            source   = src_dir,
            save     = True,
            save_txt = True,
            conf     = 0.25,
            project  = pred_dir,
            name     = '',
            exist_ok = True,
        )

# Final summary
print('\n[RESULTS] K-Fold Cross-Validation Summary:')
print('  mAP@0.5        : {:.4f} +/- {:.4f}'.format(np.mean(fold_map50), np.std(fold_map50)))
print('  mAP@0.5:0.95   : {:.4f} +/- {:.4f}'.format(np.mean(fold_map50_95), np.std(fold_map50_95)))
print('  Recall (mean)  : {:.4f} +/- {:.4f}'.format(np.mean(fold_recalls), np.std(fold_recalls)))
print('[OK] K-fold training and validation complete.')
