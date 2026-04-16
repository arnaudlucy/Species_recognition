# train_yolo.py
import os
os.system("pip install ultralytics==8.3.78 --quiet")

import comet_ml
from ultralytics import YOLO
import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()


    # Charger le modèle YOLOv11
model = YOLO('yolo11l.pt')

model.train(
    data='/home/larnaud/Yolo/bbox/data.yaml',
    epochs=150,
    plots=True,
    weight_decay=0.005,
    lr0=0.001, 
    lrf=0.05,
    iou=0.7,
    dfl=1.0,
    mosaic=0.5,
    close_mosaic=10,
    #classes=[72,33,26,19,85,61,65,32,37,38,12,86,73,68,84],
    project= '/home/larnaud/Yolo/bbox',
    val=True,
    auto_augment="randaugment",
    mixup=0.1,
    fliplr=0.5,
    copy_paste=1,
    copy_paste_mode="flip" 
    
)



