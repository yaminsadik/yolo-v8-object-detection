from os import putenv
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")


import torch
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
model.train(data="data.yaml", epochs=100)  # train the model