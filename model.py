from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results

import os
from torch.cuda import is_available

# path
DATA_PATH = "data_yolo26"
DATA_SETTINGS_PATH = DATA_PATH + "/data.yaml"
TRAIN_PATH = DATA_PATH + "/train"
VAL_PATH = DATA_PATH + "/valid"

MODEL_PATH = "models"
MODEL_NAME = "phone_detection_yolo26_v1.0.engine" if is_available() else "phone_detection_yolo26_v1.0.pt"

# training
def train(data_path: str = DATA_SETTINGS_PATH, train_path: str = TRAIN_PATH, val_path: str = VAL_PATH, model_name: str = MODEL_NAME):
    model = YOLO("yolo26b.pt")
    device = 0 # os.cpu_count()

    model.train(data=data_path, epochs=100, imgsz=512, batch=20, device=device, name=model_name)
    plot_results(train_path)

    model.val()
    plot_results(val_path)

    format = "engine" if is_available() else "onnx"
    model.export(format=format)
    # model.save()

# if phone detected - true
def get_phones(img, model_path: str = MODEL_PATH, model_name: str = MODEL_NAME) -> bool:
    # pretrain model if didn't do it before
    full_model_path = model_path + "/" + model_name
    if not os.path.exists(full_model_path):
        train()
    
    # process img
    model = YOLO(full_model_path)
    img_results = model(img)

    for name in img_results[0].names:
        if name == "phone":
            return True
    return False
    # return True if len(img_results[0]) > 0 else False