from ultralytics import YOLO
from os.path import exists
from torch.cuda import is_available

# path
DATA_SETTINGS_PATH = "data_yolo26/data.yaml"

MODEL_PATH = "models"
MODEL_NAME = "phone_detection_yolo26_v1.0"
MODEL_FULL_NAME = MODEL_PATH + "/best.engine" if is_available() else MODEL_PATH + "/best.pt"

# training
def train(data_path: str = DATA_SETTINGS_PATH, model_name: str = MODEL_NAME):
    model = YOLO("yolo26m.pt")
    device = 0 if is_available() else "cpu"

    model.train(data=data_path, epochs=50, imgsz=512, batch=20, device=device, name=model_name)
    model.val()

    format = "engine" if is_available() else "onnx"
    model.export(format=format, dynamic=True)

# if phone detected - true
def get_phones(img, model_full_name: str = MODEL_FULL_NAME) -> bool:
    # pretrain model if didn't do it before
    if not exists(model_full_name):
        train()
    
    # process img
    model = YOLO(model_full_name)
    img_results = model(img)

    for box in img_results[0].boxes:
        if model.names[int(box.cls[0])] == "phone":
            return True
    return False
    # return True if len(img_results[0]) > 0 else False