from ultralytics import YOLO
import os

# path
DATA_PATH = "data_yolo26/data.yaml"
MODEL_PATH = "models"
MODEL_NAME = "phone_detection_yolo26_v1.0.pt"

# training
def train(data_path: str = DATA_PATH, model_path: str = MODEL_PATH, model_name: str = MODEL_NAME):
    model = YOLO("yolo26b.pt")
    device = 0 # os.cpu_count()

    results = model.train(data=data_path, epochs=100, imgsz=512, batch=20, device=device, name=model_name)
    val = model.val()

    model.export(format="TensorRT")
    # model.save()

def get_phones(img, model_path: str = MODEL_PATH, model_name: str = MODEL_NAME):
    # pretrain model if didn't do it before
    if not os.path.exists(model_path + "/" + model_name):
        train()
    
    # get model and get phone