from PIL import Image

from modules import yolomodeldec


REPO_ID = "deepghs/anime_halfbody_detection"
MODEL_FILENAME = "halfbody_detect_v1.0_s/model.pt"
model_face = None
model_body = None

def load_model():
    global model
    model = yolomodeldec.load_model(REPO_ID, MODEL_FILENAME)


def get_crops(im: Image,model):
    if model is None:
        load_model()
    results = model([im])
    if len(results) == 0:
        return []
    boxes = results[0].boxes
    if len(boxes) == 0:
        return []
    return boxes[0].xyxy.tolist()[0]








