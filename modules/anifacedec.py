from PIL import Image

from modules import yolomodeldec

model = None

REPO_ID = "nyuuzyou/AnimeHeads"
MODEL_FILENAME = "weights/animeheadsv3.pt"

def load_model():
    global model
    model = yolomodeldec.load_model(REPO_ID, MODEL_FILENAME)


def get_crops(im: Image):
    global model
    if model is None:
        load_model()
    results = model([im])
    if len(results) == 0:
        return []
    boxes = results[0].boxes
    if len(boxes) == 0:
        return []
    return boxes[0].xyxy.tolist()[0]
