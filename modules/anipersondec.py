from PIL import Image
from ultralytics import YOLO

from modules import yolomodeldec

REPO_ID = "deepghs/anime_person_detection"
MODEL_FILENAME = "person_detect_v1_m/model.pt"


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


if __name__ == '__main__':
    # Load a pretrained YOLOv8n model
    model = YOLO('../assets/animeperson.pt')

    # Run inference on 'bus.jpg'
    results = model('test3.jpg')  # results list

    # Show the results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image
        im.save('results.jpg')  # save image
