import os

import huggingface_hub
from PIL import Image
from ultralytics import YOLO

model = None


def load_model():
    global model
    path = huggingface_hub.hf_hub_download(
        'nyuuzyou/AnimeHeads', "weights/animeheadsv3.pt",cache_dir='./hf_cache'
    )
    model = YOLO(path)


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


def crop_image_with_face(im: Image, ratio) -> Image:
    result = get_crops(im)
    if len(result) == 0:
        return im
    print(result)
    x1, y1, x2, y2 = result
    pwidth = x2 - x1
    pheight = y2 - y1
    width_ratio = pwidth / im.width
    height_ratio = pheight / im.height
    # img_width_scale = width / img.width
    # img_height_scale = height / img.height
    crop_x1 = max(x1 - (x1 * ratio * width_ratio), 0)
    crop_y1 = max(y1 - (y1 * ratio * height_ratio), 0)
    crop_x2 = min(x2 + (pwidth * ratio * width_ratio), im.width)
    crop_y2 = min(y2 + (pheight * ratio * height_ratio), im.height)
    # must in center
    center_x = (crop_x1 + crop_x2) / 2
    center_y = (crop_y1 + crop_y2) / 2
    width_delta = min(center_x - crop_x1, crop_x2 - center_x)
    height_delta = min(center_y - crop_y1, crop_y2 - center_y)
    crop_x1 = center_x - width_delta
    crop_x2 = center_x + width_delta
    crop_y1 = center_y - height_delta
    crop_y2 = center_y + height_delta


    return im.crop((int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)))