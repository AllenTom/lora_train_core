import os

import huggingface_hub
from PIL import Image
from ultralytics import YOLO


def load_model(repo_id, model_filename):
    path = huggingface_hub.hf_hub_download(
        repo_id, model_filename, cache_dir='./hf_cache'
    )
    model = YOLO(path)
    return model


def get_crops(im: Image, model):
    results = model([im])
    if len(results) == 0:
        return []
    boxes = results[0].boxes
    if len(boxes) == 0:
        return []
    return boxes[0].xyxy.tolist()[0]


def crop_image_with_detection(im: Image, ratio, model,box_to_top=False) -> Image:
    result = get_crops(im, model)
    if len(result) == 0:
        return im
    print(result)
    x1, y1, x2, y2 = result

    pwidth = x2 - x1
    pheight = y2 - y1
    width_ratio = pwidth / im.width
    height_ratio = pheight / im.height


    crop_x1 = max(x1 - (x1 * ratio * width_ratio), 0)
    crop_y1 = max(y1 - (y1 * ratio * height_ratio), 0)
    crop_x2 = min(x2 + (pwidth * ratio * width_ratio), im.width)
    crop_y2 = min(y2 + (pheight * ratio * height_ratio), im.height)

    box_height = crop_y2 - crop_y1
    box_width = crop_x2 - crop_x1

    if box_height > box_width and box_to_top:
        crop_y1 = 0
        crop_y2 = box_width

    return im.crop((int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)))
