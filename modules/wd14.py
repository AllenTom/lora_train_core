import os

import PIL.Image
import cv2
import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd

MOAT_MODEL_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"

SWIN_MODEL_REPO = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"

CONV_MODEL_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"

CONV2_MODEL_REPO = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"

VIT_MODEL_REPO = "SmilingWolf/wd-v1-4-vit-tagger-v2"

MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"


def load_labels() -> list[str]:
    path = huggingface_hub.hf_hub_download(
        MOAT_MODEL_REPO, LABEL_FILENAME,
    )
    df = pd.read_csv(path)
    tag_names = df["name"].tolist()
    rating_indexes = list(np.where(df["category"] == 9)[0])
    general_indexes = list(np.where(df["category"] == 0)[0])
    character_indexes = list(np.where(df["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes


def load_model(model_repo: str, model_filename: str) -> rt.InferenceSession:
    path = huggingface_hub.hf_hub_download(
        model_repo, model_filename,
    )
    model = rt.InferenceSession(path,providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
    return model


def change_model(model_name):
    if model_name == "MOAT":
        model = load_model(MOAT_MODEL_REPO, MODEL_FILENAME)
    elif model_name == "SwinV2":
        model = load_model(SWIN_MODEL_REPO, MODEL_FILENAME)
    elif model_name == "ConvNext":
        model = load_model(CONV_MODEL_REPO, MODEL_FILENAME)
    elif model_name == "ConvNextV2":
        model = load_model(CONV2_MODEL_REPO, MODEL_FILENAME)
    elif model_name == "ViT":
        model = load_model(VIT_MODEL_REPO, MODEL_FILENAME)

    return model


def make_square(img, target_size):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im


def smart_resize(img, size):
    # Assumes the image has already gone through make_square
    if img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img


class WaifuDiffusion:
    def __init__(self):
        self.model_name = None
        self.model = None
        self.general_threshold = 0.35
        self.character_threshold = 0.35

    def load(self, model_name="MOAT"):
        self.model = change_model(model_name)
        self.model_name = model_name

    def set_param(self, general_threshold, character_threshold):
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold

    def start(self):
        pass

    def stop(self):
        pass

    def tag(self, pil_image):
        self.start()
        res = self.tag_multi(pil_image)
        self.stop()

        return res

    def tag_multi(
            self,
            image: PIL.Image.Image,
            include_ranks: bool = False,
    ):
        tag_names, rating_indexes, general_indexes, character_indexes = load_labels()
        result = self.predict(
            image=image,
            general_threshold=self.general_threshold,
            character_threshold=self.character_threshold,
            tag_names=tag_names,
            rating_indexes=rating_indexes,
            general_indexes=general_indexes,
            character_indexes=character_indexes
        )
        if include_ranks:
            return result[4]
        return result[1]

    def predict(
            self,
            image: PIL.Image.Image,
            tag_names: list[str],
            rating_indexes: list[np.int64],
            general_indexes: list[np.int64],
            character_indexes: list[np.int64],
            general_threshold: float = 0.5,
            character_threshold: float = 0.5,
    ):
        rawimage = image

        model = self.model
        if model is None:
            model = change_model(self.model_name)

        _, height, width, _ = model.get_inputs()[0].shape

        # Alpha to white
        image = image.convert("RGBA")
        new_image = PIL.Image.new("RGBA", image.size, "WHITE")
        new_image.paste(image, mask=image)
        image = new_image.convert("RGB")
        image = np.asarray(image)

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        image = make_square(image, height)
        image = smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        input_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name
        probs = model.run([label_name], {input_name: image})[0]

        labels = list(zip(tag_names, probs[0].astype(float)))

        # First 4 labels are actually ratings: pick one with argmax
        ratings_names = [labels[i] for i in rating_indexes]
        rating = dict(ratings_names)

        # Then we have general tags: pick any where prediction confidence > threshold
        general_names = [labels[i] for i in general_indexes]
        general_res = [x for x in general_names if x[1] > general_threshold]
        general_res = dict(general_res)

        # Everything else is characters: pick any where prediction confidence > threshold
        character_names = [labels[i] for i in character_indexes]
        character_res = [x for x in character_names if x[1] > character_threshold]
        character_res = dict(character_res)

        b = dict(sorted(general_res.items(), key=lambda item: item[1], reverse=True))
        a = (
            ", ".join(list(b.keys()))
            .replace("_", " ")
            .replace("(", "\(")
            .replace(")", "\)")
        )
        c = ", ".join(list(b.keys()))
        return (a, c, rating, character_res, general_res)


model = WaifuDiffusion()

if __name__ == '__main__':
    model.load()
    result = model.tag_multi(
        image=PIL.Image.open("./test.jpg"),
    )
    print(result)
