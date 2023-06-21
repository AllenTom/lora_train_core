import contextlib
import os
import re

import numpy as np
import torch
from PIL import Image

from modules import deepbooru_model, share, images, modelloader

re_special = re.compile(r'([\\()])')
interrogate_deepbooru_score_threshold = 0.7
deepbooru_use_spaces = False
deepbooru_escape = True
deepbooru_sort_alpha = True
interrogate_return_ranks = False
deepbooru_filter_tags = ""


class DeepDanbooru:
    def __init__(self):
        self.model = None

    def load(self):
        if self.model is not None:
            return
        exists = os.path.exists(share.danbooru_model_path)
        if not exists:
            modelloader.load_file(
                url='https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt',
                dst=share.danbooru_model_path,
            )
            # files = modelloader.load_models(
            #     model_path=os.path.join(paths.models_path, "torch_deepdanbooru"),
            #     model_url='https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt',
            #     ext_filter=[".pt"],
            #     download_name='model-resnet_custom_v31.pt',
            # )

        self.model = deepbooru_model.DeepDanbooruModel()
        self.model.load_state_dict(torch.load(share.danbooru_model_path, map_location="cpu"))

        self.model.eval()
        self.model.to(share.devCPU, share.dtype)

    def start(self):
        self.model.to(share.device)

    def stop(self):
        if not share.interrogate_keep_models_in_memory:
            self.model.to(share.devCPU)
            # devices.torch_gc()

    def tag(self, pil_image):
        self.start()
        res = self.tag_multi(pil_image)
        self.stop()

        return res

    def tag_multi(self, pil_image, force_disable_ranks=False):
        threshold = interrogate_deepbooru_score_threshold
        use_spaces = deepbooru_use_spaces
        use_escape = deepbooru_escape
        alpha_sort = deepbooru_sort_alpha
        include_ranks = interrogate_return_ranks and not force_disable_ranks

        pic = images.resize_image(2, pil_image.convert("RGB"), 512, 512)
        a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

        with torch.no_grad(), share.autocast():
            x = torch.from_numpy(a).to(share.device)
            y = self.model(x)[0].detach().cpu().numpy()

        probability_dict = {}

        for tag, probability in zip(self.model.tags, y):
            if probability < threshold:
                continue

            if tag.startswith("rating:"):
                continue

            probability_dict[tag] = probability

        if alpha_sort:
            tags = sorted(probability_dict)
        else:
            tags = [tag for tag, _ in sorted(probability_dict.items(), key=lambda x: -x[1])]

        res = []

        filtertags = set([x.strip().replace(' ', '_') for x in deepbooru_filter_tags.split(",")])

        for tag in [x for x in tags if x not in filtertags]:
            probability = probability_dict[tag]
            tag_outformat = tag
            if use_spaces:
                tag_outformat = tag_outformat.replace('_', ' ')
            if use_escape:
                tag_outformat = re.sub(re_special, r'\\\1', tag_outformat)
            if include_ranks:
                tag_outformat = f"({tag_outformat}:{probability:.3f})"

            res.append(tag_outformat)

        return ", ".join(res)


model = DeepDanbooru()
