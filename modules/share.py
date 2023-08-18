import contextlib
import os

import torch

from modules import interrogate

device = torch.device('cuda')
devCPU = cpu = torch.device("cuda")
dtype = torch.float16
device_interrogate = torch.device("cuda")
precision = "autocast"  # autocast or full
no_half = False
interrogate_keep_models_in_memory = False
interrogator = interrogate.InterrogateModels("interrogate")
interrogate_clip_skip_categories = []
clip_models_path = "./assets"
danbooru_model_path = "./assets/model-resnet_custom_v3.pt"
blip_model_path = "./assets/model_base_caption_capfilt_large.pth"
blip_model_hash = "ee2220561f5855cb2793c58f6836a248"
vit_model_path = "./assets/ViT-L-14.pt"
vit_model_hash = "096db1af569b284eb76b3881534822d9"
med_config = os.path.join("./finetune/blip", "med_config.json")
interrogate_clip_dict_limit = 1500
interrogate_clip_num_beams = 1
interrogate_clip_min_length = 24
interrogate_clip_max_length = 48
lowvram = False
medvram = False
use_cn = False
interrogate_return_ranks = True
callbackServer = "http://localhost:6745"


def autocast(disable=False):
    if disable:
        return contextlib.nullcontext()

    if dtype == torch.float32 or precision == "full":
        return contextlib.nullcontext()

    return torch.autocast("cuda")


def get_deep_booru_model_url():
    if use_cn:
        return 'https://huggingface.co/casual02/model-resnet_custom_v3/resolve/main/model-resnet_custom_v3.pt'
    else:
        return 'https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt'
