import re
import sys
import traceback
from collections import namedtuple
from pathlib import Path

import huggingface_hub
import torch
import torch.hub
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from modules import share, lowvram

BLIP_REPO = "takayamaaren/xformers_build_pack"
BLIP_FILE = "model_base_caption_capfilt_large.pth"

blip_image_eval_size = 384
Category = namedtuple("Category", ["name", "topn", "items"])

re_topn = re.compile(r"\.top(\d+)\.")


def category_types():
    return [f.stem for f in Path(share.interrogator.content_dir).glob('*.txt')]




class InterrogateModels:
    blip_model = None
    dtype = None
    running_on_cpu = None

    def __init__(self, content_dir):
        self.loaded_categories = None
        self.skip_categories = []
        self.content_dir = content_dir
        self.running_on_cpu = share.device_interrogate == torch.device("cpu")



    def create_fake_fairscale(self):
        class FakeFairscale:
            def checkpoint_wrapper(self):
                pass

        sys.modules["fairscale.nn.checkpoint.checkpoint_activations"] = FakeFairscale

    def load_blip_model(self):
        self.create_fake_fairscale()
        import models.blip
        path = huggingface_hub.hf_hub_download(
            BLIP_REPO, BLIP_FILE,
        )
        share.blip_model_path = path
        blip_model = models.blip.blip_decoder(pretrained=share.blip_model_path,
                                              image_size=blip_image_eval_size, vit='base',
                                              med_config=share.med_config)
        blip_model.eval()

        return blip_model

    def load(self):
        if self.blip_model is None:
            self.blip_model = self.load_blip_model()
            if not share.no_half and not self.running_on_cpu:
                self.blip_model = self.blip_model.half()

        self.blip_model = self.blip_model.to(share.device_interrogate)

        self.dtype = next(self.blip_model.parameters()).dtype



    def send_blip_to_ram(self):
        if not share.interrogate_keep_models_in_memory:
            if self.blip_model is not None:
                self.blip_model = self.blip_model.to(share.devCPU)

    def unload(self):
        self.send_blip_to_ram()

        # devices.torch_gc()


    def generate_caption(self, pil_image):
        gpu_image = transforms.Compose([
            transforms.Resize((blip_image_eval_size, blip_image_eval_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(pil_image).unsqueeze(0).type(self.dtype).to(share.device_interrogate)

        with torch.no_grad():
            caption = self.blip_model.generate(gpu_image, sample=False, num_beams=share.interrogate_clip_num_beams,
                                               min_length=share.interrogate_clip_min_length,
                                               max_length=share.interrogate_clip_max_length)

        return caption[0]

    def interrogate(self, pil_image):
        res = ""
        # shared.state.begin()
        # shared.state.job = 'interrogate'
        try:
            if share.lowvram or share.medvram:
                lowvram.send_everything_to_cpu()
                # devices.torch_gc()

            self.load()

            caption = self.generate_caption(pil_image)
            self.send_blip_to_ram()
            # devices.torch_gc()

            res = caption
            result = [{
                "tag": caption,
                "rank": 1,
            }]

        except Exception:
            print("Error interrogating", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            res += "<error>"

        self.unload()
        # shared.state.end()

        return result
