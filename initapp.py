import os
import sys


def init_global():
    os.environ["HUGGINGFACE_HUB_CACHE"] = "./assets/hf_cache"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    sys.path.append(".\\sdscripts")
