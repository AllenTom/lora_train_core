import os
def init_global():
    os.environ["HUGGINGFACE_HUB_CACHE"] = "./assets/hf_cache"
