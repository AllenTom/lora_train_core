import os



def init_global():
    os.environ["HUGGINGFACE_HUB_CACHE"] = "./assets/hf_cache"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
