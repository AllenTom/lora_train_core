import hashlib
from PIL import Image


def read_sha256_from_file_path(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def generate_random_string(length: int) -> str:
    import random
    import string
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


def make_image_thumbnail(image_path, save_path) -> str:
    img = Image.open(image_path)
    img.thumbnail((128, 128))
    img.save(save_path)
    return save_path
