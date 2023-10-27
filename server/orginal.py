import hashlib
import os
import uuid

from PIL import Image

from server import project, paths, utils


def save_original_with_img(img: Image, filename, meta: project.ProjectMeta):
    original_folder = paths.get_original_image_folder(meta.project_path)
    save_name = filename
    image_path = os.path.join(original_folder, save_name)
    img.save(image_path)
    image_hash = None
    sha256_hash = hashlib.sha256()
    with open(image_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
        image_hash = sha256_hash.hexdigest()
    image_cache_path = paths.get_cache_image_folder(meta.project_path)
    uuid_str = str(uuid.uuid4())
    image_ext = os.path.splitext(filename)[1]
    thumbnail_save_name = f"{uuid_str}{image_ext}"
    thumbnail_save_path = os.path.join(image_cache_path, thumbnail_save_name)
    thumbnail_path = utils.make_image_thumbnail(image_path, thumbnail_save_path)
    item = project.OriginalItem(
        hash=image_hash,
        src=save_name,
        file_name=filename,
        thumbnail=os.path.basename(thumbnail_path),
    )
    meta = project.read_project_meta(meta.get_project_id())
    meta.add_original_item(item)
    return item
