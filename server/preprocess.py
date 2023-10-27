import asyncio
import hashlib
import os
from typing import List, Optional

from fastapi import UploadFile

from modules import preprocess
from server import project, paths, sender, orginal, utils


def save_preprocess_item(file_path: str, original: Optional[project.OriginalItem], project_id: str):
    file_hash = None
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
        file_hash = sha256_hash.hexdigest()
    original_hash = None
    if original is not None:
        original_hash = original.hash
    new_item = project.SavePreprocessItem(
        hash=file_hash,
        src=original_hash,
        dest=os.path.basename(file_path)
    )
    meta = project.read_project_meta(project_id)
    meta.add_preprocess_item(new_item)
    return new_item


def on_preprocess_output(data: dict, meta: project.ProjectMeta, original_items: List[preprocess.PreprocessImageFile]):
    src = data["src"]
    dest = data["dest"]

    original_image = next(filter(lambda x: x.filename == src, original_items), None)
    original_item = None
    if original_image is not None:
        original_item = orginal.save_original_with_img(
            img=original_image.img.copy(),
            filename=src,
            meta=meta
        )
    save_preprocess_item(dest, original_item, meta.get_project_id())

    sender.send_message_to_clients(
        type='info',
        message="output",
        vars=data,
        event="preprocess_out",
        id="global"
    )


def on_preprocess_complete(data: dict,meta: project.ProjectMeta):
    dataset_items: List[project.DatasetItem] = []
    original_items: List[project.OriginalItem] = []
    for output_item in data:
        item = project.load_dataset_item_from_file(output_item["dest"])
        original_path = paths.get_original_image_folder(meta.project_path)
        original_item = project.load_original_from_file(os.path.join(original_path, output_item["src"]))
        item.original_path = original_item.hash
        original_items.append(original_item)
        dataset_items.append(item)

    dataset_items = project.link_to_real_preprocess_image(dataset_items, meta.get_project_id())

    original_items = project.link_to_real_original_image(original_items, meta.get_project_id())
    output_dataset_item = []
    original_dataset_item = []
    for dataset_item in dataset_items:
        output_dataset_item.append(dataset_item.model_dump(by_alias=True))
    for original_item in original_items:
        original_dataset_item.append(original_item.model_dump(by_alias=True))



    sender.send_message_to_clients(
        type='info',
        message="complete",
        vars={
            "original": original_dataset_item,
            "preprocess": output_dataset_item
        },
        event="preprocess_complete",
        id="global"
    )


async def preprocess_service(project_id: str, files: List[UploadFile]):
    meta = project.read_project_meta(project_id)

    images = []

    for file in files:
        item = await preprocess.PreprocessImageFile.from_upload_file(file)
        images.append(item)
    preprocess_path = paths.get_preprocessed_image_folder(meta.project_path)

    def on_preprocess_output_handler(data: dict):
        on_preprocess_output(data, meta, images)
    def on_preprocess_complete_handler(data: dict):
        on_preprocess_complete(data, meta)

    sender.send_message_to_clients(
        type='info',
        message="output",
        vars={},
        event="preprocess_start",
        id="global"
    )

    preprocess.preprocess_work(
        process_dst=preprocess_path,
        process_width=meta.params.width,
        process_height=meta.params.height,
        process_caption=False,
        process_caption_deepbooru=False,
        process_caption_wd=False,
        process_caption_clip=False,
        wd_general_threshold=None,
        wd_character_threshold=None,
        wd_model_name=None,
        input_images=images,
        on_output_callback=on_preprocess_output_handler,
        on_complete_callback=on_preprocess_complete_handler
    )
    return "ok"
