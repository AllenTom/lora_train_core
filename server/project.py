import json
import logging
import os
import shutil
from typing import List, Optional

from PIL import Image
from fastapi import UploadFile
from pydantic import BaseModel, Field
from modules import preprocess, output, share

from server import paths, utils

log = logging.getLogger(__name__)


class SaveModel(BaseModel):
    path: str
    name: str
    img_path: Optional[str] = Field(
        serialization_alias="imgPath"
    )
    props: Optional[dict] = None

    @staticmethod
    def from_json_dict(data):
        print(data)
        return SaveModel(
            path=data["path"],
            name=data["name"],
            img_path=data.get("imgPath", None),
            props=data.get("props", None)
        )


class DatasetFolder(BaseModel):
    name: str = Field()
    step: int = Field()
    images: List[str] = Field()


class TrainConfig(BaseModel):
    id: str
    name: str
    lora_preset_name: str = Field(
        serialization_alias="loraPresetName"
    )
    model_name: str = Field(
        serialization_alias="modelName"
    )
    pretrained_model_name_or_path: str
    extra_params: Optional[dict] = Field(
        serialization_alias="extraParams"
    )

    @staticmethod
    def from_json_dict(data):
        print(data)
        return TrainConfig(
            id=data["id"],
            name=data["name"],
            lora_preset_name=data["loraPresetName"],
            model_name=data["modelName"],
            pretrained_model_name_or_path=data["pretrained_model_name_or_path"],
            extra_params=data.get("extraParams", None)

        )


class ProjectParam(BaseModel):
    width: int
    height: int

    @staticmethod
    def from_json_dict(data):
        print(data)
        return ProjectParam(
            width=data["width"],
            height=data["height"]
        )


class OriginalItem(BaseModel):
    hash: str
    src: str
    file_name: Optional[str] = Field(
        serialization_alias="fileName",
    )
    thumbnail: Optional[str] = None

    @staticmethod
    def from_json_dict(data):
        print(data)
        return OriginalItem(
            hash=data["hash"],
            src=data["src"],
            file_name=data.get("fileName", None),
            thumbnail=data.get("thumbnail", None)
        )


class SavePreprocessItem(BaseModel):
    hash: str
    src: str
    dest: str

    @staticmethod
    def from_json_dict(data):
        return SavePreprocessItem(
            hash=data["hash"],
            src=data["src"],
            dest=data["dest"]
        )


class DatasetItem(BaseModel):
    hash: str
    image_name: str
    image_path: str
    caption_path: Optional[str] = None
    captions: Optional[List[str]] = None
    original_path: Optional[str] = None

    @staticmethod
    def from_json_dict(data):
        print(data)
        return DatasetItem(
            hash=data["hash"],
            image_name=data["image_name"],
            image_path=data["image_path"],
            caption_path=data.get("caption_path", None),
            captions=data.get("captions", None),
            original_path=data.get("original_path", None)
        )


class ProjectMeta(BaseModel):
    models: List[SaveModel]
    train_configs: List[TrainConfig] = Field(
        serialization_alias="trainConfigs"
    )
    preprocess: List[SavePreprocessItem]
    dataset: List[DatasetFolder]
    original: List[OriginalItem]
    params: ProjectParam
    preview_props: Optional[dict] = Field(
        serialization_alias="previewProps"
    )
    project_path: Optional[str] = Field(
        serialization_alias="projectPath"
    )

    @staticmethod
    def new_meta(width=512, height=512, project_path=None):
        meta = ProjectMeta(
            models=[],
            train_configs=[],
            preprocess=[],
            dataset=[],
            original=[],
            params=ProjectParam(width=width, height=height),
            preview_props={},
            project_path=project_path
        )
        # id: str
        # name: str
        # lora_presetName: str
        # model_name: str
        # pretrained_model_name_or_path: str
        # extra_params: dict
        meta.train_configs.append(TrainConfig(
            id="default",
            name="默认配置",
            lora_preset_name="default",
            pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5',
            extra_params={},
            model_name="mymodel"
        ))
        meta.save()

        return meta

    def save(self):
        os.makedirs(self.project_path, exist_ok=True)
        meta_file_content = self.to_json()
        meta_file_path = os.path.join(self.project_path, "project.json")
        with open(meta_file_path, "w", encoding="utf8") as file:
            file.write(meta_file_content)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2, by_alias=True, exclude=["project_path"])

    @staticmethod
    def load_from_file(path):
        with open(path, "r", encoding="utf8") as file:
            content = file.read()
            data = json.loads(content)
            return ProjectMeta(
                models=[SaveModel.from_json_dict(x) for x in data["models"]],
                train_configs=[TrainConfig.from_json_dict(x) for x in data["trainConfigs"]],
                preprocess=[SavePreprocessItem.from_json_dict(x) for x in data["preprocess"]],
                dataset=data["dataset"],
                original=[OriginalItem.from_json_dict(x) for x in data["original"]],
                params=ProjectParam.from_json_dict(data["params"]),
                preview_props=data.get("previewProps", None),
                project_path=os.path.dirname(path)
            )

    def add_preprocess_item(self, *items: SavePreprocessItem):
        for item in items:
            # remove duplicate
            self.preprocess = [x for x in self.preprocess if x.hash != item.hash]
            self.preprocess.append(item)
        self.save()

    def add_original_item(self, *items: OriginalItem):
        for item in items:
            # remove duplicate
            self.original = [x for x in self.original if x.hash != item.hash]
            self.original.append(item)
        self.save()

    def add_dataset(self, *items: DatasetFolder):
        for item in items:
            # remove duplicate
            self.dataset = [x for x in self.dataset if x.name != item.name]
            self.dataset.append(item)
        self.save()

    def add_train_config(self, *items: TrainConfig):
        for item in items:
            # remove duplicate
            self.train_configs = [x for x in self.train_configs if x.id != item.id]
            self.train_configs.append(item)
        self.save()


async def load_original_images(meta: ProjectMeta) -> Optional[List[OriginalItem]]:
    original_folder_path = paths.get_original_image_folder()
    if not original_folder_path:
        log.error('original folder not found on load original image')
        return None

    files = os.listdir(original_folder_path)
    original_images = meta.original
    invalidate_hash_item = []
    validate_files = []

    for originalImage in original_images:
        image_path = os.path.join(original_folder_path, originalImage.src)
        if originalImage.src not in files:
            invalidate_hash_item.append(originalImage.hash)
            continue

        file_hash = utils.read_sha256_from_file_path(image_path)

        if file_hash != originalImage.hash:
            invalidate_hash_item.append(originalImage.hash)
            continue

        validate_files.append(originalImage.src)

    # Remove invalidated hash items
    original_images = [item for item in original_images if item.hash not in invalidate_hash_item]

    # Remove unlink files
    invalidate_file = [file for file in files if file not in validate_files]
    for file in invalidate_file:
        os.remove(os.path.join(original_folder_path, file))
    meta.original = original_images
    meta.save()
    return original_images


def read_caption(caption_file_path: str) -> List[str]:
    with open(caption_file_path, 'r', encoding='utf-8') as file:
        caption = file.read()
    if len(caption) == 0:
        return []
    return [it.strip() for it in caption.split(',')]


async def scan_image_files(image_path: str) -> List[DatasetItem]:
    result: List[DatasetItem] = []
    dir = os.listdir(image_path)
    image_files = [file for file in dir if os.path.splitext(file.lower())[1] in ['.png', '.jpg', '.jpeg']]
    for file in image_files:
        data_item = DatasetItem(
            hash=utils.read_sha256_from_file_path(os.path.join(image_path, file)),
            image_name=file,
            image_path=os.path.join(image_path, file)
        )

        caption_file_name = os.path.splitext(file)[0] + '.txt'
        caption_file_path = os.path.join(image_path, caption_file_name)

        if os.path.exists(caption_file_path):
            caption_tags = read_caption(caption_file_path)
            if len(caption_tags) > 0:
                data_item.caption_path = caption_file_path
                data_item.captions = caption_tags

        result.append(data_item)
    return result


async def load_preprocess_images(meta: ProjectMeta) -> Optional[List[DatasetItem]]:
    preprocess_path: str = os.path.join(meta.project_path, 'preprocess')
    if not os.path.exists(preprocess_path):
        os.makedirs(preprocess_path)

    image_items = await scan_image_files(preprocess_path)

    invalid_item_hashes = []
    item_to_remove = []

    for preprocess_item in meta.preprocess:

        is_exist = next((x for x in image_items if x.hash == preprocess_item.hash), None)
        # item not exist
        if not is_exist:
            invalid_item_hashes.append(preprocess_item.hash)
            continue

        # item exist but not match
        for original_item in meta.original:
            if original_item.hash == preprocess_item.src:
                is_exist.original_path = original_item.hash
                break

        if is_exist.original_path is None:
            invalid_item_hashes.append(preprocess_item.hash)
            continue

        # invalid hash
        if preprocess_item.hash != is_exist.hash:
            item_to_remove.append(is_exist)
            invalid_item_hashes.append(preprocess_item.hash)
            continue
    # clean up invalid item
    for datasetItem in item_to_remove:
        os.unlink(datasetItem.image_path)
        if datasetItem.caption_path is not None:
            os.unlink(datasetItem.caption_path)

    save_preprocess = []

    for preprocess_item in meta.preprocess:
        if preprocess_item.hash not in invalid_item_hashes:
            save_preprocess.append(preprocess_item)

    result: List[DatasetItem] = []

    for item in save_preprocess:
        scanned_item = next((x for x in image_items if x.image_name == item.dest), None)
        source = next((x for x in meta.original if x.hash == item.src), None)
        result.append(DatasetItem(
            hash=item.hash,
            image_name=item.dest,
            image_path=scanned_item.image_path,
            caption_path=scanned_item.caption_path,
            captions=scanned_item.captions,
            original_path=source.src if source else None
        ))

    return result


def load_project(project_path: str) -> Optional[ProjectMeta]:
    meta_file_path = os.path.join(project_path, "project.json")
    if not os.path.exists(meta_file_path):
        return None
    meta = ProjectMeta.load_from_file(meta_file_path)
    load_original_images(meta)
    load_preprocess_images(meta)
    return meta


def new_project(name, width=512, height=512):
    store_path = paths.get_project_store()
    project_path = os.path.join(store_path, name)
    meta = ProjectMeta.new_meta(width=width, height=height, project_path=project_path)
    return meta


def load_project_service(name):
    store_path = paths.get_project_store()
    project_path = os.path.join(store_path, name)
    meta = load_project(project_path)
    return meta


def get_project_path(name):
    store_path = paths.get_project_store()
    project_path = os.path.join(store_path, name)
    return project_path


def create_original_image(
        meta: ProjectMeta,
        file: Optional[UploadFile] = None,
        image: Optional[Image.Image] = None,
        filename: str = None
):
    filename = filename or file.filename
    img = image or Image.open(file.file)
    original_image_paths = paths.get_original_image_folder(meta.project_path)
    image_save_path = os.path.join(original_image_paths, filename)
    img.save(image_save_path)
    # create thumbnail
    thumbnail_name = os.path.splitext(filename)[0] + "_thumbnail.png"
    thumbnail_path = os.path.join(meta.project_path, "image", thumbnail_name)
    if not os.path.exists(os.path.dirname(thumbnail_path)):
        os.makedirs(os.path.dirname(thumbnail_path))
    MAX_SIZE = (100, 100)
    img.thumbnail(MAX_SIZE)
    img.convert('RGBA')
    img.save(thumbnail_path, "PNG")
    # save meta
    file_hash = utils.read_sha256_from_file_path(os.path.join(original_image_paths, filename))
    return OriginalItem(
        hash=file_hash,
        src=filename,
        thumbnail=thumbnail_name,
        file_name=None
    )


def add_original_image(file: UploadFile, id: str):
    project_path = get_project_path(id)
    meta = ProjectMeta.load_from_file(os.path.join(project_path, "project.json"))
    # save original image
    new_item = create_original_image(file=file, meta=meta)
    # remove duplicate
    new_original_items = meta.original
    new_original_items = [item for item in new_original_items if item.hash != new_item.hash]
    new_original_items.append(new_item)
    meta.original = new_original_items
    meta.save()

    return new_original_items


# params.outputDetail.append({
#             "dest": image_path,
#             "src": params.src,
#             "name": f"{basename}.png",
#         })
def add_preprocess_item_to_meta(outputList, meta: ProjectMeta, srcs: List[preprocess.PreprocessImageFile]):
    new_items: List[SavePreprocessItem] = []
    new_original_items: List[OriginalItem] = []
    # save original
    for item in outputList:
        original_image = next((x for x in srcs if x.filename == item['src']), None)
        if original_image is None:
            continue
        new_original_image_item = create_original_image(
            image=original_image.img,
            meta=meta,
            filename=os.path.basename(item["src"])
        )
        new_items.append(SavePreprocessItem(
            hash=utils.read_sha256_from_file_path(item["dest"]),
            src=new_original_image_item.hash,
            dest=os.path.basename(item["dest"])
        ))
        new_original_items.append(new_original_image_item)
    meta.add_original_item(*new_original_items)
    meta.add_preprocess_item(*new_items)
    return meta


def auto_caption_images(id: str, images: List[preprocess.PreprocessImageFile]):
    project_path = get_project_path(id)
    meta = ProjectMeta.load_from_file(os.path.join(project_path, "project.json"))
    params = preprocess.preprocess_work(
        process_caption_deepbooru=True,
        process_dst=paths.get_preprocessed_image_folder(meta.project_path),
        input_images=images,
    )
    add_preprocess_item_to_meta(params.outputDetail, meta, images)
    return params


def auto_caption_image_service(id: str, files: List[UploadFile]):
    images: List[preprocess.PreprocessImageFile] = []
    for file in files:
        images.append(preprocess.PreprocessImageFile.from_upload_file(file), )
    params = auto_caption_images(id, images)
    return params


def create_dataset(
        id: str,
        name: str,
        step: int = 100,
        image_hashes: List[str] = []
):
    project_path = get_project_path(id)
    preprocess_path = paths.get_preprocessed_image_folder(project_path)
    meta = ProjectMeta.load_from_file(os.path.join(project_path, "project.json"))
    dataset_path = paths.get_dataset_folder(meta.project_path)
    dataset_folder_path = os.path.join(dataset_path, f"{step}_{name}")
    if os.path.exists(dataset_folder_path):
        shutil.rmtree(dataset_folder_path)

    os.makedirs(dataset_folder_path, exist_ok=True)
    added_image_hashes = []
    for image_hash in image_hashes:
        image_item = next((x for x in meta.preprocess if x.hash == image_hash), None)
        if image_item is None:
            continue
        image_path = os.path.join(preprocess_path, image_item.dest)
        shutil.copyfile(image_path, os.path.join(dataset_folder_path, image_item.dest))
        caption_file_name = os.path.splitext(image_item.dest)[0] + '.txt'
        caption_file_path = os.path.join(preprocess_path, caption_file_name)
        if os.path.exists(caption_file_path):
            shutil.copyfile(caption_file_path, os.path.join(dataset_folder_path, caption_file_name))
        added_image_hashes.append(image_hash)
    meta.add_dataset(DatasetFolder(
        name=name,
        step=step,
        images=added_image_hashes
    ))
    return meta


def add_train_config(id: str, name: str, lora_preset_name: str, model_name: str, pretrained_model_name_or_path: str,
                     extra_params: dict,
                     config_id=None):
    if config_id is None:
        config_id = utils.generate_random_string(6)
    project_path = get_project_path(id)
    meta = ProjectMeta.load_from_file(os.path.join(project_path, "project.json"))

    config = TrainConfig(
        id=config_id,
        name=name,
        lora_preset_name=lora_preset_name,
        model_name=model_name,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        extra_params=extra_params
    )
    meta.add_train_config(config)
    return meta
