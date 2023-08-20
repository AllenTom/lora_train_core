import json
import logging
import os
from typing import List, Optional

from server import paths, utils

log = logging.getLogger(__name__)


class SaveModel:
    def __init__(
            self,
            path: str,
            name: str,
            img_path: Optional[str] = None,
            props: Optional[dict] = None
    ):
        self.path = path
        self.name = name
        self.img_path = img_path
        self.props = props


class TrainConfig:
    def __init__(
            self,
            id: Optional[str] = None,
            name: Optional[str] = None,
            lora_presetName: Optional[str] = None,
            model_name: Optional[str] = None,
            pretrained_model_name_or_path: Optional[str] = None,
            extra_params: dict = {}
    ):
        self.id = id
        self.name = name
        self.lora_presetName = lora_presetName
        self.model_name = model_name
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.extra_params = extra_params


class ProjectParam:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height


class OriginalItem:
    def __init__(
            self,
            hash: str,
            src: str,
            file_name: Optional[str] = None,
            thumbnail: Optional[str] = None
    ):
        self.hash = hash
        self.src = src
        self.fileName = file_name
        self.thumbnail = thumbnail
class SavePreprocessItem:
    def __init__(
            self,
            hash: str,
            src: str,
            dest: str,

    ):
        self.hash = hash
        self.src = src
        self.dest = dest

class DatasetItem:
    def __init__(
        self,
        hash: str,
        image_name: str,
        image_path: str,
        caption_path: Optional[str] = None,
        captions: Optional[List[str]] = None,
        original_path: Optional[str] = None
    ):
        self.hash = hash
        self.image_name = image_name
        self.image_path = image_path
        self.caption_path = caption_path
        self.captions = captions
        self.original_path = original_path
class ProjectMeta:

    def __init__(
            self,
            models: List[SaveModel],
            train_configs: List[TrainConfig],
            preprocess: List[SavePreprocessItem],
            dataset: List[dict],
            original: List[OriginalItem],
            params: ProjectParam,
            preview_props: Optional[dict] = None,
            project_path: Optional[str] = None
    ):
        self.models = models
        self.train_configs = train_configs
        self.preprocess = preprocess
        self.dataset = dataset
        self.original = original
        self.params = params
        self.preview_props = preview_props
        self.project_path = project_path

    @staticmethod
    def new_meta(width=512, height=512,project_path=None):
        meta = ProjectMeta(
            models=[],
            train_configs=[],
            preprocess=[],
            dataset=[],
            original=[],
            params=ProjectParam(width, height),
            preview_props={},
            project_path=project_path
        )
        meta.train_configs.append(TrainConfig(
            id="default",
            name="默认配置",
            lora_presetName="default",
            pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5',
            extra_params={}
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
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4, ensure_ascii=False)

    @staticmethod
    def load_from_file(path):
        with open(path, "r", encoding="utf8") as file:
            content = file.read()
            return ProjectMeta(**json.loads(content))


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
    result:List[DatasetItem] = []
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
async def load_preprocess_images(meta:ProjectMeta) -> Optional[List[DatasetItem]]:

    preprocess_path : str = os.path.join(meta.project_path, 'preprocess')
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

        if preprocess_item.hash != is_exist.hash:
            item_to_remove.append(preprocess_item)

    for datasetItem in item_to_remove:
        os.unlink(datasetItem['imagePath'])
        if datasetItem['captionPath']:
            os.unlink(datasetItem['captionPath'])

    savePreprocess = []

    for preprocess_item in meta['preprocess']:
        if preprocess_item['hash'] not in invalid_item_hashes:
            savePreprocess.append(preprocess_item)

    result = []

    for item in savePreprocess:
        scannedItem = next((x for x in image_items if x['imageName'] == item['dest']), None)
        source = next((x for x in originalImages if x['hash'] == item['src']), None)

        result.append({
            'hash': item['hash'],
            'imagePath': scannedItem['imagePath'],
            'captionPath': scannedItem['captionPath'],
            'captions': scannedItem['captions'],
            'originalPath': source['src'] if source else None,
            'imageName': item['dest']
        })

    return result
def load_project(project_path: str) -> Optional[ProjectMeta]:

def new_project(name, width=512, height=512):
    store_path = paths.get_project_store()
    project_path = os.path.join(store_path, name)
    meta = ProjectMeta.new_meta(width=width, height=height,project_path=project_path)
    return meta
