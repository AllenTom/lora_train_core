import math
import os
from typing import List, Optional

import huggingface_hub
from PIL import Image, ImageOps
from fastapi import UploadFile

from modules import autocrop, images, deepbooru, output, share, wd14, anifacedec, yolomodeldec, anipersondec, \
    anihalfpersondec,cliptagger,cliptagger2

model = deepbooru.DeepDanbooru()
FACE_CROP_REPO = "takayamaaren/xformers_build_pack"
FACE_CROP = "face_detection_yunet_2022mar.onnx"

class PreprocessImageFile:
    def __init__(self, filename: str, img: Image, raw):
        self.filename = filename
        self.img = img
        self.raw = raw

    @staticmethod
    def from_upload_file(file: UploadFile):
        return PreprocessImageFile(file.filename, Image.open(file.file).convert("RGB"), raw=file)


class PreprocessParams:
    src = None
    dstdir = None
    subindex = 0
    flip = False
    process_caption = False
    process_caption_deepbooru = False
    process_caption_wd = False
    process_caption_clip=False
    process_caption_clip2=False
    preprocess_txt_action = None
    outputFiles = []
    outputDetail = []


def center_crop(image: Image, w: int, h: int):
    iw, ih = image.size
    if ih / h < iw / w:
        sw = w * ih / h
        box = (iw - sw) / 2, 0, iw - (iw - sw) / 2, ih
    else:
        sh = h * iw / w
        box = 0, (ih - sh) / 2, iw, ih - (ih - sh) / 2
    return image.resize((w, h), Image.Resampling.LANCZOS, box)


def multicrop_pic(image: Image, mindim, maxdim, minarea, maxarea, objective, threshold):
    iw, ih = image.size
    err = lambda w, h: 1 - (lambda x: x if x < 1 else 1 / x)(iw / ih / (w / h))
    wh = max(((w, h) for w in range(mindim, maxdim + 1, 64) for h in range(mindim, maxdim + 1, 64)
              if minarea <= w * h <= maxarea and err(w, h) <= threshold),
             key=lambda wh: (wh[0] * wh[1], -err(*wh))[::1 if objective == 'Maximize area' else -1],
             default=None
             )
    return wh and center_crop(image, *wh)


def save_pic_with_caption(image, index, params: PreprocessParams, existing_caption=None):
    caption = ""

    if params.process_caption:
        caption += share.interrogator.generate_caption(image)

    if params.process_caption_deepbooru:
        if len(caption) > 0:
            caption += ", "
        caption += deepbooru.model.tag_multi(image)
    if params.process_caption_wd:
        if len(caption) > 0:
            caption += ", "
        caption += wd14.model.tag_multi(image)
    if params.process_caption_clip:
        if len(caption) > 0:
            caption += ", "
        caption += cliptagger.model.generate_caption(image)
    if params.process_caption_clip2:
        if len(caption) > 0:
            caption += ", "
        caption += cliptagger2.model.generate_caption(image)

    filename_part = params.src
    filename_part = os.path.splitext(filename_part)[0]
    filename_part = os.path.basename(filename_part)

    basename = f"{index:05}-{params.subindex}-{filename_part}"
    image_path = os.path.join(params.dstdir, f"{basename}.png")
    image.save(image_path)

    if params.preprocess_txt_action == 'prepend' and existing_caption:
        caption = existing_caption + ' ' + caption
    elif params.preprocess_txt_action == 'append' and existing_caption:
        caption = caption + ' ' + existing_caption
    elif params.preprocess_txt_action == 'copy' and existing_caption:
        caption = existing_caption

    caption = caption.strip()

    if len(caption) > 0:
        output_path = os.path.join(params.dstdir, f"{basename}.txt")
        with open(output_path, "w", encoding="utf8") as file:
            file.write(caption)
        params.subindex += 1
    params.outputFiles.append(f"{basename}.png")
    params.outputDetail.append({
        "dest": image_path,
        "src": params.src,
        "name": f"{basename}.png",
    })


def save_pic(image, index, params, existing_caption=None):
    save_pic_with_caption(image, index, params, existing_caption=existing_caption)

    if params.flip:
        save_pic_with_caption(ImageOps.mirror(image), index, params, existing_caption=existing_caption)


def split_pic(image, inverse_xy, width, height, overlap_ratio):
    if inverse_xy:
        from_w, from_h = image.height, image.width
        to_w, to_h = height, width
    else:
        from_w, from_h = image.width, image.height
        to_w, to_h = width, height
    h = from_h * to_w // from_w
    if inverse_xy:
        image = image.resize((h, to_w))
    else:
        image = image.resize((to_w, h))

    split_count = math.ceil((h - to_h * overlap_ratio) / (to_h * (1.0 - overlap_ratio)))
    y_step = (h - to_h) / (split_count - 1)
    for i in range(split_count):
        y = int(y_step * i)
        if inverse_xy:
            splitted = image.crop((y, 0, y + to_h, to_w))
        else:
            splitted = image.crop((0, y, to_w, y + to_h))
        yield splitted


def listfiles(dirname):
    return os.listdir(dirname)


def preprocess_work(
        process_src=None,
        process_dst=None,
        process_width=512,
        process_height=512,
        preprocess_txt_action=None,
        process_flip=False,
        process_split=False,
        process_caption=False,
        process_caption_deepbooru=False,
        process_caption_clip=False,
        process_caption_clip2=False,
        process_caption_wd=False,
        wd_general_threshold=None,
        wd_character_threshold=None,
        wd_model_name=None,
        split_threshold=0.5, overlap_ratio=0.2, process_focal_crop=False, process_focal_crop_face_weight=0.9,
        process_focal_crop_entropy_weight=0.3, process_focal_crop_edges_weight=0.5, process_focal_crop_debug=False,
        process_multicrop=None, process_multicrop_mindim=None, process_multicrop_maxdim=None,
        process_multicrop_minarea=None, process_multicrop_maxarea=None, process_multicrop_objective=None,
        process_multicrop_threshold=None, model_path="../assets/model-resnet_custom_v31.pt",
        process_folders=None,
        process_images=None,
        input_images: Optional[List[PreprocessImageFile]] = None,
        anime_face_detect=False,
        anime_face_detect_ratio=1.0,
        anime_person_detect=False,
        anime_person_detect_ratio=0,
        anime_half_body_detect=False,
        anime_half_body_detect_ratio=0,
        box_to_top=False,

):
    # loading model
    output.printJsonOutput(
        message="Start process",
        event="preprocess_start",
        vars={}
    )
    if process_caption:
        share.interrogator.load()
    if process_caption_deepbooru:
        deepbooru.model.load()
    if process_caption_wd:
        wd14.model.load(
            model_name=wd_model_name,
        )
        wd14.model.set_param(
            general_threshold=wd_general_threshold,
            character_threshold=wd_character_threshold,
        )
    if process_caption_clip:
        cliptagger.model.load()
    if process_caption_clip2:
        cliptagger2.model.load()

    width = process_width
    height = process_height
    files = []
    dst = os.path.abspath(process_dst)
    split_threshold = max(0.0, min(1.0, split_threshold))
    overlap_ratio = max(0.0, min(0.9, overlap_ratio))

    os.makedirs(dst, exist_ok=True)
    if input_images is None:
        folders = []
        if process_src:
            src = os.path.abspath(process_src)
            assert src != dst, 'same directory specified as source and destination'
            folders.append(src)

        if process_folders:
            folders.append(process_folders.map(lambda x: os.path.abspath(x)))
        for folder in folders:
            for folderFile in listfiles(folder):
                files.append(os.path.join(folder, folderFile))

            # files.append(map(lambda x: os.path.join(folder, x), listfiles(folder)))
        if process_images:
            files.extend(map(lambda x: os.path.abspath(x), process_images))
    else:
        files = input_images
    # shared.state.job = "preprocess"
    # shared.state.textinfo = "Preprocessing..."
    # shared.state.job_count = len(files)

    params = PreprocessParams()
    params.dstdir = dst
    params.flip = process_flip
    params.process_caption = process_caption
    params.process_caption_deepbooru = process_caption_deepbooru
    params.preprocess_txt_action = preprocess_txt_action
    params.process_caption_wd = process_caption_wd
    params.process_caption_clip = process_caption_clip
    params.process_caption_clip2 = process_caption_clip2

    # pbar = tqdm.tqdm(files)
    if anime_face_detect:
        anifacedec.load_model()
    if anime_person_detect:
        anipersondec.load_model()
    if anime_half_body_detect:
        anihalfpersondec.load_model()
    index = 0
    for imagefile in files:
        output.printJsonOutput(
            message=f"Preprocessing [Image {index + 1}/{len(files)}]",
            vars={
                "index": index,
                "total": len(files),
            },
            event="process_progress",
        )
        params.subindex = 0
        filename = imagefile
        img = None
        if type(imagefile) is PreprocessImageFile:
            img = imagefile.img
            filename = imagefile.filename
        else:
            try:
                img = Image.open(filename).convert("RGB")
            except Exception:
                continue
        if anime_face_detect:
            img = yolomodeldec.crop_image_with_detection(img, anime_face_detect_ratio, anifacedec.model)
        if anime_person_detect:
            img = yolomodeldec.crop_image_with_detection(img, anime_person_detect_ratio, anipersondec.model,
                                                         box_to_top=box_to_top)
        if anime_half_body_detect:
            img = yolomodeldec.crop_image_with_detection(img, anime_half_body_detect_ratio, anihalfpersondec.model,
                                                         box_to_top=box_to_top)
        # description = f"Preprocessing [Image {index}/{len(files)}]"
        # pbar.set_description(description)
        # shared.state.textinfo = description

        params.src = filename

        existing_caption = None
        existing_caption_filename = os.path.splitext(filename)[0] + '.txt'
        if os.path.exists(existing_caption_filename):
            with open(existing_caption_filename, 'r', encoding="utf8") as file:
                existing_caption = file.read()

        # if shared.state.interrupted:
        #     break

        if img.height > img.width:
            ratio = (img.width * height) / (img.height * width)
            inverse_xy = False
        else:
            ratio = (img.height * width) / (img.width * height)
            inverse_xy = True

        process_default_resize = True

        if process_split and ratio < 1.0 and ratio <= split_threshold:
            for splitted in split_pic(img, inverse_xy, width, height, overlap_ratio):
                save_pic(splitted, index, params, existing_caption=existing_caption)
            process_default_resize = False

        if process_focal_crop and img.height != img.width:

            dnn_model_path = None
            try:
                path = huggingface_hub.hf_hub_download(
                    FACE_CROP_REPO, FACE_CROP,
                )
                dnn_model_path = path
            except Exception as e:
                print(
                    "Unable to load face detection model for auto crop selection. Falling back to lower quality haar method.",
                    e)

            autocrop_settings = autocrop.Settings(
                crop_width=width,
                crop_height=height,
                face_points_weight=process_focal_crop_face_weight,
                entropy_points_weight=process_focal_crop_entropy_weight,
                corner_points_weight=process_focal_crop_edges_weight,
                annotate_image=process_focal_crop_debug,
                dnn_model_path=dnn_model_path,
            )
            for focal in autocrop.crop_image(img, autocrop_settings):
                save_pic(focal, index, params, existing_caption=existing_caption)
            process_default_resize = False

        if process_multicrop:
            cropped = multicrop_pic(img, process_multicrop_mindim, process_multicrop_maxdim, process_multicrop_minarea,
                                    process_multicrop_maxarea, process_multicrop_objective, process_multicrop_threshold)
            if cropped is not None:
                save_pic(cropped, index, params, existing_caption=existing_caption)
            else:
                print(
                    f"skipped {img.width}x{img.height} image {filename} (can't find suitable size within error threshold)")
            process_default_resize = False

        if process_default_resize:
            img = images.resize_image(1, img, width, height)
            save_pic(img, index, params, existing_caption=existing_caption)
        index += 1
    if output.detail_output:
        output.printJsonOutput("Done", vars=params.outputDetail, event="preprocess_done")
    else:
        output.printJsonOutput("Done", vars=params.outputFiles, event="preprocess_done")
    return params
