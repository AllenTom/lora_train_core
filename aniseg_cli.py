import base64
import json
import os
import initapp

initapp.init_global()
import argparse
import time

import cv2
import torch
import numpy as np
import glob
import huggingface_hub

REPO_ID = "skytnt/anime-seg"
MODEL_FILE = "isnetis.ckpt"
from modules import anisgemodel, aniseg, output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None,
                        help='input data dir')
    parser.add_argument('--out', type=str, default='out',
                        help='output dir')
    parser.add_argument('--img-size', type=int, default=1024,
                        help='hyperparameter, input image size of the net')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='cpu or cuda:0')
    parser.add_argument('--fp32', action='store_true', default=False,
                        help='disable mix precision')
    parser.add_argument('--json_out', action='store_true', default=None)
    parser.add_argument("--json_input_base64", action='store', default=None, type=str)

    opt = parser.parse_args()
    if opt.json_out:
        output.jsonOut = True
    input_json = None
    if opt.json_input_base64 is not None:
        decoded_bytes = base64.b64decode(opt.json_input_base64)
        decoded_string = decoded_bytes.decode('utf-8')
        input_json = decoded_string
    input_files = []
    if input_json is not None:
        json_data = json.loads(input_json)
        if "folders" in json_data:
            for folder in json_data["folders"]:
                input_files.extend(glob.glob(f'{folder}/*.*'))
        if "files" in json_data:
            input_files.extend(json_data["files"])
    if opt.data is not None:
        input_files.extend(glob.glob(f'{opt.data}/*.*'))


    device = torch.device(opt.device)
    cpkt_path = huggingface_hub.hf_hub_download(REPO_ID, MODEL_FILE)
    model = anisgemodel.AnimeSegmentation.try_load('isnet_is', cpkt_path, opt.device, img_size=opt.img_size)
    model.eval()
    model.to(device)

    if not os.path.exists(opt.out):
        os.mkdir(opt.out)
    items = input_files
    for i, path in enumerate(sorted(items)):
        img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = aniseg.get_mask(model, img, use_amp=not opt.fp32, s=opt.img_size)
        img = np.concatenate((mask * img + 1 - mask, mask * 255), axis=2).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        unix_time = int(time.time())
        unix_time_string = str(unix_time)

        output_name = f'seg_{unix_time_string}_{os.path.basename(path)}'
        cv2.imwrite(f'{opt.out}\\{output_name}', img)
        output.printJsonOutput(f"process {i + 1} of {len(items)}", event="aniseg_progress", vars={
            "current": i + 1,
            "total": len(items),
            "path": path,
            "output": f'{opt.out}\\{output_name}'
        })
