import os
import sys


import argparse
import base64
import json

import PIL

from deepbooru_cli import get_image_list_from_dir
from modules import cliptagger


def process():
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add a flag argument
    parser.add_argument('--image', action='store')
    parser.add_argument('--input_base64', action='store')
    parser.add_argument('--dir', action='store')
    parser.add_argument('--threshold', action='store', default=float(0.7), type=float)
    parser.add_argument('--with_rank', action='store_true')
    parser.add_argument('--per', action='store_true')
    parser.add_argument('--no_result', action='store_false')
    # Parse the arguments
    args = parser.parse_args()
    model = cliptagger.InterrogateModels("interrogate")
    model.load()

    srcs = []
    if args.dir:
        srcs = get_image_list_from_dir(args.dir)
    else:
        srcs = [args.image]
    if srcs is None:
        print("No images found")
        return
    result = []
    if args.input_base64:
        # decode
        decoded_bytes = base64.b64decode(args.input_base64)
        decoded_string = decoded_bytes.decode('utf-8')
        input_obj = json.loads(decoded_string)
        if "files" in input_obj:
            srcs = input_obj["files"]

    for src in srcs:
        testImage = PIL.Image.open(src)
        tag_str = model.interrogate(testImage)
        print(json.dumps({
            "filename": os.path.basename(src),
            "tags": tag_str
        },ensure_ascii=False))

if __name__ == '__main__':
    process()
