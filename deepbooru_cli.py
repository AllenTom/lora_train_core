import argparse

import PIL

from modules import deepbooru
import json
import os


def get_image_list_from_dir(dir):
    images = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            file = file.lower()
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                images.append(os.path.join(root, file))
    return images
def main():
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add a flag argument
    parser.add_argument('--image', action='store')
    parser.add_argument('--dir', action='store')
    # Parse the arguments
    args = parser.parse_args()
    dbr = deepbooru.DeepDanbooru()
    dbr.load()
    dbr.start()
    srcs = []
    if args.dir:
        srcs = get_image_list_from_dir(args.dir)
    else:
        srcs = [args.image]
    if srcs is None:
        print("No images found")
        return
    result = []
    for src in srcs:
        testImage = PIL.Image.open(src)
        tag_str = dbr.tag_multi(testImage)
        tags = tag_str.split(",")
        result.append({
            "filename": os.path.basename(src),
            "tags": [x.strip() for x in tags]
        })

    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()