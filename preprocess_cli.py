import sys

sys.path.append(".\\repositories\\BLIP")
import argparse
import base64
import json
import sys

from modules import preprocess, output, share


def main():
    parser = argparse.ArgumentParser()
    # Add a flag argument
    parser.add_argument('--src', action='store')
    parser.add_argument('--dest', action='store')
    parser.add_argument('--width', action='store', default=512, type=int)
    parser.add_argument('--height', action='store', default=512, type=int)
    parser.add_argument('--txt_action', action='store', default="prepend")
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--caption', action='store_true')
    parser.add_argument('--caption_deepbooru', action='store_true')
    parser.add_argument('--caption_wd', action='store_true')
    parser.add_argument('--wd_general_threshold', action='store', default=0.5, type=float)
    parser.add_argument('--wd_character_threshold', action='store', default=0.5, type=float)
    parser.add_argument('--wd_model_name', action='store', default='MOAT', type=str)
    parser.add_argument('--split_threshold', action='store', default=0.5, type=float)
    parser.add_argument('--overlap_ratio', action='store', default=0.2, type=float)
    parser.add_argument('--focal_crop', action='store_true')
    parser.add_argument('--focal_crop_face_weight', action='store', default=0.9, type=float)
    parser.add_argument('--focal_crop_entropy_weight', action='store', default=0.3, type=float)
    parser.add_argument('--focal_crop_edges_weight', action='store', default=0.5, type=float)
    parser.add_argument('--focal_crop_debug', action='store_true')
    parser.add_argument('--multicrop', action='store_true')
    parser.add_argument('--multicrop_mindim', action='store', default=None, type=int)
    parser.add_argument('--multicrop_maxdim', action='store', default=None, type=int)
    parser.add_argument('--multicrop_minarea', action='store', default=None, type=int)
    parser.add_argument('--multicrop_maxarea', action='store', default=None, type=int)
    parser.add_argument('--multicrop_objective', action='store', default=None)
    parser.add_argument('--multicrop_threshold', action='store', default=None, type=float)
    parser.add_argument('--json_out', action='store_true', default=None)
    parser.add_argument('--json_input', action='store', default=None, type=str)
    parser.add_argument("--json_input_base64", action='store', default=None, type=str)
    parser.add_argument("--output_detail", action='store_true')
    parser.add_argument("--anime_face", action='store_true')
    parser.add_argument("--anime_face_ratio", action='store', default=1, type=float)
    parser.add_argument("--anime_person", action='store_true')
    parser.add_argument("--anime_person_ratio", action='store', default=0, type=float)
    parser.add_argument("--anime_half",action='store_true')
    parser.add_argument("--anime_half_ratio", action='store', default=0, type=float)
    parser.add_argument("--to_anime_body_top",action='store_true')

    # Parse the arguments
    args = parser.parse_args()
    output.jsonOut = args.json_out
    output.detail_output = args.output_detail

    input_json = args.json_input
    if args.json_input_base64 is not None:
        decoded_bytes = base64.b64decode(args.json_input_base64)
        decoded_string = decoded_bytes.decode('utf-8')
        input_json = decoded_string
    input_folders = None
    input_files = None
    if input_json is not None:
        json_data = json.loads(input_json)
        if "folders" in json_data:
            input_folders = json_data["folders"]
        if "files" in json_data:
            input_files = json_data["files"]

    preprocess.preprocess_work(
        process_src=args.src,
        process_dst=args.dest,
        process_width=args.width,
        process_height=args.height,
        preprocess_txt_action=args.txt_action,
        process_flip=args.flip,
        process_split=args.split,
        process_caption=args.caption,
        process_caption_deepbooru=args.caption_deepbooru,
        process_caption_wd=args.caption_wd,
        wd_general_threshold=args.wd_general_threshold,
        wd_character_threshold=args.wd_character_threshold,
        wd_model_name=args.wd_model_name,
        split_threshold=args.split_threshold, overlap_ratio=args.overlap_ratio, process_focal_crop=args.focal_crop,
        process_focal_crop_face_weight=args.focal_crop_face_weight,
        process_focal_crop_entropy_weight=args.focal_crop_entropy_weight,
        process_focal_crop_edges_weight=args.focal_crop_edges_weight, process_focal_crop_debug=args.focal_crop_debug,
        process_multicrop=args.multicrop, process_multicrop_mindim=args.multicrop_mindim,
        process_multicrop_maxdim=args.multicrop_maxdim, process_multicrop_minarea=args.multicrop_minarea,
        process_multicrop_maxarea=args.multicrop_maxarea, process_multicrop_objective=args.multicrop_objective,
        process_multicrop_threshold=args.multicrop_threshold,
        model_path="assets/model-resnet_custom_v31.pt",
        process_folders=input_folders,
        process_images=input_files,
        anime_face_detect=args.anime_face,
        anime_face_detect_ratio=args.anime_face_ratio,
        anime_person_detect=args.anime_person,
        anime_person_detect_ratio=args.anime_person_ratio,
        anime_half_body_detect=args.anime_half,
        anime_half_body_detect_ratio=args.anime_half_ratio,
        box_to_top=args.to_anime_body_top
    )


if __name__ == '__main__':
    main()
