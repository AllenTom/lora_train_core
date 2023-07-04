import argparse
import sys

from modules import preprocess, output, share

sys.path.append("./finetune")

# def preprocess_work(
#         process_src,
#         process_dst,
#         process_width,
#         process_height,
#         preprocess_txt_action,
#         process_flip,
#         process_split,
#         process_caption,
#         process_caption_deepbooru=False,
#         split_threshold=0.5, overlap_ratio=0.2, process_focal_crop=False, process_focal_crop_face_weight=0.9,
#         process_focal_crop_entropy_weight=0.3, process_focal_crop_edges_weight=0.5, process_focal_crop_debug=False,
#         process_multicrop=None, process_multicrop_mindim=None, process_multicrop_maxdim=None,
#         process_multicrop_minarea=None, process_multicrop_maxarea=None, process_multicrop_objective=None,
#         process_multicrop_threshold=None, model_path="../assets/model-resnet_custom_v31.pt")
def main():
    parser = argparse.ArgumentParser()
    # Add a flag argument
    parser.add_argument('--src', action='store')
    parser.add_argument('--dest', action='store')
    parser.add_argument('--width', action='store',default=512, type=int)
    parser.add_argument('--height', action='store',default=512, type=int)
    parser.add_argument('--txt_action', action='store',default="prepend")
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--caption', action='store_true')
    parser.add_argument('--caption_deepbooru', action='store_true')
    parser.add_argument('--split_threshold', action='store',default=0.5, type=float)
    parser.add_argument('--overlap_ratio', action='store',default=0.2, type=float)
    parser.add_argument('--focal_crop', action='store_true')
    parser.add_argument('--focal_crop_face_weight', action='store',default=0.9, type=float)
    parser.add_argument('--focal_crop_entropy_weight', action='store',default=0.3, type=float)
    parser.add_argument('--focal_crop_edges_weight', action='store',default=0.5, type=float)
    parser.add_argument('--focal_crop_debug', action='store_true')
    parser.add_argument('--multicrop', action='store_true')
    parser.add_argument('--multicrop_mindim', action='store',default=None, type=int)
    parser.add_argument('--multicrop_maxdim', action='store',default=None, type=int)
    parser.add_argument('--multicrop_minarea', action='store',default=None, type=int)
    parser.add_argument('--multicrop_maxarea', action='store',default=None, type=int)
    parser.add_argument('--multicrop_objective', action='store',default=None)
    parser.add_argument('--multicrop_threshold', action='store',default=None, type=float)
    parser.add_argument('--json_out', action='store_true',default=None)
    # Parse the arguments
    args = parser.parse_args()
    output.jsonOut = args.json_out

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
        split_threshold=args.split_threshold, overlap_ratio=args.overlap_ratio, process_focal_crop=args.focal_crop,
        process_focal_crop_face_weight=args.focal_crop_face_weight,
        process_focal_crop_entropy_weight=args.focal_crop_entropy_weight,
        process_focal_crop_edges_weight=args.focal_crop_edges_weight, process_focal_crop_debug=args.focal_crop_debug,
        process_multicrop=args.multicrop, process_multicrop_mindim=args.multicrop_mindim,
        process_multicrop_maxdim=args.multicrop_maxdim, process_multicrop_minarea=args.multicrop_minarea,
        process_multicrop_maxarea=args.multicrop_maxarea, process_multicrop_objective=args.multicrop_objective,
        process_multicrop_threshold=args.multicrop_threshold,
        model_path="assets/model-resnet_custom_v31.pt")


if __name__ == '__main__':
    main()