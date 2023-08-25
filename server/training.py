import io
import logging
import os
import threading
from contextlib import redirect_stdout

from library import train_util
from server import project, paths
from train_network import setup_parser, train
from modules import share

out_text = ""
default_train_param = {
    'network_module': 'networks.lora',
    'save_model_as': 'safetensors',
    'train_batch_size': 1,
    'caption_extension': '.txt',
    'mixed_precision': 'fp16',
    'save_precision': 'fp16',
    'cache_latents': True,
    'seed': 1234,
    'learning_rate': 0.0001,
    'lr_scheduler': 'constant',
    'optimizer_type': 'AdamW8bit',
    'text_encoder_lr': 5e-05,
    'unet_lr': 0.0001,
    'network_dim': 128,
    'network_alpha': 128,
    'resolution': '512,512',
    'gradient_accumulation_steps': 1,
    'prior_loss_weight': 1,
    'lr_scheduler_num_cycles': 1,
    'lr_scheduler_power': 1,
    'clip_skip': 1,
    'max_token_length': 150,
    'xformers': True,
    'bucket_no_upscale': True,
    'bucket_reso_steps': 64,
    'vae_batch_size': 1,
    'max_data_loader_n_workers': 8,
    'sample_sampler': 'euler_a',
    'save_every_n_steps': 100
}


def build_train_args(meta: project.ProjectMeta, config: project.TrainConfig):
    model_output_path = paths.get_model_output_folder(meta.project_path)
    dataset_folder_path = paths.get_dataset_folder(meta.project_path)
    args = []
    # merge default args
    arg_dict = {
        **default_train_param, **config.extra_params,
        "output_dir": os.path.abspath(model_output_path),
        "output_name": config.model_name,
        "training_comment": config.model_name,
        "train_data_dir": os.path.abspath(dataset_folder_path) + "123423dw",
        "pretrained_model_name_or_path": config.pretrained_model_name_or_path,
        "disable_callbacks": True
    }

    for key, value in arg_dict.items():
        if type(value) == bool:
            if value:
                args.append(f"--{key}")
            continue
        args.append(f"--{key}")
        args.append(f"{value}")
    print(os.path.abspath(dataset_folder_path))
    return args


def train_func(args):
    try :
        train(args)
    except Exception as e:
        print(e)




def train_project(id: str, config_id: str):
    project_path = project.get_project_path(id)
    meta = project.ProjectMeta.load_from_file(os.path.join(project_path, "project.json"))
    config = next(filter(lambda x: x.id == config_id, meta.train_configs))
    parser = setup_parser()
    args = parser.parse_args(build_train_args(meta, config))
    args = train_util.read_config_from_file(args, parser)
    thread = threading.Thread(target=train_func, args=(args,))
    thread.start()
    return "ok"
