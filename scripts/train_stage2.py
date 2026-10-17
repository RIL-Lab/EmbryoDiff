"""Stage-2 training: diffusion model over the Stage-1 video features."""
import argparse
import json
import os
import shutil

import numpy as np
import torch

from data.dataset_builder import build_train_test_set
from models.model_builder import build_model
from trainer.stage2_trainer import Trainer
from utils import resolve_path, set_random_seed, setup_logger

PATH_KEYS = ("train_split", "test_split", "video_feat_folder",
             "development_stage_list", "anno_root")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage-2 training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the Stage-2 config file.")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU index, -1 for CPU.")
    parser.add_argument("--exp_name", type=str, default="debug",
                        help="Name of the experiment folder.")
    return parser.parse_args()


def main():
    args = parse_args()
    configs = json.load(open(args.config, mode="r", encoding="utf-8"))

    for key in PATH_KEYS:
        if configs["data"].get(key):
            configs["data"][key] = resolve_path(configs["data"][key])
    configs["train"]["result_dir"] = resolve_path(configs["train"]["result_dir"])

    # During training, do not set seed for better convergence.
    # Set seed only for debug purpose to test reproductivity.
    if configs["train"]["debug"]:
        set_random_seed(configs["train"]["seed"])

    experiment_dir = os.path.join(configs["train"]["result_dir"], args.exp_name)
    configs["train"]["result_dir"] = experiment_dir
    os.makedirs(experiment_dir, exist_ok=True)

    logger = setup_logger(log_file=os.path.join(experiment_dir, "train.log"))
    shutil.copy(args.config, os.path.join(experiment_dir, os.path.basename(args.config)))

    # only support single GPU training.
    if args.device != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    event_list = np.loadtxt(configs["data"]["development_stage_list"], dtype=str)
    train_set, test_set = build_train_test_set(configs)
    model = build_model(configs, device=device)

    trainer = Trainer(
        model=model,
        train_dataset=train_set,
        test_dataset=test_set,
        logger=logger,
        event_list=event_list,
        config=configs,
        device=device,
    )
    trainer.train()


if __name__ == "__main__":
    main()

