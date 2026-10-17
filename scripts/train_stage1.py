"""Stage-1 training: fine-tune a ResNet-50 frame classifier on embryo frames."""
import argparse
import json
import os

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.MFHE.image_dataset import collate_fn as collate_fn_MFHE
from data.SFHE.image_dataset import collate_fn as collate_fn_SFHE
from data.dataset_builder import build_train_test_set_stage1
from models.resnet50 import ResNet
from trainer.stage1_trainer import Trainer
from utils import resolve_path, set_random_seed, setup_logger

PATH_KEYS = ("train_split", "test_split", "video_root", "anno_root",
             "development_stage_list", "base_save_dir")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-1 training")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the Stage-1 config file.")
    parser.add_argument("--device", type=int, default=None,
                        help="GPU index. Overrides config['device'].")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name. Overrides config['exp_name'].")
    return parser.parse_args()


def main():
    args = parse_args()
    config = json.load(open(args.config, "r"))

    if args.exp_name is not None:
        config["exp_name"] = args.exp_name
    if args.device is not None:
        config["device"] = [args.device]
    for key in PATH_KEYS:
        if config.get(key):
            config[key] = resolve_path(config[key])

    set_random_seed(config["seed"])

    base_save_dir = os.path.join(config["base_save_dir"], config["exp_name"])
    config["model_save_dir"] = os.path.join(base_save_dir, "trained_model")
    config["log_dir"] = os.path.join(base_save_dir, "logging")
    config["pred_save_dir"] = os.path.join(base_save_dir, "prediction")
    for directory in (config["model_save_dir"], config["log_dir"], config["pred_save_dir"]):
        os.makedirs(directory, exist_ok=True)

    with open(os.path.join(base_save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger = setup_logger(log_file=os.path.join(config["log_dir"], "train.log"))
    writer = SummaryWriter(log_dir=os.path.join(config["log_dir"], "tensorboard"))

    # Only single-GPU training is supported.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config["device"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(config).to(device)
    logger.info(f"Model trainable parameters: {model.trainable_parameters / 1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.999),
                            lr=config["base_lr"], weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"],
                                  eta_min=config["min_lr"], last_epoch=-1)

    train_set, test_set = build_train_test_set_stage1(config)
    collate_fn = collate_fn_MFHE if config["dataset_name"] == "MFHE" else collate_fn_SFHE
    num_workers = config["num_workers"]

    train_loader = DataLoader(train_set, batch_size=config["train_batch_size"],
                              collate_fn=collate_fn, shuffle=True, num_workers=num_workers,
                              drop_last=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=config["test_batch_size"],
                             collate_fn=collate_fn, num_workers=num_workers,
                             shuffle=False, pin_memory=True)

    trainer = Trainer(config=config, model=model, logger=logger, writter=writer, device=device)
    trainer.train(train_loader=train_loader, optimizer=optimizer,
                  schedular=scheduler, test_loader=test_loader, video_list=None)


if __name__ == "__main__":
    main()

