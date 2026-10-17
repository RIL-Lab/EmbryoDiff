import os
import sys

# Allow running this file directly, e.g. "python data/MFHE/extract_video_feats.py".
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from PIL import Image
import json

import pandas as pd
from torchvision import transforms
from tqdm import tqdm
from typing import List, Dict
import argparse
from models.resnet50 import ResNet
from data.MFHE.annotation_index import (
    missing_annotations_error,
    resolve_anno_root,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-focal Video Feature Extractor using ResNet"
    )

    parser.add_argument("--video_root", type=str, required=True,
                        help="Root directory of video frames (for F0). "
                             "Other focal planes will use video_root_<focal>.")
    parser.add_argument("--resnet_path", type=str, required=True,
                        help="Path to pretrained ResNet model")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the model config JSON saved next to the "
                             "Stage-1 checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save extracted features")
    parser.add_argument("--gpu_id", type=str, default="0",
                        help="GPU ID to use, -1 for CPU")
    parser.add_argument("--resize", type=int, default=256,
                        help="Resize dimension for preprocessing")
    parser.add_argument("--crop", type=int, default=224,
                        help="Crop dimension for preprocessing")
    parser.add_argument("--split_file", type=str, required=True,
                        help="CSV file that lists sample annotation filenames")
    parser.add_argument("--anno_root", type=str, default=None,
                        help="Directory of per-video annotation CSVs. Defaults "
                             "to <split_file dir>/per_video.")
    parser.add_argument("--focal_planes", type=str, default="F0",
                        help="Comma-separated focal planes, e.g. "
                             "\"F-45,F-30,F-15,F0,F15,F30,F45\" "
                             "or \"F0\" for single focal plane.")
    parser.add_argument("--subsample_rate", type=int, default=1,
                        help="Temporal subsample rate for frames")

    return parser.parse_args()


def load_video_metadata(split_file: str, anno_root: str,
                        subsample_rate: int) -> Dict[str, List[str]]:
    if not os.path.isdir(anno_root):
        raise missing_annotations_error(split_file, anno_root)

    sample_list = pd.read_csv(split_file)["filename"].tolist()
    video_metadata: Dict[str, List[str]] = {}

    for anno_file in tqdm(sample_list, desc="Loading metadata"):
        abs_path = os.path.join(anno_root, anno_file)
        try:
            df = pd.read_csv(abs_path)
            if df.empty:
                continue
        except Exception as e:
            print(f"Skipping {abs_path}: {str(e)}")
            continue

        if subsample_rate > 1:
            df = df.iloc[::subsample_rate].reset_index(drop=True)

        video_name = anno_file.split("_annotations")[0]
        frame_names = df["frame_name"].values.tolist()
        video_metadata[video_name] = frame_names

    if not video_metadata:
        raise missing_annotations_error(split_file, anno_root)

    return video_metadata


def build_frame_path(video_name: str,
                     frame_name: str,
                     focal: str,
                     video_root: str) -> str:
    """
    - F0:   base_dir = video_root
    - others: base_dir = f"{video_root}_{focal}"
    """
    base_dir = f"{video_root}_{focal}" if focal != "F0" else video_root
    return os.path.join(base_dir, video_name, frame_name)


class ResNetFeatureExtractor:
    def __init__(self,
                 config_path: str,
                 model_path: str,
                 device: str = "cuda",
                 resize: int = 256,
                 crop: int = 224):
        self.device = device

        with open(config_path, "r") as f:
            config = json.load(f)
        self.model = ResNet(config=config)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device).eval()
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.251, 0.251, 0.251],
                std=[0.199, 0.199, 0.199]
            )
        ])

    def extract(self, images: List[Image.Image]) -> np.ndarray:
        tensor_batch = torch.stack(
            [self.transform(img) for img in images],
            dim=0
        ).to(self.device)

        with torch.no_grad():
            features = self.model.forward_features(tensor_batch)
        # [N, 2048]
        return features.cpu().numpy()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = "cpu" if args.gpu_id == "-1" else "cuda"

    os.makedirs(args.output_dir, exist_ok=True)

    focal_planes = [f.strip() for f in args.focal_planes.split(",") if f.strip()]

    extractor = ResNetFeatureExtractor(
        config_path=args.config_path,
        model_path=args.resnet_path,
        device=device,
        resize=args.resize,
        crop=args.crop
    )

    anno_root = resolve_anno_root(args.split_file, args.anno_root)
    video_meta = load_video_metadata(
        split_file=args.split_file,
        anno_root=anno_root,
        subsample_rate=args.subsample_rate
    )

    for video_name, frame_names in tqdm(video_meta.items(),
                                        desc="Processing videos"):
        all_features_per_focal: List[List[np.ndarray]] = [
            [] for _ in focal_planes
        ]

        for frame_name in frame_names:
            loaded_images: List[Image.Image] = []
            for focal in focal_planes:
                frame_path = build_frame_path(
                    video_name=video_name,
                    frame_name=frame_name,
                    focal=focal,
                    video_root=args.video_root
                )
                try:
                    img = Image.open(frame_path).convert("RGB")
                except Exception as e:
                    print(f"Error loading {frame_path}: {str(e)}")
                    img = Image.new("RGB", (args.crop, args.crop))

                loaded_images.append(img)

            # [num_focal, 2048]
            batch_features = extractor.extract(loaded_images)

            for i, feat in enumerate(batch_features):
                all_features_per_focal[i].append(feat)

        # [F, T, 2048]
        combined_features = np.array([
            np.stack(feats, axis=0)  # [T, 2048]
            for feats in all_features_per_focal
        ])  # [F, T, 2048]

        output_path = os.path.join(
            args.output_dir,
            f"{video_name}.npy"
        )
        np.save(output_path, combined_features)

    print(f"Features saved to: {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
