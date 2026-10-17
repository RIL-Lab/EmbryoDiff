import os
import numpy as np
import torch
import sys

# Allow running this file directly, e.g. "python data/SFHE/extract_video_feats.py".
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import argparse
from models.resnet50 import ResNet
from typing import List


def parse_args():
    parser = argparse.ArgumentParser(description="Video Feature Extractor using ResNet")

    parser.add_argument("--split_files", type=str, required=True,
                        help="Comma-separated paths to JSON files containing video annotations")
    parser.add_argument("--video_root", type=str, required=True,
                        help="Root directory of video frames")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to model config JSON")
    parser.add_argument("--resnet_path", type=str, required=True,
                        help="Path to pretrained ResNet model")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save extracted features")
    parser.add_argument("--gpu_id", type=str, default="0",
                        help="GPU ID to use (overrides device argument)")
    parser.add_argument("--resize", type=int, default=256,
                        help="Resize dimension for preprocessing")
    parser.add_argument("--crop", type=int, default=224,
                        help="Crop dimension for preprocessing")
    
    return parser.parse_args()
 
def load_video_metadata(split_files: List[str], video_root: str) -> dict:
    video_metadata = {}
    
    for split_file in split_files:
        with open(split_file, "r") as f:
            anno_infos = json.load(f)
        
        for vid, info in anno_infos.items():
            frame_paths = [os.path.join(video_root, p) for p in info["frame_paths"]]
            video_metadata[vid] = frame_paths
    
    return video_metadata

class ResNetFeatureExtractor:
    def __init__(self, config_path: str, model_path: str, device: str = "cuda",
                 resize: int = 256, crop: int = 224):
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def extract(self, images: list) -> np.ndarray:
        tensor_batch = torch.stack([self.transform(img) for img in images], dim=0).to(self.device)
        with torch.no_grad():
            features = self.model.forward_features(tensor_batch)
        return features.cpu().numpy()  # [N, 2048]


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cpu" if args.gpu_id == "-1" else f"cuda"

    extractor = ResNetFeatureExtractor(
        config_path=args.config_path,
        model_path=args.resnet_path,
        device=device,
        resize=args.resize,
        crop=args.crop
    )
    
    split_files = [s.strip() for s in args.split_files.split(",")]
    video_meta = load_video_metadata(split_files, args.video_root)
    
    for video_name, frame_paths in tqdm(video_meta.items(), desc="Processing videos"):
        loaded_images = []
        for path in frame_paths:
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"Warning: Failed to load {path}, using placeholder")
                img = Image.new("RGB", (args.crop, args.crop))
            loaded_images.append(img)
        
        features = extractor.extract(loaded_images)
        output_path = os.path.join(args.output_dir, f"{video_name}.npy")
        np.save(output_path, features)
    
    print(f"Feature extraction completed! Results saved to: {args.output_dir}")
 

if __name__ == "__main__":
    args = parse_args()
    main(args)

