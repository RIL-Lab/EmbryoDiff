import os
import numpy as np
import torch
import json
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import logging
from collections.abc import Mapping, Sequence
from torch.utils.data.dataloader import default_collate
from typing import Dict, List, Any, Optional


class EmbryoImageSFHE(Dataset):
    def __init__(self, anno_path: str, video_root: str, 
                num_classes: int = 15, transform: Optional[callable] = None, debug=False):
        super().__init__()
        self.anno_path = anno_path
        self.video_root = video_root
        self.num_classes = num_classes
        self.transform = transform

        self.anno_infos = json.load(open(self.anno_path, 'r'))
        self.video_list = self.anno_infos.keys()

        self.n2v_mapping = {i: name for i, name in enumerate(self.video_list)}
        self.v2n_mapping = {name: i for i, name in enumerate(self.video_list)}

        self.data = self._preload_data()

        if debug:
            self.data = self.data[:1024]

    def _preload_data(self) -> List[Dict]:
        data = []
        for vid in self.video_list:
            info = self.anno_infos[vid]

            frame_paths = info["frame_paths"]
            frame_paths = [os.path.join(self.video_root, path) for path in frame_paths]
            c_labels = info["coarse"]["frame_labels"]
            f_labels = info["fine"]["frame_labels"]

            for idx, frame_path in enumerate(frame_paths):
                data.append({
                    "video_name": vid,
                    "frame_path": frame_path,
                    "c_stage_label": c_labels[idx],
                    "f_stage_label": f_labels[idx],
                    "frame_ind": idx
                })

        return data
    
    def get_class_weights(self):
        full_event_seq = np.array([item['f_stage_label'] for item in self.data])
        class_counts = np.zeros((self.num_classes,))
        for c in range(self.num_classes):
            class_counts[c] = (full_event_seq == c).sum()
                    
        class_weights = class_counts.sum() / ((class_counts + 10) * self.num_classes)

        return class_weights

    def __len__(self) -> int:
        return len(self.data)

    def _load_image(self, path: str) -> Optional[Image.Image]:
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            logging.error(f"Failed to load {path}: {str(e)}")
            return None

    def __getitem__(self, index: int) -> Dict[str, Any]:
        anno = self.data[index]
        frame_path = anno["frame_path"]

        images = [self._load_image(frame_path)]
        if self.transform:
            anno = self.transform({"frame": images, **anno})

        return anno


class ImageTransformSFHE:
    def __init__(self, mode: str = "train"):
        self.mode = mode
        self._init_pipelines()

    def _init_pipelines(self):
        train_transforms = [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(-30, 30)),
            transforms.ColorJitter(
                brightness=0.2, 
                contrast=0.2,
                saturation=0.1,  
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  
                std=[0.229, 0.224, 0.225]
            )
        ]

        test_transforms = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  
                std=[0.229, 0.224, 0.225]
            )
        ]

        self.image_pipeline = transforms.Compose(
            train_transforms if self.mode == "train" else test_transforms
        )

        self.label_pipeline = transforms.Compose([
            self._convert_labels(["c_stage_label", "f_stage_label"])
        ])

    def _convert_labels(self, keys: List[str]) -> transforms.Lambda:
        return transforms.Lambda(
            lambda data: {
                **data,
                **{k: torch.tensor(data[k], dtype=torch.long) for k in keys}
            }
        )

    def __call__(self, data_dict: Dict) -> Dict:
        if not isinstance(data_dict.get("frame"), (list, tuple)):
            raise ValueError("Frame data must be a list/tuple of images")

        processed_frames = [self.image_pipeline(img) for img in data_dict["frame"]]
        data_dict["frame"] = torch.stack(processed_frames) if len(processed_frames) > 1 else processed_frames[0]

        return self.label_pipeline(data_dict)


def collate_fn(batch: Sequence):
    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    if isinstance(batch[0], str):
        return batch

    if isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: collate_fn([d[key] for d in batch])
            for key in batch[0]
        }
    else:
        return default_collate(batch)
    


