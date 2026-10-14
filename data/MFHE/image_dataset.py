import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import logging
from collections.abc import Mapping, Sequence
from torch.utils.data.dataloader import default_collate
from typing import Dict, List, Any, Optional


class EmbryoImageMFHE(Dataset):
    def __init__(self, split_file: str, anno_root: str, video_root: str, 
                 focal_planes: List[str] = ["F0"], num_classes: int = 15,
                 subsample_rate: int = 1, transform: Optional[callable] = None, debug=False):
        super().__init__()
        self.split_file = split_file
        self.anno_root = anno_root
        self.video_root = video_root
        self.num_classes = num_classes
        self.transform = transform
        self.subsample_rate = subsample_rate
        self.focal_planes = focal_planes

        self.sample_list = pd.read_csv(split_file)["filename"].tolist()
        self.video_list = [self._parse_video_name(fn) for fn in self.sample_list]
        self.n2v_mapping = {i: name for i, name in enumerate(self.video_list)}
        self.v2n_mapping = {name: i for i, name in enumerate(self.video_list)}

        self.data = self._preload_data_paths()

        if debug:
            self.data = self.data[:1024]

    def _parse_video_name(self, filename: str) -> str:
        return filename.split("_annotations")[0]

    def _preload_data_paths(self) -> List[Dict]:
        data = []
        for anno_file in self.sample_list:
            abs_path = os.path.join(self.anno_root, anno_file)
            try:
                df = pd.read_csv(abs_path)
                if df.empty:
                    continue
            except Exception as e:
                logging.warning(f"Skipping {abs_path}: {str(e)}")
                continue

            if self.subsample_rate > 1:
                df = df.iloc[::self.subsample_rate].reset_index(drop=True)

            video_name = self._parse_video_name(anno_file)
            frame_names = df["frame_name"].values
            c_labels = df["coarse_stage_label"].values
            f_labels = df["fine_stage_label"].values

            for idx, frame_name in enumerate(frame_names):
                data.append({
                    "video_name": video_name,
                    "frame_names": [frame_name] * len(self.focal_planes),
                    "c_stage_label": c_labels[idx],
                    "f_stage_label": f_labels[idx],
                    "frame_ind": idx + 1 
                })

        return data

    def _build_frame_path(self, video_name: str, frame_name: str, focal: str) -> str:
        base_dir = f"{self.video_root}_{focal}" if focal != "F0" else self.video_root
        return os.path.join(base_dir, video_name, frame_name)
    
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
        video_name = anno["video_name"]
        frame_names = anno["frame_names"]
        del anno["frame_names"]
        
        frame_paths = [
            self._build_frame_path(video_name, fn, focal)
            for fn, focal in zip(frame_names, self.focal_planes)
        ]
        images = [self._load_image(p) for p in frame_paths]

        if self.transform:
            anno = self.transform({"frame": images, **anno})

        return anno


class ImageTransformMFHE:
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
                mean=[0.251, 0.251, 0.251],  
                std=[0.199, 0.199, 0.199]
            )
        ]

        test_transforms = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.251, 0.251, 0.251],  
                std=[0.199, 0.199, 0.199]
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
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """

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