import os
import torch
import logging
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter1d
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional
from itertools import chain

from data.MFHE.annotation_index import missing_annotations_error, resolve_anno_root


class EmbryoVideoMFHE(Dataset):
    def __init__(self, split_file: str, anno_root: Optional[str], video_feat_folder: str,
                 focal_planes: List[str] = ["F0"], num_classes: int = 15,
                 subsample_rate: int = 1, fast_load: bool = False, transform: Optional[callable] = None, boundary_smooth=None):
        super().__init__()
        self.split_file = split_file
        self.anno_root = resolve_anno_root(split_file, anno_root)
        self.video_feat_folder = video_feat_folder
        self.num_classes = num_classes
        self.transform = transform
        self.subsample_rate = subsample_rate
        self.focal_planes = focal_planes
        self.fast_load = fast_load
        self.boundary_smooth = boundary_smooth
    
        self.sample_list = pd.read_csv(split_file)["filename"].tolist()
        self.data_list = self._load_data()

        if not self.data_list:
            raise missing_annotations_error(split_file, self.anno_root)

        self.video_list = self.get_video_list()

    def get_video_list(self):
        video_list = [anno_file.split("_annotations")[0] for anno_file in self.sample_list]

        return video_list

    def _parse_video_name(self, filename: str) -> str:
        return filename.split("_annotations")[0]
    
    def _load_data(self):
        anno_root = self.anno_root
        if not os.path.isdir(anno_root):
            raise missing_annotations_error(self.split_file, anno_root)

        data = []
        if self.fast_load:
            self.sample_list = self.sample_list[:10]
        for anno_file in self.sample_list:
            abs_path = os.path.join(anno_root, anno_file)
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
            frame_names = df["frame_name"].values.tolist()
            f_labels = df["fine_stage_label"].values.tolist()
            c_labels = df["coarse_stage_label"].values.tolist()

            video_feats_path = os.path.join(self.video_feat_folder, video_name + ".npy")
            start_end_labels, presence_labels, _, _ = self.get_labels_start_end_time(frame_wise_labels=f_labels)
            boundary_labels = self.get_boundary_seq(event_seq=f_labels, boundary_smooth=self.boundary_smooth)
            stage_num_labels = len(set(f_labels))

            data.append(
                {
                    "video_name": video_name,
                    "video_feats_path": video_feats_path,
                    "frame_names": frame_names,
                    "f_stage_label": f_labels,
                    "c_stage_label": c_labels,
                    "range_label": start_end_labels,
                    "presence_label": presence_labels,
                    "boundary_label": boundary_labels,
                    "stage_num_label": stage_num_labels
                }
            )

        return data
        

    def get_boundary_seq(self, event_seq, boundary_smooth=None):
        boundary_seq = np.zeros_like(event_seq, dtype=np.float64)
        _, _, start_times, _ = self.get_labels_start_end_time([str(int(i)) for i in event_seq])
        boundaries = start_times[1:]
        if len(boundaries) > 0 and min(boundaries) > 0:
            boundary_seq[boundaries] = 1.0
            boundary_seq[[i-1 for i in boundaries]] = 1.0

            if boundary_smooth is not None:
                boundary_seq = gaussian_filter1d(boundary_seq, boundary_smooth)
                
                # Normalize. This is ugly.
                temp_seq = np.zeros_like(boundary_seq)
                temp_seq[temp_seq.shape[0] // 2] = 1
                temp_seq[temp_seq.shape[0] // 2 - 1] = 1
                norm_z = gaussian_filter1d(temp_seq, boundary_smooth).max()
                boundary_seq[boundary_seq > norm_z] = norm_z
                boundary_seq /= boundary_seq.max()

        return boundary_seq

    def get_class_weights(self):
        full_event_seq = np.array(chain(item['f_stage_label'] for item in self.data_list))
        class_counts = np.zeros((self.num_classes,))
        for c in range(self.num_classes):
            class_counts[c] = (full_event_seq == c).sum()
                    
        class_weights = class_counts.sum() / ((class_counts + 10) * self.num_classes)

        return class_weights


    def get_labels_start_end_time(self, frame_wise_labels):
        num_classes = self.num_classes
        
        start_end_array = np.zeros((num_classes, 2), dtype=np.int32)
        presence_array = np.zeros((num_classes, 1), dtype=np.int32)

        labels, starts, ends = [], [], []
        last_label = frame_wise_labels[0]
        labels.append(frame_wise_labels[0])
        starts.append(0)
        for i in range(len(frame_wise_labels)):
            if frame_wise_labels[i] != last_label:
                labels.append(frame_wise_labels[i])
                starts.append(i)
                ends.append(i)
                last_label = frame_wise_labels[i]
        ends.append(len(frame_wise_labels)-1) 
        
        for event_id in range(num_classes):
            if event_id in labels:
                idx = labels.index(event_id)
                start_end_array[event_id] = [starts[idx], ends[idx]]
                presence_array[event_id] = 1
            else:
                pass

        return start_end_array, presence_array, starts, ends
    

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):

        sample = self.data_list[index]
        video_feats_path = sample["video_feats_path"]

        try:
            video_feats = np.load(video_feats_path)
            assert video_feats.dtype == np.float32, f"{video_feats.dtype}"
            if self.subsample_rate > 1:
                video_feats = video_feats[:, ::self.subsample_rate, :]
            
            sample["video_feats"] = video_feats
            
        except Exception as e:
            raise RuntimeError("load failed.")

        
        if self.transform:
            sample = self.transform(sample)

        return sample
        

class VideoTransformMFHE:
    def __call__(self, sample: dict) -> dict:
        processed = {}
        
        for key, value in sample.items():
            if key == "range_label":
                processed[key] = torch.from_numpy(value).to(torch.long)
            
            elif key == "presence_label":
                processed[key] = torch.from_numpy(value).to(torch.long)

            elif key == "boundary_label":
                processed[key] = torch.from_numpy(value).to(torch.long)

            elif key == "video_feats":
                processed[key] = torch.from_numpy(value)
                    
            elif key == "f_stage_label":
                processed[key] = torch.tensor(value, dtype=torch.long)
            
            elif key == "c_stage_label":
                processed[key] = torch.tensor(value, dtype=torch.long)

            elif key == "stage_num_label":
                processed[key] = torch.tensor(value, dtype=torch.long)

            else:
                processed[key] = value
 
        return processed


def collate_fn(batch):
    batch_dict = {}

    video_names = []
    frame_names_list = []
    video_feats_list = []
    f_stage_labels_list = []
    c_stage_labels_list = []
    range_labels_list = []
    presence_labels_list = []
    boundary_labels_list = []
    stage_num_labels_list = []

    for sample in batch:
        video_names.append(sample["video_name"])
        frame_names_list.append(sample["frame_names"])
        video_feats_list.append(sample["video_feats"])
        f_stage_labels_list.append(sample["f_stage_label"])
        c_stage_labels_list.append(sample["c_stage_label"])
        range_labels_list.append(sample["range_label"])
        presence_labels_list.append(sample["presence_label"])
        boundary_labels_list.append(sample["boundary_label"])
        stage_num_labels_list.append(sample["stage_num_label"])

    video_feats_transposed = [vf.transpose(0, 1) for vf in video_feats_list]
    padded_video_feats = pad_sequence(
        video_feats_transposed,
        batch_first=True,
        padding_value=-100
    )
    batch_dict["video_feats"] = padded_video_feats.transpose(1, 2)
    
    padded_f_stage_labels = pad_sequence(
        f_stage_labels_list,
        batch_first=True,
        padding_value=-100 
    )
    batch_dict["f_stage_label"] = padded_f_stage_labels

    padded_c_stage_labels = pad_sequence(
    c_stage_labels_list,
    batch_first=True,
    padding_value=-100
    )
    batch_dict["c_stage_label"] = padded_c_stage_labels

    padded_boundary_labels = pad_sequence(
        boundary_labels_list,
        batch_first=True,
        padding_value=-100
    )
    batch_dict["boundary_label"] = padded_boundary_labels

    batch_dict["range_label"] = torch.stack(range_labels_list, dim=0)
    batch_dict["presence_label"] = torch.stack(presence_labels_list, dim=0)
    batch_dict["stage_num_label"] = torch.stack(stage_num_labels_list, dim=0)

    attention_mask = (padded_f_stage_labels != -100).float()
    batch_dict["attention_mask"] = attention_mask

    batch_dict["video_name"] = video_names
    batch_dict["frame_name"] = frame_names_list
    batch_dict["seq_length"] = torch.tensor(
        [vf.shape[1] for vf in video_feats_list],
        dtype=torch.long
    )

    return batch_dict
