import os
import torch
import json
import numpy as np
from typing import Optional
from itertools import chain
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class EmbryoVideoSFHE(Dataset):
    def __init__(self, anno_path, video_feat_folder: str, 
                 num_classes: int = 12,
                 development_stage_list=None,
                fast_load: bool = False, transform: Optional[callable] = None):
        super().__init__()
        self.anno_path = anno_path
        self.video_feat_folder = video_feat_folder
        self.development_stage_list = development_stage_list
        self.num_classes = num_classes
        self.transform = transform
        self.fast_load = fast_load


        self.anno_infos = json.load(open(self.anno_path, 'r'))
        self.video_list = self.anno_infos.keys()

        self.data_list = self._load_data()

    def get_video_list(self):
        return self.video_list
    
    def _load_data(self):
        
        data = []
        for vid in self.video_list:
            info = self.anno_infos[vid]

            frame_names = info["frame_paths"]
            f_labels = info["fine"]["frame_labels"]
            c_labels = info["coarse"]["frame_labels"]

            video_feats_path = os.path.join(self.video_feat_folder, vid + ".npy")

            start_end_labels, presence_labels, _, _ = self.get_labels_start_end_time(frame_wise_labels=f_labels)
            boundary_labels = self.get_boundary_seq(event_seq=f_labels)
            stage_num_labels = len(set(f_labels))

            data.append(
                {
                    "video_name": vid,
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
        

    def get_boundary_seq(self, event_seq):

        boundary_seq = np.zeros_like(event_seq, dtype=np.float64)

        _, _, start_times, _ = self.get_labels_start_end_time([str(int(i)) for i in event_seq])
        boundaries = start_times[1:]
        if len(boundaries) > 0 and min(boundaries) > 0:
            boundary_seq[boundaries] = 1.0
            boundary_seq[[i-1 for i in boundaries]] = 1.0

        return boundary_seq

    def get_class_weights(self):
        full_event_seq = np.array(chain(item['f_stage_label'] for item in self.data_list))
        class_counts = np.zeros((self.num_classes,))
        for c in range(self.num_classes):
            class_counts[c] = (full_event_seq == c).sum()
                    
        class_weights = class_counts.sum() / ((class_counts + 10) * self.num_classes)

        return class_weights


    def get_labels_start_end_time(self, frame_wise_labels):
        all_events_path = self.development_stage_list
        with open(all_events_path, 'r') as f:
            all_events = [line.strip().split() for line in f]
        
        num_classes = len(all_events)  
        
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
        
        all_event_ids_sorted = sorted([int(event[0]) for event in all_events]) 
        for event_id in all_event_ids_sorted:
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
            sample["video_feats"] = video_feats
            
        except Exception as e:
            raise RuntimeError("load failed.")

        
        if self.transform:
            sample = self.transform(sample)

        return sample
        

class VideoTransformSFHE:
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


    video_feats_transposed = [vf for vf in video_feats_list]
    padded_video_feats = pad_sequence(
        video_feats_transposed,
        batch_first=True,
        padding_value=-100
    )
    batch_dict["video_feats"] = padded_video_feats


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
        [vf.shape[0] for vf in video_feats_list],
        dtype=torch.long
    )

    return batch_dict
