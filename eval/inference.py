import os
import sys

# Allow running this file directly, e.g. "python eval/inference.py".
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
import torch
import argparse
import numpy as np
from data.SFHE.video_dataset import EmbryoVideoSFHE, VideoTransformSFHE
from data.SFHE.video_dataset import  collate_fn as collate_fn_SFHE
from data.MFHE.video_dataset import EmbryoVideoMFHE, VideoTransformMFHE
from data.MFHE.video_dataset import  collate_fn as collate_fn_MFHE
from models.model import EmbryoDiff
from tqdm import tqdm
from utils import func_eval_modified, resolve_path
from torch.utils.data import DataLoader


class DiffusionTester:
    def __init__(self, model, configs, test_loader, event_list, device, 
                 num_runs=10, encoder_only=False, result_base_dir="results"):
        self.model = model
        self.config = configs
        self.test_loader = test_loader
        self.event_list = event_list
        self.device = device
        self.num_runs = num_runs 
        self.encoder_only = encoder_only
        self.result_base_dir = result_base_dir

    def run_testing(self):
        base_dir = self.result_base_dir
        os.makedirs(base_dir, exist_ok=True)

        all_results = []
        for run_idx in range(self.num_runs):
            run_dir = os.path.join(base_dir, f"run_{run_idx:02d}")
            os.makedirs(run_dir, exist_ok=True)

            results = self._test_single_run(run_idx, run_dir)
            all_results.append(results)

        self._aggregate_results(base_dir, all_results)

    def _test_single_run(self, run_idx, output_dir):
        pbar = tqdm(self.test_loader, total=len(self.test_loader))
        video_idx = 0
        seed = run_idx * 1000 + video_idx
        for sample in pbar:
            video_name, pred, label = self.test_single_video(sample, seed)
            video_idx += 1
            
            self.save_video_results(video_name[0], pred, label, output_dir)

        pred_dir = os.path.join(output_dir, 'predictions')
        acc, edit, f1s = func_eval_modified(
            pred_dir=pred_dir, 
            video_list=self.test_loader.dataset.video_list,
        )

        return {
            'run': run_idx,
            'Acc': acc,
            'Edit': edit,
            'F1@10': f1s[0],
            'F1@25': f1s[1],
            'F1@50': f1s[2]
        }

    def test_single_video(self, sample, seed):
        video_name, video_feats, label = sample["video_name"], sample["video_feats"], sample["f_stage_label"]
        video_feats, label = video_feats.to(self.device), label.to(self.device)

        with torch.no_grad():
            if self.encoder_only:
                output, _ = self.model.forward_condition_model(video_feats)
                output = torch.softmax(output, dim=1).cpu()
            else:
                output = self.model.ddim_sample(video_feats, seed=seed, reset_rng_after=False)

        output = np.argmax(output.squeeze(0).cpu().numpy(), axis=1)
        label = label.squeeze(0).cpu().numpy()

        return video_name, output, label

    def save_video_results(self, video_name, pred, label, output_dir):
        output_subdir = os.path.join(output_dir, 'predictions')
        os.makedirs(output_subdir, exist_ok=True)

        file_path = os.path.join(output_subdir, f'{video_name}.csv')
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['frame_inds', 'prediction', 'label'])
            for frame_idx, (p, l) in enumerate(zip(pred, label)):
                writer.writerow([frame_idx, self.event_list[p][1], self.event_list[l][1]])

    def _aggregate_results(self, base_dir, all_results):
        all_results_file = os.path.join(base_dir, 'all_results.txt')
        
        with open(all_results_file, 'w') as f:
            for result in all_results:
                run = result['run']
                f.write(f'[Run {run} Results]\n')
                for k, v in result.items():
                    if k != 'run':
                        f.write(f'{k}: {v:.4f}\n')
                f.write('\n') 

            metrics = ['Acc', 'Edit', 'F1@10', 'F1@25', 'F1@50']
            agg_results = {
                metric: np.mean([r[metric] for r in all_results])
                for metric in metrics
            }

            total_avg = np.mean([agg_results[metric] for metric in metrics])

            f.write('[Aggregated Results (Mean)]\n')
            for metric in metrics:
                f.write(f'{metric}: {agg_results[metric]:.4f}\n')
            f.write(f'Total Average Metric: {total_avg:.4f}\n') 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Tester")
    parser.add_argument("--dataset", default="SFHE", help="dataset for inference")
    parser.add_argument("--config", default=None, required=True, help="Path to config file")
    parser.add_argument("--model_path", default=None, required=True, help="Path to model ckpt")
    parser.add_argument("--events", default=None, required=True, help="Path to developmental stage mapping file")
    parser.add_argument("--result_dir", default=None, required=True, help="Base result directory")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of times to run the test")
    parser.add_argument("--encoder_only", action='store_true', help="Use encoder-only mode")
    parser.add_argument("--device", default=0, help='Used GPU index')
    parser.add_argument("--sampling_timesteps", type=int, default=1, help='Used denoising steps during inference')
    parser.add_argument("--num_workers", type=int, default=8, help='DataLoader workers')
    
    args = parser.parse_args()

    if args.device != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.events = resolve_path(args.events)
    args.result_dir = resolve_path(args.result_dir)

    event_list = np.loadtxt(args.events, dtype=str)
    configs = json.load(open(args.config, mode="r", encoding="utf-8"))
    for key in ("train_split", "test_split", "video_feat_folder", "development_stage_list", "anno_root"):
        if configs["data"].get(key):
            configs["data"][key] = resolve_path(configs["data"][key])

    if args.dataset == "SFHE":
        test_set = EmbryoVideoSFHE(
            anno_path=configs["data"]["test_split"],
            video_feat_folder=configs["data"]["video_feat_folder"],
            development_stage_list=configs["data"]["development_stage_list"],
            num_classes=configs["data"]["num_classes"],
            fast_load=False,
            transform=VideoTransformSFHE()
        )
        collate_fn = collate_fn_SFHE
    elif args.dataset == "MFHE":
        test_set = EmbryoVideoMFHE(
            split_file=configs["data"]["test_split"], 
            anno_root=configs["data"].get("anno_root"),
            video_feat_folder=configs["data"]["video_feat_folder"],
            transform=VideoTransformMFHE(),
            subsample_rate=configs["train"]["subsample_rate"],
            num_classes=configs["data"]["num_classes"],
            fast_load=False
        )
        collate_fn = collate_fn_MFHE

    test_loader = DataLoader(
        test_set, 
        batch_size=1, 
        shuffle=False, 
        num_workers=args.num_workers, 
        collate_fn=collate_fn
    )

    configs["diffusion_params"]["sampling_timesteps"] = args.sampling_timesteps
    model = EmbryoDiff(
        encoder_params=configs["encoder_params"],
        decoder_params=configs["decoder_params"],
        diffusion_params=configs["diffusion_params"],
        num_classes=len(event_list),
        device=device
    )
    
    model.load_state_dict(torch.load(args.model_path)["model"])
    model.to(device)
    model.eval()


    res_dir = os.path.join(args.result_dir, str(args.sampling_timesteps) + "_step")
    tester = DiffusionTester(
        configs=configs,
        model=model,
        test_loader=test_loader,
        event_list=event_list,
        device=device,
        num_runs=args.num_runs, 
        encoder_only=args.encoder_only,
        result_base_dir=res_dir
    )

    tester.run_testing()

