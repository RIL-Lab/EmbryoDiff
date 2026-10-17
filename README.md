# EmbryoDiff

**EmbryoDiff: A Conditional Diffusion Framework with Multi-Focal Feature Fusion for Fine-Grained Embryo Developmental Stage Recognition**

Official implementation of the paper accepted at **AAAI 2026**.
Online extended version: [arXiv:2511.11027](https://arxiv.org/abs/2511.11027)

---

## Overview

EmbryoDiff recognizes the fine-grained developmental stage of time-lapse human embryo
videos. The pipeline has two stages:

1. **Stage 1** fine-tunes an ImageNet-pretrained ResNet-50 to classify individual embryo
   frames and exports one feature vector per frame.
2. **Stage 2** trains a conditional diffusion model on those frame features. A *semantic*
   and a *boundary* temporal encoder produce the conditioning signal, and a DDIM decoder
   denoises the whole stage-label sequence for a video in one shot.

The same code base covers both datasets used in the paper:

| Name | Description | Classes |
| --- | --- | --- |
| SFHE | single-focal human embryo videos | 12 |
| MFHE | multi-focal human embryo videos | 15 |

## Repository layout

```text
EmbryoDiff/
├── configs/                  # Stage-1 and Stage-2 configs for both datasets
├── data/
│   ├── dataset_builder.py    # dataset factory used by both stages
│   ├── SFHE/                 # single-focal dataset code, stage list, annotations
│   └── MFHE/                 # multi-focal dataset code, stage list, annotations
├── eval/inference.py         # evaluation of a trained Stage-2 checkpoint
├── models/                   # ResNet backbone, temporal encoders, diffusion decoder
├── scripts/                  # training entry points (Python + shell wrappers)
├── trainer/                  # Stage-1 / Stage-2 training and testing loops
├── utils.py                  # path resolution, seeding, logging, metrics
└── requirements.txt
```

## Environment

```bash
conda create -n EmbryoDiff python=3.9 -y
conda activate EmbryoDiff

# PyTorch with the CUDA build that matches your driver, then the rest
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Training is run on a **single GPU** (or CPU for debugging); the code does not implement
multi-GPU training.

## Datasets

We do **not** redistribute the datasets. Both are public datasets released by their
authors; download them from their original releases, accept their terms, and place them
outside this repository (for example `/data/embryo/`). Please cite the corresponding paper:

| Dataset | Source paper |
| --- | --- |
| SFHE (single-focal) | *EmbryosFormer: Deformable Transformer and Collaborative Encoding-Decoding for Embryos Stage Development Classification* |
| MFHE (multi-focal) | *Towards Deep Learning-Powered IVF: A Large Public Benchmark for Morphokinetic Parameter Prediction* |

> **Refined annotations.**
> For the paper we re-checked the frame-level stage labels and fixed the noisy stage
> boundaries of the raw releases (this matters most for MFHE). These refined annotation
> files are **not** published with this repository; without them you can still run the
> code, but you will get the raw, noisier supervision. The expected file names and
> formats are documented in `data/SFHE/annotations/README.md` and
> `data/MFHE/annotations/README.md`. If you need the refined annotations,
> please send us an email.

Expected layout once the data is in place:

```text
data/SFHE/
├── annotations/                  # train_annos.json / test_annos.json (refined, on request)
├── development_stage_list.txt
├── videos/                       # raw frames, like 16052022_IM19403778_A2/xxxx.jpg
└── extracted_video_features/     # <video_name>.npy, produced by the feature extractor

data/MFHE/
├── annotations/                  # train_set.csv / test_set.csv (refined, on request)
├── development_stage_list.txt
├── videos/                       # F0 frames:  <video_name>/<frame_name>
├── videos_<focal>/               # other focal planes, same layout
└── extracted_video_features/     # <video_name>.npy, shape (num_focal, T, 2048)
```

SFHE features have shape `(T, 2048)`; MFHE features stack all focal planes as
`(num_focal, T, 2048)`.

## Training

### 0. Paths in the configs

Every config in `configs/` is written with two kinds of placeholders:

* `path_to_project/` is replaced automatically by the repository root, so you normally do
  not have to touch paths that live inside the repo.
* `path_to_*_videos` / `path_to_MFHE_annotations` are dataset locations that you must
  replace with the real paths on your machine.

Also check `device` (`[0]` = first GPU, `-1` is not supported for training) and
`base_save_dir` / `train.result_dir` in the configs.

### 1. Stage-1 training

```bash
# SFHE
bash scripts/train_stage1_SFHE.sh 0 stage1_sfhe

# MFHE
bash scripts/train_stage1_MFHE.sh 0 stage1_mfhe
```

Both wrappers call:

```bash
python -m scripts.train_stage1 --config configs/train_stage1_SFHE.json --device 0 --exp_name stage1_sfhe
```

Logs, the resolved config and the best checkpoint are written to
`<base_save_dir>/<exp_name>/{logging,config.json,trained_model/best_model.pth}`.

### 2. Extract the Stage-1 features

```bash
python data/SFHE/extract_video_feats.py \
    --split_files "data/SFHE/annotations/train_annos.json,data/SFHE/annotations/test_annos.json" \
    --video_root /path/to/SFHE/videos \
    --resnet_path result/SFHE/stage1/stage1_sfhe/trained_model/best_model.pth \
    --config_path result/SFHE/stage1/stage1_sfhe/config.json \
    --output_dir data/SFHE/extracted_video_features \
    --gpu_id 0

python data/MFHE/extract_video_feats.py \
    --split_file data/MFHE/annotations/train_set.csv \
    --anno_root /path/to/MFHE/annotations \
    --video_root /path/to/MFHE/videos \
    --focal_planes "F0" \
    --resnet_path result/MFHE/stage1/stage1_mfhe/trained_model/best_model.pth \
    --config_path result/MFHE/stage1/stage1_mfhe/config.json \
    --output_dir data/MFHE/extracted_video_features \
    --gpu_id 0
```

### 3. Stage-2 training

```bash
bash scripts/train_stage2_SFHE.sh 0 stage2_sfhe
bash scripts/train_stage2_MFHE.sh 0 stage2_mfhe
```

which runs

```bash
python -m scripts.train_stage2 --config configs/train_stage2_SFHE.json --device 0 --exp_name stage2_sfhe
```

Each epoch is followed by a test pass; the best decoder checkpoint is stored as
`best_avg_metric_model.pt` (selected by the average of Acc/Edit/F1) and `best_acc_model.pt`.

### 4. Inference / evaluation

```bash
python eval/inference.py \
    --dataset SFHE \
    --config result/SFHE/stage2/stage2_sfhe/train_stage2_SFHE.json \
    --model_path result/SFHE/stage2/stage2_sfhe/decoder/best_avg_metric_model.pt \
    --events data/SFHE/development_stage_list.txt \
    --result_dir result/SFHE/stage2_inference \
    --num_runs 10 \
    --sampling_timesteps 1 \
    --num_workers 8 \
    --device 0
```

The script writes the per-video predictions of every run plus an
`all_results.txt` holding the per-run and averaged Acc / Edit / F1@10 / F1@25 / F1@50.
Pass `--encoder_only` to evaluate the temporal encoders without the diffusion decoder.

## Notes and limitations

* **Stage 2 runs with `batch_size = 1`.** One video is processed per step; increase
  `gradient_accumulation_steps` for a larger effective batch size (the released configs
  use `24`). The same holds for `eval/inference.py`.
* Stage-1 frames are resized to 256 and cropped to 224 (`Resize(256)` + `RandomResizedCrop(224)`
  for training, `CenterCrop(224)` for testing).
* The released configs train Stage 2 for 350 epochs and sample with `sampling_timesteps = 15`
  during training; `eval/inference.py` lets you override the number of DDIM steps.

## Citation

```bibtex
@inproceedings{embryodiff2026,
  title     = {EmbryoDiff: A Conditional Diffusion Framework with Multi-Focal Feature
               Fusion for Fine-Grained Embryo Developmental Stage Recognition},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```

An extended version is available at [arXiv:2511.11027](https://arxiv.org/abs/2511.11027).

## Acknowledgements

The cosine noise schedule follows [Improved Denoising Diffusion Probabilistic
Models](https://openreview.net/forum?id=-NEXDKk8gZ).
The SFHE and MFHE datasets were released by the authors of *EmbryosFormer* and
*Towards Deep Learning-Powered IVF* respectively; please cite those papers when you use
them.

## License

Released under the MIT License, see [LICENSE](LICENSE).






