# MFHE annotations

Frame-level stage annotations for the 655 MFHE videos used in the paper. The original
release contains noisy stage boundaries; these files are our corrected version and are
distributed together with the train/test split lists. The video frames themselves are
**not** included: download them from the original MFHE release
(*Towards Deep Learning-Powered IVF: A Large Public Benchmark for Morphokinetic
Parameter Prediction*) and cite that paper when you use the data.

| split | videos | annotated frames |
| --- | --- | --- |
| `train_set.csv` | 458 | 187,860 |
| `test_set.csv` | 197 | 80,152 |
| total | 655 | 268,012 |

## Layout

```text
data/MFHE/annotations/
├── train_set.csv            # split definition, a single `filename` column
├── test_set.csv
└── per_video/
    └── <video_name>_annotations.csv
```

The split files only list the per-video files:

```csv
filename
PMDPI029-1-10_annotations.csv
...
```

Each `per_video/<video_name>_annotations.csv` has one row per annotated frame:

```csv
frame_idx,frame_name,fine_stage_label,coarse_stage_label
6,D2013.01.28_S0717_I132_WELL7_RUN6.jpg,0,0
...
```

* `frame_name` is relative to the focal-plane video root configured for Stage 1. The
  names use the `.jpg` frames of the re-saved release; if your copy still has the
  original `.jpeg` files, the loaders fall back to that suffix automatically.
* `fine_stage_label` indexes `../development_stage_list.txt`.
* Rows are in temporal order; frames before the annotated start are not listed.

## Path resolution

Both stages look for the per-video files in `per_video/` next to the split file, so
`data.train_split` and `data.test_split` are the only annotation paths you have to set.
If the files live somewhere else, point the optional `anno_root` entry (Stage-2
`data.anno_root`, Stage-1 top-level `anno_root`) at that directory.
