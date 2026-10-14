# SFHE annotations

The refined annotation files used in the paper are **not** distributed with this
repository. If you need them, please send us an email, then drop them here as:

- `train_annos.json`
- `test_annos.json`

Format (one entry per video):

```json
{
  "<video_name>": {
    "duration": 370,
    "frame_paths": ["<video_name>/frame_0000.jpg", "<video_name>/frame_0001.jpg"],
    "fine":   {"frame_labels": [0, 0, 1]},
    "coarse": {"frame_labels": [0, 0, 1]}
  }
}
```

`frame_paths` are relative to the `video_root` configured for Stage 1. `fine.frame_labels` holds the
stage index of every frame; the index to stage-name mapping is in
`../development_stage_list.txt`.


