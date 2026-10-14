# MFHE annotations

The refined annotation files used in the paper are **not** distributed with this
repository: the raw releases contain noisy stage boundaries, and the files below are our
corrected version. If you need them, please send us an email.

Two kinds of files belong here.

1. `train_set.csv` / `test_set.csv` - the split definition, a single `filename` column listing
   the per-video annotation files:

   ```csv
   filename
   PMDPI029-1-10_annotations.csv
   ...
   ```

2. `<video_name>_annotations.csv` - one row per frame with the columns

   ```csv
   frame_name,fine_stage_label,coarse_stage_label
   frame_0000.jpg,0,0
   ...
   ```

   `frame_name` is relative to the focal-plane video root configured for Stage 1.


