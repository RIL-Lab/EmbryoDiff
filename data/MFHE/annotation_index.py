"""Helpers for locating the MFHE per-video annotation files."""
import os
from typing import Optional


PER_VIDEO_DIRNAME = "per_video"


def resolve_anno_root(split_file: str, anno_root: Optional[str] = None) -> str:
    """Return the directory holding ``<video>_annotations.csv`` files.

    An explicit ``anno_root`` wins. Otherwise the files are expected in a
    ``per_video/`` sub-directory next to the split file, so that a
    self-contained annotation folder works without extra configuration:

        annotations/
        |-- train_set.csv
        |-- test_set.csv
        `-- per_video/<video>_annotations.csv
    """
    if anno_root:
        return anno_root
    split_dir = os.path.dirname(os.path.abspath(split_file))
    return os.path.join(split_dir, PER_VIDEO_DIRNAME)


def missing_annotations_error(split_file: str, anno_root: str) -> FileNotFoundError:
    return FileNotFoundError(
        f"No per-video annotation files could be read from '{anno_root}' "
        f"(split file: '{split_file}'). Expected files such as "
        f"'<video>_annotations.csv'; see data/MFHE/annotations/README.md "
        f"for the expected layout. Set 'anno_root' in the config if the "
        f"annotations live somewhere else."
    )
