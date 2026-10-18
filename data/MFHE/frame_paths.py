"""Path helpers for MFHE frame images."""
import os


# MFHE annotation files name the JPEG frames of the re-saved release. Users who
# downloaded the original `.jpeg` files can point the same annotations at them.
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def resolve_frame_path(path: str) -> str:
    """Return ``path`` if it exists, else the same frame with another image suffix."""
    if os.path.isfile(path):
        return path

    stem, extension = os.path.splitext(path)
    for candidate in _IMAGE_EXTENSIONS:
        if candidate != extension and os.path.isfile(stem + candidate):
            return stem + candidate
    return path
