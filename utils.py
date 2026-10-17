"""Shared helpers: path resolution, seeding, logging and evaluation metrics."""
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(path):
    """Resolve a path that comes from a config file.

    The shipped configs use "path_to_project" as a placeholder for the
    repository root and may also contain paths that are relative to it. Both
    are resolved here, so configs work without hand editing the project root.
    """
    if path is None:
        return None
    path = str(path).replace("path_to_project", str(PROJECT_ROOT))
    path = Path(path).expanduser()
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Evaluation metrics (frame accuracy, edit distance, F1@10/25/50)
# --------------------------------------------------------------------------- #
def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], np.float32)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,
                              D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0
    eps = 1e-7
    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / (union + eps)) * (
            [p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def _aggregate(overlap, tp, fp, fn, correct, total, edit_total, num_videos):
    acc = 100 * float(correct) / total if total > 0 else 0.0
    edit = (1.0 * edit_total) / num_videos if num_videos else 0.0

    f1s = np.zeros(len(overlap), dtype=float)
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s]) if (tp[s] + fp[s]) > 0 else 0.0
        recall = tp[s] / float(tp[s] + fn[s]) if (tp[s] + fn[s]) > 0 else 0.0
        f1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s[s] = np.nan_to_num(f1) * 100

    return acc, edit, f1s


def func_eval_modified(pred_dir, video_list):
    """Stage-2 metrics, computed from the per-video prediction CSVs."""
    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    edit = 0

    for vid in video_list:
        pred_file = os.path.join(pred_dir, f'{vid}.csv')
        pred_content = pd.read_csv(pred_file, header=0)["prediction"].to_list()
        gt_content = pd.read_csv(pred_file, header=0)["label"].to_list()

        assert (len(gt_content) == len(pred_content))

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == pred_content[i]:
                correct += 1

        edit += edit_score(pred_content, gt_content)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(pred_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

    return _aggregate(overlap, tp, fp, fn, correct, total, edit, len(video_list))


def func_eval_stage1(res_dict):
    """Stage-1 metrics, computed from frame-wise predictions kept in memory.

    Args:
        res_dict (dict): {video_name: [(frame_ind, pred, label), ...]}
    """
    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
    correct = 0
    total = 0
    edit_total = 0
    video_list = res_dict.keys()

    for vid in video_list:
        predictions = []
        ground_truths = []
        for _, pred, label in res_dict.get(vid, []):
            predictions.append(pred)
            ground_truths.append(label)

        if not predictions:
            continue

        total += len(ground_truths)
        correct += sum(p == g for p, g in zip(predictions, ground_truths))
        edit_total += edit_score(predictions, ground_truths)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(predictions, ground_truths, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

    return _aggregate(overlap, tp, fp, fn, correct, total, edit_total, len(video_list))


def setup_logger(log_file="train.log", level=logging.INFO, print_in_consule=False):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if print_in_consule:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
