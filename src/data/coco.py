# -*- coding: utf-8 -*-
"""
COCO annotation parsing, category filtering, and the train val test partition

The parser reads the standard COCO instance json and maps each enabled COCO category to a
contiguous channel index following the order of the enabled list, in the mixed mode every image
is kept so background images without an enabled object train and evaluate the detector against
false positives, otherwise only images carrying an enabled object are kept

The labelled train split and the selection val split are carved from train2017 by a fixed seed
so they are disjoint and stable, the held out test is the untouched val2017 split, the unlabelled
split feeds the student through the frozen teacher and carries no annotations
"""

import glob
import json
import os
import random

from data.paths import (
    DATA_TRAIN,
    DATA_TRAIN_ANNOTATIONS,
    DATA_UNLABELED,
    DATA_VAL,
    DATA_VAL_ANNOTATIONS,
)

_SAMPLE_SEED = 42  # Fixed seed so the sampled subset is reproducible
_TRAIN_VAL_CACHE = {}  # Memoise the parsed and partitioned train2017 split per parameter set


def _channel_map(enabled_categories):
    """Map each enabled COCO category id to a contiguous channel index"""
    return {int(cat): idx for idx, cat in enumerate(enabled_categories)}


def _parse(img_dir, ann_path, enabled_categories, mixed):
    """
    Parse a COCO split into a mapping from image path to its kept boxes

    Each kept image carries its clean boxes and, separately, its crowd boxes, the crowd regions
    become ignore areas rather than targets, when mixed is true background images without any
    enabled object are kept with empty lists so the natural distribution is preserved, otherwise
    background images are dropped

    Args:
        img_dir: Directory holding the split images
        ann_path: Path to the COCO instances json for the split
        enabled_categories: COCO category ids to keep, in channel order
        mixed: Whether to keep background images with empty box lists

    Returns:
        A dict mapping an absolute image path to a record with its clean boxes and its crowd ignore
        boxes, each box as x, y, w, h, channel
    """
    if not os.path.isfile(ann_path):
        raise FileNotFoundError(f"COCO annotations not found at {ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cat_to_channel = _channel_map(enabled_categories)
    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

    items = {}
    if mixed:  # Seed every image so background images survive with empty box lists
        items = {os.path.join(img_dir, fn): {"boxes": [], "crowd": []} for fn in id_to_file.values()}

    for ann in data["annotations"]:
        channel = cat_to_channel.get(ann["category_id"])
        if channel is None:  # Category not enabled for this task
            continue

        file_name = id_to_file.get(ann["image_id"])
        if not file_name:
            continue
        img_path = os.path.join(img_dir, file_name)
        entry = items.setdefault(img_path, {"boxes": [], "crowd": []})
        box = list(ann["bbox"]) + [channel]
        if ann.get("iscrowd", 0):  # Crowd regions become ignore areas, never rendered as centres
            entry["crowd"].append(box)
        else:
            entry["boxes"].append(box)

    return items


def _take(items, count, seed):
    """Deterministically keep a fixed number of images from a parsed split"""
    if count is None or len(items) <= count:
        return items
    rng = random.Random(seed)
    keys = rng.sample(sorted(items), count)
    return {k: items[k] for k in keys}


def load_train_val(enabled_categories, mixed=True, val_samples=5000, max_train=None, seed=_SAMPLE_SEED):
    """
    Parse train2017 and carve disjoint train and selection val splits

    The val holdout is drawn first by the seed so it stays fixed when the train cap changes, the
    remaining images form the train pool and are then capped to max_train, the result is memoised
    so the train stream and the val loader never re-parse train2017

    Args:
        enabled_categories: COCO category ids to keep, in channel order
        mixed: Whether to keep background images with an empty box list
        val_samples: Number of images held out from train2017 for selection
        max_train: Optional cap on the train pool after the val holdout
        seed: Seed for the deterministic partition and the train cap

    Returns:
        A tuple of train and val dicts mapping image path to its clean and crowd boxes record
    """
    key = (tuple(enabled_categories), mixed, val_samples, max_train, seed)
    if key in _TRAIN_VAL_CACHE:
        return _TRAIN_VAL_CACHE[key]

    items = _parse(DATA_TRAIN, DATA_TRAIN_ANNOTATIONS, enabled_categories, mixed)
    keys = sorted(items)
    rng = random.Random(seed)
    rng.shuffle(keys)

    holdout = val_samples if val_samples is not None else 0
    val_keys = keys[:holdout]  # The selection holdout is carved before any train cap
    train_keys = keys[holdout:]

    val = {k: items[k] for k in val_keys}
    train = {k: items[k] for k in train_keys}
    train = _take(train, max_train, seed)

    _TRAIN_VAL_CACHE[key] = (train, val)
    return train, val


def load_test(enabled_categories, mixed=True, max_test=None):
    """
    Load the held out test split from the untouched val2017 images

    Args:
        enabled_categories: COCO category ids to keep, in channel order
        mixed: Whether to keep background images with an empty box list
        max_test: Optional cap on the number of test images

    Returns:
        A dict mapping image path to its clean and crowd boxes record for the test split
    """
    items = _parse(DATA_VAL, DATA_VAL_ANNOTATIONS, enabled_categories, mixed)
    return _take(items, max_test, _SAMPLE_SEED)


def list_unlabeled(max_samples=None):
    """
    List the unlabelled image paths for semi supervised distillation

    Args:
        max_samples: Optional deterministic cap on the number of returned paths

    Returns:
        A sorted list of image paths from the unlabelled split
    """
    paths = sorted(glob.glob(os.path.join(DATA_UNLABELED, "*.jpg")))
    if max_samples is not None and len(paths) > max_samples:
        rng = random.Random(_SAMPLE_SEED)
        paths = sorted(rng.sample(paths, max_samples))
    return paths
