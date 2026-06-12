# -*- coding: utf-8 -*-
"""
Training streams that unify the labelled and unlabelled regimes

A batch carries the images, the optional targets, and a source tag, targets is None marks an
unlabelled batch so the engine gates the supervised term on its presence

The labelled regime yields only labelled batches, the semi regime interleaves one labelled batch
with a configurable number of cycled unlabelled batches, so distillation applies on every batch
while supervision applies only on the labelled ones
"""

from typing import NamedTuple, Optional

import torch
from torch.utils.data import DataLoader

from data.coco import list_unlabeled, load_test, load_train_val
from data.collate import collate_boxes_skip_none, collate_skip_none
from data.datasets import LabelledDetection, UnlabelledImages
from data.transforms import build_train_transform, build_val_transform


class Targets(NamedTuple):
    """Supervised targets for one batch, the per class heatmap, geometry map, and crowd ignore mask"""

    heatmap: torch.Tensor
    geometry: torch.Tensor
    ignore_mask: torch.Tensor


class Batch(NamedTuple):
    """
    One training batch shared by the labelled and unlabelled regimes

    Attributes:
        images: Input image batch
        targets: Supervised targets, or None for an unlabelled batch
        source: Tag identifying the batch as labelled or unlabelled
    """

    images: torch.Tensor
    targets: Optional[Targets]
    source: str


class TrainStream:
    """
    Iterable training stream that interleaves labelled and unlabelled batches

    The labelled loader defines the labelled steps per epoch, the optional unlabelled loader is
    cycled and drawn at the given ratio after each labelled batch

    Args:
        labelled_loader: Loader over the labelled split yielding image, heatmap, geometry, ignore mask
        unlabelled_loader: Loader over the unlabelled split, or None for the labelled regime
        ratio: Number of unlabelled batches drawn per labelled batch
        regime: Regime name, labelled or semi
    """

    def __init__(self, labelled_loader, unlabelled_loader, ratio, regime):
        self.labelled_loader = labelled_loader
        self.unlabelled_loader = unlabelled_loader
        self.ratio = ratio
        self.regime = regime

    def __len__(self) -> int:
        """Return the number of batches yielded per epoch"""
        if self.unlabelled_loader is None:
            return len(self.labelled_loader)
        return len(self.labelled_loader) * (1 + self.ratio)

    def __iter__(self):
        """Yield interleaved Batch objects for one epoch"""
        if self.unlabelled_loader is None:
            for images, heatmap, geometry, ignore_mask in self.labelled_loader:
                yield Batch(images, Targets(heatmap, geometry, ignore_mask), "labelled")
            return

        unlabelled_iter = iter(self.unlabelled_loader)
        for images, heatmap, geometry, ignore_mask in self.labelled_loader:
            yield Batch(images, Targets(heatmap, geometry, ignore_mask), "labelled")
            for _ in range(self.ratio):
                try:
                    u_images = next(unlabelled_iter)
                except StopIteration:  # Cycle the unlabelled loader within the epoch
                    unlabelled_iter = iter(self.unlabelled_loader)
                    u_images = next(unlabelled_iter)
                yield Batch(u_images, None, "unlabelled")


def _split_items(config, split):
    """
    Return the parsed boxes for the requested labelled split

    The train and val splits are carved from train2017 by the partition seed, the test split is
    the untouched val2017, the val holdout stays fixed regardless of the train cap

    Args:
        config: Resolved configuration holding the selection and limit settings
        split: One of train, val, or test

    Returns:
        A dict mapping image path to its boxes for the requested split
    """
    enabled = config["enabled_categories"]
    mixed = config.get("mixed", True)
    seed = config.get("split_seed", 42)
    val_samples = config.get("max_val_samples", 5000)

    if split == "test":
        return load_test(enabled, mixed, config.get("max_test_samples"))

    train, val = load_train_val(enabled, mixed, val_samples, config.get("max_train_samples"), seed)
    return train if split == "train" else val


def _labelled_dataset(config, split):
    """Build the labelled dataset for the train val or test split"""
    img_size = config.get("img_size", 640)
    stride = config.get("stride", 4)
    num_classes = config["num_classes"]
    transform = build_train_transform(img_size) if split == "train" else build_val_transform(img_size)
    return LabelledDetection(
        _split_items(config, split), num_classes, img_size, stride, transform, return_boxes=split != "train"
    )


def build_train_stream(config) -> TrainStream:
    """
    Build the training stream for the configured regime

    Args:
        config: Resolved configuration holding the regime, dataset, and loader settings

    Returns:
        A TrainStream that yields labelled batches, and unlabelled batches in the semi regime
    """
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)
    img_size = config.get("img_size", 640)
    regime = config.get("regime", "labelled")

    labelled = _labelled_dataset(config, "train")
    labelled_loader = DataLoader(
        labelled,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_skip_none,
        drop_last=True,
        pin_memory=True,
    )

    if regime != "semi":
        return TrainStream(labelled_loader, None, 0, regime)

    paths = list_unlabeled(config.get("max_unlabeled"))
    unlabelled = UnlabelledImages(paths, img_size, build_val_transform(img_size))
    unlabelled_loader = DataLoader(
        unlabelled,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_skip_none,
        drop_last=True,
        pin_memory=True,
    )
    return TrainStream(labelled_loader, unlabelled_loader, config.get("unlabelled_ratio", 1), regime)


def _eval_loader(config, split) -> DataLoader:
    """Build a deterministic evaluation loader yielding image and ground truth boxes"""
    return DataLoader(
        _labelled_dataset(config, split),
        batch_size=config.get("batch_size", 32),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        collate_fn=collate_boxes_skip_none,
        pin_memory=True,
    )


def build_val_loader(config) -> DataLoader:
    """
    Build the selection loader over the val holdout carved from train2017

    Args:
        config: Resolved configuration holding the dataset and loader settings

    Returns:
        A DataLoader yielding image and ground truth boxes for per epoch selection
    """
    return _eval_loader(config, "val")


def build_test_loader(config) -> DataLoader:
    """
    Build the held out test loader over the untouched val2017 split

    Args:
        config: Resolved configuration holding the dataset and loader settings

    Returns:
        A DataLoader yielding image and ground truth boxes for the final test
    """
    return _eval_loader(config, "test")


def build_fidelity_loader(config, shard=1000):
    """
    Build a deterministic loader over a fixed unlabelled shard for the teacher student fidelity metric

    The shard is the same images every epoch and every run so the agreement curve is comparable
    across epochs and methods, it is only meaningful in the semi regime so it is None otherwise

    Args:
        config: Resolved configuration holding the regime and loader settings
        shard: Number of unlabelled images held fixed for the metric

    Returns:
        A DataLoader over the fixed unlabelled shard, or None outside the semi regime
    """
    if config.get("regime", "labelled") != "semi":
        return None

    img_size = config.get("img_size", 640)
    paths = list_unlabeled(config.get("max_unlabeled"))[:shard]
    dataset = UnlabelledImages(paths, img_size, build_val_transform(img_size))
    return DataLoader(
        dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        collate_fn=collate_skip_none,
        pin_memory=True,
    )
