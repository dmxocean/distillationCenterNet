# -*- coding: utf-8 -*-
"""
Collate function that drops unreadable samples before stacking

Dataset items return None when a decode fails after the retry budget, this collate removes those
entries and stacks the rest, an entirely empty batch signals a sustained storage outage rather
than an isolated fault and is surfaced as an error
"""

from torch.utils.data import default_collate

from data.io import ImageReadError


def collate_skip_none(batch):
    """
    Drop None entries from a batch and stack the remaining samples

    Args:
        batch: List of per sample outputs where failed reads are None

    Returns:
        The default collated batch over the valid samples

    Raises:
        ImageReadError: When every sample in the batch failed to decode
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        raise ImageReadError("Entire batch unreadable, storage outage")
    return default_collate(batch)


def collate_boxes_skip_none(batch):
    """
    Drop None entries, stack the evaluation images, and keep the per image boxes as lists

    The ground truth and crowd ignore boxes carry a variable count per image, so they stay lists
    rather than being stacked, the images are collated normally

    Args:
        batch: List of per sample image, box, and ignore box tensors where failed reads are None

    Returns:
        The stacked image batch paired with the lists of per image ground truth and ignore boxes

    Raises:
        ImageReadError: When every sample in the batch failed to decode
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        raise ImageReadError("Entire batch unreadable, storage outage")
    images = default_collate([b[0] for b in batch])
    boxes = [b[1] for b in batch]
    ignore = [b[2] for b in batch]
    return images, boxes, ignore
