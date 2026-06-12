# -*- coding: utf-8 -*-
"""
PyTorch datasets for labelled detection and unlabelled images

The labelled dataset turns an image and its boxes into the input tensor, the per class heatmap,
and the geometry map, the unlabelled dataset returns only the input tensor so the teacher can
generate pseudo targets for distillation

Both datasets return None for an unreadable sample so the collate step can drop it without
crashing the run
"""

import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from data.io import ImageReadError, read_image_rgb
from data.targets import boxes_to_input_space, render_ignore_mask, render_targets

logger = logging.getLogger(__name__)


class LabelledDetection(Dataset):
    """
    Labelled detection dataset producing input, heatmap, and geometry tensors for training

    Training adds an ignore mask over the cells covered by crowd boxes so the focal loss skips them,
    in the evaluation mode the dataset returns the input, the ground truth boxes, and the crowd
    ignore boxes mapped through the same resize and pad instead of the rendered targets, so metrics
    score predictions against the real boxes rather than against centres quantised to the stride grid

    Args:
        items: Mapping from image path to its clean and crowd boxes record
        num_classes: Number of enabled classes, one heatmap channel each
        img_size: Network input resolution after resize and pad
        stride: Downsampling ratio between input and heatmap
        transform: Augmentation pipeline applied to the image
        return_boxes: Whether to return the evaluation ground truth and ignore boxes instead of targets
    """

    def __init__(self, items, num_classes, img_size=640, stride=4, transform=None, return_boxes=False):
        self.items = list(items.items())
        self.num_classes = num_classes
        self.img_size = img_size
        self.stride = stride
        self.transform = transform
        self.return_boxes = return_boxes

    def __len__(self):
        """Return the number of labelled images"""
        return len(self.items)

    def __getitem__(self, index):
        """Load one sample into training targets, or into evaluation ground truth and ignore boxes"""
        img_path, ann = self.items[index]
        gt_list, crowd_list = ann["boxes"], ann["crowd"]
        try:
            img = read_image_rgb(img_path)
        except ImageReadError:
            logger.warning("Dropping unreadable sample at %s", img_path)
            return None

        if self.return_boxes:
            boxes = boxes_to_input_space(img.shape, gt_list, self.img_size)
            ignore = boxes_to_input_space(img.shape, crowd_list, self.img_size)
            if self.transform is not None:
                img = self.transform(image=img)["image"]
            img_tensor = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).permute(2, 0, 1)
            box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 5)
            ignore_tensor = torch.tensor(ignore, dtype=torch.float32).reshape(-1, 5)
            return img_tensor, box_tensor, ignore_tensor

        heatmap, geometry = render_targets(img.shape, gt_list, self.num_classes, self.img_size, self.stride)
        ignore_mask = render_ignore_mask(img.shape, crowd_list, self.num_classes, self.img_size, self.stride)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        img_tensor = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).permute(2, 0, 1)
        heatmap_tensor = torch.from_numpy(heatmap)
        geometry_tensor = torch.from_numpy(geometry).permute(2, 0, 1)
        ignore_tensor = torch.from_numpy(ignore_mask)
        return img_tensor, heatmap_tensor, geometry_tensor, ignore_tensor


class UnlabelledImages(Dataset):
    """
    Unlabelled image dataset returning only the input tensor

    Args:
        image_paths: Image paths from the unlabelled split
        img_size: Network input resolution after resize and pad
        transform: Deterministic transform applied to the image
    """

    def __init__(self, image_paths, img_size=640, transform=None):
        self.image_paths = list(image_paths)
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        """Return the number of unlabelled images"""
        return len(self.image_paths)

    def __getitem__(self, index):
        """Load and transform one unlabelled image into the input tensor"""
        img_path = self.image_paths[index]
        try:
            img = read_image_rgb(img_path)
        except ImageReadError:
            logger.warning("Dropping unreadable sample at %s", img_path)
            return None

        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).permute(2, 0, 1)
