# -*- coding: utf-8 -*-
"""
Per class gaussian heatmap and geometry target generation

Object centres are rendered as gaussian peaks on a per class heatmap and the matching width and
height are written at each peak, the peak radius is chosen so the rendered centre keeps a minimum
overlap with the ground truth box

The rendering replicates the longest side resize and centre pad geometry of the transforms, so
the targets stay aligned with the augmented image fed to the network
"""

import numpy as np


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Compute a gaussian radius that keeps a minimum overlap with the ground truth box

    Args:
        det_size: Height and width of the object in heatmap pixels
        min_overlap: Required intersection over union between the peak and the box

    Returns:
        The radius as a continuous float
    """
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - np.sqrt(b1**2 - 4 * a1 * c1)) / (2 * a1)  # Smaller root, the minimal radius preserving the overlap

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 - np.sqrt(b2**2 - 4 * a2 * c2)) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + np.sqrt(b3**2 - 4 * a3 * c3)) / (2 * a3)

    return min(r1, r2, r3)


def gaussian_2d(shape, sigma=1):
    """Generate a normalised two dimensional gaussian kernel of the given shape"""
    center_y, center_x = [(ss - 1.0) / 2.0 for ss in shape]
    grid_y, grid_x = np.ogrid[-center_y : center_y + 1, -center_x : center_x + 1]
    kernel = np.exp(-((grid_x * grid_x) + (grid_y * grid_y)) / (2 * sigma * sigma))
    kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
    return kernel


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    Render a gaussian peak onto a single channel heatmap at the centre location

    Args:
        heatmap: Target array where the peak is accumulated by maximum
        center: Centre coordinates of the peak in heatmap pixels
        radius: Spatial spread of the gaussian
        k: Peak intensity at the centre
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left = min(x, radius)
    right = min(width - x, radius + 1)
    top = min(y, radius)
    bottom = min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def resize_pad_geometry(img_shape, img_size):
    """Compute the longest side resize ratio and the centre pad offsets in input pixels"""
    ho, wo = img_shape[0], img_shape[1]
    if wo > ho:
        ratio = wo / float(img_size)  # Longest side is the width
        wpad, hpad = 0, int((img_size - (ho / ratio)) / 2)
    else:
        ratio = ho / float(img_size)  # Longest side is the height
        wpad, hpad = int((img_size - (wo / ratio)) / 2), 0
    return ratio, wpad, hpad


def render_targets(img_shape, gt_list, num_classes, img_size, stride):
    """
    Render the per class heatmap and the geometry map for one annotated image

    The boxes are mapped through the same longest side resize and centre pad as the image, then
    each centre is drawn as a gaussian peak on its class channel with the width and height stored
    at the peak

    Args:
        img_shape: Original image shape as height, width, channels
        gt_list: Ground truth boxes as x, y, w, h, channel in original pixels
        num_classes: Number of enabled classes, one heatmap channel each
        img_size: Network input resolution after resize and pad
        stride: Downsampling ratio between input and heatmap

    Returns:
        The heatmap of shape classes, height, width and the geometry of shape height, width, two
    """
    hmap_size = img_size // stride
    heatmap = np.zeros((num_classes, hmap_size, hmap_size), dtype=np.float32)
    geometry = np.zeros((hmap_size, hmap_size, 2), dtype=np.float32)

    ratio, wpad, hpad = resize_pad_geometry(img_shape, img_size)

    for obj in gt_list:
        class_idx = int(obj[4])
        x_tl = (obj[0] / ratio) + wpad
        y_tl = (obj[1] / ratio) + hpad
        x_br = ((obj[0] + obj[2]) / ratio) + wpad
        y_br = ((obj[1] + obj[3]) / ratio) + hpad

        box_w = (x_br - x_tl) / stride
        box_h = (y_br - y_tl) / stride
        cx = int(((x_tl + x_br) / 2) / stride)  # Quantise the true centre once, no floor before averaging
        cy = int(((y_tl + y_br) / 2) / stride)
        cx = min(max(cx, 0), hmap_size - 1)  # Keep edge touching centres inside the grid
        cy = min(max(cy, 0), hmap_size - 1)

        radius = max(1, int(gaussian_radius((box_h, box_w))))
        peak = np.zeros((hmap_size, hmap_size), dtype=np.float32)
        draw_umich_gaussian(peak, (cx, cy), radius)

        heatmap[class_idx] = np.maximum(heatmap[class_idx], peak)
        geometry[peak == 1, 0] = box_w  # Store width at the exact centre
        geometry[peak == 1, 1] = box_h  # Store height at the exact centre

    return heatmap, geometry


def boxes_to_input_space(img_shape, gt_list, img_size):
    """
    Map ground truth boxes through the resize and pad into centre form in input pixels

    The boxes follow the same longest side resize and centre pad as the image and the rendered
    heatmap, but stay continuous so evaluation matches predictions against real localization
    instead of against centres quantised to the stride grid

    Args:
        img_shape: Original image shape as height, width, channels
        gt_list: Ground truth boxes as x, y, w, h, channel in original pixels
        img_size: Network input resolution after resize and pad

    Returns:
        A list of boxes as centre x, centre y, width, height, channel in input pixels
    """
    ratio, wpad, hpad = resize_pad_geometry(img_shape, img_size)
    boxes = []
    for obj in gt_list:
        w = obj[2] / ratio
        h = obj[3] / ratio
        cx = (obj[0] / ratio) + wpad + w / 2
        cy = (obj[1] / ratio) + hpad + h / 2
        boxes.append([cx, cy, w, h, int(obj[4])])
    return boxes


def render_ignore_mask(img_shape, crowd_list, num_classes, img_size, stride):
    """
    Mark the heatmap cells covered by crowd boxes so the focal loss can skip them

    Crowd boxes group many unlabelled instances of one class, the mask flags every grid cell their
    area touches on that class channel so those cells are neither rewarded nor penalised, the rest
    of the image trains normally

    Args:
        img_shape: Original image shape as height, width, channels
        crowd_list: Crowd boxes as x, y, w, h, channel in original pixels
        num_classes: Number of enabled classes, one mask channel each
        img_size: Network input resolution after resize and pad
        stride: Downsampling ratio between input and heatmap

    Returns:
        A mask of shape classes, height, width with one at the ignored cells and zero elsewhere
    """
    hmap_size = img_size // stride
    mask = np.zeros((num_classes, hmap_size, hmap_size), dtype=np.float32)
    if not crowd_list:
        return mask

    ratio, wpad, hpad = resize_pad_geometry(img_shape, img_size)
    for obj in crowd_list:
        class_idx = int(obj[4])
        x_tl = (obj[0] / ratio) + wpad
        y_tl = (obj[1] / ratio) + hpad
        x_br = ((obj[0] + obj[2]) / ratio) + wpad
        y_br = ((obj[1] + obj[3]) / ratio) + hpad

        col0 = max(0, int(x_tl // stride))
        row0 = max(0, int(y_tl // stride))
        col1 = min(hmap_size, int(np.ceil(x_br / stride)))
        row1 = min(hmap_size, int(np.ceil(y_br / stride)))
        mask[class_idx, row0:row1, col0:col1] = 1.0

    return mask
