# -*- coding: utf-8 -*-
"""
Per class detection quality over an evaluation split

The evaluator decodes predictions at a low score so the full precision recall curve is visible,
matches them against the real ground truth boxes by intersection over union, and pools every
detection across the split into one precision recall curve per threshold

Average precision is reported at the canonical and the swept thresholds, average recall is the
recall side complement, precision recall and the f one score are read at a fixed operating point,
and the final test additionally breaks the average precision down by object size
"""

from dataclasses import dataclass

import numpy as np
import torch

from evaluation.decode import decode_boxes
from losses.ops import softened_kl

_IOU_THRESHOLDS = (0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
_SIZE_LIMITS = {"small": 32 ** 2, "medium": 96 ** 2}  # Area bounds in network input pixels
_EMPTY = -1.0                 # Sentinel for a metric with no ground truth to score
_EVAL_DEFAULTS = {            # Fallback eval thresholds, overridden by the config eval block
    "decode_score": 0.05,     # Low decode score so average precision sees the full recall tail
    "operating_score": 0.3,   # Score for the precision recall and f one operating point only
    "nms_kernel": 5,          # Max pool window for local peak detection
    "max_dets": 100,          # Detections kept per image, paired with the low decode score
}
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz"))  # trapezoid replaces trapz in newer numpy


@dataclass
class DetectionQuality:
    """Headline detection metrics for one class"""

    map50_95: float
    map50: float
    ar100: float
    precision: float
    recall: float
    f1: float
    ap_small: float = _EMPTY
    ap_medium: float = _EMPTY
    ap_large: float = _EMPTY


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert centre width height boxes to corner form"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)


def _iou_matrix(boxes_a, boxes_b):
    """Pairwise intersection over union between two lists of centre width height boxes"""
    from torchvision.ops import box_iou

    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return torch.zeros(len(boxes_a), len(boxes_b))
    a = torch.tensor(boxes_a, dtype=torch.float32)
    b = torch.tensor(boxes_b, dtype=torch.float32)
    return box_iou(_cxcywh_to_xyxy(a), _cxcywh_to_xyxy(b))


def _filter_class(record, channel, with_scores):
    """Keep only the boxes of one class from a per image record"""
    boxes = [b for b, l in zip(record["boxes"], record["labels"]) if l == channel]
    out = {"boxes": boxes}
    if with_scores:
        out["scores"] = [s for s, l in zip(record["scores"], record["labels"]) if l == channel]
    return out


def _prepare(preds_c, gts_c, ignores_c=None):
    """
    Precompute the per image arrays reused across thresholds and size buckets

    Args:
        preds_c: Per image prediction records for one class with boxes and scores
        gts_c: Per image ground truth records for one class with boxes
        ignores_c: Optional per image crowd ignore records for one class with boxes

    Returns:
        A list of per image tuples with prediction scores, prediction areas, ground truth areas,
        the prediction to ground truth and the prediction to ignore intersection over union matrices
    """
    per_image = []
    for idx, (pred, gt) in enumerate(zip(preds_c, gts_c)):
        pred_boxes, gt_boxes = pred["boxes"], gt["boxes"]
        pred_scores = np.array(pred["scores"], dtype=np.float32) if pred_boxes else np.zeros(0, dtype=np.float32)
        pred_area = np.array([b[2] * b[3] for b in pred_boxes], dtype=np.float32) if pred_boxes else np.zeros(0)
        gt_area = np.array([b[2] * b[3] for b in gt_boxes], dtype=np.float32) if gt_boxes else np.zeros(0)
        iou = _iou_matrix(pred_boxes, gt_boxes).numpy()
        ignore_boxes = ignores_c[idx]["boxes"] if ignores_c is not None else []
        iou_ignore = _iou_matrix(pred_boxes, ignore_boxes).numpy()
        per_image.append((pred_scores, pred_area, gt_area, iou, iou_ignore))
    return per_image


def _collect(per_image, threshold, size_range=None, score_min=0.0):
    """
    Pool detections across the split into global score and true positive flags at one threshold

    Predictions are matched greedily in score order to the best still available ground truth, when
    a size range is given a prediction that matches only an out of range ground truth is ignored
    rather than counted as a false positive, following the COCO area rule, an unmatched prediction
    that overlaps a crowd ignore region is discarded as neither a true nor a false positive

    Args:
        per_image: Per image arrays from _prepare
        threshold: Intersection over union threshold for a positive match
        size_range: Optional low and high area bounds restricting the ground truth
        score_min: Minimum prediction score to consider, used for the operating point

    Returns:
        The parallel global scores and true positive flags and the total in range ground truth
    """
    scores_all, tps_all, total_gt = [], [], 0
    for pred_scores, pred_area, gt_area, iou, iou_ignore in per_image:
        n_gt = len(gt_area)
        n_ignore = iou_ignore.shape[1]
        if size_range is None:
            gt_in = np.ones(n_gt, dtype=bool)
            pred_in = None
        else:
            low, high = size_range
            gt_in = (gt_area >= low) & (gt_area < high)
            pred_in = (pred_area >= low) & (pred_area < high)
        total_gt += int(gt_in.sum())

        if len(pred_scores) == 0:
            continue
        available = np.ones(n_gt, dtype=bool)
        for i in np.argsort(-pred_scores):
            if pred_scores[i] < score_min:
                continue
            best, j = -1.0, -1
            if n_gt > 0:
                row = iou[i].copy()
                row[~available] = -1.0
                j = int(row.argmax())
                best = row[j]
            if j >= 0 and best >= threshold:
                available[j] = False
                if gt_in[j]:  # Matched an in range centre, a true positive
                    scores_all.append(float(pred_scores[i]))
                    tps_all.append(1)
                # A match to an out of range centre is ignored
            elif n_ignore > 0 and iou_ignore[i].max() >= threshold:  # Falls in a crowd region, discarded
                continue
            elif size_range is None or pred_in[i]:  # Unmatched, a false positive in this range
                scores_all.append(float(pred_scores[i]))
                tps_all.append(0)
    return scores_all, tps_all, total_gt


def _average_precision(scores, tps, total_gt):
    """Integrate the global precision recall curve into a single average precision"""
    if total_gt == 0:
        return _EMPTY
    if not scores:
        return 0.0
    order = np.argsort(-np.array(scores, dtype=np.float64))
    tps_arr = np.array(tps, dtype=np.float64)[order]
    tp_cum = np.cumsum(tps_arr)
    fp_cum = np.cumsum(1.0 - tps_arr)
    recall = tp_cum / total_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    mrec = np.concatenate([[0.0], recall, [recall[-1]]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):  # Monotonic precision envelope
        mpre[i] = max(mpre[i], mpre[i + 1])
    return float(_trapezoid(mpre, mrec))


def _recall(scores, tps, total_gt):
    """Global recall at the detection budget for the average recall term"""
    if total_gt == 0:
        return _EMPTY
    return float(sum(tps) / total_gt)


def _operating(per_image, threshold, score_min):
    """True false positive and false negative counts at the operating point"""
    _, tps, total_gt = _collect(per_image, threshold, None, score_min)
    tp = int(sum(tps))
    return tp, len(tps) - tp, total_gt - tp


def _mean_valid(values):
    """Mean over the metrics that had ground truth to score, otherwise the empty sentinel"""
    valid = [v for v in values if v != _EMPTY]
    return float(np.mean(valid)) if valid else _EMPTY


def _size_ap(per_image, size_range):
    """Mean average precision restricted to one object size bucket"""
    return _mean_valid([_average_precision(*_collect(per_image, t, size_range)) for t in _IOU_THRESHOLDS])


def compute_quality(predictions, ground_truth, num_classes, with_size=False, operating_score=0.3, ignore=None):
    """
    Compute the detection quality for every class

    Args:
        predictions: Per image dicts with boxes, scores, and labels
        ground_truth: Per image dicts with boxes and labels
        num_classes: Number of enabled classes
        with_size: Whether to add the small medium large average precision, used on the final test
        operating_score: Minimum score for the precision recall and f one operating point
        ignore: Optional per image dicts with crowd ignore boxes and labels

    Returns:
        A dict mapping the class channel to its DetectionQuality
    """
    result = {}
    for channel in range(num_classes):
        per_image = _prepare(
            [_filter_class(p, channel, True) for p in predictions],
            [_filter_class(g, channel, False) for g in ground_truth],
            [_filter_class(g, channel, False) for g in ignore] if ignore is not None else None,
        )

        aps, recalls = [], []
        for t in _IOU_THRESHOLDS:
            scores, tps, total_gt = _collect(per_image, t)
            aps.append(_average_precision(scores, tps, total_gt))
            recalls.append(_recall(scores, tps, total_gt))

        tp, fp, fn = _operating(per_image, _IOU_THRESHOLDS[0], operating_score)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        quality = DetectionQuality(
            map50_95=_mean_valid(aps),
            map50=aps[0],
            ar100=_mean_valid(recalls),
            precision=precision,
            recall=recall,
            f1=f1,
        )
        if with_size:
            quality.ap_small = _size_ap(per_image, (0.0, _SIZE_LIMITS["small"]))
            quality.ap_medium = _size_ap(per_image, (_SIZE_LIMITS["small"], _SIZE_LIMITS["medium"]))
            quality.ap_large = _size_ap(per_image, (_SIZE_LIMITS["medium"], float("inf")))
        result[channel] = quality
    return result


def flat_metrics(quality_by_class) -> dict:
    """Turn the per class quality into flat logging keys, omitting metrics with no ground truth"""
    out = {}
    for channel, q in quality_by_class.items():
        candidates = {
            f"map/c{channel}/50_95": q.map50_95,
            f"map/c{channel}/50": q.map50,
            f"ar/c{channel}/100": q.ar100,
            f"precision/c{channel}": q.precision,
            f"recall/c{channel}": q.recall,
            f"f1/c{channel}": q.f1,
            f"map/c{channel}/small": q.ap_small,
            f"map/c{channel}/medium": q.ap_medium,
            f"map/c{channel}/large": q.ap_large,
        }
        out.update({key: value for key, value in candidates.items() if value != _EMPTY})
    return out


def _to_record(boxes, with_scores):
    """Turn decoded box dicts into a parallel record of boxes, labels, and optional scores"""
    record = {
        "boxes": [[b["cx"], b["cy"], b["w"], b["h"]] for b in boxes],
        "labels": [b["cls"] for b in boxes],
    }
    if with_scores:
        record["scores"] = [b["score"] for b in boxes]
    return record


def _gt_record(boxes):
    """Turn the carried ground truth boxes of shape N by five into a boxes and labels record"""
    rows = boxes.tolist()
    return {
        "boxes": [row[:4] for row in rows],
        "labels": [int(row[4]) for row in rows],
    }


@torch.no_grad()
def evaluate(model, loader, device, num_classes, stride=4, with_size=False, eval_cfg=None):
    """
    Run the detector over an evaluation loader and compute per class quality

    Predictions are decoded at a low score and capped per image so the average precision sees the
    full recall tail, ground truth is the real boxes carried from the dataset through the same
    resize and pad, predictions falling in a crowd ignore region are discarded from the score, the
    thresholds come from the config eval block and fall back to the defaults

    Args:
        model: Detector to evaluate
        loader: Loader yielding image, ground truth boxes, and crowd ignore boxes
        device: Device the model runs on
        num_classes: Number of enabled classes
        stride: Downsampling ratio used by the decoder
        with_size: Whether to add the small medium large breakdown, used on the final test
        eval_cfg: Decode and metric thresholds, merged over the defaults

    Returns:
        A dict mapping the class channel to its DetectionQuality
    """
    cfg = {**_EVAL_DEFAULTS, **(eval_cfg or {})}
    model.eval()
    predictions, ground_truth, ignored = [], [], []
    for images, boxes, ignore in loader:
        out = model(images.to(device))
        prob = out.prob.detach().cpu()
        pred_geom = out.geometry.detach().cpu()
        for i in range(images.shape[0]):
            pred_boxes = decode_boxes(prob[i], pred_geom[i], cfg["decode_score"], stride, kernel=cfg["nms_kernel"], top_k=cfg["max_dets"])
            predictions.append(_to_record(pred_boxes, with_scores=True))
            ground_truth.append(_gt_record(boxes[i]))
            ignored.append(_gt_record(ignore[i]))
    return compute_quality(predictions, ground_truth, num_classes, with_size, cfg["operating_score"], ignore=ignored)


@torch.no_grad()
def fidelity_kl(student, teacher, loader, device, temperature=1.0):
    """
    Mean softened divergence between student and teacher heatmaps on a fixed unlabelled shard

    Measures how closely the student reproduces the teacher distribution on the distillation data,
    a lower value is a closer match, the teacher is the fixed reference and the shard is the same
    every epoch so the curve tracks transfer over training

    Args:
        student: Student detector under training
        teacher: Frozen teacher detector
        loader: Loader over the fixed unlabelled shard
        device: Device the models run on
        temperature: Softening temperature applied to both logits

    Returns:
        The mean divergence over the shard, or the empty sentinel when the shard is empty
    """
    student.eval()
    total, count = 0.0, 0
    for images in loader:
        images = images.to(device)
        student_out = student(images)
        teacher_out = teacher(images)
        total += softened_kl(student_out.logits, teacher_out.logits, temperature).item() * images.shape[0]
        count += images.shape[0]
    return total / count if count else _EMPTY
