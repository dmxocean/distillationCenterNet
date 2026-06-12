# -*- coding: utf-8 -*-
"""
End of run finalisation helpers shared by the trainer and the distiller

The sample saver writes a heatmap visualisation during training, the final test reloads the best
checkpoint, scores the held out split once, and saves a few heatmaps and box overlays
"""

import logging
import os

import torch

from engine.checkpoint import load_checkpoint
from evaluation.decode import decode_boxes
from evaluation.metrics import evaluate, flat_metrics

logger = logging.getLogger(__name__)


def save_sample(run_logger, model, val_loader, device, epoch):
    """Save a heatmap visualisation for the first validation image"""
    try:
        model.eval()  # Sampling must not update batch norm running statistics
        images = next(iter(val_loader))[0]
        with torch.no_grad():
            out = model(images.to(device))
        run_logger.save_heatmap(epoch, 0, out.prob[0])
    except Exception:  # Visualisation must never break a run
        logger.warning("Failed to save sample visualisation at epoch %d", epoch)


def final_test(run_logger, model, test_loader, device, num_classes, stride, eval_cfg=None):
    """
    Reload the best checkpoint and evaluate once on the held out test split

    The selected best checkpoint is restored so the reported number reflects the model that won
    selection, the test split is scored a single time and a few heatmaps are saved

    Args:
        run_logger: Coordinator owning the checkpoint paths and the visual folders
        model: Detector to restore and evaluate
        test_loader: Loader over the held out test split
        device: Device the model runs on
        num_classes: Number of enabled classes
        stride: Downsampling ratio used by the decoder

    Returns:
        The flat test metrics dict written into the run summary
    """
    if os.path.exists(run_logger.path_best()):
        load_checkpoint(run_logger.path_best(), model, map_location=device)  # Report from the selected best

    quality = evaluate(model, test_loader, device, num_classes, stride, with_size=True, eval_cfg=eval_cfg)
    score = sum(q.map50_95 for q in quality.values()) / max(len(quality), 1)
    logger.info("Final test map50_95 %.4f", score)

    try:
        images = next(iter(test_loader))[0][:3]
        with torch.no_grad():
            out = model(images.to(device))
        cfg = eval_cfg or {}
        prob, geometry = out.prob.detach().cpu(), out.geometry.detach().cpu()
        for index in range(images.shape[0]):
            boxes = decode_boxes(prob[index], geometry[index], cfg.get("operating_score", 0.3), stride, kernel=cfg.get("nms_kernel", 5), top_k=cfg.get("max_dets", 100))
            run_logger.save_test_heatmap(index, prob[index])
            run_logger.save_test_detection(index, images[index], boxes)
    except Exception:  # Visualisation must never break a run
        logger.warning("Failed to save test visualisations")

    return flat_metrics(quality)
