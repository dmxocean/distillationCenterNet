# -*- coding: utf-8 -*-
"""
Supervised fit loop for the teacher and the baseline

The trainer optimises the detector on the labelled stream with the focal and geometry losses, it
validates every epoch, checkpoints the best and last states, and stops early when the validation
score plateaus

The teacher and the baseline share this loop, they differ only in width and in whether ImageNet
weights seed the backbone, the resulting checkpoint is what the students distil from
"""

import logging
import os
import time

import torch

from data.streams import build_test_loader, build_train_stream, build_val_loader
from engine.checkpoint import load_checkpoint, save_checkpoint
from engine.early_stop import EarlyStopping
from engine.finalize import final_test, save_sample
from engine.loop import move_batch, supervised_step
from engine.optim import build_optimizer, build_scheduler
from engine.report import model_benchmark
from engine.seeding import seed_everything
from evaluation.metrics import evaluate, flat_metrics
from losses.supervised import SupervisedLoss
from models.centernet import CenterNet

logger = logging.getLogger(__name__)


def fit(config: dict, run_logger, role: str, width: float, resume: bool = False) -> dict:
    """
    Train a detector under supervision and write its artifacts

    Args:
        config: Resolved configuration for the run
        run_logger: Coordinator that owns the artifacts and the cloud mirror
        role: Run role, teacher or baseline
        width: Backbone width multiplier for the detector
        resume: Restore model, optimiser, epoch, and best from last.pt when present

    Returns:
        The run summary dict written to summary.json
    """
    seed_everything(config.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True  # Autotune convolutions for the fixed input size
    num_classes = config["num_classes"]
    img_size = config.get("img_size", 640)
    stride = config.get("stride", 4)

    model = CenterNet(num_classes, width, pretrained=(width == 1.0), img_size=img_size, stride=stride).to(device)
    stream = build_train_stream(config)
    val_loader = build_val_loader(config)
    supervised = SupervisedLoss(config.get("focal_weight", 1.0), config.get("reg_weight", 0.1))
    optimizer = build_optimizer(model, config)
    best, best_epoch, total_seconds, epoch = float("-inf"), -1, 0.0, 0
    start_epoch = 0
    if resume and os.path.exists(run_logger.path_last()):
        state = load_checkpoint(run_logger.path_last(), model, optimizer, map_location=device)
        start_epoch, best, epoch = state["epoch"] + 1, state["best"], state["epoch"]
        logger.info("Resuming from epoch %d, best %.4f", start_epoch, best)
    scheduler = build_scheduler(optimizer, config, last_epoch=start_epoch - 1)
    eval_every = config.get("eval_every", 1)
    patience = max(1, round(config.get("es_patience", 10) / eval_every))  # es_patience is in epochs
    stopper = EarlyStopping(patience, config.get("es_min_delta", 1e-4))

    epochs = config["epochs"]
    sample_every = config.get("sample_every", 5)
    use_amp = config.get("amp", "off") == "bf16" and device == "cuda"
    grad_clip = config.get("grad_clip")
    logger.info("Starting %s training on %s for %d epochs", role, device, epochs)

    for epoch in range(start_epoch, epochs):
        model.train()
        start = time.perf_counter()
        agg = {"total": 0.0, "focal": 0.0, "reg": 0.0}
        batches = 0
        for batch in stream:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, hard, _ = supervised_step(batch, model, supervised)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            agg["total"] += loss.item()
            agg["focal"] += hard.focal.item()
            agg["reg"] += hard.reg.item()
            batches += 1
        scheduler.step()
        seconds = time.perf_counter() - start
        total_seconds += seconds

        scalars = {
            "loss/total": agg["total"] / max(batches, 1),
            "loss/focal": agg["focal"] / max(batches, 1),
            "loss/reg": agg["reg"] / max(batches, 1),
            "time/epoch_seconds": seconds,
            "lr": optimizer.param_groups[0]["lr"],
        }
        is_eval = epoch % eval_every == 0 or epoch == epochs - 1
        if is_eval:
            quality = evaluate(model, val_loader, device, num_classes, stride, eval_cfg=config.get("eval"))
            score = sum(q.map50_95 for q in quality.values()) / max(len(quality), 1)
            scalars.update(flat_metrics(quality))
        run_logger.log_epoch(epoch, scalars)
        if is_eval:
            logger.info("Epoch %d loss %.4f map50_95 %.4f time %.1fs", epoch, scalars["loss/total"], score, seconds)
        else:
            logger.info("Epoch %d loss %.4f time %.1fs", epoch, scalars["loss/total"], seconds)

        save_checkpoint(run_logger.path_last(), model, optimizer, epoch, best)
        if is_eval:
            if stopper.step(score):
                best, best_epoch = score, epoch
                save_checkpoint(run_logger.path_best(), model, optimizer, epoch, best)
            if stopper.should_stop:
                logger.info("Early stopping at epoch %d", epoch)
                break
        if epoch % sample_every == 0:
            save_sample(run_logger, model, val_loader, device, epoch)

    test_loader = build_test_loader(config)
    test_metrics = final_test(run_logger, model, test_loader, device, num_classes, stride, config.get("eval"))

    run_logger.log_benchmark(model_benchmark(model, device, img_size))
    summary = {
        "role": role,
        "width": width,
        "device": device,
        "dataset": {
            "name": config.get("name", "coco"),
            "enabled_categories": config["enabled_categories"],
            "num_classes": num_classes,
            "train": len(stream.labelled_loader.dataset),
            "val": len(val_loader.dataset),
            "test": len(test_loader.dataset),
        },
        "best_epoch": best_epoch,
        "best_map50_95": best,
        "total_train_seconds": round(total_seconds, 2),
        "avg_epoch_seconds": round(total_seconds / max(epoch + 1, 1), 2),
        "final_test": test_metrics,
    }
    run_logger.log_summary(summary)
    run_logger.finish()
    return summary
