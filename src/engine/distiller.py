# -*- coding: utf-8 -*-
"""
Distillation fit loop for the student

The distiller trains the student on the labelled or the semi supervised stream, the distillation
term applies on every batch from the frozen teacher while the supervised term applies only on
labelled batches

The method is built from the registry by its configuration name, so swapping the distillation loss
never touches this loop, the run writes the same artifacts as the supervised trainer
"""

import logging
import os
import time

import torch

from data.streams import build_fidelity_loader, build_test_loader, build_train_stream, build_val_loader
from engine.checkpoint import load_checkpoint, save_checkpoint
from engine.early_stop import EarlyStopping
from engine.finalize import final_test, save_sample
from engine.loop import LossWeights, distill_step, move_batch
from engine.optim import build_optimizer, build_scheduler
from engine.report import model_benchmark
from engine.seeding import seed_everything
from evaluation.metrics import evaluate, fidelity_kl, flat_metrics
from kd.base import SignalSpec
from kd.registry import build
from kd.signals import SignalExtractor
from losses.supervised import SupervisedLoss
from models.centernet import CenterNet
from models.teacher import load_teacher

logger = logging.getLogger(__name__)


def fit(config: dict, run_logger, resume: bool = False) -> dict:
    """
    Distil a student from the frozen teacher and write its artifacts

    Args:
        config: Resolved configuration holding the method, regime, and teacher checkpoint
        run_logger: Coordinator that owns the artifacts and the cloud mirror
        resume: Restore student, optimiser, epoch, and best from last.pt when present

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
    width = config.get("student_width", 0.35)

    student = CenterNet(num_classes, width, pretrained=False, img_size=img_size, stride=stride).to(device)
    teacher = load_teacher(config["teacher_checkpoint"], num_classes, device, img_size=img_size, stride=stride)
    method = build(config["method"])
    hooked = {s for s in method.required_signals() if s in (SignalSpec.FEATURES, SignalSpec.ATTENTION)}
    extractors = (SignalExtractor(student, hooked), SignalExtractor(teacher, hooked)) if hooked else None
    stream = build_train_stream(config)
    val_loader = build_val_loader(config)
    fidelity_loader = build_fidelity_loader(config)  # Fixed unlabelled shard, semi regime only
    supervised = SupervisedLoss(config.get("focal_weight", 1.0), config.get("reg_weight", 0.1))
    weights = LossWeights(config.get("hard_weight", 1.0), config.get("distill_weight", 1.0))
    geom_weight = config.get("distill_reg_weight", 0.0)  # Geometry transfer weight, zero disables it
    peak_threshold = config.get("distill_peak_threshold", 0.3)
    optimizer = build_optimizer(student, config)
    best, best_epoch, total_seconds, epoch = float("-inf"), -1, 0.0, 0
    start_epoch = 0
    if resume and os.path.exists(run_logger.path_last()):
        state = load_checkpoint(run_logger.path_last(), student, optimizer, map_location=device)
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
    logger.info("Starting student %s on %s for %d epochs", config["method"]["name"], device, epochs)

    for epoch in range(start_epoch, epochs):
        student.train()
        start = time.perf_counter()
        agg = {"total": 0.0, "focal": 0.0, "reg": 0.0}
        kd_agg = {}
        batches, labelled_batches = 0, 0
        for batch in stream:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, distill, hard = distill_step(batch, student, teacher, method, supervised, weights, geom_weight, peak_threshold, extractors)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            optimizer.step()

            agg["total"] += loss.item()
            for name, value in distill.terms.items():
                kd_agg[name] = kd_agg.get(name, 0.0) + value.item()
            if hard is not None:
                agg["focal"] += hard.focal.item()
                agg["reg"] += hard.reg.item()
                labelled_batches += 1
            batches += 1
        scheduler.step()
        seconds = time.perf_counter() - start
        total_seconds += seconds

        scalars = {
            "loss/total": agg["total"] / max(batches, 1),
            "loss/focal": agg["focal"] / max(labelled_batches, 1),
            "loss/reg": agg["reg"] / max(labelled_batches, 1),
            "time/epoch_seconds": seconds,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"loss/{name}": total / max(batches, 1) for name, total in kd_agg.items()},
        }
        is_eval = epoch % eval_every == 0 or epoch == epochs - 1
        if is_eval:
            quality = evaluate(student, val_loader, device, num_classes, stride, eval_cfg=config.get("eval"))
            score = sum(q.map50_95 for q in quality.values()) / max(len(quality), 1)
            scalars.update(flat_metrics(quality))
            if fidelity_loader is not None:  # Teacher student agreement on the unlabelled stream
                scalars["fidelity/kl"] = fidelity_kl(student, teacher, fidelity_loader, device)
        run_logger.log_epoch(epoch, scalars)
        if is_eval:
            logger.info("Epoch %d loss %.4f map50_95 %.4f time %.1fs", epoch, scalars["loss/total"], score, seconds)
        else:
            logger.info("Epoch %d loss %.4f time %.1fs", epoch, scalars["loss/total"], seconds)

        save_checkpoint(run_logger.path_last(), student, optimizer, epoch, best)
        if is_eval:
            if stopper.step(score):
                best, best_epoch = score, epoch
                save_checkpoint(run_logger.path_best(), student, optimizer, epoch, best)
            if stopper.should_stop:
                logger.info("Early stopping at epoch %d", epoch)
                break
        if epoch % sample_every == 0:
            save_sample(run_logger, student, val_loader, device, epoch)

    test_loader = build_test_loader(config)
    test_metrics = final_test(run_logger, student, test_loader, device, num_classes, stride, config.get("eval"))

    run_logger.log_benchmark(model_benchmark(student, device, img_size))
    summary = {
        "role": "student",
        "method": config["method"]["name"],
        "regime": config.get("regime", "labelled"),
        "width": width,
        "device": device,
        "teacher_checkpoint": config["teacher_checkpoint"],
        "dataset": {
            "name": config.get("name", "coco"),
            "enabled_categories": config["enabled_categories"],
            "num_classes": num_classes,
            "train": len(stream.labelled_loader.dataset),
            "unlabeled": len(stream.unlabelled_loader.dataset) if stream.unlabelled_loader is not None else 0,
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
