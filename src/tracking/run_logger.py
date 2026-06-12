# -*- coding: utf-8 -*-
"""
Run logger that fans every call out to the local and cloud writers

The engine talks only to this coordinator, it freezes the config, exposes the checkpoint paths,
and forwards each scalar, benchmark, and summary to both backends

Adding a third backend later is one writer wired into the constructor, nothing in the engine
changes
"""

import os

from tracking.local import LocalWriter
from tracking.wandb import WandbWriter


def _flat_summary(summary: dict) -> dict:
    """Flatten the run summary into the flat key convention for the cloud run summary"""
    flat = {}
    for key in ("best_map50_95", "best_epoch", "total_train_seconds", "avg_epoch_seconds"):
        if key in summary:
            flat[key] = summary[key]
    for key, value in summary.get("final_test", {}).items():
        flat[f"test/{key}"] = value
    for key, value in summary.get("dataset", {}).items():
        if isinstance(value, (int, float)):
            flat[f"dataset/{key}"] = value
    return flat


class RunLogger:
    """
    Coordinator over the local and Weights and Biases writers

    Args:
        run_dir: Output directory of the run
        group: W&B group, the comparison the run belongs to
        name: W&B run name within the group
        config: Resolved config, frozen locally and recorded with the cloud run
    """

    def __init__(self, run_dir: str, group: str, name: str, config: dict):
        self.run_dir = run_dir
        self.local = LocalWriter(run_dir)
        self.local.write_config(config)
        self.wandb = WandbWriter(
            os.environ.get("WANDB_PROJECT"),
            os.environ.get("WANDB_ENTITY"),
            group,
            name,
            config.get("wandb_mode", "online"),
            config,
        )

    def path_best(self) -> str:
        """Path of the best validation checkpoint"""
        return os.path.join(self.run_dir, "checkpoints", "best.pt")

    def path_last(self) -> str:
        """Path of the most recent checkpoint"""
        return os.path.join(self.run_dir, "checkpoints", "last.pt")

    def log_epoch(self, epoch: int, scalars: dict):
        """Record the epoch scalars locally and in the cloud"""
        self.local.log_epoch(epoch, scalars)
        self.wandb.log(scalars, step=epoch)

    def log_benchmark(self, benchmark: dict):
        """Record the model benchmark locally and the numeric fields in the cloud"""
        self.local.write_benchmark(benchmark)
        numeric = {f"benchmark/{k}": v for k, v in benchmark.items() if isinstance(v, (int, float))}
        self.wandb.log(numeric)

    def log_summary(self, summary: dict):
        """Record the run level summary locally and its flat fields in the cloud run summary"""
        self.local.write_summary(summary)
        self.wandb.summary(_flat_summary(summary))

    def save_heatmap(self, epoch: int, index: int, prob):
        """Save a predicted heatmap visualisation"""
        self.local.save_heatmap(epoch, index, prob)

    def save_test_heatmap(self, index: int, prob):
        """Save a predicted test heatmap visualisation"""
        self.local.save_test_heatmap(index, prob)

    def save_test_detection(self, index: int, image, boxes):
        """Save a predicted test box overlay visualisation"""
        self.local.save_detection(index, image, boxes)

    def finish(self):
        """Close both backends"""
        self.wandb.finish()
        self.local.close()
