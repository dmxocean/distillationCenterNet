# -*- coding: utf-8 -*-
"""
Local writer that owns the on disk results tree for a run

The writer creates the run directory and its subfolders, freezes the resolved config, appends the
per epoch scalars to a single metrics file, and records the benchmark and summary artifacts

A file handler mirrors the python logging into train.log, the heatmap helper populates the bare
visualisation folder with a per channel view of the predicted centres
"""

import json
import logging
import os

import cv2
import numpy as np
import yaml


class LocalWriter:
    """
    Filesystem writer for one run directory

    Args:
        run_dir: Directory that receives every artifact of the run
    """

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.checkpoints = os.path.join(run_dir, "checkpoints")
        self.detections = os.path.join(run_dir, "detections")
        self.heatmaps = os.path.join(run_dir, "heatmaps")
        for directory in (run_dir, self.checkpoints, self.detections, self.heatmaps):
            os.makedirs(directory, exist_ok=True)

        self.metrics_path = os.path.join(run_dir, "metrics.json")
        self.history = []

        self._handler = logging.FileHandler(os.path.join(run_dir, "train.log"))
        self._handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(self._handler)

    def write_config(self, config: dict):
        """Freeze the resolved config as the single reproduction artifact"""
        with open(os.path.join(self.run_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    def log_epoch(self, epoch: int, scalars: dict):
        """Append the epoch scalars and rewrite the metrics history"""
        row = {"epoch": epoch}
        row.update(scalars)
        self.history.append(row)
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def write_benchmark(self, benchmark: dict):
        """Write the model benchmark artifact"""
        with open(os.path.join(self.run_dir, "benchmark.json"), "w", encoding="utf-8") as f:
            json.dump(benchmark, f, indent=2)

    def write_summary(self, summary: dict):
        """Write the run level summary artifact"""
        with open(os.path.join(self.run_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def save_heatmap(self, epoch: int, index: int, prob):
        """Save the predicted channel zero heatmap as a bare visualisation"""
        try:
            array = (prob.detach().cpu().numpy()[0] * 255).clip(0, 255).astype(np.uint8)
            name = f"epoch{epoch:03d}-c0-{index:03d}.png"
            cv2.imwrite(os.path.join(self.heatmaps, name), array)
        except Exception:  # Visualisation must never break a run
            logging.getLogger(__name__).warning("Failed to save heatmap visualisation")

    def save_test_heatmap(self, index: int, prob):
        """Save a predicted channel zero heatmap from the held out test split"""
        try:
            array = (prob.detach().cpu().numpy()[0] * 255).clip(0, 255).astype(np.uint8)
            name = f"test-c0-{index:03d}.png"
            cv2.imwrite(os.path.join(self.heatmaps, name), array)
        except Exception:  # Visualisation must never break a run
            logging.getLogger(__name__).warning("Failed to save test heatmap visualisation")

    def save_detection(self, index: int, image, boxes):
        """Save a test input image with the decoded boxes drawn as a qualitative overlay"""
        try:
            rgb = image.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
            canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # cv2 writes in blue green red order
            for box in boxes:
                x1, y1 = int(box["cx"] - box["w"] / 2), int(box["cy"] - box["h"] / 2)
                x2, y2 = int(box["cx"] + box["w"] / 2), int(box["cy"] + box["h"] / 2)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(self.detections, f"test-{index:03d}.png"), canvas)
        except Exception:  # Visualisation must never break a run
            logging.getLogger(__name__).warning("Failed to save detection overlay")

    def close(self):
        """Detach the train.log handler"""
        logging.getLogger().removeHandler(self._handler)
        self._handler.close()
