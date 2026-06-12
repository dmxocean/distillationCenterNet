# -*- coding: utf-8 -*-
"""
Checkpoint evaluation runner

Loads a detector checkpoint and reports the per class detection quality on the validation split,
the flat metrics are printed as json for quick inspection
"""

import argparse
import json

import torch

from config.resolve import resolve
from data.streams import build_test_loader, build_val_loader
from engine.checkpoint import load_checkpoint
from evaluation.metrics import evaluate, flat_metrics
from models.centernet import CenterNet


def main(argv=None):
    """Load a checkpoint and print the metrics for the chosen split as json"""
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the val or test split")
    parser.add_argument("--config", required=True, help="Experiment or component config path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate")
    parser.add_argument("--width", type=float, default=None, help="Backbone width of the checkpoint")
    parser.add_argument("--split", default="test", choices=("val", "test"), help="Split to score")
    args = parser.parse_args(argv)

    config = resolve(args.config, {})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    width = args.width if args.width is not None else config.get("student_width", 1.0)

    model = CenterNet(
        config["num_classes"], width, pretrained=False,
        img_size=config.get("img_size", 640), stride=config.get("stride", 4),
    ).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    loader = build_test_loader(config) if args.split == "test" else build_val_loader(config)
    quality = evaluate(model, loader, device, config["num_classes"], config.get("stride", 4), eval_cfg=config.get("eval"))
    print(json.dumps(flat_metrics(quality), indent=2))


if __name__ == "__main__":
    main()
