# -*- coding: utf-8 -*-
"""
Shared command line plumbing for the runners

Every runner accepts the same config and output arguments plus a small set of overrides that win
over the resolved configuration, so a short smoke run differs from a full run by command line
flags alone
"""

import argparse

_OVERRIDE_KEYS = ("wandb_mode", "epochs", "batch_size", "max_train_samples", "max_unlabeled", "num_workers")


def build_parser(description: str) -> argparse.ArgumentParser:
    """Build the common argument parser shared by every runner"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Experiment or component config path")
    parser.add_argument("--out", required=True, help="Run output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt in --out if present")
    parser.add_argument("--wandb-mode", dest="wandb_mode", default=None, help="online, offline, or disabled")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--max-train-samples", dest="max_train_samples", type=int, default=None)
    parser.add_argument("--max-unlabeled", dest="max_unlabeled", type=int, default=None)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    return parser


def overrides(args) -> dict:
    """Collect the command line overrides that take precedence over the config"""
    return {key: getattr(args, key) for key in _OVERRIDE_KEYS}
