# -*- coding: utf-8 -*-
"""
Experiment runner that executes several student experiments in sequence

Each experiment is resolved independently and written under its own method and regime directory, so
a batch produces the same per run artifacts as a single experiment without overwriting any run
"""

import argparse
import os

from config.resolve import resolve
from engine import distiller
from tracking.run_logger import RunLogger


def main(argv=None):
    """Resolve and run every passed experiment config under the output root"""
    parser = argparse.ArgumentParser(description="Run several student experiments in sequence")
    parser.add_argument("--configs", nargs="+", required=True, help="Experiment config paths")
    parser.add_argument("--out-root", dest="out_root", default="results/students", help="Output root for the runs")
    parser.add_argument("--wandb-mode", dest="wandb_mode", default=None, help="online, offline, or disabled")
    args = parser.parse_args(argv)

    for config_path in args.configs:
        config = resolve(config_path, {"wandb_mode": args.wandb_mode})
        method = config["method"]["name"]
        regime = config.get("regime", "labelled")
        out = os.path.join(args.out_root, method, regime)
        run_logger = RunLogger(out, group=method, name=regime, config=config)
        distiller.fit(config, run_logger)


if __name__ == "__main__":
    main()
