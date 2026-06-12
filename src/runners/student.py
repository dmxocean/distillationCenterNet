# -*- coding: utf-8 -*-
"""
Student distillation runner

The student trains at the student width with the configured distillation method on the labelled or
the semi supervised stream, the W&B run is grouped by the method and named by the regime
"""

from config.resolve import resolve
from engine import distiller
from runners.common import build_parser, overrides
from tracking.run_logger import RunLogger


def main(argv=None):
    """Resolve the config and run the student distillation to completion"""
    args = build_parser("Distil a student from the frozen teacher").parse_args(argv)
    config = resolve(args.config, overrides(args))
    method = config["method"]["name"]
    regime = config.get("regime", "labelled")
    run_logger = RunLogger(args.out, group=method, name=regime, config=config)
    distiller.fit(config, run_logger, resume=args.resume)


if __name__ == "__main__":
    main()
