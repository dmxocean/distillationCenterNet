# -*- coding: utf-8 -*-
"""
Teacher training runner

The teacher trains full width on the labelled split only, the resulting best checkpoint is frozen
and consumed by every student
"""

from config.resolve import resolve
from engine import trainer
from runners.common import build_parser, overrides
from tracking.run_logger import RunLogger


def main(argv=None):
    """Resolve the config and run the teacher training to completion"""
    args = build_parser("Train the teacher on labelled data").parse_args(argv)
    config = resolve(args.config, overrides(args))
    run_logger = RunLogger(args.out, group="teacher", name="teacher", config=config)
    trainer.fit(config, run_logger, role="teacher", width=config.get("width", 1.0), resume=args.resume)


if __name__ == "__main__":
    main()
