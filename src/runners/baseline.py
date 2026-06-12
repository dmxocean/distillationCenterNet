# -*- coding: utf-8 -*-
"""
Baseline training runner

The baseline trains at the student width on the labelled split with no distillation, it is the
fair comparison point the students are measured against
"""

from config.resolve import resolve
from engine import trainer
from runners.common import build_parser, overrides
from tracking.run_logger import RunLogger


def main(argv=None):
    """Resolve the config and run the supervised baseline to completion"""
    args = build_parser("Train the supervised baseline at the student width").parse_args(argv)
    config = resolve(args.config, overrides(args))
    run_logger = RunLogger(args.out, group="baseline", name="baseline", config=config)
    trainer.fit(config, run_logger, role="baseline", width=config.get("student_width", 0.35), resume=args.resume)


if __name__ == "__main__":
    main()
