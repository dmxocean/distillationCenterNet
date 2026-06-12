# -*- coding: utf-8 -*-
"""
Weights and Biases writer mirroring the run scalars to the cloud

The writer logs the same flat scalar keys that the local metrics file holds, the run is grouped by
the comparison and named by the regime so the cloud view mirrors the disk tree

The disabled mode skips initialisation entirely, so a run still writes the full local tree when the
cloud is turned off
"""

import logging

logger = logging.getLogger(__name__)


class WandbWriter:
    """
    Cloud writer that logs flat scalars to Weights and Biases

    Args:
        project: W&B project name
        entity: W&B entity, may be empty to use the default
        group: Run group, the comparison the run belongs to
        name: Run name within the group
        mode: online, offline, or disabled
        config: Resolved config recorded with the run
    """

    def __init__(self, project, entity, group, name, mode, config):
        self.run = None
        if mode == "disabled":
            return
        try:
            import wandb

            self.wandb = wandb
            self.run = wandb.init(
                project=project or None,
                entity=entity or None,
                group=group,
                name=name,
                mode=mode,
                config=config,
                reinit=True,
                settings=wandb.Settings(disable_git=True, save_code=False),
            )
        except Exception:  # A logging backend must never break a run
            logger.warning("W&B initialisation failed, continuing with local logging only")
            self.run = None

    def log(self, scalars: dict, step=None):
        """Log a flat dict of scalars at the given step"""
        if self.run is not None:
            self.wandb.log(scalars, step=step)

    def summary(self, flat: dict):
        """Record the run level summary fields so cross run plots key off the cloud run"""
        if self.run is not None:
            self.run.summary.update(flat)

    def finish(self):
        """Close the W&B run"""
        if self.run is not None:
            self.wandb.finish()
