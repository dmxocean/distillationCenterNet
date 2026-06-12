# -*- coding: utf-8 -*-
"""
Checkpoint saving and loading for the detector

A checkpoint stores the model state, the optimiser state, the epoch, and the best validation
score, so a run resumes cleanly and a student loads the frozen teacher from the same format

IMPORTANT: the teacher loader reads the model key from this format, the best checkpoint under a
run is the one students distil from
"""

import os

import torch


def save_checkpoint(path: str, model, optimizer, epoch: int, best_score: float):
    """
    Save a checkpoint holding the model, optimiser, epoch, and best score

    Args:
        path: Destination file path, parent directories are created if needed
        model: Detector whose state is saved
        optimizer: Optimiser whose state is saved for resuming
        epoch: Epoch index reached at save time
        best_score: Best validation score observed so far
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best": best_score,
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer=None, map_location="cpu"):
    """
    Restore model and optional optimiser state from a checkpoint

    Args:
        path: Source checkpoint file path
        model: Detector to restore in place
        optimizer: Optional optimiser to restore in place
        map_location: Device mapping for the loaded tensors

    Returns:
        The loaded checkpoint dictionary
    """
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state
