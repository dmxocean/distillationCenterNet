# -*- coding: utf-8 -*-
"""
Patience based early stopping on the validation score

The detector is scored on a metric where higher is better, training stops when the score fails to
improve by a minimum delta for a number of consecutive epochs
"""


class EarlyStopping:
    """
    Track the best validation score and signal when to stop

    Args:
        patience: Number of epochs without improvement before stopping
        min_delta: Minimum increase that counts as an improvement
    """

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("-inf")
        self.counter = 0

    def step(self, score: float) -> bool:
        """
        Update the tracker with a new score and report whether it improved

        Args:
            score: Validation score for the epoch, higher is better

        Returns:
            True when the score improved over the best by at least the minimum delta
        """
        if score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
            return True
        self.counter += 1
        return False

    @property
    def should_stop(self) -> bool:
        """Whether the patience has been exhausted"""
        return self.counter >= self.patience
