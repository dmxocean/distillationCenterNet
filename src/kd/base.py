# -*- coding: utf-8 -*-
"""
Distillation method contract for offline knowledge distillation

A method advertises the teacher and student signals it consumes and returns a structured loss
output so the engine can compose and log terms uniformly

IMPORTANT: compute never reads ground truth, so every method stays valid on unlabelled batches
where the teacher provides the only supervision
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple

import torch


class SignalSpec(Enum):
    """
    Teacher and student signals a method may request
    """

    LOGITS = "logits"  # Pre sigmoid heatmap logits per class
    GEOMETRY = "geometry"  # Width and height regression maps
    FEATURES = "features"  # Intermediate decoder activations
    ATTENTION = "attention"  # Attention gate maps from the decoder


@dataclass
class MethodOutput:
    """
    Structured distillation loss with named sub terms for logging
    """

    total: torch.Tensor
    terms: Dict[str, torch.Tensor] = field(default_factory=dict)


class DistillationMethod(ABC):
    """
    Abstract distillation method consumed by the training engine

    A concrete method declares the signals it needs and computes a teacher driven loss that is
    deliberately ground truth free, so the same loss trains on labelled and unlabelled images alike
    """

    @abstractmethod
    def required_signals(self) -> Tuple[SignalSpec, ...]:
        """Report the signals the engine must extract for this method"""

    @abstractmethod
    def compute(self, student_out, teacher_out, targets, student_signals=None, teacher_signals=None) -> MethodOutput:
        """
        Compute the distillation loss from aligned student and teacher outputs

        Args:
            student_out: Student forward outputs holding the requested signals
            teacher_out: Frozen teacher outputs holding the requested signals
            targets: Optional ground truth, ignored by teacher driven terms, may be None
            student_signals: Captured student stage tensors, passed when the method hooks the neck
            teacher_signals: Captured teacher stage tensors, passed when the method hooks the neck

        Returns:
            MethodOutput: Total distillation loss and its named components
        """
