"""Validation-driven learning-rate decay and early stopping controls."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PlateauDecision:
    significant_improvement: bool
    reduce_learning_rate: bool
    should_stop: bool
    validations_without_improvement: int
    validations_since_lr_reduction: int
    reference_metric: float


class ValidationPlateauController:
    """Track significant validation improvements without inspecting train loss."""

    def __init__(
        self,
        min_delta=0.003,
        early_stopping_patience=10,
        min_epochs=20,
        lr_patience=5,
        initial_metric=float("-inf"),
    ):
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        if early_stopping_patience < 0 or lr_patience < 0:
            raise ValueError("patience values must be non-negative")
        if min_epochs < 0:
            raise ValueError("min_epochs must be non-negative")
        if not math.isfinite(initial_metric) and initial_metric != float("-inf"):
            raise ValueError("initial_metric must be finite or -inf")
        self.min_delta = float(min_delta)
        self.early_stopping_patience = int(early_stopping_patience)
        self.min_epochs = int(min_epochs)
        self.lr_patience = int(lr_patience)
        self.reference_metric = float(initial_metric)
        self.validations_without_improvement = 0
        self.validations_since_lr_reduction = 0

    def update(self, metric, completed_epochs):
        metric = float(metric)
        if not math.isfinite(metric):
            raise ValueError("validation metric must be finite")
        if completed_epochs <= 0:
            raise ValueError("completed_epochs must be positive")

        improved = metric > self.reference_metric + self.min_delta
        if improved:
            self.reference_metric = metric
            self.validations_without_improvement = 0
            self.validations_since_lr_reduction = 0
        else:
            self.validations_without_improvement += 1
            self.validations_since_lr_reduction += 1

        reduce_lr = (
            self.lr_patience > 0
            and self.validations_since_lr_reduction >= self.lr_patience
        )
        if reduce_lr:
            self.validations_since_lr_reduction = 0

        should_stop = (
            self.early_stopping_patience > 0
            and completed_epochs >= self.min_epochs
            and self.validations_without_improvement
            >= self.early_stopping_patience
        )
        return PlateauDecision(
            significant_improvement=improved,
            reduce_learning_rate=reduce_lr,
            should_stop=should_stop,
            validations_without_improvement=(
                self.validations_without_improvement
            ),
            validations_since_lr_reduction=(
                self.validations_since_lr_reduction
            ),
            reference_metric=self.reference_metric,
        )
