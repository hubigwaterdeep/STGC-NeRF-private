"""Parameter-budget contracts for fair LiDAR4D scene-field comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class FieldBudgetSpec:
    """Executable fairness contract for every replaceable field sub-budget."""

    reference: "FieldBudgetReport"
    relative_tolerance: float = 0.01
    roles: tuple[str, ...] = ("static", "dynamic", "flow", "total")

    def __post_init__(self) -> None:
        if not 0 <= self.relative_tolerance < 1:
            raise ValueError("budget tolerance must be in [0, 1)")
        unknown = set(self.roles) - {"static", "dynamic", "flow", "other", "total"}
        if unknown:
            raise ValueError(f"unknown field-budget roles: {sorted(unknown)}")

    def bounds(self, role: str) -> tuple[float, float]:
        if role not in self.roles:
            raise ValueError(f"role {role!r} is not governed by this budget")
        expected = getattr(self.reference, role)
        delta = self.relative_tolerance * expected
        return expected - delta, expected + delta

    def assert_satisfied(self, report: "FieldBudgetReport") -> None:
        mismatches = []
        for role in self.roles:
            actual = getattr(report, role)
            expected = getattr(self.reference, role)
            lower, upper = self.bounds(role)
            if not lower <= actual <= upper:
                relative_error = abs(actual - expected) / max(expected, 1)
                mismatches.append(
                    f"{role}={actual:,} (expected {expected:,}, "
                    f"error {relative_error:.2%})"
                )
        if mismatches:
            raise ValueError(
                f"{report.name} does not match {self.reference.name}: "
                + "; ".join(mismatches)
            )


@dataclass(frozen=True)
class FieldBudgetReport:
    """Trainable parameters grouped by the role they play in a scene field."""

    name: str
    static: int
    dynamic: int
    flow: int
    other: int = 0

    @property
    def total(self) -> int:
        return self.static + self.dynamic + self.flow + self.other

    def ratio(self, reference: "FieldBudgetReport | None" = None) -> float:
        reference = reference or OFFICIAL_FIELD_BUDGET
        return self.total / reference.total

    def assert_matches(
        self,
        reference: "FieldBudgetReport | None" = None,
        *,
        tolerance: float = 0.01,
    ) -> None:
        """Reject hidden sub-budget changes even when the totals match."""
        reference = reference or OFFICIAL_FIELD_BUDGET
        FieldBudgetSpec(reference, tolerance).assert_satisfied(self)

    def format(self, reference: "FieldBudgetReport | None" = None) -> str:
        reference = reference or OFFICIAL_FIELD_BUDGET
        return (
            f"static={self.static:,} | dynamic={self.dynamic:,} | "
            f"flow={self.flow:,} | other={self.other:,} | total={self.total:,} "
            f"({self.ratio(reference):.4f}x {reference.name})"
        )


OFFICIAL_FIELD_BUDGET = FieldBudgetReport(
    name="official",
    static=18_866_176,
    dynamic=12_675_072,
    flow=14_947_712,
)

MATCHED_FIELD_BUDGET_SPEC = FieldBudgetSpec(OFFICIAL_FIELD_BUDGET)


def count_parameters(
    value: nn.Module | Iterable[nn.Parameter | Tensor],
) -> int:
    """Count trainable parameters without exposing module internals to callers."""
    parameters = value.parameters() if isinstance(value, nn.Module) else value
    return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
