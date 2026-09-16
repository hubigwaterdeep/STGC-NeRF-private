"""Learned routing for a frozen-Basis geometry residual.

The module accepts one complete frozen-Basis range panorama.  Absolute range
is confined to an ordered, learnable distance envelope; the contextual branch
only receives dimensionless neighborhood statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class LearnedResidualGateReadout:
    """Observable routing values returned at the module seam."""

    weight: Tensor
    distance_envelope: Tensor
    context_reliability: Tensor
    training_logit: Tensor
    transition_start_m: Tensor
    transition_end_m: Tensor


class LearnedResidualRangeGate(nn.Module):
    """Attenuate a residual using learned range support and local reliability.

    The only runtime input is a complete ``[B, H, W]`` expected-range panorama
    rendered by the frozen Basis.  It is detached internally, so this module
    cannot update the Basis through its routing features.
    """

    def __init__(
        self,
        *,
        physical_min_range_m: float,
        physical_max_range_m: float,
        context_hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        if not physical_max_range_m > physical_min_range_m:
            raise ValueError("physical maximum range must exceed the minimum")
        if context_hidden_dim < 1:
            raise ValueError("context hidden width must be positive")
        self.register_buffer(
            "physical_min_range_m", torch.tensor(float(physical_min_range_m))
        )
        self.register_buffer(
            "physical_max_range_m", torch.tensor(float(physical_max_range_m))
        )
        self.transition_start_logit = nn.Parameter(torch.tensor(0.0))
        self.transition_width_logit = nn.Parameter(torch.tensor(0.0))
        self.context_network = nn.Sequential(
            nn.Linear(5, context_hidden_dim),
            nn.SiLU(),
            nn.Linear(context_hidden_dim, 1),
        )

    def _transition_bounds_m(self) -> tuple[Tensor, Tensor]:
        physical_min = self.physical_min_range_m
        physical_max = self.physical_max_range_m
        physical_span = physical_max - physical_min
        numerical_gap = physical_span * torch.finfo(physical_span.dtype).eps**0.5
        start = physical_min + (physical_span - numerical_gap) * torch.sigmoid(
            self.transition_start_logit
        )
        remaining_after_gap = physical_max - start - numerical_gap
        end = (
            start
            + numerical_gap
            + remaining_after_gap * torch.sigmoid(self.transition_width_logit)
        )
        return start, end

    def transition_bounds_m(self) -> tuple[Tensor, Tensor]:
        """Return the current learned ordered transition in physical meters."""

        return self._transition_bounds_m()

    @staticmethod
    def _local_statistics(
        range_view_m: Tensor, physical_span_m: Tensor
    ) -> Tensor:
        panorama = range_view_m.unsqueeze(1)
        padded = F.pad(panorama, (1, 1, 0, 0), mode="circular")
        padded = F.pad(padded, (0, 0, 1, 1), mode="replicate")
        patches = F.unfold(padded, kernel_size=3).transpose(1, 2)
        center = patches[..., 4:5]
        neighbors = torch.cat((patches[..., :4], patches[..., 5:]), dim=-1)
        normalized_delta = (neighbors - center) / physical_span_m
        absolute_delta = normalized_delta.abs()
        horizontal_contrast = absolute_delta[..., [3, 4]].mean(
            dim=-1, keepdim=True
        )
        vertical_contrast = absolute_delta[..., [1, 6]].mean(
            dim=-1, keepdim=True
        )
        diagonal_contrast = absolute_delta[..., [0, 2, 5, 7]].mean(
            dim=-1, keepdim=True
        )
        return torch.cat(
            (
                normalized_delta.mean(dim=-1, keepdim=True),
                absolute_delta.mean(dim=-1, keepdim=True),
                horizontal_contrast,
                vertical_contrast,
                diagonal_contrast,
            ),
            dim=-1,
        )

    def forward(self, frozen_basis_range_view_m: Tensor) -> LearnedResidualGateReadout:
        if frozen_basis_range_view_m.ndim != 3:
            raise ValueError("frozen Basis range view must have shape [B, H, W]")
        if not torch.is_floating_point(frozen_basis_range_view_m):
            raise ValueError("frozen Basis range view must be floating-point")
        if (
            frozen_basis_range_view_m.shape[1] < 2
            or frozen_basis_range_view_m.shape[2] < 3
        ):
            raise ValueError(
                "frozen Basis range view must be a complete local panorama"
            )
        if not torch.isfinite(frozen_basis_range_view_m.float()).all():
            raise ValueError("frozen Basis range view must be finite")
        if torch.any(frozen_basis_range_view_m < 0.0):
            raise ValueError("frozen Basis range view must be nonnegative")

        source_dtype = frozen_basis_range_view_m.dtype
        range_view_m = frozen_basis_range_view_m.detach().to(
            dtype=self.transition_start_logit.dtype
        )
        start_m, end_m = self._transition_bounds_m()
        position = ((range_view_m - start_m) / (end_m - start_m)).clamp(0.0, 1.0)
        envelope = position.square() * (3.0 - 2.0 * position)

        batch, height, width = range_view_m.shape
        physical_span_m = self.physical_max_range_m - self.physical_min_range_m
        statistics = self._local_statistics(range_view_m, physical_span_m)
        context_logit = self.context_network(statistics).reshape(
            batch, height, width
        )
        reliability = torch.sigmoid(context_logit)
        weight = envelope * reliability
        transition_midpoint_m = 0.5 * (start_m + end_m)
        transition_half_width_m = 0.5 * (end_m - start_m)
        soft_distance_logit = (
            range_view_m - transition_midpoint_m
        ) / transition_half_width_m
        training_logit = -torch.logsumexp(
            torch.stack(
                (
                    -soft_distance_logit,
                    -context_logit,
                    -soft_distance_logit - context_logit,
                )
            ),
            dim=0,
        )
        return LearnedResidualGateReadout(
            weight=weight.to(source_dtype),
            distance_envelope=envelope.to(source_dtype),
            context_reliability=reliability.to(source_dtype),
            training_logit=training_logit,
            transition_start_m=start_m.to(source_dtype),
            transition_end_m=end_m.to(source_dtype),
        )
