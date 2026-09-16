"""Independent objectives for fitting the detached geometry residual basis."""

from __future__ import annotations

from typing import Any, NamedTuple

import torch
import torch.nn as nn
from torch import Tensor


LEGACY_GEOMETRY_RESIDUAL_OBJECTIVE = "legacy"
TAIL_ALIGNED_SAME_RAY_V1 = "tail_aligned_same_ray_v1"
GEOMETRY_RESIDUAL_OBJECTIVE_PRESETS = (
    LEGACY_GEOMETRY_RESIDUAL_OBJECTIVE,
    TAIL_ALIGNED_SAME_RAY_V1,
)

_MAX_POINTS = 512
_UNIFORM_POINTS = 256
_JUMP_POINTS = 64
_FAR_POINTS = 64
_FROZEN_BASE_ERROR_POINTS = 64
_TAIL_FILL_POINTS = 64
_BODY_WEIGHT = 0.5
_TAIL_WEIGHT = 0.5
_BOUNDED_SQUARE_DENOMINATOR_M2 = 25.0

_TAIL_ALIGNED_SAME_RAY_V1_CONTRACT: dict[str, Any] = {
    "preset": TAIL_ALIGNED_SAME_RAY_V1,
    "max_points": _MAX_POINTS,
    "uniform_points": _UNIFORM_POINTS,
    "jump_points": _JUMP_POINTS,
    "far_points": _FAR_POINTS,
    "frozen_base_error_points": _FROZEN_BASE_ERROR_POINTS,
    "tail_fill_points": _TAIL_FILL_POINTS,
    "body_weight": _BODY_WEIGHT,
    "tail_weight": _TAIL_WEIGHT,
    "bounded_square_denominator_m2": _BOUNDED_SQUARE_DENOMINATOR_M2,
}


class TailAlignedSelection(NamedTuple):
    """Disjoint deterministic ray masks owned by the tail-aligned preset."""

    body: Tensor
    jump: Tensor
    far: Tensor
    frozen_base_error: Tensor
    tail_fill: Tensor

    @property
    def tail(self) -> Tensor:
        return self.jump | self.far | self.frozen_base_error | self.tail_fill

    @property
    def selected(self) -> Tensor:
        return self.body | self.tail


class GeometryResidualTailTerms(NamedTuple):
    """Auditable body/tail decomposition of the residual-only objective."""

    total: Tensor
    body: Tensor
    tail: Tensor
    selection: TailAlignedSelection


def geometry_residual_objective_preset_contract(name: str) -> dict[str, Any]:
    """Return the complete immutable settings for a residual-only objective."""

    if name != TAIL_ALIGNED_SAME_RAY_V1:
        raise ValueError(f"unknown geometry residual objective preset {name!r}")
    return dict(_TAIL_ALIGNED_SAME_RAY_V1_CONTRACT)


def _uniform_subset(indices: Tensor, limit: int) -> Tensor:
    if indices.numel() <= limit:
        return indices
    positions = torch.linspace(
        0,
        indices.numel() - 1,
        limit,
        device=indices.device,
    ).long()
    return indices[positions]


def _ranked_indices(score: Tensor, eligible: Tensor) -> Tensor:
    indices = torch.nonzero(eligible, as_tuple=False).flatten()
    if indices.numel() == 0:
        return indices
    order = torch.argsort(
        score[indices],
        descending=True,
        stable=True,
    )
    return indices[order]


def _take_ranked(
    destination: Tensor,
    score: Tensor,
    support: Tensor,
    occupied: Tensor,
    limit: int,
    *,
    positive_only: bool = False,
) -> None:
    eligible = support & ~occupied
    if positive_only:
        eligible = eligible & (score > 0.0)
    ranked = _ranked_indices(score, eligible)
    chosen = ranked[:limit]
    destination[chosen] = True
    occupied[chosen] = True


def _fill_tail_round_robin(
    destination: Tensor,
    rankings: tuple[Tensor, ...],
    occupied: Tensor,
    needed: int,
) -> None:
    cursors = [0] * len(rankings)
    remaining = int(needed)
    while remaining > 0:
        progress = False
        for stream, ranked in enumerate(rankings):
            cursor = cursors[stream]
            while cursor < ranked.numel() and occupied[ranked[cursor]]:
                cursor += 1
            cursors[stream] = cursor
            if cursor >= ranked.numel():
                continue
            index = ranked[cursor]
            cursors[stream] += 1
            destination[index] = True
            occupied[index] = True
            remaining -= 1
            progress = True
            if remaining == 0:
                break
        if not progress:
            break


def _validate_inputs(*values: Tensor) -> None:
    reference = values[0]
    if reference.ndim != 2:
        raise ValueError("tail-aligned ray tensors must have shape [B, N]")
    if any(value.shape != reference.shape for value in values[1:]):
        raise ValueError("tail-aligned ray tensors must share shape [B, N]")
    if any(not torch.isfinite(value.float()).all() for value in values):
        raise ValueError("tail-aligned ray tensors must be finite")


def select_tail_aligned_support(
    target_return: Tensor,
    target_depth_m: Tensor,
    frozen_base_depth_m: Tensor,
    range_jump_score_m: Tensor,
) -> TailAlignedSelection:
    """Select a uniform body plus disjoint jump, far and base-error tails."""

    _validate_inputs(
        target_return,
        target_depth_m,
        frozen_base_depth_m,
        range_jump_score_m,
    )
    detached_target = target_depth_m.detach()
    detached_base = frozen_base_depth_m.detach()
    detached_jump = range_jump_score_m.detach()
    support = target_return.detach() > 0.5
    masks = [torch.zeros_like(support) for _ in range(5)]
    body, jump, far, frozen_base_error, tail_fill = masks

    for batch in range(support.shape[0]):
        support_indices = torch.nonzero(
            support[batch], as_tuple=False
        ).flatten()
        body_indices = _uniform_subset(support_indices, _UNIFORM_POINTS)
        body[batch, body_indices] = True
        occupied = body[batch].clone()

        jump_score = detached_jump[batch]
        far_score = detached_target[batch]
        base_error_score = (detached_base[batch] - detached_target[batch]).abs()
        _take_ranked(
            jump[batch],
            jump_score,
            support[batch],
            occupied,
            _JUMP_POINTS,
            positive_only=True,
        )
        _take_ranked(
            far[batch],
            far_score,
            support[batch],
            occupied,
            _FAR_POINTS,
        )
        _take_ranked(
            frozen_base_error[batch],
            base_error_score,
            support[batch],
            occupied,
            _FROZEN_BASE_ERROR_POINTS,
        )

        tail_target = min(
            _JUMP_POINTS
            + _FAR_POINTS
            + _FROZEN_BASE_ERROR_POINTS
            + _TAIL_FILL_POINTS,
            max(0, support_indices.numel() - body_indices.numel()),
        )
        selected_tail = int(
            jump[batch].sum()
            + far[batch].sum()
            + frozen_base_error[batch].sum()
        )
        fill_needed = tail_target - selected_tail
        rankings = (
            _ranked_indices(jump_score, support[batch] & (jump_score > 0.0)),
            _ranked_indices(far_score, support[batch]),
            _ranked_indices(base_error_score, support[batch]),
        )
        _fill_tail_round_robin(
            tail_fill[batch],
            rankings,
            occupied,
            fill_needed,
        )

        # The far and base rankings span the complete support, so this fallback
        # is reached only for unusual tensor backends with incomplete sorting.
        shortfall = tail_target - int(
            jump[batch].sum()
            + far[batch].sum()
            + frozen_base_error[batch].sum()
            + tail_fill[batch].sum()
        )
        if shortfall > 0:
            remaining = torch.nonzero(
                support[batch] & ~occupied, as_tuple=False
            ).flatten()[:shortfall]
            tail_fill[batch, remaining] = True

    return TailAlignedSelection(*masks)


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    if mask.any():
        return value[mask].mean()
    return value.sum() * 0.0


class TailAlignedSameRayObjective(nn.Module):
    """Balanced bounded-square depth risk on uniform and hard GT-return rays."""

    def forward(
        self,
        pred_depth_m: Tensor,
        target_depth_m: Tensor,
        target_return: Tensor,
        frozen_base_depth_m: Tensor,
        range_jump_score_m: Tensor,
    ) -> GeometryResidualTailTerms:
        _validate_inputs(
            pred_depth_m,
            target_depth_m,
            target_return,
            frozen_base_depth_m,
            range_jump_score_m,
        )
        selection = select_tail_aligned_support(
            target_return,
            target_depth_m,
            frozen_base_depth_m.detach(),
            range_jump_score_m,
        )
        error_squared = (pred_depth_m - target_depth_m.detach()).square()
        bounded_square = error_squared / (
            _BOUNDED_SQUARE_DENOMINATOR_M2 + error_squared
        )
        body = _masked_mean(bounded_square, selection.body)
        tail = _masked_mean(bounded_square, selection.tail)
        total = _BODY_WEIGHT * body + _TAIL_WEIGHT * tail
        return GeometryResidualTailTerms(total, body, tail, selection)


def geometry_residual_objective_from_preset(
    name: str,
) -> TailAlignedSameRayObjective:
    """Build the one registered residual-only objective."""

    geometry_residual_objective_preset_contract(name)
    return TailAlignedSameRayObjective()
