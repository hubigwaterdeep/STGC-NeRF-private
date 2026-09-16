"""Visibility-preserving scalar density redistribution for geometry fitting.

The module owns an independent fine scene-space basis.  Its learned scalar
field redistributes optical density along each ray while preserving total
optical mass, so it can move expected depth without changing terminal opacity.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from best_core.geometry_residual_basis import _sample_axis
from best_core.learned_residual_gate import LearnedResidualRangeGate


_HALF_LOG_TWO = 0.5 * math.log(2.0)
FAR_RANGE_GATE_NEAR_ZERO_M = 35.0
FAR_RANGE_GATE_FAR_FULL_M = 40.0
CONSENSUS_GATE_NEAR_ZERO = 0.20
CONSENSUS_GATE_FAR_FULL = 0.60


def spatial_consensus_gate_contract() -> dict[str, float | str]:
    """Return the fixed, readable v6.2 residual-routing contract."""

    return {
        "distance_source": "frozen_base_expected_depth_m",
        "range_near_zero_m": FAR_RANGE_GATE_NEAR_ZERO_M,
        "range_far_full_m": FAR_RANGE_GATE_FAR_FULL_M,
        "consensus_source": "complete_frozen_base_range_view_expected_depth_m",
        "consensus_neighborhood": "3x3_without_center",
        "consensus_far_vote_threshold_m": FAR_RANGE_GATE_FAR_FULL_M,
        "consensus_horizontal_boundary": "circular",
        "consensus_vertical_boundary": "replicate",
        "consensus_near_zero_fraction": CONSENSUS_GATE_NEAR_ZERO,
        "consensus_far_full_fraction": CONSENSUS_GATE_FAR_FULL,
        "interpolation": "cubic_smoothstep",
        "renderer_kwarg": "frozen_basis_consensus_gate",
    }


class FrozenBaseRangeViewConsensusGate(nn.Module):
    """Turn a complete frozen-Basis depth panorama into a coherence gate."""

    def forward(self, base_expected_depth_m: Tensor) -> Tensor:
        if base_expected_depth_m.ndim != 3:
            raise ValueError("frozen base depth panorama must have shape [B, H, W]")
        if not torch.isfinite(base_expected_depth_m.float()).all():
            raise ValueError("frozen base depth panorama must be finite")
        far = (
            base_expected_depth_m.float() >= FAR_RANGE_GATE_FAR_FULL_M
        ).float().unsqueeze(1)
        padded = F.pad(far, (1, 1, 0, 0), mode="circular")
        padded = F.pad(padded, (0, 0, 1, 1), mode="replicate")
        kernel = far.new_ones((1, 1, 3, 3))
        kernel[..., 1, 1] = 0.0
        neighbor_fraction = F.conv2d(padded, kernel).squeeze(1) / 8.0
        position = (
            (neighbor_fraction - CONSENSUS_GATE_NEAR_ZERO)
            / (CONSENSUS_GATE_FAR_FULL - CONSENSUS_GATE_NEAR_ZERO)
        ).clamp(0.0, 1.0)
        weight = position.square() * (3.0 - 2.0 * position)
        return weight.to(base_expected_depth_m.dtype)


class FrozenBasisSpatialConsensusProvider:
    """Cache row-major consensus gates produced from complete base panoramas."""

    def __init__(
        self,
        model: nn.Module,
        *,
        depth_scale: float,
        render_kwargs: Mapping[str, object],
        max_ray_batch: int = 4096,
    ) -> None:
        if depth_scale <= 0.0:
            raise ValueError("spatial consensus depth scale must be positive")
        if max_ray_batch < 1:
            raise ValueError("spatial consensus render batch must be positive")
        self.model = model
        self.depth_scale = float(depth_scale)
        self.render_kwargs = dict(render_kwargs)
        self.max_ray_batch = int(max_ray_batch)
        self.consensus_gate = FrozenBaseRangeViewConsensusGate()
        self._gates_by_frame: dict[int, Tensor] = {}

    def prime_batch(self, full_panorama: Mapping[str, object]) -> None:
        required = (
            "rays_o_lidar",
            "rays_d_lidar",
            "time",
            "frame_id",
            "H_lidar",
            "W_lidar",
        )
        missing = [name for name in required if name not in full_panorama]
        if missing:
            raise RuntimeError(
                f"spatial consensus requires complete panorama values {missing}"
            )
        rays_o = full_panorama["rays_o_lidar"]
        rays_d = full_panorama["rays_d_lidar"]
        time = full_panorama["time"]
        frame_id = full_panorama["frame_id"]
        if not all(
            isinstance(value, Tensor)
            for value in (rays_o, rays_d, time, frame_id)
        ):
            raise RuntimeError("spatial consensus panorama values must be tensors")
        assert isinstance(rays_o, Tensor)
        assert isinstance(rays_d, Tensor)
        assert isinstance(time, Tensor)
        assert isinstance(frame_id, Tensor)
        height = int(full_panorama["H_lidar"])
        width = int(full_panorama["W_lidar"])
        if rays_o.ndim != 3 or rays_o.shape != rays_d.shape:
            raise RuntimeError("spatial consensus rays must have shape [B, H*W, 3]")
        batch, ray_count = rays_o.shape[:2]
        if rays_o.shape[-1] != 3 or ray_count != height * width:
            raise RuntimeError("spatial consensus requires a complete row-major panorama")
        if frame_id.numel() != batch or time.shape[0] != batch:
            raise RuntimeError("spatial consensus frame/time batch does not match rays")
        frame_ids = tuple(int(value) for value in frame_id.reshape(-1).tolist())
        if all(value in self._gates_by_frame for value in frame_ids):
            return

        render_kwargs = dict(self.render_kwargs)
        for fixed_name in (
            "staged",
            "max_ray_batch",
            "perturb",
            "frozen_basis_consensus_gate",
        ):
            render_kwargs.pop(fixed_name, None)
        zero_consensus = rays_o.new_zeros((batch, ray_count))
        training_modes = tuple(
            (module, module.training) for module in self.model.modules()
        )
        self.model.eval()
        try:
            with torch.no_grad():
                outputs = self.model.render(
                    rays_o,
                    rays_d,
                    time,
                    staged=True,
                    max_ray_batch=self.max_ray_batch,
                    perturb=False,
                    frozen_basis_consensus_gate=zero_consensus,
                    **render_kwargs,
                )
        finally:
            for module, training in training_modes:
                module.training = training
        base_depth = outputs.get("base_depth_lidar")
        if not isinstance(base_depth, Tensor) or base_depth.shape != (batch, ray_count):
            raise RuntimeError(
                "base-only spatial consensus render must return aligned base depth"
            )
        depth_m = base_depth.float().reshape(batch, height, width) / self.depth_scale
        gates = self.consensus_gate(depth_m).reshape(batch, ray_count).detach().cpu()
        for batch_index, value in enumerate(frame_ids):
            self._gates_by_frame[value] = gates[batch_index]

    def aligned_gate(
        self,
        frame_id: Tensor,
        ray_indices: Tensor,
        *,
        like: Tensor,
    ) -> Tensor:
        if frame_id.ndim == 0:
            frame_id = frame_id.reshape(1)
        if ray_indices.ndim != 2 or frame_id.numel() != ray_indices.shape[0]:
            raise RuntimeError(
                "spatial consensus frame IDs and row-major ray indices are misaligned"
            )
        selected = []
        for batch_index, value in enumerate(frame_id.reshape(-1).tolist()):
            cached = self._gates_by_frame.get(int(value))
            if cached is None:
                raise RuntimeError(
                    f"spatial consensus frame {int(value)} has not been primed"
                )
            indices = ray_indices[batch_index].detach().cpu().long()
            if torch.any(indices < 0) or torch.any(indices >= cached.numel()):
                raise RuntimeError("spatial consensus row-major ray index is out of range")
            selected.append(cached[indices])
        return torch.stack(selected).to(device=like.device, dtype=like.dtype)


def far_range_gate_contract() -> dict[str, float | str]:
    """Return the fixed, readable v6.1 range-gate contract."""

    return {
        "distance_source": "frozen_base_expected_depth_m",
        "near_zero_m": FAR_RANGE_GATE_NEAR_ZERO_M,
        "far_full_m": FAR_RANGE_GATE_FAR_FULL_M,
        "interpolation": "cubic_smoothstep",
    }


class FrozenBaseExpectedDepthFarRangeGate(nn.Module):
    """Parameter-free gate that restricts a residual to distant rays."""

    def forward(self, base_expected_depth_m: Tensor) -> Tensor:
        if not torch.isfinite(base_expected_depth_m.float()).all():
            raise ValueError("frozen base expected depth must be finite")
        position = (
            (base_expected_depth_m.float() - FAR_RANGE_GATE_NEAR_ZERO_M)
            / (FAR_RANGE_GATE_FAR_FULL_M - FAR_RANGE_GATE_NEAR_ZERO_M)
        ).clamp(0.0, 1.0)
        weight = position.square() * (3.0 - 2.0 * position)
        return weight.to(base_expected_depth_m.dtype)


class FixedLinearTemporalPartition(nn.Module):
    """Nonnegative partition-of-unity interpolation over fixed time knots."""

    def __init__(self, rank: int = 4) -> None:
        super().__init__()
        if rank < 2:
            raise ValueError("temporal partition rank must be at least two")
        self.rank = int(rank)
        self.register_buffer("knot_indices", torch.arange(rank, dtype=torch.float32))

    def forward(self, time: Tensor) -> Tensor:
        position = time.reshape(-1).float().clamp(0.0, 1.0) * (self.rank - 1)
        distance = (position[:, None] - self.knot_indices.to(position)).abs()
        return torch.relu(1.0 - distance).to(time.dtype)


def _time_per_query(xyz: Tensor, time: Tensor) -> Tensor:
    flat_time = time.reshape(-1)
    if xyz.ndim == 2:
        if flat_time.numel() == 1:
            return flat_time.expand(xyz.shape[0])
        if flat_time.numel() == xyz.shape[0]:
            return flat_time
    elif xyz.ndim == 3:
        batch, samples = xyz.shape[:2]
        if flat_time.numel() == 1:
            return flat_time.expand(batch * samples)
        if flat_time.numel() == batch:
            return flat_time[:, None].expand(batch, samples).reshape(-1)
        if flat_time.numel() == batch * samples:
            return flat_time
    raise ValueError("time must provide one value per scene, batch, or query")


class VisibilityPreservingScalarDensityResidual(nn.Module):
    """Learn a bounded log-density redistribution with fixed ray opacity.

    ``log_residual`` is evaluated per scene-space sample.  ``redistribute``
    consumes complete rays and normalizes the multiplicative correction by
    delta-weighted base density.  The resulting density ratio is in ``[1/2, 2]``
    and each ray keeps exactly the base optical mass.
    """

    def __init__(
        self,
        *,
        spatial_resolutions: tuple[int, ...],
        channels_per_level: int = 8,
        temporal_rank: int = 4,
        num_frames: int = 51,
    ) -> None:
        super().__init__()
        if not spatial_resolutions or any(size < 2 for size in spatial_resolutions):
            raise ValueError("at least one spatial resolution of size two is required")
        if channels_per_level < 1 or temporal_rank < 1:
            raise ValueError("basis dimensions must be positive")
        if num_frames < 2:
            raise ValueError("at least two frames are required")
        self.spatial_resolutions = tuple(int(size) for size in spatial_resolutions)
        self.channels_per_level = int(channels_per_level)
        self.temporal_rank = int(temporal_rank)
        self.temporal_basis = FixedLinearTemporalPartition(temporal_rank)
        self.spatial_axes = nn.ModuleList()
        for resolution in self.spatial_resolutions:
            axes = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.empty(channels_per_level, temporal_rank, resolution)
                    )
                    for _ in range(3)
                ]
            )
            for axis in axes:
                nn.init.normal_(axis, std=0.1)
            self.spatial_axes.append(axes)
        self.output_projection = nn.Linear(
            channels_per_level * len(self.spatial_resolutions),
            1,
            bias=False,
        )
        nn.init.zeros_(self.output_projection.weight)

    def log_residual(self, xyz: Tensor, time: Tensor) -> Tensor:
        """Return the bounded scalar log-density correction per query."""

        raw = self.raw_residual(xyz, time)
        return (_HALF_LOG_TWO * torch.tanh(raw.float())).to(raw.dtype)

    def raw_residual(self, xyz: Tensor, time: Tensor) -> Tensor:
        """Shared axis-basis readout before application-specific units/bounds."""

        if xyz.ndim not in {2, 3} or xyz.shape[-1] != 3:
            raise ValueError("xyz must have shape [N, 3] or [B, N, 3]")
        output_shape = xyz.shape[:-1]
        flat_xyz = xyz.reshape(-1, 3)
        temporal = self.temporal_basis(_time_per_query(xyz, time))
        levels = []
        for axes in self.spatial_axes:
            factors = [
                _sample_axis(axis, coordinate)
                for axis, coordinate in zip(axes, flat_xyz.unbind(dim=-1))
            ]
            spatial = (
                (1.0 + factors[0])
                * (1.0 + factors[1])
                * (1.0 + factors[2])
                - 1.0
            )
            levels.append(
                torch.einsum("ncr,nr->nc", spatial, temporal.to(spatial.dtype))
            )
        raw = self.output_projection(torch.cat(levels, dim=-1)).squeeze(-1)
        return raw.reshape(output_shape)

    def redistribute(
        self,
        base_sigma: Tensor,
        log_residual: Tensor,
        deltas: Tensor,
    ) -> Tensor:
        """Redistribute density along complete rays without changing their mass."""

        if base_sigma.ndim != 2:
            raise ValueError("density redistribution requires complete [R, S] rays")
        if log_residual.shape != base_sigma.shape or deltas.shape != base_sigma.shape:
            raise ValueError("base density, residual, and deltas must share shape")
        if not torch.isfinite(base_sigma.float()).all() or torch.any(base_sigma < 0):
            raise ValueError("base density must be finite and nonnegative")
        if not torch.isfinite(log_residual.float()).all():
            raise ValueError("log-density residual must be finite")
        if not torch.isfinite(deltas.float()).all() or torch.any(deltas <= 0):
            raise ValueError("ray deltas must be finite and positive")

        base_float = base_sigma.float()
        delta_float = deltas.float()
        multiplier = torch.exp(
            log_residual.float().clamp(-_HALF_LOG_TWO, _HALF_LOG_TWO)
        )
        optical_mass = delta_float * base_float
        base_mass = optical_mass.sum(dim=-1, keepdim=True)
        if torch.any(base_mass <= 0):
            raise ValueError("each ray must have positive optical mass")
        normalizer = (optical_mass * multiplier).sum(
            dim=-1, keepdim=True
        ) / base_mass
        candidate = base_float * multiplier / normalizer
        return candidate.to(base_sigma.dtype)

    def regularization_loss(self) -> Tensor:
        return self.output_projection.weight.sum() * 0.0


class FarGatedVisibilityPreservingScalarDensityResidual(
    VisibilityPreservingScalarDensityResidual
):
    """Apply scalar density redistribution only to far frozen-Basis rays."""

    requires_base_expected_depth_m = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.range_gate = FrozenBaseExpectedDepthFarRangeGate()

    def redistribute(
        self,
        base_sigma: Tensor,
        log_residual: Tensor,
        deltas: Tensor,
        *,
        base_expected_depth_m: Tensor,
    ) -> Tensor:
        if base_expected_depth_m.shape != base_sigma.shape[:-1]:
            raise ValueError("base expected depth must provide one value per ray")
        gate = self.range_gate(base_expected_depth_m).unsqueeze(-1)
        return super().redistribute(base_sigma, log_residual * gate, deltas)


class CoherenceGatedVisibilityPreservingScalarDensityResidual(
    FarGatedVisibilityPreservingScalarDensityResidual
):
    """Restrict the far residual to frozen-Basis range-view consensus."""

    requires_frozen_basis_consensus_gate = True

    def redistribute(
        self,
        base_sigma: Tensor,
        log_residual: Tensor,
        deltas: Tensor,
        *,
        base_expected_depth_m: Tensor,
        frozen_basis_consensus_gate: Tensor,
    ) -> Tensor:
        expected_shape = base_sigma.shape[:-1]
        if frozen_basis_consensus_gate.shape != expected_shape:
            raise ValueError(
                "frozen Basis consensus gate must provide one aligned value per ray"
            )
        consensus = frozen_basis_consensus_gate.float()
        if not torch.isfinite(consensus).all():
            raise ValueError("frozen Basis consensus gate must be finite")
        if torch.any(consensus < 0.0) or torch.any(consensus > 1.0):
            raise ValueError("frozen Basis consensus gate must stay in [0, 1]")
        if frozen_basis_consensus_gate.requires_grad:
            raise ValueError("frozen Basis consensus gate must not require gradients")
        return super().redistribute(
            base_sigma,
            log_residual * consensus.to(log_residual.dtype).unsqueeze(-1),
            deltas,
            base_expected_depth_m=base_expected_depth_m,
        )


class LearnedGatedVisibilityPreservingScalarDensityResidual(
    VisibilityPreservingScalarDensityResidual
):
    """Route a frozen scalar residual with a separately learned range gate."""

    requires_learned_residual_gate_weight = True

    def __init__(
        self,
        *,
        physical_min_range_m: float,
        physical_max_range_m: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.routing_gate = LearnedResidualRangeGate(
            physical_min_range_m=physical_min_range_m,
            physical_max_range_m=physical_max_range_m,
        )

    def redistribute(
        self,
        base_sigma: Tensor,
        log_residual: Tensor,
        deltas: Tensor,
        *,
        learned_residual_gate_weight: Tensor,
    ) -> Tensor:
        expected_shape = base_sigma.shape[:-1]
        if learned_residual_gate_weight.shape != expected_shape:
            raise ValueError(
                "learned residual gate must provide one aligned value per ray"
            )
        gate = learned_residual_gate_weight.float()
        if not torch.isfinite(gate).all():
            raise ValueError("learned residual gate must be finite")
        if torch.any(gate < 0.0) or torch.any(gate > 1.0):
            raise ValueError("learned residual gate must stay in [0, 1]")
        return super().redistribute(
            base_sigma,
            log_residual * gate.to(log_residual.dtype).unsqueeze(-1),
            deltas,
        )
