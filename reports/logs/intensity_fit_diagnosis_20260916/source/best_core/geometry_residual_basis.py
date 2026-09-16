"""Compact scene-space basis for geometry-only feature corrections.

The adapter deliberately owns its temporal atoms and fine spatial
coefficients.  It receives only normalized scene coordinates, normalized time,
and the decoder-ready base feature used to enforce a pointwise safety cap.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _cubic_bspline(value: Tensor) -> Tensor:
    distance = value.abs()
    return (
        F.relu(2.0 - distance).pow(3)
        - 4.0 * F.relu(1.0 - distance).pow(3)
    ) / 6.0


def _cubic_spline_wavelet(value: Tensor) -> Tensor:
    """Compact C2 atom whose coarse component is removed analytically."""
    return 2.0 * _cubic_bspline(2.0 * value) - _cubic_bspline(value)


def _time_per_point(time: Tensor, point_count: int) -> Tensor:
    flat_time = time.reshape(-1)
    if flat_time.numel() == 1:
        return flat_time.expand(point_count)
    if flat_time.numel() != point_count:
        raise ValueError(
            f"expected one time or {point_count} times, got {flat_time.numel()}"
        )
    return flat_time


def _sample_axis(profile: Tensor, coordinate: Tensor) -> Tensor:
    """Linearly sample ``[C, R, S]`` coefficients as ``[N, C, R]``."""
    channels, rank, size = profile.shape
    image = profile.reshape(channels * rank, 1, size)
    x = coordinate.mul(2.0).sub(1.0)
    grid = torch.stack((x, torch.zeros_like(x)), dim=-1).view(1, 1, -1, 2)
    sampled = F.grid_sample(
        image.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[0, :, 0].transpose(0, 1).reshape(-1, channels, rank)


class LocalHighFrequencyTemporalBasis(nn.Module):
    """Independent compact wavelet atoms mixed to a small temporal rank."""

    def __init__(self, rank: int = 4, *, num_frames: int = 51) -> None:
        super().__init__()
        if rank < 1 or num_frames < 2:
            raise ValueError("rank and num_frames must be positive")
        centers = torch.cat(
            (
                torch.linspace(0.0, 1.0, 4),
                torch.linspace(0.0, 1.0, 8),
                torch.linspace(0.0, 1.0, 8),
            )
        )
        scales = torch.cat(
            (
                torch.full((4,), 0.25),
                torch.full((8,), 0.125),
                torch.full((8,), 0.0625),
            )
        )
        self.rank = int(rank)
        self.register_buffer("centers", centers)
        self.register_buffer("scales", scales)
        self.register_buffer("reference_times", torch.linspace(0.0, 1.0, num_frames))
        self.center_shift = nn.Parameter(torch.zeros_like(centers))
        self.mixing = nn.Parameter(torch.empty(centers.numel(), rank))
        nn.init.orthogonal_(self.mixing)

    def _dictionary(self, time: Tensor) -> Tensor:
        bounded_shift = 0.25 * self.scales * torch.tanh(self.center_shift)
        centers = self.centers + bounded_shift
        position = (
            time.reshape(-1, 1) - centers.to(time)
        ) / self.scales.to(time)
        return _cubic_spline_wavelet(position)

    def forward(self, time: Tensor) -> Tensor:
        reference = self._dictionary(self.reference_times.to(time))
        reference_output = reference @ self.mixing.to(reference)
        normalizer = reference_output.square().mean(dim=0).sqrt().clamp_min(1e-4)
        return self._dictionary(time) @ self.mixing.to(time) / normalizer.to(time)

    def regularization_loss(self) -> Tensor:
        normalized = F.normalize(self.mixing.float(), dim=0)
        gram = normalized.T @ normalized
        identity = torch.eye(self.rank, dtype=gram.dtype, device=gram.device)
        diversity = (gram - identity).square().mean()
        center_movement = torch.tanh(self.center_shift.float()).square().mean()
        return 1e-4 * diversity + 1e-6 * center_movement


class GeometryResidualBasis(nn.Module):
    """Fine scene-space coefficients contracted with independent time atoms.

    Each level is a separable four-dimensional basis: three learned 1D spatial
    factors are multiplied and contracted with a local temporal wavelet basis.
    The output projection is zero initialized, so attaching this adapter to an
    existing field is exactly inert before residual-only optimization starts.
    """

    def __init__(
        self,
        output_dim: int,
        *,
        spatial_resolutions: tuple[int, ...],
        channels_per_level: int = 8,
        temporal_rank: int = 4,
        num_frames: int = 51,
        max_feature_ratio: float = 0.10,
    ) -> None:
        super().__init__()
        if output_dim < 1 or channels_per_level < 1 or temporal_rank < 1:
            raise ValueError("feature dimensions and temporal rank must be positive")
        if not spatial_resolutions or any(size < 2 for size in spatial_resolutions):
            raise ValueError("at least one spatial resolution of size two is required")
        if max_feature_ratio <= 0.0:
            raise ValueError("max feature ratio must be positive")

        self.output_dim = int(output_dim)
        self.spatial_resolutions = tuple(int(size) for size in spatial_resolutions)
        self.channels_per_level = int(channels_per_level)
        self.temporal_rank = int(temporal_rank)
        self.max_feature_ratio = float(max_feature_ratio)
        self.temporal_basis = LocalHighFrequencyTemporalBasis(
            temporal_rank, num_frames=num_frames
        )
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
            output_dim,
            bias=False,
        )
        nn.init.zeros_(self.output_projection.weight)

    def _basis_features(self, xyz: Tensor, time: Tensor) -> Tensor:
        if xyz.ndim != 2 or xyz.shape[-1] != 3:
            raise ValueError("xyz must have shape [N, 3]")
        per_point_time = _time_per_point(time, xyz.shape[0])
        temporal = self.temporal_basis(per_point_time)
        levels = []
        for axes in self.spatial_axes:
            factors = [
                _sample_axis(axis, coordinate)
                for axis, coordinate in zip(axes, xyz.unbind(dim=-1))
            ]
            # Mirror the base field's stable multiplicative factorization while
            # removing its constant component.  This keeps the independently
            # learned fine coefficients responsive from a zero-output start.
            spatial = (
                (1.0 + factors[0])
                * (1.0 + factors[1])
                * (1.0 + factors[2])
                - 1.0
            )
            levels.append(
                torch.einsum("ncr,nr->nc", spatial, temporal.to(spatial.dtype))
            )
        return torch.cat(levels, dim=-1)

    def forward(self, xyz: Tensor, time: Tensor, base_feature: Tensor) -> Tensor:
        if base_feature.shape != (xyz.shape[0], self.output_dim):
            raise ValueError(
                "base feature must have shape "
                f"[{xyz.shape[0]}, {self.output_dim}]"
            )
        raw = self.output_projection(self._basis_features(xyz, time))
        raw_float = raw.float()
        residual_norm = raw_float.norm(dim=-1, keepdim=True)
        base_norm = base_feature.detach().float().norm(dim=-1, keepdim=True)
        limit = self.max_feature_ratio * base_norm
        scale = torch.minimum(
            torch.ones_like(residual_norm),
            limit / residual_norm.clamp_min(1e-12),
        )
        return (raw_float * scale).to(base_feature.dtype)

    def regularization_loss(self) -> Tensor:
        return self.temporal_basis.regularization_loss()
