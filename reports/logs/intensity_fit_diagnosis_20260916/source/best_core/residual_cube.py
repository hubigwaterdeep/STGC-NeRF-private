"""Parameter-matched low-rank volumetric residual fields."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TuckerResidualCube(nn.Module):
    """Implicit 3D coefficient cube reconstructed from a Tucker core."""

    def __init__(
        self,
        *,
        output_channels: int,
        temporal_rank: int,
        resolution: int = 32,
        spatial_rank: int = 8,
    ) -> None:
        super().__init__()
        if min(output_channels, temporal_rank, resolution, spatial_rank) < 1:
            raise ValueError("all Tucker cube dimensions must be positive")
        self.output_channels = output_channels
        self.temporal_rank = temporal_rank
        self.resolution = resolution
        self.spatial_rank = spatial_rank
        self.core = nn.Parameter(
            torch.empty(
                output_channels,
                temporal_rank,
                spatial_rank,
                spatial_rank,
                spatial_rank,
            )
        )
        self.factor_x = nn.Parameter(torch.empty(resolution, spatial_rank))
        self.factor_y = nn.Parameter(torch.empty(resolution, spatial_rank))
        self.factor_z = nn.Parameter(torch.empty(resolution, spatial_rank))
        nn.init.normal_(self.core, std=1e-3)
        for factor in (self.factor_x, self.factor_y, self.factor_z):
            nn.init.orthogonal_(factor)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def dense_coefficients(self) -> Tensor:
        return torch.einsum(
            "crijk,xi,yj,zk->crzyx",
            self.core,
            self.factor_x,
            self.factor_y,
            self.factor_z,
        )

    def forward(
        self,
        xyz: Tensor,
        temporal: Tensor,
        *,
        gates: Tensor | None = None,
        motion: Tensor | None = None,
        cache: dict[str, Tensor] | None = None,
    ) -> Tensor:
        if xyz.ndim != 2 or xyz.shape[-1] != 3:
            raise ValueError("xyz must have shape [N, 3]")
        if temporal.shape != (xyz.shape[0], self.temporal_rank):
            raise ValueError("temporal weights do not match the cube contract")
        dense = cache.get("tucker_dense") if cache is not None else None
        if dense is None:
            dense = self.dense_coefficients()
            if cache is not None:
                cache["tucker_dense"] = dense
        grid = xyz.mul(2.0).sub(1.0).view(1, 1, 1, -1, 3)
        sampled = F.grid_sample(
            dense.reshape(
                1,
                self.output_channels * self.temporal_rank,
                self.resolution,
                self.resolution,
                self.resolution,
            ),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        coefficients = sampled[0, :, 0, 0].T.reshape(
            -1, self.output_channels, self.temporal_rank
        )
        weights = temporal[:, None, :]
        if gates is not None:
            if gates.shape != (self.output_channels, self.temporal_rank):
                raise ValueError("cube gates must have shape [channels, temporal_rank]")
            weights = weights * gates[None].to(weights)
        if motion is not None:
            weights = weights * motion.reshape(-1, 1, 1).to(weights)
        return torch.einsum("ncr,ncr->nc", coefficients, weights.to(coefficients))
