"""Small, opt-in visibility residuals for range-view edge and far support."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


VISIBILITY_RESIDUAL_NONE = "none"
EDGE_FAR_MULTISCALE_RESIDUAL_V1 = "edge_far_multiscale_residual_v1"
VISIBILITY_RESIDUAL_BASES = (
    VISIBILITY_RESIDUAL_NONE,
    EDGE_FAR_MULTISCALE_RESIDUAL_V1,
)


def _panorama_pad(value: Tensor, radius: int) -> Tensor:
    if radius <= 0:
        return value
    value = F.pad(value, (radius, radius, 0, 0), mode="circular")
    return F.pad(value, (0, 0, radius, radius), mode="replicate")


def _panorama_average(value: Tensor, kernel_size: int) -> Tensor:
    radius = kernel_size // 2
    return F.avg_pool2d(
        _panorama_pad(value, radius),
        kernel_size=kernel_size,
        stride=1,
    )


class _PanoramaConv2d(nn.Module):
    """A convolution with azimuth wrap-around and vertical edge replication."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        zero_initialized: bool = False,
    ) -> None:
        super().__init__()
        conv_type = _ZeroInitializedConv2d if zero_initialized else nn.Conv2d
        self.conv = conv_type(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=0,
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.conv(_panorama_pad(value, 1))


class _ZeroInitializedConv2d(nn.Conv2d):
    """Keep the residual an exact identity after construction and every reset."""

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class EdgeFarVisibilityResidualBasis(nn.Module):
    """Correct frozen visibility logits using local range-view detail bases.

    The implementation analyzes two spatial scales. Horizontal operations wrap
    at the LiDAR panorama seam. Its output projection is zero initialized, so
    enabling the module does not change an existing refiner before training.
    """

    def __init__(
        self,
        *,
        input_channels: int = 3,
        hidden_channels: int = 16,
        depth_scale: float,
        max_depth_m: float = 80.0,
    ) -> None:
        super().__init__()
        if input_channels != 3:
            raise ValueError("edge/far residual basis requires three refiner inputs")
        if hidden_channels <= 0:
            raise ValueError("residual hidden channels must be positive")
        if depth_scale <= 0.0:
            raise ValueError("visibility residual depth scale must be positive")
        if max_depth_m <= 0.0:
            raise ValueError("visibility residual maximum depth must be positive")
        self.depth_scale = float(depth_scale)
        self.max_depth_m = float(max_depth_m)

        # Two three-channel high-pass bands, two base-probability bands, a
        # normalized far-range coordinate, and a fixed edge/far support gate.
        basis_channels = 2 * input_channels + 2 + 2
        self.hidden = _PanoramaConv2d(basis_channels, hidden_channels)
        self.activation = nn.SiLU()
        self.output = _PanoramaConv2d(
            hidden_channels,
            1,
            zero_initialized=True,
        )

    def _basis_and_support(
        self,
        base_logits: Tensor,
        refiner_input: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if base_logits.ndim != 4 or base_logits.shape[1] != 1:
            raise ValueError("base visibility logits must have shape [B,1,H,W]")
        if refiner_input.ndim != 4 or refiner_input.shape[1] != 3:
            raise ValueError("visibility refiner input must have shape [B,3,H,W]")
        if base_logits.shape[0] != refiner_input.shape[0] or base_logits.shape[2:] != refiner_input.shape[2:]:
            raise ValueError("visibility logits and refiner input must align")

        normalized_depth = (
            (refiner_input[:, 2:3] / self.depth_scale)
            .clamp(0.0, self.max_depth_m)
            .div(self.max_depth_m)
        )
        normalized_input = torch.cat(
            (
                refiner_input[:, 0:1].clamp(0.0, 1.0),
                refiner_input[:, 1:2].clamp(0.0, 1.0),
                normalized_depth,
            ),
            dim=1,
        )
        base_probability = torch.sigmoid(base_logits)

        input_band_3 = normalized_input - _panorama_average(normalized_input, 3)
        input_band_7 = normalized_input - _panorama_average(normalized_input, 7)
        base_band_3 = base_probability - _panorama_average(base_probability, 3)
        base_band_7 = base_probability - _panorama_average(base_probability, 7)

        edge_support = torch.maximum(
            input_band_3.abs().amax(dim=1, keepdim=True),
            input_band_7.abs().amax(dim=1, keepdim=True),
        ).clamp(0.0, 1.0)
        far_support = normalized_depth.square()
        detail_support = (
            torch.maximum(edge_support, far_support)
            .detach()
            .to(dtype=base_logits.dtype)
        )
        basis = torch.cat(
            (
                input_band_3,
                input_band_7,
                base_band_3,
                base_band_7,
                far_support,
                detail_support,
            ),
            dim=1,
        )
        return basis, detail_support

    def forward(self, base_logits: Tensor, refiner_input: Tensor) -> Tensor:
        basis, detail_support = self._basis_and_support(base_logits, refiner_input)
        residual_logits = self.output(self.activation(self.hidden(basis)))
        return base_logits + detail_support * residual_logits


def build_visibility_residual_basis(
    name: str,
    *,
    depth_scale: float,
) -> EdgeFarVisibilityResidualBasis | None:
    """Build one registered visibility residual adapter."""

    if name == VISIBILITY_RESIDUAL_NONE:
        return None
    if name == EDGE_FAR_MULTISCALE_RESIDUAL_V1:
        return EdgeFarVisibilityResidualBasis(depth_scale=depth_scale)
    raise ValueError(f"unknown visibility residual basis {name!r}")
