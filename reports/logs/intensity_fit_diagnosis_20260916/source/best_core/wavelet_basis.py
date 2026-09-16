"""Continuous compact spline-wavelet bases for LiDAR4D residual dynamics."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _cubic_bspline(value: Tensor) -> Tensor:
    """Centered cardinal cubic B-spline with compact support ``[-2, 2]``."""
    distance = value.abs()
    return (
        F.relu(2.0 - distance).pow(3)
        - 4.0 * F.relu(1.0 - distance).pow(3)
    ) / 6.0


def _cubic_spline_wavelet(value: Tensor) -> Tensor:
    """C2, compact, zero-integral wavelet derived from cubic B-splines."""
    return 2.0 * _cubic_bspline(2.0 * value) - _cubic_bspline(value)


class SplineWaveletDictionary(nn.Module):
    """Four scaling atoms and three resolutions of compact spline wavelets."""

    atom_counts = (4, 4, 8, 8)

    def __init__(self) -> None:
        super().__init__()
        scaling_centers = torch.linspace(0.0, 1.0, self.atom_counts[0])
        scaling_scales = torch.full((self.atom_counts[0],), 0.34)
        wavelet_centers = torch.cat(
            [
                torch.linspace(0.0, 1.0, self.atom_counts[1]),
                torch.linspace(0.0, 1.0, self.atom_counts[2]),
                torch.linspace(0.0, 1.0, self.atom_counts[3]),
            ]
        )
        wavelet_scales = torch.cat(
            [
                torch.full((self.atom_counts[1],), 0.25),
                torch.full((self.atom_counts[2],), 0.125),
                torch.full((self.atom_counts[3],), 0.0625),
            ]
        )
        self.register_buffer("scaling_centers", scaling_centers)
        self.register_buffer("scaling_scales", scaling_scales)
        self.register_buffer("wavelet_centers", wavelet_centers)
        self.register_buffer("wavelet_scales", wavelet_scales)
        self.register_buffer("center_adaptation", torch.tensor(0.0))
        self.center_shift = nn.Parameter(torch.zeros(wavelet_centers.numel()))

    @property
    def candidate_count(self) -> int:
        return sum(self.atom_counts)

    def set_center_adaptation(self, scale: float) -> None:
        self.center_adaptation.fill_(float(max(0.0, min(scale, 1.0))))

    def forward(self, t: Tensor) -> Tensor:
        flat_time = t.reshape(-1)
        scaling_position = (
            flat_time[:, None] - self.scaling_centers.to(flat_time)
        ) / self.scaling_scales.to(flat_time)
        scaling = _cubic_bspline(scaling_position)

        bounded_shift = 0.25 * self.wavelet_scales * torch.tanh(self.center_shift)
        centers = self.wavelet_centers + self.center_adaptation * bounded_shift
        wavelet_position = (
            flat_time[:, None] - centers.to(flat_time)
        ) / self.wavelet_scales.to(flat_time)
        wavelets = _cubic_spline_wavelet(wavelet_position)
        return torch.cat((scaling, wavelets), dim=-1)


class ProjectedSplineWaveletRefiner(nn.Module):
    """Rank-limited wavelet residual orthogonal to two analytic anchors."""

    def __init__(
        self,
        rank: int = 4,
        *,
        num_frames: int = 51,
        projection_epsilon: float = 1e-4,
        extrapolation_fade: float = 0.1,
    ) -> None:
        super().__init__()
        if rank < 1 or num_frames < 2:
            raise ValueError("rank and num_frames must be positive")
        if projection_epsilon <= 0.0 or extrapolation_fade <= 0.0:
            raise ValueError("projection epsilon and fade width must be positive")
        self.rank = rank
        self.num_frames = num_frames
        self.projection_epsilon = float(projection_epsilon)
        self.extrapolation_fade = float(extrapolation_fade)
        self.dictionary = SplineWaveletDictionary()
        self.mixing = nn.Parameter(
            torch.empty(self.dictionary.candidate_count, rank)
        )
        nn.init.orthogonal_(self.mixing)
        self.register_buffer("reference_times", torch.linspace(0.0, 1.0, num_frames))
        self.register_buffer("progress", torch.tensor(1.0))
        self.register_buffer("last_gram_loss", torch.tensor(0.0))
        self.register_buffer("last_diversity_loss", torch.tensor(0.0))
        self.register_buffer("last_center_shift", torch.tensor(0.0))
        self._last_regularization: Tensor | None = None

    def set_progress(self, scale: float) -> None:
        progress = float(max(0.0, min(scale, 1.0)))
        self.progress.fill_(progress)
        center_adaptation = max(0.0, min((progress - 0.8) / 0.2, 1.0))
        self.dictionary.set_center_adaptation(center_adaptation)

    @staticmethod
    def _joint_anchor(t: Tensor, plane_anchor: nn.Module, hash_anchor: nn.Module) -> Tensor:
        return torch.cat((plane_anchor(t), hash_anchor(t)), dim=-1)

    def _projected_dictionary(
        self,
        t: Tensor,
        plane_anchor: nn.Module,
        hash_anchor: nn.Module,
    ) -> tuple[Tensor, Tensor, Tensor]:
        reference_times = self.reference_times.to(device=t.device, dtype=t.dtype)
        base_reference = self._joint_anchor(reference_times, plane_anchor, hash_anchor)
        candidate_reference = self.dictionary(reference_times)
        base_query = self._joint_anchor(t.reshape(-1), plane_anchor, hash_anchor)
        candidate_query = self.dictionary(t)
        with torch.cuda.amp.autocast(enabled=False):
            base32 = base_reference.float()
            candidates32 = candidate_reference.float()
            identity = torch.eye(
                base32.shape[-1], device=base32.device, dtype=base32.dtype
            )
            projection = torch.linalg.solve(
                base32.T @ base32 + self.projection_epsilon * identity,
                base32.T @ candidates32,
            )
            reference_orthogonal = candidates32 - base32 @ projection
            query_orthogonal = candidate_query.float() - base_query.float() @ projection
        return query_orthogonal, reference_orthogonal, base32

    def _boundary_envelope(self, t: Tensor) -> Tensor:
        flat_time = t.reshape(-1)
        distance = torch.where(
            flat_time < 0.0,
            -flat_time,
            torch.where(flat_time > 1.0, flat_time - 1.0, torch.zeros_like(flat_time)),
        )
        value = (distance / self.extrapolation_fade).clamp(0.0, 1.0)
        smooth = 1.0 - 10.0 * value.pow(3) + 15.0 * value.pow(4) - 6.0 * value.pow(5)
        return torch.where(
            distance < self.extrapolation_fade, smooth, torch.zeros_like(smooth)
        )

    def forward(
        self,
        t: Tensor,
        plane_anchor: nn.Module,
        hash_anchor: nn.Module,
    ) -> Tensor:
        if float(self.progress) <= 0.0:
            self._last_regularization = None
            self.last_gram_loss.zero_()
            self.last_diversity_loss.zero_()
            self.last_center_shift.zero_()
            return t.new_zeros((t.numel(), self.rank))

        query_dictionary, reference_dictionary, base_reference = (
            self._projected_dictionary(t, plane_anchor, hash_anchor)
        )
        reference_output = reference_dictionary @ self.mixing.float()
        output_rms = reference_output.square().mean(dim=0).sqrt().clamp_min(1e-4)
        output = query_dictionary @ self.mixing.float() / output_rms

        base_normalized = base_reference / base_reference.square().mean(
            dim=0, keepdim=True
        ).sqrt().clamp_min(1e-4)
        output_normalized = reference_output / output_rms
        gram_loss = (
            base_normalized.T @ output_normalized / self.num_frames
        ).square().mean()
        mixing_normalized = F.normalize(self.mixing.float(), dim=0)
        mixing_gram = mixing_normalized.T @ mixing_normalized
        identity = torch.eye(self.rank, device=mixing_gram.device, dtype=mixing_gram.dtype)
        diversity_loss = (mixing_gram - identity).square().mean()
        center_shift = torch.tanh(self.dictionary.center_shift).square().mean()
        self._last_regularization = (
            1e-3 * gram_loss + 1e-4 * diversity_loss + 1e-6 * center_shift
        )
        self.last_gram_loss.copy_(gram_loss.detach().to(self.last_gram_loss))
        self.last_diversity_loss.copy_(
            diversity_loss.detach().to(self.last_diversity_loss)
        )
        self.last_center_shift.copy_(center_shift.detach().to(self.last_center_shift))

        envelope = self._boundary_envelope(t).float().unsqueeze(-1)
        output = output * envelope * self.progress.float()
        return output.to(dtype=t.dtype)

    def regularization_loss(self) -> Tensor:
        if self._last_regularization is None:
            return self.mixing.new_zeros(())
        return self._last_regularization
