"""Interchangeable scene fields for parameter-matched LiDAR4D experiments.

Every field consumes normalized ``xyz`` and time in ``[0, 1]`` and returns the
same 120-dimensional feature consumed by LiDAR4D's original density head.  The
ray renderer and the density/intensity/return heads therefore stay identical
across experiments.

The default alternative configurations are deliberately sized to the original
plane + hash + flow budget (46,488,960 trainable parameters):

* ``o2a``: stochastic static triplanes plus analytic temporal axis bases.
* ``cod_triplane``: stochastic COD-style triplanes, projected to static and
  temporal-basis planes and fused by summation.
* ``basis_time_only``: official spatial/flow field with only temporal knots
  replaced by a physical-time Koopman/Jordan basis.
* ``stochastic_basis_time_only``: the matched mean field plus a low-rank
  temporal posterior.
* ``matched_hybrid_residual``: a full mean field plus a structured stochastic
  residual, with static/dynamic/flow sub-budgets matched independently.
* ``g4d_direct``: a direct dense 4D feature grid.
* ``g4d_basis_cube``: a static cube plus analytic temporal-basis cubes.

The stochastic fields optimize their posterior parameters directly for one
scene.  This is an auto-decoder control, not the future raw-point encoder.
"""

from __future__ import annotations

import math
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import tinycudann as tcnn

from best_core.flow_field import FlowField
from best_core.geometry_residual_basis import GeometryResidualBasis
from best_core.scalar_density_residual import (
    CoherenceGatedVisibilityPreservingScalarDensityResidual,
    FarGatedVisibilityPreservingScalarDensityResidual,
    LearnedGatedVisibilityPreservingScalarDensityResidual,
    VisibilityPreservingScalarDensityResidual,
)
from best_core.field_budget import (
    FieldBudgetReport,
    OFFICIAL_FIELD_BUDGET,
    count_parameters,
)
from best_core.hash_field import HashGrid4D
from best_core.motion_router import MotionRouter
from best_core.planes_field import Planes4D
from best_core.residual_cube import TuckerResidualCube
from best_core.wavelet_basis import ProjectedSplineWaveletRefiner


OFFICIAL_FIELD_PARAMETER_BUDGET = OFFICIAL_FIELD_BUDGET.total


def _time_per_point(t: Tensor, points: int) -> Tensor:
    """Return one normalized time per query point."""
    flat = t.reshape(-1)
    if flat.numel() == 1:
        return flat.expand(points)
    if flat.numel() != points:
        raise ValueError(f"expected one time or {points} times, got {flat.numel()}")
    return flat


def _normalized_frame_time(frame_index: int, num_frames: int) -> float:
    """Map a frame index to the dataset's inclusive ``[0, 1]`` timeline."""
    if num_frames < 2:
        raise ValueError("at least two frames are required for normalized time")
    if not 0 <= frame_index < num_frames:
        raise ValueError(f"frame {frame_index} is outside [0, {num_frames - 1}]")
    return frame_index / (num_frames - 1)


def _sample_plane(plane: Tensor, coords: Tensor) -> Tensor:
    """Bilinearly query ``[C, H, W]`` at ``[N, (x, y)]`` in ``[0, 1]``."""
    grid = coords.mul(2.0).sub(1.0).view(1, 1, -1, 2)
    sampled = F.grid_sample(
        plane.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[0, :, 0].transpose(0, 1)


def _sample_axis(profile: Tensor, coordinate: Tensor) -> Tensor:
    """Linearly query ``[C, R, S]`` and return ``[N, C, R]``."""
    channels, rank, size = profile.shape
    sampled = _sample_line(profile.reshape(channels * rank, size), coordinate)
    return sampled.reshape(-1, channels, rank)


def _sample_line(profile: Tensor, coordinate: Tensor) -> Tensor:
    """Linearly query ``[C, S]`` and return ``[N, C]``."""
    channels, size = profile.shape
    image = profile.reshape(channels, 1, size)
    x = coordinate.mul(2.0).sub(1.0)
    grid = torch.stack([x, torch.zeros_like(x)], dim=-1).view(1, 1, -1, 2)
    sampled = F.grid_sample(
        image.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[0, :, 0].transpose(0, 1)


def _contract_then_sample_axis(
    profile: Tensor, mode_weights: Tensor, coordinate: Tensor
) -> Tensor:
    """Contract ``[C, R, S]`` to ``[C, S]`` before spatial interpolation.

    Linear interpolation and mode contraction commute when all points share a
    query time. This avoids materializing the former ``[N, C, R]`` activation.
    """
    contracted_profile = torch.einsum("crs,r->cs", profile, mode_weights)
    return _sample_line(contracted_profile, coordinate)


def _sample_cube(cube: Tensor, xyz: Tensor) -> Tensor:
    """Trilinearly query ``[C, Z, Y, X]`` and return ``[N, C]``."""
    grid = xyz.mul(2.0).sub(1.0).view(1, 1, 1, -1, 3)
    sampled = F.grid_sample(
        cube.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[0, :, 0, 0].transpose(0, 1)


class GaussianParameter(nn.Module):
    """A directly optimized diagonal-Gaussian posterior tensor."""

    def __init__(self, shape: tuple[int, ...], init_scale: float = 0.01) -> None:
        super().__init__()
        self.mu = nn.Parameter(torch.empty(shape))
        self.logvar = nn.Parameter(torch.full(shape, -8.0))
        nn.init.normal_(self.mu, std=init_scale)

    def forward(self) -> Tensor:
        if not self.training:
            return self.mu
        logvar = self.logvar.clamp(-30.0, 20.0)
        return self.mu + torch.randn_like(self.mu) * torch.exp(0.5 * logvar)

    def kl_loss(self) -> Tensor:
        logvar = self.logvar.clamp(-30.0, 20.0)
        return -0.5 * (1.0 + logvar - self.mu.square() - logvar.exp()).mean()


class AxisModeResidualPosterior(nn.Module):
    """Structured posterior shared by all spatial coefficients of each mode."""

    def __init__(self, axes: int, rank: int) -> None:
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(axes, rank))
        self.logvar = nn.Parameter(torch.full((axes, rank), -8.0))

    def sample(self) -> Tensor:
        if not self.training:
            return self.mu
        logvar = self.logvar.clamp(-30.0, 20.0)
        return self.mu + torch.randn_like(self.mu) * torch.exp(0.5 * logvar)

    def kl_loss(self) -> Tensor:
        logvar = self.logvar.clamp(-30.0, 20.0)
        return -0.5 * (
            1.0 + logvar - self.mu.square() - logvar.exp()
        ).mean()


class StochasticResidualPosterior(nn.Module):
    """One stochastic residual context per spatial role and temporal mode."""

    def __init__(
        self,
        *,
        plane_levels: int,
        hash_levels: int,
        axes: int,
        rank: int,
    ) -> None:
        super().__init__()
        self.plane_mu = nn.Parameter(torch.zeros(plane_levels, axes, rank))
        self.plane_logvar = nn.Parameter(
            torch.full((plane_levels, axes, rank), -8.0)
        )
        self.hash_mu = nn.Parameter(torch.zeros(axes, hash_levels, rank))
        self.hash_logvar = nn.Parameter(
            torch.full((axes, hash_levels, rank), -8.0)
        )

    def _sample(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if not self.training:
            return mu
        logvar = logvar.clamp(-30.0, 20.0)
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def sample(self) -> tuple[Tensor, Tensor]:
        return (
            self._sample(self.plane_mu, self.plane_logvar),
            self._sample(self.hash_mu, self.hash_logvar),
        )

    def kl_loss(self) -> Tensor:
        terms = []
        for mu, raw_logvar in (
            (self.plane_mu, self.plane_logvar),
            (self.hash_mu, self.hash_logvar),
        ):
            logvar = raw_logvar.clamp(-30.0, 20.0)
            terms.append(-0.5 * (1.0 + logvar - mu.square() - logvar.exp()).mean())
        return torch.stack(terms).mean()


class StructuredResidualPosterior(StochasticResidualPosterior):
    """Backward-compatible name for the structured stochastic posterior."""


class DiscreteTemporalBasis(nn.Module):
    """Learnable local hat functions centred on discrete frame knots.

    This is the non-analytic control: every temporal coefficient is local to a
    knot, while its width and gain remain trainable.  The two vectors give the
    default rank-eight control exactly the same 16-parameter basis budget as
    the analytic and Jordan controls.
    """

    def __init__(self, rank: int = 8) -> None:
        super().__init__()
        if rank < 2:
            raise ValueError("temporal rank must be at least two")
        self.rank = rank
        self.raw_width = nn.Parameter(torch.zeros(rank))
        self.raw_gain = nn.Parameter(torch.zeros(rank))
        self.register_buffer("knots", torch.arange(rank, dtype=torch.float32))

    def forward(self, t: Tensor, latent: Tensor | None = None) -> Tensor:
        position = t.unsqueeze(-1) * (self.rank - 1)
        width = torch.exp(self.raw_width).clamp(0.25, 4.0)
        output = F.relu(1.0 - (position - self.knots).abs() / width)
        output = output * (1.0 + 0.1 * torch.tanh(self.raw_gain))
        if latent is not None:
            output = output * latent.to(dtype=output.dtype, device=output.device)
        return output


class AnalyticTemporalBasis(nn.Module):
    """Learnable bounded complex modes evaluated at continuous normalized time."""

    def __init__(self, rank: int = 8) -> None:
        super().__init__()
        if rank < 2 or rank % 2:
            raise ValueError("temporal rank must be a positive even number")
        modes = rank // 2
        self.rank = rank
        self.raw_sigma = nn.Parameter(torch.zeros(modes))
        self.raw_omega = nn.Parameter(torch.empty(modes))
        self.raw_gain = nn.Parameter(torch.zeros(rank))
        ratios = torch.arange(1, modes + 1, dtype=torch.float32) / (modes + 1)
        with torch.no_grad():
            self.raw_omega.copy_(torch.atanh(ratios))
        self.register_buffer("omega_max", torch.tensor((modes + 1) * torch.pi))

    def forward(self, t: Tensor, latent: Tensor | None = None) -> Tensor:
        centered = 2.0 * t - 1.0
        sigma = 0.5 * torch.tanh(self.raw_sigma)
        omega = self.omega_max * torch.tanh(self.raw_omega)
        envelope = torch.exp(centered.unsqueeze(-1) * sigma)
        phase = centered.unsqueeze(-1) * omega
        output = torch.stack(
            [envelope * torch.cos(phase), envelope * torch.sin(phase)], dim=-1
        ).flatten(-2)
        output = output * (1.0 + 0.1 * torch.tanh(self.raw_gain))
        if latent is not None:
            output = output * latent.to(dtype=output.dtype, device=output.device)
        return output


class BSplineTemporalBasis(nn.Module):
    """Local B-spline atoms with optional, ordered adaptive knots.

    The interface deliberately exposes only basis evaluation and a scalar
    regularizer.  Positive interval parameterization hides knot ordering and
    minimum-spacing constraints from every scene-field caller.
    """

    def __init__(
        self,
        rank: int = 8,
        *,
        degree: int = 3,
        adaptive_knots: bool = False,
        minimum_spacing: float = 0.02,
    ) -> None:
        super().__init__()
        if degree < 1:
            raise ValueError("B-spline degree must be positive")
        if rank <= degree:
            raise ValueError("B-spline rank must exceed its degree")
        interval_count = rank - degree
        if not 0.0 <= minimum_spacing < 1.0 / interval_count:
            raise ValueError("minimum knot spacing leaves no free interval mass")
        self.rank = rank
        self.degree = degree
        self.adaptive_knots = adaptive_knots
        self.minimum_spacing = minimum_spacing
        self.interval_count = interval_count
        if adaptive_knots:
            self.raw_interval_logits = nn.Parameter(torch.zeros(interval_count))
        else:
            self.register_buffer(
                "fixed_intervals",
                torch.full((interval_count,), 1.0 / interval_count),
            )

    def intervals(self) -> Tensor:
        if not self.adaptive_knots:
            return self.fixed_intervals
        free_mass = 1.0 - self.minimum_spacing * self.interval_count
        return self.minimum_spacing + free_mass * torch.softmax(
            self.raw_interval_logits, dim=0
        )

    def knots(self) -> Tensor:
        intervals = self.intervals()
        interior = intervals.cumsum(dim=0)[:-1]
        endpoints = intervals.new_tensor([0.0, 1.0])
        return torch.cat(
            [
                endpoints[:1].expand(self.degree + 1),
                interior,
                endpoints[1:].expand(self.degree + 1),
            ]
        )

    def regularization_loss(self) -> Tensor:
        intervals = self.intervals()
        if not self.adaptive_knots:
            return intervals.new_zeros(())
        uniform = 1.0 / self.interval_count
        return ((intervals - uniform) / uniform).square().mean()

    def _basis_at_degree(self, x: Tensor, degree: int) -> Tensor:
        knots = self.knots().to(dtype=x.dtype, device=x.device)
        values = (
            (x.unsqueeze(-1) >= knots[:-1])
            & (x.unsqueeze(-1) < knots[1:])
        ).to(x.dtype)
        for current_degree in range(1, degree + 1):
            output_count = knots.numel() - current_degree - 1
            left_denominator = (
                knots[current_degree : current_degree + output_count]
                - knots[:output_count]
            )
            right_denominator = (
                knots[current_degree + 1 : current_degree + 1 + output_count]
                - knots[1 : 1 + output_count]
            )
            left = (
                (x.unsqueeze(-1) - knots[:output_count])
                / left_denominator.clamp_min(torch.finfo(x.dtype).eps)
                * values[..., :output_count]
            )
            right = (
                (knots[current_degree + 1 : current_degree + 1 + output_count]
                 - x.unsqueeze(-1))
                / right_denominator.clamp_min(torch.finfo(x.dtype).eps)
                * values[..., 1 : output_count + 1]
            )
            left = torch.where(left_denominator > 0, left, torch.zeros_like(left))
            right = torch.where(
                right_denominator > 0, right, torch.zeros_like(right)
            )
            values = left + right

        endpoint = x >= 1.0
        if endpoint.any():
            last = F.one_hot(
                torch.full_like(
                    x,
                    min(self.rank - 1, values.shape[-1] - 1),
                    dtype=torch.long,
                ),
                num_classes=values.shape[-1],
            ).to(values.dtype)
            values = torch.where(endpoint.unsqueeze(-1), last, values)
        return values

    def _derivative(self, x: Tensor, degree: int) -> Tensor:
        if degree == 0:
            return x.new_zeros((*x.shape, self.knots().numel() - 1))
        knots = self.knots().to(dtype=x.dtype, device=x.device)
        lower = self._basis_at_degree(x, degree - 1)
        output_count = knots.numel() - degree - 1
        left_denominator = knots[degree : degree + output_count] - knots[:output_count]
        right_denominator = (
            knots[degree + 1 : degree + 1 + output_count]
            - knots[1 : 1 + output_count]
        )
        left = degree * lower[..., :output_count] / left_denominator.clamp_min(
            torch.finfo(x.dtype).eps
        )
        right = degree * lower[..., 1 : output_count + 1] / right_denominator.clamp_min(
            torch.finfo(x.dtype).eps
        )
        left = torch.where(left_denominator > 0, left, torch.zeros_like(left))
        right = torch.where(right_denominator > 0, right, torch.zeros_like(right))
        return left - right

    def _second_derivative(self, x: Tensor) -> Tensor:
        if self.degree < 2:
            return x.new_zeros((*x.shape, self.rank))
        knots = self.knots().to(dtype=x.dtype, device=x.device)
        lower_derivative = self._derivative(x, self.degree - 1)
        output_count = self.rank
        left_denominator = (
            knots[self.degree : self.degree + output_count]
            - knots[:output_count]
        )
        right_denominator = (
            knots[self.degree + 1 : self.degree + 1 + output_count]
            - knots[1 : 1 + output_count]
        )
        left = (
            self.degree
            * lower_derivative[..., :output_count]
            / left_denominator.clamp_min(torch.finfo(x.dtype).eps)
        )
        right = (
            self.degree
            * lower_derivative[..., 1 : output_count + 1]
            / right_denominator.clamp_min(torch.finfo(x.dtype).eps)
        )
        left = torch.where(left_denominator > 0, left, torch.zeros_like(left))
        right = torch.where(right_denominator > 0, right, torch.zeros_like(right))
        return left - right

    @staticmethod
    def _quintic_extension(
        value: Tensor,
        first: Tensor,
        second: Tensor,
        normalized_distance: Tensor,
    ) -> Tensor:
        """Quintic Hermite continuation to zero with matched value/d1/d2."""
        u = normalized_distance.clamp(0.0, 1.0).unsqueeze(-1)
        u2, u3 = u.square(), u.square() * u
        u4, u5 = u3 * u, u3 * u.square()
        h00 = 1.0 - 10.0 * u3 + 15.0 * u4 - 6.0 * u5
        h10 = u - 6.0 * u3 + 8.0 * u4 - 3.0 * u5
        h20 = 0.5 * (u2 - 3.0 * u3 + 3.0 * u4 - u5)
        return value * h00 + first * h10 + second * h20

    def extrapolation_safe(self, t: Tensor, *, fade_width: float) -> Tensor:
        """Evaluate in-window and use a C2 finite-support boundary extension."""
        if fade_width <= 0:
            raise ValueError("fade_width must be positive")
        inside_time = t.clamp(0.0, 1.0)
        output = self._basis_at_degree(inside_time, self.degree)
        for boundary, side in ((0.0, -1.0), (1.0, 1.0)):
            mask = t < 0.0 if boundary == 0.0 else t > 1.0
            if not mask.any():
                continue
            boundary_time = t.new_full(t.shape, boundary)
            value = self._basis_at_degree(boundary_time, self.degree)
            first = self._derivative(boundary_time, self.degree)
            second = self._second_derivative(boundary_time)
            distance = (side * (t - boundary)) / fade_width
            extension = self._quintic_extension(
                value,
                side * fade_width * first,
                fade_width**2 * second,
                distance,
            )
            output = torch.where(mask.unsqueeze(-1), extension, output)
        return output

    def forward(self, t: Tensor, latent: Tensor | None = None) -> Tensor:
        output = self._basis_at_degree(t.clamp(0.0, 1.0), self.degree)
        if latent is not None:
            output = output * latent.to(dtype=output.dtype, device=output.device)
        return output


class AnchoredSplineTemporalBasis(nn.Module):
    """Full analytic anchor plus a gated, extrapolation-safe local residual."""

    def __init__(
        self,
        rank: int = 8,
        *,
        local_degree: int,
        adaptive_knots: bool,
        extrapolation_fade: float = 0.1,
        gate_initial_probability: float = 0.05,
    ) -> None:
        super().__init__()
        if not 0.0 < gate_initial_probability < 1.0:
            raise ValueError("gate_initial_probability must lie in (0, 1)")
        self.rank = rank
        self.anchor = AnalyticTemporalBasis(rank)
        self.local = BSplineTemporalBasis(
            rank,
            degree=local_degree,
            adaptive_knots=adaptive_knots,
        )
        gate_logit = math.log(
            gate_initial_probability / (1.0 - gate_initial_probability)
        )
        self.raw_gate = nn.Parameter(torch.full((rank,), gate_logit))
        indices = torch.arange(rank, dtype=torch.float32)
        frequencies = indices.unsqueeze(0)
        positions = (indices.unsqueeze(1) + 0.5) / rank
        projection = torch.cos(math.pi * positions * frequencies)
        projection[:, 0] *= math.sqrt(1.0 / rank)
        projection[:, 1:] *= math.sqrt(2.0 / rank)
        self.register_buffer("local_projection", projection)
        self.register_buffer("residual_scale", torch.tensor(1.0))
        self.extrapolation_fade = float(extrapolation_fade)

    def set_residual_scale(self, scale: float) -> None:
        self.residual_scale.fill_(float(max(0.0, min(scale, 1.0))))

    def local_features(self, t: Tensor) -> Tensor:
        local = self.local.extrapolation_safe(
            t, fade_width=self.extrapolation_fade
        )
        return F.linear(local, self.local_projection)

    def local_component(self, t: Tensor) -> Tensor:
        projected = self.local_features(t)
        gate = torch.sigmoid(self.raw_gate).to(projected.dtype)
        return projected * gate * self.residual_scale.to(projected.dtype)

    def regularization_loss(self) -> Tensor:
        return self.local.regularization_loss()

    def forward(self, t: Tensor, latent: Tensor | None = None) -> Tensor:
        anchor = self.anchor(t)
        mixed = anchor + self.local_component(t)
        anchor_norm = anchor.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        mixed_norm = mixed.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        output = mixed * (anchor_norm / mixed_norm)
        if latent is not None:
            output = output * latent.to(dtype=output.dtype, device=output.device)
        return output


class HighOrderTemporalRefiner(nn.Module):
    """Projected high-frequency correction with compact C2 extrapolation.

    The public seam deliberately stays small: callers provide the two existing
    temporal anchors, while candidate construction, ridge projection, output
    normalization and the boundary envelope remain private.  The 24 candidate
    atoms (six Fourier pairs and twelve adaptive quintic splines) are mixed to
    the field's unchanged rank-eight temporal interface.
    """

    def __init__(
        self,
        rank: int = 8,
        *,
        num_frames: int = 51,
        fourier_pairs: int = 6,
        spline_atoms: int = 12,
        min_cycles: float = 5.0,
        max_cycles: float = 20.0,
        projection_epsilon: float = 1e-4,
        extrapolation_fade: float = 0.1,
    ) -> None:
        super().__init__()
        if rank < 1 or num_frames < 2:
            raise ValueError("rank and num_frames must be positive")
        if fourier_pairs < 1 or spline_atoms <= 5:
            raise ValueError("the high-order dictionary requires Fourier and spline atoms")
        if not 0.0 < min_cycles < max_cycles:
            raise ValueError("Fourier cycle bounds must be positive and ordered")
        if projection_epsilon <= 0.0 or extrapolation_fade <= 0.0:
            raise ValueError("projection epsilon and fade width must be positive")

        self.rank = rank
        self.num_frames = num_frames
        self.fourier_pairs = fourier_pairs
        self.spline_atoms = spline_atoms
        self.min_cycles = float(min_cycles)
        self.max_cycles = float(max_cycles)
        self.projection_epsilon = float(projection_epsilon)
        self.extrapolation_fade = float(extrapolation_fade)
        self.local = BSplineTemporalBasis(
            spline_atoms,
            degree=5,
            adaptive_knots=True,
            minimum_spacing=0.01,
        )

        # Keep initialization strictly within the constrained interval while
        # spanning practically the complete agreed 5--20 cycles/clip band.
        margin = 1e-2
        initial_cycles = torch.linspace(
            min_cycles + margin, max_cycles - margin, fourier_pairs
        )
        initial_probability = (initial_cycles - min_cycles) / (
            max_cycles - min_cycles
        )
        self.raw_frequency = nn.Parameter(
            torch.logit(initial_probability.clamp(1e-5, 1.0 - 1e-5))
        )

        candidate_count = 2 * fourier_pairs + spline_atoms
        self.mixing = nn.Parameter(torch.empty(candidate_count, rank))
        nn.init.orthogonal_(self.mixing)
        self.register_buffer("reference_times", torch.linspace(0.0, 1.0, num_frames))
        self.register_buffer("progress", torch.tensor(1.0))
        self.register_buffer(
            "last_gram_loss", torch.tensor(0.0), persistent=False
        )
        self.register_buffer(
            "last_mixing_energy", torch.tensor(0.0), persistent=False
        )
        self.register_buffer(
            "last_curvature_loss", torch.tensor(0.0), persistent=False
        )
        self._last_regularization: Tensor | None = None

    @property
    def frequency_cycles(self) -> Tensor:
        return self.min_cycles + (self.max_cycles - self.min_cycles) * torch.sigmoid(
            self.raw_frequency
        )

    def set_progress(self, scale: float) -> None:
        self.progress.fill_(float(max(0.0, min(scale, 1.0))))

    def _candidate_dictionary(self, t: Tensor) -> Tensor:
        flat_time = t.reshape(-1)
        phase = 2.0 * math.pi * flat_time.unsqueeze(-1) * self.frequency_cycles
        fourier = torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1).flatten(-2)
        local = self.local.extrapolation_safe(
            flat_time, fade_width=self.extrapolation_fade
        )
        return torch.cat((fourier, local), dim=-1)

    def _boundary_envelope(self, t: Tensor) -> Tensor:
        flat_time = t.reshape(-1)
        distance = torch.where(
            flat_time < 0.0,
            -flat_time,
            torch.where(flat_time > 1.0, flat_time - 1.0, torch.zeros_like(flat_time)),
        )
        u = (distance / self.extrapolation_fade).clamp(0.0, 1.0)
        smooth = 1.0 - 10.0 * u.pow(3) + 15.0 * u.pow(4) - 6.0 * u.pow(5)
        return torch.where(distance < self.extrapolation_fade, smooth, torch.zeros_like(smooth))

    @staticmethod
    def _joint_anchor(
        t: Tensor,
        plane_anchor: nn.Module,
        hash_anchor: nn.Module,
    ) -> Tensor:
        return torch.cat((plane_anchor(t), hash_anchor(t)), dim=-1)

    def _projected_dictionary(
        self,
        t: Tensor,
        plane_anchor: nn.Module,
        hash_anchor: nn.Module,
    ) -> tuple[Tensor, Tensor, Tensor]:
        reference_times = self.reference_times.to(device=t.device, dtype=t.dtype)
        base_reference = self._joint_anchor(
            reference_times, plane_anchor, hash_anchor
        )
        candidate_reference = self._candidate_dictionary(reference_times)
        base_query = self._joint_anchor(t.reshape(-1), plane_anchor, hash_anchor)
        candidate_query = self._candidate_dictionary(t)

        # CUDA linalg is intentionally evaluated in fp32 under AMP.  The solve
        # stays differentiable with respect to both the anchors and dictionary.
        with torch.cuda.amp.autocast(enabled=False):
            base32 = base_reference.float()
            candidate32 = candidate_reference.float()
            identity = torch.eye(
                base32.shape[-1], device=base32.device, dtype=base32.dtype
            )
            projection = torch.linalg.solve(
                base32.T @ base32 + self.projection_epsilon * identity,
                base32.T @ candidate32,
            )
            reference_orthogonal = candidate32 - base32 @ projection
            query_orthogonal = candidate_query.float() - base_query.float() @ projection
        return query_orthogonal, reference_orthogonal, base32

    def forward(
        self,
        t: Tensor,
        plane_anchor: nn.Module,
        hash_anchor: nn.Module,
    ) -> Tensor:
        if float(self.progress) <= 0.0:
            self._last_regularization = None
            self.last_gram_loss.zero_()
            self.last_mixing_energy.zero_()
            self.last_curvature_loss.zero_()
            return t.new_zeros((t.numel(), self.rank))
        query_dictionary, reference_dictionary, base_reference = (
            self._projected_dictionary(t, plane_anchor, hash_anchor)
        )
        reference_output = reference_dictionary @ self.mixing.float()
        output_rms = reference_output.square().mean(dim=0).sqrt().clamp_min(1e-4)
        output = (query_dictionary @ self.mixing.float()) / output_rms

        base_normalized = base_reference / base_reference.square().mean(
            dim=0, keepdim=True
        ).sqrt().clamp_min(1e-4)
        output_normalized = reference_output / output_rms
        gram_loss = (
            base_normalized.T @ output_normalized / self.num_frames
        ).square().mean()
        mixing_energy = self.mixing.float().square().mean()
        second_difference = (
            output_normalized[2:] - 2.0 * output_normalized[1:-1] + output_normalized[:-2]
        )
        curvature_loss = second_difference.square().mean()
        self._last_regularization = (
            1e-3 * gram_loss + 1e-4 * mixing_energy + 1e-5 * curvature_loss
        )
        self.last_gram_loss.copy_(gram_loss.detach().to(self.last_gram_loss))
        self.last_mixing_energy.copy_(
            mixing_energy.detach().to(self.last_mixing_energy)
        )
        self.last_curvature_loss.copy_(
            curvature_loss.detach().to(self.last_curvature_loss)
        )

        envelope = self._boundary_envelope(t).float().unsqueeze(-1)
        output = output * envelope * self.progress.float()
        return output.to(dtype=t.dtype)

    def regularization_loss(self) -> Tensor:
        if self._last_regularization is None:
            return self.mixing.new_zeros(())
        return self._last_regularization


class JordanOrderOneTemporalBasis(nn.Module):
    """Four physical-time spectral modes at the default rank-eight budget.

    An order-two Jordan block spends four real channels on each complex
    frequency (cos, sin and their two polynomial companions), leaving only two
    independent frequencies at rank eight.  This order-one control keeps the
    same rank and 16 trainable basis scalars but assigns two real channels to
    each frequency, restoring four independent frequencies.  Phase is fixed at
    zero so the basis budget remains exactly matched to the other controls.
    """

    def __init__(
        self,
        rank: int = 8,
        *,
        num_frames: int = 51,
        frame_interval: float = 0.1,
    ) -> None:
        super().__init__()
        if rank < 2 or rank % 2:
            raise ValueError("temporal rank must be a positive even number")
        if num_frames < 2 or frame_interval <= 0:
            raise ValueError("physical clip time requires at least two frames")

        self.rank = rank
        self.spectral_mode_count = rank // 2
        extent = (num_frames - 1) * frame_interval
        self.raw_sigma = nn.Parameter(torch.zeros(self.spectral_mode_count))
        self.raw_omega = nn.Parameter(torch.zeros(self.spectral_mode_count))
        self.raw_gain = nn.Parameter(torch.zeros(rank))
        self.register_buffer("extent", torch.tensor(float(extent)))
        self.register_buffer("sigma_max", torch.tensor(float(2.0 / extent)))
        self.register_buffer(
            "omega_max", torch.tensor(float(math.pi / frame_interval))
        )

        harmonics = (
            torch.arange(1, self.spectral_mode_count + 1, dtype=torch.float32)
            * math.pi
            / extent
        )
        ratio = (harmonics / self.omega_max).clamp(max=0.995)
        with torch.no_grad():
            self.raw_omega.copy_(torch.atanh(ratio))

    def forward(self, t: Tensor, latent: Tensor | None = None) -> Tensor:
        seconds = (t - 0.5) * self.extent
        sigma = self.sigma_max * torch.tanh(self.raw_sigma)
        omega = self.omega_max * torch.tanh(self.raw_omega)
        envelope = torch.exp(seconds.unsqueeze(-1) * sigma)
        phase = seconds.unsqueeze(-1) * omega
        output = torch.stack(
            [envelope * torch.cos(phase), envelope * torch.sin(phase)], dim=-1
        ).flatten(-2)
        output = output * (1.0 + 0.1 * torch.tanh(self.raw_gain))
        if latent is not None:
            output = output * latent.to(dtype=output.dtype, device=output.device)
        return output


class HybridGlobalLocalTemporalBasis(nn.Module):
    """Parameter-matched continuous/local basis at fixed coefficient rank.

    The default rank-eight basis uses four real channels for two physical-time
    complex frequencies and four channels for local hat functions.  It thus
    provides six independently controlled temporal modes without increasing
    the plane/hash coefficient width or the 16-scalar basis budget.
    """

    def __init__(
        self,
        rank: int = 8,
        *,
        num_frames: int = 51,
        frame_interval: float = 0.1,
    ) -> None:
        super().__init__()
        if rank < 8 or rank % 4:
            raise ValueError("hybrid temporal rank must be a multiple of four >= 8")
        self.rank = rank
        self.global_rank = rank // 2
        self.local_rank = rank - self.global_rank
        self.global_basis = JordanOrderOneTemporalBasis(
            self.global_rank,
            num_frames=num_frames,
            frame_interval=frame_interval,
        )
        self.local_basis = DiscreteTemporalBasis(self.local_rank)
        self.spectral_mode_count = self.global_basis.spectral_mode_count
        self.local_mode_count = self.local_rank
        self.independent_mode_count = (
            self.spectral_mode_count + self.local_mode_count
        )

    def forward(self, t: Tensor, latent: Tensor | None = None) -> Tensor:
        global_latent = local_latent = None
        if latent is not None:
            global_latent = latent[..., : self.global_rank]
            local_latent = latent[..., self.global_rank :]
        return torch.cat(
            [
                self.global_basis(t, latent=global_latent),
                self.local_basis(t, latent=local_latent),
            ],
            dim=-1,
        )


class JordanTemporalBasis(nn.Module):
    """All-oscillatory Koopman/Jordan basis evaluated in physical seconds.

    LiDAR4D supplies normalized clip time, so this module is the only place that
    converts it back to a centred physical timestamp.  A rank-eight default is
    two complex modes with Jordan order two, rather than four unrelated Fourier
    modes.  This matches the Basis-Occ effective-rank accounting.
    """

    def __init__(
        self,
        rank: int = 8,
        *,
        num_frames: int = 51,
        frame_interval: float = 0.1,
        jordan_order: int = 2,
    ) -> None:
        super().__init__()
        if jordan_order < 1:
            raise ValueError("jordan_order must be positive")
        channels_per_mode = 2 * jordan_order
        if rank < channels_per_mode or rank % channels_per_mode:
            raise ValueError(
                "rank must be divisible by two times the Jordan order"
            )
        if num_frames < 2 or frame_interval <= 0:
            raise ValueError("physical clip time requires at least two frames")

        modes = rank // channels_per_mode
        extent = (num_frames - 1) * frame_interval
        self.rank = rank
        self.jordan_order = jordan_order
        self.raw_sigma = nn.Parameter(torch.zeros(modes))
        self.raw_omega = nn.Parameter(torch.zeros(modes))
        self.raw_phase = nn.Parameter(torch.zeros(modes))
        self.raw_order_scale = nn.Parameter(
            torch.zeros(modes, max(jordan_order - 1, 0))
        )
        self.raw_gain = nn.Parameter(torch.zeros(rank))
        self.register_buffer("extent", torch.tensor(float(extent)))
        self.register_buffer("tau", torch.tensor(float(extent / 2.0)))
        self.register_buffer("sigma_max", torch.tensor(float(2.0 / extent)))
        self.register_buffer("omega_max", torch.tensor(float(math.pi / frame_interval)))

        harmonics = torch.arange(1, modes + 1, dtype=torch.float32) * math.pi / extent
        ratio = (harmonics / self.omega_max).clamp(max=0.995)
        with torch.no_grad():
            self.raw_omega.copy_(torch.atanh(ratio))

    def forward(self, t: Tensor, latent: Tensor | None = None) -> Tensor:
        seconds = (t - 0.5) * self.extent
        scaled = seconds / self.tau
        polynomials = [torch.ones_like(scaled)]
        for order in range(1, self.jordan_order):
            polynomials.append(polynomials[-1] * scaled / order)
        polynomial = torch.stack(polynomials, dim=-1)

        sigma = self.sigma_max * torch.tanh(self.raw_sigma)
        omega = self.omega_max * torch.tanh(self.raw_omega)
        envelope = torch.exp(seconds.unsqueeze(-1) * sigma)
        phase = seconds.unsqueeze(-1) * omega + self.raw_phase
        cosine = envelope * torch.cos(phase)
        sine = envelope * torch.sin(phase)

        parts = []
        for mode in range(omega.numel()):
            for order in range(self.jordan_order):
                order_scale = (
                    polynomial.new_ones(())
                    if order == 0
                    else 1.0
                    + 0.1 * torch.tanh(self.raw_order_scale[mode, order - 1])
                )
                parts.append(
                    cosine[..., mode] * polynomial[..., order] * order_scale
                )
                parts.append(sine[..., mode] * polynomial[..., order] * order_scale)
        output = torch.stack(parts, dim=-1)
        output = output * (1.0 + 0.1 * torch.tanh(self.raw_gain))
        if latent is not None:
            output = output * latent.to(dtype=output.dtype, device=output.device)
        return output


class StochasticJordanTemporalBasis(JordanTemporalBasis):
    """Low-rank posterior over temporal mode gains.

    The large plane/hash coefficient tensors remain posterior means, so the
    deterministic evaluation field keeps the complete matched capacity. Only
    one diagonal posterior over the effective temporal rank is added. A single
    sample is reused for current and flow-warped neighbour queries.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latent_mu = nn.Parameter(torch.ones(self.rank))
        self.latent_logvar = nn.Parameter(torch.full((self.rank,), -8.0))

    def sample_latent(self) -> Tensor:
        if not self.training:
            return self.latent_mu
        logvar = self.latent_logvar.clamp(-30.0, 20.0)
        return self.latent_mu + torch.randn_like(self.latent_mu) * torch.exp(
            0.5 * logvar
        )

    def kl_loss(self) -> Tensor:
        logvar = self.latent_logvar.clamp(-30.0, 20.0)
        return -0.5 * (
            1.0 + logvar - self.latent_mu.square() - logvar.exp()
        ).mean()


class TemporalQueryContext(NamedTuple):
    """Internal per-query state reused for current and flow-warped neighbours."""

    basis_latent: Tensor | None = None
    plane_residual: Tensor | None = None
    hash_residual: Tensor | None = None
    high_order_cache: dict[tuple[int, float, float], Tensor] | None = None
    wavelet_cache: dict[object, Tensor] | None = None


class SceneFieldFeatures(NamedTuple):
    """Four spatial roles produced before the scene-field fusion seam."""

    plane_static: Tensor
    plane_dynamic: Tensor
    hash_static: Tensor
    hash_dynamic: Tensor

    def concatenate(self) -> Tensor:
        return torch.cat(self, dim=-1)


class SceneFieldDecomposition(NamedTuple):
    """Decoder-ready base/full features plus detached motion evidence."""

    base: Tensor
    full: Tensor
    motion_prior: Tensor
    motion_mask: Tensor
    specialized: bool


class FeatureFusion(nn.Module):
    """Parameter-matched fusion over the four scene-field feature roles.

    Every mode owns the same ``[output_dim, sum(input_dims)]`` projection and
    LayerNorm. The weight is split internally for branch-wise modes, so changing
    ``mode`` changes only the fusion algebra, never capacity or decoder shape.
    """

    MODES = ("concat", "sum", "hadamard")

    def __init__(
        self,
        input_dims: tuple[int, int, int, int],
        output_dim: int,
        *,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown fusion {mode!r}; expected one of {self.MODES}")
        if len(input_dims) != 4 or any(width <= 0 for width in input_dims):
            raise ValueError("fusion expects four positive input dimensions")
        if output_dim <= 0:
            raise ValueError("fusion output dimension must be positive")
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mode = mode
        self.weight = nn.Parameter(torch.empty(output_dim, sum(input_dims)))
        self.output_norm = nn.LayerNorm(output_dim)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: SceneFieldFeatures) -> Tensor:
        parts = tuple(features)
        observed = tuple(part.shape[-1] for part in parts)
        if observed != self.input_dims:
            raise ValueError(
                f"fusion expected feature widths {self.input_dims}, got {observed}"
            )
        if self.mode == "concat":
            fused = torch.tanh(F.linear(features.concatenate(), self.weight))
        else:
            weights = self.weight.split(self.input_dims, dim=1)
            projected = [
                torch.tanh(F.linear(part.to(weight.dtype), weight))
                for part, weight in zip(parts, weights)
            ]
            if self.mode == "sum":
                fused = torch.stack(projected).sum(dim=0) / math.sqrt(len(projected))
            else:
                fused = torch.ones_like(projected[0])
                for feature in projected:
                    fused = fused * (1.0 + feature)
                fused = fused - 1.0
        return self.output_norm(fused)


class FeatureFlowHead(nn.Module):
    """Predict adjacent-frame forward/backward flow from a queried field feature."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),
        )
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class SceneField(nn.Module):
    """Deep interface between LiDAR4D's renderer and a scene representation."""

    n_output_dims: int

    def query(self, xyz: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError

    def query_decomposition(self, xyz: Tensor, t: Tensor) -> SceneFieldDecomposition:
        """Return an inert decomposition for fields without a residual seam."""
        full = self.query(xyz, t)
        zeros = full.new_zeros(full.shape[0])
        return SceneFieldDecomposition(full, full, zeros, zeros, False)

    def flow(self, xyz: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError

    def kl_loss(self) -> Tensor:
        parameter = next(self.parameters())
        return parameter.new_zeros(())

    def basis_flow_loss(self, xyz: Tensor, t: Tensor) -> Tensor:
        """Consistency hook; non-modal fields have no basis-flow coupling."""
        return xyz.new_zeros(())

    def set_training_progress(
        self,
        *,
        spline: float = 1.0,
        high_order: float = 1.0,
        stochastic: float = 1.0,
    ) -> None:
        """Apply representation-level schedule weights; non-modal fields ignore them."""

    def temporal_regularization_loss(self) -> Tensor:
        """Return representation-internal temporal regularization."""
        parameter = next(self.parameters())
        return parameter.new_zeros(())

    def high_order_regularization_loss(self) -> Tensor:
        """Return high-order dictionary/feature regularization when available."""
        parameter = next(self.parameters())
        return parameter.new_zeros(())

    def high_order_diagnostics(self) -> dict[str, Tensor]:
        """Expose scalar diagnostics without leaking representation internals."""
        return {}

    def configure_motion_prior(self, points_by_frame: dict[int, Tensor]) -> None:
        """Install detached scene evidence when a field supports motion routing."""

    def parameter_groups(self, lr: float) -> list[dict]:
        return [{"params": self.parameters(), "lr": lr}]

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def budget_report(self, name: str | None = None) -> FieldBudgetReport:
        return FieldBudgetReport(
            name=name or type(self).__name__,
            static=0,
            dynamic=0,
            flow=0,
            other=self.parameter_count,
        )


class OfficialSceneField(SceneField):
    """The original LiDAR4D plane/hash/flow field behind the common interface."""

    def __init__(
        self,
        *,
        min_resolution: int = 32,
        base_resolution: int = 512,
        max_resolution: int = 32768,
        time_resolution: int = 8,
        n_levels_plane: int = 4,
        n_features_per_level_plane: int = 8,
        n_levels_hash: int = 8,
        n_features_per_level_hash: int = 4,
        log2_hashmap_size: int = 19,
        num_layers_flow: int = 3,
        hidden_dim_flow: int = 64,
        num_frames: int = 51,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.planes_encoder = Planes4D(
            grid_dimensions=2,
            input_dim=4,
            output_dim=n_features_per_level_plane,
            resolution=[min_resolution] * 3 + [time_resolution],
            multiscale_res=[2**level for level in range(n_levels_plane)],
        )
        self.hash_encoder = HashGrid4D(
            base_resolution=base_resolution,
            max_resolution=max_resolution,
            time_resolution=time_resolution,
            n_levels=n_levels_hash,
            n_features_per_level=n_features_per_level_hash,
            log2_hashmap_size=log2_hashmap_size,
        )
        self.flow_net = FlowField(
            input_dim=4,
            num_layers=num_layers_flow,
            hidden_dim=hidden_dim_flow,
            use_grid=True,
        )
        self.n_output_dims = (
            self.planes_encoder.n_output_dims + self.hash_encoder.n_output_dims
        )

    def _xt(self, xyz: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        per_point = _time_per_point(t, xyz.shape[0])
        return torch.cat([xyz, per_point.unsqueeze(-1)], dim=-1), per_point

    def query(self, xyz: Tensor, t: Tensor) -> Tensor:
        frame_idx = int(t.reshape(-1)[0] * (self.num_frames - 1))
        hash_static, hash_dynamic = self.hash_encoder(xyz, t.reshape(-1)[0])
        xt, _ = self._xt(xyz, t)
        plane_static, plane_dynamic = self.planes_encoder(xt)

        flow = self.flow_net(xt)
        hash_next = hash_previous = hash_dynamic
        plane_next = plane_previous = plane_dynamic

        if frame_idx < self.num_frames - 1:
            xyz_next = xyz + flow[:, :3]
            t_next = xyz.new_tensor(
                _normalized_frame_time(frame_idx + 1, self.num_frames)
            )
            with torch.no_grad():
                hash_next = self.hash_encoder.forward_dynamic(xyz_next, t_next)
            xt_next, _ = self._xt(xyz_next, t_next)
            plane_next = self.planes_encoder.forward_dynamic(xt_next)

        if frame_idx > 0:
            xyz_previous = xyz + flow[:, 3:]
            t_previous = xyz.new_tensor(
                _normalized_frame_time(frame_idx - 1, self.num_frames)
            )
            with torch.no_grad():
                hash_previous = self.hash_encoder.forward_dynamic(
                    xyz_previous, t_previous
                )
            xt_previous, _ = self._xt(xyz_previous, t_previous)
            plane_previous = self.planes_encoder.forward_dynamic(xt_previous)

        plane_dynamic = 0.5 * plane_dynamic + 0.25 * (plane_next + plane_previous)
        hash_dynamic = 0.5 * hash_dynamic + 0.25 * (hash_next + hash_previous)
        return torch.cat(
            [plane_static, plane_dynamic, hash_static, hash_dynamic], dim=-1
        )

    def flow(self, xyz: Tensor, t: Tensor) -> Tensor:
        xt, _ = self._xt(xyz, t)
        return self.flow_net(xt)

    def parameter_groups(self, lr: float) -> list[dict]:
        return [
            {"params": self.planes_encoder.parameters(), "lr": lr},
            {"params": self.hash_encoder.parameters(), "lr": lr},
            {"params": self.flow_net.parameters(), "lr": 0.1 * lr},
        ]

    def budget_report(self, name: str | None = None) -> FieldBudgetReport:
        static_plane_indices = (0, 1, 3)
        dynamic_plane_indices = (2, 4, 5)
        static_planes = sum(
            level[index].numel()
            for level in self.planes_encoder.planes
            for index in static_plane_indices
        )
        dynamic_planes = sum(
            level[index].numel()
            for level in self.planes_encoder.planes
            for index in dynamic_plane_indices
        )
        return FieldBudgetReport(
            name=name or "official",
            static=static_planes + count_parameters(self.hash_encoder.hash_static),
            dynamic=dynamic_planes
            + count_parameters(self.hash_encoder.hash_dynamic),
            flow=count_parameters(self.flow_net),
        )


class BasisTimeOnlyField(SceneField):
    """LiDAR4D spatial/flow field with only its temporal grids replaced.

    The static planes, static 3D hash, feature dimensions, multi-resolution
    schedule, dedicated FlowField and flow-guided neighbour aggregation match
    the official field.  Dynamic axis planes and dynamic 2D hashes store modal
    coefficients instead of eight discrete time knots.  Their sub-budgets are
    matched separately to the official temporal planes/hashes.
    """

    def __init__(
        self,
        *,
        min_resolution: int = 32,
        base_resolution: int = 512,
        max_resolution: int = 32768,
        time_resolution: int = 8,
        n_levels_plane: int = 4,
        n_features_per_level_plane: int = 8,
        n_levels_hash: int = 8,
        n_features_per_level_hash: int = 4,
        log2_hashmap_size: int = 19,
        num_layers_flow: int = 3,
        hidden_dim_flow: int = 64,
        num_frames: int = 51,
        frame_interval: float = 0.1,
        jordan_order: int = 2,
        stochastic_temporal: bool = False,
        basis_kind: str = "jordan",
    ) -> None:
        super().__init__()
        rank = time_resolution
        self.num_frames = num_frames
        self.rank = rank
        self.n_levels_hash = n_levels_hash
        if stochastic_temporal and basis_kind != "jordan":
            raise ValueError("stochastic temporal gains are only defined for Jordan")
        self.basis_kind = basis_kind
        if basis_kind == "discrete":
            self.basis = DiscreteTemporalBasis(rank)
        elif basis_kind == "analytic":
            self.basis = AnalyticTemporalBasis(rank)
        elif basis_kind == "jordan_o1":
            self.basis = JordanOrderOneTemporalBasis(
                rank,
                num_frames=num_frames,
                frame_interval=frame_interval,
            )
        elif basis_kind == "hybrid":
            self.basis = HybridGlobalLocalTemporalBasis(
                rank,
                num_frames=num_frames,
                frame_interval=frame_interval,
            )
        elif basis_kind == "jordan":
            basis_type = (
                StochasticJordanTemporalBasis
                if stochastic_temporal
                else JordanTemporalBasis
            )
            self.basis = basis_type(
                rank,
                num_frames=num_frames,
                frame_interval=frame_interval,
                jordan_order=jordan_order,
            )
        else:
            raise ValueError(
                "basis_kind must be one of: discrete, analytic, jordan_o1, "
                "hybrid, jordan"
            )

        self.static_planes = nn.ModuleList()
        self.modal_axes = nn.ModuleList()
        for level in range(n_levels_plane):
            resolution = min_resolution * (2**level)
            static_level = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.empty(
                            n_features_per_level_plane, resolution, resolution
                        )
                    )
                    for _ in range(3)
                ]
            )
            modal_level = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.empty(
                            n_features_per_level_plane, rank, resolution
                        )
                    )
                    for _ in range(3)
                ]
            )
            for plane in static_level:
                nn.init.uniform_(plane, a=0.1, b=0.5)
            for axis in modal_level:
                nn.init.normal_(axis, std=1e-3)
            self.static_planes.append(static_level)
            self.modal_axes.append(modal_level)

        per_level_scale = 2.0 ** (
            math.log2(max_resolution / base_resolution) / (n_levels_hash - 1)
        )
        self.hash_static = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels_hash,
                "n_features_per_level": n_features_per_level_hash,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
        )

        # One coefficient hash replaces all temporal hash knots. Raising the
        # table exponent by log2(T*C/R) keeps each official dynamic-hash
        # sub-budget exact at the default T=R=8 and C=4.
        size_shift = round(
            math.log2(time_resolution * n_features_per_level_hash / rank)
        )
        modal_hash_sizes = [15 + size_shift, 13 + size_shift, 13 + size_shift]
        self.modal_hashes = nn.ModuleList(
            [
                tcnn.Encoding(
                    n_input_dims=2,
                    encoding_config={
                        "otype": "HashGrid",
                        "n_levels": n_levels_hash,
                        "n_features_per_level": rank,
                        "log2_hashmap_size": table_size,
                        "base_resolution": base_resolution,
                        "per_level_scale": per_level_scale,
                    },
                )
                for table_size in modal_hash_sizes
            ]
        )
        self.flow_net = FlowField(
            input_dim=4,
            num_layers=num_layers_flow,
            hidden_dim=hidden_dim_flow,
            use_grid=True,
        )

        plane_dims = n_levels_plane * n_features_per_level_plane
        hash_static_dims = n_levels_hash * n_features_per_level_hash
        hash_dynamic_dims = 3 * n_levels_hash
        self.feature_dims = (
            plane_dims,
            plane_dims,
            hash_static_dims,
            hash_dynamic_dims,
        )
        self.n_output_dims = 2 * plane_dims + hash_static_dims + hash_dynamic_dims

    def _xt(self, xyz: Tensor, t: Tensor) -> Tensor:
        per_point = _time_per_point(t, xyz.shape[0])
        return torch.cat([xyz, per_point.unsqueeze(-1)], dim=-1)

    def _sample_query_context(self) -> TemporalQueryContext:
        latent = (
            self.basis.sample_latent()
            if isinstance(self.basis, StochasticJordanTemporalBasis)
            else None
        )
        return TemporalQueryContext(basis_latent=latent)

    def _plane_static(self, xyz: Tensor) -> Tensor:
        coordinates = (xyz[:, [0, 1]], xyz[:, [0, 2]], xyz[:, [1, 2]])
        levels = []
        for planes in self.static_planes:
            factors = [
                _sample_plane(plane, coordinate)
                for plane, coordinate in zip(planes, coordinates)
            ]
            levels.append(factors[0] * factors[1] * factors[2])
        return torch.cat(levels, dim=-1)

    def _plane_dynamic(
        self,
        xyz: Tensor,
        t: Tensor,
        context: TemporalQueryContext | None = None,
    ) -> Tensor:
        per_point = _time_per_point(t, xyz.shape[0])
        latent = context.basis_latent if context is not None else None
        phi = self.basis(per_point, latent=latent)
        levels = []
        for axes in self.modal_axes:
            factors = []
            for axis, coordinate in zip(axes, xyz.unbind(-1)):
                sampled = _sample_axis(axis, coordinate)
                dynamic = torch.einsum(
                    "ncr,nr->nc", sampled, phi.to(sampled.dtype)
                )
                factors.append(1.0 + dynamic)
            levels.append(factors[0] * factors[1] * factors[2])
        return torch.cat(levels, dim=-1)

    def _hash_dynamic(
        self,
        xyz: Tensor,
        t: Tensor,
        context: TemporalQueryContext | None = None,
    ) -> Tensor:
        per_point = _time_per_point(t, xyz.shape[0])
        latent = context.basis_latent if context is not None else None
        phi = self.basis(per_point, latent=latent)
        coordinates = (xyz[:, [0, 1]], xyz[:, [0, 2]], xyz[:, [1, 2]])
        features = []
        for encoder, coordinate in zip(self.modal_hashes, coordinates):
            coefficients = encoder(coordinate).reshape(
                -1, self.n_levels_hash, self.rank
            )
            features.append(
                torch.einsum(
                    "nlr,nr->nl", coefficients, phi.to(coefficients.dtype)
                )
            )
        return torch.cat(features, dim=-1)

    def _query_features(
        self,
        xyz: Tensor,
        t: Tensor,
        context: TemporalQueryContext,
    ) -> SceneFieldFeatures:
        frame_idx = int(t.reshape(-1)[0] * (self.num_frames - 1))
        frame_idx = max(0, min(self.num_frames - 1, frame_idx))
        plane_static = self._plane_static(xyz)
        hash_static = self.hash_static(xyz)
        plane_dynamic = self._plane_dynamic(xyz, t, context)
        hash_dynamic = self._hash_dynamic(xyz, t, context)

        flow = self.flow_net(self._xt(xyz, t))
        plane_next = plane_previous = plane_dynamic
        hash_next = hash_previous = hash_dynamic

        if frame_idx < self.num_frames - 1:
            xyz_next = xyz + flow[:, :3]
            t_next = xyz.new_tensor(
                _normalized_frame_time(frame_idx + 1, self.num_frames)
            )
            plane_next = self._plane_dynamic(xyz_next, t_next, context)
            with torch.no_grad():
                hash_next = self._hash_dynamic(xyz_next, t_next, context)

        if frame_idx > 0:
            xyz_previous = xyz + flow[:, 3:]
            t_previous = xyz.new_tensor(
                _normalized_frame_time(frame_idx - 1, self.num_frames)
            )
            plane_previous = self._plane_dynamic(xyz_previous, t_previous, context)
            with torch.no_grad():
                hash_previous = self._hash_dynamic(xyz_previous, t_previous, context)

        plane_dynamic = 0.5 * plane_dynamic + 0.25 * (plane_next + plane_previous)
        hash_dynamic = 0.5 * hash_dynamic + 0.25 * (hash_next + hash_previous)
        return SceneFieldFeatures(
            plane_static=plane_static,
            plane_dynamic=plane_dynamic,
            hash_static=hash_static,
            hash_dynamic=hash_dynamic,
        )

    def query_features(self, xyz: Tensor, t: Tensor) -> SceneFieldFeatures:
        """Return pre-fusion roles while keeping stochastic context internal."""
        return self._query_features(xyz, t, self._sample_query_context())

    def query(self, xyz: Tensor, t: Tensor) -> Tensor:
        return self.query_features(xyz, t).concatenate()

    def flow(self, xyz: Tensor, t: Tensor) -> Tensor:
        return self.flow_net(self._xt(xyz, t))

    def basis_flow_loss(self, xyz: Tensor, t: Tensor) -> Tensor:
        """Align dynamic modal features along the field's predicted flow.

        A single stochastic context is reused on both sides.  Consequently the
        objective measures temporal transport rather than posterior resampling
        noise, and its gradient reaches both the modal bank and FlowField.
        """
        if xyz.numel() == 0:
            return xyz.new_zeros(())
        frame_idx = int(t.reshape(-1)[0] * (self.num_frames - 1))
        context = self._sample_query_context()
        current = torch.cat(
            [
                self._plane_dynamic(xyz, t, context),
                self._hash_dynamic(xyz, t, context),
            ],
            dim=-1,
        ).float()
        flow = self.flow_net(self._xt(xyz, t))
        neighbours = []
        if frame_idx < self.num_frames - 1:
            next_t = xyz.new_tensor(
                _normalized_frame_time(frame_idx + 1, self.num_frames)
            )
            neighbours.append(
                torch.cat(
                    [
                        self._plane_dynamic(xyz + flow[:, :3], next_t, context),
                        self._hash_dynamic(xyz + flow[:, :3], next_t, context),
                    ],
                    dim=-1,
                ).float()
            )
        if frame_idx > 0:
            previous_t = xyz.new_tensor(
                _normalized_frame_time(frame_idx - 1, self.num_frames)
            )
            neighbours.append(
                torch.cat(
                    [
                        self._plane_dynamic(
                            xyz + flow[:, 3:], previous_t, context
                        ),
                        self._hash_dynamic(
                            xyz + flow[:, 3:], previous_t, context
                        ),
                    ],
                    dim=-1,
                ).float()
            )
        if not neighbours:
            return current.new_zeros(())
        return torch.stack(
            [F.smooth_l1_loss(neighbour, current) for neighbour in neighbours]
        ).mean()

    def parameter_groups(self, lr: float) -> list[dict]:
        plane_parameters = list(self.static_planes.parameters())
        plane_parameters += list(self.modal_axes.parameters())
        plane_parameters += list(self.basis.parameters())
        hash_parameters = list(self.hash_static.parameters())
        hash_parameters += list(self.modal_hashes.parameters())
        return [
            {"params": plane_parameters, "lr": lr, "stage_role": "base"},
            {"params": hash_parameters, "lr": lr, "stage_role": "base"},
            {
                "params": self.flow_net.parameters(),
                "lr": 0.1 * lr,
                "stage_role": "geometry_core",
            },
        ]

    def kl_loss(self) -> Tensor:
        if isinstance(self.basis, StochasticJordanTemporalBasis):
            return self.basis.kl_loss()
        return super().kl_loss()

    def budget_report(self, name: str | None = None) -> FieldBudgetReport:
        return FieldBudgetReport(
            name=name or type(self).__name__,
            static=count_parameters(self.static_planes)
            + count_parameters(self.hash_static),
            dynamic=count_parameters(self.modal_axes)
            + count_parameters(self.modal_hashes)
            + count_parameters(self.basis),
            flow=count_parameters(self.flow_net),
        )


class StochasticBasisTimeOnlyField(BasisTimeOnlyField):
    """Capacity-matched basis-time field with a rank-factorized posterior."""

    def __init__(self, **kwargs) -> None:
        super().__init__(stochastic_temporal=True, **kwargs)


class MatchedHybridResidualField(BasisTimeOnlyField):
    """Full-budget Jordan field with a structured stochastic residual.

    The deterministic mean retains LiDAR4D's complete static, dynamic-hash and
    flow sub-budgets. A small residual axis bank corrects the multiscale plane
    branch, while one posterior context modulates both plane residuals and the
    full-resolution modal hashes. The same sample is reused for current and
    flow-warped neighbour queries.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(stochastic_temporal=False, **kwargs)
        self.residual_modal_axes = nn.ModuleList()
        for axes in self.modal_axes:
            residual_level = nn.ParameterList(
                [nn.Parameter(torch.empty_like(axis)) for axis in axes]
            )
            for residual_axis in residual_level:
                nn.init.normal_(residual_axis, std=1e-3)
            self.residual_modal_axes.append(residual_level)
        self.residual_posterior = StochasticResidualPosterior(
            plane_levels=len(self.modal_axes),
            hash_levels=self.n_levels_hash,
            axes=3,
            rank=self.rank,
        )

    def _sample_query_context(self) -> TemporalQueryContext:
        plane_residual, hash_residual = self.residual_posterior.sample()
        return TemporalQueryContext(
            plane_residual=plane_residual,
            hash_residual=hash_residual,
        )

    def _plane_dynamic(
        self,
        xyz: Tensor,
        t: Tensor,
        context: TemporalQueryContext | None = None,
    ) -> Tensor:
        flat_time = t.reshape(-1)
        shared_time = flat_time.numel() == 1
        if shared_time:
            phi = self.basis(flat_time)[0]
        else:
            phi = self.basis(_time_per_point(t, xyz.shape[0]))
        residual_latent = context.plane_residual if context is not None else None

        levels = []
        for level_index, (mean_axes, residual_axes) in enumerate(
            zip(self.modal_axes, self.residual_modal_axes)
        ):
            factors = []
            for axis_index, (mean_axis, residual_axis, coordinate) in enumerate(
                zip(mean_axes, residual_axes, xyz.unbind(-1))
            ):
                latent = (
                    residual_latent[level_index, axis_index]
                    if residual_latent is not None
                    else torch.zeros(self.rank, dtype=phi.dtype, device=phi.device)
                )
                if shared_time:
                    mean = _contract_then_sample_axis(mean_axis, phi, coordinate)
                    residual = _contract_then_sample_axis(
                        residual_axis,
                        phi * latent,
                        coordinate,
                    )
                else:
                    mean_coefficients = _sample_axis(mean_axis, coordinate)
                    residual_coefficients = _sample_axis(residual_axis, coordinate)
                    mean = torch.einsum("ncr,nr->nc", mean_coefficients, phi)
                    residual = torch.einsum(
                        "ncr,nr->nc",
                        residual_coefficients,
                        phi * latent,
                    )
                factors.append(1.0 + mean + residual)
            levels.append(factors[0] * factors[1] * factors[2])
        return torch.cat(levels, dim=-1)

    def _hash_dynamic(
        self,
        xyz: Tensor,
        t: Tensor,
        context: TemporalQueryContext | None = None,
    ) -> Tensor:
        per_point = _time_per_point(t, xyz.shape[0])
        phi = self.basis(per_point)
        residual_latent = context.hash_residual if context is not None else None
        coordinates = (xyz[:, [0, 1]], xyz[:, [0, 2]], xyz[:, [1, 2]])
        features = []
        for axis_index, (encoder, coordinate) in enumerate(
            zip(self.modal_hashes, coordinates)
        ):
            coefficients = encoder(coordinate).reshape(
                -1, self.n_levels_hash, self.rank
            )
            mean = torch.einsum(
                "nlr,nr->nl", coefficients, phi.to(coefficients.dtype)
            )
            if residual_latent is None:
                features.append(mean)
                continue
            weights = phi[:, None, :] * residual_latent[axis_index][None, :, :]
            residual = torch.einsum(
                "nlr,nlr->nl", coefficients, weights.to(coefficients.dtype)
            )
            features.append(mean + residual)
        return torch.cat(features, dim=-1)

    def kl_loss(self) -> Tensor:
        return self.residual_posterior.kl_loss()

    def parameter_groups(self, lr: float) -> list[dict]:
        plane_parameters = list(self.static_planes.parameters())
        plane_parameters += list(self.modal_axes.parameters())
        plane_parameters += list(self.residual_modal_axes.parameters())
        plane_parameters += list(self.basis.parameters())
        plane_parameters += list(self.residual_posterior.parameters())
        hash_parameters = list(self.hash_static.parameters())
        hash_parameters += list(self.modal_hashes.parameters())
        return [
            {"params": plane_parameters, "lr": lr, "stage_role": "base"},
            {"params": hash_parameters, "lr": lr, "stage_role": "base"},
            {
                "params": self.flow_net.parameters(),
                "lr": 0.1 * lr,
                "stage_role": "geometry_core",
            },
        ]

    def budget_report(self, name: str | None = None) -> FieldBudgetReport:
        return FieldBudgetReport(
            name=name or "matched_hybrid_residual",
            static=count_parameters(self.static_planes)
            + count_parameters(self.hash_static),
            dynamic=count_parameters(self.modal_axes)
            + count_parameters(self.modal_hashes)
            + count_parameters(self.basis)
            + count_parameters(self.residual_modal_axes)
            + count_parameters(self.residual_posterior),
            flow=count_parameters(self.flow_net),
        )


class MatchedHybridSceneField(MatchedHybridResidualField):
    """Canonical matched hybrid seam; residual name remains checkpoint-compatible."""


class ModalSpatialField(MatchedHybridSceneField):
    """Matched modal spatial bank with a replaceable, fair fusion algebra.

    Spatial sampling, stochastic residuals, temporal neighbour reuse and flow
    remain hidden behind the standard ``SceneField`` interface. Fusion is the
    only experimental seam: all modes use identical parameters and emit the
    same 120-dimensional decoder feature.
    """

    def __init__(self, *, fusion: str = "concat", **kwargs) -> None:
        super().__init__(**kwargs)
        self.fusion = FeatureFusion(
            self.feature_dims,
            self.n_output_dims,
            mode=fusion,
        )

    @property
    def fusion_mode(self) -> str:
        return self.fusion.mode

    def query(self, xyz: Tensor, t: Tensor) -> Tensor:
        return self.fusion(self.query_features(xyz, t))

    def parameter_groups(self, lr: float) -> list[dict]:
        groups = super().parameter_groups(lr)
        return [
            {
                "params": list(groups[0]["params"])
                + list(self.fusion.parameters()),
                "lr": groups[0]["lr"],
            },
            groups[1],
            groups[2],
        ]

    def budget_report(self, name: str | None = None) -> FieldBudgetReport:
        base = super().budget_report(name=name or f"modal_spatial_{self.fusion_mode}")
        plane_static, plane_dynamic, hash_static, hash_dynamic = self.feature_dims
        static_projection = self.n_output_dims * (plane_static + hash_static)
        dynamic_projection = self.n_output_dims * (plane_dynamic + hash_dynamic)
        norm_parameters = count_parameters(self.fusion.output_norm)
        return FieldBudgetReport(
            name=base.name,
            static=base.static + static_projection,
            dynamic=base.dynamic + dynamic_projection,
            flow=base.flow,
            other=base.other + norm_parameters,
        )


class DeterministicModalSpatialField(BasisTimeOnlyField):
    """Deterministic modal field with the same fusion seam as the hybrid field."""

    def __init__(self, *, fusion: str = "concat", **kwargs) -> None:
        super().__init__(**kwargs)
        self.fusion = FeatureFusion(
            self.feature_dims,
            self.n_output_dims,
            mode=fusion,
        )

    @property
    def fusion_mode(self) -> str:
        return self.fusion.mode

    def query(self, xyz: Tensor, t: Tensor) -> Tensor:
        return self.fusion(self.query_features(xyz, t))

    def parameter_groups(self, lr: float) -> list[dict]:
        groups = super().parameter_groups(lr)
        return [
            {
                "params": list(groups[0]["params"])
                + list(self.fusion.parameters()),
                "lr": groups[0]["lr"],
            },
            groups[1],
            groups[2],
        ]

    def budget_report(self, name: str | None = None) -> FieldBudgetReport:
        base = super().budget_report(
            name=name or f"deterministic_modal_spatial_{self.fusion_mode}"
        )
        plane_static, plane_dynamic, hash_static, hash_dynamic = self.feature_dims
        static_projection = self.n_output_dims * (plane_static + hash_static)
        dynamic_projection = self.n_output_dims * (plane_dynamic + hash_dynamic)
        norm_parameters = count_parameters(self.fusion.output_norm)
        return FieldBudgetReport(
            name=base.name,
            static=base.static + static_projection,
            dynamic=base.dynamic + dynamic_projection,
            flow=base.flow,
            other=base.other + norm_parameters,
        )


class RoleAwareDeterministicField(DeterministicModalSpatialField):
    """Global analytic plane dynamics plus local discrete hash dynamics."""

    def __init__(self, *, plane_basis_kind: str = "analytic", **kwargs) -> None:
        kwargs.pop("basis_kind", None)
        super().__init__(
            fusion="concat", basis_kind=plane_basis_kind, **kwargs
        )
        self.hash_basis = DiscreteTemporalBasis(self.rank)

    def _hash_dynamic(
        self,
        xyz: Tensor,
        t: Tensor,
        context: TemporalQueryContext | None = None,
    ) -> Tensor:
        per_point = _time_per_point(t, xyz.shape[0])
        phi = self.hash_basis(per_point)
        coordinates = (xyz[:, [0, 1]], xyz[:, [0, 2]], xyz[:, [1, 2]])
        features = []
        for encoder, coordinate in zip(self.modal_hashes, coordinates):
            coefficients = encoder(coordinate).reshape(
                -1, self.n_levels_hash, self.rank
            )
            features.append(
                torch.einsum(
                    "nlr,nr->nl", coefficients, phi.to(coefficients.dtype)
                )
            )
        return torch.cat(features, dim=-1)

    def parameter_groups(self, lr: float) -> list[dict]:
        groups = super().parameter_groups(lr)
        return [
            groups[0],
            {
                "params": list(groups[1]["params"])
                + list(self.hash_basis.parameters()),
                "lr": groups[1]["lr"],
            },
            groups[2],
        ]

    def budget_report(self, name: str | None = None) -> FieldBudgetReport:
        base = super().budget_report(name=name or "modal_roleaware_deterministic")
        return FieldBudgetReport(
            name=base.name,
            static=base.static,
            dynamic=base.dynamic + count_parameters(self.hash_basis),
            flow=base.flow,
            other=base.other,
        )


class RoleAwareStochasticField(ModalSpatialField):
    """Role-aware temporal bases with the structured stochastic residual."""

    def __init__(self, *, plane_basis_kind: str = "analytic", **kwargs) -> None:
        kwargs.pop("basis_kind", None)
        super().__init__(
            fusion="concat", basis_kind=plane_basis_kind, **kwargs
        )
        self.hash_basis = DiscreteTemporalBasis(self.rank)

    def _hash_dynamic(
        self,
        xyz: Tensor,
        t: Tensor,
        context: TemporalQueryContext | None = None,
    ) -> Tensor:
        per_point = _time_per_point(t, xyz.shape[0])
        phi = self.hash_basis(per_point)
        residual_latent = context.hash_residual if context is not None else None
        coordinates = (xyz[:, [0, 1]], xyz[:, [0, 2]], xyz[:, [1, 2]])
        features = []
        for axis_index, (encoder, coordinate) in enumerate(
            zip(self.modal_hashes, coordinates)
        ):
            coefficients = encoder(coordinate).reshape(
                -1, self.n_levels_hash, self.rank
            )
            mean = torch.einsum(
                "nlr,nr->nl", coefficients, phi.to(coefficients.dtype)
            )
            if residual_latent is None:
                features.append(mean)
                continue
            weights = phi[:, None, :] * residual_latent[axis_index][None, :, :]
            residual = torch.einsum(
                "nlr,nlr->nl", coefficients, weights.to(coefficients.dtype)
            )
            features.append(mean + residual)
        return torch.cat(features, dim=-1)

    def parameter_groups(self, lr: float) -> list[dict]:
        groups = super().parameter_groups(lr)
        return [
            groups[0],
            {
                "params": list(groups[1]["params"])
                + list(self.hash_basis.parameters()),
                "lr": groups[1]["lr"],
            },
            groups[2],
        ]

    def budget_report(self, name: str | None = None) -> FieldBudgetReport:
        base = super().budget_report(name=name or "modal_roleaware_stochastic")
        return FieldBudgetReport(
            name=base.name,
            static=base.static,
            dynamic=base.dynamic + count_parameters(self.hash_basis),
            flow=base.flow,
            other=base.other,
        )


class AnchoredSplineBasisOnlyField(RoleAwareDeterministicField):
    """Deterministic control with analytic anchors and local spline residuals."""

    def __init__(self, **kwargs) -> None:
        super().__init__(plane_basis_kind="analytic", **kwargs)
        self.basis = AnchoredSplineTemporalBasis(
            self.rank,
            local_degree=3,
            adaptive_knots=True,
        )
        self.hash_basis = AnchoredSplineTemporalBasis(
            self.rank,
            local_degree=1,
            adaptive_knots=False,
        )

    def set_training_progress(
        self,
        *,
        spline: float = 1.0,
        high_order: float = 1.0,
        stochastic: float = 1.0,
    ) -> None:
        del high_order, stochastic
        self.basis.set_residual_scale(spline)
        self.hash_basis.set_residual_scale(spline)

    def temporal_regularization_loss(self) -> Tensor:
        return (
            self.basis.regularization_loss()
            + self.hash_basis.regularization_loss()
        )

    def budget_report(self, name: str | None = None) -> FieldBudgetReport:
        return super().budget_report(name=name or "anchored_spline_basis_only")


class AnchoredSplineChampionField(RoleAwareStochasticField):
    """Stochastic anchored-spline field used by the staged champion run.

    The large mean plane/hash coefficient banks are unchanged.  Their temporal
    weights retain all eight analytic channels and receive small local spline
    corrections.  The existing plane-only residual bank and structured
    posterior reuse the local atoms, so no second dynamic hash bank is created.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(plane_basis_kind="analytic", **kwargs)
        self.basis = AnchoredSplineTemporalBasis(
            self.rank,
            local_degree=3,
            adaptive_knots=True,
        )
        self.hash_basis = AnchoredSplineTemporalBasis(
            self.rank,
            local_degree=1,
            adaptive_knots=False,
        )
        self.register_buffer("stochastic_scale", torch.tensor(1.0))

    def set_training_progress(
        self,
        *,
        spline: float = 1.0,
        high_order: float = 1.0,
        stochastic: float = 1.0,
    ) -> None:
        del high_order
        self.basis.set_residual_scale(spline)
        self.hash_basis.set_residual_scale(spline)
        self.stochastic_scale.fill_(max(0.0, float(stochastic)))

    def temporal_regularization_loss(self) -> Tensor:
        return (
            self.basis.regularization_loss()
            + self.hash_basis.regularization_loss()
        )

    def _plane_dynamic(
        self,
        xyz: Tensor,
        t: Tensor,
        context: TemporalQueryContext | None = None,
    ) -> Tensor:
        flat_time = t.reshape(-1)
        shared_time = flat_time.numel() == 1
        per_point = flat_time if shared_time else _time_per_point(t, xyz.shape[0])
        mean_phi = self.basis(per_point)
        residual_phi = self.basis.local_features(per_point) * self.stochastic_scale
        if shared_time:
            mean_phi = mean_phi[0]
            residual_phi = residual_phi[0]
        residual_latent = context.plane_residual if context is not None else None

        levels = []
        for level_index, (mean_axes, residual_axes) in enumerate(
            zip(self.modal_axes, self.residual_modal_axes)
        ):
            factors = []
            for axis_index, (mean_axis, residual_axis, coordinate) in enumerate(
                zip(mean_axes, residual_axes, xyz.unbind(-1))
            ):
                latent = (
                    residual_latent[level_index, axis_index]
                    if residual_latent is not None
                    else torch.zeros(
                        self.rank, dtype=mean_phi.dtype, device=mean_phi.device
                    )
                )
                if shared_time:
                    mean = _contract_then_sample_axis(
                        mean_axis, mean_phi, coordinate
                    )
                    residual = _contract_then_sample_axis(
                        residual_axis,
                        residual_phi * latent,
                        coordinate,
                    )
                else:
                    mean_coefficients = _sample_axis(mean_axis, coordinate)
                    residual_coefficients = _sample_axis(residual_axis, coordinate)
                    mean = torch.einsum(
                        "ncr,nr->nc", mean_coefficients, mean_phi
                    )
                    residual = torch.einsum(
                        "ncr,nr->nc",
                        residual_coefficients,
                        residual_phi * latent,
                    )
                factors.append(1.0 + mean + residual)
            levels.append(factors[0] * factors[1] * factors[2])
        return torch.cat(levels, dim=-1)

    def _hash_dynamic(
        self,
        xyz: Tensor,
        t: Tensor,
        context: TemporalQueryContext | None = None,
    ) -> Tensor:
        per_point = _time_per_point(t, xyz.shape[0])
        mean_phi = self.hash_basis(per_point)
        residual_phi = (
            self.hash_basis.local_features(per_point) * self.stochastic_scale
        )
        residual_latent = context.hash_residual if context is not None else None
        coordinates = (xyz[:, [0, 1]], xyz[:, [0, 2]], xyz[:, [1, 2]])
        features = []
        for axis_index, (encoder, coordinate) in enumerate(
            zip(self.modal_hashes, coordinates)
        ):
            coefficients = encoder(coordinate).reshape(
                -1, self.n_levels_hash, self.rank
            )
            mean = torch.einsum(
                "nlr,nr->nl", coefficients, mean_phi.to(coefficients.dtype)
            )
            if residual_latent is None:
                features.append(mean)
                continue
            weights = (
                residual_phi[:, None, :]
                * residual_latent[axis_index][None, :, :]
            )
            residual = torch.einsum(
                "nlr,nlr->nl", coefficients, weights.to(coefficients.dtype)
            )
            features.append(mean + residual)
        return torch.cat(features, dim=-1)

    def budget_report(self, name: str | None = None) -> FieldBudgetReport:
        return super().budget_report(name=name or "anchored_spline_champion")


class _HighOrderResidualMixin:
    """Feature-level residual injection shared by deterministic and VAE fields."""

    def _init_high_order_residual(self, *, stochastic: bool) -> None:
        if not hasattr(self, "residual_modal_axes"):
            self.residual_modal_axes = nn.ModuleList()
            for axes in self.modal_axes:
                residual_level = nn.ParameterList(
                    [nn.Parameter(torch.empty_like(axis)) for axis in axes]
                )
                for residual_axis in residual_level:
                    nn.init.normal_(residual_axis, std=1e-3)
                self.residual_modal_axes.append(residual_level)

        gate_probability = 0.01
        gate_logit = math.log(gate_probability / (1.0 - gate_probability))
        self.high_order_plane_gates = nn.Parameter(
            torch.full((len(self.modal_axes), 3, self.rank), gate_logit)
        )
        self.high_order_hash_gates = nn.Parameter(
            torch.full((3, self.n_levels_hash, self.rank), gate_logit)
        )
        self.high_order_refiner = HighOrderTemporalRefiner(
            self.rank,
            num_frames=self.num_frames,
        )
        self._high_order_stochastic = bool(stochastic)
        self._high_order_energy_terms: list[Tensor] = []
        self.register_buffer(
            "last_high_order_feature_ratio", torch.tensor(0.0), persistent=False
        )

    def query_features(self, xyz: Tensor, t: Tensor) -> SceneFieldFeatures:
        self._high_order_energy_terms = []
        return super().query_features(xyz, t)

    def _sample_query_context(self) -> TemporalQueryContext:
        return super()._sample_query_context()._replace(high_order_cache={})

    def _high_order_phi(
        self, t: Tensor, context: TemporalQueryContext | None
    ) -> Tensor:
        cache = context.high_order_cache if context is not None else None
        flat_time = t.reshape(-1)
        detached_time = flat_time.detach().float()
        key = (
            flat_time.numel(),
            float(detached_time.sum()),
            float(detached_time.square().sum()),
        )
        if cache is not None and key in cache:
            return cache[key]
        output = self.high_order_refiner(t, self.basis, self.hash_basis)
        if cache is not None:
            cache[key] = output
        return output

    def _residual_modulation(
        self,
        latent: Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        if not self._high_order_stochastic or latent is None:
            return torch.ones(self.rank, dtype=dtype, device=device)
        scale = self.stochastic_scale.to(dtype=dtype, device=device)
        return 1.0 + scale * latent.to(dtype=dtype, device=device)

    def _safe_high_order_residual(
        self, base: Tensor, residual: Tensor
    ) -> Tensor:
        base_rms = (base.float().square().mean() + 1e-12).sqrt().clamp_min(1e-6)
        residual_rms = (residual.float().square().mean() + 1e-12).sqrt()
        hard_scale = (0.5 * base_rms / residual_rms).clamp(max=1.0)
        safe_residual = residual * hard_scale.to(residual.dtype)
        ratio = (safe_residual.float().square().mean() + 1e-12).sqrt() / base_rms
        self._high_order_energy_terms.append(F.relu(ratio - 0.25).square())
        self.last_high_order_feature_ratio.copy_(
            ratio.detach().to(self.last_high_order_feature_ratio)
        )
        return safe_residual

    def _plane_dynamic(
        self,
        xyz: Tensor,
        t: Tensor,
        context: TemporalQueryContext | None = None,
    ) -> Tensor:
        if float(self.high_order_refiner.progress) <= 0.0:
            return super()._plane_dynamic(xyz, t, context)
        flat_time = t.reshape(-1)
        shared_time = flat_time.numel() == 1
        query_time = (
            flat_time if shared_time else _time_per_point(t, xyz.shape[0])
        )
        mean_phi = self.basis(query_time)
        residual_phi = self._high_order_phi(query_time, context)
        if shared_time:
            mean_phi = mean_phi[0]
            residual_phi = residual_phi[0]
        residual_latent = context.plane_residual if context is not None else None
        gates = torch.sigmoid(self.high_order_plane_gates)

        levels = []
        for level_index, (mean_axes, residual_axes) in enumerate(
            zip(self.modal_axes, self.residual_modal_axes)
        ):
            factors = []
            for axis_index, (mean_axis, residual_axis, coordinate) in enumerate(
                zip(mean_axes, residual_axes, xyz.unbind(-1))
            ):
                latent = (
                    residual_latent[level_index, axis_index]
                    if residual_latent is not None
                    else None
                )
                modulation = self._residual_modulation(
                    latent,
                    dtype=residual_phi.dtype,
                    device=residual_phi.device,
                )
                weights = (
                    residual_phi
                    * gates[level_index, axis_index].to(residual_phi.dtype)
                    * modulation
                )
                if shared_time:
                    mean = _contract_then_sample_axis(
                        mean_axis, mean_phi, coordinate
                    )
                    residual = _contract_then_sample_axis(
                        residual_axis, weights, coordinate
                    )
                else:
                    mean_coefficients = _sample_axis(mean_axis, coordinate)
                    residual_coefficients = _sample_axis(
                        residual_axis, coordinate
                    )
                    mean = torch.einsum(
                        "ncr,nr->nc",
                        mean_coefficients,
                        mean_phi.to(mean_coefficients.dtype),
                    )
                    residual = torch.einsum(
                        "ncr,nr->nc",
                        residual_coefficients,
                        weights.to(residual_coefficients.dtype),
                    )
                base_factor = 1.0 + mean
                residual = self._safe_high_order_residual(base_factor, residual)
                factors.append(base_factor + residual)
            levels.append(factors[0] * factors[1] * factors[2])
        return torch.cat(levels, dim=-1)

    def _hash_dynamic(
        self,
        xyz: Tensor,
        t: Tensor,
        context: TemporalQueryContext | None = None,
    ) -> Tensor:
        if float(self.high_order_refiner.progress) <= 0.0:
            return super()._hash_dynamic(xyz, t, context)
        flat_time = t.reshape(-1)
        shared_time = flat_time.numel() == 1
        query_time = (
            flat_time if shared_time else _time_per_point(t, xyz.shape[0])
        )
        mean_phi = self.hash_basis(query_time)
        residual_phi = self._high_order_phi(query_time, context)
        if shared_time:
            mean_phi = mean_phi[0]
            residual_phi = residual_phi[0]
        residual_latent = context.hash_residual if context is not None else None
        gates = torch.sigmoid(self.high_order_hash_gates)
        coordinates = (xyz[:, [0, 1]], xyz[:, [0, 2]], xyz[:, [1, 2]])
        features = []
        for axis_index, (encoder, coordinate) in enumerate(
            zip(self.modal_hashes, coordinates)
        ):
            coefficients = encoder(coordinate).reshape(
                -1, self.n_levels_hash, self.rank
            )
            latent = (
                residual_latent[axis_index] if residual_latent is not None else None
            )
            modulation = self._residual_modulation(
                latent,
                dtype=residual_phi.dtype,
                device=residual_phi.device,
            )
            if shared_time:
                mean = torch.einsum(
                    "nlr,r->nl", coefficients, mean_phi.to(coefficients.dtype)
                )
                weights = (
                    residual_phi
                    * gates[axis_index].to(residual_phi.dtype)
                    * modulation
                )
                residual = torch.einsum(
                    "nlr,lr->nl", coefficients, weights.to(coefficients.dtype)
                )
            else:
                mean = torch.einsum(
                    "nlr,nr->nl", coefficients, mean_phi.to(coefficients.dtype)
                )
                weights = (
                    residual_phi[:, None, :]
                    * gates[axis_index][None, :, :].to(residual_phi.dtype)
                    * modulation.unsqueeze(0)
                )
                residual = torch.einsum(
                    "nlr,nlr->nl", coefficients, weights.to(coefficients.dtype)
                )
            residual = self._safe_high_order_residual(mean, residual)
            features.append(mean + residual)
        return torch.cat(features, dim=-1)

    def high_order_regularization_loss(self) -> Tensor:
        regularization = self.high_order_refiner.regularization_loss()
        if self._high_order_energy_terms:
            feature_energy = torch.stack(self._high_order_energy_terms).mean()
            regularization = regularization + 1e-4 * feature_energy
        return regularization

    def high_order_diagnostics(self) -> dict[str, Tensor]:
        return {
            "feature_ratio": self.last_high_order_feature_ratio,
            "gram": self.high_order_refiner.last_gram_loss,
            "mixing_energy": self.high_order_refiner.last_mixing_energy,
            "curvature": self.high_order_refiner.last_curvature_loss,
            "min_cycles": self.high_order_refiner.frequency_cycles.min(),
            "max_cycles": self.high_order_refiner.frequency_cycles.max(),
        }

    def parameter_groups(self, lr: float) -> list[dict]:
        groups = super().parameter_groups(lr)
        high_order_parameters = list(self.residual_modal_axes.parameters())
        high_order_parameters += list(self.high_order_refiner.parameters())
        high_order_parameters += [
            self.high_order_plane_gates,
            self.high_order_hash_gates,
        ]
        if self._high_order_stochastic:
            high_order_parameters += list(self.residual_posterior.parameters())
        high_order_ids = {id(parameter) for parameter in high_order_parameters}

        base_groups = []
        for group in groups:
            filtered = [
                parameter
                for parameter in group["params"]
                if id(parameter) not in high_order_ids
            ]
            if not filtered:
                continue
            base_group = dict(group)
            base_group["params"] = filtered
            base_group.setdefault("stage_role", "base")
            base_groups.append(base_group)
        base_groups.append(
            {
                "params": high_order_parameters,
                "lr": lr,
                "stage_role": "high_order",
            }
        )
        return base_groups


class AnchoredSplineHighOrderField(
    _HighOrderResidualMixin, AnchoredSplineBasisOnlyField
):
    """Deterministic analytic+spline base with a projected rank-eight residual."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._init_high_order_residual(stochastic=False)

    def set_training_progress(
        self,
        *,
        spline: float = 1.0,
        high_order: float = 1.0,
        stochastic: float = 1.0,
    ) -> None:
        del stochastic
        self.basis.set_residual_scale(spline)
        self.hash_basis.set_residual_scale(spline)
        self.high_order_refiner.set_progress(high_order)

    def budget_report(self, name: str | None = None) -> FieldBudgetReport:
        base = RoleAwareDeterministicField.budget_report(
            self, name=name or "anchored_spline_high_order"
        )
        extra = count_parameters(self.residual_modal_axes)
        extra += count_parameters(self.high_order_refiner)
        extra += self.high_order_plane_gates.numel()
        extra += self.high_order_hash_gates.numel()
        return FieldBudgetReport(
            name=base.name,
            static=base.static,
            dynamic=base.dynamic + extra,
            flow=base.flow,
            other=base.other,
        )


class AnchoredSplineHighOrderGeometryResidualField(AnchoredSplineHighOrderField):
    """v5.5 field with an independently parameterized fine geometry basis."""

    def __init__(self, **kwargs) -> None:
        min_resolution = int(kwargs.get("min_resolution", 32))
        plane_levels = int(kwargs.get("n_levels_plane", 4))
        plane_channels = int(kwargs.get("n_features_per_level_plane", 8))
        num_frames = int(kwargs.get("num_frames", 51))
        super().__init__(**kwargs)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        resolutions = tuple(
            min_resolution * (2**level)
            for level in range(max(0, plane_levels - 2), plane_levels)
        )
        self.geometry_residual_basis = GeometryResidualBasis(
            self.n_output_dims,
            spatial_resolutions=resolutions,
            channels_per_level=plane_channels,
            temporal_rank=min(4, self.rank),
            num_frames=num_frames,
        )

    def query_decomposition(self, xyz: Tensor, t: Tensor) -> SceneFieldDecomposition:
        base = super().query(xyz, t)
        residual = self.geometry_residual_basis(xyz, t, base)
        zeros = base.new_zeros(base.shape[0])
        return SceneFieldDecomposition(
            base=base,
            full=base + residual,
            motion_prior=zeros,
            motion_mask=zeros,
            specialized=True,
        )

    def query(self, xyz: Tensor, t: Tensor) -> Tensor:
        return self.query_decomposition(xyz, t).full

    def parameter_groups(self, lr: float) -> list[dict]:
        return super().parameter_groups(lr) + [
            {
                "params": self.geometry_residual_basis.parameters(),
                "lr": lr,
                "stage_role": "geometry_residual",
            }
        ]

    def high_order_regularization_loss(self) -> Tensor:
        return (
            super().high_order_regularization_loss()
            + self.geometry_residual_basis.regularization_loss()
        )

    def budget_report(self, name: str | None = None) -> FieldBudgetReport:
        base = super().budget_report(
            name=name or "anchored_spline_high_order_geometry_residual"
        )
        return FieldBudgetReport(
            name=base.name,
            static=base.static,
            dynamic=base.dynamic + count_parameters(self.geometry_residual_basis),
            flow=base.flow,
            other=base.other,
        )


class AnchoredSplineHighOrderScalarDensityResidualField(
    AnchoredSplineHighOrderField
):
    """Frozen Basis field with a sign-stable post-sigma geometry adapter."""

    scalar_density_residual_adapter = VisibilityPreservingScalarDensityResidual
    default_budget_name = "anchored_spline_high_order_scalar_density_residual"

    def scalar_density_residual_adapter_options(self) -> dict[str, object]:
        """Return adapter-only construction values owned by a specialization."""

        return {}

    def __init__(self, **kwargs) -> None:
        residual_resolutions = kwargs.pop("scalar_residual_resolutions", None)
        representation = kwargs.pop("scalar_residual_representation", "basis")
        if representation not in {"basis", "latent_planes", "basis_depth_displacement"}:
            raise ValueError("unknown scalar residual representation")
        adapter = self.scalar_density_residual_adapter
        if representation == "latent_planes":
            if adapter is not VisibilityPreservingScalarDensityResidual:
                raise ValueError("latent representation requires an ungated adapter")
            from best_core.latent_density_residual import LatentFeatureDensityResidual
            adapter = LatentFeatureDensityResidual
        if representation == "basis_depth_displacement":
            if adapter is not VisibilityPreservingScalarDensityResidual:
                raise ValueError("depth displacement requires an ungated adapter")
            from best_core.depth_displacement_residual import BasisDepthDisplacementResidual
            adapter = BasisDepthDisplacementResidual
        self.scalar_residual_representation = representation
        if residual_resolutions is not None:
            residual_resolutions = tuple(residual_resolutions)
            if (len(residual_resolutions) != 2
                or any(not isinstance(r, int) or r < 2 for r in residual_resolutions)
                or residual_resolutions[0] >= residual_resolutions[1]):
                raise ValueError("scalar residual resolutions must be two increasing integers >= 2")
        min_resolution = int(kwargs.get("min_resolution", 32))
        plane_levels = int(kwargs.get("n_levels_plane", 4))
        plane_channels = int(kwargs.get("n_features_per_level_plane", 8))
        num_frames = int(kwargs.get("num_frames", 51))
        super().__init__(**kwargs)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        resolutions = tuple(
            min_resolution * (2**level)
            for level in range(max(0, plane_levels - 2), plane_levels)
        )
        self.scalar_density_residual = adapter(
            spatial_resolutions=residual_resolutions or resolutions,
            channels_per_level=plane_channels,
            temporal_rank=min(4, self.rank),
            num_frames=num_frames,
            **self.scalar_density_residual_adapter_options(),
        )

    def query_decomposition(self, xyz: Tensor, t: Tensor) -> SceneFieldDecomposition:
        base = super().query(xyz, t)
        zeros = base.new_zeros(base.shape[0])
        return SceneFieldDecomposition(
            base=base,
            full=base,
            motion_prior=zeros,
            motion_mask=zeros,
            specialized=True,
        )

    def parameter_groups(self, lr: float) -> list[dict]:
        return super().parameter_groups(lr) + [
            {
                "params": self.scalar_density_residual.parameters(),
                "lr": lr,
                "stage_role": "scalar_density_residual",
            }
        ]

    def high_order_regularization_loss(self) -> Tensor:
        return (
            super().high_order_regularization_loss()
            + self.scalar_density_residual.regularization_loss()
        )

    def budget_report(self, name: str | None = None) -> FieldBudgetReport:
        base = super().budget_report(
            name=name or self.default_budget_name
        )
        return FieldBudgetReport(
            name=base.name,
            static=base.static,
            dynamic=base.dynamic + count_parameters(self.scalar_density_residual),
            flow=base.flow,
            other=base.other,
        )


class AnchoredSplineHighOrderFarGatedScalarDensityResidualField(
    AnchoredSplineHighOrderScalarDensityResidualField
):
    """Scalar density adapter restricted by frozen-Basis expected ray depth."""

    scalar_density_residual_adapter = (
        FarGatedVisibilityPreservingScalarDensityResidual
    )
    default_budget_name = (
        "anchored_spline_high_order_far_gated_scalar_density_residual"
    )


class AnchoredSplineHighOrderCoherenceGatedScalarDensityResidualField(
    AnchoredSplineHighOrderScalarDensityResidualField
):
    """Far scalar density adapter gated by frozen range-view consensus."""

    scalar_density_residual_adapter = (
        CoherenceGatedVisibilityPreservingScalarDensityResidual
    )
    default_budget_name = (
        "anchored_spline_high_order_coherence_gated_scalar_density_residual"
    )


class AnchoredSplineHighOrderLearnedGatedScalarDensityResidualField(
    AnchoredSplineHighOrderScalarDensityResidualField
):
    """Scalar density adapter routed by learned distance and local reliability."""

    scalar_density_residual_adapter = (
        LearnedGatedVisibilityPreservingScalarDensityResidual
    )
    default_budget_name = (
        "anchored_spline_high_order_learned_gated_scalar_density_residual"
    )

    def __init__(
        self,
        *,
        physical_min_range_m: float,
        physical_max_range_m: float,
        **kwargs,
    ) -> None:
        self._physical_min_range_m = float(physical_min_range_m)
        self._physical_max_range_m = float(physical_max_range_m)
        super().__init__(**kwargs)

    def scalar_density_residual_adapter_options(self) -> dict[str, object]:
        return {
            "physical_min_range_m": self._physical_min_range_m,
            "physical_max_range_m": self._physical_max_range_m,
        }


class AnchoredSplineHighOrderChampionField(
    _HighOrderResidualMixin, AnchoredSplineChampionField
):
    """Champion objectives/posterior wrapped around the same high-order residual."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._init_high_order_residual(stochastic=True)

    def set_training_progress(
        self,
        *,
        spline: float = 1.0,
        high_order: float = 1.0,
        stochastic: float = 1.0,
    ) -> None:
        self.basis.set_residual_scale(spline)
        self.hash_basis.set_residual_scale(spline)
        self.high_order_refiner.set_progress(high_order)
        self.stochastic_scale.fill_(max(0.0, float(stochastic)))

    def budget_report(self, name: str | None = None) -> FieldBudgetReport:
        base = RoleAwareStochasticField.budget_report(
            self, name=name or "anchored_spline_high_order_champion"
        )
        extra = count_parameters(self.high_order_refiner)
        extra += self.high_order_plane_gates.numel()
        extra += self.high_order_hash_gates.numel()
        return FieldBudgetReport(
            name=base.name,
            static=base.static,
            dynamic=base.dynamic + extra,
            flow=base.flow,
            other=base.other,
        )


class _RoutedTemporalResidualField(AnchoredSplineBasisOnlyField):
    """Shared implementation for the preregistered wavelet-vNext pilots."""

    def __init__(
        self,
        *,
        temporal_kind: str,
        use_motion_routing: bool,
        use_residual_cube: bool,
        calibrated_routing: bool = False,
        specialization_enabled: bool = False,
        enforce_current_motion_support: bool = False,
        residual_rank_override: int | None = None,
        motion_router_init_seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if temporal_kind not in ("high_order", "wavelet"):
            raise ValueError("temporal_kind must be high_order or wavelet")
        self.temporal_kind = temporal_kind
        self.use_motion_routing = bool(use_motion_routing)
        self.use_residual_cube = bool(use_residual_cube)
        self.specialization_enabled = bool(specialization_enabled)
        self.enforce_current_motion_support = bool(
            enforce_current_motion_support
        )
        default_residual_rank = 4 if temporal_kind == "wavelet" else self.rank
        self.residual_rank = (
            default_residual_rank
            if residual_rank_override is None
            else int(residual_rank_override)
        )
        if self.residual_rank < 1 or self.residual_rank > self.rank:
            raise ValueError("residual rank must lie in [1, base rank]")

        self.wavelet_residual_modal_axes = nn.ModuleList()
        for axes in self.modal_axes:
            residual_level = nn.ParameterList()
            for axis in axes:
                channels, _, resolution = axis.shape
                residual = nn.Parameter(
                    torch.empty(channels, self.residual_rank, resolution)
                )
                nn.init.normal_(residual, std=1e-3)
                residual_level.append(residual)
            self.wavelet_residual_modal_axes.append(residual_level)

        if temporal_kind == "wavelet":
            self.temporal_residual_refiner = ProjectedSplineWaveletRefiner(
                self.residual_rank,
                num_frames=self.num_frames,
            )
        else:
            self.temporal_residual_refiner = HighOrderTemporalRefiner(
                self.residual_rank,
                num_frames=self.num_frames,
            )

        # Start the residual close enough to the preregistered 5% utilization
        # floor to receive a useful reconstruction gradient.  The previous
        # 5% *gate* initialization yielded only 0.65--2.72% feature-RMS after
        # temporal projection and routing, so every pilot collapsed before the
        # gate could escape its low-slope sigmoid tail.
        gate_probability = 0.15
        gate_logit = math.log(gate_probability / (1.0 - gate_probability))
        plane_channels = self.modal_axes[0][0].shape[0]
        self.wavelet_plane_gates = nn.Parameter(
            torch.full(
                (
                    len(self.modal_axes),
                    3,
                    plane_channels,
                    self.residual_rank,
                ),
                gate_logit,
            )
        )
        self.wavelet_hash_gates = nn.Parameter(
            torch.full((3, self.n_levels_hash, self.rank), gate_logit)
        )
        self.hash_residual_projection = nn.Parameter(
            torch.empty(self.residual_rank, self.rank)
        )
        if self.residual_rank == self.rank:
            nn.init.eye_(self.hash_residual_projection)
        else:
            nn.init.orthogonal_(self.hash_residual_projection)

        if motion_router_init_seed is None:
            self.motion_router = MotionRouter(calibrated=calibrated_routing)
        else:
            # The rank-4 and rank-8 treatment arms allocate a different number
            # of residual parameters before the router.  Isolate its RNG so a
            # basis comparison cannot silently receive different router starts.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(motion_router_init_seed))
                self.motion_router = MotionRouter(calibrated=calibrated_routing)
        if self.use_residual_cube:
            cube_channels = 3 * self.n_levels_hash
            self.residual_cube = TuckerResidualCube(
                output_channels=cube_channels,
                temporal_rank=self.residual_rank,
                resolution=32,
                spatial_rank=8,
            )
            self.wavelet_cube_gates = nn.Parameter(
                torch.full((cube_channels, self.residual_rank), gate_logit)
            )

        self.register_buffer(
            "occupancy_change_grid",
            torch.zeros(
                self.num_frames, 1, 32, 32, 32, dtype=torch.uint8
            ),
        )
        self.register_buffer("motion_prior_ready", torch.tensor(False))
        self.register_buffer(
            "motion_prior_observed", torch.zeros(self.num_frames, dtype=torch.bool)
        )
        self.register_buffer("utilization_floor_scale", torch.tensor(0.0))
        self.register_buffer(
            "last_high_order_feature_ratio", torch.tensor(0.0), persistent=False
        )
        self.register_buffer("last_dynamic_feature_ratio", torch.tensor(0.0))
        self.register_buffer("last_static_feature_ratio", torch.tensor(0.0))
        self.register_buffer("last_output_feature_ratio", torch.tensor(0.0))
        self.register_buffer("last_output_dynamic_feature_ratio", torch.tensor(0.0))
        self.register_buffer("last_output_static_feature_ratio", torch.tensor(0.0))
        self.register_buffer("last_motion_mask", torch.tensor(0.0))
        self.register_buffer("last_motion_prior", torch.tensor(0.0))
        self.register_buffer("last_feature_decorrelation", torch.tensor(0.0))
        self._wavelet_energy_terms: list[tuple[Tensor, Tensor, Tensor]] = []
        self._wavelet_output_energy_terms: list[tuple[Tensor, Tensor, Tensor]] = []
        self._wavelet_router_terms: list[Tensor] = []
        self._wavelet_decorrelation_terms: list[Tensor] = []

    @property
    def wavelet_refiner(self) -> ProjectedSplineWaveletRefiner:
        if self.temporal_kind != "wavelet":
            raise AttributeError("the high-order control does not own a wavelet refiner")
        return self.temporal_residual_refiner

    def configure_motion_prior(self, points_by_frame: dict[int, Tensor]) -> None:
        """Voxelize world-aligned scans and store adjacent-frame change evidence."""
        device = self.occupancy_change_grid.device
        resolution = self.occupancy_change_grid.shape[-1]
        occupancy = torch.zeros(
            self.num_frames,
            1,
            resolution,
            resolution,
            resolution,
            device=device,
        )
        valid_frames = []
        for frame, points in points_by_frame.items():
            frame_index = int(frame)
            if not 0 <= frame_index < self.num_frames:
                continue
            values = torch.as_tensor(points, device=device, dtype=torch.float32)
            if values.numel() == 0:
                continue
            values = values.reshape(-1, 3).clamp(0.0, 1.0)
            indices = (values * (resolution - 1)).round().long()
            occupancy[
                frame_index,
                0,
                indices[:, 2],
                indices[:, 1],
                indices[:, 0],
            ] = 1.0
            valid_frames.append(frame_index)
        if not valid_frames:
            self.occupancy_change_grid.zero_()
            self.motion_prior_observed.zero_()
            self.motion_prior_ready.fill_(False)
            return

        occupancy = F.max_pool3d(occupancy, kernel_size=3, stride=1, padding=1)
        change = torch.zeros_like(occupancy)
        valid_frames = sorted(set(valid_frames))
        observed = torch.zeros(self.num_frames, dtype=torch.bool, device=device)
        observed[valid_frames] = True
        for position, frame in enumerate(valid_frames):
            neighbours = []
            if position > 0:
                previous = valid_frames[position - 1]
                neighbours.append(
                    (occupancy[frame] - occupancy[previous]).abs()
                    / max(frame - previous, 1)
                )
            if position + 1 < len(valid_frames):
                following = valid_frames[position + 1]
                neighbours.append(
                    (occupancy[frame] - occupancy[following]).abs()
                    / max(following - frame, 1)
                )
            if neighbours:
                change[frame] = torch.stack(neighbours).amax(dim=0)

        # Held-out evaluation frames have no scan of their own.  Interpolate
        # adjacent *change rates* rather than silently replacing their primary
        # routing evidence with zero or reading held-out LiDAR labels.
        for left, right in zip(valid_frames, valid_frames[1:]):
            gap = right - left
            for frame in range(left + 1, right):
                alpha = (frame - left) / gap
                change[frame] = torch.lerp(change[left], change[right], alpha)
        change = F.avg_pool3d(change, kernel_size=3, stride=1, padding=1)
        self.occupancy_change_grid.copy_((255.0 * change).round().to(torch.uint8))
        self.motion_prior_observed.copy_(observed)
        self.motion_prior_ready.fill_(True)

    def _sample_occupancy_change(self, xyz: Tensor, t: Tensor) -> Tensor:
        if not bool(self.motion_prior_ready):
            return xyz.new_zeros(xyz.shape[0])
        times = _time_per_point(t, xyz.shape[0])
        frame_indices = (times * (self.num_frames - 1)).round().long().clamp(
            0, self.num_frames - 1
        )
        output = xyz.new_zeros(xyz.shape[0], dtype=torch.float32)
        for frame in frame_indices.unique():
            selected = frame_indices == frame
            grid = xyz[selected].float().mul(2.0).sub(1.0).view(1, 1, 1, -1, 3)
            volume = (
                self.occupancy_change_grid[int(frame), None].float() / 255.0
            )
            sampled = F.grid_sample(
                volume,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            output[selected] = sampled[0, 0, 0, 0]
        return output.to(xyz)

    def _sample_query_context(self) -> TemporalQueryContext:
        return super()._sample_query_context()._replace(
            high_order_cache={}, wavelet_cache={}
        )

    def query_features(self, xyz: Tensor, t: Tensor) -> SceneFieldFeatures:
        self._wavelet_energy_terms = []
        self._wavelet_output_energy_terms = []
        self._wavelet_router_terms = []
        self._wavelet_decorrelation_terms = []
        return super().query_features(xyz, t)

    def _residual_phi(
        self, t: Tensor, context: TemporalQueryContext | None
    ) -> Tensor:
        cache = context.high_order_cache if context is not None else None
        flat_time = t.reshape(-1)
        detached = flat_time.detach().float()
        key = (flat_time.numel(), float(detached.sum()), float(detached.square().sum()))
        if cache is not None and key in cache:
            return cache[key]
        output = self.temporal_residual_refiner(t, self.basis, self.hash_basis)
        if cache is not None:
            cache[key] = output
        return output

    def _motion_mask(
        self,
        xyz: Tensor,
        t: Tensor,
        context: TemporalQueryContext | None,
    ) -> tuple[Tensor, Tensor]:
        cache = context.wavelet_cache if context is not None else None
        detached_time = t.reshape(-1).detach().float()
        key = (
            "motion",
            xyz.data_ptr(),
            xyz.shape[0],
            float(detached_time.sum()),
        )
        if cache is not None and key in cache:
            stacked = cache[key]
            return stacked[:, 0], stacked[:, 1]
        flow = cache.get("root_flow") if cache is not None else None
        if flow is None or flow.shape[0] != xyz.shape[0]:
            flow = self.flow_net(self._xt(xyz, t))
        occupancy = self._sample_occupancy_change(xyz, t)
        routed, prior = self.motion_router(xyz, t, flow, occupancy)
        mask = routed if self.use_motion_routing else torch.ones_like(routed)
        if self.use_motion_routing:
            router_loss = (
                self.motion_router.supervision_loss(routed, prior)
                if self.specialization_enabled
                else F.mse_loss(routed, prior)
            )
            self._wavelet_router_terms.append(router_loss)
        self.last_motion_mask.copy_(mask.detach().mean().to(self.last_motion_mask))
        self.last_motion_prior.copy_(prior.detach().mean().to(self.last_motion_prior))
        if cache is not None:
            cache[key] = torch.stack((mask, prior), dim=-1)
        return mask, prior

    @staticmethod
    def _rms(value: Tensor) -> Tensor:
        # ``sqrt`` has an infinite derivative at zero.  The residual schedule
        # intentionally starts below FP16 resolution, so a clamped FP32 mean
        # square is required to keep the first activation steps finite.
        return value.float().square().mean().clamp_min(1e-12).sqrt()

    @staticmethod
    def _weighted_rms(value: Tensor, weight: Tensor) -> Tensor:
        expanded = weight.reshape(-1, *([1] * (value.ndim - 1))).float()
        denominator = expanded.sum() * value[0].numel()
        if float(denominator.detach()) <= 1e-6:
            return value.new_zeros((), dtype=torch.float32)
        mean_square = (
            (value.float().square() * expanded).sum() / denominator.clamp_min(1e-6)
        )
        return mean_square.clamp_min(1e-12).sqrt()

    def _safe_wavelet_residual(
        self,
        base: Tensor,
        residual: Tensor,
        motion_prior: Tensor,
    ) -> Tensor:
        base_rms = self._rms(base).clamp_min(1e-6)
        residual_rms = self._rms(residual)
        hard_scale = (0.5 * base_rms / residual_rms).clamp(max=1.0)
        safe = residual * hard_scale.to(residual)
        global_ratio = self._rms(safe) / base_rms
        dynamic_base = self._weighted_rms(base, motion_prior).clamp_min(1e-6)
        dynamic_ratio = self._weighted_rms(safe, motion_prior) / dynamic_base
        static_weight = 1.0 - motion_prior
        static_base = self._weighted_rms(base, static_weight).clamp_min(1e-6)
        static_ratio = self._weighted_rms(safe, static_weight) / static_base
        self._wavelet_energy_terms.append(
            (global_ratio, dynamic_ratio, static_ratio)
        )
        return safe

    def _safe_output_residual(
        self,
        base: Tensor,
        residual: Tensor,
        motion_prior: Tensor,
        motion_mask: Tensor | None = None,
    ) -> Tensor:
        """Cap, route and measure the residual at the final fusion seam."""
        if motion_mask is not None:
            expanded_mask = motion_mask.detach().reshape(
                -1, *([1] * (residual.ndim - 1))
            ).to(residual)
            residual = residual * expanded_mask
        base_rms = self._rms(base).clamp_min(1e-6)
        residual_rms = self._rms(residual)
        hard_scale = (0.5 * base_rms / residual_rms).clamp(max=1.0)
        safe = residual * hard_scale.to(residual)
        global_ratio = self._rms(safe) / base_rms
        dynamic_base = self._weighted_rms(base, motion_prior).clamp_min(1e-6)
        dynamic_ratio = self._weighted_rms(safe, motion_prior) / dynamic_base
        static_weight = 1.0 - motion_prior
        static_base = self._weighted_rms(base, static_weight).clamp_min(1e-6)
        static_ratio = self._weighted_rms(safe, static_weight) / static_base
        self._wavelet_output_energy_terms.append(
            (global_ratio, dynamic_ratio, static_ratio)
        )
        self.last_output_feature_ratio.copy_(
            global_ratio.detach().to(self.last_output_feature_ratio)
        )
        self.last_output_dynamic_feature_ratio.copy_(
            dynamic_ratio.detach().to(self.last_output_dynamic_feature_ratio)
        )
        self.last_output_static_feature_ratio.copy_(
            static_ratio.detach().to(self.last_output_static_feature_ratio)
        )
        return safe

    def _plane_dynamic_parts(
        self,
        xyz: Tensor,
        t: Tensor,
        context: TemporalQueryContext | None = None,
    ) -> tuple[Tensor, Tensor]:
        if float(self.temporal_residual_refiner.progress) <= 0.0:
            base = super()._plane_dynamic(xyz, t, context)
            return base, base
        flat_time = t.reshape(-1)
        shared_time = flat_time.numel() == 1
        query_time = flat_time if shared_time else _time_per_point(t, xyz.shape[0])
        mean_phi = self.basis(query_time)
        residual_phi = self._residual_phi(query_time, context)
        if shared_time:
            mean_phi = mean_phi[0]
            residual_phi = residual_phi[0]
        motion, prior = self._motion_mask(xyz, t, context)
        gates = torch.sigmoid(self.wavelet_plane_gates)

        base_levels = []
        full_levels = []
        for level_index, (mean_axes, residual_axes) in enumerate(
            zip(self.modal_axes, self.wavelet_residual_modal_axes)
        ):
            base_factors = []
            full_factors = []
            for axis_index, (mean_axis, residual_axis, coordinate) in enumerate(
                zip(mean_axes, residual_axes, xyz.unbind(-1))
            ):
                if shared_time:
                    mean = _contract_then_sample_axis(mean_axis, mean_phi, coordinate)
                    weights = gates[level_index, axis_index] * residual_phi[None]
                    profile = torch.einsum("crs,cr->cs", residual_axis, weights)
                    residual = _sample_line(profile, coordinate)
                else:
                    mean_coefficients = _sample_axis(mean_axis, coordinate)
                    residual_coefficients = _sample_axis(residual_axis, coordinate)
                    mean = torch.einsum(
                        "ncr,nr->nc", mean_coefficients, mean_phi
                    )
                    weights = (
                        residual_phi[:, None, :]
                        * gates[level_index, axis_index][None]
                    )
                    residual = torch.einsum(
                        "ncr,ncr->nc", residual_coefficients, weights
                    )
                residual = residual * motion[:, None].to(residual)
                base_factor = 1.0 + mean
                residual = self._safe_wavelet_residual(
                    base_factor, residual, prior
                )
                base_factors.append(base_factor)
                full_factors.append(base_factor + residual)
            base_levels.append(base_factors[0] * base_factors[1] * base_factors[2])
            full_levels.append(full_factors[0] * full_factors[1] * full_factors[2])
        return torch.cat(base_levels, dim=-1), torch.cat(full_levels, dim=-1)

    def _plane_dynamic(
        self,
        xyz: Tensor,
        t: Tensor,
        context: TemporalQueryContext | None = None,
    ) -> Tensor:
        return self._plane_dynamic_parts(xyz, t, context)[1]

    def _hash_dynamic_parts(
        self,
        xyz: Tensor,
        t: Tensor,
        context: TemporalQueryContext | None = None,
    ) -> tuple[Tensor, Tensor]:
        if float(self.temporal_residual_refiner.progress) <= 0.0:
            base = super()._hash_dynamic(xyz, t, context)
            return base, base
        flat_time = t.reshape(-1)
        shared_time = flat_time.numel() == 1
        query_time = flat_time if shared_time else _time_per_point(t, xyz.shape[0])
        mean_phi = self.hash_basis(query_time)
        residual_phi = self._residual_phi(query_time, context)
        if shared_time:
            mean_phi = mean_phi[0]
            residual_phi = residual_phi[0]
        motion, prior = self._motion_mask(xyz, t, context)
        gates = torch.sigmoid(self.wavelet_hash_gates)
        coordinates = (xyz[:, [0, 1]], xyz[:, [0, 2]], xyz[:, [1, 2]])
        base_features = []
        full_features = []
        for axis_index, (encoder, coordinate) in enumerate(
            zip(self.modal_hashes, coordinates)
        ):
            coefficients = encoder(coordinate).reshape(
                -1, self.n_levels_hash, self.rank
            )
            if shared_time:
                mean = torch.einsum(
                    "nlr,r->nl", coefficients, mean_phi.to(coefficients)
                )
                projected = residual_phi @ self.hash_residual_projection
                weights = gates[axis_index] * projected[None]
                residual = torch.einsum(
                    "nlr,lr->nl", coefficients, weights.to(coefficients)
                )
            else:
                mean = torch.einsum(
                    "nlr,nr->nl", coefficients, mean_phi.to(coefficients)
                )
                projected = residual_phi @ self.hash_residual_projection
                weights = gates[axis_index][None] * projected[:, None]
                residual = torch.einsum(
                    "nlr,nlr->nl", coefficients, weights.to(coefficients)
                )
            residual = residual * motion[:, None].to(residual)
            residual = self._safe_wavelet_residual(mean, residual, prior)
            base_features.append(mean)
            full_features.append(mean + residual)
        base_output = torch.cat(base_features, dim=-1)
        full_output = torch.cat(full_features, dim=-1)

        if self.use_residual_cube:
            temporal = (
                residual_phi.expand(xyz.shape[0], -1)
                if shared_time
                else residual_phi
            )
            cache = context.wavelet_cache if context is not None else None
            cube_residual = self.residual_cube(
                xyz,
                temporal,
                gates=torch.sigmoid(self.wavelet_cube_gates),
                motion=motion,
                cache=cache,
            )
            cube_residual = self._safe_wavelet_residual(
                full_output, cube_residual, prior
            )
            full_output = full_output + cube_residual
        return base_output, full_output

    def _hash_dynamic(
        self,
        xyz: Tensor,
        t: Tensor,
        context: TemporalQueryContext | None = None,
    ) -> Tensor:
        return self._hash_dynamic_parts(xyz, t, context)[1]

    def _query_feature_decomposition(
        self,
        xyz: Tensor,
        t: Tensor,
        context: TemporalQueryContext,
    ) -> tuple[SceneFieldFeatures, SceneFieldFeatures, Tensor, Tensor]:
        """Compute base/full roles once with a shared temporal context."""
        frame_idx = int(t.reshape(-1)[0] * (self.num_frames - 1))
        flow = self.flow_net(self._xt(xyz, t))
        if context.wavelet_cache is not None:
            context.wavelet_cache["root_flow"] = flow
        plane_static = self._plane_static(xyz)
        hash_static = self.hash_static(xyz)
        plane_base, plane_full = self._plane_dynamic_parts(xyz, t, context)
        hash_base, hash_full = self._hash_dynamic_parts(xyz, t, context)
        motion_mask, motion_prior = self._motion_mask(xyz, t, context)

        plane_next_base = plane_previous_base = plane_base
        plane_next_full = plane_previous_full = plane_full
        hash_next_base = hash_previous_base = hash_base
        hash_next_full = hash_previous_full = hash_full
        if frame_idx < self.num_frames - 1:
            xyz_next = xyz + flow[:, :3]
            t_next = xyz.new_tensor(
                _normalized_frame_time(frame_idx + 1, self.num_frames)
            )
            plane_next_base, plane_next_full = self._plane_dynamic_parts(
                xyz_next, t_next, context
            )
            with torch.no_grad():
                hash_next_base, hash_next_full = self._hash_dynamic_parts(
                    xyz_next, t_next, context
                )
        if frame_idx > 0:
            xyz_previous = xyz + flow[:, 3:]
            t_previous = xyz.new_tensor(
                _normalized_frame_time(frame_idx - 1, self.num_frames)
            )
            plane_previous_base, plane_previous_full = self._plane_dynamic_parts(
                xyz_previous, t_previous, context
            )
            with torch.no_grad():
                hash_previous_base, hash_previous_full = self._hash_dynamic_parts(
                    xyz_previous, t_previous, context
                )

        def smooth(current: Tensor, following: Tensor, previous: Tensor) -> Tensor:
            return 0.5 * current + 0.25 * (following + previous)

        base = SceneFieldFeatures(
            plane_static=plane_static,
            plane_dynamic=smooth(plane_base, plane_next_base, plane_previous_base),
            hash_static=hash_static,
            hash_dynamic=smooth(hash_base, hash_next_base, hash_previous_base),
        )
        full = SceneFieldFeatures(
            plane_static=plane_static,
            plane_dynamic=smooth(plane_full, plane_next_full, plane_previous_full),
            hash_static=hash_static,
            hash_dynamic=smooth(hash_full, hash_next_full, hash_previous_full),
        )
        return base, full, motion_prior, motion_mask

    def _query_features(
        self,
        xyz: Tensor,
        t: Tensor,
        context: TemporalQueryContext,
    ) -> SceneFieldFeatures:
        _, full, _, _ = self._query_feature_decomposition(xyz, t, context)
        return full

    def query_decomposition(self, xyz: Tensor, t: Tensor) -> SceneFieldDecomposition:
        self._wavelet_energy_terms = []
        self._wavelet_output_energy_terms = []
        self._wavelet_router_terms = []
        self._wavelet_decorrelation_terms = []
        context = self._sample_query_context()
        base_roles, full_roles, motion_prior, motion_mask = (
            self._query_feature_decomposition(xyz, t, context)
        )
        base = self.fusion(base_roles)
        full = self.fusion(full_roles)
        residual = full - base
        if self.specialization_enabled:
            output_motion_mask = (
                motion_mask if self.enforce_current_motion_support else None
            )
            residual = self._safe_output_residual(
                base, residual, motion_prior, output_motion_mask
            )
            full = base + residual
        base_unit = F.normalize(base.float(), dim=-1, eps=1e-6)
        residual_unit = F.normalize(residual.float(), dim=-1, eps=1e-6)
        cosine_square = (base_unit * residual_unit).sum(dim=-1).square()
        motion_weight = motion_prior.detach().float()
        decorrelation = (cosine_square * motion_weight).sum() / motion_weight.sum().clamp_min(1.0)
        self._wavelet_decorrelation_terms.append(decorrelation)
        self.last_feature_decorrelation.copy_(
            decorrelation.detach().to(self.last_feature_decorrelation)
        )
        active = self.specialization_enabled and (
            float(self.temporal_residual_refiner.progress) > 0.0
        )
        return SceneFieldDecomposition(
            base, full, motion_prior.detach(), motion_mask.detach(), active
        )

    def set_training_progress(
        self,
        *,
        spline: float = 1.0,
        high_order: float = 1.0,
        stochastic: float = 1.0,
    ) -> None:
        del stochastic
        self.basis.set_residual_scale(spline)
        self.hash_basis.set_residual_scale(spline)
        self.temporal_residual_refiner.set_progress(high_order)
        progress = float(high_order)
        if self.specialization_enabled:
            self.motion_router.set_specialization_progress(
                max(0.0, min((progress - 0.5) / 0.5, 1.0))
            )
        # The utilization constraint is an architectural identifiability
        # condition, not merely a warm-up aid.  Keep it active after the ramp;
        # otherwise the established field absorbs the residual again before
        # the final pilot window is measured.
        floor_scale = min(max(progress / 0.25, 0.0), 1.0)
        self.utilization_floor_scale.fill_(floor_scale)

    def high_order_regularization_loss(self) -> Tensor:
        regularization = self.temporal_residual_refiner.regularization_loss()
        if self._wavelet_router_terms:
            regularization = regularization + 1e-3 * torch.stack(
                self._wavelet_router_terms
            ).mean()
        energy_terms = (
            self._wavelet_output_energy_terms
            if self.specialization_enabled and self._wavelet_output_energy_terms
            else self._wavelet_energy_terms
        )
        if energy_terms:
            ratios = torch.stack(
                [torch.stack(values) for values in energy_terms]
            )
            global_ratio, dynamic_ratio, static_ratio = ratios.mean(dim=0)
            lower = dynamic_ratio.new_tensor(0.05)
            upper = dynamic_ratio.new_tensor(0.15)
            below_band = F.relu((lower - dynamic_ratio) / lower).square()
            above_band = F.relu((dynamic_ratio - upper) / upper).square()
            # Normalizing by the band endpoints avoids the vanishing gradient
            # of the old absolute 1e-4 penalty.  This loss is exactly zero in
            # [5%, 15%], so it cannot keep inflating a valid residual.
            utilization_band = below_band + above_band
            if self.specialization_enabled:
                static_excess = F.relu((static_ratio - 0.02) / 0.02).square()
                static_penalty = 5e-3 * static_excess
            else:
                static_penalty = 1e-4 * static_ratio.square()
            regularization = regularization + (
                1e-2 * self.utilization_floor_scale * utilization_band
                + static_penalty
            )
            self.last_high_order_feature_ratio.copy_(
                global_ratio.detach().to(self.last_high_order_feature_ratio)
            )
            self.last_dynamic_feature_ratio.copy_(
                dynamic_ratio.detach().to(self.last_dynamic_feature_ratio)
            )
            self.last_static_feature_ratio.copy_(
                static_ratio.detach().to(self.last_static_feature_ratio)
            )
        if self._wavelet_decorrelation_terms:
            regularization = regularization + 1e-4 * torch.stack(
                self._wavelet_decorrelation_terms
            ).mean()
        return regularization

    def high_order_diagnostics(self) -> dict[str, Tensor]:
        diagnostics = {
            "feature_ratio": self.last_high_order_feature_ratio,
            "dynamic_feature_ratio": self.last_dynamic_feature_ratio,
            "static_feature_ratio": self.last_static_feature_ratio,
            "output_feature_ratio": self.last_output_feature_ratio,
            "output_dynamic_feature_ratio": self.last_output_dynamic_feature_ratio,
            "output_static_feature_ratio": self.last_output_static_feature_ratio,
            "motion_mask": self.last_motion_mask,
            "motion_prior": self.last_motion_prior,
            "flow_scale": self.motion_router.flow_scale,
            "feature_decorrelation": self.last_feature_decorrelation,
        }
        if self.temporal_kind == "wavelet":
            diagnostics.update(
                {
                    "gram": self.wavelet_refiner.last_gram_loss,
                    "mixing_diversity": self.wavelet_refiner.last_diversity_loss,
                    "center_shift": self.wavelet_refiner.last_center_shift,
                    "center_adaptation": self.wavelet_refiner.dictionary.center_adaptation,
                }
            )
        else:
            diagnostics.update(
                {
                    "gram": self.temporal_residual_refiner.last_gram_loss,
                    "mixing_energy": self.temporal_residual_refiner.last_mixing_energy,
                    "curvature": self.temporal_residual_refiner.last_curvature_loss,
                }
            )
        return diagnostics

    def parameter_groups(self, lr: float) -> list[dict]:
        groups = super().parameter_groups(lr)
        high_order_parameters = list(self.wavelet_residual_modal_axes.parameters())
        high_order_parameters += list(self.temporal_residual_refiner.parameters())
        high_order_parameters.append(self.hash_residual_projection)
        if self.use_residual_cube:
            high_order_parameters += list(self.residual_cube.parameters())
        gate_parameters = [self.wavelet_plane_gates, self.wavelet_hash_gates]
        gate_parameters += list(self.motion_router.parameters())
        if self.use_residual_cube:
            gate_parameters.append(self.wavelet_cube_gates)
        excluded = {id(parameter) for parameter in high_order_parameters + gate_parameters}
        base_groups = []
        for group in groups:
            parameters = [
                parameter for parameter in group["params"] if id(parameter) not in excluded
            ]
            if parameters:
                copied = dict(group)
                copied["params"] = parameters
                copied.setdefault("stage_role", "base")
                base_groups.append(copied)
        base_groups.extend(
            [
                {
                    "params": high_order_parameters,
                    "lr": lr,
                    "stage_role": "high_order",
                },
                {
                    "params": gate_parameters,
                    "lr": 5.0 * lr,
                    "stage_role": "high_order_gate",
                },
            ]
        )
        return base_groups

    def budget_report(self, name: str | None = None) -> FieldBudgetReport:
        base = RoleAwareDeterministicField.budget_report(
            self, name=name or type(self).__name__
        )
        extra = count_parameters(self.wavelet_residual_modal_axes)
        extra += count_parameters(self.temporal_residual_refiner)
        extra += self.hash_residual_projection.numel()
        extra += self.wavelet_plane_gates.numel()
        extra += self.wavelet_hash_gates.numel()
        extra += count_parameters(self.motion_router)
        if self.use_residual_cube:
            extra += count_parameters(self.residual_cube)
            extra += self.wavelet_cube_gates.numel()
        return FieldBudgetReport(
            name=base.name,
            static=base.static,
            dynamic=base.dynamic + extra,
            flow=base.flow,
            other=base.other,
        )


class WaveletOnlyField(_RoutedTemporalResidualField):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            temporal_kind="wavelet",
            use_motion_routing=False,
            use_residual_cube=False,
            **kwargs,
        )


class HighOrderMotionRoutingField(_RoutedTemporalResidualField):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            temporal_kind="high_order",
            use_motion_routing=True,
            use_residual_cube=False,
            **kwargs,
        )


class HighOrderMotionSpecializedField(_RoutedTemporalResidualField):
    """Matched specialization control that changes only the temporal dictionary."""

    def __init__(self, **kwargs) -> None:
        super().__init__(
            temporal_kind="high_order",
            use_motion_routing=True,
            use_residual_cube=False,
            calibrated_routing=True,
            specialization_enabled=True,
            **kwargs,
        )


class WaveletMotionField(_RoutedTemporalResidualField):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            temporal_kind="wavelet",
            use_motion_routing=True,
            use_residual_cube=False,
            **kwargs,
        )


class WaveletMotionSpecializedField(_RoutedTemporalResidualField):
    """Calibrated motion-routed wavelet residual with base/full supervision."""

    def __init__(self, **kwargs) -> None:
        super().__init__(
            temporal_kind="wavelet",
            use_motion_routing=True,
            use_residual_cube=False,
            calibrated_routing=True,
            specialization_enabled=True,
            **kwargs,
        )


class WaveletCubeField(_RoutedTemporalResidualField):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            temporal_kind="wavelet",
            use_motion_routing=False,
            use_residual_cube=True,
            **kwargs,
        )


class WaveletMotionCubeField(_RoutedTemporalResidualField):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            temporal_kind="wavelet",
            use_motion_routing=True,
            use_residual_cube=True,
            **kwargs,
        )


class _ProjectedFeatureField(SceneField):
    def __init__(self, feature_dim: int = 120, flow_hidden_dim: int = 128) -> None:
        super().__init__()
        self.n_output_dims = feature_dim
        self.output_norm = nn.LayerNorm(feature_dim)
        self.flow_head = FeatureFlowHead(feature_dim, flow_hidden_dim)

    def flow(self, xyz: Tensor, t: Tensor) -> Tensor:
        return self.flow_head(self.query(xyz, t))


class O2AField(_ProjectedFeatureField):
    """Hybrid stochastic static-triplane/analytic-axis StateBank auto-decoder."""

    def __init__(
        self,
        feature_dim: int = 120,
        channels: int = 29,
        resolution: int = 512,
        rank: int = 8,
        flow_hidden_dim: int = 128,
    ) -> None:
        super().__init__(feature_dim, flow_hidden_dim)
        self.basis = AnalyticTemporalBasis(rank)
        self.static = nn.ModuleDict(
            {
                name: GaussianParameter((channels, resolution, resolution))
                for name in ("xy", "xz", "yz")
            }
        )
        self.modal = nn.ModuleDict(
            {
                name: GaussianParameter((channels, rank, resolution))
                for name in ("x", "y", "z")
            }
        )
        self.projections = nn.ModuleDict(
            {
                name: nn.Linear(channels, feature_dim)
                for name in ("xy", "xz", "yz", "x", "y", "z")
            }
        )

    def query(self, xyz: Tensor, t: Tensor) -> Tensor:
        per_point = _time_per_point(t, xyz.shape[0])
        phi = self.basis(per_point)
        factors = [
            _sample_plane(self.static["xy"](), xyz[:, [0, 1]]),
            _sample_plane(self.static["xz"](), xyz[:, [0, 2]]),
            _sample_plane(self.static["yz"](), xyz[:, [1, 2]]),
            torch.einsum(
                "ncr,nr->nc", _sample_axis(self.modal["x"](), xyz[:, 0]), phi
            ),
            torch.einsum(
                "ncr,nr->nc", _sample_axis(self.modal["y"](), xyz[:, 1]), phi
            ),
            torch.einsum(
                "ncr,nr->nc", _sample_axis(self.modal["z"](), xyz[:, 2]), phi
            ),
        ]
        names = ("xy", "xz", "yz", "x", "y", "z")
        fused = torch.ones(
            xyz.shape[0], self.n_output_dims, dtype=xyz.dtype, device=xyz.device
        )
        for name, factor in zip(names, factors):
            fused = fused * (1.0 + torch.tanh(self.projections[name](factor)))
        return self.output_norm(fused)

    def kl_loss(self) -> Tensor:
        losses = [posterior.kl_loss() for posterior in self.static.values()]
        losses += [posterior.kl_loss() for posterior in self.modal.values()]
        return torch.stack(losses).mean()


class O2AResidualField(SceneField):
    """Parameter-matched residual upgrade of the original H-O2a field.

    H-O2a's static triplanes, analytic spatial axes and Hadamard fusion remain
    intact. Dynamic factors contain a full deterministic mean plus a separate
    stochastic residual. The residual posterior is factorized by axis and
    temporal mode instead of duplicating every spatial coefficient as logvar.
    LiDAR4D's complete FlowField and neighbour-warp refinement are restored.
    """

    def __init__(
        self,
        *,
        feature_dim: int = 120,
        channels: int = 39,
        resolution: int = 511,
        rank: int = 8,
        num_layers_flow: int = 3,
        hidden_dim_flow: int = 64,
        num_frames: int = 51,
        **_unused_official_kwargs,
    ) -> None:
        super().__init__()
        self.n_output_dims = feature_dim
        self.num_frames = num_frames
        self.basis = AnalyticTemporalBasis(rank)
        self.posterior = AxisModeResidualPosterior(axes=3, rank=rank)

        self.static = nn.ParameterDict(
            {
                name: nn.Parameter(torch.empty(channels, resolution, resolution))
                for name in ("xy", "xz", "yz")
            }
        )
        self.mean_axes = nn.ParameterDict(
            {
                name: nn.Parameter(torch.empty(channels, rank, resolution))
                for name in ("x", "y", "z")
            }
        )
        self.residual_axes = nn.ParameterDict(
            {
                name: nn.Parameter(torch.empty(channels, rank, resolution))
                for name in ("x", "y", "z")
            }
        )
        for plane in self.static.values():
            nn.init.normal_(plane, std=0.01)
        for axis in self.mean_axes.values():
            nn.init.normal_(axis, std=0.01)
        for axis in self.residual_axes.values():
            nn.init.normal_(axis, std=1e-3)

        self.projections = nn.ModuleDict(
            {
                name: nn.Linear(channels, feature_dim)
                for name in ("xy", "xz", "yz", "x", "y", "z")
            }
        )
        self.output_norm = nn.LayerNorm(feature_dim)
        self.flow_net = FlowField(
            input_dim=4,
            num_layers=num_layers_flow,
            hidden_dim=hidden_dim_flow,
            use_grid=True,
        )

    def _xt(self, xyz: Tensor, t: Tensor) -> Tensor:
        per_point = _time_per_point(t, xyz.shape[0])
        return torch.cat([xyz, per_point.unsqueeze(-1)], dim=-1)

    def _static_feature(self, xyz: Tensor) -> Tensor:
        coordinates = {
            "xy": xyz[:, [0, 1]],
            "xz": xyz[:, [0, 2]],
            "yz": xyz[:, [1, 2]],
        }
        feature = torch.ones(
            xyz.shape[0],
            self.n_output_dims,
            dtype=xyz.dtype,
            device=xyz.device,
        )
        for name, coordinate in coordinates.items():
            factor = _sample_plane(self.static[name], coordinate)
            feature = feature * (
                1.0 + torch.tanh(self.projections[name](factor))
            )
        return feature

    def _dynamic_feature(
        self, xyz: Tensor, t: Tensor, residual_latent: Tensor
    ) -> Tensor:
        flat_time = t.reshape(-1)
        shared_time = flat_time.numel() == 1
        if shared_time:
            phi = self.basis(flat_time)[0]
        else:
            per_point = _time_per_point(t, xyz.shape[0])
            phi = self.basis(per_point)
        feature = torch.ones(
            xyz.shape[0],
            self.n_output_dims,
            dtype=xyz.dtype,
            device=xyz.device,
        )
        for axis_index, (name, coordinate) in enumerate(
            zip(("x", "y", "z"), xyz.unbind(-1))
        ):
            axis_latent = residual_latent[axis_index].to(
                dtype=phi.dtype, device=phi.device
            )
            if shared_time:
                mean = _contract_then_sample_axis(
                    self.mean_axes[name], phi, coordinate
                )
                residual = _contract_then_sample_axis(
                    self.residual_axes[name], phi * axis_latent, coordinate
                )
            else:
                mean_coefficients = _sample_axis(
                    self.mean_axes[name], coordinate
                )
                mean = torch.einsum("ncr,nr->nc", mean_coefficients, phi)
                residual_coefficients = _sample_axis(
                    self.residual_axes[name], coordinate
                )
                residual = torch.einsum(
                    "ncr,nr->nc",
                    residual_coefficients,
                    phi * axis_latent,
                )
            factor = mean + residual
            feature = feature * (
                1.0 + torch.tanh(self.projections[name](factor))
            )
        return feature

    def query(self, xyz: Tensor, t: Tensor) -> Tensor:
        residual_latent = self.posterior.sample()
        frame_idx = int(t.reshape(-1)[0] * (self.num_frames - 1))
        static = self._static_feature(xyz)
        dynamic = self._dynamic_feature(xyz, t, residual_latent)
        dynamic_next = dynamic_previous = dynamic

        flow = self.flow_net(self._xt(xyz, t))
        if frame_idx < self.num_frames - 1:
            xyz_next = xyz + flow[:, :3]
            t_next = xyz.new_tensor(
                _normalized_frame_time(frame_idx + 1, self.num_frames)
            )
            dynamic_next = self._dynamic_feature(
                xyz_next, t_next, residual_latent
            )
        if frame_idx > 0:
            xyz_previous = xyz + flow[:, 3:]
            t_previous = xyz.new_tensor(
                _normalized_frame_time(frame_idx - 1, self.num_frames)
            )
            dynamic_previous = self._dynamic_feature(
                xyz_previous, t_previous, residual_latent
            )

        dynamic = 0.5 * dynamic + 0.25 * (dynamic_next + dynamic_previous)
        return self.output_norm(static * dynamic)

    def flow(self, xyz: Tensor, t: Tensor) -> Tensor:
        return self.flow_net(self._xt(xyz, t))

    def kl_loss(self) -> Tensor:
        return self.posterior.kl_loss()

    def parameter_groups(self, lr: float) -> list[dict]:
        field_parameters = list(self.static.parameters())
        field_parameters += list(self.mean_axes.parameters())
        field_parameters += list(self.residual_axes.parameters())
        field_parameters += list(self.basis.parameters())
        field_parameters += list(self.posterior.parameters())
        field_parameters += list(self.projections.parameters())
        field_parameters += list(self.output_norm.parameters())
        return [
            {"params": field_parameters, "lr": lr},
            {"params": self.flow_net.parameters(), "lr": 0.1 * lr},
        ]


class CODTriPlaneField(_ProjectedFeatureField):
    """COD-style sum-fused triplanes projected to static and basis planes."""

    def __init__(
        self,
        feature_dim: int = 120,
        channels: int = 10,
        static_resolution: int = 512,
        modal_resolution: int = 253,
        rank: int = 8,
        flow_hidden_dim: int = 128,
    ) -> None:
        super().__init__(feature_dim, flow_hidden_dim)
        self.channels = channels
        self.rank = rank
        self.basis = AnalyticTemporalBasis(rank)
        self.static = nn.ModuleDict(
            {
                name: GaussianParameter(
                    (channels, static_resolution, static_resolution)
                )
                for name in ("xy", "xz", "yz")
            }
        )
        self.modal = nn.ModuleDict(
            {
                name: GaussianParameter(
                    (channels, rank, modal_resolution, modal_resolution)
                )
                for name in ("xy", "xz", "yz")
            }
        )
        self.projections = nn.ModuleDict(
            {
                name: nn.Linear(channels, feature_dim)
                for name in ("s_xy", "s_xz", "s_yz", "d_xy", "d_xz", "d_yz")
            }
        )

    def _sample_modal_plane(self, plane: Tensor, coords: Tensor) -> Tensor:
        flat = plane.reshape(self.channels * self.rank, *plane.shape[-2:])
        return _sample_plane(flat, coords).reshape(-1, self.channels, self.rank)

    def query(self, xyz: Tensor, t: Tensor) -> Tensor:
        per_point = _time_per_point(t, xyz.shape[0])
        phi = self.basis(per_point)
        coordinates = {
            "xy": xyz[:, [0, 1]],
            "xz": xyz[:, [0, 2]],
            "yz": xyz[:, [1, 2]],
        }
        factors = []
        for name in ("xy", "xz", "yz"):
            factors.append(
                self.projections[f"s_{name}"](
                    _sample_plane(self.static[name](), coordinates[name])
                )
            )
        for name in ("xy", "xz", "yz"):
            modal = self._sample_modal_plane(self.modal[name](), coordinates[name])
            dynamic = torch.einsum("ncr,nr->nc", modal, phi)
            factors.append(self.projections[f"d_{name}"](dynamic))
        return self.output_norm(sum(factors))

    def kl_loss(self) -> Tensor:
        losses = [posterior.kl_loss() for posterior in self.static.values()]
        losses += [posterior.kl_loss() for posterior in self.modal.values()]
        return torch.stack(losses).mean()


class Direct4DGridField(_ProjectedFeatureField):
    """A dense spatial cube at each discrete time slice with linear time sampling."""

    def __init__(
        self,
        feature_dim: int = 120,
        channels: int = 7,
        resolution: int = 94,
        time_resolution: int = 8,
        flow_hidden_dim: int = 128,
    ) -> None:
        super().__init__(feature_dim, flow_hidden_dim)
        self.time_resolution = time_resolution
        self.grid = nn.Parameter(
            torch.empty(
                time_resolution, channels, resolution, resolution, resolution
            )
        )
        nn.init.normal_(self.grid, std=0.01)
        self.projection = nn.Linear(channels, feature_dim)

    def query(self, xyz: Tensor, t: Tensor) -> Tensor:
        per_point = _time_per_point(t, xyz.shape[0])
        position = per_point * (self.time_resolution - 1)
        lower = position.floor().long().clamp(0, self.time_resolution - 1)
        upper = (lower + 1).clamp(max=self.time_resolution - 1)

        # Render batches share one time.  Keep a general per-point fallback for
        # standalone continuous queries and tests.
        if torch.equal(lower, lower[:1].expand_as(lower)) and torch.equal(
            upper, upper[:1].expand_as(upper)
        ):
            low_feature = _sample_cube(self.grid[lower[0]], xyz)
            high_feature = _sample_cube(self.grid[upper[0]], xyz)
        else:
            all_times = self.grid.permute(1, 0, 2, 3, 4).flatten(0, 1)
            sampled = _sample_cube(all_times, xyz).reshape(
                xyz.shape[0], -1, self.time_resolution
            )
            gather_low = lower.view(-1, 1, 1).expand(-1, sampled.shape[1], 1)
            gather_high = upper.view(-1, 1, 1).expand_as(gather_low)
            low_feature = sampled.gather(2, gather_low).squeeze(-1)
            high_feature = sampled.gather(2, gather_high).squeeze(-1)
        weight = (position - lower).unsqueeze(-1)
        feature = low_feature.lerp(high_feature, weight)
        return self.output_norm(self.projection(feature))


class BasisCubeField(_ProjectedFeatureField):
    """A static 3D cube plus analytic temporal-basis coefficient cubes."""

    def __init__(
        self,
        feature_dim: int = 120,
        channels: int = 6,
        resolution: int = 95,
        rank: int = 8,
        flow_hidden_dim: int = 128,
    ) -> None:
        super().__init__(feature_dim, flow_hidden_dim)
        self.channels = channels
        self.rank = rank
        self.basis = AnalyticTemporalBasis(rank)
        self.static_cube = nn.Parameter(
            torch.empty(channels, resolution, resolution, resolution)
        )
        self.modal_cubes = nn.Parameter(
            torch.empty(channels, rank, resolution, resolution, resolution)
        )
        nn.init.normal_(self.static_cube, std=0.01)
        nn.init.normal_(self.modal_cubes, std=0.01)
        self.static_projection = nn.Linear(channels, feature_dim)
        self.dynamic_projection = nn.Linear(channels, feature_dim)

    def query(self, xyz: Tensor, t: Tensor) -> Tensor:
        per_point = _time_per_point(t, xyz.shape[0])
        phi = self.basis(per_point)
        static = _sample_cube(self.static_cube, xyz)
        modal = _sample_cube(
            self.modal_cubes.reshape(
                self.channels * self.rank, *self.modal_cubes.shape[-3:]
            ),
            xyz,
        ).reshape(-1, self.channels, self.rank)
        dynamic = torch.einsum("ncr,nr->nc", modal, phi)
        return self.output_norm(
            self.static_projection(static) + self.dynamic_projection(dynamic)
        )


def field_budget_report(name: str, field: SceneField) -> FieldBudgetReport:
    return field.budget_report(name=name)


def build_scene_field(name: str, **official_kwargs) -> SceneField:
    """Construct a field using parameter-matched defaults."""
    if name == "official":
        return OfficialSceneField(**official_kwargs)
    if name == "basis_time_only":
        return BasisTimeOnlyField(**official_kwargs)
    if name == "stochastic_basis_time_only":
        return StochasticBasisTimeOnlyField(**official_kwargs)
    if name == "matched_hybrid_residual":
        return MatchedHybridResidualField(**official_kwargs)
    if name.startswith("modal_spatial_"):
        fusion = name.removeprefix("modal_spatial_")
        return ModalSpatialField(fusion=fusion, **official_kwargs)
    if name.startswith("modal_basis_"):
        basis_kind = name.removeprefix("modal_basis_")
        return ModalSpatialField(
            fusion="concat", basis_kind=basis_kind, **official_kwargs
        )
    if name == "modal_roleaware_deterministic":
        return RoleAwareDeterministicField(**official_kwargs)
    if name == "modal_roleaware_stochastic":
        return RoleAwareStochasticField(**official_kwargs)
    if name == "modal_roleaware_jordan_o1_stochastic":
        return RoleAwareStochasticField(
            plane_basis_kind="jordan_o1", **official_kwargs
        )
    if name == "modal_roleaware_hybrid_stochastic":
        return RoleAwareStochasticField(
            plane_basis_kind="hybrid", **official_kwargs
        )
    if name == "anchored_spline_basis_only":
        return AnchoredSplineBasisOnlyField(**official_kwargs)
    if name == "anchored_spline_champion":
        return AnchoredSplineChampionField(**official_kwargs)
    if name == "anchored_spline_high_order":
        return AnchoredSplineHighOrderField(**official_kwargs)
    if name == "anchored_spline_high_order_geometry_residual":
        return AnchoredSplineHighOrderGeometryResidualField(**official_kwargs)
    if name == "anchored_spline_high_order_scalar_density_residual":
        return AnchoredSplineHighOrderScalarDensityResidualField(**official_kwargs)
    if name == "anchored_spline_high_order_far_gated_scalar_density_residual":
        return AnchoredSplineHighOrderFarGatedScalarDensityResidualField(
            **official_kwargs
        )
    if name == "anchored_spline_high_order_coherence_gated_scalar_density_residual":
        return AnchoredSplineHighOrderCoherenceGatedScalarDensityResidualField(
            **official_kwargs
        )
    if name == "anchored_spline_high_order_learned_gated_scalar_density_residual":
        return AnchoredSplineHighOrderLearnedGatedScalarDensityResidualField(
            **official_kwargs
        )
    if name == "anchored_spline_high_order_champion":
        return AnchoredSplineHighOrderChampionField(**official_kwargs)
    if name == "wavelet_only":
        return WaveletOnlyField(**official_kwargs)
    if name == "high_order_motion_routing":
        return HighOrderMotionRoutingField(**official_kwargs)
    if name == "high_order_motion_specialized":
        return HighOrderMotionSpecializedField(**official_kwargs)
    if name == "wavelet_motion":
        return WaveletMotionField(**official_kwargs)
    if name == "wavelet_motion_specialized":
        return WaveletMotionSpecializedField(**official_kwargs)
    if name == "wavelet_cube":
        return WaveletCubeField(**official_kwargs)
    if name == "wavelet_motion_cube":
        return WaveletMotionCubeField(**official_kwargs)
    if name == "o2a":
        return O2AField()
    if name == "h_o2a_residual":
        return O2AResidualField(**official_kwargs)
    if name == "cod_triplane":
        return CODTriPlaneField()
    if name == "g4d_direct":
        return Direct4DGridField()
    if name == "g4d_basis_cube":
        return BasisCubeField()
    raise ValueError(f"unknown scene field {name!r}")
