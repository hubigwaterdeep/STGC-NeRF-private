"""Native CUDA hash lookups with an explicit, unsplit spatial scale schedule.

TCNN accepts only integer base resolutions. For a selected level with continuous
scale s, a one-level grid of resolution ceil(s)+1 is queried at x*s/ceil(s).
Its lattice coordinate is consequently x*s+0.5, just like the unsplit grid.
This avoids rounding a new base and recomputing a new multilevel schedule.
Only ordinary FP32 evaluation-order roundoff differs. No input clamping: flow
queries retain the original behavior outside the unit square.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
import tinycudann as tcnn


@dataclass(frozen=True)
class SpatialSchedule:
    base: int = 512
    maximum: int = 32768
    levels: int = 8

    def __post_init__(self):
        if self.levels < 2 or self.base < 2 or self.maximum < self.base:
            raise ValueError('require levels >= 2 and 2 <= base <= maximum')

    @property
    def factor(self):
        return 2.0 ** (math.log2(self.maximum / self.base) / (self.levels - 1))

    def scale(self, level):
        if not 0 <= level < self.levels:
            raise ValueError('level outside original schedule')
        # Match the float configuration and grid_scale arithmetic in TCNN.
        exponent = np.float32(level) * np.log2(np.float32(self.factor))
        return float(np.float32(np.exp2(exponent) * np.float32(self.base) - np.float32(1)))

    def resolution(self, level):
        return math.ceil(self.scale(level)) + 1

    def as_dict(self):
        return dict(base=self.base, maximum=self.maximum, levels=self.levels,
                    factor=self.factor, scales=[self.scale(i) for i in range(self.levels)],
                    resolutions=[self.resolution(i) for i in range(self.levels)])


def prefix_encoding(schedule, levels, channels, log2_size):
    return tcnn.Encoding(2, dict(otype='HashGrid', n_levels=levels,
        n_features_per_level=channels, log2_hashmap_size=log2_size,
        base_resolution=schedule.base, per_level_scale=schedule.factor))


class SelectedHash(nn.Module):
    """Only allocate selected levels, retaining their original lattice scales."""
    def __init__(self, schedule, levels, channels, log2_size):
        super().__init__()
        self.levels = tuple(levels)
        self.channels = channels
        self.grids = nn.ModuleList()
        ratios = []
        sizes = [log2_size] * len(self.levels) if isinstance(log2_size, int) else list(log2_size)
        if len(sizes) != len(self.levels):
            raise ValueError("one table capacity is required per selected level")
        for level, size in zip(self.levels, sizes):
            resolution = schedule.resolution(level)
            self.grids.append(tcnn.Encoding(2, dict(otype='HashGrid', n_levels=1,
                n_features_per_level=channels, log2_hashmap_size=size,
                base_resolution=resolution, per_level_scale=1.0)))
            ratios.append(schedule.scale(level) / (resolution - 1))
        self.register_buffer('coordinate_ratios', torch.tensor(ratios, dtype=torch.float32))

    def forward(self, xy):
        if not self.grids:
            return xy.new_empty((len(xy), 0, self.channels))
        return torch.stack([grid(xy * self.coordinate_ratios[i])[:, :self.channels]
                            for i, grid in enumerate(self.grids)], dim=1)


def normalized_times(t, points, *, device, dtype):
    t = torch.as_tensor(t, device=device, dtype=dtype).reshape(-1)
    if t.numel() not in (1, points) or not torch.isfinite(t).all() or ((t < 0) | (t > 1)).any():
        raise ValueError('time must be scalar or per-point, finite, and normalized to [0, 1]')
    return t


def lagrange_weights(t, count=4):
    knots = [i / (count - 1) for i in range(count)]
    return torch.stack([math.prod([(t - knots[m]) / (knots[j] - knots[m])
                        for m in range(count) if m != j]) for j in range(count)], -1)


class LocalTemporalHash(nn.Module):
    """Eight time-indexed encoders followed by STGC's four-channel interpT."""
    def __init__(self, schedule, levels, log2_size, time_resolution=8):
        super().__init__()
        self.time_resolution = time_resolution
        self.levels = tuple(levels)
        self.hash_t = nn.ModuleList([SelectedHash(schedule, levels, 4, log2_size)
                                     for _ in range(time_resolution)])
        self.n_output_dims = len(self.levels)

    def forward(self, xy, t):
        times = normalized_times(t, len(xy), device=xy.device, dtype=xy.dtype)
        if not len(xy):
            return xy.new_empty((0, self.n_output_dims))
        index = times * (self.time_resolution - 1)
        lower, upper = index.floor().long(), index.ceil().long()
        if times.numel() == 1:
            lo, hi = int(lower.item()), int(upper.item())
            features = self.hash_t[lo](xy)
            if lo != hi:
                features = (upper[0] - index[0]) * features + (index[0] - lower[0]) * self.hash_t[hi](xy)
            # The reference multiplies half-precision channels by scalar times.
            weights = lagrange_weights(times)[0].to(features.dtype)
            return sum(features[..., j] * weights[j] for j in range(4))
        # Group by exact knot intervals: no detached/cached trainable features.
        output = xy.new_zeros((len(xy), self.n_output_dims))
        for lo in lower.unique().tolist():
            mask = lower == lo
            hi = min(lo + 1, self.time_resolution - 1)
            mix = (index[mask] - lo).reshape(-1, 1, 1)
            features = self.hash_t[lo](xy[mask])
            if hi != lo:
                features = (1 - mix) * features + mix * self.hash_t[hi](xy[mask])
            weights = lagrange_weights(times[mask]).to(features.dtype)
            output[mask] = sum(features[..., j] * weights[:, j:j+1] for j in range(4)).to(output.dtype)
        return output


class NeuralCoefficients(nn.Module):
    """Interpolate z(xy), THEN decode; time and the omitted spatial axis are absent."""
    def __init__(self, schedule, levels, log2_sizes, rank=8, latent_dim=4,
                 width=32, embedding_dim=4, frequencies=2):
        super().__init__()
        self.levels = tuple(levels)
        self.frequencies = frequencies
        self.latents = nn.ModuleList([SelectedHash(schedule, levels, latent_dim, size)
                                      for size in log2_sizes])
        self.role_embedding = nn.Embedding(3, embedding_dim)
        self.level_embedding = nn.Embedding(len(levels), embedding_dim)
        inputs = latent_dim + 2 * (1 + 2 * frequencies) + 2 * embedding_dim
        self.decoder = nn.Sequential(nn.Linear(inputs, width), nn.ReLU(), nn.Linear(width, rank))
        nn.init.normal_(self.decoder[-1].weight, std=1e-3)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, xy, role):
        z = self.latents[role](xy).float()
        position = [xy.float()]
        for i in range(self.frequencies):
            phase = xy.float() * (2**i * math.pi)
            position.extend([phase.sin(), phase.cos()])
        encoded = torch.cat(position, -1)[:, None, :].expand(-1, len(self.levels), -1)
        roles = self.role_embedding.weight[role].view(1, 1, -1).expand(len(xy), len(self.levels), -1)
        levels = self.level_embedding.weight[None].expand(len(xy), -1, -1)
        return self.decoder(torch.cat([z, encoded, roles, levels], -1))
