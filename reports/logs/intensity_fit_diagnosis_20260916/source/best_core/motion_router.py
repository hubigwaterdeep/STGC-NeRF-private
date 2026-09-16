"""Detached motion evidence and a compact learned residual router."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MotionRouter(nn.Module):
    """Turn flow and occupancy-change evidence into a soft motion mask."""

    def __init__(
        self,
        hidden_dim: int = 32,
        flow_scale: float = 0.01,
        *,
        calibrated: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or flow_scale <= 0.0:
            raise ValueError("router dimensions and flow scale must be positive")
        self.net = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.calibrated = bool(calibrated)
        self.register_buffer("flow_scale", torch.tensor(float(flow_scale)))
        self.register_buffer("specialization_progress", torch.tensor(0.0))

    def set_specialization_progress(self, progress: float) -> None:
        self.specialization_progress.fill_(max(0.0, min(float(progress), 1.0)))

    def _temperature(self) -> Tensor:
        return 1.0 - 0.35 * self.specialization_progress.float()

    def supervision_loss(self, mask: Tensor, prior: Tensor) -> Tensor:
        """Balanced soft-label routing loss, detached from motion evidence."""
        target = prior.detach().float().clamp(0.01, 0.99)
        if self.calibrated:
            target = torch.sigmoid(torch.logit(target) / self._temperature())
        positive = target.mean().clamp(0.02, 0.98)
        positive_weight = (1.0 - positive) / positive
        weights = target * positive_weight + (1.0 - target)
        # Manual probability BCE keeps the operation in float32 and is safe
        # inside the renderer's AMP autocast context.
        probability = mask.float().clamp(1e-5, 1.0 - 1e-5)
        binary_cross_entropy = -(
            target * probability.log()
            + (1.0 - target) * (1.0 - probability).log()
        )
        return (binary_cross_entropy * weights).mean()

    def forward(
        self,
        xyz: Tensor,
        t: Tensor,
        flow: Tensor,
        occupancy_change: Tensor,
    ) -> tuple[Tensor, Tensor]:
        points = xyz.shape[0]
        time = t.reshape(-1)
        if time.numel() == 1:
            time = time.expand(points)
        if time.numel() != points or flow.shape != (points, 6):
            raise ValueError("router inputs must provide one time and flow per point")
        occupancy = occupancy_change.reshape(-1)
        if occupancy.numel() != points:
            raise ValueError("occupancy change must provide one value per point")

        detached_flow = flow.detach().float()
        forward_magnitude = detached_flow[:, :3].norm(dim=-1)
        backward_magnitude = detached_flow[:, 3:].norm(dim=-1)
        if self.training and forward_magnitude.numel() > 1:
            observed = torch.quantile(
                torch.maximum(forward_magnitude, backward_magnitude), 0.8
            ).clamp_min(1e-4)
            self.flow_scale.mul_(0.99).add_(0.01 * observed.to(self.flow_scale))
        scale = self.flow_scale.detach().float().clamp_min(1e-4)
        occupancy = occupancy.detach().float().clamp(0.0, 1.0)
        maximum_magnitude = torch.maximum(forward_magnitude, backward_magnitude)
        if self.calibrated:
            flow_temperature = (0.25 * scale).clamp_min(1e-4)
            flow_score = torch.sigmoid(
                (maximum_magnitude - scale) / flow_temperature
            )
            consistency = 1.0 - (
                (forward_magnitude - backward_magnitude).abs()
                / (forward_magnitude + backward_magnitude).clamp_min(1e-4)
            ).clamp(0.0, 1.0)
            occupancy_score = (occupancy / 0.15).clamp(0.0, 1.0)
            occupancy_score = occupancy_score.square() * (
                3.0 - 2.0 * occupancy_score
            )
            flow_evidence = 0.5 * flow_score * consistency
            prior = occupancy_score + (1.0 - occupancy_score) * flow_evidence
        else:
            flow_score = (maximum_magnitude / scale).clamp(0.0, 1.0)
            prior = torch.maximum(flow_score, occupancy)
        disagreement = (forward_magnitude - backward_magnitude).abs() / scale
        evidence = torch.cat(
            (
                xyz.detach().float(),
                time.detach().float().unsqueeze(-1),
                (forward_magnitude / scale).clamp(0.0, 2.0).unsqueeze(-1),
                (backward_magnitude / scale).clamp(0.0, 2.0).unsqueeze(-1),
                disagreement.clamp(0.0, 2.0).unsqueeze(-1),
                occupancy.unsqueeze(-1),
            ),
            dim=-1,
        )
        correction = 0.5 * torch.tanh(self.net(evidence).squeeze(-1))
        prior_logit = torch.logit(prior.clamp(0.02, 0.98))
        temperature = self._temperature() if self.calibrated else 1.0
        return torch.sigmoid((prior_logit + correction) / temperature), prior
