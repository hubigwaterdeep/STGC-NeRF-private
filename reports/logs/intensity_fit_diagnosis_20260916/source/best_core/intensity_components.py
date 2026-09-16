"""Read mode-indexed C_k phi_k terms before their sum, with bounded memory.

The original field remains frozen. Independent, zero-initialized projections
read each plane level / grid orientation separately; only their small outputs
are retained by autograd, rather than one giant per-sample coefficient vector.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def cap_terms(base, terms):
    base_rms = (base.float().square().mean() + 1e-12).sqrt().clamp_min(1e-6)
    residual_rms = (terms.sum(-1).float().square().mean() + 1e-12).sqrt()
    scale = (.5 * base_rms / residual_rms).clamp(max=1.0)
    return terms * scale.to(terms.dtype)


def term_groups(field, xyz, time, context):
    """Yield frozen component matrices and retain their temporal-mode order."""
    from best_core.scene_field import _sample_axis
    with torch.no_grad():
        query_time = time.reshape(-1)
        mean_phi = field.basis(query_time).expand(len(xyz), -1)
        grid_phi = field.hash_basis(query_time).expand(len(xyz), -1)
        active = float(field.high_order_refiner.progress) > 0
        residual_phi = (field._high_order_phi(query_time, context).expand(len(xyz), -1)
                        if active else torch.zeros_like(mean_phi))
        plane_gates = field.high_order_plane_gates.sigmoid()
        grid_gates = field.high_order_hash_gates.sigmoid()
    for level, (axes, residual_axes) in enumerate(zip(field.modal_axes, field.residual_modal_axes)):
        with torch.no_grad():
            vectors = []
            for axis, (mean_axis, residual_axis, coordinate) in enumerate(
                    zip(axes, residual_axes, xyz.unbind(-1))):
                mean = _sample_axis(mean_axis, coordinate) * mean_phi[:, None, :]
                residual = (_sample_axis(residual_axis, coordinate) * residual_phi[:, None, :]
                            * plane_gates[level, axis])
                residual = cap_terms(1.0 + mean.sum(-1), residual)
                vectors.append(torch.cat([mean, residual], dim=-1).flatten(1))
            terms = torch.cat(vectors, dim=-1)
        yield "plane", level, terms
    coordinates = (xyz[:, [0, 1]], xyz[:, [0, 2]], xyz[:, [1, 2]])
    for axis, (encoder, coordinate) in enumerate(zip(field.modal_hashes, coordinates)):
        with torch.no_grad():
            coefficients = encoder(coordinate).reshape(len(xyz), field.n_levels_hash, field.rank)
            mean = coefficients * grid_phi[:, None, :]
            residual = coefficients * residual_phi[:, None, :] * grid_gates[axis]
            residual = cap_terms(mean.sum(-1), residual)
            terms = torch.cat([mean, residual], dim=-1).flatten(1)
        yield "grid", axis, terms


class ModalComponentProjectors(nn.Module):
    def __init__(self, field):
        super().__init__()
        if getattr(field, "_high_order_stochastic", True):
            raise ValueError("component readout requires the deterministic Best high-order field")
        self.planes = nn.ModuleList([
            nn.Linear(3 * axes[0].shape[0] * field.rank * 2, axes[0].shape[0], bias=False)
            for axes in field.modal_axes])
        self.grids = nn.ModuleList([
            nn.Linear(field.n_levels_hash * field.rank * 2, field.n_levels_hash, bias=False)
            for _ in field.modal_hashes])
        for module in (*self.planes, *self.grids):
            nn.init.zeros_(module.weight)

    def forward(self, field, xyz, time, context):
        if torch.is_grad_enabled():
            # Recompute frozen terms during backward instead of retaining all
            # mode vectors merely to differentiate the projection weights.
            return checkpoint(self._project, field, xyz, time, context, use_reentrant=False)
        return self._project(field, xyz, time, context)

    def _project(self, field, xyz, time, context):
        plane, grid = [], []
        for kind, index, terms in term_groups(field, xyz, time, context):
            modules, outputs = (self.planes, plane) if kind == "plane" else (self.grids, grid)
            outputs.append(modules[index](terms.float()))
        return torch.cat(plane, dim=-1), torch.cat(grid, dim=-1)
