"""One bounded metre-scale output correction per frozen-Basis ray."""
from __future__ import annotations

import torch
from torch import Tensor

from best_core.scalar_density_residual import VisibilityPreservingScalarDensityResidual

MAX_DISPLACEMENT_M = 1.0


def depth_displacement_contract():
    return {
        'kind': 'basis_depth_displacement_v1',
        'max_displacement_m': MAX_DISPLACEMENT_M,
        'bound_selection': 'ceil_q90_absolute_raw_Basis_error_on_v65_cached_optimization_rays',
        'training_abs_error_q90_m': 0.6917853951454163,
        'query': 'one_detached_frozen_predicted_point_per_ray',
        'coordinate_units': 'scene_unit_cube',
        'output_units': 'metres',
        'formula': 'base_depth_plus_one_metre_tanh_basis_readout',
        'physical_range': 'clip_offset_to_available_headroom_preserving_zero_identity',
        'density_redistribution': False,
        'geometry_role': 'post_render_depth_adapter_not_reconstructed_density_field',
    }


class BasisDepthDisplacementResidual(VisibilityPreservingScalarDensityResidual):
    """Same parameters and basis as v6.5, applied at the predicted surface only."""

    applies_depth_displacement = True

    def correct_depth(self, depth: Tensor, rays_o: Tensor, rays_d: Tensor, time: Tensor,
                      *, bound: float, depth_scale: float, near: float, far: float) -> Tensor:
        if depth_scale <= 0 or bound <= 0 or far <= near:
            raise ValueError('depth displacement requires positive scale/bound and ordered range')
        if depth.ndim != 1 or rays_o.shape != (depth.numel(),3) or rays_d.shape != rays_o.shape:
            raise ValueError('depth displacement requires flat aligned rays')
        base = depth.detach().float()
        point = rays_o.detach().float() + base[:,None]*rays_d.detach().float()
        xyz = ((point+bound)/(2*bound)).clamp(0,1)
        offset = MAX_DISPLACEMENT_M*torch.tanh(self.raw_residual(xyz,time.detach()).float())*depth_scale
        # Do not clamp the base itself: zero residual must also preserve rays
        # whose unnormalized expected depth lies outside the physical interval.
        lower = -(base-near).clamp_min(0)
        upper = (far-base).clamp_min(0)
        offset = torch.maximum(torch.minimum(offset,upper),lower)
        return base+offset
