"""Independent dense spacetime latent planes decoded into a density residual."""
from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from best_core.scalar_density_residual import (
    VisibilityPreservingScalarDensityResidual, _HALF_LOG_TWO, _time_per_query,
)


def latent_feature_contract():
    return {
        'kind': 'dense_spacetime_latent_planes_mlp_v1',
        'planes_per_level': ['xy', 'xz', 'yz', 'xt', 'yt', 'zt'],
        'channels_per_plane': 4, 'time_resolution': 4,
        'fusion': 'concat_xy_times_zt_xz_times_yt_yz_times_xt_across_levels',
        'decoder': [24, 16, 1], 'activation': 'relu',
        'interpolation': 'bilinear_border_align_corners_true',
        'output_initialization': 'zero_weight_no_bias',
        'backbone_feature_input': False,
    }


def sample_plane(plane: Tensor, uv: Tensor) -> Tensor:
    """Sample [1,C,H(v),W(u)] at [N,2] normalized coordinates."""
    grid = (2*uv.float()-1).reshape(1,-1,1,2)
    sampled = F.grid_sample(plane,grid,mode='bilinear',padding_mode='border',align_corners=True)
    return sampled[0,:,:,0].T


class LatentFeatureDensityResidual(VisibilityPreservingScalarDensityResidual):
    """Replace the separable basis, keeping exactly its ray redistribution seam."""

    def __init__(self, *, spatial_resolutions, channels_per_level=8,
                 temporal_rank=4, num_frames=51):
        nn.Module.__init__(self)
        self.spatial_resolutions = tuple(spatial_resolutions)
        if (len(self.spatial_resolutions) != 2 or any(r < 2 for r in self.spatial_resolutions)
            or self.spatial_resolutions[0] >= self.spatial_resolutions[1]):
            raise ValueError('latent residual requires two increasing spatial resolutions')
        if temporal_rank != 4 or num_frames < 2:
            raise ValueError('latent pilot requires four temporal nodes and multiple frames')
        # The pilot's latent width is independent of the frozen field's width.
        self.channels_per_level = 4
        self.temporal_rank = 4
        self.planes = nn.ModuleList()
        for r in self.spatial_resolutions:
            level = nn.ParameterList([nn.Parameter(torch.empty(1,4,r,r)) for _ in range(3)] +
                                     [nn.Parameter(torch.empty(1,4,4,r)) for _ in range(3)])
            for i, plane in enumerate(level):
                nn.init.normal_(plane,mean=0. if i < 3 else 1.,std=.1)
            self.planes.append(level)
        self.decoder = nn.Linear(24,16)
        self.output_projection = nn.Linear(16,1,bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def log_residual(self, xyz: Tensor, time: Tensor) -> Tensor:
        if xyz.ndim not in (2,3) or xyz.shape[-1] != 3:
            raise ValueError('xyz must have shape [N,3] or [B,N,3]')
        flat = xyz.reshape(-1,3)
        t = _time_per_query(xyz,time).to(flat)
        x,y,z = flat.unbind(-1)
        coords = [torch.stack(pair,-1) for pair in ((x,y),(x,z),(y,z),(x,t),(y,t),(z,t))]
        features = []
        for level in self.planes:
            xy,xz,yz,xt,yt,zt = [sample_plane(p,uv) for p,uv in zip(level,coords)]
            features.extend((xy*zt,xz*yt,yz*xt))
        hidden = F.relu(self.decoder(torch.cat(features,-1)))
        raw = self.output_projection(hidden).squeeze(-1)
        return (_HALF_LOG_TWO*torch.tanh(raw.float())).to(raw.dtype).reshape(xyz.shape[:-1])
