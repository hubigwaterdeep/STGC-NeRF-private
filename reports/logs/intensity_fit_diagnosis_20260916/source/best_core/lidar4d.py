# ==============================================================================
# Copyright (c) 2024 Zehan Zheng. All Rights Reserved.
# LiDAR4D: Dynamic Neural Fields for Novel Space-time View LiDAR Synthesis
# CVPR 2024
# https://github.com/ispc-lab/LiDAR4D
# Apache License 2.0
# ==============================================================================

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import tinycudann as tcnn
from best_core.activation import trunc_exp
from best_core.renderer import LiDAR_Renderer
from best_core.scene_field import build_scene_field
from best_core.unet import UNet


class LiDAR4D(LiDAR_Renderer):
    def __init__(
        self,
        min_resolution=32,
        base_resolution=512,
        max_resolution=32768,
        time_resolution=8,
        n_levels_plane=4,
        n_features_per_level_plane=8,
        n_levels_hash=8,
        n_features_per_level_hash=4,
        log2_hashmap_size=19,
        num_layers_flow=3,
        hidden_dim_flow=64,
        num_layers_sigma=2,
        hidden_dim_sigma=64,
        geo_feat_dim=15,
        num_layers_lidar=3,
        hidden_dim_lidar=64,
        out_lidar_dim=2,
        num_frames=51,
        scene_field="official",
        enforce_current_motion_support=False,
        residual_rank_override=0,
        scalar_residual_resolutions=None,
        scalar_residual_representation="basis",
        motion_router_init_seed=-1,
        visibility_residual_basis="none",
        visibility_depth_scale=1.0,
        bound=1,
        **kwargs,
    ):
        super().__init__(bound, **kwargs)

        self.out_lidar_dim = out_lidar_dim
        self.num_frames = num_frames
        self.scene_field_name = scene_field
        field_kwargs = dict(
            min_resolution=min_resolution,
            base_resolution=base_resolution,
            max_resolution=max_resolution,
            time_resolution=time_resolution,
            n_levels_plane=n_levels_plane,
            n_features_per_level_plane=n_features_per_level_plane,
            n_levels_hash=n_levels_hash,
            n_features_per_level_hash=n_features_per_level_hash,
            log2_hashmap_size=log2_hashmap_size,
            num_layers_flow=num_layers_flow,
            hidden_dim_flow=hidden_dim_flow,
            num_frames=num_frames,
        )
        if (
            scene_field
            == "anchored_spline_high_order_learned_gated_scalar_density_residual"
        ):
            depth_scale = float(visibility_depth_scale)
            if depth_scale <= 0.0:
                raise ValueError("learned residual gate requires a positive depth scale")
            normalized_near = float(kwargs.get("near_lidar", 0.0))
            normalized_far = float(kwargs.get("far_lidar", 0.0))
            if normalized_far <= normalized_near:
                raise ValueError("learned residual gate requires a valid LiDAR range")
            field_kwargs.update(
                physical_min_range_m=normalized_near / depth_scale,
                physical_max_range_m=normalized_far / depth_scale,
            )
        if scalar_residual_resolutions is not None:
            if scene_field != "anchored_spline_high_order_scalar_density_residual":
                raise ValueError("residual resolution override requires the ungated scalar field")
            field_kwargs["scalar_residual_resolutions"] = tuple(scalar_residual_resolutions)
        if scalar_residual_representation != "basis":
            if (scene_field != "anchored_spline_high_order_scalar_density_residual"
                or scalar_residual_representation not in {"latent_planes", "basis_depth_displacement"}):
                raise ValueError("alternative residual requires the ungated scalar field")
            field_kwargs["scalar_residual_representation"] = scalar_residual_representation
        if enforce_current_motion_support:
            if scene_field not in {
                "high_order_motion_specialized",
                "wavelet_motion_specialized",
            }:
                raise ValueError(
                    "current-motion output support requires a specialized motion field"
                )
            field_kwargs["enforce_current_motion_support"] = True
        if residual_rank_override:
            if scene_field not in {
                "high_order_motion_specialized",
                "wavelet_motion_specialized",
            }:
                raise ValueError(
                    "residual rank override requires a specialized motion field"
                )
            field_kwargs["residual_rank_override"] = int(residual_rank_override)
        if motion_router_init_seed >= 0:
            if scene_field not in {
                "high_order_motion_specialized",
                "wavelet_motion_specialized",
            }:
                raise ValueError(
                    "router init seed requires a specialized motion field"
                )
            field_kwargs["motion_router_init_seed"] = int(motion_router_init_seed)
        shared_initialization_seed = torch.initial_seed()
        fork_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=fork_devices):
            self.scene_field = build_scene_field(scene_field, **field_kwargs)
            flow_net = getattr(self.scene_field, "flow_net", None)
            if hasattr(flow_net, "reset_mlp_parameters"):
                flow_net.reset_mlp_parameters(seed=shared_initialization_seed)

        self.view_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Frequency",
                "degree": 12,
            },
        )

        self.sigma_net = tcnn.Network(
            n_input_dims=self.scene_field.n_output_dims,
            n_output_dims=1 + geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_sigma,
                "n_hidden_layers": num_layers_sigma - 1,
            },
        )

        self.intensity_net = tcnn.Network(
            n_input_dims=self.view_encoder.n_output_dims + geo_feat_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_lidar,
                "n_hidden_layers": num_layers_lidar - 1,
            },
        )

        self.raydrop_net = tcnn.Network(
            n_input_dims=self.view_encoder.n_output_dims + geo_feat_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_lidar,
                "n_hidden_layers": num_layers_lidar - 1,
            },
        )

        self.unet = UNet(
            in_channels=3,
            out_channels=1,
            visibility_residual_basis=visibility_residual_basis,
            visibility_depth_scale=visibility_depth_scale,
        )

    def forward(self, x, d, t):
        pass

    def flow(self, x, t):
        # x: [N, 3] in [-bound, bound] for point clouds
        x = (x + self.bound) / (2 * self.bound)
        flow = self.scene_field.flow(x, t)
        # Scene fields predict displacements in their [0, 1] unit cube.  The
        # public API accepts world-normalized coordinates, so its vectors must
        # use that same coordinate system before callers add or rotate them.
        flow = flow * (2 * self.bound)

        return {
            "forward": flow[:, :3],
            "backward": flow[:, 3:],
        }

    def density(self, x, t=None):
        # x: [N, 3], in [-bound, bound]
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]

        decomposition = self.scene_field.query_decomposition(x, t)
        full_h = self.sigma_net(decomposition.full)
        sigma = trunc_exp(full_h[..., 0])
        if not decomposition.specialized:
            return {
                "sigma": sigma,
                "geo_feat": full_h[..., 1:],
            }

        # The residual is a geometry adapter: it changes occupancy/depth while
        # view-dependent attributes continue to consume the base geometry
        # feature. This keeps a second appearance head out of the comparison.
        base_h = self.sigma_net(decomposition.base)
        scalar_density_residual = getattr(
            self.scene_field, "scalar_density_residual", None
        )
        if scalar_density_residual is not None:
            # The frozen tcnn head emits FP16 values.  Materialize its density
            # exponent in FP32 so the scalar path has identical numerics both
            # inside and outside autocast, without FP16 underflow or overflow.
            base_sigma = trunc_exp(base_h[..., 0].float())
            if not torch.isfinite(base_sigma).all() or torch.any(base_sigma < 0):
                raise RuntimeError(
                    "scalar density residual could not materialize finite base density"
                )
            # This field's decomposition guarantees full == base.  Reuse the
            # same materialized tensor so zero residual remains exactly inert.
            sigma = base_sigma
        else:
            base_sigma = trunc_exp(base_h[..., 0])
        outputs = {
            "sigma": sigma,
            "geo_feat": base_h[..., 1:],
            "base_sigma": base_sigma,
            "motion_prior": decomposition.motion_prior,
            "motion_mask": decomposition.motion_mask,
        }
        readout = getattr(self, "intensity_readout", None)
        if readout is not None:
            outputs["intensity_features"] = readout.encode(
                self.scene_field, x, t, decomposition.base, outputs["geo_feat"])
        if getattr(self, "intensity_feature_mode", "none") != "none":
            outputs["full_geo_feat"] = full_h[..., 1:]
        if scalar_density_residual is not None:
            if getattr(scalar_density_residual, "applies_depth_displacement", False):
                # Keep the base-attribute rendering path, but never evaluate
                # the displacement basis at volume samples.
                outputs["density_log_residual"] = torch.zeros_like(base_sigma)
            else:
                outputs["density_log_residual"] = scalar_density_residual.log_residual(x, t)
        return outputs

    # allow masked inference
    def attribute(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]

        if mask is not None:
            output = torch.zeros(
                mask.shape[0], self.out_lidar_dim, dtype=x.dtype, device=x.device
            )  # [N, 3]
            # in case of empty mask
            if not mask.any():
                return output
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = (d + 1) / 2  # to [0, 1]
        d = self.view_encoder(d)

        intensity = self.intensity_net(torch.cat([d, geo_feat], dim=-1))
        intensity = torch.sigmoid(intensity)

        raydrop = self.raydrop_net(torch.cat([d, geo_feat], dim=-1))
        raydrop = torch.sigmoid(raydrop)

        h = torch.cat([raydrop, intensity], dim=-1)

        if mask is not None:
            output[mask] = h.to(output.dtype)  # fp16 --> fp32
        else:
            output = h

        return output

    # optimizer utils
    def get_params(self, lr):
        params = self.scene_field.parameter_groups(lr) + [
            {
                "params": self.view_encoder.parameters(),
                "lr": lr,
                "stage_role": "readout",
            },
            {
                "params": self.sigma_net.parameters(),
                "lr": 0.1 * lr,
                "stage_role": "geometry_core",
            },
            {
                "params": self.intensity_net.parameters(),
                "lr": 0.1 * lr,
                "stage_role": "readout",
            },
            {
                "params": self.raydrop_net.parameters(),
                "lr": 0.1 * lr,
                "stage_role": "readout",
            },
        ]

        return params

    def kl_loss(self):
        """KL term for stochastic fields; exactly zero for deterministic fields."""
        return self.scene_field.kl_loss()

    def set_training_progress(
        self, *, spline=1.0, high_order=1.0, stochastic=1.0
    ):
        """Apply staged representation weights through the scene-field seam."""
        self.scene_field.set_training_progress(
            spline=spline, high_order=high_order, stochastic=stochastic
        )

    def temporal_regularization_loss(self):
        """Return spline-knot regularization without exposing field internals."""
        return self.scene_field.temporal_regularization_loss()

    def high_order_regularization_loss(self):
        """Return projected high-order residual regularization when present."""
        return self.scene_field.high_order_regularization_loss()

    def high_order_diagnostics(self):
        """Return representation-owned high-order scalar diagnostics."""
        return self.scene_field.high_order_diagnostics()

    def configure_motion_prior(self, points_by_frame):
        """Normalize world-aligned scans and install detached routing evidence."""
        normalized = {
            int(frame): (
                torch.as_tensor(points, dtype=torch.float32)
                .reshape(-1, 3)
                .add(self.bound)
                .div(2 * self.bound)
            )
            for frame, points in points_by_frame.items()
        }
        self.scene_field.configure_motion_prior(normalized)

    def basis_flow_loss(self, x, t, max_points=512):
        """Evaluate modal transport consistency on a bounded point subset."""
        if x.shape[0] > max_points:
            stride = max(1, x.shape[0] // max_points)
            x = x[::stride][:max_points]
        x = (x + self.bound) / (2 * self.bound)
        return self.scene_field.basis_flow_loss(x, t)


if __name__ == '__main__':
    model = LiDAR4D().cuda()
    x = torch.rand(100, 3).cuda()
    t = torch.tensor([0.2]).cuda()
    result = model.density(x, t)
    print(result)
