# ==============================================================================
# Copyright (c) 2024 Zehan Zheng. All Rights Reserved.
# LiDAR4D: Dynamic Neural Fields for Novel Space-time View LiDAR Synthesis
# CVPR 2024
# https://github.com/ispc-lab/LiDAR4D
# Apache License 2.0
# ==============================================================================

import torch
import torch.nn as nn


class LiDAR_Renderer(nn.Module):
    def __init__(
        self,
        bound=1,
        near_lidar=0.01,
        far_lidar=0.81,
        density_scale=1,
        active_sensor=False,
    ):
        super().__init__()

        self.bound = bound
        self.near_lidar = near_lidar
        self.far_lidar = far_lidar
        self.density_scale = density_scale
        self.active_sensor = active_sensor

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        aabb = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        self.register_buffer("aabb", aabb)

    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and intensity/raydrop query (can accelerate non-cuda-ray mode.)
    def density(self, x):
        raise NotImplementedError()

    def attribute(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def run(
        self,
        rays_o,
        rays_d,
        time,
        num_steps=768,
        perturb=False,
        **kwargs
    ):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # time: [B, 1]
        # return: image: [B, N, 3], depth: [B, N]

        out_lidar_dim = self.out_lidar_dim

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        aabb = self.aabb

        # hard code
        nears = torch.ones(N, dtype=rays_o.dtype, device=rays_o.device) * self.near_lidar
        fars = torch.ones(N, dtype=rays_o.dtype, device=rays_o.device) * self.far_lidar

        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        # print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0)  # [1, T]
        z_vals = z_vals.expand((N, num_steps))  # [N, T]
        z_vals = nears + (fars - nears) * z_vals  # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            # z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.

        # query SDF and RGB
        density_outputs = self.density(xyzs.reshape(-1, 3), time)

        # sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)

        deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        preserve_base_attributes = "density_log_residual" in density_outputs
        if preserve_base_attributes:
            if "base_sigma" not in density_outputs:
                raise RuntimeError(
                    "scalar density residual requires an explicit base density"
                )
            try:
                scalar_density_residual = self.scene_field.scalar_density_residual
            except AttributeError as error:
                raise RuntimeError(
                    "scalar density residual output has no redistribution module"
                ) from error
            base_sigma = density_outputs["base_sigma"].squeeze(-1)
            log_residual = density_outputs["density_log_residual"].squeeze(-1)
            if getattr(
                scalar_density_residual,
                "requires_learned_residual_gate_weight",
                False,
            ):
                learned_gate = kwargs.get("learned_residual_gate_weight")
                if not isinstance(learned_gate, torch.Tensor):
                    raise RuntimeError(
                        "learned-gated scalar density residual requires an "
                        "aligned learned residual gate"
                    )
                if learned_gate.shape != prefix:
                    raise RuntimeError(
                        "learned residual gate shape must match the ray prefix"
                    )
                candidate_sigma = scalar_density_residual.redistribute(
                    base_sigma,
                    log_residual,
                    deltas,
                    learned_residual_gate_weight=learned_gate.reshape(-1),
                )
            elif getattr(
                scalar_density_residual,
                "requires_base_expected_depth_m",
                False,
            ):
                depth_scale = float(kwargs.get("scale", 0.0))
                if depth_scale <= 0.0:
                    raise RuntimeError(
                        "far-gated scalar density residual requires a positive depth scale"
                    )
                gate_base_alphas = 1 - torch.exp(
                    -deltas * self.density_scale * base_sigma
                )
                if self.active_sensor:
                    gate_base_alphas = 1 - torch.exp(
                        -2 * deltas * self.density_scale * base_sigma
                    )
                gate_base_shifted = torch.cat(
                    [
                        torch.ones_like(gate_base_alphas[..., :1]),
                        1 - gate_base_alphas + 1e-15,
                    ],
                    dim=-1,
                )
                gate_base_weights = gate_base_alphas * torch.cumprod(
                    gate_base_shifted, dim=-1
                )[..., :-1]
                base_expected_depth_m = torch.sum(
                    gate_base_weights * z_vals, dim=-1
                ) / depth_scale
                if getattr(
                    scalar_density_residual,
                    "requires_frozen_basis_consensus_gate",
                    False,
                ):
                    consensus_gate = kwargs.get("frozen_basis_consensus_gate")
                    if not isinstance(consensus_gate, torch.Tensor):
                        raise RuntimeError(
                            "coherence-gated scalar density residual requires an "
                            "aligned frozen Basis consensus gate"
                        )
                    if consensus_gate.shape != prefix:
                        raise RuntimeError(
                            "frozen Basis consensus gate shape must match the ray prefix"
                        )
                    candidate_sigma = scalar_density_residual.redistribute(
                        base_sigma,
                        log_residual,
                        deltas,
                        base_expected_depth_m=base_expected_depth_m,
                        frozen_basis_consensus_gate=consensus_gate.reshape(-1),
                    )
                else:
                    candidate_sigma = scalar_density_residual.redistribute(
                        base_sigma,
                        log_residual,
                        deltas,
                        base_expected_depth_m=base_expected_depth_m,
                    )
            else:
                candidate_sigma = scalar_density_residual.redistribute(
                    base_sigma,
                    log_residual,
                    deltas,
                )
            density_outputs["sigma"] = candidate_sigma.unsqueeze(-1)
        alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs["sigma"].squeeze(-1))  # [N, T+t]
        if self.active_sensor:
            alphas = 1 - torch.exp(-2 * deltas * self.density_scale * density_outputs["sigma"].squeeze(-1))  # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)  # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]  # [N, T+t]

        base_weights = None
        if "base_sigma" in density_outputs:
            base_alphas = 1 - torch.exp(
                -deltas
                * self.density_scale
                * density_outputs["base_sigma"].squeeze(-1)
            )
            if self.active_sensor:
                base_alphas = 1 - torch.exp(
                    -2
                    * deltas
                    * self.density_scale
                    * density_outputs["base_sigma"].squeeze(-1)
                )
            base_shifted = torch.cat(
                [
                    torch.ones_like(base_alphas[..., :1]),
                    1 - base_alphas + 1e-15,
                ],
                dim=-1,
            )
            base_weights = base_alphas * torch.cumprod(
                base_shifted, dim=-1
            )[..., :-1]

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        mask = weights > 1e-4  # hard coded
        if preserve_base_attributes:
            # The scalar candidate may move a sample across the attribute
            # evaluation threshold.  Attributes must nevertheless follow the
            # exact frozen Basis support, otherwise a geometry-only adapter
            # would silently change ray-drop or intensity.
            mask = base_weights > 1e-4
        elif base_weights is not None:
            # The base teacher must remain observable even where the learned
            # residual suppresses full density; otherwise the candidate can
            # alter the reference used by its own paired evaluation.
            mask = mask | (base_weights > 1e-4)
        query = getattr(self, "attribute_with_reference", None)
        query_args = (xyzs.reshape(-1, 3), dirs.reshape(-1, 3))
        if query is None:
            attr = self.attribute(*query_args, mask=mask.reshape(-1), **density_outputs)
            reference_intensity = None
        else:
            attr, reference_intensity = query(
                *query_args, mask=mask.reshape(-1), **density_outputs)

        attr = attr.view(N, -1, out_lidar_dim)  # [N, T+t, 3]

        attribute_weights = (
            base_weights if preserve_base_attributes else weights
        )

        # Scalar density fitting changes geometry only.  Its inherited
        # visibility/intensity aggregation stays bitwise on the base weights.
        weights_sum = attribute_weights.sum(dim=-1)  # [N]

        # calculate depth  Note: not real depth!!
        # ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        # depth = torch.sum(weights * ori_z_vals, dim=-1)
        depth = torch.sum(weights * z_vals, dim=-1)
        displacement = getattr(getattr(self, "scene_field", None), "scalar_density_residual", None)
        if getattr(displacement, "applies_depth_displacement", False):
            depth = displacement.correct_depth(
                depth, rays_o, rays_d, time, bound=self.bound,
                depth_scale=float(kwargs.get("scale", 0.0)),
                near=self.near_lidar, far=self.far_lidar,
            )

        # calculate lidar attributes
        image = torch.sum(attribute_weights.unsqueeze(-1) * attr, dim=-2)  # [N, 3], in [0, 1]

        image = image.view(*prefix, out_lidar_dim)
        depth = depth.view(*prefix)

        results = {
            "depth_lidar": depth,
            "image_lidar": image,
            "weights_sum_lidar": weights_sum,
            "weights": weights,
            "z_vals": z_vals,
        }
        if reference_intensity is not None:
            reference = torch.sum(
                attribute_weights * reference_intensity.reshape_as(attribute_weights), dim=-1)
            results["image_lidar_reference"] = torch.stack(
                [image[..., 0], reference.view(*prefix)], dim=-1)
        if base_weights is not None:
            base_depth = torch.sum(base_weights * z_vals, dim=-1)
            base_image = torch.sum(base_weights.unsqueeze(-1) * attr, dim=-2)
            motion = density_outputs["motion_prior"].view(N, num_steps, -1)
            base_weights_sum = base_weights.sum(dim=-1)
            motion = torch.sum(
                base_weights.unsqueeze(-1) * motion, dim=-2
            ).squeeze(-1)
            motion = motion / base_weights_sum.clamp_min(1e-6)
            results.update(
                {
                    "base_depth_lidar": base_depth.view(*prefix),
                    "base_image_lidar": base_image.view(*prefix, out_lidar_dim),
                    "motion_lidar": motion.view(*prefix).clamp(0.0, 1.0),
                }
            )
        return results

    def render(
        self,
        rays_o,
        rays_d,
        time,
        staged=False,
        max_ray_batch=4096,
        **kwargs
    ):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device
        aligned_consensus_gate = kwargs.get("frozen_basis_consensus_gate")
        if aligned_consensus_gate is not None:
            if not isinstance(aligned_consensus_gate, torch.Tensor):
                raise RuntimeError("frozen Basis consensus gate must be a tensor")
            if aligned_consensus_gate.shape != (B, N):
                raise RuntimeError(
                    "frozen Basis consensus gate shape must match the ray prefix"
                )
        aligned_learned_gate = kwargs.get("learned_residual_gate_weight")
        if aligned_learned_gate is not None:
            if not isinstance(aligned_learned_gate, torch.Tensor):
                raise RuntimeError("learned residual gate must be a tensor")
            if aligned_learned_gate.shape != (B, N):
                raise RuntimeError(
                    "learned residual gate shape must match the ray prefix"
                )

        if staged:
            out_lidar_dim = self.out_lidar_dim
            res_keys = ["depth_lidar", "image_lidar"]
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, out_lidar_dim), device=device)
            optional = {}

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    run_kwargs = kwargs
                    if (
                        aligned_consensus_gate is not None
                        or aligned_learned_gate is not None
                    ):
                        run_kwargs = dict(kwargs)
                    if aligned_consensus_gate is not None:
                        run_kwargs["frozen_basis_consensus_gate"] = (
                            aligned_consensus_gate[b : b + 1, head:tail]
                        )
                    if aligned_learned_gate is not None:
                        run_kwargs["learned_residual_gate_weight"] = (
                            aligned_learned_gate[b : b + 1, head:tail]
                        )
                    results_ = _run(
                        rays_o[b : b + 1, head:tail],
                        rays_d[b : b + 1, head:tail],
                        time[b:b+1],
                        **run_kwargs
                    )
                    depth[b : b + 1, head:tail] = results_[res_keys[0]]
                    image[b : b + 1, head:tail] = results_[res_keys[1]]
                    for key in (
                        "base_depth_lidar",
                        "base_image_lidar",
                        "motion_lidar",
                        "image_lidar_reference",
                    ):
                        if key not in results_:
                            continue
                        value = results_[key]
                        if key not in optional:
                            optional[key] = torch.empty(
                                (B, N, *value.shape[2:]),
                                dtype=value.dtype,
                                device=value.device,
                            )
                        optional[key][b : b + 1, head:tail] = value
                    head += max_ray_batch

            results = {}
            results[res_keys[0]] = depth
            results[res_keys[1]] = image
            results.update(optional)

        else:
            results = _run(rays_o, rays_d, time, **kwargs)

        return results
