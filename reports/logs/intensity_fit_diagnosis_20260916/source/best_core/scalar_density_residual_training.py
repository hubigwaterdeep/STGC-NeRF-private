"""Training contract for the sign-stable scalar density residual pilot."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import torch
import torch.nn as nn

from best_core.scalar_density_residual import (
    far_range_gate_contract,
    spatial_consensus_gate_contract,
)


SCALAR_DENSITY_RESIDUAL_MUTABLE_PREFIX = "scene_field.scalar_density_residual."
FROZEN_SCALAR_DENSITY_RESIDUAL_PILOT_V1 = (
    "frozen_scalar_density_residual_pilot_v1"
)
TAIL_ALIGNED_FULL_1024_V1 = "tail_aligned_full_1024_bounded_same_ray_v1"
SCALAR_DENSITY_RESIDUAL_FIELD = (
    "anchored_spline_high_order_scalar_density_residual"
)
FAR_GATED_SCALAR_DENSITY_RESIDUAL_FIELD = (
    "anchored_spline_high_order_far_gated_scalar_density_residual"
)
COHERENCE_GATED_SCALAR_DENSITY_RESIDUAL_FIELD = (
    "anchored_spline_high_order_coherence_gated_scalar_density_residual"
)


class FullSamplerTailTerms(NamedTuple):
    """Balanced body/tail risk over the complete upstream sampler output."""

    total: torch.Tensor
    body: torch.Tensor
    tail: torch.Tensor


class FullSamplerTailObjective(nn.Module):
    """Consume all 1024 pre-render selected returns without another ranking."""

    def forward(
        self,
        pred_depth_m: torch.Tensor,
        target_depth_m: torch.Tensor,
        target_return: torch.Tensor,
        body_mask: torch.Tensor,
        tail_mask: torch.Tensor,
    ) -> FullSamplerTailTerms:
        values = (
            pred_depth_m,
            target_depth_m,
            target_return,
            body_mask,
            tail_mask,
        )
        if pred_depth_m.ndim != 2 or pred_depth_m.shape[1] != 1024:
            raise ValueError("scalar density objective requires exactly 1024 rays")
        if any(value.shape != pred_depth_m.shape for value in values[1:]):
            raise ValueError("scalar density objective tensors must share shape")
        if not all(
            torch.isfinite(value.float()).all()
            for value in (pred_depth_m, target_depth_m, target_return)
        ):
            raise ValueError("scalar density objective tensors must be finite")
        body = body_mask.bool()
        tail = tail_mask.bool()
        support = target_return.detach() > 0.5
        if not support.all():
            raise ValueError("scalar density objective requires GT-return rays")
        if torch.any(body & tail) or not torch.all(body | tail):
            raise ValueError("body and tail masks must partition every sampler ray")
        if not torch.all(body.sum(dim=-1) == 512):
            raise ValueError("scalar density objective requires 512 body rays")
        if not torch.all(tail.sum(dim=-1) == 512):
            raise ValueError("scalar density objective requires 512 tail rays")

        squared_error = (pred_depth_m - target_depth_m.detach()).square()
        bounded_square = squared_error / (25.0 + squared_error)
        body_risk = bounded_square[body].reshape(pred_depth_m.shape[0], 512).mean()
        tail_risk = bounded_square[tail].reshape(pred_depth_m.shape[0], 512).mean()
        return FullSamplerTailTerms(
            total=0.5 * body_risk + 0.5 * tail_risk,
            body=body_risk,
            tail=tail_risk,
        )


def configure_scalar_density_residual_trainability(
    model: nn.Module,
) -> list[nn.Parameter]:
    """Freeze Basis, sigma, readout, and visibility; expose only the adapter."""

    model.eval()
    try:
        residual = model.get_submodule(
            SCALAR_DENSITY_RESIDUAL_MUTABLE_PREFIX.rstrip(".")
        )
    except AttributeError as error:
        raise RuntimeError(
            "scalar density fitting requires "
            f"{SCALAR_DENSITY_RESIDUAL_MUTABLE_PREFIX.rstrip('.')}"
        ) from error
    residual.train()
    selected = []
    for name, parameter in model.named_parameters():
        mutable = name.startswith(SCALAR_DENSITY_RESIDUAL_MUTABLE_PREFIX)
        parameter.requires_grad_(mutable)
        if mutable:
            selected.append(parameter)
    if not selected:
        raise RuntimeError("scalar density residual selected no trainable parameters")
    return selected


def load_scalar_density_residual_source(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Materialize the raw 30000-step v5.4 Basis EMA into a fresh adapter model."""

    source_path = Path(checkpoint).resolve()
    if not source_path.is_file():
        raise RuntimeError(f"scalar density source does not exist: {source_path}")
    state = torch.load(source_path, map_location=device)
    if not isinstance(state, Mapping) or not isinstance(state.get("model"), Mapping):
        raise RuntimeError("scalar density source must contain model weights")
    if int(state.get("global_step", -1)) != 30_000:
        raise RuntimeError("scalar density source must be the 30000-step endpoint")
    if "refine_contract" in state:
        raise RuntimeError("scalar density source must precede visibility refinement")
    ema_state = state.get("ema")
    if not isinstance(ema_state, Mapping):
        raise RuntimeError("scalar density source requires raw EMA weights")

    source_model = state["model"]
    if any(
        str(name).startswith(SCALAR_DENSITY_RESIDUAL_MUTABLE_PREFIX)
        for name in source_model
    ):
        raise RuntimeError("scalar density source already contains residual weights")
    target_state = model.state_dict()
    new_state_names = {
        name
        for name in target_state
        if name.startswith(SCALAR_DENSITY_RESIDUAL_MUTABLE_PREFIX)
    }
    if not new_state_names:
        raise RuntimeError("target model has no scalar density residual state")
    inherited_names = set(target_state) - new_state_names
    source_names = {str(name) for name in source_model}
    if source_names != inherited_names:
        raise RuntimeError("raw Basis source does not exactly match inherited state")
    incompatible = [
        name
        for name in inherited_names
        if not torch.is_tensor(source_model[name])
        or source_model[name].shape != target_state[name].shape
    ]
    if incompatible:
        raise RuntimeError(
            f"raw Basis source has incompatible inherited tensors: {sorted(incompatible)}"
        )

    inherited_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith(SCALAR_DENSITY_RESIDUAL_MUTABLE_PREFIX)
    ]
    shadows = ema_state.get("shadow_params")
    if not isinstance(shadows, (list, tuple)) or len(shadows) != len(
        inherited_parameters
    ):
        raise RuntimeError("raw Basis source has incomplete EMA weights")
    load_state = dict(source_model)
    for (name, parameter), shadow in zip(inherited_parameters, shadows):
        if (
            name not in source_model
            or not torch.is_tensor(shadow)
            or shadow.shape != parameter.shape
            or shadow.dtype != parameter.dtype
        ):
            raise RuntimeError(f"raw Basis EMA parameter mismatch: {name}")
        load_state[name] = shadow.detach().clone()
    missing, unexpected = model.load_state_dict(load_state, strict=False)
    if set(missing) != new_state_names or unexpected:
        raise RuntimeError("validated raw Basis source failed strict loading")
    return {
        "source_checkpoint": str(source_path),
        "source_checkpoint_name": source_path.name,
        "source_global_step": 30_000,
        "source_epoch": int(state.get("epoch", 0)),
        "source_weights_kind": "ema",
        "source_protocol": "raw_basis_ema_step30000",
    }


def build_scalar_density_residual_contract(
    *,
    source_checkpoint: str | Path,
    source_epoch: int,
    train_transform: str | Path,
    args_file: str | Path,
    steps: int,
    mutable_parameter_names: tuple[str, ...] | list[str],
    optimizer_updates: int,
    gradient_overflows: int,
    target_field: str = SCALAR_DENSITY_RESIDUAL_FIELD,
    residual_holdout_frame_ids: Sequence[int] | None = None,
    optimization_frame_ids: Sequence[int] | None = None,
    surface_localization: bool = False,
    spatial_resolutions: Sequence[int] = (128, 256),
    representation: str = "basis",
) -> dict[str, Any]:
    """Describe the complete, readable scalar-only continuation."""

    if int(steps) not in {16, 1_000}:
        raise ValueError("scalar density pilot supports 16 or 1000 steps")
    if optimizer_updates < 0 or gradient_overflows < 0:
        raise ValueError("optimizer counters must be non-negative")
    if int(optimizer_updates) != int(steps) or int(gradient_overflows) != 0:
        raise ValueError(
            "every step must update and scalar density fitting permits no gradient overflows"
        )
    parameter_names = [str(name) for name in mutable_parameter_names]
    if not parameter_names or len(parameter_names) != len(set(parameter_names)):
        raise ValueError("mutable scalar density parameter names must be unique")
    if any(
        not name.startswith(SCALAR_DENSITY_RESIDUAL_MUTABLE_PREFIX)
        for name in parameter_names
    ):
        raise ValueError("mutable parameter escaped scalar density namespace")
    target_field = str(target_field)
    if target_field not in {
        SCALAR_DENSITY_RESIDUAL_FIELD,
        FAR_GATED_SCALAR_DENSITY_RESIDUAL_FIELD,
        COHERENCE_GATED_SCALAR_DENSITY_RESIDUAL_FIELD,
    }:
        raise ValueError("unsupported scalar density residual target field")
    if (residual_holdout_frame_ids is None) != (optimization_frame_ids is None):
        raise ValueError("residual holdout and optimization frames must be recorded together")
    readable_holdout: list[int] | None = None
    readable_optimization: list[int] | None = None
    if residual_holdout_frame_ids is not None and optimization_frame_ids is not None:
        readable_holdout = [int(value) for value in residual_holdout_frame_ids]
        readable_optimization = [int(value) for value in optimization_frame_ids]
        if len(readable_holdout) != 4 or len(set(readable_holdout)) != 4:
            raise ValueError("scalar residual contract requires four unique holdout frames")
        if len(readable_optimization) != 39 or len(set(readable_optimization)) != 39:
            raise ValueError("scalar residual contract requires 39 unique optimization frames")
        if set(readable_holdout) & set(readable_optimization):
            raise ValueError("scalar residual holdout and optimization frames overlap")

    resolutions = list(spatial_resolutions)
    if (len(resolutions) != 2 or any(not isinstance(r, int) or r < 2 for r in resolutions)
        or resolutions[0] >= resolutions[1]):
        raise ValueError("scalar residual resolutions must be two increasing integers >= 2")
    source_path = Path(source_checkpoint).resolve()
    contract = {
        "protocol": FROZEN_SCALAR_DENSITY_RESIDUAL_PILOT_V1,
        "source_checkpoint": str(source_path),
        "source_checkpoint_name": source_path.name,
        "source_global_step": 30_000,
        "source_epoch": int(source_epoch),
        "source_weights_kind": "ema",
        "source_field": "anchored_spline_high_order",
        "target_field": target_field,
        "mutable_state_prefix": SCALAR_DENSITY_RESIDUAL_MUTABLE_PREFIX,
        "mutable_parameter_names": parameter_names,
        "residual_application": "post_sigma",
        "temporal_basis": "fixed_linear_nonnegative_partition_of_unity_rank4",
        "spatial_resolutions": resolutions,
        "density_ratio_bounds": [0.5, 2.0],
        "ray_optical_mass": "preserved_by_delta_sigma_normalization",
        "attribute_aggregation": "frozen_base_weights",
        "visibility_refinement": False,
        "steps": int(steps),
        "max_learning_rate": 0.005,
        "scheduler": "one_cycle",
        "sampler": {
            "rays_per_step": 1024,
            "body": 512,
            "jump": 256,
            "far": 128,
            "jump_far": 128,
            "selection_stage": "before_render",
        },
        "loss_terms": [
            TAIL_ALIGNED_FULL_1024_V1,
            "gt_support_huber_geometry_v1",
        ],
        "same_ray_loss": {
            "candidate_selection": "all_1024_sampler_rays",
            "body_weight": 0.5,
            "tail_weight": 0.5,
            "bounded_square_denominator_m2": 25.0,
            "second_stage_top_k": False,
        },
        "optimizer_updates": int(optimizer_updates),
        "gradient_overflows": int(gradient_overflows),
        "output_global_step": 30_000 + int(steps),
    }
    if target_field == FAR_GATED_SCALAR_DENSITY_RESIDUAL_FIELD:
        contract["range_gate"] = far_range_gate_contract()
    if target_field == COHERENCE_GATED_SCALAR_DENSITY_RESIDUAL_FIELD:
        contract["spatial_consensus_gate"] = spatial_consensus_gate_contract()
    if readable_holdout is not None and readable_optimization is not None:
        contract["residual_holdout_frame_ids"] = readable_holdout
        contract["optimization_frame_ids"] = readable_optimization
    if surface_localization:
        from best_core.surface_localization import surface_localization_contract
        if target_field != SCALAR_DENSITY_RESIDUAL_FIELD:
            raise ValueError("surface localization requires the ungated scalar field")
        contract["protocol"] = "frozen_scalar_surface_localization_pilot_v1"
        contract["surface_localization"] = surface_localization_contract()
        contract["loss_terms"] = [*contract["loss_terms"], contract["surface_localization"]["preset"]]
    if representation not in {"basis", "latent_planes", "basis_depth_displacement"}:
        raise ValueError("unknown scalar residual representation")
    if representation == "latent_planes":
        from best_core.latent_density_residual import latent_feature_contract
        if target_field != SCALAR_DENSITY_RESIDUAL_FIELD or surface_localization:
            raise ValueError("latent pilot requires ungated original scalar objective")
        contract["protocol"] = "frozen_latent_feature_density_residual_pilot_v1"
        contract["representation"] = latent_feature_contract()
        contract["temporal_basis"] = "learned_spacetime_planes_with_four_bilinear_time_nodes"
    if representation == "basis_depth_displacement":
        from best_core.depth_displacement_residual import depth_displacement_contract
        if target_field != SCALAR_DENSITY_RESIDUAL_FIELD or surface_localization:
            raise ValueError("depth displacement requires ungated original scalar objective")
        contract["protocol"] = "frozen_basis_depth_displacement_pilot_v1"
        contract["representation"] = depth_displacement_contract()
        contract["residual_application"] = "post_render_depth"
        contract["density_ratio_bounds"] = [1.0, 1.0]
        contract["ray_optical_mass"] = "unchanged_base_density_and_weights"
    return contract
