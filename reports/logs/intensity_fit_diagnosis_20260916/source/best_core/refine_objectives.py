"""Visibility-refinement objectives with explicit, fixed preset contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from best_core.geometry_residual_objectives import TAIL_ALIGNED_SAME_RAY_V1
from best_core.scalar_density_residual_training import (
    FROZEN_SCALAR_DENSITY_RESIDUAL_PILOT_V1,
)


HARD_TARGET_PROBABILITY_BCE = "hard_target_probability_bce"
BCE_EXPECTED_MASKED_DEPTH_SUPPORT_V1 = (
    "bce_expected_masked_depth_support_v1"
)
FROZEN_VISIBILITY_REFINE_V3 = "frozen_visibility_refine_v3"
BCE_DEPTH_INTENSITY_RISK_V1 = "bce_depth_intensity_risk_v1"
FROZEN_VISIBILITY_AFTER_GEOMETRY_RESIDUAL_V1 = (
    "frozen_visibility_after_geometry_residual_v1"
)
FROZEN_VISIBILITY_AFTER_TAIL_RESIDUAL_PILOT_V1 = (
    "frozen_visibility_after_tail_residual_pilot_v1"
)
FROZEN_VISIBILITY_AFTER_SCALAR_DENSITY_RESIDUAL_PILOT_V1 = (
    "frozen_visibility_after_scalar_density_residual_pilot_v1"
)
FROZEN_VISIBILITY_RESIDUAL_BASIS_V1 = (
    "frozen_visibility_residual_basis_v1"
)
VISIBILITY_REFINE_LOSS_PRESETS = (
    HARD_TARGET_PROBABILITY_BCE,
    BCE_EXPECTED_MASKED_DEPTH_SUPPORT_V1,
    BCE_DEPTH_INTENSITY_RISK_V1,
)

_PRESET_CONTRACTS: dict[str, dict[str, Any]] = {
    BCE_DEPTH_INTENSITY_RISK_V1: {
        "preset": BCE_DEPTH_INTENSITY_RISK_V1,
        "bce": True, "expected_masked_depth_support_risk": True,
        "expected_intensity_risk": True, "intensity_risk_weight": 1.0,
        "risk_weight": 1.0, "max_depth_m": 80.0,
    },
    HARD_TARGET_PROBABILITY_BCE: {
        "preset": HARD_TARGET_PROBABILITY_BCE,
        "bce": True,
        "expected_masked_depth_support_risk": False,
    },
    BCE_EXPECTED_MASKED_DEPTH_SUPPORT_V1: {
        "preset": BCE_EXPECTED_MASKED_DEPTH_SUPPORT_V1,
        "bce": True,
        "expected_masked_depth_support_risk": True,
        "risk_weight": 1.0,
        "max_depth_m": 80.0,
    },
}


def visibility_refine_loss_preset_contract(name: str) -> dict[str, Any]:
    """Return a copy of one registered visibility-refine loss contract."""

    try:
        return dict(_PRESET_CONTRACTS[name])
    except KeyError as error:
        raise ValueError(f"unknown visibility-refine loss preset {name!r}") from error


class VisibilityRefineLossTerms(NamedTuple):
    total: Tensor
    bce: Tensor
    expected_depth_risk: Tensor


class VisibilityRefineLoss(nn.Module):
    """Optimize hard return BCE with an optional frozen-depth support risk."""

    def __init__(self, *, preset: str, max_depth_m: float = 80.0) -> None:
        super().__init__()
        if preset not in VISIBILITY_REFINE_LOSS_PRESETS:
            raise ValueError(f"unknown visibility-refine loss preset {preset!r}")
        if max_depth_m <= 0.0:
            raise ValueError("maximum refine depth must be positive")
        self.preset = preset
        self.max_depth_m = float(max_depth_m)

    def forward(
        self,
        predicted_return: Tensor,
        target_return: Tensor,
        predicted_depth: Tensor | None = None,
        target_depth: Tensor | None = None,
        *,
        depth_scale: float = 1.0,
        predicted_intensity: Tensor | None = None,
        target_intensity: Tensor | None = None,
    ) -> VisibilityRefineLossTerms:
        if predicted_return.shape != target_return.shape:
            raise ValueError("predicted and target return tensors must share shape")
        bce = F.binary_cross_entropy(predicted_return, target_return)
        if self.preset == HARD_TARGET_PROBABILITY_BCE:
            zero = predicted_return.sum() * 0.0
            return VisibilityRefineLossTerms(bce, bce, zero)
        if predicted_depth is None or target_depth is None:
            raise ValueError("expected-depth refine loss requires both depth tensors")
        if predicted_depth.shape != predicted_return.shape:
            raise ValueError("predicted depth and return tensors must share shape")
        if target_depth.shape != target_return.shape:
            raise ValueError("target depth and return tensors must share shape")
        if depth_scale <= 0.0:
            raise ValueError("depth scale must be positive")

        predicted_depth_normalized = (
            (predicted_depth.detach() / float(depth_scale))
            .clamp(0.0, self.max_depth_m)
            .div(self.max_depth_m)
        )
        target_depth_normalized = (
            (target_depth.detach() / float(depth_scale))
            .clamp(0.0, self.max_depth_m)
            .div(self.max_depth_m)
        )
        risk = (
            (1.0 - target_return)
            * predicted_return
            * predicted_depth_normalized.square()
            + target_return
            * (1.0 - predicted_return)
            * target_depth_normalized.square()
        ).mean()
        total = bce + risk
        if self.preset == BCE_DEPTH_INTENSITY_RISK_V1:
            from best_core.appearance_objectives import expected_intensity_risk
            if predicted_intensity is None or target_intensity is None:
                raise ValueError("intensity-risk refine requires intensity tensors")
            total = total + expected_intensity_risk(predicted_return,
                                                    predicted_intensity, target_intensity)
        return VisibilityRefineLossTerms(total, bce, risk)


def visibility_refine_loss_from_preset(name: str) -> VisibilityRefineLoss:
    """Construct a visibility-refine loss from its immutable preset."""

    contract = visibility_refine_loss_preset_contract(name)
    return VisibilityRefineLoss(
        preset=name,
        max_depth_m=float(contract.get("max_depth_m", 80.0)),
    )


def unaugmented_visibility_targets(images_lidar: Tensor) -> tuple[Tensor, Tensor]:
    """Extract return/depth supervision before input masking is applied."""

    if images_lidar.ndim != 4 or images_lidar.shape[-1] < 3:
        raise ValueError("LiDAR images must have shape [B, H, W, C>=3]")
    return (
        images_lidar[..., 0].unsqueeze(1),
        images_lidar[..., 2].unsqueeze(1),
    )


def build_readable_refine_checkpoint_contract(
    *,
    source_checkpoint: str | Path,
    source_global_step: int,
    source_weights_kind: str,
    train_transform: str | Path,
    args_file: str | Path,
    init_seed: int,
    augmentation_seed: int,
    optimization_seed: int,
    steps: int,
    learning_rate: float,
    loss_preset: str,
    depth_scale: float,
    evaluation_split: str,
) -> dict[str, Any]:
    """Build the fixed, human-readable v5.5 refinement contract."""

    expected = {
        "source_global_step": (int(source_global_step), 30_000),
        "source_weights_kind": (source_weights_kind, "ema"),
        "init_seed": (int(init_seed), 0),
        "augmentation_seed": (int(augmentation_seed), 0),
        "optimization_seed": (int(optimization_seed), 0),
        "steps": (int(steps), 1_000),
        "learning_rate": (float(learning_rate), 0.001),
        "loss_preset": (
            loss_preset,
            BCE_EXPECTED_MASKED_DEPTH_SUPPORT_V1,
        ),
        "evaluation_split": (evaluation_split, "val"),
    }
    for field, (actual, required) in expected.items():
        if actual != required:
            raise ValueError(
                f"{FROZEN_VISIBILITY_REFINE_V3} requires "
                f"{field}={required!r}, got {actual!r}"
            )
    if depth_scale <= 0.0:
        raise ValueError("depth scale must be positive")

    source_path = Path(source_checkpoint).resolve()
    return {
        "protocol": FROZEN_VISIBILITY_REFINE_V3,
        "source_checkpoint": str(source_path),
        "source_checkpoint_name": source_path.name,
        "source_global_step": int(source_global_step),
        "source_weights_kind": source_weights_kind,
        "mutable_state_prefix": "unet.",
        "train_transform": str(Path(train_transform).resolve()),
        "args_file": str(Path(args_file).resolve()),
        "init_seed": int(init_seed),
        "augmentation_seed": int(augmentation_seed),
        "optimization_seed": int(optimization_seed),
        "optimization_rng": "torch_and_cuda_global_before_refine_optimizer_v1",
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "loss_preset": loss_preset,
        "risk_weight": 1.0,
        "max_depth_m": 80.0,
        "depth_source": "unaugmented_frozen_renderer_detached",
        "depth_scale": float(depth_scale),
        "evaluation_split": evaluation_split,
        "return_threshold": 0.5,
    }


def build_readable_refine_after_geometry_residual_checkpoint_contract(
    *,
    source_checkpoint: str | Path,
    source_global_step: int,
    source_weights_kind: str,
    source_geometry_protocol: str,
    train_transform: str | Path,
    args_file: str | Path,
    init_seed: int,
    augmentation_seed: int,
    optimization_seed: int,
    steps: int,
    learning_rate: float,
    loss_preset: str,
    depth_scale: float,
    evaluation_split: str,
) -> dict[str, Any]:
    """Describe the shared visibility refiner applied after geometry residual fitting."""

    expected = {
        "source_global_step": (int(source_global_step), 34_000),
        "source_weights_kind": (source_weights_kind, "model"),
        "source_geometry_protocol": (
            source_geometry_protocol,
            "frozen_geometry_residual_basis_v1",
        ),
        "init_seed": (int(init_seed), 0),
        "augmentation_seed": (int(augmentation_seed), 0),
        "optimization_seed": (int(optimization_seed), 0),
        "steps": (int(steps), 1_000),
        "learning_rate": (float(learning_rate), 0.001),
        "loss_preset": (loss_preset, BCE_EXPECTED_MASKED_DEPTH_SUPPORT_V1),
        "evaluation_split": (evaluation_split, "val"),
    }
    for field, (actual, required) in expected.items():
        if actual != required:
            raise ValueError(
                f"{FROZEN_VISIBILITY_AFTER_GEOMETRY_RESIDUAL_V1} requires "
                f"{field}={required!r}, got {actual!r}"
            )
    if depth_scale <= 0.0:
        raise ValueError("depth scale must be positive")

    source_path = Path(source_checkpoint).resolve()
    return {
        "protocol": FROZEN_VISIBILITY_AFTER_GEOMETRY_RESIDUAL_V1,
        "source_checkpoint": str(source_path),
        "source_checkpoint_name": source_path.name,
        "source_global_step": 34_000,
        "source_weights_kind": "model",
        "source_geometry_protocol": source_geometry_protocol,
        "mutable_state_prefix": "unet.",
        "train_transform": str(Path(train_transform).resolve()),
        "args_file": str(Path(args_file).resolve()),
        "init_seed": 0,
        "augmentation_seed": 0,
        "optimization_seed": 0,
        "optimization_rng": "torch_and_cuda_global_before_refine_optimizer_v1",
        "steps": 1_000,
        "learning_rate": 0.001,
        "loss_preset": BCE_EXPECTED_MASKED_DEPTH_SUPPORT_V1,
        "risk_weight": 1.0,
        "max_depth_m": 80.0,
        "depth_source": "unaugmented_frozen_renderer_detached",
        "depth_scale": float(depth_scale),
        "evaluation_split": "val",
        "return_threshold": 0.5,
        "geometry_residual_frozen": True,
    }


def build_readable_refine_after_tail_residual_pilot_checkpoint_contract(
    *,
    source_checkpoint: str | Path,
    source_global_step: int,
    source_weights_kind: str,
    source_geometry_protocol: str,
    source_geometry_objective_preset: str,
    train_transform: str | Path,
    args_file: str | Path,
    init_seed: int,
    augmentation_seed: int,
    optimization_seed: int,
    steps: int,
    learning_rate: float,
    loss_preset: str,
    depth_scale: float,
    evaluation_split: str,
) -> dict[str, Any]:
    """Describe visibility refinement of the fixed 1000-step tail pilot."""

    expected = {
        "source_global_step": (int(source_global_step), 31_000),
        "source_weights_kind": (source_weights_kind, "model"),
        "source_geometry_protocol": (
            source_geometry_protocol,
            "frozen_geometry_residual_basis_v1",
        ),
        "source_geometry_objective_preset": (
            source_geometry_objective_preset,
            TAIL_ALIGNED_SAME_RAY_V1,
        ),
        "init_seed": (int(init_seed), 0),
        "augmentation_seed": (int(augmentation_seed), 0),
        "optimization_seed": (int(optimization_seed), 0),
        "steps": (int(steps), 1_000),
        "learning_rate": (float(learning_rate), 0.001),
        "loss_preset": (
            loss_preset,
            BCE_EXPECTED_MASKED_DEPTH_SUPPORT_V1,
        ),
        "evaluation_split": (evaluation_split, "val"),
    }
    for field, (actual, required) in expected.items():
        if actual != required:
            raise ValueError(
                f"{FROZEN_VISIBILITY_AFTER_TAIL_RESIDUAL_PILOT_V1} "
                f"requires {field}={required!r}, got {actual!r}"
            )
    if depth_scale <= 0.0:
        raise ValueError("depth scale must be positive")

    source_path = Path(source_checkpoint).resolve()
    return {
        "protocol": FROZEN_VISIBILITY_AFTER_TAIL_RESIDUAL_PILOT_V1,
        "source_checkpoint": str(source_path),
        "source_checkpoint_name": source_path.name,
        "source_global_step": 31_000,
        "source_weights_kind": "model",
        "source_geometry_protocol": source_geometry_protocol,
        "source_geometry_objective_preset": (
            source_geometry_objective_preset
        ),
        "mutable_state_prefix": "unet.",
        "train_transform": str(Path(train_transform).resolve()),
        "args_file": str(Path(args_file).resolve()),
        "init_seed": 0,
        "augmentation_seed": 0,
        "optimization_seed": 0,
        "optimization_rng": "torch_and_cuda_global_before_refine_optimizer_v1",
        "steps": 1_000,
        "learning_rate": 0.001,
        "loss_preset": BCE_EXPECTED_MASKED_DEPTH_SUPPORT_V1,
        "risk_weight": 1.0,
        "max_depth_m": 80.0,
        "depth_source": "unaugmented_frozen_renderer_detached",
        "depth_scale": float(depth_scale),
        "evaluation_split": "val",
        "return_threshold": 0.5,
        "geometry_residual_frozen": True,
    }


def build_readable_refine_after_scalar_density_residual_pilot_checkpoint_contract(
    *,
    source_checkpoint: str | Path,
    source_global_step: int,
    source_weights_kind: str,
    source_scalar_protocol: str,
    source_scalar_target_field: str,
    train_transform: str | Path,
    args_file: str | Path,
    init_seed: int,
    augmentation_seed: int,
    optimization_seed: int,
    steps: int,
    learning_rate: float,
    loss_preset: str,
    depth_scale: float,
    evaluation_split: str,
) -> dict[str, Any]:
    """Describe visibility refinement of the fixed scalar-density pilot."""

    expected = {
        "source_global_step": (int(source_global_step), 31_000),
        "source_weights_kind": (source_weights_kind, "model"),
        "source_scalar_protocol": (
            source_scalar_protocol,
            FROZEN_SCALAR_DENSITY_RESIDUAL_PILOT_V1,
        ),
        "init_seed": (int(init_seed), 0),
        "augmentation_seed": (int(augmentation_seed), 0),
        "optimization_seed": (int(optimization_seed), 0),
        "steps": (int(steps), 1_000),
        "learning_rate": (float(learning_rate), 0.001),
        "loss_preset": (
            loss_preset,
            BCE_EXPECTED_MASKED_DEPTH_SUPPORT_V1,
        ),
        "evaluation_split": (evaluation_split, "val"),
    }
    for field, (actual, required) in expected.items():
        if actual != required:
            raise ValueError(
                f"{FROZEN_VISIBILITY_AFTER_SCALAR_DENSITY_RESIDUAL_PILOT_V1} "
                f"requires {field}={required!r}, got {actual!r}"
            )
    if source_scalar_target_field not in {
        "anchored_spline_high_order_scalar_density_residual",
        "anchored_spline_high_order_far_gated_scalar_density_residual",
        "anchored_spline_high_order_coherence_gated_scalar_density_residual",
    }:
        raise ValueError(
            f"{FROZEN_VISIBILITY_AFTER_SCALAR_DENSITY_RESIDUAL_PILOT_V1} "
            "requires source_scalar_target_field to be a registered "
            "scalar-density residual field"
        )
    if depth_scale <= 0.0:
        raise ValueError("depth scale must be positive")

    source_path = Path(source_checkpoint).resolve()
    return {
        "protocol": FROZEN_VISIBILITY_AFTER_SCALAR_DENSITY_RESIDUAL_PILOT_V1,
        "source_checkpoint": str(source_path),
        "source_checkpoint_name": source_path.name,
        "source_global_step": 31_000,
        "source_weights_kind": "model",
        "source_scalar_protocol": source_scalar_protocol,
        "source_scalar_target_field": source_scalar_target_field,
        "mutable_state_prefix": "unet.",
        "train_transform": str(Path(train_transform).resolve()),
        "args_file": str(Path(args_file).resolve()),
        "init_seed": 0,
        "augmentation_seed": 0,
        "optimization_seed": 0,
        "optimization_rng": "torch_and_cuda_global_before_refine_optimizer_v1",
        "steps": 1_000,
        "learning_rate": 0.001,
        "loss_preset": BCE_EXPECTED_MASKED_DEPTH_SUPPORT_V1,
        "risk_weight": 1.0,
        "max_depth_m": 80.0,
        "depth_source": "unaugmented_frozen_renderer_detached",
        "depth_scale": float(depth_scale),
        "evaluation_split": "val",
        "return_threshold": 0.5,
        "scalar_density_residual_frozen": True,
    }


def build_readable_residual_basis_checkpoint_contract(
    *,
    source_checkpoint: str | Path,
    source_global_step: int,
    source_weights_kind: str,
    train_transform: str | Path,
    args_file: str | Path,
    init_seed: int,
    augmentation_seed: int,
    optimization_seed: int,
    steps: int,
    learning_rate: float,
    loss_preset: str,
    residual_basis: str,
    depth_scale: float,
    evaluation_split: str,
) -> dict[str, Any]:
    """Build the readable residual-only visibility refinement contract."""

    from best_core.visibility_residual_basis import EDGE_FAR_MULTISCALE_RESIDUAL_V1

    expected = {
        "source_global_step": (int(source_global_step), 30_000),
        "source_weights_kind": (source_weights_kind, "model"),
        "init_seed": (int(init_seed), 0),
        "augmentation_seed": (int(augmentation_seed), 0),
        "optimization_seed": (int(optimization_seed), 0),
        "steps": (int(steps), 1_000),
        "learning_rate": (float(learning_rate), 0.001),
        "loss_preset": (
            loss_preset,
            BCE_EXPECTED_MASKED_DEPTH_SUPPORT_V1,
        ),
        "residual_basis": (
            residual_basis,
            EDGE_FAR_MULTISCALE_RESIDUAL_V1,
        ),
        "evaluation_split": (evaluation_split, "val"),
    }
    for field, (actual, required) in expected.items():
        if actual != required:
            raise ValueError(
                f"{FROZEN_VISIBILITY_RESIDUAL_BASIS_V1} requires "
                f"{field}={required!r}, got {actual!r}"
            )
    if depth_scale <= 0.0:
        raise ValueError("depth scale must be positive")

    source_path = Path(source_checkpoint).resolve()
    return {
        "protocol": FROZEN_VISIBILITY_RESIDUAL_BASIS_V1,
        "source_checkpoint": str(source_path),
        "source_checkpoint_name": source_path.name,
        "source_global_step": int(source_global_step),
        "source_weights_kind": source_weights_kind,
        "mutable_state_prefix": "unet.residual_basis.",
        "train_transform": str(Path(train_transform).resolve()),
        "args_file": str(Path(args_file).resolve()),
        "init_seed": int(init_seed),
        "augmentation_seed": int(augmentation_seed),
        "optimization_seed": int(optimization_seed),
        "optimization_rng": "torch_and_cuda_global_before_refine_optimizer_v1",
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "loss_preset": loss_preset,
        "risk_weight": 1.0,
        "max_depth_m": 80.0,
        "depth_source": "unaugmented_frozen_renderer_detached",
        "depth_scale": float(depth_scale),
        "evaluation_split": evaluation_split,
        "return_threshold": 0.5,
        "residual_basis": residual_basis,
        "residual_target": "visibility_logit",
        "base_refiner_frozen": True,
    }


def visibility_refine_mutable_prefixes(protocol: str) -> tuple[str, ...]:
    """Return the sole mutable namespace for a readable refine protocol."""

    if protocol in {
        FROZEN_VISIBILITY_REFINE_V3,
        FROZEN_VISIBILITY_AFTER_GEOMETRY_RESIDUAL_V1,
        FROZEN_VISIBILITY_AFTER_TAIL_RESIDUAL_PILOT_V1,
        FROZEN_VISIBILITY_AFTER_SCALAR_DENSITY_RESIDUAL_PILOT_V1,
    }:
        return ("unet.",)
    if protocol == FROZEN_VISIBILITY_RESIDUAL_BASIS_V1:
        return ("unet.residual_basis.",)
    raise ValueError(f"unknown readable visibility refine protocol {protocol!r}")


def configure_visibility_refine_trainability(
    model: nn.Module,
    *,
    protocol: str,
    model_prefix: str = "",
) -> list[nn.Parameter]:
    """Configure modes/ownership and return one protocol's trainable parameters."""

    prefixes = visibility_refine_mutable_prefixes(protocol)
    model.eval()
    for prefix in prefixes:
        if not prefix.startswith(model_prefix):
            raise ValueError(
                f"mutable namespace {prefix!r} is outside model prefix {model_prefix!r}"
            )
        local_module = prefix[len(model_prefix) :].rstrip(".")
        if not local_module:
            raise ValueError("mutable namespace must select a child module")
        model.get_submodule(local_module).train()
    selected = []
    for name, parameter in model.named_parameters():
        qualified_name = f"{model_prefix}{name}"
        mutable = any(qualified_name.startswith(prefix) for prefix in prefixes)
        parameter.requires_grad_(mutable)
        if mutable:
            selected.append(parameter)
    if not selected:
        raise RuntimeError(f"{protocol} selected no trainable parameters")
    return selected


def snapshot_frozen_state(
    model: nn.Module,
    *,
    mutable_prefixes: tuple[str, ...],
) -> dict[str, Tensor]:
    """Clone all state tensors outside explicitly mutable namespaces."""

    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not any(name.startswith(prefix) for prefix in mutable_prefixes)
    }


def verify_frozen_state(
    model: nn.Module,
    expected: Mapping[str, Tensor],
    *,
    mutable_prefixes: tuple[str, ...],
) -> None:
    """Verify exact frozen-state equality through readable tensor names."""

    current = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not any(name.startswith(prefix) for prefix in mutable_prefixes)
    }
    for name in expected:
        if name not in current:
            raise RuntimeError(f"frozen model tensor is missing: {name}")
        if not torch.equal(current[name], expected[name]):
            raise RuntimeError(
                f"visibility refinement modified frozen model tensor: {name}"
            )
    for name in current:
        if name not in expected:
            raise RuntimeError(f"unexpected frozen model tensor appeared: {name}")


def snapshot_frozen_non_unet_state(
    model: nn.Module,
) -> dict[str, Tensor]:
    """Clone every state tensor outside the mutable refiner namespace."""

    return snapshot_frozen_state(model, mutable_prefixes=("unet.",))


def verify_frozen_non_unet_state(
    model: nn.Module,
    expected: Mapping[str, Tensor],
) -> None:
    """Fail with the concrete tensor name when frozen state changes."""

    verify_frozen_state(
        model,
        expected,
        mutable_prefixes=("unet.",),
    )
