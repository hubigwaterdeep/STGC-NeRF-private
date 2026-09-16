"""Independent appearance readouts for frozen-field controlled experiments."""

import torch
from torch import nn
from contextlib import contextmanager
import math


def matched_decoder_width(input_dim, parameter_budget, extra_parameters=0):
    """Closest integer width including biases and any component projectors."""
    available = parameter_budget - extra_parameters
    if available < input_dim + 5:
        raise ValueError("parameter budget is too small for this intensity readout")
    root = (-(input_dim + 3) + math.sqrt((input_dim + 3) ** 2 + 4 * (available - 1))) / 2
    candidates = {max(1, math.floor(root)), max(1, math.ceil(root))}
    return min(candidates, key=lambda h: (abs(h * h + (input_dim + 3) * h + 1 - available), h))


@contextmanager
def preserve_field_regularization(field):
    """Appearance queries must not add terms to the geometry objective."""
    saved = getattr(field, "_high_order_energy_terms", None)
    if saved is not None:
        field._high_order_energy_terms = []
    try:
        yield
    finally:
        if saved is not None:
            field._high_order_energy_terms = saved


def temporal_features(field, xyz, time, current_weight, query_dynamic, *, apply_fusion=True):
    """Query a separate appearance path; never change the density field's routing."""
    from best_core.scene_field import SceneFieldFeatures, _normalized_frame_time
    if time is None or time.numel() != 1:
        raise ValueError("appearance temporal queries require one frame time per render batch")
    frame = max(0, min(field.num_frames - 1, int(time.item() * (field.num_frames - 1))))
    with preserve_field_regularization(field):
        context = field._sample_query_context()
        with torch.no_grad():
            plane_static, grid_static = field._plane_static(xyz), field.hash_static(xyz)
        plane, grid = query_dynamic(field, xyz, time, context)
        if current_weight != 1.0:
            next_plane = previous_plane = plane
            next_grid = previous_grid = grid
            with torch.no_grad():
                flow = field.flow_net(field._xt(xyz, time))
            if frame < field.num_frames - 1:
                next_time = xyz.new_tensor(_normalized_frame_time(frame + 1, field.num_frames))
                next_plane, next_grid = query_dynamic(field, xyz + flow[:, :3], next_time, context)
            if frame > 0:
                previous_time = xyz.new_tensor(_normalized_frame_time(frame - 1, field.num_frames))
                previous_plane, previous_grid = query_dynamic(field, xyz + flow[:, 3:], previous_time, context)
            neighbor_weight = (1.0 - current_weight) / 2.0
            plane = current_weight * plane + neighbor_weight * (next_plane + previous_plane)
            grid = current_weight * grid + neighbor_weight * (next_grid + previous_grid)
        # The fusion weights are frozen, but later modal projectors may need
        # gradients through this operation to their own outputs.
        features = SceneFieldFeatures(plane_static, plane, grid_static, grid)
        return field.fusion(features) if apply_fusion else features.concatenate()


class IndependentIntensityReadout(nn.Module):
    MODES = ("geo", "field", "components", "field_pre_fusion")

    def __init__(self, field, direction_dim, *, mode="field", geo_dim=15,
                 width=64, seed=0, current_weight=0.5, parameter_budget=0):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown independent intensity mode: {mode}")
        if width < 1:
            raise ValueError("intensity decoder width must be positive")
        if not 0.0 <= current_weight <= 1.0:
            raise ValueError("current frame weight must be in [0, 1]")
        if mode == "geo" and current_weight != 0.5:
            raise ValueError("geo control uses the original pooled features; temporal override needs field mode")
        self.mode, self.seed = mode, seed
        self.current_weight = current_weight
        self.feature_dim = geo_dim if mode == "geo" else field.n_output_dims
        if mode == "field_pre_fusion":
            self.feature_dim = sum(field.fusion.input_dims)
        self.parameter_budget = parameter_budget
        if parameter_budget < 0:
            raise ValueError("parameter budget must be nonnegative")
        if parameter_budget:
            extra = 0
            if mode == "components":
                extra = sum(3 * axes[0].shape[0] ** 2 * field.rank * 2 for axes in field.modal_axes)
                extra += len(field.modal_hashes) * field.n_levels_hash ** 2 * field.rank * 2
            width = matched_decoder_width(self.feature_dim + direction_dim, parameter_budget, extra)
        self.width = width
        self.register_buffer("configuration", torch.tensor(
            [self.MODES.index(mode), current_weight, width, self.feature_dim, seed, parameter_budget],
            dtype=torch.float64))
        # Fork the CPU RNG only: constructing this control cannot change the
        # original model's initialization, CUDA RNG, or ray-sampling stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.decoder = nn.Sequential(
                nn.Linear(self.feature_dim + direction_dim, width), nn.ReLU(),
                nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
            if mode == "components":
                from best_core.intensity_components import ModalComponentProjectors
                self.components = ModalComponentProjectors(field)

    def encode(self, field, xyz, time, base, geo):
        if self.mode == "components":
            return temporal_features(field, xyz, time, self.current_weight, self.query_dynamic)
        if self.mode == "field_pre_fusion":
            with torch.no_grad():
                return temporal_features(field, xyz, time, self.current_weight, self.query_dynamic,
                                         apply_fusion=False)
        if self.mode == "field" and self.current_weight != 0.5:
            with torch.no_grad():
                return temporal_features(field, xyz, time, self.current_weight, self.query_dynamic)
        return (geo if self.mode == "geo" else base).detach()

    def query_dynamic(self, field, xyz, time, context):
        with torch.no_grad():
            plane, grid = (field._plane_dynamic(xyz, time, context),
                           field._hash_dynamic(xyz, time, context))
        if self.mode == "components":
            delta_plane, delta_grid = self.components(field, xyz, time, context)
            plane = plane + delta_plane.to(plane.dtype)
            grid = grid + delta_grid.to(grid.dtype)
        return plane, grid

    def forward(self, features, direction):
        return torch.sigmoid(self.decoder(torch.cat([features.float(), direction.float()], -1)))

    def contract(self):
        return {"version": 1, "mode": self.mode, "width": self.width,
                "seed": self.seed, "feature_dim": self.feature_dim,
                "current_weight": self.current_weight,
                "parameter_budget": self.parameter_budget,
                "trainable_parameters": sum(p.numel() for p in self.parameters()),
                "refiner_input": "frozen_original_intensity"}
