"""STGC's renderer interface backed by the complete named Best field."""

import torch
from best_core.lidar4d import LiDAR4D

BEST_FIELD = "anchored_spline_high_order_geometry_residual"


class STGC_NeRF_Best(LiDAR4D):
    def __init__(self, intensity_feature_mode="none", intensity_readout_mode="none",
                 intensity_readout_width=64, intensity_readout_seed=0,
                 intensity_current_weight=0.5, intensity_parameter_budget=0, **kwargs):
        super().__init__(scene_field=BEST_FIELD, **kwargs)
        # Best's historical residual-fitting constructor freezes the trunk.
        # A new STGC training run must optimize spatial/time/flow coefficients.
        self.requires_grad_(True)
        self.unet.requires_grad_(False)
        if intensity_feature_mode not in ("none", "base", "delta"):
            raise ValueError("unknown intensity feature mode")
        self.intensity_feature_mode = intensity_feature_mode
        if intensity_feature_mode != "none":
            # Isolate adapter initialization so all existing parameter/RNG
            # states remain identical between capacity-matched arms.
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(0)
                dim = kwargs.get("geo_feat_dim", 15)
                self.intensity_adapter = torch.nn.Sequential(
                    torch.nn.Linear(dim, 32), torch.nn.ReLU(), torch.nn.Linear(32, dim))
                torch.nn.init.zeros_(self.intensity_adapter[-1].weight)
                torch.nn.init.zeros_(self.intensity_adapter[-1].bias)
        if intensity_readout_mode != "none":
            if intensity_feature_mode != "none":
                raise ValueError("independent readouts cannot be combined with legacy adapters")
            from best_core.intensity_readout import IndependentIntensityReadout
            self.intensity_readout = IndependentIntensityReadout(
                self.scene_field, self.view_encoder.n_output_dims,
                mode=intensity_readout_mode, geo_dim=kwargs.get("geo_feat_dim", 15),
                width=intensity_readout_width, seed=intensity_readout_seed,
                current_weight=intensity_current_weight, parameter_budget=intensity_parameter_budget)
            self.requires_grad_(False)
            self.intensity_readout.requires_grad_(True)

    def attribute_with_reference(self, x, d, mask=None, geo_feat=None,
                                 intensity_features=None, **kwargs):
        if not hasattr(self, "intensity_readout"):
            return self.attribute(x, d, mask=mask, geo_feat=geo_feat, **kwargs), None
        reference = super().attribute(x, d, mask=mask, geo_feat=geo_feat, **kwargs)
        if intensity_features is None:
            raise ValueError("independent readout requires density's intensity_features")
        output = reference.clone()
        if mask is None or mask.any():
            selected_d = d if mask is None else d[mask]
            features = intensity_features if mask is None else intensity_features[mask]
            direction = self.view_encoder((selected_d + 1) / 2)
            intensity = self.intensity_readout(features, direction).squeeze(-1)
            if mask is None:
                output[:, 1] = intensity.to(output.dtype)
            else:
                output[mask, 1] = intensity.to(output.dtype)
        return output, reference[:, 1]

    def load_state_dict(self, state_dict, strict=True, assign=False):
        incoming = state_dict.get("intensity_readout.configuration")
        readout = getattr(self, "intensity_readout", None)
        has_readout_keys = any(k.startswith("intensity_readout.") for k in state_dict)
        if incoming is None and has_readout_keys:
            raise ValueError("intensity checkpoint is missing its readout configuration")
        if (incoming is not None) != (readout is not None):
            raise ValueError("intensity checkpoint requires its matching readout model; use the ablation loader")
        if incoming is not None and not torch.equal(incoming.cpu(), readout.configuration.cpu()):
            raise ValueError("intensity checkpoint has different readout/temporal settings")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def attribute(self, x, d, mask=None, geo_feat=None, full_geo_feat=None, **kwargs):
        if hasattr(self, "intensity_readout"):
            return self.attribute_with_reference(x, d, mask=mask, geo_feat=geo_feat, **kwargs)[0]
        if self.intensity_feature_mode == "none":
            return super().attribute(x, d, mask=mask, geo_feat=geo_feat, **kwargs)
        if full_geo_feat is None:
            raise ValueError("intensity adapter requires full and base geometry features")
        if mask is not None:
            output = torch.zeros(mask.shape[0], self.out_lidar_dim, dtype=x.dtype, device=x.device)
            if not mask.any():
                return output
            d, geo_feat, full_geo_feat = d[mask], geo_feat[mask], full_geo_feat[mask]
        encoded_d = self.view_encoder((d + 1) / 2)
        adapter_input = (geo_feat if self.intensity_feature_mode == "base"
                         else full_geo_feat - geo_feat)
        correction = self.intensity_adapter(adapter_input.detach().float())
        corrected = geo_feat + correction.to(geo_feat.dtype)
        intensity = torch.sigmoid(self.intensity_net(torch.cat([encoded_d, corrected], -1)))
        # The raw ray-drop pathway is deliberately identical to the baseline.
        raydrop = torch.sigmoid(self.raydrop_net(torch.cat([encoded_d, geo_feat], -1)))
        values = torch.cat([raydrop, intensity], -1)
        if mask is None:
            return values
        output[mask] = values.to(output.dtype)
        return output

    def get_params(self, lr):
        if hasattr(self, "intensity_readout"):
            return [{"params": self.intensity_readout.parameters(), "lr": 0.1 * lr,
                     "stage_role": "intensity_readout"}]
        groups = super().get_params(lr)
        if self.intensity_feature_mode != "none":
            groups.append({"params": self.intensity_adapter.parameters(),
                           "lr": 0.1 * lr, "stage_role": "intensity_adapter"})
        return groups


def load_best_weights(model, checkpoint):
    """Load all endpoint weights without inheriting its training counters."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contract = state.get("refine_contract", {})
    if contract.get("protocol") != "frozen_visibility_after_geometry_residual_v1":
        raise ValueError("initialization requires the named Best refined checkpoint")
    if contract.get("loss_preset") != "bce_expected_masked_depth_support_v1":
        raise ValueError("Best checkpoint has a different visibility objective")
    model.load_state_dict(state["model"], strict=True)
    return {"checkpoint": str(checkpoint), "source_global_step": state["global_step"]}
