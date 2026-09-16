"""Small independent plane coefficients, with the existing field's query rules."""
from contextlib import contextmanager
import torch
from torch import nn
from best_core.scene_field import BasisTimeOnlyField, _HighOrderResidualMixin
from model.stgc_best import STGC_NeRF_Best


@contextmanager
def preserve_diagnostics(field):
    # The original field records diagnostics on queries. The appearance query
    # must not change those diagnostics or its regularization bookkeeping.
    modules = (field, field.high_order_refiner)
    saved = [(m, name, value.detach().clone()) for m in modules
             for name, value in m.named_buffers(recurse=False)
             if name in m._non_persistent_buffers_set]
    energy = field._high_order_energy_terms
    regularization = field.high_order_refiner._last_regularization
    field._high_order_energy_terms = []
    try:
        yield
    finally:
        with torch.no_grad():
            for module, name, value in saved:
                getattr(module, name).copy_(value)
        field._high_order_energy_terms = energy
        field.high_order_refiner._last_regularization = regularization


class QueryView:
    """Read-only facade: own subset tensors plus frozen remaining field values.

    No module swapping, functional_call mutation, or writable parameter alias.
    Existing query implementations operate on this private view.
    """
    def __init__(self, field, bank):
        self.field = field
        for role in bank.roles:
            original = getattr(field, role)
            setattr(self, role, [
                [torch.cat((bank.coefficients[f'{role}_{level}_{axis}'], tensor[1:].detach()), 0)
                 if level < 2 else tensor.detach()
                 for axis, tensor in enumerate(axes)]
                for level, axes in enumerate(original)])

    def __getattr__(self, name):
        return getattr(self.field, name)

    _plane_static = BasisTimeOnlyField._plane_static
    _plane_dynamic = _HighOrderResidualMixin._plane_dynamic

    def _safe_high_order_residual(self, base, residual):
        # Same amplitude rule, with no shared diagnostic writes.
        base_rms = (base.float().square().mean() + 1e-12).sqrt().clamp_min(1e-6)
        residual_rms = (residual.float().square().mean() + 1e-12).sqrt()
        return residual * (0.5 * base_rms / residual_rms).clamp(max=1.0).to(residual.dtype)

    def _hash_dynamic(self, *args):
        with torch.no_grad():
            return self.field._hash_dynamic(*args)


class CoefficientBank(nn.Module):
    roles = ('static_planes', 'modal_axes', 'residual_modal_axes')

    def __init__(self, field):
        super().__init__()
        if field._high_order_stochastic or field.rank != 8:
            raise ValueError('requires deterministic rank-eight Best field')
        self.coefficients = nn.ParameterDict()
        self.mapping = []
        for role in self.roles:
            for level in range(2):
                for axis, source in enumerate(getattr(field, role)[level]):
                    key = f'{role}_{level}_{axis}'
                    value = nn.Parameter(source[0:1].detach().clone())
                    assert value.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
                    self.coefficients[key] = value
                    self.mapping.append(dict(name=key, source=f'scene_field.{role}.{level}.{axis}',
                                             channels=[0], shape=list(value.shape), count=value.numel()))
        assert sum(p.numel() for p in self.parameters()) == 19968

    def forward(self, field, xyz, time):
        with preserve_diagnostics(field):
            view = QueryView(field, self)
            features = BasisTimeOnlyField._query_features(view, xyz.detach(), time.detach(),
                                                          field._sample_query_context())
            return field.fusion(features)


class IntensityCorrection(nn.Module):
    def __init__(self, field, direction_dim, train_coefficients, seed=0):
        super().__init__()
        self.bank = CoefficientBank(field)
        self.bank.requires_grad_(train_coefficients)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.decoder = nn.Sequential(nn.Linear(120 + 15 + direction_dim, 64), nn.ReLU(),
                                         nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
            nn.init.zeros_(self.decoder[-1].weight)
            nn.init.zeros_(self.decoder[-1].bias)
        assert sum(p.numel() for p in self.decoder.parameters()) == 17537

    def forward(self, features, geo, direction):
        return self.decoder(torch.cat((features.float(), geo.detach().float(), direction.detach().float()), -1))


def attach_correction(model, *, train_coefficients, seed=0):
    # Class has no constructor/state changes; only an isolated new module is added.
    model.requires_grad_(False).eval()
    model.__class__ = CoefficientIntensityModel
    model.intensity_correction = IntensityCorrection(model.scene_field, model.view_encoder.n_output_dims,
                                                     train_coefficients, seed).cuda()
    return model


class CoefficientIntensityModel(STGC_NeRF_Best):
    def density(self, x, t=None):
        with torch.no_grad():
            output = super().density(x, t)
        output['coefficient_features'] = self.intensity_correction.bank(
            self.scene_field, (x + self.bound) / (2 * self.bound), t)
        return output

    def attribute_with_reference(self, x, d, mask=None, geo_feat=None, coefficient_features=None, **kwargs):
        assert mask is not None
        original = torch.zeros(len(x), self.out_lidar_dim, dtype=x.dtype, device=x.device)
        candidate = original.clone()
        if mask.any():
            with torch.no_grad():
                direction = self.view_encoder((d[mask] + 1) / 2)
                joined = torch.cat((direction, geo_feat[mask]), -1)
                logits = self.intensity_net(joined)
                intensity = torch.sigmoid(logits)
                raydrop = torch.sigmoid(self.raydrop_net(joined))
                original[mask] = torch.cat((raydrop, intensity), -1).to(original.dtype)
            correction = self.intensity_correction(coefficient_features[mask], geo_feat[mask], direction)
            corrected = torch.sigmoid(logits.detach() + correction.to(logits.dtype))
            candidate[mask] = torch.cat((raydrop, corrected), -1).to(candidate.dtype)
        self._candidate_samples = candidate[:, 1]
        return candidate, original[:, 1]

    def run(self, *args, **kwargs):
        result = super().run(*args, **kwargs)
        weights = result['weights'].detach()
        # Canonical paired one-channel integration on BOTH paths ensures exact
        # stage-zero identity, without changing raydrop/depth/weights or U-Net.
        integrated = (weights * self._candidate_samples.reshape_as(weights)).sum(-1)
        result['image_lidar'] = torch.stack((result['image_lidar'][..., 0],
                                           integrated.reshape_as(result['depth_lidar'])), -1)
        self._candidate_samples = None
        return result
