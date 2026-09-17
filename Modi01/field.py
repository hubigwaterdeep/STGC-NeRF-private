"""Hash-only representation changes around the audited geometry-residual field."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
from torch import nn

from Modi01.runtime import activate_source
activate_source()
from best_core.scene_field import AnchoredSplineHighOrderGeometryResidualField
from best_core.field_budget import FieldBudgetReport
from Modi01.hash_encoding import (SpatialSchedule, prefix_encoding, LocalTemporalHash,
                                 NeuralCoefficients, normalized_times)


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())


@dataclass(frozen=True)
class RepresentationConfig:
    modal_levels: int = 4
    coefficients: str = 'tied'
    latent_dim: int = 4
    decoder_width: int = 32
    embedding_dim: int = 4
    spatial_frequencies: int = 2

    def validate(self, levels):
        for name in ('modal_levels', 'latent_dim', 'decoder_width', 'embedding_dim', 'spatial_frequencies'):
            if type(getattr(self, name)) is not int:
                raise ValueError(f'{name} must be an integer')
        if not 0 <= self.modal_levels <= levels:
            raise ValueError('modal_levels must be between 0 and n_levels_hash')
        if self.coefficients not in ('tied', 'untied', 'neural'):
            raise ValueError('coefficients must be tied, untied, or neural')
        if not self.modal_levels and self.coefficients != 'tied':
            raise ValueError('independent coefficients require retained modal levels')
        if self.latent_dim not in (1, 2, 4, 8) or self.decoder_width < 1 or self.embedding_dim < 1:
            raise ValueError('invalid decoder dimensions')
        if self.spatial_frequencies < 0:
            raise ValueError('spatial_frequencies cannot be negative')


class HybridTemporalField(AnchoredSplineHighOrderGeometryResidualField):
    def __init__(self, representation=None, **kwargs):
        config = representation or RepresentationConfig()
        config.validate(kwargs.get('n_levels_hash', 8))
        if kwargs.get('time_resolution', 8) != 8 or kwargs.get('n_features_per_level_hash', 4) != 4:
            raise ValueError('registered experiment requires rank/time_resolution=8 and four hash channels')
        super().__init__(**kwargs)
        self.representation = config
        self.spatial_schedule = SpatialSchedule(kwargs.get('base_resolution', 512),
            kwargs.get('max_resolution', 32768), self.n_levels_hash)
        self.reference_hash_parameters = parameter_count(self.modal_hashes)
        self.reference_field_parameters = parameter_count(self)
        modal = config.modal_levels
        self.local_temporal_hashes = nn.ModuleList()
        self.high_coefficient_hashes = nn.ModuleList()
        self.neural_coefficients = None
        self._original_forward = modal == self.n_levels_hash and config.coefficients == 'tied'
        if not self._original_forward:
            original = self.modal_hashes
            logs = [grid.encoding_config['log2_hashmap_size'] for grid in original]
            low_logs = [size - int(config.coefficients != 'tied') for size in logs]
            # Preserve shared module initialization and the caller's RNG state.
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                lows = nn.ModuleList()
                for role, size in enumerate(low_logs):
                    if modal:
                        low = prefix_encoding(self.spatial_schedule, modal, self.rank, size)
                        if config.coefficients == 'tied':
                            with torch.no_grad():
                                low.params.copy_(original[role].params[:low.params.numel()])
                        lows.append(low)
                    if modal and config.coefficients == 'untied':
                        high = prefix_encoding(self.spatial_schedule, modal, self.rank, size)
                        # Independent storage, same initialized coefficients.
                        with torch.no_grad():
                            high.params.copy_(lows[-1].params)
                        self.high_coefficient_hashes.append(high)
                    if modal < self.n_levels_hash:
                        self.local_temporal_hashes.append(LocalTemporalHash(
                            self.spatial_schedule, range(modal, self.n_levels_hash), logs[role] - 2))
                self.modal_hashes = lows
                if config.coefficients == 'neural':
                    # A bounded, native-table budget. Decoder and embeddings are
                    # included; power-of-two granularity can leave unused budget.
                    remaining = self.reference_hash_parameters - self.hash_parameter_count()
                    latent_logs = [[size] * modal for size in logs]
                    while True:
                        candidate = NeuralCoefficients(self.spatial_schedule, range(modal),
                            latent_logs, self.rank, config.latent_dim, config.decoder_width,
                            config.embedding_dim, config.spatial_frequencies)
                        if parameter_count(candidate) <= remaining:
                            self.neural_coefficients = candidate
                            break
                        excess = parameter_count(candidate) - remaining
                        choices = [(parameter_count(grid) // 2, r, level)
                                   for r, latent in enumerate(candidate.latents)
                                   for level, grid in enumerate(latent.grids)
                                   if latent_logs[r][level] > 1]
                        if not choices:
                            raise ValueError('decoder cannot fit the registered hash budget')
                        enough = [choice for choice in choices if choice[0] >= excess]
                        _, role, level = min(enough) if enough else max(choices)
                        latent_logs[role][level] -= 1
                        del candidate
        self.requires_grad_(True)
        # Native dense tables may be below their caps in nondefault tiny configs.
        # Enforce fairness where independent coefficients require a fixed budget.
        if config.coefficients != 'tied' and self.hash_parameter_count() > self.reference_hash_parameters:
            raise ValueError('independent coefficient allocation exceeds reference hash budget')

    def hash_parameter_count(self):
        modules = [self.modal_hashes, self.local_temporal_hashes, self.high_coefficient_hashes]
        if self.neural_coefficients is not None:
            modules.append(self.neural_coefficients)
        return sum(parameter_count(m) for m in modules)

    def get_extra_state(self):
        return dict(version=1, representation=asdict(self.representation),
                    spatial_schedule=self.spatial_schedule.as_dict())

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('checkpoint has a different temporal representation or spatial schedule')

    def _hash_dynamic(self, xyz, t, context=None):
        times = normalized_times(t, len(xyz), device=xyz.device, dtype=xyz.dtype)
        if self._original_forward:
            return super()._hash_dynamic(xyz, t, context)
        modal = self.representation.modal_levels
        active = modal and float(self.high_order_refiner.progress) > 0
        shared = times.numel() == 1
        if modal:
            mean_phi = self.hash_basis(times)
            if shared:
                mean_phi = mean_phi[0]
        if active:
            residual_phi = self._high_order_phi(times, context)
            if shared:
                residual_phi = residual_phi[0]
            gates = self.high_order_hash_gates.sigmoid()[:, :modal]
        features = []
        for role, axes in enumerate(((0, 1), (0, 2), (1, 2))):
            coordinate = xyz[:, axes]
            pieces = []
            if modal:
                low = self.modal_hashes[role](coordinate).reshape(-1, modal, self.rank)
                contraction = 'nlr,r->nl' if shared else 'nlr,nr->nl'
                mean = torch.einsum(contraction, low, mean_phi.to(low.dtype))
                if active:
                    high = low
                    if self.representation.coefficients == 'untied':
                        high = self.high_coefficient_hashes[role](coordinate).reshape(-1, modal, self.rank)
                    elif self.representation.coefficients == 'neural':
                        high = self.neural_coefficients(coordinate, role).to(low.dtype)
                    weights = (residual_phi * gates[role].to(residual_phi.dtype) if shared else
                        residual_phi[:, None, :] * gates[role][None].to(residual_phi.dtype))
                    residual = torch.einsum('nlr,lr->nl' if shared else 'nlr,nlr->nl',
                                             high, weights.to(high.dtype))
                    mean = mean + self._safe_high_order_residual(mean, residual)
                pieces.append(mean)
            if modal < self.n_levels_hash:
                pieces.append(self.local_temporal_hashes[role](coordinate, t))
            features.append(torch.cat(pieces, -1))
        return torch.cat(features, -1)

    def parameter_groups(self, lr):
        groups = super().parameter_groups(lr)
        if self.local_temporal_hashes:
            groups.append(dict(params=list(self.local_temporal_hashes.parameters()), lr=lr, stage_role='base'))
        high = list(self.high_coefficient_hashes.parameters())
        if self.neural_coefficients is not None:
            high += list(self.neural_coefficients.parameters())
        if high:
            groups.append(dict(params=high, lr=lr, stage_role='high_order'))
        return groups

    def budget_report(self, name=None):
        base = super().budget_report(name or 'modi01')
        added = parameter_count(self.local_temporal_hashes) + parameter_count(self.high_coefficient_hashes)
        if self.neural_coefficients is not None:
            added += parameter_count(self.neural_coefficients)
        return FieldBudgetReport(base.name, base.static, base.dynamic + added, base.flow, base.other)

    def representation_report(self):
        actual = self.hash_parameter_count()
        return dict(configuration=asdict(self.representation), spatial_schedule=self.spatial_schedule.as_dict(),
            source_class=f'{type(self).__module__}.{type(self).__name__}',
            n_output_dims=self.n_output_dims, dynamic_hash_outputs=3 * self.n_levels_hash,
            roles=['xy', 'xz', 'yz'],
            inactive_hash_gate_values=3*(self.n_levels_hash-self.representation.modal_levels)*self.rank,
            reference_hash_parameters=self.reference_hash_parameters,
            hash_parameters=actual, hash_parameter_delta=actual-self.reference_hash_parameters,
            exact_hash_capacity_match=actual == self.reference_hash_parameters,
            field_parameters=parameter_count(self), trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            reference_field_parameters=self.reference_field_parameters,
            neural_order='interpolate spatial latent, then nonlinear decode; no time input',
            cap_scope='retained modal levels within each role; inherited RMS formula',
            table_parameters={name: parameter_count(module) for name, module in
                [('low', self.modal_hashes), ('local_temporal', self.local_temporal_hashes),
                 ('independent_high', self.high_coefficient_hashes)]},
            neural_parameters=parameter_count(self.neural_coefficients) if self.neural_coefficients is not None else 0)
