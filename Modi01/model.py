"""From-scratch renderer wrapper; the historical renderer and heads are reused."""
from __future__ import annotations
import torch
from Modi01.runtime import activate_source
activate_source()
from model.stgc_best import STGC_NeRF_Best
from Modi01.field import HybridTemporalField, RepresentationConfig


class STGCNeRFModi01(STGC_NeRF_Best):
    def __init__(self, representation=None, **kwargs):
        for option in ('intensity_feature_mode', 'intensity_readout_mode'):
            if kwargs.get(option, 'none') != 'none':
                raise ValueError('representation controls require unchanged appearance heads')
        cpu_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all()
        super().__init__(**kwargs)
        names = ('min_resolution', 'base_resolution', 'max_resolution', 'time_resolution',
                 'n_levels_plane', 'n_features_per_level_plane', 'n_levels_hash',
                 'n_features_per_level_hash', 'log2_hashmap_size', 'num_layers_flow',
                 'hidden_dim_flow', 'num_frames')
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.set_rng_state(cpu_state)
            torch.cuda.set_rng_state_all(cuda_states)
            replacement = HybridTemporalField(representation or RepresentationConfig(),
                                               **{k: kwargs[k] for k in names if k in kwargs})
            replacement.flow_net.reset_mlp_parameters(seed=torch.initial_seed())
        self.scene_field = replacement
        self.scene_field_name = 'modi01'
        self.requires_grad_(True)
        self.unet.requires_grad_(False)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        expected = self.scene_field.get_extra_state()
        if state_dict.get('scene_field._extra_state') != expected:
            raise ValueError('use a checkpoint with the same Modi01 representation configuration')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
