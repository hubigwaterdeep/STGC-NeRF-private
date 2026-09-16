"""Same-ray surface supervision for the frozen-Basis scalar residual."""

import torch

PRESET = 'normalized_ray_surface_bounded_square_v1'


def surface_localization_contract():
    return {
        'preset': PRESET,
        'weight': 0.1,
        'distance_units': 'metres',
        'bounded_square_denominator_m2': 25.0,
        'support': 'all_1024_preselected_gt_return_rays',
        'ray_weight_normalization': 'sum_of_candidate_weights',
        'target_gradient': False,
        'additional_sampling': False,
    }


def surface_localization_loss(weights, sample_depth_m, target_depth_m):
    """Penalize distributed response, including cancellation in expected depth.

    The bounded metric and its 25 m² scale match the existing same-ray risk.
    Normalizing each ray prevents changing opacity to evade supervision.
    """
    w = weights.float()
    z = sample_depth_m.detach().float()
    target = target_depth_m.detach().float()
    if w.ndim != 2 or z.shape != w.shape or target.shape != w.shape[:-1]:
        raise ValueError('surface localization requires [rays,samples] and [rays]')
    if not all(torch.isfinite(x).all() for x in (w, z, target)):
        raise ValueError('surface localization inputs must be finite')
    mass = w.sum(-1, keepdim=True)
    if (w < 0).any() or (mass <= 0).any():
        raise ValueError('surface localization requires positive ray mass')
    distance_square = (z - target[:, None]).square()
    cost = distance_square / (25.0 + distance_square)
    return ((w / mass) * cost).sum(-1).mean()
