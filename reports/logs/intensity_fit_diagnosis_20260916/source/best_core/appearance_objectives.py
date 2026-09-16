"""Intensity objectives for the official-frame controlled experiments."""

import torch


def expected_intensity_risk(probability, prediction, target):
    """Expected hard-mask squared error; only probability receives gradients."""
    if probability.shape != prediction.shape or prediction.shape != target.shape:
        raise ValueError("intensity risk tensors must have identical shapes")
    prediction, target = prediction.detach().float(), target.detach().float()
    probability = probability.float()
    return (probability * (prediction - target).square()
            + (1 - probability) * target.square()).mean()


def intensity_gradient_loss(prediction, target, mask, depth_m, patch_size,
                            max_depth_jump_m=0.5):
    """Signed GT-gradient matching on valid same-surface neighbors only.

    Horizontal panorama wrapping is provided by the existing patch sampler;
    no connection is made between opposite vertical borders or patches.
    """
    shape = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
    if len(shape) == 1:
        shape = (shape[0], shape[0])
    height, width = shape
    if height <= 1:
        return prediction.sum() * 0.0
    if prediction.numel() % (height * width):
        raise ValueError("ray count must contain complete patches")
    pred = prediction.float().reshape(-1, height, width)
    gt = target.detach().float().reshape_as(pred)
    valid = mask.detach().reshape_as(pred) > 0.5
    depth = depth_m.detach().float().reshape_as(pred)
    total, count = pred.sum() * 0.0, pred.new_zeros(())
    for dim in (1, 2):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[dim], b[dim] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        support = valid[a] & valid[b] & ((depth[a] - depth[b]).abs() < max_depth_jump_m)
        error = ((pred[a] - pred[b]) - (gt[a] - gt[b])).abs()
        total = total + (error * support).sum()
        count = count + support.sum()
    return total / count.clamp_min(1)
