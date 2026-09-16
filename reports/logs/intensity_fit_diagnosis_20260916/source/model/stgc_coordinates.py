"""Coordinate conversions between the Best world field and STGC teacher."""

import torch


def field_flow_in_sensor(model, world_points, time, pose):
    """Query world-normalized positions, rotate displacements into the sensor."""
    rotation = pose[0, :3, :3].to(world_points)
    flow = model.flow(world_points[0], time)
    return {direction: vector @ rotation for direction, vector in flow.items()}


def nearest_in_source_frame(source, target, source_pose, target_pose):
    """Select nearest world points, then express them in the source sensor."""
    if source.ndim != 3 or source.shape[0] != 1 or source.shape[-1] != 3:
        raise ValueError("STGC nearest-point alignment expects [1, N, 3]")
    target = torch.as_tensor(target, device=source.device, dtype=source.dtype).reshape(-1, 3)
    if not len(target):
        raise ValueError("cannot align an empty neighboring scan")
    pose_a = torch.as_tensor(source_pose, device=source.device, dtype=source.dtype).reshape(4, 4)
    pose_b = torch.as_tensor(target_pose, device=source.device, dtype=source.dtype).reshape(4, 4)
    world_a = source[0] @ pose_a[:3, :3].T + pose_a[:3, 3]
    world_b = target @ pose_b[:3, :3].T + pose_b[:3, 3]
    # Only correspondence selection is discrete; retaining full N x scan-size
    # distances wastes memory on dense KITTI panoramas.
    with torch.no_grad():
        indices = torch.cat([torch.cdist(chunk, world_b).argmin(-1)
                             for chunk in world_a.split(128)])
    selected = (world_b[indices] - pose_a[:3, 3]) @ pose_a[:3, :3]
    return selected.unsqueeze(0)
