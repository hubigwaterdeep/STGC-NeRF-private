"""Keep the original appearance channel available to the frozen refiner."""

import torch


def refiner_input(output, height, width, *, probability_bce=False):
    attributes = output.get("image_lidar_reference", output["image_lidar"])
    attributes = attributes.reshape(-1, height, width, 2).permute(0, 3, 1, 2)
    if probability_bce:
        attributes = torch.cat([attributes[:, :1].sigmoid(), attributes[:, 1:]], dim=1)
    depth = output["depth_lidar"].reshape(-1, 1, height, width)
    return torch.cat([attributes, depth], dim=1)
