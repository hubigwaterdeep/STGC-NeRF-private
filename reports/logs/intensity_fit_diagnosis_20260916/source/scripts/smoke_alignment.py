#!/usr/bin/env python3
"""One disposable real-training-frame update through all STGC losses."""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".deps"))

import torch
from torch.utils.data import DataLoader
from main_ours import build_model, get_arg_parser
from model.runner import Trainer
from flow.gmsf import GMSF
from data.kitti360_dataset import KITTI360Dataset


def main():
    started = time.monotonic()
    torch.manual_seed(0)
    opt = get_arg_parser().parse_args([
        "--config", str(ROOT / "configs/kitti360_4950_stgc_best.txt"),
        "--num_rays_lidar", "128", "--num_steps", "96", "--max_train_steps", "1",
        "--skip_refine", "--skip_final_eval",
    ])
    model = build_model(opt).cuda().train()
    teacher = GMSF(backbone=opt.backbone, feature_channels=opt.feature_channels,
                   ffn_dim_expansion=opt.ffn_dim_expansion,
                   num_transformer_pt_layers=opt.num_transformer_pt_layers,
                   num_transformer_layers=opt.num_transformer_layers)
    teacher.load_state_dict(torch.load(ROOT / opt.resume, map_location="cpu", weights_only=False)["model"], strict=True)
    criterion = {"depth": torch.nn.L1Loss(reduction="none"),
                 "raydrop": torch.nn.MSELoss(reduction="none"),
                 "intensity": torch.nn.MSELoss(reduction="none"),
                 "grad": torch.nn.L1Loss(reduction="none")}
    trainer = Trainer("stgc_best_smoke", opt, model, teacher, criterion=criterion,
                       optimizer=lambda m: torch.optim.Adam(m.get_params(opt.lr), betas=(.9, .99), eps=1e-15),
                       device=torch.device("cuda"), use_checkpoint="scratch", fp16=opt.fp16,
                       use_tensorboardX=False, workspace=str(ROOT / "log/smoke_alignment"))
    ds = KITTI360Dataset(device="cuda", split="refine", root_path=str(ROOT / "data/kitti360"),
                         sequence_id=opt.sequence_id, split_protocol=opt.split_protocol,
                         preload=False, scale=opt.scale, offset=opt.offset, fp16=opt.fp16,
                         fov_lidar=opt.fov_lidar)
    # Five actual training frames activate both forward/backward STGC neighbors
    # at offsets 1 and 2. No validation/test point cloud enters supervision.
    indices = [ds.frame_ids.tolist().index(frame) for frame in range(4950, 4955)]
    loader = DataLoader(indices, batch_size=1, collate_fn=ds.collate)
    loader._data = ds
    trainer.process_pointcloud(loader)
    ds.training = True
    ds.num_rays_lidar = opt.num_rays_lidar
    batch = ds.collate([indices[2]])
    before = model.scene_field.geometry_residual_basis.output_projection.weight.detach().clone()
    with torch.autocast("cuda", enabled=opt.fp16):
        loss = trainer.train_step(batch)[-1]
    if not torch.isfinite(loss):
        raise RuntimeError("STGC smoke produced a non-finite loss")
    trainer.scaler.scale(loss).backward()
    trainer.scaler.unscale_(trainer.optimizer)
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and parameter.grad.numel():
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"non-finite gradient: {name}")
            gradients[name] = float(parameter.grad.abs().max())
    for prefix in ("scene_field.basis.", "scene_field.hash_basis.", "scene_field.flow_net.",
                   "scene_field.geometry_residual_basis.", "sigma_net.", "raydrop_net.", "intensity_net."):
        if not any(value > 0 for name, value in gradients.items() if name.startswith(prefix)):
            raise RuntimeError(f"missing nonzero STGC gradient: {prefix}")
    if any(p.grad is not None for p in teacher.parameters()):
        raise RuntimeError("teacher received gradients")
    trainer.scaler.step(trainer.optimizer)
    trainer.scaler.update()
    if torch.equal(before, model.scene_field.geometry_residual_basis.output_projection.weight):
        raise RuntimeError("geometry residual did not update")
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError("optimizer produced non-finite model parameters")
    report = {"state": "pass", "initialization": "fresh_parameters", "scene_checkpoint_loaded": False,
              "split": "train", "frame": 4952,
              "neighbor_frames": list(range(4950, 4955)), "optimizer_updates": 1,
              "rays": opt.num_rays_lidar, "samples_per_ray": opt.num_steps,
              "loss": float(loss.detach()), "gradient_tensor_count": len(gradients),
              "teacher_frozen": True, "checkpoint_written": False,
              "elapsed_seconds": round(time.monotonic() - started, 2)}
    output = ROOT / "docs/smoke_from_scratch_result.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
