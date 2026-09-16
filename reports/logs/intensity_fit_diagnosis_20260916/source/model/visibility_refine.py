"""Best's unified return refinement inside the STGC training pipeline."""

from pathlib import Path
import numpy as np
import torch
from best_core.refine_objectives import visibility_refine_loss_from_preset


def refinement_settings(opt):
    return {"protocol": "stgc_best_unified_refine_v1",
            "loss_preset": opt.refine_loss_preset,
            "steps": opt.refine_steps, "learning_rate": opt.refine_lr,
            "depth_scale": opt.scale,
            "seed": opt.refine_init_seed if opt.refine_init_seed >= 0 else opt.seed}


def refine(trainer, loader):
    opt, model = trainer.opt, trainer.model
    if hasattr(model, "intensity_readout"):
        raise ValueError("independent intensity controls preserve the original refiner; do not reinitialize it")
    if trainer.ema is not None:
        trainer.ema.copy_to()
        trainer.ema = None
    model.eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("unet."))
    if opt.refine_init_seed >= 0:
        devices = list(range(torch.cuda.device_count()))
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(opt.refine_init_seed)
            for module in model.unet.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

    inputs, targets, target_depths, target_intensities = [], [], [], []
    for index, data in enumerate(loader):
        height, width = data["H_lidar"], data["W_lidar"]
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=opt.fp16):
            output = model.render(data["rays_o_lidar"], data["rays_d_lidar"], data["time"],
                                  staged=True, perturb=False, **vars(opt))
        attributes = output["image_lidar"].reshape(-1, height, width, 2).permute(0, 3, 1, 2)
        depth = output["depth_lidar"].reshape(-1, 1, height, width)
        inputs.append(torch.cat([attributes, depth], dim=1).float())
        targets.append(data["images_lidar"][..., 0].unsqueeze(1).float())
        target_depths.append(data["images_lidar"][..., 2].unsqueeze(1).float())
        target_intensities.append(data["images_lidar"][..., 1].unsqueeze(1).float() * targets[-1])
        if index % 10 == 0:
            trainer.log(f"Refine inputs: {index + 1}/{len(loader)}")
    inputs, targets, target_depths = map(torch.cat, (inputs, targets, target_depths))
    target_intensities = torch.cat(target_intensities)
    frozen = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
              if not name.startswith("unet.")}
    objective = visibility_refine_loss_from_preset(opt.refine_loss_preset)
    model.unet.train()
    optimizer = torch.optim.Adam(model.unet.parameters(), lr=opt.refine_lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.refine_lr,
                                                   total_steps=opt.refine_steps)
    seed = opt.refine_init_seed if opt.refine_init_seed >= 0 else opt.seed
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    for step in range(opt.refine_steps):
        optimizer.zero_grad(set_to_none=True)
        mask = torch.ones_like(inputs)
        height, width = inputs.shape[-2:]
        for _ in range(rng.integers(32)):
            h = int(rng.integers(1, max(2, int(.1 * height))))
            w = int(rng.integers(1, max(2, int(.1 * width))))
            y, x = int(rng.integers(height - h)), int(rng.integers(width - w))
            mask[:, :, y:y + h, x:x + w] = 0
        probability = model.unet(inputs * mask)
        terms = objective(probability, targets, inputs[:, 2:3], target_depths,
                          depth_scale=opt.scale, predicted_intensity=inputs[:, 1:2],
                          target_intensity=target_intensities)
        if not torch.isfinite(terms.total):
            raise RuntimeError(f"non-finite refinement loss at step {step + 1}")
        terms.total.backward()
        optimizer.step()
        scheduler.step()
        if step % 50 == 0:
            trainer.log(f"Refine step {step + 1}: BCE={terms.bce.item():.6f}, "
                        f"depth support={terms.expected_depth_risk.item():.6f}, "
                        f"intensity risk={(terms.total - terms.bce - terms.expected_depth_risk).item():.6f}")
    for name, expected in frozen.items():
        if not torch.equal(expected, model.state_dict()[name].detach().cpu()):
            raise RuntimeError(f"refinement modified frozen field tensor {name}")
    trainer.use_refine = True
    model.eval()
    path = Path(trainer.ckpt_path) / f"{trainer.name}_ep{trainer.epoch:04d}_refine.pth"
    torch.save({"stage": "refined", "model": model.state_dict(), "global_step": trainer.global_step,
                "epoch": trainer.epoch,
                "refine_contract": {**refinement_settings(opt),
                                    "split": loader._data.split}}, path)
    trainer.log(f"Saved refined STGC checkpoint: {path}")
