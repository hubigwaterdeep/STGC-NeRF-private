#!/usr/bin/env python3
"""Prepare and explicitly run official47 frozen-field intensity controls.

Preparation never starts training. All jobs inherit the same completed field,
data, original refiner, sampling settings, seed and appearance parameter budget.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / ".deps"), str(ROOT)]
TRAIN_IDS = [i for i in range(4950, 5001) if i not in (4960, 4970, 4980, 4990)]
EVAL_IDS = [4960, 4970, 4980, 4990]
ARMS = (("H0_geo", "geo", .5), ("H1_field", "field", .5),
        ("H1a_pre_fusion", "field_pre_fusion", .5),
        ("H1_current", "field", 1.), ("H2_components", "components", .5))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_baseline(baseline):
    """Audit inherited artifacts, without file digests or new preprocessing."""
    import numpy as np
    baseline = Path(baseline).resolve()
    args_path = baseline / "resolved_args.json"
    opt = json.loads(args_path.read_text())
    if (opt.get("field_backend"), opt.get("dataloader"), str(opt.get("sequence_id")),
            opt.get("split_protocol")) != ("best", "kitti360", "4950", "legacy"):
        raise ValueError("intensity controls require a completed Best official47 / legacy 4950 baseline")
    if opt.get("intensity_feature_mode", "none") != "none":
        raise ValueError("baseline must use the original intensity head")
    if not np.isfinite(opt["scale"]) or opt["scale"] <= 0:
        raise ValueError("invalid source normalization scale")
    if len(opt["offset"]) != 3 or not np.isfinite(opt["offset"]).all():
        raise ValueError("invalid source normalization offset")
    checkpoints = sorted((baseline / "scratch/checkpoints").glob("*_refine.pth"))
    if not checkpoints:
        raise FileNotFoundError("baseline has no completed refined checkpoint")
    data_root = Path(opt["path"]).resolve()
    report = {"protocol": "STGC official47 / legacy", "baseline": str(baseline),
              "source_args": str(args_path), "checkpoint": str(checkpoints[-1]),
              "normalization": {k: opt[k] for k in ("scale", "offset", "fov_lidar")},
              "splits": {}, "file_digest_policy": "disabled by user",
              "preprocessing_scope": "inherit baseline range views, poses and normalization; no new raw-point preprocessing"}
    frame_data = {}
    for split, expected in (("train", TRAIN_IDS), ("val", EVAL_IDS), ("test", EVAL_IDS)):
        manifest = data_root / f"transforms_4950_{split}.json"
        data = json.loads(manifest.read_text())
        rows = sorted(data["frames"], key=lambda row: row["lidar_file_path"])
        ids = [int(row["frame_id"]) for row in rows]
        if ids != expected:
            raise ValueError(f"incorrect official frame IDs in {manifest}")
        shape = (data["h_lidar"], data["w_lidar"], 3)
        details = []
        for row in rows:
            fid = int(row["frame_id"])
            pose = np.asarray(row["lidar2world"], dtype=np.float64)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f"invalid pose for frame {fid}")
            path = (data_root / row["lidar_file_path"]).resolve()
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"invalid range view for frame {fid}")
            if (array[..., 1] < 0).any() or (array[..., 1] > 1).any() or (array[..., 2] < 0).any():
                raise ValueError(f"invalid intensity/depth range for frame {fid}")
            normalized = pose.copy()
            normalized[:3, 3] = (normalized[:3, 3] - opt["offset"]) * opt["scale"]
            detail = {"frame_id": fid, "range_view": str(path), "shape": list(shape),
                      "time": (fid - 4950) / 50, "lidar2world": pose.tolist(),
                      "normalized_pose": normalized.tolist()}
            if fid in frame_data and frame_data[fid] != detail:
                raise ValueError(f"val/test data or pose mismatch for frame {fid}")
            frame_data[fid] = detail
            details.append(detail)
        report["splits"][split] = {"manifest": str(manifest), "count": len(ids),
                                    "frame_ids": ids, "frames": details}
    return opt, report


def prepare(baseline, output, *, steps=3000, rays=128, parameter_budget=32768, max_ray_batch=512):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if steps < 1 or rays < 16 or rays % 16 or parameter_budget < 1 or max_ray_batch < 1:
        raise ValueError("steps/budget must be positive and rays must be a positive multiple of 16")
    opt, alignment = validate_baseline(baseline)
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    # Snapshot code only, with no datasets, checkpoints, generated files or
    # ignored dependency trees. A running job never reads the live launcher.
    for name in ("best_core", "model", "utils", "data", "flow", "scripts", "configs"):
        for path in (ROOT / name).rglob("*"):
            if path.is_file() and path.suffix in (".py", ".cpp", ".cu", ".h", ".txt", ".json"):
                if any(part in ("__pycache__", "tmp", "checkpoints") for part in path.relative_to(ROOT).parts):
                    continue
                target = source / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    for name in ("main_ours.py", "AGENTS.md", "requirements.txt"):
        shutil.copy2(ROOT / name, source / name)
    for name in (".venv", ".deps"):
        if (ROOT / name).exists():
            (source / name).symlink_to((ROOT / name).resolve(), target_is_directory=True)
    write_json(output / "data_alignment.json", alignment)
    write_json(output / "source_args.json", opt)
    jobs = []
    for name, mode, current_weight in ARMS:
        spec = {"name": name, "baseline": str(Path(baseline).resolve()),
                "source_args": str(output / "source_args.json"),
                "checkpoint": alignment["checkpoint"], "workspace": str(output / "runs" / name),
                "mode": mode, "current_weight": current_weight, "seed": 0,
                "steps": steps, "num_rays": rays, "num_steps": opt["num_steps"],
                "max_ray_batch": max_ray_batch, "parameter_budget": parameter_budget,
                "learning_rate": .001, "ema_decay": .95}
        path = output / "jobs" / f"{name}.json"
        write_json(path, spec)
        jobs.append(str(path))
    python = str(source / ".venv/bin/python") if (source / ".venv/bin/python").exists() else sys.executable
    plan = {"source": str(source), "python": python, "jobs": jobs,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "frozen-field intensity fine-tuning; not from-scratch scene training",
            "serial": True, "policy": "no automatic retries; preserve original geometry/refiner; no file digests"}
    write_json(output / "plan.json", plan)
    (output / "EXPERIMENT_POLICY.md").write_text(
        "# Intensity controls\n\nOfficial47 / legacy. Train: " + str(TRAIN_IDS) +
        "\n\nVal/test: " + str(EVAL_IDS) +
        "\n\nFull manifest paths, range-view shapes, poses, times and inherited normalization: data_alignment.json."
        "\n\nFrozen-field fine-tuning, not from-scratch training. No SHA or replacement file digests."
        "\n\nThe original intensity channel remains the refiner input. Parameter counts are approximately matched,"
        " with exact counts recorded per job.\n")
    return plan


def construct_model(opt, state, spec, *, candidate_checkpoint=False):
    import torch
    from model.stgc_best import STGC_NeRF_Best
    keys = ("min_resolution", "base_resolution", "max_resolution", "time_resolution",
            "n_levels_plane", "n_features_per_level_plane", "n_levels_hash", "n_features_per_level_hash",
            "log2_hashmap_size", "num_layers_flow", "hidden_dim_flow", "num_layers_sigma",
            "hidden_dim_sigma", "geo_feat_dim", "num_layers_lidar", "hidden_dim_lidar",
            "out_lidar_dim", "num_frames", "bound", "density_scale", "active_sensor")
    kwargs = {k: opt[k] for k in keys}
    kwargs.update(near_lidar=opt["near_lidar"] * opt["scale"], far_lidar=opt["far_lidar"] * opt["scale"])
    model = STGC_NeRF_Best(**kwargs, intensity_readout_mode=spec["mode"],
                          intensity_current_weight=spec["current_weight"], intensity_readout_seed=spec["seed"],
                          intensity_parameter_budget=spec["parameter_budget"]).cuda().eval()
    incoming = state["model"]
    expected = model.state_dict()
    if candidate_checkpoint:
        if state.get("intensity_contract") != model.intensity_readout.contract():
            raise ValueError("checkpoint does not match the requested intensity configuration")
        model.load_state_dict(incoming, strict=True)
    else:
        if not state.get("refine_contract") or state.get("ema") is not None:
            raise ValueError("baseline must be a completed refined endpoint with materialized field weights")
        original_keys = {k for k in expected if not k.startswith("intensity_readout.")}
        if set(incoming) != original_keys or any(incoming[k].shape != expected[k].shape for k in original_keys):
            raise ValueError("source checkpoint keys/shapes do not match the original Best field")
        expected.update(incoming)
        model.load_state_dict(expected, strict=True)
    return model


def evaluate_model(model, opt, spec, workspace):
    import torch
    from scripts.improvement_worker import dataset, evaluate
    from utils.refiner_input import refiner_input
    ds = dataset(opt, "val")
    predictions, references, targets, probabilities = [], [], [], []
    model.eval()
    for index in range(len(EVAL_IDS)):
        data = ds.collate([index])
        h, w = data["H_lidar"], data["W_lidar"]
        with torch.no_grad(), torch.autocast("cuda", enabled=opt["fp16"]):
            output = model.render(data["rays_o_lidar"], data["rays_d_lidar"], data["time"],
                                  staged=True, perturb=False, num_steps=spec["num_steps"],
                                  max_ray_batch=spec["max_ray_batch"])
            original_input = refiner_input(output, h, w)
            probability = model.unet(original_input).float().cpu()
            attributes = output["image_lidar"].reshape(1, h, w, 2).permute(0, 3, 1, 2)
            candidate_input = torch.cat([attributes, original_input[:, 2:]], 1)
        references.append(original_input.float().cpu())
        predictions.append(candidate_input.float().cpu())
        targets.append(data["images_lidar"].permute(0, 3, 1, 2).float().cpu())
        probabilities.append(probability)
    predictions, references, targets, probabilities = map(torch.cat, (predictions, references, targets, probabilities))
    if not torch.equal(predictions[:, [0, 2]], references[:, [0, 2]]):
        raise RuntimeError("candidate changed raw raydrop or depth")
    reference_metrics = evaluate(references, targets, probabilities, EVAL_IDS, opt,
                                 workspace / "reference", "original field at identical rendering settings")
    metrics = evaluate(predictions, targets, probabilities, EVAL_IDS, opt,
                       workspace / "evaluation", "independent intensity; original field/refiner and mask preserved")
    # Useful fitting diagnostic on GT-positive pixels, without claiming it is
    # the official masked full-frame metric.
    mask = targets[:, 0] > .5
    metrics["gt_valid_intensity_rmse"] = float((predictions[:, 1][mask] - targets[:, 1][mask]).square().mean().sqrt())
    return metrics, reference_metrics


@contextmanager
def gpu_slot(spec):
    lock_path = Path(spec["baseline"]).parent / ".intensity_ablation_gpu.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_gpu_available()
        yield


def run_job(job_path):
    spec = json.loads(Path(job_path).read_text())
    with gpu_slot(spec):
        return _run_job(job_path)


def ensure_gpu_available():
    # Check before importing the CUDA field modules. Some installations create
    # a context during import; that must not preempt another running campaign.
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if any(pid.strip() != str(os.getpid()) for pid in occupied.splitlines()):
        raise RuntimeError("GPU has an existing compute job; run intensity controls after it finishes")


def _run_job(job_path):
    import numpy as np
    import torch
    from torch_ema import ExponentialMovingAverage
    from scripts.improvement_worker import dataset
    spec = json.loads(Path(job_path).read_text())
    workspace = Path(spec["workspace"])
    if workspace.exists():
        raise FileExistsError(f"job output already exists; no implicit restart: {workspace}")
    opt = json.loads(Path(spec["source_args"]).read_text())
    current_opt, alignment = validate_baseline(spec["baseline"])
    if current_opt != opt or alignment["checkpoint"] != spec["checkpoint"]:
        raise ValueError("baseline inputs changed since preparation")
    workspace.mkdir(parents=True)
    write_json(workspace / "spec.json", spec)
    write_json(workspace / "data_alignment.json", alignment)
    write_json(workspace / "status.json", {"state": "running"})
    started = time.monotonic()
    try:
        torch.manual_seed(spec["seed"])
        state = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
        model = construct_model(opt, state, spec)
        frozen = {k: v.clone() for k, v in state["model"].items()}
        source_step, source_refine = state.get("global_step"), state["refine_contract"]
        del state
        mutable = list(model.intensity_readout.parameters())
        optimizer = torch.optim.Adam(mutable, lr=spec["learning_rate"], eps=1e-15)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: .1 ** min(s / spec["steps"], 1))
        scaler = torch.cuda.amp.GradScaler(enabled=opt["fp16"], init_scale=1)
        ema = ExponentialMovingAverage(mutable, decay=spec["ema_decay"])
        ds = dataset(opt, "train")
        ds.num_rays_lidar = spec["num_rays"]
        torch.manual_seed(spec["seed"])
        rng = np.random.default_rng(spec["seed"])
        write_json(workspace / "readout_contract.json", model.intensity_readout.contract())
        for step in range(spec["steps"]):
            if step % 47 == 0:
                order = rng.permutation(47)
            ds.patch_size_lidar = [2, 8] if (step // 47 + 1) % 2 == 0 else 1
            data = ds.collate([int(order[step % 47])])
            model.intensity_readout.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=opt["fp16"]):
                output = model.render(data["rays_o_lidar"], data["rays_d_lidar"], data["time"],
                                      staged=False, num_steps=spec["num_steps"], perturb=True)
                gt = data["images_lidar"].float()
                loss = opt["alpha_i"] * ((output["image_lidar"][..., 1].float() - gt[..., 1]) * gt[..., 0]).square().sum()
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite intensity loss at step {step + 1}")
            if not loss.requires_grad:
                # Entirely empty density support has no appearance signal.
                # Preserve the sampling/update schedule with a zero gradient.
                loss = loss + sum(p.sum() * 0 for p in mutable)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in mutable):
                raise RuntimeError(f"nonfinite intensity gradients at step {step + 1}")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update()
            if step % 50 == 0 or step + 1 == spec["steps"]:
                progress = {"step": step + 1, "loss": float(loss.detach()), "seconds": time.monotonic() - started}
                write_json(workspace / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        ema.copy_to()
        for key, value in frozen.items():
            if not torch.equal(value, model.state_dict()[key].detach().cpu()):
                raise RuntimeError(f"intensity control changed frozen field/refiner tensor: {key}")
        del frozen, ds, optimizer, scheduler, ema
        checkpoint = workspace / "intensity.pth"
        torch.save({"stage": "intensity_only", "model": model.state_dict(), "source_checkpoint": spec["checkpoint"],
                    "global_step": source_step, "attribute_steps": spec["steps"], "refine_contract": source_refine,
                    "intensity_contract": model.intensity_readout.contract(), "spec": spec,
                    "source_args": opt, "data_alignment": alignment}, checkpoint)
        metrics, reference = evaluate_model(model, opt, spec, workspace)
        result = {"state": "completed", "checkpoint": str(checkpoint), "metrics": metrics,
                  "reference_metrics": reference, "seconds": time.monotonic() - started,
                  "frozen_field_and_refiner_unchanged": True, "intensity_contract": model.intensity_readout.contract()}
        write_json(workspace / "result.json", result)
        write_json(workspace / "status.json", {"state": "completed"})
    except BaseException as error:
        write_json(workspace / "status.json", {"state": "failed", "error": str(error)})
        raise


def run_suite(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    for job in plan["jobs"]:
        subprocess.run([plan["python"], "-u", str(Path(plan["source"]) / "scripts/intensity_ablation.py"),
                        "run", "--job", job], check=True, cwd=plan["source"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--baseline-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--rays", type=int, default=128)
    p.add_argument("--parameter-budget", type=int, default=32768)
    p.add_argument("--max-ray-batch", type=int, default=512)
    p = commands.add_parser("run")
    p.add_argument("--job", type=Path, required=True)
    p = commands.add_parser("run-suite")
    p.add_argument("--plan", type=Path, required=True)
    p = commands.add_parser("evaluate")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.baseline_run, args.output, steps=args.steps, rays=args.rays,
                         parameter_budget=args.parameter_budget, max_ray_batch=args.max_ray_batch)
        print(json.dumps(result, indent=2))
    elif args.command == "run":
        run_job(args.job)
    elif args.command == "run-suite":
        run_suite(args.plan)
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        ensure_gpu_available()
        import torch
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        with gpu_slot(state["spec"]):
            model = construct_model(state["source_args"], state, state["spec"], candidate_checkpoint=True)
            result, reference = evaluate_model(model, state["source_args"], state["spec"], args.output)
            write_json(args.output / "result.json", {"metrics": result, "reference_metrics": reference})


if __name__ == "__main__":
    main()
