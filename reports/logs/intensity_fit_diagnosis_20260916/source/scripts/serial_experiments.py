#!/usr/bin/env python3
"""Detached, fail-stop queue: Best fine-tuning followed by fresh STGC training."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(ZoneInfo("Australia/Perth")).isoformat(timespec="seconds")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare(queue_dir, *, warm_lr=.001, scratch_lr=.01):
    if queue_dir.exists():
        raise FileExistsError(f"queue directory already exists: {queue_dir}")
    for relative in ("checkpoints/Best.pth", "flow/checkpoints/FTD_o.pth", ".venv/bin/python"):
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(ROOT / relative)
    data = ROOT / "data/kitti360"
    frame_ids = {}
    for split in ("train", "val", "test"):
        records = json.loads((data / f"transforms_4950_{split}.json").read_text())["frames"]
        frame_ids[split] = [row["frame_id"] for row in records]
        for row in records:
            if not (data / row["lidar_file_path"]).is_file():
                raise FileNotFoundError(data / row["lidar_file_path"])
    evaluation_frames = [4960, 4970, 4980, 4990]
    expected_frames = {
        "train": [frame for frame in range(4950, 5001) if frame not in evaluation_frames],
        "val": evaluation_frames,
        "test": evaluation_frames,
    }
    if any(sorted(frame_ids[split]) != expected_frames[split] for split in expected_frames):
        raise ValueError("expected STGC official 4950 split: 47 train frames; val/test = 4960, 4970, 4980, 4990")
    queue_dir.mkdir(parents=True)
    source = queue_dir / "source"
    source.mkdir()
    for directory in ("model", "best_core", "flow", "data", "utils", "configs", "scripts"):
        shutil.copytree(ROOT / directory, source / directory, symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__", "tmp"))
    for filename in ("main_ours.py", "requirements.txt"):
        shutil.copy2(ROOT / filename, source / filename)
    for directory in (".venv", ".deps", "checkpoints"):
        (source / directory).symlink_to((ROOT / directory).resolve(), target_is_directory=True)
    # These shared inputs are read-only to the training entry point. The code
    # itself is a snapshot, so future live edits cannot change arm 2 midway.
    python = str(source / ".venv/bin/python")
    arms = []
    for name, lr, initialization in (
        ("01_warm_best", warm_lr, "Best.pth"),
        ("02_from_scratch", scratch_lr, "fresh_parameters"),
    ):
        workspace = queue_dir / name
        command = [python, "-u", str(source / "main_ours.py"),
                   "--config", str(source / "configs/kitti360_4950_stgc_best.txt"),
                   "--workspace", str(workspace), "--seed", "0", "--ckpt", "scratch",
                   "--lr", str(lr), "--iters", "30000", "--refine_steps", "1000",
                   "--refine_lr", "0.001", "--split_protocol", "legacy", "--final_eval_split", "val"]
        if name == "01_warm_best":
            command += ["--init_best", str(source / "checkpoints/Best.pth"),
                        "--refine_init_seed", "-1"]
        else:
            command += ["--refine_init_seed", "0"]
        arms.append({"name": name, "initialization": initialization, "learning_rate": lr,
                     "field_steps": 30000, "refine_steps": 1000, "refine_lr": .001,
                     "refiner_initialization": "keep_Best_weights" if name == "01_warm_best" else "fresh_seed_0",
                     "workspace": str(workspace), "output_log": str(queue_dir / f"{name}.log"),
                     "command": command})
    plan = {"name": queue_dir.name, "created_at": now(), "source_snapshot": str(source),
            "queue_directory": str(queue_dir), "gpu": "0", "seed": 0,
            "frames": frame_ids, "split_protocol": "legacy", "sampling": {"rays": 1024, "samples_per_ray": 768},
            "final_eval_split": "val", "failure_policy": "stop before the next arm",
            "lock_file": str(ROOT / "log/.stgc_serial_queue.lock"), "arms": arms}
    write_json(queue_dir / "plan.json", plan)
    write_json(queue_dir / "status.json", {"state": "prepared", "updated_at": now(),
                                           "arms": [{"name": arm["name"], "state": "pending"} for arm in arms]})
    return plan


def validate_arm_outputs(arm):
    import torch

    workspace = Path(arm["workspace"])
    checkpoints = sorted((workspace / "checkpoints").glob("*_refine.pth"))
    if not checkpoints:
        raise RuntimeError("training exited without a refined checkpoint")
    checkpoint = checkpoints[-1]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("global_step") != arm["field_steps"]:
        raise RuntimeError("field checkpoint did not reach the registered update budget")
    contract = state.get("refine_contract", {})
    expected = {"steps": arm["refine_steps"], "learning_rate": arm["refine_lr"],
                "loss_preset": "bce_expected_masked_depth_support_v1"}
    if any(contract.get(key) != value for key, value in expected.items()):
        raise RuntimeError("refined checkpoint does not match the registered refinement settings")
    if not all(torch.isfinite(value).all() for value in state["model"].values() if value.is_floating_point()):
        raise RuntimeError("refined checkpoint contains non-finite weights")
    outputs = list((workspace / "results").glob("*_depth_lidar.ply"))
    if len(outputs) != 4:
        raise RuntimeError(f"expected four final development-frame exports, found {len(outputs)}")
    return {"checkpoint": str(checkpoint), "global_step": state["global_step"],
            "refinement_updates": contract["steps"], "development_exports": len(outputs)}


def run_queue(plan, validator=validate_arm_outputs):
    queue_dir = Path(plan["queue_directory"])
    status_path = queue_dir / "status.json"
    status = json.loads(status_path.read_text())
    if status["state"] != "prepared":
        raise ValueError(f"queue already started: {status['state']}")
    child = None

    def save():
        status["updated_at"] = now()
        write_json(status_path, status)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"queue received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with open(plan["lock_file"], "a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            status.update(state="failed", error="another STGC queue holds the GPU queue lock", finished_at=now())
            save()
            raise
        status.update(state="running", supervisor_pid=os.getpid(), started_at=now())
        save()
        try:
            for arm, record in zip(plan["arms"], status["arms"]):
                workspace = Path(arm["workspace"])
                if workspace.exists() and any(workspace.iterdir()):
                    resume = arm.get("resume_checkpoint")
                    command = arm["command"]
                    if (not resume or not Path(resume).is_file() or "--ckpt" not in command
                            or command[command.index("--ckpt") + 1] != resume):
                        raise FileExistsError(f"refusing to overwrite existing run without an explicit resume checkpoint: {workspace}")
                record.update(state="running", started_at=now())
                status["active_arm"] = arm["name"]
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=plan["gpu"], PYTHONUNBUFFERED="1")
                with open(arm["output_log"], "a", buffering=1) as output:
                    child = subprocess.Popen(arm["command"], cwd=plan["source_snapshot"],
                                             env=environment, stdin=subprocess.DEVNULL,
                                             stdout=output, stderr=subprocess.STDOUT)
                    record["pid"] = child.pid
                    save()
                    print(f"{now()} START {arm['name']} pid={child.pid}", flush=True)
                    code = child.wait()
                record.update(exit_code=code, finished_at=now())
                child = None
                if code:
                    record["state"] = "failed"
                    raise RuntimeError(f"{arm['name']} exited with code {code}; see {arm['output_log']}")
                record.update(validator(arm), state="completed")
                save()
                print(f"{now()} COMPLETE {arm['name']}", flush=True)
            status.update(state="completed", active_arm=None, finished_at=now())
            save()
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            for record in status["arms"]:
                if record["state"] == "running":
                    record.update(state="failed", finished_at=now())
            status.update(state="failed", error=str(error), finished_at=now())
            save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "launch", "run", "status"))
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--warm-lr", type=float, default=.001)
    parser.add_argument("--scratch-lr", type=float, default=.01)
    args = parser.parse_args()
    queue_dir = args.queue.resolve()
    if args.mode == "prepare":
        print(json.dumps(prepare(queue_dir, warm_lr=args.warm_lr, scratch_lr=args.scratch_lr), indent=2))
        return
    if args.mode == "status":
        print((queue_dir / "status.json").read_text())
        return
    plan = json.loads((queue_dir / "plan.json").read_text())
    if args.mode == "run":
        run_queue(plan)
    else:
        if json.loads((queue_dir / "status.json").read_text())["state"] != "prepared":
            raise ValueError("queue has already been launched")
        # Claim launch atomically, including the short interval before the
        # detached supervisor writes its running status.
        claim = os.open(queue_dir / "launch.pid", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with open(queue_dir / "queue.log", "a", buffering=1) as output:
            command = [str(Path(plan["source_snapshot"]) / ".venv/bin/python"), "-u",
                       str(Path(plan["source_snapshot"]) / "scripts/serial_experiments.py"),
                       "run", "--queue", str(queue_dir)]
            process = subprocess.Popen(command, start_new_session=True, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT)
        with os.fdopen(claim, "w") as pid_file:
            pid_file.write(str(process.pid) + "\n")
        print(json.dumps({"supervisor_pid": process.pid, "queue_directory": str(queue_dir)}))


if __name__ == "__main__":
    main()
