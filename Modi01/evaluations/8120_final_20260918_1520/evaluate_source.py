"""Read-only final-checkpoint evaluation using each run's archived implementation.

Run from the repository with its activated environment:
  python Modi01/evaluate.py --output Modi01/evaluations/<new-directory>

Each arm runs in a fresh process. Main metrics call the archived Trainer.eval_step
and metric classes directly, with the original AMP, masking and frame averaging.
No Trainer is constructed, optimizer created, or checkpoint written.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
METRIC_NAMES = [
    ['return_rmse', 'return_accuracy', 'return_f1'],
    ['intensity_rmse', 'intensity_medae', 'intensity_lpips', 'intensity_ssim', 'intensity_psnr'],
    ['depth_rmse_m', 'depth_medae_m', 'depth_lpips', 'depth_ssim', 'depth_psnr'],
    ['cd_m2', 'point_fscore'],
]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def summarize_strata(rows):
    """Pool pixels, unlike the official main metrics' unweighted frame average."""
    output = {}
    for name in rows[0]:
        totals = {key: sum(row[name][key] for row in rows) for key in rows[0][name]}
        n = totals['pixels']
        tp, fp, fn, tn = (totals[key] for key in ('tp', 'fp', 'fn', 'tn'))
        output[name] = dict(
            **totals,
            depth_rmse_m=math.sqrt(totals['depth_sse'] / n) if n else None,
            intensity_rmse=math.sqrt(totals['intensity_sse'] / n) if n else None,
            return_precision=tp / (tp + fp) if tp + fp else None,
            return_recall=tp / (tp + fn) if tp + fn else None,
            return_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            return_accuracy=(tp + tn) / n if n else None,
        )
    return output


def pixel_strata(pred_i, pred_d, pred_r, gt_i, gt_d, gt_r, scale):
    """Explicit supplemental GT-range/depth-discontinuity diagnostics, in float64."""
    import numpy as np

    p_i, p_d, p_r, g_i, g_d, g_r = [
        value.detach().cpu().numpy()[0].astype(np.float64)
        for value in (pred_i, pred_d, pred_r, gt_i, gt_d, gt_r)
    ]
    p_d, g_d = p_d / scale, g_d / scale
    pm, gm = p_r > 0.5, g_r > 0.5
    edge = np.zeros(gm.shape, dtype=bool)
    # Horizontal panorama wraps; vertical neighbors do not wrap. Both endpoints
    # must have GT returns, so missing returns are not mistaken for depth edges.
    horizontal = gm & np.roll(gm, 1, axis=1) & (np.abs(g_d - np.roll(g_d, 1, axis=1)) > 1.0)
    edge |= horizontal | np.roll(horizontal, -1, axis=1)
    vertical = gm[1:] & gm[:-1] & (np.abs(g_d[1:] - g_d[:-1]) > 1.0)
    edge[1:] |= vertical
    edge[:-1] |= vertical
    masks = {'all_pixels': np.ones(gm.shape, dtype=bool), 'gt_return': gm,
             'gt_no_return': ~gm, 'depth_edge': edge, 'non_edge_gt_return': gm & ~edge}
    for lo, hi in [(0, 10), (10, 30), (30, 50), (50, 80), (80, float('inf'))]:
        masks[f'range_{lo}_{hi}m'] = gm & (g_d >= lo) & (g_d < hi)
    # Same hard prediction mask and clipping bounds as main metrics; arithmetic
    # deliberately promoted for stable pooling. GT retains loader quantization.
    ds = (np.clip(p_d * pm, 1e-6, 80) - np.clip(g_d, 1e-6, 80)) ** 2
    ins = (np.clip(p_i * pm, 1e-6, 1) - np.clip(g_i, 1e-6, 1)) ** 2
    return {name: dict(
        pixels=int(mask.sum()), depth_sse=float(ds[mask].sum()), intensity_sse=float(ins[mask].sum()),
        tp=int((mask & pm & gm).sum()), fp=int((mask & pm & ~gm).sum()),
        fn=int((mask & ~pm & gm).sum()), tn=int((mask & ~pm & ~gm).sum()),
    ) for name, mask in masks.items()}


def evaluate_worker(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    out = Path(spec['output'])
    workspace = Path(spec['workspace'])
    registration = json.loads((workspace / 'registration.json').read_text())
    # Source selection happens before any model/data/utils import. Never mix
    # archived implementations within one interpreter.
    # Direct script execution adds Modi01/ to sys.path; its model.py would
    # shadow the upstream namespace package named model, even after insertion.
    sys.path[:] = [entry for entry in sys.path if Path(entry or '.').resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(Path(spec['modi_source']).parent))
    import Modi01.runtime as runtime
    runtime.ROOT = ROOT
    runtime.SOURCE = Path(spec['audited_source'])
    runtime.activate_source()
    import numpy as np
    import torch
    from torch_ema import ExponentialMovingAverage
    from Modi01.field import RepresentationConfig
    from Modi01.model import STGCNeRFModi01
    from data.kitti360_dataset import KITTI360Dataset
    from model.runner import Trainer
    from utils.metrics import RaydropMeter, IntensityMeter, DepthMeter, PointsMeter

    torch.set_num_threads(4)
    torch.manual_seed(registration['fixed_seed'])
    np.random.seed(registration['fixed_seed'])
    opt = SimpleNamespace(**registration['configuration'])
    assert opt.split_protocol == 'legacy' and opt.skip_refine and opt.raydrop_ratio == 0.5
    assert opt.sequence_id == '8120' and opt.num_steps == 768
    opt.max_ray_batch = 4096
    names = ('min_resolution', 'base_resolution', 'max_resolution', 'time_resolution',
             'n_levels_plane', 'n_features_per_level_plane', 'n_levels_hash', 'n_features_per_level_hash',
             'log2_hashmap_size', 'num_layers_flow', 'hidden_dim_flow', 'num_layers_sigma',
             'hidden_dim_sigma', 'geo_feat_dim', 'num_layers_lidar', 'hidden_dim_lidar',
             'out_lidar_dim', 'num_frames', 'bound', 'density_scale', 'active_sensor')
    model = STGCNeRFModi01(
        representation=RepresentationConfig(**registration['representation']),
        **{key: getattr(opt, key) for key in names},
        near_lidar=opt.near_lidar * opt.scale, far_lidar=opt.far_lidar * opt.scale,
    ).cuda()
    checkpoint_path = Path(spec['checkpoint'])
    before = (checkpoint_path.stat().st_size, checkpoint_path.stat().st_mtime_ns)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=False)
    assert checkpoint['stage'] == 'field' and checkpoint['global_step'] == 30000
    assert not checkpoint.get('refine_contract')
    model.load_state_dict(checkpoint['model'], strict=True)
    ema = ExponentialMovingAverage(model.parameters(), decay=opt.ema_decay)
    shadows = checkpoint['ema']['shadow_params']
    assert len(shadows) == len(list(model.parameters()))
    assert all(a.shape == b.shape for a, b in zip(shadows, model.parameters()))
    ema.load_state_dict(checkpoint['ema'])
    model.eval().requires_grad_(False)
    losses = {'l1': torch.nn.L1Loss(reduction='none'), 'mse': torch.nn.MSELoss(reduction='none')}
    runner = SimpleNamespace(model=model, opt=opt, use_refine=False, criterion={
        key: losses[getattr(opt, key + '_loss')] for key in ['depth', 'raydrop', 'intensity']})
    metrics = [RaydropMeter(ratio=opt.raydrop_ratio), IntensityMeter(scale=opt.intensity_scale),
               DepthMeter(scale=opt.scale), PointsMeter(scale=opt.scale, intrinsics=opt.fov_lidar)]
    datasets = {}
    for label, split in [('development', 'val'), ('train', 'refine')]:
        ds = KITTI360Dataset(device='cuda', split=split, root_path=opt.path,
            sequence_id=opt.sequence_id, split_protocol=opt.split_protocol, preload=opt.preload,
            scale=opt.scale, offset=opt.offset, fp16=opt.fp16, fov_lidar=opt.fov_lidar)
        expected = registration['assets']['splits']['val' if label == 'development' else 'train']['frame_ids']
        assert ds.frame_ids.tolist() == expected
        assert not ds.training and ds.num_rays_lidar == -1
        np.testing.assert_allclose(ds.times.cpu().numpy().ravel(), (np.asarray(expected) - 8120) / 50, atol=1e-7)
        datasets[label] = ds
    assert datasets['development'].frame_ids.tolist() == [8130, 8140, 8150, 8160]
    assert len(datasets['train']) == 47
    assert registration['assets']['splits']['val']['frame_ids'] == registration['assets']['splits']['test']['frame_ids']
    record = dict(**spec, status='RUNNING', steps=checkpoint['global_step'], epoch=checkpoint['epoch'],
        ema_updates=checkpoint['ema']['num_updates'], representation=registration['representation'],
        frames={key: ds.frame_ids.tolist() for key, ds in datasets.items()},
        evaluation='post-hoc development; val and test share frames; pre-refiner only',
        rendering=dict(fp16=opt.fp16, num_steps=opt.num_steps, max_ray_batch=opt.max_ray_batch,
                       staged=True, perturb=False, return_threshold=0.5),
        main_metric_aggregation='arithmetic mean of per-frame archived metric values',
        point_metrics='CD=sum of two mean squared distances (m^2); F-score squared threshold 0.05 m^2 (radius sqrt(0.05) m)',
        supplemental_strata='pooled float64 pixel errors; GT range bins and >1m adjacent valid-GT depth edges; no moving/static labels',
        parameter_count=sum(p.numel() for p in model.parameters()),
        source_modules={name: str(Path(sys.modules[name].__file__).resolve()) for name in
                        ['Modi01.model', 'Modi01.field', 'model.runner', 'utils.metrics', 'data.kitti360_dataset']},
        groups={})
    write_json(out / 'results.json', record)
    rows = []
    start = time.monotonic()
    # Finish both development weight conventions first; then full training views.
    for weights, split in [('ema', 'development'), ('raw', 'development'), ('ema', 'train')]:
        model.load_state_dict(checkpoint['model'], strict=True)
        if weights == 'ema':
            ema.copy_to()
        model.eval()
        for metric in metrics:
            metric.clear()
        group_rows, strata_rows = [], []
        for index, data in enumerate(datasets[split].dataloader()):
            frame_start = time.monotonic()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=opt.fp16):
                pred_i, pred_d, pred_r, gt_i, gt_d, gt_r, loss = Trainer.eval_step(runner, data)
            values = [pred_i, pred_d, pred_r, gt_i, gt_d, gt_r]
            assert all(torch.isfinite(value).all() for value in values), 'non-finite rendered data'
            mask = torch.where(pred_r > 0.5, 1, 0)
            for metric, pred, gt in zip(metrics, [pred_r, pred_i * mask, pred_d * mask, pred_d * mask],
                                       [gt_r, gt_i, gt_d, gt_d]):
                metric.update(pred.clone(), gt.clone())
            row = dict(variant=spec['variant'], weights=weights, split=split,
                       frame_id=int(data['frame_id'].item()), time=float(data['time'].item()), loss=float(loss.item()))
            for metric, names in zip(metrics, METRIC_NAMES):
                row.update({name: float(value) for name, value in zip(names, metric.V[-1])})
            assert all(math.isfinite(value) for value in row.values() if isinstance(value, float))
            rows.append(row)
            group_rows.append(row)
            strata_rows.append(pixel_strata(*values, scale=opt.scale))
            if split == 'development':
                destination = out / 'predictions' / weights
                destination.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination / f'{row["frame_id"]}.npz',
                    **{name: value.detach().cpu().numpy() for name, value in zip(
                        ['intensity', 'depth_normalized', 'return_probability', 'gt_intensity', 'gt_depth_normalized', 'gt_return'], values)},
                    scale=np.array(opt.scale), frame_id=np.array(row['frame_id']))
            print(json.dumps(dict(variant=spec['variant'], weights=weights, split=split,
                                  frame=index+1, total=len(datasets[split]), frame_id=row['frame_id'],
                                  cd=row['cd_m2'], seconds=round(time.monotonic()-frame_start, 2))), flush=True)
            write_json(out / 'progress.json', dict(variant=spec['variant'], weights=weights, split=split,
                completed=index+1, total=len(datasets[split]), elapsed_seconds=time.monotonic()-start))
        aggregate = {}
        for metric, names in zip(metrics, METRIC_NAMES):
            aggregate.update({name: float(value) for name, value in zip(names, metric.measure())})
        # Read-only verification compares values directly; no file digests.
        expected_parameters = shadows if weights == 'ema' else [checkpoint['model'][name] for name, _ in model.named_parameters()]
        assert all(torch.equal(p.detach().cpu(), expected.to(dtype=p.dtype))
                   for p, expected in zip(model.parameters(), expected_parameters)), 'parameters changed during evaluation'
        record['groups'][f'{weights}_{split}'] = dict(
            frames=len(group_rows), metrics=aggregate, strata=summarize_strata(strata_rows),
            parameters_unchanged=True)
        write_json(out / 'results.json', record)
        with (out / 'per_frame.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    assert before == (checkpoint_path.stat().st_size, checkpoint_path.stat().st_mtime_ns)
    record.update(status='COMPLETED', elapsed_seconds=time.monotonic()-start, checkpoint_stat_unchanged=True)
    write_json(out / 'results.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--queue', type=Path)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        evaluate_worker(args.worker)
        return
    if args.output is None:
        parser.error('--output is required')
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / 'Modi01/evaluations'):
        parser.error('output must be a fresh directory within Modi01/evaluations')
    out.mkdir(parents=True, exist_ok=False)
    queue_path = args.queue or Path(json.loads((ROOT / 'Modi01/current_queue.json').read_text())['manifest'])
    queue = json.loads(queue_path.read_text())
    completion = json.loads((queue_path.parent / 'completion_summary.json').read_text())
    assert completion['status'] == 'COMPLETED'
    jobs = []
    for job, finished in zip(queue['jobs'], completion['jobs']):
        assert job['variant'] == finished['variant'] and finished['steps'] == 30000
        workspace = Path(job['workspace'])
        reg = json.loads((workspace / 'registration.json').read_text())
        source = Path(job.get('source_snapshot', Path(job['launch']) / 'source'))
        jobs.append(dict(variant=job['variant'], workspace=str(workspace), checkpoint=finished['checkpoint'],
                         modi_source=str(source / 'Modi01'), audited_source=reg['source'],
                         output=str(out / job['variant'])))
    state = dict(status='RUNNING', queue=str(queue_path), jobs=jobs, completed=[])
    write_json(out / 'evaluation.json', state)
    (out / 'evaluate_source.py').write_text(Path(__file__).read_text())
    for job in jobs:
        job_out = Path(job['output'])
        job_out.mkdir()
        spec_path = job_out / 'job.json'
        write_json(spec_path, job)
        with (job_out / 'evaluate.log').open('w') as log:
            result = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(spec_path)],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        if result.returncode:
            state.update(status='FAILED', failed_variant=job['variant'], exit_code=result.returncode)
            write_json(out / 'evaluation.json', state)
            raise RuntimeError(f'Evaluation failed: {job["variant"]}; see {job_out / "evaluate.log"}')
        state['completed'].append(job['variant'])
        write_json(out / 'evaluation.json', state)
    state['status'] = 'COMPLETED'
    write_json(out / 'evaluation.json', state)
    print(f'Completed evaluation: {out}', flush=True)


if __name__ == '__main__':
    main()
