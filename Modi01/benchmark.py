"""Bounded synthetic rendering profile. Not a trained-scene quality evaluation."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import statistics
import time
import torch
from Modi01.field import RepresentationConfig
from Modi01.model import STGCNeRFModi01


def profile(config, rays, samples, steps, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = STGCNeRFModi01(config, near_lidar=.01, far_lidar=.5).cuda()
    model.set_training_progress(spline=1., high_order=1.)
    optimizer = torch.optim.Adam(model.get_params(.001), betas=(.9, .99), eps=1e-15)
    # Dedicated RNG gives exactly the same rays to all candidates.
    generator = torch.Generator(device='cuda').manual_seed(seed + 1)
    origins = torch.rand(1, rays, 3, generator=generator, device='cuda') * .2 - .1
    directions = torch.randn(1, rays, 3, generator=generator, device='cuda')
    directions /= directions.norm(dim=-1, keepdim=True)
    t = torch.tensor([[.37]], device='cuda')
    target_depth = torch.full((1, rays), .2, device='cuda')
    target_image = torch.full((1, rays, 2), .5, device='cuda')
    def update():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            output = model.render(origins, directions, t, num_steps=samples, perturb=False)
            loss = (output['depth_lidar'].float()-target_depth).square().mean()
            loss += (output['image_lidar'].float()-target_image).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('synthetic renderer produced non-finite loss')
        loss.backward()
        optimizer.step()
        return float(loss.detach())
    # Touch every temporal table before measuring Adam-state memory.
    for knot in range(8):
        t.fill_(knot / 7)
        update()
    t.fill_(.37)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    losses = []
    for _ in range(steps):
        start = time.perf_counter()
        losses.append(update())
        torch.cuda.synchronize()
        timings.append(time.perf_counter()-start)
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    model.eval()
    inference = []
    with torch.no_grad():
        for _ in range(steps):
            start = time.perf_counter()
            model.render(origins, directions, t, num_steps=samples, perturb=False)
            torch.cuda.synchronize()
            inference.append(time.perf_counter()-start)
    result = model.scene_field.representation_report()
    result.update(model_parameters=sum(p.numel() for p in model.parameters()),
        model_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
        step_ms_median=statistics.median(timings)*1000,
        training_rays_per_second=rays/statistics.median(timings),
        inference_rays_per_second=rays/statistics.median(inference),
        step_seconds=timings, synthetic_losses=losses)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rays', type=int, default=128)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20260918)
    parser.add_argument('--output', type=Path, default=Path('Modi01/reports/benchmark.json'))
    args = parser.parse_args()
    if min(args.rays, args.samples, args.steps) < 1:
        parser.error('rays, samples and steps must be positive')
    configurations = {'all_modal': RepresentationConfig(8), 'hybrid44': RepresentationConfig(),
        'hybrid44_untied': RepresentationConfig(coefficients='untied'),
        'hybrid44_neural': RepresentationConfig(coefficients='neural')}
    report = dict(scope='synthetic raw pre-refiner forward/backward + Adam; not scene training or quality evidence',
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, seed=args.seed,
        rays=args.rays, samples_per_ray=args.samples, warmup_updates=8,
        timed_updates=args.steps, includes_flow_teacher=False, includes_task_regularizers=False,
        source='reports/logs/intensity_fit_diagnosis_20260916/source', results={})
    for name, config in configurations.items():
        report['results'][name] = profile(config, args.rays, args.samples, args.steps, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(name, report['results'][name]['step_ms_median'], flush=True)


if __name__ == '__main__':
    main()
