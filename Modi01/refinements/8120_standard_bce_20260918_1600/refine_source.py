"""Run standard STGC ray-drop refinement on completed Modi01 EMA fields.

Calls an archived copy of the repository's original Trainer.refine unchanged:
47 full training frames, BCE only, 1000 Adam/OneCycleLR updates at max_lr .001.
The field is frozen; only unet parameters/buffers may change. All outputs go in
a fresh directory. Cached full-frame renders permit train evaluation without
recomputing an unchanged field; development renders are computed afresh.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now().astimezone().isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_fixed_field(torch, model, expected):
    current = model.state_dict()
    for name, value in expected.items():
        if torch.is_tensor(value):
            same = torch.equal(value, current[name].detach().cpu())
        else:
            same = value == current[name]
        if not same:
            raise RuntimeError(f'refinement changed the frozen field: {name}')


def worker(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    out = Path(spec['output'])
    reg = json.loads((Path(spec['workspace']) / 'registration.json').read_text())
    sys.path[:] = [entry for entry in sys.path if Path(entry or '.').resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(Path(spec['modi_source']).parent))
    import Modi01.runtime as runtime
    runtime.ROOT, runtime.SOURCE = Path(spec['repository']), Path(spec['audited_source'])
    runtime.activate_source()
    import numpy as np
    import torch
    from torch_ema import ExponentialMovingAverage
    from Modi01.field import RepresentationConfig
    from Modi01.model import STGCNeRFModi01
    from data.kitti360_dataset import KITTI360Dataset
    from utils.metrics import RaydropMeter, IntensityMeter, DepthMeter, PointsMeter
    from utils.refiner_input import refiner_input

    support = load_module('refine_evaluation_support', Path(spec['evaluation_source']))
    standard = load_module('refine_standard_runner', Path(spec['standard_runner']))
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    opt = SimpleNamespace(**reg['configuration'])
    assert opt.sequence_id == '8120' and opt.split_protocol == 'legacy'
    assert opt.num_steps == 768 and opt.raydrop_loss == 'mse'
    assert 'max_ray_batch' not in vars(opt)  # original refine supplies it explicitly
    names = ('min_resolution', 'base_resolution', 'max_resolution', 'time_resolution',
        'n_levels_plane', 'n_features_per_level_plane', 'n_levels_hash', 'n_features_per_level_hash',
        'log2_hashmap_size', 'num_layers_flow', 'hidden_dim_flow', 'num_layers_sigma',
        'hidden_dim_sigma', 'geo_feat_dim', 'num_layers_lidar', 'hidden_dim_lidar',
        'out_lidar_dim', 'num_frames', 'bound', 'density_scale', 'active_sensor')
    model = STGCNeRFModi01(representation=RepresentationConfig(**reg['representation']),
        **{key: getattr(opt, key) for key in names},
        near_lidar=opt.near_lidar * opt.scale, far_lidar=opt.far_lidar * opt.scale).cuda()
    source = Path(spec['checkpoint'])
    source_stat = (source.stat().st_size, source.stat().st_mtime_ns)
    ckpt = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    assert ckpt['stage'] == 'field' and ckpt['global_step'] == 30000
    model.load_state_dict(ckpt['model'], strict=True)
    ema = ExponentialMovingAverage(model.parameters(), decay=opt.ema_decay)
    assert len(ckpt['ema']['shadow_params']) == len(list(model.parameters()))
    assert all(a.shape == b.shape for a, b in zip(ckpt['ema']['shadow_params'], model.parameters()))
    ema.load_state_dict(ckpt['ema'])
    ema.copy_to()
    del ema
    model.eval().requires_grad_(False)
    model.unet.requires_grad_(True)
    assert model.unet.residual_basis is None
    expected = {name: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
                for name, value in model.state_dict().items() if not name.startswith('unet.')}
    initial_unet = {name: value.detach().cpu().clone() for name, value in model.unet.state_dict().items()}
    assert all(torch.equal(value, ckpt['model']['unet.'+name]) for name, value in initial_unet.items())
    assert all(name.startswith('unet.') for name, p in model.named_parameters() if p.requires_grad)
    # Verify the archived wrapper's UNet reproduces the standard implementation.
    native_unet = load_module('refine_standard_unet', Path(spec['standard_unet'])).UNet(3, out_channels=1).cuda().eval()
    native_unet.load_state_dict(model.unet.state_dict(), strict=True)
    with torch.no_grad():
        probe = torch.linspace(0, 1, 3*32*64, device='cuda').reshape(1, 3, 32, 64)
        torch.testing.assert_close(native_unet(probe), model.unet(probe), rtol=0, atol=0)
    del native_unet, probe

    datasets = {}
    for label, split in [('train', 'refine'), ('development', 'val')]:
        ds = KITTI360Dataset(device='cuda', split=split, root_path=opt.path, sequence_id=opt.sequence_id,
            split_protocol='legacy', preload=opt.preload, scale=opt.scale, offset=opt.offset,
            fp16=opt.fp16, fov_lidar=opt.fov_lidar)
        expected_ids = reg['assets']['splits']['train' if label == 'train' else 'val']['frame_ids']
        assert ds.frame_ids.tolist() == expected_ids and not ds.training and ds.num_rays_lidar == -1
        np.testing.assert_allclose(ds.times.cpu().numpy().ravel(), (np.asarray(expected_ids)-8120)/50, atol=1e-7)
        datasets[label] = ds
    assert len(datasets['train']) == 47 and datasets['development'].frame_ids.tolist() == [8130, 8140, 8150, 8160]
    assert not set(datasets['train'].frame_ids.tolist()) & set(datasets['development'].frame_ids.tolist())
    contract = dict(protocol='standard_stgc_bce_refinement_v1', source_weights='final_ema', source_steps=30000,
        source_ema_updates=ckpt['ema']['num_updates'], steps=1000, optimizer='Adam', maximum_learning_rate=0.001,
        scheduler='OneCycleLR', loss='probability BCE only', batch_size=47, train_frames=datasets['train'].frame_ids.tolist(),
        development_frames=datasets['development'].frame_ids.tolist(), augmentation_seed=0, optimization_seed=0,
        initialization='unchanged frozen UNet from field checkpoint; identical across three arms',
        initialization_replaced=False, mutable_prefix='unet.', field_fp16=opt.fp16, unet_training_dtype='float32',
        render_samples=768, render_batch=4096, random_boxes='original np.random.randint; 0..31 boxes, <10% height/width',
        previous_unused_refine_loss_preset=opt.refine_loss_preset,
        reason_for_protocol='user requested refinement after comparison with standard STGC; use its original BCE-only procedure',
        posthoc_development=True, untouched_test=False)
    state = dict(**spec, status='RUNNING', phase='PREPARING_TRAIN_RENDERS', started_at=now(),
                 contract=contract, checks=dict(standard_unet_eval_parity=True, no_development_training=True))
    write_json(out / 'status.json', state)
    start = time.monotonic()
    cache = []
    original_render = model.render

    def capture_render(*args, **kwargs):
        frame_start = time.monotonic()
        result = original_render(*args, **kwargs)
        # These are outputs of the frozen field, not trainable features.
        cache.append({key: result[key].detach().clone() for key in ['image_lidar', 'depth_lidar']})
        state.update(phase='PREPARING_TRAIN_RENDERS', rendered_frames=len(cache), total_train_frames=47,
                     elapsed_seconds=time.monotonic()-start, last_frame_seconds=time.monotonic()-frame_start)
        write_json(out / 'status.json', state)
        print(json.dumps(dict(phase=state['phase'], frame=len(cache), seconds=state['last_frame_seconds'])), flush=True)
        return result

    training_started = [None]

    def log(message):
        print(message, flush=True)
        if message == 'Start UNet Optimization ...':
            training_started[0] = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            state.update(phase='REFINING', optimizer_steps_reported=0)
        elif ' iter:' in message:
            step = int(message.split(' iter:')[1].split(',')[0])
            state.update(phase='REFINING', optimizer_steps_reported=step,
                         optimization_seconds=time.monotonic()-training_started[0])
        state.update(elapsed_seconds=time.monotonic()-start, updated_at=now())
        write_json(out / 'status.json', state)

    checkpoints = out / 'checkpoints'
    checkpoints.mkdir()
    trainer = SimpleNamespace(model=model, ema=None, opt=opt, log=log, bce_fn=torch.nn.BCELoss(),
        name='stgc_nerf', epoch=ckpt['epoch'], ckpt_path=str(checkpoints))
    model.render = capture_render
    # Common seeded augmentation/dropout stream. The original code itself is not
    # edited or monkeypatched; it performs exactly 1000 full-batch optimizer steps.
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    try:
        standard.Trainer.refine(trainer, datasets['train'].dataloader())
    finally:
        del model.render  # restore class method
    assert len(cache) == 47
    check_fixed_field(torch, model, expected)
    assert any(not torch.equal(initial_unet[name], value.detach().cpu()) for name, value in model.unet.state_dict().items())
    refined_path = checkpoints / f'stgc_nerf_ep{ckpt["epoch"]:04d}_refine.pth'
    saved = torch.load(refined_path, map_location='cpu', mmap=True, weights_only=False)
    saved_model = saved['model']
    assert saved['epoch'] == ckpt['epoch']
    # Original procedure already saved; check it, then publish rich metadata in a
    # fresh path instead of overwriting or modifying the source field checkpoint.
    for name, value in model.state_dict().items():
        assert (torch.equal(value.detach().cpu(), saved_model[name]) if torch.is_tensor(value) else value == saved_model[name])
    final_path = checkpoints / 'stgc_nerf_ep0639_refine_standard_bce.pth'
    torch.save(dict(stage='refined', epoch=ckpt['epoch'], global_step=30000, model=model.state_dict(),
                    refine_contract=contract, source_checkpoint=str(source)), final_path)
    state.update(phase='EVALUATING', refined_checkpoint=str(final_path), refinement_completed_at=now(),
                 optimization_steps=1000, optimization_seconds=time.monotonic()-training_started[0],
                 peak_refinement_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                 preparation_and_refinement_seconds=time.monotonic()-start)
    state['checks'].update(frozen_field_unchanged=True, unet_updated=True, original_refined_save_verified=True)
    write_json(out / 'status.json', state)
    del saved_model, saved
    # Reload the artifact actually handed to users before computing final metrics.
    reloaded = torch.load(final_path, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(reloaded['model'], strict=True)
    model.eval().requires_grad_(False)
    check_fixed_field(torch, model, expected)
    metrics = [RaydropMeter(.5), IntensityMeter(opt.intensity_scale), DepthMeter(opt.scale), PointsMeter(opt.scale, opt.fov_lidar)]
    rows, groups = [], {}
    for split in ['development', 'train']:
        for metric in metrics:
            metric.clear()
        strata_rows = []
        for index, data in enumerate(datasets[split].dataloader()):
            height, width = data['H_lidar'], data['W_lidar']
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=opt.fp16):
                result = original_render(data['rays_o_lidar'], data['rays_d_lidar'], data['time'],
                    staged=True, max_ray_batch=4096, perturb=False, **vars(opt)) if split == 'development' else cache[index]
                attrs = result['image_lidar'].reshape(-1, height, width, 2)
                p_i = attrs[..., 1]
                p_d = result['depth_lidar'].reshape(-1, height, width)
                p_r = model.unet(refiner_input(result, height, width)).squeeze(1)
            gt = data['images_lidar']
            g_r, g_i, g_d = gt[..., 0], gt[..., 1]*gt[..., 0], gt[..., 2]*gt[..., 0]
            frame_id = int(data['frame_id'].item())
            if split == 'development':
                with np.load(Path(spec['pre_evaluation']) / 'predictions/ema' / f'{frame_id}.npz') as previous:
                    # Same field and rendering settings must reproduce the saved
                    # pre-refiner depth/intensity, before applying a changed mask.
                    np.testing.assert_array_equal(p_d.cpu().numpy(), previous['depth_normalized'])
                    np.testing.assert_array_equal(p_i.cpu().numpy(), previous['intensity'])
                    np.testing.assert_array_equal(g_r.cpu().numpy(), previous['gt_return'])
            mask = torch.where(p_r > .5, 1, 0)
            for metric, pred, target in zip(metrics, [p_r, p_i*mask, p_d*mask, p_d*mask], [g_r,g_i,g_d,g_d]):
                metric.update(pred.clone(), target.clone())
            row = dict(variant=spec['variant'], weights='ema_field_refined_unet', split=split, frame_id=frame_id)
            for metric, names in zip(metrics, support.METRIC_NAMES):
                row.update({key:float(value) for key,value in zip(names,metric.V[-1])})
            rows.append(row)
            strata_rows.append(support.pixel_strata(p_i,p_d,p_r,g_i,g_d,g_r,scale=opt.scale))
            if split == 'development':
                dest = out / 'predictions'
                dest.mkdir(exist_ok=True)
                np.savez_compressed(dest/f'{frame_id}.npz', intensity=p_i.cpu().numpy(),depth_normalized=p_d.cpu().numpy(),
                    return_probability=p_r.cpu().numpy(),gt_intensity=g_i.cpu().numpy(),gt_depth_normalized=g_d.cpu().numpy(),
                    gt_return=g_r.cpu().numpy(),scale=np.array(opt.scale))
            print(json.dumps(dict(phase='EVALUATING',split=split,frame=index+1,total=len(datasets[split]),cd=row['cd_m2'])),flush=True)
            state.update(phase='EVALUATING',evaluation_split=split,evaluated_frames=index+1,elapsed_seconds=time.monotonic()-start)
            write_json(out / 'status.json',state)
        aggregate = {key:float(value) for metric,names in zip(metrics,support.METRIC_NAMES)
                     for key,value in zip(names,metric.measure())}
        groups[split] = dict(frames=len(datasets[split]),metrics=aggregate,strata=support.summarize_strata(strata_rows))
        write_json(out/'results.json',dict(variant=spec['variant'],status='RUNNING',contract=contract,groups=groups))
        with (out/'per_frame.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    check_fixed_field(torch,model,expected)
    assert source_stat==(source.stat().st_size,source.stat().st_mtime_ns)
    state['checks'].update(source_checkpoint_stat_unchanged=True,development_unmasked_depth_intensity_unchanged=True)
    state.update(status='COMPLETED',phase='COMPLETED',completed_at=now(),elapsed_seconds=time.monotonic()-start)
    write_json(out/'status.json',state)
    write_json(out/'results.json',dict(variant=spec['variant'],status='COMPLETED',contract=contract,groups=groups,
        checkpoint=str(final_path),checks=state['checks'],elapsed_seconds=state['elapsed_seconds']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-evaluation',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if args.from_evaluation is None or args.output is None:
        parser.error('--from-evaluation and --output are required')
    out=args.output.resolve()
    if not out.is_relative_to(ROOT/'Modi01/refinements'):
        parser.error('output must be a fresh directory in Modi01/refinements')
    out.mkdir(parents=True,exist_ok=False)
    previous=json.loads((args.from_evaluation/'evaluation.json').read_text())
    assert previous['status']=='COMPLETED'
    for source,dest in [(Path(__file__),out/'refine_source.py'),(ROOT/'model/runner.py',out/'standard_runner.py'),
                        (ROOT/'model/unet.py',out/'standard_unet.py'),(ROOT/'Modi01/evaluate.py',out/'evaluation_support.py')]:
        dest.write_bytes(source.read_bytes())
    # The three checkpoint UNets must begin from identical states, including BN
    # buffers. Direct tensor comparison only, without file digests.
    import torch
    reference=None
    for job in previous['jobs']:
        c=torch.load(job['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
        unet={k:v for k,v in c['model'].items() if k.startswith('unet.')}
        if reference is None:
            reference={k:v.clone() for k,v in unet.items()}
        else:
            assert unet.keys()==reference.keys() and all(torch.equal(v,reference[k]) for k,v in unet.items())
    state=dict(status='RUNNING',started_at=now(),from_evaluation=str(args.from_evaluation.resolve()),
               common_initial_unet_verified=True,jobs=[],completed=[])
    for job in previous['jobs']:
        destination=out/job['variant']
        destination.mkdir()
        spec=dict(job,repository=str(ROOT),pre_evaluation=job['output'],output=str(destination),standard_runner=str(out/'standard_runner.py'),
                  standard_unet=str(out/'standard_unet.py'),evaluation_source=str(out/'evaluation_support.py'))
        write_json(destination/'job.json',spec)
        state['jobs'].append(spec)
    write_json(out/'refinement.json',state)
    for job in state['jobs']:
        state['active_variant']=job['variant']
        write_json(out/'refinement.json',state)
        with (Path(job['output'])/'refine.log').open('w') as log:
            result=subprocess.run([sys.executable,'-u',str(out/'refine_source.py'),'--worker',str(Path(job['output'])/'job.json')],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        if result.returncode:
            state.update(status='FAILED',exit_code=result.returncode)
            write_json(out/'refinement.json',state)
            raise RuntimeError(f"Refinement failed for {job['variant']}; see its refine.log")
        state['completed'].append(job['variant'])
        write_json(out/'refinement.json',state)
    state.update(status='COMPLETED',completed_at=now())
    write_json(out/'refinement.json',state)
    print(f'Completed refinement and evaluation: {out}',flush=True)


if __name__=='__main__':
    main()
