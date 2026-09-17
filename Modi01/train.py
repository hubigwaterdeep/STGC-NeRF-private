"""Run one registered from-scratch arm, or prepare its assets/manifest only.

Usage: python -m Modi01.train --config configs/kitti360_8120.txt --prepare-only
The original runner supplies losses, ray sampling, flow, EMA, and refinement.
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import torch
import numpy as np

from Modi01.runtime import ROOT, SOURCE, activate_source
activate_source()
from Modi01.field import RepresentationConfig
from Modi01.model import STGCNeRFModi01


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def upstream_entry():
    spec = importlib.util.spec_from_file_location('modi01_upstream_entry', SOURCE / 'main_ours.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def data_manifest(opt):
    root = Path(opt.path).resolve()
    sequence = int(opt.sequence_id)
    manifest_path = ROOT / 'data/experiment_manifest.json'
    existing = json.loads(manifest_path.read_text())
    scene = next(s for s in existing['scenes'] if s['dataset'] == 'kitti360' and s['sequence_id'] == sequence)
    count = scene['num_frames']
    config = scene['configuration']
    if opt.num_frames != count or not np.isclose(opt.scale, config['scale'], rtol=1e-9):
        raise ValueError('num_frames/scale differ from the registered dataset manifest')
    if not np.allclose(opt.offset, config['offset'], rtol=0, atol=1e-8) or opt.fov_lidar != config['fov_lidar']:
        raise ValueError('offset/FOV differ from the registered dataset manifest')
    held_out = [sequence + i for i in scene['test_zero_based_indices']]
    train = [i for i in range(sequence, sequence+count) if i not in held_out]
    splits, missing = {}, []
    for split, expected in [('train', train), ('val', held_out), ('test', held_out)]:
        path = root / f'transforms_{sequence}_{split}.json'
        if not path.is_file():
            missing.append(str(path))
            continue
        data = json.loads(path.read_text())
        ids = [f['frame_id'] for f in data['frames']]
        if ids != expected or data['num_frames'] != count or data['num_frames_split'] != len(expected):
            raise ValueError(f'{path}: frame IDs/counts disagree with registered official protocol')
        shapes = set()
        for frame in data['frames']:
            pose = np.asarray(frame['lidar2world'])
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f'invalid pose for frame {frame["frame_id"]}')
            asset = root / frame['lidar_file_path']
            if not asset.is_file():
                missing.append(str(asset))
                continue
            array = np.load(asset, mmap_mode='r', allow_pickle=False)
            if array.shape != (data['h_lidar'], data['w_lidar'], 3):
                raise ValueError(f'invalid range view shape: {asset}: {array.shape}')
            shapes.add(tuple(array.shape))
        splits[split] = dict(manifest=str(path), frame_ids=ids, count=len(ids),
            normalized_times=[(i-sequence)/(count-1) for i in ids], range_view_shapes=sorted(shapes))
    if not Path(opt.resume).is_file():
        missing.append(str(Path(opt.resume).resolve()))
    return dict(source_manifest=str(manifest_path), protocol='legacy', splits=splits,
        scale=opt.scale, offset=opt.offset, fov_lidar=opt.fov_lidar,
        missing_assets=sorted(set(missing)),
        protocol_scope='local official-protocol manifest and loader agreement; no file digests computed',
        evaluation_scope='post-hoc development; legacy val/test overlap, no untouched evaluation claim')


def create_parser(upstream):
    parser = upstream.get_arg_parser()
    parser.set_defaults(config=str(ROOT / 'configs/kitti360_8120.txt'), workspace='',
                        skip_final_eval=True, skip_refine=True)
    parser.add_argument('--representation', default=str(ROOT / 'Modi01/configs/hybrid44.json'))
    parser.add_argument('--prepare-only', action='store_true')
    return parser


def build_model(opt, config):
    names = ('min_resolution', 'base_resolution', 'max_resolution', 'time_resolution',
        'n_levels_plane', 'n_features_per_level_plane', 'n_levels_hash', 'n_features_per_level_hash',
        'log2_hashmap_size', 'num_layers_flow', 'hidden_dim_flow', 'num_layers_sigma',
        'hidden_dim_sigma', 'geo_feat_dim', 'num_layers_lidar', 'hidden_dim_lidar',
        'out_lidar_dim', 'num_frames', 'bound', 'density_scale', 'active_sensor')
    model = STGCNeRFModi01(representation=config, **{k: getattr(opt, k) for k in names},
        near_lidar=opt.near_lidar*opt.scale, far_lidar=opt.far_lidar*opt.scale)
    groups = model.get_params(opt.lr)
    report = model.scene_field.representation_report()
    report['optimizer_groups'] = [dict(stage_role=g.get('stage_role', 'base'), lr=g['lr'],
        parameters=sum(p.numel() for p in g['params'])) for g in groups]
    report['model_parameters'] = sum(p.numel() for p in model.parameters())
    report['model_trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_json(Path(opt.workspace) / 'representation.json', report)
    print(json.dumps(report, indent=2))
    return model


def main():
    upstream = upstream_entry()
    parser = create_parser(upstream)
    opt = parser.parse_args()
    config = RepresentationConfig(**json.loads(Path(opt.representation).read_text()))
    config.validate(opt.n_levels_hash)
    if opt.dataloader != 'kitti360' or opt.split_protocol != 'legacy' or opt.sequence_id not in ('8120', '10200', '3353'):
        parser.error('registered diagnosis uses KITTI-360 8120/10200/3353 with the legacy official frame protocol')
    if opt.field_backend != 'best' or opt.intensity_feature_mode != 'none' or opt.init_best:
        parser.error('representation study requires unchanged Best heads and no pretrained scene initialization')
    if opt.test or opt.test_eval or opt.refine or opt.ckpt != 'scratch':
        parser.error('this entry is from-scratch training only; use a separate, explicitly labeled evaluation job')
    if opt.iters < 1 or opt.max_train_steps < 0 or opt.max_train_steps > opt.iters:
        parser.error('require 0 <= max_train_steps <= iters, with iters positive')
    label = f'modal{config.modal_levels}_{config.coefficients}_seed{opt.seed}'
    workspace = Path(opt.workspace or ROOT / 'Modi01/runs' / opt.sequence_id / label).resolve()
    # Never let inherited runner names overwrite named Best/lastBest experiments.
    if not workspace.is_relative_to(ROOT / 'Modi01') or workspace == ROOT / 'Modi01':
        parser.error('workspace must be a dedicated directory inside Modi01')
    if workspace.exists() and not opt.prepare_only:
        allowed = {'registration.json', 'log/EXPERIMENT_POLICY.md'}
        occupied = [p for p in workspace.rglob('*') if p.is_file() and str(p.relative_to(workspace)) not in allowed]
        if occupied:
            parser.error('workspace contains run artifacts; choose a fresh directory')
    opt.workspace = str(workspace)
    assets = data_manifest(opt)
    record = dict(status='BLOCKED' if assets['missing_assets'] else 'READY_NOT_RUN',
        source=str(SOURCE), model='Modi01.model.STGCNeRFModi01', configuration=vars(opt),
        representation=json.loads(Path(opt.representation).read_text()), assets=assets,
        training_steps_executed=0, fixed_seed=opt.seed,
        requested_training_budget=opt.max_train_steps or opt.iters,
        evaluation_conventions=dict(training='raw', development='EMA when enabled',
            refiner='disabled by default; run separately with explicit pre/post labels'),
        required_metrics=['CD', 'F-score', 'depth RMSE', 'intensity RMSE', 'return metrics'],
        required_strata=['train/development', 'moving/static/edge/range when labels are reliable'],
        sampling='inherited STGC sampler and patch schedule; shared initialization preserves RNG state',
        no_automatic_queue=True, no_promotion=True)
    write_json(workspace / 'registration.json', record)
    policy = workspace / 'log/EXPERIMENT_POLICY.md'
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_text('# Modi01 experiment registration\n\n'
        'From scratch; official legacy frame IDs. No file digests are computed.\n'
        'This is post-hoc development; val/test overlap and cannot establish untouched-test performance.\n'
        f'Source: `{SOURCE}`\n\n'
        f'Frame IDs, normalization and manifests: `{workspace / "registration.json"}`\n\n'
        f'Fixed seed {opt.seed}; maximum {opt.max_train_steps or opt.iters} optimizer updates.\n'
        'Use identical budgets, seed, input ordering, ray sampler and patch schedule for controls.\n'
        'Report all five metric families together; label raw/EMA and pre/post-refiner separately.\n'
        'Moving/static/edge/range strata require reliable labels; otherwise report unavailable.\n'
        'Stop after this single arm. No automatic experiment queue, benchmark sweep or promotion.\n')
    print(json.dumps(record, indent=2, ensure_ascii=False))
    if opt.prepare_only:
        return
    if assets['missing_assets']:
        raise RuntimeError('real training BLOCKED by missing assets; see registration.json')
    # Only this imported entry's two explicit hooks change. The historical files
    # and global model/field registries remain untouched.
    upstream.get_arg_parser = lambda: parser
    upstream.build_model = lambda _: build_model(opt, config)
    # Upstream parses the same argv again, so provide the resolved workspace.
    import sys
    sys.argv += ['--workspace', str(workspace)]
    record['status'] = 'RUNNING'
    record.pop('training_steps_executed')
    record['training_counters'] = 'runner checkpoint and optimizer logs'
    write_json(workspace / 'registration.json', record)
    try:
        upstream.main()
    except BaseException:
        record['status'] = 'FAILED'
        write_json(workspace / 'registration.json', record)
        raise
    record['status'] = 'FINISHED_REQUIRES_JOINT_METRIC_REVIEW'
    # Actual counters are recorded in runner checkpoints/logs, not guessed here.
    write_json(workspace / 'registration.json', record)


if __name__ == '__main__':
    main()
