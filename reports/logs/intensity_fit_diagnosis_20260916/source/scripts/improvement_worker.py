"""One GPU job in the official47 improvement campaign; no file digests."""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.deps'), str(ROOT)]
import numpy as np
import torch
from best_core.appearance_objectives import intensity_gradient_loss
from best_core.refine_objectives import visibility_refine_loss_from_preset
from model.stgc_best import STGC_NeRF_Best
from data.kitti360_dataset import KITTI360Dataset
from utils.refiner_input import refiner_input

EVAL_IDS = [4960, 4970, 4980, 4990]
TRAIN_IDS = [i for i in range(4950, 5001) if i not in EVAL_IDS]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def seed(value=0):
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def options(campaign):
    plan = json.loads((campaign / 'plan.json').read_text())
    return plan, json.loads(Path(plan['baseline_args']).read_text())


def model_from(opt, checkpoint, feature='none', materialize_ema=False):
    seed(0)
    keys = ('min_resolution', 'base_resolution', 'max_resolution', 'time_resolution',
            'n_levels_plane', 'n_features_per_level_plane', 'n_levels_hash',
            'n_features_per_level_hash', 'log2_hashmap_size', 'num_layers_flow',
            'hidden_dim_flow', 'num_layers_sigma', 'hidden_dim_sigma', 'geo_feat_dim',
            'num_layers_lidar', 'hidden_dim_lidar', 'out_lidar_dim', 'num_frames',
            'bound', 'density_scale', 'active_sensor')
    kwargs = {k: opt[k] for k in keys}
    kwargs.update(near_lidar=opt['near_lidar'] * opt['scale'],
                  far_lidar=opt['far_lidar'] * opt['scale'])
    model = STGC_NeRF_Best(intensity_feature_mode=feature, **kwargs).cuda().eval()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    incoming = state['model']
    missing = set(model.state_dict()) - set(incoming)
    if any(not key.startswith('intensity_adapter.') for key in missing):
        raise ValueError(f'checkpoint missing non-adapter parameters: {missing}')
    initial = model.state_dict()
    initial.update(incoming)
    model.load_state_dict(initial, strict=True)
    if materialize_ema and 'ema' in state:
        from torch_ema import ExponentialMovingAverage
        ema = ExponentialMovingAverage(model.parameters(), decay=opt['ema_decay'])
        ema.load_state_dict(state['ema'])
        ema.copy_to()
    del state, incoming, initial
    gc.collect()
    return model


def dataset(opt, split):
    ds = KITTI360Dataset(device='cuda', split=split, root_path=opt['path'],
        sequence_id='4950', split_protocol='legacy', preload=True,
        scale=opt['scale'], offset=opt['offset'], fp16=True, num_rays_lidar=1024,
        fov_lidar=opt['fov_lidar'])
    expected = TRAIN_IDS if split in ('train', 'refine') else EVAL_IDS
    if ds.frame_ids.tolist() != expected:
        raise ValueError(f'incorrect official frames for {split}: {ds.frame_ids.tolist()}')
    return ds


def alignment(opt):
    result = {'protocol': 'STGC official47 / legacy', 'hash_policy': 'disabled by user',
              'normalization': {k: opt[k] for k in ('scale', 'offset', 'fov_lidar')},
              'splits': {}}
    for split in ('train', 'val', 'test'):
        path = Path(opt['path']) / f'transforms_4950_{split}.json'
        manifest = json.loads(path.read_text())
        rows = sorted(manifest['frames'], key=lambda x: x['lidar_file_path'])
        ids = [int(row['frame_id']) for row in rows]
        if ids != (TRAIN_IDS if split == 'train' else EVAL_IDS):
            raise ValueError(f'incorrect manifest: {path}')
        for row in rows:
            if not (Path(opt['path']) / row['lidar_file_path']).is_file():
                raise FileNotFoundError(row['lidar_file_path'])
        result['splits'][split] = {'manifest': str(path), 'count': len(ids), 'frame_ids': ids}
    result['splits']['refine'] = dict(result['splits']['train'])
    return result


def render_arrays(model, data, opt):
    h, w = data['H_lidar'], data['W_lidar']
    with torch.no_grad(), torch.autocast('cuda', enabled=opt['fp16']):
        output = model.render(data['rays_o_lidar'], data['rays_d_lidar'], data['time'],
                              staged=True, max_ray_batch=4096, num_steps=768, perturb=False)
        image = output['image_lidar'].reshape(1, h, w, 2).permute(0, 3, 1, 2)
        depth = output['depth_lidar'].reshape(1, 1, h, w)
        inputs = torch.cat([image, depth], 1)
        probability = model.unet(refiner_input(output, h, w)).float()
    return inputs.float().cpu(), data['images_lidar'].permute(0, 3, 1, 2).float().cpu(), probability.cpu()


def cache(campaign, name='baseline', checkpoint=None, feature='none'):
    plan, opt = options(campaign)
    checkpoint = str(checkpoint or plan['baseline_checkpoint'])
    folder = campaign / 'cache' / name
    folder.mkdir(parents=True, exist_ok=True)
    contract = {'checkpoint': checkpoint, 'feature': feature, 'num_steps': 768,
                'fp16': True, 'alignment': alignment(opt)}
    if (folder / 'complete.json').exists():
        if json.loads((folder / 'complete.json').read_text()) != contract:
            raise ValueError('cache contract mismatch')
        return folder
    model = model_from(opt, checkpoint, feature)
    for split in ('refine', 'val'):
        ds = dataset(opt, split)
        rows, targets, probabilities = [], [], []
        for index in range(len(ds.frame_ids)):
            row, target, probability = render_arrays(model, ds.collate([index]), opt)
            rows.append(row); targets.append(target); probabilities.append(probability)
            if index % 5 == 0:
                write_json(folder / 'progress.json', {'split': split, 'rendered': index + 1,
                                                     'total': len(ds.frame_ids)})
                print(f'CACHE {split} {index + 1}/{len(ds.frame_ids)}', flush=True)
        torch.save({'inputs': torch.cat(rows), 'targets': torch.cat(targets),
                    'probabilities': torch.cat(probabilities), 'frame_ids': ds.frame_ids.tolist()},
                   folder / f'{split}.pt')
        del ds, rows, targets, probabilities
        gc.collect(); torch.cuda.empty_cache()
    del model
    gc.collect(); torch.cuda.empty_cache()
    write_json(folder / 'complete.json', contract)
    return folder


def evaluate(inputs, targets, probabilities, frame_ids, opt, out, scope):
    from utils.metrics import IntensityMeter, DepthMeter, PointsMeter, RaydropMeter
    meters = [IntensityMeter(1), DepthMeter(opt['scale']),
              PointsMeter(opt['scale'], opt['fov_lidar']), RaydropMeter()]
    rows = []
    out.mkdir(parents=True, exist_ok=True)
    for j, fid in enumerate(frame_ids):
        raw_i = inputs[j:j+1, 1].numpy().copy()
        raw_d = inputs[j:j+1, 2].numpy().copy()
        gt_m = targets[j:j+1, 0].numpy().copy()
        gt_i = targets[j:j+1, 1].numpy().copy() * gt_m
        gt_d = targets[j:j+1, 2].numpy().copy() * gt_m
        probability = probabilities[j:j+1, 0].numpy().copy()
        keep = probability > .5
        pred_i, pred_d = raw_i * keep, raw_d * keep
        for meter in meters:
            meter.clear()
        # Match the official runner's dtypes and operation order: predictions
        # are float32, GT range-view is fp16, and unet probabilities are fp16.
        # In particular, GT depth division by scale occurs before numpy conversion.
        meters[0].update(torch.from_numpy(pred_i.copy()), torch.from_numpy(gt_i.copy()).half())
        meters[1].update(torch.from_numpy(pred_d.copy()), torch.from_numpy(gt_d.copy()).half())
        meters[2].update(torch.from_numpy(pred_d.copy()), torch.from_numpy(gt_d.copy()).half())
        # RMSE measures probabilities; accuracy/F1 threshold them internally.
        meters[3].update(torch.from_numpy(probability.copy()).half(), torch.from_numpy(gt_m.copy()).half())
        names = ('rmse', 'medae', 'lpips', 'ssim', 'psnr')
        row = {'frame_id': fid}
        row.update({f'intensity_{k}': float(v) for k,v in zip(names, meters[0].measure())})
        row.update({f'depth_{k}': float(v) for k,v in zip(names, meters[1].measure())})
        row.update(dict(zip(('cd','fscore'), map(float, meters[2].measure()))))
        row.update(dict(zip(('ray_rmse','ray_accuracy','ray_f1'), map(float, meters[3].measure()))))
        false_negative = (gt_m > .5) & ~keep
        false_positive = (gt_m < .5) & keep
        common = (gt_m > .5) & keep
        row.update(fn_count=int(false_negative.sum()), fp_count=int(false_positive.sum()),
                   fn_intensity_sse=float(np.square(gt_i[false_negative]).sum()),
                   fp_intensity_sse=float(np.square(raw_i[false_positive]).sum()),
                   retained_true_intensity_rmse=float(np.sqrt(np.square(raw_i[common]-gt_i[common]).mean())))
        rows.append(row)
        np.savez_compressed(out / f'{fid}.npz', intensity=raw_i[0], depth_m=raw_d[0]/opt['scale'],
                            probability=probability[0], gt_mask=gt_m[0], gt_intensity=gt_i[0],
                            gt_depth_m=(torch.from_numpy(gt_d[0]).half()/opt['scale']).numpy())
    summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != 'frame_id'}
    report = {'scope': scope, 'mean': summary, 'per_frame': rows}
    write_json(out / 'metrics.json', report)
    with (out / 'per_frame.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({'metrics': summary, 'scope': scope}), flush=True)
    return report


def refine_job(campaign, spec, out):
    plan, opt = options(campaign)
    checkpoint = spec.get('checkpoint', plan['baseline_checkpoint'])
    feature = spec.get('feature', 'none')
    folder = cache(campaign, spec.get('cache', 'baseline'), checkpoint, feature)
    model = model_from(opt, checkpoint, feature)
    model.requires_grad_(False)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(0)
        for module in model.unet.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
    model.unet.requires_grad_(True).train()
    field_before = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()
                    if not k.startswith('unet.')}
    train = torch.load(folder / 'refine.pt', weights_only=True)
    inputs, gt = train['inputs'].cuda(), train['targets'].cuda()
    if train['frame_ids'] != TRAIN_IDS:
        raise ValueError('refine cache is not official47')
    criterion = visibility_refine_loss_from_preset(spec['preset'])
    optimizer = torch.optim.Adam(model.unet.parameters(), lr=.001, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=.001, total_steps=1000)
    seed(0)
    rng = np.random.default_rng(0)
    started = time.time()
    peak = torch.cuda.max_memory_allocated()
    for step in range(1000):
        optimizer.zero_grad(set_to_none=True)
        augmentation = torch.ones_like(inputs)
        height, width = inputs.shape[-2:]
        for _ in range(rng.integers(32)):
            h = int(rng.integers(1, max(2, int(.1 * height))))
            w = int(rng.integers(1, max(2, int(.1 * width))))
            y, x = int(rng.integers(height-h)), int(rng.integers(width-w))
            augmentation[:, :, y:y+h, x:x+w] = 0
        probability = model.unet(inputs * augmentation)
        terms = criterion(probability, gt[:,0:1], inputs[:,2:3], gt[:,2:3],
                          depth_scale=opt['scale'], predicted_intensity=inputs[:,1:2],
                          target_intensity=gt[:,1:2] * gt[:,0:1])
        if not torch.isfinite(terms.total):
            raise RuntimeError(f'non-finite refine loss at {step+1}')
        terms.total.backward()
        optimizer.step(); scheduler.step()
        if step % 25 == 0 or step == 999:
            progress = {'step': step+1, 'total': 1000, 'seconds': time.time()-started,
                        'loss': float(terms.total), 'bce': float(terms.bce),
                        'depth_risk': float(terms.expected_depth_risk),
                        'intensity_risk': float(terms.total-terms.bce-terms.expected_depth_risk)}
            write_json(out / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    for name, value in field_before.items():
        if not torch.equal(value, model.state_dict()[name].detach().cpu()):
            raise RuntimeError(f'refine changed field tensor: {name}')
    del field_before, inputs, gt, train, optimizer, scheduler
    model.eval()
    state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    torch.save({'model': state, 'global_step': 30000, 'epoch': 639,
                'feature_mode': feature, 'refine_contract': {
                    'steps': 1000, 'learning_rate': .001, 'seed': 0,
                    'loss_preset': spec['preset'], 'source': str(checkpoint),
                    'protocol': 'official47_intensity_campaign_v1'}}, out / 'checkpoint.pth')
    validation = torch.load(folder / 'val.pt', weights_only=True)
    probabilities = []
    with torch.no_grad(), torch.autocast('cuda', enabled=opt['fp16']):
        for row in validation['inputs'].split(1):
            probabilities.append(model.unet(row.cuda()).float().cpu())
    metrics = evaluate(validation['inputs'], validation['targets'], torch.cat(probabilities),
                       EVAL_IDS, opt, out / 'evaluation', 'official final prediction')
    write_json(out / 'result.json', {'state': 'completed', 'spec': spec, 'metrics': metrics,
        'checkpoint': str(out / 'checkpoint.pth'), 'frozen_field_unchanged': True,
        'parameters': sum(p.numel() for p in model.parameters()),
        'training_seconds': time.time()-started,
        'peak_gpu_bytes': max(peak, torch.cuda.max_memory_allocated())})


def attribute_job(campaign, spec, out):
    plan, opt = options(campaign)
    model = model_from(opt, plan['baseline_checkpoint'], spec['feature'])
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(('intensity_net.', 'intensity_adapter.')))
    frozen = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()
              if not k.startswith(('intensity_net.', 'intensity_adapter.'))}
    mutable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(mutable, lr=.001, eps=1e-15)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: .1 ** min(s/3000, 1))
    scaler = torch.amp.GradScaler('cuda', init_scale=1)
    from torch_ema import ExponentialMovingAverage
    ema = ExponentialMovingAverage(mutable, decay=.95)
    ds = dataset(opt, 'train')
    seed(0)
    rng = np.random.default_rng(0)
    order = []
    started = time.time()
    for step in range(3000):
        if step % 47 == 0:
            order = rng.permutation(47).tolist()
        epoch = step // 47 + 1
        ds.patch_size_lidar = [2,8] if epoch % 2 == 0 else 1
        data = ds.collate([order[step % 47]])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', enabled=True):
            output = model.render(data['rays_o_lidar'], data['rays_d_lidar'], data['time'],
                                  staged=False, num_steps=768, perturb=True)
            prediction = output['image_lidar'][...,1].float()
            gt = data['images_lidar'].float()
            pixel = ((prediction-gt[...,1]) * gt[...,0]).square().sum() * .1
            gradient = intensity_gradient_loss(prediction, gt[...,1], gt[...,0],
                                               gt[...,2]/opt['scale'], ds.patch_size_lidar)
            loss = pixel + .1 * spec.get('gradient',0) * prediction.numel() * gradient
        if not torch.isfinite(loss):
            raise RuntimeError(f'non-finite attribute loss at {step+1}')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in mutable):
            raise RuntimeError(f'non-finite attribute gradient at {step+1}')
        scaler.step(optimizer); scaler.update(); scheduler.step(); ema.update()
        if step % 50 == 0 or step == 2999:
            progress = {'step': step+1, 'total': 3000, 'seconds': time.time()-started,
                        'loss': float(loss), 'pixel': float(pixel), 'gradient': float(gradient)}
            write_json(out / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    ema.copy_to()
    for name,value in frozen.items():
        if not torch.equal(value, model.state_dict()[name].detach().cpu()):
            raise RuntimeError(f'attribute stage changed frozen tensor: {name}')
    del frozen, ds, optimizer, scheduler, ema
    model.eval()
    torch.save({'model': {k:v.detach().cpu() for k,v in model.state_dict().items()},
                'global_step': 30000, 'attribute_steps': 3000, 'feature_mode': spec['feature'],
                'attribute_contract': spec, 'weights_kind': 'attribute EMA, frozen field'},
               out / 'checkpoint.pth')
    ds = dataset(opt, 'val')
    rows, targets = [], []
    surface_rows = []
    for index, fid in enumerate(EVAL_IDS):
        data = ds.collate([index])
        row, target, _ = render_arrays(model, data, opt)
        rows.append(row); targets.append(target)
        valid = data['images_lidar'][0,...,0].reshape(-1) > .5
        ranges = data['images_lidar'][0,...,2].reshape(-1)[valid].float()
        directions = data['rays_d_lidar'][0,valid].float()
        positions = data['rays_o_lidar'][0,valid].float() + directions * ranges[:,None]
        if ((positions < model.aabb[:3]) | (positions > model.aabb[3:])).any():
            raise RuntimeError('GT-surface diagnostic unexpectedly outside bounds')
        predicted = []
        with torch.no_grad(), torch.autocast('cuda', enabled=True):
            for start in range(0,len(positions),4096):
                xyz = positions[start:start+4096]
                predicted.append(model.attribute(xyz, directions[start:start+4096],
                                  **model.density(xyz, data['time']))[:,1].float())
        gt_i = data['images_lidar'][0,...,1].reshape(-1)[valid].float()
        surface_rows.append({'frame_id':fid, 'rmse':float((torch.cat(predicted)-gt_i).square().mean().sqrt())})
    validation = torch.load(campaign / 'cache/baseline/val.pt', weights_only=True)
    rows, targets = torch.cat(rows), torch.cat(targets)
    # Frozen geometry/raw return must also preserve their rendered outputs.
    if not torch.equal(rows[:,[0,2]], validation['inputs'][:,[0,2]]):
        raise RuntimeError('attribute-only training changed depth/raw-return rendering')
    metrics = evaluate(rows, targets, validation['probabilities'], EVAL_IDS, opt,
                       out / 'evaluation', 'diagnostic: frozen original final mask')
    write_json(out / 'surface_diagnostic.json', surface_rows)
    write_json(out / 'result.json', {'state':'completed', 'spec':spec, 'metrics':metrics,
        'checkpoint':str(out/'checkpoint.pth'), 'frozen_field_and_mask':True,
        'parameters':sum(p.numel() for p in model.parameters()),
        'trainable_parameters':sum(p.numel() for p in mutable),
        'training_seconds':time.time()-started, 'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
        'surface_probe_scope':'GT-only diagnostic, not official metric'})


def evaluate_checkpoint(campaign, spec, out):
    _, opt = options(campaign)
    model = model_from(opt, spec['checkpoint'], spec.get('feature','none'))
    ds = dataset(opt,'val')
    rows, targets, probs = [], [], []
    for index in range(4):
        row, target, prob = render_arrays(model, ds.collate([index]), opt)
        rows.append(row); targets.append(target); probs.append(prob)
    metrics = evaluate(torch.cat(rows),torch.cat(targets),torch.cat(probs),EVAL_IDS,opt,
                       out/'evaluation','official final prediction')
    write_json(out/'result.json',{'state':'completed','spec':spec,'metrics':metrics,
                                'checkpoint':spec['checkpoint']})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--job',required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    spec = json.loads((campaign/'jobs'/f'{args.job}.json').read_text())
    out = campaign/'runs'/args.job
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        raise FileExistsError('completed job cannot be overwritten')
    _,opt = options(campaign)
    write_json(out/'data_alignment.json',alignment(opt))
    write_json(out/'spec.json',spec)
    torch.cuda.reset_peak_memory_stats()
    if spec['kind'] == 'cache':
        folder = cache(campaign)
        validation = torch.load(folder/'val.pt',weights_only=True)
        metrics = evaluate(validation['inputs'],validation['targets'],validation['probabilities'],
                           EVAL_IDS,opt,out/'evaluation','baseline reproduced final prediction')
        write_json(out/'result.json',{'state':'completed','cache':str(folder),'metrics':metrics})
    elif spec['kind'] == 'refine':
        refine_job(campaign,spec,out)
    elif spec['kind'] == 'attribute':
        attribute_job(campaign,spec,out)
    elif spec['kind'] == 'evaluate':
        evaluate_checkpoint(campaign,spec,out)
    else:
        raise ValueError(spec['kind'])


if __name__ == '__main__':
    main()
