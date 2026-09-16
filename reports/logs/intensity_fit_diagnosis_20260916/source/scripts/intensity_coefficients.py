"""T0/T1 preflight and pre-registered finite training; never runs test."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import types
from datetime import datetime, timezone

SOURCE = Path(__file__).resolve().parents[1]
RUN = SOURCE.parent
sys.path[:0] = [str(SOURCE/'.deps'), str(SOURCE)]
AUDIT = Path('/home/zijiewu/Code/STGC-NeRF-private/reports/logs/intensity_stage1_audit_20260915/continuation_20260915')
sys.path.append(str(AUDIT))
from scripts.intensity_ablation import write_json, validate_baseline, construct_model, gpu_slot

BASELINE = Path('/home/zijiewu/Code/basis4D/log/best_official47_baseline')
IDS = [4960,4970,4980,4990]
SPEC = dict(baseline=str(BASELINE), seed=0, steps=3000, num_rays=128, num_steps=768,
            max_ray_batch=512, learning_rate=.001, ema_decay=.95,
            decoder_width=64, bank_levels=[0,1], bank_channels=[0], evaluate_at=[0,1000,2000,3000],
            loss='alpha_i * sum(((rendered_intensity - gt_intensity) * gt_mask)^2)',
            selection='fixed final step 3000; all intermediate EMA curves reported',
            test_executed=False, file_digest_policy='disabled by user',
            canonical_intensity_integration='paired separate-channel sum')


def inputs():
    import torch
    opt,alignment=validate_baseline(BASELINE)
    state=torch.load(alignment['checkpoint'],map_location='cpu',weights_only=False)
    return opt,alignment,state


def make_model(opt,state,arm):
    import torch
    from best_core.intensity_coefficients import attach_correction
    torch.manual_seed(0)
    model=construct_model(opt,state,dict(mode='none',current_weight=.5,seed=0,parameter_budget=32768))
    return attach_correction(model,train_coefficients=arm=='T1',seed=0)


def frozen_equal(model,state):
    import torch
    current=model.state_dict()
    for key,value in state['model'].items():
        assert torch.equal(value,current[key].detach().cpu()),key
    assert all(not p.requires_grad and p.grad is None for n,p in model.named_parameters()
               if not n.startswith('intensity_correction.'))


def render(model,data,staged,perturb=False):
    return model.render(data['rays_o_lidar'],data['rays_d_lidar'],data['time'],
                        staged=staged,perturb=perturb,num_steps=768,max_ray_batch=512)


def evaluate(model,opt,folder,step):
    import numpy as np
    import torch
    from scripts.improvement_worker import dataset, evaluate as original_evaluate
    from utils.refiner_input import refiner_input
    from audit_stage1 import decompose,aggregate
    ds=dataset(opt,'val'); predictions=[]; targets=[]; probabilities=[]
    model.eval()
    for j,fid in enumerate(IDS):
        data=ds.collate([j]); h,w=data['H_lidar'],data['W_lidar']
        with torch.no_grad(),torch.autocast('cuda',enabled=opt['fp16']):
            output=render(model,data,True)
            original=refiner_input(output,h,w)
            probability=model.unet(original).float().cpu()
            attributes=output['image_lidar'].reshape(1,h,w,2).permute(0,3,1,2)
            prediction=torch.cat((attributes,original[:,2:]),1).float().cpu()
        previous=torch.load(AUDIT/'cuda_replay'/f'{fid}_raw.pt',map_location='cpu',weights_only=True)
        assert torch.equal(original.float().cpu(),previous['inputs']),(step,fid,'refiner_input')
        assert torch.equal(probability,previous['probability']),(step,fid,'probability')
        assert torch.equal(prediction[:,[0,2]],previous['inputs'][:,[0,2]]),(step,fid,'raw_ray_depth')
        if step==0:
            assert torch.equal(prediction,previous['inputs']),(step,fid,'zero-correction')
        target=data['images_lidar'].permute(0,3,1,2).float().cpu()
        assert torch.equal(target,previous['target'])
        predictions.append(prediction); targets.append(target); probabilities.append(probability)
    metrics=original_evaluate(torch.cat(predictions),torch.cat(targets),torch.cat(probabilities),
                              IDS,opt,folder,'T0/T1 development val; original field and original refiner input preserved')
    rows=[decompose(dict(np.load(folder/f'{fid}.npz')),fid) for fid in IDS]
    result=dict(step=step,metrics=metrics,partition_summary=aggregate(rows),partitions=rows,
                original_refiner_input_probability_mask_depth_bitwise_equal=True)
    write_json(folder/'audit.json',result)
    print(json.dumps(dict(evaluation_step=step,folder=str(folder),summary=result['partition_summary'])),flush=True)
    return result


def loss_fn(model,data,opt):
    import torch
    with torch.autocast('cuda',enabled=opt['fp16']):
        out=render(model,data,False,True)
        gt=data['images_lidar'].float()
        loss=opt['alpha_i']*((out['image_lidar'][...,1].float()-gt[...,1])*gt[...,0]).square().sum()
    if not loss.requires_grad:
        loss=loss+sum(p.sum()*0 for p in model.parameters() if p.requires_grad)
    return loss,out


def preflight():
    import numpy as np
    import torch
    from scripts.improvement_worker import dataset
    from best_core.intensity_coefficients import preserve_diagnostics
    opt,alignment,state=inputs()
    assert json.loads((AUDIT/'cuda_replay_audit.json').read_text())['exact_saved_reference_replay']
    saved=json.loads((AUDIT/'saved_array_audit.json').read_text())['models']
    assert all(v['summary']['tp_sse']>saved['reference']['summary']['tp_sse'] for k,v in saved.items() if k!='reference')
    write_json(RUN/'data_alignment.json',alignment); write_json(RUN/'source_args.json',opt)
    write_json(RUN/'spec.json',SPEC)
    results={}; initial=None
    for arm in ('T0','T1'):
        model=make_model(opt,state,arm)
        current={k:v.detach().cpu().clone() for k,v in model.intensity_correction.state_dict().items()}
        if initial is None:
            initial=current
        else:
            assert set(initial)==set(current)
            assert all(torch.equal(v,current[k]) for k,v in initial.items())
        bank=model.intensity_correction.bank
        write_json(RUN/f'{arm}_parameter_mapping.json',bank.mapping)
        for row in bank.mapping:
            value=bank.coefficients[row['name']]
            assert torch.equal(value.detach().cpu(),state['model'][row['source']][0:1])
        # Query identity on real normalized points, and nonpersistent state isolation.
        torch.manual_seed(91)
        xyz=torch.rand(512,3,device='cuda'); t=torch.tensor([[.2]],device='cuda')
        with torch.no_grad(),torch.autocast('cuda',enabled=opt['fp16']):
            with preserve_diagnostics(model.scene_field):
                original=model.scene_field.fusion(model.scene_field.query_features(xyz,t))
            before={n:v.clone() for n,v in model.scene_field.named_buffers()}
            dedicated=bank(model.scene_field,xyz,t)
            assert torch.equal(original,dedicated),float((original-dedicated).abs().max())
            assert all(torch.equal(before[n],v) for n,v in model.scene_field.named_buffers())
        evaluate(model,opt,RUN/'preflight'/arm/'step0000',0)
        ds=dataset(opt,'train'); ds.num_rays_lidar=128; ds.patch_size_lidar=1
        torch.manual_seed(0); data=ds.collate([0])
        mutable=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.Adam(mutable,lr=.001,eps=1e-15)
        scaler=torch.amp.GradScaler('cuda',enabled=opt['fp16'],init_scale=1)
        gradients=[]
        anchor=None
        for step in range(4):
            optimizer.zero_grad(set_to_none=True)
            # Same perturbation for isolation comparison, not the formal ray stream.
            torch.manual_seed(103)
            loss,out=loss_fn(model,data,opt)
            observed={k:out[k].detach().cpu() for k in ('weights','z_vals','depth_lidar','image_lidar_reference')}
            if anchor is None: anchor=observed
            else: assert all(torch.equal(anchor[k],v) for k,v in observed.items())
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            assert torch.isfinite(loss)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in mutable)
            row=dict(step=step+1,loss=float(loss.detach()),
                     decoder_gradient_l1=sum(float(p.grad.abs().sum()) for p in model.intensity_correction.decoder.parameters() if p.grad is not None),
                     bank_gradient_l1=sum(float(p.grad.abs().sum()) for p in bank.parameters() if p.grad is not None),
                     upstream_decoder_gradient_l1=sum(float(p.grad.abs().sum()) for p in model.intensity_correction.decoder[:4].parameters() if p.grad is not None))
            assert row['decoder_gradient_l1']>0
            if step==0: assert row['bank_gradient_l1']==row['upstream_decoder_gradient_l1']==0
            if arm=='T1' and step>0: assert row['bank_gradient_l1']>0
            scaler.step(optimizer); scaler.update()
            gradients.append(row)
        frozen_equal(model,state)
        changes={k:float((v.detach().cpu()-initial[k]).abs().max()) for k,v in model.intensity_correction.state_dict().items()}
        assert all(v==0 for k,v in changes.items() if k.startswith('bank.')) if arm=='T0' else any(v>0 for k,v in changes.items() if k.startswith('bank.'))
        results[arm]=dict(gradients=gradients,trainable_parameters=sum(p.numel() for p in mutable),
                          zero_step_full_validation_identity=True,field_and_buffers_unchanged=True,
                          weights_samples_raw_depth_reference_unchanged_after_updates=True,
                          maximum_parameter_changes=changes,smoke_weights_discarded=True)
        print(json.dumps(dict(preflight_arm=arm,gradients=gradients)),flush=True)
        del model,bank,mutable,optimizer,scaler,data,ds,out,loss
        gc.collect();torch.cuda.empty_cache()
    write_json(RUN/'preflight.json',dict(passed=True,arms=results,identical_initialization=True))


def train_probe(model,ds,opt):
    import torch
    rows=[]; old_rays,old_patch=ds.num_rays_lidar,ds.patch_size_lidar
    ds.num_rays_lidar=1024;ds.patch_size_lidar=1
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(700)
        for index in range(4):
            data=ds.collate([index])
            with torch.no_grad(),torch.autocast('cuda',enabled=opt['fp16']):
                output=render(model,data,True,False)
            gt=data['images_lidar'].float(); mask=gt[...,0]>.5
            error=(output['image_lidar'][...,1].float()-gt[...,1]).square()
            rows.append(dict(frame_id=int(ds.frame_ids[index]),gt_valid_count=int(mask.sum()),
                             gt_valid_sse=float(error[mask].sum()),gt_valid_rmse=float(error[mask].mean().sqrt())))
    ds.num_rays_lidar,ds.patch_size_lidar=old_rays,old_patch
    return dict(scope='fixed 1024 sampled training rays for each of frames 4950-4953; unmasked GT-positive intensity, no perturbation',
                rows=rows,rmse=(sum(r['gt_valid_sse'] for r in rows)/sum(r['gt_valid_count'] for r in rows))**.5)


def run_arm(arm):
    import numpy as np
    import torch
    from torch_ema import ExponentialMovingAverage
    from scripts.improvement_worker import dataset
    import data.kitti360_dataset as dataset_module
    assert json.loads((RUN/'preflight.json').read_text())['passed']
    assert json.loads((RUN/'spec.json').read_text())==SPEC
    out=RUN/'runs'/arm
    if out.exists(): raise FileExistsError(f'no implicit restart: {out}')
    out.mkdir(parents=True)
    write_json(out/'status.json',dict(state='running',started_at=datetime.now(timezone.utc).isoformat()))
    opt,alignment,state=inputs();model=make_model(opt,state,arm)
    initial={k:v.detach().cpu().clone() for k,v in model.intensity_correction.state_dict().items()}
    torch.save(initial,out/'initial_correction.pth')
    mutable=[p for p in model.parameters() if p.requires_grad]
    write_json(out/'spec.json',dict(SPEC,arm=arm,source_checkpoint=alignment['checkpoint'],
                                   trainable_parameters=sum(p.numel() for p in mutable),
                                   trainable_coefficients=arm=='T1'))
    write_json(out/'data_alignment.json',alignment)
    optimizer=torch.optim.Adam(mutable,lr=.001,eps=1e-15)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:.1**min(s/3000,1))
    scaler=torch.amp.GradScaler('cuda',enabled=opt['fp16'],init_scale=1)
    ema=ExponentialMovingAverage(mutable,decay=.95)
    ds=dataset(opt,'train');ds.num_rays_lidar=128
    # Record actual sampled ray indices, not a digest of the sample stream.
    original_rays=dataset_module.get_lidar_rays; last_indices=[]
    def tracked_rays(*args,**kwargs):
        result=original_rays(*args,**kwargs)
        last_indices[:] = [result['inds'].detach().cpu().numpy().copy()]
        return result
    dataset_module.get_lidar_rays=tracked_rays
    indices=[];frames=[];perturbation_samples=[];cuda_rng_states=[];curves=[]
    curves.append(dict(step=0,val=evaluate(model,opt,out/'step0000',0),train_probe=train_probe(model,ds,opt)))
    torch.manual_seed(0);rng=np.random.default_rng(0);counter=[0]
    def counted_update(*args,**kwargs):counter[0]+=1
    hook=optimizer.register_step_post_hook(counted_update)
    started=time.monotonic()
    with (out/'optimizer_updates.jsonl').open('w',buffering=1) as log:
        for step in range(3000):
            if step%47==0:order=rng.permutation(47)
            ds.patch_size_lidar=[2,8] if (step//47+1)%2==0 else 1
            index=int(order[step%47]);data=ds.collate([index])
            indices.append(last_indices[0].copy());frames.append(int(ds.frame_ids[index]))
            cuda_rng_states.append(torch.cuda.get_rng_state().numpy().copy())
            model.eval();model.intensity_correction.train()
            optimizer.zero_grad(set_to_none=True)
            loss,output=loss_fn(model,data,opt)
            perturbation_samples.append(output['z_vals'][0,:8].detach().cpu().numpy().copy())
            assert torch.isfinite(loss),(arm,step,'loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in mutable),(arm,step,'gradients')
            decoder_grad=sum(float(p.grad.abs().sum()) for p in model.intensity_correction.decoder.parameters())
            bank_grad=sum(float(p.grad.abs().sum()) for p in model.intensity_correction.bank.parameters() if p.grad is not None)
            scale_before=scaler.get_scale();scaler.step(optimizer);scaler.update()
            assert counter[0]==step+1,(arm,step,'skipped_optimizer_update')
            states=[int(optimizer.state[p]['step']) for p in mutable]
            assert min(states)==max(states)==step+1,(arm,step,'Adam_step_counters')
            row=dict(step=step+1,frame_id=frames[-1],loss=float(loss.detach()),
                     optimizer_updates=counter[0],adam_step_min=min(states),adam_step_max=max(states),
                     lr=optimizer.param_groups[0]['lr'],amp_scale_before=scale_before,amp_scale_after=scaler.get_scale(),
                     decoder_gradient_l1=decoder_grad,bank_gradient_l1=bank_grad,
                     seconds=time.monotonic()-started)
            log.write(json.dumps(row,allow_nan=False)+'\n')
            scheduler.step();ema.update()
            if step%100==0 or step==2999:
                write_json(out/'progress.json',row);print(json.dumps(dict(arm=arm,**row)),flush=True)
            if (step+1)%1000==0:
                frozen_equal(model,state)
                checkpoint_dir=out/f'step{step+1:04d}'
                checkpoint_dir.mkdir()
                torch.save(dict(correction=model.intensity_correction.state_dict(),optimizer=optimizer.state_dict(),
                                scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),ema=ema.state_dict(),
                                step=step+1,optimizer_updates=counter[0],spec=SPEC,arm=arm),checkpoint_dir/'training_state.pth')
                with ema.average_parameters():
                    model.eval()
                    torch.save(dict(correction=model.intensity_correction.state_dict(),step=step+1,arm=arm,
                                    source_checkpoint=alignment['checkpoint'],spec=SPEC),checkpoint_dir/'correction_ema.pth')
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                        val=evaluate(model,opt,checkpoint_dir,step+1)
                        probe=train_probe(model,ds,opt)
                    curves.append(dict(step=step+1,val=val,train_probe=probe))
                write_json(out/'curves.json',curves)
                frozen_equal(model,state)
                print(json.dumps(dict(arm=arm,step=step+1,train_probe=probe)),flush=True)
    hook.remove();dataset_module.get_lidar_rays=original_rays
    np.savez_compressed(out/'actual_sampling.npz',frame_ids=np.asarray(frames),ray_indices=np.stack(indices),
                        first_ray_first_eight_z=np.stack(perturbation_samples),cuda_rng_states_before_render=np.stack(cuda_rng_states))
    final={k:v.detach().cpu() for k,v in model.intensity_correction.state_dict().items()}
    bank_unchanged=all(torch.equal(initial[k],v) for k,v in final.items() if k.startswith('bank.'))
    assert bank_unchanged if arm=='T0' else not bank_unchanged
    result=dict(state='completed',arm=arm,steps=3000,verified_optimizer_updates=counter[0],
                bank_unchanged=bank_unchanged,original_state_unchanged=True,curves=curves,
                elapsed_seconds=time.monotonic()-started,test_executed=False)
    write_json(out/'result.json',result);write_json(out/'status.json',dict(state='completed'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['preflight','T0','T1']);args=parser.parse_args()
    with gpu_slot(SPEC):
        if args.phase=='preflight':preflight()
        else:
            try:run_arm(args.phase)
            except BaseException as error:
                write_json(RUN/'runs'/args.phase/'failure.json',dict(state='failed',error=str(error)))
                write_json(RUN/'runs'/args.phase/'status.json',dict(state='failed',error=str(error)))
                raise


if __name__=='__main__':main()
