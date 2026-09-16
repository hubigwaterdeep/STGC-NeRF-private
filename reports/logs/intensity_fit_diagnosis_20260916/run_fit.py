"""Exactly300 diagnostic updates per arm; only fixed train caches, never val/test."""
import subprocess
occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not occupied,f'GPU occupied: {occupied}'
from diagnostic_core import *
import csv
import gc
import time


def frozen_snapshot(model):
    return dict(parameters={n:p.detach().cpu().clone() for n,p in model.named_parameters()},
                buffers={n:b.detach().cpu().clone() for n,b in model.named_buffers()})


def check_isolation(model,before,caches,cache_before):
    for n,p in model.named_parameters():
        assert torch.equal(p.detach().cpu(),before['parameters'][n]),n
        assert not p.requires_grad and p.grad is None,n
    for n,b in model.named_buffers():assert torch.equal(b.cpu(),before['buffers'][n]),n
    for c,old in zip(caches,cache_before):
        for key in old:assert torch.equal(c[key].cpu(),old[key]),(c['frame_id'],c['block'],key)
    # U-Net still sees its same original train-frame inputs; no geometry replay.
    for fid in FRAMES:
        data=torch.load(ROOT/'cache'/f'{fid}_refiner_context.pt',map_location='cuda',weights_only=True)
        with torch.no_grad(),torch.autocast('cuda',enabled=True):
            probability=model.unet(data['refiner_input']).float().flatten()
        assert torch.equal(probability,data['probability']),(fid,'refiner')
    return dict(old_parameters_buffers_unchanged=True,all_fixed_teacher_arrays_unchanged=True,
                original_refiner_input_and_probability_unchanged=True,no_val_test_or_geometry_rerender=True)


def evaluate(branch,model,caches,arm,step,folder,initial,previous=None,gradients=None):
    branch.eval()
    with torch.no_grad():
        outputs=[apply_path(branch,model,c,'amp',use_frozen_bank_cache=arm=='T0') for c in caches]
    stats=numeric_stats(caches,outputs)
    metrics={s:pool_metrics(caches,outputs,s) for s in ['fit','probe']}
    params=parameter_report(branch,initial,previous,gradients)
    result=dict(arm=arm,step=step,metrics=metrics,numeric=stats,parameters=params,parameter_view='raw, not EMA')
    write(folder/f'step{step:04d}.json',result)
    arrays={f"{c['frame_id']}_{c['block']}":dict(J=o['J'].cpu(),J0=o['J0'].cpu(),delta_J=o['delta_J'].cpu())
            for c,o in zip(caches,outputs)}
    torch.save(arrays,folder/f'predictions_step{step:04d}.pt')
    rows=[]
    for subset,m in metrics.items():
        populations=[(c,o) for c,o in zip(caches,outputs) if c['subset']==subset]
        da=distribution(torch.cat([o['delta'].detach().float().cpu() for c,o in populations]))
        di=distribution(torch.cat([o['delta_I'].detach().float().cpu() for c,o in populations]))
        dj=m['delta_J']
        row=dict(arm=arm,step=step,subset=subset,**{k:v for k,v in m.items() if not isinstance(v,dict)})
        for label,data in [('delta_a',da),('delta_I',di),('delta_J',dj)]:
            row.update({label+'_'+k:data[k] for k in ('p50','p95','max','nonzero_fraction')})
        row.update(decoder_relative_cumulative_update=params['decoder']['relative_cumulative_l2'],
                   coefficient_relative_cumulative_update=params['coefficients']['relative_cumulative_l2'])
        rows.append(row)
    print(json.dumps(dict(arm=arm,checkpoint_step=step,metrics=metrics)),flush=True)
    return result,rows


def main():
    if (ROOT/'fit_status.json').exists():raise FileExistsError('no implicit diagnostic restart')
    write(ROOT/'fit_status.json',dict(state='running',budget=300))
    model,opt=model_and_options();caches=load_cache()
    index={(c['frame_id'],c['block']):c for c in caches};schedule=read(ROOT/'schedule.json')
    old=frozen_snapshot(model)
    fixed_keys=['weights','z_vals','xyz_unit_full','a0','geo','direction','teacher','depth','probability','target','support']
    cache_before=[{k:c[k].cpu().clone() for k in fixed_keys} for c in caches]
    initial_reference=None;all_rows=[];arm_results={}
    for arm in ['T0','T1']:
        folder=ROOT/'fit'/arm;folder.mkdir(parents=True)
        branch,initial=new_branch(model,arm)
        if initial_reference is None:initial_reference=initial
        else:assert all(torch.equal(v,initial_reference[k]) for k,v in initial.items())
        mutable=[p for p in branch.parameters() if p.requires_grad]
        optimizer=torch.optim.Adam(mutable,lr=.001,eps=1e-15,weight_decay=0)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:.1**min(s/300,1))
        scaler=torch.amp.GradScaler('cuda',init_scale=1,enabled=True)
        updates=[0]
        def count_step(*args,**kwargs):updates[0]+=1
        hook=optimizer.register_step_post_hook(count_step)
        # Initial backward-only diagnostic; no optimizer state/update is created.
        first=schedule[0];c=index[first['frame_id'],first['block']]
        initial_out=apply_path(branch,model,c,'amp',use_frozen_bank_cache=arm=='T0')
        initial_loss=loss_components(initial_out['J'],c,torch.tensor(first['local_ray_indices'],device='cuda'))
        initial_loss['total'].backward()
        grads={n:p.grad.cpu().clone() if p.grad is not None else torch.zeros_like(p,device='cpu') for n,p in branch.named_parameters()}
        branch.zero_grad(set_to_none=True)
        zero,rows=evaluate(branch,model,caches,arm,0,folder,initial,initial,grads);all_rows+=rows
        results=[zero];start=time.monotonic()
        with (folder/'optimizer_updates.jsonl').open('w',buffering=1) as log:
            for job in schedule:
                step=job['step'];c=index[job['frame_id'],job['block']]
                selected=torch.tensor(job['local_ray_indices'],device='cuda')
                branch.train();optimizer.zero_grad(set_to_none=True)
                previous=tensor_state(branch)
                output=apply_path(branch,model,c,'amp',use_frozen_bank_cache=arm=='T0')
                losses=loss_components(output['J'],c,selected)
                loss=losses['total'];assert torch.isfinite(loss)
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in mutable)
                grads={n:p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p,device='cpu') for n,p in branch.named_parameters()}
                scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                assert updates[0]==step,'skipped optimizer step'
                adam=[int(optimizer.state[p]['step']) for p in mutable]
                assert min(adam)==max(adam)==step
                parameters=parameter_report(branch,initial,previous,grads)
                entry=dict(step=step,optimizer_updates=updates[0],adam_step_min=min(adam),adam_step_max=max(adam),
                           frame_id=c['frame_id'],block=c['block'],fit_indices=job['local_ray_indices'],
                           losses={k:float(v.detach()) for k,v in losses.items()},
                           lr=optimizer.param_groups[0]['lr'],amp_scale_before=scale,amp_scale_after=scaler.get_scale(),
                           parameters=parameters,seconds=time.monotonic()-start)
                log.write(json.dumps(entry,allow_nan=False)+'\n');scheduler.step()
                if step in CHECKS:
                    result,rows=evaluate(branch,model,caches,arm,step,folder,initial,previous,grads)
                    results.append(result);all_rows+=rows
                    torch.save(dict(correction=tensor_state(branch),optimizer=optimizer.state_dict(),step=step,
                                    source_checkpoint=str(BASELINE),diagnostic_only=True),folder/f'diagnostic_step{step:04d}.pth')
                if step%25==0:
                    write(folder/'progress.json',dict(step=step,updates=updates[0],loss=entry['losses'],seconds=entry['seconds']))
                    print(json.dumps(dict(arm=arm,step=step,loss=entry['losses'],seconds=entry['seconds'])),flush=True)
        hook.remove()
        isolation=check_isolation(model,old,caches,cache_before)
        arm_results[arm]=dict(updates=updates[0],isolation=isolation,final=results[-1]['metrics'],
                              parameter_view='raw, no EMA',elapsed_seconds=time.monotonic()-start)
        # Read-only numerical endpoints after fitting, no FP32 training.
        end_paths={}
        for path in ['amp','fp32_add','fp32_branch']:
            with torch.no_grad():outs=[apply_path(branch,model,c,path) for c in caches]
            end_paths[path]=dict(metrics={s:pool_metrics(caches,outs,s) for s in ['fit','probe']},numeric=numeric_stats(caches,outs))
        write(folder/'final_numeric_paths.json',end_paths)
        del branch,mutable,optimizer,scheduler,scaler,output,losses,loss,initial_out,initial_loss
        gc.collect();torch.cuda.empty_cache()
    with (ROOT/'fit_curves.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    write(ROOT/'fit_results.json',arm_results)
    write(ROOT/'fit_status.json',dict(state='completed',actual_updates_per_arm=300,no_val_test_loaded=True,
                                     stop_after_this_diagnosis=True))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'fit_failure.json',dict(error=str(error)))
        raise
