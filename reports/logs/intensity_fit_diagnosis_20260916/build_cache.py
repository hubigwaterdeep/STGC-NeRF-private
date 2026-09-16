"""Four TRAIN frames only; exact full-context teacher and fixed query cache."""
import os
import subprocess
occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not occupied,f'GPU occupied: {occupied}'
from diagnostic_core import *
import types
from scripts.improvement_worker import dataset
from utils.refiner_input import refiner_input


def range_row(target,upper,loose,J0):
    g=target[:,0]>.5;y=target[:,1].float();excess=(y-upper).clamp_min(0)
    irreducible=excess[g].double().square().sum()
    original=(J0[g]-y[g]).double().square().sum()
    return dict(rays=len(y),gt_valid_count=int(g.sum()),outside_count=int((g&(excess>0)).sum()),
                outside_fraction=float((excess[g]>0).float().mean()),unreachable_sse=float(irreducible),
                optimistic_projected_gt_valid_rmse=float((irreducible/g.sum()).sqrt()),
                original_gt_valid_sse=float(original),unreachable_fraction_of_original_sse=float(irreducible/original),
                max_excess=float(excess[g].max()),upper=distribution(upper[g]),
                loose_outside_count=int((g&(y>loose)).sum()))


def main():
    if (ROOT/'cache').exists():raise FileExistsError('cache exists; no implicit overwrite')
    (ROOT/'cache').mkdir()
    model,opt=model_and_options();branch,initial=new_branch(model,'T0')
    inherited=read(OLD/'data_alignment.json')
    write(ROOT/'inherited_data_alignment.json',inherited)
    write(ROOT/'source_args.json',opt)
    ds=dataset(opt,'train');ds.num_rays_lidar=-1
    assert ds.frame_ids.tolist()==[f for f in range(4950,5001) if f not in (4960,4970,4980,4990)]
    old_query=model.attribute_with_reference;old_run=model.run
    state=dict(chunk=0,selected={},active=None)
    captured={};bounds=[];loose_bounds=[];splits=[];ranges=[]
    def head_hook(module,inputs,output):
        if state['active'] is not None:
            captured['a0']=output.detach().squeeze(-1).cpu().clone()
            captured['direction']=inputs[0][:,:model.view_encoder.n_output_dims].detach().cpu().clone()
    handle=model.intensity_net.register_forward_hook(head_hook)
    def query(self,x,d,mask=None,geo_feat=None,**kwargs):
        state['last_support']=mask.detach()
        if state['active'] is not None:
            captured.update(xyz_unit_full=((x+self.bound)/(2*self.bound)).detach().cpu().clone(),
                            support=mask.reshape(-1,768).detach().cpu().clone(),
                            geo=geo_feat[mask].detach().cpu().clone())
        attr,reference=old_query(x,d,mask=mask,geo_feat=geo_feat,**kwargs)
        return attr,attr[:,1]
    def run(self,*args,**kwargs):
        block=state['chunk'];state['active']=block if block in state['selected'] else None
        captured.clear()
        result=old_run(*args,**kwargs)
        weights=result['weights'];support=state['last_support'].reshape_as(weights)
        upper=(weights*support).sum(-1)
        bounds.append(upper.cpu());loose_bounds.append(weights.sum(-1).cpu())
        if state['active'] is not None:
            c=dict(captured,weights=weights.detach().cpu().clone(),z_vals=result['z_vals'].detach().cpu().clone(),
                   teacher=result['image_lidar_reference'][0].detach().cpu().clone(),
                   depth=result['depth_lidar'][0].detach().cpu().clone(),time=args[2].detach().cpu().clone(),
                   effective_upper=upper.cpu(),loose_upper=weights.sum(-1).cpu(),
                   frame_id=state['frame_id'],block=block,subset=state['selected'][block],
                   pixel_indices=torch.arange(block*512,(block+1)*512))
            state['selected_cache'][block]=c
        state['chunk']+=1
        return result
    model.attribute_with_reference=types.MethodType(query,model);model.run=types.MethodType(run,model)
    for frame_index,(fid,blocks) in enumerate(zip(FRAMES,BLOCKS)):
        state.update(chunk=0,frame_id=fid,selected=dict(zip(blocks,['fit','probe'])),selected_cache={})
        bounds.clear();loose_bounds.clear()
        data=ds.collate([frame_index]);h,w=data['H_lidar'],data['W_lidar']
        assert int(data['frame_id'][0])==fid
        with torch.no_grad(),torch.autocast('cuda',enabled=True):
            out=model.render(data['rays_o_lidar'],data['rays_d_lidar'],data['time'],staged=True,
                             perturb=False,num_steps=768,max_ray_batch=512)
            original=refiner_input(out,h,w)
            probability=model.unet(original).float().cpu().flatten()
        target=data['images_lidar'][0].reshape(-1,3).float().cpu()
        J0=original[0,1].flatten().float().cpu()
        row=range_row(target,torch.cat(bounds),torch.cat(loose_bounds),J0);row['frame_id']=fid;ranges.append(row)
        torch.save(dict(refiner_input=original.float().cpu(),probability=probability,target=target,
                        frame_id=fid),ROOT/'cache'/f'{fid}_refiner_context.pt')
        for block,c in state['selected_cache'].items():
            idx=c['pixel_indices'];c['target']=target[idx].clone();c['probability']=probability[idx].clone()
            with torch.no_grad(),torch.autocast('cuda',enabled=True):
                features=branch.bank(model.scene_field,c['xyz_unit_full'].cuda(),c['time'].cuda())
                c['bank_features_zero']=features[c['support'].cuda().flatten()].detach().cpu()
            # Check just the new fixed cache against its teacher; not full old audit.
            gpu={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in c.items()}
            with torch.no_grad():value=apply_path(branch,model,gpu,'amp',use_frozen_bank_cache=True)
            assert torch.equal(value['J'].cpu(),c['teacher'][:,1]),(fid,block,'teacher cache')
            torch.save(c,ROOT/'cache'/f'{fid}_{block}.pt')
            splits.append(dict(frame_id=fid,block=block,subset=c['subset'],pixel_indices=idx.tolist(),
                               range_statistics=range_row(c['target'],c['effective_upper'],c['loose_upper'],c['teacher'][:,1]),
                               support_count=int(c['support'].sum()),samples_in_full_context=c['support'].numel()))
        print(json.dumps(dict(cache_frame=fid,range_summary={k:v for k,v in row.items() if k!='upper'})),flush=True)
    handle.remove()
    fit={(r['frame_id'],i) for r in splits if r['subset']=='fit' for i in r['pixel_indices']}
    probe={(r['frame_id'],i) for r in splits if r['subset']=='probe' for i in r['pixel_indices']}
    assert len(fit)==len(probe)==2048 and not fit&probe
    schedule=[];rng=np.random.default_rng(19)
    for step in range(300):
        if step%4==0:order=rng.permutation(4)
        frame_index=int(order[step%4]);block=BLOCKS[frame_index][0]
        schedule.append(dict(step=step+1,frame_id=FRAMES[frame_index],block=block,
                             local_ray_indices=rng.choice(512,128,replace=False).tolist()))
    write(ROOT/'schedule.json',schedule)
    write(ROOT/'data_split.json',dict(protocol='official47/legacy train only',frames=FRAMES,selection_seed=700,
          train_manifest=inherited['splits']['train']['manifest'],blocks=splits,fit_probe_disjoint=True,
          fit_count=2048,probe_count=2048,context='complete512 rays x768 samples, no perturbation',
          new_fixed_pool_not_historical_probe=True,no_val_test_loaded=True))
    write(ROOT/'reachability.json',dict(activation='unbounded-logit correction then sigmoid, no explicit delta bound',
          interval='[0,sum(original weights*actual support)]',loose_interval='[0,sum(original weights)]',
          bounded_delta=False,tighter_delta_bound=None,scope='four TRAIN full frames and preselected fixed blocks, not all47',
          full_train_frames=ranges,fixed_blocks=[dict(frame_id=r['frame_id'],block=r['block'],subset=r['subset'],**r['range_statistics']) for r in splits],
          caveat='Independent sample/ray relaxation only; not a deployable model or evidence of attainable shared-field fitting.'))
    evidence={}
    verification=read(OLD/'verification.json')
    for arm in ['T0','T1']:
        records=[json.loads(line) for line in (OLD/'runs'/arm/'optimizer_updates.jsonl').read_text().splitlines()]
        evidence[arm]=dict(actual_spec=read(OLD/'runs'/arm/'spec.json'),previously_verified=verification['arms'][arm],
                           historical_record_count=len(records),first=records[0],last=records[-1],
                           historical_gradients={k:distribution([r[k] for r in records]) for k in ['decoder_gradient_l1','bank_gradient_l1']},
                           source_checkpoint=str(BASELINE),correction_file=str(OLD/'runs'/arm/'step3000/correction_ema.pth'))
    write(ROOT/'existing_evidence.json',dict(scope='Read previous artifacts, not a rerun of their verification',arms=evidence))
    write(ROOT/'cache_status.json',dict(completed=True,fixed_query_teacher_parity=True,no_val_test_loaded=True))

if __name__=='__main__':main()
