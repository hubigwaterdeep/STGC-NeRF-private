"""Diagnostic-only fixed-query machinery. No writes to historical artifacts."""
from pathlib import Path
import json
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
OLD=Path('/home/zijiewu/Code/basis4D/log/intensity_coefficients_20260915')
sys.path[:0]=[str(ROOT/'source/.deps'),str(ROOT/'source')]
from scripts.intensity_ablation import construct_model
from best_core.intensity_coefficients import IntensityCorrection
import best_core.intensity_coefficients as coefficient_module

FRAMES=[4950,4951,4952,4953]
BLOCKS=[[32,86],[60,127],[29,120],[44,51]]
CHECKS=[0,1,10,50,100,300]
BASELINE=Path('/home/zijiewu/Code/basis4D/log/best_official47_baseline/scratch/checkpoints/stgc_nerf_ep0639_refine.pth')

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)

def read(path):return json.loads(Path(path).read_text())

def distribution(values):
    if isinstance(values,torch.Tensor):values=values.detach().float().cpu().numpy()
    a=np.asarray(values,dtype=np.float64).reshape(-1)
    if not len(a):return dict(count=0,p50=None,p95=None,max=None,nonzero_fraction=None,signed_mean=None)
    assert np.isfinite(a).all()
    ab=np.abs(a)
    return dict(count=len(a),p50=float(np.quantile(ab,.5)),p95=float(np.quantile(ab,.95)),
                max=float(ab.max()),nonzero_fraction=float(np.count_nonzero(a)/len(a)),
                signed_mean=float(a.mean()),l2=float(np.linalg.norm(a)))

def tensor_state(module):return {n:v.detach().cpu().clone() for n,v in module.state_dict().items()}

def parameter_report(branch,initial,previous=None,gradients=None):
    result={}
    for name,condition in [('decoder',lambda n:n.startswith('decoder.')),('coefficients',lambda n:n.startswith('bank.')),
                           ('last_decoder_layer',lambda n:n.startswith('decoder.4.'))]:
        selected=[(n,p) for n,p in branch.named_parameters() if condition(n)]
        current=torch.cat([p.detach().cpu().flatten() for _,p in selected])
        start=torch.cat([initial[n].flatten() for n,_ in selected])
        delta=current-start
        row=dict(parameter=distribution(current),cumulative_update=distribution(delta),
                 relative_cumulative_l2=float(delta.norm()/start.norm()) if start.norm()>0 else None,
                 initial_l2=float(start.norm()),zero_initial_norm=bool(start.norm()==0))
        if previous is not None:
            old=torch.cat([previous[n].flatten() for n,_ in selected])
            update=current-old
            row.update(last_update=distribution(update),relative_last_update_l2=float(update.norm()/old.norm()) if old.norm()>0 else None)
        if gradients is not None:
            grad=torch.cat([gradients[n].flatten() for n,_ in selected])
            row['gradient']=distribution(grad)
            row['relative_gradient_l2']=float(grad.norm()/current.norm()) if current.norm()>0 else None
        result[name]=row
    return result

def model_and_options():
    opt=read(OLD/'source_args.json')
    state=torch.load(BASELINE,map_location='cpu',weights_only=False)
    torch.manual_seed(0)
    model=construct_model(opt,state,dict(mode='none',current_weight=.5,seed=0,parameter_budget=32768))
    model.requires_grad_(False).eval()
    return model,opt

def new_branch(model,arm,*,historical=False):
    branch=IntensityCorrection(model.scene_field,model.view_encoder.n_output_dims,arm=='T1',seed=0).cuda()
    initial=torch.load(OLD/'runs'/arm/'initial_correction.pth',map_location='cpu',weights_only=True)
    if historical:
        payload=torch.load(OLD/'runs'/arm/'step3000/correction_ema.pth',map_location='cpu',weights_only=False)
        assert payload['source_checkpoint']==str(BASELINE)
        branch.load_state_dict(payload['correction'],strict=True)
    else:branch.load_state_dict(initial,strict=True)
    return branch,initial

def load_cache(device='cuda'):
    return [torch.load(ROOT/'cache'/f'{fid}_{block}.pt',map_location=device,weights_only=True)
            for fid,blocks in zip(FRAMES,BLOCKS) for block in blocks]

CAP_SCALES=[]
class ObservedQueryView(coefficient_module.QueryView):
    def _safe_high_order_residual(self,base,residual):
        # Only inspect detached values; the returned expression is the original.
        with torch.no_grad():
            base_rms=(base.float().square().mean()+1e-12).sqrt().clamp_min(1e-6)
            residual_rms=(residual.float().square().mean()+1e-12).sqrt()
            CAP_SCALES.append((.5*base_rms/residual_rms).clamp(max=1).detach())
        return super()._safe_high_order_residual(base,residual)
coefficient_module.QueryView=ObservedQueryView

def apply_path(branch,model,cache,path='amp',*,use_frozen_bank_cache=False):
    """The only trainable computation is the private branch; teacher is constant."""
    amp=path!='fp32_branch'
    CAP_SCALES.clear()
    with torch.autocast('cuda',enabled=amp):
        if use_frozen_bank_cache and path!='fp32_branch':
            features=cache['bank_features_zero']
        else:
            # NEVER restrict xyz to loss rays or attribute-support points.
            full=branch.bank(model.scene_field,cache['xyz_unit_full'],cache['time'])
            features=full[cache['support'].flatten()]
        delta=branch(features,cache['geo'],cache['direction']).squeeze(-1)
    a0=cache['a0']
    if path=='amp':
        converted=delta.to(a0.dtype)
        actual=a0+converted
        sample_i=torch.sigmoid(actual)
        zero_i=torch.sigmoid(a0)
    else:
        converted=delta.float()
        actual=a0.float()+converted
        sample_i=torch.sigmoid(actual)
        zero_i=torch.sigmoid(a0.float())
    support=cache['support']
    values=torch.zeros_like(cache['weights']).flatten()
    zeros=torch.zeros_like(cache['weights']).flatten()
    values[support.flatten()]=sample_i.float();zeros[support.flatten()]=zero_i.float()
    J=(cache['weights']*values.reshape_as(support)).sum(-1)
    J0=(cache['weights']*zeros.reshape_as(support)).sum(-1)
    return dict(delta=delta,converted=converted,actual_delta=actual.float()-a0.float(),
                intensity=sample_i,zero_intensity=zero_i,delta_I=sample_i.float()-zero_i.float(),
                J=J,J0=J0,delta_J=J-J0,features=features,
                cap_scales=torch.stack(CAP_SCALES) if CAP_SCALES else torch.empty(0,device=J.device),
                dtypes=dict(a0=str(a0.dtype),features=str(features.dtype),decoder_output=str(delta.dtype),
                            converted=str(converted.dtype),added_logit=str(actual.dtype),activation=str(sample_i.dtype),
                            weights=str(cache['weights'].dtype),integrated=str(J.dtype)))

def loss_components(J,c,selection=None):
    if selection is None:selection=torch.arange(len(J),device=J.device)
    gt=c['target'][selection];g=gt[:,0]>.5;m=c['probability'][selection]>.5
    sq=((J[selection].float()-gt[:,1].float())*gt[:,0].float()).square()
    return dict(total=.1*sq.sum(),tp=.1*sq[g&m].sum(),fn=.1*sq[g&~m].sum(),
                fp=.1*sq[~g&m].sum(),tn=.1*sq[~g&~m].sum())

def pool_metrics(caches,outputs,subset):
    groups=[(c,o) for c,o in zip(caches,outputs) if c['subset']==subset]
    gt=torch.cat([c['target'] for c,o in groups]).float()
    J=torch.cat([o['J'].detach() for c,o in groups]).float()
    J0=torch.cat([o['J0'].detach() for c,o in groups]).float()
    g=gt[:,0]>.5;m=torch.cat([c['probability'] for c,o in groups])>.5
    raw=(J-gt[:,1]).square();final=(m*J-g*gt[:,1]).square()
    result=dict(ray_count=len(J),gt_valid_count=int(g.sum()),gt_valid_sse=float(raw[g].sum()),
                gt_valid_rmse=float(raw[g].mean().sqrt()),original_loss_total=float(.1*raw[g].sum()),
                original_loss_per_ray=float(.1*raw[g].sum()/len(J)),
                final_masked_rmse=float(final.mean().sqrt()),
                delta_J=distribution(J-J0),own_zero_gt_valid_rmse=float((J0[g]-gt[g,1]).square().mean().sqrt()))
    for name,mask in dict(tp=g&m,fp=~g&m,fn=g&~m,tn=~g&~m).items():
        result[name+'_count']=int(mask.sum())
        result[name+'_sse']=float(final[mask].sum())
        result['loss_'+name]=float(.1*raw[mask].sum()) if name in ('tp','fn') else 0.
        result[name+'_rmse']=float(final[mask].mean().sqrt()) if mask.any() else None
    return result

def numeric_stats(caches,outputs):
    """All populations are explicit; statistics are absolute unless labeled mean."""
    storage={};ray_storage={};sat={};dtype_rows=[];caps=[]
    for c,o in zip(caches,outputs):
        support=c['support'];n,s=support.shape
        g=c['target'][:,0]>.5;m=c['probability']>.5
        w=c['weights'][support]
        tp=(g&m)[:,None].expand_as(support)[support];fp=(~g&m)[:,None].expand_as(support)[support]
        high=w>=.01
        strata=dict(supported=torch.ones_like(high),high_weight=high,tp=tp,fp=fp,tp_high=tp&high,fp_high=fp&high)
        I=o['intensity'].float();I0=o['zero_intensity'].float()
        stats=dict(delta_a=o['delta'].float(),cast_delta_a=o['converted'].float(),actual_logit_change=o['actual_delta'],
                   delta_I=o['delta_I'],weight=w,sigmoid_slope=I*(1-I),weighted_sensitivity=w*I*(1-I),
                   linearized_delta_J_contribution=w*I0*(1-I0)*o['delta'].float())
        with torch.enable_grad():
            requested=o['delta'].detach().clone().requires_grad_(True)
            if o['dtypes']['added_logit']=='torch.float16':
                active=torch.sigmoid(c['a0']+requested.to(c['a0'].dtype))
            else:active=torch.sigmoid(c['a0'].float()+requested.float())
            derivative=torch.autograd.grad((w*active.float()).sum(),requested)[0].float()
        stats['autograd_sensitivity']=derivative.detach()
        for name,mask in strata.items():
            target=storage.setdefault(name,{k:[] for k in stats})
            for key,value in stats.items():target[key].append(value[mask].detach().float().cpu())
            bucket=sat.setdefault(name,dict(count=0,saturated=0,exact_endpoint=0,requested_nonzero=0,add_erased=0,
                                            intensity_erased=0,sensitivity_below_1e_6=0))
            requested=o['delta']!=0
            bucket['count']+=int(mask.sum())
            bucket['saturated']+=int((mask&((I<=.01)|(I>=.99))).sum())
            bucket['exact_endpoint']+=int((mask&((I==0)|(I==1))).sum())
            bucket['requested_nonzero']+=int((mask&requested).sum())
            bucket['add_erased']+=int((mask&requested&(o['actual_delta']==0)).sum())
            bucket['intensity_erased']+=int((mask&requested&(o['delta_I']==0)).sum())
            bucket['sensitivity_below_1e_6']+=int((mask&(stats['weighted_sensitivity']<1e-6)).sum())
        for name,mask in dict(all=torch.ones_like(g),gt_valid=g,tp=g&m,fp=~g&m).items():
            key=c['subset']+'/'+name
            ray_storage.setdefault(key,[]).append(o['delta_J'][mask].detach().cpu())
        dtype_rows.append(o['dtypes']);caps.append(o['cap_scales'].detach().cpu())
    sample_stats={s:{k:distribution(torch.cat(v)) for k,v in row.items()} for s,row in storage.items()}
    for name,row in sat.items():
        for key in ('saturated','exact_endpoint','sensitivity_below_1e_6'):
            row[key+'_fraction']=row[key]/row['count'] if row['count'] else None
        for key in ('add_erased','intensity_erased'):
            row[key+'_fraction_of_nonzero_request']=row[key]/row['requested_nonzero'] if row['requested_nonzero'] else None
    cap=torch.cat(caps)
    return dict(sample_statistics=sample_stats,saturation_and_erasure=sat,
                ray_delta_J={k:distribution(torch.cat(v)) for k,v in ray_storage.items()},
                high_order_cap=dict(scales=distribution(cap),active_fraction=float((cap<1).float().mean()) if len(cap) else None),
                dtypes=dtype_rows[0])
