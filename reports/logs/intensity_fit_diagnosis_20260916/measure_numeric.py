"""Read-only historical endpoints, own-zero precision controls and local gradients."""
import subprocess
occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not occupied,f'GPU occupied: {occupied}'
from diagnostic_core import *
import gc


def main():
    model,opt=model_and_options();caches=load_cache()
    report=dict(formula={
        'amp':'delta16=decoder_AMP(bank_AMP,geo0,d0); b16=round16(a0_16+cast16(delta16)); I16=sigmoid16(b16); J32=sum(w32*support*float32(I16)); final=m0*J32',
        'derivative':'Smooth-chain sensitivity w*I*(1-I) on supported samples; actual PyTorch autograd sensitivity also measured. Quantized forward map has plateaus, so smooth derivative is not a finite-change guarantee.',
        'fp32_add':'Same AMP features and AMP decoder delta; promote a0 and delta before addition/sigmoid, retain original weights and mask.',
        'fp32_branch':'Private appearance bank and decoder under autocast disabled; cached old a0/geo/direction promoted, FP32 addition and sigmoid. Intrinsically FP16 frozen tcnn outputs remain as implemented.',
        'delta_bounds':'No explicit bound/clamp on decoder correction. High-order cap constrains intermediate feature residuals, not delta logits.',
        'own_zero_rule':'Every learned J is compared with its same-path delta=0 output; zero-path offsets are reported separately.'},
        scope='Preselected TRAIN fit/probe pool only; no optimization in numerical paths',historical_endpoints={})
    schedule=read(ROOT/'schedule.json');first=schedule[0]
    for arm in ['T0','T1']:
        branch,initial=new_branch(model,arm,historical=True)
        before=tensor_state(branch)
        paths={};predictions={}
        for path in ['amp','fp32_add','fp32_branch']:
            with torch.no_grad():
                outputs=[apply_path(branch,model,c,path) for c in caches]
            stats=numeric_stats(caches,outputs)
            stats['pool_metrics']={split:pool_metrics(caches,outputs,split) for split in ['fit','probe']}
            # A backward-only measurement on the same fixed fit minibatch.
            branch.zero_grad(set_to_none=True)
            c=next(c for c in caches if c['frame_id']==first['frame_id'] and c['block']==first['block'])
            value=apply_path(branch,model,c,path)
            components=loss_components(value['J'],c,torch.tensor(first['local_ray_indices'],device='cuda'))
            components['total'].backward()
            grads={n:p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p,device='cpu')
                   for n,p in branch.named_parameters()}
            stats['parameter_and_gradient']=parameter_report(branch,initial,gradients=grads)
            stats['gradient_scope']='Backward only; first fixed128-ray fit minibatch, original objective; no optimizer step'
            stats['gradient_loss_components']={k:float(v.detach()) for k,v in components.items()}
            paths[path]=stats
            predictions[path]=dict(J=torch.cat([o['J'] for o in outputs]).cpu(),
                                   J0=torch.cat([o['J0'] for o in outputs]).cpu(),
                                   delta_J=torch.cat([o['delta_J'] for o in outputs]).cpu())
            print(json.dumps(dict(arm=arm,path=path,fit=stats['pool_metrics']['fit'],
                                  high_weight_erasure=stats['saturation_and_erasure']['high_weight'])),flush=True)
            del outputs,value,components;branch.zero_grad(set_to_none=True)
        assert all(torch.equal(before[n],v.cpu()) for n,v in branch.state_dict().items())
        comparisons={}
        for path in ['fp32_add','fp32_branch']:
            a=predictions['amp'];p=predictions[path]
            comparisons[path]=dict(precision_only_zero_offset=distribution(p['J0']-a['J0']),
                                   learned_delta_J_difference=distribution(p['delta_J']-a['delta_J']),
                                   learned_delta_J_l2_ratio=float(p['delta_J'].norm()/a['delta_J'].norm()),
                                   learned_change_cosine=float(torch.nn.functional.cosine_similarity(p['delta_J'],a['delta_J'],dim=0)))
        report['historical_endpoints'][arm]=dict(paths=paths,paired_precision_comparisons=comparisons,
                                                no_parameter_updates=True)
        (ROOT/'numeric_arrays').mkdir(exist_ok=True)
        torch.save(predictions,ROOT/'numeric_arrays'/f'{arm}_historical_paths.pt')
        del branch;gc.collect();torch.cuda.empty_cache()
    write(ROOT/'numeric_path.json',report)

if __name__=='__main__':main()
