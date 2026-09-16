"""Real-checkpoint CUDA checks for attribute isolation before launching jobs."""
import gc
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from improvement_worker import model_from,dataset,seed

root=Path(__file__).resolve().parents[1]
baseline=root/'log/stgc_best_official47_from_scratch_20260913_seed0'
opt=json.loads((baseline/'resolved_args.json').read_text())
checkpoint=baseline/'scratch/checkpoints/stgc_nerf_ep0639_refine.pth'
ds=dataset(opt,'train')
ds.num_rays_lidar=64
seed(123)
data=ds.collate([0])

def render(model):
    with torch.autocast('cuda',enabled=True):
        return model.render(data['rays_o_lidar'],data['rays_d_lidar'],data['time'],
                            staged=False,num_steps=768,perturb=False)

model=model_from(opt,checkpoint)
model.requires_grad_(False)
with torch.no_grad():
    output=render(model)
reference={k:output[k].detach().clone() for k in ('image_lidar','depth_lidar')}
del model,output
gc.collect();torch.cuda.empty_cache()
results=[]
for feature in ('base','delta'):
    model=model_from(opt,checkpoint,feature)
    for name,param in model.named_parameters():
        param.requires_grad_(name.startswith(('intensity_net.','intensity_adapter.')))
    frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items()
            if not k.startswith(('intensity_net.','intensity_adapter.'))}
    output=render(model)
    for name in reference:
        if not torch.equal(reference[name],output[name]):
            raise RuntimeError(f'zero adapter changed {feature}/{name}')
    mutable=[p for p in model.parameters() if p.requires_grad]
    initial=[p.detach().clone() for p in mutable]
    optimizer=torch.optim.Adam(mutable,lr=.001)
    gt=data['images_lidar'].float()
    loss=((output['image_lidar'][...,1].float()-gt[...,1])*gt[...,0]).square().mean()
    loss.backward()
    if not any(p.grad is not None and p.grad.abs().max()>0 for p in mutable):
        raise RuntimeError('missing attribute gradients')
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in mutable):
        raise RuntimeError('non-finite attribute gradients')
    optimizer.step()
    if not any(not torch.equal(before,after) for before,after in zip(initial,mutable)):
        raise RuntimeError('attribute optimizer did not update weights')
    for name,value in frozen.items():
        if not torch.equal(value,model.state_dict()[name].detach().cpu()):
            raise RuntimeError(f'changed frozen parameter {name}')
    with torch.no_grad():
        after=render(model)
    if not torch.equal(after['depth_lidar'],reference['depth_lidar']):
        raise RuntimeError('attribute step changed rendered depth')
    if not torch.equal(after['image_lidar'][...,0],reference['image_lidar'][...,0]):
        raise RuntimeError('attribute step changed rendered return')
    results.append({'feature':feature,'zero_adapter_identical':True,
                    'attribute_gradient_finite':True,'attribute_weights_updated':True,
                    'frozen_parameters_unchanged':True,'depth_and_return_identical':True})
    del model,optimizer,output,after,frozen,initial,mutable
    gc.collect();torch.cuda.empty_cache()
print(json.dumps({'state':'passed','checks':results}),flush=True)
