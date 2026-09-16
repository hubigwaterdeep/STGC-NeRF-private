"""Sequential, fail-stop official47 campaign with predeclared selection rules."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DEPTH = 'bce_expected_masked_depth_support_v1'
INTENSITY = 'bce_depth_intensity_risk_v1'
BCE = 'hard_target_probability_bce'


def now():
    return datetime.now(ZoneInfo('Australia/Perth')).isoformat(timespec='seconds')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def score(metrics):
    # Equal relative importance for RMSE, median absolute error and LPIPS.
    return (metrics['intensity_rmse'],metrics['intensity_medae'],metrics['intensity_lpips'])


def relative_score(candidate, reference):
    return sum(a/b for a,b in zip(score(candidate),score(reference))) / 3


def acceptable(candidate, reference, geometry=True):
    checks = {
        'intensity_rmse': candidate['intensity_rmse'] <= reference['intensity_rmse']*1.001,
        'intensity_medae': candidate['intensity_medae'] <= reference['intensity_medae']*1.005,
        'intensity_lpips': candidate['intensity_lpips'] <= reference['intensity_lpips']*1.005,
        'intensity_ssim': candidate['intensity_ssim'] >= reference['intensity_ssim']-.001,
        'intensity_psnr': candidate['intensity_psnr'] >= reference['intensity_psnr']-.01,
    }
    if geometry:
        checks.update(depth=candidate['depth_rmse'] <= reference['depth_rmse']*1.005,
                      cd=candidate['cd'] <= reference['cd']*1.01,
                      fscore=candidate['fscore'] >= reference['fscore']-.001,
                      ray_f1=candidate['ray_f1'] >= reference['ray_f1']-.001)
    return checks


def choose(candidates, reference_name, geometry=True):
    reference = candidates[reference_name]
    audit = {}
    eligible = []
    for name, metrics in candidates.items():
        checks = acceptable(metrics,reference,geometry)
        value = relative_score(metrics,reference)
        audit[name] = {'relative_score':value,'checks':checks}
        if all(checks.values()) and value < .999:
            eligible.append((value,name))
    selected = min(eligible)[1] if eligible else reference_name
    return selected,audit


def prepare(campaign):
    if campaign.exists():
        raise FileExistsError(campaign)
    baseline = ROOT/'log/stgc_best_official47_from_scratch_20260913_seed0'
    campaign.mkdir(parents=True)
    source = campaign/'source'
    source.mkdir()
    for directory in ('model','best_core','flow','data','utils','configs','scripts','tests'):
        shutil.copytree(ROOT/directory,source/directory,symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','checkpoints','tmp','*.pth'))
    for filename in ('main_ours.py','requirements.txt','AGENTS.md'):
        shutil.copy2(ROOT/filename,source/filename)
    for directory in ('.venv','.deps'):
        (source/directory).symlink_to((ROOT/directory).resolve(),target_is_directory=True)
    shutil.copy2(ROOT/'log/EXPERIMENT_POLICY.md',campaign/'EXPERIMENT_POLICY.md')
    python = str(source/'.venv/bin/python')
    specs = [
        ('00_cache',{'kind':'cache'}),
        ('R0_bce',{'kind':'refine','preset':BCE}),
        ('R1_depth',{'kind':'refine','preset':DEPTH}),
        ('R2_intensity',{'kind':'refine','preset':INTENSITY}),
        ('select_refine',{'kind':'selection'}),
        ('A0_head',{'kind':'attribute','feature':'none','gradient':0}),
        ('A1_base',{'kind':'attribute','feature':'base','gradient':0}),
        ('A2_delta',{'kind':'attribute','feature':'delta','gradient':0}),
        ('select_feature',{'kind':'selection'}),
        ('A3_gradient',{'kind':'attribute','feature':'selected','gradient':.05}),
        ('select_attribute',{'kind':'selection'}),
        ('A4_deployment_refine',{'kind':'refine','preset':'selected','checkpoint':'selected','cache':'selected_attribute'}),
        ('select_combination',{'kind':'selection'}),
        ('C0_combined_seed0',{'kind':'scratch','variant':'combined','seed':0}),
        ('C1_baseline_seed1',{'kind':'scratch','variant':'baseline','seed':1}),
        ('C1_combined_seed1',{'kind':'scratch','variant':'combined','seed':1}),
        ('C2_baseline_seed2',{'kind':'scratch','variant':'baseline','seed':2}),
        ('C2_combined_seed2',{'kind':'scratch','variant':'combined','seed':2}),
    ]
    plan = {'created_at':now(),'source':str(source),'python':python,
            'campaign':str(campaign),'baseline_run':str(baseline),
            'baseline_args':str(baseline/'resolved_args.json'),
            'baseline_checkpoint':str(baseline/'scratch/checkpoints/stgc_nerf_ep0639_refine.pth'),
            'gpu':0,'max_concurrent_jobs':1,'failure_policy':'stop, no automatic retry',
            'lock_file':str(ROOT/'log/.stgc_improvement_gpu0.lock'),
            'hash_policy':'never compute or verify any SHA or alternative file digest',
            'attribute_steps':3000,'field_steps':30000,'refine_steps':1000,
            'refine_seed':0,'threshold':.5,'num_rays':1024,'num_steps':768,
            'selection':{'score':'mean relative intensity RMSE, MedAE, LPIPS',
                         'minimum_relative_gain':.001,'max_rmse_ratio':1.001,
                         'max_medae_lpips_ratio':1.005,'max_ssim_drop':.001,
                         'max_psnr_drop':.01,'max_depth_ratio':1.005,
                         'max_cd_ratio':1.01,'max_fscore_f1_drop':.001,
                         'note':'provisional development selection, not statistical significance'},
            'jobs':[name for name,_ in specs]}
    for name,spec in specs:
        write(campaign/'jobs'/f'{name}.json',spec)
    write(campaign/'plan.json',plan)
    write(campaign/'status.json',{'state':'prepared','created_at':now(),'active_job':None,
                                'jobs':[{'name':name,'state':'pending'} for name,_ in specs]})
    (campaign/'local_env.sh').write_text(
        '#!/usr/bin/env bash\n'
        f'export LIDAR4D_REPO_ROOT={source}\n'
        f'export LIDAR4D_VENV={source/".venv"}\n'
        'export TORCH_HOME=/home/zijiewu/.cache/torch\n'
        'export CUDA_VISIBLE_DEVICES=0\n'
        'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4\n'
        'export PYTHONUNBUFFERED=1\n'
        'source /home/zijiewu/Code/LiDAR4D-basis-portable-20260904/env.sh\n'
        f'export PYTHONPATH={source/".deps"}:{source}\n')
    (campaign/'launch.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
        f'source {campaign/"local_env.sh"}\ncd {source}\n'
        f'exec {python} -u {source/"scripts/improvement_queue.py"} run --campaign {campaign}\n')
    (campaign/'EXPERIMENT.md').write_text(
        '## Material Passport\n\n- Origin Skill: academic-research-suite / experiment-agent\n'
        '- Origin Mode: run\n- Verification Status: PREPARED\n\n'
        '用户已授权按诊断方案实施并串行排队。每个 GPU 作业结束并输出指标后再运行下一项；失败停队，不自动重试。\n\n'
        '全部训练和 refine 使用官方 47 帧：4950–5000 排除 4960、4970、4980、4990。'
        '评估固定这四帧，不使用它们的 GT 训练；各任务记录实际 manifest、帧 ID、数量和归一化。\n\n'
        '严禁计算或校验 SHA 或替代哈希。仅检查模型加载、键与形状、数值、帧清单。\n\n'
        '顺序：缓存并复现基线 → R0/R1/R2 → A0/A1/A2 → 所选结构加梯度 A3 '
        '→ 所选属性重新 refine A4 → 若真实输出证实改进，进行种子 0/1/2 组合从头训练和种子 1/2 原方法配对。'
        '种子 0 原方法使用已完成的官方47基线，并在本轮重新评估。\n\n'
        'R 组均为 1000 步、同初始 U-Net、同增强、Adam .001、阈值 .5；R2 仅新增权重 1 的强度期望风险。'
        'A 组均从同一场 checkpoint 重新开始，只更新强度参数 3000 步、Adam .001 指数衰减到 .0001、EMA .95；'
        'A1/A2 都为 15→32→15 adapter，输出层为零，输入分别为基础特征和完整/基础特征差分。'
        'A3 从原起点重训相同步数，梯度权重 .05，同表面深度差阈值 .5 m；采样沿用单射线/2×8 patch 交替。\n\n'
        '选择规则在 plan.json 中预先固定，不做无限参数搜索。无受控收益则跳过昂贵的组合训练并报告原因。'
        '当前 val/test 共用且参与方案选择，结果标为开发评估；后续跨场景仍须各自官方协议。\n\n'
        '运行状态 status.json；进度 QUEUE.md；每项 runs/<名称>/train.log、progress.json、result.json。\n')
    return plan


def metrics(campaign,name):
    return read(campaign/'runs'/name/'result.json')['metrics']['mean']


def selection(campaign,name):
    path = campaign/'decisions.json'
    decisions = read(path) if path.exists() else {}
    if name == 'select_refine':
        candidates = {n:metrics(campaign,n) for n in ('R0_bce','R1_depth','R2_intensity')}
        selected,audit = choose(candidates,'R1_depth')
        decisions['refine'] = {'job':selected,'preset':read(campaign/'jobs'/f'{selected}.json')['preset'],
                               'audit':audit}
    elif name == 'select_feature':
        candidates = {n:metrics(campaign,n) for n in ('A0_head','A1_base','A2_delta')}
        selected,audit = choose(candidates,'A0_head',False)
        feature = read(campaign/'jobs'/f'{selected}.json')['feature']
        decisions['feature'] = {'job':selected,'feature':feature,'audit':audit}
        spec = read(campaign/'jobs/A3_gradient.json')
        spec['feature'] = feature
        write(campaign/'jobs/A3_gradient.json',spec)
    elif name == 'select_attribute':
        reference = decisions['feature']['job']
        candidates = {n:metrics(campaign,n) for n in (reference,'A3_gradient')}
        selected,audit = choose(candidates,reference,False)
        spec = read(campaign/'jobs'/f'{selected}.json')
        decisions['attribute'] = {'job':selected,'feature':spec['feature'],
                                  'gradient':spec.get('gradient',0),'audit':audit}
        deploy = read(campaign/'jobs/A4_deployment_refine.json')
        deploy.update(preset=decisions['refine']['preset'],feature=spec['feature'],
                      checkpoint=str(campaign/'runs'/selected/'checkpoint.pth'))
        write(campaign/'jobs/A4_deployment_refine.json',deploy)
    elif name == 'select_combination':
        reference = metrics(campaign,decisions['refine']['job'])
        deployed = metrics(campaign,'A4_deployment_refine')
        attr = decisions['attribute']
        checks = acceptable(deployed,reference)
        attr_has_change = attr['feature'] != 'none' or attr['gradient'] > 0
        use_attribute = (attr_has_change and all(checks.values())
                         and relative_score(deployed,reference) < .999)
        use_refine = decisions['refine']['preset'] != DEPTH
        decisions['combination'] = {
            'enabled':use_attribute or use_refine,
            'feature':attr['feature'] if use_attribute else 'none',
            'gradient':attr['gradient'] if use_attribute else 0,
            'preset':decisions['refine']['preset'],
            'attribute_deployment_passed':use_attribute,'deployment_checks':checks,
            'deployment_relative_score':relative_score(deployed,reference),
            'note':'no new configuration benefit means scratch jobs are skipped; A0 extra fitting is not a scratch architecture change'}
    else:
        raise ValueError(name)
    write(path,decisions)
    return decisions


def dashboard(campaign,status):
    lines = ['# 官方 47 帧改进实验串行队列','',f'状态：{status["state"]}；更新：{status.get("updated_at",now())}',
             '', '| 实验 | 状态 | 开始 | 完成 |','| --- | --- | --- | --- |']
    for row in status['jobs']:
        lines.append(f'| {row["name"]} | {row["state"]} | {row.get("started_at","")} | {row.get("finished_at","")} |')
    lines += ['', '同一时刻仅一个 GPU 作业。失败停队，不自动重试。官方 47 帧；禁止 SHA/哈希计算与校验。']
    (campaign/'QUEUE.md').write_text('\n'.join(lines)+'\n')


def run(campaign):
    plan,status = read(campaign/'plan.json'),read(campaign/'status.json')
    if status['state'] != 'prepared':
        raise RuntimeError('queue is not prepared; duplicate launch/retry refused')
    child = None
    def save():
        status['updated_at']=now()
        write(campaign/'status.json',status)
        dashboard(campaign,status)
    def interrupt(signum,frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupt)
    signal.signal(signal.SIGINT,interrupt)
    def execute(command,log,timeout_hours):
        nonlocal child
        started=time.monotonic()
        with log.open('a',buffering=1) as output:
            child=subprocess.Popen(command,cwd=plan['source'],stdin=subprocess.DEVNULL,
                                   stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            status['child_pid']=child.pid
            save()
            while True:
                try:
                    code=child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    status['active_elapsed_seconds']=round(time.monotonic()-started)
                    save()
                    if time.monotonic()-started > timeout_hours*3600:
                        raise TimeoutError(f'job exceeded {timeout_hours} hour timeout')
        child=None
        if code:
            raise RuntimeError(f'child exit {code}; see {log}')
    with open(plan['lock_file'],'a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        occupied=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if occupied:
            raise RuntimeError(f'GPU compute process already running: {occupied}')
        status.update(state='running',started_at=now(),supervisor_pid=os.getpid())
        save()
        try:
            for record in status['jobs']:
                name=record['name']
                if record['state'] == 'completed':
                    if read(campaign/'runs'/name/'result.json').get('state') != 'completed':
                        raise RuntimeError(f'completed job has no valid result: {name}')
                    continue
                spec=read(campaign/'jobs'/f'{name}.json')
                if spec['kind']=='scratch' and not read(campaign/'decisions.json')['combination']['enabled']:
                    record.update(state='skipped',reason='no controlled configuration benefit',finished_at=now())
                    save(); continue
                record.update(state='running',started_at=now())
                status.update(active_job=name,active_elapsed_seconds=0,child_pid=None)
                save()
                print(f'{now()} START {name}',flush=True)
                out=campaign/'runs'/name
                out.mkdir(parents=True,exist_ok=True)
                if spec['kind']=='selection':
                    result=selection(campaign,name)
                    write(out/'result.json',{'state':'completed','decisions':result})
                elif spec['kind']=='scratch':
                    combo=read(campaign/'decisions.json')['combination']
                    chosen=combo if spec['variant']=='combined' else {'feature':'none','gradient':0,'preset':DEPTH}
                    original=read(plan['baseline_args'])
                    command=[plan['python'],'-u',str(Path(plan['source'])/'main_ours.py'),
                             '--config',str(Path(plan['source'])/'configs/kitti360_4950_stgc_best.txt'),
                             '--workspace',str(out),'--path',original['path'],'--resume',original['resume'],
                             '--ckpt','scratch','--seed',str(spec['seed']),'--lr','.01',
                             '--iters','30000','--max_train_steps','30000',
                             '--refine_steps','1000','--refine_init_seed','0',
                             '--refine_loss_preset',chosen['preset'],
                             '--intensity_feature_mode',chosen['feature'],
                             '--alpha_intensity_gradient',str(chosen['gradient'])]
                    write(out/'command.json',command)
                    shutil.copy2(campaign/'runs/00_cache/data_alignment.json',out/'data_alignment.json')
                    execute(command,out/'train.log',12)
                    checkpoints=sorted((out/'checkpoints').glob('*_refine.pth'))
                    if not checkpoints:
                        raise RuntimeError('scratch job missing completed refine checkpoint')
                    eval_name=name+'__evaluate'
                    write(campaign/'jobs'/f'{eval_name}.json',{'kind':'evaluate',
                          'checkpoint':str(checkpoints[-1]),'feature':chosen['feature']})
                    execute([plan['python'],'-u',str(Path(plan['source'])/'scripts/improvement_worker.py'),
                             '--campaign',str(campaign),'--job',eval_name],out/'evaluation.log',1)
                    result=read(campaign/'runs'/eval_name/'result.json')
                    result.update(training_spec=spec,configuration=chosen)
                    write(out/'result.json',result)
                else:
                    command=[plan['python'],'-u',str(Path(plan['source'])/'scripts/improvement_worker.py'),
                             '--campaign',str(campaign),'--job',name]
                    write(out/'command.json',command)
                    execute(command,out/'train.log',3)
                    result=read(out/'result.json')
                    if result.get('state')!='completed':
                        raise RuntimeError('job output is incomplete')
                record.update(state='completed',finished_at=now())
                save()
                print(f'{now()} COMPLETE {name}',flush=True)
            status.update(state='completed',active_job=None,child_pid=None,finished_at=now())
            save()
        except BaseException as error:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
            for record in status['jobs']:
                if record['state']=='running':
                    record.update(state='failed',finished_at=now())
            status.update(state='failed',error=str(error),child_pid=None,finished_at=now())
            save()
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('--campaign',type=Path,required=True)
    args=parser.parse_args()
    campaign=args.campaign.resolve()
    if args.mode=='prepare':
        print(json.dumps(prepare(campaign),indent=2))
    else:
        run(campaign)
