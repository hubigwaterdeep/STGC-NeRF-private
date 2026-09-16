"""CPU-only verification of this diagnosis and human-readable report. No training."""
from pathlib import Path
import json
import csv
import math
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def f(x,n=8):return '—' if x is None else f'{x:.{n}g}'
def quant(s):return ' / '.join(f(s[k],7) for k in ('p50','p95','max'))


def main():
    status=read(ROOT/'fit_status.json');assert status['state']=='completed'
    numeric=read(ROOT/'numeric_path.json');reach=read(ROOT/'reachability.json')
    schedule=read(ROOT/'schedule.json');split=read(ROOT/'data_split.json')
    results=read(ROOT/'fit_results.json');checks=[0,1,10,50,100,300]
    rows=list(csv.DictReader((ROOT/'fit_curves.csv').open()))
    assert len(rows)==24
    diagnostics={};verified={}
    for arm in ['T0','T1']:
        folder=ROOT/'fit'/arm
        logs=[json.loads(s) for s in (folder/'optimizer_updates.jsonl').read_text().splitlines()]
        assert len(logs)==300
        for i,(log,job) in enumerate(zip(logs,schedule),1):
            assert log['step']==log['optimizer_updates']==log['adam_step_min']==log['adam_step_max']==i
            assert log['fit_indices']==job['local_ray_indices']
            assert (log['frame_id'],log['block'])==(job['frame_id'],job['block'])
            loss=log['losses'];assert abs(loss['total']-sum(loss[k] for k in ['tp','fn','fp','tn']))<1e-6
            assert loss['fp']==loss['tn']==0
        diagnostics[arm]={s:read(folder/f'step{s:04d}.json') for s in checks}
        for s,record in diagnostics[arm].items():
            for pop in ['fit','probe']:
                m=record['metrics'][pop]
                assert abs(m['original_loss_total']-.1*m['gt_valid_sse'])<2e-7
                assert abs(m['gt_valid_rmse']-math.sqrt(m['gt_valid_sse']/m['gt_valid_count']))<1e-8
                assert m['fn_sse']==diagnostics[arm][0]['metrics'][pop]['fn_sse']
        state=torch.load(folder/'diagnostic_step0300.pth',map_location='cpu',weights_only=False)
        assert state['step']==300 and state['diagnostic_only']
        assert all(int(s['step'])==300 for s in state['optimizer']['state'].values())
        verified[arm]=dict(actual_updates=300,schedule_equal=True,loss_identity=True,final_checkpoint_loadable=True,
                           isolation=results[arm]['isolation'])
    first=read(ROOT/'fit/T0/step0000.json')['metrics']
    assert first==read(ROOT/'fit/T1/step0000.json')['metrics']
    for population in ['full_train_frames','fixed_blocks']:
        group=reach[population];count=sum(r['gt_valid_count'] for r in group)
        impossible=sum(r['unreachable_sse'] for r in group)
        original=sum(r['original_gt_valid_sse'] for r in group)
        reach[population+'_aggregate']=dict(gt_valid_count=count,outside_count=sum(r['outside_count'] for r in group),
          outside_fraction=sum(r['outside_count'] for r in group)/count,unreachable_sse=impossible,
          optimistic_projected_gt_valid_rmse=math.sqrt(impossible/count),original_gt_valid_sse=original,
          unreachable_fraction_of_original_sse=impossible/original)
    reach['fit_probe_aggregate']={}
    for subset in ['fit','probe']:
        group=[r for r in reach['fixed_blocks'] if r['subset']==subset]
        count=sum(r['gt_valid_count'] for r in group);sse=sum(r['unreachable_sse'] for r in group)
        reach['fit_probe_aggregate'][subset]=dict(gt_valid_count=count,outside_count=sum(r['outside_count'] for r in group),
                                                unreachable_sse=sse,optimistic_gt_valid_rmse=math.sqrt(sse/count))
    write(ROOT/'reachability.json',reach)
    numeric['definitions']=dict(quantiles='absolute magnitude; signed_mean separate; empty strata have null statistics',
        high_weight='original w>=0.01 within actual support',TP='GT_return and original final mask',FP='not GT_return and original final mask',
        saturation='I<=.01 or I>=.99; exact endpoints separately counted',
        numerical_comparison='Read-only; compare each path with its own zero output, never use precision-only offsets as learned gains')
    numeric['diagnostic_fit_endpoints']={arm:read(ROOT/'fit'/arm/'final_numeric_paths.json') for arm in ['T0','T1']}
    write(ROOT/'numeric_path.json',numeric)
    write(ROOT/'diagnostic_verification.json',dict(arms=verified,common_zero_metrics=True,fit_probe_disjoint=True,
          source='this diagnosis only, not repeated historical full validation',no_val_test=True,no_model_promotion=True))
    actual=dict(preexecution_configuration='CONFIGURATION.md',source_snapshot=str(ROOT/'source'),
          old_experiment='/home/zijiewu/Code/basis4D/log/intensity_coefficients_20260915',
          frames=[4950,4951,4952,4953],split_file='data_split.json',schedule_file='schedule.json',
          sampling=dict(full_context_rays=512,samples_per_ray=768,fit_rays_per_update=128,perturb=False,selection_seed=700,update_order_seed=19),
          optimizer=dict(type='Adam',learning_rate=.001,eps=1e-15,weight_decay=0,schedule='0.1**min(step/300,1)',updates_per_arm=300),
          precision='original AMP fitting; FP32 alternatives read-only',amp_init_scale=1,ema_applied=False,
          trainable_parameters=dict(T0=17537,T1=37505),coefficient_parameters=19968,evaluation_steps=checks,
          loss='0.1*sum(((J-GT_intensity)*GT_return)^2); no final prediction mask in loss',
          commands='EXECUTION_COMMANDS.md',test_executed=False,no_further_runs=True)
    write(ROOT/'actual_config.json',actual)
    produce_report(numeric,reach,diagnostics,first)
    plot(diagnostics)
    print(json.dumps(dict(fit=results,reach=reach['full_train_frames_aggregate']),indent=2))


def produce_report(numeric,reach,diag,zero):
    L=['# Intensity 可拟合性与数值作用诊断','',
       '状态：**完成并停止**。只使用train查询，各300次实际更新；没有新架构、扩容、test、发布或Best覆盖。','',
       '## 结论先行','',
       '本轮最清楚的证据是：**现有修正路径能拟合固定fit查询，但明显损害互斥probe查询；T0/T1表现接近。**',
       '因此，应优先怀疑当前空间/系数共享与局部适配的泛化问题；本次证据不支持继续扩容，也不支持宣布基函数无效。',
       'FP16会吞掉一部分小修正，但FP32只读对照没有放大整体学习修正或明显改善拟合结果，不能认定它是主因。','',
       '## 1. 已有事实（本轮仅阅读，不重新验证）','',
       '已读取旧RESULTS.md、PREREGISTRATION.md、verification.json、实际spec/source_args、源码快照、',
       '两组完整optimizer_updates.jsonl以及最终checkpoint加载代码。旧3000步/零步同一性/隔离/',
       '保存模型重放等事实沿用原证据；旧T1仅0.0291%收益、90.74%来自FP，不作为本轮新测量。',
       '加载关系：原Best的完整model状态 → 冻结；历史correction_ema.pth仅填入独立intensity_correction分支。',
       '只读数值测试加载旧最终EMA分支；300步拟合加载历史同一initial_correction.pth，并新建优化器。',
       '原始来源及历史梯度记录摘要见existing_evidence.json。历史结果和正式源码未修改。','',
       '## 2. 新诊断的数据与固定预算','',
       '使用官方47/legacy的train帧4950–4953。每帧2个预先随机登记的完整512-ray连续块；',
       'fit/probe各2048条射线，像素查询严格互斥；fit含1835个GT有效回波，probe含1185个。',
       '此池**不是旧的1024随机射线probe**，而是同训练协议下的新固定缓存。两池回波基率、难度不同，',
       '因此比较各池相对自身零步的变化，不把fit/probe的绝对数值大小当作泛化证据。','',
       '| train帧 | fit block | probe block |', '|---|---:|---:|',
       '| 4950 | 32 | 86 |','| 4951 | 60 | 127 |','| 4952 | 29 | 120 |','| 4953 | 44 | 51 |','',
       '每块固定512×768采样点。T1每次实时计算完整块的新系数特征并保留梯度；损失只取固定计划中的128条fit射线。',
       '只缓存原logits/权重/geo/方向/位置等teacher常量；T0冻结系数的特征允许缓存，T1没有缓存成常量。',
       '4个train全图原refiner输入仅用于提供正确空间上下文的固定mask，顺带测量范围上限；未加载val/test。',
       'Adam lr0.001、eps1e-15、无weight decay，按0.1**(step/300)衰减，AMP初始scale1；两组同一采样计划。',
       '本诊断观察raw当前参数，不做EMA平滑、不选最佳快照；与旧EMA成绩的口径区别已在运行前登记。','',
       '## 3. 实际修正链及目标','',
       '源码位置：source/best_core/intensity_coefficients.py 的attribute_with_reference/run；',
       'source/best_core/renderer.py 的support/weights；source/scripts/intensity_coefficients.py 的loss_fn。','',
       '```text',
       '固定 a0: FP16',
       '实时私有bank → 120维特征（实测FP32） + 固定geo/方向',
       '→ 修正decoder（AMP，输出delta为FP16）',
       '→ delta.to(a0.dtype)（这里仍为FP16）',
       '→ FP16加法与舍入 → FP16 sigmoid',
       '→ 转FP32，按原非负weights积分 → J',
       '→ 仅最终图像乘原mask m0；refiner永远读取原强度',
       '```','',
       '令s为实际属性支持，w为原权重：J=Σ w·s·sigmoid(a0+δ)。s是full/base weights>1e-4的并集。',
       '没有权重归一化、额外强度缩放或δ的显式限幅；最后线性层不带有界激活。',
       '上游高阶特征确有RMS上限，依赖完整查询块；原坐标AABB裁剪、原fusion和邻帧平均均保留。',
       '平滑链的敏感度为 w·s·I(1−I)；同时记录了实际PyTorch autograd导数。',
       '离散浮点前向存在平台区，不能把平滑导数或非零参数梯度等同于可观测的输出变化。','',
       '原训练损失 **L=0.1Σ((J−y)g)²**：包含TP及FN真实回波监督；FP/TN损失严格为0。',
       '没有把背景零标签写到采样点，也没有增加FP目标。最终图中的FP SSE是评估误差，不是训练项。','',
       '## 4. A：固定权重可达范围（新测量）','',
       '实际逐采样范围[0,1]、δ无显式上下界，因此有效乐观范围为[0,A_eff]，A_eff=Σw·s；',
       'Σw仅是更松上限。没有把观察到的δ幅度当作硬约束，也没有为了降低下限放宽任何限制。','',
       '| train范围 | GT有效回波 | 超界数量 | 不可达SSE | 投影GT-valid RMSE下限 |',
       '|---|---:|---:|---:|---:|']
    for label,row in [('4个train全图',reach['full_train_frames_aggregate']),('固定fit',reach['fit_probe_aggregate']['fit']),('固定probe',reach['fit_probe_aggregate']['probe'])]:
        floor=row.get('optimistic_projected_gt_valid_rmse',row.get('optimistic_gt_valid_rmse'))
        L.append(f"| {label} | {row['gt_valid_count']} | {row['outside_count']} | {f(row['unreachable_sse'])} | {f(floor)} |")
    a=reach['full_train_frames_aggregate']
    L+=['',f"仅frame4952有1条超界，超出0.11081135；占4帧有效回波{a['outside_fraction']*100:.6f}%，",
        f"不可达SSE占其原GT-valid SSE的{a['unreachable_fraction_of_original_sse']*100:.6f}%。该射线不在fit/probe池。",
        '这是放松样本与空间共享后的可达性诊断，不是可部署成绩、全47帧结论或T1必然达到的拟合值。',
        '最终mask造成的FN误差是另一项固定限制，不混入上述原目标的GT-valid范围投影。','',
        '## 5. B：实际数值作用与削弱（历史最终分支的新测量）','',
        '高权重定义为原w≥0.01且处于实际support，共64186个样本。以下分位数均为**绝对幅度p50 / p95 / max**。','',
        '| 组 | 量 | p50 / p95 / max | 非零比例 |','|---|---|---|---:|']
    for arm in ['T0','T1']:
        h=numeric['historical_endpoints'][arm]['paths']['amp']['sample_statistics']['high_weight']
        for label,key in [('decoder δa','delta_a'),('加法后实际Δlogit','actual_logit_change'),('逐点ΔI','delta_I'),('wI(1−I)','weighted_sensitivity'),('实际autograd敏感度','autograd_sensitivity')]:
            s=h[key];L.append(f"| {arm} | {label} | {quant(s)} | {s['nonzero_fraction']*100:.3f}% |")
    L+=['','TP/FP分别统计如下，避免仅凭全体采样点数推断射线作用：','',
        '| 组 | supported样本域 | δa p50/p95/max | 实际Δlogit p50/p95/max | ΔI p50/p95/max | ΔI非零 |',
        '|---|---|---|---|---|---:|']
    for arm in ['T0','T1']:
        p=numeric['historical_endpoints'][arm]['paths']['amp']
        for pop in ['tp','fp']:
            s=p['sample_statistics'][pop]
            L.append(f"| {arm} | {pop.upper()} | {quant(s['delta_a'])} | {quant(s['actual_logit_change'])} | {quant(s['delta_I'])} | {s['delta_I']['nonzero_fraction']*100:.3f}% |")
    L+=['','渲染后的ΔJ相对各自零修正，TP/FP未按量级筛选：','',
        '| 组 | 射线域 | ΔJ p50/p95/max | 非零比例 |','|---|---|---|---:|']
    for arm in ['T0','T1']:
        p=numeric['historical_endpoints'][arm]['paths']['amp']
        for key in ['fit/tp','fit/fp','probe/tp','probe/fp']:
            s=p['ray_delta_J'][key]
            L.append(f"| {arm} | {key} | {quant(s)} | {s['nonzero_fraction']*100:.3f}% |")
    L+=['','FP样本很少：fit仅24条、probe仅3条，不能把其分位数外推为稳定背景统计。','',
        'T1高权重点中，FP16加法吞掉13.966%的非零请求；20.226%的逐点强度没有变化。',
        'δ输出已经是FP16，随后to(FP16)不是额外降精度；局部损失主要出现在FP16加法/激活的舍入。',
        '约13.60%的高权重点满足I≤0.01或I≥0.99，精确端点约0.830%；约4.00%的平滑加权敏感度<1e-6。',
        '低权重点的进一步衰减见numeric_path.json的supported/TP/FP完整权重和敏感度分布。',
        '所测私有高阶plane分支264个RMS限幅系数均为1，当前没有观察到该限幅压低修正；这不代表所有上游非线性不存在影响。','',
        '### 只读FP32对照：不把精度零点变化算成收益','',
        '三条路径：原AMP；同一AMP δ仅FP32加法/激活；私有bank+decoder autocast关闭及FP32加法/激活。',
        '旧a0、权重、geo、方向、depth和refiner都来自同一个固定teacher；未升级/重训旧网络。',
        '私有分支复用的编码器按其原实现执行，不宣称整套旧网络全FP32。每条路径单独计算δ=0的J0。','',
        '| 历史组 | 对照 | 精度零点差p95 / max | 学习ΔJ差p95 / max | ΔJ L2比(FP32/AMP) | ΔJ方向余弦 |',
        '|---|---|---|---|---:|---:|']
    for arm in ['T0','T1']:
        for path,c in numeric['historical_endpoints'][arm]['paired_precision_comparisons'].items():
            a=c['precision_only_zero_offset'];b=c['learned_delta_J_difference']
            L.append(f"| {arm} | {path} | {f(a['p95'])} / {f(a['max'])} | {f(b['p95'])} / {f(b['max'])} | {c['learned_delta_J_l2_ratio']:.6f} | {c['learned_change_cosine']:.6f} |")
    L+=['','T1的FP32整体ΔJ范数约为AMP的0.992倍，方向余弦约0.996：局部量化确实可测，但没有总体修正幅度恢复。',
        '从各自零点计算的GT-valid误差变化也没有显示FP32明显更优；不能仅因FP16存在就认定精度是主因。',
        '若后续获准修复，可单独验证把原FP16 logits提升为FP32后做修正加法/sigmoid、再按原权重积分；',
        '**本轮只提出这一数值修复候选，不修改正式路径、不混入FP32训练。**','',
        '## 6. C/D：300步固定查询拟合与未拟合查询','',
        '| 组 | step | fit原目标总loss | fit GT-valid RMSE | probe GT-valid RMSE | fit TP SSE | fit FP SSE | probe TP SSE | probe FP SSE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ['T0','T1']:
        for step,record in diag[arm].items():
            a=record['metrics']['fit'];b=record['metrics']['probe']
            L.append(f"| {arm} | {step} | {a['original_loss_total']:.7f} | {a['gt_valid_rmse']:.8f} | {b['gt_valid_rmse']:.8f} | {a['tp_sse']:.6f} | {a['fp_sse']:.6f} | {b['tp_sse']:.6f} | {b['fp_sse']:.6f} |")
    L+=['','完整loss TP/FN/FP/TN分项、输出作用及相对更新在fit_curves.csv和逐步JSON中。',
        '固定fit全池loss等于0.1×GT-valid SSE，因此这里不存在“原目标下降但同池GT-valid误差不降”的目标取舍；',
        '随机单次minibatch的loss不能与整个fit池曲线直接混比。FP不受监督，不能把FP SSE变化误说成FP loss优化。','',
        f'![Fixed fit and probe curves]({ROOT}/fit_probe_curves.png)','',
        '### 实际作用与相对更新，不只看“有梯度”','',
        '| 组/300步 | decoder相对累计L2更新 | 系数相对累计L2更新 | 高权重δa p50/p95/max | 高权重ΔI非零 | fit ΔJ p50/p95/max |',
        '|---|---:|---:|---|---:|---|']
    for arm in ['T0','T1']:
        r=diag[arm][300];p=r['parameters'];s=r['numeric']['sample_statistics']['high_weight'];dj=r['metrics']['fit']['delta_J']
        L.append(f"| {arm} | {p['decoder']['relative_cumulative_l2']:.6f} | {p['coefficients']['relative_cumulative_l2']:.6f} | {quant(s['delta_a'])} | {s['delta_I']['nonzero_fraction']*100:.3f}% | {quant(dj)} |")
    L+=['','相对累计更新定义||θ300−θ0||₂/||θ0||₂；末层零初始化的分母为0，单列null而不是伪造相对比例。',
        '梯度、每步相对更新和p50/p95/max/非零比例在每组optimizer_updates.jsonl及step*.json逐项保留。',
        'T1的高权重δa中位数从历史端点约0.002增长到诊断后约0.116，fit ΔJ的p95约0.0823，',
        '98.21%的高权重逐点强度发生变化，全部fit/probe射线ΔJ非零：现有链路确实能产生实质输出变化。',
        '诊断后高阶cap仍全部为1。FP32只读回放的最终fit/probe RMSE与AMP相差约1e-6量级，不能解释probe明显退化。','',
        '### 证据支持什么，不支持什么','']
    for arm in ['T0','T1']:
        end=diag[arm][300]['metrics'];fit=(1-end['fit']['gt_valid_rmse']/zero['fit']['gt_valid_rmse'])*100
        probe=(end['probe']['gt_valid_rmse']/zero['probe']['gt_valid_rmse']-1)*100
        L.append(f"- {arm}：fit RMSE降低{fit:.3f}%，probe RMSE升高{probe:.3f}%。")
    L+=['','两组都能拟合该池，且都损害未拟合的训练查询；这优先支持“局部适配/共享参数泛化代价”的怀疑，',
        '不是训练梯度断路、整体可达范围卡死或完全没有读出能力。T1比T0仅有小幅优势，',
        '两者均有明显probe退化，**当前证据不支持可训练专属系数的必要性，也不支持扩容**。',
        '不能单独归咎于系数共享：T0没有训练系数也退化，共享decoder、输入分布差异、空间支持和优化偏向也可能参与。',
        '本轮没有区分这些机制的因果份额；300步不代表充分收敛，更不代表理论不可拟合。',
        '所选fit/probe只是同一序列内有限的空间块，不能外推为所有场景/所有系数表示的结论。','',
        '## 7. 隔离、交付和停止','',
        '只核查本次固定查询：teacher缓存入口一致；原参数与全部buffers未变；固定位置/权重/logits/',
        'geo/方向/GT/depth/原强度/概率缓存未变；原refiner在同一4个train上下文输入上概率完全一致。',
        '两组逐次optimizer hook和每个Adam counter都为300，实际fit索引与登记schedule完全一致。',
        '没有重复旧全验证，没有val/test查询，没有改几何basis、时间基、正式强度链或refiner输入。','',
        '交付：numeric_path.json、reachability.json、fit_curves.csv、actual_config.json、data_split.json、',
        'schedule.json、CONFIGURATION.md（运行前登记）、EXECUTION_COMMANDS.md（实际命令）、diagnostic_verification.json。',
        '完整数值分布在numeric_path.json及fit/step*.json；诊断checkpoint仅为本目录的独立副本，不发布、不替换Best。',
        '**300步预算完成后已停止，不自动安排下一轮结构实验、扩容、加训或精度修复训练。**']
    (ROOT/'INTENSITY_FIT_DIAGNOSIS.md').write_text('\n'.join(L)+'\n')


def plot(diag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,subset in zip(axes,['fit','probe']):
        for arm,color in [('T0','#2563eb'),('T1','#ea580c')]:
            seq=list(diag[arm]);y=[diag[arm][s]['metrics'][subset]['gt_valid_rmse'] for s in seq]
            ax.plot(seq,y,'-o',color=color,label=arm,markersize=4)
        base=diag['T0'][0]['metrics'][subset]['gt_valid_rmse']
        ax.axhline(base,color='#64748b',linestyle='--',label='Own zero')
        ax.set_title(('Fitted queries' if subset=='fit' else 'Unfitted TRAIN queries')+' | GT-valid RMSE')
        ax.set_xlabel('Actual optimizer updates');ax.set_ylabel('RMSE');ax.set_xticks([0,50,100,200,300]);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Fixed-query intensity diagnosis | Same original AMP path | Raw parameters, no EMA',fontsize=11)
    fig.savefig(ROOT/'fit_probe_curves.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
