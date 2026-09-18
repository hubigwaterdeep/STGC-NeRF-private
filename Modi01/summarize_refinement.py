"""Report standard-BCE refinement, pre/post metrics, and historical STGC context."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ['cd_m2', 'point_fscore', 'depth_rmse_m', 'intensity_rmse', 'return_rmse', 'return_accuracy', 'return_f1']
HEADER = ['| 方法 | CD ↓ (m²) | 点云 F-score ↑ | 深度 RMSE ↓ (m) | 强度 RMSE ↓ | 回波 RMSE ↓ | 回波 Acc ↑ | 回波 F1 ↑ |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']


def metric_row(name, metrics):
    return '| ' + name + ' | ' + ' | '.join('未记录' if metrics.get(key) is None else f'{metrics[key]:.6f}' for key in KEYS) + ' |'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('refinement', type=Path)
    args = parser.parse_args()
    out = args.refinement.resolve()
    state = json.loads((out / 'refinement.json').read_text())
    assert state['status'] == 'COMPLETED' and state['common_initial_unet_verified']
    records = [json.loads((Path(job['output']) / 'results.json').read_text()) for job in state['jobs']]
    statuses = [json.loads((Path(job['output']) / 'status.json').read_text()) for job in state['jobs']]
    before = [json.loads((Path(job['pre_evaluation']) / 'results.json').read_text()) for job in state['jobs']]
    assert all(r['status'] == 'COMPLETED' and all(r['checks'].values()) for r in records)
    assert all(s['optimization_steps'] == 1000 for s in statuses)
    assets = [json.loads((Path(j['workspace'])/'registration.json').read_text())['assets'] for j in state['jobs']]
    assert all(a == assets[0] for a in assets)
    for record in records:
        assert record['contract']['train_frames'] == assets[0]['splits']['train']['frame_ids']
        assert record['contract']['development_frames'] == assets[0]['splits']['val']['frame_ids']
    policy = dict(protocol='legacy', sequence_id='8120', scale=assets[0]['scale'], offset=assets[0]['offset'],
                  fov_lidar=assets[0]['fov_lidar'], splits=assets[0]['splits'], normalization='(frame_id-8120)/50',
                  refinement_supervision='train only; 47 full frames', development_use='fixed final-step evaluation only',
                  weights='30000-step final EMA field; only UNet updated', digests_computed=False)
    (out/'data_protocol.json').write_text(json.dumps(policy,indent=2)+'\n')
    policy_lines = ['# Refinement experiment policy', '',
                    'Three authorized Modi01 candidates; standard STGC BCE-only refinement, 1000 steps.', '',
                    'Final 30000-step EMA field frozen; UNet only; separate output checkpoints.', '',
                    'Official legacy KITTI-360 8120–8170; time=(frame_id-8120)/50; val/test overlap: post-hoc development.', '']
    for split, details in assets[0]['splits'].items():
        policy_lines += [f'{split}: {details["count"]} frames; IDs: {details["frame_ids"]}',
                         f'Manifest: {details["manifest"]}', '']
    policy_lines += ['No development/test training, early stopping or checkpoint selection. No model promotion.',
                     'No file digests are computed. Full normalization: ../data_protocol.json', '']
    (out/'log').mkdir(exist_ok=True)
    (out/'log/EXPERIMENT_POLICY.md').write_text('\n'.join(policy_lines))
    original = next(row for row in csv.DictReader((ROOT/'reports/data/stgc_all_scene_metrics.csv').open(encoding='utf-8-sig')) if row['scene'] == '8120')
    standard = {k: float(original[v]) for k, v in [('cd_m2', 'CD'), ('point_fscore', 'F-score(repo)'),
        ('depth_rmse_m', 'Depth RMSE'), ('intensity_rmse', 'Intensity RMSE')]}
    rows, frame_rows, deltas = [], [], []
    for record, pre, job in zip(records, before, state['jobs']):
        with (Path(job['output'])/'per_frame.csv').open() as handle:
            frame_rows.extend(csv.DictReader(handle))
        for split, group in record['groups'].items():
            assert group['frames'] == (4 if split == 'development' else 47)
            rows.append(dict(variant=record['variant'], split=split, stage='post_refiner', frames=group['frames'], **group['metrics']))
        pre_m, post_m = pre['groups']['ema_development']['metrics'], record['groups']['development']['metrics']
        deltas.append(dict(variant=record['variant'],
            vs_pre_percent={k:100*(post_m[k]/pre_m[k]-1) for k in ['cd_m2','depth_rmse_m','intensity_rmse']},
            vs_standard_percent={k:100*(post_m[k]/standard[k]-1) for k in ['cd_m2','depth_rmse_m','intensity_rmse']},
            vs_pre_point_fscore_percentage_points=100*(post_m['point_fscore']-pre_m['point_fscore']),
            vs_standard_point_fscore_percentage_points=100*(post_m['point_fscore']-standard['point_fscore']),
            vs_pre_return_f1_percentage_points=100*(post_m['return_f1']-pre_m['return_f1'])))
    assert len(frame_rows) == 153
    assert len({(r['variant'],r['split'],r['frame_id']) for r in frame_rows}) == 153
    for name, data in [('summary.csv',rows),('per_frame.csv',frame_rows)]:
        with (out/name).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data[0]),lineterminator='\n')
            writer.writeheader()
            writer.writerows(data)
    (out/'summary.json').write_text(json.dumps(dict(standard_historical=standard,records=records,deltas=deltas),indent=2,ensure_ascii=False)+'\n')
    winners = {}
    for key in ['cd_m2','point_fscore','depth_rmse_m','intensity_rmse','return_f1']:
        sign = 1 if key in ['point_fscore','return_f1'] else -1
        winners[key] = max(records,key=lambda r:sign*r['groups']['development']['metrics'][key])['variant']
    neural_delta = next(d for d in deltas if d['variant'] == 'hybrid44_neural')
    d = neural_delta['vs_standard_percent']
    conclusion = (f'neural 相对标准历史结果：CD {d["cd_m2"]:+.2f}%，深度 RMSE {d["depth_rmse_m"]:+.2f}%，'
                  f'强度 RMSE {d["intensity_rmse"]:+.2f}%，点云 F-score {neural_delta["vs_standard_point_fscore_percentage_points"]:+.3f} 个百分点。'
                  '应联合判断这些指标，不能仅依据 CD 宣称整体优于标准 STGC。')
    lines = ['# Modi01：标准 BCE refinement 与最终比较', '',
        '三个 8120 / seed 0 候选均从各自 30,000 步最终 EMA 主干出发，完成 1,000 步标准 STGC ray-drop refinement。只更新 U-Net，主干及原始 checkpoint 未变化。', '',
        '本次 refinement 后的候选最优指标：' + '；'.join(f'{k}={v}' for k,v in winners.items()) + '。', '',
        conclusion, '',
        '## Refinement 后：4 帧开发集与标准历史结果', '', *HEADER,
        metric_row('标准 STGC（历史，refinement 后）',standard)]
    for record in records:
        lines.append(metric_row(record['variant'],record['groups']['development']['metrics']))
    lines += ['', '以上候选使用 EMA 主干 + 最终 U-Net；U-Net 自身未做 EMA。标准行来自历史汇总，未在本次重新运行。', '',
              '相对标准历史结果的数值差（误差项负值更好，F-score 正值更好）：', '',
              '| 方法 | CD 相对变化 | 深度 RMSE 相对变化 | 强度 RMSE 相对变化 | 点云 F-score 差（百分点） |',
              '|---|---:|---:|---:|---:|']
    for delta in deltas:
        d=delta['vs_standard_percent']
        lines.append(f'| {delta["variant"]} | {d["cd_m2"]:+.2f}% | {d["depth_rmse_m"]:+.2f}% | {d["intensity_rmse"]:+.2f}% | {delta["vs_standard_point_fscore_percentage_points"]:+.3f} |')
    lines += ['', '## Refinement 前后', '', *HEADER]
    for pre, record in zip(before,records):
        lines.append(metric_row(record['variant']+' / 前',pre['groups']['ema_development']['metrics']))
        lines.append(metric_row(record['variant']+' / 后',record['groups']['development']['metrics']))
    lines += ['', '未加回波掩码的深度与强度逐元素保持不变，改善来自训练后的回波预测及其掩码；refinement 没有修改深度或强度网络。', '',
              '## Refinement 后：47 帧训练集', '', *HEADER]
    for record in records:
        lines.append(metric_row(record['variant'],record['groups']['train']['metrics']))
    lines += ['', '## Refinement 后：距离与边缘', '',
              '开发集按像素合并误差；距离采用有效 GT 深度。边缘为相邻有效 GT 深度差 >1 m 的两侧像素，水平环绕、垂直不环绕。该表不与主表的逐帧 RMSE 平均混为一谈。', '',
              '| 区域 | 像素数 | 方法 | 深度 RMSE (m) | 强度 RMSE | GT 回波召回 |', '|---|---:|---|---:|---:|---:|']
    for key in ['gt_return','depth_edge','non_edge_gt_return','range_0_10m','range_10_30m','range_30_50m','range_50_80m']:
        for record in records:
            s=record['groups']['development']['strata'][key]
            lines.append(f'| {key} | {s["pixels"]} | {record["variant"]} | {s["depth_rmse_m"]:.6f} | {s["intensity_rmse"]:.6f} | {s["return_recall"]:.6f} |')
    lines += ['', '## 固定协议与检查', '',
        '- 直接调用本仓库标准 `model/runner.py` 中的 `Trainer.refine`，保存独立源码副本后运行，函数内容未修改。',
        '- 47 个官方 legacy 训练帧，全 batch；BCE；Adam（weight_decay=0）、OneCycleLR，max_lr=0.001，1000 步；标准矩形随机遮挡。优化和遮挡随机种子均固定为 0。',
        '- 这次明确采用标准 STGC 的 BCE-only refinement。原注册信息中未执行过的 `bce_expected_masked_depth_support_v1` 不是本次损失；未加入深度或强度风险项。',
        '- 三个初始 U-Net 权重及 BatchNorm 状态直接比较相同；从 checkpoint 中冻结的初始 U-Net 开始，没有重新随机初始化。其前向与标准 STGC U-Net 逐元素相等。',
        '- 只允许 unet.* 参数和缓冲区变化。refinement 前后直接比较所有主干状态（包括非张量配置），确认冻结；原 checkpoint 大小和修改时间不变。没有计算文件摘要。',
        '- 新 refined checkpoint 保存后重新严格载入再评估。4 帧开发集的未掩码深度/强度与 refinement 前已保存预测逐元素相同；训练集复用相同 4096 射线分块下的冻结主干渲染输出。',
        '- 主干渲染沿用 AMP、每条射线 768 采样、无扰动；U-Net 训练使用原标准 FP32 路径。最终推理和指标沿用原 AMP/GT 精度、>0.5 回波阈值和逐帧平均。',
        '- train 排除 8130/8140/8150/8160，这 4 帧只用于最终开发评估；没有用于 refinement、早停或 checkpoint 选择。val/test 仍共用这些帧，是 post-hoc development。',
        '- CD 单位 m²，为双向平均平方距离之和；F-score 阈值为平方距离 0.05 m²（半径约 22.36 cm），深度截断 80 m，强度范围 [0,1]。',
        '- moving/static 无可靠标签，未报告该分层。标准历史 CSV 缺回波指标，表中标为未记录。',
        '- 已补齐 refinement 阶段，并采用标准代码的损失与优化流程；标准 8120 权重仍未找到，历史 seed、EMA 状态和原始逐帧输入未独立核实，所以不是严格配对的同环境重测。',
        '- 本轮没有自动晋升 Best/lastBest，没有额外主干训练；all-modal 仍是改进架构消融，不能替代标准 STGC 对照。', '',
        '## 耗时与产物', '', '| 方法 | 输入准备+refinement (秒) | 优化阶段含保存 (秒) | 全部耗时 (秒) | PyTorch 峰值 allocated (MiB) |',
        '|---|---:|---:|---:|---:|']
    for s in statuses:
        lines.append(f'| {s["variant"]} | {s["preparation_and_refinement_seconds"]:.1f} | {s["optimization_seconds"]:.1f} | {s["elapsed_seconds"]:.1f} | {s["peak_refinement_allocated_mib"]:.1f} |')
    lines += ['', '预先固定展示首个开发帧 8130，各方法色标一致：', '',
              '![Pre/post refinement](frame_8130_pre_post.png)', '',
              '- summary.csv / summary.json：全部主指标和分层；per_frame.csv：153 次帧评估。',
              '- 各方法 checkpoints/stgc_nerf_ep0639_refine_standard_bce.pth：EMA 主干与训练后的 U-Net，含完整 refinement 协议。',
              '- 各方法 status.json / refine.log：训练进度、冻结检查和执行日志；predictions/：开发帧数值预测。checkpoint 与数值预测保留在本机，未上传 Git。',
              '- refine_source.py / evaluation_support.py：本次执行源码快照。standard_runner.py / standard_unet.py 的原版快照保留在本机，已直接核对内容与仓库 model/runner.py / model/unet.py 完全一致；Git 复用仓库中的原文件。', '',
              '- data_protocol.json / log/EXPERIMENT_POLICY.md：实际帧 ID、划分、manifest 路径及归一化约定。', '',
              '标准历史来源：[逐场景 CSV](../../../reports/data/stgc_all_scene_metrics.csv)、[历史测试报告](../../../reports/STGC-NeRF-three-benchmark-metrics-report.md)。', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    plot(out,state,records)
    print('\n'.join(HEADER+[metric_row('standard historical',standard)]+[metric_row(r['variant'],r['groups']['development']['metrics']) for r in records]))
    print(json.dumps(deltas,indent=2))
    print(f'Report: {out/"REPORT.md"}')


def plot(out,state,records):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,4,figsize=(20,8),layout='constrained')
    columns=['Before: masked depth (m)','After: masked depth (m)','Before: masked intensity','After: masked intensity']
    previous=[np.load(Path(j['pre_evaluation'])/'predictions/ema/8130.npz') for j in state['jobs']]
    current=[np.load(Path(j['output'])/'predictions/8130.npz') for j in state['jobs']]
    reference=previous[0]
    gt_d=reference['gt_depth_normalized'][0].astype(float)/float(reference['scale'])
    gt_i=reference['gt_intensity'][0].astype(float)
    for row in range(4):
        data=[gt_d,gt_d,gt_i,gt_i] if row==0 else [
            previous[row-1]['depth_normalized'][0]/float(previous[row-1]['scale'])*(previous[row-1]['return_probability'][0]>.5),
            current[row-1]['depth_normalized'][0]/float(current[row-1]['scale'])*(current[row-1]['return_probability'][0]>.5),
            previous[row-1]['intensity'][0]*(previous[row-1]['return_probability'][0]>.5),
            current[row-1]['intensity'][0]*(current[row-1]['return_probability'][0]>.5)]
        images=[]
        for col,value in enumerate(data):
            images.append(axes[row,col].imshow(value,vmin=0,vmax=80 if col<2 else 1,cmap='viridis' if col<2 else 'gray',aspect='auto',interpolation='nearest'))
            axes[row,col].set_xticks([])
            axes[row,col].set_yticks([])
            if row==0: axes[row,col].set_title(columns[col])
        axes[row,0].set_ylabel(['GT','Tied','Untied','Neural'][row],fontsize=12)
    for col,image in enumerate(images):
        fig.colorbar(image,ax=list(axes[:,col]),orientation='horizontal',shrink=.8,aspect=30,pad=.02)
    fig.suptitle('KITTI-360 8130 | frozen final EMA field | standard BCE refinement, 1000 steps\nIdentical unmasked depth/intensity; only the predicted-return mask changes',fontsize=14)
    fig.savefig(out/'frame_8130_pre_post.png',dpi=160)
    plt.close(fig)
    for array in previous+current: array.close()


if __name__=='__main__':
    main()
