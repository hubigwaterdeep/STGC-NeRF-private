"""Combine completed final evaluations and produce a reviewable report/figure."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

LABELS = {'hybrid44': 'Hybrid 4/4（tied）', 'hybrid44_untied': 'Hybrid 4/4（untied）',
          'hybrid44_neural': 'Hybrid 4/4（neural）'}
KEYS = ['cd_m2', 'point_fscore', 'depth_rmse_m', 'intensity_rmse', 'return_rmse', 'return_accuracy', 'return_f1']


def compare_run_inputs(records):
    registrations = [json.loads((Path(r['workspace']) / 'registration.json').read_text()) for r in records]
    ignored = {'config', 'workspace', 'representation'}
    keys = set().union(*(r['configuration'] for r in registrations)) - ignored
    varying = {k: [r['configuration'].get(k) for r in registrations] for k in sorted(keys)
               if any(r['configuration'].get(k) != registrations[0]['configuration'].get(k) for r in registrations)}
    reference = Path(records[0]['audited_source'])
    different, checked = [], set()
    for record in records[1:]:
        source = Path(record['audited_source'])
        for path in source.rglob('*.py'):
            rel = path.relative_to(source)
            if '.deps' in rel.parts or '.venv' in rel.parts:
                continue
            checked.add(str(rel))
            other = reference / rel
            if not other.is_file() or path.read_bytes() != other.read_bytes():
                different.append(f'{record["variant"]}/{rel}')
    modi_diff = []
    for path in Path(records[0]['modi_source']).glob('*.py'):
        for record in records[1:]:
            other = Path(record['modi_source']) / path.name
            if not other.is_file() or path.read_bytes() != other.read_bytes():
                modi_diff.append(f'{record["variant"]}/{path.name}')
    return dict(configuration_differences_excluding_paths=varying,
                archived_audited_python_files_compared=len(checked), audited_source_differences=different,
                archived_modi_source_differences=modi_diff, comparison_method='direct content comparison, no digests',
                val_equals_test=all(r['assets']['splits']['val']['frame_ids'] == r['assets']['splits']['test']['frame_ids'] for r in registrations))


def table(records, group):
    lines = ['| 方法 | CD ↓ (m²) | 点云 F-score ↑ | 深度 RMSE ↓ (m) | 强度 RMSE ↓ | 回波 RMSE ↓ | 回波 Acc ↑ | 回波 F1 ↑ |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for record in records:
        m = record['groups'][group]['metrics']
        lines.append('| ' + LABELS[record['variant']] + ' | ' + ' | '.join(f'{m[k]:.6f}' for k in KEYS) + ' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation', type=Path)
    args = parser.parse_args()
    out = args.evaluation.resolve()
    state = json.loads((out / 'evaluation.json').read_text())
    assert state['status'] == 'COMPLETED'
    records = [json.loads((Path(j['output']) / 'results.json').read_text()) for j in state['jobs']]
    assert all(r['status'] == 'COMPLETED' and r['checkpoint_stat_unchanged'] for r in records)
    assert all(r['groups'][g]['parameters_unchanged'] for r in records for g in r['groups'])
    completion = json.loads((Path(state['queue']).parent / 'completion_summary.json').read_text())
    audit = compare_run_inputs(records)
    (out / 'comparability.json').write_text(json.dumps(audit, indent=2) + '\n')
    assert not audit['configuration_differences_excluding_paths']
    assert not audit['audited_source_differences'] and not audit['archived_modi_source_differences']
    rows = []
    frame_rows = []
    for record in records:
        with (Path(record['output']) / 'per_frame.csv').open() as handle:
            frame_rows.extend(csv.DictReader(handle))
        for group, result in record['groups'].items():
            rows.append(dict(variant=record['variant'], group=group, frames=result['frames'], **result['metrics']))
    for name, data in [('summary.csv', rows), ('per_frame.csv', frame_rows)]:
        with (out / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(data)
    (out / 'summary.json').write_text(json.dumps(dict(records=records, comparability=audit), indent=2, ensure_ascii=False) + '\n')

    base, untied, neural = [r['groups']['ema_development']['metrics'] for r in records]
    changes = []
    for record, m in zip(records[1:], [untied, neural]):
        changes.append(f'- {LABELS[record["variant"]]} 相对 tied：CD {(m["cd_m2"] / base["cd_m2"]-1)*100:+.2f}%，'
                       f'深度 RMSE {(m["depth_rmse_m"] / base["depth_rmse_m"]-1)*100:+.2f}%，'
                       f'强度 RMSE {(m["intensity_rmse"] / base["intensity_rmse"]-1)*100:+.2f}%，'
                       f'点云 F-score {(m["point_fscore"]-base["point_fscore"])*100:+.3f} 个百分点，'
                       f'回波 F1 {(m["return_f1"]-base["return_f1"])*100:+.3f} 个百分点。')
    wins = {}
    for key in KEYS:
        bigger = key in ['point_fscore', 'return_accuracy', 'return_f1']
        wins[key] = max(records, key=lambda r: r['groups']['ema_development']['metrics'][key])['variant'] if bigger else min(records, key=lambda r: r['groups']['ema_development']['metrics'][key])['variant']
    core_keys = ['cd_m2', 'point_fscore', 'depth_rmse_m', 'intensity_rmse', 'return_f1']
    paired_wins = {}
    dev_rows = {r['variant']: {row['frame_id']: row for row in frame_rows if row['variant'] == r['variant']
                and row['weights'] == 'ema' and row['split'] == 'development'} for r in records}
    for key in core_keys:
        sign = 1 if key in ['point_fscore', 'return_f1'] else -1
        paired_wins[key] = sum(sign * float(dev_rows['hybrid44_neural'][f][key]) > sign * float(dev_rows['hybrid44'][f][key])
                              for f in dev_rows['hybrid44'])
    decision = ('本轮优先保留 neural 作为后续候选：五类核心指标的 EMA 开发集均值均最好。'
                if all(wins[k] == 'hybrid44_neural' for k in core_keys)
                else '本轮存在指标权衡，应结合五类指标判断，不以 CD 单项排名替代决策。')
    train_base, train_untied = [r['groups']['ema_train']['metrics'] for r in records[:2]]
    untied_tradeoff = (train_untied['cd_m2'] < train_base['cd_m2'] and train_untied['intensity_rmse'] < train_base['intensity_rmse']
                      and train_untied['return_f1'] > train_base['return_f1'] and untied['cd_m2'] > base['cd_m2']
                      and untied['intensity_rmse'] > base['intensity_rmse'] and untied['return_f1'] < base['return_f1'])
    untied_note = ('untied 的训练集 CD、强度 RMSE 与回波 F1 优于 tied，但开发集这三项均较差；当前预算下没有可见的开发集收益。该现象符合拟合与泛化差距增大的表现，尚不能据此确定原因。'
                   if untied_tradeoff else 'untied 的训练/开发集差异见下方完整表格。')
    gap_lines = ['| 方法 | CD 开发−训练 (m²) | 深度 RMSE 开发−训练 (m) | 强度 RMSE 开发−训练 |',
                 '|---|---:|---:|---:|']
    for r in records:
        dev, train = [r['groups'][g]['metrics'] for g in ['ema_development', 'ema_train']]
        gap_lines.append('| ' + r['variant'] + ' | ' + ' | '.join(f'{dev[k]-train[k]:.6f}' for k in ['cd_m2', 'depth_rmse_m', 'intensity_rmse']) + ' |')
    lines = ['# Modi01：8120 最终 checkpoint 联合评估', '',
        '三个方法均完成 30,000 步；以下为最终 epoch 639 checkpoint 的新评估，不是训练中 epoch 600 的旧指标。', '',
        decision, '', '## 结论依据', '', *changes, '',
        'neural 对 tied 的逐帧胜出数（共 4 帧）：' + '；'.join(f'{k}={v}/4' for k, v in paired_wins.items()) + '。', '',
        untied_note, '',
        '各指标最优：' + '；'.join(f'{key}={variant}' for key, variant in wins.items()) + '。', '',
        '这是单场景、单 seed、三个候选之间的开发集比较。尚未训练同配置 all-modal 对照，不能据此声称超过原始 STGC 或证明混合表示本身有效；没有自动选为 Best/lastBest。', '',
        '## 最终 EMA / pre-refiner：开发集主表', '', table(records, 'ema_development'), '',
        '## 最终 raw / pre-refiner：开发集对照', '', table(records, 'raw_development'), '',
        '## 最终 EMA / pre-refiner：47 帧训练集', '', table(records, 'ema_train'), '',
        '下表是不同帧集合指标均值之差，仅用于拟合差距诊断，不是泛化误差界或显著性结论。', '',
        *gap_lines, '',
        '下一项有信息价值的对照是相同 8120、seed 0、30k 预算的 all-modal，用来判断 4/4 混合相对原表示的收益。本次仅评估已完成的三个候选，没有启动该训练。', '',
        '## 距离与深度边缘诊断', '',
        '下表使用 EMA 开发集，按像素合并平方误差后开根号；不同于主表的逐帧 RMSE 均值。深度/强度均使用预测回波硬掩码，漏回波按零预测计入误差。距离由有效 GT 深度定义。边缘定义为相邻有效 GT 深度差 >1 m 的两侧像素（水平环绕、垂直不环绕）；这是几何诊断，不是语义边缘标签。', '',
        '| 区域 | 像素数 | 方法 | 深度 RMSE (m) | 强度 RMSE | GT 回波召回 |',
        '|---|---:|---|---:|---:|---:|']
    for key in ['gt_return', 'depth_edge', 'non_edge_gt_return', 'range_0_10m', 'range_10_30m', 'range_30_50m', 'range_50_80m', 'range_80_infm']:
        for record in records:
            s = record['groups']['ema_development']['strata'][key]
            fmt = lambda v: 'N/A' if v is None else f'{v:.6f}'
            lines.append(f'| {key} | {s["pixels"]} | {record["variant"]} | {fmt(s["depth_rmse_m"])} | {fmt(s["intensity_rmse"])} | {fmt(s["return_recall"])} |')
    lines += ['', '所有像素的合并回波统计：', '',
              '| 方法 | Precision | Recall | F1 | TP | FP | FN | TN |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for record in records:
        s = record['groups']['ema_development']['strata']['all_pixels']
        lines.append(f'| {record["variant"]} | {s["return_precision"]:.6f} | {s["return_recall"]:.6f} | {s["return_f1"]:.6f} | {s["tp"]} | {s["fp"]} | {s["fn"]} | {s["tn"]} |')
    lines += ['', '## 四帧一致性', '',
              '| EMA 帧 | 方法 | CD (m²) | 深度 RMSE (m) | 强度 RMSE | 回波 F1 |',
              '|---:|---|---:|---:|---:|---:|']
    for frame_id in ['8130', '8140', '8150', '8160']:
        for record in records:
            row = next(row for row in frame_rows if row['variant'] == record['variant'] and row['frame_id'] == frame_id and row['weights'] == 'ema' and row['split'] == 'development')
            lines.append('| ' + frame_id + ' | ' + record['variant'] + ' | ' + ' | '.join(f'{float(row[key]):.6f}' for key in ['cd_m2', 'depth_rmse_m', 'intensity_rmse', 'return_f1']) + ' |')
    lines += ['', '## 预算与可比性', '', '| 方法 | 参数总数 | 训练时长 | 最终步数 | EMA 更新次数 |', '|---|---:|---:|---:|---:|']
    for record, job in zip(records, completion['jobs']):
        seconds = int(job['duration_seconds'])
        lines.append(f'| {record["variant"]} | {record["parameter_count"]:,} | {seconds//3600}h {(seconds%3600)//60}m | {record["steps"]} | {record["ema_updates"]} |')
    lines += ['',
        f'- neural 比 tied/untied 少 {records[0]["parameter_count"]-records[2]["parameter_count"]:,} 个参数（总模型约 {(1-records[2]["parameter_count"]/records[0]["parameter_count"])*100:.3f}%；按场景场统计约 0.138%）。未建立完全等容量的 neural 对照，不能仅凭结果作严格的表示机制归因。',
        f'- neural 训练总耗时比 tied 增加 {(completion["jobs"][2]["duration_seconds"]/completion["jobs"][0]["duration_seconds"]-1)*100:.2f}%，包括相同周期的开发评估开销；这不是纯 kernel 性能测量。',
        f'- 非路径训练选项一致，seed=0、同一预算与数据协议。通过直接读取内容比较 {audit["archived_audited_python_files_compared"]} 个原始 Python 源文件和三个 Modi01 快照，未发现差异；未计算任何文件摘要。',
        '- KITTI-360 8120–8170，共 51 个时刻；train 为排除 8130/8140/8150/8160 后的 47 帧，val/test 共用这 4 帧。时间为 (frame_id−8120)/50。',
        '- 直接调用归档 Trainer.eval_step 与四类指标；AMP、预加载 GT 精度、每条射线 768 采样、4096 射线渲染分块、无 perturb、回波阈值 >0.5 与训练时周期评估一致。主表每帧独立计算后等权平均；额外完整指标在 summary.csv。',
        '- CD 是双向最近邻平方距离均值之和，单位 m²；点云 F-score 使用平方距离阈值 0.05 m²，即半径约 0.2236 m。深度截断至 80 m；强度范围 [0,1]。',
        '- EMA 每个 epoch 更新一次，最终 639 次；raw 与 EMA 分别严格载入。无优化器、无训练或 refiner 更新。每组结束直接比较参数值，确认未变化；checkpoint 大小及修改时间未变化。',
        '- refiner 未训练，因此本次只有 pre-refiner；post-refiner 未评估。没有可靠 moving/static 标签，未报告该分层。原始权重只评估开发集；训练集使用与主比较一致的 EMA。',
        '- 只有 4 个彼此相关的开发帧，未作统计显著性或跨场景泛化结论。', '',
        '## 预测可视化与文件', '',
        '预先固定展示首个开发帧 8130；各方法采用共同色标。完整 4 帧的 EMA/raw 浮点预测保存在本机各方法 predictions/ 目录，未上传到 Git。', '',
        '![Frame 8130 comparison](frame_8130_comparison.png)', '',
        '- summary.csv / summary.json：汇总及协议；per_frame.csv：全部 165 次帧评估。',
        '- 每个方法 results.json：完整主指标、分层与源码路径；evaluate.log：执行日志。',
        '- evaluate_source.py：本次评估入口副本；comparability.json：配置与源代码比较结果。', '',
        '执行方式：在仓库激活环境后，使用 `python Modi01/evaluate.py --output Modi01/evaluations/<新目录>`，完成后运行 `python Modi01/summarize_evaluation.py <评估目录>`。', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    draw_comparison(out, records)
    print(table(records, 'ema_development'))
    print('\n'.join(changes))
    print(f'Report: {out / "REPORT.md"}')


def draw_comparison(out, records):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    arrays = [np.load(Path(record['output']) / 'predictions/ema/8130.npz') for record in records]
    scale = float(arrays[0]['scale'])
    gt_d = arrays[0]['gt_depth_normalized'][0].astype(float) / scale
    gt_i = arrays[0]['gt_intensity'][0].astype(float)
    fig, axes = plt.subplots(4, 4, figsize=(20, 8), layout='constrained')
    columns = [('Depth (m)', 0, 80, 'viridis'), ('|Depth error| (m)', 0, 10, 'magma'),
               ('Intensity', 0, 1, 'gray'), ('|Intensity error|', 0, 0.3, 'magma')]
    images = []
    for row in range(4):
        if row == 0:
            depth, intensity = gt_d, gt_i
        else:
            a = arrays[row-1]
            pm = a['return_probability'][0] > 0.5
            depth = a['depth_normalized'][0] / scale * pm
            intensity = a['intensity'][0] * pm
        images = []
        for col, data in enumerate([depth, np.abs(depth-gt_d), intensity, np.abs(intensity-gt_i)]):
            _, lo, hi, cmap = columns[col]
            image = axes[row, col].imshow(data, vmin=lo, vmax=hi, cmap=cmap, aspect='auto', interpolation='nearest')
            images.append(image)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(columns[col][0])
        axes[row, 0].set_ylabel(['Ground truth', 'Tied', 'Untied', 'Neural'][row], fontsize=12)
    for col, image in enumerate(images):
        fig.colorbar(image, ax=list(axes[:, col]), orientation='horizontal', shrink=0.8, aspect=30, pad=0.02)
    fig.suptitle('KITTI-360 8130 | final 30k-step EMA | pre-refiner\nShared scales; predicted-return mask; error colors saturate at shown maximum', fontsize=14)
    fig.savefig(out / 'frame_8130_comparison.png', dpi=160)
    plt.close(fig)
    for array in arrays:
        array.close()


if __name__ == '__main__':
    main()
