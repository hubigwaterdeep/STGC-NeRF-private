# 与标准 STGC 的历史结果对照：KITTI-360 8120

当前 neural 只在三个 Modi01 候选中表现最好。与仓库保存的标准 STGC 历史结果相比，它没有取得更好的数值；但双方 refinement 阶段不同，这不是已经完成的同阶段架构对照。

## 已有结果

两批结果的场景均为 8120，主干训练预算均为 30,000 步，每项开发/测试指标均汇总 4 帧。标准行来自历史汇总，并非本次重新评估。

| 方法 | 阶段及权重口径 | CD ↓ (m²) | 点云 F-score ↑ | 深度 RMSE ↓ (m) | 强度 RMSE ↓ | 回波 F1 ↑ |
|---|---|---:|---:|---:|---:|---:|
| 标准 STGC（历史） | refinement 后；raw/EMA 未独立核实 | 0.06738932 | 0.95346636 | 2.32898498 | 0.09016316 | 未记录 |
| Modi01 tied | refinement 前，最终 EMA | 0.07805865 | 0.93189228 | 3.18703765 | 0.11887343 | 0.92537853 |
| Modi01 untied | refinement 前，最终 EMA | 0.08009097 | 0.91699630 | 3.26065832 | 0.12272872 | 0.92347550 |
| Modi01 neural | refinement 前，最终 EMA | 0.06783888 | 0.94632077 | 3.07467026 | 0.11627119 | 0.92829491 |

F-score 沿用仓库平方距离阈值 0.05 m²，欧氏半径约 22.36 cm；不是严格 5 cm 指标。

neural 相对标准历史结果的数值差：

- CD 高 **0.67%**。
- 点云 F-score 低 **0.715 个百分点**。
- 深度 RMSE 高 **32.02%**。
- 强度 RMSE 高 **28.96%**。

## 可比性的缺口

标准历史报告明确包含 ray-drop refinement，而本次三个 Modi01 实验的 refiner 均未训练。回波掩码直接影响深度、强度和点云指标，因此不能把上述差值全部解释为主干表示的优劣，也不能假定补上 refiner 就一定能追回差距。

标准历史 CSV 未记录回波指标、seed、最终 checkpoint 的 raw/EMA 状态及逐帧 ID。虽然报告记载 30k 主干训练和 4 个留出帧，但本次未找到足够的原始记录核实完整采样与数据版本一致性。本次 Modi01 的 4 个开发帧为 8130、8140、8150、8160，val/test 重叠。

已检查本机三个项目目录中的模型文件（包含被忽略的日志目录）和 STGC-NeRF-private 的 GitHub Release 清单，未找到标准 8120 checkpoint；该仓库 Release 当前只有标准 4950 checkpoint。没有用其他场景权重代替，也没有启动额外训练或 refiner 更新。

下一步同阶段比较需要取得标准 8120 checkpoint 并关闭 refiner重评，或按标准协议给三个候选补齐 refiner 后比较。要作严格的表示消融，还需统一 seed、数据、采样、训练预算、权重口径和其余模块。

`all_modal` 是保留改进版平面、flow、融合与 geometry residual 的全 modal 消融配置；标准 STGC 使用其原始结构及 time-indexed hash/interpT。**all-modal 消融不能代替标准 STGC 对照。**

## 来源

- [标准 STGC 历史逐场景 CSV](../../../reports/data/stgc_all_scene_metrics.csv)：8120 行。
- [标准 STGC 历史三组数据报告](../../../reports/STGC-NeRF-three-benchmark-metrics-report.md)：开头注明 30k 训练、ray-drop refinement 和最终测试；单场景表记录 8120 数值。
- [本次 Modi01 最终评估](REPORT.md)：最终 30k、EMA/raw、pre-refiner、逐帧及分层结果。

历史标准结果仅作带有阶段标注的参考，未伪装成本次同口径复测。
