# Modi01 实现与验证报告

> 本文记录首次实现时的状态。2026-09-18 后续按用户授权完成了三个 8120 / seed 0 候选的各 30,000 步训练及最终评估，见 [最终评估报告](evaluations/8120_final_20260918_1520/REPORT.md)。下文的 NOT_RUN 保留为当时记录，不代表当前训练和评估状态。

状态：三项架构代码、独立配置、从头训练接入、容量报告和验证工具已实现。真实场景训练未启动，未修改或晋升 Best/lastBest。

## 实现范围

- 首个候选：4 层 modal / 4 层实际 time-indexed hash + interpT，包含全 modal 对照。
- 后续候选：独立低/高阶空间系数，默认表预算与 tied 对照相同。
- 后续候选：小型 decoder 在空间 latent 插值之后生成高阶系数，低阶继续显式优化，保持 xy/xz/yz 域且无时间输入。
- 训练接入：重用审计快照的损失、采样、flow、120 维特征、融合、几何残差、原头部、EMA 及训练器。新增 checkpoint 配置检查；from-scratch 主干启用训练，UNet 保持原阶段的冻结策略。
- 所有工作仅位于 Modi01；依赖的历史 source 文件保持原样。没有文件摘要计算。

## 代码验证

11 项 unittest 全部通过，原始记录：`reports/tests.log`。

- 全 modal 在 high-order progress=0/1、t=0/0.37/1 上与旧场前向逐元素相等；反向梯度在 CUDA 归约容差内一致。
- 共享平面、时间基、flow、融合、geometry residual、密度/强度/return 头和 UNet 初始化一致；新增分支不改变后续 CUDA RNG。
- 时间端点、8 个精确 knot、Lagrange 分区和 4 个多项式节点通过；标量/逐点时间次序一致，非法时间拒绝。
- 原 HashGridT 与所选层实现对比，含单位域边界与越界 flow 坐标，最大绝对输出误差 `2.384185791015625e-7`，来源于 FP32 运算次序及 FP16 输出。
- 0、1、4、7、8 个 modal 层的形状/前向通过；默认保留 24 维动态 hash、120 维场。
- 所有选中的 modal/time-knot/high-coefficient/latent 表和 decoder/embedding 参数都有有限非零梯度，无训练特征 detach/cache。optimizer group 无遗漏和重复。
- current/next/previous 仍为 0.5/0.25/0.25；邻帧 hash 查询仍处于 no_grad，坐标按原 flow 移动，端点仍复用 current。
- density 使用 full 特征、appearance 使用 base geometry 的接口通过；完整 renderer 的反向可达 hash 与 flow。
- 模型 checkpoint 正常回读，不同表示配置即使 strict=False 也被拒绝。
- CPU 时间代数仍为示例 rank 29；此结果没有扩展为全模型秩或训练结论。

## 合成性能实测

RTX 5090；PyTorch 2.7.1+cu128；固定 seed 20260918；128 rays × 64 samples；8 次 warmup 覆盖全部时间表、分配 Adam 状态，再计时 5 次更新。下面是 raw、pre-refiner 的合成 renderer + backward + Adam；不含场景流教师、真实任务约束或数据加载。显存为 PyTorch peak allocated，吞吐为本测量输入下的 rays/s，不能当作完整训练速度或质量结果。

| 配置 | 场景场参数 | hash 容量差 | 峰值 MiB | 步时中位 ms | 训练 rays/s | 推理 rays/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all_modal | 46,635,190 | +0 | 878.6 | 69.37 | 1845 | 5264 |
| hybrid44 | 46,635,190 | +0 | 839.3 | 76.71 | 1669 | 4182 |
| hybrid44_untied | 46,635,190 | +0 | 828.1 | 78.17 | 1637 | 4202 |
| hybrid44_neural | 46,570,682 | -64,508 | 825.6 | 81.74 | 1566 | 3619 |

全 modal / 4/4 tied / 4/4 untied 的实际 hash 参数均为 12,582,912；总场景场参数相同。neural 的 decoder 和 embeddings 已计入，实际少 64,508 个参数（约占总场景场 0.138%）。保留原 hash gate 张量，其中混合方案有 96 个原有 gate 值不再使用；没有新增闲置参数作为容量填充。混合候选在此次小批次合成测试中更慢，当前没有速度提升或重建收益的结论。

详见 `reports/benchmark.json`，含完整表/网络参数、所有时间样本和峰值 allocated/reserved 数据。

## 数据与真实训练状态

8120、10200：47 train / 4 val / 4 test；3353：60 train / 4 val / 4 test。三个 preflight 核对了实际帧 ID、归一化时间、range-view shape、pose shape/有限性、scale、offset、FOV；val/test 保持官方 legacy 重叠协议。这验证本机 manifest 与 loader 一致，未声称字节等同于其他远端实验数据。

初始默认教师路径不存在；随后在本机 `basis4D-intensity-mtl-diagnosis/flow/checkpoints/FTD_o.pth` 找到权重，按原 GMSF 严格载入，模型键和形状通过。审计快照的 Trainer/GMSF 依赖可导入。最新三个注册报告应为 `READY_NOT_RUN`，不是训练成功；详细路径见 `reports/runtime_assets.json` 和 `reports/preflight_*/registration.json`。

CD、F-score、depth RMSE、intensity RMSE、return 指标及分层成绩均为 **NOT_RUN**，不是零；raw/EMA 与 pre/post-refiner 的真实评估也尚未执行。下一步仅应按注册预算先运行全 modal 对照和 4/4 候选，再写联合指标决策；后续两个候选已经可选，但没有自动加入实验队列。
