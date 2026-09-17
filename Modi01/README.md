# Modi01：时间表示改进

实现了计划中的三个候选，默认是 **4 层 modal + 4 层 local temporal latent**。新增代码全部位于 Modi01；通过明确的源码入口复用 `reports/logs/intensity_fit_diagnosis_20260916/source` 的场景场、renderer 和训练器。根目录标准 STGC 和历史 Best/lastBest 不作修改。

## 配置

| 配置文件 | hash 层 0–3 | hash 层 4–7 | 高阶空间系数 |
| --- | --- | --- | --- |
| `configs/all_modal.json` | modal | modal | 与低阶共享，旧 forward |
| `configs/hybrid44.json` | modal | 8 个时间 knot + 4 通道 Lagrange 插值 | 与低阶共享 |
| `configs/hybrid44_untied.json` | modal | 同上 | 独立显式表 |
| `configs/hybrid44_neural.json` | modal | 同上 | 空间 latent 插值后由小 MLP 生成 |

所有配置保留 xy/xz/yz 三个二维角色、每层一个输出、24 维动态 hash 和 120 维整体特征。平面、flow、融合、geometry residual、密度/强度/return 头以及邻帧聚合均继承原实现。代码没有添加渲染后修正或 canonical warp。

`untied` 将保留 modal 层的表容量均分给低阶和高阶，所有显式表都训练；`neural` 保留显式低阶系数，只生成保留 hash 层的高阶系数。MLP 输入为该角色二维坐标的编码、先插值得到的空间 latent、role embedding 和 level embedding，**不输入时间，也不补入第三个空间坐标**。plane 的系数策略不变。

## 空间尺度与初始化

拆分后的每个层仍使用原八层 schedule 的连续 grid scale。TCNN 只接受整数 base resolution，因此单层 encoder 使用 `ceil(scale)+1`，输入乘 `scale/ceil(scale)`，保持其连续格点坐标 `x*scale+0.5`。不重新计算四层的增长率，也不裁剪 flow 移动后越界的坐标。重排 FP32 运算会有小量舍入差异，测试对比原 TCNN/HashGridT 路径并量化误差。

保留 modal 前缀直接复制同 seed 原始表的对应参数；其余共享组件和初始化后的 CPU/CUDA RNG 保持不变。新分支在隔离的 RNG 作用域中创建。全 modal 模式直接调用旧的 `_hash_dynamic`。

原高阶时间原子、progress、sigmoid gates 和 RMS limiter 公式均保持不变。混合模式的 hash RMS cap 作用于每个角色保留的 modal 层；自由时间通道不受该 cap。原 gate 张量保持原形状，4/4 配置中有 96 个原有 gate 值不再参与 hash 计算，报告明确列出，不将它们算成新增可用表容量。

## 运行与检查

从仓库根目录运行，环境需要本项目已有的 PyTorch/CUDA/tiny-cuda-nn：

```bash
cd /home/zijiewu/Code/STGC-NeRF-private
source activate_env.sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s Modi01/tests -v
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python Modi01/check_stgc_temporal_span.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m Modi01.benchmark
```

仅核对数据与生成注册信息，不训练：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m Modi01.train \
  --config configs/kitti360_8120.txt \
  --representation Modi01/configs/hybrid44.json \
  --workspace Modi01/runs/8120/hybrid44_seed0 \
  --resume /home/zijiewu/Code/basis4D-intensity-mtl-diagnosis/flow/checkpoints/FTD_o.pth \
  --seed 0 --iters 30000 --max_train_steps 30000 --prepare-only
```

之后去掉 `--prepare-only` 才会启动该单个实验。训练总预算、学习率 schedule、seed、射线数和 patch schedule 必须在对照中一致；只修改 representation 和输出目录来运行 `all_modal.json`。不要用全 modal 的 checkpoint 初始化候选；这里明确要求从头训练并拒绝 `init_best`。

`--resume` 是原冻结场景流教师权重，不是场景模型预训练。上面的本机路径已按 GMSF 模型键和张量形状严格载入检查，没有计算文件摘要。默认 `flow/checkpoints/FTD_o.pth` 目前不存在，不传有效教师路径时会生成 `BLOCKED` 注册报告并终止训练，不会换用简化训练或重新训练教师。

训练入口只接受计划中的 8120、10200、3353 和 legacy 官方帧协议。它检查实际帧 ID、inclusive 时间、range-view 形状、位姿形状/有限性、scale、offset、FOV，并记录 manifest。所有输出必须写入 Modi01 的独立目录，已有训练产物禁止覆盖。仅有 preflight 注册文件的目录可以继续启动。训练入口不自动启动其他候选、场景、测试集评估或模型晋升；默认跳过最后评估及 refiner 训练，训练期间仍沿用原 development 评估。

## 容量与报告

`representation.json` 记录实际模型/场景场参数数、表/decoder/embedding 数量、各 optimizer group 和学习率；checkpoint 保存配置与空间 schedule，并拒绝不同配置的权重。`log/EXPERIMENT_POLICY.md` 和 `registration.json` 保存实际帧划分和训练预算。

默认全 modal、4/4 tied、4/4 untied 的 hash 表总量完全一致。neural 的 decoder/embedding 计入预算；受 TCNN 表大小二次幂粒度限制，默认总场景场比对照少 64,508 个参数。没有用闲置参数补齐容量；因此 neural 与 untied 的差异不能全部归因于表示形式。

本轮只运行了合成数据的代码验证和有明确步数上限的性能测试。实际场景训练、CD/F-score/depth RMSE/intensity RMSE/return 指标仍待运行。历史已观察场景按 post-hoc development 处理；legacy val/test 使用相同帧，不能宣称拥有未见测试集。正式结果还须区分 raw/EMA、pre/post-refiner，并在可靠标签存在时分 moving/static/edge/range 层报告；缺标签时明确标为 unavailable。

详见 `IMPLEMENTATION_REPORT.md` 和 `reports/` 的原始 JSON/日志。
