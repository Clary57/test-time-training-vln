# 项目代码变更与目录说明

本文说明从基线提交 `b8af38601cc51e0d768ffd0234f38f4b651cbda4` 到当前开发历史所做的代码变更，以及各目录在实验中的职责。

## 1. 总体结构

```text
test-time-training-vln/
├── scripts/                 # 选择实验 episode
├── ttrl-src/
│   ├── eval/                # A/B/C evaluator、动作 trace 和结果聚合
│   ├── trainer/             # consensus 与在线更新控制
│   ├── tests/               # CPU 单元测试
│   ├── config/              # 实验配置，本轮未修改
│   ├── data/                # 数据路径相关，本轮未修改
│   ├── dataset/             # dataset 代码，本轮未修改
│   ├── model/               # 模型代码，本轮未修改
│   └── train/               # 训练代码，本轮未修改
└── PROJECT_CHANGES_GUIDE.md # 本文
```

整体改动集中在实验选择、evaluator 控制、动作日志、严格聚合和 CPU 测试。没有重新设计模型架构、prompt、reward、medoid 或默认 TTRL 算法。

## 2. `scripts/`：决定运行哪些 episode

### `select_representative_r2r_episodes.py`

从 R2R `val_unseen` 候选中确定性选择代表性子集：

- 读取 scene 和参考路径；
- 按路径长度分成 short、middle、long；
- 使用固定 seed 保证可复现；
- 输出 episode ID 和选择理由；
- 不读取模型结果、SR、SPL、nDTW 等表现。

该脚本用于 representative-20 的选择。

### `select_balanced_r2r_extension.py`

把已有 selection 扩展为更大的 balanced selection：

- 保留已有正式 episode；
- 排除重复 ID；
- 按长度类别补足名额；
- 按 scene 的真实候选数量分配配额；
- 输出新增列表和最终完整列表。

该脚本用于从 representative-20 扩展到 balanced representative-72。

这两个脚本只决定“跑谁”，不改变单个 episode 内的导航决策。

## 3. `ttrl-src/eval/`：评测入口与结果记录

### `eval_ttt.py`

这是 A 和 B 共用的 evaluator：

- A：三候选 consensus/medoid，加在线 LoRA 更新；
- B：同一条 evaluator 路径，但通过 `--disable-update` 禁止参数更新。

新增能力：

1. 按 selection 文件精确运行指定 episode ID；
2. 使用 `--stop-trace` 记录逐原子动作的结构化日志；
3. 修复直接执行脚本时 `eval.stop_trace` 的 Python package 解析问题。

默认 TTRL 路径没有改变 prompt、candidate generation、reward、medoid、action parser、pending queue、loss 或 optimizer 更新行为。

### `eval.py`

这是 C frozen NaVIDA evaluator。它不执行 trajectory consensus、teacher-forcing、backward、optimizer 或在线 LoRA 更新。

新增能力：

- 按指定 ID 精确运行与 A/B 相同的 episode；
- 支持 `--stop-trace`；
- 支持 `--max-executed-chunks {2,3}`。

`--max-executed-chunks` 的含义：

- 默认值 `3`：保持项目历史 frozen C 行为；
- 显式 `3`：与默认完全一致；
- 显式 `2`：仅截断第三个 parsed action chunk，用于复现官方 NaVIDA 的前两个 chunks 协议。

最终值写入 episode manifest、run metadata 和 stop-trace metadata。

### `episode_subset.py`

A/B/C 共用的 episode 过滤工具：

- 读取 selection JSON；
- 保持文件中的 ID 顺序；
- 检查重复、未知和缺失 ID；
- 对指定 selection 做 `split-num/split-id` 分片。

它保证 evaluator 只运行锁定的 episode，并且顺序不会隐式改变。

### `stop_trace.py`

默认关闭的非干扰式动作日志模块，记录：

- 模型原始输出；
- candidate 及 reward；
- selected candidate；
- 解析后的 action chunk；
- pending queue；
- 实际执行动作；
- 动作前后距离；
- 是否进入 success radius；
- stop 的生成、选择、排队和执行；
- termination reason。

它只记录已经发生的事件，不参与动作选择。CPU 测试验证 trace 开关不会改变动作或 RNG 状态。

### `aggregate_ttt_results.py`

严格结果聚合器：

- 支持显式全局 expected episode IDs；
- 可以合并旧结果和新增结果；
- 拒绝 duplicate、missing 和 unexpected episode；
- 拒绝非有限指标；
- 检查 manifest 和 stats 完整性。

它只处理完成后的结果，不参与导航推理。

### `__init__.py`

把 `ttrl-src/eval` 明确定义为 Python package，避免直接执行 evaluator 时同名 `eval.py` 遮蔽 `eval.stop_trace`。

## 4. `ttrl-src/trainer/`：A/B 更新控制

### `trajectory_consensus_updater.py`

这是 A/B 共用的 consensus/TTRL 组件。

A 的路径：

```text
生成3个candidate
→ 计算reward
→ medoid选择
→ teacher-forcing loss
→ backward
→ gradient clipping
→ optimizer.step
→ LoRA参数变化
```

B 的 `--disable-update` 路径：

```text
生成3个candidate
→ 计算reward
→ medoid选择
→ teacher-forcing loss
→ backward
→ gradient clipping
→ 跳过optimizer.step
→ LoRA参数保持不变
```

B 不是 frozen C。它保留 A 的候选生成、reward、medoid、loss、backward、梯度裁剪、action parsing 和 pending queue，只隔离参数更新。

运行记录还包含 `update_enabled`、adapter norm、checksum 和 parameters unchanged 等验证字段。默认不传 `--disable-update` 时，A 的行为保持原样。

## 5. `ttrl-src/tests/`：CPU 保护网

### `test_episode_subset.py`

验证 selection 顺序、精确 ID、重复/缺失/未知 ID 检查和分片逻辑。

### `test_balanced_extension.py`

验证保留旧 episode、按类别补足、去重、配额和确定性回填。

### `test_trajectory_consensus_updater.py`

验证 reward/medoid、selected candidate、正常 update、disable-update、参数变化、RNG 一致性、严格聚合和 evaluator 协议。

### `test_stop_instrumentation.py`

验证：

- tracing disabled 不产生 trace；
- trace on/off 动作、candidate、pending queue 和 RNG 一致；
- distance before/after 正确；
- future/current/executed stop 可区分；
- early-stop、max-step 和 environment done 可区分；
- JSONL 连续可解析；
- evaluator import precedence 正确。

### `test_frozen_action_chunk_control.py`

验证：

- 默认 chunk 数为 3；
- 默认 3 与显式 3 等价；
- 显式 2 只删除第三个 chunk；
- pending queue 和 stop 顺序正确；
- invalid fallback 的 RNG 调用符合执行 chunk 数；
- trace on/off 不改变动作或 RNG；
- metadata 记录最终 chunk 数；
- 非 2/3 参数被 CLI 拒绝；
- early-stop 阈值保持不变。

## 6. 没有修改的核心部分

以下目录和外部资产没有修改：

```text
ttrl-src/model/
ttrl-src/config/
ttrl-src/data/
ttrl-src/dataset/
ttrl-src/train/
外部 NaVIDA 仓库
模型 checkpoint
R2R dataset / GT
MP3D scenes
```

因此没有修改：

- 模型架构和权重；
- tokenizer/processor；
- R2R 和 Habitat 数据；
- prompt；
- success distance；
- forward distance 和 turn angle；
- reward 和 medoid 定义；
- TTRL loss；
- 默认 optimizer update；
- early-stop 阈值。

## 7. 实验结果不属于 Git 源码

实验结果位于：

```text
/hdd3/wangziyu/runs/test-time-training-vln/
```

其中包括 selection、launch、stdout/stderr、stats、steps、stop trace、summary、audit 和 Git bundle。这些是实验现场与证据，不应作为源码随 Git 推送。

模型、数据集、checkpoint、外部 NaVIDA 文件和软链接目标也不应推送。

## 8. 提交演进

```text
b8af386  原始起点
   │
   ├─ 273e778  精确 episode 子集
   ├─ 28a0842  representative 扩展和严格聚合
   ├─ 0918a19  disable-update 与 frozen 子集支持
   ├─ 48527ae  stop instrumentation
   ├─ 15c3193  instrumentation import/selection metadata 修正
   ├─ 19bb979  evaluator package precedence 修正
   └─ 6adbb49  frozen action chunk 2/3 控制
```

## 9. 一句话工作流

```text
scripts选择episode
→ episode_subset严格锁定运行对象
→ eval_ttt运行A/B，eval.py运行C
→ stop_trace记录逐动作证据
→ aggregate严格验证和聚合结果
→ tests用CPU验证所有控制开关没有隐藏副作用
```

准确地说，本项目修改了 evaluator 与实验控制基础设施，但没有重新设计模型核心算法。新增的行为差异都通过显式开关控制，默认路径尽量保持历史兼容。
