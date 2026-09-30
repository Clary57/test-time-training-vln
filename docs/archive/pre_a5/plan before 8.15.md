# 单卡、逐 Episode 重置、逐推理步更新的 TTRL 改造计划

## 一、方案结论

将原来的“多条测试流分别持续适应”修改为：

> 完全复用 evaluation 的逐 episode 导航逻辑。每个 episode 开始前，将 LoRA 参数和优化器恢复到同一个 base checkpoint 状态；episode 内每次模型推理产生候选动作后，立即根据轨迹共识损失执行一次反向传播和优化器更新；更新只影响该 episode 后续的推理。不同 episode 之间不继承任何适应结果。

多卡仅用于并行评测不同 episode：

```text
GPU 0：episode 子集 0，每个 episode 独立 reset → step-wise TTT
GPU 1：episode 子集 1，每个 episode 独立 reset → step-wise TTT
GPU 2：episode 子集 2，每个 episode 独立 reset → step-wise TTT
GPU 3：episode 子集 3，每个 episode 独立 reset → step-wise TTT
```

各进程不建立分布式进程组，不使用 DDP、DeepSpeed、Accelerate 多进程或梯度同步。由于每个 episode 都从相同参数开始，数据分片只是并行加速手段，不再代表不同的持续适应测试流。

## 二、核心执行协议

每个 episode 严格执行：

```text
恢复初始 LoRA 参数
重新创建优化器，清空 Adam 动量等状态
清空视觉历史、待执行动作和环境状态
        ↓
根据当前观察采样 G 个动作块
        ↓
解析候选动作并计算轨迹共识奖励
        ↓
计算组相对策略梯度损失
        ↓
zero_grad → backward → gradient clipping → optimizer.step
        ↓
执行本次采样得到的 medoid 动作块
        ↓
使用更新后的模型处理该 episode 的下一个决策点
        ↓
episode 结束并记录指标
        ↓
下一个 episode 再次恢复初始状态
```

更新粒度定义为“每次模型推理后更新一次”，不是每个 Habitat 原子动作更新一次。一个动作块中的 pending actions 直接依次执行，不重复推理和更新。

第 \(e\) 个 episode 的参数生命周期写为：

\[
\phi_{e,0}=\phi_{\text{base}},
\qquad
\phi_{e,t+1}
=
\phi_{e,t}-\eta\nabla_\phi\mathcal{L}_{e,t}.
\]

当前决策点的候选动作由更新前的 \(\phi_{e,t}\) 产生；该步更新后的 \(\phi_{e,t+1}\) 只用于下一次推理，避免用更新后的模型重新评价或重新选择当前动作。

## 三、代码改造

### 1. 以 evaluation 循环作为主入口

在 `eval/eval_ttt.py` 中实现真正的 TTT evaluator：

- 保留现有 `for episode → while not episode_over → agent.act → env.step` 结构。
- 将 `agent.reset()` 拆成：
  - `reset_episode_state()`：清空图像历史、pending actions、视频状态；
  - `restore_base_model()`：恢复初始 LoRA、重建优化器。
- 两个 reset 必须在 episode 第一次推理前完成。
- `agent.act()` 在需要新动作块时执行“生成—奖励—反向传播—optimizer.step—返回 medoid 动作”；存在 pending action 时只返回动作。
- episode 结束后立即保存导航指标，不使用适应后的模型重新运行该 episode。
- 已完成 episode 可直接跳过；采用 episode-ID 派生随机种子，使断点续跑和不同 GPU 分片尽可能得到一致结果。

### 2. 重构 TTRL 计算模块

从 `trainer/test_time_trainer.py` 中提取并复用以下纯计算能力：

- 多候选生成及 completion mask；
- 动作文本解析与离散动作块转换；
- FastDTW 距离矩阵、medoid 动作和组相对奖励；
- completion log-probability；
- advantage、有效候选 mask 和策略梯度损失。

新的在线更新接口固定为：

```python
result = updater.adapt_step(prompt_inputs)
```

返回：

```python
{
    "selected_actions": list[int],
    "loss": float,
    "rewards": list[float],
    "reward_std": float,
    "valid_count": int,
    "timing": {
        "generation": float,
        "loss_forward": float,
        "backward": float,
        "optimizer": float,
    },
}
```

同时修复奖励计算中使用 `G` 遍历有效轨迹数组的问题：距离矩阵循环必须使用有效候选数量 `V`，防止候选解析失败后越界。

旧 `TTTRLTrainer.training_step()` 不再作为论文主实验入口，因为 Hugging Face Trainer 只会在整个 episode 返回后调用外层 optimizer step，无法满足逐推理步更新。

### 3. 模型和优化器重置

模型初始化规则：

- 只训练语言模型侧 LoRA，视觉编码器和 base 权重全部冻结；
- 若指定初始 LoRA checkpoint，则以该 checkpoint 作为每个 episode 的起点；
- 未指定时，创建一份新 LoRA，并将其初始化状态作为起点；
- 不再 `merge_and_unload()`，否则无法只恢复和更新 LoRA。

初始化完成后，将所有可训练 LoRA tensor 克隆到 CPU：

```python
base_adapter_state = {
    name: parameter.detach().cpu().clone()
    for name, parameter in model.named_parameters()
    if parameter.requires_grad
}
```

每个 episode 开始时原位复制回来，并重新创建 AdamW。禁止只清梯度而保留上一 episode 的参数或优化器动量。采用内存快照恢复，避免每个 episode 从磁盘重新加载完整模型。

### 4. 命令行与启动方式

为 `eval/eval_ttt.py` 增加或统一参数：

```text
--model-path
--lora-path
--learning-rate
--weight-decay
--max-grad-norm
--num-generations
--temperature
--split-num
--split-id
--seed
--result-path
```

强制检查：

- `WORLD_SIZE` 不得大于 1；
- `torch.distributed.is_initialized()` 必须为 false；
- 单进程 batch size 固定为一个 episode；
- 每个进程使用独立结果目录，如 `result/split_0`；
- 不加载 DeepSpeed 配置。

四卡启动为四个普通 Python 进程，例如：

```bash
CUDA_VISIBLE_DEVICES=0 python eval/eval_ttt.py --split-num 4 --split-id 0 ...
CUDA_VISIBLE_DEVICES=1 python eval/eval_ttt.py --split-num 4 --split-id 1 ...
CUDA_VISIBLE_DEVICES=2 python eval/eval_ttt.py --split-num 4 --split-id 2 ...
CUDA_VISIBLE_DEVICES=3 python eval/eval_ttt.py --split-num 4 --split-id 3 ...
```

不使用 `torchrun`。每个进程只看见一张物理 GPU，并在进程内使用逻辑设备 `cuda:0`。

### 5. 数据与结果汇总

- 固定并保存完整 episode ID 顺序及四个互斥分片，不再使用 `TTTRLDataset` 中无清单记录的 `random.shuffle`。
- 汇总前校验 episode ID 没有重复或遗漏。
- 从各分片的逐 episode 原始指标重新计算 SR、SPL、nDTW 等总体指标，不能简单平均分片均值。
- 每个决策点额外保存 loss、reward、有效候选数、参数更新前后 LoRA norm 和各阶段耗时。
- 最终模型不作为方法产物；最后一个 episode 的适应参数可以不保存，因为它对其他 episode 无效。

## 四、论文框架同步修改

更新 `VLN_TTA论文框架.md` 中所有协议描述：

- 将“episode-level continual test-time update”改为“episode-local step-wise TTRL”。
- 删除“参数跨 episode 持续保留”“不同机器人形成不同参数演化轨迹”等表述。
- 方法伪代码改为每个 episode 先执行 \(\phi\leftarrow\phi_{\text{base}}\)，每个决策点执行一次 optimizer step。
- 将多 GPU 描述为无通信的 episode-level evaluation parallelism，而不是多条持续适应流。
- 删除一条、两条、四条测试流的性能敏感性主张；理论上相同 episode 的起始参数不受分片数量影响，仅运行速度变化。
- 将持续适应曲线改为 episode 内随决策步变化的 reward、loss、参数距离和导航表现分析。
- 保留“跨 episode 持续更新”作为可选消融，而不是主方法。
- 贡献点改为“episode 内在线适应、episode 间恢复基础策略”，避免声称跨场景积累知识。
- 明确候选生成和反向传播仍是主要计算成本，取消通信只消除同步、等待和梯度交换。

主要对照固定为：

1. 冻结 NaVIDA，单候选，不更新；
2. 多候选 medoid 动作，不更新；
3. 多候选、逐推理步 TTRL、每 episode reset，主方法；
4. 多候选、episode 结束后更新；
5. 多候选、逐步更新但跨 episode 持续保留，作为协议消融。

## 五、测试与验收

### 正确性测试

- episode 开始前 LoRA 参数哈希与初始快照完全一致。
- optimizer 在 episode 开始时没有上一 episode 的动量状态。
- 非零 advantage 时，一次 `adapt_step()` 后至少一个 LoRA 参数发生变化。
- base 模型和视觉编码器在任何时刻均不发生变化。
- 同一 episode 的第二次推理使用第一次更新后的参数。
- 切换 episode 后，参数恢复到初始值，而不是上一 episode 的结束值。
- reward 全相等导致 advantage 为零时，参数不应产生有效更新。
- 存在无效候选时奖励计算不越界，且无效候选不贡献梯度。
- 当前 episode 的 success、distance-to-goal、SPL 等指标只用于记录，不参与奖励或参数更新。

### 无通信测试

- 进程中没有初始化 `torch.distributed`。
- 不调用 `accelerator.reduce`、`gather_for_metrics`、barrier 或广播。
- 暂停或结束一张 GPU 的进程不会阻塞其他进程。
- 每个进程退出时不等待其他分片。
- 使用 1 卡和 4 卡运行同一 episode，在相同随机种子下得到一致的候选序列、更新次数和指标。

### 集成与效率测试

使用两个短 episode 做端到端 smoke test，验证：

- 导航循环可以完成；
- 每个模型决策点恰好调用一次 optimizer step；
- pending action 不额外触发更新；
- episode 指标和 step 日志均正确落盘；
- 四个分片合并后覆盖完整测试子集；
- 对比旧同步实现的墙钟时间、每 episode 时间、生成时间、前向/反向时间、优化器时间、GPU 利用率和峰值显存。

## 默认假设

- “每步推理后优化”指每次生成动作块后优化一次，而不是每个 Habitat 原子动作后优化。
- base checkpoint 指“NaVIDA 基础权重加初始 LoRA 状态”；每个 episode 恢复同一适配起点。
- 主协议采用 episode 间完全 reset，不跨 episode、场景或 GPU 继承参数。
- 每张 GPU 同时只运行一个 episode。
- 当前动作仍执行本次更新前采样得到的 medoid；更新后的模型从下一次推理开始生效。
