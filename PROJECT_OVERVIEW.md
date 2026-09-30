# Test-Time Training for VLN：项目全貌与进展总览

> 更新日期：2026-09-30（Asia/Shanghai）  
> 证据覆盖：仓库提交历史、当前工作区、锁定实验包，以及截至 2026-09-23 生成的原始运行与验收产物。  
> 本文面向第一次接触本项目的读者，是当前唯一按“研究问题—方法—演进—结果—结论—复现”串起来的单文件总览。

## 1. 一页摘要

本项目研究视觉语言导航（Vision-and-Language Navigation，VLN）中的测试时训练：机器人拿到自然语言指令和当前视觉观察后，不直接照基础模型的单次输出行动，而是在每个新的决策点生成多个动作候选，根据候选之间的轨迹共识和基础模型置信度构造局部伪专家，并尝试在当前 episode 内更新 LoRA，使后续决策更好。

基础策略是 NaVIDA，评测任务是 R2R-VLNCE `val_unseen`。核心约束是：正式方法不能使用测试时目标位置、最短路径或专家动作作为 reward；这些信息只允许用于评测或明确标记的离线诊断。

截至目前，最重要的结果是：

- canonical frozen NaVIDA（C0）在完整 1,839 条 `val_unseen` 上为 **1,075/1,839，SR 58.46%**；在项目反复使用的 600 条开发集上为 **347/600，SR 57.83%**。
- 早期 A5（K=3、T=1.5、全随机候选、在线更新）在同一 600 条上只有 **305/600，SR 50.83%**，显著落后于同 ID C0，但当时多项协议同时不同，不能把退化单独归因于在线更新。
- A5.2-K3 修正“gate 拒绝后仍执行被拒候选”的行为，并保留一个低温候选作为 fallback，达到 **353/600，SR 58.83%**，回到 baseline 附近；相对同 ID C0 仅 +1.00 个百分点，区间跨 0。
- 最新 A5.2-K8 更新组 A 达到 **359/600，SR 59.83%**；相对 K3 仅 +1.00 个百分点，相对同 ID C0 +2.00 个百分点，两者都没有统计上明确的优势。
- 最新同协议禁用更新组 B 达到 **365/600，SR 60.83%**。A−B 为 **−1.00 个百分点**，paired bootstrap 95% CI 为 **[−4.00, +1.83]**，exact McNemar `p=0.581`。因此，当前证据**没有证明 episode 内 optimizer 更新带来收益**，也没有证明它必然有害。

当前最稳妥的总体判断是：**候选生成、可靠性 gate 和 fallback 选择已经把系统从早期退化状态恢复到 baseline 附近，但 LoRA 在线更新的边际价值仍未被证明。** 所有 A5.2 结果都来自同一 600 条开发集，尚无独立确认集结果。

## 2. 项目在解决什么问题

### 2.1 VLN 任务

VLN 要求智能体在未见过的室内环境中，根据自然语言指令和第一视角视觉观察连续执行离散动作，例如：

- `STOP`；
- 向前 25 cm；
- 左转 15°；
- 右转 15°。

本项目使用 Habitat 中的 R2R-VLNCE `val_unseen`。配置中的成功半径为 3 m。模型每次输出文本动作块，evaluator 最多解析并执行前两个逗号分隔的动作 chunk；一个 chunk 可能展开为若干原子动作。动作块进入 pending queue 后会逐个执行，期间不会重新生成候选或再次更新。

### 2.2 为什么做测试时训练

基础 NaVIDA 在测试时通常只生成一个动作方案。项目的出发点是：同一观察下多采样几个方案，如果若干方案在局部运动轨迹上彼此接近，它们可能形成一个无需人工标签的“局部伪专家”。系统可以：

1. 用这个局部共识选择更可靠的动作；
2. 用伪专家与其他候选之间的相对关系更新 LoRA；
3. 让更新只影响当前 episode 后续的决策；
4. 在下一个 episode 开始时恢复同一个初始 adapter，避免跨场景污染。

这是一项经验研究，而不是已被证明成立的前提。“候选相似”“模型置信度高”和“实际导航方向正确”不是同义词，必须通过对照实验区分。

### 2.3 项目的核心研究问题

项目目前实际在回答四个问题：

1. 多候选的局部轨迹共识能否比单候选 NaVIDA 做出更好的动作选择？
2. confidence、轨迹一致性和 gate 能否筛出足够可靠的伪专家？
3. 在相同候选和动作选择下，episode-local LoRA 更新是否比不更新更好？
4. 候选数量从 K=3 增加到 K=8，是否改善候选覆盖、排序和最终导航性能？

截至目前，第 3、4 个问题都没有获得肯定证据；第 1、2 个问题显示出恢复 baseline 性能的潜力，但仍缺独立确认。

## 3. 系统组成与数据流

项目不是从头训练一个 VLN 模型，而是在既有 NaVIDA 推理路径旁增加 episode-local 的候选生成、共识选择、LoRA 更新、审计和严格评测设施。

```text
语言指令 + 历史视觉 + 当前观察
                 │
                 ▼
        NaVIDA / Qwen2.5-VL + LoRA
                 │
                 ▼
      顺序生成 K 个文本动作候选
                 │
                 ▼
  解析前两个 action chunks，展开原子动作
                 │
                 ▼
 构造局部虚拟轨迹 (x, y, θ, stopped)
                 │
                 ▼
 DTW 共识 + 冻结 adapter confidence + 反转惩罚
                 │
          ┌──────┴──────┐
          ▼             ▼
   可靠 cluster      gate 拒绝
   选 cluster 内动作   执行低温候选 fallback
          │
          ▼
  若更新合格：cluster 对 outsiders 的 pairwise loss
          │
          ▼
  A：optimizer.step       B：跳过 optimizer.step
          │
          ▼
  执行更新前已经选好的动作；更新只影响下一次生成
```

### 3.1 模型与 LoRA

- 基础 checkpoint：`/hdd3/wangziyu/models/NaVIDA`。
- evaluator 加载 `Qwen2_5_VLForConditionalGeneration`。
- LoRA 配置为 `r=8`、`alpha=8`、dropout `0.05`，作用于语言模型的 attention/MLP 投影层，显式排除 visual 模块。
- 基础模型和视觉编码器冻结，只允许 LoRA 参数训练。
- 每个 episode 开始时恢复初始 LoRA tensor、重建 AdamW、清空历史与 pending queue，并按 `seed + episode_id` 派生随机状态。
- 多 GPU 只是把互斥 episode 分片交给多个独立单进程 worker；没有 DDP、梯度同步或跨 worker 参数共享。

### 3.2 候选与虚拟轨迹

候选文本被解析为 `STOP / FORWARD / LEFT / RIGHT`。每个非 STOP chunk 最多展开为 3 个原子动作。虚拟轨迹只根据动作几何在局部坐标中展开，不实际分叉运行 Habitat，也不建模碰撞。

当前 `se2_stop` 轨迹点为 `(x, y, θ, stopped)`：

- 前进改变 `(x,y)`；
- 转向改变并环绕归一化 `θ`；
- STOP 把 `stopped` 设为 1 并终止虚拟轨迹；
- 点距离综合平移、圆周角差和 STOP 状态差；
- 候选轨迹之间用 FastDTW 距离衡量相似性。

正式 reward 不读取目标点、专家动作、最短路径进度或 success 标签。

### 3.3 confidence 与 quality

为了避免在线更新后的模型给自己制造越来越高的置信度，候选 confidence 使用 **episode 初始 adapter** 重新计算。A5.1 之后只统计实际可执行前缀的 action-token 平均 log-probability：最多前两个 chunk，遇 STOP 提前结束。

候选内 confidence 做 min-max 归一化，再减去相邻 LEFT/RIGHT 反转惩罚，得到 quality。当前综合选择分数为：

```text
combined = 0.75 × consensus_reward + 0.25 × quality
```

confidence 是描述性模型信号，不是导航正确性的 oracle。早期 128 条 B 审计中，confidence 与 consistency 的候选内相关约为 `r≈0.259`，不能称为高度冗余，也不能据此称其为独立正确性信号。

### 3.4 K=3 与 K=8 的 cluster gate

K=3 时：

- 找 DTW 距离最近的两个候选作为 pair，第三个作为 outlier；
- cohesion 是 pair 内距离；
- separation 是 pair 到 outlier 的平均距离；
- 要求 `cohesion / separation <= 0.5`，且 cluster 平均 quality 不低于 outlier。

K=8 时：

- 枚举大小为 2–7 的 246 个候选子集；
- cohesion 使用子集内部最大距离，即 complete-link；
- separation 使用子集成员到外部候选的平均距离；
- 先在满足几何阈值的子集中优先选最大 cluster，再按 ratio、cohesion 和索引顺序打破平局；
- 几何 cluster 选定后再做 quality veto，不因 quality 失败而寻找另一 cluster。

gate 可信时从 cluster 内按 combined score 选动作；gate 因质量、分离度、候选有效性或 action span 等原因拒绝时，A5.2 执行 candidate 0 作为 canonical fallback。达到每 episode 更新上限只阻止继续更新，不否定已经找到的 cluster，因此仍执行 cluster 内动作。

### 3.5 在线更新与 no-update 对照

更新合格时，当前 K=8 loss 提高 cluster 内候选的平均 action-prefix log-probability，相对于全部 cluster 外候选的平均 log-probability：

```text
loss = -log sigmoid(beta × (mean_logp_cluster - mean_logp_outsiders))
```

其中 `beta=1`。每个 episode 最多 5 次合格更新机会，学习率 `2e-6`，weight decay 为 0，gradient norm 上限为 1。anchor L2 和 adapter drift 阈值已有代码支持，但当前正式实验均设为 0；reference-policy KL 未实现。

- A 组执行 forward、backward、gradient clipping 和 `optimizer.step()`。
- B 组执行相同候选生成、gate、选择、loss、backward 和 clipping，但通过 `--disable-update` 跳过 `optimizer.step()`，并逐决策验证 LoRA checksum 不变。
- C0 是单候选 frozen NaVIDA，不经过上述多候选路径。B 不等于 C0。

K=8 首次运行曾因八个 trainable forward 的计算图同时保留而在 48 GiB GPU 上 OOM。之后仅对 `K>3` 的训练前向启用 PyTorch activation checkpointing，并保留 dropout RNG；反向时重算激活，未改变 loss 定义。

## 4. 实验组命名：A、B、C 到底是什么

| 名称 | 候选/选择 | 是否 backward | 是否 optimizer step | 用途 |
|---|---|---:|---:|---|
| C0 canonical | 单候选、官方 NaVIDA 输入与前两个 chunks | 否 | 否 | 当前可信 frozen baseline |
| C_project | 历史 frozen evaluator，默认行为与 canonical 不完全一致 | 否 | 否 | 历史记录，不能冒充 C0 |
| B / disable-update | 与对应 A 相同的多候选、reward、gate、loss | 是 | 否 | 隔离参数更新的边际作用 |
| A / TTRL | 多候选、reward、gate、loss | 是 | 是 | episode-local 在线更新主方法 |

“A 比 C0 好/差”同时混入了候选数、采样温度、动作选择和在线更新等因素；只有相同协议的 A/B 才更接近回答 optimizer 更新本身的作用。

## 5. 方法版本演进

| 阶段 | 核心变化 | 主要边界 |
|---|---|---|
| 初始 full TTRL | 多候选轨迹共识、medoid、逐决策 LoRA 更新 | 早期输入和 baseline 协议尚未完全对齐 |
| A4 | K3 全 stochastic；XY 轨迹；cluster/quality gate；action-token loss | 不含朝向和 STOP 几何 |
| A5 | 轨迹改为 `se2_stop`；标准 named profile 为 greedy + stochastic | 后续 T=1.5 全 stochastic 运行是显式 custom 协议，不等于 profile 默认值 |
| A5 T1.5 | K3、三个 T=1.5 stochastic 候选、最多 5 次更新 | 600 条上明显落后 C0，且缺匹配 B600 |
| A5.1 | confidence 和 loss 只看实际执行的前两个 chunks/STOP 前缀 | 单独没有恢复性能 |
| A5.2-K3 | candidate0 改为 sampled T=0.2；gate 真正控制执行；拒绝时回退 candidate0 | 同时改变 gate 执行和候选组成，不是单因素实验 |
| A5.2-K8 | 1 个 T=0.2 + 7 个 T=1.5；cluster 泛化为 complete-link 子集选择；activation checkpointing | K 与 cluster 算法同时变化；仍是同一开发集 |

## 6. 从项目开始到现在的时间线

### 2026-08-12 至 08-16：建立 episode-local TTRL

- 08-12 创建最初版本和论文框架。
- 08-15 从旧的多卡/持续适应设想转为当前主协议：每个 episode 独立 reset，episode 内每个模型决策点更新，多卡仅并行不同 episode。
- 08-16 修复 teacher-forcing forward，加入 nDTW 配置和 episode ID 规范化。

### 2026-08-17 至 08-20：小规模 pilot、A/B/C 控制与 STOP 日志

- 建立确定性的 representative20 和扩展 representative72 selection。
- 增加 `--disable-update` 和 frozen 控制，开始区分 A、B、C。
- 增加逐原子动作 STOP trace、严格 episode 子集和结果聚合。
- 早期结果：A representative20 为 8/20；A representative72 为 25/72；旧 B72 为 22/72；旧 C_project72 为 15/72。
- 历史 confirmatory128 中 full TTRL A 为 26/128，旧 C_project 为 27/128。由于 baseline 和输入协议后来被发现存在混杂，这些数字只保留为历史，不代表当前方法能力。

### 2026-08-20 至 08-27：STOP 诊断与 canonical baseline 对齐

- 对 18 条 STOP 相关 episode 做多轮诊断、salvage 和配对分析。
- 明确 frozen evaluator 的执行 chunk 数差异，并增加显式的 2/3 chunks 控制。
- 新建 `eval_official_subset.py`，沿官方 NaVIDA 输入/解析谱系建立 canonical C0。
- 增加 official-protocol TTRL、native LoRA 注入和视觉输入 cardinality 修复。
- 历史 C0-128 跨原运行和 continuation 合计 72/128；official-protocol 历史 TTRL 只完成 48 条、成功 27 条，不能报告为完整 128。

### 2026-08-28 至 09-03：从 reward ablation 走向 A4/A5

- 依次加入 confidence gate、朝向感知 reward、混合采样、cluster/quality gate。
- 08-31 固化 named A5 profile；09-02 锁定 new128 selection，并记录 frozen candidate confidence。
- new128 上：C0 74/128，A4 72/128，R13 A5 71/128。该 new128 随后反复用于开发，不能再称独立确认集。
- confidence48 为 29/48，但它是 new128 的前缀且拓扑不同，不能单独形成确认性结论。

### 2026-09-09 至 09-14：T1.5 B128、A600 与全量 C0

- T1.5、K3、全 stochastic 的 B-only 完成 128 条：66/128，SR 51.56%，验收为 `PASS_WITH_PROTOCOL_AMENDMENT`；0 optimizer step、参数未变化。
- 同类 A 在扩大后的 600 条开发集上完成：305/600，SR 50.83%。
- canonical C0 完整 `val_unseen` 1,839 条完成：1,075/1,839，SR 58.46%；同 A 的 600 个 ID 上为 347/600，SR 57.83%。
- A5−C0 为 −7.00 个百分点，paired bootstrap 95% CI `[−11.33, −2.67]`，差距主要集中在 long 组；但这不是相同候选协议的 A/B，不能隔离 online update。

### 2026-09-15：A5.1 executed-prefix 修正

- 把 confidence 和 policy-loss mask 截断到实际会执行的动作前缀。
- 完整重跑得到 300/600，SR 50.00%。首次自动分析因 action-span token count 校验逻辑报错；后续 `analysis_corrected_v1` 对原始产物完成严格验收并标为 PASS。
- 结果说明，仅修正 token span 不足以恢复性能。

### 2026-09-16 至 09-17：A5.2-K3 修复 gate 执行

- 修复此前的关键语义问题：旧实现即使 reliability gate 拒绝 cluster，动作选择仍可能落在被拒的 pair 内。
- A5.2 让 gate 真正控制执行；拒绝时回退 sampled T=0.2 的 candidate0，另两个候选仍为 T=1.5。
- 结果为 353/600，SR 58.83%，相对 A5.1 +8.83 个百分点；相对同 ID C0 +1.00 个百分点，95% CI `[−2.50, +4.67]`，`p=0.661`。
- 三个 worker 均 exit 0。原 coordinator 因运行期间新增无关只读监控文件而在 clean-tree 检查退出 1；独立验收重做输入哈希和 episode 级检查后通过，保留该流程例外。

### 2026-09-18 至 09-20：A5.2-K8、OOM 与成功重跑

- 根据组会方向将候选数扩大为 8，并保留一个 T=0.2 低温候选和七个 T=1.5 探索候选。
- 首个四卡批次在第一个 episode 前后全部 CUDA OOM，0/600 有效完成；失败现场保留且未 resume。
- 加入 `K>3` activation checkpointing 后，以 GPU4/6/7 三个独立 worker、每卡 200 条完整重跑。
- 09-20 严格验收通过：359/600，SR 59.83%。相对 A5.2-K3 +1.00 个百分点，95% CI `[−3.17, +5.00]`，`p=0.689`；相对同 ID C0 +2.00 个百分点，95% CI `[−1.83, +5.67]`，`p=0.342`。

### 2026-09-21 至 09-23：同协议 K8 B-only

- 准备与 A5.2-K8 相同的 B-only 协议，唯一 evaluator CLI 差异是 `--disable-update`。
- 三个固定的 200-episode shard 由动态调度器分配给空闲 GPU；无自动重试。
- 09-23 三个 worker 全部 exit 0，严格验收通过：365/600，SR 60.83%，0 optimizer step，2,999 次合格机会均被 disable-update 抑制。
- 与历史非同期 K8 A 配对：A 359、B 365；A−B = −1.00 个百分点，95% CI `[−4.00, +1.83]`，`p=0.581`。

### 2026-09-24 至 09-30：仓库观察状态

截至本文整理，没有发现晚于 09-23 的新结果目录或新的 Git 提交。已有 `CURRENT_STATUS.md` 仍停留在 K8 A 运行中的 09-19 状态，因此本文补入了 09-20 的 K8 A 最终结果和 09-23 的 K8 B 最终结果。

## 7. 主要实验结果总表

下表按证据角色排列。不同协议之间不要只看成功率就做因果比较。

| 实验 | 数据 | Success / SR | 角色与结论边界 |
|---|---:|---:|---|
| historical A representative20 | 20 | 8/20 = 40.00% | 早期 pilot |
| historical A representative72 | 72 | 25/72 = 34.72% | 旧 full TTRL |
| historical B72 | 72 | 22/72 = 30.56% | 旧 no-update，不是当前 B |
| historical C_project72 | 72 | 15/72 = 20.83% | 非 canonical baseline |
| historical confirmatory A | 128 | 26/128 = 20.31% | 旧 full TTRL |
| historical confirmatory C_project | 128 | 27/128 = 21.09% | baseline 协议混杂 |
| historical canonical C0-128 | 128 | 72/128 = 56.25% | 跨 continuation 合并 |
| new128 C0 | 128 | 74/128 = 57.81% | 后来被用于开发 |
| new128 A4 | 128 | 72/128 = 56.25% | K3、XY、T0.3 |
| new128 R13 A5 | 128 | 71/128 = 55.47% | K3、se2_stop、T0.3 |
| T1.5 K3 B-only | 128 | 66/128 = 51.56% | 无同期匹配 A128 |
| A5 T1.5 K3 A | 600 | 305/600 = 50.83% | 比同 ID C0 低 7 pp |
| canonical C0 same600 | 600 | 347/600 = 57.83% | 600 开发集参考线 |
| canonical C0 full | 1,839 | 1,075/1,839 = 58.46% | 当前全量 frozen baseline |
| A5.1 K3 A | 600 | 300/600 = 50.00% | executed-prefix 修正 |
| A5.2 K3 A | 600 | 353/600 = 58.83% | gate 控制执行 + 低温 fallback |
| A5.2 K8 A | 600 | 359/600 = 59.83% | optimizer 开启；最新 A |
| A5.2 K8 B | 600 | 365/600 = 60.83% | optimizer 禁用；最新 B |

最新四个关键方案的更多指标：

| 指标 | C0 same600 | A5.2-K3 A | A5.2-K8 A | A5.2-K8 B |
|---|---:|---:|---:|---:|
| SR | 57.83% | 58.83% | 59.83% | 60.83% |
| SPL | 52.07% | 52.66% | 53.75% | 54.16% |
| OSR | 67.00% | 69.33% | 67.83% | 68.50% |
| nDTW | 未记录 | 67.47% | 66.27% | 67.53% |
| SDTW | 未记录 | 51.01% | 52.20% | 52.43% |
| DTG | 4.572 m | 4.313 m | 4.544 m | 4.326 m |
| Path Length | 10.530 m | 10.521 m | 10.731 m | 10.418 m |
| optimizer steps | 0 | 2,986 | 3,000 | 0 |

注：OSR 对应 `oracle_success`，表示轨迹是否曾进入成功半径，不等于最终正确 STOP；C0 原始产物没有 nDTW/SDTW，不做填补。A5.2-K3 的 SDTW 按逐 episode `success × nDTW` 推导。

## 8. 目前可以下什么结论

### 8.1 已被证据直接支持

- episode 间 adapter 和 optimizer reset 已实现并有运行审计。
- 当前动作由更新前候选决定；optimizer 更新只影响同一 episode 的后续生成。
- pending 原子动作期间不重新生成、不重新评分、不更新。
- canonical C0 全量结果为 SR 58.46%，同 600 条为 57.83%。
- 早期 A5 T1.5 K3 在同 600 条上明显低于 C0。
- A5.2 的 gate-controlled execution 与低温 fallback 组合使 K3 结果恢复到 C0 附近。
- K 从 3 扩到 8 后，A 的 SR 只增加 1 个百分点，当前数据不能确认真实提升。
- 在最新 K8 协议下，B 的点估计高于 A 1 个百分点，但区间跨 0；当前数据不能确认 online update 有益或有害。
- A5.2-K8 A 和 B 都完成 600 条严格验收，所有 worker exit 0；A 有 3,000 次有效参数变化、B 为 0。

### 8.2 合理但尚未确认的解释

- A5 到 A5.2 的主要恢复更像来自“拒绝不可靠 cluster 后执行低温 fallback”这一动作选择修复，而不是在线更新本身；但 A5.2 同时改变了 gate 语义和 candidate0 采样，无法单独归因。
- K8 提供了更高的候选多样性：每次决策平均约 5.5 个唯一原子动作候选，但更大的候选池没有转化为明确的 SR 增益，问题可能从“覆盖不足”转向“排序/伪专家可靠性不足”。
- B 略高于 A 提示更新至少不是当前性能提升的必要条件，但这是点估计层面的线索，不是更新有害的证明。
- STOP 仍可能是主要瓶颈之一。K8 A 中仅 18.53% 的决策存在至少一个 STOP 候选；在成功半径内该比例为 46.99%，仍有一半以上的半径内决策没有 STOP 候选。

### 8.3 现在不能宣称的内容

- 不能宣称 A5.2-K8 显著超过 baseline。
- 不能宣称 K8 显著优于 K3。
- 不能宣称 online LoRA update 有效；也不能仅凭 B 点估计更高就宣称 update 必然退化。
- 不能把 A5.2 的改进单独归因于 gate、低温候选、cluster 算法或某一个 reward 分量。
- 不能把 same600、new128 或 confidence48 称为独立确认集；它们已经参与方法开发。
- 不能把历史 full TTRL、C_project、canonical C0 和当前 A5.2 混为同一协议。
- 不能把会议纪要、用户转述或旧论文草稿中的数字当作实验验收证据。

## 9. 数据、指标与统计边界

### 9.1 数据角色

- 完整评测集：R2R-VLNCE `val_unseen`，共 1,839 条。
- same600：short/middle/long 各 200 条，selection seed 43，SHA256 为 `850fe3f1eb2af02fe52e6f36c610a5f6c3e482b8a07a5c5a4f3996edae369066`。
- same600 包含 new128 全部 128 条，因此两者都属于开发证据。
- 历史 confirmatory128 与 new128 是不同 selection；文件名中的 `independent` 或 `confirmatory` 不自动保证当前仍具独立性。

### 9.2 指标解释

- SR：最终是否成功停止在目标 3 m 半径内。
- OSR：导航过程中是否曾进入成功半径，不要求最终 STOP 正确。
- SPL：成功率与路径效率的联合指标。
- nDTW：执行轨迹与参考路径的归一化相似度。
- SDTW：以成功条件加权的 nDTW。
- DTG：episode 结束时到目标的距离，越低越好。
- Path Length：实际导航路径长度。

### 9.3 统计解释

同一 episode ID 的两种方法应使用 paired 分析：报告 discordant pairs、exact McNemar 和 episode-level paired bootstrap。区间跨 0 时不能称为明确优于或劣于。当前 K8 A/B 虽然使用同 600 条、相同算法和 seed，但并非同期同 GPU 运行，报告中已明确标为 historical nonconcurrent comparison。

## 10. 代码和目录怎么读

```text
README.md / CURRENT_STATUS.md      旧的项目入口与状态摘要；后者未纳入最终 K8 A/B
PROJECT_OVERVIEW.md                本文；截至 2026-09-30 的综合总览
docs/
  method/CURRENT_METHOD.md         A4/A5 边界；早于最终 A5.2-K8
  experiments/                     复现、结果索引、A5/C0 与 A5.2 验收说明
  meetings/                        会议方向，不是实验事实
  archive/pre_a5/                  早期计划、论文框架和旧说明
experiments/
  official_navida_*/               canonical C0 与历史 official TTRL 锁定包
  preparations/                    A5.2/K8/A/B 的命令、manifest、hash 与验收代码
scripts/
  launch/                          A5.2-K8 准备、协调和调度入口
  maintenance/                     静态检查与只读进度工具
ttrl-src/
  eval/eval_ttt.py                 A/B episode-local evaluator 与 profile
  eval/eval_official_subset.py     canonical C0 evaluator
  eval/stop_trace.py               逐动作 STOP/执行证据
  eval/aggregate_ttt_results.py    严格聚合
  trainer/trajectory_consensus_updater.py
                                   候选、DTW、gate、loss、reset、optimizer
  config/vln_r2r_ttrl.yaml         Habitat/R2R/nDTW 配置
  tests/                            CPU 单元测试
```

最值得直接阅读的实现是：

1. [`ttrl-src/eval/eval_ttt.py`](ttrl-src/eval/eval_ttt.py)：完整导航循环、A4/A5/A5.2 profile、episode reset 和结果记录。
2. [`ttrl-src/trainer/trajectory_consensus_updater.py`](ttrl-src/trainer/trajectory_consensus_updater.py)：候选生成、轨迹 reward、confidence、cluster gate、loss、更新和 rollback。
3. [`ttrl-src/eval/eval_official_subset.py`](ttrl-src/eval/eval_official_subset.py)：canonical C0。
4. [`docs/experiments/a5_2_acceptance_20260917/README.md`](docs/experiments/a5_2_acceptance_20260917/README.md)：A5.2-K3 的独立验收。
5. 最新 K8 A/B 的原始 report 和 paired analysis，见下一节。

## 11. 当前机器上的关键路径

| 用途 | 路径 |
|---|---|
| 本仓库 | `/hdd3/wangziyu/code/test-time-training-vln` |
| 外部 NaVIDA | `/hdd3/wangziyu/code/Self-TTRL4VLN/NaVIDA` |
| Habitat 代码 | `/hdd3/wangziyu/code/habitat-lab` |
| Python 环境 | `/hdd3/wangziyu/envs/navida/bin/python` |
| NaVIDA checkpoint | `/hdd3/wangziyu/models/NaVIDA` |
| R2R 数据与 GT | `/hdd3/wangziyu/data/R2R_VLNCE_v1-3_preprocessed/val_unseen` |
| 场景数据 | `/hdd3/wangziyu/data/scene_dataset` |
| 仓库外实验产物 | `/hdd3/wangziyu/runs/test-time-training-vln` |

真实 evaluator 必须以外部 NaVIDA 根为工作目录，因为 YAML 中的 `data/...` 使用该目录的软链接语义。源码仓库没有完整可移植的环境锁，不能只凭环境名复现实验。

### 最新 K8 A 证据

- 结果根：`/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01`
- [严格 report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)
- [validation](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/validation.json)
- [相对 K3 配对分析](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/paired_vs_a5_2_k3.json)
- [相对 C0 配对分析](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/paired_vs_canonical_c0.json)

### 最新 K8 B 证据

- 结果根：`/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01`
- [严格 report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/report.json)
- [validation](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/validation.json)
- [A−B 配对分析](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/paired_A_minus_B.json)

## 12. 复现与版本边界

### 12.1 当前 Git 状态不是完整的 K8 版本号

本文整理时：

- 当前分支为 `codex/a5.1-executed-span-20260915`；
- HEAD 为 `af8e58b5d04cf384f576981ed5167186eec7a438`；
- 工作区包含尚未提交的 A5.2-K8 cluster 泛化、activation checkpointing、测试、启动器、准备包和文档。

因此，不能只 checkout `af8e58b` 就声称得到 K8 实际运行代码。K8 A 和 B 的准备包都记录了相同的 runtime diff SHA256：

```text
ce1da2fdc6288d9f1ac314e29fb8abbed7ee2bc207fc66f08a4d5475b5a05e19
```

准确复现必须同时核对 base commit、runtime source hashes/diff、selection、模型、数据、环境、GPU/worker 拓扑和实际展开参数。

### 12.2 锁定包的原则

每个正式运行包把以下内容放在一起：

- `selection*.json`：固定 episode ID 和顺序；
- `protocol_manifest.json`：方法、参数、分片和比较边界；
- `analysis_plan.json`：验收和统计计划；
- `inputs.lock.json` / `package.lock.json`：源码、环境和资产哈希；
- launcher/coordinator/audit worker：启动和运行时审计；
- evaluator 展开参数和 GPU UUID。

旧锁定包是证据，不应修改成适配当前 HEAD。新实验应创建新的准备包和新的结果目录；失败批次保留现场，不在原目录 resume、覆盖或自动挑结果重跑。

### 12.3 CPU 检查与真实运行不是一回事

仓库提供 layout 检查和 CPU 单元测试，用假模型验证 selection、解析、gate、trace、聚合和协议开关。A5.2-K8 准备阶段记录了 104 项 CPU 测试通过，但本文整理没有重新运行测试。CPU 测试不加载完整场景，不能替代 GPU/Habitat 端到端验收。

真实运行入口和 cwd/PYTHONPATH 规则见 [`docs/experiments/REPRODUCTION.md`](docs/experiments/REPRODUCTION.md)。不要直接把历史 smoke、旧 `C_project` 或 `official_navida_ttrl_a128` 当成当前 A5.2 入口。

## 13. 已知问题与研究缺口

### 方法层面

- 没有强制 STOP 候选。
- 没有独立 progress、waypoint 或 stop critic。
- 没有 reference-policy token KL。
- 没有 delayed post-observation update。
- pending queue 期间不逐原子动作重规划。
- anchor L2 和 drift threshold 虽有实现，但正式 A5.2 实验没有启用。
- 当前轨迹是无碰撞的局部动作几何，不是环境中的真实分支 rollout。

### 证据层面

- same600 已被多轮开发使用，存在过拟合方法选择的风险。
- K8 A/B 是非同期运行，尽管输入和协议被严格锁定，仍存在 GPU、时序和运行环境差异。
- K8 与 K3 的比较同时改变候选数和 K>3 cluster 泛化，不能称为纯 K 消融。
- A5.2 相对 A5.1 同时改变 gate 执行和低温 fallback，不能称为纯 gate 修复消融。
- 一般导航的 oracle candidate recall@K、oracle-best upper bound、全导航 gate precision 尚未完成。
- instruction/vision ablation、entropy minimization 对照和独立重复尚未完成。
- 目前没有在冻结方法后使用未参与开发的数据做确认实验。

### 工程层面

- 最新 K8 算法仍是 base commit 加未提交 diff，版本管理风险较高。
- 根 `README.md`、`CURRENT_STATUS.md`、`docs/method/CURRENT_METHOD.md` 和 `RESULTS_INDEX.md` 均早于最终 K8 A/B 结果。
- 原始结果在仓库外，迁移项目时必须同时保护 runs 目录、锁定包和外部依赖。
- 没有一个完整的、可跨机器安装的依赖锁文件。

## 14. 推荐的下一步

以下是基于现有证据的研究优先级，不代表已经获准执行的新实验。

1. **先冻结当前方法与代码版本。** 把已运行的 K8 源码变成可引用的明确提交，同时保留并核对现有 runtime diff/hash，避免“结果对应哪个代码”继续依赖脏工作区。
2. **把最新 K8 A/B 纳入正式状态与结果索引。** 当前总状态停在运行中，容易让新成员误判项目尚未完成 K8。
3. **不要继续在 same600 上追逐小幅 SR。** K8 A/K3 和 A/B 的区间都跨 0；继续反复调参会进一步削弱该开发集的确认价值。
4. **做最小候选覆盖与排序诊断。** 在开发集上区分“候选池没有合适动作”和“已有合适候选但 gate/quality 排错”，oracle 仅用于离线诊断，绝不进入正式 reward。
5. **若保留在线更新贡献，必须设计更强的 A/B 证据。** 当前 A−B 点估计为负且不显著；应先解释更新何时改变后续候选、哪些 episode 获益/受损，再决定是否值得新的独立确认运行。
6. **最后才做独立确认。** 冻结方法、统计计划、selection 和运行协议后，在未用于开发的数据或预先声明的完整评测上运行，避免把同一 600 条继续包装成确认实验。

## 15. 新成员建议阅读顺序

如果只想在 20 分钟内了解项目：

1. 先读本文第 1、2、5、7、8、13 节；
2. 再看最新 K8 A/B 的三个 JSON report/paired 文件；
3. 阅读 `eval_ttt.py` 的 `TTRL_PROFILES` 和 `trajectory_consensus_updater.py` 的 `_cluster_gate()`、`adapt_step()`；
4. 需要复现实验时再读锁定 package，而不是从旧 README 中拼命令。

如果要写论文或汇报，建议使用以下一句话概括当前阶段：

> 项目已经构建并严格验收了 episode-local、多候选轨迹共识的 VLN 测试时适应系统；A5.2 的可靠性 gate 和低温 fallback 将性能恢复到 canonical NaVIDA baseline 附近，但 K=8 相对 K=3 的提升以及在线 LoRA 更新相对 no-update 的收益均未被当前 600 条开发集证据确认。

## 16. 证据标签与维护约定

本文采用以下语义：

- **VERIFIED**：由源码、Git、原始 stats、锁定 manifest 或验收产物直接支持。
- **DOCUMENTED**：旧文档记载，但本次没有从原始产物重新验收。
- **INFERENCE**：根据证据作出的解释，不是实验直接输出。
- **MEETING DECISION**：会议提出的方向，不是结果。
- **UNKNOWN**：当前材料不足，禁止补成已完成或已验证。

今后更新本文时，至少应同时核对：最新 Git 状态、`/hdd3/wangziyu/runs/test-time-training-vln` 的新结果目录、每次运行的 `execution_status.json`、`analysis/validation.json`、`analysis/report.json`、配对分析和对应准备包。目录存在、`SUCCESS` 文件名、会议陈述或用户转述都不能单独代替验收。
