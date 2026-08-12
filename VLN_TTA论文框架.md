# 基于轨迹共识的视觉—语言导航测试时强化学习：论文框架

> 本文档依据 `/nvme-ssd2/lixintao/projects/论文框架.md` 中的问答，以及 `/nvme-ssd1/zwy/navida/src` 的当前代码整理。它是一份可直接扩写成论文的写作提纲，不是最终论文正文。
>
> **事实标记约定**：
>
> - **[已实现]**：当前代码中已有对应实现；
> - **[待实验]**：需要补充实验后才能写成确定结论；
> - **[待确认]**：代码或协议尚有歧义，投稿前必须核实；
> - **[占位符]**：不可凭现有材料填写，应由最终实验结果替换。
>
> **术语边界**：上游 NaVIDA 是基于 VLN 与 inverse-dynamics 数据的监督微调方法，不使用强化学习；本文研究的是在该冻结基础策略之上新增的轨迹共识 Test-Time Reinforcement Learning（TTRL）。ATENA 属于带 episodic feedback 的熵优化式 Active TTA，不属于本文所采用的 TTRL 路线。

---

## 一、论文定位

### 1.1 暂定题目

中文题目建议：

1. **基于轨迹共识的视觉—语言导航测试时强化学习**
2. **无需外部反馈的视觉—语言导航测试时强化学习**
3. **从动作采样到策略更新：基于轨迹共识奖励的视觉—语言导航**

英文题目建议：

1. **Trajectory-Consensus Test-Time Reinforcement Learning for Vision-and-Language Navigation**
2. **Feedback-Free Test-Time Reinforcement Learning for Vision-and-Language Navigation**
3. **Adapting Vision-Language Navigation Policies at Test Time via Action-Trajectory Consensus**

建议暂用第一个英文题目。当前代码具备多 rollout 采样、标量伪奖励、组内优势和策略梯度更新，称为 TTRL 比笼统的 TTA 更准确；同时该标题不会过早承诺尚未验证的“实时”“稳定”或“显著提升”。

### 1.2 一句话概括

> We introduce test-time reinforcement learning for a NaVIDA-based navigation policy by sampling multiple action chunks, measuring their trajectory-level agreement, and optimizing lightweight LoRA parameters with a group-relative consensus reward, without ground-truth actions, goal locations, or human feedback.

中文表述：

> 本文以监督训练的 NaVIDA 为基础策略，在测试过程中对同一导航状态采样多个动作块，以候选动作轨迹之间的一致性构造组相对伪奖励，并通过策略梯度仅更新轻量 LoRA 参数，从而在不使用真实动作、目标位置或人工反馈的情况下实现测试时强化学习。

### 1.3 核心研究问题

论文应始终围绕以下问题展开：

> 在连续视觉—语言导航中，如果测试阶段既没有专家轨迹，也没有人工或成功反馈，能否利用策略自身多次采样产生的动作轨迹结构，为模型构造可靠的伪奖励并执行测试时强化学习？

进一步拆成三个 Research Questions：

- **RQ1：有效性。** 轨迹共识驱动的测试时强化学习能否优于冻结的 NaVIDA、仅执行共识动作的 Test-Time Scaling 基线和朴素测试时适应方法？
- **RQ2：机制。** 性能变化究竟来自多样化采样、轨迹级共识奖励、组相对优化，还是 LoRA 更新？
- **RQ3：稳定性与代价。** 该方法在长轨迹、连续 episode、不同采样数和不同分布偏移下是否稳定，额外时间和显存成本是多少？

### 1.4 论文主线

全文建议保持如下论证链：

```text
VLN 模型在训练后通常固定
        ↓
未知场景与分布偏移会削弱部署泛化
        ↓
普通 TTA 多面向独立感知样本，缺少序贯导航特有的无标签学习信号
        ↓
已有 VLN-TTA 往往依赖熵、快慢更新或 episodic feedback；通用 TTRL 则主要研究独立推理问题上的答案投票
        ↓
同一导航状态下，多次采样得到的动作可以转换为可比较的局部轨迹
        ↓
候选轨迹的“共识程度”可形成不依赖外部标注的内在奖励
        ↓
将轨迹共识作为伪奖励，通过组相对策略梯度更新 LoRA，并执行共识轨迹
        ↓
在严格的在线协议下验证 TTRL 收益、稳定性和计算代价
```

### 1.5 建议贡献点

在实验完成前，贡献应使用保守措辞：

1. **问题与框架贡献。** 将 Test-Time Reinforcement Learning 从独立语言推理扩展到生成式视觉—语言导航，建立“采样—伪奖励—策略更新—环境交互”的闭环框架。
2. **学习信号贡献。** 提出轨迹共识伪奖励：将多个生成动作块映射为局部运动轨迹，以轨迹间 FastDTW 距离衡量共识，将离散答案 majority voting 推广为结构化轨迹空间中的软共识。
3. **参数高效适应贡献。** 冻结视觉编码器和基础权重，仅对语言模型中的 LoRA 参数进行 episode 级持续更新，以减少可训练参数与优化器状态；这不代表整体计算高效，因为多候选生成与反向传播仍会显著增加时间和显存开销。
4. **实验贡献。[待实验]** 在 R2R、RxR 及受控分布偏移设置中，与冻结策略、通用 TTA 和 VLN 专用 TTA 进行比较，并系统分析奖励设计、更新协议和计算开销。

如果最终实验不能覆盖多个数据集或不能稳定超过直接基线，第四点应缩小，不应写“comprehensive”或“state-of-the-art”。

---

## 二、摘要框架

摘要建议采用“背景—缺口—方法—结果—意义”五句话结构，控制在 180～220 个英文词。

### 2.1 中文摘要草稿

> 视觉—语言导航智能体需要依据自然语言指令在未见环境中执行连续决策，但多数方法在离线训练后固定参数，难以利用部署过程中不断到来的观察适应测试分布。现有测试时适应方法主要面向独立感知样本，而导航中的动作会改变后续观察；同时，测试阶段通常缺少专家动作、目标位置或可靠的成功反馈。为此，本文提出一种基于**动作轨迹共识**的测试时强化学习方法。该方法以监督训练的 NaVIDA 为基础策略，在每个决策状态采样多个动作块，将其转换为局部运动轨迹，通过候选轨迹间的 FastDTW 距离构造组相对伪奖励，并使用策略梯度更新轻量 LoRA 参数；执行阶段选择与其他候选最一致的轨迹，从而形成无外部监督的 TTRL 闭环。在 [数据集] 上的实验表明，本方法相较冻结的 NaVIDA 将 SR/SPL 分别提升 [X]/[Y]，并在 [分布偏移设置] 下表现出 [待填写的核心结论]。进一步的消融与效率分析验证了 [轨迹共识奖励/共识动作选择/LoRA 策略更新] 的作用。结果说明，生成策略自身的轨迹分布可为序贯导航提供伪奖励，但共识本身并不保证动作正确。

### 2.2 摘要中不能提前写的内容

- 当前不能写“显著提升”“state-of-the-art”或具体百分比；现有 TTRL 日志只覆盖 R2R val-unseen 的 152 个 episode。
- 当前不能写“每一步在线更新”；现有训练器在一个 episode（多卡时为一批并行 episode）内部累计梯度，外层优化器在该训练 step 结束后才更新。
- 当前没有启用 KL/reference-model 约束，不能将其描述为现有稳定机制。
- 当前没有实现参数 episodic reset；代码表现为参数跨 episode 持续保留。

---

## 三、1. Introduction

建议写 5 段，正文约 1.5～2 页。

### 第 1 段：任务背景与泛化问题

写作目标：定义 VLN，说明连续环境中的泛化为何重要。

建议内容：

> Vision-and-Language Navigation (VLN) requires an embodied agent to interpret a natural-language instruction and navigate through a visually perceived environment. Recent agents have substantially improved multimodal reasoning and long-horizon control, yet deployment in unseen buildings, sensor conditions, and instruction distributions remains challenging.

引用位置：

- VLN 定义和 R2R：Anderson et al., 2018；
- 历史建模代表：HAMT；
- 基于视频/VLM 的连续 VLN：NaVid、NaVIDA。

### 第 2 段：NaVIDA 的进展与仍存限制

写作目标：交代直接基础工作，同时明确继承关系。

建议内容：

> NaVIDA introduces inverse-dynamics supervision and hierarchical probabilistic action chunking to learn action-grounded visual dynamics with a lightweight vision-language model. Its released training pipeline is supervised fine-tuning rather than reinforcement learning; like most offline-trained VLN policies, its parameters remain fixed during evaluation and cannot explicitly exploit the stream of observations encountered during deployment.

必须说明：

- 当前工作直接建立在 NaVIDA 代码和 checkpoint 之上；
- IDS、HPAC、Qwen2.5-VL 基础模型及原始导航提示不属于本文贡献；
- 本文新增部分是测试时多 rollout 采样、轨迹共识伪奖励、组相对策略梯度和 LoRA TTRL 闭环；不能把这些机制归入原始 NaVIDA 的贡献。

### 第 3 段：为什么普通 TTA 不够

写作目标：建立 research gap。

建议从以下差异展开：

- 图像分类 TTA 常假设样本相对独立，而 VLN 的动作决定下一时刻可见观察；
- 动作错误和错误更新可能共同累积；
- 生成式 VLN 的输出是可变长度动作文本或动作块，不是单一类别概率；
- 熵最小化可能强化已经错误但高置信的动作；
- 测试阶段通常拿不到专家动作、最短路径或终点坐标。

引用：TTA Survey、Tent、CoTTA/EATA、VLN-TTA 的 FSTTA 与 ATENA，以及使用 majority voting 伪奖励的 TTRL。

### 第 4 段：本文方法

建议用一个直观问题引入：

> Although no candidate action can be verified against an expert label, multiple stochastic predictions reveal the local structure of the policy's action distribution. We hypothesize that a candidate trajectory supported by many other samples is a more reliable self-training target than an isolated prediction.

然后概括四步：

1. 对同一状态采样 \(G\) 个动作块；
2. 转换为二维局部轨迹；
3. 用候选之间的轨迹距离构造共识奖励并选出 medoid 轨迹执行；
4. 用组内标准化优势更新 LoRA 参数。

此处应强调：共识是**可靠性代理信号**，不是“正确性证明”。如果所有候选共同犯错，共识仍可能失败，这应在局限性中承认。

### 第 5 段：贡献列表

采用第 1.5 节的三或四条贡献。最终结果出来后，再把第四点改成定量结论。

---

## 四、2. Related Work

### 2.1 Vision-and-Language Navigation

建议按照技术演进而非逐篇罗列：

1. 经典 VLN 与 R2R 任务定义；
2. Transformer 与长程历史建模，如 HAMT；
3. 基于视频、VLM/VLA 的连续环境导航，如 NaVid、NaVILA、StreamVLN；
4. NaVIDA 的 inverse dynamics supervision 与 action chunking。

本节结尾落到：这些方法主要提升离线学习或基础策略能力，本文研究部署期间如何利用无标签交互进行参数适应。

### 2.2 Test-Time Adaptation

建议分为三类：

- **熵最小化与选择性更新**：Tent、EATA；
- **持续测试时适应与抗遗忘**：CoTTA；
- **视觉—语言模型 TTA**：TDA 等。

本节不要暗示本文使用了 entropy、EMA teacher 或 memory bank；当前代码没有这些机制。可将本文区别写为：现有方法通常在预测空间约束置信度或一致性，本文在**生成动作所诱导的轨迹空间**中构造学习信号。

### 2.3 Test-Time Reinforcement Learning

重点讨论：

- **TTRL**：在无标签测试问题上采样多个推理结果，以 majority voting 产生伪标签或伪奖励，再通过强化学习更新策略；
- **本文扩展**：把独立问答中的离散答案投票推广到交互式 VLN，将动作文本投影为局部轨迹，并用 FastDTW 中心性形成连续的组相对奖励；
- **必要边界**：只做多数投票或 medoid 选择而不更新参数属于 Test-Time Scaling；使用伪奖励、advantage 和策略梯度更新后才构成本文所说的 TTRL。

本节还应说明：当前方法使用的是模型内部共识奖励，而不是 Habitat 的 success、distance-to-goal 或 SPL 任务奖励，因此属于自监督 TTRL，而不是传统的环境任务奖励导航 RL。

### 2.4 Test-Time Adaptation for VLN

重点讨论：

- **FSTTA**：通过 fast/slow 更新平衡在线适应速度与稳定性；
- **ATENA**：属于 Active TTA；根据 episodic success/failure 对 mixture entropy 分别最小化或最大化，并训练 self-prediction head，不使用 rollout reward 与策略梯度；
- **本文区别**：不需要外部成功/失败反馈，将多动作采样的轨迹共识转化为内部伪奖励，并面向生成式动作块策略执行 TTRL。

投稿前必须复核各方法在以下维度的公平性：是否使用反馈、是否访问目标信息、参数何时更新、是否跨 episode 保留、是否允许回看整条轨迹。

---

## 五、3. Method

### 3.1 Problem Formulation

#### 3.1.1 VLN 策略

给定语言指令 \(I\)、当前观察 \(o_t\) 和历史视觉观察 \(H_t=\{o_0,\ldots,o_{t-1}\}\)，生成式导航策略为：

\[
y_t \sim \pi_{\theta,\phi}(\cdot\mid I,o_t,H_t),
\]

其中：

- \(\theta\) 表示冻结的基础模型参数；
- \(\phi\) 表示测试阶段可更新的 LoRA 参数；
- \(y_t\) 是文本形式的动作块，例如 forward、turn left、turn right 或 stop。

通过解析与离散化函数 \(f_{\mathrm{act}}\)，得到 Habitat 离散动作序列：

\[
A_t=f_{\mathrm{act}}(y_t)=(a_{t,1},\ldots,a_{t,L}).
\]

当前代码对应的原子动作空间为：停止、前进 0.25 m、左转 \(15^\circ\)、右转 \(15^\circ\)。每次只保留生成结果中的前两个子动作，每个子动作最多展开为 3 个原子动作。以上实现细节应在正文或附录中报告。

#### 3.1.2 测试时强化学习约束

测试阶段不使用：

- 专家动作或真实轨迹；
- 目标位置或最短路径；
- 人工标注的成功/失败反馈。

允许使用：指令、截至当前时刻的视觉历史、策略生成结果以及已执行动作产生的新观察。

需要在实验协议中严格定义参数状态：

\[
\phi_{e+1}=\operatorname{Update}(\phi_e,\mathcal{D}_e),
\]

其中 \(e\) 为 episode。**[已实现]** 当前参数跨 episode 持续保留，episode 顺序被随机打乱；没有 episodic reset。

### 3.2 NaVIDA Base Policy

本节用一段或一张小图交代继承关系：

```text
上游 NaVIDA（监督学习基础策略，继承）
├── Qwen2.5-VL 视觉—语言骨干
├── 历史帧 + 当前帧提示形式
├── 文本动作块预测
├── inverse dynamics supervision（原始离线训练）
└── hierarchical probabilistic action chunking（原始方法）

本文新增
├── 同状态多候选随机采样
├── 动作块到局部轨迹的转换
├── FastDTW 轨迹共识奖励
├── 共识候选执行
└── 组相对 LoRA 策略梯度更新（TTRL）
```

历史视觉输入在当前代码中最多均匀采样 8 帧，并附加当前帧。视觉编码器被冻结，LoRA 插入语言模型的注意力投影和 MLP 投影层。原始 NaVIDA 的 VLN/IDM 学习仍是监督交叉熵训练；强化学习仅出现在本文新增的 TTRL 路径。

### 3.3 Framework Overview

建议将总框架图设计为一条闭环：

```text
Instruction + Visual History + Current Observation
                         │
                         ▼
                NaVIDA Policy + LoRA
                         │  stochastic sampling
                         ▼
               G Candidate Action Chunks
                         │
             parse and kinematic rollout
                         ▼
               G Local Action Trajectories
                         │
          pairwise FastDTW + group ranking
                  ┌──────┴──────┐
                  ▼             ▼
          Consensus Reward   Medoid Action
                  │             │
                  ▼             ▼
          LoRA Optimization   Environment Step
                  └──────┬──────┘
                         ▼
                  Next Observation
```

图注应明确：环境仅执行一个共识候选，其余候选只在内部运动学模型中展开，不会真实执行。

### 3.4 Diverse Action-Chunk Sampling

在状态 \(s_t=(I,o_t,H_t)\) 下采样 \(G\) 个候选：

\[
y_t^{(g)}\sim\pi_{\theta,\phi}(\cdot\mid s_t;\tau),\qquad g=1,\ldots,G,
\]

其中 \(\tau\) 为采样温度。当前脚本设置：

- \(G=8\)；
- temperature = 0.3；
- top-p = 1.0；
- repetition penalty = 1.05；
- max new tokens = 512。

无法解析的候选标记为无效，并在当前实现中赋予 \(-1\) 奖励，同时不参与有效候选的共识轨迹计算。

### 3.5 Action-to-Trajectory Projection

把离散动作序列 \(A_t^{(g)}\) 通过无障碍局部运动学模型映射为轨迹：

\[
T_t^{(g)}=\bigl\{(x_l^{(g)},y_l^{(g)},\vartheta_l^{(g)})\bigr\}_{l=0}^{L_g}.
\]

前进一步的更新为：

\[
x_{l+1}=x_l+d\cos\vartheta_l,\qquad
y_{l+1}=y_l+d\sin\vartheta_l,
\]

左转或右转分别使朝向增加或减少 \(\Delta\vartheta\)。当前实现采用 \(d=0.25\) m、\(\Delta\vartheta=15^\circ\)。

这一步的意义是把不同文本长度、不同动作组合统一到可比较的几何轨迹空间，而不是直接比较字符串或 token。

### 3.6 Trajectory-Consensus Reward

对所有有效候选轨迹计算两两 FastDTW 距离：

\[
D_{ij}=\operatorname{FastDTW}(T_t^{(i)},T_t^{(j)}),
\]

其中局部距离使用二维位置欧氏距离。候选 \(i\) 与其他候选的平均距离为：

\[
c_i=\frac{1}{G_v}\sum_{j=1}^{G_v}D_{ij},
\]

其中 \(G_v\) 是有效候选数。共识轨迹，即候选集合的 medoid，为：

\[
i^*=\arg\min_i c_i.
\]

当前实现对 \(c_i\) 做组内 min-max 线性映射，得到 \([-1,1]\) 的奖励：

\[
r_i=1-2\frac{c_i-c_{\min}}{c_{\max}-c_{\min}}.
\]

若所有有效候选的平均距离相同，则其奖励均设为 1；此时组内标准化优势全部为 0，实际不会产生策略更新。若只有一个有效候选，则该候选奖励为 1；无效候选奖励为 \(-1\)。环境执行 \(A_t^{(i^*)}\)。

本节需要给出直觉解释：靠近候选分布中心的轨迹得到更高奖励，孤立轨迹得到更低奖励。它相当于在局部动作轨迹空间中进行自一致性选择。

### 3.7 Group-Relative Test-Time Reinforcement Learning

对每组奖励计算标准化优势：

\[
\hat A_i=\frac{r_i-\mu_r}{\sigma_r+\epsilon}.
\]

对候选输出 token 的平均对数概率进行加权，测试时优化目标可写为：

\[
\mathcal{L}_{\mathrm{TC}}
=-\frac{1}{G}\sum_{i=1}^{G}
\hat A_i\frac{1}{|y_i|}\sum_{k=1}^{|y_i|}
\log\pi_{\theta,\phi}(y_{i,k}\mid s_t,y_{i,<k}).
\]

代码通过 \(\exp(\log p-\operatorname{sg}(\log p))\hat A_i\) 构造 group-relative REINFORCE/GRPO-style 的一阶策略梯度代理，其中 \(\operatorname{sg}\) 表示 stop-gradient。无效输出和已经结束的 episode 使用 mask 排除。当前实现没有启用 reference-policy KL 或 PPO clipping，因此不应直接宣称使用了完整 GRPO/PPO 算法。

需要准确描述当前更新时机：

- episode 内每个决策点都会计算损失并反向传播；
- 梯度在 episode 内累积；
- Hugging Face Trainer 的外层优化器在该 training step 返回后更新一次；
- 多 GPU 时，每张卡并行一个 episode，需等待全局 batch 中所有 episode 结束。

因此，当前方法更准确地称为 **episode-level continual test-time update**，而不是每个环境 step 后即时更新。

### 3.8 Parameter-Efficient Adaptation

当前实现设置如下：

| 项目 | 当前代码设置 |
|---|---:|
| 基础模型 | NaVIDA 的 Qwen2.5-VL checkpoint |
| 视觉编码器 | 冻结 |
| 可训练参数 | LoRA |
| LoRA 目标层 | q/k/v/o projection，gate/up/down projection |
| LoRA rank | 8 |
| LoRA alpha | 8 |
| LoRA dropout | 0.05 |
| 学习率 | \(2\times10^{-6}\) |
| 调度器 | constant |
| 精度 | bfloat16 |
| 分布式训练 | DeepSpeed ZeRO-2，4 GPU 脚本 |

### 3.9 Algorithm

```text
Algorithm 1: Trajectory-Consensus Test-Time Reinforcement Learning

Input: pretrained NaVIDA policy πθ, LoRA parameters φ,
       test episode stream E, sample number G

for each episode e in E do
    reset environment; keep φ from previous episodes
    initialize visual history H
    while episode is not terminated do
        build state s = (instruction, H, current observation)
        sample G action chunks from πθ,φ(·|s)
        parse each valid chunk into a discrete action sequence
        project each sequence into a local kinematic trajectory
        compute pairwise FastDTW distances
        calculate trajectory-consensus rewards and group advantages
        accumulate gradients of the group-relative policy loss
        execute the medoid candidate in the environment
        append the new observation to H
    end while
    update φ using the accumulated episode gradients
end for
```

---

## 六、4. Experiments

实验章节应按 Research Questions 组织，而不是只列结果表。

### 4.1 Experimental Setup

#### Datasets and splits

优先级建议：

1. **R2R VLN-CE val-unseen**：当前训练与日志已使用，共 1,839 个 episode；
2. **RxR VLN-CE val-unseen**：仓库已有冻结 NaVIDA 评测日志，共 3,669 个 episode；
3. **受控分布偏移**：视觉 corruption、相机参数变化、场景域变化或指令语言变化，至少选一种可复现设置。

如果将同一个 val-unseen 流同时用于模型适应和报告，需要明确这属于 transductive/continual TTA，而非传统冻结评测。不能用适应后的模型再次从头评测同一批 episode，并将结果描述为严格 online TTA，除非单独标记为 transductive adaptation。

#### Metrics

至少报告：

- Success Rate（SR，越高越好）；
- Success weighted by Path Length（SPL，越高越好）；
- Navigation Error（NE，越低越好）；
- Oracle Success Rate（OSR，越高越好）；
- RxR 建议报告 nDTW；
- 平均推理时间、单 episode 更新时间和峰值显存。

#### Current reproducible configuration

| 设置 | 值 |
|---|---|
| 模拟器 | Habitat-Sim/Lab 0.2.4 |
| RGB 分辨率（环境） | 640 × 480 |
| 模型输入图像 | 308 × 252 |
| 前进步长 | 0.25 m |
| 转向角 | 15° |
| 最大 episode 步数 | 500 |
| 成功半径 | 3.0 m |
| 历史采样帧数 | 最多 8 个历史帧 + 当前帧 |
| episode 顺序 | 随机打乱，seed 42（训练器） |

#### TTA protocol checklist

正文中必须回答：

- 参数在每个 step、每个 episode 还是每个场景后更新？
- 参数是否跨 episode、跨场景保留？
- episode 顺序是否固定，并对所有方法完全一致？
- 是否允许未来观察、真实距离、success、SPL 等指标参与更新？
- 超参数在哪个 split 上选择？
- 多个随机种子如何汇总？
- baseline 是否得到相同的测试数据流、反馈预算和更新预算？

### 4.2 Main Results

#### 表 1：R2R val-unseen 主结果

| Method | External feedback | Updated params | SR ↑ | SPL ↑ | NE ↓ | OSR ↑ | Time/step ↓ |
|---|---|---|---:|---:|---:|---:|---:|
| Original NaVIDA (Frozen, SFT) | None | None | 60.96 | 54.68 | 4.44 | 68.52 | [待测] |
| NaVIDA + Self-Consistency Only | None | None | [待填] | [待填] | [待填] | [待填] | [待填] |
| NaVIDA + Tent | None | Norm/LoRA | [待填] | [待填] | [待填] | [待填] | [待填] |
| NaVIDA + CoTTA/EATA | None | [统一设置] | [待填] | [待填] | [待填] | [待填] | [待填] |
| FSTTA | 按原协议 | [待注明] | [待填] | [待填] | [待填] | [待填] | [待填] |
| ATENA | Episodic feedback | [待注明] | [待填] | [待填] | [待填] | [待填] | [待填] |
| **NaVIDA + Trajectory-Consensus TTRL (Ours)** | None | LoRA | **[待填]** | **[待填]** | **[待填]** | **[待填]** | [待填] |

表中的 NaVIDA 冻结结果来自当前目录完整的 1,839 个 JSON 日志汇总，与 README 报告值接近。投稿时应由统一脚本重新复现并报告多随机种子结果。

#### 表 2：RxR 或跨分布结果

| Method | Setting | SR ↑ | SPL ↑ | NE ↓ | nDTW ↑ |
|---|---|---:|---:|---:|---:|
| NaVIDA (Frozen) | RxR val-unseen | [重算] | [重算] | [重算] | 67.30 |
| Ours | RxR val-unseen | [待填] | [待填] | [待填] | [待填] |
| NaVIDA (Frozen) | Corruption/Domain shift | [待填] | [待填] | [待填] | [待填] |
| Ours | Corruption/Domain shift | [待填] | [待填] | [待填] | [待填] |

当前 RxR 日志中存在极大/无穷 NE 或 SPL 异常值，必须先修正统计脚本中的 NaN/Inf 处理再填写。

### 4.3 Ablation Studies

#### 表 3：组件消融

| Variant | Multi-sampling | Trajectory reward | Relative optimization | LoRA update | SR | SPL |
|---|:---:|:---:|:---:|:---:|---:|---:|
| Frozen NaVIDA |  |  |  |  | [ ] | [ ] |
| Self-consistency execution | ✓ | ✓ |  |  | [ ] | [ ] |
| Token/action voting | ✓ | 替代奖励 | ✓ | ✓ | [ ] | [ ] |
| Endpoint Euclidean consensus | ✓ | 替代奖励 | ✓ | ✓ | [ ] | [ ] |
| w/o invalid-output filtering | ✓ | ✓ | ✓ | ✓ | [ ] | [ ] |
| **Full model** | ✓ | ✓ | ✓ | ✓ | [ ] | [ ] |

该表必须包含“Self-consistency Only”。否则无法判断收益来自更好的动作选择，还是来自参数更新。

#### 表 4：采样数和温度

测试 \(G\in\{1,2,4,8,16\}\)，\(\tau\in\{0.1,0.3,0.7,1.0\}\)。\(G=1\) 时没有真正的组内共识，应作为退化边界讨论。

#### 表 5：更新范围与时机

比较：

- 不更新；
- 每 step 更新；
- 每 episode 更新（当前实现）；
- episodic reset；
- continual update；
- 只更新 attention LoRA；
- attention + MLP LoRA（当前实现）。

### 4.4 Analysis

建议至少包含以下四项：

1. **轨迹共识与导航正确性的相关性。** 将平均共识距离与最终 success、距离目标变化量做相关性/分桶分析。这是证明自监督信号合理性的关键。
2. **适应过程曲线。** 按 episode 序号绘制滑动窗口 SR/SPL 和奖励均值，观察持续适应还是灾难性漂移。
3. **成功与失败案例。** 展示候选轨迹分布、medoid 轨迹和实际执行路径，分析共识何时有效、何时共同犯错。
4. **效率分析。** 报告相对冻结 NaVIDA 的生成次数、每步延迟、反向传播时间、显存和总 GPU 小时。

可进一步分析：

- 不同 episode 长度下的性能；
- 场景切换前后的遗忘；
- 无效动作文本比例；
- 停止动作的精度与召回；
- FastDTW 与普通 DTW、Chamfer distance、终点距离的比较；
- 对输入顺序和随机种子的敏感性。

### 4.5 Statistical Reporting

- 至少运行 3 个随机种子；
- 报告 mean ± std；
- 对相同 episode 的 success 可采用配对 bootstrap 置信区间；
- 同时报告绝对提升和相对提升；
- 不要只从部分 152 个 episode 推断完整测试集结论。

---

## 七、5. Discussion and Limitations

建议诚实讨论以下限制：

1. **共识不等于正确。** 当基础策略在新域中发生系统性偏差时，多个候选可能高度一致但全部错误。
2. **计算成本高。** 每个状态需采样 \(G\) 个生成序列、重新计算其 token log-probability 并反向传播；当前 \(G=8\)，总 FLOPs 会随 \(G\) 明显增长。候选可以批处理，因此墙钟时间不一定严格达到 8 倍，但仍显著高于单候选冻结推理；FastDTW 通常不是主要瓶颈。
3. **局部无障碍运动学近似。** 候选轨迹投影未建模障碍物、碰撞和真实视觉变化，只描述动作的局部几何结果。
4. **持续更新可能漂移。** 当前没有 KL、EMA teacher、源模型正则或周期重置，跨 episode 累积更新可能破坏已有能力。
5. **停止决策较脆弱。** 轨迹距离对“是否应该停止”的语义刻画有限，需要单独分析。
6. **协议依赖。** continual、episodic 和 transductive TTA 的结论不可混用，测试流顺序也可能显著影响结果。

潜在改进方向可以放在本节末尾：置信度门控、源模型 KL 正则、场景变化检测、低共识时跳过更新、视觉变化一致性或 inverse-dynamics consistency。

注意：这些改进目前不是代码已实现贡献，不应写进 Method 主体。

---

## 八、6. Conclusion

结论只回答三个问题：

1. 研究了什么：无外部反馈的生成式 VLN 测试时强化学习；
2. 提出了什么：轨迹共识伪奖励、medoid 动作执行与组相对 LoRA 策略梯度更新；
3. 得到了什么结论：[由完整实验填写]。

中文结论模板：

> 本文研究生成式视觉—语言导航策略在无专家标注和无外部反馈条件下的测试时强化学习问题。我们提出基于动作轨迹共识的 TTRL 框架，通过多样化采样、局部轨迹投影和组相对伪奖励，从策略自身的动作分布中构造部署时强化信号，并以 LoRA 执行参数高效的策略梯度更新。实验在 [数据集与偏移设置] 上表明，[填写经实验支持的主要结论]。该研究说明，动作空间中的结构化自一致性可以作为序贯具身决策的测试时伪奖励，同时也揭示了共识失真、持续更新漂移和显著计算开销等仍待解决的问题。

---

## 九、建议图表清单

### 主文图

- **Figure 1：方法总览。** 从视觉历史和指令，到多动作采样、轨迹投影、共识奖励、LoRA 更新与环境交互。
- **Figure 2：轨迹共识奖励示意。** 展示 8 个候选轨迹、medoid、高/低奖励候选。
- **Figure 3：持续适应曲线。** episode 序号—滑动窗口 SR/SPL/奖励。
- **Figure 4：定性案例。** 成功案例、共同错误案例、停止失败案例。

### 主文表

- **Table 1：R2R 主结果；**
- **Table 2：RxR/分布偏移结果；**
- **Table 3：核心组件消融；**
- **Table 4：效率与资源开销。**

### 附录

- 完整超参数；
- 不同 \(G\)、温度、学习率、LoRA rank 的敏感性；
- episodic/continual/reset 协议对比；
- 更多轨迹可视化；
- 伪代码与动作解析细节；
- 所有随机种子与置信区间。

---

## 十、代码—论文内容对照表

| 论文内容 | 当前代码位置 | 状态 |
|---|---|---|
| 原始 NaVIDA 监督微调 | `NaVIDA/src/train/train.py` | 已实现；交叉熵训练，无 reward/advantage/policy gradient |
| TTRL 训练入口与超参数 | `src/train/tttrl_main.py` | 已实现 |
| episode 数据流 | `src/dataset/dataset.py` | 已实现；随机打乱 episode |
| 环境状态、提示与视觉历史 | `src/trainer/episode_state.py` | 已实现 |
| 多候选生成 | `src/trainer/test_time_trainer.py::compute_loss` | 已实现 |
| 文本动作解析与离散化 | `compute_reward_and_execute_actions` | 已实现 |
| 动作到局部轨迹映射 | `get_traj_from_action` | 已实现 |
| FastDTW 共识奖励 | `rewards_function` | 已实现 |
| 组相对优势和策略损失 | `compute_loss` | 已实现 |
| 共识候选执行 | `compute_reward_and_execute_actions` | 已实现 |
| LoRA 配置 | `src/train/tttrl_main.py` | 已实现 |
| R2R 配置与指标 | `config/vln_r2r.yaml` | 已实现 |
| 冻结 NaVIDA 评测 | `src/eval/eval.py`、`eval_log/navida_best_r2r` | 已完成完整日志 |
| TTRL 完整评测 | `eval_log/tttrl_test` | 未完成，仅 152 个 episode |
| KL/reference model 正则 | `test_time_trainer.py` 中被注释 | 未启用 |
| episodic reset | 无 | 未实现 |
| 分布偏移实验 | 未发现统一实现 | 待补 |

---

## 十一、投稿前必须解决的实现与协议问题

这些问题会直接影响论文结论可信度，应优先于润色正文：

1. **修正有效候选索引。** `rewards_function` 构造的 `traj_list` 长度为有效候选数 \(V\)，但 pairwise 循环当前使用 `range(G)`；一旦出现无效候选，可能越界。应统一使用 `range(V)`。
2. **确认更新时机。** 当前代码在 episode 内多次 backward、episode 后一次 optimizer step。若论文要声称 step-wise TTA，需要修改实现并重新实验。
3. **统一评测协议。** 训练阶段直接交互 R2R val-unseen；必须说明每个 episode 的指标是在更新前还是更新后产生，以及最终模型是否重新访问相同 episode。
4. **避免测试集调参。** 学习率、温度、采样数和 LoRA rank 应在独立 validation stream 或受控开发 split 上选择。
5. **固定 episode 顺序。** 当前数据集初始化后随机 shuffle；所有方法和随机种子需要保存并复用完全相同的 episode order。
6. **处理 RxR 数值异常。** 当前日志含 Inf/极大 NE 或异常 SPL；统计时应明确过滤规则并报告有效 episode 数。
7. **补充稳定机制或稳定性证据。** 当前 KL 项和 reference model 均被注释。若不增加机制，论文应把稳定性作为实证分析而非方法贡献。
8. **区分 action selection 与 adaptation。** 必须加入“只用 medoid 执行、不更新参数”的 baseline。
9. **报告真实计算成本。** \(G=8\) 的生成和反向传播很昂贵；不能只报告导航指标。
10. **确认命名与算法边界。** 当前类名是 `TTTRLTrainer`。正文可称 Test-Time Reinforcement Learning，但应说明它是简化的 group-relative policy-gradient 实现：奖励来自内部轨迹共识，不是环境任务奖励；reference KL 与 clipping 均未启用，不能未经说明直接称为完整 GRPO/PPO。

---

## 十二、核心参考文献与引用位置

下列文献来自原问答中已经整理的核心文献包。正式写作时请导入 BibTeX，并再次核对作者、版本与发表信息。

1. **Anderson et al. Vision-and-Language Navigation, CVPR 2018.** 用于 VLN 定义和 R2R 数据集。  
   <https://openaccess.thecvf.com/content_cvpr_2018/html/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.html>
2. **HAMT, NeurIPS 2021.** 用于历史感知 Transformer VLN。  
   <https://papers.nips.cc/paper/2021/hash/2e5c2cb8d13e8fba78d95211440ba326-Abstract.html>
3. **NaVid, 2024.** 用于基于视频/VLM 的连续 VLN。  
   <https://arxiv.org/abs/2402.15852>
4. **NaVIDA, 2026.** 直接基础策略；Introduction、Related Work、Method 和 Experiments 均应引用。  
   <https://arxiv.org/abs/2601.18188>
5. **Tent, ICLR 2021.** 用于熵最小化 TTA。  
   <https://openreview.net/forum?id=uXl3bZLkr3c>
6. **CoTTA, CVPR 2022.** 用于持续测试时适应和抗遗忘。  
   <https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html>
7. **EATA, ICML 2022.** 用于选择性适应与防遗忘。  
   <https://proceedings.mlr.press/v162/niu22a.html>
8. **FSTTA, ICML 2024.** VLN 测试时适应的直接相关工作。  
   <https://proceedings.mlr.press/v235/gao24p.html>
9. **TDA, CVPR 2024.** 用于视觉—语言模型测试时适应。  
   <https://arxiv.org/abs/2403.18293>
10. **TTRL: Test-Time Reinforcement Learning, NeurIPS 2025.** 本文最直接的方法学基础：使用 majority voting 为无标签测试样本构造伪奖励并进行强化学习。  
    <https://arxiv.org/abs/2504.16084>
11. **ATENA, NeurIPS 2025.** Active VLN-TTA：使用 episodic feedback 控制 mixture entropy 的最小化或最大化；作为 TTA 对照，而非 TTRL 文献。  
    <https://arxiv.org/abs/2506.06630>
12. **TTA Survey.** 用于 TTA 定义、分类和背景。  
    <https://arxiv.org/abs/2303.15361>

如果实现直接借鉴了 GRPO/相关工作，还必须补充对应原始文献并准确说明差异；当前代码未启用 reference KL 或 clipping，投稿前应追溯策略损失来源。

---

## 十三、推荐写作顺序

1. 先固定实验协议，完成冻结、只做共识动作选择、完整 TTRL 三个最关键 baseline；
2. 再写 Method 3.1～3.9，并同步绘制 Figure 1；
3. 完成主结果和组件消融后写 Experiments；
4. 根据实验事实回写 Introduction 的 research gap 和贡献；
5. 最后写 Abstract、Conclusion 和标题。

当前最优先的不是继续增加 Related Work，而是回答一个决定论文是否成立的问题：

> **在完全相同的候选采样与 medoid 动作选择条件下，更新 LoRA 是否确实优于不更新参数？**

只有这个对照成立，论文的核心贡献才能被可靠归因于“测试时强化学习”，而不仅是“多次采样后的自一致性动作选择”。
