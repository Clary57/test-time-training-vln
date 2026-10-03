# 文献地图与来源核验状态

更新：2026-10-03。本页把“论文原始页面可直接核对的内容”与“旧笔记中的推荐或推断”分开。**PRIMARY-SOURCE CHECKED** 仅表示题名及下述摘要层面的描述已对照作者论文或正式会议页面；不是完整复现、协议公平性审计或本项目结果的证据。**UNVERIFIED** 表示目前只见旧笔记/会议线索，不在论文中当作已核实事实引用。

## 与当前问题直接相关的原始来源

| 来源及核验状态 | 原始来源能支持的有限陈述 | 对本项目的作用与不能推出的结论 |
|---|---|---|
| [Anderson 等，VLN/R2R，CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.html) — **PRIMARY-SOURCE CHECKED** | 论文提出视觉条件下执行自然语言导航指令的 VLN 任务及 R2R 数据集。 | 任务背景；不能把原始 R2R 与本项目 R2R-VLNCE `val_unseen` 的数值协议视为同一结果。 |
| [NaVIDA，arXiv:2601.18188](https://arxiv.org/abs/2601.18188) — **PRIMARY-SOURCE CHECKED** | 作者摘要描述 inverse-dynamics supervision 与 hierarchical probabilistic action chunking。 | 基础策略来源；该摘要不证明本项目的测试时更新有效，也不提供本机 C0 的 58.46% 数字。项目使用的 checkpoint 与评测协议仍以[本机记录](../experiments/A5_C0_20260914.md)为准。 |
| [Tent，arXiv:2006.10726](https://arxiv.org/abs/2006.10726) — **PRIMARY-SOURCE CHECKED** | 作者摘要描述在测试时最小化预测熵并在线更新归一化相关参数，实验主要涵盖分类等任务。 | 无标签 TTA 背景；本项目正式目标并非完整熵最小化，不得把 Tent 的效果移植为 VLN 结果。 |
| [FSTTA，ICML 2024/PMLR](https://proceedings.mlr.press/v235/gao24p.html) — **PRIMARY-SOURCE CHECKED** | 作者摘要提出在线 VLN 的 fast-slow 测试时适应，讨论多步决策与更新频率。 | VLN-TTA 相关工作；反馈预算、数据、骨干模型、动作空间和 reset 协议是否与本项目可比仍需逐项核查。 |
| [ATENA，NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/3f510f82323c1293ae3e343893dd77b1-Abstract-Conference.html) — **PRIMARY-SOURCE CHECKED** | 作者摘要描述 episodic feedback、mixture entropy optimization 与 active learning。 | 与无外部反馈的本项目比较时，必须标明反馈条件不同；不能把其数值直接列为同预算对照。 |
| [TTRL，arXiv:2504.16084](https://arxiv.org/abs/2504.16084) — **PRIMARY-SOURCE CHECKED** | 作者摘要研究语言模型无标签推理任务，用多样本 majority voting 估计测试时强化学习奖励。 | 术语与动机背景；多数投票在语言推理中的结果不证明局部导航轨迹共识可作为正确 reward。 |

## 旧材料中的待核验线索

- [历史写作问答](../archive/pre_a5/资料.md)提及 CoTTA、EATA、TDA、NaVid、NaVILA、HAMT、TTA survey 等；本次没有逐篇核对原文、数据及比较条件，均标 **UNVERIFIED**，不进入当前论文的事实性比较或“必须引用”清单。
- [旧论文框架](../archive/pre_a5/VLN_TTA论文框架.md)中的方法优先级、预设贡献、外部 baseline 数字及 RxR/分布偏移表格属于历史构想或占位，**UNVERIFIED**；不能从其中推出已有实验结果。
- [2026-09-17 会议记录](../meetings/2026-09-17.md)给出 [arXiv:2511.18960](https://arxiv.org/abs/2511.18960)。该原始页面的题名是 *AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention*，摘要讨论 recurrent state 与视觉注意力；会议中的 soft prompt、TTT 层/K/V 解释不能据此归到该论文。具体所指另一论文：**UNKNOWN**。这条会议讨论不作为本项目已实施方法或文献事实。

## 尚未完成的文献审查

1. FSTTA、ATENA 与本项目的任务、反馈预算、数据选择、骨干模型及更新时序是否可构成公平定量对照：**UNKNOWN**。目前只核对了原始页面摘要，不做跨论文性能排序。
2. NaVIDA 原文报告数字与本机 canonical C0 的 checkpoint、场景/数据版本、动作块执行及评测口径能否逐项对齐：**UNKNOWN**。[本机 C0 结果](../experiments/A5_C0_20260914.md)、[本机 C0 分析计划](../../experiments/official_navida_c_full/analysis_plan.json)
3. 本项目的局部共识、推理时多候选选择与测试时参数更新，分别相对已有工作的真正新颖性边界：**UNKNOWN**。在完成全文与协议核验前，不写“首次”或“优于已有 VLN-TTA”。
