# 研究问题与证据边界

更新：2026-10-03。本页陈述当前研究对象和待回答的问题；运行参数以对应实验 manifest 为准，结果以验收产物为准。术语：**VERIFIED FACT** 是协议、源码或验收文件直接支持的事实；**EMPIRICAL OBSERVATION** 是测得的结果；**INTERPRETATION** 是对结果的解释；**HYPOTHESIS** 待证伪；**PROPOSED IDEA** 尚未实施；**UNKNOWN** 缺少足够证据。

## 研究对象

**VERIFIED FACT：** 本项目在 NaVIDA 基础策略上研究 R2R-VLNCE `val_unseen` 的测试时适应。智能体在每次需要新动作块时看到指令与截至当前时刻的观察，生成多个动作候选，将可执行前缀投影为局部虚拟轨迹，用轨迹一致性及 episode 初始 adapter 的候选置信度选择动作。A 组还在当前 episode 内更新 LoRA；B 组走相同的候选、选择及前反向流程，但跳过 optimizer step；canonical C0 是单候选冻结策略。当前动作在更新前已选定，更新只可能影响后续生成；每个 episode 重置 adapter 和 optimizer。[项目总览](../../PROJECT_OVERVIEW.md)、[K8 A 协议](../../experiments/preparations/a5_2_k8_checkpoint_gpu467_af8e58b_20260919/protocol_manifest.json)、[K8 B 协议](../../experiments/preparations/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921/protocol_manifest.json)

**VERIFIED FACT：** 正式方法不把测试时目标位置、最短路径、专家动作或最终 success 当作伪奖励；这些信息只用于评测或明确标注的离线诊断。局部虚拟轨迹没有真实执行候选分支，也不模拟碰撞。[项目总览](../../PROJECT_OVERVIEW.md)、[当前方法旧版本说明](../method/CURRENT_METHOD.md)

## 科学问题

在没有测试时专家标签或目标反馈的序贯导航中，**策略自身生成的局部候选结构是否提供足够可靠的信息，用来改善动作选择和同 episode 后续决策？** 需要分别判断：

1. 可用动作是否出现在候选集合里，以及已有可用动作时是否被正确选择。两者分别是候选覆盖与排序问题，现有 STOP 审计不能代替一般导航动作的 oracle 覆盖率。[候选诊断草案](../plans/CANDIDATE_DIAGNOSTIC.md)
2. 轨迹共识是否携带导航正确性信息，而不只是产生与低温候选不同的动作。这个机制目前是 **HYPOTHESIS**，不是结果。[HYPOTHESES](HYPOTHESES.md)
3. 在相同多候选选择协议下，在线参数更新是否有独立增益。K8 A/B 是现有最接近的对照；两次运行非同期，结果尚不支持更新收益。[A/B 配对结果](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/paired_A_minus_B.json)

## 目前的实验范围

**VERIFIED FACT：** canonical C0 已覆盖全部 1,839 条 `val_unseen`，本项目的主要 A5/A5.2 比较使用同一批 600 个 ID。same600 包含此前用于开发的 new128 全部 128 条。[A5/C0 验收说明](../experiments/A5_C0_20260914.md)、[K8 A 分析计划](../../experiments/preparations/a5_2_k8_checkpoint_gpu467_af8e58b_20260919/analysis_plan.json)

> **核心证据边界：** same600 已被反复用于方法开发；K8 A/B 非同期运行；K8/K3 同时改变候选数与 cluster 泛化规则；目前没有独立确认集证明本方法优于冻结基线。[K8 A 分析计划](../../experiments/preparations/a5_2_k8_checkpoint_gpu467_af8e58b_20260919/analysis_plan.json)、[K8 B 分析计划](../../experiments/preparations/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921/analysis_plan.json)

因此，本项目已经能够评估已运行协议在开发集上的表现，却仍不能把观察到的跨版本差异归因为某一个方法部件。当前论文主张的上限见 [PAPER_STORY](PAPER_STORY.md)，逐项结果见 [FINDINGS](FINDINGS.md)。

## UNKNOWN

- 一般导航场景中，可执行的正确候选覆盖率及已覆盖时的排序错误率：**UNKNOWN**。现有日志不足以把未执行候选的反事实导航结果补成事实。[候选诊断草案](../plans/CANDIDATE_DIAGNOSTIC.md)
- 在线更新在独立确认数据上的效果与跨运行稳定性：**UNKNOWN**。[A/B 配对结果](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/paired_A_minus_B.json)
- 当前本机 C0 数字与旧文档所引外部基线数字能否按完全相同协议比较：**UNKNOWN**。[C0 全量计划](../../experiments/official_navida_c_full/analysis_plan.json)、[A5/C0 验收说明](../experiments/A5_C0_20260914.md)
