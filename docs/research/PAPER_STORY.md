# 暂定论文叙述及主张上限

更新：2026-10-03。本文是研究叙述的审查边界，不是已确认的摘要或贡献列表。文中 **VERIFIED FACT / EMPIRICAL OBSERVATION / INTERPRETATION / HYPOTHESIS / UNSUPPORTED CLAIM** 各有不同证据地位；相关质疑见 [ATTACK_LOG](ATTACK_LOG.md)。

## 已展示的系统能力（VERIFIED FACT）

本项目实现了 NaVIDA 上的 episode-local 多候选生成、局部轨迹共识与 gate 选择，以及 A 组的 LoRA 更新和 B 组的禁用更新对照。更新前选出的动作先执行；参数只影响后续生成；新 episode 恢复初始 adapter。K8 A 的严格验收记录了 3,000 次有效参数变化；K8 B 记录了 0 次 optimizer step 与 0 次有效参数变化。这证明两种执行路径确实不同，**不证明更新改善导航**。[K8 A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)、[K8 B report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/report.json)

## 已观察到的结果（EMPIRICAL OBSERVATION）

canonical C0 在全量 1,839 条上为 1,075/1,839，same600 上为 347/600；早期 A5 T1.5 K3 为 305/600。A5.2-K3 为 353/600，K8 A 为 359/600，K8 B 为 365/600。K8 A 比 K3 高 1 个百分点、比同 ID C0 高 2 个百分点，两个配对区间均跨 0；K8 A−B 为 −1 个百分点，95% 区间 [−4.00，+1.83]，同样跨 0。[A5/C0 结果](../experiments/A5_C0_20260914.md)、[K3 验收](../experiments/a5_2_acceptance_20260917/report.json)、[K8/K3 配对](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/paired_vs_a5_2_k3.json)、[K8/C0 配对](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/paired_vs_canonical_c0.json)、[K8 A/B 配对](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/paired_A_minus_B.json)

**证据边界：** same600 已反复用于开发；K8 A/B 非同期；K8/K3 不只改变 K；本项目没有独立确认集证明优于 frozen C0。数值差异是开发集结果，不能写成独立确认或单因素效果。[K8 A 分析计划](../../experiments/preparations/a5_2_k8_checkpoint_gpu467_af8e58b_20260919/analysis_plan.json)、[K8 B 分析计划](../../experiments/preparations/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921/analysis_plan.json)

## 机制解释（INTERPRETATION / HYPOTHESIS）

- A5.2-K3 相对早期 A5/A5.1 的恢复，与 gate 控制执行及低温 fallback 的组合同时出现；候选组成、动作选择和运行拓扑都存在变化，不能断定是哪一项导致恢复。[K3 验收说明](../experiments/a5_2_acceptance_20260917/README.md)
- K8 的平均唯一原子动作候选约为 5.50，但 K8 A 对 K3 的 SR 差异不明确。究竟是候选覆盖已足够、排序不可靠，还是其他因素限制收益，仍是待检验解释。[K8 A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)、[H001/H002](HYPOTHESES.md)
- K8 A/B 的观测没有显示在线更新的独立收益；区间也容许小幅正或负效果，且存在非同期运行差异。[A/B 配对结果](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/paired_A_minus_B.json)、[H003](HYPOTHESES.md)
- STOP 候选覆盖不足可能解释部分近目标失败，但覆盖率是沿各自已执行轨迹的描述性统计，没有证明其对最终失败的因果作用。[K8 A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)、[H004](HYPOTHESES.md)

## 当前可辩护的论文表述

> 我们建立并验收了一个以 NaVIDA 为基础、在单个导航 episode 内进行多候选局部轨迹共识选择与可选 LoRA 更新的测试时适应系统。在反复使用的 600 条开发集上，A5.2-K3/K8 的成功率与本机冻结基线接近；现有配对结果尚不能确认 K8 相对 K3 的额外收益，也未确认在线更新相对同协议禁用更新的收益。候选覆盖、排序及更新作用仍需进一步区分。

此段陈述系统能力与观测范围；它不宣称独立确认的性能优势。

## 当前不成立的主张（UNSUPPORTED CLAIM）

1. “在线测试时参数更新提升导航性能”：同协议 K8 A/B 没有给出正向证据。[A/B 配对结果](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/paired_A_minus_B.json)
2. “K8 因候选数增加而优于 K3”：区间跨 0，且 K 与 cluster 规则一同变化。[K8/K3 配对结果](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/paired_vs_a5_2_k3.json)
3. “当前方法在独立确认集上超过 frozen NaVIDA”：没有相应独立确认集。[K8 A 分析计划](../../experiments/preparations/a5_2_k8_checkpoint_gpu467_af8e58b_20260919/analysis_plan.json)
4. “共识 reward 已证明能识别导航正确动作”或“STOP 缺失是主要失败原因”：目前只有机制与覆盖诊断，没有足够的反事实正确性证据。[候选诊断草案](../plans/CANDIDATE_DIAGNOSTIC.md)

旧[论文框架](../archive/pre_a5/VLN_TTA论文框架.md)是历史提纲，其占位的 RxR、分布偏移、效率和性能优势不能写入当前结果。
