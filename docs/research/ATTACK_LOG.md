# Research Attack Log

更新：2026-10-03。这里记录由**已有证据**暴露的主张漏洞。`OPEN` 表示尚未由独立证据排除；它不是已确定的错误，也不自动授权下一项实验。研究问题与证据分级见 [PROBLEM](PROBLEM.md)。

## ATTACK-001

### Target Claim

“A5.2 在冻结 NaVIDA 之上取得了可推广的性能优势。”

### Attack / Alternative Explanation

same600 反复用于选择和调整方法，且完整包含之前用于开发的 new128。开发过程中的选择偏差可以让该集合上的点估计高于未来独立数据；K3 与 K8 相对 C0 的配对区间也都跨 0。

### Why It Matters

如果不能在未参与开发的集合上成立，论文只能报告开发集观察，不能主张已确认的泛化优势。

### Existing Evidence

[A5/C0 selection 与结果](../experiments/A5_C0_20260914.md)、[K8 A 分析计划](../../experiments/preparations/a5_2_k8_checkpoint_gpu467_af8e58b_20260919/analysis_plan.json)、[K3/C0 配对](../experiments/a5_2_acceptance_20260917/paired_vs_canonical_c0.json)、[K8/C0 配对](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/paired_vs_canonical_c0.json)。

### Evidence Needed

独立于方法开发的预先定义评估集合、锁定协议和完整验收；集合作用、选择过程及比较规则均需说明。目前是否存在可用的独立确认集合：**UNKNOWN**。

### Status

**OPEN**。当前不存在独立确认集优于 frozen C0 的证据。

## ATTACK-002

### Target Claim

“在线 LoRA 更新在相同多候选推理选择之外提供独立收益。”

### Attack / Alternative Explanation

K8 A 有 3,000 次有效参数变化，却成功 359/600；禁用更新的 B 成功 365/600。A−B 为 −1.00 个百分点，95% 区间 [−4.00，+1.83]。两次运行非同期，可能使用不同物理 GPU；区间也不能排除小幅正或负效果。

### Why It Matters

系统能完成在线更新与“更新使导航更好”是不同命题。如果收益仅来自推理时多候选选择，论文的 TTRL 效果主张便不成立。

### Existing Evidence

[A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)、[B report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/report.json)、[配对分析](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/paired_A_minus_B.json)、[B 预锁定分析边界](../../experiments/preparations/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921/analysis_plan.json)。

### Evidence Needed

预先规定最小有意义收益、未用于开发的匹配 A/B 验收，以及更新后决策的解释范围。是否有足够资源与独立数据完成该检验：**UNKNOWN**。

### Status

**OPEN**；当前正向主张未获支持，尚不能判定更新必然有害。

## ATTACK-003

### Target Claim

“候选数从 K3 增加到 K8 改善了导航，或纯 K 是改善原因。”

### Attack / Alternative Explanation

K8 A 相对 K3 仅 +1.00 个百分点，95% 区间 [−3.17，+5.00]，`p=0.689`；K8 使用了适应更大候选集的 complete-link cluster 规则，参考 K3 是历史非同期运行。K、聚类规则及运行条件不能分离。

### Why It Matters

把组合变更误写为 K 的单因素收益，会使方法机制与论文消融结论失真。

### Existing Evidence

[K8/K3 配对](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/paired_vs_a5_2_k3.json)、[K8 A 预锁定分析计划](../../experiments/preparations/a5_2_k8_checkpoint_gpu467_af8e58b_20260919/analysis_plan.json)、[K8 A 协议](../../experiments/preparations/a5_2_k8_checkpoint_gpu467_af8e58b_20260919/protocol_manifest.json)。

### Evidence Needed

若要讨论纯 K，需要可解释且锁定的等价选择规则、独立评估及重复性；这些条件在现有材料中是否可实现：**UNKNOWN**。在此前只能报告“现有 K8 整套协议对历史 K3 的开发集比较”。

### Status

**OPEN**；纯 K 作用未被识别。

## ATTACK-004

### Target Claim

“A5.2-K3 的恢复证明 gate、共识 reward 或低温 fallback 中某一项有效。”

### Attack / Alternative Explanation

A5.2 同时改变 gate 对执行的控制和 candidate 0 的采样温度/回退行为，并与 A5.1 使用不同 worker 拓扑。K3 相对 A5.1 的 +8.83 个百分点是整套协议差异。原 coordinator 的 clean-tree 检查也失败，最终结论依赖附例外的独立产物验收。

### Why It Matters

该结果是论文“为什么恢复”的主要诱因，但不能据此指认单一机制，更不能推断在线更新是原因。

### Existing Evidence

[K3/A5.1 配对](../experiments/a5_2_acceptance_20260917/paired_vs_a5_1.json)、[K3 验收说明](../experiments/a5_2_acceptance_20260917/README.md)、[A5.1 修正验收](/hdd3/wangziyu/runs/test-time-training-vln/a5_1_executed_span_A600_gpu28_19a2e5e_20260915_batch02/analysis_corrected_v1/validation.json)。

### Evidence Needed

若主张特定机制，需要同候选、同选择规则及相同数据角色下的隔离比较，并保留对原 coordinator 例外的完整说明；现有材料不能补出未运行的反事实。

### Status

**OPEN**；整套协议的经验改善可报告，单因素解释不可报告。

## ATTACK-005

### Target Claim

“轨迹共识伪奖励或 STOP 覆盖诊断已经证明选择机制找到了正确导航动作。”

### Attack / Alternative Explanation

局部轨迹一致性是候选之间的关系，不是与目标指令一致的标签；虚拟轨迹不执行分支，也不模拟碰撞。K8 A 在成功半径内只有 1,473/3,135 次决策含 STOP 候选，但这既不判定该 STOP 是否在执行前缀，也不说明最终失败是否由缺失造成。旧 B 审计中 50/62 个失败从未进入成功半径，STOP 缺失不能解释它们。

### Why It Matters

论文若将描述性覆盖率写成伪奖励有效性或失败原因，便跨过了最关键的科学论证缺口。

### Existing Evidence

[K8 A 诊断 report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)、[旧 B STOP 审计](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01/offline_stop_audit_20260910/REPORT.md)、[候选诊断草案](../plans/CANDIDATE_DIAGNOSTIC.md)。

### Evidence Needed

独立于正式 reward 的候选正确性定义、可恢复的相同状态、覆盖与排序分开计量，以及失败 episode 的完整分母。未执行动作分支能否可靠回放：**UNKNOWN**。

### Status

**OPEN**；目前只有描述性诊断。

## ATTACK-006

### Target Claim

“本机 canonical C0 与旧资料引用的 NaVIDA 基线数字已完全对齐，可以直接用于论文跨来源比较。”

### Attack / Alternative Explanation

全量 C0 分析计划记载一个外部 `published_sr=0.614` 参考；本机只读复核得到 1,075/1,839（58.46%）。旧论文框架又列出 60.96%。这些外部数字的原始协议、版本和可比性未在当前迁移中独立核定，不能把数值差异直接解释为模型退化或复现失败。

### Why It Matters

基线来源不清会影响论文主表的公平性和“超过 NaVIDA”这一表述。

### Existing Evidence

[C0 全量分析计划](../../experiments/official_navida_c_full/analysis_plan.json)、[本机 C0 验收说明](../experiments/A5_C0_20260914.md)、[旧论文框架](../archive/pre_a5/VLN_TTA论文框架.md)。

### Evidence Needed

逐项核对原论文数字所用 checkpoint、数据版本、动作执行、历史长度、评测脚本和统计口径，并明确只比较同协议结果。旧资料中的 60.96% 来源与适用条件：**UNKNOWN / UNVERIFIED**。

### Status

**OPEN**；本项目内部 same600 比较应继续使用本机 canonical C0，跨来源数值比较暂不成立。
