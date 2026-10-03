# 科学假设登记

更新：2026-10-03。假设是可被证伪的科学解释，不是“增加 K”“打开更新”或“修改 gate”之类实施步骤。以下所有结果均来自反复使用的 same600 开发集；K8 A/B 非同期，K8/K3 同时改变 K 与 cluster 泛化。没有独立确认集证明当前方法优于 frozen C0。[问题与边界](PROBLEM.md)、[研究攻击记录](ATTACK_LOG.md)

状态含义：**OPEN**＝证据不足；**NOT SUPPORTED**＝现有比较没有显示预期效果，但尚未排除小效果；**FALSIFIED** 仅在预定的反证标准得到满足时使用。目前没有一项达到确认或证伪标准。每项的“证伪实验”只是判断该解释需要什么证据，不是已批准的下一实验；[NEXT_EXPERIMENT](NEXT_EXPERIMENT.md)保持空白。

## H001 — 候选覆盖与排序谁更限制表现

- **Observation（观察）：** K8 A 每次决策平均有约 5.50 个不同的原子动作候选，而相对 K3 的 SR 仅高 1.00 个百分点，95% 配对区间 [−3.17，+5.00]；在成功半径内，约 53% 的决策仍没有 STOP 候选。[K8 A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)、[K8/K3 配对](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/paired_vs_a5_2_k3.json)
- **Hypothesis（科学假设）：** 对一般导航动作，候选中已经包含合适动作却未被选中的情况，可能比“合适动作根本未生成”更限制当前 K8 协议的表现。这里的“合适”必须用独立于正式 reward 的导航结果标准定义，不能用候选之间的相似度代替。
- **Prediction（预测）：** 若假设成立，在可可靠回放的相同决策状态上，候选集的 oracle 可行/正确动作覆盖会明显高于实际被选动作的正确率；仅增加候选多样性而不改善排序，不会稳定提高 SR。所需差距大小与判定阈值目前 **UNKNOWN**。
- **Alternative explanations（替代解释）：** 新候选虽不同却都错误；STOP 候选仍不足；K8 的 cluster 规则与 K3 不同；随机运行、轨迹长度及非同期拓扑造成差异。
- **Supporting evidence（支持证据）：** “多样性增加而 SR 差异不明确”与该解释相容，但不是一般导航正确候选覆盖的直接证据。[K8 A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)、[K8/K3 配对](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/paired_vs_a5_2_k3.json)
- **Contradicting evidence（反证）：** 近目标 STOP 覆盖依然不足，说明至少这一类动作可能仍受生成覆盖限制；一般导航动作是否覆盖 **UNKNOWN**。[K8 A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)
- **Required falsification experiment（必要证伪检验）：** 在事先固定的决策状态与正确性定义上，分别估计候选覆盖和实际选择的正确率；必要时回放候选分支，完整保留决策及 episode 分母。如果可靠覆盖很低，或在有正确候选时现有选择几乎总能选对，则该假设不成立。日志能否恢复足够一致的状态：**UNKNOWN**。[候选诊断草案](../plans/CANDIDATE_DIAGNOSTIC.md)
- **Status（状态）：** **OPEN；关键 oracle 覆盖证据 UNKNOWN**。

## H002 — 轨迹共识是否含有导航正确性信息

- **Observation（观察）：** A5.2-K3 相对 A5.1 为 +8.83 个百分点，且 K8 禁用更新的 B 达到 365/600；这些协议同时涉及候选组成、gate 与 fallback，未隔离共识排序的作用。[K3/A5.1 配对](../experiments/a5_2_acceptance_20260917/paired_vs_a5_1.json)、[K3 验收说明](../experiments/a5_2_acceptance_20260917/README.md)、[K8 B report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/report.json)
- **Hypothesis（科学假设）：** 当候选中有可用动作时，局部轨迹一致性包含相对低温候选额外的导航正确性信息，因此能帮助选择动作；这个假设不等于“现有 gate 一定有效”。
- **Prediction（预测）：** 在同一候选集合和状态下，共识排序应与独立的导航正确性标签正相关，并比始终选择低温候选更常选到合适动作；该关系应在预先定义的状态子组中可复现。预测的效应大小 **UNKNOWN**。
- **Alternative explanations（替代解释）：** A5.2 恢复主要来自低温 fallback、纠正执行语义、不同采样随机性或 worker 拓扑；共识也可能把系统性错误候选聚成高分簇。
- **Supporting evidence（支持证据）：** K3 的较高 SR 与有 gate 的执行路径同时出现，仅是间接相容证据；没有锁定候选、单独改变排序的实验。[K3 验收说明](../experiments/a5_2_acceptance_20260917/README.md)
- **Contradicting evidence（反证）：** **UNKNOWN：没有直接检验该假设的反证实验。** 现有材料没有显示共识分数可独立预测导航正确性；早期 B 的 confidence/consistency 候选内相关约 `r≈0.259` 只描述两个模型信号之间的关系，不能证明正确性。[旧 B STOP 审计](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01/offline_stop_audit_20260910/REPORT.md)
- **Required falsification experiment（必要证伪检验）：** 锁定相同状态和候选，用事先定义、与正式 reward 独立的正确性标签比较共识选择与低温候选的选择；若共识排序无正相关或不优于低温候选，该假设不成立。未执行分支的真实结局能否可靠回放：**UNKNOWN**。
- **Status（状态）：** **OPEN；独立正确性标签与隔离对照 UNKNOWN**。

## H003 — 在线参数更新有无独立增益

- **Observation（观察）：** 同协议 K8 A 为 359/600，禁用更新 B 为 365/600；A−B 为 −1.00 个百分点，95% 配对区间 [−4.00，+1.83]，`p=0.581`。A 有 3,000 次有效参数变化，B 为 0，但两次运行非同期。[A/B 配对](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/paired_A_minus_B.json)、[A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)、[B report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/report.json)
- **Hypothesis（科学假设）：** 在控制候选生成及推理时选择后，episode 内参数更新能改善后续决策，带来独立于推理时选择的导航收益。
- **Prediction（预测）：** 相同协议的更新组应在预先确定的评估数据上稳定优于禁用更新组；若收益来自适应，差异应与先前更新之后的决策相关，而不能仅出现在更新前的决策。所需最小有意义效果 **UNKNOWN**。
- **Alternative explanations（替代解释）：** 多候选推理选择已提供全部收益；伪奖励方向不可靠；更新有正负抵消；非同期 GPU、随机路径或采样差异掩盖小效果。
- **Supporting evidence（支持证据）：** **UNKNOWN：没有观测到导航收益的直接支持证据。** A 组确实更新了参数，且协议计划尽可能对齐；这只支持“可以测试更新效应”，**不支持“更新有效”**。[A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)、[B 分析计划](../../experiments/preparations/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921/analysis_plan.json)
- **Contradicting evidence（反证）：** 当前 B 点估计高 1 个百分点，配对区间跨 0；没有观测到预期的正向优势。[A/B 配对](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/paired_A_minus_B.json)
- **Required falsification experiment（必要证伪检验）：** 事先规定有意义的最小收益及分析方案，在未用于开发的数据上复核匹配 A/B，并分析更新后决策；若足够精确的区间排除该最小收益，则此版本的增益假设被否定。独立数据与可控重复运行条件：**UNKNOWN**。
- **Status（状态）：** **NOT SUPPORTED；未证伪，小幅正负效果仍可能**。

## H004 — STOP 候选缺失是否解释部分近目标失败

- **Observation（观察）：** K8 A 的 17,516 次决策中，3,246 次含 STOP 候选；进入成功半径内的 3,135 次决策中，1,473 次含 STOP 候选，即 46.99%。早期 128 条 B 的 62 个失败 episode 中，12 个曾进入成功半径，50 个未进入。[K8 A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)、[旧 B STOP 审计](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01/offline_stop_audit_20260910/REPORT.md)
- **Hypothesis（科学假设）：** 在已经到达可成功停止区域的部分失败中，决策时没有及时生成可执行的 STOP 候选，是造成失败的一个必要限制；它不能解释从未到达区域的失败。
- **Prediction（预测）：** 对已到达半径且最终失败的决策，缺少立即可执行 STOP 的比例应高；在同状态下存在及时 STOP 时，部分失败应转为成功，前提是该候选确实可执行且被正确选择。具体可挽回数量 **UNKNOWN**。
- **Alternative explanations（替代解释）：** STOP 已生成但排序错误；STOP 位于不可执行的后续 chunk；半径标签与实际可成功停止时刻不一致；继续前进、碰撞或轨迹状态差异导致失败。
- **Supporting evidence（支持证据）：** 近目标 STOP 覆盖不足是直接观测；旧 B 审计还有半径内漏选 STOP 的案例，但协议与 K8 不同。[K8 A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)、[旧 B STOP 审计](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01/offline_stop_audit_20260910/REPORT.md)
- **Contradicting evidence（反证）：** 旧 B 的多数失败（50/62）从未进入成功半径；因此 STOP 缺失不可能单独解释全部失败。现有 K8 覆盖统计不是最终失败的反事实分析。[旧 B STOP 审计](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01/offline_stop_audit_20260910/REPORT.md)
- **Required falsification experiment（必要证伪检验）：** 预先界定“及时、可执行 STOP”及成功时刻，对全部 episode 的决策进行轨迹级核查，并在可可靠恢复的状态做分支回放；若近目标失败并不集中于无 STOP 的可停止时刻，或补足候选仍不减少该类失败，假设不成立。K8 现有记录能否支持完整反事实：**UNKNOWN**。
- **Status（状态）：** **OPEN；仅有覆盖诊断，没有因果证据**。
