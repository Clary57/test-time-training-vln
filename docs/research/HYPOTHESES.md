# 科学假设登记

证据截至：2026-10-03；本次仅重组登记结构。假设是可被证伪的科学解释，不是“增加 K”“打开更新”或“修改 gate”之类实施步骤。以下所有结果均来自反复使用的 same600 开发集；K8 A/B 非同期，K8/K3 同时改变 K 与 cluster 泛化。没有独立确认集证明当前方法优于 frozen C0。[问题与边界](PROBLEM.md)、[研究攻击记录](ATTACK_LOG.md)

四个科学问题各列出一对竞争解释，以不同预测区分它们；这些解释并不穷尽所有可能，机制也可能并存。比较须固定状态范围、正确性定义和有意义的效应阈值；“部分失败中的重要作用”不等于“所有失败的主导原因”。下述预测和反证条件均不是新观测。

状态含义：**OPEN**＝证据不足；**NOT SUPPORTED**＝现有比较没有显示预期效果，但尚未排除小效果；**FALSIFIED** 仅在预定的反证标准得到满足时使用。目前没有一项达到确认或证伪标准。每项的“削弱／证伪条件”只是判断该解释需要什么证据，不是已批准的下一实验；[NEXT_EXPERIMENT](NEXT_EXPERIMENT.md)保持空白。

## Q1 — Candidate generation vs candidate selection

问题：一般导航动作的正确候选未被生成，还是已经生成却未被选中，更限制当前 K8 协议？“合适／正确”必须用独立于正式 reward 的导航结果标准定义，不能用候选之间的相似度代替。STOP 的具体因果作用另见 Q4。

### H1a — Generation / coverage bottleneck

- **Observation / motivation（观察／动机）：** K8 A 每次决策平均有约 5.50 个不同的原子动作候选，但不同不等于正确；在成功半径内，约 53% 的决策仍没有 STOP 候选，提示至少这一类动作存在覆盖缺口。[K8 A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)
- **Hypothesis（科学假设）：** 对一般导航动作，合适动作根本未生成的覆盖缺口，比已有合适动作却未被选中的排序错误更限制表现。
- **Prediction（预测）：** 在相同决策状态上，候选集的 oracle 可行／正确动作覆盖应较低；在确有正确候选时，现有选择应较少错失它。保持选择规则不变而真正提高正确候选覆盖，应比仅重排原候选更能改善导航结果。所需覆盖差距与判定阈值 **UNKNOWN**。
- **Alternative explanation（替代解释）：** 正确候选已存在但排序错误（H1b）；STOP 覆盖不足不能代表一般导航覆盖不足；K8 的 cluster 规则与 K3 不同，随机运行、轨迹长度及非同期拓扑也可能影响差异。
- **Supporting evidence（支持证据）：** 近目标 STOP 覆盖不足与生成瓶颈相容，但不是一般导航正确候选覆盖的直接证据；一般导航的 oracle 覆盖 **UNKNOWN**。[K8 A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)
- **Contradicting evidence（反证）：** **UNKNOWN：没有直接的一般导航覆盖检验。** K8 相对 K3 的 SR 仅高 1.00 个百分点，95% 配对区间 [−3.17，+5.00]；这不支持“增加不同候选就足够”的简化解释，也不能反证正确候选覆盖瓶颈，因为 K8/K3 同时改变多个因素。[K8/K3 配对](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/paired_vs_a5_2_k3.json)
- **What would weaken / falsify it（削弱／证伪条件）：** 在事先固定的状态与正确性定义上，若可靠 oracle 覆盖明显高于实际选择正确率，且失败主要发生在已有正确候选却未选中的状态，则该解释被削弱；若达到预定差距标准，则反对其作为主要瓶颈。需分别计量覆盖与选择、必要时回放候选分支，并完整保留决策及 episode 分母。日志能否恢复足够一致的状态：**UNKNOWN**。[候选诊断草案](../plans/CANDIDATE_DIAGNOSTIC.md)
- **Status（状态）：** **OPEN；一般导航的关键 oracle 覆盖证据 UNKNOWN**。

### H1b — Selection / ranking bottleneck

- **Observation / motivation（观察／动机）：** K8 的候选多样性增加而相对 K3 的 SR 差异不明确，促使区分“没有正确候选”和“没有选中正确候选”；该比较不能单独识别 K 或排序的作用。[K8 A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)、[K8/K3 配对](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/paired_vs_a5_2_k3.json)
- **Hypothesis（科学假设）：** 对一般导航动作，候选中已经包含合适动作却未被选中的情况，比合适动作根本未生成更限制当前 K8 协议的表现。
- **Prediction（预测）：** 在可可靠回放的相同决策状态上，候选集的 oracle 可行／正确动作覆盖应明显高于实际被选动作的正确率；固定候选而改善排序应更能减少此类错失，仅增加候选多样性而不改善排序，不会稳定提高 SR。所需差距大小与判定阈值 **UNKNOWN**。
- **Alternative explanation（替代解释）：** 新候选虽不同却都错误，生成覆盖才是主要瓶颈（H1a）；STOP 候选仍不足；K8 的 cluster 规则、随机运行、轨迹长度及非同期拓扑造成差异。
- **Supporting evidence（支持证据）：** “多样性增加而 SR 差异不明确”与该解释相容，但不是一般导航正确候选已被覆盖的直接证据。[K8 A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)、[K8/K3 配对](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/paired_vs_a5_2_k3.json)
- **Contradicting evidence（反证）：** 近目标 STOP 覆盖依然不足，说明至少这一类动作可能仍受生成覆盖限制，不能把选择瓶颈外推至所有动作；一般导航动作是否覆盖 **UNKNOWN**。[K8 A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)
- **What would weaken / falsify it（削弱／证伪条件）：** 在事先固定的决策状态与正确性定义上，若可靠覆盖很低，或在有正确候选时现有选择几乎总能选对，则该假设不成立。需分别估计候选覆盖和实际选择正确率，必要时回放候选分支，并完整保留决策及 episode 分母。日志能否恢复足够一致的状态：**UNKNOWN**。[候选诊断草案](../plans/CANDIDATE_DIAGNOSTIC.md)
- **Status（状态）：** **OPEN；关键 oracle 覆盖证据 UNKNOWN**。

## Q2 — Does trajectory consensus contain correctness information?

问题：固定状态与候选后，轨迹共识是否包含相对低温候选额外的导航正确性信息？共识分数、候选间一致性和独立正确性标签须分开；这一问题不等于现有 gate 是否有效。

### H2a — Consensus contains additional correctness signal

- **Observation / motivation（观察／动机）：** A5.2-K3 相对 A5.1 为 +8.83 个百分点，且 K8 禁用更新的 B 达到 365/600；这些协议同时涉及候选组成、gate 与 fallback，未隔离共识排序的作用。[K3/A5.1 配对](../experiments/a5_2_acceptance_20260917/paired_vs_a5_1.json)、[K3 验收说明](../experiments/a5_2_acceptance_20260917/README.md)、[K8 B report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/report.json)
- **Hypothesis（科学假设）：** 当候选中有可用动作时，局部轨迹一致性包含相对低温候选额外的导航正确性信息，因此能帮助选择动作；不据此断言现有 gate 有效。
- **Prediction（预测）：** 在同一候选集合和状态下，共识排序应与独立的导航正确性标签正相关，并比始终选择低温候选更常选到合适动作；在控制低温候选偏好后，额外正确性信息仍应存在，该关系应在预先定义的状态子组中可复现。预测的效应大小 **UNKNOWN**。
- **Alternative explanation（替代解释）：** 共识主要反映 base-policy 偏好或共同错误（H2b）；A5.2 恢复主要来自低温 fallback、纠正执行语义、不同采样随机性或 worker 拓扑。
- **Supporting evidence（支持证据）：** K3 的较高 SR 与有 gate 的执行路径同时出现，仅是间接相容证据；没有锁定候选、单独改变排序的实验。[K3 验收说明](../experiments/a5_2_acceptance_20260917/README.md)
- **Contradicting evidence（反证）：** **UNKNOWN：没有直接检验该假设的反证实验。** 现有材料没有显示共识分数可独立预测导航正确性；早期 B 的 confidence/consistency 候选内相关约 `r≈0.259` 只描述两个模型信号之间的关系，不能证明正确性。[旧 B STOP 审计](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01/offline_stop_audit_20260910/REPORT.md)
- **What would weaken / falsify it（削弱／证伪条件）：** 锁定相同状态和候选，用事先定义、与正式 reward 独立的正确性标签比较共识选择与低温候选；若共识排序无正相关、不优于低温候选，或控制 base-policy 偏好后不再有额外信号，则该假设被削弱，是否达到证伪标准须按预定精度与效应阈值判断。未执行分支的真实结局能否可靠回放：**UNKNOWN**。
- **Status（状态）：** **OPEN；独立正确性标签与隔离对照 UNKNOWN**。

### H2b — Consensus mainly reflects base-policy preference or shared error

- **Observation / motivation（观察／动机）：** 现有协议比较未隔离共识，且 confidence/consistency 的候选内相关不能区分正确动作与共同错误；这使“共识只是模型偏好的一致呈现”仍是可行解释。[K3 验收说明](../experiments/a5_2_acceptance_20260917/README.md)、[旧 B STOP 审计](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01/offline_stop_audit_20260910/REPORT.md)
- **Hypothesis（科学假设）：** 局部轨迹共识主要反映 base-policy 已有偏好，或把系统性错误候选聚成高分簇；相对低温候选，它不提供稳定的额外正确性信息。
- **Prediction（预测）：** 固定状态和候选、控制低温候选偏好后，共识分数与独立正确性标签的额外关系应很弱或不存在，共识选择不应稳定优于低温候选；在共同错误的状态中，高共识仍可对应错误动作。判定“主要”及额外信号大小的阈值 **UNKNOWN**。
- **Alternative explanation（替代解释）：** 共识确有额外正确性信号（H2a），但现有 gate、fallback、采样或执行路径掩盖了它；不同状态子组的作用可能不同。
- **Supporting evidence（支持证据）：** **UNKNOWN：没有把 base-policy 偏好、共同错误与正确性分开的直接支持证据。** 未隔离的协议结果与该解释相容，但不能据此证明高分簇错误或共识冗余；候选内信号相关也不证明 H2b。[K3 验收说明](../experiments/a5_2_acceptance_20260917/README.md)、[旧 B STOP 审计](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01/offline_stop_audit_20260910/REPORT.md)
- **Contradicting evidence（反证）：** **UNKNOWN：没有同状态、同候选的隔离反证。** K3 的较高 SR 不能反证该解释，因为候选组成、gate、fallback 与运行条件未被分离。[K3/A5.1 配对](../experiments/a5_2_acceptance_20260917/paired_vs_a5_1.json)、[K3 验收说明](../experiments/a5_2_acceptance_20260917/README.md)
- **What would weaken / falsify it（削弱／证伪条件）：** 若在锁定状态与候选、使用独立正确性标签并控制 base-policy 偏好后，共识仍可复现地提供达到预定阈值的额外正确性信息，且稳定比低温候选更常选对，则该解释被削弱或否定。独立标签、效应阈值与分支回放可靠性：**UNKNOWN**。
- **Status（状态）：** **OPEN；未证明共识冗余或共同错误主导**。

## Q3 — Does online parameter adaptation provide independent value?

问题：控制候选生成及推理时选择后，episode 内在线参数更新是否带来独立收益？“能更新参数”不等于“更新有效”，“没有观测到优势”也不等于“已证明不需要更新”。有意义的最小收益尚为 **UNKNOWN**。

### H3a — Online adaptation provides independent benefit

- **Observation / motivation（观察／动机）：** 同协议 K8 A 为 359/600，禁用更新 B 为 365/600；A−B 为 −1.00 个百分点，95% 配对区间 [−4.00，+1.83]，`p=0.581`。A 有 3,000 次有效参数变化，B 为 0，但两次运行非同期。[A/B 配对](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/paired_A_minus_B.json)、[A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)、[B report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/report.json)
- **Hypothesis（科学假设）：** 在控制候选生成及推理时选择后，episode 内参数更新能改善后续决策，带来独立于推理时选择的导航收益。
- **Prediction（预测）：** 相同协议的更新组应在预先确定的评估数据上稳定优于禁用更新组，并达到预定的最小有意义收益；若收益来自适应，差异应与先前更新之后的决策相关，而不能仅出现在更新前的决策。所需最小有意义效果 **UNKNOWN**。
- **Alternative explanation（替代解释）：** 当前表现不需要有意义的在线参数适应（H3b）；多候选推理选择已提供全部收益；伪奖励方向不可靠；更新有正负抵消；非同期 GPU、随机路径或采样差异掩盖小效果。
- **Supporting evidence（支持证据）：** **UNKNOWN：没有观测到导航收益的直接支持证据。** A 组确实更新了参数，且协议计划尽可能对齐；这只支持“可以测试更新效应”，**不支持“更新有效”**。[A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)、[B 分析计划](../../experiments/preparations/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921/analysis_plan.json)
- **Contradicting evidence（反证）：** 当前 B 点估计高 1 个百分点，配对区间跨 0；没有观测到预期的正向优势。[A/B 配对](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/paired_A_minus_B.json)
- **What would weaken / falsify it（削弱／证伪条件）：** 事先规定有意义的最小收益及分析方案，在未用于开发的数据上复核匹配 A/B，并分析更新后决策；若足够精确的区间排除该最小收益，则此版本的增益假设被否定。独立数据与可控重复运行条件：**UNKNOWN**。
- **Status（状态）：** **NOT SUPPORTED；未证伪，小幅正负效果仍可能**。

### H3b — Current performance does not require meaningful online parameter adaptation

- **Observation / motivation（观察／动机）：** 禁用更新的 B 仍有已记录的导航成功，点估计高于 A；但 A/B 非同期，区间跨 0 且仍容许小幅正负效果，因此只能提出“不需要有意义适应”的解释，不能视为等效性结论。[A/B 配对](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/paired_A_minus_B.json)、[B report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/report.json)
- **Hypothesis（科学假设）：** 在当前协议范围内，多候选推理时选择等机制足以维持当前表现，不需要达到预定有意义收益阈值的在线参数适应；这不等于更新效应严格为零、更新必然有害，或所有适应方法都无效。
- **Prediction（预测）：** 控制候选生成及推理时选择后，禁用更新应可复现地维持表现；足够精确的更新组相对禁用组收益上界应低于预定最小有意义收益，更新后决策也不应显示达到该阈值的独立收益。最小收益阈值 **UNKNOWN**；仅有 A/B 相似或不显著不足以满足此预测。
- **Alternative explanation（替代解释）：** 在线更新确有独立收益（H3a），但被非同期 GPU、随机路径或采样差异掩盖；正负更新效果抵消，或收益只存在于某些预先定义的状态子组。
- **Supporting evidence（支持证据）：** B 在没有参数变化时仍达到现有点估计，与该解释相容；**没有证明“不需要有意义适应”的精确、独立对照证据**，不能因 A/B 相似而将 H3b 标为 SUPPORTED。[B report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/report.json)、[A/B 配对](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/paired_A_minus_B.json)
- **Contradicting evidence（反证）：** **UNKNOWN：没有已确认的独立更新收益反证。** 当前配对区间仍允许正向收益，且没有预定最小有意义效果，故不能据此排除 H3a；参数确有变化也不构成 H3b 的反证。[A/B 配对](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/paired_A_minus_B.json)
- **What would weaken / falsify it（削弱／证伪条件）：** 若在未用于开发的数据、锁定协议和可控重复条件下，更新组稳定取得达到预定最小有意义阈值的独立优势，并能在更新后决策中定位该收益，则此版本的“不需要有意义适应”解释被削弱或否定。独立数据、阈值与运行条件：**UNKNOWN**。
- **Status（状态）：** **OPEN；未确认不需要有意义的在线参数适应，不能由 H3a 未获支持推出 H3b 成立**。

## Q4 — What role does STOP candidate availability play?

问题：缺少及时、可执行的 STOP 是否是部分近目标失败的重要原因，还是主要提供诊断信息而非主导因果瓶颈？候选包含 STOP、STOP 位于可执行部分、STOP 被选中执行，以及当时能否成功停止，须分别判断；部分近目标作用不能外推至从未到达目标区域的失败。

两项预测须在同一预先定义的近目标失败子组、同一选择规则与同一最小有意义挽回幅度下比较；这些定义与阈值目前 **UNKNOWN**。

### H4a — Missing executable STOP is an important causal bottleneck for some near-goal failures

- **Observation / motivation（观察／动机）：** K8 A 的 17,516 次决策中，3,246 次含 STOP 候选；进入成功半径内的 3,135 次决策中，1,473 次含 STOP 候选，即 46.99%。早期 128 条 B 的 62 个失败 episode 中，12 个曾进入成功半径，50 个未进入。[K8 A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)、[旧 B STOP 审计](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01/offline_stop_audit_20260910/REPORT.md)
- **Hypothesis（科学假设）：** 在已经到达可成功停止区域的部分失败中，未及时生成可执行 STOP 是一个重要因果瓶颈；这是待验证的解释，不是覆盖统计已经证明的必要原因，且不能解释从未到达区域的失败。
- **Prediction（预测）：** 对已到达半径且最终失败的决策，缺少立即可执行 STOP 的比例应高；固定选择规则的可靠同状态回放中，补足及时、可执行 STOP 应使预先定义的近目标失败子组得到达到最小有意义幅度的挽回，需核查该候选确实被正确选择和执行。具体可挽回数量及“重要”的判定阈值 **UNKNOWN**。
- **Alternative explanation（替代解释）：** STOP 缺失主要是诊断信号，其他因素才是主导瓶颈（H4b）；STOP 已生成但排序错误，或位于不可执行的后续 chunk；半径标签与实际可成功停止时刻不一致；继续前进、碰撞或轨迹状态差异导致失败。
- **Supporting evidence（支持证据）：** 近目标 STOP 覆盖不足是直接观测；旧 B 审计还有半径内漏选 STOP 的案例，但协议与 K8 不同。覆盖不足支持动机，漏选案例提示选择机制的替代解释，二者均不是缺失的直接因果证据。[K8 A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)、[旧 B STOP 审计](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01/offline_stop_audit_20260910/REPORT.md)
- **Contradicting evidence（反证）：** 旧 B 的多数失败（50/62）从未进入成功半径，因此 STOP 缺失不可能单独解释全部失败；这限制外推范围，不直接反证部分近目标作用。现有 K8 覆盖统计不是最终失败的反事实分析。[旧 B STOP 审计](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01/offline_stop_audit_20260910/REPORT.md)
- **What would weaken / falsify it（削弱／证伪条件）：** 预先界定“及时、可执行 STOP”、成功时刻和有意义的挽回幅度，对全部 episode 的决策进行轨迹级核查，并在可可靠恢复的状态做分支回放；若近目标失败并不集中于无 STOP 的可停止时刻，或补足候选仍不减少该类失败，则该解释被削弱或不成立。K8 现有记录能否支持完整反事实：**UNKNOWN**。
- **Status（状态）：** **OPEN；仅有覆盖诊断，没有因果证据**。

### H4b — STOP absence is mainly diagnostic rather than the dominant causal bottleneck

- **Observation / motivation（观察／动机）：** K8 的半径内 STOP 覆盖统计没有判定可执行性或最终失败原因；旧 B 既有半径内漏选 STOP 的案例，也有大量从未进入成功半径的失败，但协议与 K8 不同。[K8 A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)、[旧 B STOP 审计](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01/offline_stop_audit_20260910/REPORT.md)
- **Hypothesis（科学假设）：** STOP 缺失主要提示候选、轨迹或停止时刻的问题；在预先界定的近目标失败子组中，它不是主导因果瓶颈，选择、执行位置或实际可停止时刻等因素更能解释失败。该解释允许少量缺失效应；不同状态子组可有不同瓶颈。
- **Prediction（预测）：** 固定状态和选择规则，仅补足及时、可执行 STOP 对同一近目标失败子组的挽回应很小或低于 H4a 使用的最小有意义幅度；多数失败仍需由选择、执行或停止时刻等因素解释。总体失败中从未到达区域的部分也不应因补足 STOP 而被解释为可挽回。具体幅度与“主导”的判定阈值 **UNKNOWN**。
- **Alternative explanation（替代解释）：** 缺少可执行 STOP 确实是某些近目标失败的重要原因（H4a），但决策级覆盖汇总和旧协议审计掩盖了它；不同状态子组可能存在不同瓶颈。
- **Supporting evidence（支持证据）：** 旧 B 的 50/62 个失败从未进入成功半径，限制 STOP 对总体失败的解释；半径内漏选 STOP 的案例与选择瓶颈相容。但这些只提供诊断与适用范围，**不直接支持 K8 近目标失败由其他因素主导**。[旧 B STOP 审计](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01/offline_stop_audit_20260910/REPORT.md)
- **Contradicting evidence（反证）：** K8 的近目标 STOP 覆盖不足使缺失的因果作用仍然可能；它与 H4a 相容，但尚无补足候选的反事实结果，不能据此反证 H4b。[K8 A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)
- **What would weaken / falsify it（削弱／证伪条件）：** 若完整轨迹核查与可靠同状态回放显示，固定选择规则而补足及时、可执行 STOP 可挽回达到同一预定最小有意义幅度的近目标失败，且改善可归于候选可用性，则其低挽回幅度预测被削弱或否定；若进一步达到预定主导阈值，则反对其非主导解释。需保留全部 episode 分母，不能将旧 B 的总体失败分布当作 K8 子组的因果证据。阈值与完整反事实可恢复性：**UNKNOWN**。
- **Status（状态）：** **OPEN；诊断边界明确，但“缺失不是主导原因”未获因果确认**。

## Mapping from previous registry

Previous H001 -> Q1

Previous H002 -> Q2

Previous H003 -> Q3

Previous H004 -> Q4
