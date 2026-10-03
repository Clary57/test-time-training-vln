# 经验发现

更新：2026-10-03。此处只登记已完成运行的测量结果和验收范围。**Confidence** 指证据完整性及分析范围，不表示因果确信度。每项的 **Scientific interpretation** 都受其证据边界约束；假设与攻击分别见 [HYPOTHESES](HYPOTHESES.md) 和 [ATTACK_LOG](ATTACK_LOG.md)。

> same600 已反复用于开发；K8 A/B 非同期；K8/K3 同时改变 K 与 cluster 规则；当前没有独立确认集证明本方法优于 frozen C0。配对区间与 `p` 值均不得越过这些设计边界。[K8 A 分析计划](../../experiments/preparations/a5_2_k8_checkpoint_gpu467_af8e58b_20260919/analysis_plan.json)、[K8 B 分析计划](../../experiments/preparations/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921/analysis_plan.json)

## F001 — 本机 canonical C0 基线

- **Evidence：** 完整 `val_unseen` 1,839 条中成功 1,075 条，SR 58.46%；与开发集同 ID 的 600 条中成功 347 条，SR 57.83%。全量运行 8 个 worker exit 0，1,839 个唯一 stats 与 selection/manifest 对应；验收是只读复核，没有单独保存自动生成的统一 validation/report。[A5/C0 验收说明](../experiments/A5_C0_20260914.md)、[结果索引](../experiments/RESULTS_INDEX.md)
- **Confidence：** 较高，限于本机该 frozen 协议和文档所述只读验收；不把旧论文表的外部数字混入。[C0 全量分析计划](../../experiments/official_navida_c_full/analysis_plan.json)
- **Scientific interpretation：** 为本项目同 ID 开发比较提供 frozen 参考值；不单独证明任何适应方法优劣。
- **Remaining uncertainty：** 本机与文献基线报告数字是否完全同协议、可直接比较：**UNKNOWN**。

## F002 — 早期 A5 T1.5 K3 在 same600 低于 C0

- **Evidence：** A5 成功 305/600（50.83%），同 ID C0 成功 347/600（57.83%），A5−C0 为 −7.00 个百分点；配对 bootstrap 95% 区间 [−11.33，−2.67]，事后 exact McNemar `p=0.002414`。[A5/C0 验收说明](../experiments/A5_C0_20260914.md)、[A5 report](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_A600_gpu67_83a4d2f_20260911_batch01/analysis/report.json)
- **Confidence：** 对这两次已验收运行的数值较高；配对检验是事后探索性补充，且 A5 的单臂计划并未预锁定该因果比较。
- **Scientific interpretation：** 本机开发集上的早期 A5 整体协议表现低于同 ID C0；不能隔离在线更新、候选数、温度或动作选择的作用。
- **Remaining uncertainty：** 退化由哪一因素导致：**UNKNOWN**；同配置跨运行稳定性 **UNKNOWN**。

## F003 — A5.1 executed-prefix 修正后的开发集结果

- **Evidence：** A5.1 成功 300/600（50.00%）；相对旧 A5 305/600 为 −0.83 个百分点，95% 配对区间 [−4.50，+2.83]、exact McNemar `p=0.719`。首轮自动 analysis 因校验逻辑报错，`analysis_corrected_v1/validation.json` 对原产物标记 PASS。[A5.1 修正验收](/hdd3/wangziyu/runs/test-time-training-vln/a5_1_executed_span_A600_gpu28_19a2e5e_20260915_batch02/analysis_corrected_v1/validation.json)、[A5.1 report](/hdd3/wangziyu/runs/test-time-training-vln/a5_1_executed_span_A600_gpu28_19a2e5e_20260915_batch02/analysis_corrected_v1/report.json)、[A5.1/A5 配对](/hdd3/wangziyu/runs/test-time-training-vln/a5_1_executed_span_A600_gpu28_19a2e5e_20260915_batch02/analysis_corrected_v1/paired_vs_old_a5.json)
- **Confidence：** 较高，限于修正后的验收；不能把初次 FAIL 抹去或将其误写为 episode 运行失败。
- **Scientific interpretation：** 这次修正没有观察到明确的 SR 恢复；不能证明修正本身有害或无效。
- **Remaining uncertainty：** 非同期运行和随机过程对该小差异的影响 **UNKNOWN**。

## F004 — A5.2-K3 开发集表现回到本机 C0 附近

- **Evidence：** K3 成功 353/600（58.83%）；相对 A5.1 为 +8.83 个百分点，95% 配对区间 [+4.50，+13.33]，`p=0.000195`；相对同 ID C0 为 +1.00 个百分点，95% 区间 [−2.50，+4.67]，`p=0.661`。[K3 独立验收 report](../experiments/a5_2_acceptance_20260917/report.json)、[K3/A5.1 配对](../experiments/a5_2_acceptance_20260917/paired_vs_a5_1.json)、[K3/C0 配对](../experiments/a5_2_acceptance_20260917/paired_vs_canonical_c0.json)
- **Confidence：** 600 条原始产物获独立验收，但须注明原 coordinator clean-tree 检查退出 1；结论是“独立产物验收通过，附原始流程例外”，不是原 coordinator PASS。[验收例外说明](../experiments/a5_2_acceptance_20260917/README.md)
- **Scientific interpretation：** A5.2-K3 整套协议在开发集上接近 C0，并高于历史 A5.1；不能把差额单独归因于 gate、fallback、采样温度或在线更新。
- **Remaining uncertainty：** 各项变动的独立作用和独立确认集表现：**UNKNOWN**。

## F005 — K8 A 与历史 K3、C0 的比较

- **Evidence：** K8 A 严格验收 PASS，成功 359/600（59.83%），3,000 次 optimizer step 均有有效参数变化。相对 K3 为 +1.00 个百分点，95% 配对区间 [−3.17，+5.00]，`p=0.689`；相对同 ID C0 为 +2.00 个百分点，区间 [−1.83，+5.67]，`p=0.342`。[K8 A validation](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/validation.json)、[K8 A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)、[K8/K3 配对](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/paired_vs_a5_2_k3.json)、[K8/C0 配对](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/paired_vs_canonical_c0.json)
- **Confidence：** 对该运行及配对计数较高；对 K 的因果效果低，因为 K 和 cluster 泛化共同变化且参考为非同期开发运行。[K8 A 分析计划](../../experiments/preparations/a5_2_k8_checkpoint_gpu467_af8e58b_20260919/analysis_plan.json)
- **Scientific interpretation：** 当前观察没有确立 K8 相对 K3 或 C0 的优势；参数发生更新不等于更新产生收益。
- **Remaining uncertainty：** 纯 K 效应、cluster 规则效应和独立确认表现：**UNKNOWN**。

## F006 — 同协议 K8 禁用更新 B 与更新 A

- **Evidence：** B 严格验收 PASS，成功 365/600（60.83%），0 次 optimizer step、0 次有效参数变化；A 成功 359/600。A−B 为 −1.00 个百分点，95% 配对区间 [−4.00，+1.83]，exact McNemar `p=0.581`，discordant pairs 为 A-only 38、B-only 44。[B validation](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/validation.json)、[B report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/report.json)、[A/B 配对](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/paired_A_minus_B.json)
- **Confidence：** 对两个已验收运行和同 ID 配对较高；因果解释受非同期、可能不同物理 GPU 及开发集复用限制。[B 分析计划](../../experiments/preparations/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921/analysis_plan.json)
- **Scientific interpretation：** 当前数据没有显示在线更新的独立正向收益，也不能据 B 的较高点估计断言更新有害。
- **Remaining uncertainty：** 更小效果、跨运行重复性及独立集上的更新效应：**UNKNOWN**。

## F007 — K8 A 的候选与 STOP 覆盖诊断

- **Evidence：** K8 A 共 17,516 次决策，平均 5.50 个唯一原子动作候选；3,246 次决策存在 STOP 候选，占 18.53%。进入成功半径内的 3,135 次决策中，1,473 次存在 STOP 候选，占 46.99%。K8 B 对应为 16,984 次决策、平均 5.49 个唯一候选，半径内 STOP 覆盖为 1,444/3,116（46.34%）。[A report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)、[B report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/report.json)
- **Confidence：** 对日志内的描述性分母和计数较高；不同组经过不同实际轨迹，决策并非可直接配对的相同状态。
- **Scientific interpretation：** 多样性和 STOP 覆盖是候选日志性质；它们尚不能证明候选正确、共识选错或 STOP 缺失导致最终失败。
- **Remaining uncertainty：** 一般导航 oracle 覆盖、近目标 STOP 的可执行位置及反事实成功率：**UNKNOWN**。
