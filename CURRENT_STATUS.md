# 当前研究状态

更新：2026-10-03。本文是项目的简短入口；实证结论以所链接的验收产物为准。

## 当前研究问题

在没有测试时专家动作或目标反馈的视觉语言导航中，多候选局部轨迹共识能否帮助选择动作？episode 内参数更新能否在相同推理时选择之外提供独立收益？[问题定义](docs/research/PROBLEM.md)

## 当前方法与版本

当前主要协议为 **A5.2-K8**：同一决策状态顺序采样 1 个 T=0.2 与 7 个 T=1.5 候选，依据局部轨迹共识、候选质量与 gate 选择动作；gate 拒绝时执行低温 candidate 0。A 在 episode 内更新 LoRA，B 使用相同候选与选择流程但跳过 optimizer step；两者均在 episode 间重置。canonical C0 是单候选冻结 NaVIDA。[A 协议](experiments/preparations/a5_2_k8_checkpoint_gpu467_af8e58b_20260919/protocol_manifest.json)、[B 协议](experiments/preparations/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921/protocol_manifest.json)、[版本演进](PROJECT_OVERVIEW.md)

## 最强的当前证据

| 已验收运行 | 成功数 / 样本数 | 证据 |
|---|---:|---|
| canonical C0，全量 `val_unseen` | 1,075/1,839（58.46%） | [只读验收说明](docs/experiments/A5_C0_20260914.md) |
| canonical C0，same600 | 347/600（57.83%） | [同上](docs/experiments/A5_C0_20260914.md) |
| A5.2-K3 A，same600 | 353/600（58.83%） | [独立产物验收，附 coordinator 例外](docs/experiments/a5_2_acceptance_20260917/README.md) |
| A5.2-K8 A，same600 | 359/600（59.83%） | [严格 report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/report.json)、[validation](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/validation.json) |
| A5.2-K8 B，same600 | 365/600（60.83%） | [严格 report](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/report.json)、[validation](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/validation.json) |

同 ID 的 K8 A−B 为 **−1.00 个百分点**，95% 配对 bootstrap 区间 **[−4.00，+1.83]**；K8 A−K3 为 **+1.00 个百分点**，区间 **[−3.17，+5.00]**。两项均没有确立正向优势。[A/B 配对](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921_batch01/analysis/paired_A_minus_B.json)、[K8/K3 配对](/hdd3/wangziyu/runs/test-time-training-vln/a5_2_k8_checkpoint_gpu467_af8e58b_20260919_batch01/analysis/paired_vs_a5_2_k3.json)

> **证据边界：** same600 已被反复用于开发；A/B 运行非同期；K8/K3 比较同时改变 K 与 cluster 泛化规则；目前没有独立确认集证明当前方法优于冻结基线。[A 分析计划](experiments/preparations/a5_2_k8_checkpoint_gpu467_af8e58b_20260919/analysis_plan.json)、[B 分析计划](experiments/preparations/a5_2_k8_B_only_same600_dynamic3x200_af8e58b_20260921/analysis_plan.json)

## 当前有效的发现与假设

**经验发现：** A5.2-K3/K8 的 same600 SR 接近本机 C0；K8 相对 K3 的差异不明确；同协议 K8 B 的点估计高于 A，但区间跨 0；K8 A 在成功半径内约 47% 的决策含 STOP 候选。这些都是描述性结果，不能直接推断原因。[F001–F007](docs/research/FINDINGS.md)

**待检验假设：** 候选覆盖与排序的相对限制；轨迹共识对导航正确性的独立信息；在线更新的额外收益；STOP 候选缺失对部分近目标失败的影响。[H001–H004](docs/research/HYPOTHESES.md)

## 最大不确定性与可辩护主张

最大不确定性是：**现有局部共识信号是否可靠识别正确动作，以及在线参数更新是否提供推理时多候选选择之外的收益。** 目前缺少一般导航候选的 oracle 覆盖/排序证据和独立确认集。[研究攻击记录](docs/research/ATTACK_LOG.md)

当前可辩护的论文主张是：**项目已经构建并验收了一个无测试时外部标签、episode-local 的多候选选择及可选在线更新系统；开发集上的 A5.2 表现接近本机冻结基线，但尚未证明 K8 的独立收益、在线更新收益或独立确认的基线优势。** [论文叙述边界](docs/research/PAPER_STORY.md)

## 下一科学问题与详细文档

下一步先完成独立科学审查：判断现有证据能否把候选覆盖、共识排序与更新作用分开，确定哪些主张能被反证。**尚未选定下一实验；[NEXT_EXPERIMENT.md](docs/research/NEXT_EXPERIMENT.md)保持空白。**

详细入口：[问题](docs/research/PROBLEM.md) · [假设](docs/research/HYPOTHESES.md) · [发现](docs/research/FINDINGS.md) · [论文叙述](docs/research/PAPER_STORY.md) · [攻击记录](docs/research/ATTACK_LOG.md) · [文献地图](docs/literature/LITERATURE_MAP.md) · [早期实验结果索引](docs/experiments/RESULTS_INDEX.md) · [历史会议与存档](docs/README.md)。
