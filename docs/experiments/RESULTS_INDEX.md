# 实验结果索引

路径基准：`/hdd3/wangziyu/runs/test-time-training-vln`。2026-09-14 增补 A5 600 与 C0 全量结果说明；未移动或重写原始结果。VERIFIED只针对明确列出的检查；DOCUMENTED是既有分析文档的记录；UNKNOWN不得补成完成。原有目录存在性快照见[RUN_DIRECTORIES.md](RUN_DIRECTORIES.md)，新增运行的完整说明见[A5/C0实验说明](A5_C0_20260914.md)。

| 实验 | 样本 / Success | 证据与验收范围 | 比较边界 | 结果根 |
|---|---|---|---|---|
| pilot representative20 | 20，8成功 | summary已读；未在根发现final_validation | 旧pilot，非确认性 | [representative20_g3_seed41_20260817_133613](/hdd3/wangziyu/runs/test-time-training-vln/representative20_g3_seed41_20260817_133613) |
| 历史A representative72 | 72，25成功 | summary与final_validation passed=true | 含原20与扩展52，旧full TTRL | [representative72_extend52_seed41_20260817_204147](/hdd3/wangziyu/runs/test-time-training-vln/representative72_extend52_seed41_20260817_204147) |
| 历史B_disable_update72 | 72，22成功 | summary与final_validation passed=true | 不是当前A5/B | [consensus_only72_disable_update_seed41_20260818_005837](/hdd3/wangziyu/runs/test-time-training-vln/consensus_only72_disable_update_seed41_20260818_005837) |
| 历史C_project72 | 72，15成功 | summary与final_validation passed=true | 旧frozen协议，不是canonical | [frozen_navida72_seed41_20260818_005837](/hdd3/wangziyu/runs/test-time-training-vln/frozen_navida72_seed41_20260818_005837) |
| 历史confirmatory128 A | 128，26成功 | summary与final_validation支持 | full TTRL；不是A5 | [confirmatory128_seed42_20260818_144052/A_full_ttrl](/hdd3/wangziyu/runs/test-time-training-vln/confirmatory128_seed42_20260818_144052/A_full_ttrl) |
| 历史confirmatory128 C_project | 128，27成功 | summary支持；历史验收文件存在 | 旧baseline协议混杂 | [confirmatory128_seed42_20260818_144052/C_frozen_navida](/hdd3/wangziyu/runs/test-time-training-vln/confirmatory128_seed42_20260818_144052/C_frozen_navida) |
| canonical C0历史128起点 | 原根及3次continuation合计128，72成功 | 重新计数128条唯一stats；本次未重做完整strict验收 | 需合并原根及三个continuation，不能只看最终根 | [official_navida_c128_20260827_095359](/hdd3/wangziyu/runs/test-time-training-vln/official_navida_c128_20260827_095359) |
| official-protocol历史TTRL | 仅48，27成功 | 三个worker exit=0，48条唯一stats | 部分运行；不能报告成128已完成 | [official_navida_ttrl_a128_20260827_204743](/hdd3/wangziyu/runs/test-time-training-vln/official_navida_ttrl_a128_20260827_204743) |
| new128 C0 | 128，74成功 | 128条唯一stats；8个worker exit=0 | 仅本次计数，不冒称重做全部strict validation | [official_navida_c128_seed43_matched_20260830_095323](/hdd3/wangziyu/runs/test-time-training-vln/official_navida_c128_seed43_matched_20260830_095323) |
| new128 A4 | 128，72成功 | 128条唯一stats；worker exit=0 | 已用于开发 | [ttrl_a4_independent128_seed43_20260829_164245](/hdd3/wangziyu/runs/test-time-training-vln/ttrl_a4_independent128_seed43_20260829_164245) |
| new128 R13 A5 | 128，71成功 | 128条唯一stats；4个worker exit=0 | T0.3、greedy+stochastic、更新开启；与当前B不配对 | [ttrl_a5_independent128_seed43_multigpu_20260902_141135](/hdd3/wangziyu/runs/test-time-training-vln/ttrl_a5_independent128_seed43_multigpu_20260902_141135) |
| A5 confidence48 | DOCUMENTED：48，29成功 | 见audits/a5_confidence48_analysis_20260903_092453/HANDOFF.md | 前缀选择，非独立确认；本次未重算原始48 | [a5_confidence48_retry_20260902_231602](/hdd3/wangziyu/runs/test-time-training-vln/a5_confidence48_retry_20260902_231602) |
| 先前T1.5 B-only | 128，66成功 | PASS_WITH_PROTOCOL_AMENDMENT；下方直达验收 | 完整new128开发；无同批A，不作配对更新效应结论 | [a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01) |
| A5 T1.5 全 stochastic A600 | 600，305成功，SR 50.83% | 机器生成 `STRICT_VALIDATION_PASS`；2个worker exit=0，600条唯一stats/steps/STOP trace | 扩大开发集，重叠new128；A-only，不能单独证明更新收益 | [a5_t15_stochastic_A600_gpu67_83a4d2f_20260911_batch01](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_A600_gpu67_83a4d2f_20260911_batch01) |
| canonical C0 full val_unseen | 1,839，1,075成功，SR 58.46% | 8个worker exit=0；2026-09-14只读验收核对1,839唯一ID、manifest和有限指标；没有单独保存自动validation | 全量baseline；与A5同ID的600条347成功，跨协议比较不能隔离reward/optimizer | [canonical_c0_full1839_gpu34_83a4d2f_20260912_batch01](/hdd3/wangziyu/runs/test-time-training-vln/canonical_c0_full1839_gpu34_83a4d2f_20260912_batch01) |
| stop diagnostic18 | 18设计；多臂、多次修订 | 存在多个validation/salvage/paired报告，须按适用范围读取 | 机制诊断，不能把条件性salvage当完整严格配对 | [stop_diagnostic18_20260820_103444](/hdd3/wangziyu/runs/test-time-training-vln/stop_diagnostic18_20260820_103444) |
| dev16 | 目录存在；本次不补写指标 | UNKNOWN：未独立验收 | 开发 | [ttrl_cluster_quality_dev16_20260829_111444](/hdd3/wangziyu/runs/test-time-training-vln/ttrl_cluster_quality_dev16_20260829_111444) |
| dev48 extension32 | 目录存在；本次不补写指标 | UNKNOWN：未独立验收 | 开发、扩展设计，不视为独立确认 | [ttrl_cluster_quality_dev48_extension32_20260829_125828](/hdd3/wangziyu/runs/test-time-training-vln/ttrl_cluster_quality_dev48_extension32_20260829_125828) |

## 先前new128 B的最终证据

- [最终report](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01/analysis_after_continuation/report.json)：SR=51.5625%，Wilson 95% CI=[42.986%,60.048%]；OSR=60.9375%；SPL=0.471364、nDTW=0.654690、SDTW=0.456181、DTG=4.718697 m、PL=9.726449 m。
- [最终validation](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01/analysis_after_continuation/validation.json)：PASS_WITH_PROTOCOL_AMENDMENT。
- [continuation说明](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01/continuation_20260909_2023/README.md)：原两个worker完成后更改资源门槛，仅完成未启动worker；原execution_status仍保留中断状态。
- [实际展开参数](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01/preparation/expanded_args.json)、[准备HANDOFF](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01/preparation/HANDOFF.md)：T1.5、K3全stochastic、disable-update、4分片在GPU4串行。
- [离线STOP审计](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01/offline_stop_audit_20260910/REPORT.md)、[汇总](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01/offline_stop_audit_20260910/summary.json)、[回放验收](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01/offline_stop_audit_20260910/validation.json)：3269决策回放一致，当前用于方法诊断。
- [封存输入哈希](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01/offline_stop_audit_20260910/input_hashes.json)：该审计程序锁定原代码HEAD；整理分支不修改它来规避锁。

## 其他关键证据入口

- [历史baseline fidelity审计](/hdd3/wangziyu/runs/test-time-training-vln/audits/navida_canonical_baseline_fidelity_audit_20260826_002859/HANDOFF.md)
- [R13 A5 reward/update诊断](/hdd3/wangziyu/runs/test-time-training-vln/audits/a5_reward_update_diagnostic_20260902_214100/HANDOFF.md)
- [confidence48诊断](/hdd3/wangziyu/runs/test-time-training-vln/audits/a5_confidence48_analysis_20260903_092453/HANDOFF.md)
- [STOP B严格验收历史](/hdd3/wangziyu/runs/test-time-training-vln/audits/stop_diagnostic18_B_final_validation_and_paired_analysis_20260824_102743/HANDOFF.md)
- [STOP匹配和条件性分析](/hdd3/wangziyu/runs/test-time-training-vln/audits/stop_diagnostic18_A_gpu7_match_B_paired_analysis_20260825_133110/HANDOFF.md)
- [旧C0 continuation根目录链](/hdd3/wangziyu/runs/test-time-training-vln/official_navida_c128_20260827_095359_continuation3_gpu7_20260827_175847/previous_roots.txt)

## 不可直接比较与尚无证据的部分

历史full TTRL的26/128不是A5；旧C_project的27/128也不是canonical C0。historical128与new128是不同selection。旧C0与新C0成功数72和74的差异不能脱离ID集合解读。new128已经用于A4/A5及温度开发，不能因目录名含independent就继续称独立确认集。

R13 A5与当前B同时存在代码版本、采样温度/组成、optimizer、GPU/worker拓扑和配置/观测记录路径差异；71与66的差值不隔离其中任何单因素。当前B缺少同批A，不补写配对bootstrap/McNemar。confidence48也不替代完整new128。

仓库official_navida_c_full的1839准备包已有完成的全量C0结果，对应本页新增行；原始启动脚本的分支锁在封存准备包中按新HEAD调整，算法与评估协议未变。2026-09-12 的C0 selection600尝试在worker分别完成5条和8条时按用户要求中止，不与全量结果合并。其他smoke、retry及aborted启动目录只列存在性，不据名称填结果。会议、用户转述以及旧论文草稿数字不能替代原始stats与验收。

后续可复现的新实验应保留同批命令、manifest、环境和资产SHA256、ID顺序/分片、物理GPU UUID、exit code及重试/amendment。旧包是证据，不能改写成适配整理后的新HEAD。
