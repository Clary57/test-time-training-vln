# 实验目录存在性索引

检查日期2026-09-10。结果根：`/hdd3/wangziyu/runs/test-time-training-vln`。本表仅列根目录的直接子项；未标明验收的条目为**UNKNOWN：完成状态未在本次独立验收**，不根据目录名补全结果。目录mtime只用于定位，不能当实验完成时间。小型证据检查另受最大4层深度限制，跳过模型、权重、图片、视频、缓存等目录。

| 目录（绝对路径） | 目录mtime（本地） | 说明 |
|---|---|---|
| [a5_confidence48_20260902_231212](/hdd3/wangziyu/runs/test-time-training-vln/a5_confidence48_20260902_231212) | 2026-09-02T23:13:23 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [a5_confidence48_retry_20260902_231602](/hdd3/wangziyu/runs/test-time-training-vln/a5_confidence48_retry_20260902_231602) | 2026-09-02T23:16:36 | 历史48前缀诊断；不是独立确认集 |
| [a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01) | 2026-09-10T11:31:21 | 当前B已完成；读analysis_after_continuation及offline_stop_audit，不读原中断标志推断完成状态 |
| [audits](/hdd3/wangziyu/runs/test-time-training-vln/audits) | 2026-09-03T09:24:53 | 历史CPU审计、验收、失败/续跑记录；不修改 |
| [confirmatory128_seed42_20260818_144052](/hdd3/wangziyu/runs/test-time-training-vln/confirmatory128_seed42_20260818_144052) | 2026-08-18T14:49:39 | 历史full TTRL与C_project；不是A5 |
| [consensus_only72_disable_update_seed41_20260818_005837](/hdd3/wangziyu/runs/test-time-training-vln/consensus_only72_disable_update_seed41_20260818_005837) | 2026-08-18T05:02:41 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [frozen_navida72_seed41_20260818_005837](/hdd3/wangziyu/runs/test-time-training-vln/frozen_navida72_seed41_20260818_005837) | 2026-08-18T12:24:59 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [launch_logs](/hdd3/wangziyu/runs/test-time-training-vln/launch_logs) | 2026-09-09T14:00:01 | 启动记录，不是最终指标 |
| [navida_frozen_factor_smoke4_20260826_165446](/hdd3/wangziyu/runs/test-time-training-vln/navida_frozen_factor_smoke4_20260826_165446) | 2026-08-26T16:59:49 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [official_navida_c128_20260827_095359](/hdd3/wangziyu/runs/test-time-training-vln/official_navida_c128_20260827_095359) | 2026-08-27T10:28:18 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [official_navida_c128_20260827_095359_continuation2_gpu7_20260827_161548](/hdd3/wangziyu/runs/test-time-training-vln/official_navida_c128_20260827_095359_continuation2_gpu7_20260827_161548) | 2026-08-27T16:28:12 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [official_navida_c128_20260827_095359_continuation3_gpu7_20260827_175847](/hdd3/wangziyu/runs/test-time-training-vln/official_navida_c128_20260827_095359_continuation3_gpu7_20260827_175847) | 2026-08-27T18:31:02 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [official_navida_c128_20260827_095359_continuation_gpu7_20260827_125857](/hdd3/wangziyu/runs/test-time-training-vln/official_navida_c128_20260827_095359_continuation_gpu7_20260827_125857) | 2026-08-27T13:45:01 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [official_navida_c128_seed43_matched_20260830_095323](/hdd3/wangziyu/runs/test-time-training-vln/official_navida_c128_seed43_matched_20260830_095323) | 2026-08-30T09:56:15 | new128 C0，128条唯一stats，74成功 |
| [official_navida_ttrl_a128_20260827_192115](/hdd3/wangziyu/runs/test-time-training-vln/official_navida_ttrl_a128_20260827_192115) | 2026-08-27T19:23:09 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [official_navida_ttrl_a128_20260827_194811](/hdd3/wangziyu/runs/test-time-training-vln/official_navida_ttrl_a128_20260827_194811) | 2026-08-27T19:50:08 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [official_navida_ttrl_a128_20260827_204743](/hdd3/wangziyu/runs/test-time-training-vln/official_navida_ttrl_a128_20260827_204743) | 2026-08-27T23:22:29 | 本目录48条stats，不能称完整A128 |
| [official_navida_ttrl_native_lora_smoke1_20260827_201422](/hdd3/wangziyu/runs/test-time-training-vln/official_navida_ttrl_native_lora_smoke1_20260827_201422) | 2026-08-27T20:15:43 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [official_navida_ttrl_native_lora_smoke1_retry_20260827_203220](/hdd3/wangziyu/runs/test-time-training-vln/official_navida_ttrl_native_lora_smoke1_retry_20260827_203220) | 2026-08-27T20:33:02 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [official_navida_ttrl_visual_fix_smoke1_20260827_204000](/hdd3/wangziyu/runs/test-time-training-vln/official_navida_ttrl_visual_fix_smoke1_20260827_204000) | 2026-08-27T20:41:40 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [preparations](/hdd3/wangziyu/runs/test-time-training-vln/preparations) | 2026-09-09T13:57:50 | 多个封存准备包，不等于多个已完成实验 |
| [representative20_g3_seed41_20260817_133613](/hdd3/wangziyu/runs/test-time-training-vln/representative20_g3_seed41_20260817_133613) | 2026-08-17T15:21:49 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [representative72_extend52_seed41_20260817_204147](/hdd3/wangziyu/runs/test-time-training-vln/representative72_extend52_seed41_20260817_204147) | 2026-08-17T23:23:07 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [stop_diagnostic18_20260820_103444](/hdd3/wangziyu/runs/test-time-training-vln/stop_diagnostic18_20260820_103444) | 2026-08-24T20:53:51 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [stop_trace_smoke_20260819_184639](/hdd3/wangziyu/runs/test-time-training-vln/stop_trace_smoke_20260819_184639) | 2026-08-19T18:51:39 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [ttrl_a4_independent128_seed43_20260829_164245](/hdd3/wangziyu/runs/test-time-training-vln/ttrl_a4_independent128_seed43_20260829_164245) | 2026-08-29T16:44:26 | A4，128条唯一stats，72成功；开发材料 |
| [ttrl_a5_independent128_seed43_20260902_140204](/hdd3/wangziyu/runs/test-time-training-vln/ttrl_a5_independent128_seed43_20260902_140204) | 2026-09-02T14:11:35 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [ttrl_a5_independent128_seed43_multigpu_20260902_141135](/hdd3/wangziyu/runs/test-time-training-vln/ttrl_a5_independent128_seed43_multigpu_20260902_141135) | 2026-09-02T14:11:36 | R13 A5，128条唯一stats，71成功；开发材料 |
| [ttrl_cluster_quality_dev16_20260829_111444](/hdd3/wangziyu/runs/test-time-training-vln/ttrl_cluster_quality_dev16_20260829_111444) | 2026-08-29T11:18:59 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [ttrl_cluster_quality_dev48_extension32_20260829_125828](/hdd3/wangziyu/runs/test-time-training-vln/ttrl_cluster_quality_dev48_extension32_20260829_125828) | 2026-08-29T13:03:43 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [ttrl_reward_ablation_smoke4_20260828_225826](/hdd3/wangziyu/runs/test-time-training-vln/ttrl_reward_ablation_smoke4_20260828_225826) | 2026-08-29T10:24:43 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [ttt_full2_gpu3_20260816_150734](/hdd3/wangziyu/runs/test-time-training-vln/ttt_full2_gpu3_20260816_150734) | 2026-08-16T15:38:57 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [ttt_full4_ndtw_gpu3_20260816_172112](/hdd3/wangziyu/runs/test-time-training-vln/ttt_full4_ndtw_gpu3_20260816_172112) | 2026-08-16T21:42:06 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [ttt_ndtw_smoke2_gpu3_20260816_165931](/hdd3/wangziyu/runs/test-time-training-vln/ttt_ndtw_smoke2_gpu3_20260816_165931) | 2026-08-16T17:01:27 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [ttt_ndtw_smoke_gpu3_20260816_162925](/hdd3/wangziyu/runs/test-time-training-vln/ttt_ndtw_smoke_gpu3_20260816_162925) | 2026-08-16T16:29:45 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [ttt_smoke_gpu3_20260816_135001](/hdd3/wangziyu/runs/test-time-training-vln/ttt_smoke_gpu3_20260816_135001) | 2026-08-16T13:50:09 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [ttt_smoke_gpu3_20260816_140152](/hdd3/wangziyu/runs/test-time-training-vln/ttt_smoke_gpu3_20260816_140152) | 2026-08-16T14:02:29 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |
| [ttt_smoke_gpu3_20260816_143049](/hdd3/wangziyu/runs/test-time-training-vln/ttt_smoke_gpu3_20260816_143049) | 2026-08-16T14:32:39 | 历史/辅助目录；完成状态UNKNOWN，需查其manifest、stats、exit及验收 |

根目录散落的小型启动日志亦保持原位：

| 文件 | bytes |
|---|---:|
| [official_navida_c128_20260827_095359.tmux.log](/hdd3/wangziyu/runs/test-time-training-vln/official_navida_c128_20260827_095359.tmux.log) | 28 |
| [official_navida_c128_20260827_095359_continuation2_gpu7_20260827_161548.monitor.log](/hdd3/wangziyu/runs/test-time-training-vln/official_navida_c128_20260827_095359_continuation2_gpu7_20260827_161548.monitor.log) | 172 |
| [official_navida_c128_20260827_095359_continuation3_gpu7_20260827_175847.tmux.log](/hdd3/wangziyu/runs/test-time-training-vln/official_navida_c128_20260827_095359_continuation3_gpu7_20260827_175847.tmux.log) | 0 |
| [official_navida_c128_20260827_095359_continuation_gpu7_20260827_125857.tmux.log](/hdd3/wangziyu/runs/test-time-training-vln/official_navida_c128_20260827_095359_continuation_gpu7_20260827_125857.tmux.log) | 56 |
| [official_navida_ttrl_a128_20260827_192115.monitor.log](/hdd3/wangziyu/runs/test-time-training-vln/official_navida_ttrl_a128_20260827_192115.monitor.log) | 202 |
| [official_navida_ttrl_a128_20260827_194811.monitor.log](/hdd3/wangziyu/runs/test-time-training-vln/official_navida_ttrl_a128_20260827_194811.monitor.log) | 222 |
| [official_navida_ttrl_a128_20260827_204743_tmux.log](/hdd3/wangziyu/runs/test-time-training-vln/official_navida_ttrl_a128_20260827_204743_tmux.log) | 28 |

主要研究结果与比较边界见[结果索引](RESULTS_INDEX.md)。本次没有移动或修改上述任何目录、日志或文件。
