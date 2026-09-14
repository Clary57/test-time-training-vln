# 文档索引

路径均相对于本文件；服务器外部产物使用绝对路径。状态以实际内容和适用版本区分，不以文件名中的 independent、confirmatory 或 SUCCESS 自动认定证据等级。

| 文档 | 状态 | 用途 |
|---|---|---|
| [根 README](../README.md) | 当前有效 | 新成员总入口、结构、环境、阅读顺序 |
| [CURRENT_STATUS](../CURRENT_STATUS.md) | 当前有效，更新至2026-09-14 | 实现、验收、未知项和待验证问题 |
| [当前方法](method/CURRENT_METHOD.md) | 当前有效 | A4/A5/profile/当前B的边界 |
| [复现说明](experiments/REPRODUCTION.md) | 当前有效 | CPU检查、cwd、canonical C0与A5入口 |
| [结果索引](experiments/RESULTS_INDEX.md) | 当前有效，产物快照 | 已读结果、验收边界、不能直接比较的实验 |
| [A5 600 / C0 全量实验说明](experiments/A5_C0_20260914.md) | 当前有效 | 完整结果、同 ID 对照与统计边界 |
| [运行目录清单](experiments/RUN_DIRECTORIES.md) | 目录存在性快照 | 包含未独立验收的历史目录，不代表全部完成 |
| [脚本索引](../scripts/README.md) | 当前有效 | 启动、selection、分析、验收及维护工具 |
| [实验包说明](../experiments/README.md) | 当前有效 | 保留selection和包内路径的原因 |
| [2026-09-04会议](meetings/2026-09-04.md) | MEETING DECISION / 用户提供的纪要摘要 | 研究方向，不能当实验日志 |
| [候选诊断计划](plans/CANDIDATE_DIAGNOSTIC.md) | 尚未执行；分析建议 | 尚未锁定的诊断准备工作 |
| [目录整理记录](maintenance/ORGANIZATION_20260910.md) | 当前有效 | 移动映射、遗留风险、验证和管理约定 |
| [早期论文框架](archive/pre_a5/VLN_TTA论文框架.md) | 历史记录；当前协议说明已被替代 | 含K=8、XY与部分自相矛盾的reset/旧日志描述，不用于填写当前结果 |
| [写作问答资料](archive/pre_a5/资料.md) | 历史记录 | 与论文框架主题重叠；保留原始问答，外部引文未在本次核验 |
| [旧改造计划](<archive/pre_a5/plan before 8.15.md>) | 历史记录；部分已实现 | 含未被当前实验保证的跨拓扑一致性要求，不是现行analysis plan |
| [旧变更说明](archive/pre_a5/PROJECT_CHANGES_GUIDE.md) | 历史记录，覆盖早期基础设施 | “未修改reward”等语句仅指当时，不适用于A4/A5整个历史 |
| [旧评测README](archive/pre_a5/README_ttt_legacy.md) | 已被当前复现说明替代 | 旧服务器、K=8/history=10示例和旧清理建议仅供追溯 |

仓库三个 `experiments/official_navida_*/HANDOFF.md` 和相邻 `analysis_plan.json`、`protocol_manifest.json` 仍在原位：**对相应历史实验包有效**，不是当前 A5/B 的通用配置。[C128](../experiments/official_navida_c128/HANDOFF.md)、[C full](../experiments/official_navida_c_full/HANDOFF.md)、[历史 TTRL A128](../experiments/official_navida_ttrl_a128/HANDOFF.md)。

没有发现仓库内或适用祖先目录的 AGENTS.md。整理前根 README.md、CURRENT_STATUS.md 和 docs/ 均不存在。原先8份Markdown均已阅读；5份历史文档完整保留并迁移，其余3份HANDOFF保留原位。
