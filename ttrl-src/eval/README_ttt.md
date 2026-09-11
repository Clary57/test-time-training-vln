# Evaluator 入口

当前入口见 [仓库 README](../../README.md)、[当前状态](../../CURRENT_STATUS.md) 和 [复现说明](../../docs/experiments/REPRODUCTION.md)。

`eval_official_subset.py` 是 canonical frozen C0；`eval_ttt.py` 是 episode-local A/B；`eval.py` 保留历史 C_project 行为，不能默认当 canonical C0。严格聚合工具为 `aggregate_ttt_results.py`。

每个 evaluator 是普通单进程，不使用 torchrun/DDP；多卡运行互斥 episode 分片。配置中的 `data/...` 以外部 NaVIDA 工作目录为基准，不能从任意目录直接启动。

早期 K=8 / history=10 和旧服务器 smoke 示例已原文保存在 [历史 README](../../docs/archive/pre_a5/README_ttt_legacy.md)，不作为当前运行建议。改变协议时使用新的结果目录，保留已有结果与原始日志。
