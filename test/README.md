# 历史测试启动材料

`before 8.15/` 中3个Shell和1个兼容runner原位保留，均为历史材料，不是当前CPU测试入口。

- Shell硬编码的`/nvme-ssd2/lixintao/...`工作区在本服务器不存在。
- `ori_run_test.sh`在旧仓库cwd调用相对`ori_tttrl_compat_runner.py`，与现在子目录布局不一致。
- runner依赖其相邻`ori_ttrl-src/`，该目录不存在。恢复旧环境前不能运行。
- 不移动或修补这些脚本，以免把历史协议修复混入目录整理。

当前CPU测试位于[ttrl-src/tests](../ttrl-src/tests)，命令见[根README](../README.md)；当前启动和分析入口见[脚本索引](../scripts/README.md)。
