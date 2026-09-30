# 脚本职责索引

实现保持原路径，通过本索引分类。相对链接以本文件为基准；Python/Shell命令使用绝对路径或明确cwd，见[复现说明](../docs/experiments/REPRODUCTION.md)。

| 类别 | 入口 | 使用范围与路径约束 |
|---|---|---|
| canonical baseline启动 | [C128 launch](../experiments/official_navida_c128/launch.sh) | 锁定selection/hash，绝对外部NaVIDA cwd，8进程串行 |
| canonical full启动 | [C full launch](../experiments/official_navida_c_full/launch.sh) | 1839配置准备包；目录存在不代表已运行 |
| 历史TTRL启动 | [A128 launch](../experiments/official_navida_ttrl_a128/launch.sh) | 历史full TTRL，不是named A5 |
| A5入口 | [eval_ttt.py](../ttrl-src/eval/eval_ttt.py) | named profile + 完整实验参数；当前B封存入口见结果索引 |
| selection生成 | [代表性选择](select_representative_r2r_episodes.py)、[扩展选择](select_balanced_r2r_extension.py) | 现有选择文件不可覆盖；脚本相互导入、CPU测试按原路径动态加载 |
| 严格结果聚合 | [aggregate_ttt_results.py](../ttrl-src/eval/aggregate_ttt_results.py) | 检查ID和manifest，包内import依赖，保持原位 |
| 历史结果分析 | [analyze_results.py](../ttrl-src/eval/analyze_results.py)、[IDM分析](../ttrl-src/eval/analyze_results_idm.py) | 硬编码历史路径/指标，不能替代当前严格验收，原位保留 |
| 准备包验收 | [C128](../experiments/official_navida_c128/validate_preparation.py)、[C full](../experiments/official_navida_c_full/validate_preparation.py)、[A128](../experiments/official_navida_ttrl_a128/validate_preparation.py) | CPU-only；依赖`__file__`相邻selection/launch |
| 目录与语法维护 | [check_layout.py](maintenance/check_layout.py) | 只读CPU检查新文档本地链接、Python语法、Shell语法和配置解析；不import evaluator |
| CPU单元测试 | [tests](../ttrl-src/tests) | 需navida环境和外部habitat_extensions的PYTHONPATH；不运行场景 |
| 历史smoke启动 | [旧test目录](../test/README.md) | 旧服务器路径及缺失archive，不能直接运行 |

runs中的已封存分析脚本属于对应实验证据，不迁入源码或修改。新工具若以后泛化为复用脚本，再在独立改动中整理到明确职责子目录；本次不复制旧代码制造第二份实现。
