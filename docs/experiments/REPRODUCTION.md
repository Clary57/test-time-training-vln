# 运行与复现入口

本页区分CPU检查、历史命令索引和未来新运行模板。目录整理没有启动GPU、模型或真实episode。

## 路径基准

```bash
PROJECT=/hdd3/wangziyu/code/test-time-training-vln
NAVIDA_ROOT=/hdd3/wangziyu/code/Self-TTRL4VLN/NaVIDA
PYTHON=/hdd3/wangziyu/envs/navida/bin/python
export PYTHONPATH="$NAVIDA_ROOT:$PROJECT/ttrl-src"
export PYTHONDONTWRITEBYTECODE=1
```

工具环境`/hdd3/wangziyu/envs/codex-cli`不是已验证的导航依赖环境，不在其中安装、更改或整理文件。CPU单元测试也用已有navida Python。使用`-B`避免更改已跟踪的旧pyc。

YAML中的`data/...`相对路径以`$NAVIDA_ROOT`为基准：其数据软链接指向`/hdd3/wangziyu/data`。真实evaluator必须先`cd "$NAVIDA_ROOT"`；只把配置文件改成绝对路径不够。配置保留在原位置，没有新建一份默认配置。

## CPU检查

设置上述变量后，从任意cwd可运行：

```bash
"$PYTHON" -B "$PROJECT/scripts/maintenance/check_layout.py"
"$PYTHON" -B "$PROJECT/experiments/official_navida_c128/validate_preparation.py"
"$PYTHON" -B "$PROJECT/experiments/official_navida_c_full/validate_preparation.py"
"$PYTHON" -B "$PROJECT/experiments/official_navida_ttrl_a128/validate_preparation.py"
CUDA_VISIBLE_DEVICES='' "$PYTHON" -B -m unittest discover -s "$PROJECT/ttrl-src/tests" -p 'test_*.py'
```

不设置PYTHONPATH会使frozen evaluator测试找不到`habitat_extensions`。这些测试使用CPU假模型和解析检查，不构造真实Habitat场景。维护检查只做语法/链接/配置解析，不能替代完整实验验收。

## Canonical C0与历史TTRL

这些是**未来运行的命令接口说明，不是本次要执行的命令**。RESULT_ROOT必须是新绝对路径，GPU须由当时资源审查明确指定。

```text
bash /hdd3/wangziyu/code/test-time-training-vln/experiments/official_navida_c128/launch.sh RESULT_ROOT GPU_ID
bash /hdd3/wangziyu/code/test-time-training-vln/experiments/official_navida_c_full/launch.sh RESULT_ROOT GPU_IDS_CSV
bash /hdd3/wangziyu/code/test-time-training-vln/experiments/official_navida_ttrl_a128/launch.sh RESULT_ROOT GPU_IDS_CSV
```

三个脚本内部已设置外部NaVIDA cwd和PYTHONPATH。前两个是canonical frozen，第三个是历史full TTRL，不是A5。保留8个分片/进程边界，不默认用DDP。配置哈希和ID验证失败时不要改旧锁强行启动。

## A5方法入口

R13 A5的真实worker命令见[原command.txt](/hdd3/wangziyu/runs/test-time-training-vln/ttrl_a5_independent128_seed43_multigpu_20260902_141135/worker_0/command.txt)。其代码commit为`55509a037c83579ceacf6cc72d0890f30017b9e5`，K3、T0.3、greedy＋2 stochastic、四个GPU worker；不是当前T1.5 B的对应A。

下面是同类A5的**单worker参数模板**，不是已锁定复现实验包。先设置GPU_ID、全局SPLIT_NUM、该worker的SPLIT_ID和唯一WORKER_RESULT绝对路径，再审查完整运行计划。全体worker共享selection/seed/起始adapter，每个worker用独立结果目录。模板保留标准A5采样组成，没有替当前B改参数。

```bash
: "${GPU_ID:?set physical GPU index}"
: "${SPLIT_NUM:?set global shard count}"
: "${SPLIT_ID:?set this shard index}"
: "${WORKER_RESULT:?set a new absolute worker result path}"
cd "$NAVIDA_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" -B "$PROJECT/ttrl-src/eval/eval_ttt.py" \
  --exp-config "$NAVIDA_ROOT/config/vln_r2r.yaml" \
  --model-path /hdd3/wangziyu/models/NaVIDA \
  --episode-ids-file "$PROJECT/experiments/a4_a5_independent128/selection128.json" \
  --result-path "$WORKER_RESULT" --split-num "$SPLIT_NUM" --split-id "$SPLIT_ID" \
  --official-navida-protocol --max-action-history 200 \
  --forward-distance 25 --turn-angle 15 --resolution-ratio 0.5 \
  --num-generations 3 --temperature 0.3 --seed 41 \
  --learning-rate 2e-6 --weight-decay 0 --max-grad-norm 1 \
  --early-stop-steps 400 --early-stop-rotation 25 --stop-trace --ttrl-profile a5
```

如要隔离更新，B应在完全相同的冻结方案上加`--disable-update`；不能同时改温度、GPU拓扑或候选组成。用户此前将实验改为T1.5 B-only，因此不能声称当前已存在这样的匹配A/B。

## 已完成的T1.5 B：只读封存入口

准备包：[/preparations/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909](/hdd3/wangziyu/runs/test-time-training-vln/preparations/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909)。原`launch_B.sh`调用相邻绝对路径的`launch_B.py`；有wrapper、eval_config.yaml、expanded_args.json、analysis_plan和输入锁。

准确运行参数看[实际展开配置](/hdd3/wangziyu/runs/test-time-training-vln/a5_t15_stochastic_B_new128_gpu4_3dcfc89_20260909_batch01/preparation/expanded_args.json)，而非复制named A5模板。该B为custom、K3全stochastic、T1.5、seed41、4个modulo shard在物理GPU4串行，显式保留A5其余机制；带nDTW配置和额外观测/验收wrapper。正式结果在[结果索引](RESULTS_INDEX.md)。

**整理后的代码commit不同于封存包锁定的3dcfc89。** 即使算法文件相同，原launcher可能因HEAD、文件路径或完整输入哈希锁拒绝运行，属于正常保护。原封存包和原结果不更新、不覆盖、不直接再执行；复现旧实验需使用其原commit的独立checkout及原环境证据，新实验则新建准备包并记录新commit。旧离线审计的verify_artifacts.py也锁定旧HEAD，不能在当前整理分支上通过“修改旧校验器”解决。

## 结果验收和失败处理

从逐episode统计重新聚合；每臂ID恰好一次、各worker exit=0，命令/manifest/版本齐全，adapter校验与更新日志可追踪。报告SR及配对差值时必须有真实匹配两臂；单臂不编造McNemar或paired bootstrap。

已有`aggregate_ttt_results.py`接受`--result-dirs`及`--expected-episode-ids-file`等参数，先阅读其解析器。它不是所有外部封存wrapper的统一验收替代品。不得把不同协议或不同selection的run通过通配符混合聚合。

使用新结果目录保存新协议；保留失败现场。重试/续跑必须事前定义，并在新amendment中记录适用worker和原结果保护规则，不能回写原成功状态掩盖中断。
