# VLN 的测试时轨迹共识学习

2026-09-11：用户新授权A5扩大600集，当前正明确采样与selection，尚未启动；下述已完成结果和诊断建议截至2026-09-10。最新状态见[CURRENT_STATUS](CURRENT_STATUS.md)。

研究目标：以 NaVIDA 为基础策略，利用当前状态生成的多个动作候选构造局部伪专家，在不使用测试目标位置或专家动作作为 reward 的条件下，探索 episode 内 LoRA 更新能否改善导航。

先读 [当前状态](CURRENT_STATUS.md)，再读 [方法与版本边界](docs/method/CURRENT_METHOD.md)、[复现入口](docs/experiments/REPRODUCTION.md) 和 [实验结果索引](docs/experiments/RESULTS_INDEX.md)。全部文档见 [docs/README.md](docs/README.md)。

## 当前方法与进度

A5 将动作转换为 `(x,y,theta,stop)` 虚拟轨迹，用 DTW 衡量一致性，结合 episode 初始模型的候选 confidence 与相邻转向反转惩罚，形成最近 pair / outlier 门控和 action-token loss。每个 episode 恢复初始 adapter 和优化器，最多 5 次合格更新机会；pending 动作期间不重新生成。

**已实现不等于已证明有效。** 最近完成的是 A5 机制的 **B-only 开发实验**：new128，T=1.5，K=3，全 stochastic，禁止 optimizer step；SR=66/128=51.56%。它不是默认采样 A5，也没有同批 A 可用于隔离更新收益。离线 STOP 审计已完成，当前唯一优先事项是准备候选覆盖与排序的 oracle-only 离线诊断，尚未启动补充环境回放或 600+ 实验。

## 目录结构

```text
README.md / CURRENT_STATUS.md    项目入口 / 研究状态
docs/
  README.md                      文档有效性与阅读顺序
  method/                        当前方法、参数及证据边界
  experiments/                   复现说明、结果和运行目录索引
  meetings/                      人类讨论摘要，不充当实验证据
  plans/                         尚未执行的诊断方案
  archive/pre_a5/                原文保留的旧论文、计划、变更说明
  maintenance/                   整理记录、迁移映射和文件盘点
scripts/
  README.md                      按职责索引所有脚本（不复制实现）
  select_*.py                    历史 selection 生成工具，保持原路径
  maintenance/check_layout.py    CPU 静态检查
experiments/
  README.md                      锁定实验包说明
  official_navida_*/              启动脚本、selection、manifest、analysis plan
  a4_a5_independent128/           固定 new128 selection（现为开发集）
ttrl-src/
  eval/                          canonical C0、A/B evaluator 与分析工具
  trainer/                       TTRL updater 与旧训练器
  config/                        Habitat YAML；保留数据相对路径语义
  tests/                         CPU 单元测试
  model/ dataset/ data/ train/ utils/  原有模型及训练/数据工具
test/before 8.15/                 旧服务器启动材料，保留原位
```

脚本及配置没有机械搬到新目录：实验包内存在 `__file__`、固定 SHA256、相邻 selection 和测试动态导入依赖。它们按职责通过索引组织，运行路径保持不变。历史文件迁移见 [整理记录](docs/maintenance/ORGANIZATION_20260910.md)。

## 环境与数据

本服务器仓库根：`/hdd3/wangziyu/code/test-time-training-vln`。Markdown 相对链接以文档所在目录为基准；下述 CPU 命令从仓库根运行。真实 evaluator 的工作目录必须是外部 NaVIDA 根，详情见复现说明。

| 用途 | 绝对路径 |
|---|---|
| Python/Node 工具环境（本次未改动） | `/hdd3/wangziyu/envs/codex-cli` |
| 已有导航运行与 CPU 测试 Python | `/hdd3/wangziyu/envs/navida/bin/python` |
| 外部 NaVIDA / habitat_extensions | `/hdd3/wangziyu/code/Self-TTRL4VLN/NaVIDA` |
| 基础 checkpoint | `/hdd3/wangziyu/models/NaVIDA` |
| R2R 数据及 GT | `/hdd3/wangziyu/data/R2R_VLNCE_v1-3_preprocessed/val_unseen` |
| 场景目录 | `/hdd3/wangziyu/data/scene_dataset` |
| 实验产物（独立于源码） | `/hdd3/wangziyu/runs/test-time-training-vln` |

不在本次整理中安装依赖或迁移环境。源仓库没有完整的可移植环境锁；历史复现应使用相应运行包中的版本和输入哈希，不能仅凭环境名字认定版本相同。

## 基线、A5 与检查入口

- canonical C0：`experiments/official_navida_c128/launch.sh`，核心 `ttrl-src/eval/eval_official_subset.py`。单候选、history=200、前两个 chunks、无 LoRA 更新。
- A5：`ttrl-src/eval/eval_ttt.py --ttrl-profile a5`；完整命令必须包含数据、模型、分片、官方输入协议等参数，见 [复现说明](docs/experiments/REPRODUCTION.md)。
- `experiments/official_navida_ttrl_a128/launch.sh` 是历史 full TTRL，**不是 A5 启动脚本**。
- 最近 B 的精确命令、wrapper、配置和版本锁在 runs 的封存准备包中，只作为历史入口引用；整理后 HEAD 已变化，不能直接复用锁定 `3dcfc89` 的包启动新实验。

仅运行 CPU 检查：

```bash
cd /hdd3/wangziyu/code/test-time-training-vln
export PYTHONPATH=/hdd3/wangziyu/code/Self-TTRL4VLN/NaVIDA:/hdd3/wangziyu/code/test-time-training-vln/ttrl-src
PYTHONDONTWRITEBYTECODE=1 /hdd3/wangziyu/envs/navida/bin/python -B scripts/maintenance/check_layout.py
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /hdd3/wangziyu/envs/navida/bin/python -B -m unittest discover -s ttrl-src/tests -p 'test_*.py'
```

## 推荐复现顺序与比较边界

1. 阅读状态与对应 manifest；核对 code commit、selection、数据/模型、环境和 GPU/进程拓扑。
2. 先完成 CPU 静态检查和单元测试；不把历史 smoke 脚本当正式入口。
3. 如需新运行，先建立独立结果根和新协议锁，再做小规模协议验收，确认 canonical baseline 输入一致。
4. 研究实验只能按冻结的开发诊断或配对方案执行；不能把旧结果直接拼成完美反事实。600+ 确认实验留待方法冻结与独立数据审查之后。

历史 full TTRL A128、旧 C_project、canonical C0、A4、R13 A5 和当前 T1.5 B 的协议不同。new128 虽保留 `independent128` 文件夹名，现已被用于开发；confidence48 是其前缀子集，不是独立确认集。旧论文草稿中的数字也不能直接当作本服务器当前结果。详见 [结果索引](docs/experiments/RESULTS_INDEX.md)。
