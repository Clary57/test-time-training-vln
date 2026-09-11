# 当前研究状态

**2026-09-11新指令：** 用户授权有空闲GPU时启动A5扩大至600集。已发现GPU6/7空闲；采样配置与600集组成正在明确，尚未启动。以下研究结果及“唯一优先实验”段是2026-09-10的诊断建议快照，不能代替这条最新运行决定。旧结果、selection和封存协议不改写，新运行须单独记录配置与commit。

更新：2026-09-10。算法基准 HEAD：`3dcfc892c2919438eb587e5d33861ca3baf97758`。之后的目录整理仅变更文档、索引与维护检查，不改变该算法。整理提交完整 hash 以 `git rev-parse HEAD` 为准；历史实验的 commit 不随整理更新。

证据标签：**VERIFIED**＝源码、Git 或原始产物直接支持；**DOCUMENTED**＝文档记载、尚未重新验收；**MEETING DECISION**＝人类讨论方向；**USER-REPORTED**＝用户转述；**INFERENCE**＝分析判断；**UNKNOWN**＝缺证据。

## 已实现

**VERIFIED：** episode-local adapter/optimizer reset、XY 与 se2_stop DTW、圆周朝向距离、STOP 后虚拟轨迹结束、A5 greedy＋stochastic、A4 全 stochastic、最近 pair/outlier、cluster 与 quality gate、frozen action-token confidence、反转惩罚、action-span loss、每 episode 更新上限、disable-update、非有限参数/漂移 rollback 框架、canonical NaVIDA 子集入口。

支持 anchor L2 和 adapter drift 阈值，但 A5 及当前 B 使用 0；不能写成已启用。当前动作在更新前生成，adapt_step 在真实执行前发生；pending 原子动作没有额外生成或更新。参数不跨 episode 保留。详见 [方法](docs/method/CURRENT_METHOD.md)。

## 已验证

**VERIFIED：** 最近 T1.5 全 stochastic B 完成 128/128，四个 worker exit=0，最终验收为 `PASS_WITH_PROTOCOL_AMENDMENT`；0 optimizer step、0 有效参数变化。SR=66/128=51.5625%，OSR=78/128=60.9375%，SPL=0.471364、nDTW=0.654690、SDTW=0.456181、DTG=4.718697 m、PL=9.726449 m。

**VERIFIED：** 其 CPU 审计完整回放 3,269 次决策。半径内 512 次新决策中，59 次有立即 STOP、25 次选中、34 次漏选；另外 377 次没有任何 STOP，76 次只有 future STOP。confidence 与 consistency 候选内描述性 r≈0.259，不能写成高度冗余，也不能因此声称提供独立正确性。

**VERIFIED：** 62 个失败中，12 个曾进入成功半径，50 个从未进入。一个已记录案例在半径内选中 `[FORWARD, STOP]`，先前进后越界停止。

原始数据、验收和报告直达链接见 [结果索引](docs/experiments/RESULTS_INDEX.md)。

## 已运行但结论不充分

| 实验 | 现有结果 | 不能据此宣称 |
|---|---|---|
| 历史 full TTRL / C_project confirmatory128 | summary 分别为 26/128、27/128 | 这是 A5 结果，或与 canonical C0 可混用 |
| canonical C0 在历史128上 | 跨原运行及 continuation 共 128 条唯一 stats、72 次成功 | 完整 1839 复现已完成 |
| new128 C0 / A4 / R13 A5 | 各128条唯一 stats，成功74/72/71 | 当前 T1.5 B 的匹配反事实，或方法有确认性提升 |
| 最近 T1.5 B-only | 66/128 | optimizer 更新有效/无效，或温度单因素导致变化 |
| confidence48 | DOCUMENTED：已分析48条，数据前缀、拓扑也不同 | 独立确认性结论或与完整128等价 |

所有比例是描述性记录；新128已经用于开发。当前 B 没有同批 A，不能做 A−B 的配对因果归因。离线替代评分只改变已记录状态的排名，没有重跑导航，不能计算反事实 SR。

## 尚未实现或尚未完成

**VERIFIED / UNKNOWN：** 当前正式路径没有强制 STOP 候选、独立 progress/waypoint/stop critic、reference-policy token KL、delayed post-observation update 或逐原子动作重规划。现有日志不足以完成一般导航 oracle candidate recall@K、oracle-best upper bound 和全导航 gate precision。instruction/vision ablation 与 entropy minimization 对照未完成；平均候选 log-probability 不能替代完整 entropy。

**USER-REPORTED：** 李欣陶另一服务器的 confidence 伪专家实验出现两个 episode 退化；没有本地配置/日志可判断严格重复或统计意义。

## 当前唯一优先实验

**INFERENCE / 尚未执行：** 准备开发集上的最小 **oracle-only 候选覆盖与排序诊断**，区分“候选不覆盖合适动作”和“已有合适候选却排错”。现有 STOP 审计是已完成的特例；一般导航诊断仍需定义标签、审查状态恢复和采集条件。

重点检查50个未到达失败，但不能把这个事后子组当成独立测试集；需保留完整128的分母及成功episode对照。Oracle 只用于离线诊断，绝不进入正式 reward。此项仍处于方案准备，不是已经锁定的运行许可；本次目录整理不启动任何任务。见 [待执行计划](docs/plans/CANDIDATE_DIAGNOSTIC.md)。

**MEETING DECISION：** 保持局部伪专家、confidence＋轨迹相似性方向；朝向纳入统一距离；暂缓全局池和复杂新监督。600+ 与重复运行的讨论要求应在方法/协议冻结和数据独立性检查后落实。见 [会议摘要](docs/meetings/2026-09-04.md)。
