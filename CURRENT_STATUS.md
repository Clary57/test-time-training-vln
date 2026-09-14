# 当前研究状态

**2026-09-14结果更新：** A5 T=1.5、K=3 全 stochastic、optimizer 启用的 600 条扩大开发集及 canonical C0 全量 1,839 条均已完成。两次运行的代码 HEAD 都是 `83a4d2f48077165b15c469e55faf465472b5054d`；该整理提交没有改变 A5 算法。完整产物、验收、统计与比较边界见[实验说明](docs/experiments/A5_C0_20260914.md)。

更新：2026-09-14。A5 算法基准 HEAD：`3dcfc892c2919438eb587e5d33861ca3baf97758`；目录整理后的运行 HEAD：`83a4d2f48077165b15c469e55faf465472b5054d`。历史实验的 commit 不随整理或结果文档提交更新。

证据标签：**VERIFIED**＝源码、Git 或原始产物直接支持；**DOCUMENTED**＝文档记载、尚未重新验收；**MEETING DECISION**＝人类讨论方向；**USER-REPORTED**＝用户转述；**INFERENCE**＝分析判断；**UNKNOWN**＝缺证据。

## 已实现

**VERIFIED：** episode-local adapter/optimizer reset、XY 与 se2_stop DTW、圆周朝向距离、STOP 后虚拟轨迹结束、A5 greedy＋stochastic、A4 全 stochastic、最近 pair/outlier、cluster 与 quality gate、frozen action-token confidence、反转惩罚、action-span loss、每 episode 更新上限、disable-update、非有限参数/漂移 rollback 框架、canonical NaVIDA 子集入口。

支持 anchor L2 和 adapter drift 阈值，但 A5 及当前 B 使用 0；不能写成已启用。当前动作在更新前生成，adapt_step 在真实执行前发生；pending 原子动作没有额外生成或更新。参数不跨 episode 保留。详见 [方法](docs/method/CURRENT_METHOD.md)。

## 已验证

**VERIFIED：** 最新 A5 600 条开发集严格验收 `PASS`，两个 worker exit=0，SR=305/600=50.83%（Wilson 95% CI 46.84%—54.82%），OSR=58.17%，SPL=45.59%，nDTW=62.64%，SDTW=43.93%。共 2,927 次 optimizer step、0 rollback；其中 562/600 个 episode 达到每 episode 5 次更新上限。selection 包含旧 new128 的全部 128 条，不能视为独立确认集。

**VERIFIED：** canonical C0 完整 val_unseen 1,839 条在 8 个官方分片上完成，8 个 exit code 均为 0；只读核对 1,839 条唯一 stats 与 selection/manifest 一一对应。SR=1,075/1,839=58.46%，OSR=67.43%，SPL=52.51%。与 A5 同 ID 的 600 条 C0 为 347/600=57.83%。C0 未记录 nDTW/SDTW。

**VERIFIED：** 先前 T1.5 全 stochastic B 完成 128/128，四个 worker exit=0，最终验收为 `PASS_WITH_PROTOCOL_AMENDMENT`；0 optimizer step、0 有效参数变化。SR=66/128=51.5625%，OSR=78/128=60.9375%，SPL=0.471364、nDTW=0.654690、SDTW=0.456181、DTG=4.718697 m、PL=9.726449 m。

**VERIFIED：** 其 CPU 审计完整回放 3,269 次决策。半径内 512 次新决策中，59 次有立即 STOP、25 次选中、34 次漏选；另外 377 次没有任何 STOP，76 次只有 future STOP。confidence 与 consistency 候选内描述性 r≈0.259，不能写成高度冗余，也不能因此声称提供独立正确性。

**VERIFIED：** 62 个失败中，12 个曾进入成功半径，50 个从未进入。一个已记录案例在半径内选中 `[FORWARD, STOP]`，先前进后越界停止。

原始数据、验收和报告直达链接见 [结果索引](docs/experiments/RESULTS_INDEX.md)。

## 已运行但结论不充分

| 实验 | 现有结果 | 不能据此宣称 |
|---|---|---|
| 历史 full TTRL / C_project confirmatory128 | summary 分别为 26/128、27/128 | 这是 A5 结果，或与 canonical C0 可混用 |
| canonical C0 在历史128上 | 跨原运行及 continuation 共 128 条唯一 stats、72 次成功 | 与新全量运行是同一批数据或同一启动拓扑 |
| new128 C0 / A4 / R13 A5 | 各128条唯一 stats，成功74/72/71 | 当前 T1.5 B 的匹配反事实，或方法有确认性提升 |
| 先前 T1.5 B-only | 66/128 | optimizer 更新有效/无效，或温度单因素导致变化 |
| A5 600 与 C0 同 ID 比较 | 305/600 对 347/600，A5−C0 为 −7.00 个百分点 | 单独认定 reward/optimizer 是退化原因，或把开发集差值当独立确认性结论 |
| confidence48 | DOCUMENTED：已分析48条，数据前缀、拓扑也不同 | 独立确认性结论或与完整128等价 |

同 600 条 ID 的事后配对探索性分析：仅 A5 成功 71、仅 C0 成功 113，exact McNemar `p=0.002414`；10,000 次配对 bootstrap 的经验 95% CI 为 [−11.33，−2.67] 个百分点。C0 与 A5 的候选数、温度、选择、更新和拓扑均不同；这个统计差距不能隔离单一机制。旧 new128 已用于开发，当前 B 只跑了 128 条，没有同 600 条的 A−B 配对反事实。离线替代评分只改变已记录状态的排名，没有重跑导航，不能计算反事实 SR。

## 尚未实现或尚未完成

**VERIFIED / UNKNOWN：** 当前正式路径没有强制 STOP 候选、独立 progress/waypoint/stop critic、reference-policy token KL、delayed post-observation update 或逐原子动作重规划。现有日志不足以完成一般导航 oracle candidate recall@K、oracle-best upper bound 和全导航 gate precision。instruction/vision ablation 与 entropy minimization 对照未完成；平均候选 log-probability 不能替代完整 entropy。

**USER-REPORTED：** 李欣陶另一服务器的 confidence 伪专家实验出现两个 episode 退化；没有本地配置/日志可判断严格重复或统计意义。

## 当前待验证问题

**INFERENCE / 尚未执行：** 若要定位 A5 与 C0 的差距中有多少来自在线更新，应在相同 600 个 ID、相同 T=1.5 全 stochastic A5 协议上运行 `--disable-update` 对照，并锁定其余参数、seed 和 worker 过程证据。已有 128 条 B 不能与新 600 条 A 构成完整配对。另一个方法诊断方向是开发集上的最小 **oracle-only 候选覆盖与排序诊断**，区分“候选不覆盖合适动作”和“已有合适候选却排错”；现有 STOP 审计是已完成的特例。

候选诊断若重点检查此前 50 个未到达失败，也不能把这个事后子组当成独立测试集；需保留完整128的分母及成功episode对照。Oracle 只用于离线诊断，绝不进入正式 reward。此项仍处于方案准备，不是已经锁定的运行许可。见 [待执行计划](docs/plans/CANDIDATE_DIAGNOSTIC.md)。

**MEETING DECISION：** 保持局部伪专家、confidence＋轨迹相似性方向；朝向纳入统一距离；暂缓全局池和复杂新监督。会议提出扩大到600+与重复运行；当前600条已作为开发性扩大实验执行，但独立确认与重复运行尚未完成。见 [会议摘要](docs/meetings/2026-09-04.md)。
