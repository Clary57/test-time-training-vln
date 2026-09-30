# 当前方法与配置边界

证据基准：算法提交 `3dcfc892c2919438eb587e5d33861ca3baf97758`。本次整理不变更任何 Python 算法、Shell 启动配置、selection 或 YAML/JSON 实验配置。

通俗解释：在同一个观察下提出几个短动作方案，看哪些运动轨迹相近，再结合基础模型对动作的置信度选局部示范。A 可以据此更新当前 episode 的 adapter；B 做同样计算但不 optimizer step。相近与自信不保证导航方向正确。

## 版本定义

| 字段 | named A4 | named A5 | 最近B实验 |
|---|---|---|---|
| CLI profile | a4 | a5 | custom，显式展开A5机制 |
| update_policy | cluster_gated | cluster_gated | cluster_gated |
| action_span_only | true | true | true |
| max_updates_per_episode | 5 | 5 | 5次合格机会，step=0 |
| reward_geometry | xy | se2_stop | se2_stop |
| candidate_sampling | all_stochastic | greedy_plus_stochastic | all_stochastic |
| cluster_quality_weight | 0.25 | 0.25 | 0.25 |
| max_cluster_ratio / min_quality_margin | 0.5 / 0.0 | 0.5 / 0.0 | 0.5 / 0.0 |
| 实际历史 K / temperature | 3 / 0.3 | R13：3 / 0.3 | 3 / 1.5 |
| optimizer | 启用 | 启用 | disable-update |

代码位置：[TTRL_PROFILES](../../ttrl-src/eval/eval_ttt.py)。named profile 会覆盖同名字段；temperature、K不由该字典定义，必须同时查看CLI与manifest。A5 profile不会因用户后来运行T1.5 B而自动变成全stochastic。anchor_l2_weight 与 max_adapter_delta_norm 的CLI默认仍为0。

## 实际执行顺序

1. episode开始，恢复同一初始adapter、重建optimizer、清空视觉历史与pending queue。实际随机配置与顺序还受manifest、分片及CUDA栈影响，不能宣称换拓扑后逐位一致。
2. 没有pending动作时构建视觉/语言输入，生成K个候选。官方输入路径使用sequential generation；named A5为第一个greedy、后两个stochastic，当前B为三个stochastic。
3. 解析前两个动作chunks，展开为原子动作；局部虚拟轨迹不执行真实分支、不建模碰撞。se2_stop包含位置、圆周朝向差和STOP状态，遇STOP结束虚拟轨迹。
4. 由两两DTW距离得到consensus reward；临时使用episode初始adapter计算候选action-token平均log-probability，组内min-max归一化，减相邻LEFT/RIGHT反转惩罚并clip形成quality。
5. 三个有效候选中选距离最近pair及outlier，检查分离、ratio、quality margin、action span和episode更新上限。即使gate拒绝更新，只要返回pair，动作仍在pair内按`0.75*reward+0.25*quality`选择。
6. 完成loss forward、backward和裁剪。合格A执行optimizer，并可触发非有限/漂移rollback；B只跳过step，保留计算。合格机会计数受上限控制。
7. 当前选中的动作来自更新前候选，随后入pending queue；先更新再真实执行。pending期间逐个返回原子动作，不重新生成/打分/更新。下一次生成才受A此前更新影响。

证据：[updater](../../ttrl-src/trainer/trajectory_consensus_updater.py)、[evaluator](../../ttrl-src/eval/eval_ttt.py)、[STOP trace](../../ttrl-src/eval/stop_trace.py)。

## 不同基线

- historical full TTRL：旧XY consensus/medoid与更新协议，不可把它的A128写成A5。
- B_disable_update：保持对应A机制、前反向和候选选择，跳过optimizer；不是单候选 frozen NaVIDA。
- C_project：`eval.py`保留默认前3chunks等历史行为；不能冒充canonical C0。
- canonical C0：`eval_official_subset.py`沿官方输入/解析谱系，前2chunks、history200、单候选、无adaptation。

正式reward没有独立progress/waypoint/stop critic，也没有测试oracle目标输入；未实现强制STOP、token-level reference KL、逐原子动作重规划、post-observation delayed update。完整entropy和输入ablation不在本轮审计范围。方法有效性仍需候选覆盖/排序与更新效应证据，不能由机制存在性代替。
