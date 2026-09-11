# 2026-09-10目录整理记录

整理基准：`3dcfc892c2919438eb587e5d33861ca3baf97758`，原分支`codex/cluster-quality-gated-ttrl`，相对本地upstream领先1提交。整理前工作区干净，无来源不明的用户修改。没有fetch/pull/push或修改远端分支。

## 整理前问题与盘点范围

- 根README、CURRENT_STATUS和docs均不存在；论文草稿、问答与早期变更指南承担了不一致的“当前说明”。
- 86个文件全部已跟踪，总计1,929,557字节；按字节SHA256无完全重复的Markdown，但论文框架和问答主题重叠。
- 两个historical128 selection完全相同，这是配对输入，完整保留，不去重。
- 6个pyc已被Git跟踪，时间早于最新源码；不删除、不移动、不重生成。不把ignore误称为已清除追踪。
- 旧test启动脚本包含不存在的`/nvme-ssd2/lixintao`路径；compat runner所需相邻`ori_ttrl-src`缺失。用途属于历史基准，原位保留。
- 已读全部8份原有Markdown、3组experiment analysis plan/manifest及validators。未发现适用AGENTS.md。
- 文件大小、mtime、Git状态和SHA256详见[inventory_before.tsv](inventory_before.tsv)。runs只读检查根目录和最多4层的小型证据路径，跳过模型/权重/图片/视频/缓存；不声称遍历验证整个资产树。

## 移动清单

以下均用`git mv`，历史正文逐字保留；5份文件内容哈希应与基准一致。

| 原路径（相对仓库根） | 新路径（相对仓库根） |
|---|---|
| PROJECT_CHANGES_GUIDE.md | docs/archive/pre_a5/PROJECT_CHANGES_GUIDE.md |
| VLN_TTA论文框架.md | docs/archive/pre_a5/VLN_TTA论文框架.md |
| 资料.md | docs/archive/pre_a5/资料.md |
| plans/plan before 8.15.md | docs/archive/pre_a5/plan before 8.15.md |
| ttrl-src/eval/README_ttt.md | docs/archive/pre_a5/README_ttt_legacy.md |

机器可读映射见[moves.json](moves.json)。原eval README路径补充新的当前导航页；其完整旧正文在archive，不被删除或替换成不可追溯内容。旧文本中的历史路径/旧政策不作为当前命令使用，当前文档索引明确标记适用范围。

## 新增与修改

- 根README和CURRENT_STATUS：项目、环境、数据、版本、已验证结果、未知项、唯一优先诊断。
- docs/README：当前/历史/已替代材料统一索引。
- docs/method：当前A4/A5/最近B定义及执行流程。
- docs/experiments：复现入口、精确结果路径、全根目录存在性索引。
- docs/meetings：用户提供的会议方向摘要，单独标记非实验证据。
- docs/plans：当前未执行的候选诊断准备，不冒称锁定analysis plan。
- scripts/README、experiments/README、test/README、eval/README：为保留原位的代码、协议与历史脚本提供清晰入口。
- scripts/maintenance/check_layout.py：新增CPU只读静态检查；不import导航代码。
- docs/maintenance：本记录、基准盘点、迁移映射、验证证据。
- .gitignore：仅补Python生成缓存规则，保留原规则；不忽略stats/selection/manifest/log等重要实验记录。

## 保持原位及跳过事项

| 材料 | 原因 |
|---|---|
| ttrl-src核心包、旧trainer、model/data/train与notebook/fig | 包导入、研究历史与资源依赖；不重构或判断无用后移动 |
| experiments包内所有sh/py/json | validator使用__file__和相邻selection，launcher含绝对路径和固定hash |
| scripts/select_*.py | 脚本间import、测试按固定路径加载；改变位置涉及代码调整 |
| eval/aggregate及analyze脚本 | 包依赖或旧硬编码路径；通过脚本索引分类，不复制第二份实现 |
| test/before 8.15全部脚本 | 旧服务器与缺失archive，恢复属于单独复现任务 |
| 6个已跟踪pyc | 遵守不删除；本次使用-B避免改写 |
| datasets、checkpoint、selection、原始日志与正式结果 | 禁止移动/重命名/修改 |
| /hdd3/wangziyu/envs/codex-cli | 未整理、未安装、未修改 |
| runs中的封存准备和分析脚本 | 属于实验原始证据，仅读取和建立链接 |

## 路径与协议风险

没有移动任何Python、Shell、配置或selection，故未引入执行路径迁移。旧文档路径仅在archive和迁移记录中作为历史引用保留；新入口使用有效相对链接或服务器绝对路径。既有历史代码的失效路径没有为了“全绿”被悄悄修补。

Git commit变化会使旧封存包的HEAD锁、输入路径锁及旧审计校验器不再适用于当前分支，即使算法内容相同。这是已知边界，不更新旧锁来绕过。旧实验复现用原commit的独立checkout，新运行创建新的准备包和结果根。源代码不再假定与旧GitHub/main或旧准备包一致。

## 验证记录

CPU单元测试首次缺少外部NaVIDA的PYTHONPATH，84项加载中出现1个导入错误；按真实环境路径补全命令后，92个测试全部通过，无跳过。没有修改测试或算法来掩盖错误。Gym打印既有维护提示，未触发依赖升级。

静态检查已通过：47个Python文件AST解析、6个Shell的bash -n、11个JSON（随后新增的验证JSON也纳入检查）和1个YAML解析；当前文档本地链接通过。3个既有实验准备包validator均通过。盘点TSV初次使用CRLF被git diff --check识别为尾空白，已仅将新增TSV改为LF；未格式化旧文件。

保护复核：除.gitignore外，原86文件中的85份经迁移映射核对均与基准字节一致；5334个范围受控的runs文件大小/mtime未变，391个当前B审计输入哈希一致。包括核心源码、Shell、配置、selection、notebook、图和pyc在内的原文件均未改变内容。[结构化验证](validation_20260910.json)记录检查边界，未递归读取所有权重。测试命令详见[复现说明](../experiments/REPRODUCTION.md)。

2026-09-11续接时用户新授权有空闲GPU就启动A5 600集；这是目录整理之外的新运行工作。本整理提交只包含目录、文档和CPU维护工具，不混入新实验启动代码或selection。根状态页标明新指令，旧实验证据不改写。

## 后续管理约定

每个新实验使用唯一名称与独立结果根，准备包固定commit、参数、selection、环境/模型/数据hash、物理GPU与worker拓扑、分析方案和重试规则。保留原始产物，后续分析用独立子目录与脚本版本；失败/续跑用显式amendment。文档状态索引与CURRENT_STATUS同步更新，勿用文件夹名字暗示确认性。

新的通用工具可按职责放scripts子目录；已封存实验脚本不为统一目录而迁移。不要在目录整理提交中混入算法、超参数、selection或格式化变更。
