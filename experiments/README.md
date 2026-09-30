# 锁定实验包

一个历史实验包将selection、manifest、analysis plan、launch和validator放在一起。包内使用固定hash、相邻文件和`__file__`路径，故不拆分到顶层configs/scripts。

| 目录 | 状态 / 适用协议 |
|---|---|
| [official_navida_c128](official_navida_c128/HANDOFF.md) | 历史128 canonical frozen准备包；结果分布在原运行与continuation |
| [official_navida_c_full](official_navida_c_full/HANDOFF.md) | 全1839准备包；本次未发现对应完整运行根，不据此认定已跑完 |
| [official_navida_ttrl_a128](official_navida_ttrl_a128/HANDOFF.md) | 官方输入协议的历史full TTRL，不能当A5；至少一条运行只有48条stats |
| [a4_a5_independent128/selection128.json](a4_a5_independent128/selection128.json) | new128固定selection，selection seed43，现已用于开发；名称保留 |

C128和历史TTRL A128的selection内容相同是有意配对，不去重、不移动、不修改。new128与历史128的selection不同。结果与验收在[结果索引](../docs/experiments/RESULTS_INDEX.md)，不存入或搬入本目录。

以后新实验应使用新的包和结果目录，记录父代码commit、完整展开参数、环境/资产hash、ID顺序、GPU UUID和worker拓扑、重试规则与analysis plan。不要反向修改旧manifest适配新代码。
