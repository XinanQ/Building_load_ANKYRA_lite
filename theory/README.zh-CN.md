# 理论：ankyra-lite 背后的精确性质

[English](README.md) · [证明](PROOFS.md) · [算子](operators.py) · [数值检验](test_operators.py) · [载体检验](test_carrier_properties.py) · [加权检验](test_weighting_properties.py) · [接续层检验](test_continuation_properties.py) · [返回预测器](../README.zh-CN.md)

本文件夹收录放在预测器旁边的数学性质。预测器是 [`ankyra_lite/`](../ankyra_lite/) 这个包，包里没有任何代码引用本文件夹。这些性质让预测器可以逐步检查；它们不是精度的证据，精度是测出来的（[results/SUMMARY.md](../results/SUMMARY.md)）。性质分三组，对应 ankyra-lite 的三个部分：

- **P1–P20，锚定层。** `ankyra_lite.anchor` 是 ANKYRA 2.2.1 包的逐字节副本，所以这些就是 ANKYRA 2.2.1 理论部分的性质、算子和检验，只改了导入路径。它们解释并限定锚定层：
  - 为什么把月前预测拆成水平、日路径和日内形状；
  - 权重需要多少历史，最多能离开等权多远；
  - 电量读出和峰值读出能保证什么、不能保证什么；
  - 评估为什么并列报告几种估计目标。
- **C1–C14，载体。** ankyra-lite 新增。C1–C10：在每条性质各自的运算与输入假设下，载体网络的构造对任意权重（训练过或没训练过）保证什么：
  - 它的预测是一份非负的电量预算，均值等于水平；
  - 它只用起报时刻之前的数据，没有单位标识输入，并随负荷同比例缩放；
  - 它的复制指针从零开始，而且从不改变水平；
  - 训练截止点与测试集之间留出 186 天，锚定层评分的每一个伪起报点预测对载体都是样本外的。

  C11–C14（0.2.0 新增）：对满足其支撑与正下限条件的训练池（任意训练池不一定满足），配方 Rw 第二阶段的逆噪声损失加权保证什么：
  - 权重全部相等时，它与配方 R 逐位相同；
  - 每个人群在抽样中所占的份额不变；
  - 每个权重都有限、为正，并受下限约束；
  - 权重只从该窗口自己起报前最后 28 天的上下文读出。
- **O1–O9，起报接续层。** 0.2.0 新增：在每条性质的假设下，放在锚定层之后的起报接续层对锚定层交付的、被接受的轨迹保证什么（浮点下的有限性与严格为正另需可表示性条件；任意自定义常数不在保证范围内）：
  - 它把每个小时乘以一个有明确界的正因子，所以符号和精确的零都保持不变；
  - 它只读起报前 672 小时和日历中的八个小时；
  - 状态窗口不受影响；关闭时不改变该载体对应的预测流程；复现 0.1 产品还须用原配方 R 的种子 0–2 检查点、相同输入和原流程；
  - 电量读数（`energy_kwh`）、水平和峰值在它之前读出。交付曲线之和一般与 电量读数（`energy_kwh`）不同，差额由接续前曲线与水平读数的原有差额及接续层变化组成，也可能为零。

每条性质有三种支撑：

| 文件 | 内容 |
|---|---|
| [PROOFS.md](PROOFS.md) | 陈述、证明，以及标出适用边界的反例；本仓库结果文件在数据上测过后果的，注明该文件 |
| [operators.py](operators.py) | 把 P1–P20 写成函数，适用于任何预测器的 744 小时轨迹 |
| [test_operators.py](test_operators.py) | P1–P20 的数值检验，含反例（31 项） |
| [test_carrier_properties.py](test_carrier_properties.py) | C1–C10 在随机与扰动后的载体权重上的数值检验，含反例（27 项） |
| [test_weighting_properties.py](test_weighting_properties.py) | C11–C14 在合成训练池和未训练载体上的数值检验，含反例（15 项） |
| [test_continuation_properties.py](test_continuation_properties.py) | O1–O9 在合成窗口上、以及在一栋人造建筑上端到端的数值检验，含反例（36 项） |

```bash
python -m unittest discover -s theory -t .
```

这条命令运行四个检验文件（共 111 项；只用 CPU，单线程约 35 秒），不需要检查点，也不需要研究数据。请在仓库根目录下运行。

![一个测试窗口上的精确性质](../figures/fig7_operators.png)

*图 7。图 a 在图 6 的剑桥测试窗口上，把载体预测的水平、日路径和日内块逐块换成 ankyra-lite 的：每根条都精确地分成三块的损失（P1、P4），三块全部换完就是 ankyra-lite 的预测。图 b–d 说明权重需要多少历史、收缩让权重最多移动多远（P7–P9），以及为什么电量在截断之前读出（P11、P12）；图 e 和 f 是峰值读出及其误差分解（P15、P16）。由 [`figures/make_figures.py`](../figures/README.md) 绘制。*

## 锚定层的二十条性质

标为“标准”的性质是设计所依赖的已知结论；它们收录在这里，但不作为新贡献。其余性质在第三列的条件下成立。没有一条是精度保证。

| | 性质 | 成立条件 | 锚定层用它做什么 | 代码 | 检验 |
|---|---|---|---|---|---|
| P1 | [块恒等式](PROOFS.md#blocks)（标准） | 总是 | 把每一块交给一个来源 | `anchor.blocks.block_losses` | `BlockGeometryTests` |
| P2 | [四块；没有周频的傅里叶分量](PROOFS.md#blocks) | 总是（744 小时窗口） | 用日类型代替周分量 | `operators.four_block_losses` | `BlockGeometryTests` |
| P3 | [电量就是水平](PROOFS.md#blocks) | 总是 | 电量读出 | `anchor.blocks.level` | `BlockGeometryTests` |
| P4 | [精确替换与水平份额界](PROOFS.md#replacement) | 替换用的估计是固定的 | 由块损失给任意分工打分 | `operators.replace_blocks` | `ReplacementTests` |
| P5 | [两来源分工：合并最优与单位层面的不对称](PROOFS.md#replacement) | 两个来源、按块 | 固定分工为什么不够 | `operators.division_log_ratio` | `ReplacementTests` |
| P6 | [修正量核算](PROOFS.md#replacement)（标准） | 总是 | 诊断一项修正 | `operators.correction_accounting` | `ReplacementTests` |
| P7 | [已完成误差的支撑](PROOFS.md#historical-weights) | 记录完整；缺失的小时只会减少个数 | 权重需要多少历史 | `operators.completed_windows` | `SupportTests` |
| P8 | [收缩界](PROOFS.md#historical-weights) | 支撑一致，起报时没有候选缺失 | 权重最多能移动多远 | `operators.mixture_mass_bounds` | `ShrinkageTests` |
| P9 | [候选缺失时混合重新归一](PROOFS.md#historical-weights) | 总是；一个反例标出 P8 在哪里失效 | 界何时不再成立 | `operators.retained_data_share` | `ShrinkageTests` |
| P10 | [共享误差下的最小二乘权与逆 MSE 权](PROOFS.md#historical-weights) | 各预测含共同的误差成分 | 分周交接的权重 | `operators.two_source_weights` | `ShrinkageTests` |
| P11 | [截断到零不会增大任何一个小时的误差](PROOFS.md#nonnegativity-and-energy)（标准） | 负荷非负 | 交付的轨迹 | `operators.clip_nonnegative` | `FeasibilityTests` |
| P12 | [电量、非负与不增误差三者不可兼得](PROOFS.md#nonnegativity-and-energy) | 总是（两小时的反例） | 电量从水平读出 | `operators.project_to_mean` | `FeasibilityTests` |
| P13 | [日层面的投影：有条件的保证](PROOFS.md#nonnegativity-and-energy) | 水平与负荷非负；水平的低估不超过最小的真实日均值 | 一个可选的投影 | `operators.daily_projection` | `FeasibilityTests` |
| P14 | [均值路径低估期望峰值](PROOFS.md#peak)（标准） | 精确的条件均值 | 单独的峰值读出 | — | `PeakTests` |
| P15 | [峰值算子的标量形式与界](PROOFS.md#peak) | 总是 | 峰值读出 | `anchor.readouts.peak_readout` | [tests/anchor/](../tests/anchor/test_blocks_and_readouts.py) 里的 `ReadoutTests` |
| P16 | [峰值误差分解](PROOFS.md#peak) | 总是；幅度项没有固定的符号 | 水平的收益怎样传到峰值 | `operators.peak_error_decomposition` | `PeakTests` |
| P17 | [有限支撑中位数性质](PROOFS.md#peak) | 同类型日内水平不变、独立抽取、$((m-1)/m)^n<1/2$ | 包络估计的是什么 | `operators.max_lower_median` | `PeakTests` |
| P18 | [非对称的峰值成本](PROOFS.md#peak)（标准） | 总是 | 在给定成本下选择峰值方法 | `operators.asymmetric_peak_cost` | `PeakTests` |
| P19 | [合并汇总与单位等权汇总](PROOFS.md#estimands) | 总是（恒等式） | 为什么并列报告几种估计目标 | `anchor.metrics.pooled_ratio_decomposition` | `EstimandTests` |
| P20 | [候选选择的证据界](PROOFS.md#evidence-bound)（标准结果；零基准为恒等式） | 每个候选有 T 个伪起报点误差 | 为强收缩水平权重提供统计背景（最坏情形的速率，不是在这批数据上的测量） | `operators.aggregation_rate`、`operators.exchangeable_hit_rate` | `EvidenceBoundTests` |

`operators.*` 指本文件夹里的 [operators.py](operators.py)；`anchor.*` 是锚定层 `ankyra_lite.anchor`，P1、P3、P15 和 P19 由它自身的函数承载。检验名是 [test_operators.py](test_operators.py) 里的测试类，注明了文件夹的除外。

## 载体的十四条性质

十四条都由 `ankyra_lite.carrier` 的代码推出。C1–C10 在每条性质各自的运算与输入假设下对任意权重（训练过或没训练过）成立；只有 C9(a) 讲的是初始化。证明里注明了代码行号。关于实数运算的陈述用 float64 检验，关于交付的载体预测的陈述用包自身的 float32 路径检验。C11–C14（0.2.0 新增）在其支撑与正下限条件下成立，任意训练池不一定满足这些条件；它们讲的是配方 Rw 第二阶段的损失加权（`train.noise_weights`、`train.weighted_log1p_loss`），证明里注明了函数名。

| | 性质 | 成立条件 | 对 ankyra-lite 的意义 | 代码 | 检验 |
|---|---|---|---|---|---|
| C1 | [轨迹非负](PROOFS.md#exact-properties-of-the-carrier-network) | 任意权重；只在实数运算中严格为正（份额可能下溢为 0） | 交给锚定层的载体候选非负；非有限的预测在锚定前被拒绝 | `model.decode_budget`、`inference.carrier_forecasts` | `NonnegativityTests` |
| C2 | [电量预算：轨迹均值等于水平](PROOFS.md#exact-properties-of-the-carrier-network) | 总是（实数运算中精确） | 载体的水平、日路径和日内形状就是其轨迹的 P1 分块 | `model.decode_budget` | `EnergyBudgetTests` |
| C3 | [日份额、小时份额和指针注意力都是概率分布](PROOFS.md#exact-properties-of-the-carrier-network) | 总是 | 复制来的日内形状是该单位自己若干完整日的混合 | `model.CopyPointerNetwork` | `ShareTests` |
| C4 | [不读起报时刻及之后的任何输入](PROOFS.md#exact-properties-of-the-carrier-network) | 总是；预测月的日类型会被读取 | 每个伪起报点预测都是从它自己的起报时刻出发的真实预测 | `features.features`、`inference.carrier_forecasts` | `CausalityTests` |
| C5 | [编码器按词元顺序因果](PROOFS.md#exact-properties-of-the-carrier-network) | 总是 | 指针对某个近期日的键只概括该日及更早的日 | `model.SharedCausalBlock` | `EncoderOrderTests` |
| C6 | [被屏蔽的词元不起作用](PROOFS.md#exact-properties-of-the-carrier-network) | 总是；屏蔽掩码本身是输入 | 缺失日的占位值进不了预测 | `model.StructuredLoadNetwork` | `MaskTests` |
| C7 | [尺度等变](PROOFS.md#exact-properties-of-the-carrier-network) | $c>0$ 且最近 8,760 小时有正负荷；2 的幂次逐位成立；否则有反例 | 载体预测不依赖负荷的计量单位 | `features.features`、`model.StructuredLoadNetwork` | `ScaleTests` |
| C8 | [没有单位标识输入](PROOFS.md#exact-properties-of-the-carrier-network) | 总是；类别是输入 | 同一类别中同样的历史得到同样的预测 | `features.features`、`model.build_carrier` | `UnitIdentityTests` |
| C9 | [复制指针：初始化时为零、不改变水平、近期没有完整日时不起作用](PROOFS.md#exact-properties-of-the-carrier-network) | (a) 初始化时；(b)–(d) 任意权重 | 指针只重新分配一个月的电量，不移动电量本身 | `model.CopyPointerNetwork` | `PointerTests` |
| C10 | [伪起报点的样本外保证](PROOFS.md#exact-properties-of-the-carrier-network) | 测试起报点至少在训练截止点之后 4,464 小时（186 天） | 锚定层评分的每一个已完成伪预测对载体都是样本外的 | `inference.pseudo_origins`、`train.make_windows` | `PseudoOriginTests` |
| C11 | [权重全为 1 就是配方 R](PROOFS.md#the-loss-weighting-of-recipe-rw-c11c14) | CPU 上逐位成立；任意常数权重在实数运算中成立 | 配方 Rw 可以退化为配方 R；0.1 的各配方及其记录不变 | `train.weighted_log1p_loss`、`train.train_step`、`train.recipe` | `UnitWeightTests` |
| C12 | [每个人群保持它在抽样中的份额](PROOFS.md#the-loss-weighting-of-recipe-rw-c11c14) | 对抽样取期望（精确） | 加权只在一个人群内部移动权重，不在目标人群和借来的窗口之间移动 | `train.draw_probabilities`、`train.noise_weights` | `SourceBalanceTests` |
| C13 | [权重有界，以及下限](PROOFS.md#the-loss-weighting-of-recipe-rw-c11c14) | 下限为正（一个反例给出下限为零的情形） | 训练池里每个窗口的权重都有限、为正、不超过下限对应的权重；至少四分之一的抽样概率落在下限上 | `train.noise_weights`、`train.weighted_quantile` | `WeightBoundTests` |
| C14 | [权重只读窗口自己最近的上下文](PROOFS.md#the-loss-weighting-of-recipe-rw-c11c14) | 总是；float16 词元会让估计有舍入 | 标签、评估窗口和随机数都进不了权重；权重不依赖负荷的计量单位 | `train.noise_sigma2`、`train.pool_sigma2` | `WeightInputTests` |

`model.*`、`features.*`、`inference.*` 和 `train.*` 是 `ankyra_lite.carrier` 的模块；检验名是 [test_carrier_properties.py](test_carrier_properties.py)（C1–C10）和 [test_weighting_properties.py](test_weighting_properties.py)（C11–C14）里的测试类。

## 起报接续层的九条性质

九条都由 `ankyra_lite.continuation`（0.2.0 新增）和 `ankyra_lite.forecast` 的代码推出，对锚定层交付的任意轨迹成立。除非给出具体数字，解析陈述允许任意常数 κ₁、κ₂ ≥ 0，ρ₁、ρ₂ > 0；在浮点运算中还需要可表示性条件：自定义常数极端时，普通窗口上出现非有限的修正量或输出会抛出 `ValueError`（0.2.1 起），非零的小时也可能下溢为零。接续层把锚定后的轨迹 F 乘以 e^{Δ(h)}，Δ(h) = κ₁x_f e^{−h/12} + κ₂x_d e^{−h/168}，其中 x_f 和 x_d 是起报前最后 3 小时和最后 24 小时相对于预测自己参照日的对数比（截在 [−1, 1]）；界 E(h) = κ₁e^{−h/12} + κ₂e^{−h/168} 在 h = 0 时为 0.809，从第 8 天起不超过 0.1215，最后一小时为 0.004。

| | 性质 | 成立条件 | 对 ankyra-lite 的意义 | 代码 | 检验 |
|---|---|---|---|---|---|
| O1 | [非负、符号与零](PROOFS.md#exact-properties-of-the-continuation-layer) | 实数运算中成立；浮点下次正规数的小时可能变成零（反例），自定义常数极端时抛出 `ValueError` 而不是给出非有限的小时；最后几小时均值不高于 −ε（负负荷）时该异常取 0 | 交付的逐时预测保持非负，精确的零保持为零 | `continuation.apply_batch` | `NonnegativityTests` |
| O2 | [修正有界](PROOFS.md#exact-properties-of-the-continuation-layer) | 总是（对包络逐位成立） | 每个小时乘以 [e^{−E(h)}, e^{E(h)}] 内的因子：起报时为 [0.445, 2.246]，从第 8 天起为 [0.885, 1.130] | `continuation.delta`、`continuation.envelope` | `BoundTests` |
| O3 | [因果](PROOFS.md#exact-properties-of-the-continuation-layer) | 总是；会读取第一个预测周的日历 | 接续层只读起报前 672 小时和日历中的八个小时 | `continuation.window_inputs` | `CausalityTests` |
| O4 | [旁路](PROOFS.md#exact-properties-of-the-continuation-layer) | 状态窗口；最后几小时有缺测（经 `forecast` 不会出现）；最后几小时均值不高于 −ε（负负荷，发布版规则） | 状态规则的预测原样交付；最后几小时缺测时接续层不起作用；某段均值非正时该段异常取 0（并记录标记），两段都非正时接续层不起作用 | `continuation.apply_batch`、`forecast.forecast` | `BypassTests`、`ForecastBypassTests` |
| O5 | [可退化](PROOFS.md#exact-properties-of-the-continuation-layer) | 常数为零，或关闭接续层 | 关闭时该载体对应的 0.1 流程不变；复现 0.1 产品还要求原配方 R 的种子 0–2 检查点、相同输入和原流程 | `continuation.ZERO`、`forecast.forecast` | `DegeneracyTests` |
| O6 | [读数在接续层之前读出](PROOFS.md#exact-properties-of-the-continuation-layer) | 总是（读出规则） | 固定载体时，开关层不改变水平、电量读数（`energy_kwh`）和峰值；交付曲线之和一般与 电量读数（`energy_kwh`）相差接续前曲线/水平的原有差额及接续变化，也可能为零 | `forecast.forecast`、`readouts.continuation_energy_change` | `ReadoutTests` |
| O7 | [一致性与单调性](PROOFS.md#exact-properties-of-the-continuation-layer) | (a) 浮点数精确相等；(b) 两个异常同号（异号时符号改变一次），最后几小时均值高于 −ε | 已经接着最后几小时往下走的预测保持不变；修正方向与异常方向一致 | `continuation.anomalies`、`continuation.delta` | `ConsistencyTests` |
| O8 | [尺度等变](PROOFS.md#exact-properties-of-the-continuation-layer) | $c>0$ 且尺度高于 10⁻³ kW 的下限；2 的幂次逐位成立；否则有反例 | 接续层不依赖负荷的计量单位 | `continuation.anomalies` | `ScaleTests` |
| O9 | [参照日与身份](PROOFS.md#exact-properties-of-the-continuation-layer) | 总是；没有同类日时取 d* = 6，与开发代码不同 | 两个异常比较的是相同钟点、相同工作日类别；常数和规则由一个摘要唯一标识 | `continuation.reference_day`、`continuation.digest` | `ReferenceDayTests` |

`continuation.*` 是 `ankyra_lite.continuation`，`forecast.*` 和 `readouts.*` 是 `ankyra_lite` 的模块；检验名是 [test_continuation_properties.py](test_continuation_properties.py) 里的测试类。

## 这些性质怎样进入预测器

- **块（P1–P3）。** 预测由三个正交的块拼成，所以每一块可以单独定权、单独评分、单独读出，电量只从水平读出。
- **分工（P4–P6）。** 两个预测器之间的任何分工都可以由块损失打分，但在合并误差上最优的分工，对典型单位仍可能是亏的。所以锚定层按每个单位自己的记录给它定权。
- **权重（P7–P10）。** $m$ 小时的记录支撑 $\min\lbrace12,\lfloor(m-1344)/744\rfloor\rbrace_+$ 个已完成误差；不足两个时权重相等。收缩限定权重能移动多远；交接用最小二乘权，因为共享误差会把逆 MSE 权拉向二分之一。
- **电量与非负（P11–P13）。** 截断不会让任何一个小时变差，但没有映射能同时保持电量、非负和不增误差，所以电量在截断之前读出。（在关停窗口和微负荷窗口上，锚定层原样返回载体的轨迹，不做截断；ankyra-lite 的默认状态规则这时改为交付周季节朴素预测。）
- **峰值（P14–P18）。** 平滑预测的最大值会低估峰值，所以峰值从历史偏移包络读出；它有精确的标量形式、误差分解，以及在写明的工作分布下的中位数性质。
- **估计目标（P19）。** 合并汇总和单位等权汇总可能因代数原因符号相反，所以结果文件并列报告单位等权 log 比、合并 MSE 比、平均单位排名和常规指标。
- **证据界（P20）。** 聚合的极小极大速率是最坏情形的结果：七个水平候选、至多六个伪起报点误差时，它按量级占误差方差的可观比例。它为强收缩权重提供统计背景；它不是在这批数据上的测量，单凭它也不能说明这里的候选无法区分（[PROOFS.md](PROOFS.md#evidence-bound)）。在测试窗口上，权重最大的候选在起报时刻是最好候选的频率高于随机水平，但大约只有五个窗口中的一个（`results/level_candidate_hit_rate.csv`）。锚定层报告的记录统计量（`anchoring_record`）是描述性指标；预测器不声称选出了最好的候选。
- **载体（C1–C10）。** 载体交给锚定层的是一条非负、均值等于其水平的轨迹（C1–C3），只用每个起报时刻之前的数据，并且不受占位值影响（C4–C6），按负荷自身的尺度、没有单位标识（C7、C8）。它的复制指针重塑一个月的形状但不移动水平，并从基础网络出发（C9）。训练截止点与测试集之间留出 186 天，锚定层拿单位自己的历史给伪起报点预测评分时，每一个都对载体是样本外的（C10）；锚定层按单位定权正是以此为前提。
- **配方 Rw 的损失加权（C11–C14）。** 在第二阶段，配方 Rw 给每个训练窗口一个权重：它最近三周相对周季节朴素预测的噪声的负二分之一次方，噪声以训练池的第 25 百分位为下限。权重全部相等时，它与配方 R 逐位相同（C11）；加权只在一个人群内部移动侧重，不在目标人群和借来的窗口之间移动（C12）；权重有限、为正、有界（C13）；它们从每个窗口起报前自己的上下文读出，不用标签，也不用随机数（C14）。
- **起报接续层（O1–O9）。** 在锚定层和状态规则之后，接续层把轨迹的每个小时乘以 e^{Δ(h)}，Δ 由起报前最后 3 小时和最后 24 小时离预测自己的参照日有多远决定。这个因子为正且有界，从第一小时的 [0.445, 2.246] 收窄到最后一小时的 [0.996, 1.004]（O1、O2）；它只读起报前 672 小时和第一周的日历（O3）；状态窗口绕过它，关闭时不改变该载体对应的预测流程（O4、O5）。电量读数（`energy_kwh`）、水平和峰值在它之前读出；交付曲线之和一般与 电量读数（`energy_kwh`）不同，差额由接续前曲线与水平读数的原有差额和接续层变化组成，也可能为零（O6）。已经接着最后几小时往下走的预测保持不变，两个异常同号时整个窗口朝同一方向移动（O7）；它不依赖负荷的计量单位（O8）；它比较的是同类的量：相同的钟点、相同的工作日类别（O9）。

## 本仓库测量的后果

这些性质对任意权重成立；本仓库的结果文件在测试窗口上测量了其中某条性质的后果时，文件列在下表，并在 [PROOFS.md](PROOFS.md) 的相应陈述里注明。这些读数都是描述性的；数字及其解读见 [docs/EVALUATION.md](../docs/EVALUATION.md)。

| | 测量的内容 | 文件 |
|---|---|---|
| P1 | ankyra-lite 与各基线的逐时平方误差中每一块所占的份额 | [`results/block_shares.csv`](../results/block_shares.csv)；图 [14](../figures/fig14_block_attribution.png) |
| P4 | 用锚定后的块替换载体的块：ankyra-lite 对单独载体的逐块对比；一个窗口上的精确替换 | [`results/contrast_ii_blocks.csv`](../results/contrast_ii_blocks.csv)（`anchored vs carrier alone`）；图 [7a](../figures/fig7_operators.png) |
| P5、P10 | 交接对固定分工和固定权重（不加权、½、每个窗口一个最小二乘权、分周权重）的对比，以及按预测周估计出的权重 | [`results/handover_granularity.csv`](../results/handover_granularity.csv)、[`results/lead_weeks.csv`](../results/lead_weeks.csv) |
| P7 | 按起报前记录长度和 K_eff 分层的 ankyra-lite | [`results/history_length.csv`](../results/history_length.csv) |
| P11、P12 | 投影起作用的窗口，以及改用水平读出时的电量对比 | [`results/energy_level_readout.csv`](../results/energy_level_readout.csv) |
| P15 | 峰值读出对轨迹最大值、上一窗口峰值以及各基线上同一读出的对比 | [`results/peak_readout.csv`](../results/peak_readout.csv) |
| P16 | 单独载体上的同一读出：锚定后的日均值是否传到峰值 | [`results/peak_readout.csv`](../results/peak_readout.csv)（`same readout on the carrier alone`） |
| P19 | 并列报告单位等权 log 比、合并 MSE 比、平均单位排名和常规指标 | `results/benchmark_pairwise.csv`、`benchmark_mean_unit_rank.csv`、`conventional_metrics_late.csv` |
| P20 | 权重最大的水平候选的命中率对随机水平；锚定记录对实现收益 | [`results/level_candidate_hit_rate.csv`](../results/level_candidate_hit_rate.csv)、[`results/anchoring_gain_deciles.csv`](../results/anchoring_gain_deciles.csv) |
| C12、C13 | 按人群：损失权重的下限、归一化系数、各人群的抽样份额、Kish 有效样本量比，以及最大权重与中位权重之比 | [`checkpoints/MANIFEST.json`](../checkpoints/MANIFEST.json) 中每个 Rw 检查点的 `weights_record` |
| O2、O7 | 同一载体上开启与关闭接续层的对比，分第 0 小时、第 0–2 小时（文件中的 `lead hours 0-3`）、第 1 天、第 2–7 天、第 8–31 天和整个窗口 | [`results/continuation_layer.csv`](../results/continuation_layer.csv) |
| O4、O6、O9 | 被旁路的状态窗口、被截断的异常所占的份额、用了参照日后备值的窗口、最大的 \|Δ\|，以及交付曲线均值与电量读数的相对差 | [`results/continuation_record.csv`](../results/continuation_record.csv) |

P1–P20 及其文件来自 ANKYRA 2.2.1 的理论文件夹，只把导入从 `ankyra` 改为 `ankyra_lite.anchor`。该理论里的数据证据（来自方法早期开发和 ANKYRA 研究，载体是 TimesFM 2.5）不是本仓库的结果，已删去；本仓库结果文件测过后果的，PROOFS.md 注明该文件。C1–C10 和 `test_carrier_properties.py` 是 ankyra-lite 新增的；C11–C14 及 `test_weighting_properties.py`、O1–O9 及 `test_continuation_properties.py` 是 0.2.0 新增的。
