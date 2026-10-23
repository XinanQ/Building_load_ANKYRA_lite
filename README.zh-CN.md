# Building_load_ANKYRA_lite

**0.2.1 版（2026 年 10 月 22 日）是维护版本。** 修复了代码审核发现的接口与保护问题，并更正了文档；默认输出、检查点、`results/` 和 `figures/` 都与 0.2.0 版相同，本页的每个数字都由 0.2.0 版得出（[更正说明](docs/CODE_AUDIT_CORRECTIONS_20261022.md)）。没有重新训练或评分。日历公告例外仍然存在，不能据此认证所有输入在起报时都已可得。

**不用基础模型的月前逐时负荷预测：一个 61,613 个参数的载体网络，每个人群单独训练，并锚定到每个单位自己的历史**

[English](README.md) · [安装与运行](#安装与运行) · [快速上手](#快速上手) · [用自己的数据](#输入与输出) ·
[模型卡](#模型卡) · [接续层](#接续层) · [训练配方 Rw](#训练配方-rw) · [数据划分](#数据划分) ·
[评估](#评估) · [检查点](#检查点) · [方法](docs/METHOD.md) ·
[评估细节](docs/EVALUATION.md) · [外部检验](docs/EXTERNAL_TESTS.md) · [数据](docs/DATA.md) ·
[理论](theory/README.zh-CN.md) · [结果数据](results/) · [图](figures/)

中文页面只有本页和 [theory/README.zh-CN.md](theory/README.zh-CN.md)；其余文档（`docs/`、`results/README.md`、`evaluation/README.md`、`figures/README.md`、`theory/PROOFS.md`）为英文。

ankyra-lite（包名 `ankyra-lite`，发行名 `building-load-ankyra-lite`，导入名 `ankyra_lite`）根据一个单位（一栋楼、一块电表、一个用电点或一户住户）截至起报点的自身记录，预测其后 744 小时（31 天）的逐时负荷。它由三部分组成。一个小型**载体网络**（复制指针结构化负荷网络，61,613 个参数，每个人群从零训练，不用任何预训练权重）根据单位的历史和已知日历，在起报点和六个更早的伪起报点上预测这个月；六个训练好的副本（种子）取平均。一个**锚定层**（从 ANKYRA 2.2.1 包逐字节复制而来）在单位已经经历过的月份上给载体的伪起报点预测打分，并用这些误差逐块（月水平、日路径、日内形状）在载体和单位的长期历史之间定权。一个有四个冻结常数的**接续层**随后把锚定后这个月的头几个小时和头几天向最后观测到的几小时校正。输出是 744 小时的轨迹、月电量和月峰值读出；电量、水平和峰值都在接续层之前读出。用六个种子时，一次预测在单个 CPU 线程上约需 0.66 秒（[耗时](#每个窗口的耗时)），不下载任何东西。本仓库包含这个包、81 个训练好的检查点（九个人群：本版本的六个种子和 0.1 版的三个种子）、训练配方、评分代码、与 21 个外部基线在九个人群上的评分比较、图，以及三部分各自的精确性质。

> **状态（0.2.1 版，2026 年 10 月 22 日；全部结果来自 2026 年 10 月 21 日的 0.2.0 版）。**
>
> - **与 0.1 相比改了什么。** 载体改用配方 Rw 训练（配方 R 的第二阶段加上逆噪声损失权重），并对六个种子取平均；新增的接续层校正这个月的头几个小时和头几天。电量、水平和峰值在接续层之前读出（读出规则 A），所以交付的逐时曲线的均值不等于电量读数（[读出](#读出)）。
> - **证据性质：在已被多次读过的测试窗口上再读一次。** 0.2.0 的配置（配方、种子、接续层及其四个常数、读出规则）在读这些测试窗口之前就只根据开发窗口冻结，然后按读数前写好的规则把测试窗口为它读了一次。这些窗口此前已被读过多次（发布版评分、载体的早期版本、0.1 的配方研究），所以这次读数只定每项改动的方向，不是确认；本页九个人群的每个数字描述的都是发布的预测器在这些窗口上的表现（[先读这一节](#先读这一节)）。本页引用的开发集数字是在选出设计的开发窗口上的复用评估。
> - **外部检验重跑一次**：用本版本在 0.1 已评分的四个测试集上重跑（同样的窗口、比较对象和估计量，此前读过一次）；赫尔辛基仍然封存（[外部检验](#外部检验)，完整报告 [docs/EXTERNAL_TESTS.md](docs/EXTERNAL_TESTS.md) 为英文）。
> - **含权重。** 本版本的 54 个检查点和 0.1 版的 27 个检查点都在 [`checkpoints/`](checkpoints/MANIFEST.json) 中（[检查点](#检查点)）。复现 0.1 的预测需要 0.1 的检查点、三个种子并关闭接续层；在六种子 Rw 载体上关闭接续层得到的是 0.2.0 载体在接续层之前的预测，不是 0.1 版。

## 一眼看懂

ankyra-lite 与 21 个外部基线（ANKYRA 2.2.1 发布版的 20 个基线加上 LSTM-T-KAN-M；共 22 个预测器）在九个人群测试窗口上的比较：1,891 个单位的 5,716 个窗口。“明确”指该对比的 95% 单位与月份自助区间不含零。

| | 结果 | 范围 | 文件 |
|---|---|---|---|
| 在 22 个预测器中的位置 | **全部九个**人群上单位平均名次都最低 | 测试窗口 | `benchmark_mean_unit_rank.csv` |
| 逐时误差 | 每个人群上明确好于 21 个基线中的 13–21 个（189 个对比中的 153 个）；明确更差的有 1 个对比：GoiEner 住户对 TimesFM（−4.6%） | 九个人群 | `benchmark_pairwise.csv` |
| 月电量 | 每个人群上明确好于 21 个中的 6–20 个（189 个中的 114 个）；没有明确更差的 | 九个人群 | `energy_error.csv` |
| 秩检验 | 平均名次 5.234，下一个预测器（逐单位岭回归）为 7.451；Nemenyi 临界差 0.786 | 六个测试人群合并，1,762 个单位 | `rank_tests.csv` |
| 峰值读出 | 峰值误差比它自身交付轨迹的最大值低 21.3–68.2%；全部九个人群上都明确 | 九个人群 | `peak_readout.csv` |
| 对 0.1 版，逐时 | 整月 **+1.15% [+0.60, +1.71]**；提前 0–3 小时（第 0–2 小时）+28.46%，第 1 天 +13.97%，第 2–7 天 +2.67%，第 8–31 天 +0.67%，全部明确 | 九个人群，等权 | `recipe_history.csv`（`primary: 0.2.0 vs 0.1.0`） |
| 对 0.1 版，电量与峰值 | 月电量 +1.03% [−1.77, +3.74]，峰值 −0.35% [−2.18, +1.45]；都不明确 | 九个人群，等权 | `recipe_history.csv` |
| 接续层开对关（同一载体） | 整月 +0.46%，第 0 小时 +28.0%，第 1 天 +11.1%，第 2–7 天 +1.9%，第 8–31 天 +0.17%，全部明确 | 1,891 个单位合并 | `continuation_layer.csv` |
| 对比 (i)：锚定层对单独的载体 | 人群数（明确更好 / 不明确 / 明确更差）：逐时误差 **2 / 7 / 0**，月电量 **0 / 9 / 0** | 九个人群 | `carriers_in_anchoring_layer.csv` |
| 对比 (ii)：含载体对纯历史配置 | 逐时 **9 / 0 / 0**（+6.1% 到 +20.6%），月电量 **5 / 4 / 0** | 九个人群 | `carriers_in_anchoring_layer.csv` |
| 对无载体对照（同一流程，把纯历史预测放在载体的位置上） | 逐时 **9 / 0 / 0**（+3.8% 到 +15.6%） | 九个人群 | `contrast_ii_control.csv` |
| 80% 预测区间 | 覆盖 0.766–0.855 的小时（90%：0.858–0.920）；TimesFM 原生分位数为 0.345–0.606，Chronos-2 为 0.629–0.804 | 九个人群 | `intervals_by_population.csv` |
| 耗时 | 每次预测 0.655 秒（载体在 7 个发布时刻 × 6 个种子、锚定层、接续层、峰值读出） | 单个 CPU 线程，6 个奥斯陆窗口 | `cost_per_window.csv` |

对比 (i) 和 (ii) 比较的是接续层之前的锚定预测，与 0.1 版相同；含接续层的行在同一文件中。

- **读出（规则 A）。** 电量由水平读出（744 × 水平，在非负投影之前），月峰值由偏移包络读出（每天的预测均值加上同日类型近期的最大偏移），作用在接续层之前的轨迹上；预测区间由单位自己纯历史伪预测的残差得到，放在交付轨迹周围（[读出](#读出)）。因为接续层在这些读出之后起作用，交付逐时曲线的均值与电量读数不同：在测试窗口上中位数 0.77%，第 99 百分位 7.8%，最大 310%（`continuation_record.csv`，`all nine populations` 行；[细节](#读出)）。
- **诊断记录。** 每个预测都精确拆成水平、日路径和日内三块（P1），每一块的权重随预测一起返回（水平权重、按周的交接权重、日内信任），`anchoring_record` 报告载体在单位自己过去几个月上相对历史的表现，`continuation` 报告接续层做了什么（两个异常、参照日、Δ(h) 和是否旁路）（[输出](#输出)）。
- **对比不是包含检验。** 对比 (i) 和 (ii) 衡量的是在这些窗口上一种配置相对另一种的增量精度，不是正式的预测包含检验（Chong 和 Hendry 1986；Harvey、Leybourne 和 Newbold 1998）：不明确的对比不说明一个预测包含了另一个，锚定层收缩后的权重是用于预测的估计量，不是检验统计量（[参考文献](#来源许可与引用)）。

**它的边界在哪里。**

- **是复用评估，不是检验。** 这些窗口此前已被读过多次。外部重跑读的是 0.1 的冻结外部检验读过一次的测试集，不是未见过的人群；在那里的第 1 个预测日，零样本 TimesFM 2.5 在 LCL 上的逐时误差仍然明确更低（−10.43%；LCL 在 TimesFM 2.5 的预训练数据中），在头三个小时里在 LCL、UNICON 和 GoiEner 检验上也明确更低（[外部检验](#外部检验)）。
- **住户的逐时误差。** 在 GoiEner 住户上，零样本 TimesFM 的逐时误差明确更低（−4.6%），其余三种零样本变体的点估计也更低（−0.9% 到 −2.9%，不明确）。ankyra-lite 在那里的月电量误差和峰值误差明确低于全部四种零样本基础模型变体。
- **月初几天。** 接续层缩小了这里的差距（第 1 天相对同一载体不加接续层 +11.1%），但在第 1 个预测日，它在六个测试人群上仍排 22 个中的第 2 到第 6，在每个人群上都落后于零样本基础模型。
- **四个人群上对零样本基础模型。** 在 BDG2 2017、GoiEner 非住户、HEEW 和 EWELD 上，与 TimesFM、TimesFM-X、Chronos-2 或 Chronos-2-X 的逐时对比没有一项在任一方向上明确，有些点估计为负（BDG2 对 Chronos-2-X −4.0%）。
- **配方有已披露的损失**（来自 0.1 版的配方研究）：与不归还验证段的同一继续训练相比，归还验证段在德拉门 −2.90%，在有开发窗口的六个人群的第 2–7 个预测日上 −1.35%（明确）。
- **每个人群一个载体。** 检查点没有在它自己以外的任何人群上评估过。

![ankyra-lite 的结构](figures/fig1_architecture.png)

*图 1. 结构，按时间展开。a，载体网络在起报点 o 和至多六个伪起报点 o − 744k（Δ = 744 小时）上各发布一次预测，每次只用该发布时刻之前的数据；每个伪起报点预测覆盖的那个月在 o 时已经观测到，并在这个月上打分（e_k）。b，载体网络：日 token、月 token 和预测日 token，因果空洞卷积编码器，注意力读出（31 个日查询和一个学到的全局查询），覆盖单位最近 56 日位置内完整日的复制指针，以及能量预算解码器 F(d, h) = 744 · ℓ · p_d · q_d(h)。c，锚定层（ANKYRA 2.2.1 的副本，其中没有任何东西被训练）按打分 e_k 设定权重，在载体和单位自己的历史之间为月水平、日均值和日内形状定权；电量和峰值从它的轨迹读出，接续层随后把头几个小时和头几天向最后观测到的几小时校正。载体为六个种子、配方 Rw。图中的序列以及所有曲线、柱和权重都是合成的示意，不是结果。每次预测的耗时取自 `results/cost_per_window.csv`。*

**简要原理。** 术语的定义见[本页用到的术语](#本页用到的术语)。

- **载体**把单位最近 366 天读作日 token，把更早的 24 个 744 小时块读作月 token，再读入 31 个预测日的日历。一个因果空洞卷积编码器和一个注意力读出，接到一个能量预算解码器上：正的月水平乘以日份额再乘以小时份额，所以预测非负，均值就是水平。复制指针让每个预测日借用单位自己最近完整日的日内曲线和相对日均值。
- **每次预测要发布七次载体预测**：在起报点 o 和 o − 744k（k = 1 … 6）处。每个伪起报点的那个月在起报时已经观测到，所以锚定层可以在已经发生的月份上，拿载体和单位自己的历史作比较、打分。
- **锚定层**把一个预测精确拆成水平、居中的日路径和日内形状。它把载体的窗口均值与六个历史水平候选放在一起定权，按周从载体取一部分日均值（交接），并把单位的相似日形状混入载体的日内形状；每个权重都来自单位自己已完成的伪预测，并向先验收缩。
- **接续层**把最后 3 个和最后 24 个观测小时，与锚定预测在同一工作日类别的第一个预测日上的对应值比较，然后把整个月乘以一个因子：起点最多 e^(±0.809)，按 12 小时和一周两个时间常数衰减，到最后一小时不超过 ±0.4%。状态窗口绕过它；电量、水平和峰值在它之前读出。
- **六个种子，加权训练。** 用不同种子训练的六个载体在每个发布时刻取平均。训练第二阶段中，每个窗口的损失按它自己近期周朴素噪声的平方根倒数加权，让被不可预测噪声主导的窗口对载体的牵引更小。
- **测试集从载体训练截点之后 186 天开始**，这样锚定层打分的每个伪起报点预测对载体都是样本外的。
- **预测时不训练任何东西。** 载体每个人群训练一次；锚定层在每个起报点上根据单位的记录估计逐单位的权重。

## 目录

- **使用：** [仓库结构](#仓库结构) · [安装与运行](#安装与运行) · [快速上手](#快速上手) ·
  [输入与输出](#输入与输出)（[输入](#输入) · [输出](#输出) · [读出](#读出) ·
  [需要多少历史](#需要多少历史) · [什么时候不该用](#什么时候不该用)） · [检查点](#检查点)
- **原理：** [模型卡](#模型卡) ·
  [锚定层怎样使用载体](#锚定层怎样使用载体) · [接续层](#接续层) ·
  [训练配方 Rw](#训练配方-rw) · [数据划分](#数据划分) · [本页用到的术语](#本页用到的术语) ·
  [起报点之后无信息](#起报点之后无信息) · [基线](#基线)
- **读结果：** [评估](#评估)（[先读这一节](#先读这一节) · [0.2.0 改了什么](#020-改了什么) · [总览](#总览) ·
  [零样本基础模型](#与零样本基础模型比较) · [秩检验](#秩检验) ·
  [常规指标](#常规指标) · [按预测日](#按预测日) · [按小时看接续层](#按小时看接续层) ·
  [月电量](#月电量与月内累计电量) · [峰值](#峰值读出) ·
  [误差分块](#误差分块与尺度化误差) · [区间](#预测区间) ·
  [各部分带来什么](#各部分带来什么) · [种子](#种子) ·
  [锚定层中的其他载体](#同一锚定层中的其他载体) ·
  [历史长度](#历史长度与锚定记录) · [稳健性](#对受损上下文的稳健性) ·
  [训练预算](#稳健性训练型基线的训练预算) · [一年上下文](#一年上下文的基础模型) ·
  [耗时](#每个窗口的耗时) · [配方历史](#配方历史r-是怎样选定的以及它的负面结果)） ·
  [ankyra-lite 的薄弱之处](#ankyra-lite-的薄弱之处) · [局限](#局限) · [外部检验](#外部检验)
- **核对：** [测试](#测试) · [理论](#理论精确性质) · [可复现性](#可复现性) ·
  [版本历史](#版本历史) · [来源、许可与引用](#来源许可与引用)

## 仓库结构

```
ankyra_lite/                 the forecaster (import ankyra_lite)
  forecast.py                forecast(): carrier forecasts -> anchoring layer -> state rule -> readouts -> continuation layer;
                             ForecastResult; forecast(history, None) is the history-only configuration
  continuation.py            the origin-continuity layer: apply, apply_batch, the four frozen constants (CONSTANTS, DIGEST)
  _release.py                the release switches (version, layer default, recipe, seeds, checkpoint folder); checkpoint_paths
  readouts.py                readouts of a forecast: interval, blocks, raw_trajectory, projection_energy_increase,
                             continuation_energy_change, peak
  carrier/                   the carrier network
    model.py                 CopyPointerNetwork, build_carrier, load_carrier, forecast_loss, param_groups (61,613 parameters)
    features.py              the carrier's inputs at one issue time, from data before it only
    inference.py             carrier_forecasts (origin and pseudo-origins), forecast_windows (batched), pseudo_origins
    train.py                 the training recipes R and Rw: TrainConfig, NoiseWeights, recipe(), prepare_source, build_pool,
                             train_one, train_ensemble
  anchor/                    byte-identical copy of the ANKYRA 2.2.1 anchoring layer (17 files; PROVENANCE.md lists their hashes)
checkpoints/                 81 carriers: <population>/rw/seed0-5.pt (0.2.0, recipe Rw) and <population>/seed0-2.pt (0.1, recipe R),
                             and MANIFEST.json (hashes, cuts, source runs, loss weighting)
examples/quickstart.py       end to end on an artificial building with the shipped checkpoints
examples/own_unit.py         your own unit's record, on an artificial meter: a record that starts in the middle of a year,
                             holidays, gaps, the history-only call, the interval and the anchoring record
tests/                       275 tests: the anchoring layer's own tests, equivalence with the original package and the
                             research code, the carrier, the training recipes, the continuation layer, the forecast API
                             against 0.1.1, the evaluation module, the comparison script and the external-test table
theory/                      the exact properties: P1-P20 (anchoring layer), C1-C14 (carrier and recipe Rw) and O1-O9
                             (continuation layer), proofs, operators, 111 checks
evaluation/                  the scoring module, a runnable example, and the scripts that wrote results/ (need the study's data)
results/                     every scored statistic (CSV / JSON), SUMMARY.md, the reproduction record; README.md lists the columns
figures/                     make_figures.py, architecture.py (the drawing of Figure 1) and the figures (PDF and PNG), drawn
                             from results/ (Figure 7 also from theory/)
docs/                        METHOD.md (network, recipes, anchoring, continuation layer, constants), EVALUATION.md (protocol,
                             baselines, full tables), DATA.md (sources, preparation, splits, borrowed windows),
                             EXTERNAL_TESTS.md (the external tests and the 0.2.0 re-run)
tools/export_checkpoints.py  the export of the checkpoints with their verification (needs the study's data)
tools/products_for_configuration.py  the products mapping of one configuration (recipe, seeds, layer on or off)
.github/workflows/tests.yml  CI (Python 3.10-3.13, CPU): tests, theory checks, quickstart, own-unit example, evaluation
                             example; the figures regenerated from results/ on Python 3.11
pyproject.toml, LICENSE (MIT), CITATION.cff, README.zh-CN.md
```

有意不包含的内容：

- **原始数据、整理好的数据仓库和保存的预测。** 九个人群都是公开数据；[docs/DATA.md](docs/DATA.md) 给出每个来源及其整理方法。`results/` 存放的是评分得到的统计量，不是计算它们所用的预测，所以只靠本仓库无法重新生成这些表格。
- **基线的代码。** 只包含三个简单的参照预测器（`evaluation/baselines.py`）；其他基线在 [docs/EVALUATION.md](docs/EVALUATION.md#baselines) 中说明。
- **基础模型的权重。** ankyra-lite 不用任何基础模型；零样本基础模型只作为基线出现。

## 安装与运行

需要 Python ≥ 3.10、NumPy、SciPy 和 PyTorch。预测、快速上手、测试和画图用 CPU 就够；研究中发布的载体是在 GPU 上训练的。

```bash
git clone https://github.com/XinanQ/Building_load_ANKYRA_lite.git
cd Building_load_ANKYRA_lite
pip install -e .
python -B examples/quickstart.py
```

缺少 PyTorch 时，`pip install -e .` 会安装它。如果只要 CPU 版，先装好它：`pip install torch --index-url https://download.pytorch.org/whl/cpu`。`pip install -e ".[figures]"` 会另外安装 `figures/make_figures.py` 需要的 Matplotlib。运行时不下载任何东西：检查点就在仓库里。

**软件环境。** 仓库记录了以下内容，仅此而已：

- `pyproject.toml` 只给出版本下限：Python ≥ 3.10、NumPy ≥ 1.26、SciPy ≥ 1.11、PyTorch ≥ 2.4；画图可选 Matplotlib ≥ 3.8。
- CI（`.github/workflows/tests.yml`）在 Python 3.10、3.11、3.12 和 3.13（Ubuntu，只用 CPU 的 PyTorch）上运行两套测试、快速上手、自己单位的示例和评估示例，并在 Python 3.11 上重新生成图。需要研究本地数据或 ANKYRA 仓库副本的测试在那里会跳过。
- 发布的图用 Python 3.11、Matplotlib 3.11.0 和 NumPy 2.4.4 绘制（[figures/README.md](figures/README.md)）。快速上手和本页的代码于 2026 年 10 月 21 日在 CPU 单线程上用 Python 3.11、NumPy 2.4.4 和 PyTorch 2.11 运行过。
- 发布的检查点来自研究在 GPU 上的运行；这些运行所用的软件包版本没有记录在本仓库中。

## 快速上手

```bash
python -B examples/quickstart.py                          # the release's checkpoints of Cambridge, an artificial building
python -B examples/quickstart.py --population households  # the release's checkpoints of another population
python -B examples/quickstart.py --checkpoints DIR        # every seed*.pt in DIR, averaged
python -B examples/quickstart.py --untrained              # an untrained network of the same shape (format only)
```

这个脚本载入发布版剑桥的六个检查点（`checkpoints/cambridge/rw/seed0.pt` … `seed5.pt`），用 `ankyra_lite.anchor.synthetic` 构造一栋有 16,000 小时历史的人造办公楼，计算载体在起报点和六个伪起报点上的预测，做锚定，施加状态规则和接续层，然后打印各项读出和预测背后的权重。它不需要任何数据，一次预测在单个 CPU 线程上约需一秒。输出节选（列表压缩成一行）：

```
 "carrier": "6 checkpoint(s) in cambridge/rw/: seed0.pt, seed1.pt, seed2.pt, seed3.pt, seed4.pt, seed5.pt",
 "carrier_parameters": 61613,
 "carrier_issue_times_k": [0, 1, 2, 3, 4, 5, 6],
 "state_window": false,
 "energy_kwh": 17048.084,
 "level_kw": 22.9141,
 "peak_kw": 27.706,
 "continuation_x_fast_x_day": [0.0701, 0.0685],
 "continuation_energy_change_kwh": 95.644,
 "interval_pseudo_windows": 12,
 "lead_week_weights_on_carrier": [0.145, 0.125, 0.125, 0.125],
 "pseudo_origin_pairs": 6,
 "within_day_trust_on_analog_shape": [0.5, 0.5, 0.5, 0.5],
 "within_day_pseudo_origin_triples": 3,
```

这些检查点属于剑桥人群及其训练截点，而人造建筑几乎是严格周期的（日内信任停在上限 ½），所以输出说明的是格式，不是典型数值。这里接续层给交付的这个月增加了 95.6 kWh（`energy_kwh` 的 0.56%）；`energy_kwh`、`level_kw` 和 `peak_kw` 都在它之前读出。各行的含义见[输出](#输出)。

同样的步骤用 Python 写（2026 年 10 月 21 日已从仓库根目录完整运行核对过）：

```python
import numpy as np
from ankyra_lite import checkpoint_paths, load_carrier, carrier_forecasts, forecast, readouts
from ankyra_lite.anchor.synthetic import synthetic_history      # an artificial building, for the example only

carriers = [load_carrier(p) for p in checkpoint_paths("cambridge")]   # checkpoints/cambridge/rw/seed0..5.pt, ascending
history = synthetic_history(16000)       # or ankyra_lite.anchor.History(load_kw, temperature_c, day_types, start_timestamp)

# step 1: the carrier's forecasts at the origin (k = 0) and at the pseudo-origins origin - 744 k (k = 1 ... 6)
load = np.asarray(history.load_kw, dtype=np.float64)
F = carrier_forecasts(carriers, load, history.temperature_c, history.day_types, "Office",
                      history.start_timestamp, unit=0, origin=len(load))
print(sorted(F))                         # [0, 1, 2, 3, 4, 5, 6]

# step 2: the anchoring layer, the state rule, the readouts and the continuation layer (on by default)
r = forecast(history, F, group="Office", temp_sigma_std=0.25, dst_region="EU")
print(r.trajectory_kw.shape, r.state_window, r.level_kw, r.energy_kwh, r.peak_kw)
print(r.continuation.x_fast, r.continuation.x_day, r.continuation.d_star)        # what the layer did
print(r.trajectory_kw.sum() - r.energy_kwh,                                      # generally not zero (rule A):
      readouts.projection_energy_increase(r) + readouts.continuation_energy_change(r))   # the same, up to rounding
off = forecast(history, F, group="Office", temp_sigma_std=0.25, dst_region="EU", continuation=False)
assert off.energy_kwh == r.energy_kwh and np.array_equal(off.trajectory_kw, r.trajectory_before_continuation_kw)
```

第 1 步和第 2 步可以合成一次调用：`forecast(history, carriers, group="Office", temp_sigma_std=0.25, dst_region="EU")` 给出相同的轨迹。传入载体列表即为集成（各成员预测的简单平均，按给定顺序求和）；只传一个载体也可以。在某个人群自己的记录上预测时，用 `cut=` 传入该人群的训练截点（[检查点](#检查点)）。

## 输入与输出

`ankyra_lite.forecast` 预测 `load_kw` 最后一个值之后的 744 小时。先把单位放到一个规则的逐时网格上：两个部分都不做重采样，不填补起报点处的上下文，也不转换时区或夏令时（[docs/DATA.md](docs/DATA.md#the-input-format)）。违反锚定层约定的输入会报错。

### 输入

记录 `ankyra_lite.anchor.History`（锚定层的输入记录，与 ANKYRA 2.2.1 相同，未改动）：

| 字段 | 要提供什么 | 载体怎样读它 | 锚定层怎样读它 |
|---|---|---|---|
| `load_kw` | 逐时平均负荷，单位 kW；下标 0 是第一年的 1 月 1 日 00:00；最后一个值是起报点前一小时；缺测处为 NaN | 只读每个发布时刻之前的小时；负值或非有限值算作缺测；没有任何观测小时的日子被屏蔽 | 常规载体推理要求最后 1,344 小时完整观测（从不填补）；预计算映射的停机分支例外见下文；更早的、在同一个 744 小时块内不超过 6 小时的缺口，在其估计器读取的副本中做插值 |
| `temperature_c` | 同样各小时的室外逐时气温，单位 °C | 只读每个发布时刻之前的小时，经日均值及其覆盖率读入；允许缺测 | 起报点之前处处有限；不需要未来气温 |
| `day_types` | 历史加上 744 个预测小时，每小时一个整数：周一 = 0 … 周日 = 6，法定假日 = 7 | 读到每个发布时刻的预测月为止（日历按起报时已知处理） | 同左 |
| `start_timestamp` | 第一年的 1 月 1 日 00:00，带 `+00:00` 偏移，例如 `"2019-01-01T00:00:00+00:00"`（固定偏移的网格） | 每个 token 的年相位和记录的绝对时间 | 年温度谐波的相位，以及相似日的夏令时状态 |
| `observed`（可选） | 负荷的布尔掩码 | 被屏蔽的小时按缺测处理 | `False` 的小时被忽略 |

**记录从年中开始。** 用 NaN 把负荷向前补到第一年的 1 月 1 日 00:00，补齐的小时也要给出有限的温度（有档案温度就用档案温度），写出补齐那些天的日类型，并把这个 1 月 1 日 00:00 作为 `start_timestamp`。[`examples/own_unit.py`](examples/own_unit.py) 在一块从 2019 年 3 月 18 日开始、有法定假日和三处缺测的人工电表上从头到尾做了这些，并用发布的检查点（`cut=None`）运行预测、纯历史配置、区间读出和锚定记录（`python -B examples/own_unit.py`；细节见 [docs/DATA.md](docs/DATA.md#how-much-history)）。

`forecast(history, carriers, *, group, temp_sigma_std, dst_region="none", cut=None, state_rule="seasonal_naive", continuation=True, device="cpu", **anchor_kwargs)` 的其他参数：

- `carriers`：一个载体网络、一个载体列表（集成，在每个发布时刻按给定顺序取平均；发布版的种子请按升序传入，即 `ankyra_lite.checkpoint_paths(population)`，并对它们只传一个训练截点），或预先算好的映射 `{k: (744,) carrier forecast issued at origin − 744k}`；传 `None` 得到**纯历史配置**（没有载体的锚定层：历史水平和日路径，加上同日类型的日内默认形状；不用状态规则；`carrier` 为 `{}`），即对比 (ii) 中的简化配置。
- `group`：`"Industrial"`、`"Office"`、`"Public"`、`"Residential"` 或 `"Commercial"`。它既是载体的类别输入，也是锚定层选择固定温度特征先验的依据。`"Commercial"` 在锚定层中没有先验曲线（温度特征为零），与 ANKYRA 2.2.1 相同。
- `temp_sigma_std`：该人群的温度异常尺度，一个正数，单位为 10 °C，在训练截点之前的日子上拟合（[研究中怎样拟合](docs/DATA.md#the-input-format)）。
- `dst_region`：`"EU"`、`"US"` 或 `"none"`；只用于从相同的夏令时状态中取相似日。
- `cut`：载体的训练截点，以记录的小时下标表示。截点之前的伪起报点不使用（不计算；传入预先算好的映射时则从中删去）；起报点早于截点会报错。只要预测的记录是载体训练时用过的，就要传入。
- `state_rule`：`"seasonal_naive"`（评估过的配置，见下文）或 `None`。
- `continuation`：`True`（发布默认值，见 `ankyra_lite/_release.py`：使用冻结常数的接续层）、`False` 或 `None`（关闭接续层：交付接续层之前的轨迹），或一个 `ankyra_lite.continuation.Constants`（`continuation.ZERO` 在每个轨迹有限的窗口上逐位给出接续层之前的轨迹，并附上接续层的记录）。纯历史配置从不接续。复现 0.1 版的预测需要 0.1 的检查点（`checkpoints/<population>/seed0-2.pt`）、相同输入并设 `continuation=False`；在六种子 Rw 载体上关闭接续层不是 0.1 版。
- `**anchor_kwargs`：传给锚定层（`gap_tolerance`、`single_trust`、`within_anchor`、`micro_load_rule`）；评估用的都是它们的默认值（全部为 `True`）。

在顶层导入 `forecast` 会把属性 `ankyra_lite.forecast` 重新绑定到这个函数；该模块的其他名字请显式导入，例如 `from ankyra_lite.forecast import ForecastResult, seasonal_naive`。

### 输出

`forecast` 返回一个 `ForecastResult`：

| 字段 | 含义 |
|---|---|
| `trajectory_kw` | (744,) 交付的逐时预测，单位 kW，在接续层之后；第 0 个元素是最后一个负荷值之后的那一小时 |
| `trajectory_before_continuation_kw` | (744,) 接续层之前的轨迹：锚定层非负投影之后的输出，状态窗口上为季节性朴素预测。`peak_kw` 从它读出；关闭接续层时它与 `trajectory_kw` 是同一个数组 |
| `level_kw`、`energy_kwh` | 锚定层的水平，在其非负投影之前、也在接续层之前读出（状态窗口上采用季节性朴素规则时，是那个预测的均值）；以及月电量，即 744 × 水平。开启接续层时，`trajectory_kw` 的均值一般不等于 `level_kw`（[读出](#读出)） |
| `peak_kw` | `trajectory_before_continuation_kw` 的月峰值读出：在 31 天中，取当天预测均值加上同日类型近期最大偏移的最大者（`anchor.readouts.peak_readout`）；不一定等于这条曲线的最大值 |
| `continuation` | 接续层做了什么（`ankyra_lite.continuation.ContinuationRecord`）：`x_fast`、`x_day`（截断后的异常）、`d_star`（参照日）、`matched`、`delta`（Δ(h)）、`bypass`（`"state window"`、`"missing hours"`、`"nonpositive load"` 或 None）、标记 `nonpositive_fast`、`nonpositive_day`，以及所用常数的摘要；关闭接续层或纯历史配置时为 None |
| `state_window`、`state_rule` | 锚定层是否报告了停机或微负荷窗口，以及实际采用的状态规则（`"seasonal_naive"` 或 `None`） |
| `carrier` | {k: (744,)} 交给锚定层的载体预测 |
| `anchored` | 锚定层自己的结果；其他属性都从这里读取 |

透传到锚定层的属性，按快速上手打印的样子列出：

| 快速上手中的行 | 字段 | 含义 |
|---|---|---|
| `carrier_issue_times_k` | `carrier` 的键 | 载体发布预测的时刻 o − 744k；0 是起报点 |
| `continuation_x_fast_x_day` | `continuation.x_fast`、`continuation.x_day` | 接续层的两个截断后的异常（最后 3 个和最后 24 个观测小时对预测参照日）；0 表示该时段不做校正 |
| `continuation_energy_change_kwh` | `ankyra_lite.readouts.continuation_energy_change(r)` | 接续层给交付的这个月增加的电量（负值为减少）；不计入 `energy_kwh` |
| `lead_week_weights_on_carrier` | `lead_week_weights` | 交接：第 1–7、8–14、15–21 和 22–31 天中载体日均值所占的权重；½ 表示没有证据 |
| `level_weights` | `level_weights` | 每个水平候选的权重：`s`、`l` 和 `a` 分别是最近 7 天均值、长期历史均值和一年前同一窗口的均值（`_u` 不作、`_w` 作天气修正）；`fm` 是载体自己的窗口均值（键名沿用锚定层的名字） |
| `pseudo_origin_pairs` | `pseudo_pairs` | 交接权重背后已完成的伪起报点个数（最多 6） |
| `within_day_trust_on_analog_shape` | `within_trust` | 单位自己相似日形状的权重，介于 0 和 ½ 之间；每个窗口一个值，在月内四段上重复 |
| `within_day_pseudo_origin_triples` | `within_pseudo_pairs` | 日内信任背后已完成的伪起报点个数（最多 3） |
| `analog_shape_kept` | `analog_kept` | 相似日形状大得不合理、整个窗口改用载体形状时为 `False` |
| — | `off_state`、`micro_load` | 锚定层的哪条状态规则被触发 |
| — | `anchoring_record` | 单位伪预测记录在水平上的两个统计量，预测本身不使用：`carrier_level_weight` 和 `history_vs_carrier_log_ratio`（[表](#历史长度与锚定记录)） |
| — | `daily_means_kw` | (31,) 交接之后、非负投影之前的日均值；`level_kw` 是它们的均值 |
| — | `fixed_division_daily_means_kw` | (31,) **固定分工**：历史水平和日路径，不含载体的水平候选，也不做交接 |
| — | `within_day_kw` | (744,) 交付的日内块：载体的日内块按信任混入相似日形状 |
| — | `foundation_within_day_kw` | (744,) 载体自己的日内块（字段名沿用锚定层的名字） |
| — | `analog_shape_kw` | (744,) 单位的相似日形状（合理性检查回退时为载体的块） |
| `interval_80pct_first_hour_kw`、`interval_pseudo_windows` | `ankyra_lite.interval(history, r, group=..., temp_sigma_std=...)` | 80% 和 90% 区间（(5, 744) 区间带的第 1、3 行和第 0、4 行）：由单位纯历史伪预测的残差分位数得到，加到轨迹上；这是单独的读出，不属于预测器（[读出](#读出)） |

在状态窗口上，这些字段描述的是锚定层的返回值（载体的预测，权重、固定分工和相似日字段取默认值），而不是默认状态规则交付的季节性朴素预测。`np.repeat(r.daily_means_kw, 24) + r.within_day_kw` 是投影之前的预测（`ankyra_lite.readouts.raw_trajectory`），`ankyra_lite.readouts.blocks` 把任何轨迹拆成三块。全部字段见 [docs/METHOD.md](docs/METHOD.md#output-fields)。

**状态窗口。** 预计算预测映射路径可能在完整上下文检查前进入继承的停机分支，以末周已观测值的最大值判定，再由季节性朴素规则把缺测小时填零；这一继承规则不能证明缺测小时意味着停机。常规载体推理仍拒绝不完整上下文。当最后 168 小时都不超过 10⁻⁶ kW（停机），或整个 1,344 小时上下文都在 0 附近 10⁻³ kW 以内（微负荷）时，锚定层原样交回载体的预测。在默认的 `state_rule="seasonal_naive"` 下，ankyra-lite 改为交付最后一个观测周的重复（缺测小时记为 0）；这是评估过的配置。`state_rule=None` 保留锚定层的返回值（[docs/METHOD.md](docs/METHOD.md#state-windows-and-the-seasonal-naive-rule)）。

### 读出

每个预测附带三种读出，都只用起报时可得的信息（[docs/METHOD.md](docs/METHOD.md#readouts)）：

读出规则 A：电量、水平和峰值都在接续层**之前**读出。对固定的载体，开关接续层不会改变它们的任何一位；交付的逐时曲线是接续层之后的那条。

- **电量**（`r.energy_kwh`）：744 × 水平，在非负投影之前（P3）、也在接续层之前（O6）读出。它与交付曲线之间隔着两项起报时已知的量：`ankyra_lite.readouts.projection_energy_increase(r)`（投影起作用的地方轨迹的电量更多；P11–P12）和 `ankyra_lite.readouts.continuation_energy_change(r)`（接续层增加或减少的电量），`r.trajectory_kw.sum()` = `r.energy_kwh` + 这两项。两项可能为零或相互抵消，所以一般而言、但不是每个窗口上，求和都不等于电量读数。**披露。** 在接续层起作用的 5,390 个测试窗口上（不含状态窗口），交付曲线均值与水平读数的相对差异中位数为 0.77%，第 99 百分位为 7.8%，最大为 310%（`results/continuation_record.csv`，`all nine populations` 行；各人群的最大值为 6.5% 到 13.3%，只有 GoiEner 非住户为 89.9%、GoiEner 住户为 309.8%）。接续层本身对月电量的改变在 −14.8% 到 +33.4% 之间（`layer_energy_change_pct_min`、`_max` 两列），所以凡是更大的差异，都是非负投影起作用、接续层之前的轨迹本来就比 744 × 水平电量更多的窗口。在用来选定接续层的开发窗口上，这三个数字为 0.6%、7% 和 9%（复用评估，在读测试窗口之前）。对一条平坦的轨迹，接续层最多把月电量改变 −7.34% 到 +9.31%；电量更集中在头几天的轨迹可以改变更多，极端情况下为 −55.5% 到 +124.6%（向外取整；理论 O6）。评估对 ankyra-lite 按接续层之前轨迹的电量评分，对每个基线按它自己的轨迹评分；水平读数作为敏感性分析（[月电量](#月电量与月内累计电量)）。
- **峰值**（`r.peak_kw`），从 `trajectory_before_continuation_kw` 读出：U = 各天 d 上 (L̂_d + A_τd) 的最大值，即每个预测日的均值，加上同日类型最近四个上下文日中最大的偏移（日最大值减日均值）。56 个上下文日中没有出现的日类型取全部上下文日的最大偏移，没有出现的假日类型取周日的值。由于最大值可交换，U 等于逐小时的包络；U ≥ max_d L̂_d ≥ 水平，其误差可拆成水平、日路径、幅度和选择四项，所以水平上的改进会传到峰值（P15–P17）。`ankyra_lite.readouts.peak` 可对任何轨迹做同样的读出（[结果](#峰值读出)）。
- **区间**（`ankyra_lite.interval(history, r, group=..., temp_sigma_std=...)`，单独调用）：单位自己纯历史伪预测在最多 12 个已完成伪起报点上的逐时残差，各除以该窗口的起报尺度，按一天中的小时和工作日合并，得到分位数，再乘以起报尺度加到接续层之后的交付轨迹上（在零处截断，各水平单调）。残差不涉及载体（[结果](#预测区间)）。

### 需要多少历史

- **锚定层**通常需要起报点之前的 1,344 小时完整观测，预计算映射的停机分支例外见上文。随着已完成的伪起报点增多（每个是一步 744 小时），它的权重逐渐离开先验：有 2,088 个完整小时时交接权重离开 ½（一个已完成的伪起报点），有 2,832 个时水平权重离开等权（两个），用到一年前窗口的候选需要 10,248 小时（[完整列表](docs/DATA.md#how-much-history)）。评估的每个测试窗口都至少有 9 个完整的 744 小时伪起报点窗口（`results/history_length.csv`：`K_eff` 为 9–11 的有 451 个窗口，为 12 的有 5,265 个）。
- **缺口。** 更早记录中、同一个 744 小时块内不超过 6 小时的缺口，在锚定层估计器读取的副本中做插值（短缺口规则）；对自己的记录，`ankyra_lite.fill_short_gaps(load_kw)` 返回这个副本。起报点处的 1,344 小时上下文从不填补。
- **载体**最多读最近 366 天（日 token）和其前的 24 个 744 小时块；缺测的日和块被屏蔽，所以它在更短的记录上也能运行。
- **使用检查点时**，起报点至少要在其训练截点之后 186 天：这样锚定层打分的每个伪起报点预测都是在载体训练期之后发布的（[原因](#锚定层怎样使用载体)）。

按历史长度观察到的情况是描述性的，见[下文](#历史长度与锚定记录)。

### 什么时候不该用

按本仓库的证据，下列情况下 ankyra-lite 不是更好的选择：

- **一个没有自己载体的人群里的单位。** 检查点只在它们自己人群的测试窗口上评估过。对新数据，请用[配方](#训练配方-rw)训练一个载体。
- **住户，且看重逐时轨迹时。** 在 GoiEner 住户上，零样本 TimesFM 的逐时误差明确更低（−4.6%），TimesFM-X、Chronos-2 和 Chronos-2-X 的点估计也更低（−0.9% 到 −2.9%，不明确）。如果看重的是月电量或峰值，ankyra-lite 在那里的误差明确低于全部四种零样本变体。
- **月初几天。** 在第 1 个预测日，即使有接续层，零样本基础模型在每个测试人群上都领先。
- **逐时曲线必须积分到电量读数时。** 开启接续层时，`trajectory_kw` 之和与 `energy_kwh` 之间还多出一项起报时已知的差额（`continuation_energy_change`）；用 `trajectory_before_continuation_kw` 或 `continuation=False` 可以去掉这一项，只剩投影带来的 `projection_energy_increase`（投影不起作用时为零）。
- **净计量记录。** 接续层需要最后 3 个和最后 24 个观测小时的均值（加上一个小偏移）为正。某个时段的均值不大于它时，接续层把该时段的异常置为零并标记该窗口（`nonpositive_fast`、`nonpositive_day`），也就是不再根据这个时段做校正。这条发布规则是在研究之后加的（研究的代码在这里会报错）；没有测试窗口遇到这种情况。
- **上下文里有单个大尖峰。** 一个达到上下文最大值十倍的尖峰使逐时误差上升 52.5%，并使峰值读出失效（[稳健性](#对受损上下文的稳健性)）。
- **会关停的单位，在用区间时。** 区间读出的宽度在那里不含信息。
- **不是逐时的数据，或单位不是 kW 的数据。** 状态阈值是以 kW 计的绝对值；预测长度固定为 744 小时。
- **南半球的站点。** 锚定层年温度谐波的相位是固定的，最冷的一天在 1 月（继承而来，未改动）；九个人群都不在南半球。

## 模型卡

每一项都对照代码核对过（`ankyra_lite/carrier/model.py`、`features.py`、`inference.py`、`train.py`、`ankyra_lite/forecast.py`、`ankyra_lite/continuation.py`）；完整说明见 [docs/METHOD.md](docs/METHOD.md#the-carrier-network)。

| 项目 | 内容 |
|---|---|
| 任务 | 一个单位的月前逐时负荷预测：起报点之后的 744 小时（31 天 × 24 小时） |
| 结构 | 复制指针结构化负荷网络（`CopyPointerNetwork`，`build_carrier()`），输出送入锚定层，其后是状态规则和接续层 |
| 参数 | **61,613** 个可训练参数（`CARRIER_PARAMETERS`；在 `checkpoints/MANIFEST.json` 中对每个发布的检查点重新计数）；没有预训练权重；没有单位编号嵌入 |
| 日 token | **366 × 69**，最近 366 天，从旧到新：24 × arcsinh(负荷 / s)、24 个逐时观测标记、日均值和日标准差除以 s 后的 arcsinh、arcsinh(日均气温 / 30) 及其覆盖率、8 维日类型独热（周一 … 周日、假日）、年相位（sin、cos）、相对位置、负荷覆盖率、5 维类别独热 |
| 月 token | **24 × 16**：比日跨度更早的 744 小时块（均值、标准差、第 10 和第 90 百分位、工作日和周末均值、零负荷比例、覆盖率、气温及其覆盖率、年相位、位置、一个常数、均值变化、有效标记） |
| 未来日 token | **31 × 16**：预测各日的日历，起报时已知（8 维日类型独热、年相位、相对日位置、5 维类别独热） |
| 尺度 s | max（最近 744 小时中观测小时的均值，最近 8,760 小时中观测小时的均值）；不是有限正数时取 1 |
| 编码器 | 宽度 **48**；token 嵌入、时龄嵌入和分辨率嵌入；**两个共享的门控因果块**（层归一化、核长 3 的逐深度因果卷积、门 tanh · σ、残差），在 366 个日 token 上按**空洞 1, 2, …, 128**、在 24 个月 token 上按 **1, 2, 4, 8** 重复使用 |
| 读出 | 一次多头注意力（**4 个头**），由一个全局查询和 31 个以日历为条件的日查询对 390 个记忆位置做注意力，然后接 48 → 96 → 48 的 MLP |
| 解码器 | 正的能量预算 F(d, h) = 744 · L · p_d · q_d(h)：月水平 L = s · softplus(·)，日份额 p（31 天上的 softmax），小时份额 q_d（24 小时上的 softmax）；小时 logits 来自 23 个零均值小时方向上的**秩 8 字典**（一个学到的基础加上一个有界的、随起报点变化的修正，0.1 · tanh），31 天共享，每天各有系数 |
| 复制指针 | 每个预测日对单位自己**最近 56 日位置内的完整日**做注意力（16 维的查询和键，日类型相同时另有一个学到的加分），把注意到的日内份额曲线加到它的小时 logits 上，把注意到的相对日均值加到它的日 logits 上，都经过初始化为零的门 |
| 输出 | 载体：744 小时，非负，均值等于 L（在实数运算下），水平、日路径和日内形状是轨迹的精确分块。ankyra-lite：接续层之后的 744 个非负小时；电量、水平和峰值在接续层之前读出，所以交付曲线的均值不是电量读数（[读出](#读出)） |
| 接续层 | F′(h) = F(h) e^(Δ(h))，Δ(h) = 0.479 x_f e^(−h/12) + 0.330 x_d e^(−h/168)；x_f、x_d 是最后 3 个和 24 个观测小时对预测参照日的截断对数比；四个常数在开发窗口上一次拟合，从不重新拟合；状态窗口旁路（[小节](#接续层)） |
| 训练 | 每个人群从零训练，然后归还验证段并用逆噪声损失权重再训练 8 轮（[配方 Rw](#训练配方-rw)）；种子 0–5；预测是六个种子的简单平均 |
| 训练数据 | 该人群自己在训练截点之前的窗口（第 2 阶段加入验证段），加上从其他研究人群借来的窗口，每个都在该人群的验证段之前结束（[docs/DATA.md](docs/DATA.md#windows-borrowed-from-other-populations)） |
| 检查点 | 81 个文件：本版本 54 个（`checkpoints/<population>/rw/seed0–5.pt`，配方 Rw）和 0.1 版 27 个（`checkpoints/<population>/seed0–2.pt`，配方 R），`{"model": state_dict, "epoch": 8}`，每个 265,653 字节 |
| 推理耗时 | 每窗最多 42 个成员预测（7 个发布时刻 × 6 个种子，可批处理；伪起报不可用时减少）；六个 Oslo 窗的测量为：0.135 秒；加上锚定层、接续层和峰值读出，每个窗口 0.655 秒，单个 CPU 线程（`results/cost_per_window.csv`） |
| 预期用途 | 为载体所针对的人群里的单位做月前逐时预测，起报点至少在其训练截点之后 186 天 |
| 未评估 | 不重新训练就用于其他人群；小时以下分辨率的数据；744 小时以外的预测长度；为这个载体设计的预测区间 |
| 证据 | 在已被多次读过的测试窗口上再读一次，只定方向（[先读这一节](#先读这一节)） |
| 许可 | MIT |

## 锚定层怎样使用载体

`ankyra_lite.anchor.forecast(history, foundation={k: forecast issued at o − 744k}, ...)` 就是 ANKYRA 2.2.1 的锚定层，没有改动；它的代码和文档字符串里说到基础模型的地方，接收的是载体的预测。

- **伪起报点。** 锚定层读取载体在起报点 o（k = 0）和 q = o − 744k（k = 1 … 6，相邻两个相隔 31 天）发布的预测，前提是 q 之前的 1,344 小时上下文完整观测。伪起报点的目标月 [q, q + 744) 在 o 时已经观测到，所以锚定层可以在已经发生的月份上，拿载体和单位自己的历史作比较、打分。`carrier_forecasts(..., origin=o, cut=C′)` 正好计算这些预测，每个只用它自己发布时刻之前的数据，并丢掉截点之前的伪起报点。
- **它为每个单位估计什么**（[docs/METHOD.md](docs/METHOD.md#how-the-anchoring-layer-uses-the-carrier)）：
  - **水平权重**：载体的窗口均值作为第七个候选，与六个历史水平候选（近期、长期历史和一年前的均值，各有不作和作天气修正两种）并列，按误差倒数定权，并向等权收缩；
  - **交接权重**：在第 1–7、8–14、15–21 和 22–31 天中，载体日均值相对历史侧的权重，用最小二乘估计，并向 ½ 收缩；
  - **日内信任**：在载体的日内形状里混入多少单位自己的相似日形状，每个窗口一个值，向零收缩，上限为 ½。
- **为什么测试集从截点之后 186 天开始。** 载体在伪起报点上的每个预测，对载体来说都必须是真正的预测：在训练截点 C′ 或之后发布，这样它预测的那个月从来不是训练标签。测试起报点至少在 C′ 之后 6 × 744 小时 = 186 天，所以对每个 k ≤ 6 都有 q = o − 744k ≥ C′（性质 C10）。没有这段间隔，逐单位权重就会在载体的样本内误差上估计，从而过度信任载体。代价是这里的载体比实际使用时至少旧半年。
- **交接在这些人群上怎样起作用**（`handover_granularity.csv`、`lead_weeks.csv`、图 4d–f；描述性，在此前读过的测试窗口上）。按周的权重在单位已完成的伪起报点上，用**固定分工**（历史侧的水平和日路径）对载体来估计，然后用到含载体水平候选的历史侧日均值上（[细节](docs/METHOD.md#week-by-week-handover)）。
  - 在这里，单独的载体几乎在每个预测周都好于固定分工：在七个人群上至少有一周明确（BDG2、剑桥、HEEW、EWELD、GoiEner 住户、奥斯陆、德拉门；按周为 −3.4% 到 −15.7%），在 GoiEner 非住户和 CINELDI 上不明确。锚定层因此把一半以上的日均值放在载体上：BDG2 上每周的平均权重为 0.48–0.56，其余为 0.53–0.76（没有证据时为 ½）。
  - ankyra-lite 在全部九个人群上的逐时误差都好于固定分工（+2.7% 到 +8.4%，全部明确），在五个人群上的月电量明确更好（剑桥 +20.2%、HEEW +16.3%、EWELD +25.4%、奥斯陆 +20.7%、德拉门 +12.5%）。固定分工也没有载体的水平候选和日内锚定，所以这部分收益不只来自交接（`ablation.csv`，`fixed division ...`）。
  - **哪一部分起作用。** 以固定的 ½ 权重混入载体的日均值，就已经得到按周权重相对不交接的大部分收益：在这一收益明确的八个人群上（+1.3% 到 +6.2%），固定 ½ 权重按对数计给出其中的 44%（BDG2）到 99%（CINELDI）。根据单位自己的伪预测逐单位估计权重，相对固定 ½ 增加 −0.1% 到 +1.2%（BDG2 +0.9%、德拉门 +1.0% 明确；从未明确更差）；每周分开的权重相对每个窗口一个权重没有增益（−0.5% 到 −0.1%，都不明确）。带来收益的是把载体的日均值按单位自己的误差混入历史侧；按周的权重描述的是平衡怎样随提前期变化。
  - **按预测周**对单独的载体，ankyra-lite（含接续层）在 BDG2 的第 1、2、4 周，GoiEner 非住户和 HEEW 的第 1 周，以及 CINELDI 的第 4 周明确更好，在 GoiEner 住户的第 2、3 周明确更差（−2.9% 和 −4.5%），其余在 −3.0% 到 +3.7% 之间，都不明确。
  - **按预测周对 TimesFM**（同一文件），ankyra-lite 在 HEEW（−11.5%）和 GoiEner 非住户（−7.8%）的第 1 周、BDG2 的第 2 周（−13.1%）以及住户的每一周（−12.2% 到 −19.2%）明确更差，在剑桥、奥斯陆、德拉门和 CINELDI 的后几周明确更好。BDG2 第 1、2 周的点估计（−81.2%，不明确；−13.1%）反映的是少数几块电表，不是整个人群：第 1 周有三块电表（Lamb_office_Vasiliki、Lamb_assembly_Bertie、Lamb_office_Velma）整周读数是同一个很小的非零值，TimesFM 几乎原样重复这个值，所以这几块电表的误差比主导了单位均值；第 2 周另有两块读数恒定的电表起同样的作用（[细节](results/README.md#handover-within-day-trust-and-combinations)）。

![一个剑桥测试窗口](figures/fig6_example_window.png)

*图 6. 一个剑桥测试窗口（建筑 b37，2013 年 12 月）：实测负荷、接续层之前（F）和之后（F′，交付）的 ankyra-lite、单独的载体和零样本 TimesFM 2.5，另有头三天的放大面板，以及日均值和按预测周的误差。这些预测都没有预见到圣诞周的低负荷。一个窗口只用来说明机制，不是证据。数据：剑桥大学校产档案（CC BY 4.0）；`results/example_window_cambridge.csv`。*

## 接续层

载体根据记录的日汇总和月汇总预测这个月；它没有一条从最后观测到的几小时接着往下走的通路，而锚定层的历史侧又会把预测拉回单位的过去。在开发窗口上，锚定预测在第一小时的对数误差几乎与最后观测的几小时离预测自身曲线有多远一比一地上升，这种依赖大约在一天（快的部分）和一周（慢的部分）内消失。接续层（`ankyra_lite.continuation`，由 `forecast` 在状态规则和读出之后施加）就校正这一点，而且放在锚定层之后，这样校正不会被稀释，锚定层也保持逐字节不变。

- **两个异常。** x_f 比较最后 3 个观测小时的均值与预测在参照日 d*（第 0–6 个预测日中，第一个与最后观测日同属一个工作日类别的那天）相同钟点的值；x_d 比较最后 24 个观测小时与预测的参照日。两者都是带一个小偏移（单位最近四周平均负荷的 1%）的对数比，截断到 ±1；其中有一个小时未观测时置为 0。
- **一个校正。** F′(h) = F(h) e^(Δ(h))，Δ(h) = 0.479 x_f e^(−h/12) + 0.330 x_d e^(−h/168)：快项在一天内消退，慢项大约在一周内消退。
- **有界。** 每个小时被乘以介于 e^(−E(h)) 和 e^(E(h)) 之间的因子，E(h) = 0.479 e^(−h/12) + 0.330 e^(−h/168)：第一小时在 ×0.445 到 ×2.246 之间，从第 8 天起在 ×0.885 到 ×1.130 之间，最后一小时在 ±0.4% 以内（O2）。非负的预测保持非负，恰好为零的小时保持为零（O1）。接续的浮点承诺限于接受的有限输出：保留非负性并恢复输入的正零和负零；非零值可能下溢为零，自定义常数导致非状态修正或最终输出非有限时抛出 `ValueError`。实数正因子界不保证任意常数下浮点结果严格为正。
- **因果、可开关。** 它只读预测本身、起报点之前 672 小时的负荷，以及最后观测日和第一个预测周的日类型（O3）；状态窗口绕过它（O4）；`continuation=False` 关闭它。
- **常数。** 在六个人群较早的开发窗口上合并一次拟合，所有人群共用（包括没有开发窗口的三个人群），外部检验也用同一组；从不在测试窗口上重新拟合。用 `continuation.DIGEST` 标识。
- **读出。** 电量、水平和峰值在它之前读出（读出规则 A）；对固定的载体，接续层一项都不改变，交付曲线的均值则不等于电量读数（[读出](#读出)）。
- **两条不改变任何研究数字的发布规则。** 第 0–6 个预测日中没有一天与最后观测日同属一个工作日类别时，参照日取 d* = 6，这是接续层设计中的回退（研究的开发代码用 d* = 0；只要存在同类别的一天两者就一致，每个开发窗口都是如此）；在测试窗口上，这个回退只用在 33 个 EWELD 窗口上，它们的第一个预测周里没有与最后观测日同类别的日子（`continuation_record.csv`，`no_match_reference_day`）。最后 3 个或 24 个小时的均值加偏移不大于零（负负荷）时，把该异常置为 0 并标记窗口，而不是报错；没有测试窗口遇到这种情况（`nonpositive_fast_span`、`nonpositive_day_span` 均为 0）。

在测试窗口上，同一载体上开启对关闭接续层（九个人群 1,891 个单位合并；`continuation_layer.csv`，`all nine populations` 行）：整月 +0.46% [+0.33, +0.70]，第 0 小时 +28.0%，第 0–2 小时 +23.4%，第 1 天 +11.1%，第 2–7 天 +1.9%，第 8–31 天 +0.17%，全部明确；交付轨迹的电量 +1.9%，不明确。在选出其形式的开发窗口上（复用评估），同一对比为整月 +0.48%、第 1 天 +9.49%（`recipe_history.csv`，`Rw6_c vs Rw6` 的开发行）。细节、形式的选择和全部保证见 [docs/METHOD.md](docs/METHOD.md#the-continuation-layer) 和 [theory/PROOFS.md](theory/PROOFS.md#exact-properties-of-the-continuation-layer)。

## 训练配方 Rw

发布的配方 Rw 分两个阶段，对每个人群完全相同（训练、选出停止的轮数，然后按原抽样密度加入验证时段的一部分窗口并对窗口加权）。第 1 阶段就是配方 R（0.1 版）的第 1 阶段；第 2 阶段与 R 的唯一区别是损失权重。每个常数都是 `ankyra_lite.carrier.train.TrainConfig` 的字段；`recipe(name, init_checkpoint)` 构造各个命名的配置（[docs/METHOD.md](docs/METHOD.md#training)）。

**第 1 阶段：从零训练，带早停**（`TrainConfig()` = `recipe("from_scratch")`）。

1. **窗口。** 一个窗口是（单位，起报点 o）：上下文 [o − 1344, o)，标签 [o, o + 744)。只用训练截点 C′ 之前的数据（o + 744 ≤ C′；C′ 及之后的负荷和气温在计算任何特征之前就置为缺测）。训练起报点按天取；上下文小时和标签小时都至少有 75% 被观测到，窗口才合格。
2. **划分。** 验证段是截点之前的最后 93 天，从 v0 = max(2,088, C′ − 2,232) 开始。训练窗口在 v0 或之前结束；验证窗口在 v0 或之后开始、不晚于 C′ 结束；早停用的验证集是目标人群自己按周取起报点的验证窗口。
3. **上限。** 每个人群最多 16,000 个训练窗口和 2,000 个验证窗口，按单位均衡选取。
4. **合并训练集。** 每个研究数据源（九个人群和苏州工业园区）都贡献其标签在目标人群验证段开始时刻（绝对时间）或之前结束的窗口；对目标人群来说就是它的训练窗口，对其他来源则是取自它们自己训练集或验证段的“借来的”窗口（从不取自它们的测试期）。每个来源的窗口数与它按周网格的窗口数相同。
5. **抽样。** 每轮有放回地抽 16,000 个窗口：一半来自目标人群（每个单位等权），一半来自借出窗口的来源（先均匀选一个来源，再每个单位等权）。
6. **损失。** 每个窗口取 log(1 + MSE + 0.1 · KL)，在按尺度归一化后的观测小时上计算（MSE 加上一个广义 Kullback–Leibler 项），再对窗口平均。
7. **优化。** AdamW，学习率 3 × 10⁻⁴，权重衰减 10⁻⁴，批大小 64；梯度范数裁剪到 1.0，基础参数和复制指针参数分开裁剪。
8. **早停。** 验证准则：对每个单位取 log(1 + 该单位平方误差均值除以尺度的平方)，再在单位间平均。保留最好的一轮；最多 30 轮，patience 为 5。

**第 2 阶段：归还验证段，并加逆噪声损失权重**（`recipe("span_return_weighted", <stage-1 checkpoint>)`，也可写 `recipe("Rw", ...)`）。

9. **核对。** 重新计算起始权重的第 1 阶段准则，必须复现记录的值。
10. **归还的验证段。** 把目标人群自己在 [v0, C′) 段内按天取起报点、标签至少在截点之前 24 小时结束的窗口加入它的训练窗口，按该人群自己的抽样密度。
11. **损失权重。** 每个训练窗口的权重为 max(σ̂², f)^(−½)：σ̂² 是该窗口起报点之前最后 21 天自己的周朴素相对平方误差（每个小时与一周前同一小时配对，两者都观测到；不足 168 对时无定义，取中位权重），f 是训练池上 σ̂² 的第 25 百分位，两者都按抽样器的抽取概率加权；然后缩放权重，使每个来源人群在这些概率下的平均权重为 1。一个批的损失是 Σ w ℓ / Σ w，ℓ 为第 6 步的逐窗口项。权重只读每个窗口起报点之前的上下文，不用标签，不用随机数；权重相等时这一步逐位等于配方 R（C11–C14）。在发布的训练池上，最大权重是按抽取概率加权的中位权重的 1.6 倍（剑桥）到 6.2 倍（GoiEner 非住户），抽取的有效样本量是不加权时的 0.69–0.82（`checkpoints/MANIFEST.json` 的 `kish_ess_ratio_draws` 及其指向的权重记录）。
12. **训练。** 新建一个 AdamW，借来的窗口和时间限制不变，正好训练 8 轮，保留最后的权重；这一阶段不做任何选择。
13. **种子与集成。** 种子 0 到 5：种子 0–2 接着 0.1 版背后的第 1 阶段运行继续训练，种子 3–5 是为本版本用同一第 1 阶段配方新训练的。载体预测是六个种子的预测在每个发布时刻的简单平均。

```python
from ankyra_lite.carrier.train import TrainConfig, recipe, prepare_source, train_ensemble

stage1 = TrainConfig()                                                    # from scratch, early stopping
sources = {n: prepare_source(n, data[n], cut[n], stage1, seed_index=i) for i, n in enumerate(names)}
first = train_ensemble(target, sources, stage1, seeds=range(6), checkpoint_dir=f"runs/{target}/stage1")

stage2 = recipe("span_return_weighted", "runs/{target}/stage1/seed{seed}.pt")   # recipe Rw: span returned, weighted
sources = {n: prepare_source(n, data[n], cut[n], stage2, seed_index=i, with_validation=(n == target))
           for i, n in enumerate(names)}
final = train_ensemble(target, sources, stage2, seeds=range(6), checkpoint_dir=f"runs/{target}/rw")
```

`data[name]` 包含 `load`（单位 × 小时，kW，缺测为 NaN）、`temp`、`day_types`、`groups` 和 `start_iso`。示例写入 `runs/`（被 git 忽略），从不写入 `checkpoints/`。

**训练记录显示了什么**（`results/training_curves.csv`；种子 0–5 的第 1 阶段最佳轮数）：BDG2 12、9、27、16、24、29；GoiEner 非住户 15、3、16、5、19、9；住户 15、16、5、6、10、7；剑桥 16、12、16、29、21、29；HEEW 27、20、23、27、27、30；EWELD 10、9、7、6、14、10；奥斯陆 14、9、30、30、15、12；德拉门 21、4、23、3、2、13；CINELDI 14、6、6、6、5、13。有三次运行（奥斯陆种子 2 和 3、HEEW 种子 5）选中了第 30 轮，也就是最后一轮；每个第 1 阶段检查点都在第 2 阶段的起始核对中得到复现。第 2 阶段的准则是在已经回到训练集的窗口上计算的，所以只是描述，不是留出数据上的分数。

**包能复现什么。** 在小规模运行上，本包逐行复现研究的训练（奥斯陆和 BDG2 的窗口、划分、上限、训练池和归还的验证段），并以零参数差复现每个阶段的一步优化（`tests/test_train.py`，需要研究的数据）。配方 Rw 的权重与研究的规则逐位相同，权重相等时加权的一步等于配方 R 的一步（`tests/test_train_rw.py`）。发布的检查点来自研究的 GPU 运行，不是本包的完整规模运行；GPU 训练不能逐位重复。

**包里的其他配方。** `recipe("span_return", ...)` 就是配方 R，即 0.1 版发布的配方；它的 27 个检查点随仓库发布，R 和 Rw 只差 `noise_weighting`。`recipe("plain_continuation", ...)`（不归还验证段的第 2 阶段）和 `recipe("proportional_validation", ...)` 是选定 R 的那项研究里的另外两个方案；保留它们是为了复现（[配方历史](#配方历史r-是怎样选定的以及它的负面结果)）。

## 数据划分

**规则，对每个人群相同。**

- **测试起点 C**：该人群评估起报点中位数当天或之后的第一个月初。它是 ANKYRA 2.2.1 发布版中训练型基线的训练截止时间，所以每个基线在每个测试窗口上都有预测。测试集是该人群从 C 起的月初起报点（发布版的 `late` 窗口）。
- **训练截点 C′ = C − 186 天**（4,464 小时）。C′ 及之后的任何数据都不用于拟合载体的权重。
- **验证段**：C′ 之前的 93 天。它决定第 1 阶段何时停止，然后在第 2 阶段归还给训练集，但标签在 C′ 之前 24 小时以内结束的窗口除外。
- **训练集**：该人群自己在验证段之前结束的窗口，加上其他人群在同一时刻之前结束的借来窗口。

```
 first training origin ...... training set ...... | validation span (93 days) | C' | 186-day gap | C: test set ...
 stage 1:  trained on ............................  | early stopping only       |
 stage 2:  trained on ................................................ returned, up to 24 h before C'
```

用来更新权重的最新标签，至少在第一个测试起报点之前 186 天又 24 小时结束；预测输入始终使用截至起报点的负荷。

**日期。** 真实的日历日期（CINELDI 用修正后的数据仓库），取自 [`results/datasets.csv`](results/README.md#populations-and-splits)：

| 人群 | 层级 | 第一个训练起报点 | 验证段起点 | 训练截点 C′ | 测试起点 C（第一个测试起报点） | 最后一个测试起报点 |
|---|---|---|---|---|---|---|
| BDG2 2017 | 测试 | 2016-02-26 | 2016-10-26 | 2017-01-27 | 2017-08-01 | 2017-12-01 |
| GoiEner 非住户 | 测试 | 2018-02-26 | 2018-07-26 | 2018-10-27 | 2019-05-01 | 2019-12-01 |
| GoiEner 住户 | 测试 | 2018-02-26 | 2018-09-25 | 2018-12-27 | 2019-07-01 | 2019-12-01 |
| 剑桥 | 测试 | 2000-02-26 | 2012-09-25 | 2012-12-27 | 2013-07-01 | 2022-12-01 |
| HEEW（亚利桑那） | 测试 | 2014-02-26 | 2018-02-25 | 2018-05-29 | 2018-12-01 | 2022-12-01 |
| EWELD | 测试 | 2016-07-15 | 2018-10-26 | 2019-01-27 | 2019-08-01 | 2022-07-01 |
| 奥斯陆 | 预览 | 2012-11-12 | 2018-12-26 | 2019-03-29 | 2019-10-01 | 2024-06-01 |
| 德拉门 | 预览 | 2018-02-26 | 2019-11-27 | 2020-02-28 | 2020-09-01 | 2022-02-01 |
| CINELDI | 预览 | 2019-04-12 | 2020-06-26 | 2020-09-27 | 2021-04-01 | 2022-02-01 |

**数量**（同一文件）：

| 人群 | 开发窗口 | 数据仓库 | 测试窗口 / 单位 | 测试起报月份 / 跨越月份 | C′ 之前的训练起报天数 | 验证段占自身窗口的比例 | 自身训练窗口（第 1 阶段） | 归还的验证窗口（第 2 阶段） |
|---|---|---|---:|---:|---:|---:|---:|---:|
| BDG2 2017 | 无 | 原始 | 474 / 142 | 5 / 5 | 336 | 23.7% | 16,000 | 5,042 |
| GoiEner 非住户 | 无 | 原始 | 890 / 481 | 8 / 8 | 243 | 38.3% | 8,272 | 5,381 |
| GoiEner 住户 | 无 | 原始 | 696 / 696 | 6 / 6 | 304 | 28.2% | 16,000 | 6,292 |
| 剑桥 | 有 | 原始 | 730 / 108 | 107 / 114 | 4,688 | 1.9% | 16,000 | 312 |
| HEEW（亚利桑那） | 有 | 原始 | 658 / 138 | 39 / 49 | 1,553 | 4.5% | 16,000 | 749 |
| EWELD | 有 | 原始 | 523 / 197 | 36 / 36 | 926 | 11.0% | 16,000 | 1,969 |
| 奥斯陆 | 有 | 原始 | 581 / 41 | 52 / 57 | 2,328 | 3.6% | 8,158 | 304 |
| 德拉门 | 有 | 修正 | 689 / 43 | 18 / 18 | 732 | 9.2% | 3,898 | 396 |
| CINELDI | 有 | 修正 | 475 / 45 | 11 / 11 | 534 | 13.7% | 2,447 | 389 |

- **开发窗口**：研究在开发截点（训练截点之前 279 天）与训练截点之间的窗口上检查载体的设计选择。三个短记录人群的记录太短，没有这样的窗口，所以在读它们的测试窗口之前，没有任何设计选择在它们上面检查过。
- **数据仓库**：“修正”（文件中为 `corrected`）标出用修正后的 CINELDI 数据仓库训练的载体。最初建成的数据仓库把 CINELDI 的第一个小时标为 2019 年 1 月 1 日，而数据实际从 2019 年 3 月 1 日开始，所以其日期早了 59 天；修正后的数据仓库在前面补上缺失的 1,416 小时。CINELDI 在修正后的数据仓库上训练、锚定和评分；德拉门自己的数据不受影响，但它的合并训练集里有 CINELDI 窗口，所以它的载体也用修正后的数据仓库从零重新训练（[docs/DATA.md](docs/DATA.md#9-cineldi-industrial-customers-norway)）。
- **同一规则，后果不同。** BDG2 2017 和两个 GoiEner 人群在截点之前只有 243–336 天的训练起报点；93 天的验证段占了它们自身窗口的 23.7–38.3%（其他六个人群为 1.9–13.7%），它们的测试集只覆盖同一年的 5–8 个月（应理解为那一年的下半年）。在这三个人群上，验证段里有训练集原本缺少的季节，这正是配方需要第二阶段的原因（[配方历史](#配方历史r-是怎样选定的以及它的负面结果)）。
- **借来的窗口。** 每个训练池从哪些人群借窗口，列在 [docs/DATA.md](docs/DATA.md#windows-borrowed-from-other-populations)；剑桥不借任何窗口。

![数据划分](figures/fig1b_data_splits.png)

*图 1b. 每个人群的训练集、验证段、训练截点、186 天间隔和测试集，按日历时间排列（a），以及按训练截点对齐（b）（`results/datasets.csv`）。*

## 本页用到的术语

- **单位**：一条计量序列：一栋楼、一块电表、一个用电点或一户住户。
- **起报点**：发出预测的那个小时；预测覆盖其后的 744 小时（31 天）。
- **窗口**：一个单位的一个起报点，连同它的 1,344 小时**上下文**和 744 小时的目标。
- **载体**：小型载体网络，锚定层锚定的就是它的预测。**单独的载体**指它在起报点上的六种子预测，不经过锚定层、状态规则和接续层。
- **接续层**：锚定层和状态规则之后的起报接续层（[小节](#接续层)）。**读出规则 A**：电量、水平和峰值在接续层之前读出。**开发窗口**：训练截点之前、用来检查设计的窗口（[数据划分](#数据划分)）。
- **伪起报点**：单位记录内部更早的起报点 o − 744k（k = 1 … 6），它的那个月在 o 时已经观测到；载体在那里的预测称为**伪预测**。
- **块**：744 小时预测的三个精确部分：**水平**（其均值）、**日路径**（31 个日均值减去水平）和**日内形状**（每个小时减去当天的均值）。它们的平方误差可以直接相加。
- **交接**：日均值中取自载体而不是历史侧的份额，每个预测周一个权重（第 1–7、8–14、15–21、22–31 天）。**信任**：日内块中单位相似日形状的权重（0 到 ½）。
- **状态窗口**：锚定层的停机规则或微负荷规则被触发的窗口；ankyra-lite 在那里交付周季节性朴素预测。
- **训练截点 C′、测试起点 C**：见[数据划分](#数据划分)。**测试窗口**：从 C 起的月初起报点，结果文件中记为 `late`。
- **配方 Rw**（0.2.0 发布：验证段归还加逆噪声损失权重）、**配方 R**（验证段归还，0.1 发布）、**F**（同样 8 轮的继续训练，但不归还验证段）、**V**（按比例划分的验证集，带早停）、**P3s**（只有第 1 阶段）：各配方以及 0.1 版配方研究的各个方案。在 `recipe_history.csv` 的 0.2.0 行中，配置按配方和种子命名（R3：配方 R、种子 0–2；R6、Rw3、Rw6：配方 R 或 Rw，三个或六个种子），`_c` 表示开启接续层；0.2.0 即 Rw6_c，0.1.0 即 R3。
- **短记录人群**：BDG2 2017、GoiEner 非住户和 GoiEner 住户。**切换层**：它们的测试窗口中，目标月在归还验证段之前没有同季训练标签、归还之后有了的那些窗口。
- **纯历史配置**：没有载体的锚定层（`foundation=None`）。**无载体对照**：把纯历史配置自己的预测放在载体的位置上，调用未改动的锚定层。
- **固定分工**：水平和日路径只用历史侧（历史估计器的水平和日路径，不含载体的水平候选，也不做交接），日内块用载体的，再投影到非负负荷。交接权重在伪起报点上就是对它估计的（`fixed_division_daily_means_kw`）。
- **对比 (i)**：锚定后对单独的载体（单位的历史对载体有没有帮助？）。**对比 (ii)**：锚定后对纯历史配置（含载体的配置是否好于只用历史？）。
- **明确**：一个对比的 95% 区间不含零。**不明确**不等于一样好。
- **层级**：该人群在 ANKYRA 2.2.1 发布版中的角色：测试（`test`，六个人群）或预览（`preview`，奥斯陆、德拉门、CINELDI）。在这里，每个人群都有自己的训练截点和测试集，评分方式相同；层级只用于给表格分组。

## 起报点之后无信息

起报点 o 上的预测只是下列各项的函数，此外别无其他（[docs/METHOD.md](docs/METHOD.md#information-at-the-origin)）：

- o 之前各小时的负荷和气温（载体最多读最近 366 天和其前的 24 个 744 小时块；锚定层读起报点之前的整个记录）；
- 31 个预测日的日类型，按起报时已知处理；
- 单位的类别、该人群固定的温度异常尺度（在训练截点之前的日子上拟合）和所在地区的夏令时规则；
- 载体的权重，拟合所用的标签至少在第一个测试起报点之前 186 天又 24 小时结束；
- 载体在 o 和每个伪起报点 q 上的预测，每个只用 q 之前的数据构造；
- 对接续层而言：锚定预测本身、o 之前 672 小时的负荷，以及最后观测日和第一个预测周的日类型（O3）。

**由测试检验。** `tests/test_carrier_unit.py`（不读发布时刻及之后的数据、伪起报点规则）、锚定层自己的 `tests/anchor/test_no_future_information.py`、`tests/test_forecast_v2.py`（含接续层时起报点之后无信息），以及 [theory/](theory/README.zh-CN.md) 中的性质 C4、C10、C14 和 O3。它们证明的是实现层面的输入隔离，并不能说明输入本身不含更晚的信息：有两个日历在部分测试起报点之后才公布假日（[局限](#局限)），数据提供方的补缺测和气象档案的事后修订都在本包的上游。

## 基线

ankyra-lite 与 **21 个基线**（共 22 个预测器）比较：ANKYRA 2.2.1 发布版的 20 个外部基线（按它们保存的预测评分），以及 LSTM-T-KAN-M。ANKYRA 本身及其各版本不在其中。这 21 个在每张表和每个名次里都是普通的基线（[docs/EVALUATION.md](docs/EVALUATION.md#baselines)）。

| 信息 | 预测器（结果文件中的模型 id） |
|---|---|
| 拿到发布版的共用协变量集（过去 11 维、未来 10 维、静态 6 维；不含实测的未来天气）；各自使用其结构能接收的部分 | 线性上下文模型（TiDE 实现）（`TiDE`，训练型）、iTransformer-X（训练型）、GBT-T（在 21 个派生特征上训练的梯度提升）、Chronos-2-X（零样本）、TimesFM-X（零样本，协变量经线性回归进入） |
| 负荷和日历，训练型 | LSTM-T-KAN-M |
| 只用负荷，零样本基础模型 | TimesFM 2.5（`TimesFM`）、Chronos-2 |
| 只用负荷，每个人群训练 | DLinear、PatchTST、iTransformer、LSTM |
| 只用负荷，统计方法 | Holt–Winters、MSTL |
| 日历和气候态温度 | 每个起报点重新拟合的逐单位岭回归（`RIDGE-L`） |
| 简单方法 | 上月、去年同期和 4 周剖面；周期为一天和一周的季节性朴素；零样本梯度提升（`GBT`） |

- **发布版的训练型基线**每个人群训练一次，用 C 之前的全部目标（比 ankyra-lite 的载体新至多半年），最多 15 轮，patience 为 3，三个种子平均。在 CINELDI 上，用的是在修正后的数据仓库上重新评分的基线：起始日期起作用的地方重新预测，读日历的训练型基线（GBT-T、线性上下文模型、iTransformer-X）用相同的代码和种子重新训练。
- **线性上下文模型（TiDE 实现）。** 发布版的 TiDE 实现在其时间解码器末尾对单个输出做层归一化，结果是常数。因此它的预测是 1,344 小时负荷上下文的线性映射加一个常数，协变量不起作用。它保留模型 id `TiDE`，标为 `Linear-context (TiDE implementation)`，并且因为确实收到了共用协变量集，仍列在拿到这些协变量的基线中。
- **GBT-T** 是发布版的修正版本；在 EWELD 上，它的逐窗口缩放仍在很多窗口上过冲。

**LSTM-T-KAN-M。** Shen、Yang、Song 和 Liang（2026，doi:10.1016/j.scsadv.2026.100065）的多变量 LSTM-T-KAN：**模型和训练代码逐行复制自作者的公开仓库（提交 `ce24938`，MIT 许可），外加作者的 exp2 特征列表**（16 个通道，来自过去的负荷和过去的日历；不含气温）。它**不是对论文的严格复现**。十处偏离如下：

| # | 论文 / 作者的代码 | 这里的 LSTM-T-KAN-M |
|---|---|---|
| 1 | 预测长度 1–30 天，每个长度一个模型，最长 720 步 | 一个模型，覆盖全部 744 步 |
| 2 | 每栋建筑一个模型 | 每个人群一个模型，跨单位训练 |
| 3 | 在前 70% 上拟合的 Min-Max 缩放 | 用截止时间之前的小时做逐单位 z 标准化 |
| 4 | 70 / 10 / 20 的时间顺序划分，步长为 1 的窗口 | 发布版的截止时间，训练起报点的最后 20% 用于选择，起报点步长 24 小时，最多 16,000 个训练对 |
| 5 | 对序列做 3 点平滑（论文） | 不做 |
| 6 | 最多 100 轮，早停 patience 10 | 最多 30 轮，patience 8（协议附录 A，在任何预测评分之前定下；协议最初定的是 50 和 10） |
| 7 | 学习率 10⁻⁴（配置文件）或 10⁻³（论文） | 在选择对上从 {10⁻⁴, 3 × 10⁻⁴, 10⁻³} 中选 |
| 8 | 单次运行，种子 42 | 种子 0、1 和 2 的平均 |
| 9 | 预先构造的特征列，每个预测长度单独做特征选择 | 一份固定的 16 个 exp2 通道列表，按本研究的定义构造；不含天气 |
| 10 | 验证集划分内的验证窗口 | 选择对 = 训练起报点的最后 20% |

它的预算（30 轮、patience 8、plateau 学习率调度、恢复最佳权重）比发布版其他训练型基线的预算大。在 CINELDI 上，它与发布版读日历的基线一样，用相同的代码、协议、学习率网格和种子在修正后的数据仓库上重新训练；在此之前，同样的代码在原始数据仓库上逐位复现了它保存的预测，重新训练也通过了检查（`results/compare_baselines.json`，`populations` → `cineldi` → `extra_baselines`）。细节见 [docs/EVALUATION.md](docs/EVALUATION.md#lstm-t-kan-m)。

**双方各能知道什么。** ankyra-lite 读起报点之前的负荷和气温、预测月的日历和类别。拿到协变量的基线只以气候态期望的形式得到未来气温。零样本基础模型没有本地训练；它们的预训练语料是这里任何预测器都控制不了的唯一暴露（[局限](#局限)）。

## 评估

本节的每个数字都是 [`results/`](results/README.md) 中某个文件的一行，由 2026 年 10 月 21 日用 ankyra-lite 0.2.0（配方 Rw、六个种子、接续层；0.1.1 的补充分析包含在同一次运行中）完成的全量重新生成写出；逐时、名次、常规指标、按日和分块表按交付轨迹（接续层之后）评分，电量表和峰值读出按接续层之前读 ankyra-lite（读出规则 A）；每一部分都注明了文件，[`results/SUMMARY.md`](results/SUMMARY.md) 是同一次运行的可读摘要。图由 [`figures/make_figures.py`](figures/README.md) 从同样的文件绘制。完整表格和评估协议见 [docs/EVALUATION.md](docs/EVALUATION.md)。

![ankyra-lite 与四种零样本基础模型在逐时误差、月电量和月峰值上的比较](figures/fig0_advantage.png)

*图 0. ankyra-lite 0.2.0 对四种零样本基础模型变体（TimesFM、TimesFM-X、Chronos-2、Chronos-2-X），在每个人群的测试窗口上，按发布版的主估计量：交付轨迹的逐时误差（a）、在接续层之前读出的月电量误差（b），以及月峰值，对比把同一读出用在各变体轨迹上的结果（c）。改进附 95% 单位与月份区间；实心 = 明确更好，空心 = 不明确，红色 = 明确更差。*

在月电量和月峰值上，与四种变体的对比（每种读出 36 项）没有一项明确不利于 ankyra-lite：电量有 19 项明确更好（+17.7% 到 +38.6%；剑桥、GoiEner 住户和奥斯陆上对全部四种变体，德拉门上三种，GoiEner 非住户和 CINELDI 上各两种），峰值有 9 项（+9.8% 到 +16.8%；住户上全部四种，奥斯陆上三种，剑桥上两种），两种读出各有 36 项中的 31 项点估计偏向 ankyra-lite。逐时误差在剑桥（+10.0% 到 +16.8%）、奥斯陆（+14.6% 到 +25.0%）和 CINELDI（+6.7% 到 +10.5%）上明确好于全部四种变体，在德拉门上明确好于其中三种（+16.6% 到 +20.9%；Chronos-2-X +7.3%，不明确）；在 BDG2、HEEW、EWELD 和 GoiEner 非住户上两个方向都不明确（−4.0% 到 +5.4%）；在 GoiEner 住户上对 TimesFM 明确更差（−4.6%；TimesFM-X −0.9%、Chronos-2 −2.8%、Chronos-2-X −2.9% 不明确）（`benchmark_pairwise.csv`、`energy_error.csv`、`peak_readout.csv`）。逐时误差逐日怎样变化、按预测日的面板为什么读起来不同，见[按预测日](#按预测日)；接续层在头几天的作用见[按小时看接续层](#按小时看接续层)。

### 先读这一节

- **0.2.0 再读一次。** 0.2.0 的配置（配方、种子、接续层及其常数、读出规则）在本版本读测试窗口之前只根据开发窗口冻结：开发检查（在同样选出接续层形式的开发窗口上的复用评估）给出接续层整月 +0.48%、第 1 天 +9.49%，完整的 0.2.0 配置对 0.1 版整月 +0.97%、第 1 天 +10.17%；在那里损失权重使峰值读出明确变差（Rw6 对 R6 −0.92%），仍在读数前写好的保护线以内（`recipe_history.csv`，开发行）。随后按读数前写好的规则把测试窗口读了一次，没有触发任何放弃某项改动的规则；这些窗口此前已被读过多次（见下），所以这次读数确定每项改动的方向，不是确认的效果。
- **是复用评估，不是首次检验。** 这些测试窗口是 ANKYRA 2.2.1 发布版面板的截点后（late）窗口，在那里为发布版的基线评过分。研究之后又在这些窗口上给小载体的早期版本评过分，做过一次只看逐时误差的与基线的比较，并在选定配方 R 的研究中再读过一次。复制指针的方向来自对载体在这些窗口上误差的分析，季节性朴素状态规则是在对它们的一次早先评分之后写的。这些表格描述的是发布的预测器在这些窗口上的表现；它们不能估计它在没有参与设计的窗口上会表现如何。
- **配方 R 是在这些窗口上选定的**，是在对四个配方方案做了一次方向性读数之后（[配方历史](#配方历史r-是怎样选定的以及它的负面结果)）；配方 Rw 在它之上加了损失权重。
- **没有为这次比较训练或调整任何基线。** 已保存的预测用发布版的评分函数打分；在此之前先核对了这些函数能复现发布版公布的表格（[可复现性](#可复现性)）。
- **状态窗口。** ankyra-lite 在 33 个 BDG2、13 个 GoiEner 非住户、35 个住户测试窗口，以及 523 个 EWELD 测试窗口中的 245 个上交付周季节性朴素预测（`results/compare_baselines.json`）；在这些窗口上，它的预测与 `SN-week` 基线相同，接续层被绕过。在每个窗口都读数接近零的三块 BDG2 电表上（10 个窗口），它的误差恰好为零，所以这些电表不进入单位等权的对比，但计入平均名次（[细节](docs/EVALUATION.md#state-windows-and-the-three-near-zero-bdg2-meters)）。
- 发布版的**苏州园区**没有 ankyra-lite 预测，不在任何表中；它只提供借来的训练窗口。

### 怎样读这些数字

- **改进** = 100[1 − exp(r)]，r 是单位等权的对数 RMS 比：对每个单位取它在各测试窗口上误差的 RMS，r 是 log(ankyra-lite 的 RMS / 基线的 RMS) 在各单位上的均值。正数表示 ankyra-lite 更好。
- **区间**：95%，来自 2,000 次重抽单位和目标月份的自助重复。`*` 表示明确偏向 ankyra-lite 的对比，`(−)` 表示明确不利于它的对比。
- **逐时**：交付轨迹（接续层之后）744 小时上的误差。**电量**：月电量误差，即轨迹窗口均值误差的 744 倍；ankyra-lite 用接续层之前的轨迹（读出规则 A），每个基线用它自己的轨迹。**峰值**：ankyra-lite 的峰值读出（作用在接续层之前的轨迹上），对比把同一读出用在每个基线轨迹上的结果（发布版的规则），以及对比每个基线自己轨迹的最大值。
- **名次**：按单位平均名次在 22 个预测器中的位置（在每个单位内按 RMSE 给预测器排名）。
- 报告几种汇总，是因为合并汇总和单位等权汇总可能符号相反（[theory/](theory/README.zh-CN.md) 中的 P19），而且绝对误差类指标偏向中位数型的预测，平方误差类指标偏向均值型的预测。

### 0.2.0 改了什么

相对 0.1 版（配方 R、三个种子、没有接续层）的每项改动，九个人群等权，逐时改进（`recipe_history.csv` 中 `reading` 为 0.2.0 测试读数的行；0.2.0 对 0.1 和加权两行把单位与月份自助法与两组三种子之间的差异合在一起，其他行只有单位与月份区间；除标“不明确”外全部明确）：

| 对比 | 整窗 | 第 0–2 小时 | 第 1 天 | 第 2–7 天 | 第 8–31 天 |
|---|---|---|---|---|---|
| **0.2.0 对 0.1**（`primary: 0.2.0 vs 0.1.0`） | **+1.15% [+0.60, +1.71]** | +28.46% | +13.97% | +2.67% | +0.67% |
| 六个种子对三个（配方 R、无接续层；`R6 vs R3`） | +0.41% [+0.15, +0.70] | +0.84% 不明确 | +0.57% | +0.75% | +0.32% |
| 损失权重：配方 Rw 对 R（六个种子、无接续层；`Rw6 vs R6`） | +0.31% [+0.06, +0.56] | +0.76% 不明确 | +0.81% | +0.39% 不明确 | +0.30% |
| 接续层开对关（同一 Rw6 载体） | +0.43% [+0.14, +0.78] | +27.30% | +12.77% | +1.55% | +0.04% 不明确 |

- **按人群**，0.2.0 对 0.1 的整月改进：BDG2 +1.71%、GoiEner 非住户 +1.91%、GoiEner 住户 +0.94%（明确，[+0.37, +1.51]）、剑桥 +1.42%、HEEW +1.44%、EWELD +0.19%、奥斯陆 −0.50%、德拉门 +2.42%、CINELDI +0.81%（`by_population_pct`；`component_contributions.csv` 给出各自的区间）。三个没有开发窗口的人群合并：+1.52% [+0.61, +2.42]；有开发窗口的六个：+0.97% [+0.28, +1.65]。
- **对 0.1 的电量和峰值**：月电量（接续层之前的曲线）+1.03% [−1.77, +3.74]，按水平读数的电量 +1.04% [−1.77, +3.76]，峰值 −0.35% [−2.18, +1.45]；都不明确，按读数前写好的规则也没有标出可信的伤害。对固定的载体，接续层不改变其中任何一项（它们在接续层之前读出），所以这些差别来自载体。
- 各部分不可相加，三项对比在同一批窗口上读出、互不独立，种子和加权两项还带有重新训练的噪声，而接续层那一项没有。接续层对比按单位合并的版本（`continuation_layer.csv`：+0.46%）按单位而不是按人群加权。

### 总览

明确更好 / 明确更差的基线个数，共 21 个（`benchmark_pairwise.csv`、`energy_error.csv`、`peak_readout.csv`；名次来自 `benchmark_mean_unit_rank.csv`；状态窗口来自 `compare_baselines.json`）：

| 人群 | 层级 | 窗口 / 单位 | 状态窗口 | 22 个中的名次（单位平均名次） | 单位平均名次第二 | 逐时 | 电量 | 峰值（同一读出） | 峰值（对基线轨迹最大值） |
|---|---|---|---:|---|---|---|---|---|---|
| BDG2 2017 | test | 474 / 142 | 33 | 1 (4.68) | 逐单位岭回归 (7.01) | 13 / 0 | 7 / 0 | 4 / 0 | 17 / 0 |
| GoiEner 非住户 | test | 890 / 481 | 13 | 1 (5.02) | iTransformer (7.20) | 14 / 0 | 13 / 0 | 2 / 0 | 19 / 0 |
| GoiEner 住户 | test | 696 / 696 | 35 | 1 (5.34) | 逐单位岭回归 (7.06) | 17 / 1 | 16 / 0 | 11 / 0 | 21 / 0 |
| 剑桥 | test | 730 / 108 | 0 | 1 (3.04) | iTransformer-X (5.99) | 21 / 0 | 20 / 0 | 12 / 0 | 21 / 0 |
| HEEW 亚利桑那 | test | 658 / 138 | 0 | 1 (5.11) | Chronos-2-X (6.80) | 14 / 0 | 7 / 0 | 7 / 0 | 16 / 0 |
| EWELD | test | 523 / 197 | 245 | 1 (7.09) | TimesFM (8.16) | 13 / 0 | 6 / 0 | 5 / 0 | 19 / 0 |
| 奥斯陆 | preview | 581 / 41 | 0 | 1 (1.78) | GBT-T (2.63) | 20 / 0 | 19 / 0 | 17 / 0 | 17 / 0 |
| 德拉门 | preview | 689 / 43 | 0 | 1 (1.67) | Chronos-2-X (3.58) | 20 / 0 | 13 / 0 | 6 / 1 | 19 / 0 |
| CINELDI | preview | 475 / 45 | 0 | 1 (2.89) | 逐单位岭回归 (5.69) | 21 / 0 | 13 / 0 | 3 / 0 | 19 / 0 |

- **逐时。** 唯一明确的劣势在 GoiEner 住户上，对 TimesFM（−4.6%）。明确的优势从 +4.3%（逐单位岭回归，住户）到 +54.0%（奥斯陆）。不明确的例如：奥斯陆上的 GBT-T（+2.9%）、德拉门上的 Chronos-2-X（+7.3%）和 BDG2 上的 Chronos-2-X（−4.0%）。
- **去掉三块接近零的电表后的 BDG2**：明确更好的逐时对比从 13 个变为 12 个（iTransformer 不再明确），139 个单位上的单位平均名次为 4.683，仍是 22 个中的第一（`bdg2_near_zero_sensitivity.csv`）。
- **电量**在任何人群上都没有明确差于任何基线。把 ankyra-lite 接续层之前轨迹的电量换成它的水平读数，计数不变（`energy_level_readout.csv`）；两者只在非负投影起作用的地方不同（BDG2 1 个、GoiEner 非住户 138 个、住户 21 个、剑桥 10 个、EWELD 18 个、德拉门 2 个、CINELDI 27 个窗口）。
- 明确对比的计数是对几百个区间的描述性汇总，不是经过多重性校正的检验。

![测试集名次与两两对比](figures/fig2_test_ranks.png)

*图 2. 每个预测器在每个人群测试窗口上的单位平均名次（a），以及跨人群的两两对比计数（b）。*

![带区间的逐时对比](figures/fig3_test_pairwise.png)

*图 3. ankyra-lite 的单位等权逐时改进及 95% 单位与月份区间，对比拿到同样信息的基线、LSTM-T-KAN-M 和部分只用负荷的基线；其他基线见图 2 和图 8。*

![每个人群上的位置](figures/fig12_consistency.png)

*图 12. 每个预测器在每个人群上于 22 个中的位置（单位平均名次），按模型类别着色。*

### 与零样本基础模型比较

改进百分比（`benchmark_pairwise.csv`、`energy_error.csv`，以及按同一读出的 `peak_readout.csv`）：

| 人群 | 逐时对 TimesFM | 电量对 TimesFM | 峰值对 TimesFM | 逐时对 Chronos-2-X | 电量对 Chronos-2-X | 峰值对 Chronos-2-X |
|---|---|---|---|---|---|---|
| BDG2 2017 | +0.7 | +14.7 | +4.0 | −4.0 | +3.3 | −0.2 |
| GoiEner 非住户 | +4.7 | +20.2 * | +0.8 | −0.5 | +12.4 | +3.4 |
| GoiEner 住户 | −4.6 (−) | +32.8 * | +10.2 * | −2.9 | +32.8 * | +11.7 * |
| 剑桥 | +15.4 * | +24.6 * | +10.0 * | +10.0 * | +19.7 * | +4.5 |
| HEEW 亚利桑那 | +2.5 | −0.1 | −1.2 | +0.8 | −0.8 | +0.0 |
| EWELD | +1.5 | −2.8 | +2.9 | +4.2 | +16.8 | +0.6 |
| 奥斯陆 | +23.0 * | +34.9 * | +16.4 * | +14.6 * | +20.4 * | +7.1 |
| 德拉门 | +20.9 * | +29.0 * | +14.6 | +7.3 | +4.3 | −1.1 |
| CINELDI | +10.5 * | +22.4 * | +6.9 | +6.7 * | +16.7 | +7.6 |

- 在 BDG2、GoiEner 非住户、HEEW 和 EWELD 上，与四种零样本变体（TimesFM、TimesFM-X、Chronos-2、Chronos-2-X）的逐时对比没有一项明确，有些点估计为负（BDG2 对 Chronos-2-X −4.0%，GoiEner 非住户对 Chronos-2-X −0.5%，EWELD 对 Chronos-2 −0.2%）。
- 在住户上，电量对比对全部四种变体都明确偏向 ankyra-lite（+28.7% 到 +33.6%），逐时对比对 TimesFM 明确不利于它（−4.6%；TimesFM-X −0.9%、Chronos-2 −2.8%、Chronos-2-X −2.9% 不明确）。
- 七个 BDG2 站点列在 TimesFM 2.5 的预训练集合中，这可能有利于 TimesFM 和 TimesFM-X 在 BDG2 上的表现（[局限](#局限)）。

### 秩检验

逐单位 RMSE，以单位为区组（`rank_tests.csv`；评分之后计算，描述性）：

- **六个测试人群合并**（1,762 个单位）：Friedman p < 10⁻³⁰⁰；Nemenyi 临界差 0.786。ankyra-lite 的平均名次是 22 个中的 5.234，下一个预测器（逐单位岭回归）为 7.451，第三个（iTransformer-X）为 8.568，差距大于临界差。经 Holm 校正的 Wilcoxon 检验认为它好于全部 21 个基线，不差于任何一个。
- **逐个人群**，它在七个人群上显著好于 21 个基线中的 21 个，在奥斯陆上好于 20 个（GBT-T 无显著差异），在 EWELD 上好于 19 个（TimesFM 和 Chronos-2 无显著差异）。没有任何基线在任何人群上显著更好。

![秩检验](figures/fig15_rank_tests.png)

*图 15. 合并后的平均名次及 Nemenyi 临界差和 Holm 校正的 Wilcoxon 检验（a）；各人群的配对检验（b）。*

### 常规指标

ankyra-lite 在 22 个预测器中的位置（`conventional_metrics_late.csv`；RMSE 和 MAE 为单位均值，单位 kW，CV(RMSE) 和 WAPE 取中位单位）：

| 指标 | BDG2 | GoiEner 非住户 | GoiEner 住户 | 剑桥 | HEEW | EWELD | 奥斯陆 | 德拉门 | CINELDI |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RMSE（单位均值） | 1 | 1 | 1 | 1 | 2 | 2 | 1 | 1 | 1 |
| MAE（单位均值） | 2 | 2 | 4 | 1 | 4 | 4 | 1 | 1 | 4 |
| CV(RMSE)（中位单位） | 3 | 1 | 1 | 1 | 1 | 3 | 1 | 1 | 1 |
| WAPE（中位单位） | 2 | 4 | 5 | 1 | 4 | 5 | 1 | 1 | 2 |
| WAPE（合并） | 2 | 2 | 5 | 1 | 5 | 6 | 1 | 1 | 4 |

位置 = 1 加上在文件的两位小数下严格领先的预测器个数。两位小数下的并列：住户上逐单位岭回归的 RMSE 相同（位置 1，若并列判给它则为 2），逐单位岭回归和 TimesFM-X 的 MAE 相同（位置 4，最多到 6）；HEEW 上 TimesFM-X 的 MAE 相同（位置 4，或 5）。

领先它的大多是零样本基础模型（Chronos-2-X、Chronos-2、TimesFM、TimesFM-X）。GBT-T 在 BDG2 的 CV(RMSE)、HEEW 的 RMSE、MAE 和合并 WAPE、CINELDI 的 MAE 和合并 WAPE 上领先；按日季节性朴素在 EWELD 的合并 WAPE 上领先。MAE 和 WAPE 由预测分布的中位数最小化，平方误差类指标由均值最小化；ankyra-lite 的载体按平方误差训练，锚定权重也按平方误差估计。

![22 个预测器的各项损失](figures/fig8_loss_metrics.png)

*图 8. 每个预测器在每个人群上的 RMSE、MAE、CV(RMSE) 和 WAPE，按相对 ankyra-lite 的损失着色。*

### 按预测日

六个测试人群，22 个预测器（`lead_day_metrics.csv`：每天 24 小时上的 CV(RMSE)，在一组固定单位上取几何平均；评分之后计算，描述性）：

| | BDG2 | GoiEner 非住户 | GoiEner 住户 | 剑桥 | HEEW | EWELD |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 第 1 天的位置 | 6 | 2 | 5 | 4 | 5 | 5 |
| 31 天中它最低的天数 | 8 | 0 | 0 | 27 | 11 | 0 |
| 任一天中最差的位置 | 6 | 6 | 5 | 4 | 6 | 11 |

在文件的两位小数下，最低值没有并列。

- **月初几天仍是它最弱的地方。** 第 1 天，零样本基础模型在每个人群上都领先于它（GoiEner 非住户上只有 Chronos-2-X；剑桥上 TimesFM、Chronos-2 和 Chronos-2-X；BDG2、住户、HEEW 和 EWELD 上全部四种变体），BDG2 上另有按日季节性朴素领先；第 1 天没有训练型基线领先（0.1 版时在六个人群中的五个上有训练型基线领先）。让第 1 天发生变化的是接续层（[下文](#按小时看接续层)）；它缩小了差距，没有消除差距。
- **第一周之后**，它的逐时损失在剑桥第 8 天起的全部 24 天、HEEW 的 11 天、BDG2 的 8 天上最低，在 EWELD 和两个 GoiEner 人群上一天也没有，那里最低的是零样本 Chronos-2-X 或 Chronos-2。
- **按日损失衡量的是什么。** 上表以及图 9 和图 10 用的是发布版的按日 CV(RMSE) 几何均值：每个单位在其各窗口上某一天 24 小时的 RMS 误差（住户每个单位一个窗口），表示为其平均负荷的百分比，在一组固定单位上按对数平均。它更像绝对误差类指标，偏向较低的、中位数型的预测；而主估计量，即每个单位各窗口全部 744 小时上的逐时 RMS，更重视误差大的日子。按同一文件中以 kW 计的单位均值 RMSE，ankyra-lite 在 GoiEner 住户 31 天中的 22 天、剑桥的 30 天、BDG2 的 12 天、GoiEner 非住户的 12 天、HEEW 的 7 天和 EWELD 的 2 天上是 22 个中最低的；按以 kW 计的单位均值 MAE，它在住户上一天也不是最低，在 GoiEner 非住户上有一天。在住户上，四种零样本变体的预测低于实测负荷（中位单位 NMBE，实测减预测：+17.7% 到 +22.4%，ankyra-lite 为 +4.3%；`conventional_metrics_late.csv`）。

![按预测日的损失](figures/fig9_loss_by_day.png)

*图 9. 第 1–31 个预测日的逐时 CV(RMSE)，在一组固定单位上取几何平均，六个测试人群；ankyra-lite 为交付轨迹，即接续层之后。紧挨着的图 9c 给出接续层按提前期的作用。*

![按小时看接续层](figures/fig9c_continuation_layer.png)

*图 9c（与图 9 并列）。同一载体和锚定层上开启对关闭接续层（两者在状态窗口上都用季节性朴素预测，接续层被绕过），按提前期：a，九个人群 1,891 个单位合并（第 0 小时；第 0–2 小时；第 1 天；第 2–7 天；第 8–31 天；整个窗口；以及交付轨迹的月电量）；b，各人群。改进附 95% 单位与月份区间；实心加粗 = 明确。*

![相对 ankyra-lite 的按日损失](figures/fig10_loss_by_day_relative.png)

*图 10. 每个列出的基线的按日损失相对 ankyra-lite（ankyra-lite = 0）。*

### 按小时看接续层

上面与图 9 并列的图 9c 画的就是这些对比。`continuation_layer.csv`（单位与月份区间；描述性，两边是同一载体）：

| 人群 | 第 0 小时 | 第 0–2 小时 | 第 1 天 | 第 2–7 天 | 第 8–31 天 | 整窗 |
|---|---|---|---|---|---|---|
| BDG2 2017 | +39.2 * | +32.2 * | +19.2 * | +3.7 * | +0.2 | +0.8 * |
| GoiEner 非住户 | +32.4 * | +29.8 * | +12.0 * | +2.4 * | +0.3 * | +0.5 * |
| GoiEner 住户 | +18.4 * | +13.5 * | +8.4 * | +1.7 * | +0.1 * | +0.4 * |
| 剑桥 | +22.6 * | +17.7 * | +7.6 * | +0.4 | −0.1 | +0.2 |
| HEEW 亚利桑那 | +31.9 * | +27.5 * | +14.3 * | +3.1 * | +0.3 * | +0.9 * |
| EWELD | +37.0 * | +28.4 * | +7.5 * | +1.1 | +0.3 | +0.5 * |
| 奥斯陆 | +35.1 * | +30.0 * | +12.6 * | −1.0 | −0.5 (−) | −0.3 |
| 德拉门 | +42.8 * | +39.5 * | +25.1 * | +2.3 | −0.1 | +1.0 |
| CINELDI | +24.8 * | +23.9 * | +6.4 * | +0.2 | −0.1 | +0.0 |
| 九个人群合并（1,891 个单位） | +28.0 * | +23.4 * | +11.1 * | +1.9 * | +0.2 * | +0.5 * |

- 接续层在每个人群上都改善了第 1 天；第一周之后作用接近零，在奥斯陆的第 8–31 天明确为负（−0.5%），整窗点估计为 −0.3%。
- 交付轨迹的电量合并后变化 +1.9%（不明确；各人群 −2.8% 到 +3.3%，都不明确）；电量读数本身不变（规则 A）。
- `continuation_record.csv`：326 个状态窗口绕过了接续层；33 个 EWELD 窗口用了参照日回退 d* = 6；没有窗口出现未观测或非正的时段；快异常和日异常分别在 6.4% 和 4.2% 的窗口上被截断；最大的 |Δ| 为 0.809；交付均值与电量读数的差异中位数 0.77%，第 99 百分位 7.8%，最大 310%（[读出](#读出)）。

### 月电量与月内累计电量

ankyra-lite 的电量取接续层之前轨迹的电量（读出规则 A）；每个基线取它自己轨迹的电量。

- **月电量**（`energy_error.csv`，九个人群）：明确好于 21 个基线中的 6 个（EWELD）到 20 个（剑桥），没有明确更差的。按中位单位的绝对百分比电量误差，它在 GoiEner 非住户（12.09%）、住户（12.99%）、奥斯陆（7.61%）和德拉门（7.72%）上是 22 个中的第一，在 BDG2 上第二，在剑桥和 CINELDI 上第三，在 HEEW 上第五，在 EWELD 上第八。
- **对 TimesFM**，电量误差在六个人群上明确更低（+20.2% 到 +34.9%），在 BDG2 上不明确（+14.7%），在 HEEW（−0.1%）和 EWELD（−2.8%）上更高但不明确。
- **月内累计**（`lead_day_energy.csv`：从第 1 天到第 d 天的电量误差，六个测试人群；描述性）：

  | | BDG2 | GoiEner 非住户 | GoiEner 住户 | 剑桥 | HEEW | EWELD |
  |---|:---:|:---:|:---:|:---:|:---:|:---:|
  | 31 天中它最低的天数 | 12 | 12 | 4 | 24 | 6 | 0 |
  | 第 31 天（即月电量）的位置 | 1 | 1 | 3 | 1 | 1 | 6 |
  | 第 8–31 天中低于全部四种零样本变体的天数（共 24 天） | 12 | 20 | 24 | 24 | 6 | 0 |

  HEEW 上第 31 天对 Chronos-2-X 的领先很小（7.546 对 7.671）。住户上第 31 天它排第三，在逐单位岭回归和 `SP-profile` 基线之后；EWELD 上排第六，在 PatchTST、TimesFM-X、Chronos-2、TimesFM 和周季节性朴素之后。

![随月份累积的电量误差](figures/fig9b_energy_by_day.png)

*图 9b. 截至每个预测日的电量误差，汇总方式同图 9；ankyra-lite 的累计电量从接续层之前的轨迹读出（读出规则 A）。*

![对每个基线的月电量误差](figures/fig11_energy_error.png)

*图 11. 相对每个基线的月电量误差改进，按人群（BDG2 另给出去掉三块接近零电表的结果）（a），以及两两对比计数（b）；ankyra-lite 的电量是接续层之前的读数。*

### 峰值读出

ankyra-lite 的峰值读出是锚定层的包络算子，作用在接续层之前轨迹的日均值上：U = 各天 d 上 (L̂_d + A_τd) 的最大值，即当天的预测均值加上同日类型近期的最大偏移（P15–P17）。`peak_readout.csv`：

- **对把同一读出用在每个基线轨迹上**（发布版的规则）：每个人群上明确好于 21 个中的 2 个（GoiEner 非住户）到 17 个（奥斯陆）；唯一明确的劣势是德拉门上对 GBT-T（−10.6%）。
- **对每个基线自己轨迹的最大值**：每个人群上明确好于 21 个中的 16–21 个，没有明确更差的。
- **对它自己轨迹的最大值**（接续层之后交付轨迹的最大值）：在全部九个人群上明确更好，从 +21.3%（奥斯陆）到 +68.2%（住户）。
- **对上个月的实测峰值**（最后 744 个上下文小时的最大值）：在奥斯陆上明确更好（+26.8%），在 CINELDI 上明确更差（−18.2%）；其余都不明确，从 −34.7%（GoiEner 非住户）到 +15.7%（德拉门）。
- **对把同一读出用在单独的载体上**（起报点上的六种子载体预测，比较对象 `same readout on the carrier alone`）：在七个人群上不明确（HEEW −6.9% 到 GoiEner 非住户 +5.6%），在奥斯陆（−2.5%）和德拉门（−2.9%）上明确更差。这个读出会把日均值上的改进带到峰值（P16），但这里锚定层在水平上相对载体的收益很小（月电量在任何人群上都不明确，对比 (i)），峰值上也就看不出来。

![峰值读出](figures/fig5_peak_operator.png)

*图 5. 峰值读出对它自身交付轨迹的最大值（a）和对上个月的实测峰值（b）；对把同一读出用在每个基线轨迹上的两两对比计数（c）。*

### 误差分块与尺度化误差

每个窗口的逐时平方误差可以精确拆成水平、日路径和日内三部分（P1）。`block_shares.csv`（六个测试人群，交付轨迹，描述性；分块对比不带区间）：

| | BDG2 | GoiEner 非住户 | GoiEner 住户 | 剑桥 | HEEW | EWELD |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| ankyra-lite 误差的合并份额：水平 | 36.6% | 12.4% | 14.5% | 15.9% | 38.8% | 48.9% |
| 日路径 | 43.2% | 32.4% | 16.2% | 20.1% | 32.2% | 35.4% |
| 日内 | 20.2% | 55.2% | 69.3% | 64.0% | 29.1% | 15.7% |
| 对 TimesFM 的改进：水平 | +15.3 | +22.8 | +34.7 | +23.7 | +2.5 | −4.3 |
| 日路径 | −2.4 | −0.4 | −9.8 | +12.0 | +3.2 | +3.5 |
| 日内 | −1.8 | +5.6 | −3.1 | +12.1 | +2.0 | +3.0 |

在住户上，对 TimesFM 的逐时劣势在日路径和日内块里，不在水平上。这里的水平块是交付轨迹（接续层之后）的水平块；电量读数见 `energy_error.csv`。

`scaled_errors.csv`（每个窗口的误差除以周季节性朴素在其上下文上的样本内误差；描述性）：按中位单位 RMSSE，ankyra-lite 在七个人群（GoiEner 非住户、住户、剑桥、HEEW、奥斯陆、德拉门、CINELDI）上排第一，在 BDG2 和 EWELD 上排第二。按中位单位 MASE（一种绝对误差尺度），它在剑桥、奥斯陆、德拉门和 CINELDI 上第一，在 BDG2、GoiEner 非住户、住户和 HEEW 上第五，在 EWELD 上第六。

![分块归因](figures/fig14_block_attribution.png)

*图 14. ankyra-lite、TimesFM 和 Chronos-2-X 的逐时平方误差中各块的中位单位份额（a）；ankyra-lite 按块对 TimesFM（b）和对 Chronos-2-X（c）的改进。*

### 预测区间

ankyra-lite 自己不产生区间。作为描述，把锚定层的区间读出放在它的交付轨迹（接续层之后）周围：单位在最多 12 个已完成伪起报点上**纯历史**预测器的残差分位数，与载体无关。与 TimesFM 2.5 和 Chronos-2 的原生分位数比较（`intervals_by_population.csv`、`intervals_winkler_contrasts.csv`、`intervals_households.json`；描述性）：

| 人群 | 80% 覆盖率：ankyra-lite / TimesFM / Chronos-2 | 90% 覆盖率：ankyra-lite / Chronos-2 | Winkler 80 对 TimesFM | 对 Chronos-2 |
|---|---|---|---|---|
| BDG2 2017 | 0.766 / 0.349 / 0.629 | 0.858 / 0.776 | −2.7 | −17.2 |
| GoiEner 非住户 | 0.792 / 0.430 / 0.720 | 0.876 / 0.851 | +6.2 | −9.1 (−) |
| GoiEner 住户 | 0.782 / 0.345 / 0.680 | 0.874 / 0.815 | +6.8 * | −1.4 |
| 剑桥 | 0.790 / 0.393 / 0.710 | 0.880 / 0.836 | +15.3 * | −1.1 |
| HEEW 亚利桑那 | 0.776 / 0.373 / 0.726 | 0.863 / 0.848 | +9.3 | −2.5 |
| EWELD | 0.844 / 0.606 / 0.804 | 0.894 / 0.883 | −3.7 | −39.6 (−) |
| 奥斯陆 | 0.847 / 0.379 / 0.678 | 0.920 / 0.813 | +21.0 * | +4.5 * |
| 德拉门 | 0.855 / 0.388 / 0.725 | 0.917 / 0.841 | +20.6 * | +3.2 |
| CINELDI | 0.818 / 0.438 / 0.736 | 0.901 / 0.859 | +15.5 * | −4.3 (−) |

- 80% 区间带覆盖 0.766–0.855 的小时，TimesFM 原生分位数为 0.345–0.606，Chronos-2 为 0.629–0.804。它的覆盖率随提前期下降（第一周 0.808–0.884，第四周 0.718–0.829），因为残差是在整个窗口上合并的。
- 在 80% Winkler 分数上，它在五个人群上明确好于 TimesFM 的区间带，没有明确更差的；对 Chronos-2 的分位数，它在奥斯陆上明确更好，在 EWELD、GoiEner 非住户和 CINELDI 上明确更差。
- 在 EWELD 上，区间带对会关停的单位不含信息（80% 平均宽度 388,321 kW）。
- 在住户上，把同一区间放在单独的载体周围，Winkler 分数变化 0.10%（不明确）：区间来自纯历史的残差，而不是来自载体。

![住户上的预测区间](figures/fig13_intervals.png)

*图 13. GoiEner 住户上各区间的覆盖率和 Winkler 分数。*

![每个人群上的区间覆盖率](figures/fig16_intervals_populations.png)

*图 16. 每个人群上以及按预测周的名义 80% 区间覆盖率。*

### 各部分带来什么

ankyra-lite 0.2.0 相对去掉某一部分的版本、或相对列出的替代方案的逐时改进（`component_contributions.csv`、`ablation.csv`；“相同”：两个预测在每个窗口上都一样）。关掉锚定层某些部分的行，是在保存的六种子载体预测上重新运行未改动的锚定层；在用任何一行之前，重新运行先逐位复现了被评估的预测。配方和载体的行把整个 0.2.0 配置与研究保存的产物比较，所以其中也包含了损失权重、六个种子和接续层。

| 部分 | BDG2 | GoiEner NH | GoiEner HH | 剑桥 | HEEW | EWELD | 奥斯陆 | 德拉门 | CINELDI |
|---|---|---|---|---|---|---|---|---|---|
| 锚定层、状态规则和接续层合在一起（对单独的载体） | +3.6 * | +3.3 | −1.3 | +0.1 | +2.2 * | +0.5 | −0.5 | +0.8 | +1.3 |
| 锚定层中的载体（对纯历史配置） | +8.0 * | +21.1 * | +11.9 * | +15.0 * | +13.8 * | +16.8 * | +20.1 * | +17.4 * | +6.2 * |
| 接续层（同一载体和锚定层） | +0.8 * | +0.5 * | +0.4 * | +0.2 | +0.9 * | +0.5 * | −0.3 | +1.0 | +0.0 |
| 日内锚定 | +0.7 * | +2.4 | −0.0 | +0.3 | +0.8 * | −0.0 | +0.6 * | +0.9 * | +0.9 * |
| 季节性朴素状态规则 | +0.3 | −0.0 | +0.3 | 相同 | 相同 | +2.1 | 相同 | 相同 | 相同 |
| **0.2.0 对 0.1**（全部改动合在一起） | +1.7 | +1.9 * | +0.9 * | +1.4 * | +1.4 * | +0.2 | −0.5 | +2.4 * | +0.8 |
| 损失权重（配方 Rw 对 R，六个种子，含接续层） | +0.2 | +0.9 | +0.3 | +0.5 | +0.5 | −0.3 | −0.2 | +0.1 | +0.6 * |
| 六个种子对种子 0–2（配方 Rw，含接续层） | +0.7 | +0.4 | +0.2 | +0.8 | −0.0 | +0.0 | −0.3 | +0.9 | +0.5 |
| 六种子集成对一个三种子组（两组的平均；点估计） | +0.5 | +0.2 | +0.1 | +0.3 | +0.4 | +0.2 | +0.3 | +0.5 | +0.2 |
| 0.2.0 对只有从零训练阶段（0.1 配方研究） | +2.9 * | +4.6 * | +2.3 * | +3.1 * | +2.3 * | +1.3 | +1.7 * | +0.2 | +1.6 |
| 0.2.0 对普通继续训练 F（0.1 配方研究） | +2.2 * | +4.1 | +1.7 * | +2.0 * | +1.2 * | −0.2 | −0.4 | −0.4 | +1.1 |
| the carrier's level candidate, the handover and the 日内锚定 (against the fixed division) | +2.9 * | +4.3 * | +2.9 * | +7.9 * | +6.8 * | +8.4 * | +8.3 * | +7.5 * | +2.7 * |
| 按周的交接权重（对固定 ½ 权重） | +0.9 * | −0.1 | +0.4 | +1.2 | +0.9 | +1.0 | +0.4 | +1.0 * | +0.0 |
| 分块权重（对同样两个预测的单个整窗权重） | −0.2 | −0.2 | +1.5 * | +0.4 | +0.4 | +0.5 | +0.4 | −0.0 | −0.2 |
| 分块权重（对同样两个预测的各半平均） | +1.0 | +1.7 | +4.0 * | +3.9 * | +2.9 * | +3.1 * | +2.5 * | +1.5 * | +0.3 |

- 在 `component_contributions.csv` 中，0.1 配方研究的两行保留了 0.1 的名字（`continuation with the span return (from-scratch stage -> released recipe)`、`span return (...)`）；在 0.2.0 中它们是整个 0.2.0 配置对研究保存的从零训练阶段和 F 的三种子产物的比较，所以不是验证段归还本身的贡献。后者（同样种子下 R 对 F）见[配方历史](#配方历史r-是怎样选定的以及它的负面结果)。
- **对比 (i)**，锚定层对单独的载体（接续层之前，`carriers_in_anchoring_layer.csv`），在两个人群上逐时误差明确（BDG2 +2.8%，CINELDI +1.3%），月电量在任何人群上都不明确；在住户上为 −1.7%（不明确）。**对比 (ii)**，对纯历史配置，在全部九个人群上逐时误差明确（+6.1% 到 +20.6%），在五个人群上月电量明确（剑桥 +20.0%、HEEW +14.5%、EWELD +34.8%、奥斯陆 +20.7%、德拉门 +12.7%）。
- **小开关。** 每个窗口一个日内信任值（相对按提前段各一个）、短缺口规则和微负荷规则，各自对逐时误差的影响不超过 0.2%。状态规则只在存在状态窗口的地方起作用；小开关的电量对比没有一项明确。
- **更早的载体**（八个人群；修正后的 CINELDI 数据仓库上没有；`ablation.csv`）：相对同一网络去掉复制指针、用从零配方训练，ankyra-lite 0.2.0 好 +1.8% 到 +5.0%（七个人群明确，德拉门 +2.2% 不明确）；相对更早的、没有指针、在按周窗口上训练的结构化网络，好 +2.0% 到 +11.7%（八个人群都明确）。这些行也包含了 0.2.0 的全部改动。
- 这些行的区间只重抽单位和月份；带种子层的 0.2.0 行见 [0.2.0 改了什么](#020-改了什么)。
- **日内锚定在哪里起作用**（`within_day_trust.csv`）。在八个非住户人群上，对单位相似日形状的信任平均为 0.13–0.21（上限 ½），在 76–88% 的锚定窗口上为正；在 GoiEner 住户上平均为 0.05（中位数 0，43% 为正），所以那里基本保留载体的日内形状，与该行的 −0.0% 一致。合理性检查在 94–100% 的窗口上保留了相似日形状。
- 交接各行由锚定层在每个窗口上自己的字段重建（`evaluation/lite_reanchor.py`，`fields` 模式；重建的按周权重、信任、水平权重和日均值与锚定层自己的完全相等，按周权重那一臂逐位复现锚定后的预测）；它们比较的是接续层之前的预测，讨论见[锚定层怎样使用载体](#锚定层怎样使用载体)。

**分块定权对单个整窗权重。** 这个对照检验收益的一种替代解释：收益来自对水平、日路径和日内三块分开定权。它把与锚定层相同的两个预测，即历史侧（固定分工的日均值加相似日形状）和载体，用每个窗口**一个**覆盖整个 744 小时的最小二乘权重组合起来，权重由单位自己已完成的伪预测（交接对和信任三元组）确定，并向 ½ 收缩；ANKYRA 2.2.1 发布版称之为 B2，各半平均称为 B1。载体上的平均权重为 0.56–0.73。

- 在逐时误差上，单个整窗权重与 ankyra-lite 相差 −0.2% 到 +1.5%：只有在 GoiEner 住户上 ankyra-lite 明确更好（+1.5%），其余八个人群分不开。在月电量上没有一个人群分得开（−3.9% 到 +3.0%）。
- 各半平均更弱：ankyra-lite 在六个人群上逐时误差明确好于它（+1.5% 到 +4.0%），月电量上一个也没有。
- 解读：在这些人群上，分块定权不是收益的必要解释。这个比较留下的是历史侧预测，以及按单位自己伪预测误差定权；分块分解仍是用于解释、读出和诊断的精确恒等式（P1）。这个对照只检验锚定层自己的分块定权对它自己的整窗特例，不涉及其他按块定权的方法。

**通用组合伙伴**（`generic_combinations.csv`）。历史侧提供的是否多于一个通用的局部预测？B2(X) 是把历史侧换成 X 的同一种整窗组合：周季节性朴素、4 周剖面或去年同期剖面，各自用每个伪起报点之前的负荷计算；EW(X) 是载体与 X 的等权平均（X 也可以是逐单位岭回归或 MSTL）。九个人群中，前者明确更好 / 不明确 / 明确更差的人群数：

| 对比 | 逐时 | 月电量 |
|---|---|---|
| B2(历史侧) 对 B2(季节性朴素，周) | 1 / 8 / 0 | 0 / 9 / 0 |
| B2(历史侧) 对 B2(4 周剖面) | 0 / 9 / 0 | 0 / 9 / 0 |
| B2(历史侧) 对 B2(去年同期剖面) | 1 / 8 / 0 | 0 / 9 / 0 |
| B2(历史侧) 对最好的 B2(X) | 0 / 9 / 0 | 0 / 9 / 0 |
| B2(历史侧) 对最好的等权平均 | 2 / 7 / 0 | 0 / 8 / 1 |
| ankyra-lite 对其中最好的 | 0 / 9 / 0 | 0 / 9 / 0 |

“最好”是在评分之后按人群和误差挑出的，这偏向通用一侧。在同一种组合中，历史侧对三个通用预测器中的每一个都是明确更好（两项对比）或分不开；对其中最好的，在任何人群上都分不开。唯一明确的劣势是德拉门的月电量，B2(历史侧) 对载体与逐单位岭回归的等权平均（−13.4%）。

![消融](figures/fig4_ablation.png)

*图 4. ankyra-lite 0.2.0 对它自己的简化版和替代版（逐时误差）：关掉锚定层的部分、状态规则和接续层，单独的载体和纯历史配置（a）；用一个三种子组代替六个种子（b）；同一锚定层中的其他载体、0.1 版的配方方案和 0.2.0 的各配置（配方 R 或 Rw，三个或六个种子，有无接续层）（c）；交接各臂和单个整窗权重（d）；ankyra-lite 和固定分工按预测周对单独的载体（e）；按周的载体平均交接权重（f）。*

### 种子

`seed_spread.csv`：单独锚定的每组三个种子（种子 0–2、种子 3–5，都含接续层）以及六种子集成对 TimesFM 的逐时改进：

| 人群 | 种子 0–2 | 种子 3–5 | 六个种子 |
|---|---|---|---|
| BDG2 2017 | +0.0 | +0.4 | +0.7 |
| GoiEner 非住户 | +4.3 | +4.6 | +4.7 |
| GoiEner 住户 | −4.9 (−) | −4.7 (−) | −4.6 (−) |
| 剑桥 | +14.7 * | +15.6 * | +15.4 * |
| HEEW 亚利桑那 | +2.5 | +1.8 | +2.5 |
| EWELD | +1.5 | +1.2 | +1.5 |
| 奥斯陆 | +23.2 * | +22.3 * | +23.0 * |
| 德拉门 | +20.2 * | +20.8 * | +20.9 * |
| CINELDI | +10.1 * | +10.5 * | +10.5 * |

两组之间最多相差 0.97 个百分点；在每个人群上，两组和集成的结论都相同。同一文件还给出每个种子单独的载体。除 `recipe_history.csv` 中 0.2.0 行的种子层外，种子差异是 `results/` 中衡量重新训练噪声的唯一量；主表的区间不包含它。

### 同一锚定层中的其他载体

`carriers_in_anchoring_layer.csv`：同一锚定层配上另外三个训练好的载体（LSTM、PatchTST 和 LSTM-T-KAN-M，只在各自人群上训练，训练截点与小型载体相同；在状态窗口上锚定层返回它们自己的预测；在 CINELDI 上它们在原始数据仓库上训练、在修正后的数据仓库上锚定；它们都不加接续层）。九个人群中明确更好 / 不明确 / 明确更差的人群数；小型载体的对比 (i) 和 (ii) 在接续层之前，最后两列比较交付的 ankyra-lite 的逐时误差和它的电量读数（接续层之前）：

| 载体 | (i) 锚定后对单独载体：逐时 | 电量 | (ii) 锚定后对纯历史：逐时 | 电量 | ankyra-lite 对这个锚定后的载体：逐时 | 电量 |
|---|---|---|---|---|---|---|
| 小型载体网络（ankyra-lite） | 2 / 7 / 0 | 0 / 9 / 0 | 9 / 0 / 0 | 5 / 4 / 0 | — | — |
| LSTM | 8 / 1 / 0 | 6 / 3 / 0 | 3 / 5 / 1 | 0 / 7 / 2 | 9 / 0 / 0 | 6 / 3 / 0 |
| PatchTST | 7 / 2 / 0 | 3 / 6 / 0 | 7 / 2 / 0 | 3 / 6 / 0 | 7 / 2 / 0 | 1 / 8 / 0 |
| LSTM-T-KAN-M | 8 / 1 / 0 | 7 / 2 / 0 | 1 / 6 / 2 | 0 / 5 / 4 | 9 / 0 / 0 | 6 / 3 / 0 |

- 锚定层对较弱的载体改进更多（对比 (i)）。配上小型载体，组合在全部九个人群上的逐时误差好于纯历史配置，配上 LSTM 为三个，配上 LSTM-T-KAN-M 为一个。对比 (ii) 的明确劣势包括两者在 EWELD 上的（逐时 −83.1% 和 −104.4%），那里锚定层在 245 个状态窗口上返回这些载体自己的预测；以及 LSTM-T-KAN-M 在剑桥上的（−10.3%）。
- 没有任何锚定后的载体在任何人群上明确好于 ankyra-lite。

**按块看对比 (ii)，以及无载体对照**（`contrast_ii_blocks.csv`、`contrast_ii_control.csv`；描述性）。对比 (ii) 比较的是两个完整的配置：纯历史配置用日类型默认形状代替相似日形状，也没有状态规则。无载体对照用纯历史预测代替载体，运行同一流程。九个人群中明确更好 / 不明确 / 明确更差的人群数，全部测试窗口（“锚定后”指接续层之前的 ankyra-lite）：

| 对比 | 逐时 | 水平 | 日路径 | 日内 |
|---|---|---|---|---|
| 锚定后对纯历史 | 9 / 0 / 0 | 5 / 4 / 0 | 6 / 3 / 0 | 8 / 1 / 0 |
| 含接续层的锚定后对纯历史 | 9 / 0 / 0 | 5 / 4 / 0 | 5 / 4 / 0 | 8 / 1 / 0 |
| 锚定后对单独载体 | 2 / 7 / 0 | 0 / 9 / 0 | 1 / 6 / 2 | 6 / 3 / 0 |
| 单独载体对纯历史 | 7 / 2 / 0 | 3 / 6 / 0 | 5 / 3 / 1 | 8 / 1 / 0 |
| 锚定后对无载体对照 | 9 / 0 / 0 | 5 / 4 / 0 | 5 / 4 / 0 | 8 / 1 / 0 |
| 无载体对照对纯历史 | 8 / 1 / 0 | 0 / 9 / 0 | 1 / 8 / 0 | 8 / 1 / 0 |

对无载体对照，ankyra-lite 在全部九个人群上逐时误差明确更好（CINELDI +3.8% 到奥斯陆 +15.6%）。两边运行同一流程，只是放在载体位置上的预测不同（以及状态窗口，ankyra-lite 在那里交付季节性朴素预测）。这个预测同时进入水平候选、交接和日内混合，所以按块拆分只能粗略定位差别。

### 历史长度与锚定记录

`history_length.csv`，九个人群合并（描述性；分层与人群混杂）：

| 起报时距单位首次观测的小时数 | 窗口 | 对单独的载体 | 对 TimesFM | 对 Chronos-2-X |
|---|---:|---|---|---|
| < 10,248 小时（没有年度候选） | 450 | −0.6 | −2.4 | −2.6 |
| 10,248 小时 – 2 年 | 1,715 | +1.7 | +0.4 | −2.7 |
| 2–3 年 | 829 | +2.5 | +11.8 | +7.2 |
| ≥ 3 年 | 2,722 | +0.3 | +9.1 * | +5.8 |

对零样本基础模型，点估计在 10,248 小时以下为负，两年以内有正有负，两年以上为正；只有 ≥ 3 年这一层对 TimesFM 明确。这是描述，不是各人群共同的门槛。

`anchoring_gain_deciles.csv`（`anchoring_record` 的两个统计量对 ankyra-lite 相对单独载体的实际收益；九个人群上每个十分位 538–539 个窗口，去掉状态窗口和平均负荷接近零的单位；描述性）：月水平收益在载体水平权重最低的十分位最大（+8.8%），在第七和第八个十分位为负（−7.2%、−3.7%）；按 `history_vs_carrier_log_ratio`，收益在最低的两个十分位最大，即单位历史在更早伪起报点上胜过载体的地方（+10.5%、+9.4%；那里逐时收益明确，+3.1% 和 +2.4%）。两个统计量都不能把收益单调排序，而且载体权重就是锚定层自己的权重，所以它与收益的关联部分是机械的。

**伪预测多常选中最好的水平候选**（`level_candidate_hit_rate.csv`，P20；描述性）。权重最大的水平候选，即在单位已完成伪起报点上误差最小的那个，恰好是窗口均值最接近实际值的候选，九个人群合并时占 21.7% 的窗口，七个候选中随机为 14.4%（差 +7.3 个百分点，95% 单位自助区间 +6.1 到 +8.4）；各人群为 17.5% 到 25.7%，差值在七个人群上明确（EWELD 和 CINELDI 不明确）。只看六个历史候选时为 23.3% 对 16.9%。载体的窗口均值在 40–71% 的窗口上权重最大，在 15–29% 的窗口上是最接近的候选。所以伪起报点误差对哪个候选会最好有一定信息，但绝对量不大，这正是对水平权重做强收缩所假设的。

### 对受损上下文的稳健性

`robustness.csv`：64 个德拉门测试窗口（43 个单位），用发布的检查点在 CPU 上从损坏的上下文重新预测（每个发布时刻的载体、锚定层、状态规则、接续层）；目标不动。逐时 RMS 误差相对干净上下文的变化（正数 = 更差）：

| 上下文的损坏方式 | ankyra-lite | 单独的载体 | 损坏后 ankyra-lite 对单独的载体 |
|---|---:|---:|---:|
| 5% 的小时缺测，线性插值填补 | +0.0% | −0.1% | −2.9% |
| 20% 的小时缺测，填补 | +0.2% | +0.1% | −2.9% |
| 最后一周中有一段 24 小时被置零 | +2.3% | +2.4% | −3.0% |
| 起报前 36 小时有一个 10 倍上下文最大值的尖峰 | +52.5% | +40.7% | +5.2% |
| 上下文平移 +1 小时 | +10.8% | +10.8% | −2.9% |
| 上下文平移 −1 小时 | +13.3% | +13.9% | −3.4% |
| 整个历史乘以 0.5 | +0.0% | +0.0% | −2.9% |
| 整个历史乘以 2 | +0.0% | +0.0% | −2.9% |

- 把整个历史乘以 0.5 或 2，预测严格按比例变化（最大偏差 0），只要记录没有跨过某个状态阈值（阈值是以 kW 计的绝对值）。
- 尖峰使峰值读出的中位变化为 +997%：读出取的是近期最大偏移。落在最后 24 小时内的尖峰或置零段也会进入接续层的异常，这些异常被截断在 ±1。
- 任何损坏的上下文都没有触发状态规则。

### 稳健性：训练型基线的训练预算

对与训练型模型比较的一种解释是，它们被较小的训练预算拖了后腿。ANKYRA 2.2.1 发布版用最多 30 轮、早停 patience 8（代替 15 和 3）在每个人群上重新训练了 DLinear、iTransformer、LSTM 和 PatchTST（一次放宽，训练前定下；截止时间、训练对、选择集、学习率网格和种子不变）。`results/training_budget_sensitivity.csv` 在同样的测试窗口上把这些保存的预测与 ankyra-lite 比较（这里没有重新训练任何东西）：

- **逐时误差：** 36 个结论（九个人群 × 四个模型）中有 35 个不变；HEEW 上对 iTransformer 的对比变为不明确（+8.3% → +6.5%）。没有点估计改变符号；把四个放宽后的模型换进来，ankyra-lite 在每个人群上的单位平均名次仍是第一。
- **月电量：** 36 个结论中 33 个不变。三个对比变为不明确：BDG2（+18.9% → +15.8%）和 CINELDI（+28.1% → +11.4%）上的 LSTM，住户上的 PatchTST（+17.8% → +12.4%）；没有点估计改变符号。
- **模型本身**（放宽对原始，逐时；与 ankyra-lite 无关）：8 个明确更好，11 个不明确，3 个明确更差（EWELD 和住户上的 DLinear、EWELD 上的 iTransformer），14 个与原始预测相同（学习率和轮数相同）。LSTM 收益最大，在剑桥和 GoiEner 非住户上它选中的是第 30 轮，所以更大的预算对它可能仍然不够。

在这一次放宽下，结论大多保持。它是一个替代预算，不是对预算的搜索；拿到协变量的训练型模型没有重新训练，LSTM-T-KAN-M 本来就是用 30 轮和 patience 8 训练的（[细节](docs/EVALUATION.md)）。

### 一年上下文的基础模型

零样本 TimesFM 2.5 用 8,736 小时（52 周）上下文、Chronos-2 用 8,192 小时（其配置的上限），作为 22 个预测器及其名次之外的两个额外基线评分（`results/long_context_baselines.csv`；ANKYRA 2.2.1 发布版保存的预测；记录更短时，取在起报点结束的最长完整观测段，这些窗口上至少 5,809 小时）。九个人群中 ankyra-lite 明确更好 / 不明确 / 明确更差的人群数：

| 对 | 逐时 | 月电量 |
|---|---|---|
| TimesFM-8736 | 4 / 4 / 1 | 5 / 4 / 0 |
| Chronos-2-8192 | 4 / 4 / 1 | 4 / 5 / 0 |
| TimesFM（1,344 小时，主表） | 4 / 4 / 1 | 6 / 3 / 0 |
| Chronos-2（1,344 小时，主表） | 4 / 5 / 0 | 4 / 5 / 0 |

- 有一年上下文时，TimesFM 在 GoiEner 住户上的逐时误差明确低于 ankyra-lite（−5.3%），Chronos-2 也是（−5.2%）；在 BDG2 上 TimesFM-8736 领先但不明确（−6.1%）。ankyra-lite 在剑桥、奥斯陆、德拉门和 CINELDI 上明确好于两者（+7.7% 到 +24.8%）。
- 更长的上下文在四个人群上明确帮助了 TimesFM（剑桥、GoiEner 非住户、德拉门、CINELDI：逐时 +3.1% 到 +6.6%），在住户上帮助了 Chronos-2（+4.4%）；8,192 小时的 Chronos-2 在 BDG2 和 EWELD 上明确差于 1,344 小时的。
- 在 BDG2 上，ankyra-lite 的对比去掉了三块接近零的电表（它在这些电表上的误差恰好为零），与主表相同；两种 TimesFM 上下文之间的对比保留它们。

### 每个窗口的耗时

`cost_per_window.csv`：本仓库的包配发布的检查点，6 个奥斯陆测试窗口，先做一次不计时的预热，CPU，单线程，不用 GPU：

| 步骤 | 耗时 |
|---|---:|
| 一次完整预测（载体预测、含状态规则的锚定层、接续层、峰值读出） | 0.655 s 每个窗口 |
| 起报点和伪起报点上的载体预测（平均 7.0 个发布时刻）× 6 个种子 | 0.135 s 每个窗口 |
| 一次载体预测（一个发布时刻、一个种子；平均） | 0.0032 秒 |
| 给定载体预测时的锚定层，含状态规则和接续层 | 0.520 s 每个窗口 |
| 峰值读出 | 0.0003 s 每个窗口 |
| 载入六个检查点（一次） | 0.057 秒 |

耗时主要在锚定层，而不是载体；六个种子的载体耗时约是三个种子的两倍（0.1 版为 0.075 秒）。一个检查点 0.2657 MB。保存的评估预测是在 GPU 上算的；在 CPU 上，本包复现它们的最大相对差为：载体预测 1.89 × 10⁻⁶，接续层之前的轨迹 1.09 × 10⁻⁶，交付轨迹 9.8 × 10⁻⁷（文件中的 `check:` 各行）。

### 配方历史：R 是怎样选定的，以及它的负面结果

这是 0.1 版的配方研究；它的各行在 0.2.0 中不变，0.2.0 的配方 Rw 就是在配方 R 的第二阶段加上损失权重（0.2.0 的研究见 [0.2.0 改了什么](#020-改了什么)）。配方是 ankyra-lite 0.1 最后定下的部分。它的第一阶段（从零训练，在固定的 93 天验证段上早停）把三个短记录人群 23.7–38.3% 的自身窗口拿出了训练，也连带拿走了它们部分测试月份需要的季节。研究比较了同一批第 1 阶段检查点的几种继续训练（各三个种子，在全部九个人群的测试窗口上；德拉门和 CINELDI 用修正后的数据仓库）。计划及其预测是在对各方案唯一一次读数之前写下的；这些测试窗口此前已经读过很多次，所以这次读数给出的是方向，不是确认。作者于 2026 年 10 月 19 日选定 R，并披露下面的损失。

| 方案 | 做法 | `recipe_history.csv` 中的代码 |
|---|---|---|
| 只有从零训练阶段 | 只有第 1 阶段 | `P3s` |
| 普通继续训练（对照） | 第 1 阶段，然后用新的优化器再训练 8 轮，验证集仍然留出 | `F` |
| 验证段归还（**0.1 发布**；Rw 的基础） | 同 F，另把验证段归还给训练集 | `R` |
| 按比例划分的验证集 | 第 1 阶段，然后继续训练，验证段为该人群记录的 10%（38–93 天），并带早停 | `V` |

验证段归还对普通继续训练（`R vs F`；单位等权对数 RMS 比，人群等权；Welch 95% 区间，结合三种子集成的单位与月份自助法和三个单种子对比之间的差异；正数偏向 R）。每个数字都是 `results/recipe_history.csv` 的一行：

| 读数（除注明外均为逐时） | R 对 F |
|---|---|
| 三个短记录人群合并 | +1.17% [−0.64, +2.94]，不明确 |
| 它们的切换层（BDG2 150 个、GoiEner 非住户 406 个、住户 415 个窗口） | **+3.11% [+0.98, +5.21]，明确** |
| 它们的非切换层 | −0.52% [−2.08, +1.02]，不明确 |
| 九个人群合并 | +0.11% [−0.82, +1.03]，不明确 |
| 有开发窗口的六个人群 | −0.42% [−1.22, +0.37]，不明确 |
| **有开发窗口的六个人群，第 2–7 个预测日** | **−1.35% [−2.43, −0.28]，明确不利于 R** |
| **德拉门**（单个人群的点估计） | **−2.90%**（月电量 −8.76%，峰值 −4.24%） |
| 三个短记录人群：月电量（交付轨迹）、峰值 | +3.95% [−3.93, +11.23]；+0.67% [−3.88, +5.02]；都不明确 |
| 九个人群：月电量（交付轨迹）、峰值 | +0.32% [−3.30, +3.81]；−0.61% [−2.52, +1.27]；都不明确 |

- **验证段归还并非处处更好。** 它的收益出现在训练集缺少测试月季节的地方，与计划的预测一致；九个人群合并后，它与普通继续训练分不出高下；在德拉门上它更差（单点损失最大，逐时误差、电量和峰值都是如此）；在有开发窗口的六个人群上，第 2–7 个预测日明确更差。这些损失是配方 R 的一部分，因而也是保留了验证段归还的配方 Rw 的一部分。
- **机制没有被分离出来。** 归还验证段增加了同季数据，但也改变了训练数据的新近程度和多样性，以及旧窗口的重放；这些读数支持季节解释，但不能证明它。
- **按比例划分的验证集没有带来收益**：对普通继续训练，在四个长记录人群上为 −0.35% [−1.15, +0.43]，在三个短记录人群上为 +0.39% [−0.89, +1.65]，都不明确；在切换层上它明确差于验证段归还（−2.15% [−3.96, −0.36]）。
- **单纯继续训练的作用。** 相对只有从零训练阶段（九个人群；只有单位与月份区间，没有种子层）：普通继续训练 +0.97% [+0.55, +1.52]，验证段归还 +1.08% [+0.36, +1.86]，按比例验证 +0.86% [+0.56, +1.32]。九个人群合并时，R 与从零训练阶段之间的差别，大部分来自训练得更久，而不是来自归还的验证段。在德拉门上，R 比只有从零训练阶段差 2.30%。

![配方历史](figures/fig18_recipe_history.png)

*图 18. a–c：配方 R（验证段归还，配方 Rw 在其上加了损失权重）背后的训练配方研究，在测试窗口上（此前读过：是描述，不是确认）：按人群组和分层的 R 对 F、V 对 F 和 V 对 R（a），按读出的 R 对 F（b），以及按人群的 R 对 F（c）；脚注列出所有明确不利于 R 的对比。d–e：0.2.0 研究的一次测试读数（只定方向）：0.2.0 对 0.1 以及各单项改动（同一 Rw6 载体上的接续层、损失权重 Rw6 对 R6、六个种子对三个 R6 对 R3）按读出（d），以及 0.2.0 对 0.1 按人群组（e）；电量和峰值在接续层之前读出。*

## ankyra-lite 的薄弱之处

对外部基线，在这些测试窗口上：

- **住户的逐时误差。** 在 GoiEner 住户上，TimesFM 的逐时误差明确更低（−4.6%），其余三种零样本变体的点估计更低（−0.9% 到 −2.9%，不明确），ankyra-lite 按 MAE 排第四、按 WAPE 排第五，在零样本变体之后。0.2.0 缩小了与 TimesFM 的差距（0.1.1 中为 −5.6%），但没有消除。它在那里的月电量误差和峰值误差明确低于全部四种变体。
- **月初几天。** 第 1 天它在六个测试人群上排 22 个中的第 2 到第 6（BDG2 第 6），在每个人群上都落后于零样本基础模型。接续层缩小了第 1 天的差距（相对同一载体不加接续层，合并 +11.1%），没有消除差距。
- **短记录与零样本基础模型。** 在没有开发窗口、训练记录最短的 BDG2 和 GoiEner 非住户，以及 HEEW 和 EWELD 上，与四种零样本变体的逐时对比都不明确，BDG2（Chronos-2-X −4.0%）、GoiEner 非住户（Chronos-2-X −0.5%）和 EWELD（Chronos-2 −0.2%）上有负的点估计。在两个 GoiEner 人群和 EWELD 上，它的逐时损失在任何一个预测日都不是最低。
- **HEEW 和 EWELD 的电量。** 对 TimesFM 的月电量在 HEEW 上为 −0.1%、在 EWELD 上为 −2.8%（不明确），EWELD 上第 31 天的电量排 22 个中的第六。
- **绝对误差类指标。** 按 MAE，它在四个人群（住户、HEEW、EWELD、CINELDI）上排第四；按中位单位 WAPE，在四个人群（GoiEner 非住户、住户、HEEW、EWELD）上排第四或第五；按中位单位 MASE，在五个人群上排第五或第六，主要落后于零样本基础模型。
- **峰值对简单参照。** 把同一读出用在 GBT-T 的轨迹上，在德拉门上明确更接近（−10.6%）；上个月的实测峰值在 CINELDI 上明确更接近（−18.2%）；在 GoiEner 非住户和 EWELD 上，对上个月峰值的点估计为 −34.7% 和 −23.7%。
- **区间对 Chronos-2。** 放在 ankyra-lite 周围的区间在 EWELD、GoiEner 非住户和 CINELDI 上明确差于 Chronos-2 的原生分位数（80% Winkler 分数）。
- **一年上下文的基础模型。** 用 8,736 小时上下文的零样本 TimesFM 2.5 和用 8,192 小时的 Chronos-2 在住户上的逐时误差明确更低（−5.3% 和 −5.2%）（[细节](#一年上下文的基础模型)）。

它自身设计的弱点在出现的地方报告：配方 R 在德拉门和第 2–7 个预测日上的损失（[配方历史](#配方历史r-是怎样选定的以及它的负面结果)）、接续层在奥斯陆第 8–31 天的明确劣势（[接续层](#按小时看接续层)）、交付曲线与电量读数之间的差异（[读出](#读出)），以及上下文中尖峰的影响（[稳健性](#对受损上下文的稳健性)）。

## 局限

- **在九个人群上只是复用评估。** 九个人群比较中的测试集没有一个对设计是未见过的；配方 R 是在读过这些窗口之后选定的，0.2.0 又在它们上面读了一次（只定方向）。外部重跑读的是 0.1 的冻结外部检验读过一次的测试集，即此前读过的人群和一项同源跨时段检验，不是未见过的人群（[外部检验](#外部检验)）。
- **接续层的形式是在开发窗口上选定的，部分是在读过它们之后，** 四个常数在六个人群的开发窗口上拟合；三个没有开发窗口的人群在读数之前从未检验过就用了同一组常数。它的参照日回退（d* = 6）与研究的开发代码（d* = 0）不同，用在了 33 个 EWELD 测试窗口上；它处理负负荷的规则是在研究之后加的。
- **读出规则。** 电量、水平和峰值在接续层之前读出；交付的逐时曲线积分不等于电量读数（测试窗口上中位数 0.77%，最大 310%；[读出](#读出)，理论 O6）。
- **损失权重的作用很小。** 测试窗口上配方 Rw 对 R 的整月改进为 +0.31% [+0.06, +0.56]，与种子组之间的差异同一量级；在开发窗口上损失权重使峰值读出明确变差（−0.92%），在测试窗口上峰值为 +0.04%（不明确）。它只作为方向报告。
- **每个人群一个载体。** 每个检查点只为一个人群训练，也只在该人群的测试窗口上评分。检查点迁移到其他建筑或人群没有评估过。
- **借来的窗口。** 一个载体的合并训练集包含其他研究人群的窗口（至多另外八个人群和苏州园区；剑桥没有），每个都取自该人群自己的训练期或验证段，并在目标人群验证段开始之前结束；没有借来的窗口落在其他人群的测试期内（[docs/DATA.md](docs/DATA.md#windows-borrowed-from-other-populations)）。因此九个载体共享数据，彼此并不独立；起报点距检查点截点不足 186 天时，它的伪起报点会落进训练期。
- **日历问题。** 预测月的日类型按起报时已知处理，ankyra-lite（载体和接续层的参照日）和每个读日历的基线都是如此。在 **523 个 EWELD** 测试窗口中的 **26 个**上（起报点 2020-01-01），假日日历包含 2020 年 1 月 27 日才公布的春节假期延长；在 **730 个剑桥**测试窗口中的 **10 个**上（起报点 2022-09-01），包含 2022 年 9 月 10 日才公布的 2022 年 9 月 19 日银行假日。预测没有按起报时已知的日历重做；结果文件包含这些窗口（[docs/DATA.md](docs/DATA.md#calendar-caveat-holidays-announced-after-the-origin)）。
- **BDG2 与 TimesFM 2.5 预训练数据的重叠。** 七个 BDG2 站点（Bear、Bull、Cockatoo、Fox、Hog、Panther、Rat）列在 TimesFM 2.5 模型卡称为训练数据的 GIFT-Eval 预训练集合中。这可能有利于 TimesFM 和 TimesFM-X 在 BDG2 上的比较；影响多大没有测量。ankyra-lite 没有预训练权重（[docs/DATA.md](docs/DATA.md#bdg2-and-the-pretraining-collection-of-timesfm-25)）。
- **短记录。** BDG2 2017 和两个 GoiEner 人群的训练记录最短，没有开发窗口，测试集只覆盖一年中的 5–8 个月。
- **新近程度。** 载体的训练标签止于 C′，训练型基线的止于 C，晚 186 天；这是伪起报点样本外的代价。
- **基线。** 发布版训练型基线的预算是 15 轮、patience 3，LSTM-T-KAN-M 是 30 和 8。在一次放宽预算（30 轮、patience 8）下，DLinear、iTransformer、LSTM 和 PatchTST 的 36 个逐时结论中 35 个不变，36 个电量结论中有三个变为不明确（[训练预算](#稳健性训练型基线的训练预算)）；这是一个替代预算，不是搜索，拿到协变量的训练型模型没有重新训练。有一年上下文时，零样本 TimesFM 2.5 和 Chronos-2 在住户上逐时误差明确更好（[一年上下文](#一年上下文的基础模型)）。线性上下文模型的协变量不起作用。GBT-T 在 EWELD 上过冲。LSTM-T-KAN-M 不是对论文的严格复现（[十处偏离](#基线)）。
- **区间**是描述性的；区间读出不是为这个载体设计或调整的，并且放在接续层之后的交付轨迹周围。
- **训练不能逐位重复**（在 GPU 上）。六个种子取平均，并报告两组之间的差异，但主表的区间只重抽单位和月份，不包括重新训练。
- **多重性。** 21 个基线 × 9 个人群 × 几种量，得到几百个区间；明确对比的计数是描述性的，不是经过多重性校正的检验。
- **继承自锚定层**：只用于逐时数据，只有 744 小时的预测长度，起报点处需要完整的 1,344 小时上下文，类别 Commercial 没有温度特征先验（13 个 BDG2 单位属于此类），年温度谐波的相位按北半球设定。

## 外部检验

**0.2.0 版：重跑一次（2026 年 10 月 20 日）。** 0.2.0 配置在 0.1 冻结外部检验的四个已评分测试集上重跑（同样的窗口、比较对象和估计量，在那里用冻结的候选，即配方 R 六种子，读过一次），并评分一次。每个测试集用该检验的第 1 阶段载体（六个种子），在该测试集自己的训练池上用配方 Rw 继续训练，取平均，由未改动的锚定层配季节性朴素状态规则锚定，再用冻结常数的接续层校正（不重新拟合），按规则 A 读出，并用冻结的估计量评分；没有用发布的检查点。这是每个测试集的第二次读数，也是本版本的第一次。完整报告：[docs/EXTERNAL_TESTS.md](docs/EXTERNAL_TESTS.md)（英文）；每个数字都是 [`results/external_tests.csv`](results/external_tests.csv) 的一行，由 `evaluation/external_tables.py` 从评分记录复制而来。

- **LCL、UNICON 和 HKUST** 是此前读过的人群：研究中更早的预测器（包括小网络的早期版本）在那里评过分，锚定层是在那些读数之后定下的，0.1 的冻结外部检验又读过一次。HKUST 是样本较小的次要测试集。
- **GoiEner 住户跨时段：同源跨时段补充检验**（研究人群 6 的数据源中新的住户和更晚的时段），不是新的人群。
- **赫尔辛基（回顾）：已训练、锚定并封存，未评分**，重跑也没有碰它。**UCI370：跳过。**

主读数，逐时误差，六个种子（\* 明确更好；没有及格线）。它们是 15 个预先指定的名义 95% 区间中的 12 个（3 个对比 × 5 个测试集，赫尔辛基的封存），没有做多重性校正；如果没有真实差别，15 个中预计约 0.75 个会碰巧显示为明确。LCL 在 TimesFM 2.5、Chronos 和 Chronos-2 的预训练数据中；GoiEner 是否在其中没有核实；UNICON 和 HKUST 不在公布的列表上，但这不能证明不在。最后一列是 0.1 冻结候选在同样窗口上对 TimesFM 2.5 的估计（历史）。

| 测试集 | 窗口 / 单位 / 月份 | 对 TimesFM 2.5 | 对季节性朴素（周） | 对带状态规则的无载体对照 | 0.1 候选对 TimesFM 2.5 |
|---|---|---|---|---|---|
| LCL households (London) | 710 / 710 / 5 | +2.74% [−1.16, +6.48] | +24.37% [+17.71, +30.48] * | +3.13% [+2.06, +4.19] * | +2.68% |
| UNICON (La Trobe University campuses) | 450 / 59 / 23 | −2.42% [−14.33, +8.24] | +11.83% [+0.73, +21.68] * | +10.71% [+5.51, +15.62] * | −5.54% |
| HKUST campus incomer meters (secondary) | 89 / 29 / 8 | +10.88% [−9.01, +27.14] | +26.18% [+6.34, +41.81] * | +4.63% [−2.96, +11.67] | +10.24% |
| GoiEner households, out of time (supplementary) | 1,417 / 282 / 6 | +4.25% [+1.88, +6.57] * | +26.04% [+23.82, +28.20] * | +5.55% [+3.49, +7.57] * | +4.19% * |

- 在四个已评分的测试集上，12 个主区间中有 8 个明确偏向 ankyra-lite（0.1 候选为 7 个；UNICON 对周季节性朴素新变为明确），4 个不明确，没有明确不利的。
- **对零样本 TimesFM 2.5**，只有 GoiEner 检验明确（+4.25%）；在 LCL（+2.74%；LCL 在 TimesFM 2.5 的预训练数据中）、UNICON（−2.42%）和 HKUST（+10.88%）上区间都含零；不明确不等于没有差别。
- **对 0.1 冻结候选**（配方 R、六个种子、无接续层），0.2.0 整月只在 UNICON 上明确更好（+2.95%），其余三个不明确（+0.06% 到 +0.72%）；第 1 天在 LCL（+4.89%）和 UNICON（+16.67%）上明确更好。
- **第 1 个预测日。** 零样本 TimesFM 2.5 在 LCL 上第 1 天的误差仍明确更低（−10.43%；0.1 配置时为 −16.63%）；在 UNICON（−13.29%，之前 −37.30%）、HKUST（−1.43%）和 GoiEner（−1.99%）上第 1 天的对比不明确。在第 0–2 小时，TimesFM 2.5 在 LCL（−25.38%）、UNICON（−73.88%）和 GoiEner（−18.26%）上明确领先。UNICON 上 ankyra-lite 的单位平均名次为 14 个中的第三。
- **第 1 天规则，读数前写好。** 如果外部测试集上第 1 天对 TimesFM 2.5 的平均对数差距，从 0.1 配置到 0.2.0 缩小的量不到开发窗口上缩小量的三分之一，本 README 就不说 0.2.0 缩小了短期差距。结果：允许这样说（外部平均缩小量 +0.085 对数单位，判定线 +0.036，开发缩小量 +0.107；各测试集 LCL +0.055、UNICON +0.192、HKUST +0.088、GoiEner +0.004；`external_tests.csv` 的 `day-1 rule (design 3.3)` 和 `day-1 gap to TimesFM 2.5` 两块）。能说的只是第 1 天对 TimesFM 2.5 的差距缩小了；ankyra-lite 在第 1 天并不领先 TimesFM 2.5。

![外部检验](figures/fig17_external_tests.png)

*图 17. ankyra-lite 0.2.0 的外部检验：0.2.0 配置在冻结外部检验的测试集上重跑，各评分一次：三个主读数（a–c），按提前期（含第 0–2 小时）的同样对比（d–f），以及 ankyra-lite 自身各配置之间按提前期的描述性对比（g：0.2.0 对 0.1 配置；h：接续层开对关；i：配方 Rw 的损失权重对配方 R，不含接续层）。实心：95% 区间不含零；超出坐标轴的区间端点画成箭头。LCL、UNICON、HKUST：此前读过的人群；GoiEner 住户跨时段：同源跨时段补充检验；赫尔辛基：封存，未评分。*

**0.1 版（2026 年 10 月 19 日），历史。** 0.1 的冻结外部检验训练配方 R 的载体（六个种子，发布配置的三个种子作为次要读数），每个测试集评分一次；它的数字见上表最后一列和 [docs/EXTERNAL_TESTS.md](docs/EXTERNAL_TESTS.md#version-01-external-test-19-october-2026)。12 个主区间中有 7 个明确偏向 ankyra-lite，没有明确不利的。

## 检查点

仓库发布 81 个载体，即研究运行第 2 阶段的最终权重，由 `tools/export_checkpoints.py --out checkpoints` 逐字节复制：

```
checkpoints/<population>/rw/seed<s>.pt  version 0.2.0 (the default): recipe Rw, s in {0, 1, 2, 3, 4, 5}
checkpoints/<population>/seed<s>.pt     version 0.1: recipe R, s in {0, 1, 2} (unchanged)
                                        population in {bdg2, cambridge, heew, eweld, goiener_confirm, households,
                                                       oslo, drammen, cineldi}
checkpoints/MANIFEST.json               per file: SHA-256, bytes, parameters, recipe and seed group, loss weighting and
                                        its weights record, training cut (hour index and date), hour 0 of the record,
                                        store, source run, training record and stage-1 run with their SHA-256; the two
                                        configurations (release: Rw, six seeds, layer on; 0.1.x: R, three seeds, layer off)
```

- **格式。** 每个文件是 `{"model": state_dict, "epoch": 8}`（265,653 字节）。`ankyra_lite.load_carrier(path)` 构造网络并严格载入，在 CPU 上、处于评估模式；它也接受裸的 state dict。
- **核对。** 导出时用运行记录中的 SHA-256 核对每个副本，严格载入并数出 61,613 个参数。对奥斯陆和 CINELDI（修正后的数据仓库）各三个测试窗口、发布配置的每个种子，它在 CPU 上于相同的发布时刻复现保存的 GPU 载体预测；规则是相对差低于 10⁻⁴，`MANIFEST.json`（`verification`）中记录的最大值为 2.0 × 10⁻⁵（CINELDI，种子 2）。0.1 版的 27 个文件与 0.1.1 逐字节相同。
- **六个种子，一个截点。** 按种子升序载入六个文件（`ankyra_lite.checkpoint_paths(population)`；发布开关在 `ankyra_lite/_release.py` 中），只传一次该人群的截点；集成平均按这个顺序对成员求和。
- **配合该人群自己的截点使用。** 一个检查点属于一个人群及其训练截点。用 `cut=` 传入截点（以第 0 小时为下表所列时刻的记录的小时下标），并使用至少在其后 186 天的起报点：

| 文件夹 | 人群 | 记录的第 0 小时 | 训练截点（`training_cut_hour`） | 训练截点（日期） | 数据仓库 |
|---|---|---|---:|---|---|
| `bdg2` | BDG2 2017 | 2016-01-01 | 9,408 | 2017-01-27 | 原始 |
| `goiener_confirm` | GoiEner 非住户 | 2018-01-01 | 7,176 | 2018-10-27 | 原始 |
| `households` | GoiEner 住户 | 2018-01-01 | 8,640 | 2018-12-27 | 原始 |
| `cambridge` | 剑桥 | 2000-01-01 | 113,856 | 2012-12-27 | 原始 |
| `heew` | HEEW 亚利桑那 | 2014-01-01 | 38,616 | 2018-05-29 | 原始 |
| `eweld` | EWELD | 2016-01-01 | 26,928 | 2019-01-27 | 原始 |
| `oslo` | 奥斯陆 | 2012-01-01 | 63,456 | 2019-03-29 | 原始 |
| `drammen` | 德拉门 | 2018-01-01 | 18,912 | 2020-02-28 | 修正后（其训练池含 CINELDI 窗口） |
| `cineldi` | CINELDI | 2019-01-01 | 15,240 | 2020-09-27 | 修正后（第 0 小时确为 2019-01-01） |

```python
from ankyra_lite import checkpoint_paths, forecast, load_carrier
carriers = [load_carrier(p) for p in checkpoint_paths("oslo")]                # checkpoints/oslo/rw/seed0..5.pt
# history: an Oslo school's ankyra_lite.anchor.History whose hour 0 is 2012-01-01, ending at an origin >= 63,456 + 4,464;
# sigma: Oslo's temperature-anomaly scale fitted on days before the cut
r = forecast(history, carriers, group="Public", temp_sigma_std=sigma, dst_region="EU", cut=63456)

# version 0.1 (recipe R, three seeds, no continuation layer):
old = [load_carrier(f"checkpoints/oslo/seed{s}.pt") for s in (0, 1, 2)]
r01 = forecast(history, old, group="Public", temp_sigma_std=sigma, dst_region="EU", cut=63456, continuation=False)
```

- 不要把自己的训练运行写进 `checkpoints/`：`.gitignore` 忽略除发布的 81 个以外的所有权重文件。

## 测试

不需要 `pytest`；测试用的是 `unittest`。在仓库根目录下：

```bash
python -m unittest discover -s tests -t .      # 275 tests
python -m unittest discover -s theory -t .     # 111 checks of the exact properties
```

各模块的测试数与检查内容见英文 README 的 [Tests](README.md#tests) 一节（同样的表格）：

| Module | Tests | What it checks |
|---|---:|---|
| `tests/anchor/` | 62 | the anchoring layer's own tests, ported to the copy (no information after the origin, the handover, the state rules, gap tolerance, analog shapes, readouts, the reference estimator) |
| `tests/test_anchor_equivalence.py` | 14 | the copy is byte-identical to ANKYRA 2.2.1 and gives bit-identical forecasts, readouts and metrics (needs the ANKYRA checkout) |
| `tests/test_carrier_unit.py` | 14 | parameter count, shapes, no read at or after the issue time, the pseudo-origin rule, the ensemble, the end-to-end forecast, the state rule |
| `tests/test_carrier_equivalence.py` | 7 | features, initialisation and forecasts against the research code and the stored forecasts (needs the study's data) |
| `tests/test_train.py` | 19 | sampler, splits, caps, pools, the returned span and one optimisation step of each stage against the research training loop (part needs the study's data) |
| `tests/test_train_rw.py` | 19 | recipe Rw: the reduction to recipe R with equal weights, the weights against the study's rule bit for bit, the row-alignment check (part needs the study's research folder or the 0.1.1 checkout) |
| `tests/test_continuation.py` | 24 | the continuation layer against its definition, the frozen constants and digest, the release rules, and the study's reference implementation bit for bit (part needs the study's research folder) |
| `tests/test_forecast_v2.py` | 17 | the forecast with the layer off against version 0.1.1 field by field (in the same process and in a separate one; needs the 0.1.1 checkout), readout rule A, the bypasses, six-seed ensembles and the release switches |
| `tests/test_store_arrays.py` | 9 | the evaluation's store reader against the study's loader, per population (needs the study's data) |
| `tests/test_compare_baselines.py` | 42 | the comparison script's guards, the products (and the continued products against the release layer), the checkpoint export, and the result files: columns, forecaster set, flags, labels, no absolute paths, the added rows, the gates of the fields cache and of the handover rebuild (part needs the ANKYRA release files, the study's data or `results/`) |
| `tests/test_readouts_lite.py` | 13 | the readouts around a forecast (interval, blocks, raw trajectory, energy added by the projection and by the continuation layer, peak), the history-only configuration, the fields read through to the anchoring layer, no information after the origin end to end, the top-level names and the two examples (synthetic data, untrained carrier) |
| `tests/test_evaluation.py` | 15 | the scoring module on artificial data |
| `tests/test_external_tables.py` | 12 | the external-test table: columns, forecaster names, no absolute path, the sealed Helsinki set never read, equality with the score records field by field, no contrast of the scorer's other output files, the primary table of docs/EXTERNAL_TESTS.md (part needs the study's score records) |
| `theory/test_operators.py`, `theory/test_carrier_properties.py`, `theory/test_weighting_properties.py`, `theory/test_continuation_properties.py` | 31, 27, 15, 36 | P1–P20, C1–C10, C11–C14 and O1–O9, counterexamples included |

2026 年 10 月 22 日（0.2.1 版；CPU，单线程）完整运行通过：收集到的 275 项测试中运行了 269 项（整类跳过的只计一次），没有失败，14 项跳过；111 项理论检验全部通过。与 0.1 版的对照测试在由提交 c27affc 导出的 0.1.1 副本上运行（`ANKYRA_LITE_REFERENCE_REPO`、`ANKYRA_LITE_REFERENCE_TRAIN`），全部通过。跳过的是：9 项数据存储数组检查（空闲内存不足，需要 5、6.5 或 11 GB；10 月 21 日的运行中另行运行 7 项通过，EWELD 和住户没有运行），以及 5 项本次运行缺少研究记录或文件的测试。需要 ANKYRA 仓库副本、0.1.1 副本或研究数据的测试，在缺少这些时（如 CI 上）会说明原因并跳过。环境变量：`ANKYRA_ORIGINAL_ROOT`（ANKYRA 仓库副本）、`ANKYRA_LITE_RESEARCH_ROOT`（研究文件夹）、`ANKYRA_LITE_REFERENCE_REPO`（本仓库的 0.1.1 副本）、`ANKYRA_LITE_REFERENCE_TRAIN`（0.1 版的 `ankyra_lite/carrier/train.py`）、`ANKYRA_LITE_RESULTS`（另一个要检查的结果目录）和 `ANKYRA_LITE_TEST_MIN_FREE_GB`（读数据的测试要等到的空闲内存）。

## 理论：精确性质

![一个测试窗口上的精确性质](figures/fig7_operators.png)

*图 7. 锚定层在图 6 那个剑桥窗口上的精确性质：从单独的载体到 ankyra-lite 接续层之前的锚定轨迹 F 的分块替换（a，P4），历史支撑（b，P7），收缩界（c，P8、P9），截断对保电量投影（d，P11、P12），峰值读出及其误差分解（e、f；P15、P16）。*

这个构造之所以可审计，是因为每一步都有一条明确的性质。它们放在预测器旁边的 [theory/](theory/README.zh-CN.md) 中，包本身不导入它们：

- **P1–P20，锚定层**（ANKYRA 2.2.1 理论中的性质，只改了导入路径）：精确的分块恒等式与替换（P1–P6），权重需要多少历史、收缩让它们能移动多远（P7–P10），为什么电量在截断之前读出（P11–P13），峰值算子（P14–P18），为什么报告几种估计量（P19），以及对水平权重做强收缩的最坏情况背景（P20）。
- **C1–C14，载体网络与配方 Rw**：非负、均值等于水平的轨迹（C1、C2）；份额和指针注意力都是分布（C3）；发布时刻及之后没有输入（C4）；因果、尊重掩码的编码器（C5、C6）；尺度等变（C7）；没有单位编号（C8）；复制指针从零开始、从不移动水平（C9）；186 天的间隔使每个打分的伪起报点预测都是样本外的（C10）；以及 0.2.0 新增的：损失权重相等时退化为配方 R（C11），保持每个人群的抽取份额（C12），有限、为正且有下限（C13），只读每个窗口自己起报点之前的上下文（C14）。
- **O1–O9，接续层**（0.2.0 新增）：保持符号和零（O1）；校正被 E(h) = 0.479 e^(−h/12) + 0.330 e^(−h/168) 界住（O2）；不读起报点及之后的负荷，只读第一个预测周的日历（O3）；状态窗口不动（O4）；关闭时原样交付接续层之前的轨迹（O5）；电量、水平和峰值在它之前读出，交付曲线之和与电量读数相差已知的项（O6）；一致性和单调性（O7）；尺度等变（O8）；同一工作日类别的参照日，以及常数的唯一摘要（O9）。

[theory/PROOFS.md](theory/PROOFS.md) 给出陈述、证明和反例；`theory/operators.py` 把 P1–P20 写成函数；四个检验文件共有 111 项数值检验。这些性质都不是精度保证；精度是测出来的。

## 可复现性

- **评分代码核对。** 在把 ankyra-lite 放到参照位置之前，`evaluation/compare_baselines.py` 用发布版自己的参照预测器重新计算 ANKYRA 2.2.1 发布版十张结果表中截点后窗口的各行，并按公布时的舍入，逐字段与发布仓库提交 `487e0d2` 的文件比较（用 `git show` 只读地读取）。十张全部复现：94,131 个比较字段中 0 处不一致（[`results/REPRODUCTION_CHECK.json`](results/REPRODUCTION_CHECK.json)）。不能复现的表不会写出，重新计算出的发布版数字不在这里的任何文件中。
- **产物核对。** 对每个人群，锚定后的产物必须正好具有面板的测试窗口、状态标记和目标，否则脚本停止。由锚定后的产物重新计算的各人群 R 对 F、R 对从零训练阶段的对比，在 9 个人群中的 9 个上与研究自己的评分器结果相同（最大差 0.0 个百分点；0.1 版的运行，`study_scorer_crosscheck`）。在 0.2.0 中，对六种子集成重新锚定，在接续层之前和之后都逐位复现了每个保存的产物（`results/compare_baselines.json` 中的 `extra` → `reanchor` → `ens`）。
- **接续层核对。** 对每个人群，产物清单列出接续层之前和之后的锚定产物；`evaluation/compare_baselines.py --check-products` 在不读任何目标的情况下核对：两者的窗口、状态标记、水平和峰值相等，状态窗口不动，其余每个窗口的轨迹都是发布的接续层作用在之前产物上的结果，在 float32 下逐位相同（`results/compare_baselines.json` 中的 `continuation_gate`；九个人群全部通过）。
- **包对研究代码。** 锚定层与 ANKYRA 2.2.1 逐字节相同，并给出逐位相同的预测；载体的特征、初始化和预测，以及训练配方，都复现研究代码（[测试](#测试)）。在 CPU 上，本包复现保存的 GPU 预测，相对差约为 10⁻⁶（[耗时](#每个窗口的耗时)）。
- **哪些预测被评分。** `evaluation/products.json` 按人群列出配置（配方 Rw、六个种子、接续层）、接续层之前和之后的锚定后六种子产物、两个三种子组的产物，以及它们背后的研究运行（路径相对研究文件夹）；由 `tools/products_for_configuration.py` 写出。`evaluation/products_0_1.json` 是 0.1 版的清单，`python -B evaluation/compare_baselines.py --recipe r010 --out DIR` 由它重新生成 0.1 的表格。
- **核对一个数字。** 上面表格中的每个数字都是 `results/` 中某个文件的一行。例如总览中剑桥那一行：

  ```python
  import csv
  rows = [r for r in csv.DictReader(open("results/benchmark_pairwise.csv", encoding="utf-8")) if r["set"] == "Cambridge"]
  print(sum(float(r["um_high"]) < 0 for r in rows), "of", len(rows))      # 21 of 21 resolved better
  ```
- **哪些不能只靠本仓库重新运行。** `evaluation/compare_baselines.py`、`tools/export_checkpoints.py` 和依赖数据的测试，要读研究的本地数据仓库、保存的预测和 ANKYRA 仓库副本（只读；[evaluation/README.md](evaluation/README.md)）；`evaluation/external_tables.py` 要读外部检验的评分记录（[docs/EXTERNAL_TESTS.md](docs/EXTERNAL_TESTS.md#files)）。`python -m evaluation.run_example` 用发布的检查点在十二栋人造建筑上演示数据流程（其数字没有意义），`python -B figures/make_figures.py --strict` 从 `results/` 重新生成所有图。

## 版本历史

- **0.2.1（2026 年 10 月 22 日）**：代码审核后的维护版本；默认输出、检查点、结果和图都不变（[更正说明](docs/CODE_AUDIT_CORRECTIONS_20261022.md)）。`evaluation/cost_timing.py` 和 `evaluation/robustness.py` 中关闭接续层的开关现在真正生效（发布的运行都开启接续层）；较早 744 小时特征块中缺失的日类型改用星期回退；Wilcoxon 的方向改由每一对自己的符号秩判断（186 个 Holm 显著的比较方向全部不变）；自定义接续常数极端时抛出 `ValueError`，不再给出非有限的小时；重锚定缓存绑定内容身份并加独占认领；外部检验的子群按其构造改称“没有第一阶段自身训练窗口的单位”；更正方法、读出和理论文档；回归测试见 `tests/test_audit_fixes.py`。
- **0.2.0（2026 年 10 月 21 日）**：载体改用配方 Rw 训练（第二阶段加逆噪声损失权重）并对六个种子取平均；在锚定层和状态规则之后加入起报接续层（`ankyra_lite.continuation`，四个冻结常数），电量、水平和峰值在它之前读出（读出规则 A），结果新增两个字段（`trajectory_before_continuation_kw`、`continuation`）；发布开关在 `ankyra_lite/_release.py` 中，另有 `ankyra_lite.checkpoint_paths`；54 个新检查点（`checkpoints/<population>/rw/`；0.1 版的 27 个不变）；比较全量重新生成（新文件 `continuation_layer.csv`、`continuation_record.csv`）；外部检验重跑一次；精确性质 C11–C14 和 O1–O9；与图 9 并列的图 9c。接续层的两条发布规则不改变任何研究数字：参照日回退 d* = 6，以及对负负荷改为标记而不报错。在测试窗口上对 0.1 版（此前读过；只定方向）：整月逐时 +1.15%，第 1 天 +13.97%，电量和峰值不明确。
- **0.1.1（2026 年 10 月 19 日）**：读出接口（`ankyra_lite.readouts`）、纯历史配置（`forecast(history, None)`）、自己单位的示例（`examples/own_unit.py`）、补充分析（交接、按预测周、日内信任、训练预算、通用组合、一年上下文）、图 0（优势汇总）和重新绘制的图 1。0.1.0 的预测、检查点和结果不变。
- **0.1.0（2026 年 10 月 19 日）**：第一个版本。复制指针载体网络（61,613 个参数），配上逐字节相同的 ANKYRA 2.2.1 锚定层和季节性朴素状态规则；发布的训练配方 R（第 1 阶段从零训练，在 93 天验证段上早停；第 2 阶段归还验证段训练 8 轮；三个种子）；27 个检查点及其清单；CINELDI 用修正后的数据仓库，德拉门载体也用它重新训练；与 21 个基线（发布版的 20 个和 LSTM-T-KAN-M）在九个人群上的比较，用发布配方全量重新生成；载体的精确性质 C1–C10。
- **0.1.0 之前的设计历史**（研究中的各个载体，都由同一个锚定层锚定；`ablation.csv` 和 `recipe_history.csv`）：一个更早的、没有复制指针、在按周窗口上训练的结构化网络；同一网络去掉复制指针、用从零配方训练；更早的几项继续训练研究（不加权和加权的损失，有和没有预热）；只有从零训练阶段的复制指针网络；以及继续训练方案 F、V 和 R，其中发布的是 R。

## 来源、许可与引用

- **锚定层。** `ankyra_lite/anchor/` 是 ANKYRA 2.2.1 的 `ankyra` 包（仓库 [XinanQ/Building_load_ANKYRA](https://github.com/XinanQ/Building_load_ANKYRA)，提交 `487e0d2`，其 `ankyra/` 目录树与标签 `v2.2.1` 相同），逐字节复制；[`PROVENANCE.md`](ankyra_lite/anchor/PROVENANCE.md) 列出每个文件的哈希。它的文档字符串仍写作 `ankyra.*`，请读作 `ankyra_lite.anchor.*`。不包含 `ankyra.timesfm_adapter`。这 17 个复制的文件从不改动。
- **载体。** 网络、特征、推理和训练代码复现研究代码（参数名和顺序、初始化一致，特征逐位一致）。
- **测试窗口、发布版的 20 个基线和评分函数**都来自 ANKYRA 2.2.1 发布版。
- **许可。** MIT（[LICENSE](LICENSE)），Copyright (c) 2026 Xinan Qin；锚定层使用同一许可。LSTM-T-KAN 的代码（MIT）不包含在内；只对它的预测评了分。
- **数据。** 评估使用公开数据集：BDG2（Building Data Genome Project 2）、剑桥大学校产档案、HEEW（亚利桑那州立大学）、EWELD、GoiEner、COFACTOR-SBHUB Oslo 和 COFACTOR Drammen、CINELDI，以及苏州工业园区（只用作训练窗口），都使用 CC BY 4.0 许可；来源和整理见 [docs/DATA.md](docs/DATA.md)。这些数据都没有再分发；检查点包含在这些数据上拟合的权重，不包含数据本身。`results/` 中的示例窗口来自剑桥大学校产档案（CC BY 4.0）。
- **外部检验的数据**（[docs/EXTERNAL_TESTS.md](docs/EXTERNAL_TESTS.md)）：Low Carbon London 智能电表住户（UK Power Networks，London Datastore，Creative Commons Attribution）；UNICON（La Trobe University；CC BY-NC-SA 4.0，仅限研究使用，所以数据和在其上训练的载体都不在本仓库中）；HKUST 智能电表数据库（Dryad，CC0）；GoiEner 住户的跨时段检验（GoiEner 档案，CC BY 4.0）。这些数据都没有再分发。HKUST 和 UNICON 的气温是 ERA5 再分析：Contains modified Copernicus Climate Change Service information [2026]；ERA5 hourly data（Hersbach et al., 2020, doi:10.1002/qj.3803；dataset doi:10.24381/cds.adbb2d47），经 Open-Meteo historical weather API 获取（Zippenfenig, doi:10.5281/zenodo.7970649, CC BY 4.0）。
- **参考文献**（[一眼看懂](#一眼看懂)中两个对比与预测包含检验的区别）：
  - Chong, Y. Y., Hendry, D. F. (1986). Econometric evaluation of linear macro-economic models. *Review of Economic Studies* 53(4), 671–690.
  - Harvey, D. I., Leybourne, S. J., Newbold, P. (1998). Tests for forecast encompassing. *Journal of Business & Economic Statistics* 16(2), 254–259.
- **引用。** [CITATION.cff](CITATION.cff)；也请引用 ANKYRA（列在其中的 references 下）。存档 DOI 尚未分配。

问题和错误报告：[issue 跟踪页](https://github.com/XinanQ/Building_load_ANKYRA_lite/issues)。
