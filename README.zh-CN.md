# Building_load_ANKYRA_lite

**不用基础模型的月前逐时负荷预测：一个 61,613 个参数的载体网络，每个人群单独训练，并锚定到每个单位自己的历史**

[English](README.md) · [安装与运行](#安装与运行) · [快速上手](#快速上手) · [用自己的数据](#输入与输出) ·
[模型卡](#模型卡) · [训练配方 R](#训练配方-r) · [数据划分](#数据划分) ·
[评估](#评估) · [检查点](#检查点) · [方法](docs/METHOD.md) ·
[评估细节](docs/EVALUATION.md) · [外部检验](docs/EXTERNAL_TESTS.md) · [数据](docs/DATA.md) ·
[理论](theory/README.zh-CN.md) · [结果数据](results/) · [图](figures/)

中文页面只有本页和 [theory/README.zh-CN.md](theory/README.zh-CN.md)；其余文档（`docs/`、`results/README.md`、`evaluation/README.md`、`figures/README.md`、`theory/PROOFS.md`）为英文。

ankyra-lite（包名 `ankyra-lite`，发行名 `building-load-ankyra-lite`，导入名 `ankyra_lite`）根据一个单位（一栋楼、一块电表、一个用电点或一户住户）截至起报点的自身记录，预测其后 744 小时（31 天）的逐时负荷。它由两部分组成。一个小型**载体网络**（复制指针结构化负荷网络，61,613 个参数，每个人群从零训练，不用任何预训练权重）根据单位的历史和已知日历，在起报点和六个更早的伪起报点上预测这个月。一个**锚定层**（从 ANKYRA 2.2.1 包逐字节复制而来）在单位已经经历过的月份上给载体的伪起报点预测打分，并用这些误差逐块（月水平、日路径、日内形状）在载体和单位的长期历史之间定权。输出是 744 小时的轨迹、月电量和月峰值读出。用三个种子时，一次预测在单个 CPU 线程上约需 0.58 秒（[耗时](#每个窗口的耗时)），不下载任何东西。本仓库包含这个包、27 个训练好的检查点（九个人群 × 三个种子）、训练配方、评分代码、与 21 个外部基线在九个人群上的评分比较、图，以及两部分各自的精确性质。

> **状态（0.1.0 版，2026 年 10 月 19 日）。**
>
> - **证据性质：在此前读过的测试窗口上的复用评估。** 这些测试窗口在发布配置定下之前已经读过，部分设计（复制指针的方向、季节性朴素状态规则）是看过它们之后选定的。本页九个人群比较中的每个数字描述的都是发布的预测器在这些窗口上的表现，没有一个是首次检验（[先读这一节](#先读这一节)）。外部检验的数字有各自的结果身份，与每个数字写在一起。
> - **配方是在一次方向性读数之后选定的。** 发布的训练配方 R（验证段归还）是 2026 年 10 月 19 日、在这些测试窗口上对验证段归还研究做了一次方向性读数之后选定的。它的负面结果是发布内容的一部分，并已披露（[配方历史](#配方历史r-是怎样选定的以及它的负面结果)）。
> - **外部检验：四个测试集各评分一次，一个封存。** 每个测试集都用发布配方单独训练载体：LCL、UNICON 和 HKUST（本候选在此前被读过的人群上的首读），以及 GoiEner 住户跨时段（同源跨时段补充检验）；赫尔辛基回顾已训练并封存，尚未评分；UCI370 本轮跳过（[外部检验](#外部检验)，完整报告 [docs/EXTERNAL_TESTS.md](docs/EXTERNAL_TESTS.md) 为英文）。
> - **含权重。** 九个人群所有结果背后的 27 个检查点都在 [`checkpoints/`](checkpoints/MANIFEST.json) 中（[检查点](#检查点)）；外部检验各自训练了载体，这些载体不随仓库发布。

## 一眼看懂

ankyra-lite 与 21 个外部基线（ANKYRA 2.2.1 发布版的 20 个基线加上 LSTM-T-KAN-M；共 22 个预测器）在九个人群测试窗口上的比较：1,891 个单位的 5,716 个窗口。“明确”指该对比的 95% 单位与月份自助区间不含零。

| | 结果 | 范围 | 文件 |
|---|---|---|---|
| 在 22 个预测器中的位置 | **全部九个**人群上单位平均名次都最低 | 测试窗口 | `benchmark_mean_unit_rank.csv` |
| 逐时误差 | 每个人群上明确好于 21 个基线中的 12–21 个（189 个对比中的 149 个）；明确更差的有 2 个对比，都在 GoiEner 住户上（TimesFM −5.6%，Chronos-2-X −3.8%） | 九个人群 | `benchmark_pairwise.csv` |
| 月电量 | 每个人群上明确好于 21 个中的 4–20 个（189 个中的 105 个）；没有明确更差的 | 九个人群 | `energy_error.csv` |
| 秩检验 | 平均名次 5.496，下一个预测器（逐单位岭回归）为 7.438；Nemenyi 临界差 0.786 | 六个测试人群合并，1,762 个单位 | `rank_tests.csv` |
| 峰值读出 | 峰值误差比它自身轨迹的最大值低 18.1–67.9%；在八个人群上明确（奥斯陆除外） | 九个人群 | `peak_readout.csv` |
| 耗时 | 每次预测 0.584 秒（载体在 7 个发布时刻 × 3 个种子、锚定层、峰值读出） | 单个 CPU 线程，6 个奥斯陆窗口 | `cost_per_window.csv` |

**它的边界在哪里。**

- **是复用评估，不是检验。** 这些窗口此前已经读过。外部检验是本候选在此前被读过的人群上的首读和一项同源跨时段检验，不是在未见过的人群上的检验；在那里的第 1 个预测日，零样本 TimesFM 2.5 在 LCL（LCL 在 TimesFM 2.5 的预训练数据中）和 UNICON 上的逐时误差明确更低（[外部检验](#外部检验)）。
- **住户的逐时误差。** 在 GoiEner 住户上，零样本 TimesFM 和 Chronos-2-X 的逐时误差明确更低；ankyra-lite 在那里的月电量误差和峰值误差明确低于全部四种零样本基础模型变体。
- **月初几天。** 在第 1 个预测日，它在六个测试人群上排 22 个中的第 4 到第 15，在每个人群上都落后于零样本基础模型。
- **四个人群上对零样本基础模型。** 在 BDG2 2017、GoiEner 非住户、HEEW 和 EWELD 上，与 TimesFM、TimesFM-X、Chronos-2 或 Chronos-2-X 的逐时对比没有一项在任一方向上明确，有几项点估计为负。
- **配方有已披露的损失**：与不归还验证段的同一继续训练相比，德拉门 −2.90%，有开发窗口的六个人群在第 2–7 个预测日上 −1.35%（明确）。
- **每个人群一个载体。** 检查点没有在它自己以外的任何人群上评估过。

![ankyra-lite 的结构](figures/fig1_architecture.png)

*图 1. 结构：起报时已知的输入、载体网络、它在起报点和六个伪起报点上的预测、锚定层（水平权重、交接权重、日内信任、状态窗口）和输出。每次预测的耗时取自 `results/cost_per_window.csv`。*

**简要原理。** 术语的定义见[本页用到的术语](#本页用到的术语)。

- **载体**把单位最近 366 天读作日 token，把更早的 24 个 744 小时块读作月 token，再读入 31 个预测日的日历。一个因果空洞卷积编码器和一个注意力读出，接到一个能量预算解码器上：正的月水平乘以日份额再乘以小时份额，所以预测非负，均值就是水平。复制指针让每个预测日借用单位自己最近完整日的日内曲线和相对日均值。
- **每次预测要发布七次载体预测**：在起报点 o 和 o − 744k（k = 1 … 6）处。每个伪起报点的那个月在起报时已经观测到，所以锚定层可以在已经发生的月份上，拿载体和单位自己的历史作比较、打分。
- **锚定层**把一个预测精确拆成水平、居中的日路径和日内形状。它把载体的窗口均值与六个历史水平候选放在一起定权，按周从载体取一部分日均值（交接），并把单位的相似日形状混入载体的日内形状；每个权重都来自单位自己已完成的伪预测，并向先验收缩。
- **测试集从载体训练截点之后 186 天开始**，这样锚定层打分的每个伪起报点预测对载体都是样本外的。
- **预测时不训练任何东西。** 载体每个人群训练一次；锚定层在每个起报点上根据单位的记录估计逐单位的权重。

## 目录

- **使用：** [仓库结构](#仓库结构) · [安装与运行](#安装与运行) · [快速上手](#快速上手) ·
  [输入与输出](#输入与输出)（[输入](#输入) · [输出](#输出) ·
  [需要多少历史](#需要多少历史) · [什么时候不该用](#什么时候不该用)） · [检查点](#检查点)
- **原理：** [模型卡](#模型卡) ·
  [锚定层怎样使用载体](#锚定层怎样使用载体) ·
  [训练配方 R](#训练配方-r) · [数据划分](#数据划分) · [本页用到的术语](#本页用到的术语) ·
  [起报点之后无信息](#起报点之后无信息) · [基线](#基线)
- **读结果：** [评估](#评估)（[先读这一节](#先读这一节) · [总览](#总览) ·
  [零样本基础模型](#与零样本基础模型比较) · [秩检验](#秩检验) ·
  [常规指标](#常规指标) · [按预测日](#按预测日) ·
  [月电量](#月电量与月内累计电量) · [峰值](#峰值读出) ·
  [误差分块](#误差分块与尺度化误差) · [区间](#预测区间) ·
  [各部分带来什么](#各部分带来什么) · [种子](#种子) ·
  [锚定层中的其他载体](#同一锚定层中的其他载体) ·
  [历史长度](#历史长度与锚定记录) · [稳健性](#对受损上下文的稳健性) ·
  [耗时](#每个窗口的耗时) · [配方历史](#配方历史r-是怎样选定的以及它的负面结果)） ·
  [ankyra-lite 的薄弱之处](#ankyra-lite-的薄弱之处) · [局限](#局限) · [外部检验](#外部检验)
- **核对：** [测试](#测试) · [理论](#理论精确性质) · [可复现性](#可复现性) ·
  [版本历史](#版本历史) · [来源、许可与引用](#来源许可与引用)

## 仓库结构

```
ankyra_lite/                 the forecaster (import ankyra_lite)
  forecast.py                forecast(): carrier forecasts -> anchoring layer -> state rule -> peak readout; ForecastResult
  carrier/                   the carrier network
    model.py                 CopyPointerNetwork, build_carrier, load_carrier, forecast_loss, param_groups (61,613 parameters)
    features.py              the carrier's inputs at one issue time, from data before it only
    inference.py             carrier_forecasts (origin and pseudo-origins), forecast_windows (batched), pseudo_origins
    train.py                 the training recipe: TrainConfig, recipe(), prepare_source, build_pool, train_one, train_ensemble
  anchor/                    byte-identical copy of the ANKYRA 2.2.1 anchoring layer (17 files; PROVENANCE.md lists their hashes)
checkpoints/                 the 27 released carriers (9 populations x 3 seeds) and MANIFEST.json (hashes, cuts, source runs)
examples/quickstart.py       end to end on an artificial building with the shipped checkpoints
tests/                       176 tests: the anchoring layer's own tests, equivalence with the original package and the
                             research code, the carrier, the training recipe, the evaluation module, the comparison script
                             and the external-test table
theory/                      the exact properties: P1-P20 (anchoring layer) and C1-C10 (carrier), proofs, operators, 58 checks
evaluation/                  the scoring module, a runnable example, and the scripts that wrote results/ (need the study's data)
results/                     every scored statistic (CSV / JSON), SUMMARY.md, the reproduction record; README.md lists the columns
figures/                     make_figures.py and the figures (PDF and PNG), drawn from results/ (Figure 7 also from theory/)
docs/                        METHOD.md (network, recipe, anchoring, constants), EVALUATION.md (protocol, baselines, full tables),
                             DATA.md (sources, preparation, splits, borrowed windows), EXTERNAL_TESTS.md (the external tests)
tools/export_checkpoints.py  the export of the 27 checkpoints with their verification (needs the study's data)
.github/workflows/tests.yml  CI: tests, theory checks, quickstart, evaluation example, figures (Python 3.10-3.13, CPU)
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
- CI（`.github/workflows/tests.yml`）在 Python 3.10、3.11、3.12 和 3.13（Ubuntu，只用 CPU 的 PyTorch）上运行两套测试、快速上手和评估示例，并在 Python 3.11 上重新生成图。需要研究本地数据或 ANKYRA 仓库副本的测试在那里会跳过。
- 发布的图用 Python 3.11、Matplotlib 3.11.0 和 NumPy 2.4.4 绘制（[figures/README.md](figures/README.md)）。快速上手和本页的代码于 2026 年 10 月 19 日在 CPU 单线程上用 Python 3.11、NumPy 2.4.4 和 PyTorch 2.11 运行过。
- 发布的检查点来自研究在 GPU 上的运行；这些运行所用的软件包版本没有记录在本仓库中。

## 快速上手

```bash
python -B examples/quickstart.py                          # checkpoints/cambridge/seed*.pt, an artificial building
python -B examples/quickstart.py --population households  # the three checkpoints of another population
python -B examples/quickstart.py --checkpoints DIR        # every seed*.pt in DIR, averaged
python -B examples/quickstart.py --untrained              # an untrained network of the same shape (format only)
```

这个脚本载入剑桥的三个检查点，用 `ankyra_lite.anchor.synthetic` 构造一栋有 16,000 小时历史的人造办公楼，计算载体在起报点和六个伪起报点上的预测，做锚定，然后打印各项读出和预测背后的权重。它不需要任何数据，一次预测在单个 CPU 线程上约需一秒。输出节选（列表压缩成一行）：

```
 "carrier": "3 checkpoint(s) in cambridge/: seed0.pt, seed1.pt, seed2.pt",
 "carrier_parameters": 61613,
 "carrier_issue_times_k": [0, 1, 2, 3, 4, 5, 6],
 "state_window": false,
 "energy_kwh": 17309.978,
 "level_kw": 23.2661,
 "peak_kw": 28.001,
 "interval_pseudo_windows": 12,
 "lead_week_weights_on_carrier": [0.231, 0.214, 0.216, 0.205],
 "pseudo_origin_pairs": 6,
 "within_day_trust_on_analog_shape": [0.5, 0.5, 0.5, 0.5],
 "within_day_pseudo_origin_triples": 3,
```

这些检查点属于剑桥人群及其训练截点，而人造建筑几乎是严格周期的（日内信任停在上限 ½），所以输出说明的是格式，不是典型数值。各行的含义见[输出](#输出)。

同样的步骤用 Python 写（已从仓库根目录完整运行核对过）：

```python
import numpy as np
from ankyra_lite import load_carrier, carrier_forecasts, forecast
from ankyra_lite.anchor.synthetic import synthetic_history      # an artificial building, for the example only

carriers = [load_carrier(f"checkpoints/cambridge/seed{s}.pt") for s in (0, 1, 2)]   # one population, three seeds
history = synthetic_history(16000)       # or ankyra_lite.anchor.History(load_kw, temperature_c, day_types, start_timestamp)

# step 1: the carrier's forecasts at the origin (k = 0) and at the pseudo-origins origin - 744 k (k = 1 ... 6)
load = np.asarray(history.load_kw, dtype=np.float64)
F = carrier_forecasts(carriers, load, history.temperature_c, history.day_types, "Office",
                      history.start_timestamp, unit=0, origin=len(load))
print(sorted(F))                         # [0, 1, 2, 3, 4, 5, 6]

# step 2: the anchoring layer on those forecasts, the state rule and the peak readout
r = forecast(history, F, group="Office", temp_sigma_std=0.25, dst_region="EU")
print(r.trajectory_kw.shape, r.state_window, r.level_kw, r.energy_kwh, r.peak_kw)
print(r.lead_week_weights, r.pseudo_pairs)   # read through to the anchoring layer's own result
```

第 1 步和第 2 步可以合成一次调用：`forecast(history, carriers, group="Office", temp_sigma_std=0.25, dst_region="EU")` 给出相同的轨迹。传入载体列表即为集成（各成员预测的简单平均）；只传一个载体也可以。在某个人群自己的记录上预测时，用 `cut=` 传入该人群的训练截点（[检查点](#检查点)）。

## 输入与输出

`ankyra_lite.forecast` 预测 `load_kw` 最后一个值之后的 744 小时。先把单位放到一个规则的逐时网格上：两个部分都不做重采样，不填补起报点处的上下文，也不转换时区或夏令时（[docs/DATA.md](docs/DATA.md#the-input-format)）。违反锚定层约定的输入会报错。

### 输入

记录 `ankyra_lite.anchor.History`（锚定层的输入记录，与 ANKYRA 2.2.1 相同，未改动）：

| 字段 | 要提供什么 | 载体怎样读它 | 锚定层怎样读它 |
|---|---|---|---|
| `load_kw` | 逐时平均负荷，单位 kW；下标 0 是第一年的 1 月 1 日 00:00；最后一个值是起报点前一小时；缺测处为 NaN | 只读每个发布时刻之前的小时；负值或非有限值算作缺测；没有任何观测小时的日子被屏蔽 | 最后 1,344 小时必须完整观测（从不填补；有缺口就报错）；更早的、在同一个 744 小时块内不超过 6 小时的缺口，在其估计器读取的副本中做插值 |
| `temperature_c` | 同样各小时的室外逐时气温，单位 °C | 只读每个发布时刻之前的小时，经日均值及其覆盖率读入；允许缺测 | 起报点之前处处有限；不需要未来气温 |
| `day_types` | 历史加上 744 个预测小时，每小时一个整数：周一 = 0 … 周日 = 6，法定假日 = 7 | 读到每个发布时刻的预测月为止（日历按起报时已知处理） | 同左 |
| `start_timestamp` | 第一年的 1 月 1 日 00:00，带 `+00:00` 偏移，例如 `"2019-01-01T00:00:00+00:00"`（固定偏移的网格） | 每个 token 的年相位和记录的绝对时间 | 年温度谐波的相位，以及相似日的夏令时状态 |
| `observed`（可选） | 负荷的布尔掩码 | 被屏蔽的小时按缺测处理 | `False` 的小时被忽略 |

`forecast(history, carriers, *, group, temp_sigma_std, dst_region="none", cut=None, state_rule="seasonal_naive", device="cpu", **anchor_kwargs)` 的其他参数：

- `carriers`：一个载体网络、一个载体列表（集成，取平均），或预先算好的映射 `{k: (744,) carrier forecast issued at origin − 744k}`。
- `group`：`"Industrial"`、`"Office"`、`"Public"`、`"Residential"` 或 `"Commercial"`。它既是载体的类别输入，也是锚定层选择固定温度特征先验的依据。`"Commercial"` 在锚定层中没有先验曲线（温度特征为零），与 ANKYRA 2.2.1 相同。
- `temp_sigma_std`：该人群的温度异常尺度，一个正数，单位为 10 °C，在训练截点之前的日子上拟合（[研究中怎样拟合](docs/DATA.md#the-input-format)）。
- `dst_region`：`"EU"`、`"US"` 或 `"none"`；只用于从相同的夏令时状态中取相似日。
- `cut`：载体的训练截点，以记录的小时下标表示。截点之前的伪起报点不使用（不计算；传入预先算好的映射时则从中删去）；起报点早于截点会报错。只要预测的记录是载体训练时用过的，就要传入。
- `state_rule`：`"seasonal_naive"`（评估过的配置，见下文）或 `None`。
- `**anchor_kwargs`：传给锚定层（`gap_tolerance`、`single_trust`、`within_anchor`、`micro_load_rule`）；评估用的都是它们的默认值（全部为 `True`）。

在顶层导入 `forecast` 会把属性 `ankyra_lite.forecast` 重新绑定到这个函数；该模块的其他名字请显式导入，例如 `from ankyra_lite.forecast import ForecastResult, seasonal_naive`。

### 输出

`forecast` 返回一个 `ForecastResult`：

| 字段 | 含义 |
|---|---|
| `trajectory_kw` | (744,) 交付的逐时预测，单位 kW；第 0 个元素是最后一个负荷值之后的那一小时 |
| `level_kw`、`energy_kwh` | 锚定层的水平，在其非负投影之前读出（状态窗口上采用季节性朴素规则时，是那个预测的均值）；以及月电量，即 744 × 水平 |
| `peak_kw` | 交付轨迹的月峰值读出：在 31 天中，取当天预测均值加上同日类型近期最大偏移的最大者（`anchor.readouts.peak_readout`） |
| `state_window`、`state_rule` | 锚定层是否报告了停机或微负荷窗口，以及实际采用的状态规则（`"seasonal_naive"` 或 `None`） |
| `carrier` | {k: (744,)} 交给锚定层的载体预测 |
| `anchored` | 锚定层自己的结果；其他属性都从这里读取 |

透传到锚定层的属性，按快速上手打印的样子列出：

| 快速上手中的行 | 字段 | 含义 |
|---|---|---|
| `carrier_issue_times_k` | `carrier` 的键 | 载体发布预测的时刻 o − 744k；0 是起报点 |
| `lead_week_weights_on_carrier` | `lead_week_weights` | 交接：第 1–7、8–14、15–21 和 22–31 天中载体日均值所占的权重；½ 表示没有证据 |
| `level_weights` | `level_weights` | 每个水平候选的权重：`s`、`l` 和 `a` 分别是最近 7 天均值、长期历史均值和一年前同一窗口的均值（`_u` 不作、`_w` 作天气修正）；`fm` 是载体自己的窗口均值（键名沿用锚定层的名字） |
| `pseudo_origin_pairs` | `pseudo_pairs` | 交接权重背后已完成的伪起报点个数（最多 6） |
| `within_day_trust_on_analog_shape` | `within_trust` | 单位自己相似日形状的权重，介于 0 和 ½ 之间；每个窗口一个值，在月内四段上重复 |
| `within_day_pseudo_origin_triples` | `within_pseudo_pairs` | 日内信任背后已完成的伪起报点个数（最多 3） |
| `analog_shape_kept` | `analog_kept` | 相似日形状大得不合理、整个窗口改用载体形状时为 `False` |
| — | `off_state`、`micro_load` | 锚定层的哪条状态规则被触发 |
| — | `anchoring_record` | 单位伪预测记录在水平上的两个统计量，预测本身不使用：`carrier_level_weight` 和 `history_vs_carrier_log_ratio`（[表](#历史长度与锚定记录)） |
| `interval_80pct_first_hour_kw`、`interval_pseudo_windows` | `anchor.readouts.interval_bands`、`residual_quantiles` | 一个 80% 区间：由单位纯历史伪预测的残差分位数得到，加到轨迹上；这是单独的调用，不属于预测器；它配合这个载体时的覆盖率见[预测区间](#预测区间) |

**状态窗口。** 当最后 168 小时都不超过 10⁻⁶ kW（停机），或整个 1,344 小时上下文都在 0 附近 10⁻³ kW 以内（微负荷）时，锚定层原样交回载体的预测。在默认的 `state_rule="seasonal_naive"` 下，ankyra-lite 改为交付最后一个观测周的重复（缺测小时记为 0）；这是评估过的配置。`state_rule=None` 保留锚定层的返回值（[docs/METHOD.md](docs/METHOD.md#state-windows-and-the-seasonal-naive-rule)）。

### 需要多少历史

- **锚定层**需要起报点之前的 1,344 小时完整观测。随着已完成的伪起报点增多（每个是一步 744 小时），它的权重逐渐离开先验；用到一年前窗口的候选需要 10,248 小时。评估的每个测试窗口都至少有 9 个完整的 744 小时伪起报点窗口（`results/history_length.csv`：`K_eff` 为 9–11 的有 451 个窗口，为 12 的有 5,265 个）。
- **载体**最多读最近 366 天（日 token）和其前的 24 个 744 小时块；缺测的日和块被屏蔽，所以它在更短的记录上也能运行。
- **使用检查点时**，起报点至少要在其训练截点之后 186 天：这样锚定层打分的每个伪起报点预测都是在载体训练期之后发布的（[原因](#锚定层怎样使用载体)）。

按历史长度观察到的情况是描述性的，见[下文](#历史长度与锚定记录)。

### 什么时候不该用

按本仓库的证据，下列情况下 ankyra-lite 不是更好的选择：

- **一个没有自己载体的人群里的单位。** 检查点只在它们自己人群的测试窗口上评估过。对新数据，请用[配方](#训练配方-r)训练一个载体。
- **住户，且看重逐时轨迹时。** 在 GoiEner 住户上，零样本 TimesFM（−5.6%）和 Chronos-2-X（−3.8%）的逐时误差明确更低。如果看重的是月电量或峰值，ankyra-lite 在那里的误差明确低于全部四种零样本变体。
- **月初几天。** 在第 1 个预测日，零样本基础模型在每个测试人群上都领先。
- **上下文里有单个大尖峰。** 一个达到上下文最大值十倍的尖峰使逐时误差上升 41.1%，并使峰值读出失效（[稳健性](#对受损上下文的稳健性)）。
- **会关停的单位，在用区间时。** 区间读出的宽度在那里不含信息。
- **不是逐时的数据，或单位不是 kW 的数据。** 状态阈值是以 kW 计的绝对值；预测长度固定为 744 小时。
- **南半球的站点。** 锚定层年温度谐波的相位是固定的，最冷的一天在 1 月（继承而来，未改动）；九个人群都不在南半球。

## 模型卡

每一项都对照代码核对过（`ankyra_lite/carrier/model.py`、`features.py`、`inference.py`、`train.py`、`ankyra_lite/forecast.py`）；完整说明见 [docs/METHOD.md](docs/METHOD.md#the-carrier-network)。

| 项目 | 内容 |
|---|---|
| 任务 | 一个单位的月前逐时负荷预测：起报点之后的 744 小时（31 天 × 24 小时） |
| 结构 | 复制指针结构化负荷网络（`CopyPointerNetwork`，`build_carrier()`），输出送入锚定层 |
| 参数 | **61,613** 个可训练参数（`CARRIER_PARAMETERS`；在 `checkpoints/MANIFEST.json` 中对每个发布的检查点重新计数）；没有预训练权重；没有单位编号嵌入 |
| 日 token | **366 × 69**，最近 366 天，从旧到新：24 × arcsinh(负荷 / s)、24 个逐时观测标记、日均值和日标准差除以 s 后的 arcsinh、arcsinh(日均气温 / 30) 及其覆盖率、8 维日类型独热（周一 … 周日、假日）、年相位（sin、cos）、相对位置、负荷覆盖率、5 维类别独热 |
| 月 token | **24 × 16**：比日跨度更早的 744 小时块（均值、标准差、第 10 和第 90 百分位、工作日和周末均值、零负荷比例、覆盖率、气温及其覆盖率、年相位、位置、一个常数、均值变化、有效标记） |
| 未来日 token | **31 × 16**：预测各日的日历，起报时已知（8 维日类型独热、年相位、相对日位置、5 维类别独热） |
| 尺度 s | max（最近 744 小时中观测小时的均值，最近 8,760 小时中观测小时的均值）；不是有限正数时取 1 |
| 编码器 | 宽度 **48**；token 嵌入、时龄嵌入和分辨率嵌入；**两个共享的门控因果块**（层归一化、核长 3 的逐深度因果卷积、门 tanh · σ、残差），在 366 个日 token 上按**空洞 1, 2, …, 128**、在 24 个月 token 上按 **1, 2, 4, 8** 重复使用 |
| 读出 | 一次多头注意力（**4 个头**），由一个全局查询和 31 个以日历为条件的日查询对 390 个记忆位置做注意力，然后接 48 → 96 → 48 的 MLP |
| 解码器 | 正的能量预算 F(d, h) = 744 · L · p_d · q_d(h)：月水平 L = s · softplus(·)，日份额 p（31 天上的 softmax），小时份额 q_d（24 小时上的 softmax）；小时 logits 来自 23 个零均值小时方向上的**秩 8 字典**（一个学到的基础加上一个有界的、随起报点变化的修正，0.1 · tanh），31 天共享，每天各有系数 |
| 复制指针 | 每个预测日对单位自己的**最近 56 个完整日**做注意力（16 维的查询和键，日类型相同时另有一个学到的加分），把注意到的日内份额曲线加到它的小时 logits 上，把注意到的相对日均值加到它的日 logits 上，都经过初始化为零的门 |
| 输出 | 744 小时，非负；轨迹均值等于 L（在实数运算下）；水平、日路径和日内形状就是轨迹的精确分块 |
| 训练 | 每个人群从零训练，然后归还验证段再训练 8 轮（[配方 R](#训练配方-r)）；种子 0、1、2；预测是三个种子的简单平均 |
| 训练数据 | 该人群自己在训练截点之前的窗口（第 2 阶段加入验证段），加上从其他研究人群借来的窗口，每个都在该人群的验证段之前结束（[docs/DATA.md](docs/DATA.md#windows-borrowed-from-other-populations)） |
| 检查点 | 27 个文件（9 个人群 × 3 个种子），`{"model": state_dict, "epoch": 8}`，每个 265,653 字节 |
| 推理耗时 | 每次预测 21 次网络计算（7 个发布时刻 × 3 个种子）：0.075 秒；加上锚定层和峰值读出，每个窗口 0.584 秒，单个 CPU 线程（`results/cost_per_window.csv`） |
| 预期用途 | 为载体所针对的人群里的单位做月前逐时预测，起报点至少在其训练截点之后 186 天 |
| 未评估 | 不重新训练就用于其他人群；小时以下分辨率的数据；744 小时以外的预测长度；为这个载体设计的预测区间 |
| 证据 | 在此前读过的测试窗口上的复用评估（[先读这一节](#先读这一节)） |
| 许可 | MIT |

## 锚定层怎样使用载体

`ankyra_lite.anchor.forecast(history, foundation={k: forecast issued at o − 744k}, ...)` 就是 ANKYRA 2.2.1 的锚定层，没有改动；它的代码和文档字符串里说到基础模型的地方，接收的是载体的预测。

- **伪起报点。** 锚定层读取载体在起报点 o（k = 0）和 q = o − 744k（k = 1 … 6，相邻两个相隔 31 天）发布的预测，前提是 q 之前的 1,344 小时上下文完整观测。伪起报点的目标月 [q, q + 744) 在 o 时已经观测到，所以锚定层可以在已经发生的月份上，拿载体和单位自己的历史作比较、打分。`carrier_forecasts(..., origin=o, cut=C′)` 正好计算这些预测，每个只用它自己发布时刻之前的数据，并丢掉截点之前的伪起报点。
- **它为每个单位估计什么**（[docs/METHOD.md](docs/METHOD.md#what-the-layer-estimates-from-the-completed-pseudo-forecasts)）：
  - **水平权重**：载体的窗口均值作为第七个候选，与六个历史水平候选（近期、长期历史和一年前的均值，各有不作和作天气修正两种）并列，按误差倒数定权，并向等权收缩；
  - **交接权重**：在第 1–7、8–14、15–21 和 22–31 天中，载体日均值相对历史侧的权重，用最小二乘估计，并向 ½ 收缩；
  - **日内信任**：在载体的日内形状里混入多少单位自己的相似日形状，每个窗口一个值，向零收缩，上限为 ½。
- **为什么测试集从截点之后 186 天开始。** 载体在伪起报点上的每个预测，对载体来说都必须是真正的预测：在训练截点 C′ 或之后发布，这样它预测的那个月从来不是训练标签。测试起报点至少在 C′ 之后 6 × 744 小时 = 186 天，所以对每个 k ≤ 6 都有 q = o − 744k ≥ C′（性质 C10）。没有这段间隔，逐单位权重就会在载体的样本内误差上估计，从而过度信任载体。代价是这里的载体比实际使用时至少旧半年。

![一个剑桥测试窗口](figures/fig6_example_window.png)

*图 6. 一个剑桥测试窗口（建筑 b37，2013 年 12 月）：实测负荷、ankyra-lite、单独的载体和零样本 TimesFM 2.5，以及日均值和按预测周的误差。在这个窗口上，锚定后的预测与载体的预测很接近；三个预测都没有预见到圣诞周的低负荷。一个窗口只用来说明机制，不是证据。数据：剑桥大学校产档案（CC BY 4.0）；`results/example_window_cambridge.csv`。*

## 训练配方 R

发布的配方分两个阶段，对每个人群完全相同（训练、选出停止的轮数，然后用上截点之前的全部数据）。每个常数都是 `ankyra_lite.carrier.train.TrainConfig` 的字段；`recipe(name, init_checkpoint)` 构造各个命名的配置（[docs/METHOD.md](docs/METHOD.md#training)）。

**第 1 阶段：从零训练，带早停**（`TrainConfig()` = `recipe("from_scratch")`）。

1. **窗口。** 一个窗口是（单位，起报点 o）：上下文 [o − 1344, o)，标签 [o, o + 744)。只用训练截点 C′ 之前的数据（o + 744 ≤ C′；C′ 及之后的负荷和气温在计算任何特征之前就置为缺测）。训练起报点按天取；上下文小时和标签小时都至少有 75% 被观测到，窗口才合格。
2. **划分。** 验证段是截点之前的最后 93 天，从 v0 = max(2,088, C′ − 2,232) 开始。训练窗口在 v0 或之前结束；验证窗口在 v0 或之后开始、不晚于 C′ 结束；早停用的验证集是目标人群自己按周取起报点的验证窗口。
3. **上限。** 每个人群最多 16,000 个训练窗口和 2,000 个验证窗口，按单位均衡选取。
4. **合并训练集。** 每个研究数据源（九个人群和苏州工业园区）都贡献其标签在目标人群验证段开始时刻（绝对时间）或之前结束的窗口；对目标人群来说就是它的训练窗口，对其他来源则是取自它们自己训练集或验证段的“借来的”窗口（从不取自它们的测试期）。每个来源的窗口数与它按周网格的窗口数相同。
5. **抽样。** 每轮有放回地抽 16,000 个窗口：一半来自目标人群（每个单位等权），一半来自借出窗口的来源（先均匀选一个来源，再每个单位等权）。
6. **损失。** 每个窗口取 log(1 + MSE + 0.1 · KL)，在按尺度归一化后的观测小时上计算（MSE 加上一个广义 Kullback–Leibler 项），再对窗口平均。
7. **优化。** AdamW，学习率 3 × 10⁻⁴，权重衰减 10⁻⁴，批大小 64；梯度范数裁剪到 1.0，基础参数和复制指针参数分开裁剪。
8. **早停。** 验证准则：对每个单位取 log(1 + 该单位平方误差均值除以尺度的平方)，再在单位间平均。保留最好的一轮；最多 30 轮，patience 为 5。

**第 2 阶段：归还验证段**（`recipe("span_return", <stage-1 checkpoint>)`）。

9. **核对。** 重新计算起始权重的第 1 阶段准则，必须复现记录的值。
10. **归还的验证段。** 把目标人群自己在 [v0, C′) 段内按天取起报点、标签至少在截点之前 24 小时结束的窗口加入它的训练窗口，按该人群自己的抽样密度。
11. **训练。** 新建一个 AdamW，借来的窗口和时间限制不变，正好训练 8 轮，保留最后的权重；这一阶段不做任何选择。
12. **种子与集成。** 种子 0、1 和 2；载体预测是三个种子预测的简单平均。

```python
from ankyra_lite.carrier.train import TrainConfig, recipe, prepare_source, train_ensemble

stage1 = TrainConfig()                                                    # from scratch, early stopping
sources = {n: prepare_source(n, data[n], cut[n], stage1, seed_index=i) for i, n in enumerate(names)}
first = train_ensemble(target, sources, stage1, checkpoint_dir=f"runs/{target}/stage1")

stage2 = recipe("span_return", "runs/{target}/stage1/seed{seed}.pt")     # validation span returned, 8 epochs
sources = {n: prepare_source(n, data[n], cut[n], stage2, seed_index=i, with_validation=(n == target))
           for i, n in enumerate(names)}
final = train_ensemble(target, sources, stage2, checkpoint_dir=f"runs/{target}")
```

`data[name]` 包含 `load`（单位 × 小时，kW，缺测为 NaN）、`temp`、`day_types`、`groups` 和 `start_iso`。示例写入 `runs/`（被 git 忽略），从不写入 `checkpoints/`。

**训练记录显示了什么**（`results/training_curves.csv`；种子 0、1、2 的第 1 阶段最佳轮数）：BDG2 12、9、27；GoiEner 非住户 15、3、16；住户 15、16、5；剑桥 16、12、16；HEEW 27、20、23；EWELD 10、9、7；奥斯陆 14、9、30；德拉门 21、4、23；CINELDI 14、6、6。有一次运行（奥斯陆，种子 2）选中了第 30 轮，也就是最后一轮；每个第 1 阶段检查点都在第 2 阶段的起始核对中得到复现。第 2 阶段的准则是在已经回到训练集的窗口上计算的，所以只是描述，不是留出数据上的分数。

**包能复现什么。** 在小规模运行上，本包逐行复现研究的训练（奥斯陆和 BDG2 的窗口、划分、上限、训练池和归还的验证段），并以零参数差复现每个阶段的一步优化（`tests/test_train.py`，需要研究的数据）。发布的检查点来自研究的 GPU 运行，不是本包的完整规模运行；GPU 训练不能逐位重复。

**包里的其他配方。** `recipe("plain_continuation", ...)`（不归还验证段的第 2 阶段）和 `recipe("proportional_validation", ...)` 是选定 R 的那项研究里的另外两个方案；保留它们是为了复现，它们不是发布的配方（[配方历史](#配方历史r-是怎样选定的以及它的负面结果)）。

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
- **载体**：小型载体网络，锚定层锚定的就是它的预测。**单独的载体**指它在起报点上的三种子预测，不经过锚定层，也不用状态规则。
- **伪起报点**：单位记录内部更早的起报点 o − 744k（k = 1 … 6），它的那个月在 o 时已经观测到；载体在那里的预测称为**伪预测**。
- **块**：744 小时预测的三个精确部分：**水平**（其均值）、**日路径**（31 个日均值减去水平）和**日内形状**（每个小时减去当天的均值）。它们的平方误差可以直接相加。
- **交接**：日均值中取自载体而不是历史侧的份额，每个预测周一个权重（第 1–7、8–14、15–21、22–31 天）。**信任**：日内块中单位相似日形状的权重（0 到 ½）。
- **状态窗口**：锚定层的停机规则或微负荷规则被触发的窗口；ankyra-lite 在那里交付周季节性朴素预测。
- **训练截点 C′、测试起点 C**：见[数据划分](#数据划分)。**测试窗口**：从 C 起的月初起报点，结果文件中记为 `late`。
- **配方 R**（验证段归还，已发布）、**F**（同样 8 轮的继续训练，但不归还验证段）、**V**（按比例划分的验证集，带早停）、**P3s**（只有第 1 阶段）：配方研究的各个方案。
- **短记录人群**：BDG2 2017、GoiEner 非住户和 GoiEner 住户。**切换层**：它们的测试窗口中，目标月在归还验证段之前没有同季训练标签、归还之后有了的那些窗口。
- **纯历史配置**：没有载体的锚定层（`foundation=None`）。**无载体对照**：把纯历史配置自己的预测放在载体的位置上，调用未改动的锚定层。
- **对比 (i)**：锚定后对单独的载体（单位的历史对载体有没有帮助？）。**对比 (ii)**：锚定后对纯历史配置（含载体的配置是否好于只用历史？）。
- **明确**：一个对比的 95% 区间不含零。**不明确**不等于一样好。
- **层级**：该人群在 ANKYRA 2.2.1 发布版中的角色：测试（`test`，六个人群）或预览（`preview`，奥斯陆、德拉门、CINELDI）。在这里，每个人群都有自己的训练截点和测试集，评分方式相同；层级只用于给表格分组。

## 起报点之后无信息

起报点 o 上的预测只是下列各项的函数，此外别无其他（[docs/METHOD.md](docs/METHOD.md#information-at-the-origin)）：

- o 之前各小时的负荷和气温（载体最多读最近 366 天和其前的 24 个 744 小时块；锚定层读起报点之前的整个记录）；
- 31 个预测日的日类型，按起报时已知处理；
- 单位的类别、该人群固定的温度异常尺度（在训练截点之前的日子上拟合）和所在地区的夏令时规则；
- 载体的权重，拟合所用的标签至少在第一个测试起报点之前 186 天又 24 小时结束；
- 载体在 o 和每个伪起报点 q 上的预测，每个只用 q 之前的数据构造。

**由测试检验。** `tests/test_carrier_unit.py`（不读发布时刻及之后的数据、伪起报点规则）、锚定层自己的 `tests/anchor/test_no_future_information.py`，以及 [theory/](theory/README.zh-CN.md) 中的性质 C4 和 C10。它们证明的是实现层面的输入隔离，并不能说明输入本身不含更晚的信息：有两个日历在部分测试起报点之后才公布假日（[局限](#局限)），数据提供方的补缺测和气象档案的事后修订都在本包的上游。

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

本节的每个数字都是 [`results/`](results/README.md) 中某个文件的一行，由 2026 年 10 月 19 日用发布配方完成的全量重新生成写出；每一部分都注明了文件，[`results/SUMMARY.md`](results/SUMMARY.md) 是同一次运行的可读摘要。图由 [`figures/make_figures.py`](figures/README.md) 从同样的文件绘制。完整表格和评估协议见 [docs/EVALUATION.md](docs/EVALUATION.md)。

### 先读这一节

- **是复用评估，不是首次检验。** 这些测试窗口是 ANKYRA 2.2.1 发布版面板的截点后（late）窗口，在那里为发布版的基线评过分。研究之后又在这些窗口上给小载体的早期版本评过分，做过一次只看逐时误差的与基线的比较，并在选定配方的研究中再读过一次。复制指针的方向来自对载体在这些窗口上误差的分析，季节性朴素状态规则是在对它们的一次早先评分之后写的。这些表格描述的是发布的预测器在这些窗口上的表现；它们不能估计它在没有参与设计的窗口上会表现如何。
- **配方是在这些窗口上选定的**，是在对四个配方方案做了一次方向性读数之后（[配方历史](#配方历史r-是怎样选定的以及它的负面结果)）。
- **没有为这次比较训练或调整任何模型。** 已保存的预测用发布版的评分函数打分；在此之前先核对了这些函数能复现发布版公布的表格（[可复现性](#可复现性)）。
- **状态窗口。** ankyra-lite 在 33 个 BDG2、13 个 GoiEner 非住户、35 个住户测试窗口，以及 523 个 EWELD 测试窗口中的 245 个上交付周季节性朴素预测（`results/compare_baselines.json`）；在这些窗口上，它的预测与 `SN-week` 基线相同。在每个窗口都读数接近零的三块 BDG2 电表上（10 个窗口），它的误差恰好为零，所以这些电表不进入单位等权的对比，但计入平均名次（[细节](docs/EVALUATION.md#state-windows-and-the-three-near-zero-bdg2-meters)）。
- 发布版的**苏州园区**没有 ankyra-lite 预测，不在任何表中；它只提供借来的训练窗口。

### 怎样读这些数字

- **改进** = 100[1 − exp(r)]，r 是单位等权的对数 RMS 比：对每个单位取它在各测试窗口上误差的 RMS，r 是 log(ankyra-lite 的 RMS / 基线的 RMS) 在各单位上的均值。正数表示 ankyra-lite 更好。
- **区间**：95%，来自 2,000 次重抽单位和目标月份的自助重复。`*` 表示明确偏向 ankyra-lite 的对比，`(−)` 表示明确不利于它的对比。
- **逐时**：744 小时上的误差。**电量**：月电量误差，即交付轨迹窗口均值误差的 744 倍，对每个预测器都这样计算。**峰值**：ankyra-lite 的峰值读出，对比把同一读出用在每个基线轨迹上的结果（发布版的规则），以及对比每个基线自己轨迹的最大值。
- **名次**：按单位平均名次在 22 个预测器中的位置（在每个单位内按 RMSE 给预测器排名）。
- 报告几种汇总，是因为合并汇总和单位等权汇总可能符号相反（[theory/](theory/README.zh-CN.md) 中的 P19），而且绝对误差类指标偏向中位数型的预测，平方误差类指标偏向均值型的预测。

### 总览

明确更好 / 明确更差的基线个数，共 21 个（`benchmark_pairwise.csv`、`energy_error.csv`、`peak_readout.csv`；名次来自 `benchmark_mean_unit_rank.csv`；状态窗口来自 `compare_baselines.json`）：

| 人群 | 层级 | 窗口 / 单位 | 状态窗口 | 22 个中的名次（单位平均名次） | 按单位平均名次的下一个 | 逐时 | 电量 | 峰值（同一读出） | 峰值（对基线轨迹最大值） |
|---|---|---|---:|---|---|---|---|---|---|
| BDG2 2017 | 测试 | 474 / 142 | 33 | 1（5.35） | 逐单位岭回归（6.91） | 12 / 0 | 4 / 0 | 4 / 0 | 17 / 0 |
| GoiEner 非住户 | 测试 | 890 / 481 | 13 | 1（5.42） | iTransformer（7.15） | 14 / 0 | 14 / 0 | 1 / 0 | 18 / 0 |
| GoiEner 住户 | 测试 | 696 / 696 | 35 | 1（5.49） | 逐单位岭回归（7.06） | 17 / 2 | 17 / 0 | 11 / 0 | 21 / 0 |
| 剑桥 | 测试 | 730 / 108 | 0 | 1（3.23） | iTransformer-X（5.98） | 21 / 0 | 20 / 0 | 10 / 0 | 21 / 0 |
| HEEW（亚利桑那） | 测试 | 658 / 138 | 0 | 1（5.56） | Chronos-2-X（6.78） | 12 / 0 | 5 / 0 | 5 / 0 | 16 / 0 |
| EWELD | 测试 | 523 / 197 | 245 | 1（7.02） | TimesFM（8.17） | 13 / 0 | 6 / 0 | 6 / 0 | 19 / 0 |
| 奥斯陆 | 预览 | 581 / 41 | 0 | 1（1.76） | GBT-T（2.71） | 20 / 0 | 19 / 0 | 17 / 0 | 17 / 0 |
| 德拉门 | 预览 | 689 / 43 | 0 | 1（2.14） | Chronos-2-X（3.58） | 20 / 0 | 9 / 0 | 6 / 1 | 19 / 0 |
| CINELDI | 预览 | 475 / 45 | 0 | 1（3.36） | 逐单位岭回归（5.67） | 20 / 0 | 11 / 0 | 3 / 0 | 19 / 0 |

- **逐时。** 唯一的明确落后在 GoiEner 住户上，对 TimesFM（−5.6%）和 Chronos-2-X（−3.8%）。明确的领先从 +3.4%（逐单位岭回归，住户）到 +54.2%（奥斯陆）。不明确的例子：奥斯陆上的 GBT-T（+3.4%），以及德拉门（+5.0%）和 CINELDI（+6.0%）上的 Chronos-2-X。
- **去掉三块近零电表后的 BDG2**：点估计相同；明确更好的逐时对比是 10 个而不是 12 个（PatchTST 和 iTransformer-X 不再明确），在 139 个单位上的单位平均名次为 5.367，22 个中第一（`bdg2_near_zero_sensitivity.csv`）。
- **电量**在任何人群上都没有明确差于任何基线。把交付轨迹的均值换成 ankyra-lite 的水平读出，个数相同；两者只在非负投影起作用的窗口上不同（BDG2 1 个、GoiEner 非住户 142 个、住户 29 个、剑桥 9 个、EWELD 21 个、德拉门 2 个、CINELDI 33 个窗口）。
- 明确对比的个数是对数百个区间的描述性汇总，不是经过多重比较校正的检验。

![测试集名次与两两胜负](figures/fig2_test_ranks.png)

*图 2. 每个预测器在各人群测试窗口上的单位平均名次（a），以及跨人群的两两胜负计数（b）。*

![逐时对比及其区间](figures/fig3_test_pairwise.png)

*图 3. ankyra-lite 的单位等权逐时改进及 95% 单位与月份区间，对比对象是拿到相同信息的基线、LSTM-T-KAN-M 和部分只用负荷的基线；其他基线见图 2 和图 8。*

![在每个人群上的位置](figures/fig12_consistency.png)

*图 12. 每个预测器在每个人群上于 22 个中的位置（单位平均名次），按模型类别着色。*

### 与零样本基础模型比较

改进，单位为 %（`benchmark_pairwise.csv`、`energy_error.csv`，以及按同一读出的 `peak_readout.csv`）：

| 人群 | 逐时 对 TimesFM | 电量 对 TimesFM | 峰值 对 TimesFM | 逐时 对 Chronos-2-X | 电量 对 Chronos-2-X | 峰值 对 Chronos-2-X |
|---|---|---|---|---|---|---|
| BDG2 2017 | −1.0 | +12.0 | +5.3 | −5.8 | +0.1 | +1.2 |
| GoiEner 非住户 | +2.8 | +19.7 * | +0.1 | −2.4 | +11.9 | +2.7 |
| GoiEner 住户 | −5.6 (−) | +36.4 * | +11.0 * | −3.8 (−) | +36.4 * | +12.5 * |
| 剑桥 | +14.2 * | +23.1 * | +10.4 | +8.7 * | +18.2 * | +4.9 |
| HEEW（亚利桑那） | +1.1 | −1.6 | −1.7 | −0.7 | −2.4 | −0.5 |
| EWELD | +1.4 | −5.3 | +5.7 | +4.0 | +14.7 | +3.5 |
| 奥斯陆 | +23.4 * | +35.4 * | +17.1 * | +15.0 * | +21.0 * | +7.9 |
| 德拉门 | +18.9 * | +25.7 * | +14.3 | +5.0 | −0.1 | −1.4 |
| CINELDI | +9.8 * | +21.2 * | +5.2 | +6.0 | +15.4 | +5.8 |

- 在 BDG2、GoiEner 非住户、HEEW 和 EWELD 上，与四种零样本变体（TimesFM、TimesFM-X、Chronos-2、Chronos-2-X）的逐时对比没有一项明确，有几项点估计为负（BDG2 对 Chronos-2-X −5.8%，EWELD 对 Chronos-2 −0.4%）。
- 在住户上，电量对比对全部四种变体都明确偏向 ankyra-lite（+32.6% 到 +37.1%），逐时对比对 TimesFM 和 Chronos-2-X 明确不利于它（Chronos-2 −3.8% 和 TimesFM-X −1.9% 不明确）。
- 有七个 BDG2 站点列在 TimesFM 2.5 的预训练集合中，这可能使 TimesFM 和 TimesFM-X 在 BDG2 上占优（[局限](#局限)）。

### 秩检验

逐单位 RMSE，以单位为区组（`rank_tests.csv`；评分之后计算，描述性）：

- **六个测试人群合并**（1,762 个单位）：Friedman p < 10⁻³⁰⁰；Nemenyi 临界差 0.786。ankyra-lite 的平均名次是 22 个中的 5.496，下一个预测器（逐单位岭回归）为 7.438，第三个（iTransformer-X）为 8.556，差距大于临界差。Holm 校正的 Wilcoxon 检验显示它好于全部 21 个基线，没有差于任何一个。
- **按人群**，它在八个人群上显著好于 21 个基线中的 21 个，在 HEEW 上好于 20 个，那里 Chronos-2-X 与它没有显著差异。没有任何基线在任何人群上显著更好。

![秩检验](figures/fig15_rank_tests.png)

*图 15. 合并的平均名次，附 Nemenyi 临界差和 Holm 校正的 Wilcoxon 检验（a）；各人群的配对检验（b）。*

### 常规指标

ankyra-lite 在 22 个预测器中的位置（`conventional_metrics_late.csv`；RMSE 和 MAE 是单位平均，单位 kW；CV(RMSE) 和 WAPE 取中位单位）：

| 指标 | BDG2 | GoiEner 非住户 | GoiEner 住户 | 剑桥 | HEEW | EWELD | 奥斯陆 | 德拉门 | CINELDI |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RMSE（单位平均） | 1 | 1 | 1 | 1 | 2 | 2 | 1 | 1 | 1 |
| MAE（单位平均） | 2 | 2 | 5 | 1 | 5 | 4 | 1 | 1 | 5 |
| CV(RMSE)（中位单位） | 3 | 1 | 1 | 1 | 1 | 3 | 1 | 1 | 1 |
| WAPE（中位单位） | 2 | 4 | 5 | 1 | 4 | 4 | 1 | 1 | 2 |
| WAPE（合并） | 3 | 3 | 5 | 1 | 5 | 6 | 1 | 1 | 5 |

位置是按文件中的两位小数严格排在它前面的预测器个数加一。表中唯一的并列是 BDG2 的合并 WAPE：ankyra-lite 与 TimesFM 都是 12.09%，所以位置是第 3；若把并列判给 TimesFM，则是第 4。

排在它前面的主要是零样本基础模型（Chronos-2-X、Chronos-2、TimesFM、TimesFM-X）。GBT-T 在 BDG2 的 CV(RMSE)、HEEW 的 RMSE、MAE 和合并 WAPE，以及 CINELDI 的 MAE 和合并 WAPE 上领先；逐单位岭回归在 BDG2 的合并 WAPE 上领先，日季节性朴素在 EWELD 的合并 WAPE 上领先，4 周剖面在 CINELDI 的 MAE 上领先，iTransformer 在 CINELDI 的合并 WAPE 上领先。MAE 和 WAPE 由预测分布的中位数最小化，平方误差类指标由其均值最小化；ankyra-lite 的载体按平方误差训练，锚定权重也按平方误差估计。

![全部 22 个预测器的损失指标](figures/fig8_loss_metrics.png)

*图 8. 每个预测器在每个人群上的 RMSE、MAE、CV(RMSE) 和 WAPE，按相对 ankyra-lite 的损失着色。*

### 按预测日

六个测试人群，22 个预测器（`lead_day_metrics.csv`：每天 24 小时上的 CV(RMSE)，在一组固定的单位上取几何平均；评分之后计算，描述性）：

| | BDG2 | GoiEner 非住户 | GoiEner 住户 | 剑桥 | HEEW | EWELD |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 第 1 天的位置 | 15 | 4 | 7 | 6 | 10 | 8 |
| 31 天中最低的天数 | 4 | 0 | 0 | 24 | 6 | 1 |
| 任一天中的最差位置 | 15 | 7 | 7 | 6 | 10 | 11 |

“最低”按文件中的两位小数计入并列（严格最低：剑桥 23 天，HEEW 3 天）。

- **月初几天是它最弱的时候。** 第 1 天，零样本基础模型在每个人群上都领先于它（五个人群上是全部四种变体；GoiEner 非住户上是 Chronos-2-X、Chronos-2 和 TimesFM），而且除 GoiEner 非住户外，每个人群上还有训练型基线领先（BDG2 上除四种变体外还有十个基线）。
- **第一周之后**，它的逐时损失在剑桥第 8 天起的 24 天中有 23 天最低，HEEW 6 天，BDG2 4 天，EWELD 1 天；在两个 GoiEner 人群上一天也没有，那里最低的是零样本 Chronos-2-X 或 Chronos-2。

![按预测日的损失](figures/fig9_loss_by_day.png)

*图 9. 按预测日 1–31 的逐时 CV(RMSE)，在一组固定的单位上取几何平均，六个测试人群。*

![相对 ankyra-lite 的按预测日损失](figures/fig10_loss_by_day_relative.png)

*图 10. 各个点名基线的每日损失相对 ankyra-lite 的值（ankyra-lite = 0）。*

### 月电量与月内累计电量

- **月电量**（`energy_error.csv`，九个人群）：明确好于 21 个基线中的 4 个（BDG2）到 20 个（剑桥），没有明确差于任何一个。按中位单位的电量绝对百分比误差，它在 GoiEner 非住户（11.85%）、住户（13.22%）、剑桥（4.55%）、奥斯陆（7.80%）和德拉门（8.96%）上是 22 个中的第一，在 BDG2 上第三，CINELDI 第四，HEEW 第五，EWELD 第六。
- **对 TimesFM**，电量误差在六个人群上明确更低（+19.7% 到 +36.4%），在 BDG2 上不明确（+12.0%），在 HEEW（−1.6%）和 EWELD（−5.3%）上更高但不明确。
- **月内累计**（`lead_day_energy.csv`：从第 1 天到第 d 天交付的电量的误差，六个测试人群；描述性）：

  | | BDG2 | GoiEner 非住户 | GoiEner 住户 | 剑桥 | HEEW | EWELD |
  |---|:---:|:---:|:---:|:---:|:---:|:---:|
  | 31 天中最低的天数 | 6 | 8 | 6 | 24 | 5 | 0 |
  | 第 31 天的位置（即月电量） | 1 | 1 | 1 | 1 | 1 | 8 |
  | 第 8–31 天中低于全部四种零样本变体的天数（共 24 天） | 6 | 15 | 24 | 24 | 5 | 0 |

  在 HEEW 上，第 31 天对 Chronos-2-X 的领先在第三位小数（7.67 对 7.6714）。在 EWELD 上，它在第 31 天排第八，落后于 PatchTST、TimesFM-X、Chronos-2、TimesFM、周季节性朴素、iTransformer 和逐单位岭回归。

![随月内累积的电量误差](figures/fig9b_energy_by_day.png)

*图 9b. 截至每个预测日交付的电量的误差，汇总方式与图 9 相同。*

![对每个基线的月电量误差](figures/fig11_energy_error.png)

*图 11. 每个人群上月电量误差相对每个基线的改进（BDG2 另给出去掉三块近零电表的结果）（a），以及两两胜负计数（b）。*

### 峰值读出

ankyra-lite 的峰值读出是锚定层的包络算子，作用在其交付轨迹的日均值上：U 取各日 d 上 (L̂_d + A_τd) 的最大值，即当天的预测均值加上同日类型近期的最大偏移（P15–P17）。`peak_readout.csv`：

- **对每个基线轨迹上的同一读出**（发布版的规则）：每个人群上明确好于 21 个中的 1 个（GoiEner 非住户）到 17 个（奥斯陆）；唯一的明确落后是德拉门上对 GBT-T（−10.9%）。
- **对每个基线自己轨迹的最大值**：每个人群上明确好于 21 个中的 16–21 个，没有明确更差的。
- **对它自己轨迹的最大值**：在八个人群上明确更好（从德拉门的 +30.8% 到住户的 +67.9%）；奥斯陆 +18.1%，不明确。
- **对上个月的实测峰值**（最后 744 个上下文小时的最大值）：奥斯陆明确更好（+27.4%），CINELDI 明确更差（−20.4%）；其余不明确，从 −35.8%（GoiEner 非住户）到 +15.4%（德拉门）。

![峰值读出](figures/fig5_peak_operator.png)

*图 5. 峰值读出对它自己的轨迹最大值（a）和对上个月实测峰值（b）；对每个基线轨迹上同一读出的两两胜负计数（c）。*

### 误差分块与尺度化误差

每个窗口的逐时平方误差都可以精确拆成水平、日路径和日内三部分（P1）。`block_shares.csv`（六个测试人群，描述性；分块对比不带区间）：

| | BDG2 | GoiEner 非住户 | GoiEner 住户 | 剑桥 | HEEW | EWELD |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| ankyra-lite 误差中的份额，合并：水平 | 36.4% | 11.4% | 14.2% | 15.3% | 38.5% | 50.7% |
| 日路径 | 43.2% | 33.1% | 16.8% | 21.2% | 32.2% | 34.1% |
| 日内 | 20.4% | 55.4% | 69.0% | 63.5% | 29.3% | 15.2% |
| 对 TimesFM 的改进：水平 | +12.0 | +19.7 | +36.4 | +23.1 | −1.6 | −5.3 |
| 日路径 | −3.8 | +2.1 | −9.3 | +9.9 | +2.6 | +4.0 |
| 日内 | −2.8 | +3.7 | −4.6 | +11.2 | +1.5 | +2.5 |

在住户上，相对 TimesFM 的逐时差距在日路径块和日内块，不在水平块。

`scaled_errors.csv`（每个窗口的误差除以周季节性朴素在其上下文上的样本内误差；描述性）：按中位单位的 RMSSE，ankyra-lite 在六个人群上排第一（住户、剑桥、HEEW、奥斯陆、德拉门、CINELDI），GoiEner 非住户第二，BDG2 第三，EWELD 第四。按中位单位的 MASE（一种绝对误差尺度），它在剑桥、奥斯陆和德拉门上第一，CINELDI 第二，住户第五，BDG2、GoiEner 非住户、HEEW 和 EWELD 第六。

![分块归因](figures/fig14_block_attribution.png)

*图 14. ankyra-lite、TimesFM 和 Chronos-2-X 逐时平方误差中每一块的中位单位份额（a）；ankyra-lite 按块相对 TimesFM（b）和相对 Chronos-2-X（c）的改进。*

### 预测区间

ankyra-lite 自己不产生区间。为了描述，把锚定层的区间读出放在它的交付轨迹周围：用**纯历史**预测器在单位至多 12 个已完成伪起报点上的残差分位数，它们与载体无关。比较对象是 TimesFM 2.5 和 Chronos-2 的原生分位数（`intervals_by_population.csv`、`intervals_winkler_contrasts.csv`、`intervals_households.json`；描述性）：

| 人群 | 80% 覆盖率：ankyra-lite / TimesFM / Chronos-2 | 90% 覆盖率：ankyra-lite / Chronos-2 | Winkler 80 对 TimesFM | 对 Chronos-2 |
|---|---|---|---|---|
| BDG2 2017 | 0.758 / 0.349 / 0.629 | 0.850 / 0.776 | −3.9 | −18.6 |
| GoiEner 非住户 | 0.783 / 0.430 / 0.720 | 0.868 / 0.851 | +5.5 | −9.9 (−) |
| GoiEner 住户 | 0.777 / 0.345 / 0.680 | 0.870 / 0.815 | +6.3 * | −1.9 |
| 剑桥 | 0.791 / 0.393 / 0.710 | 0.881 / 0.836 | +15.2 * | −1.3 |
| HEEW（亚利桑那） | 0.779 / 0.373 / 0.726 | 0.865 / 0.848 | +9.4 * | −2.4 |
| EWELD | 0.841 / 0.606 / 0.804 | 0.893 / 0.883 | −4.0 | −40.0 (−) |
| 奥斯陆 | 0.851 / 0.379 / 0.678 | 0.921 / 0.813 | +21.2 * | +4.7 * |
| 德拉门 | 0.848 / 0.388 / 0.725 | 0.915 / 0.841 | +20.4 * | +2.9 |
| CINELDI | 0.812 / 0.438 / 0.736 | 0.896 / 0.859 | +15.2 * | −4.7 (−) |

- 80% 区间覆盖 0.758–0.851 的小时，TimesFM 的原生分位数为 0.345–0.606，Chronos-2 的为 0.629–0.804。它的覆盖率随提前期下降（第一周 0.792–0.874，第四周 0.715–0.831），因为残差是在整个窗口上合并的。
- 按 80% Winkler 分数，它在六个人群上明确好于 TimesFM 的区间，没有一个人群更差；对 Chronos-2 的分位数，它在奥斯陆上明确更好，在 EWELD、GoiEner 非住户和 CINELDI 上明确更差。
- 在 EWELD 上，对会关停的单位，区间不含信息（80% 区间的平均宽度为 388,322 kW）。
- 在住户上，把同样的区间放在单独的载体周围，Winkler 分数变化 0.15%（不明确）：区间来自纯历史残差，而不是来自载体。

![住户上的预测区间](figures/fig13_intervals.png)

*图 13. GoiEner 住户上区间的覆盖率和 Winkler 分数。*

![每个人群上的区间覆盖率](figures/fig16_intervals_populations.png)

*图 16. 每个人群上名义 80% 区间的覆盖率，以及按预测周的覆盖率。*

### 各部分带来什么

ankyra-lite 相对去掉某一部分的版本的逐时改进（`component_contributions.csv`、`ablation.csv`；“相同”表示两个预测在每个窗口上都一样）。关掉锚定层某些部分的各行，是在保存的载体预测上重新运行未改动的锚定层；在用到其中任何一行之前，发布集成的重新运行先逐位复现了评估所用的预测。

| 部分 | BDG2 | GoiEner 非住户 | GoiEner 住户 | 剑桥 | HEEW | EWELD | 奥斯陆 | 德拉门 | CINELDI |
|---|---|---|---|---|---|---|---|---|---|
| 整个锚定层（对单独的载体） | +3.2 * | +3.5 | −1.0 | +0.3 | +1.4 | +1.2 | +0.0 | +0.4 | +2.6 * |
| 锚定层中的载体（对纯历史配置） | +6.4 * | +19.5 * | +11.1 * | +13.8 * | +12.6 * | +16.6 * | +20.5 * | +15.3 * | +5.4 * |
| 日内锚定 | +0.9 * | +3.2 * | −0.1 | +0.5 * | +0.9 * | −0.0 | +0.5 * | +1.0 * | +1.2 * |
| 季节性朴素状态规则 | +0.2 | −0.0 | +0.3 | 相同 | 相同 | +2.6 | 相同 | 相同 | 相同 |
| 三种子集成（对单个种子；三个对比的平均） | +0.8 | +0.8 | +0.3 | +1.5 | +1.9 | +0.6 | +1.8 | +1.4 | +0.8 |
| 发布的配方（对只有从零训练阶段） | +1.2 | +2.7 | +1.4 * | +1.7 * | +0.9 * | +1.1 | +2.2 * | −2.3 | +0.8 |
| 验证段归还（对不归还的同一继续训练） | +0.5 | +2.2 | +0.7 | +0.5 | −0.2 | −0.3 | +0.1 | −2.9 (−) | +0.3 |

- **对比 (i)**，即整个锚定层对单独的载体，在逐时误差上有两个人群明确，在月电量上有一个（CINELDI +7.8%）；在住户上为 −1.0%（不明确）。**对比 (ii)**，即对纯历史配置，在逐时误差上九个人群全部明确（+5.4% 到 +20.5%），在月电量上有四个（剑桥 +18.5%、HEEW +13.2%、EWELD +33.2%、奥斯陆 +21.4%）。
- **小开关。** 每个窗口一个日内信任值（相对于每个提前期块一个值）、短缺口规则和微负荷规则，各自对逐时误差的影响都不超过 0.2%。状态规则只在有状态窗口的地方起作用；在 GoiEner 非住户上它使月电量变差（−0.9%，明确不利）。
- **早期载体**（八个人群；修正后的 CINELDI 数据仓库上没有）：对去掉复制指针、用从零配方训练的同一网络，ankyra-lite 在七个人群上好 +1.2% 到 +3.7%（剑桥、HEEW 和奥斯陆上明确），在德拉门上为 −0.3%；对更早的、没有指针、在按周窗口上训练的结构化网络，好 +1.1% 到 +10.4%（六个人群上明确）。
- 这些行的区间只重抽单位和月份；带种子层的配方各行见[配方历史](#配方历史r-是怎样选定的以及它的负面结果)。

![消融](figures/fig4_ablation.png)

*图 4. ankyra-lite 对它自己的简化版和替代版（逐时误差）：关掉锚定层部分组件和状态规则、单独的载体和纯历史配置（a）；一个种子代替三个（b）；同一锚定层中的早期载体和其他配方方案（c）。*

### 种子

`seed_spread.csv`：每个种子的载体单独锚定，以及三种子集成，相对 TimesFM 的逐时改进：

| 人群 | 种子 0 | 种子 1 | 种子 2 | 集成 |
|---|---|---|---|---|
| BDG2 2017 | −1.6 | −3.6 | −0.4 | −1.0 |
| GoiEner 非住户 | +3.3 | +1.7 | +1.2 | +2.8 |
| GoiEner 住户 | −6.3 (−) | −5.6 (−) | −6.0 (−) | −5.6 (−) |
| 剑桥 | +13.4 * | +12.8 * | +12.3 * | +14.2 * |
| HEEW（亚利桑那） | +0.6 | +0.4 | −3.5 | +1.1 |
| EWELD | +0.9 | +0.8 | +0.5 | +1.4 |
| 奥斯陆 | +23.0 * | +20.5 * | +22.4 * | +23.4 * |
| 德拉门 | +17.5 * | +20.2 * | +15.7 * | +18.9 * |
| CINELDI | +9.0 * | +8.7 * | +9.4 * | +9.8 * |

三个种子之间最多相差 4.5 个百分点（德拉门；HEEW 4.2）；在每个人群上，每个种子和集成的判定都相同。种子间的差异是 `results/` 中唯一的重新训练噪声度量；主要表格的区间不包含它。

### 同一锚定层中的其他载体

`carriers_in_anchoring_layer.csv`：同一锚定层配上另外三个训练型载体（LSTM、PatchTST 和 LSTM-T-KAN-M，只在它们自己的人群上训练，训练截点与小载体相同；在状态窗口上锚定层返回它们自己的预测；在 CINELDI 上，它们在原始数据仓库上训练、在修正后的数据仓库上锚定）。九个人群中明确更好 / 不明确 / 明确更差的个数：

| 载体 | (i) 锚定后对单独的载体：逐时 | 电量 | (ii) 锚定后对纯历史：逐时 | 电量 | ankyra-lite 对这个锚定后的载体：逐时 | 电量 |
|---|---|---|---|---|---|---|
| 小型载体网络（ankyra-lite） | 2 / 7 / 0 | 1 / 8 / 0 | 9 / 0 / 0 | 4 / 5 / 0 | — | — |
| LSTM | 8 / 1 / 0 | 6 / 3 / 0 | 3 / 5 / 1 | 0 / 7 / 2 | 9 / 0 / 0 | 5 / 4 / 0 |
| PatchTST | 7 / 2 / 0 | 3 / 6 / 0 | 7 / 2 / 0 | 3 / 6 / 0 | 5 / 4 / 0 | 1 / 8 / 0 |
| LSTM-T-KAN-M | 8 / 1 / 0 | 7 / 2 / 0 | 1 / 6 / 2 | 0 / 5 / 4 | 9 / 0 / 0 | 6 / 3 / 0 |

- 锚定层对较弱的载体改进更多（对比 (i)）。配上小载体时，组合在全部九个人群上的逐时误差都好于纯历史配置；配上 LSTM 有三个，配上 LSTM-T-KAN-M 有一个。对比 (ii) 中明确的落后包括两者在 EWELD 上的结果（逐时 −83.1% 和 −104.4%），那里锚定层在它的 245 个状态窗口上返回这些载体自己的预测。
- 没有任何锚定后的载体在任何人群上明确好于 ankyra-lite。

**按块的对比 (ii)，以及无载体对照**（`contrast_ii_blocks.csv`、`contrast_ii_control.csv`；描述性）。对比 (ii) 比较的是两个完整配置：纯历史配置用日类型默认形状代替相似日形状，并且没有状态规则。无载体对照用同一流程运行，只是把纯历史预测放在载体的位置上。九个人群中明确更好 / 不明确 / 明确更差的个数，全部测试窗口：

| 对比 | 逐时 | 水平 | 日路径 | 日内 |
|---|---|---|---|---|
| ankyra-lite 对纯历史 | 9 / 0 / 0 | 4 / 5 / 0 | 5 / 4 / 0 | 8 / 1 / 0 |
| ankyra-lite 对单独的载体 | 2 / 7 / 0 | 1 / 8 / 0 | 1 / 6 / 2 | 7 / 2 / 0 |
| 单独的载体对纯历史 | 6 / 3 / 0 | 2 / 7 / 0 | 5 / 3 / 1 | 7 / 2 / 0 |
| ankyra-lite 对无载体对照 | 9 / 0 / 0 | 4 / 5 / 0 | 5 / 4 / 0 | 7 / 2 / 0 |
| 无载体对照对纯历史 | 8 / 1 / 0 | 0 / 9 / 0 | 1 / 8 / 0 | 8 / 1 / 0 |

对无载体对照，ankyra-lite 在全部九个人群上的逐时误差都明确更好（从 CINELDI 的 +3.0% 到奥斯陆的 +15.8%）。两边运行同一流程，差别在于放在载体位置上的那个预测（以及状态窗口，ankyra-lite 在那里交付季节性朴素预测）。这个预测同时进入水平候选、交接和日内混合，所以按块拆分只能粗略地定位差别所在。

### 历史长度与锚定记录

`history_length.csv`，九个人群合并（描述性；分层与人群相互混杂）：

| 起报时距单位第一个观测的小时数 | 窗口 | 对单独的载体 | 对 TimesFM | 对 Chronos-2-X |
|---|---:|---|---|---|
| < 10,248 小时（没有年度候选） | 450 | −0.6 | −3.7 | −3.9 |
| 10,248 小时 – 2 年 | 1,715 | +1.8 * | −1.3 | −4.4 |
| 2–3 年 | 829 | +2.6 | +10.8 | +6.1 |
| ≥ 3 年 | 2,722 | +0.3 | +8.3 * | +5.0 |

对零样本基础模型，点估计在两年以下为负，两年以上为正；只有 ≥ 3 年分层对 TimesFM 的对比明确。这是描述，不是各人群共同的阈值。

`anchoring_gain_deciles.csv`（`anchoring_record` 的两个统计量与 ankyra-lite 相对单独载体的实际收益之间的关系；九个人群上每个十分位 538–539 个窗口，不含状态窗口和平均负荷接近零的单位；描述性）：月水平收益在载体水平权重最低的十分位中最大（+7.1%），在最高的两个十分位中略为负（−1.4%、−1.1%）；按 `history_vs_carrier_log_ratio` 分，收益在最低的两个十分位中最大，在那里单位的历史在更早的伪起报点上胜过了载体（+9.5%、+8.0%；那里的逐时收益明确，分别为 +2.9% 和 +2.5%）。两个统计量都不能把收益单调排序；载体权重是锚定层自己的权重，所以它与收益的关联有一部分是机械性的。

### 对受损上下文的稳健性

`robustness.csv`：64 个德拉门测试窗口（43 个单位），用发布的检查点在 CPU 上从受损的上下文重新预测（各发布时刻的载体、锚定层、状态规则）；目标不变。逐时 RMS 误差相对干净上下文的变化（正数 = 更差）：

| 对上下文的破坏 | ankyra-lite | 单独的载体 | 受损时 ankyra-lite 对单独的载体 |
|---|---:|---:|---:|
| 5% 的小时缺测，用线性插值填补 | −0.0% | −0.1% | −2.0% |
| 20% 的小时缺测，填补 | +0.1% | −0.1% | −2.0% |
| 最后一周中一个 24 小时的补零块 | +3.2% | +3.0% | −2.0% |
| 起报点前 36 小时一个达到上下文最大值 10 倍的尖峰 | +41.1% | +23.6% | +11.8% |
| 上下文平移 +1 小时 | +9.4% | +9.2% | −1.9% |
| 上下文平移 −1 小时 | +11.7% | +12.7% | −3.1% |
| 整个历史乘以 0.5 | +0.0% | +0.0% | −2.2% |
| 整个历史乘以 2 | +0.0% | +0.0% | −2.2% |

- 整个历史乘以 0.5 或 2 时，预测严格按同一比例变化（最大偏差为 0），只要没有记录跨过状态阈值（阈值是以 kW 计的绝对值）。
- 尖峰使峰值读出的中位数变化 +998%：读出取的是近期最大偏移。
- 在任何受损的上下文上都没有触发状态规则。

### 每个窗口的耗时

`cost_per_window.csv`：本仓库的包加发布的检查点，6 个奥斯陆测试窗口，先做一次不计时的预热，CPU，单线程，不用 GPU：

| 步骤 | 时间 |
|---|---:|
| 一次完整预测（载体预测、带状态规则的锚定层、峰值读出） | 每个窗口 0.584 秒 |
| 载体在起报点和各伪起报点上的预测（平均 7.0 个发布时刻）× 3 个种子 | 每个窗口 0.075 秒 |
| 一次载体预测（一个发布时刻、一个种子；平均） | 0.0036 秒 |
| 给定载体预测时的锚定层，带状态规则 | 每个窗口 0.509 秒 |
| 峰值读出 | 每个窗口 0.0003 秒 |
| 载入三个检查点（一次） | 0.045 秒 |

大部分时间花在锚定层上，而不是载体上。一个检查点为 0.2657 MB。保存的评估预测是在 GPU 上算的；在 CPU 上，本包复现它们的最大相对差为：载体预测 1.98 × 10⁻⁶，交付轨迹 1.47 × 10⁻⁶（文件中的 `check:` 行）。

### 配方历史：R 是怎样选定的，以及它的负面结果

配方是 ankyra-lite 最后定下的部分。它的第一阶段（从零训练，在固定的 93 天验证段上早停）把三个短记录人群 23.7–38.3% 的自身窗口拿出了训练，也连带拿走了它们部分测试月份需要的季节。研究比较了同一批第 1 阶段检查点的几种继续训练（各三个种子，在全部九个人群的测试窗口上；德拉门和 CINELDI 用修正后的数据仓库）。计划及其预测是在对各方案唯一一次读数之前写下的；这些测试窗口此前已经读过很多次，所以这次读数给出的是方向，不是确认。作者于 2026 年 10 月 19 日选定 R，并披露下面的损失。

| 方案 | 做法 | `recipe_history.csv` 中的代码 |
|---|---|---|
| 只有从零训练阶段 | 只有第 1 阶段 | `P3s` |
| 普通继续训练（对照） | 第 1 阶段，然后用新的优化器再训练 8 轮，验证集仍然留出 | `F` |
| 验证段归还（**已发布**） | 同 F，另把验证段归还给训练集 | `R` |
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

- **验证段归还并非处处更好。** 它的收益出现在训练集缺少测试月季节的地方，与计划的预测一致；九个人群合并后，它与普通继续训练分不出高下；在德拉门上它更差（单点损失最大，逐时误差、电量和峰值都是如此）；在有开发窗口的六个人群上，第 2–7 个预测日明确更差。这些损失是发布配方的一部分。
- **机制没有被分离出来。** 归还验证段增加了同季数据，但也改变了训练数据的新近程度和多样性，以及旧窗口的重放；这些读数支持季节解释，但不能证明它。
- **按比例划分的验证集没有带来收益**：对普通继续训练，在四个长记录人群上为 −0.35% [−1.15, +0.43]，在三个短记录人群上为 +0.39% [−0.89, +1.65]，都不明确；在切换层上它明确差于验证段归还（−2.15% [−3.96, −0.36]）。
- **单纯继续训练的作用。** 相对只有从零训练阶段（九个人群；只有单位与月份区间，没有种子层）：普通继续训练 +0.97% [+0.55, +1.52]，验证段归还 +1.08% [+0.36, +1.86]，按比例验证 +0.86% [+0.56, +1.32]。九个人群合并时，R 与从零训练阶段之间的差别，大部分来自训练得更久，而不是来自归还的验证段。在德拉门上，R 比只有从零训练阶段差 2.30%。

![配方历史](figures/fig18_recipe_history.png)

*图 18. 发布配方背后的训练配方研究，在测试窗口上（此前读过：是描述，不是确认）：按人群组和分层的 R 对 F、V 对 F 和 V 对 R（a），按读出的 R 对 F（b），以及按人群的 R 对 F（c）。脚注列出所有明确不利于 R 的对比。*

## ankyra-lite 的薄弱之处

对外部基线，在这些测试窗口上：

- **住户的逐时误差。** 在 GoiEner 住户上，TimesFM（−5.6%）和 Chronos-2-X（−3.8%）的逐时误差明确更低；按 MAE 和 WAPE，ankyra-lite 排在四种零样本变体之后，位列第五。它在那里的月电量和峰值误差明确低于全部四种。
- **月初几天。** 第 1 天，它在六个测试人群上排 22 个中的第 4 到第 15（BDG2 第 15），在全部六个人群上落后于零样本基础模型，在五个人群上还落后于训练型基线。
- **短记录与零样本基础模型。** 在没有开发窗口、训练记录最短的 BDG2 和 GoiEner 非住户上，以及在 HEEW 和 EWELD 上，与四种零样本变体的逐时对比都不明确，BDG2（Chronos-2-X −5.8%，TimesFM −1.0%）和 GoiEner 非住户（Chronos-2-X −2.4%）的点估计为负。在两个 GoiEner 人群上，它没有一个预测日的逐时损失是最低的。
- **HEEW 和 EWELD 上的电量。** 对 TimesFM 的月电量，HEEW 为 −1.6%，EWELD 为 −5.3%（不明确）；在 EWELD 上，第 31 天的电量在 22 个中排第八。
- **绝对误差类指标。** 按 MAE，它在四个人群（住户、HEEW、EWELD、CINELDI）上排第四或第五；按中位单位 WAPE，在四个人群（GoiEner 非住户、住户、HEEW、EWELD）上也是如此；按中位单位 MASE，在五个人群上排第五或第六，大多落后于零样本基础模型。
- **峰值对简单参照。** 用在 GBT-T 轨迹上的同一读出在德拉门上明确更接近（−10.9%），上个月的实测峰值在 CINELDI 上明确更接近（−20.4%）；在 GoiEner 非住户和 EWELD 上，对上个月峰值的点估计为 −35.8% 和 −20.2%。
- **区间对 Chronos-2。** 放在 ankyra-lite 周围的区间在 EWELD、GoiEner 非住户和 CINELDI 上明确差于 Chronos-2 的原生分位数（80% Winkler 分数）。

它自身设计上的弱点在出现的地方报告：配方在德拉门和第 2–7 个预测日上的损失（[配方历史](#配方历史r-是怎样选定的以及它的负面结果)），以及上下文中尖峰的影响（[稳健性](#对受损上下文的稳健性)）。

## 局限

- **九个人群上只有复用评估。** 九个人群的比较中没有一个测试集是设计时没看过的；配方是在读过这些窗口之后选定的。外部检验是本候选在此前被读过的人群上的首读和一项同源跨时段检验，不是在未见过的人群上的检验（[外部检验](#外部检验)）。
- **每个人群一个载体。** 每个检查点为一个人群训练，只在该人群的测试窗口上评分。把检查点迁移到其他建筑或人群没有评估过。
- **借来的窗口。** 载体的合并训练集里有其他研究人群的窗口（最多另外八个人群和苏州园区；剑桥没有），每个都取自该人群自己的训练期或验证段，并在目标人群的验证段开始之前结束；没有借来的窗口落在另一个人群的测试期内（[docs/DATA.md](docs/DATA.md#windows-borrowed-from-other-populations)）。因此九个载体共享数据，彼此并不独立；起报点距检查点截点不足 186 天时，它的伪起报点会落进它的训练期。
- **日历说明。** 预测月的日类型按起报时已知处理，ankyra-lite 和每个读日历的基线都是如此。在 **EWELD 的 523 个测试窗口中有 26 个**（起报点 2020-01-01），其假日日历包含 2020 年 1 月 27 日才公布的 2020 年春节假期延长；在**剑桥的 730 个测试窗口中有 10 个**（起报点 2022-09-01），包含 2022 年 9 月 10 日才公布的 2022 年 9 月 19 日银行假日。这些预测没有用起报时已知的日历重做；结果文件包含这些窗口（[docs/DATA.md](docs/DATA.md#calendar-caveat-holidays-announced-after-the-origin)）。
- **BDG2 与 TimesFM 2.5 预训练的重叠。** 七个 BDG2 站点（Bear、Bull、Cockatoo、Fox、Hog、Panther、Rat）列在 TimesFM 2.5 模型卡称为训练数据的 GIFT-Eval 预训练集合中。这可能使 TimesFM 和 TimesFM-X 在 BDG2 的比较中占优，程度没有测量。ankyra-lite 没有预训练权重（[docs/DATA.md](docs/DATA.md#bdg2-and-the-pretraining-collection-of-timesfm-25)）。
- **短记录。** BDG2 2017 和两个 GoiEner 人群的训练记录最短，没有开发窗口，测试集只覆盖某一年的 5–8 个月。
- **新近程度。** 载体的训练标签在 C′ 结束，训练型基线的在 C 结束，晚 186 天；这是让伪起报点保持样本外的代价。
- **基线。** 发布版训练型基线的预算是 15 轮、patience 3，LSTM-T-KAN-M 是 30 轮和 8；这里没有在更大的预算下重新核对判定。线性上下文模型的协变量不起作用。GBT-T 在 EWELD 上过冲。LSTM-T-KAN-M 不是对其论文的严格复现（[十处偏离](#基线)）。
- **区间**是描述性的；区间读出不是为这个载体设计或调整的。
- **GPU 上的训练不能逐位重复。** 三个种子取平均并报告了它们的差异，但主要表格的区间只重抽单位和月份，不包含重新训练。
- **多重比较。** 21 个基线 × 9 个人群 × 几种量，给出数百个区间；明确对比的个数是描述性的，不是经过多重比较校正的检验。
- **继承自锚定层的限制**：只支持逐时数据，只支持 744 小时的预测长度，起报点处需要完整的 1,344 小时上下文，类别 Commercial 没有温度特征先验（13 个 BDG2 单位属于这一类），年温度谐波的相位是北半球的。

## 外部检验

本版本在九个人群测试窗口以外的数据上做了检验；这一轮在 2026 年 10 月 19 日、在这些数据上的任何训练开始之前冻结。每个测试集都用发布的配方 R 单独训练载体（六个种子；三个种子的发布配置是次要读数），由未改动的锚定层锚定，状态窗口用季节性朴素规则，并在预测记下哈希之后只评分一次；没有使用仓库附带的检查点。完整报告（英文）：[docs/EXTERNAL_TESTS.md](docs/EXTERNAL_TESTS.md)；每个数字都是 [`results/external_tests.csv`](results/external_tests.csv) 中的一行，由 `evaluation/external_tables.py` 从评分记录中复制。

- **LCL、UNICON 和 HKUST：本候选在此前被读过的人群上的首读。** 研究中更早的预测器（包括小网络的早期版本）在这些人群上评过分，锚定层是在这些读数之后定下的。HKUST 是次要测试集，样本小。
- **GoiEner 住户跨时段：同源跨时段补充检验**（研究人群 6 同一来源的新住户和更晚的时段），不是外部新人群。
- **赫尔辛基回顾：已训练、锚定并封存，尚未评分。** 它的评分原本与赫尔辛基前瞻检验挂钩，前瞻检验暂不进行（作者 2026 年 10 月 19 日决定），封存的结果保持不读。**UCI370：本轮跳过。**

主读数，逐时误差，六个种子（\* 表示明确更好；不设通过线）。它们是事先写定的 15 个名义 95% 区间中的 12 个（3 个对比 × 5 个测试集，赫尔辛基的 3 个封存），没有做多重比较校正；如果真实差别都不存在，15 个中约有 0.75 个会偶然显示“明确”。LCL 在 TimesFM 2.5、Chronos 和 Chronos-2 的预训练数据中；GoiEner 是否在其中未核实；UNICON 和 HKUST 不在公开的预训练清单上，但这不能证明不在其中。

| 测试集 | 结果身份 | 窗口 / 单位 / 月 | 对 TimesFM 2.5 | 对周季节性朴素 | 对带状态规则的无载体对照 |
|---|---|---|---|---|---|
| LCL 住户（伦敦） | 本候选在此前被读过的人群上的首读（回顾性外部重评） | 710 / 710 / 5 | +2.68% [−1.22, +6.43] | +24.32% [+17.65, +30.46] * | +3.08% [+2.08, +4.07] * |
| UNICON（拉筹伯大学各校区） | 本候选在此前被读过的人群上的首读（回顾性外部重评）；锚定层和网络家族在设计时已见过这些目标 | 450 / 59 / 23 | −5.54% [−19.45, +6.76] | +9.15% [−3.73, +20.42] | +8.00% [+3.69, +12.11] * |
| HKUST 校园进线电表（次要） | 本候选在此前被读过的人群上的首读（回顾性外部重评）（次要测试集，样本小） | 89 / 29 / 8 | +10.24% [−9.22, +26.23] | +25.65% [+6.33, +40.98] * | +3.95% [−4.11, +11.38] |
| GoiEner 住户跨时段（补充） | 同源跨时段补充检验（新单位、新时段），不是外部新人群；与研究人群 6（GoiEner 住户）的结果相关 | 1,417 / 282 / 6 | +4.19% [+1.59, +6.72] * | +25.99% [+23.87, +28.05] * | +5.48% [+3.59, +7.34] * |

- 在已评分的四个测试集上（三个是本候选在此前被读过的人群上的首读，一个是同源跨时段补充检验），12 个主读数区间中有 7 个明确更好、5 个不明确、没有明确更差；三个种子的发布配置在全部 12 个上标签相同。
- **对零样本 TimesFM 2.5**，只有 GoiEner 检验明确（+4.19%，同源跨时段补充检验；GoiEner 是否在其预训练数据中未核实）。在 LCL（+2.68%；LCL 在 TimesFM 2.5 的预训练数据中）、UNICON（−5.54%）和 HKUST（+10.24%）上（本候选在此前被读过的人群上的首读），区间都含零；不明确不等于没有差别。
- **第 1 个预测日**，在 LCL（−16.11%；LCL 在 TimesFM 2.5 的预训练数据中）和 UNICON（−35.95%）上明确差于零样本 TimesFM 2.5（本候选在此前被读过的人群上的首读）；在 UNICON 上，ankyra-lite 的单位平均名次是 14 个中的第 5，排在四种零样本基础模型变体之后（本候选在此前被读过的人群上的首读）。

![外部检验](figures/fig17_external_tests.png)

*图 17. 外部检验，每个测试集评分一次：主读数（a–c；圆点为六个种子，灰色菱形为三个种子的发布配置）、按预测日的同样对比（d–f），以及与 ankyra-lite 自身简化配置的描述性对比（g–i）。实心：95% 区间不含零；区间端点超出坐标轴时画成箭头。LCL、UNICON、HKUST：本候选在此前被读过的人群上的首读；GoiEner 住户跨时段：同源跨时段补充检验；赫尔辛基：封存，未评分。*

## 检查点

仓库附带发布配方的 27 个载体，即研究运行的第 2 阶段最终权重，由 `tools/export_checkpoints.py --out checkpoints` 逐字节复制：

```
checkpoints/<population>/seed<s>.pt     population in {bdg2, cambridge, heew, eweld, goiener_confirm, households,
                                                       oslo, drammen, cineldi}; s in {0, 1, 2}
checkpoints/MANIFEST.json               per file: SHA-256, bytes, parameters, training cut (hour index and date),
                                        hour 0 of the record, store, source run and its record's SHA-256, the stage-1 run
```

- **格式。** 每个文件是 `{"model": state_dict, "epoch": 8}`（265,653 字节）。`ankyra_lite.load_carrier(path)` 构建网络并严格载入，放在 CPU 上，处于 eval 模式；它也接受裸状态字典。
- **核对。** 导出时把每个副本与其运行记录中的 SHA-256 核对，严格载入，并数出 61,613 个参数。对奥斯陆和 CINELDI（修正后的数据仓库）各三个测试窗口、每个种子，它在 CPU 上于相同的发布时刻复现保存的 GPU 载体预测；规则是相对差小于 10⁻⁴，`MANIFEST.json`（`verification`）中记录的最大值为 2.1 × 10⁻⁵（CINELDI，种子 2）。
- **配合该人群自己的截点使用。** 一个检查点属于一个人群及其训练截点。用 `cut=` 传入截点（一个记录的小时下标，该记录的第 0 小时见下表），并使用至少在截点之后 186 天的起报点：

| 文件夹 | 人群 | 记录的第 0 小时 | 训练截点（`training_cut_hour`） | 训练截点（日期） | 数据仓库 |
|---|---|---|---:|---|---|
| `bdg2` | BDG2 2017 | 2016-01-01 | 9,408 | 2017-01-27 | 原始 |
| `goiener_confirm` | GoiEner 非住户 | 2018-01-01 | 7,176 | 2018-10-27 | 原始 |
| `households` | GoiEner 住户 | 2018-01-01 | 8,640 | 2018-12-27 | 原始 |
| `cambridge` | 剑桥 | 2000-01-01 | 113,856 | 2012-12-27 | 原始 |
| `heew` | HEEW（亚利桑那） | 2014-01-01 | 38,616 | 2018-05-29 | 原始 |
| `eweld` | EWELD | 2016-01-01 | 26,928 | 2019-01-27 | 原始 |
| `oslo` | 奥斯陆 | 2012-01-01 | 63,456 | 2019-03-29 | 原始 |
| `drammen` | 德拉门 | 2018-01-01 | 18,912 | 2020-02-28 | 修正（训练池中有 CINELDI 窗口） |
| `cineldi` | CINELDI | 2019-01-01 | 15,240 | 2020-09-27 | 修正（第 0 小时确为 2019-01-01） |

```python
from ankyra_lite import load_carrier, forecast
carriers = [load_carrier(f"checkpoints/oslo/seed{s}.pt") for s in (0, 1, 2)]
# history: an Oslo school's ankyra_lite.anchor.History whose hour 0 is 2012-01-01, ending at an origin >= 63,456 + 4,464;
# sigma: Oslo's temperature-anomaly scale fitted on days before the cut
r = forecast(history, carriers, group="Public", temp_sigma_std=sigma, dst_region="EU", cut=63456)
```

- 不要把自己的训练运行写进 `checkpoints/`：`.gitignore` 忽略除这 27 个以外的所有权重文件。

## 测试

不需要 `pytest`；测试用 `unittest`。在仓库根目录下运行：

```bash
python -m unittest discover -s tests -t .      # 176 tests
python -m unittest discover -s theory -t .     # 58 checks of the exact properties
```

| 模块 | 测试数 | 检查内容 |
|---|---:|---|
| `tests/anchor/` | 62 | 锚定层自己的测试，移植到副本上（起报点之后无信息、交接、状态规则、缺口容忍、相似日形状、读出、参照估计器） |
| `tests/test_anchor_equivalence.py` | 14 | 副本与 ANKYRA 2.2.1 逐字节相同，并给出逐位相同的预测、读出和指标（需要 ANKYRA 仓库副本） |
| `tests/test_carrier_unit.py` | 14 | 参数个数、形状、不读发布时刻及之后的数据、伪起报点规则、集成、端到端预测、状态规则 |
| `tests/test_carrier_equivalence.py` | 7 | 特征、初始化和预测对研究代码与保存的预测的一致性（需要研究的数据） |
| `tests/test_train.py` | 19 | 抽样器、划分、上限、训练池、归还的验证段，以及每个阶段的一步优化对研究训练循环的一致性（部分需要研究的数据） |
| `tests/test_compare_baselines.py` | 33 | 比较脚本的保护机制、产物、检查点导出，以及结果文件：列、预测器集合、标记、标签、没有绝对路径（部分需要 ANKYRA 发布版文件、研究的数据或 `results/`） |
| `tests/test_evaluation.py` | 15 | 评分模块在人造数据上的表现 |
| `tests/test_external_tables.py` | 12 | 外部检验表：列、预测器名称、没有绝对路径、从不读取封存的赫尔辛基测试集、与评分记录逐字段一致、不含评分器其他输出文件中的任何对比、docs/EXTERNAL_TESTS.md 的主读数表（部分需要研究的评分记录） |
| `theory/test_operators.py`、`theory/test_carrier_properties.py` | 31、27 | P1–P20 和 C1–C10，含反例 |

在作者的机器上，在有 ANKYRA 仓库副本和研究数据的条件下，并在重新生成的 `results/` 上（2026 年 10 月 19 日，CPU，单线程），176 个测试中 175 个通过、1 个跳过，58 项理论检验全部通过。跳过的那个测试把逐时数字与更早那次只看逐时误差的比较核对，而那次比较评分的是从零训练阶段，不是发布的产物。需要 ANKYRA 仓库副本或研究数据的测试，在缺少这些时会跳过并说明原因（CI 上就是这样）：研究文件夹不可用（ANKYRA 仓库副本和 `results/` 都在）时，同一天的运行在 176 个测试中有 147 个通过、29 个跳过，58 项理论检验中 58 项通过。环境变量：`ANKYRA_ORIGINAL_ROOT`（ANKYRA 仓库副本）、`ANKYRA_LITE_RESEARCH_ROOT`（研究文件夹）、`ANKYRA_LITE_RESULTS`（要核对的另一个结果目录）和 `ANKYRA_LITE_TEST_MIN_FREE_GB`（读数据的测试要等待的空闲内存）。

## 理论：精确性质

![测试窗口上的精确性质](figures/fig7_operators.png)

*图 7. 锚定层在图 6 那个剑桥窗口上的精确性质：从单独的载体出发的块替换（a，P4）、历史支撑（b，P7）、收缩界（c，P8、P9）、裁剪与保电量投影的对比（d，P11、P12）、峰值读出及其误差分解（e、f；P15、P16）。*

这套构造可以审计，因为每一步都有明确陈述的性质。它们放在预测器旁边的 [theory/](theory/README.zh-CN.md) 中，包本身不导入它：

- **P1–P20，锚定层**（ANKYRA 2.2.1 理论部分的性质，只改了导入）：精确的块恒等式和块替换（P1–P6），权重需要多少历史、收缩让它们最多移动多远（P7–P10），为什么在裁剪之前读电量（P11–P13），峰值算子（P14–P18），为什么报告几种估计目标（P19），以及对水平权重做强收缩的最坏情形背景（P20）。
- **C1–C10，载体网络**（新增）：非负的轨迹，均值等于水平（C1、C2）；份额和指针注意力都是概率分布（C3）；不用发布时刻及之后的任何输入（C4）；因果的、遵守掩码的编码器（C5、C6）；尺度等变（C7）；没有单位标识（C8）；复制指针从零开始，并且从不改变水平（C9）；以及让每个打分的伪起报点预测都是样本外的 186 天间隔（C10）。

[theory/PROOFS.md](theory/PROOFS.md) 给出陈述、证明和反例；`theory/operators.py` 把 P1–P20 写成函数；两个检验文件包含 58 项数值检验。这些性质都不是精度保证；精度是测出来的。

## 可复现性

- **评分代码核对。** 在把 ankyra-lite 放到参照位置之前，`evaluation/compare_baselines.py` 用发布版自己的参照预测器重新计算 ANKYRA 2.2.1 发布版十张结果表中截点后窗口的各行，并按公布时的舍入，逐字段与发布仓库提交 `487e0d2` 的文件比较（用 `git show` 只读地读取）。十张全部复现：94,131 个比较字段中 0 处不一致（[`results/REPRODUCTION_CHECK.json`](results/REPRODUCTION_CHECK.json)）。不能复现的表不会写出，重新计算出的发布版数字不在这里的任何文件中。
- **产物核对。** 对每个人群，锚定后的产物必须正好具有面板的测试窗口、状态标记和目标，否则脚本停止。由锚定后的产物重新计算的各人群 R 对 F、R 对从零训练阶段的对比，在 9 个人群中的 9 个上与研究自己的评分器结果相同（最大差 0.0 个百分点；`results/compare_baselines.json` 中的 `study_scorer_crosscheck`）。对三种子集成重新锚定，逐位复现了每个保存的产物。
- **包对研究代码。** 锚定层与 ANKYRA 2.2.1 逐字节相同，并给出逐位相同的预测；载体的特征、初始化和预测，以及训练配方，都复现研究代码（[测试](#测试)）。在 CPU 上，本包复现保存的 GPU 预测，相对差约为 10⁻⁶（[耗时](#每个窗口的耗时)）。
- **哪些预测被评分。** `evaluation/products.json` 按人群列出发布配方的锚定后三种子产物，以及它背后的研究运行（路径相对研究文件夹）。
- **核对一个数字。** 上面表格中的每个数字都是 `results/` 中某个文件的一行。例如总览中剑桥那一行：

  ```python
  import csv
  rows = [r for r in csv.DictReader(open("results/benchmark_pairwise.csv", encoding="utf-8")) if r["set"] == "Cambridge"]
  print(sum(float(r["um_high"]) < 0 for r in rows), "of", len(rows))      # 21 of 21 resolved better
  ```
- **哪些不能只靠本仓库重新运行。** `evaluation/compare_baselines.py`、`tools/export_checkpoints.py` 和依赖数据的测试，要读研究的本地数据仓库、保存的预测和 ANKYRA 仓库副本（只读；[evaluation/README.md](evaluation/README.md)）；`evaluation/external_tables.py` 要读外部检验的评分记录（[docs/EXTERNAL_TESTS.md](docs/EXTERNAL_TESTS.md#files)）。`python -m evaluation.run_example` 用发布的检查点在十二栋人造建筑上演示数据流程（其数字没有意义），`python -B figures/make_figures.py --strict` 从 `results/` 重新生成所有图。

## 版本历史

- **0.1.0（2026 年 10 月 19 日）**：第一个版本。复制指针载体网络（61,613 个参数），配上逐字节相同的 ANKYRA 2.2.1 锚定层和季节性朴素状态规则；发布的训练配方 R（第 1 阶段从零训练，在 93 天验证段上早停；第 2 阶段归还验证段训练 8 轮；三个种子）；27 个检查点及其清单；CINELDI 用修正后的数据仓库，德拉门载体也用它重新训练；与 21 个基线（发布版的 20 个和 LSTM-T-KAN-M）在九个人群上的比较，用发布配方全量重新生成；载体的精确性质 C1–C10。
- **0.1.0 之前的设计历史**（研究中的各个载体，都由同一个锚定层锚定；`ablation.csv` 和 `recipe_history.csv`）：一个更早的、没有复制指针、在按周窗口上训练的结构化网络；同一网络去掉复制指针、用从零配方训练；更早的几项继续训练研究（不加权和加权的损失，有和没有预热）；只有从零训练阶段的复制指针网络；以及继续训练方案 F、V 和 R，其中发布的是 R。

## 来源、许可与引用

- **锚定层。** `ankyra_lite/anchor/` 是 ANKYRA 2.2.1 的 `ankyra` 包（仓库 [XinanQ/Building_load_ANKYRA](https://github.com/XinanQ/Building_load_ANKYRA)，提交 `487e0d2`，其 `ankyra/` 目录树与标签 `v2.2.1` 相同），逐字节复制；[`PROVENANCE.md`](ankyra_lite/anchor/PROVENANCE.md) 列出每个文件的哈希。它的文档字符串仍写作 `ankyra.*`，请读作 `ankyra_lite.anchor.*`。不包含 `ankyra.timesfm_adapter`。这 17 个复制的文件从不改动。
- **载体。** 网络、特征、推理和训练代码复现研究代码（参数名和顺序、初始化一致，特征逐位一致）。
- **测试窗口、发布版的 20 个基线和评分函数**都来自 ANKYRA 2.2.1 发布版。
- **许可。** MIT（[LICENSE](LICENSE)），Copyright (c) 2026 Xinan Qin；锚定层使用同一许可。LSTM-T-KAN 的代码（MIT）不包含在内；只对它的预测评了分。
- **数据。** 评估使用公开数据集：BDG2（Building Data Genome Project 2）、剑桥大学校产档案、HEEW（亚利桑那州立大学）、EWELD、GoiEner、COFACTOR-SBHUB Oslo 和 COFACTOR Drammen、CINELDI，以及苏州工业园区（只用作训练窗口），都使用 CC BY 4.0 许可；来源和整理见 [docs/DATA.md](docs/DATA.md)。这些数据都没有再分发；检查点包含在这些数据上拟合的权重，不包含数据本身。`results/` 中的示例窗口来自剑桥大学校产档案（CC BY 4.0）。
- **引用。** [CITATION.cff](CITATION.cff)；也请引用 ANKYRA（列在其中的 references 下）。存档 DOI 尚未分配。

问题和错误报告：[issue 跟踪页](https://github.com/XinanQ/Building_load_ANKYRA_lite/issues)。
