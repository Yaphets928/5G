# QPSO_cleaned 项目说明

## 1. 项目概览

本仓库当前是一个**单文件仿真脚本项目**，核心文件为 `QPSO_cleaned.py`。脚本实现了一个面向异构蜂窝网络（微站+宏站）的联合优化流程，结合：

- 多时隙（`Xi`）场景建模（白天/夜晚业务负载）
- UA（User Association）离散优化（OT + rounding）
- 波束成形迭代优化（WMMSE）
- 外层智能优化（QPSO/PSO）
- 多快照 Monte Carlo 评估

程序入口为 `main()`，默认会进行多轮快照仿真并打印统计结果。

---

## 2. 代码结构分析

`QPSO_cleaned.py` 可以按“从底层物理建模 → 中层优化 → 外层搜索 → 主流程调度”理解：

### 2.1 全局配置与状态

- 环境变量初始化（例如 `XI`）用于控制时隙数量。  
- 一组全局常量用于初始化策略、随机种子与动态负载记忆：
  - `INIT_SPACING_MULT_MICRO` / `INIT_SPACING_MULT_MACRO`
  - `UA_PREV_LOADS` / `UA_EPS_LOAD`
  - `CRE_BIAS_DB` / `CRE_INIT_USED`
  - `CHAN_EPOCH`

这些变量影响 OT 负载平衡、信道随机性注入与初始化稳定性。

### 2.2 信道与阵列建模层（基础物理层）

主要函数：

- `steering_matrix_from_pos_3d(...)`：构建方向向量矩阵。
- `fspl_power(...)`：自由空间路径损耗功率。
- `make_rx_array(...)`：构建 UE 侧接收阵列。
- `build_H_for_link_tx_UPA(...)`：单链路 MIMO 信道构建。
- `build_channels(...)`：批量构建所有 UE 与所有基站的信道矩阵。

这一层输出 `H[k][j]` 信道对象，供后续 UA/WMMSE 使用。

### 2.3 关联与资源分配层（UA + OT）

主要函数：

- `_ua_cre_init(...)`：首次 UA 的 CRE 粗初始化。
- `lp_and_integerize(...)`：
  - 先做 OT（Sinkhorn）得到软关联 `a_frac`
  - 再做整数化得到 `a_int`
  - 包含 dummy BS（阻塞建模）、容量约束、局部搜索修复
  - 使用 `UA_PREV_LOADS` 实现 `1/U` 的动态负载权重

这一层的核心作用是：把连续代价优化结果转换为符合容量约束的离散关联矩阵。

### 2.4 波束成形优化层（WMMSE）

主要函数：

- `wmmse_given_a(...)`：在固定 UA 下执行加权 WMMSE，输出频谱效率矩阵与收发波束相关变量。
- `inner_wmmse_cvxp(...)`：
  - 先通过简化 SINR 构造 surrogate rate
  - 与 `lp_and_integerize(...)` 交替迭代
  - 在最优迭代点返回 `best_Rspeff`、`best_a`、`best_obj`

该层将“离散 UA”与“连续波束”弱耦合地联合优化，是项目的算法核心。

### 2.5 外层全局搜索层（QPSO / PSO）

主要函数：

- `run_qpso_unified(...)`：量子粒子群优化（当前主流程默认调用）。
- `run_pso_unified(...)`：标准粒子群优化（保留对照版本）。
- `evalUnified(...)`：统一目标函数，将 `Xi` 个时隙的加权目标聚合。
- `UnifiedPack`：
  - `pack/unpack`：几何参数向量化与反向恢复
  - `clip`：约束修复（边界与阵元最小间距）

这一层负责搜索宏站位置、微站位置和各时隙 MA offset 等高维决策变量。

### 2.6 主流程与实验控制

- `main()` 负责：
  1. 解析配置（含 `XI`、FAST 模式、PSO 参数覆盖）
  2. 生成 UE 与场景快照
  3. 运行外层优化与每时隙评估
  4. 汇总 Monte Carlo 结果
  5. 可选绘图（matplotlib）

从工程组织上看，`main()` 更偏实验脚本式编排，尚未拆分为模块化 package。

---

## 3. 当前工程结构优缺点

### 优点

- 算法链路完整：信道→UA→WMMSE→QPSO→MC 评估闭环。
- 关键函数注释较充分，能看出论文/实验导向设计思路。
- 提供了 QPSO 与 PSO 两套外层优化器，便于对比实验。

### 可改进点

- **单文件过大**：维护、测试与复用成本较高。
- **全局变量较多**：`UA_PREV_LOADS`、`CHAN_EPOCH` 等带来隐式状态耦合。
- **配置管理分散**：环境变量与硬编码参数混用，复现实验不够标准化。
- **可测试性偏弱**：缺少单元测试与基准测试脚本。

---

## 4. 建议的模块化拆分（后续可选）

建议按如下结构重构：

```text
.
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── src/
│   ├── channel.py          # 信道/阵列构建
│   ├── ua_ot.py            # OT + rounding + CRE
│   ├── wmmse.py            # 固定UA下的WMMSE
│   ├── objective.py        # evalUnified/多时隙聚合
│   ├── optim_qpso.py       # QPSO
│   ├── optim_pso.py        # PSO
│   └── packing.py          # UnifiedPack与clip约束
├── scripts/
│   └── run_mc.py           # 主实验入口
└── tests/
    ├── test_channel.py
    ├── test_ua_ot.py
    ├── test_wmmse.py
    └── test_packing.py
```

---

## 5. 运行说明（基于当前仓库）

### 5.1 环境准备

建议 Python 3.9+，依赖至少包含：

- `numpy`
- `matplotlib`（若需绘图）
- `cvxpy`（可选，脚本会自动检测是否可用）

### 5.2 直接运行

```bash
python3 QPSO_cleaned.py
```

### 5.3 常用环境变量

- `XI`：时隙数量（要求偶数且 `>=2`）
- `FAST_MODE=1`：快速模式（降低粒子数与迭代数）
- `PSO_PARTICLES`：覆盖粒子数
- `PSO_ITERS`：覆盖迭代数

示例：

```bash
FAST_MODE=1 XI=4 PSO_PARTICLES=6 PSO_ITERS=3 python3 QPSO_cleaned.py
```

---

## 6. 结果输出说明

脚本会输出：

- 每个快照下的优化过程日志
- 每时隙原始/加权目标值
- 每个夜间 UE 规模下的 Monte Carlo 平均加权和速率
- 明细 `(nightUE, epoch, weighted_sum_rate_bps)`

如环境支持图形界面，还会弹出代表性 day/night 场景关联可视化。

---

## 7. 总结

这是一个典型的“研究型仿真脚本”：算法完整、实验导向强，但工程化程度（模块化、配置化、测试化）还有较大提升空间。若下一步要做长期迭代，优先建议先拆模块并补充自动化测试。
