# Inductor Pass NPU 调研：现状、环境选择与背景知识

## 1. 当前结论

截至 2026-08-17，本任务已经完成静态清单、候选问题分级、运行探针草案和变更控制框架，但尚未形成任何可信的 NPU pass 性能结论，也没有修改 PyTorch、torch_npu 或 Triton 源码。

当前必须暂停动态测试。用户已确认有其他进程正在修改 `/home/z50063656/Benchmark` 相关环境；依赖、源码路径、动态库和编译缓存都有可能变化。在环境锁定前，已有动态结果只用于诊断，不能作为 baseline。

关于“使用 PyTorch wheel 还是源码编译”，建议如下：

| 阶段 | 推荐形态 | 是否需要源码编译 |
|---|---|---|
| 静态枚举和源码机制分析 | 直接读取 PyTorch/torch_npu 源码树 | 不需要 |
| 第一轮可用性与性能基线 | 独立、冻结、版本严格匹配的 PyTorch + torch_npu + Triton Ascend wheel 环境 | 通常不需要，优先使用 wheel |
| 修改纯 Python pass、pattern、lowering | 对应 commit 的源码树，加受控源码导入或构建出的 wheel | 通常不需要重编 C++，但必须保证运行代码确实来自该源码 |
| 修改 PyTorch/torch_npu C++、注册、ABI 或编译扩展 | 对应 commit 的源码树 | 需要构建，建议产出 wheel 后安装到任务专属环境 |
| 只开发独立手写 Triton kernel | 匹配的 Triton Ascend + PyTorch/torch_npu wheel | 通常不需要重编 PyTorch |
| 将 Triton kernel 正式接入内置 pass | 取决于接入点；Python 注册可不重编，C++/打包变更需要构建 | 按修改范围决定 |

因此，本任务不应一开始就全量源码编译。先用冻结 wheel 环境建立可复现基线；确认某个 pass 需要修改后，再使用完全匹配的源码做开发。最终验证时至少要有一套干净 wheel 安装结果，避免 `PYTHONPATH`、源码树和 site-packages 混用。

## 2. 已完成工作

### 2.1 静态 pass 清单

已对以下源码范围做 AST/文本级静态扫描：

- `/home/z50063656/Dynamo/pytorch/torch/_inductor/fx_passes`
- `/home/z50063656/Dynamo/torch_npu/torch_npu/_inductor`
- NPU Triton experimental pass
- `inductor-npu-ext` 自定义 pass 入口

当前清单共 194 条记录。它们是 pipeline、pattern、扩展函数和 NPU 自定义 pass 的“审计记录”，不等于 194 个可以逐一独立运行的测试用例。

按阶段统计：

| 阶段 | 数量 |
|---|---:|
| inductor-extension | 74 |
| joint_graph | 13 |
| npu-custom | 3 |
| npu-ext | 5 |
| npu-pattern | 1 |
| npu-triton-experimental | 3 |
| post_grad | 69 |
| pre_grad | 26 |

按机制统计：

| 机制 | 数量 |
|---|---:|
| extension-function | 25 |
| npu-custom-pass | 27 |
| pattern-entry | 76 |
| pattern-registry | 25 |
| pipeline | 41 |

静态路由标签统计：

| 标签 | 数量 | 含义 |
|---|---:|---|
| backend-sensitive | 2 | 明确依赖后端行为，需要优先验证 |
| generic-needs-validation | 101 | 通用逻辑，但必须在 NPU 图和生成代码中验证 |
| generic-pipeline | 41 | pipeline/observer 类入口 |
| needs-npu-validation | 10 | 已发现 NPU 相关风险点 |
| npu-specific | 40 | NPU 专用逻辑 |

产物：

- `report/pass_inventory.json`：机器可读清单。
- `report/pass_inventory.md`：人工审阅索引。
- `audit_passes.py`：不导入 torch 的静态清单生成器。

### 2.2 已识别的首批候选

1. `mm_plus_mm`

   上游存在 `mm + mm -> tuned_mm_plus_mm` pattern；当前 torch_npu patch 对 NPU 禁用该匹配，NPU Python kernel 注册覆盖 `mm/addmm/bmm/grouped_mm`，但没有等价的 `mm_plus_mm` lowering。候选方向是 NPU lowering + vendor/CATLASS/AscendC；只有测量证明有收益时才考虑 Triton `tl.dot`。

2. `pad_mm`

   NPU Triton experimental backend 明确关闭 `shape_padding`。需要先确认问题是 layout、padding/slice lowering、动态 shape，还是 padding 本身在 NPU 上没有收益。不能只复制 CUDA 的 padding 策略。

3. `addmm` fusion

   NPU Triton experimental backend 可以关闭上游 add+mm -> addmm pattern。需要比较原始图、NPU addmm lowering、CATLASS/vendor 和 fallback，并按 dtype、shape、layout 做 capability gate。

三个候选都已在 `change_control.md` 登记为 `draft`，没有进入实现。

### 2.3 探针框架

`run_npu_probe.py` 已形成草案，设计目标包括：

- 每个 backend 使用独立进程，避免全局 monkeypatch 和 cache 污染。
- 比较 eager 与 compiled 正确性。
- 记录首次编译/执行延迟、稳态 mean/stdev/p50/p99、峰值 NPU 内存。
- 通过 `GraphTransformObserver` 和 `PatternMatcherPass` 观察实际运行的 pass。
- 缺少 NPU 时记录 `skip`，不能记为成功。

该脚本尚未在稳定的任务环境中验证，不应直接作为正式 benchmark。

## 3. 已知运行环境事实

通过 `/home/z50063656/Benchmark/env.sh` 启动时，曾采集到以下快照：

| 项目 | 诊断快照 |
|---|---|
| Python | 3.11.15，`/home/z50063656/envs/benchmark-py311/bin/python` |
| PyTorch | `2.14.0a0+git8e86e0a` |
| PyTorch 导入路径 | `/home/z50063656/Benchmark/pytorch-upstream/torch/__init__.py` |
| torch_npu | `2.14.0`，来自 site-packages |
| CANN | 9.0.1，`/usr/local/Ascend/cann9.0.1/cann-9.0.1` |
| NPU | 8 x Ascend910B2，设备 0-7 Health OK，单卡 HBM 65536 MiB |
| Triton | clean process 中 `triton.__version__=3.2.0` |
| Triton 元数据 | 同时存在 `triton=3.5.0` 与 `triton-ascend=3.2.1` 元数据 |
| torchair | 导入失败，缺少 protobuf 分发元数据 |

这是一套混合环境：PyTorch 从 Benchmark 源码树导入，torch_npu 从 site-packages 导入；静态清单分析的则是 `/home/z50063656/Dynamo` 下的另一组源码。两组源码没有完成 commit 对齐，因此不能把静态清单直接等同于当前运行时行为。

最小 eager NPU 加法曾成功：`npu:0`、fp16、`128x128`，输出 shape/device 正确且数值有限。首次样本约 6.78 ms，只能说明基本设备链路可用，不能代表性能。

最小 `torch.compile` 尚未进入 Inductor：导入 `torch._dynamo` 时遇到 NumPy ABI 错误：

```text
numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88 from PyObject
```

由于环境正在被其他进程修改，暂不判断这是最终环境缺陷，也不在当前环境中修包。

另有一个旧探针使用了不同环境，得到 CPU PyTorch、NPU 不可见等结果。该结果保留为历史诊断，不是 Benchmark 环境的 baseline。

## 4. 为什么优先使用 wheel 基线

wheel 基线的主要价值不是“安装方便”，而是可复现：

- PyTorch、torch_npu、Triton Ascend 和 CANN 的版本边界明确。
- 不会因为当前目录、`PYTHONPATH` 或源码树中未提交修改而改变导入代码。
- 每次 fresh process 使用相同二进制和 Python 文件，性能比较才有意义。
- 可将 baseline wheel 和 candidate wheel 安装到两个独立环境，做成真正的 paired benchmark。

建议的第一套正式环境是“发布/基线环境”：

1. 创建任务专属环境，不复用正在变化的 `benchmark-py311`。
2. 选择一组官方支持矩阵内的 PyTorch、torch_npu、Triton Ascend、CANN 版本。
3. 清除源码树注入的 `PYTHONPATH`，确认 `torch.__file__` 和 `torch_npu.__file__` 都指向该环境的 site-packages。
4. 固化 wheel 文件名、SHA256、Python 版本、CANN 路径、driver/npu-smi 版本和 SoC。
5. 运行 eager、Dynamo import、最小 Inductor compile，再开始 pass 测试。

## 5. 什么时候必须源码编译

以下情况需要 PyTorch 或 torch_npu 源码构建：

- 修改了 C++ dispatcher、设备注册、codegen extension 或 ABI。
- 修改的 commit 没有对应 wheel，且不能用纯 Python 覆盖验证。
- 需要验证 PyTorch 与 torch_npu 的跨仓库接口变更。
- 最终交付要求证明修改能从干净源码构建并打包。

推荐构建策略不是在工作环境里原地编译，而是：

1. 锁定 PyTorch commit。
2. 以该 commit 构建 PyTorch wheel。
3. 用这个 PyTorch wheel 对齐构建 torch_npu wheel。
4. 将两个 wheel 安装到新的 candidate 环境。
5. baseline 与 candidate 分环境测试，避免同一 site-packages 被覆盖。

纯 Python pass/pattern/lowering 的早期开发通常可以不重编 C++，但必须记录导入路径，并证明运行时加载了 candidate 文件。最终仍建议产出 wheel 做交付验证。

## 6. Inductor pass 背景

简化后的 `torch.compile(backend="inductor")` 链路如下：

```text
Python model
  -> TorchDynamo 捕获 FX graph
  -> AOTAutograd 拆分 forward/backward
  -> pre_grad passes
  -> joint_graph passes
  -> post_grad passes
  -> lowering 到 Inductor IR
  -> scheduler / fusion / codegen
  -> NPU backend kernel
  -> runtime execution
```

pass 的作用不只是一种“图融合”。常见类型包括：

- pattern rewrite：把一组算子替换成更适合后端的等价图。
- decomposition：把高级算子拆成后端已经支持的基础算子。
- canonicalization：统一 shape、layout、dtype 或表达式形式，帮助后续匹配。
- fusion：减少 kernel 数、访存和 launch 开销。
- lowering selection：为同一语义选择 vendor op、template、Triton 或 fallback。
- scheduler/codegen hook：改变分组、分块、并行和最终内核生成。

一个 pass 在 NPU 上“可用”，至少要同时满足：

1. 能触发：输入 FX graph 符合 pattern，设备 gate 没有把 NPU 排除。
2. 语义正确：dtype、shape、stride、动态 shape、forward/backward 和边界值正确。
3. 能 lowering：替换后的算子有 NPU lowering 或合法 fallback。
4. 能 codegen：backend 依赖、编译器和运行时能够生成、加载内核。
5. 没有隐性 graph break 或 CPU fallback。
6. 性能合理：稳定态和端到端收益抵消编译、额外内存与同步开销。

因此，“源码里注册了 pass”不等于“在 NPU 上可用”；“输出正确”也不等于“pass 实际触发”。必须同时检查 pass observer、FX 图、generated code、kernel 数和 profiler。

## 7. NPU backend 与替代实现

当前 torch_npu 代码中可见多种 Inductor NPU backend loader：`default`、`ascendc`、`mlir`、`dvm`、`triton_experimental`。典型选择方式是：

```python
torch.compile(
    fn,
    backend="inductor",
    options={"npu_backend": "ascendc"},
)
```

不同 backend 不是简单的性能开关。它们可能注册不同 lowering、scheduler、codegen 和 pass，且会修改 Inductor 全局状态。因此正式测试必须每个 backend 使用 fresh process。

当某个 pass 不可用或性能差时，替代方案的优先级应按问题类型选择：

| 问题 | 优先方案 |
|---|---|
| pattern 没触发 | 修 pattern、decomposition 或 device capability gate |
| 替换图正确但没有 NPU lowering | 增加 NPU lowering 或选择已有 vendor op |
| GEMM/attention/collective 性能差 | 先比较 vendor/CATLASS/AscendC，不默认手写 Triton |
| pointwise/reduction 多 kernel、访存主导 | 适合评估手写 Triton fusion |
| 动态 shape/layout 不支持 | 限定 capability gate，保留原图 fallback |
| Triton 编译失败 | 先修工具链/版本匹配，不能归因到 pass |

手写 Triton 是实现手段，不是目标。它最适合表达规则、可融合、访存主导的 pointwise/reduction；对于矩阵乘、attention 等算子，成熟 vendor kernel 往往有更完整的 tiling、流水和硬件特性支持。只有在正确性和 paired benchmark 都证明收益时，才能把 Triton 替代接入 pass。

## 8. 正式性能方法

每个 pass/case/backend 至少需要一组同机、同环境、fresh process 的 baseline/candidate 数据：

- CANN、driver、SoC、PyTorch、torch_npu、Triton 版本。
- 输入 dtype、shape、stride、dynamic/static、forward/backward。
- warmup 次数和 runs 次数；建议起点为 warmup 10、runs 100。
- 首次编译/首轮延迟。
- 稳态 mean、stdev、p50、p99。
- 峰值 NPU 内存、kernel 数、graph break/fallback。
- 每次计时前后正确同步 NPU。
- 记录设备占用，避免和其他进程共享同一张卡做性能结论。

验收建议分两层：

- 可用度替身：原 pass 在 NPU 不支持时，candidate 能正确执行、无非预期 CPU fallback，并有明确 capability gate 和回退路径。
- 性能提升：candidate 相对相同输入的 baseline 在稳态 p50 有明确收益，同时 p99、编译时间和峰值内存没有不可接受回退。

## 9. 环境稳定后的恢复顺序

1. 用户确认其他环境修改进程已结束。
2. 从 `/home/z50063656/tmp` 启动测试，并重新 source `/home/z50063656/Benchmark/env.sh` 或新的任务环境脚本。
3. 重新采集 `which python`、`torch.__file__`、`torch_npu.__file__`、版本、wheel 元数据、CANN、npu-smi 和 Triton backend。
4. 校验静态审计源码 commit 与运行环境 commit 是否一致；不一致则重新生成清单。
5. 先通过 eager、`import torch._dynamo`、最小 `torch.compile`。
6. 以一个 pointwise case 验证 observer、generated code 和 backend 隔离。
7. 再分批执行 pre_grad、joint_graph、post_grad、lowering/scheduler 类 pass。
8. 在出现真实缺口后更新 `change_control.md`，经审核再实施 Triton/AscendC/vendor 方案。

## 10. 当前文件导航

- `README.md`：入口与运行约束。
- `change_control.md`：环境、提案和源码修改冻结记录。
- `replacement_plan.md`：NPU 替代实现总体策略。
- `audit_passes.py`：静态清单生成器。
- `run_npu_probe.py`：动态探针草案。
- `report/pass_inventory.md`：静态 pass 索引。
- `report/pass_inventory.json`：静态 pass 机器可读数据。
- `report/npu_probe.json`：旧环境历史诊断，不能作为 baseline。

