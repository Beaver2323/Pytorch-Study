```
Tracing rules and policies for TorchDynamo compilation decisions.

This module defines the rules that govern what code TorchDynamo should trace and compile
versus what should be executed eagerly. It contains functions and classes that determine:

- Which modules, functions, and objects should be skipped during tracing
- Which parts of the code should cause graph breaks
- How to handle different Python libraries and third-party packages
- Rules for determining when to inline functions vs calling them eagerly

Key components:
- Skip rules: Functions that return True if an object should be skipped during tracing
- Inlining rules: Policies for when to inline function calls during compilation
- Library-specific handling: Special cases for popular Python packages
- Performance heuristics: Rules that balance compilation overhead vs runtime benefits

These rules are critical for TorchDynamo's ability to automatically determine
compilation boundaries and optimize PyTorch programs effectively.

A note on skip/inline rules:

Dynamo consults this file to determine whether function should be inlined or skipped.

A skip applies at the frame boundary, meaning dynamo either triggers a graph break
at the beginning of the frame or attempts to trace/inline the whole frame. When skipping
a frame, recursively called frames are still traced by dynamo unless also skipped.

Skipfiles (skipped at the file level instead of function level) still apply on a
frame-by-frame boundary as dynamo traces, but apply to all functions in that file.

@skip is a helper decorator that can be applied to your function to cause it to be
included here.

Dynamo skip/inline rules & priorities are defined as follows:
* Inline is the default behavior and will be used unless explicitly skipped.
* Dynamo has two SKIPLIST: BUILTIN_SKIPLIST and THIRDPARTY_SKIPLIST.
    * BUILTIN_SKIPLIST contains builtin python modules, such as abc, collections, etc.
    * THIRDPARTY_SKIPLIST contains common third party libraries, such as numpy, pandas, etc.
* Functions in these two SKIPLISTs are always skipped, except:
    * They have explicitly defined rule in `manual_torch_name_rule_map`;
    * The corresponding python module has been put into MOD_INLINELIST.
* PyTorch(torch) is in the BUILTIN_SKIPLIST by default, but there are many cases
    where we want inline the functions under torch namespace.
    We should specify inline for the functions in `manual_torch_name_rule_map` or
    put the corresponding python module into MOD_INLINELIST to make dynamo inline them.
* If you call functions under skipped modules/files, Dynamo will wrap these functions
    as SkipFunctionVariable. There are a few functions(e.g, collections.OrderedDict) that
    we have special handling at SkipFunctionVariable.call_function.

Overall: *_INLINELIST has precedence over *_SKIPLIST has precedence over DEFAULT (inline)

To figure out what the behavior is, check the following list in order:
* `manual_torch_name_rule_map` (Inline if YES)
* MOD_INLINELIST (Inline if YES)
* BUILTIN_SKIPLIST & THIRDPARTY_SKIPLIST (Skip if YES)
* MOD_SKIPLIST (Skip if YES)
* Inline by default

In general, if you want to force inline a function or module, please consider adding
the function's python module to MOD_INLINELIST first.
Use the `manual_torch_name_rule_map` only when there are other functions under the same module that
you don't want to inline them.
"""

"""
Map of function objects to their tracing rules (Dynamo variables).
* TorchInGraphFunctionVariable: The functions should be put into the FX graph or can be constant folded. E.g.,
  - torch.add: should be put into the FX graph.
  - torch.is_floating_point: constant folded.
* SkipFunctionVariable: The objects should be skipped from tracing.
* UserFunctionVariable: The functions should be inlined.

For developers: If you add/remove a torch level API, it may trigger failures from
test/dynamo/test_trace_rules.py:test_torch_name_rule_map_updated. To fix the failures:
If you are adding a new torch level API or Dynamo implementation:
* Add the name with the corresponding tracing rule to this map
  if you are adding a new in graph function or Dynamo implementation for an existing function.
* Remove the object name from test/dynamo/test_trace_rules.ignored_c_binding_in_graph_function_names if it's there.

If you are removing an existing torch level API:
* Remove the entry represented the API from this map or test/dynamo/test_trace_rules.ignored_c_binding_in_graph_function_names
  depends on where it is.

```
| 字典名称 | 包含内容 | 映射目标 | 核心目的 |
| :--- | :--- | :--- | :--- |
| torch_non_c_binding_in_graph_functions_npu | NPU 的上层纯 Python API（如 `torch.npu.stream`, `empty_cache`） | TorchInGraphFunctionVariable | 阻止展开，直接作为黑盒节点保留在 FX Graph 中。 |
| torch_c_binding_in_graph_functions_npu | NPU 的底层 C++ 绑定 API（如 `_C._npu_emptyCache`） | TorchInGraphFunctionVariable | 防止 Dynamo 尝试追踪二进制 C++ 代码导致 Unsupported 崩溃，同样作为图内节点保留。 |
| skip_functions_npu | 具有全局副作用的 API（如 `torch.npu.set_device`） | SkipFunctionVariable | 触发图断裂（Graph Break），让这段代码回退给 Python 原生解释器立刻执行。 |
``` python
# torch_npu/_init/registry/dynamo.py
def register_dynamo_backends():
    from torch_npu.dynamo import _register_backends

    _register_backends()


def register_dynamo_trace_rules():
    """
    # Support stream into Dynamo charts. Enable Dynamo to recognize NPU
    stream/device/memory/random APIs and related torch_npu._C bindings during graph capture.
    """
    from torch_npu.dynamo.trace_rule import _patch_npu_trace_rules

    _patch_npu_trace_rules()
```

# PyTorch Dynamo `trace_rule` 路由机制与 NPU 适配深度解析

在 PyTorch 2.x 的 `torch.compile` (Dynamo) 编译体系中，`trace_rule` 是决定代码能否成功转换为计算图（FX Graph）的“最高宪法”。本文档提炼了 Dynamo 底层路由优先级规则、NPU 适配的痛点，以及如何通过 PDB 源码级调试揭示图断裂（Graph Break）的根因。

---

## 一、 Dynamo 路由机制的 5 级优先级判定

根据 PyTorch 官方设计指南，当 Dynamo 遇到一个函数时，会按照以下优先级顺序决定是“内联步入（Inline）”、“跳过（Skip）”还是“转为图节点（InGraph）”：

1. **`manual_torch_name_rule_map` (Inline/InGraph if YES) —— 👑 第一优先级**
2. **`MOD_INLINELIST`** (模块级白名单：尝试内联展开)
3. **`BUILTIN_SKIPLIST` & `THIRDPARTY_SKIPLIST`** (默认黑名单：包含 `torch`, `numpy` 等，遇之即跳过)
4. **`MOD_SKIPLIST`** (文件级黑名单：跳过特定模块)
5. **Inline by default** (默认行为：内联步入追踪源码)

**⚠️ NPU 踩坑的根本原因（命名空间陷阱）：**
官方规定，整个 `torch.*` 命名空间默认处于第 3 优先级的 `BUILTIN_SKIPLIST` 中。NPU 的扩展 API（如 `torch.npu.stream`）会因此被 Dynamo 误判为需要跳过的黑盒。如果在 `fullgraph=True`（严禁图断裂）模式下，这会导致直接触发 `Unsupported` 崩溃。

---

## 二、 核心解法：三种 Variable 语义与 VIP 截胡

为了打破上述的默认黑名单限制，NPU 必须利用 **“第 1 优先级可以覆盖一切”** 的规则，将 API 注入全局的 `torch_name_rule_map`。注入时，需严格对齐以下三种 Variable 语义：

| 变量类型 | Dynamo 行为 | 适用 NPU 场景 |
| :--- | :--- | :--- |
| **`TorchInGraphFunctionVariable`** | **图内黑盒 / 常量折叠**。直接生成 FX Graph 节点，不追踪内部 Python 源码。 | NPU 的 C++ 底层算子（如 `_C._npu_emptyCache`）、张量操作、需直接下发给 Inductor 的流/事件控制。 |
| **`SkipFunctionVariable`** | **触发 Graph Break**。打断计算图，退回 Python 原生解释器立即执行。 | 带有全局副作用、绝对不能延迟执行的函数（如 `torch.npu.set_device`）。 |
| **`UserFunctionVariable`** | **内联展开 (Inline)**。强行步入该函数，逐行解析内部字节码。 | NPU 侧编写的无副作用纯 Python 辅助工具函数。 |

---

## 三、 Dynamo 查表拦截的底层全流程剖析

当我们执行 `s = torch.npu.stream(torch.npu.Stream())` 时，Dynamo 在解析字节码（`VariableBuilder` 加载阶段）会调用核心分发函数：`torch._dynamo.trace_rules._lookup_inner`。

该函数分为三个关键步骤（Step 1 到 Step 3）：

### 🟢 Step 1: VIP 快速通道 (补丁生效点)
```python
# torch/_dynamo/trace_rules.py -> _lookup_inner()
rule = get_torch_obj_rule_map().get(obj, None)
if rule is not None:
    return rule  # 完美截胡！直接返回 TorchInGraphFunctionVariable
