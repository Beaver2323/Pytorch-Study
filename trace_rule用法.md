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
