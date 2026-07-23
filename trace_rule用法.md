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
```
| 字典名称 | 包含内容 | 映射目标 | 核心目的 |
| :--- | :--- | :--- | :--- |
| torch_non_c_binding_in_graph_functions_npu | NPU 的上层纯 Python API（如 `torch.npu.stream`, `empty_cache`） | TorchInGraphFunctionVariable | 阻止展开，直接作为黑盒节点保留在 FX Graph 中。 |
| torch_c_binding_in_graph_functions_npu | NPU 的底层 C++ 绑定 API（如 `_C._npu_emptyCache`） | TorchInGraphFunctionVariable | 防止 Dynamo 尝试追踪二进制 C++ 代码导致 Unsupported 崩溃，同样作为图内节点保留。 |
| skip_functions_npu | 具有全局副作用的 API（如 `torch.npu.set_device`） | SkipFunctionVariable | 触发图断裂（Graph Break），让这段代码回退给 Python 原生解释器立刻执行。 |
