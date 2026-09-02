# Inductor benchmark 解耦修改讲解

本文只回答三个问题：原来有什么问题、现在准备怎么做、每段代码修改是什么意思。

当前范围只有三个改动：

| 编号 | 仓库 | 作用 |
| --- | --- | --- |
| P1 | PyTorch | 为 Graph benchmark 提供设备无关的注册和分发能力 |
| N1 | torch_npu | 把普通 NPU benchmark 接入 PyTorch 已有的注册机制 |
| N3 | torch_npu | 实现并注册 NPU Graph benchmark |

## 1. 解决的问题是什么，原来 NPU 是怎么做的

### 1.1 普通 benchmark

benchmark 是 Inductor autotune 选择实现时使用的计时过程：同一个算子有多个候选实现时，
分别运行并测量耗时，最后选择最快的实现。

PyTorch 原本已经提供普通 benchmark 注册表：

```python
_BENCHMARK_DISPATCH: dict[str, Callable[..., Any]] = {}


def register_benchmarker(device_type, fn, *, override=False):
    ...
    _BENCHMARK_DISPATCH[device_type] = fn
```

设备可以用 `register_benchmarker("设备名", handler)` 注册自己的计时入口。但修改前
torch_npu 没有注册 NPU handler，而是在自己控制的三个位置直接调用
`benchmark_gpu(..., device_type="npu")`：

| torch_npu 原调用点 | 用途 |
| --- | --- |
| `select_algorithm.py::_NPUTritonBenchmarkRequestMixin.do_bench` | 测量候选 kernel |
| `runtime/triton_heuristics.py::NPUCachingAutotuner._bench_with_launch_args` | 测量 Triton launcher |
| `codegen/scheduling.py::NPUTritonScheduling.benchmark_fused_nodes` | 测量融合 kernel，并扣除克隆输入开销 |

核心调用如下：

```python
# select_algorithm.py：候选实现计时
class _NPUTritonBenchmarkRequestMixin:
    kernel_has_output_arg = True

    def do_bench(
        self,
        fn: Callable[[], None],
        *input_tensors: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> float:
        from torch._inductor.runtime.benchmarking import benchmarker

        device_idx_set = OrderedSet(
            tensor.device.index
            for tensor in (*input_tensors, out)
            if isinstance(tensor, torch.Tensor)
            and tensor.device.type == "npu"
            and tensor.device.index is not None
        )
        assert len(device_idx_set) <= 1, f"Can not mix devices {device_idx_set}"
        device_interface = get_interface_for_device("npu")
        device_idx = (
            next(iter(device_idx_set))
            if device_idx_set
            else device_interface.current_device()
        )
        with device_interface.device(device_idx):
            result = benchmarker.benchmark_gpu(fn, device_type="npu")
            device_interface.synchronize()

# runtime/triton_heuristics.py：Triton launcher 计时
from torch._inductor.runtime.benchmarking import TritonBenchmarker
benchmarker = TritonBenchmarker()
def _bench_with_launch_args(self, launcher, launch_args, reset_args, **kwargs):
    device_interface = self.get_device_interface()
    stream = device_interface.get_raw_stream(device_interface.current_device())

    def kernel_call():
        cloned_args, cloned_kwargs = self.clone_args(*launch_args, **kwargs)
        self.reset_to_zero_args(*reset_args, **kwargs)
        launcher(
            *cloned_args,
            **cloned_kwargs,
            stream=stream,
        )

    if (
        self.inductor_meta.get("profile_bandwidth_with_do_bench_using_profiling", False)
        and not npu_profiler_session_active()
    ):
        return do_bench_using_profiling_npu(kernel_call, rep=1)

    return benchmarker.benchmark_gpu(kernel_call, rep=1, device_type='npu')

# codegen/scheduling.py：融合 kernel 计时
ms = benchmarker.benchmark_gpu(
    lambda: call(wrapped_jit_function.clone_args(*args)[0]),
    device_type="npu",
)
```

这三种旧用法本身能够正常计时，但共同特点是：**调用点都在 torch_npu 仓内，并且绕过
`Benchmarker.benchmark()` 的设备注册表，直接进入 `benchmark_gpu()`。**

当 PyTorch 通用 autotune 代码调用 `Benchmarker.benchmark()` 时，torch_npu 无法修改
这个上游调用点，也没有已注册的 NPU handler 接管它。因此，原有直接调用不能覆盖
PyTorch 通用路径。这正是 N1 要补齐的范围。

N1 要解决的就是这个问题：不新增 PyTorch 接口，只在 torch_npu 中注册 NPU handler。

### 1.2 Graph benchmark

Graph benchmark 不是普通地重复执行函数，而是先把函数捕获成设备图，再反复 replay
这张图进行计时。

修改前，PyTorch 的四个通用 autotune 调用点直接写死：

```python
benchmarker.benchmark_gpu_with_cuda_graph(...)
```

也就是说，只要开启 Graph benchmark，通用代码就直接进入 CUDA Graph。这里没有按
Tensor 的实际设备进行分发，也没有让其他设备注册 Graph 捕获、回放和计时实现的入口。

torch_npu 原本已经有完整的 NPUGraph 使用方式，但它服务的是“把编译后的模型捕获成图
并在模型运行时 replay”，不是“把 autotune 候选捕获成图后测量耗时”。

#### 原用法一：通过 `npugraphs` backend 捕获编译模型

用户可以这样启用原有能力：

```python
compiled_model = torch.compile(model, backend="npugraphs")
result = compiled_model(*inputs)
```

对应入口是
`torch_npu/dynamo/__init__.py::_NpugraphsBackendEntryPoint.__call__`：

```python
class _NpugraphsBackendEntryPoint:
    compiler_name = "npugraphs"

    def __call__(self, gm, example_inputs, **kwargs):
        from torch_npu.utils._dynamo import _lazy_dynamo_setup, _lazy_inductor_setup

        _lazy_dynamo_setup()
        _lazy_inductor_setup()
        from torch_npu.utils._graph_tree import NpugraphsBackend

        return NpugraphsBackend()(gm, example_inputs)
```

随后
`torch_npu/utils/_graph_tree.py::npugraphs` 使用 `aot_autograd` 分别包装前向、反向和推理
编译器，再把模型交给 `npugraphify_impl`：

```python
aot_npugraphs = aot_autograd(
    fw_compiler=forward_npugraphs,
    bw_compiler=backward_npugraphs,
    inference_compiler=functools.partial(
        forward_npugraphs,
        is_inference=True,
    ),
    keep_inference_input_mutations=(
        torch._dynamo.config.cudagraph_backend_keep_input_mutation
    ),
)
return aot_npugraphs(dynamo_model, dynamo_inputs)
```

#### 原用法二：接管 Inductor 的整图捕获函数

NPU Inductor 懒初始化时会执行
`torch_npu/utils/_graph_tree.py::_apply_npugraph_tree_methods`：

```python
def _apply_npugraph_tree_methods():
    if "npugraphs" not in _COMPILER_FNS:
        from torch_npu.dynamo import _npugraphs_backend_entrypoint

        register_backend(
            name="npugraphs",
            compiler_fn=_npugraphs_backend_entrypoint,
        )

    torch._inductor.compile_fx.cudagraphify = npugraphify
```

这里把 Inductor 的 `cudagraphify` 替换为 NPU 的 `npugraphify`，作用对象仍是编译完成的
模型或图分区。

#### 原来的捕获与 replay

`torch_npu/utils/_graph_tree.py::npugraphify_impl` 的核心逻辑是：准备静态输入，在独立流
预热，捕获模型，然后返回一个运行函数；每次模型运行时复制新输入并 replay：

```python
# warmup
torch.npu.synchronize()
stream = torch.npu.Stream()
stream.wait_stream(torch.npu.current_stream())
with torch.npu.stream(stream):
    model(list(static_inputs))
stream.synchronize()

# capture
graph = torch.npu.NPUGraph()
with torch.npu.graph(
    graph,
    stream=stream,
    capture_error_mode="thread_local",
):
    static_outputs = model(list(static_inputs))

# runtime replay
def run(new_inputs):
    for idx in copy_indices:
        index_expanded_dims_and_copy_(
            static_inputs[idx],
            new_inputs[idx],
            inps_expanded_dims[idx],
        )
    new_inputs.clear()
    graph.replay()
    return static_outputs
```

原有 Graph 调用链可以简化为：

```text
torch.compile(..., backend="npugraphs")
    -> _NpugraphsBackendEntryPoint
    -> NpugraphsBackend / aot_autograd
    -> npugraphify 或 NPUGraph tree
    -> 捕获编译模型
    -> 模型实际运行时 graph.replay()
```

而本任务要接入的是：

```text
Inductor autotune 的某个候选实现
    -> Graph benchmark
    -> 捕获该候选
    -> 重复 replay 并返回候选耗时
```

两条路径的捕获对象和返回结果不同：旧路径捕获编译模型并返回可执行函数；Graph
benchmark 捕获候选 callable 并返回耗时。因此，不能直接用旧 `npugraphs` backend
代替 Graph benchmark handler。

因此问题不是“torch_npu 没有图功能”，而是“PyTorch 的通用 Graph benchmark 入口只会
调用 CUDA Graph，NPU 已有的图能力没有接入位置”。

## 2. 现在打算怎么做

整体方案如下：

```text
普通 benchmark：
PyTorch Benchmarker.benchmark
    -> 已有普通 benchmark 注册表
    -> N1 注册的 NPU handler
    -> benchmark_gpu(..., device_type="npu")

Graph benchmark：
PyTorch 四个通用调用点
    -> 从输入/输出 Tensor 推断实际 device
    -> 查询该设备是否注册 Graph benchmarker
       -> 已注册：按 device.type 分发
       -> 未注册：保持原来的普通 benchmark 路径
    -> N3 注册的 NPU handler
    -> NPUGraph 捕获并 replay
    -> benchmark_gpu(replay, device_type="npu")
```

三个改动的职责边界是：

- **P1（PyTorch）**：只提供设备无关的 Graph benchmark 注册、查询和分发接口；不出现
  NPU 或厂商逻辑。CUDA 通过同一个接口注册原有实现。
- **N1（torch_npu）**：使用 PyTorch 已有的普通 benchmark 注册槽位，为普通计时补上
  `device_type="npu"`。
- **N3（torch_npu）**：负责 NPU 特有的预热、流同步、NPUGraph 捕获、回放和计时。

P1 不改变未注册设备的行为：只有 `has_graph_benchmarker(device)` 返回 `True` 才进入
Graph benchmark，否则继续执行该调用点原来的普通 benchmark。CUDA 已默认注册，所以
CUDA 继续使用原 CUDA Graph 实现；XPU 没有被修改，也不会被强制送进 CUDA Graph。

N1 不依赖 P1。N3 使用 P1 新增的 Graph 注册接口，因此依赖 P1。

## 3. 代码 diff 解释

下面只展示生产代码 diff；测试代码不放进讲解正文。

### 3.1 P1：PyTorch 新增设备无关的 Graph benchmark 分发

#### 3.1.1 新增注册表、注册接口和能力查询

文件：`torch/_inductor/runtime/benchmarking.py`

```diff
@@
 _BENCHMARK_DISPATCH: dict[str, Callable[..., Any]] = {}
+_GRAPH_BENCHMARK_DISPATCH: dict[str, Callable[..., Any]] = {}
 
 
+def register_graph_benchmarker(
+    device_type: str,
+    fn: Callable[..., Any],
+    *,
+    override: bool = False,
+) -> None:
+    """Register a graph benchmark handler for one torch.device.type."""
+    if not isinstance(device_type, str) or not device_type:
+        raise ValueError(
+            "device_type must be a non-empty string matching torch.device.type"
+        )
+    if not callable(fn):
+        raise TypeError("fn must be callable")
+    if not override and device_type in _GRAPH_BENCHMARK_DISPATCH:
+        raise ValueError(
+            f"Graph benchmarker for device_type '{device_type}' already registered"
+        )
+    _GRAPH_BENCHMARK_DISPATCH[device_type] = fn
+
+
+def has_graph_benchmarker(device: str | torch.device) -> bool:
+    resolved_device = torch.device(device) if isinstance(device, str) else device
+    if not isinstance(resolved_device, torch.device):
+        raise TypeError(f"Expected torch.device, got {type(resolved_device)}")
+    return resolved_device.type in _GRAPH_BENCHMARK_DISPATCH
```

解释：

- `_GRAPH_BENCHMARK_DISPATCH` 单独保存各设备的 Graph benchmark handler；它与普通
  benchmark 注册表分开，因为 Graph 捕获策略由设备决定。
- `register_graph_benchmarker()` 供设备后端注册实现。
- `has_graph_benchmarker()` 供通用调用点先查询能力。未注册设备可以安全回退到普通计时。

#### 3.1.2 新增统一 Graph benchmark 调用入口

文件：`torch/_inductor/runtime/benchmarking.py::Benchmarker`

```diff
@@
 class Benchmarker:
+    def benchmark_gpu_with_graph(
+        self: Self,
+        _callable: Callable[[], Any],
+        *,
+        device: str | torch.device,
+        grad_to_none: list[torch.Tensor] | None = None,
+        **kwargs: Any,
+    ) -> float:
+        resolved_device = torch.device(device) if isinstance(device, str) else device
+        if not isinstance(resolved_device, torch.device):
+            raise TypeError(f"Expected torch.device, got {type(resolved_device)}")
+
+        graph_benchmark_fn = _GRAPH_BENCHMARK_DISPATCH.get(resolved_device.type)
+        if graph_benchmark_fn is None:
+            raise RuntimeError(
+                "No graph benchmarker registered for "
+                f"device_type '{resolved_device.type}'"
+            )
+        return graph_benchmark_fn(
+            self,
+            _callable,
+            grad_to_none=grad_to_none,
+            device_type=resolved_device.type,
+            **kwargs,
+        )
```

解释：入口接收实际 `device`，取出 `device.type` 查注册表，再调用对应 handler。PyTorch
只负责分发，不负责实现某种设备图的捕获方式。

#### 3.1.3 把原 CUDA Graph 实现注册为默认 CUDA handler

文件：`torch/_inductor/runtime/benchmarking.py`

```diff
@@
+def _default_cuda_graph_bench(self, f, *, grad_to_none=None, **kw):
+    kw.pop("device_type", None)
+    return self.benchmark_gpu_with_cuda_graph(
+        f,
+        grad_to_none=grad_to_none,
+        **kw,
+    )
+
+
+register_graph_benchmarker("cuda", _default_cuda_graph_bench, override=True)
```

解释：没有重写 CUDA Graph 的实现，只增加一个适配函数，把新统一入口转回原来的
`benchmark_gpu_with_cuda_graph()`。因此 CUDA 的捕获和计时逻辑保持不变。

#### 3.1.4 四个通用调用点改为“先判断能力，再分发”

文件：`torch/_inductor/autotune_process.py`

```diff
@@
-from .runtime.benchmarking import benchmarker
+from .runtime.benchmarking import benchmarker, has_graph_benchmarker
@@ class BenchmarkRequest:
             if self.benchmark_with_cudagraphs:
-                res = benchmarker.benchmark_gpu_with_cuda_graph(fn)
+                device = benchmarker.infer_device(*input_tensors, out)
+                if has_graph_benchmarker(device):
+                    res = benchmarker.benchmark_gpu_with_graph(fn, device=device)
+                else:
+                    res = self.do_bench(fn, *input_tensors, out)
             else:
                 res = self.do_bench(fn, *input_tensors, out)
@@ class ExternKernelBenchmarkRequest:
             if self.benchmark_with_cudagraphs:
-                return benchmarker.benchmark_gpu_with_cuda_graph(
-                    lambda: algo(*input_tensors)
-                )
+                device = benchmarker.infer_device(*input_tensors, out_new)
+                if has_graph_benchmarker(device):
+                    return benchmarker.benchmark_gpu_with_graph(
+                        lambda: algo(*input_tensors), device=device
+                    )
             if config.profile_bandwidth_with_do_bench_using_profiling:
```

文件：`torch/_inductor/codegen/subgraph.py`

```diff
@@
-from torch._inductor.runtime.benchmarking import benchmarker
+from torch._inductor.runtime.benchmarking import benchmarker, has_graph_benchmarker
@@ class SubgraphChoiceCaller:
         if self._benchmark_with_cudagraphs:
-            return benchmarker.benchmark_gpu_with_cuda_graph(fn)
+            device = benchmarker.infer_device(*sym_inputs, *args, out)
+            if has_graph_benchmarker(device):
+                return benchmarker.benchmark_gpu_with_graph(fn, device=device)
```

文件：`torch/_inductor/ir.py`

```diff
@@
-from .runtime.benchmarking import benchmarker
+from .runtime.benchmarking import benchmarker, has_graph_benchmarker
@@ class ChoiceCaller:
         if self._benchmark_with_cudagraphs:
-            return benchmarker.benchmark_gpu_with_cuda_graph(lambda: algo(*args))
+            device = benchmarker.infer_device(*args, out)
+            if has_graph_benchmarker(device):
+                return benchmarker.benchmark_gpu_with_graph(
+                    lambda: algo(*args), device=device
+                )
```

解释：四处改法完全相同。

1. 从参与 benchmark 的 Tensor 推断真实设备；
2. 只有设备注册了 Graph handler 才进入统一 Graph 入口；
3. 未注册时自然继续执行后面的普通 benchmark 逻辑。

没有修改 NVIDIA 专属的 `nv_universal_gemm`，因为它本来就是 CUDA 专用代码，不属于
需要设备解耦的通用调用点。

### 3.2 N1：torch_npu 注册普通 benchmark handler

文件：`torch_npu/utils/_dynamo.py`

```diff
@@
+def _register_npu_benchmarker():
+    from torch._inductor.runtime.benchmarking import register_benchmarker
+
+    def npu_benchmark(self, f, *, warmup, rep, **kw):
+        kw.setdefault("device_type", "npu")
+        return self.benchmark_gpu(f, warmup=warmup, rep=rep, **kw)
+
+    register_benchmarker("npu", npu_benchmark, override=True)
+
+
 @run_once
 def _inject_inductor_npu_backend_config():
@@
 def _lazy_inductor_setup():
     _inject_inductor_npu_backend_config()
+    _run_dynamo_setup_step(
+        "npu_benchmarker",
+        _register_npu_benchmarker,
+    )
```

解释：

- `npu_benchmark()` 仍复用 PyTorch 的 `benchmark_gpu()`，没有另造一套普通计时实现。
- `kw.setdefault(...)` 只在调用方没有指定设备时补上 `npu`，不会覆盖已有参数。
- 注册放在 `_lazy_inductor_setup()`，只有使用 NPU Inductor 时才安装，并由现有初始化机制
  保证只执行一次。

### 3.3 N3：torch_npu 实现并注册 NPU Graph benchmark

新增文件：`torch_npu/_inductor/runtime/npu_graph_benchmarking.py`

```diff
+from collections.abc import Callable
+from typing import Any
+
+import torch
+from torch._inductor.runtime.benchmarking import gpu_benchmark_lock
+
+
+@gpu_benchmark_lock
+def npu_graph_benchmark(
+    self: Any,
+    _callable: Callable[[], Any],
+    *,
+    grad_to_none: list[torch.Tensor] | None = None,
+    device_type: str | torch.device | None = None,
+    **kwargs: Any,
+) -> float:
+    resolved_device_type = (
+        "npu"
+        if device_type is None
+        else torch.device(device_type).type
+    )
+    if resolved_device_type != "npu":
+        raise ValueError(
+            "NPU graph benchmarker received "
+            f"device_type '{resolved_device_type}' instead of 'npu'"
+        )
+
+    device_module = torch.npu
+
+    device_module.synchronize()
+    _callable()
+    device_module.synchronize()
+
+    stream = device_module.Stream()
+    stream.wait_stream(device_module.current_stream())
+    with device_module.stream(stream):
+        if grad_to_none is not None:
+            for tensor in grad_to_none:
+                tensor.grad = None
+        _callable()
+    stream.synchronize()
+
+    graph = device_module.NPUGraph()
+    with device_module.graph(
+        graph,
+        stream=stream,
+        capture_error_mode="thread_local",
+    ):
+        if grad_to_none is not None:
+            for tensor in grad_to_none:
+                tensor.grad = None
+        _callable()
+
+    device_module.current_stream().wait_stream(stream)
+    device_module.synchronize()
+
+    return self.benchmark_gpu(
+        graph.replay,
+        device_type="npu",
+        **kwargs,
+    )
```

这段实现按顺序完成：

1. 校验分发进来的设备确实是 NPU；
2. 在当前流执行一次，完成基础预热；
3. 创建独立流并再预热一次，满足图捕获前的流和内存准备要求；
4. 使用 `torch.npu.NPUGraph` 和 `torch.npu.graph(...)` 捕获 `_callable`；
5. 恢复当前流与捕获流的同步关系；
6. 把 `graph.replay` 交给已有 `benchmark_gpu()` 重复执行并计时。

`grad_to_none` 在预热和捕获前都被清理，避免梯度状态进入图后与实际重复执行语义不一致。
`@gpu_benchmark_lock` 用来避免多个 Graph benchmark 同时操作设备流和捕获状态。

注册代码仍放在 `torch_npu/utils/_dynamo.py`：

```diff
@@
+def _register_npu_graph_benchmarker():
+    from torch._inductor.runtime.benchmarking import register_graph_benchmarker
+    from torch_npu._inductor.runtime.npu_graph_benchmarking import (
+        npu_graph_benchmark,
+    )
+
+    register_graph_benchmarker(
+        "npu",
+        npu_graph_benchmark,
+        override=True,
+    )
+
+
 @run_once
 def _inject_inductor_npu_backend_config():
@@
 def _lazy_inductor_setup():
     _inject_inductor_npu_backend_config()
+    _run_dynamo_setup_step(
+        "npu_graph_benchmarker",
+        _register_npu_graph_benchmarker,
+    )
```

解释：P1 只提供注册槽位；N3 在 torch_npu 内注册设备名 `npu` 及其具体 handler。
PyTorch 上游不需要知道 NPUGraph，也不会出现 NPU 或厂商判断。

N1 和 N3 都修改 `_lazy_inductor_setup()` 附近。两个 MR 最终合并时必须同时保留普通
benchmarker 和 Graph benchmarker 两个 setup step，不能让后合入的修改覆盖先合入内容。

一句话总结：**PyTorch 通用代码只按设备能力分发，torch_npu 自己实现并注册 NPU
计时行为。**
