* 包装器 torch/_dynamo/eval_frame.py compile_wrapper()
``` python
@functools.wraps(fn)
def compile_wrapper(*args: Any, **kwargs: Any) -> Any:
```
* 相当于代理，包装器中使用set_eval_frame设置自定义的钩子（回调）拦截python帧
``` python
prior = set_eval_frame(None)  # 保存旧钩子
_maybe_set_eval_frame(_callback_from_stance(callback))  # 设置 Dynamo 的钩子
result = fn(*args, **kwargs)  # 执行时触发钩子
```
``` python
if DISABLE_JUSTKNOBS:
    _maybe_set_eval_frame = set_eval_frame  # 简单直通
else:
    def _maybe_set_eval_frame(callback):    # 带检查的包装
        if not justknobs_check("pytorch/compiler:enable_compiler_set_eval_frame"):
            # JustKnobs 关闭 → 不设置钩子
            return callback
        else:
            # JustKnobs 开启 → 正常设置钩子
            return set_eval_frame(callback)
```
* 经过各种跳过逻辑后到达真正执行的入口
``` bash
> /data/env_common/miniconda3/envs/zyf_2.14_inductor/lib/python3.13/site-packages/torch/_dynamo/convert_frame.py(2642)__call__()
-> result = self._torchdynamo_orig_backend(
```
* 然后进入class ConvertFrame ， 将python字节码转换成FX Graph的主入口
``` bash
> /data/env_common/miniconda3/envs/zyf_2.14_inductor/lib/python3.13/site-packages/torch/_dynamo/convert_frame.py(2322)__call__()
-> def __call__(
```
``` python
    try:
        result = self._inner_convert(
            frame, cache_entry, hooks, frame_state, skip=skip + 1
        )
        counters["frames"]["ok"] += 1  # 编译成功
        return result
    except Exception as e:
```
* _inner_convert进入ConvertFrameAssert，ConvertFrameAssert.__call__调用_compile编译
``` python
increment_frame()  # 增加帧计数器
code = frame.f_code  # 获取代码对象
.....
#检查是否有缓存
isolate_recompiles_id = get_eval_frame_isolate_recompiles_id()
cache_entries = _get_cache_entries_for_region(code, isolate_recompiles_id)
```

|跳过条件	|原因|
|  ----  | ----  |
|code in output_codes	|已经是 Dynamo 生成的代码|
TORCHDYNAMO_DEBUG_FUNCTION 过滤	|只调试特定函数
特定的 genexpr	|处理 transformers/diffusers 中的生成器表达式
__setattr__ 帧	|属性设置函数不编译
torch.optim 的 __init__	|优化器初始化不编译
exec() 生成的帧|	动态生成的代码不编译
namedtuple 构造函数	|特殊数据结构
生成器函数	|生成器无法直接编译
没有张量的帧	|没有张量操作，没必要编译
 
* 调用_compile编译
``` python
        try:
            compile_ctx = compile_context(CompileContext(compile_id))
            # When recompile_limit is set, temporarily override the global
            # config so the existing exceeds_recompile_limit check uses it.
            recompile_ctx = (
                config.patch(recompile_limit=self._recompile_limit)
                if self._recompile_limit is not None
                else contextlib.nullcontext()
            )
            with compile_ctx, recompile_ctx:
                result = _compile(
                    frame.f_code,
                    frame.f_globals,
                    frame.f_locals,
                    frame.f_builtins,
                    frame.closure,
                    self._torchdynamo_orig_backend,
                    self._one_graph,
                    self._export,
                    self._export_constraints,
                    hooks,
                    cache_entry,
                    cache_entries,
                    cache_entries_for_reasons,
                    cache_size,
                    frame,
                    frame_state=frame_state,
                    compile_id=compile_id,
                    skip=skip + 1,
                    package=self._package,
                    convert_frame_box=self._box,
                )
        finally:
            # Restore the previous initial_global_state for nested compilation handling
            initial_global_state = prev_initial_global_state
```
* 进入_compile前
``` bash
(Pdb) p frame.f_locals
{'config': <__main__.NpuModelConfig object at 0xfffddc6f3e00>, 'x': tensor([[-1.0217,  1.0408],
        [ 0.3088, -1.0210]], device='npu:0')}
(Pdb) p frame.f_globals.keys()
dict_keys(['__name__', '__doc__', '__package__', '__loader__', '__spec__', '__annotations__', '__builtins__', '__file__', '__cached__', 'torch', 'torch_npu', 'pdb', 'NpuModelConfig', 'mock_npu_c_api', 'train_step', 'config_obj', 'input_tensor'])
```
_compile (总入口)<br>
    ├── compile_inner (编译执行器)<br>
    │   └── _compile_inner (核心编译器)<br>
    │       ├── compile_frame (帧编译)<br>
    │       │   └── InstructionTranslator (字节码解析)<br>
    │       ├── build_guards (守卫构建)<br>
    │       └── wrap_guarded_code (结果包装)<br>
    ├── 异常处理<br>
    └── 指标收集<br>
* 进入compile_inner
``` bash
> /data/env_common/miniconda3/envs/zyf_2.14_inductor/lib/python3.13/site-packages/torch/_dynamo/convert_frame.py(2116)_compile()
-> guarded_code, tracer_output = compile_inner(code, one_graph, hooks)
```
_compile_inner<br>
    ├── 1. 记录原始字节码 (log_bytecode)<br>
    ├── 2. 执行帧编译 (compile_frame) ← 最核心！<br>
    ├── 3. 获取编译结果 (dynamo_output)<br>
    ├── 4. 记录修改后的字节码 (log_bytecode)<br>
    ├── 5. 处理字节码钩子 (_bytecode_hooks)<br>
    ├── 6. 验证代码对象一致性 (arg/free/cell vars)<br>
    ├── 7. 构建守卫 (build_guards) ← 关键！<br>
    ├── 8. 包装成 GuardedCode<br>
    └── 9. 返回结果<br>
* 将进入compile_frame
``` bash
> /data/env_common/miniconda3/envs/zyf_2.14_inductor/lib/python3.13/site-packages/torch/_dynamo/convert_frame.py(1761)_compile_inner()
-> dynamo_output = compile_frame(
```
compile_frame<br>
    ├── 定义 transform 函数 (字节码转换器)<br>
    ├── 循环尝试编译 (支持重试)<br>
    │   ├── transform_code_object 调用<br>
    │   │   └── transform 回调<br>
    │   │       └── trace_frame (真正的字节码追踪)<br>
    │   ├── 成功 → 返回 DynamoOutput<br>
    │   └── 失败 → RestartAnalysis → 重试<br>
    └── SkipFrame 异常 → 向上传递<br>
``` bash
> /data/env_common/miniconda3/envs/zyf_2.14_inductor/lib/python3.13/site-packages/torch/_dynamo/convert_frame.py(1598)compile_frame()
-> bytecode, tracer_output = transform_code_object(code, transform)
```
``` bash
> /data/env_common/miniconda3/envs/zyf_2.14_inductor/lib/python3.13/site-packages/torch/_dynamo/bytecode_transformation.py(1822)transform_code_object()
-> tracer_output = transformations(instructions, code_options)
```
``` bash
> /data/env_common/miniconda3/envs/zyf_2.14_inductor/lib/python3.13/site-packages/torch/_dynamo/convert_frame.py(1569)transform()
-> tracer_output = trace_frame(
```
* 之后经过几次跳转，至InstructionTranslatorBase
``` bash
> /data/env_common/miniconda3/envs/zyf_2.14_inductor/lib/python3.13/site-packages/torch/_dynamo/symbolic_convert.py(2026)run()
-> def run(self) -> None:
```
* 查看当前指令列表的情况
``` bash
(Pdb) for i, inst in enumerate(self.instructions[:10]):print(f"{i}: {inst.opname}")
0: RESUME
1: LOAD_GLOBAL
2: LOAD_FAST
3: CALL
4: STORE_FAST
5: LOAD_FAST
6: LOAD_FAST
7: LOAD_ATTR
8: BINARY_OP
9: STORE_FAST
```
| 索引 | 指令 | 对应代码 | 作用 |
| :--- | :--- | :--- | :--- |
| 0 | RESUME | `def train_step(...):` | Python 3.11+ 函数入口标记 |
| 1 | LOAD_GLOBAL | `mock_npu_c_api` | 加载全局函数 `mock_npu_c_api` |
| 2 | LOAD_FAST | `x` | 加载局部变量 `x`（输入张量） |
| 3 | CALL | `mock_npu_c_api(x)` | 调用函数，将结果压入栈 |
| 4 | STORE_FAST | `y = ...` | 将结果弹出栈并存储到局部变量 `y` |
| 5 | LOAD_FAST | `y` | 加载局部变量 `y`（`mock_npu_c_api` 的返回值） |
| 6 | LOAD_FAST | `config` |  加载局部变量 `config`（你的自定义对象） |
| 7 | LOAD_ATTR | `config.scale_factor` |  从 `config` 对象中加载属性 `scale_factor` |
| 8 | BINARY_OP | `y * config.scale_factor` | 执行乘法运算 |
| 9 | STORE_FAST | `out = ...` | 将结果存储到局部变量 `out` |
``` bash
> /data/env_common/miniconda3/envs/zyf_2.14_inductor/lib/python3.13/site-packages/torch/_dynamo/symbolic_convert.py(1639)step()
-> def step(self) -> bool:
```
step负责单步追踪字节码，构建FX Graph
* 示例代码
``` python
import torch
import torch_npu
import pdb


# ==========================================
# 触发点 1：自定义 Python 类
# 预期：Dynamo 会用 UserDefinedClassVariable 包装它
# ==========================================
class NpuModelConfig:
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor


# ==========================================
# 触发点 2：被显式禁用的函数（模拟 NPU 底层 C++ API）
# 预期：Dynamo 会用 SkipFunctionVariable 包装它
# ==========================================
@torch._dynamo.disable
def mock_npu_c_api(tensor):
    # 模拟一个无法被 Dynamo 解析的底层调用
    print("[Runtime] 执行被 Skip 的底层 NPU 函数...")
    return tensor + 1.0


# ==========================================
# 编译入口
# ==========================================
@torch.compile(backend="eager", fullgraph=False)
def train_step(config, x):
    # 触发 SkipFunctionVariable 追踪
    y = mock_npu_c_api(x)

    # 触发 UserDefinedClassVariable 属性读取
    out = y * config.scale_factor
    return out


if __name__ == "__main__":
    # 初始化 NPU 设备和数据
    torch.npu.set_device(0)
    config_obj = NpuModelConfig(scale_factor=2.0)
    input_tensor = torch.randn(2, 2).npu()

    print(">>> 准备进入编译阶段，启动 PDB...")

    # 在 Dynamo 触发前打断点
    pdb.set_trace()

    # 初次运行会触发 Dynamo Tracing
    result = train_step(config_obj, input_tensor)

    print(">>> 运行结果:", result)
```
* continue 到CALL指令，dispatch_table分发到具体的功能函数, b /data/env_common/miniconda3/envs/zyf_2.14_inductor/lib/python3.13/site-packages/torch/_dynamo/symbolic_convert.py:1685
``` bash
(Pdb) c
> /data/env_common/miniconda3/envs/zyf_2.14_inductor/lib/python3.13/site-packages/torch/_dynamo/symbolic_convert.py(1685)step()
-> try:
(Pdb) self.current_instruction
Instruction(opcode=53, opname='CALL', arg=1, argval=1, offset=14, starts_line=32, is_jump_target=False, positions=Positions(lineno=32, end_lineno=32, col_offset=8, end_col_offset=25), target=None, exn_tab_entry=None, argrepr=None)
(Pdb) n
> /data/env_common/miniconda3/envs/zyf_2.14_inductor/lib/python3.13/site-packages/torch/_dynamo/symbolic_convert.py(1686)step()
-> self.dispatch_table[inst.opcode](self, inst)
(Pdb) p self.dispatch_table[inst.opcode]
<function InstructionTranslatorBase.CALL at 0xfffdd9bb2160>
```
* 经过断图保护逻辑
``` bash
(Pdb) l
1087        ) -> Callable[[InstructionTranslatorBase, Instruction], None]:
1088            @functools.wraps(inner_fn)
1089            def wrapper(self: InstructionTranslatorBase, inst: Instruction) -> None:
1090                prev_push = self.current_instruction_push
1091                self.current_instruction_push = push
1092 ->             speculation = self.speculate()
1093                if speculation.failed(self):
1094                    # no need to restore current_instruction_push if speculation failed
1095                    if speculation.reason is None:
1096                        raise AssertionError(
1097                            "expected speculation.reason is not None to be true"
(Pdb) n
> /data/env_common/miniconda3/envs/zyf_2.14_inductor/lib/python3.13/site-packages/torch/_dynamo/symbolic_convert.py(1093)wrapper()
-> if speculation.failed(self):
(Pdb)
> /data/env_common/miniconda3/envs/zyf_2.14_inductor/lib/python3.13/site-packages/torch/_dynamo/symbolic_convert.py(1100)wrapper()
-> try:
(Pdb) n
> /data/env_common/miniconda3/envs/zyf_2.14_inductor/lib/python3.13/site-packages/torch/_dynamo/symbolic_convert.py(1101)wrapper()
-> return inner_fn(self, inst)
```
