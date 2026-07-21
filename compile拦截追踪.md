包装器 torch/_dynamo/eval_frame.py compile_wrapper()
``` python
@functools.wraps(fn)
def compile_wrapper(*args: Any, **kwargs: Any) -> Any:
```
相当于代理，包装器中使用set_eval_frame设置自定义的钩子（回调）拦截python帧
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
经过各种跳过逻辑后到达真正执行的入口
``` bash
> /data/env_common/miniconda3/envs/zyf_2.14_inductor/lib/python3.13/site-packages/torch/_dynamo/convert_frame.py(2642)__call__()
-> result = self._torchdynamo_orig_backend(
```
然后进入class ConvertFrame ， 将python字节码转换成FX Graph的主入口
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
_inner_convert进入ConvertFrameAssert
ConvertFrameAssert.__call__调用_compile编译
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
调用_compile编译
```
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
