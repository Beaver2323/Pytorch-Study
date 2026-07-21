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
然后进入class ConvertFrame
``` bash
> /data/env_common/miniconda3/envs/zyf_2.14_inductor/lib/python3.13/site-packages/torch/_dynamo/convert_frame.py(2322)__call__()
-> def __call__(
```
