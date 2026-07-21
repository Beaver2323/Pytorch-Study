包装器 torch/_dynamo/eval_frame.py compile_wrapper()
``` python
@functools.wraps(fn)
def compile_wrapper(*args: Any, **kwargs: Any) -> Any:
```
相当于代理，包装器中使用钩子追踪python帧
``` python
prior = set_eval_frame(None)  # 保存旧钩子
_maybe_set_eval_frame(_callback_from_stance(callback))  # 设置 Dynamo 的钩子
result = fn(*args, **kwargs)  # 执行时触发钩子
```
