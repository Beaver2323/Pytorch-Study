### `torch_npu` 劫持 Dynamo 的底层动因

**核心原因：应对 Dynamo 的设备硬编码与兜底降级**
PyTorch Dynamo 的变量分发器 (`VariableBuilder._wrap`) 高度硬编码了 CUDA/CPU 逻辑。当遇到 `torch_npu` 的特有对象（如 `npu.Stream`、`npu.amp.autocast`）时，由于无法匹配任何原生的 `elif` 分支，这些对象会“穿透”分发树，跌落进纯 Python 类的兜底容器 `UserDefinedClassVariable`。如果不加干预，Dynamo 会对它们进行普通的方法内省，导致图内语义丢失（如无法正确展开混合精度上下文）或引发 Graph Break。

为了以非侵入式的第三方插件形态接入（避免直接修改上游核心库），`torch_npu` 选择在兜底容器的实例化入口（`__new__` 方法）注入 Monkey Patch，强行拦截 NPU 对象并将其“偷梁换柱”为正确的 `VariableTracker`。

**代码对照解析：**

```python
# ==========================================
# 1. 原生 PyTorch 侧：硬编码匹配与兜底穿透
# 位置：torch/_dynamo/variables/builder.py
# ==========================================
def _id_dispatch(self, value):
    # ... 上千行的 if-elif 硬编码 ...
    elif isinstance(value, torch.cuda.StreamContext):
        return StreamContextVariable.create(...) # 认识 CUDA
        
    # [穿透]：npu.StreamContext 等 NPU 对象匹配全部失败，掉入最底层
    # 返回兜底容器，将其视为毫无特殊语义的纯 Python 类
    return UserDefinedClassVariable(value, ...) 
```
``` python
# ==========================================
# 2. torch_npu 侧：在兜底容器的入口实施拦截
# 位置：torch_npu/utils/_dynamo.py
# ==========================================
def patch_user_defined_class_variable():
    from torch._dynamo.variables.user_defined import UserDefinedClassVariable
    
    # 截获原生类原本隐式继承的 __new__ 方法
    UserDefinedClassVariable.__new__raw = UserDefinedClassVariable.__new__
    
    def UserDefinedClassVariable__new__(cls, value, **kwargs):
        # 关卡拦截：发现是被 Dynamo 误判为普通类的 NPU 特殊组件
        if value in [torch_npu.npu.amp.autocast, ...]:
            # 狸猫换太子：强行返回专用的上下文管理器 Tracker，恢复图内语义
            return NPUTorchCtxManagerClassVariable(value, **kwargs)
            
        # 真正的普通纯 Python 类（放行）
        return cls.__new__raw(cls)
        
    # 动态覆盖，完成 Monkey Patch
    UserDefinedClassVariable.__new__ = UserDefinedClassVariable__new__
```

