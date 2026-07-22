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
# NPU 在 Dynamo 中被拦截的原因与 Patch 解决机制分析

## 1. 移除 Patch 后的拦截原因（结合 PDB 运行记录）

当移除 NPU 的 patch 并在 `fullgraph=True` 模式下运行时，代码在执行 `s = torch.npu.stream(torch.npu.Stream())` 时发生崩溃。结合之前的 PDB 堆栈追踪，拦截的根本原因可归结为以下过程：

**PDB 堆栈核心片段：**
> `.../polyfills/__init__.py(369) instantiate_user_defined_class_object`
> `.../npu/streams.py(25) __new__`
> `.../npu/utils.py(108) __init__`
> `.../npu/utils.py(158) _get_device_index`
> `torch._dynamo.exc.Unsupported: ... marked as skipped ... qualname: _get_device_index, skip reason: file matches MOD_SKIPLIST`

**拦截原因剖析：**
1. **设备硬编码导致兜底：** 原生 Dynamo（`_id_dispatch`）只认识 `torch.cuda.Stream`。由于移除了 patch，`torch.npu.Stream` 匹配失败，被兜底降级为普通的 `UserDefinedClassVariable`。
2. **强制内联展开（Polyfill）：** Dynamo 试图将这个“普通类”翻译进计算图，于是进入 `polyfills/instantiate_user_defined_class_object`，强行步入（step into）该类的 `__new__` 和 `__init__` 方法。
3. **触碰禁区（MOD_SKIPLIST）：** 在追踪 `__init__` 时，调用链进入了 `torch._utils._get_device_index`。原生 PyTorch 将 `torch._utils` 模块硬编码在 `MOD_SKIPLIST`（禁止追踪黑名单）中。Dynamo 发现越界，立刻抛出 `Unsupported` 异常导致编译崩溃。

---

## 2. NPU 源码如何解决该问题（结合 Patch 代码解析）

`torch_npu` 通过动态修改 Dynamo 的底层行为（Monkey Patch），在不修改 PyTorch 核心代码的前提下，让 NPU 组件获得了与 CUDA 相同的“图内特权”。

### 2.1 扩充 `_in_graph_classes` 白名单（阻断 `__init__` 追踪）

**对应源码：**
```python
def patch_user_defined_class_variable():
    original_method = UserDefinedClassVariable._in_graph_classes

    @staticmethod
    @functools.lru_cache(None)
    def patched_in_graph_classes():
        result = original_method()
        # 核心修复：强行将 NPU 类加入免检白名单
        result.add(torch.npu.Event)
        result.add(torch.npu.Stream)
        return result

    UserDefinedClassVariable._in_graph_classes = patched_in_graph_classes
