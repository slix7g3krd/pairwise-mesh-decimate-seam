# pairwise-mesh-decimate-seam

## 领域

游戏、图形与多媒体

## 主题

网格减面接缝

## 说明

接缝锁定、切线重算与确定性坍缩。

初始状态下 `graphics/mesh_decimate_seam.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
