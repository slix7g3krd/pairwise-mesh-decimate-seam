# pairwise-mesh-decimate-seam

## 领域

游戏、图形与多媒体

## 主题

网格减面接缝

## 说明

接缝锁定、切线重算与确定性坍缩。`graphics/mesh_decimate_seam.py` 提供一组纯函数
（无第三方依赖），覆盖题目列出的六个缺陷：

| 现象 | 修复入口 | 关键做法 |
| --- | --- | --- |
| UV 接缝处边坍缩把接缝撕裂 | `seam_lock` / `edge_cost` | 按位置焊接后检测 UV 接缝、法线接缝、边界与非流形边；接缝边代价为 `inf`，接缝端点被锁定，其它坍缩不得移动/合并它们 |
| 法线接缝与切线空间不一致，高光断裂 | `tangent_rebuild` | 按「焊接位置 + UV 图」而不是按顶点下标累加 UV 导数，UV 拆点两侧得到同一切线；同图内的法线接缝只做 Gram-Schmidt，方向一致 |
| LOD 切换顶点数突变，阴影边缘跳动 | `lod_morph` / `bounds_update` | 用「投影到高模表面」建立对应并做连续形变；简化/形变后重算 AABB 与 Ritter 包围球 |
| 异步简化完成顺序不定 | `async_version` | 校验请求令牌、源网格哈希、`current_version` 与请求版本，过期（`stale`）与乱序（`future`）结果一律丢弃 |
| 重复导入同一模型得到不同拓扑 | `collapse_order` / `import_dedupe` / `determinism` | 顺序键 = 量化代价 → 两端位置几何键 → 焊接组号，全部与输入排列无关；输出顶点/三角形做规范排序；`determinism` 用固定种子重排输入自检 |

辅助入口：`validation`（结构与拓扑检查）、`audit`（汇总审计）、`fallback`
（失败时返回未减面网格）、`uv_guard`（UV 翻转/退化/跨周期保护）、`metrics`
（统计）、`normal_seam`（法线接缝检测与分组平滑）。

## 数据模型

宽松的 `dict`（也接受同名属性对象），所有字段除 `positions` 外均可选：

```python
{
    "positions":      [(x, y, z), ...],
    "triangles":      [(a, b, c), ...],      # 或 "indices" 扁平列表
    "uvs":            [(u, v), ...],         # 逐顶点
    "corner_uvs":     [(u, v), ...],         # 逐角，长度 3*T
    "normals":        [(x, y, z), ...],
    "corner_normals": [(x, y, z), ...],
    "seam_edges":     [(a, b), ...],         # 显式接缝
}
```

拓扑分析前会按位置（默认 `weld_eps=1e-6`）焊接重复顶点：UV 拆点产生的同位置
副本进入同一个「几何组」，因此接缝两侧共享同一条几何边，而各自的角 UV/法线
仍被保留。所有入口对非网格输入返回空结果而不是抛错，便于降级路径继续运行。

## 运行

```bash
python -m pytest -q
```

期望全部通过。
