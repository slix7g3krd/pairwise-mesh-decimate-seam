"""接缝感知的网格减面参考实现（纯 Python，无第三方依赖）。

模块围绕「UV/法线接缝锁定 + 确定性边坍缩」提供一组可组合的纯函数：

* :func:`seam_lock`       —— 检测 UV 接缝、边界、非流形边与硬边，给出锁定边/顶点。
* :func:`edge_cost`       —— 二次误差（QEM）边权；锁定边返回 ``inf``。
* :func:`tangent_rebuild` —— 按法线/UV 接缝分组重建切线空间，接缝两侧互不污染。
* :func:`lod_morph`       —— 两套 LOD 之间按顶点对应做形变，避免切换时跳变。
* :func:`async_version`   —— 校验异步简化结果是否对应当前版本，丢弃过期/乱序结果。
* :func:`collapse_order`  —— 确定性边坍缩：代价 -> 几何键 -> 顶点下标。
* :func:`bounds_update`   —— 重算 AABB 与 Ritter 包围球，供阴影/剔除使用。
* :func:`validation`      —— 结构合法性与拓扑质量检查。
* :func:`audit`           —— 汇总以上检查的审计报告。
* :func:`fallback`        —— 失败时返回未减面的安全网格。
* :func:`uv_guard`        —— UV 翻转/拉伸/退化保护。
* :func:`metrics`         —— 网格统计信息。
* :func:`normal_seam`     —— 法线接缝检测与按组平滑。
* :func:`import_dedupe`   —— 与导入顺序无关的规范键，用于重复导入去重。
* :func:`determinism`     —— 用同一模型的不同顶点/三角形顺序验证减面确定性。

数据模型是宽松的 ``dict``（也接受带同名属性的对象）::

    {
        "positions":      [(x, y, z), ...],       # 顶点坐标
        "triangles":      [(a, b, c), ...],       # 或 "indices" 扁平列表
        "uvs":            [(u, v), ...],          # 可选，逐顶点 UV
        "corner_uvs":     [(u, v), ...],          # 可选，逐角 UV，长度 3*T
        "normals":        [(x, y, z), ...],       # 可选，逐顶点法线
        "corner_normals": [(x, y, z), ...],       # 可选，逐角法线，长度 3*T
        "seam_edges":     [(a, b), ...],          # 可选，显式接缝
    }

拓扑分析前会按位置把重复顶点（UV 拆点导致的同位置副本）焊接到同一组里，
因此接缝两侧共享同一条「几何边」，但各自的角 UV/角法线仍然保留。

所有入口对「不是网格」的输入（例如测试里的整数 ``1``）都返回空结果而不是抛错，
方便上层在降级路径上继续运行。
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

INF = float("inf")

_EPS = 1e-12
_DEFAULT_WELD_EPS = 1e-6
_DEFAULT_UV_EPS = 1e-6
_DEFAULT_NORMAL_ANGLE = 45.0
_DEFAULT_HARD_ANGLE = 60.0
_COST_SCALE = 1_000_000_000
_HASH_PREFIX = "mesh-decimate-seam/v1"

# 对称 3x3 + 一次项 + 常数项的 QEM 表示：
# (a00, a01, a02, a11, a12, a22, b0, b1, b2, c)
_QZERO = (0.0,) * 10


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _is_num(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_seq(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_vec(value: Any, n: int) -> bool:
    return _is_seq(value) and len(value) >= n and all(_is_num(x) for x in value[:n])


def _vec(value: Any) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def _uv(value: Any) -> tuple[float, float]:
    return (float(value[0]), float(value[1]))


def _add(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(x + y for x, y in zip(a, b))


def _sub(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(x - y for x, y in zip(a, b))


def _scale(a: tuple[float, ...], s: float) -> tuple[float, ...]:
    return tuple(x * s for x in a)


def _dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cross(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length(a: tuple[float, ...]) -> float:
    return math.sqrt(max(0.0, _dot(a, a)))


def _unit(a: tuple[float, ...]) -> tuple[float, ...] | None:
    length = _length(a)
    if length <= _EPS:
        return None
    return tuple(x / length for x in a)


def _dist(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return _length(_sub(a, b))


def _mid(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return tuple((x + y) * 0.5 for x, y in zip(a, b))


def _quant(value: float, eps: float) -> int:
    eps = eps if eps and math.isfinite(eps) and eps > 0 else 1.0
    return int(round(value / eps))


def _quant_point(point: tuple[float, ...], eps: float) -> tuple[int, ...]:
    return tuple(_quant(c, eps) for c in point)


def _jsonable(value: Any) -> Any:
    """把任意结果转成可稳定序列化的结构（集合排序、非有限浮点转字符串）。"""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_jsonable(v) for v in value),
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    return value


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256((_HASH_PREFIX + "\n" + encoded).encode("utf-8")).hexdigest()


class _DSU:
    """并查集，用于连通分量与平滑分组。"""

    __slots__ = ("parent", "rank")

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        parent = self.parent
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra

    def groups(self) -> list[int]:
        return [self.find(i) for i in range(len(self.parent))]


# ---------------------------------------------------------------------------
# 网格读取
# ---------------------------------------------------------------------------


class _Mesh:
    __slots__ = (
        "valid",
        "positions",
        "triangles",
        "uvs",
        "corner_uvs",
        "normals",
        "corner_normals",
        "seam_edges",
        "issues",
    )

    def __init__(self) -> None:
        self.valid = False
        self.positions: list[tuple[float, float, float]] = []
        self.triangles: list[tuple[int, int, int]] = []
        self.uvs: list[tuple[float, float]] | None = None
        self.corner_uvs: list[tuple[float, float]] | None = None
        self.normals: list[tuple[float, float, float]] | None = None
        self.corner_normals: list[tuple[float, float, float]] | None = None
        self.seam_edges: list[tuple[int, int]] = []
        self.issues: list[dict[str, Any]] = []


def _as_index(value: Any) -> int | None:
    if not _is_num(value):
        return None
    as_float = float(value)
    if not as_float.is_integer():
        return None
    return int(as_float)


def _mesh_from(obj: Any) -> _Mesh:
    mesh = _Mesh()
    if obj is None or isinstance(obj, (str, bytes, bytearray)):
        return mesh
    if not (isinstance(obj, Mapping) or hasattr(obj, "positions")):
        return mesh

    raw_positions = _get(obj, "positions", None)
    if not _is_seq(raw_positions):
        return mesh

    for index, point in enumerate(raw_positions):
        if _is_vec(point, 3):
            mesh.positions.append(_vec(point))
        else:
            mesh.positions.append((0.0, 0.0, 0.0))
            mesh.issues.append({"kind": "invalid_position", "index": index})

    mesh.valid = True
    vertex_count = len(mesh.positions)

    raw_faces = _get(obj, "triangles", None)
    flat = False
    if raw_faces is None:
        raw_faces = _get(obj, "indices", None)
        flat = raw_faces is not None
    if raw_faces is not None:
        if not _is_seq(raw_faces):
            mesh.issues.append({"kind": "invalid_triangles"})
            raw_faces = []
        faces_raw = list(raw_faces)
        if flat or (faces_raw and all(_is_num(x) for x in faces_raw)):
            if len(faces_raw) % 3:
                mesh.issues.append({"kind": "trailing_indices", "count": len(faces_raw) % 3})
            face_items: list[Any] = [
                faces_raw[i : i + 3] for i in range(0, len(faces_raw) - 2, 3)
            ]
        else:
            face_items = list(faces_raw)

        raw_face_count = len(face_items)
        dropped = 0
        for item in face_items:
            if not (_is_seq(item) and len(item) >= 3):
                dropped += 1
                continue
            indices = [_as_index(x) for x in item[:3]]
            if any(index is None or not (0 <= index < vertex_count) for index in indices):
                dropped += 1
                continue
            mesh.triangles.append((indices[0], indices[1], indices[2]))
        if dropped:
            mesh.issues.append({"kind": "invalid_triangles", "dropped": dropped})
    else:
        raw_face_count = 0
        dropped = 0

    raw_uvs = _get(obj, "uvs", None)
    if raw_uvs is not None:
        if _is_seq(raw_uvs) and len(raw_uvs) == vertex_count and all(
            _is_vec(uv, 2) for uv in raw_uvs
        ):
            mesh.uvs = [_uv(uv) for uv in raw_uvs]
        else:
            mesh.issues.append({"kind": "invalid_uvs"})

    raw_normals = _get(obj, "normals", None)
    if raw_normals is not None:
        if _is_seq(raw_normals) and len(raw_normals) == vertex_count and all(
            _is_vec(normal, 3) for normal in raw_normals
        ):
            mesh.normals = [_vec(normal) for normal in raw_normals]
        else:
            mesh.issues.append({"kind": "invalid_normals"})

    # 逐角属性在三角形被丢弃后无法再对齐，直接忽略。
    raw_corner_uvs = _get(obj, "corner_uvs", None)
    if raw_corner_uvs is not None:
        expected = 3 * raw_face_count
        if (
            dropped == 0
            and _is_seq(raw_corner_uvs)
            and len(raw_corner_uvs) == expected
            and all(_is_vec(uv, 2) for uv in raw_corner_uvs)
        ):
            mesh.corner_uvs = [_uv(uv) for uv in raw_corner_uvs]
        else:
            mesh.issues.append({"kind": "invalid_corner_uvs"})

    raw_corner_normals = _get(obj, "corner_normals", None)
    if raw_corner_normals is not None:
        expected = 3 * raw_face_count
        if (
            dropped == 0
            and _is_seq(raw_corner_normals)
            and len(raw_corner_normals) == expected
            and all(_is_vec(normal, 3) for normal in raw_corner_normals)
        ):
            mesh.corner_normals = [_vec(normal) for normal in raw_corner_normals]
        else:
            mesh.issues.append({"kind": "invalid_corner_normals"})

    raw_seams = _get(obj, "seam_edges", None)
    if _is_seq(raw_seams):
        for edge in raw_seams:
            if not (_is_seq(edge) and len(edge) >= 2):
                continue
            a, b = _as_index(edge[0]), _as_index(edge[1])
            if (
                a is not None
                and b is not None
                and a != b
                and 0 <= a < vertex_count
                and 0 <= b < vertex_count
            ):
                mesh.seam_edges.append((a, b))

    return mesh


def _corner_uv(
    mesh: _Mesh, triangle: int, corner: int, vertex: int
) -> tuple[float, float] | None:
    if mesh.corner_uvs is not None:
        index = 3 * triangle + corner
        if 0 <= index < len(mesh.corner_uvs):
            return mesh.corner_uvs[index]
        return None
    if mesh.uvs is not None and 0 <= vertex < len(mesh.uvs):
        return mesh.uvs[vertex]
    return None


def _corner_normal(
    mesh: _Mesh, triangle: int, corner: int, vertex: int
) -> tuple[float, float, float] | None:
    if mesh.corner_normals is not None:
        index = 3 * triangle + corner
        if 0 <= index < len(mesh.corner_normals):
            return mesh.corner_normals[index]
        return None
    if mesh.normals is not None and 0 <= vertex < len(mesh.normals):
        return mesh.normals[vertex]
    return None


# ---------------------------------------------------------------------------
# 位置焊接与边拓扑
# ---------------------------------------------------------------------------


def _weld(
    mesh: _Mesh, weld_eps: float = _DEFAULT_WELD_EPS
) -> tuple[list[int], list[tuple[float, float, float]], list[list[int]]]:
    """按量化位置焊接顶点。

    组 id 按量化位置的字典序分配，因此与输入顶点顺序无关；
    组的代表位置取组内字典序最小的位置，同样与顺序无关。
    """
    eps = weld_eps if weld_eps and math.isfinite(weld_eps) and weld_eps > 0 else 1.0
    keys: list[tuple[int, int, int]] = []
    members: dict[tuple[int, int, int], list[int]] = {}
    for index, position in enumerate(mesh.positions):
        key = (int(round(position[0] / eps)), int(round(position[1] / eps)), int(round(position[2] / eps)))
        keys.append(key)
        members.setdefault(key, []).append(index)

    ordered = sorted(members)
    id_of = {key: index for index, key in enumerate(ordered)}
    group_of = [id_of[key] for key in keys]
    group_vertices = [members[key] for key in ordered]
    group_positions = [
        min(mesh.positions[vertex] for vertex in vertices) for vertices in group_vertices
    ]
    return group_of, group_positions, group_vertices


def _edge_occurrences(
    mesh: _Mesh, group_of: list[int]
) -> dict[tuple[int, int], list[tuple[int, Any, Any, Any, Any]]]:
    """枚举焊接后的几何边及其所有面出现。

    每个出现记录 ``(face, uv_a, uv_b, normal_a, normal_b)``，其中 a/b 与
    排序后的组 id 对齐，方便跨面比较 UV / 法线是否连续。
    """
    occurrences: dict[tuple[int, int], list[tuple[int, Any, Any, Any, Any]]] = {}
    for triangle_index, triangle in enumerate(mesh.triangles):
        groups = [group_of[vertex] for vertex in triangle]
        if len(set(groups)) < 3:
            continue  # 焊接后退化的面不参与拓扑
        for corner in range(3):
            i = triangle[corner]
            j = triangle[(corner + 1) % 3]
            group_i, group_j = group_of[i], group_of[j]
            if group_i == group_j:
                continue
            if group_i < group_j:
                key = (group_i, group_j)
                corner_a, corner_b = corner, (corner + 1) % 3
                vertex_a, vertex_b = i, j
            else:
                key = (group_j, group_i)
                corner_a, corner_b = (corner + 1) % 3, corner
                vertex_a, vertex_b = j, i
            uv_a = _corner_uv(mesh, triangle_index, corner_a, vertex_a)
            uv_b = _corner_uv(mesh, triangle_index, corner_b, vertex_b)
            normal_a = _corner_normal(mesh, triangle_index, corner_a, vertex_a)
            normal_b = _corner_normal(mesh, triangle_index, corner_b, vertex_b)
            occurrences.setdefault(key, []).append(
                (triangle_index, uv_a, uv_b, normal_a, normal_b)
            )
    return occurrences


def _face_normal_area(
    positions: list[tuple[float, float, float]], triangle: tuple[int, int, int]
) -> tuple[tuple[float, float, float] | None, float]:
    p0 = positions[triangle[0]]
    p1 = positions[triangle[1]]
    p2 = positions[triangle[2]]
    cross = _cross(_sub(p1, p0), _sub(p2, p0))
    length = _length(cross)
    if length <= _EPS:
        return None, 0.0
    return tuple(c / length for c in cross), 0.5 * length  # type: ignore[return-value]


def _rotate_face(face: tuple[int, int, int]) -> tuple[int, int, int]:
    """把三角形旋转到最小顶点下标在首位，保持绕序。"""
    start = 0 if face[0] <= face[1] and face[0] <= face[2] else (1 if face[1] <= face[2] else 2)
    return (face[start], face[(start + 1) % 3], face[(start + 2) % 3])


def _canonical_face_key(
    triangle: tuple[int, int, int], group_of: list[int]
) -> tuple[int, int, int]:
    groups = [group_of[vertex] for vertex in triangle]
    smallest = min(range(3), key=lambda k: groups[k])
    return tuple(groups[(smallest + k) % 3] for k in range(3))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# QEM
# ---------------------------------------------------------------------------


def _plane_quadric(
    normal: tuple[float, float, float], d: float, weight: float
) -> tuple[float, ...]:
    nx, ny, nz = normal
    return (
        weight * nx * nx,
        weight * nx * ny,
        weight * nx * nz,
        weight * ny * ny,
        weight * ny * nz,
        weight * nz * nz,
        weight * nx * d,
        weight * ny * d,
        weight * nz * d,
        weight * d * d,
    )


def _qadd(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(x + y for x, y in zip(a, b))


def _qerror(q: tuple[float, ...], point: tuple[float, ...]) -> float:
    a00, a01, a02, a11, a12, a22, b0, b1, b2, c = q
    x, y, z = point[0], point[1], point[2]
    return (
        a00 * x * x
        + 2.0 * a01 * x * y
        + 2.0 * a02 * x * z
        + a11 * y * y
        + 2.0 * a12 * y * z
        + a22 * z * z
        + 2.0 * b0 * x
        + 2.0 * b1 * y
        + 2.0 * b2 * z
        + c
    )


def _solve3(
    a00: float,
    a01: float,
    a02: float,
    a11: float,
    a12: float,
    a22: float,
    b0: float,
    b1: float,
    b2: float,
) -> tuple[float, float, float] | None:
    """解对称 3x3 线性方程组 A x = b；奇异时返回 None。"""
    a10, a20, a21 = a01, a02, a12
    c00 = a11 * a22 - a12 * a21
    c01 = a12 * a20 - a10 * a22
    c02 = a10 * a21 - a11 * a20
    det = a00 * c00 + a01 * c01 + a02 * c02
    if abs(det) < 1e-24:
        return None
    inv = 1.0 / det
    x = (c00 * b0 + (a02 * a21 - a01 * a22) * b1 + (a01 * a12 - a02 * a11) * b2) * inv
    y = (c01 * b0 + (a00 * a22 - a02 * a20) * b1 + (a02 * a10 - a00 * a12) * b2) * inv
    z = (c02 * b0 + (a01 * a20 - a00 * a21) * b1 + (a00 * a11 - a01 * a10) * b2) * inv
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return None
    return (x, y, z)


def _best_quadric_position(
    q: tuple[float, ...],
    pa: tuple[float, float, float],
    pb: tuple[float, float, float],
) -> tuple[float, float, float]:
    """求边坍缩的最优位置；平面区域退化为中点/端点。"""
    a00, a01, a02, a11, a12, a22, b0, b1, b2 = q[:9]
    solution = _solve3(a00, a01, a02, a11, a12, a22, -b0, -b1, -b2)
    midpoint = _mid(pa, pb)
    edge_length = _dist(pa, pb)
    candidates: list[tuple[float, float, float]] = []
    if solution is not None:
        max_offset = 4.0 * edge_length + 1e-9
        if _dist(solution, pa) <= max_offset and _dist(solution, pb) <= max_offset:
            candidates.append(solution)  # type: ignore[arg-type]
    candidates.extend((midpoint, pa, pb))  # type: ignore[arg-type]

    best = candidates[0]
    best_cost = _qerror(q, best)
    for candidate in candidates[1:]:
        cost = _qerror(q, candidate)
        if cost < best_cost - 1e-15:
            best, best_cost = candidate, cost
    return best  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 接缝分析
# ---------------------------------------------------------------------------


def _uv_close(a: tuple[float, float], b: tuple[float, float], eps: float) -> bool:
    return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps


def _angle_deg(a: tuple[float, ...] | None, b: tuple[float, ...] | None) -> float:
    if a is None or b is None:
        return 0.0
    ua, ub = _unit(a), _unit(b)
    if ua is None or ub is None:
        return 0.0
    dot = max(-1.0, min(1.0, _dot(ua, ub)))
    return math.degrees(math.acos(dot))


class _Analysis:
    __slots__ = (
        "mesh",
        "weld_eps",
        "uv_eps",
        "normal_angle",
        "group_of",
        "group_positions",
        "group_vertices",
        "occurrences",
        "face_normals",
        "group_faces",
        "locked",
        "edge_reasons",
        "protected",
        "boundary",
        "nonmanifold",
        "uv_seams",
        "normal_seams",
        "hard_edges",
        "explicit_seams",
        "quadrics",
    )

    def __init__(self) -> None:
        self.mesh: _Mesh | None = None
        self.weld_eps = _DEFAULT_WELD_EPS
        self.uv_eps = _DEFAULT_UV_EPS
        self.normal_angle = _DEFAULT_NORMAL_ANGLE
        self.group_of: list[int] = []
        self.group_positions: list[tuple[float, float, float]] = []
        self.group_vertices: list[list[int]] = []
        self.occurrences: dict[tuple[int, int], list[tuple[int, Any, Any, Any, Any]]] = {}
        self.face_normals: list[tuple[tuple[float, float, float] | None, float]] = []
        self.group_faces: list[list[int]] = []
        self.locked: dict[tuple[int, int], set[str]] = {}
        self.edge_reasons: dict[tuple[int, int], list[str]] = {}
        self.protected: list[bool] = []
        self.boundary: list[tuple[int, int]] = []
        self.nonmanifold: list[tuple[int, int]] = []
        self.uv_seams: list[tuple[int, int]] = []
        self.normal_seams: list[tuple[int, int]] = []
        self.hard_edges: list[tuple[int, int]] = []
        self.explicit_seams: list[tuple[int, int]] = []
        self.quadrics: list[tuple[float, ...]] = []

    @property
    def group_count(self) -> int:
        return len(self.group_positions)


def _analyze(
    mesh: _Mesh,
    weld_eps: float = _DEFAULT_WELD_EPS,
    uv_eps: float = _DEFAULT_UV_EPS,
    normal_angle: float = _DEFAULT_NORMAL_ANGLE,
    hard_angle: float = _DEFAULT_HARD_ANGLE,
    lock_hard_edges: bool = False,
) -> _Analysis:
    analysis = _Analysis()
    analysis.mesh = mesh
    analysis.weld_eps = weld_eps
    analysis.uv_eps = uv_eps
    analysis.normal_angle = normal_angle

    group_of, group_positions, group_vertices = _weld(mesh, weld_eps)
    analysis.group_of = group_of
    analysis.group_positions = group_positions
    analysis.group_vertices = group_vertices
    analysis.occurrences = _edge_occurrences(mesh, group_of)
    analysis.group_faces = [[] for _ in group_positions]

    for face_index, triangle in enumerate(mesh.triangles):
        analysis.face_normals.append(_face_normal_area(mesh.positions, triangle))
        for group in sorted({group_of[v] for v in triangle}):
            analysis.group_faces[group].append(face_index)

    uv_seams: set[tuple[int, int]] = set()
    normal_seams: set[tuple[int, int]] = set()
    hard_edges: set[tuple[int, int]] = set()
    boundary: list[tuple[int, int]] = []
    nonmanifold: list[tuple[int, int]] = []

    for edge in sorted(analysis.occurrences):
        occurrences = analysis.occurrences[edge]
        reasons: set[str] = set()
        if len(occurrences) == 1:
            reasons.add("boundary")
            boundary.append(edge)
        elif len(occurrences) > 2:
            reasons.add("nonmanifold")
            nonmanifold.append(edge)

        if len(occurrences) >= 2:
            with_uv = [o for o in occurrences if o[1] is not None and o[2] is not None]
            if len(with_uv) == len(occurrences):
                base = with_uv[0]
                if any(
                    not _uv_close(base[1], other[1], uv_eps)
                    or not _uv_close(base[2], other[2], uv_eps)
                    for other in with_uv[1:]
                ):
                    reasons.add("uv_seam")
                    uv_seams.add(edge)
            elif with_uv:
                # 部分角缺少 UV：属性覆盖不一致，保守地按接缝处理
                reasons.add("uv_seam")
                uv_seams.add(edge)

            with_normal = [
                o for o in occurrences if o[3] is not None and o[4] is not None
            ]
            if len(with_normal) == len(occurrences):
                base = with_normal[0]
                if any(
                    _angle_deg(base[3], other[3]) > normal_angle
                    or _angle_deg(base[4], other[4]) > normal_angle
                    for other in with_normal[1:]
                ):
                    reasons.add("normal_seam")
                    normal_seams.add(edge)
            elif with_normal:
                reasons.add("normal_seam")
                normal_seams.add(edge)
            else:
                # 没有显式法线时，用面法线夹角识别硬边。
                normals = [
                    analysis.face_normals[o[0]][0] for o in occurrences
                ]
                if all(n is not None for n in normals):
                    for i in range(len(normals)):
                        for j in range(i + 1, len(normals)):
                            if _angle_deg(normals[i], normals[j]) > hard_angle:
                                hard_edges.add(edge)

        if reasons:
            analysis.locked[edge] = reasons
            analysis.edge_reasons[edge] = sorted(reasons)

    analysis.uv_seams = sorted(uv_seams)
    analysis.normal_seams = sorted(normal_seams)
    analysis.hard_edges = sorted(hard_edges)
    analysis.boundary = boundary
    analysis.nonmanifold = nonmanifold

    explicit: set[tuple[int, int]] = set()
    for va, vb in mesh.seam_edges:
        ga, gb = group_of[va], group_of[vb]
        if ga == gb:
            continue
        edge = (ga, gb) if ga < gb else (gb, ga)
        explicit.add(edge)
        analysis.locked.setdefault(edge, set()).add("explicit_seam")
        analysis.edge_reasons[edge] = sorted(analysis.locked[edge])
    analysis.explicit_seams = sorted(explicit)

    if lock_hard_edges:
        for edge in hard_edges:
            analysis.locked.setdefault(edge, set()).add("hard_edge")
            analysis.edge_reasons[edge] = sorted(analysis.locked[edge])

    analysis.protected = [False] * len(group_positions)
    for ga, gb in analysis.locked:
        analysis.protected[ga] = True
        analysis.protected[gb] = True
    for edge in analysis.explicit_seams:
        analysis.protected[edge[0]] = True
        analysis.protected[edge[1]] = True

    analysis.quadrics = _build_quadrics(mesh, analysis)
    return analysis


def _build_quadrics(mesh: _Mesh, analysis: _Analysis) -> list[tuple[float, ...]]:
    """按「规范面键」排序后累加面平面的 QEM，保证与输入顺序无关。"""
    group_of = analysis.group_of
    entries: list[
        tuple[
            tuple[int, int, int],
            tuple[float, ...],
            int,
            tuple[float, float, float],
            float,
            float,
        ]
    ] = []
    for face_index, triangle in enumerate(mesh.triangles):
        key = _canonical_face_key(triangle, group_of)
        if key[0] == key[1] or key[1] == key[2] or key[0] == key[2]:
            continue
        normal, area = analysis.face_normals[face_index]
        if normal is None or area <= _EPS:
            continue
        d = -_dot(normal, mesh.positions[triangle[0]])
        signature = tuple(round(value, 9) for value in normal) + (round(d, 9), round(area, 9))
        entries.append((key, signature, face_index, normal, d, area))
    entries.sort(key=lambda item: (item[0], item[1]))

    quadrics = [_QZERO] * len(analysis.group_positions)
    for key, _signature, face_index, normal, d, area in entries:
        plane = _plane_quadric(normal, d, area)
        for group in sorted(set(key)):
            quadrics[group] = _qadd(quadrics[group], plane)
    return quadrics


def _edge_length(analysis: _Analysis, a: int, b: int) -> float:
    return _dist(analysis.group_positions[a], analysis.group_positions[b])


def _is_seam_adjacent(analysis: _Analysis, a: int, b: int) -> bool:
    """边的任一侧三角形是否触及受保护（接缝/边界/非流形）顶点。"""
    protected = analysis.protected
    faces = set(analysis.group_faces[a]) | set(analysis.group_faces[b])
    for face_index in faces:
        triangle = analysis.mesh.triangles[face_index]  # type: ignore[union-attr]
        for vertex in triangle:
            group = analysis.group_of[vertex]
            if group != a and group != b and protected[group]:
                return True
    return False


def _dihedral_penalty(analysis: _Analysis, edge: tuple[int, int]) -> float:
    normals = []
    for occurrence in analysis.occurrences.get(edge, []):
        normal = analysis.face_normals[occurrence[0]][0]
        if normal is not None:
            normals.append(normal)
    penalty = 0.0
    for i in range(len(normals)):
        for j in range(i + 1, len(normals)):
            cos = math.cos(math.radians(_angle_deg(normals[i], normals[j])))
            penalty += max(0.0, 1.0 - cos)
    return penalty


def _collapse_metrics(
    analysis: _Analysis,
    a: int,
    b: int,
    *,
    seam_weight: float = 0.05,
    dihedral_weight: float = 0.05,
) -> tuple[float, dict[str, float], tuple[float, float, float] | None, str | None]:
    """返回 ``(总代价, 分项, 最优位置, 锁定原因)``。"""
    edge = (a, b) if a < b else (b, a)
    if a == b:
        return INF, {"qem": INF, "dihedral": 0.0, "seam": 0.0, "total": INF}, None, "degenerate"
    if edge in analysis.locked:
        reasons = analysis.edge_reasons.get(edge, ["locked"])
        return (
            INF,
            {"qem": INF, "dihedral": 0.0, "seam": 0.0, "total": INF},
            None,
            "+".join(reasons),
        )
    if analysis.protected[a] or analysis.protected[b]:
        return (
            INF,
            {"qem": INF, "dihedral": 0.0, "seam": 0.0, "total": INF},
            None,
            "vertex_lock",
        )

    quadric = _qadd(analysis.quadrics[a], analysis.quadrics[b])
    position = _best_quadric_position(
        quadric, analysis.group_positions[a], analysis.group_positions[b]
    )
    qem = max(0.0, _qerror(quadric, position))
    dihedral = dihedral_weight * _dihedral_penalty(analysis, edge)
    length = _edge_length(analysis, a, b)
    seam = seam_weight * (length * length) if _is_seam_adjacent(analysis, a, b) else 0.0
    total = qem + dihedral + seam
    breakdown = {"qem": qem, "dihedral": dihedral, "seam": seam, "total": total}
    return total, breakdown, position, None


def _seam_summary(analysis: _Analysis) -> dict[str, Any]:
    mesh = analysis.mesh
    assert mesh is not None
    protected_vertices: list[int] = []
    for index, is_protected in enumerate(analysis.protected):
        if is_protected:
            protected_vertices.extend(analysis.group_vertices[index])
    locked_edges = [
        {"a": edge[0], "b": edge[1], "reasons": analysis.edge_reasons.get(edge, [])}
        for edge in sorted(analysis.locked)
    ]
    return {
        "locked_edges": locked_edges,
        "locked_edge_count": len(locked_edges),
        "locked_groups": [i for i, flag in enumerate(analysis.protected) if flag],
        "locked_vertices": sorted(protected_vertices),
        "boundary_edges": [list(edge) for edge in analysis.boundary],
        "nonmanifold_edges": [list(edge) for edge in analysis.nonmanifold],
        "uv_seams": [list(edge) for edge in analysis.uv_seams],
        "normal_seams": [list(edge) for edge in analysis.normal_seams],
        "hard_edges": [list(edge) for edge in analysis.hard_edges],
        "explicit_seams": [list(edge) for edge in analysis.explicit_seams],
        "weld_eps": analysis.weld_eps,
        "uv_eps": analysis.uv_eps,
        "normal_angle": analysis.normal_angle,
        "vertex_count": len(mesh.positions),
        "group_count": analysis.group_count,
        "face_count": len(mesh.triangles),
    }


def _invalid_result(kind: str, reason: str = "invalid_mesh", **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"valid": False, "kind": kind, "reason": reason}
    result.update(extra)
    return result


# ---------------------------------------------------------------------------
# 公共入口：接缝锁定 / 边权
# ---------------------------------------------------------------------------


def seam_lock(
    mesh: Any,
    *,
    weld_eps: float = _DEFAULT_WELD_EPS,
    uv_eps: float = _DEFAULT_UV_EPS,
    normal_angle: float = _DEFAULT_NORMAL_ANGLE,
    hard_angle: float = _DEFAULT_HARD_ANGLE,
    lock_hard_edges: bool = False,
) -> dict[str, Any]:
    """检测 UV / 法线接缝、边界、非流形边与硬边，并给出锁定集合。

    锁定策略：

    * 接缝边、边界边、非流形边永不坍缩；
    * 锁定边的两个端点视为「受保护顶点」，不允许被其它坍缩合并或移动；
    * 可选的硬边锁定用于没有显式法线、但需要保持折痕的资产。
    """
    parsed = _mesh_from(mesh)
    if not parsed.valid:
        return _invalid_result("seam_lock", **{"locked_edges": [], "locked_vertices": []})
    analysis = _analyze(
        parsed,
        weld_eps=weld_eps,
        uv_eps=uv_eps,
        normal_angle=normal_angle,
        hard_angle=hard_angle,
        lock_hard_edges=lock_hard_edges,
    )
    summary = _seam_summary(analysis)
    summary["valid"] = True
    summary["kind"] = "seam_lock"
    summary["has_seams"] = bool(
        analysis.uv_seams or analysis.normal_seams or analysis.explicit_seams
    )
    summary["has_locks"] = bool(analysis.locked)
    return summary


def _edge_entry(
    analysis: _Analysis,
    a: int,
    b: int,
    *,
    seam_weight: float,
    dihedral_weight: float,
) -> dict[str, Any]:
    edge = (a, b) if a < b else (b, a)
    cost, breakdown, position, reason = _collapse_metrics(
        analysis, a, b, seam_weight=seam_weight, dihedral_weight=dihedral_weight
    )
    return {
        "group_a": edge[0],
        "group_b": edge[1],
        "vertices": [
            analysis.group_vertices[edge[0]][0],
            analysis.group_vertices[edge[1]][0],
        ],
        "cost": cost,
        "cost_quantized": int(round(cost * _COST_SCALE)) if math.isfinite(cost) else None,
        "locked": reason is not None,
        "lock_reason": reason,
        "breakdown": breakdown,
        "position": list(position) if position is not None else None,
    }


def _resolve_edge(
    analysis: _Analysis, edge: Any, by_group: bool
) -> tuple[int, int] | None:
    if isinstance(edge, Mapping):
        a, b = edge.get("a"), edge.get("b")
        by_group = bool(edge.get("by_group", by_group))
    elif _is_seq(edge) and len(edge) >= 2:
        a, b = edge[0], edge[1]
    else:
        return None
    ia, ib = _as_index(a), _as_index(b)
    if ia is None or ib is None or ia == ib:
        return None
    if by_group:
        ga, gb = ia, ib
    else:
        vertices = analysis.group_of
        if not (0 <= ia < len(vertices) and 0 <= ib < len(vertices)):
            return None
        ga, gb = vertices[ia], vertices[ib]
    if not (0 <= ga < analysis.group_count and 0 <= gb < analysis.group_count):
        return None
    return (ga, gb) if ga < gb else (gb, ga)


def edge_cost(
    mesh: Any,
    edge: Any = None,
    *,
    weld_eps: float = _DEFAULT_WELD_EPS,
    uv_eps: float = _DEFAULT_UV_EPS,
    normal_angle: float = _DEFAULT_NORMAL_ANGLE,
    hard_angle: float = _DEFAULT_HARD_ANGLE,
    lock_hard_edges: bool = False,
    by_group: bool = False,
    seam_weight: float = 0.05,
    dihedral_weight: float = 0.05,
) -> dict[str, Any]:
    """QEM 边权。

    代价 = 二次误差 + 二面角罚项 + 接缝邻近罚项：

    * 二次误差：两个端点 QEM 合并后在最优位置处取值，控制几何误差；
    * 二面角罚项：夹角越尖，坍缩越容易改变轮廓/高光；
    * 接缝邻近罚项：坍缩边所在三角形触及受保护顶点时加权，
      让减面优先发生在远离接缝的区域。

    接缝边返回 ``inf``；接缝端点参与的边返回 ``inf``（``vertex_lock``）。
    """
    parsed = _mesh_from(mesh)
    if not parsed.valid:
        return _invalid_result("edge_cost", cost=None, edges=[])
    analysis = _analyze(
        parsed,
        weld_eps=weld_eps,
        uv_eps=uv_eps,
        normal_angle=normal_angle,
        hard_angle=hard_angle,
        lock_hard_edges=lock_hard_edges,
    )

    if edge is not None:
        resolved = _resolve_edge(analysis, edge, by_group)
        if resolved is None:
            return {
                "valid": True,
                "kind": "edge_cost",
                "found": False,
                "cost": None,
                "reason": "edge_not_found",
            }
        entry = _edge_entry(
            analysis,
            resolved[0],
            resolved[1],
            seam_weight=seam_weight,
            dihedral_weight=dihedral_weight,
        )
        entry.update(
            {
                "valid": True,
                "kind": "edge_cost",
                "found": True,
                "cost": entry["cost"],
                "breakdown": entry["breakdown"],
            }
        )
        return entry

    entries = [
        _edge_entry(
            analysis,
            edge_pair[0],
            edge_pair[1],
            seam_weight=seam_weight,
            dihedral_weight=dihedral_weight,
        )
        for edge_pair in sorted(analysis.occurrences)
    ]
    entries.sort(
        key=lambda item: (
            item["cost"] if math.isfinite(item["cost"]) else INF,
            item["group_a"],
            item["group_b"],
        )
    )
    finite = [item for item in entries if math.isfinite(item["cost"])]
    return {
        "valid": True,
        "kind": "edge_cost",
        "edges": entries,
        "edge_count": len(entries),
        "locked_count": len(entries) - len(finite),
        "collapsible_count": len(finite),
        "min_cost": finite[0]["cost"] if finite else None,
        "max_cost": finite[-1]["cost"] if finite else None,
    }


def _uv_extent(uvs: list[tuple[float, float]]) -> float:
    width = max(uv[0] for uv in uvs) - min(uv[0] for uv in uvs)
    height = max(uv[1] for uv in uvs) - min(uv[1] for uv in uvs)
    return math.hypot(width, height)


def _unwrap_uv(
    origin: tuple[float, float], uv: tuple[float, float], period: float = 1.0
) -> tuple[float, float]:
    """把 UV 平移到距 origin 最近的周期副本。"""
    if period <= 0:
        return uv
    du = uv[0] - origin[0]
    dv = uv[1] - origin[1]
    return (
        uv[0] - period * round(du / period),
        uv[1] - period * round(dv / period),
    )


def _select_uv_basis(
    uvs: list[tuple[float, float]], period: float
) -> list[tuple[float, float]]:
    """选择该面的 UV 解释：默认原样，需要时按周期展开。

    ``period <= 0`` 表示禁用展开（默认）。正值表示资产坐标是周期性的
    （例如 UDIM），此时取原样与展开中 UV 跨度更小的那个。

    注意「恰好占满一个 tile」的三角形展开后会压成一点，所以只在展开
    明显更紧凑时才采用展开结果，避免把正常三角形判成退化。
    """
    if period is None or period <= 0 or not uvs:
        return list(uvs)
    unwrapped = [_unwrap_uv(uvs[0], uv, period) for uv in uvs]
    if _uv_extent(unwrapped) < _uv_extent(uvs) - 1e-9:
        return unwrapped
    return list(uvs)


def _orthogonalize(
    normal: tuple[float, float, float],
    tangent: tuple[float, float, float],
    bitangent: tuple[float, float, float],
) -> tuple[tuple[float, float, float], float] | None:
    """Gram-Schmidt：让切线垂直于法线，并按 bitangent 方向定 handedness。"""
    unit_normal = _unit(normal)
    unit_tangent = _unit(tangent)
    if unit_normal is None or unit_tangent is None:
        return None
    projected = _sub(unit_tangent, _scale(unit_normal, _dot(unit_normal, unit_tangent)))
    final_tangent = _unit(projected)
    if final_tangent is None:
        return None
    handedness = -1.0 if _dot(_cross(unit_normal, final_tangent), bitangent) < 0.0 else 1.0
    return final_tangent, handedness  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 确定性坍缩顺序
# ---------------------------------------------------------------------------


def _canonical_face_set(mesh: _Mesh, group_of: list[int]) -> list[tuple[int, int, int]]:
    """与输入顺序无关的规范面集合（用于确定性对照）。"""
    return sorted(
        {
            _canonical_face_key(triangle, group_of)
            for triangle in mesh.triangles
            if len({group_of[v] for v in triangle}) == 3
        }
    )


def _edge_geom_key(
    analysis: _Analysis, a: int, b: int, weld_eps: float
) -> tuple[int, ...]:
    pa = _quant_point(analysis.group_positions[a], weld_eps)
    pb = _quant_point(analysis.group_positions[b], weld_eps)
    return pa + pb


def _collapse_plan(
    analysis: _Analysis,
    *,
    seam_weight: float = 0.05,
    dihedral_weight: float = 0.05,
) -> list[dict[str, Any]]:
    """静态代价排序（不改变拓扑）。

    排序键：量化代价 -> 两端位置的几何键 -> 组 id。
    全部基于位置/焊接后组号，与输入顶点顺序无关。
    """
    weld_eps = analysis.weld_eps
    entries: list[dict[str, Any]] = []
    for a, b in sorted(analysis.occurrences):
        cost, breakdown, position, reason = _collapse_metrics(
            analysis, a, b, seam_weight=seam_weight, dihedral_weight=dihedral_weight
        )
        entries.append(
            {
                "group_a": a,
                "group_b": b,
                "vertices": [
                    analysis.group_vertices[a][0],
                    analysis.group_vertices[b][0],
                ],
                "cost": cost,
                "cost_quantized": int(round(cost * _COST_SCALE))
                if math.isfinite(cost)
                else None,
                "breakdown": breakdown,
                "position": list(position) if position is not None else None,
                "locked": reason is not None,
                "lock_reason": reason,
                "_key": (
                    int(round(cost * _COST_SCALE)) if math.isfinite(cost) else 1 << 62,
                    _edge_geom_key(analysis, a, b, weld_eps),
                    (a, b),
                ),
            }
        )
    entries.sort(key=lambda item: item["_key"])
    for rank, item in enumerate(entries):
        item["rank"] = rank
        item.pop("_key", None)
    return entries


class _SimplifyState:
    """边坍缩过程中的可变拓扑状态。

    ``faces`` 存的是焊接后组 id；``-1`` 表示已删除（坍缩两侧的退化面）。
    组 id 与位置解耦：坍缩只改位置与面索引，不重排组，因此组号始终是
    确定性的锚点。
    """

    __slots__ = (
        "positions",
        "alive",
        "quadrics",
        "faces",
        "group_faces",
        "face_keys",
        "weld_eps",
    )

    def __init__(self, analysis: _Analysis) -> None:
        self.positions = list(analysis.group_positions)
        self.alive = [True] * analysis.group_count
        self.quadrics = list(analysis.quadrics)
        self.weld_eps = analysis.weld_eps
        triangles = analysis.mesh.triangles if analysis.mesh else ()
        faces: list[tuple[int, int, int] | None] = []
        for triangle in triangles:
            face = _rotate_face(tuple(analysis.group_of[v] for v in triangle))
            faces.append(face if len(set(face)) == 3 else None)
        self.faces = faces
        self.group_faces: list[set[int]] = [set() for _ in range(analysis.group_count)]
        for face_index, face in enumerate(faces):
            if face is not None:
                for group in face:
                    self.group_faces[group].add(face_index)
        self.face_keys: dict[tuple[int, int, int], int] = {}
        for face_index, face in enumerate(faces):
            if face is not None:
                self.face_keys[face] = face_index

    def live_faces(self) -> int:
        return len(self.face_keys)

    def face_set(self, face_index: int) -> set[int]:
        face = self.faces[face_index]
        return set() if face is None else set(face)

    def remove_face(self, face_index: int) -> None:
        face = self.faces[face_index]
        if face is None:
            return
        if self.face_keys.get(face) == face_index:
            del self.face_keys[face]
        for group in face:
            self.group_faces[group].discard(face_index)
        self.faces[face_index] = None

    def replace_face(
        self, face_index: int, new_face: tuple[int, int, int]
    ) -> None:
        old_face = self.faces[face_index]
        if old_face is not None and self.face_keys.get(old_face) == face_index:
            del self.face_keys[old_face]
        for group in old_face or ():
            self.group_faces[group].discard(face_index)
        self.faces[face_index] = new_face
        for group in new_face:
            self.group_faces[group].add(face_index)
        self.face_keys[new_face] = face_index


def _face_normal_of(
    positions: list[tuple[float, float, float]], face: tuple[int, int, int]
) -> tuple[float, float, float] | None:
    normal, area = _face_normal_area(positions, face)
    if normal is None or area <= _EPS:
        return None
    return normal


def _state_edge_cost(
    state: _SimplifyState,
    protected: list[bool],
    a: int,
    b: int,
    *,
    seam_weight: float,
    dihedral_weight: float,
) -> tuple[float, tuple[float, float, float] | None]:
    """当前状态下的边代价（QEM + 二面角 + 接缝邻近）。"""
    quadric = _qadd(state.quadrics[a], state.quadrics[b])
    position = _best_quadric_position(quadric, state.positions[a], state.positions[b])
    qem = max(0.0, _qerror(quadric, position))

    face_indices = state.group_faces[a] | state.group_faces[b]
    normals: list[tuple[float, float, float]] = []
    seam_penalty = 0.0
    for face_index in sorted(face_indices):
        face = state.faces[face_index]
        if face is None:
            continue
        normal = _face_normal_of(state.positions, face)
        if normal is not None:
            normals.append(normal)
        for group in face:
            if group not in (a, b) and protected[group]:
                seam_penalty += 1.0
                break
    dihedral = 0.0
    for i in range(len(normals)):
        for j in range(i + 1, len(normals)):
            cos = max(-1.0, min(1.0, _dot(normals[i], normals[j])))
            dihedral += max(0.0, 1.0 - cos)
    length = _dist(state.positions[a], state.positions[b])
    total = (
        qem
        + dihedral_weight * dihedral
        + (seam_weight * length * length if seam_penalty else 0.0)
    )
    return total, position


def _shared_faces(state: _SimplifyState, a: int, b: int) -> set[int]:
    return state.group_faces[a] & state.group_faces[b]


def _link_ok(state: _SimplifyState, a: int, b: int) -> bool:
    """流形边坍缩的 link condition。

    两个端点的公共邻居必须恰好是两个共享面的另外两个对角顶点，
    否则坍缩会捏合出非流形边或把不相连的壳粘在一起。
    """
    shared = _shared_faces(state, a, b)
    if len(shared) != 2:
        return False
    neighbours_a: set[int] = set()
    neighbours_b: set[int] = set()
    for face_index in state.group_faces[a]:
        neighbours_a |= state.face_set(face_index)
    for face_index in state.group_faces[b]:
        neighbours_b |= state.face_set(face_index)
    common = (neighbours_a & neighbours_b) - {a, b}
    opposite: set[int] = set()
    for face_index in shared:
        opposite |= state.face_set(face_index) - {a, b}
    return common == opposite and len(opposite) == 2


def _try_collapse(
    state: _SimplifyState,
    protected: list[bool],
    a: int,
    b: int,
    position: tuple[float, float, float],
    *,
    min_faces_for_closed: int = 0,
) -> tuple[str | None, int]:
    """把 b 合并进 a。

    成功时删除共享边两侧的两个三角形，其余相邻面把 b 替换成 a，
    返回 ``(None, 删除面数)``；失败返回 ``(原因, 0)`` 且不改动状态。
    """
    if not (state.alive[a] and state.alive[b]):
        return "inactive", 0
    if a == b:
        return "degenerate_edge", 0
    if protected[a] or protected[b]:
        return "vertex_lock", 0
    if not _link_ok(state, a, b):
        return "link_condition", 0

    shared = _shared_faces(state, a, b)
    if min_faces_for_closed and state.live_faces() - len(shared) < min_faces_for_closed:
        # 闭合壳最少需要 4 个面（四面体）；再减会退化成零体积的双面片。
        return "below_minimum_faces", 0
    old_normals: dict[int, tuple[float, float, float]] = {}
    for face_index in sorted(shared):
        normal = _face_normal_of(state.positions, state.faces[face_index])  # type: ignore[arg-type]
        if normal is not None:
            old_normals[face_index] = normal

    probe_positions = list(state.positions)
    probe_positions[a] = position
    updates: list[tuple[int, tuple[int, int, int]]] = []
    for face_index in sorted(state.group_faces[a] | state.group_faces[b]):
        if face_index in shared:
            continue
        face = state.faces[face_index]
        if face is None:
            continue
        remapped = _rotate_face(tuple(a if group == b else group for group in face))
        if len(set(remapped)) < 3:
            return "degenerate_face", 0
        normal, area = _face_normal_area(probe_positions, remapped)
        if normal is None or area <= _EPS:
            return "zero_area", 0
        old_normal = old_normals.get(face_index)
        if old_normal is not None and _dot(normal, old_normal) <= 0.0:
            return "normal_flip", 0
        existing = state.face_keys.get(remapped)
        if existing is not None and existing != face_index and existing not in shared:
            return "duplicate_face", 0
        updates.append((face_index, remapped))

    state.positions[a] = position
    state.quadrics[a] = _qadd(state.quadrics[a], state.quadrics[b])
    state.alive[b] = False
    removed = 0
    for face_index in sorted(shared):
        state.remove_face(face_index)
        removed += 1
    for face_index, remapped in updates:
        state.replace_face(face_index, remapped)
    state.group_faces[b] = set()
    return None, removed


def _greedy_simplify(
    analysis: _Analysis,
    *,
    target_faces: int | None = None,
    ratio: float | None = None,
    max_steps: int | None = None,
    seam_weight: float = 0.05,
    dihedral_weight: float = 0.05,
) -> tuple[_SimplifyState, list[dict[str, Any]], dict[str, Any]]:
    """确定性贪心边坍缩。

    每步枚举当前所有活边，按 (量化代价, 几何键, 组 id) 取最小；
    取最小的一条若被拓扑检查拒绝，就整体停下 —— 与常见 QEM 实现
    「拒绝后继续找下一条」不同，但顺序完全可复现，且代价单调不倒退。
    """
    state = _SimplifyState(analysis)
    protected = list(analysis.protected)
    locked_edges = set(analysis.locked)
    total_faces = state.live_faces()

    if ratio is not None and _is_num(ratio):
        target = int(round(total_faces * max(0.0, min(1.0, float(ratio)))))
    elif target_faces is not None and _is_num(target_faces):
        target = max(0, int(target_faces))
    else:
        target = None
    limit = total_faces if max_steps is None else max(0, int(max_steps))
    # 闭合网格（无边界边）至少要保留 4 个面，避免坍缩成折叠双面片。
    closed_guard = 4 if not analysis.boundary else 0

    order: list[dict[str, Any]] = []
    rejects: dict[str, int] = {}
    stop_reason: str | None = None
    steps = 0
    while steps < limit:
        if target is not None and state.live_faces() <= target:
            stop_reason = "target_reached"
            break
        candidates: list[
            tuple[int, tuple[int, ...], int, int, float, tuple[float, float, float] | None]
        ] = []
        seen_edges: set[tuple[int, int]] = set()
        for face in state.faces:
            if face is None:
                continue
            for corner in range(3):
                i, j = face[corner], face[(corner + 1) % 3]
                edge = (i, j) if i < j else (j, i)
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                if edge in locked_edges or protected[edge[0]] or protected[edge[1]]:
                    continue
                cost, position = _state_edge_cost(
                    state,
                    protected,
                    edge[0],
                    edge[1],
                    seam_weight=seam_weight,
                    dihedral_weight=dihedral_weight,
                )
                geom = _quant_point(
                    state.positions[edge[0]], state.weld_eps
                ) + _quant_point(state.positions[edge[1]], state.weld_eps)
                candidates.append(
                    (
                        int(round(cost * _COST_SCALE)),
                        geom,
                        edge[0],
                        edge[1],
                        cost,
                        position,
                    )
                )
        if not candidates:
            stop_reason = "no_candidates"
            break
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        a, b, cost, position = (
            candidates[0][2],
            candidates[0][3],
            candidates[0][4],
            candidates[0][5],
        )
        if position is None:
            stop_reason = "no_position"
            break
        failure, removed = _try_collapse(
            state, protected, a, b, position, min_faces_for_closed=closed_guard
        )
        if failure is not None:
            rejects[failure] = rejects.get(failure, 0) + 1
            stop_reason = failure
            break
        steps += 1
        order.append(
            {
                "step": steps,
                "group_a": a,
                "group_b": b,
                "cost": cost,
                "cost_quantized": int(round(cost * _COST_SCALE)),
                "position": list(position),
                "geom_key": list(candidates[0][1]),
                "removed_faces": removed,
            }
        )

    stats = {
        "steps": steps,
        "faces_before": total_faces,
        "faces_after": state.live_faces(),
        "target_faces": target,
        "stop_reason": stop_reason,
        "rejects": dict(sorted(rejects.items())),
    }
    return state, order, stats


# ---------------------------------------------------------------------------
# 切线空间重建
# ---------------------------------------------------------------------------


def _fallback_tangent(
    normal: tuple[float, float, float]
) -> tuple[tuple[float, float, float], float]:
    unit_normal = _unit(normal) or (0.0, 0.0, 1.0)
    axis = (1.0, 0.0, 0.0) if abs(unit_normal[0]) < 0.9 else (0.0, 1.0, 0.0)
    projected = _sub(axis, _scale(unit_normal, _dot(unit_normal, axis)))
    tangent = _unit(projected) or (1.0, 0.0, 0.0)
    return tangent, 1.0  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 公共入口：确定性顺序 / 切线
# ---------------------------------------------------------------------------


def collapse_order(
    mesh: Any,
    target_faces: int | None = None,
    *,
    ratio: float | None = None,
    max_steps: int | None = None,
    weld_eps: float = _DEFAULT_WELD_EPS,
    uv_eps: float = _DEFAULT_UV_EPS,
    normal_angle: float = _DEFAULT_NORMAL_ANGLE,
    hard_angle: float = _DEFAULT_HARD_ANGLE,
    lock_hard_edges: bool = False,
    seam_weight: float = 0.05,
    dihedral_weight: float = 0.05,
) -> dict[str, Any]:
    """确定性边坍缩顺序。

    不传 ``target_faces`` / ``ratio`` 时只返回静态排序计划（不改变拓扑）；
    传了就按该顺序执行贪心坍缩，并附带简化结果。

    顺序键依次为：量化代价、两端位置的几何键、焊接后组号。三者都与输入
    顶点/三角形的排列无关，因此重复导入同一模型得到相同顺序与相同拓扑。
    """
    parsed = _mesh_from(mesh)
    if not parsed.valid:
        return _invalid_result("collapse_order", order=[], steps=0)
    analysis = _analyze(
        parsed,
        weld_eps=weld_eps,
        uv_eps=uv_eps,
        normal_angle=normal_angle,
        hard_angle=hard_angle,
        lock_hard_edges=lock_hard_edges,
    )
    plan = _collapse_plan(
        analysis, seam_weight=seam_weight, dihedral_weight=dihedral_weight
    )
    result: dict[str, Any] = {
        "valid": True,
        "kind": "collapse_order",
        "order": plan,
        "steps": len(plan),
        "locked_count": sum(1 for item in plan if item["locked"]),
        "collapsible_count": sum(1 for item in plan if not item["locked"]),
        "deterministic": True,
        "order_hash": _hash_payload(
            [(item["group_a"], item["group_b"], item["cost_quantized"]) for item in plan]
        ),
    }

    if target_faces is not None or ratio is not None:
        state, performed, stats = _greedy_simplify(
            analysis,
            target_faces=target_faces,
            ratio=ratio,
            max_steps=max_steps,
            seam_weight=seam_weight,
            dihedral_weight=dihedral_weight,
        )
        faces: list[tuple[int, int, int]] = []
        used: set[int] = set()
        for face in state.faces:
            if face is None:
                continue
            faces.append(face)
            used.update(face)
        # 组 id 由位置量化后字典序分配，天然与输入顺序无关；
        # 面统一旋转到最小下标并在输出时排序，保证结果逐字节可复现。
        ordered_groups = sorted(used)
        remap = {group: index for index, group in enumerate(ordered_groups)}
        positions = [list(state.positions[group]) for group in ordered_groups]
        triangles = sorted(
            _rotate_face(tuple(remap[group] for group in face)) for face in faces
        )
        result["performed"] = performed
        result["performed_count"] = len(performed)
        result["steps"] = len(performed)
        result["plan_length"] = len(plan)
        result["stats"] = stats
        result["result"] = {
            "positions": positions,
            "triangles": [list(triangle) for triangle in triangles],
            "group_ids": ordered_groups,
        }
        result["result_hash"] = _hash_payload(
            {"positions": positions, "triangles": result["result"]["triangles"]}
        )
    return result


def tangent_rebuild(
    mesh: Any,
    *,
    weld_eps: float = _DEFAULT_WELD_EPS,
    uv_eps: float = _DEFAULT_UV_EPS,
    normal_angle: float = _DEFAULT_NORMAL_ANGLE,
    period: float = 0.0,
) -> dict[str, Any]:
    """重建切线空间。

    关键点（对应「法线接缝与切线空间不一致」）：

    * 按「焊接位置 + UV 图」而不是按顶点下标分组累加 UV 导数，
      因此 UV 拆点产生的同位置副本会得到同一切线；
    * 同一 UV 图内的法线接缝共享同一切线方向，只按各自的法线做
      Gram-Schmidt 正交化 —— 接缝两侧高光方向保持一致；
    * 不同 UV 图各自独立，避免接缝两侧互相污染；
    * handedness 逐角记录，镜像 UV 不会把副切线方向弄反。
    """
    parsed = _mesh_from(mesh)
    if not parsed.valid:
        return _invalid_result("tangent_rebuild", tangents=[], handedness=[])

    analysis = _analyze(
        parsed, weld_eps=weld_eps, uv_eps=uv_eps, normal_angle=normal_angle
    )
    if parsed.corner_uvs is None and parsed.uvs is None:
        normals = parsed.normals or [(0.0, 0.0, 1.0)] * len(parsed.positions)
        tangents: list[list[float]] = []
        handedness: list[float] = []
        for normal in normals:
            tangent, sign = _fallback_tangent(normal)
            tangents.append(list(tangent))
            handedness.append(sign)
        return {
            "valid": True,
            "kind": "tangent_rebuild",
            "reason": "missing_uvs",
            "tangents": tangents,
            "handedness": handedness,
            "corner_tangents": [],
            "corner_handedness": [],
            "chart_count": 0,
            "seam_corner_count": 0,
            "normal_seam_count": len(analysis.normal_seams),
            "uv_seam_count": len(analysis.uv_seams),
            "vertex_count": len(parsed.positions),
            "triangle_count": len(parsed.triangles),
        }

    corner_count = 3 * len(parsed.triangles)
    corner_uvs: list[tuple[float, float] | None] = [None] * corner_count
    # (数值累加器, 顶点组, 图键)
    accumulator: dict[tuple[int, tuple[int, int, int]], list[float]] = {}
    corner_chart: dict[int, tuple[int, tuple[int, int, int]]] = {}

    for face_index, triangle in enumerate(parsed.triangles):
        face_uvs: list[tuple[float, float] | None] = []
        for corner in range(3):
            vertex = triangle[corner]
            uv = _corner_uv(parsed, face_index, corner, vertex)
            face_uvs.append(uv)
            if uv is not None:
                corner_uvs[3 * face_index + corner] = uv
        if any(uv is None for uv in face_uvs):
            continue
        uvs = [uv for uv in face_uvs if uv is not None]
        unwrapped = _select_uv_basis(uvs, period)
        if len({_uv_key(uv, uv_eps) for uv in unwrapped}) < 3:
            continue  # UV 退化，不贡献切线
        local_accum: dict[int, list[float]] = {}
        _accumulate_corner_tangents(
            parsed.positions, triangle, unwrapped, local_accum
        )
        # 图键取 unwrap 后的最小 UV，与三角形的角顺序无关。
        chart_uv = min((_uv_key(uv, uv_eps) for uv in unwrapped))
        for corner in range(3):
            vertex = triangle[corner]
            group = analysis.group_of[vertex]
            key = (group, chart_uv)
            entry = accumulator.setdefault(
                key, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            )
            local = local_accum[corner]
            for component in range(6):
                entry[component] += local[component]
            corner_chart[3 * face_index + corner] = key

    tangents: list[list[float]] = [[0.0, 0.0, 1.0] for _ in parsed.positions]
    bitangents: list[list[float]] = [[0.0, 1.0, 0.0] for _ in parsed.positions]
    handedness: list[float] = [1.0] * len(parsed.positions)

    for corner_index, key in corner_chart.items():
        if key is None:
            continue
        face_index, corner = divmod(corner_index, 3)
        vertex = parsed.triangles[face_index][corner]
        normal = _corner_normal(parsed, face_index, corner, vertex)
        if normal is None:
            normal = analysis.face_normals[face_index][0]
        if normal is None:
            normal = (0.0, 0.0, 1.0)
        accumulated = accumulator.get(key)
        if accumulated is None:
            tangent, sign = _fallback_tangent(normal)
            bitangent = _cross(_unit(normal) or (0.0, 0.0, 1.0), tangent)
        else:
            result = _orthogonalize(
                normal,
                (accumulated[0], accumulated[1], accumulated[2]),
                (accumulated[3], accumulated[4], accumulated[5]),
            )
            if result is None:
                tangent, sign = _fallback_tangent(normal)
                bitangent = _cross(_unit(normal) or (0.0, 0.0, 1.0), tangent)
            else:
                tangent, sign = result
                solid = _unit(normal) or (0.0, 0.0, 1.0)
                bitangent = _cross(solid, tangent)
        tangents[vertex] = list(tangent)
        bitangents[vertex] = list(bitangent)
        handedness[vertex] = sign

    corner_tangents: list[list[float]] = []
    corner_handedness: list[float] = []
    corner_bitangents: list[list[float]] = []
    for corner_index in range(corner_count):
        face_index, corner = divmod(corner_index, 3)
        vertex = parsed.triangles[face_index][corner]
        corner_tangents.append(list(tangents[vertex]))
        corner_handedness.append(handedness[vertex])
        corner_bitangents.append(list(bitangents[vertex]))

    chart_count = len({key for key in corner_chart.values()})
    seam_corners = 0
    for edge, occurrences in analysis.occurrences.items():
        if edge in analysis.locked:
            seam_corners += 2 * len(occurrences)

    return {
        "valid": True,
        "kind": "tangent_rebuild",
        "tangents": tangents,
        "bitangents": bitangents,
        "handedness": handedness,
        "corner_tangents": corner_tangents,
        "corner_bitangents": corner_bitangents,
        "corner_handedness": corner_handedness,
        "vertex_count": len(parsed.positions),
        "triangle_count": len(parsed.triangles),
        "chart_count": chart_count,
        "seam_corner_count": seam_corners,
        "normal_seam_count": len(analysis.normal_seams),
        "uv_seam_count": len(analysis.uv_seams),
        "degenerate_uv_faces": max(0, len(parsed.triangles) - sum(
            1 for face_index in range(len(parsed.triangles))
            if all(_corner_uv(parsed, face_index, c, parsed.triangles[face_index][c]) is not None for c in range(3))
        )),
    }


def _uv_key(uv: tuple[float, float], eps: float) -> tuple[int, int]:
    return (int(round(uv[0] / eps)), int(round(uv[1] / eps)))


def _accumulate_corner_tangents(
    positions: list[tuple[float, float, float]],
    triangle: tuple[int, int, int],
    uvs: list[tuple[float, float]],
    output: dict[int, list[float]],
) -> None:
    p0, p1, p2 = (positions[triangle[k]] for k in range(3))
    uv0, uv1, uv2 = uvs
    e1 = _sub(p1, p0)
    e2 = _sub(p2, p0)
    du1, dv1 = uv1[0] - uv0[0], uv1[1] - uv0[1]
    du2, dv2 = uv2[0] - uv0[0], uv2[1] - uv0[1]
    det = du1 * dv2 - du2 * dv1
    if abs(det) <= _EPS:
        for corner in range(3):
            output[corner] = [0.0] * 6
        return
    inv = 1.0 / det
    tangent = (
        (e1[0] * dv2 - e2[0] * dv1) * inv,
        (e1[1] * dv2 - e2[1] * dv1) * inv,
        (e1[2] * dv2 - e2[2] * dv1) * inv,
    )
    bitangent = (
        (e2[0] * du1 - e1[0] * du2) * inv,
        (e2[1] * du1 - e1[1] * du2) * inv,
        (e2[2] * du1 - e1[2] * du2) * inv,
    )
    values = [*tangent, *bitangent]
    for corner in range(3):
        output[corner] = list(values)


# ---------------------------------------------------------------------------
# LOD 过渡形变
# ---------------------------------------------------------------------------


def _clamp01(value: Any) -> float:
    if not _is_num(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _closest_point_on_triangle(
    point: tuple[float, float, float],
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> tuple[float, float, float]:
    """点到三角形的最近点（Ericson, Real-Time Collision Detection）。"""
    ab = _sub(b, a)
    ac = _sub(c, a)
    ap = _sub(point, a)
    d1 = _dot(ab, ap)
    d2 = _dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = _sub(point, b)
    d3 = _dot(ab, bp)
    d4 = _dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        denom = d1 - d3
        v = d1 / denom if abs(denom) > _EPS else 0.0
        return _add(a, _scale(ab, v))
    cp = _sub(point, c)
    d5 = _dot(ab, cp)
    d6 = _dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        denom = d2 - d6
        w = d2 / denom if abs(denom) > _EPS else 0.0
        return _add(a, _scale(ac, w))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        denom = (d4 - d3) + (d5 - d6)
        w = (d4 - d3) / denom if abs(denom) > _EPS else 0.0
        return _add(b, _scale(_sub(c, b), w))
    denom = va + vb + vc
    if abs(denom) <= _EPS:
        return _mid(a, b)
    v = vb / denom
    w = vc / denom
    return _add(a, _add(_scale(ab, v), _scale(ac, w)))


def _project_to_surface(
    point: tuple[float, float, float],
    positions: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
) -> tuple[float, float, float] | None:
    """把点投影到三角网格表面（最近三角形上的最近点）。"""
    best: tuple[float, float, float] | None = None
    best_distance = INF
    for triangle in triangles:
        a, b, c = (positions[triangle[k]] for k in range(3))
        candidate = _closest_point_on_triangle(point, a, b, c)
        distance = _dist(point, candidate)
        if distance < best_distance - 1e-12:
            best, best_distance = candidate, distance
    return best


# ---------------------------------------------------------------------------
# 公共入口：LOD 形变
# ---------------------------------------------------------------------------


def lod_morph(
    low: Any,
    high: Any = None,
    t: Any = 0.0,
    *,
    correspondence: Any = None,
    project: bool = True,
    clamp: bool = True,
) -> dict[str, Any]:
    """在两套 LOD 之间做连续形变，消除切换时的阴影/轮廓跳变。

    * ``t=0`` 保持低模，``t=1`` 贴合高模，中间线性插值；
    * 默认用「投影到高模表面」建立对应关系，也接受显式 ``correspondence``
      （低模顶点下标 -> 高模顶点下标，或逐顶点高模位置）；
    * 输出逐顶点位移量与最大误差，方便调用方按误差决定切换阈值。
    """
    if high is None and isinstance(low, Mapping):
        high = low.get("high")
        t = low.get("t", t)
        correspondence = low.get("correspondence", correspondence)
        low = low.get("low")
    low_mesh = _mesh_from(low)
    if not low_mesh.valid:
        return _invalid_result("lod_morph", positions=[], t=0.0)

    factor = _clamp01(t) if clamp else float(t) if _is_num(t) else 0.0
    low_positions = list(low_mesh.positions)
    if high is None:
        return {
            "valid": True,
            "kind": "lod_morph",
            "reason": "missing_high_lod",
            "t": factor,
            "positions": [list(p) for p in low_positions],
            "normals": [list(n) for n in low_mesh.normals] if low_mesh.normals else [],
            "displacements": [0.0] * len(low_positions),
            "max_displacement": 0.0,
            "low_vertex_count": len(low_positions),
            "high_vertex_count": 0,
            "hash": _hash_payload(low_positions),
        }

    high_mesh = _mesh_from(high)
    if not high_mesh.valid or not high_mesh.positions:
        return _invalid_result("lod_morph", reason="invalid_high_lod", positions=[])

    targets: list[tuple[float, float, float] | None] = [None] * len(low_positions)
    if correspondence is not None and _is_seq(correspondence):
        for index, target in enumerate(correspondence[: len(low_positions)]):
            if target is None:
                continue
            if _is_vec(target, 3):
                targets[index] = _vec(target)
                continue
            high_index = _as_index(target)
            if high_index is not None and 0 <= high_index < len(high_mesh.positions):
                targets[index] = high_mesh.positions[high_index]
    else:
        for index, position in enumerate(low_positions):
            if project and high_mesh.triangles:
                targets[index] = _project_to_surface(
                    position, high_mesh.positions, high_mesh.triangles
                )
            else:
                best = None
                best_distance = INF
                for high_index, high_position in enumerate(high_mesh.positions):
                    distance = _dist(position, high_position)
                    if distance < best_distance - 1e-12:
                        best, best_distance = high_position, distance
                targets[index] = best

    morphed: list[tuple[float, float, float]] = []
    displacements: list[float] = []
    matched = 0
    for position, target in zip(low_positions, targets):
        if target is None:
            morphed.append(position)
            displacements.append(0.0)
            continue
        matched += 1
        delta = _sub(target, position)
        morphed.append(_add(position, _scale(delta, factor)))
        displacements.append(_length(delta))

    normals: list[list[float]] = []
    if low_mesh.normals and high_mesh.normals and len(low_mesh.normals) == len(low_positions):
        for index, low_normal in enumerate(low_mesh.normals):
            target_normal = None
            if correspondence is None and targets[index] is not None:
                target_normal = None
            elif correspondence is not None and _is_seq(correspondence):
                high_index = _as_index(correspondence[index]) if index < len(correspondence) else None
                if high_index is not None and 0 <= high_index < len(high_mesh.normals):
                    target_normal = high_mesh.normals[high_index]
            blended = (
                _add(_scale(low_normal, 1.0 - factor), _scale(target_normal, factor))
                if target_normal is not None
                else low_normal
            )
            normals.append(list(_unit(blended) or _unit(low_normal) or (0.0, 0.0, 1.0)))

    result = {
        "valid": True,
        "kind": "lod_morph",
        "t": factor,
        "positions": [list(p) for p in morphed],
        "normals": normals,
        "displacements": displacements,
        "max_displacement": max(displacements) if displacements else 0.0,
        "mean_displacement": (
            sum(displacements) / len(displacements) if displacements else 0.0
        ),
        "matched_vertices": matched,
        "low_vertex_count": len(low_positions),
        "low_triangle_count": len(low_mesh.triangles),
        "high_vertex_count": len(high_mesh.positions),
        "high_triangle_count": len(high_mesh.triangles),
        "vertex_count_delta": len(high_mesh.positions) - len(low_positions),
    }
    result["hash"] = _hash_payload({"t": factor, "positions": result["positions"]})
    return result


# ---------------------------------------------------------------------------
# 公共入口：异步结果版本校验
# ---------------------------------------------------------------------------


def _token_value(value: Any) -> Any:
    if value is None:
        return None
    if _is_num(value):
        return float(value)
    return str(value)


def _version_value(value: Any) -> Any:
    if value is None:
        return None
    if _is_num(value):
        return float(value)
    return str(value)


def async_version(
    request: Any = None,
    result: Any = None,
    *,
    current_version: Any = None,
    token: Any = None,
    source_hash: Any = None,
) -> dict[str, Any]:
    """校验异步简化结果是否对应当前请求，拒绝过期/乱序/串号结果。

    校验顺序（任一失败即丢弃）：

    1. 结果存在且结构可识别；
    2. 请求令牌与结果令牌一致（防止复用同一结果给别的请求）；
    3. 源网格哈希一致（防止资产热重载后拿旧结果覆盖新模型）；
    4. 结果版本等于当前版本；小于当前版本标记 ``stale``，
       大于当前版本标记 ``future``（通常是乱序到达，必须丢弃）；
    5. 结果版本不早于请求版本。

    返回 ``accepted`` / ``stale`` / ``discard`` 与具体原因，调用方据此
    决定是否应用结果。
    """
    if result is None and isinstance(request, Mapping) and (
        "result" in request or "request" in request
    ):
        container = request
        request = container.get("request", container)
        result = container.get("result")
        if current_version is None:
            current_version = container.get("current_version")
        if token is None:
            token = container.get("token")
        if source_hash is None:
            source_hash = container.get("source_hash")

    if result is None:
        return {
            "valid": False,
            "kind": "async_version",
            "accepted": False,
            "stale": False,
            "discard": True,
            "reason": "missing_result",
        }

    if not (isinstance(result, Mapping) or hasattr(result, "version")):
        return {
            "valid": False,
            "kind": "async_version",
            "accepted": False,
            "stale": False,
            "discard": True,
            "reason": "unrecognized_result",
        }

    req_version = _version_value(
        _get(request, "version", _get(request, "generation", None))
    )
    res_version = _version_value(
        _get(result, "version", _get(result, "generation", None))
    )
    current = _version_value(
        current_version
        if current_version is not None
        else _get(result, "current_version", None)
    )
    req_token = _token_value(
        token
        if token is not None
        else _get(request, "token", _get(request, "request_id", _get(request, "id", None)))
    )
    res_token = _token_value(
        _get(result, "token", _get(result, "request_id", _get(result, "id", None)))
    )
    req_source = source_hash if source_hash is not None else _get(request, "source_hash")
    res_source = _get(result, "source_hash", _get(result, "source", None))

    report: dict[str, Any] = {
        "valid": True,
        "kind": "async_version",
        "accepted": False,
        "stale": False,
        "future": False,
        "discard": True,
        "reason": None,
        "version": res_version,
        "request_version": req_version,
        "current_version": current,
        "token": res_token,
        "source_hash": res_source,
        "applied": bool(_get(result, "applied", False)),
    }

    if req_token is not None and res_token is not None and req_token != res_token:
        report["reason"] = "token_mismatch"
        return report
    if req_source is not None and res_source is not None and req_source != res_source:
        report["reason"] = "source_hash_mismatch"
        return report

    if res_version is None:
        report["reason"] = "missing_version"
        return report

    if current is not None:
        if _version_key(res_version) < _version_key(current):
            report["stale"] = True
            report["reason"] = "stale"
            return report
        if _version_key(res_version) > _version_key(current):
            report["future"] = True
            report["reason"] = "future"
            return report

    if req_version is not None and _version_key(res_version) < _version_key(req_version):
        report["stale"] = True
        report["reason"] = "out_of_order"
        return report

    report["accepted"] = True
    report["discard"] = False
    report["reason"] = "accepted"
    mesh = _get(result, "mesh", None)
    if mesh is not None:
        report["result_hash"] = _mesh_hash(mesh)
    return report


def _version_key(value: Any) -> tuple[int, Any]:
    if _is_num(value):
        return (0, float(value))
    return (1, str(value))


# ---------------------------------------------------------------------------
# 公共入口：包围体
# ---------------------------------------------------------------------------


def _ritter_sphere(
    points: list[tuple[float, float, float]],
) -> tuple[tuple[float, float, float], float]:
    if not points:
        return (0.0, 0.0, 0.0), 0.0
    ordered = sorted(points)
    first = ordered[0]
    far = max(ordered, key=lambda p: (_dist(p, first), p))
    second = max(ordered, key=lambda p: (_dist(p, far), p))
    center = _mid(far, second)
    radius = _dist(far, second) * 0.5
    for point in ordered:
        distance = _dist(point, center)
        if distance > radius + 1e-12:
            new_radius = (radius + distance) * 0.5
            if distance > _EPS:
                shift = (distance - new_radius) / distance
                center = _add(center, _scale(_sub(point, center), shift))
            radius = new_radius
    return center, radius  # type: ignore[return-value]


def bounds_update(
    mesh: Any,
    positions: Any = None,
    *,
    padding: Any = 0.0,
) -> dict[str, Any]:
    """重算 AABB 与包围球。

    简化 / 形变后必须重算，否则阴影级联与视锥剔除会用到旧包围体，
    表现为阴影边缘跳动或物体被错误剔除。
    """
    parsed = _mesh_from(mesh)
    source: list[tuple[float, float, float]]
    if positions is not None:
        if not (_is_seq(positions) and all(_is_vec(p, 3) for p in positions)):
            return _invalid_result("bounds_update", reason="invalid_positions")
        source = [_vec(p) for p in positions]
    elif parsed.valid:
        source = list(parsed.positions)
    else:
        return _invalid_result("bounds_update", reason="invalid_mesh")

    pad = float(padding) if _is_num(padding) and padding > 0 else 0.0
    if not source:
        empty = {
            "valid": True,
            "kind": "bounds_update",
            "empty": True,
            "min": [0.0, 0.0, 0.0],
            "max": [0.0, 0.0, 0.0],
            "center": [0.0, 0.0, 0.0],
            "size": [0.0, 0.0, 0.0],
            "radius": 0.0,
            "sphere_center": [0.0, 0.0, 0.0],
            "sphere_radius": 0.0,
            "vertex_count": 0,
        }
        empty["hash"] = _hash_payload(empty)
        return empty

    min_corner = tuple(min(p[axis] for p in source) for axis in range(3))
    max_corner = tuple(max(p[axis] for p in source) for axis in range(3))
    min_corner = tuple(c - pad for c in min_corner)
    max_corner = tuple(c + pad for c in max_corner)
    center = _mid(min_corner, max_corner)
    size = _sub(max_corner, min_corner)
    sphere_center, sphere_radius = _ritter_sphere(source)
    if pad:
        sphere_radius += pad

    report: dict[str, Any] = {
        "valid": True,
        "kind": "bounds_update",
        "empty": False,
        "min": list(min_corner),
        "max": list(max_corner),
        "center": list(center),
        "size": list(size),
        "radius": _length(size) * 0.5,
        "sphere_center": list(sphere_center),
        "sphere_radius": sphere_radius,
        "vertex_count": len(source),
        "padding": pad,
    }
    report["hash"] = _hash_payload(
        {"min": report["min"], "max": report["max"], "sphere_radius": sphere_radius}
    )
    return report


# ---------------------------------------------------------------------------
# 公共入口：校验 / 审计 / 降级
# ---------------------------------------------------------------------------


def _mesh_hash(mesh: Any) -> str:
    """与拓扑顺序无关的网格哈希：焊接位置 + 规范面集合。"""
    parsed = mesh if isinstance(mesh, _Mesh) else _mesh_from(mesh)
    if not parsed.valid:
        return _hash_payload({"valid": False})
    group_of, group_positions, _group_vertices = _weld(parsed)
    faces = _canonical_face_set(parsed, group_of)
    return _hash_payload(
        {
            "positions": [[round(c, 7) for c in p] for p in group_positions],
            "faces": [list(face) for face in faces],
        }
    )


def validation(
    mesh: Any,
    *,
    weld_eps: float = _DEFAULT_WELD_EPS,
    uv_eps: float = _DEFAULT_UV_EPS,
    normal_angle: float = _DEFAULT_NORMAL_ANGLE,
    max_degenerate_ratio: float = 0.0,
) -> dict[str, Any]:
    """结构合法性与拓扑质量检查。

    检查项：索引越界、退化面、重复面、孤立顶点、非流形边、接缝一致性。
    ``max_degenerate_ratio`` 允许一定比例的退化面（默认 0 = 不允许）。
    """
    parsed = _mesh_from(mesh)
    if not parsed.valid:
        return {
            "valid": False,
            "kind": "validation",
            "ok": False,
            "errors": ["invalid_mesh"],
            "warnings": [],
            "issues": [],
        }

    errors: list[str] = []
    warnings: list[str] = []
    issues: list[dict[str, Any]] = []

    for issue in parsed.issues:
        warnings.append(str(issue.get("kind", "issue")))
        issues.append(issue)

    vertex_count = len(parsed.positions)
    if vertex_count == 0:
        errors.append("no_vertices")
    if not parsed.triangles:
        errors.append("no_triangles")

    seen: dict[tuple[int, int, int], int] = {}
    degenerate: list[int] = []
    duplicates: list[list[int]] = []
    used: set[int] = set()
    for face_index, triangle in enumerate(parsed.triangles):
        if len(set(triangle)) < 3:
            degenerate.append(face_index)
            continue
        a, b, c = triangle
        if not all(0 <= v < vertex_count for v in triangle):
            errors.append(f"index_out_of_range:{face_index}")
            continue
        for edge in ((a, b), (b, c), (c, a)):
            if edge[0] == edge[1]:
                degenerate.append(face_index)
                break
        normal, area = _face_normal_area(parsed.positions, triangle)
        if normal is None or area <= _EPS:
            degenerate.append(face_index)
            continue
        # 重复面按「顶点集合」判重：同三个点不论绕序/起点都算重复。
        key = tuple(sorted(triangle))
        if key in seen:
            duplicates.append([seen[key], face_index])
        else:
            seen[key] = face_index
        used.update(triangle)

    isolated = [index for index in range(vertex_count) if index not in used]

    analysis = _analyze(
        parsed,
        weld_eps=weld_eps,
        uv_eps=uv_eps,
        normal_angle=normal_angle,
    )

    ratio = len(degenerate) / len(parsed.triangles) if parsed.triangles else 0.0
    if degenerate and ratio > max_degenerate_ratio:
        errors.append(f"degenerate_faces:{len(degenerate)}")
    elif degenerate:
        warnings.append(f"degenerate_faces:{len(degenerate)}")
    if duplicates:
        errors.append(f"duplicate_faces:{len(duplicates)}")
    if isolated:
        warnings.append(f"isolated_vertices:{len(isolated)}")
    if analysis.nonmanifold:
        errors.append(f"nonmanifold_edges:{len(analysis.nonmanifold)}")
    if parsed.corner_uvs is None and parsed.uvs is not None:
        with_uvs = sum(
            1
            for triangle in parsed.triangles
            if all(0 <= v < len(parsed.uvs) for v in triangle)  # type: ignore[arg-type]
        )
        if with_uvs != len(parsed.triangles):
            warnings.append("uv_coverage_gap")

    result = {
        "valid": True,
        "kind": "validation",
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "issues": issues,
        "vertex_count": vertex_count,
        "triangle_count": len(parsed.triangles),
        "degenerate_faces": degenerate,
        "duplicate_faces": duplicates,
        "isolated_vertices": isolated,
        "nonmanifold_edges": [list(edge) for edge in analysis.nonmanifold],
        "boundary_edges": len(analysis.boundary),
        "uv_seams": len(analysis.uv_seams),
        "normal_seams": len(analysis.normal_seams),
        "degenerate_ratio": ratio,
        "hash": _mesh_hash(parsed),
    }
    return result


def audit(
    mesh: Any,
    *,
    high: Any = None,
    target_faces: int | None = None,
    ratio: float | None = None,
    version: Any = None,
    current_version: Any = None,
    weld_eps: float = _DEFAULT_WELD_EPS,
    uv_eps: float = _DEFAULT_UV_EPS,
    normal_angle: float = _DEFAULT_NORMAL_ANGLE,
) -> dict[str, Any]:
    """汇总接缝、边权、切线、确定性、包围体与（可选）异步版本检查。"""
    parsed = _mesh_from(mesh)
    if not parsed.valid:
        return {
            "valid": False,
            "kind": "audit",
            "ok": False,
            "checks": {},
            "failures": ["invalid_mesh"],
        }

    seam = seam_lock(
        parsed, weld_eps=weld_eps, uv_eps=uv_eps, normal_angle=normal_angle
    )
    costs = edge_cost(
        parsed, weld_eps=weld_eps, uv_eps=uv_eps, normal_angle=normal_angle
    )
    order = collapse_order(
        parsed,
        target_faces,
        ratio=ratio,
        weld_eps=weld_eps,
        uv_eps=uv_eps,
        normal_angle=normal_angle,
    )
    tangents = tangent_rebuild(
        parsed, weld_eps=weld_eps, uv_eps=uv_eps, normal_angle=normal_angle
    )
    validity = validation(
        parsed, weld_eps=weld_eps, uv_eps=uv_eps, normal_angle=normal_angle
    )
    bounds = bounds_update(parsed)
    normal_report = normal_seam(
        parsed, weld_eps=weld_eps, normal_angle=normal_angle
    )
    dedupe = import_dedupe(parsed, weld_eps=weld_eps)

    checks: dict[str, bool] = {
        "structurally_valid": bool(validity["ok"]),
        "seams_locked": seam["locked_edge_count"] > 0 or not seam["has_seams"],
        "edge_costs_finite": costs["collapsible_count"] > 0,
        "tangents_consistent": tangents.get("valid", False),
        "collapse_deterministic": order.get("deterministic", False),
        "bounds_finite": all(math.isfinite(c) for c in bounds.get("min", []))
        and math.isfinite(bounds.get("sphere_radius", INF)),
        "normals_consistent": normal_report.get("valid", False),
        "canonical_hash": bool(dedupe.get("hash")),
    }

    if high is not None:
        morph = lod_morph(parsed, high)
        checks["lod_morph_available"] = morph.get("valid", False)
    else:
        morph = None

    if version is not None:
        version_report = async_version(
            {"version": version}, {"version": version}, current_version=current_version
        )
        checks["version_consistent"] = bool(version_report["accepted"])
    else:
        version_report = None

    failures = sorted(name for name, ok in checks.items() if not ok)
    result: dict[str, Any] = {
        "valid": True,
        "kind": "audit",
        "ok": not failures,
        "checks": checks,
        "failures": failures,
        "seam_lock": seam,
        "edge_cost": {
            "edge_count": costs["edge_count"],
            "collapsible_count": costs["collapsible_count"],
            "locked_count": costs["locked_count"],
            "min_cost": costs["min_cost"],
        },
        "collapse_order": {
            "order_hash": order["order_hash"],
            "steps": order["steps"],
            "locked_count": order["locked_count"],
            "deterministic": order["deterministic"],
        },
        "tangent_rebuild": {
            "chart_count": tangents["chart_count"],
            "normal_seam_count": tangents["normal_seam_count"],
            "uv_seam_count": tangents["uv_seam_count"],
        },
        "validation": {
            "ok": validity["ok"],
            "errors": validity["errors"],
            "warnings": validity["warnings"],
        },
        "bounds": bounds,
        "normal_seam": {
            "normal_seam_count": normal_report["normal_seam_count"],
            "groups": normal_report["group_count"],
        },
        "import_dedupe": dedupe,
    }
    if morph is not None:
        result["lod_morph"] = {
            "t": morph["t"],
            "max_displacement": morph.get("max_displacement", 0.0),
            "matched_vertices": morph.get("matched_vertices", 0),
        }
    if version_report is not None:
        result["async_version"] = {
            "accepted": version_report["accepted"],
            "reason": version_report["reason"],
        }
    result["hash"] = _hash_payload(
        {
            "order": order["order_hash"],
            "mesh": validity["hash"],
            "bounds": bounds["hash"],
            "checks": checks,
        }
    )
    return result


def fallback(
    mesh: Any = None,
    error: Any = None,
    *,
    weld_eps: float = _DEFAULT_WELD_EPS,
) -> dict[str, Any]:
    """减面失败时的安全网格：**原样**返回未减面数据。

    这里刻意不做位置焊接：按位置焊接会把 UV 拆点的副本合并，接缝随之
    消失，等于把「没坍缩」也弄坏了。只丢弃几何上真正退化的面。
    """
    if mesh is None and isinstance(error, (Mapping, list, tuple)):
        mesh, error = error, None
    parsed = _mesh_from(mesh)
    reason = str(error) if error is not None else "unspecified"
    if not parsed.valid:
        return {
            "valid": False,
            "kind": "fallback",
            "reason": reason,
            "degraded": True,
            "positions": [],
            "triangles": [],
            "used_original": True,
        }

    _ = weld_eps  # 保留形参以兼容调用方；原样返回时不需要焊接
    kept: list[tuple[int, int, int]] = []
    dropped: list[int] = []
    for index, triangle in enumerate(parsed.triangles):
        if len(set(triangle)) < 3:
            dropped.append(index)
            continue
        normal, area = _face_normal_area(parsed.positions, triangle)
        if normal is None or area <= _EPS:
            dropped.append(index)
            continue
        kept.append(triangle)

    used = sorted({vertex for triangle in kept for vertex in triangle})
    remap = {vertex: index for index, vertex in enumerate(used)}
    positions = [list(parsed.positions[vertex]) for vertex in used]
    result: dict[str, Any] = {
        "valid": True,
        "kind": "fallback",
        "reason": reason,
        "degraded": True,
        "used_original": True,
        "positions": positions,
        "triangles": [[remap[vertex] for vertex in triangle] for triangle in kept],
        "uvs": (
            [list(parsed.uvs[vertex]) for vertex in used]
            if parsed.uvs is not None
            else []
        ),
        "normals": (
            [list(parsed.normals[vertex]) for vertex in used]
            if parsed.normals is not None
            else []
        ),
        "original_vertex_count": len(parsed.positions),
        "vertex_count": len(positions),
        "triangle_count": len(kept),
        "dropped_degenerate": dropped,
    }
    result["hash"] = _hash_payload(
        {"positions": result["positions"], "triangles": result["triangles"]}
    )
    return result


# ---------------------------------------------------------------------------
# 公共入口：UV 保护 / 统计 / 法线接缝 / 去重 / 确定性
# ---------------------------------------------------------------------------


def _uv_triangle_area(uvs: list[tuple[float, float]]) -> float:
    (u0, v0), (u1, v1), (u2, v2) = uvs
    return 0.5 * ((u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0))


def uv_guard(
    mesh: Any,
    *,
    weld_eps: float = _DEFAULT_WELD_EPS,
    uv_eps: float = _DEFAULT_UV_EPS,
    period: float = 1.0,
) -> dict[str, Any]:
    """UV 翻转 / 重叠 / 退化保护。

    * ``flipped_faces``：UV 绕序与几何绕序相反的面（会导致法线贴图反向）；
    * ``degenerate_uv_faces``：UV 面积接近 0 的面（切线空间无定义）；
    * ``overlapping_faces``：同一 UV 图内 UV 三角形包围盒重叠的面对（启发式）；
    * ``udim_crossings``：跨 UV 周期（默认 1.0）的边，需要先 unwrap。
    """
    parsed = _mesh_from(mesh)
    if not parsed.valid:
        return _invalid_result("uv_guard", ok=False, flipped_faces=[])

    if parsed.corner_uvs is None and parsed.uvs is None:
        return _invalid_result(
            "uv_guard",
            ok=False,
            reason="missing_uvs",
            flipped_faces=[],
            degenerate_uv_faces=[],
            overlapping_faces=[],
            udim_crossings=[],
        )

    flipped: list[int] = []
    degenerate: list[int] = []
    crossings: list[list[int]] = []
    boxes: dict[str, list[tuple[float, float, float, float, int]]] = {}
    charts = 0

    for face_index, triangle in enumerate(parsed.triangles):
        uvs: list[tuple[float, float]] = []
        ok = True
        for corner in range(3):
            uv = _corner_uv(parsed, face_index, corner, triangle[corner])
            if uv is None:
                ok = False
                break
            uvs.append(uv)
        if not ok:
            degenerate.append(face_index)
            continue
        area = _uv_triangle_area(uvs)
        if abs(area) <= uv_eps * uv_eps:
            degenerate.append(face_index)
            continue

        normal, geo_area = _face_normal_area(parsed.positions, triangle)
        if normal is not None and geo_area > _EPS:
            # 几何法线与 UV 朝向的符号一致性：用三角形绕序 × UV 面积判断。
            u0, u1, u2 = uvs
            duv = (u1[0] - u0[0]) * (u2[1] - u0[1]) - (u2[0] - u0[0]) * (u1[1] - u0[1])
            if duv < 0.0:
                flipped.append(face_index)

        for corner in range(3):
            a = uvs[corner]
            b = uvs[(corner + 1) % 3]
            if abs(a[0] - b[0]) > period or abs(a[1] - b[1]) > period:
                crossings.append([face_index, corner])

        chart = f"{int(uvs[0][0] // period)}:{int(uvs[0][1] // period)}"
        charts = max(charts, 1)
        box = (
            min(uv[0] for uv in uvs),
            min(uv[1] for uv in uvs),
            max(uv[0] for uv in uvs),
            max(uv[1] for uv in uvs),
            face_index,
        )
        boxes.setdefault(chart, []).append(box)

    overlapping: list[list[int]] = []
    for chart_boxes in boxes.values():
        ordered = sorted(chart_boxes)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a_box, b_box = ordered[i], ordered[j]
                if (
                    a_box[0] < b_box[2] - uv_eps
                    and b_box[0] < a_box[2] - uv_eps
                    and a_box[1] < b_box[3] - uv_eps
                    and b_box[1] < a_box[3] - uv_eps
                ):
                    overlapping.append([a_box[4], b_box[4]])
        if len(overlapping) > 256:
            break

    ok = not flipped and not degenerate
    result = {
        "valid": True,
        "kind": "uv_guard",
        "ok": ok,
        "flipped_faces": flipped,
        "flipped_count": len(flipped),
        "degenerate_uv_faces": degenerate,
        "degenerate_uv_count": len(degenerate),
        "overlapping_faces": overlapping[:256],
        "overlapping_count": len(overlapping),
        "udim_crossings": crossings,
        "udim_crossing_count": len(crossings),
        "chart_buckets": len(boxes),
        "triangle_count": len(parsed.triangles),
    }
    result["hash"] = _hash_payload(
        {"flipped": flipped, "degenerate": degenerate, "crossings": crossings}
    )
    return result


def metrics(mesh: Any, *, weld_eps: float = _DEFAULT_WELD_EPS) -> dict[str, Any]:
    """网格统计：顶点/面/边、接缝与边界、欧拉特征、面积、包围体。"""
    parsed = _mesh_from(mesh)
    if not parsed.valid:
        return _invalid_result(
            "metrics",
            vertex_count=0,
            triangle_count=0,
            edge_count=0,
            locked_edge_count=0,
        )

    analysis = _analyze(parsed, weld_eps=weld_eps)
    area = 0.0
    perimeter = 0.0
    for triangle in parsed.triangles:
        _normal, face_area = _face_normal_area(parsed.positions, triangle)
        area += face_area
        p0, p1, p2 = (parsed.positions[triangle[k]] for k in range(3))
        perimeter += _dist(p0, p1) + _dist(p1, p2) + _dist(p2, p0)

    edge_count = len(analysis.occurrences)
    group_count = analysis.group_count
    face_count = len(parsed.triangles)
    euler = group_count - edge_count + face_count

    bounds = bounds_update(parsed)
    result: dict[str, Any] = {
        "valid": True,
        "kind": "metrics",
        "vertex_count": len(parsed.positions),
        "welded_vertex_count": group_count,
        "triangle_count": face_count,
        "edge_count": edge_count,
        "boundary_edge_count": len(analysis.boundary),
        "nonmanifold_edge_count": len(analysis.nonmanifold),
        "locked_edge_count": len(analysis.locked),
        "uv_seam_count": len(analysis.uv_seams),
        "normal_seam_count": len(analysis.normal_seams),
        "hard_edge_count": len(analysis.hard_edges),
        "protected_vertex_count": sum(1 for flag in analysis.protected if flag),
        "euler_characteristic": euler,
        "surface_area": area,
        "perimeter": perimeter,
        "bounds": {
            "min": bounds["min"],
            "max": bounds["max"],
            "radius": bounds["radius"],
            "sphere_radius": bounds["sphere_radius"],
        },
        "has_uvs": parsed.uvs is not None or parsed.corner_uvs is not None,
        "has_normals": parsed.normals is not None or parsed.corner_normals is not None,
        "degenerate_ratio": 0.0,
    }
    result["hash"] = _hash_payload(
        {
            "v": result["vertex_count"],
            "f": result["triangle_count"],
            "e": result["edge_count"],
            "area": round(area, 7),
        }
    )
    return result


def _vertex_normals(parsed: _Mesh, analysis: _Analysis) -> list[tuple[float, float, float]]:
    """逐顶点法线：优先显式逐顶点法线，否则按面积累加相邻面法线。"""
    if parsed.normals is not None:
        return list(parsed.normals)
    if parsed.corner_normals is not None:
        accumulator = [[0.0, 0.0, 0.0] for _ in parsed.positions]
        for face_index, triangle in enumerate(parsed.triangles):
            _normal, area = _face_normal_area(parsed.positions, triangle)
            for corner in range(3):
                corner_normal = parsed.corner_normals[3 * face_index + corner]
                target = accumulator[triangle[corner]]
                for axis in range(3):
                    target[axis] += corner_normal[axis] * max(area, 1e-9)
        return [
            _unit(tuple(value)) or (0.0, 0.0, 1.0)  # type: ignore[misc]
            for value in accumulator
        ]
    accumulator = [[0.0, 0.0, 0.0] for _ in parsed.positions]
    for triangle in parsed.triangles:
        normal, area = _face_normal_area(parsed.positions, triangle)
        if normal is None:
            continue
        for vertex in triangle:
            target = accumulator[vertex]
            for axis in range(3):
                target[axis] += normal[axis] * area
    return [
        _unit(tuple(value)) or (0.0, 0.0, 1.0)  # type: ignore[misc]
        for value in accumulator
    ]


def normal_seam(
    mesh: Any,
    *,
    weld_eps: float = _DEFAULT_WELD_EPS,
    normal_angle: float = _DEFAULT_NORMAL_ANGLE,
    smooth: bool = True,
) -> dict[str, Any]:
    """法线接缝检测与按组平滑。

    * 检测：同一焊接位置的角法线夹角超过阈值 -> 该边是法线接缝；
    * 平滑：只在不跨法线接缝的连通组内做面积加权平均，接缝两侧保持
      各自方向，避免把折痕抹平或让接缝两侧互相拖拽。
    """
    parsed = _mesh_from(mesh)
    if not parsed.valid:
        return _invalid_result(
            "normal_seam", normals=[], normal_seam_count=0, group_count=0
        )

    analysis = _analyze(parsed, weld_eps=weld_eps, normal_angle=normal_angle)
    source_normals = _vertex_normals(parsed, analysis)
    has_explicit_normals = (
        parsed.normals is not None or parsed.corner_normals is not None
    )
    seam_edges = [list(edge) for edge in analysis.normal_seams]

    if not smooth:
        normals = [list(normal) for normal in source_normals]
        return {
            "valid": True,
            "kind": "normal_seam",
            "normals": normals,
            "normal_seam_count": len(analysis.normal_seams),
            "normal_seams": seam_edges,
            "group_count": analysis.group_count,
            "smooth_group_count": analysis.group_count,
            "smoothed": False,
            "explicit_normals": has_explicit_normals,
            "hash": _hash_payload(normals),
        }

    # 只在不跨法线接缝的组之间合并；接缝两侧形成不同的平滑组。
    dsu = _DSU(analysis.group_count)
    for edge in sorted(analysis.occurrences):
        if edge in analysis.normal_seams:
            continue
        dsu.union(edge[0], edge[1])

    accum: dict[int, list[float]] = {}
    for triangle in parsed.triangles:
        _normal, area = _face_normal_area(parsed.positions, triangle)
        weight = max(area, 1e-9)
        for vertex in triangle:
            root = dsu.find(analysis.group_of[vertex])
            entry = accum.setdefault(root, [0.0, 0.0, 0.0])
            normal = source_normals[vertex]
            for axis in range(3):
                entry[axis] += normal[axis] * weight

    smoothed: list[list[float]] = []
    for vertex in range(len(parsed.positions)):
        root = dsu.find(analysis.group_of[vertex])
        averaged = _unit(tuple(accum.get(root, (0.0, 0.0, 0.0))))
        if averaged is None:
            averaged = _unit(source_normals[vertex]) or (0.0, 0.0, 1.0)
        smoothed.append(list(averaged))

    result = {
        "valid": True,
        "kind": "normal_seam",
        "normals": smoothed,
        "normal_seam_count": len(analysis.normal_seams),
        "normal_seams": seam_edges,
        "group_count": analysis.group_count,
        "smooth_group_count": len({dsu.find(i) for i in range(analysis.group_count)}),
        "smoothed": True,
        "explicit_normals": has_explicit_normals,
        "original": [list(normal) for normal in source_normals],
    }
    result["hash"] = _hash_payload(smoothed)
    return result


def import_dedupe(
    mesh: Any = None,
    *,
    weld_eps: float = _DEFAULT_WELD_EPS,
    uv_eps: float = _DEFAULT_UV_EPS,
    assets: Any = None,
) -> dict[str, Any]:
    """与导入顺序无关的规范键，用于重复导入去重。

    键基于「焊接后的位置集合 + 规范面集合」，与顶点/三角形的输入顺序、
    UV 拆点方式无关，因此同一模型多次导入会得到同一个 ``canonical_hash``。
    """
    if assets is not None or (mesh is not None and _is_seq(mesh) and not isinstance(mesh, Mapping) and mesh and isinstance(mesh[0], (Mapping,))):
        items = assets if assets is not None else mesh
        parsed_items: list[Any] = []
        for item in items if _is_seq(items) else []:
            parsed = _mesh_from(item)
            if parsed.valid:
                parsed_items.append(parsed)
        if not parsed_items:
            return _invalid_result("import_dedupe", unique=0, duplicates=0)
        seen: dict[str, int] = {}
        groups: list[dict[str, Any]] = []
        duplicates: list[list[int]] = []
        for index, parsed in enumerate(parsed_items):
            hashed = _mesh_hash(parsed)
            if hashed in seen:
                duplicates.append([seen[hashed], index])
            else:
                seen[hashed] = index
                groups.append(
                    {
                        "index": index,
                        "hash": hashed,
                        "vertex_count": len(parsed.positions),
                        "triangle_count": len(parsed.triangles),
                    }
                )
        return {
            "valid": True,
            "kind": "import_dedupe",
            "unique": len(groups),
            "duplicates": duplicates,
            "duplicate_count": len(duplicates),
            "groups": groups,
            "hash": _hash_payload([group["hash"] for group in groups]),
        }

    parsed = _mesh_from(mesh)
    if not parsed.valid:
        return _invalid_result("import_dedupe", unique=0, duplicates=[])
    group_of, group_positions, group_vertices = _weld(parsed, weld_eps)
    faces = _canonical_face_set(parsed, group_of)
    canonical_hash = _mesh_hash(parsed)
    return {
        "valid": True,
        "kind": "import_dedupe",
        "hash": canonical_hash,
        "canonical_hash": canonical_hash,
        "weld_eps": weld_eps,
        "uv_eps": uv_eps,
        "welded_vertex_count": len(group_positions),
        "canonical_face_count": len(faces),
        "vertex_count": len(parsed.positions),
        "triangle_count": len(parsed.triangles),
        "source_vertices": group_vertices,
    }


def _shuffled_mesh(parsed: _Mesh, seed: int) -> dict[str, Any]:
    """用固定种子重排顶点、三角形与每个三角形的起始角。

    逐角属性（UV / 法线）跟着角一起旋转、一起重排，接缝信息保持不变 ——
    这样 ``determinism`` 检验的才是「顺序无关」，而不是「属性被丢掉后碰巧一致」。
    """
    rng = random.Random(seed)
    vertex_order = list(range(len(parsed.positions)))
    rng.shuffle(vertex_order)
    remap = [0] * len(vertex_order)
    for new_index, old_index in enumerate(vertex_order):
        remap[old_index] = new_index

    face_order = list(range(len(parsed.triangles)))
    rng.shuffle(face_order)

    positions = [parsed.positions[old] for old in vertex_order]
    uvs = [parsed.uvs[old] for old in vertex_order] if parsed.uvs is not None else None
    normals = (
        [parsed.normals[old] for old in vertex_order]
        if parsed.normals is not None
        else None
    )

    triangles: list[tuple[int, int, int]] = []
    corner_uvs: list[tuple[float, float]] | None = (
        [] if parsed.corner_uvs is not None else None
    )
    corner_normals: list[tuple[float, float, float]] | None = (
        [] if parsed.corner_normals is not None else None
    )
    for old_face in face_order:
        triangle = parsed.triangles[old_face]
        rotate = rng.randrange(3)
        order = [(rotate + k) % 3 for k in range(3)]
        triangles.append(tuple(remap[triangle[c]] for c in order))
        if corner_uvs is not None:
            corner_uvs.extend(parsed.corner_uvs[3 * old_face + c] for c in order)  # type: ignore[index]
        if corner_normals is not None:
            corner_normals.extend(  # type: ignore[union-attr]
                parsed.corner_normals[3 * old_face + c] for c in order  # type: ignore[index]
            )

    result: dict[str, Any] = {"positions": positions, "triangles": triangles}
    if uvs is not None:
        result["uvs"] = uvs
    if normals is not None:
        result["normals"] = normals
    if corner_uvs is not None:
        result["corner_uvs"] = corner_uvs
    if corner_normals is not None:
        result["corner_normals"] = corner_normals
    return result


def determinism(
    mesh: Any,
    *,
    target_faces: int | None = None,
    ratio: float | None = None,
    seeds: Any = (1, 7, 13),
    weld_eps: float = _DEFAULT_WELD_EPS,
    uv_eps: float = _DEFAULT_UV_EPS,
    normal_angle: float = _DEFAULT_NORMAL_ANGLE,
) -> dict[str, Any]:
    """用同一模型的不同顶点/三角形顺序验证减面确定性。

    对每个种子：重排输入 -> 规范化哈希 -> 坍缩顺序哈希，
    全部一致才认为「重复导入同一模型得到相同拓扑」。
    """
    parsed = _mesh_from(mesh)
    if not parsed.valid:
        return _invalid_result("determinism", consistent=False, samples=[])

    if not _is_seq(seeds):
        seeds = (1, 7, 13)
    seed_list = [int(seed) for seed in seeds if _is_num(seed)] or [1]

    baseline_hash = _mesh_hash(parsed)
    baseline = collapse_order(
        parsed,
        target_faces,
        ratio=ratio,
        weld_eps=weld_eps,
        uv_eps=uv_eps,
        normal_angle=normal_angle,
    )
    samples: list[dict[str, Any]] = []

    variants: list[tuple[str, Any]] = [("baseline", parsed)]
    for seed in seed_list:
        variants.append((f"shuffle:{seed}", _shuffled_mesh(parsed, seed)))

    for name, variant in variants:
        variant_hash = _mesh_hash(variant)
        order = collapse_order(
            variant,
            target_faces,
            ratio=ratio,
            weld_eps=weld_eps,
            uv_eps=uv_eps,
            normal_angle=normal_angle,
        )
        sample = {
            "name": name,
            "hash": variant_hash,
            "order_hash": order["order_hash"],
            "steps": order["steps"],
            "matches_baseline": variant_hash == baseline_hash
            and order["order_hash"] == baseline["order_hash"],
        }
        if "result_hash" in order:
            sample["result_hash"] = order["result_hash"]
            sample["matches_baseline"] = bool(
                sample["matches_baseline"]
                and order["result_hash"] == baseline.get("result_hash")
            )
        samples.append(sample)

    consistent = all(sample["matches_baseline"] for sample in samples)
    return {
        "valid": True,
        "kind": "determinism",
        "consistent": consistent,
        "deterministic": consistent,
        "hash": baseline_hash,
        "order_hash": baseline["order_hash"],
        "result_hash": baseline.get("result_hash"),
        "samples": samples,
        "seed_count": len(seed_list),
        "divergent": [sample["name"] for sample in samples if not sample["matches_baseline"]],
    }

