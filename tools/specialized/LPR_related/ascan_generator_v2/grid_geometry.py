"""
grid_geometry.py
================

格子上の対象形状（gprMax と同じ規則でのラスタ化）と、gprMax が出力する .vti の読み込み。

gprMax（v3.1.6, 誘電率の平滑化なし）の規則
------------------------------------------
* セル (i, j) は [i dx, (i+1) dx] x [j dy, (j+1) dy]。
* #box はセル番号 round(x0/dx) .. round(x1/dx)-1 を材料で埋める。
* #cylinder（z 方向）はセル中心と円の中心の距離が r 以下のセルを埋める。
* 平滑化なし（averaging = n。分散性材料は gprMax が自動で平滑化を切る）では、
  埋めたセルの 4 隅の Ez 節点にもその材料が入る。
  したがって **2D TMz の計算で Ez が感じる対象は、セルの範囲より 1 節点ぶん広い。**
  Ez 節点を中心とする幅 dx の区間で考えると、実効的な厚さは「セルの厚さ + dx」になる。

このモジュールは .in 生成時の予測（ケース JSON の as_built）と、
計算後の .vti からの読み取りの両方で使う（同じ関数で数えるため）。
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# ラスタ化（局所窓の中だけで計算する）
# =============================================================================
@dataclass
class LocalMask:
    """全体のセル番号 (i0, j0) を原点とする局所的なセルマスク mask[i - i0, j - j0]"""
    i0: int
    j0: int
    mask: np.ndarray            # bool, shape (ni, nj)
    dx: float

    @property
    def n_cells(self) -> int:
        return int(self.mask.sum())


def _window(xmin: float, xmax: float, ymin: float, ymax: float, dx: float, pad: int = 2):
    i0 = int(np.floor(xmin / dx)) - pad
    i1 = int(np.ceil(xmax / dx)) + pad
    j0 = int(np.floor(ymin / dx)) - pad
    j1 = int(np.ceil(ymax / dx)) + pad
    ic = (np.arange(i0, i1) + 0.5) * dx
    jc = (np.arange(j0, j1) + 0.5) * dx
    X, Y = np.meshgrid(ic, jc, indexing="ij")
    return i0, j0, X, Y


def rasterize_circle(xc: float, yc: float, r: float, dx: float) -> LocalMask:
    """gprMax の z 方向 #cylinder と同じ規則（セル中心が円内）"""
    i0, j0, X, Y = _window(xc - r, xc + r, yc - r, yc + r, dx)
    mask = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2) <= r
    return LocalMask(i0, j0, mask, dx)


def rasterize_box(x0: float, y0: float, x1: float, y1: float, dx: float) -> LocalMask:
    """gprMax の #box と同じ規則"""
    ia, ib = int(round(x0 / dx)), int(round(x1 / dx))
    ja, jb = int(round(y0 / dx)), int(round(y1 / dx))
    return LocalMask(ia, ja, np.ones((ib - ia, jb - ja), dtype=bool), dx)


def rasterize_polygon(vertices: Sequence[Tuple[float, float]], dx: float) -> LocalMask:
    """セル中心が多角形の内側にあるセル（偶奇規則）"""
    v = np.asarray(vertices, dtype=float)
    i0, j0, X, Y = _window(v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max(), dx)
    inside = np.zeros(X.shape, dtype=bool)
    n = len(v)
    for k in range(n):
        xa, ya = v[k]
        xb, yb = v[(k + 1) % n]
        cond = (ya > Y) != (yb > Y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (xb - xa) * (Y - ya) / (yb - ya) + xa
        inside ^= cond & (X < xint)
    return LocalMask(i0, j0, inside, dx)


def mask_to_column_boxes(lm: LocalMask) -> List[Tuple[float, float, float, float]]:
    """セルマスクを、列ごとの連続区間の #box（x0, y0, x1, y1）に分解する"""
    boxes = []
    dx = lm.dx
    for a in range(lm.mask.shape[0]):
        col = lm.mask[a]
        if not col.any():
            continue
        idx = np.flatnonzero(col)
        # 連続区間に分ける
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends = np.r_[idx[breaks], idx[-1]]
        i = lm.i0 + a
        for s, e in zip(starts, ends):
            boxes.append((i * dx, (lm.j0 + s) * dx, (i + 1) * dx, (lm.j0 + e + 1) * dx))
    return boxes


# =============================================================================
# Ez 節点と、中点の真下（nadir）での寸法
# =============================================================================
def ez_node_mask(cell_mask: np.ndarray) -> np.ndarray:
    """セルマスク (ni, nj) から Ez 節点マスク (ni+1, nj+1)：4 隅のどれかのセルが対象なら対象"""
    m = cell_mask
    out = np.zeros((m.shape[0] + 1, m.shape[1] + 1), dtype=bool)
    out[:-1, :-1] |= m
    out[1:, :-1] |= m
    out[:-1, 1:] |= m
    out[1:, 1:] |= m
    return out


def _runs(col: np.ndarray) -> List[Tuple[int, int]]:
    idx = np.flatnonzero(col)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def as_built_summary(lm: LocalMask, x_nadir_m: float, surface_y_m: float) -> Dict:
    """
    ラスタ化した対象の寸法。中点の真下（x = x_nadir_m の Ez 節点列）での厚さと上端深さを返す。

    * cells: 対象のセルの範囲（vti の Material と同じ）
    * ez_nodes: Ez 節点の範囲。実効的な境界は外側の節点から dx/2 外。
      地表も同じ規則で dx/2 上に実効境界があるので、実効的な上端深さはセル基準と同じになる。
    """
    dx = lm.dx
    m = lm.mask
    n_cells = int(m.sum())
    out: Dict = dict(n_cells=n_cells, area_m2=n_cells * dx * dx,
                     area_equivalent_diameter_m=float(np.sqrt(4.0 * n_cells * dx * dx / np.pi)))
    if n_cells == 0:
        out["empty"] = True
        return out

    ii, jj = np.nonzero(m)
    out["cells_x_range_m"] = [float((lm.i0 + ii.min()) * dx), float((lm.i0 + ii.max() + 1) * dx)]
    out["cells_y_range_m"] = [float((lm.j0 + jj.min()) * dx), float((lm.j0 + jj.max() + 1) * dx)]
    out["cells_width_m"] = out["cells_x_range_m"][1] - out["cells_x_range_m"][0]
    out["cells_height_m"] = out["cells_y_range_m"][1] - out["cells_y_range_m"][0]

    nodes = ez_node_mask(m)                      # 節点 (i0 + a, j0 + b)
    a_n = int(round(x_nadir_m / dx)) - lm.i0
    if 0 <= a_n < nodes.shape[0]:
        runs = _runs(nodes[a_n])
    else:
        runs = []
    if runs:
        # 最も上の区間（上端エコーを返す部分）
        s, e = max(runs, key=lambda r: r[1])
        y_top_node = (lm.j0 + e) * dx
        y_bot_node = (lm.j0 + s) * dx
        n_nodes = e - s + 1
        out["nadir"] = dict(
            x_m=x_nadir_m,
            n_segments=len(runs),
            ez_top_node_y_m=y_top_node,
            ez_bottom_node_y_m=y_bot_node,
            ez_node_count=n_nodes,
            cells_thickness_m=(n_nodes - 1) * dx,          # セル基準の厚さ（節点間距離）
            effective_thickness_m=n_nodes * dx,            # Ez が感じる実効的な厚さ（= セル基準 + dx）
            depth_top_m=surface_y_m - y_top_node,          # 地表と上端の深さ（実効境界どうしでも同じ）
        )
    else:
        out["nadir"] = None
    return out


# =============================================================================
# gprMax .vti の読み込み
# =============================================================================
@dataclass
class VtiGeometry:
    material: np.ndarray          # (nx, ny, nz) uint32
    sources_pml: Optional[np.ndarray]
    receivers: Optional[np.ndarray]
    spacing: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    material_ids: Dict[str, int]  # 材料名 -> 番号

    def mask(self, name: str) -> np.ndarray:
        """材料名のセルマスク (nx, ny, nz)"""
        return self.material == self.material_ids[name]


def read_gprmax_vti(path: str) -> VtiGeometry:
    """gprMax v3 の #geometry_view（n: normal）が出力する .vti を読む"""
    raw = open(path, "rb").read()
    head_end = raw.index(b"<AppendedData")
    header = raw[:head_end].decode("utf-8", errors="replace")
    m = re.search(r'WholeExtent="([^"]+)"', header)
    ext = [int(v) for v in m.group(1).split()]
    nx, ny, nz = ext[1] - ext[0], ext[3] - ext[2], ext[5] - ext[4]
    spacing = tuple(float(v) for v in re.search(r'Spacing="([^"]+)"', header).group(1).split())
    origin = tuple(float(v) for v in re.search(r'Origin="([^"]+)"', header).group(1).split())
    htype = re.search(r'header_type="([^"]+)"', header)
    hdt = np.dtype("<u8") if (htype and htype.group(1) == "UInt64") else np.dtype("<u4")

    start = raw.index(b"_", raw.index(b">", head_end)) + 1
    arrays = {}
    for am in re.finditer(r'<DataArray type="(\w+)" Name="(\w+)" format="appended" offset="(\d+)"', header):
        vtype, name, off = am.group(1), am.group(2), int(am.group(3))
        dt = {"UInt32": "<u4", "Int8": "i1", "UInt8": "u1", "Int16": "<i2", "Float32": "<f4"}[vtype]
        p = start + off
        nbytes = int(np.frombuffer(raw[p:p + hdt.itemsize], hdt)[0])
        p += hdt.itemsize
        a = np.frombuffer(raw[p:p + nbytes], dtype=dt)
        arrays[name] = a.reshape((nz, ny, nx)).transpose(2, 1, 0)   # VTK は x が最速

    ids = {}
    tail = raw[raw.index(b"</VTKFile>"):].decode("utf-8", errors="replace")
    gm = re.search(r"<gprMax>.*</gprMax>", tail, re.S)
    if gm:
        root = ET.fromstring(gm.group(0))
        for el in root.findall("Material"):
            ids[el.attrib["name"]] = int(el.text)

    return VtiGeometry(material=arrays["Material"], sources_pml=arrays.get("Sources_PML"),
                       receivers=arrays.get("Receivers"), spacing=spacing, origin=origin,
                       material_ids=ids)


def target_mask_from_vti(geo: VtiGeometry, material_name: str, surface_y_m: float) -> LocalMask:
    """vti から、地表より下にある指定材料のセル（= 対象）を取り出す"""
    dx = geo.spacing[0]
    m = geo.mask(material_name)[:, :, 0].copy()
    j_surface = int(round(surface_y_m / dx))
    m[:, j_surface:] = False                       # 地表より上（空気）は除く
    if not m.any():
        return LocalMask(0, 0, np.zeros((1, 1), dtype=bool), dx)
    ii, jj = np.nonzero(m)
    i0, i1, j0, j1 = ii.min() - 2, ii.max() + 3, jj.min() - 2, jj.max() + 3
    return LocalMask(int(i0), int(j0), m[i0:i1, j0:j1], dx)
