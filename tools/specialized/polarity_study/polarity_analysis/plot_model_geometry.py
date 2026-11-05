#!/usr/bin/env python3
"""
plot_model_geometry.py  --  計算モデルの図（図 1 の素材）
=========================================================

gprMax の形状ファイル（.vti、#geometry_view の 'n'）から、媒質を固定の色で塗り分けたモデル図を描く。
全体図と拡大図を、別々の図として出力する。

    python plot_model_geometry.py [キャンペーンフォルダ または .vti ...] [オプション]

  キャンペーンフォルダを指定すると、manifest.json の順に全ケースの図を順番に作る（既にあるものは飛ばす）。
  .vti を直接指定すると、そのケースだけを作る。

読み取るもの
  .vti        媒質（Material）、PML と送信点（Sources_PML: 1 = PML, 2 = 送信点）、受信点（Receivers: 1）
  ケース JSON  媒質の比誘電率（凡例の表示）、地表の高さ、アンテナの位置、反射体の寸法とフレネル半径（--annotate）
              .vti と同じフォルダの <case_id>.json。なければ媒質の名前だけで描く。

出力（.in と同じ階層の geometry/ フォルダ）
  <case_id>_model_full.png / .pdf   計算領域の全体
  <case_id>_model_zoom.png / .pdf   アンテナと反射体の周りの拡大

色（論文の本文やキャプションで呼びやすい名前）
  真空・空洞 white、レゴリス light brown、岩片 dark gray、PML blue（半透明）、送受信点 red（星）

オプション
  --groups / --targets / --cases   対象を絞る（キャンペーンフォルダを指定したとき）
  --annotate      拡大図に寸法線（w, h, d, h_antenna）とフレネル帯の範囲（2r_F、破線）を描く
  --no-title      図の上のケース名を描かない
  --force         既にある図も作り直す
  --formats png pdf
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_style                                 # noqa: E402
figure_style.apply()
from matplotlib.lines import Line2D                 # noqa: E402
from matplotlib.patches import Patch                # noqa: E402

# =============================================================================
# [EDIT HERE]
# =============================================================================
CAMPAIGN_DIR = Path("/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_10x10/Polarity_EPS/polarity_v01")

# 媒質の色（gprMax の材料名 -> (色の名前, RGB)）。色の名前は凡例と README に使う
MATERIAL_COLORS = {
    "free_space": ("white", "#ffffff"),
    "regolith": ("light brown", "#d2b48c"),
    "rock": ("dark gray", "#505050"),
    "pec": ("black", "#000000"),
}
# 上の表にない材料に順に割り当てる色
SPARE_COLORS = [("orange", "#ff8c00"), ("green", "#2ca02c"), ("purple", "#7f3fbf")]
PML_COLOR = ("blue", "#1f4fd8", 0.55)               # (名前, RGB, 不透明度)
ANTENNA_COLOR = ("red", "#e00000")

ZOOM_MARGIN_X_M = 1.0       # 拡大図: 反射体の左右の余白（幅の半分 + これ。最小 ±1.5 m、平行平板は ±3 m）
ZOOM_MARGIN_BOTTOM_M = 0.5  # 拡大図: 反射体の底面の下の余白
ZOOM_MARGIN_TOP_M = 0.3     # 拡大図: アンテナの上の余白
# =============================================================================


# -----------------------------------------------------------------------------
# .vti
# -----------------------------------------------------------------------------
def read_vti(path: Path) -> Dict:
    """gprMax v3 の .vti（'n' 形式）: Material, Sources_PML, Receivers と、番号 -> 名前の対応"""
    raw = path.read_bytes()
    head_end = raw.index(b"<AppendedData")
    header = raw[:head_end].decode("utf-8", errors="replace")
    ext = [int(v) for v in re.search(r'WholeExtent="([^"]+)"', header).group(1).split()]
    nx, ny, nz = ext[1] - ext[0], ext[3] - ext[2], ext[5] - ext[4]
    spacing = [float(v) for v in re.search(r'Spacing="([^"]+)"', header).group(1).split()]
    origin = [float(v) for v in re.search(r'Origin="([^"]+)"', header).group(1).split()]
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
        arrays[name] = a.reshape((nz, ny, nx)).transpose(2, 1, 0)[:, :, 0]     # (nx, ny)
    names = {}
    tail = raw[raw.index(b"</VTKFile>"):].decode("utf-8", errors="replace")
    gm = re.search(r"<gprMax>.*</gprMax>", tail, re.S)
    if gm:
        for el in ET.fromstring(gm.group(0)).findall("Material"):
            names[int(el.text)] = el.attrib["name"]
    return dict(material=arrays["Material"], sources_pml=arrays.get("Sources_PML"),
                receivers=arrays.get("Receivers"), dx=spacing[0], origin=origin, names=names)


# -----------------------------------------------------------------------------
# ケースの情報
# -----------------------------------------------------------------------------
def case_json_for(vti: Path) -> Optional[Dict]:
    """<case_id>_geometry.vti と同じフォルダの <case_id>.json"""
    stem = re.sub(r"_geometry\d*$", "", vti.stem)
    p = vti.parent / f"{stem}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    cands = list(vti.parent.glob("*.json"))
    return json.loads(cands[0].read_text(encoding="utf-8")) if len(cands) == 1 else None


def material_labels(d: Optional[Dict], names: Dict[int, str]) -> Dict[str, str]:
    """凡例の文字: 材料名 -> 'Regolith (eps = 3)' など"""
    eps = {"free_space": 1.0}
    kind = None
    if d:
        mats = d.get("materials", {})
        for role in ("background", "target"):
            m = mats.get(role)
            if m:
                eps[m["gprmax_name"]] = m["eps_r"]
        kind = (d.get("target") or {}).get("kind")
    out = {}
    for n in names.values():
        if n == "free_space":
            base = "Vacuum / void" if kind == "void" else "Vacuum"
        elif n == "regolith":
            base = "Regolith"
        elif n == "rock":
            base = "Rock"
        else:
            base = n
        out[n] = base                      # 凡例は名称だけ（比誘電率はキャプションに書く）
    return out


def color_table(names: Dict[int, str]) -> Tuple[np.ndarray, Dict[str, Tuple[str, str]]]:
    """材料番号 -> RGBA（uint8）の表と、使った色"""
    used = {}
    spare = list(SPARE_COLORS)
    n_ids = max(names) + 1 if names else 1
    lut = np.zeros((n_ids, 4), dtype=np.uint8)
    for i, n in names.items():
        if n in MATERIAL_COLORS:
            cname, hexc = MATERIAL_COLORS[n]
        else:
            cname, hexc = spare.pop(0) if spare else ("gray", "#999999")
            print(f"    note: material '{n}' has no fixed color; using {cname}")
        rgb = matplotlib.colors.to_rgb(hexc)
        lut[i] = [int(round(255 * c)) for c in rgb] + [255]
        used[n] = (cname, hexc)
    return lut, used


# -----------------------------------------------------------------------------
# 描画
# -----------------------------------------------------------------------------
def draw(geo: Dict, d: Optional[Dict], out_base: Path, zoom: bool, annotate: bool,
         title: bool, formats: List[str]) -> None:
    dx = geo["dx"]
    mat = geo["material"]
    nx, ny = mat.shape
    x0, y0 = geo["origin"][0], geo["origin"][1]

    # 座標: 横 = アンテナからの水平距離、縦 = 地表からの深さ（下が正）
    x_ant = d["antenna"]["x_m"] if d else x0 + nx * dx / 2
    if d and "model" in d:
        y_surf = d["model"]["surface_y_m"]
    else:
        reg = [i for i, n in geo["names"].items() if n not in ("free_space", "pec")]
        solid = np.isin(mat, reg)
        rows = [np.flatnonzero(c).max() + 1 for c in solid if c.any()]
        y_surf = y0 + (np.median(rows) if rows else ny) * dx

    lut, used = color_table(geo["names"])
    rgb = lut[mat][:, :, :3].astype(np.float32)             # (nx, ny, 3)
    # PML は半透明の青を重ねた色として、1 枚の画像に合成する（描画が速い）
    sp = geo.get("sources_pml")
    has_pml = sp is not None and np.any(sp == 1)
    if has_pml:
        a = PML_COLOR[2]
        blue = np.array(matplotlib.colors.to_rgb(PML_COLOR[1])) * 255
        m = sp == 1
        rgb[m] = (1 - a) * rgb[m] + a * blue
    img = np.transpose(rgb.astype(np.uint8), (1, 0, 2))[::-1]   # 行 = 上から（y の大きい方から）
    extent = [x0 - x_ant, x0 + nx * dx - x_ant, y_surf - y0, y_surf - (y0 + ny * dx)]

    # 表示範囲
    if zoom:
        t = (d or {}).get("target") or {}
        h_ant = d["antenna"]["height_m"] if d else 0.3
        if t:
            w = t.get("width_m")
            half = 3.0 if w is None else max(1.5, w / 2 + ZOOM_MARGIN_X_M)
            bottom = t["depth_top_m"] + t["height_m"] + ZOOM_MARGIN_BOTTOM_M
        else:
            half, bottom = 3.0, 2.0
            if d and d["antenna"]["rx_xy_m"] != d["antenna"]["tx_xy_m"]:
                ry = d["antenna"]["rx_xy_m"][1]
                bottom = max(bottom, y_surf - ry + 0.5)
        xlim = (-half, half)
        ylim = (min(bottom, extent[2]), -(h_ant + ZOOM_MARGIN_TOP_M))
    else:
        xlim = (extent[0], extent[1])
        ylim = (extent[2], extent[3])

    w_in = 6.0
    h_in = w_in * (ylim[0] - ylim[1]) / (xlim[1] - xlim[0])
    h_in = float(np.clip(h_in, 2.0, 9.0))
    fig, ax = plt.subplots(figsize=(w_in + 2.6, h_in + 0.8))
    if zoom:                                              # 表示範囲だけを切り出して描く
        c0 = max(int(np.floor((xlim[0] - extent[0]) / dx)) - 1, 0)
        c1 = min(int(np.ceil((xlim[1] - extent[0]) / dx)) + 1, img.shape[1])
        r0 = max(int(np.floor((ylim[1] - extent[3]) / dx)) - 1, 0)
        r1 = min(int(np.ceil((ylim[0] - extent[3]) / dx)) + 1, img.shape[0])
        sub = img[r0:r1, c0:c1]
        ext = [extent[0] + c0 * dx, extent[0] + c1 * dx, extent[3] + r1 * dx, extent[3] + r0 * dx]
        ax.imshow(sub, extent=ext, interpolation="nearest", aspect="equal")
    else:
        ax.imshow(img, extent=extent, interpolation="nearest", aspect="equal")

    handles = []
    for n in sorted(used, key=lambda k: list(MATERIAL_COLORS).index(k) if k in MATERIAL_COLORS else 99):
        if n == "pec" and not np.any(mat == [i for i, m in geo["names"].items() if m == "pec"][0]):
            continue
        cname, hexc = used[n]
        handles.append(Patch(facecolor=hexc, edgecolor="0.4", linewidth=0.6,
                             label=f"{material_labels(d, geo['names'])[n]}"))

    # PML（全体図だけ凡例に出す。拡大図には写らない）
    if has_pml and not zoom:
        n_pml = int(np.flatnonzero(sp[:, ny // 2] == 1).size // 2)
        handles.append(Patch(facecolor=PML_COLOR[1], alpha=PML_COLOR[2], edgecolor="none",
                             label="PML"))

    # 送受信点（gprMax の座標は節点 i * dx）
    tx = np.argwhere(sp == 2) if sp is not None else np.empty((0, 2))
    rx = np.argwhere(geo["receivers"] == 1) if geo.get("receivers") is not None else np.empty((0, 2))
    to_xy = lambda ij: (x0 + ij[0] * dx - x_ant, y_surf - (y0 + ij[1] * dx))
    same = len(tx) == 1 and len(rx) == 1 and list(tx[0]) == list(rx[0])
    if same:
        x, y = to_xy(tx[0])
        ax.plot(x, y, marker="*", ms=14, color=ANTENNA_COLOR[1], mec="k", mew=0.6, ls="")
        handles.append(Line2D([], [], marker="*", ms=12, color=ANTENNA_COLOR[1], mec="k", mew=0.6,
                              ls="", label="Tx/Rx"))
    else:
        for ij in tx:
            ax.plot(*to_xy(ij), marker="*", ms=14, color=ANTENNA_COLOR[1], mec="k", mew=0.6, ls="")
        for ij in rx:
            ax.plot(*to_xy(ij), marker="v", ms=9, color=ANTENNA_COLOR[1], mec="k", mew=0.6, ls="")
        if len(tx):
            handles.append(Line2D([], [], marker="*", ms=12, color=ANTENNA_COLOR[1], mec="k", ls="",
                                  label="Tx"))
        if len(rx):
            handles.append(Line2D([], [], marker="v", ms=8, color=ANTENNA_COLOR[1], mec="k", ls="",
                                  label="Rx"))

    if annotate and zoom and d and d.get("target"):
        annotate_dimensions(ax, d, handles)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Horizontal distance from antenna (m)")
    ax.set_ylabel("Depth below surface (m)")
    if title:
        ax.set_title(d["case_id"] if d else out_base.stem)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(out_base.with_suffix("." + ext), dpi=300, bbox_inches="tight")
    plt.close(fig)


def annotate_dimensions(ax, d: Dict, handles: List) -> None:
    """寸法線（w, h, d, h_antenna）と、上面・底面のフレネル帯の範囲（2 r_F）"""
    t = d["target"]
    th = d["theory"]["nominal"]
    w, h, dep = t.get("width_m"), t["height_m"], t["depth_top_m"]
    h_ant = d["antenna"]["height_m"]
    kw = dict(arrowstyle="<->", color="k", lw=0.9, shrinkA=0, shrinkB=0)
    xr = (w / 2 if w is not None else 2.0) + 0.25
    ax.annotate("", (xr, 0), (xr, dep), arrowprops=kw)
    ax.text(xr + 0.05, dep / 2, "d", va="center")
    ax.annotate("", (xr, dep), (xr, dep + h), arrowprops=kw)
    ax.text(xr + 0.05, dep + h / 2, "h", va="center")
    ax.annotate("", (-0.25, -h_ant), (-0.25, 0), arrowprops=kw)
    ax.text(-0.3, -h_ant / 2, r"$h_{\rm antenna}$", va="center", ha="right")
    if w is not None:
        yw = dep + h + 0.2
        ax.annotate("", (-w / 2, yw), (w / 2, yw), arrowprops=kw)
        ax.text(0, yw + 0.05, "w", ha="center", va="top")
    for face, y in (("top", dep), ("bottom", dep + h)):
        r = th[face]["fresnel_radius_m"]
        ax.plot([-r, r], [y - 0.06, y - 0.06] if face == "top" else [y + 0.06, y + 0.06],
                color=ANTENNA_COLOR[1], ls="--", lw=1.3)
    handles.append(Line2D([], [], color=ANTENNA_COLOR[1], ls="--", lw=1.3,
                          label=r"$2r_{\rm F}$"))


# -----------------------------------------------------------------------------
def collect(args) -> List[Path]:
    paths = []
    for p in args.paths:
        p = p.resolve()
        if p.suffix == ".vti":
            paths.append(p)
            continue
        man = p / "manifest.json"
        if not man.exists():
            print(f"{p}: neither a .vti nor a campaign folder (manifest.json not found)")
            continue
        for r in json.loads(man.read_text(encoding="utf-8"))["cases"]:
            if args.groups and r["group"] not in args.groups:
                continue
            if args.targets and r.get("target") not in args.targets:
                continue
            if args.cases and r["case_id"] not in args.cases:
                continue
            dj = json.loads((p / r["json"]).read_text(encoding="utf-8"))
            paths.append(p / dj["files"]["geometry_vti"])
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path, default=[CAMPAIGN_DIR])
    ap.add_argument("--groups", nargs="+", choices=["reference", "slab", "nominal", "depth"])
    ap.add_argument("--targets", nargs="+", choices=["rock", "void"])
    ap.add_argument("--cases", nargs="+")
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--no-title", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    a = ap.parse_args(argv)

    vtis = collect(a)
    n = len(vtis)
    print(f"{n} geometry files")
    t0 = time.time()
    done = skipped = missing = 0
    for k, vti in enumerate(vtis, 1):
        d = case_json_for(vti) if vti.exists() else None
        cid = d["case_id"] if d else vti.stem
        out_dir = vti.parent / "geometry"
        suffix = "_annotated" if a.annotate else ""
        bases = [out_dir / f"{cid}_model_full", out_dir / f"{cid}_model_zoom{suffix}"]
        if not vti.exists():
            missing += 1
            print(f"[{k:3d}/{n}] {cid:40s} no .vti (skipped)")
            continue
        if not a.force and all(b.with_suffix("." + a.formats[0]).exists() for b in bases):
            skipped += 1
            continue
        geo = read_vti(vti)
        draw(geo, d, bases[0], zoom=False, annotate=False, title=not a.no_title, formats=a.formats)
        draw(geo, d, bases[1], zoom=True, annotate=a.annotate, title=not a.no_title, formats=a.formats)
        done += 1
        el = time.time() - t0
        eta = el / k * (n - k)
        print(f"[{k:3d}/{n}] {100*k/n:5.1f}%  {int(el)//60:02d}:{int(el)%60:02d} elapsed, "
              f"ETA {int(eta)//60:02d}:{int(eta)%60:02d}  {cid}", flush=True)
    print(f"\ndone: {done} drawn, {skipped} already existed (use --force to redraw), {missing} without .vti")
    return 0


if __name__ == "__main__":
    sys.exit(main())
