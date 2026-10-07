#!/usr/bin/env python3
"""
generate_ascan_inputs.py
========================

CE-4 LPR CH2 を模擬した A-scan 計算（岩石・空隙のサイズと波形の関係）用に、
gprMax の .in ファイルと、ケースごとのパラメータ JSON、一覧ファイル、実行スクリプトを作る。

    python generate_ascan_inputs.py --dry-run      # 件数・容量・検証結果だけ表示（ファイルは作らない）
    python generate_ascan_inputs.py                # 生成
    python generate_ascan_inputs.py --depths 1 --targets basalt void   # 一部だけ生成

出力（キャンペーンフォルダ = ROOT / CAMPAIGN）
------------------------------------------
    campaign.json           共通設定（材料、計算領域、アンテナ、波形、生成スクリプトのハッシュ）
    manifest.json / .csv    全ケースの一覧（フォルダ内のケース JSON を走査して毎回作り直す）
    in_list.txt             .in の一覧（キャンペーンフォルダからの相対パス、実行順）
    out_list.txt            .out の一覧（絶対パス）
    run_all.sh              一括実行（.done があるケースは飛ばす）
    check_geometry.sh       --geometry-only で形状だけ作る
    scripts/                生成に使ったスクリプトのコピー
    reference/ref_background/, reference/ref_freespace/
    <target>/<shape>/depthXXm/DXXXcm/<case_id>.in, <case_id>.json, A-scan/<case_id>.out

設定を変えるときは CAMPAIGN を新しい名前にする（同じ名前で物理設定が変わると停止する）。
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import materials as mat                                   # noqa: E402
import theory                                             # noqa: E402
from grid_geometry import (LocalMask, as_built_summary, mask_to_column_boxes,   # noqa: E402
                           rasterize_box, rasterize_circle, rasterize_polygon)

GENERATOR_VERSION = "1.0.0"
CASE_SCHEMA = "lpr_ascan_case/1"

# =============================================================================
# [EDIT HERE] 設定
# =============================================================================
ROOT = Path("/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_5x14/size_waveform_investigation")
CAMPAIGN = "ascan_v01"

# ---- 計算領域（2D, TMz） ----------------------------------------------------
DOMAIN_X_M = 5.0
DOMAIN_Y_M = 14.0
DX_M = 0.0025
SURFACE_Y_M = 13.0                  # 地表（レゴリス上面）。レゴリスは y = 0 - 13 m
TIME_WINDOW_S = 170e-9
PML_CELLS = 10                      # gprMax の既定値（記録用。.in には書かない）

# ---- アンテナ（Fang et al., 2014） ---------------------------------------------
ANT_MID_X_M = 2.5                   # 送受信点の中点（対象はこの真下）
ANT_HEIGHT_M = 0.30                 # 地表からの高さ
ANT_OFFSET_M = 0.32                 # 送受信点の間隔
WAVEFORM = dict(type="gaussiandot", amplitude=1.0, frequency_hz=400e6, name="my_src")
TRANSMITTED_POLARITY = "NPN"        # 送信波の極性（理論値の極性ラベルに使う。ref_freespace で確認）

# ---- 計算パターン ---------------------------------------------------------------
TARGETS = ["basalt", "highland", "void"]          # materials.MEDIA のキー
SHAPES = ["circle", "square"]
SIZES_CM = list(range(1, 16)) + [20, 30, 40, 50]
DEPTHS_M = [1, 6, 12]                              # 地表から対象上端までの深さ

# 不規則な形状（様子見用）。形状は variant ごとに固定し、サイズで相似に拡大縮小する。
# サイズは多角形の面積相当直径。
IRREGULAR = dict(
    enabled=True,
    targets=["basalt", "void"],
    depths_m=[1],
    sizes_cm=[6, 10, 30],
    variants=[1, 2],
    n_vertices=(7, 11),
    radius_jitter=0.25,
    base_seed=20261007,
)

# ---- スナップショット（既定ではコメントアウトした形で .in に残す） -------------
SNAPSHOT = dict(dt_s=0.5e-9, t_start_s=0.0, t_end_s=TIME_WINDOW_S, dxdy_m=0.005)

# ---- 検証に使う値 ---------------------------------------------------------------
SOURCE_DELAY_EST_S = 6e-9           # 波源のピークの遅れ + パルスの後半（時間窓の確認用の目安）
TIME_MARGIN_S = 10e-9               # 下端エコーの後に確保する余裕
ATTENUATION_FREQUENCY_HZ = 500e6    # 理論減衰量を計算する周波数（パルスの中心周波数）
FRESNEL_WAVELENGTH_M = mat.C0 / 500e6   # フレネル半径に使う波長（EPS 論文と同じく真空中、500 MHz）

TARGET_ORDER = {k: i for i, k in enumerate(["basalt", "highland", "void"])}

# =============================================================================
# 導出量
# =============================================================================
NX = int(round(DOMAIN_X_M / DX_M))
NY = int(round(DOMAIN_Y_M / DX_M))
DT_S = 1.0 / (mat.C0 * math.sqrt(1.0 / DX_M ** 2 + 1.0 / DX_M ** 2))     # 2D の CFL 条件
ITERATIONS = int(math.ceil(TIME_WINDOW_S / DT_S)) + 1          # gprMax と同じ数え方
TX_X_M = ANT_MID_X_M - ANT_OFFSET_M / 2.0
RX_X_M = ANT_MID_X_M + ANT_OFFSET_M / 2.0
ANT_Y_M = SURFACE_Y_M + ANT_HEIGHT_M


def fm(v: float) -> str:
    """座標などを .in に書く書式（浮動小数点の誤差を落とす）"""
    s = f"{round(float(v), 9):.9g}"
    return "0" if s in ("-0", "0") else s


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def depth_tag(d: float) -> str:
    if abs(d - round(d)) < 1e-9:
        return f"depth{int(round(d)):02d}m"
    return "depth" + f"{d:05.2f}".replace(".", "p") + "m"


# =============================================================================
# ケース
# =============================================================================
@dataclass
class Case:
    group: str                       # reference / main / irregular
    case_id: str
    target: Optional[str] = None     # materials.MEDIA のキー
    shape: Optional[str] = None      # circle / square / irregNN
    size_cm: Optional[int] = None
    depth_m: Optional[float] = None
    variant: Optional[int] = None
    ref_kind: Optional[str] = None   # background / freespace

    @property
    def rel_dir(self) -> Path:
        if self.group == "reference":
            return Path("reference") / self.case_id
        return Path(self.target) / self.shape / depth_tag(self.depth_m) / f"D{self.size_cm:03d}cm"

    @property
    def pair_key(self) -> Optional[str]:
        if self.group == "reference":
            return None
        return f"{self.shape}_{depth_tag(self.depth_m)}_D{self.size_cm:03d}cm"

    def sort_key(self):
        g = {"reference": 0, "main": 1, "irregular": 2}[self.group]
        if self.group == "reference":
            return (g, 0 if self.ref_kind == "background" else 1)
        return (g, self.depth_m, TARGET_ORDER.get(self.target, 99), self.shape, self.size_cm)


def make_case(group: str, target: str, shape: str, size_cm: int, depth_m: float,
              variant: Optional[int] = None) -> Case:
    cid = f"{target}_{shape}_{depth_tag(depth_m)}_D{size_cm:03d}cm"
    return Case(group=group, case_id=cid, target=target, shape=shape, size_cm=size_cm,
                depth_m=depth_m, variant=variant)


def planned_cases() -> List[Case]:
    cases = [Case(group="reference", case_id="ref_background", ref_kind="background"),
             Case(group="reference", case_id="ref_freespace", ref_kind="freespace")]
    for d in DEPTHS_M:
        for t in TARGETS:
            for s in SHAPES:
                for sz in SIZES_CM:
                    cases.append(make_case("main", t, s, sz, d))
    if IRREGULAR["enabled"]:
        for d in IRREGULAR["depths_m"]:
            for t in IRREGULAR["targets"]:
                for v in IRREGULAR["variants"]:
                    for sz in IRREGULAR["sizes_cm"]:
                        cases.append(make_case("irregular", t, f"irreg{v:02d}", sz, d, variant=v))
    return sorted(cases, key=Case.sort_key)


# =============================================================================
# 対象の形状
# =============================================================================
def irregular_unit_polygon(variant: int) -> Tuple[np.ndarray, int]:
    """面積相当直径 1 の、星形（自己交差しない）ランダム多角形。variant ごとに固定。"""
    seed = int(IRREGULAR["base_seed"]) + int(variant)
    rng = np.random.default_rng(seed)
    nmin, nmax = IRREGULAR["n_vertices"]
    n = int(rng.integers(nmin, nmax + 1))
    ang = (np.arange(n) + rng.uniform(-0.35, 0.35, n)) * 2.0 * np.pi / n + rng.uniform(0, 2 * np.pi)
    ang = np.sort(np.mod(ang, 2.0 * np.pi))
    rad = np.clip(1.0 + IRREGULAR["radius_jitter"] * rng.standard_normal(n), 0.55, 1.45)
    v = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
    x, y = v[:, 0], v[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    v *= math.sqrt((math.pi / 4.0) / area)
    return v, seed


def polygon_centroid(v: np.ndarray) -> Tuple[float, float]:
    x, y = v[:, 0], v[:, 1]
    cr = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = 0.5 * cr.sum()
    return float(((x + np.roll(x, -1)) * cr).sum() / (6 * a)), float(((y + np.roll(y, -1)) * cr).sum() / (6 * a))


def vertical_chord(v: np.ndarray, x0: float) -> Optional[Tuple[float, float]]:
    """多角形と鉛直線 x = x0 の交点の (最上点 y, 最下点 y)"""
    ys = []
    n = len(v)
    for k in range(n):
        (xa, ya), (xb, yb) = v[k], v[(k + 1) % n]
        if (xa - x0) * (xb - x0) <= 0 and xa != xb:
            ys.append(ya + (yb - ya) * (x0 - xa) / (xb - xa))
    if len(ys) < 2:
        return None
    return max(ys), min(ys)


def target_geometry(case: Case) -> Dict:
    """対象の gprMax コマンド、連続形状の寸法、ラスタ化マスクを返す"""
    m = mat.MEDIA[case.target]
    name = m.gprmax_name
    D = case.size_cm / 100.0
    y_top = SURFACE_Y_M - case.depth_m
    xc = ANT_MID_X_M
    info: Dict = dict(size_m=D)

    if case.shape == "circle":
        r = D / 2.0
        yc = y_top - r
        cmds = [f"#cylinder: {fm(xc)} {fm(yc)} 0 {fm(xc)} {fm(yc)} {fm(DX_M)} {fm(r)} {name} n"]
        lm = rasterize_circle(xc, yc, r, DX_M)
        info.update(size_definition="diameter", center_xy_m=[xc, yc],
                    x_extent_m=[xc - r, xc + r], y_extent_m=[yc - r, y_top],
                    nominal_nadir=dict(depth_top_m=case.depth_m, thickness_m=D))
        align = [xc, yc, 2 * r]
    elif case.shape == "square":
        x0, x1, y0, y1 = xc - D / 2.0, xc + D / 2.0, y_top - D, y_top
        cmds = [f"#box: {fm(x0)} {fm(y0)} 0 {fm(x1)} {fm(y1)} {fm(DX_M)} {name} n"]
        lm = rasterize_box(x0, y0, x1, y1, DX_M)
        info.update(size_definition="side length", center_xy_m=[xc, y_top - D / 2.0],
                    x_extent_m=[x0, x1], y_extent_m=[y0, y1],
                    nominal_nadir=dict(depth_top_m=case.depth_m, thickness_m=D))
        align = [x0, x1, y0, y1]
    elif case.shape.startswith("irreg"):
        unit, seed = irregular_unit_polygon(case.variant)
        v = unit * D
        cx, _ = polygon_centroid(v)
        v[:, 0] += xc - cx
        v[:, 1] += y_top - v[:, 1].max()
        lm = rasterize_polygon(v, DX_M)
        boxes = mask_to_column_boxes(lm)
        cmds = [f"#box: {fm(a)} {fm(b)} 0 {fm(c)} {fm(d)} {fm(DX_M)} {name} n" for a, b, c, d in boxes]
        chord = vertical_chord(v, xc)
        info.update(size_definition="area-equivalent diameter of the polygon",
                    center_xy_m=list(polygon_centroid(v)),
                    x_extent_m=[float(v[:, 0].min()), float(v[:, 0].max())],
                    y_extent_m=[float(v[:, 1].min()), float(v[:, 1].max())],
                    irregular=dict(variant=case.variant, seed=seed, n_vertices=int(len(v)),
                                   vertices_m=v.round(9).tolist(),
                                   unit_vertices=unit.round(9).tolist(),
                                   construction="star-shaped random polygon; rasterized by cell centres "
                                                "and written as one #box per contiguous column run"),
                    nominal_nadir=None if chord is None else dict(
                        depth_top_m=SURFACE_Y_M - chord[0], thickness_m=chord[0] - chord[1]))
        align = [a for b in boxes for a in b]
    else:
        raise ValueError(case.shape)

    info["gprmax_commands"] = cmds
    info["dielectric_smoothing"] = False
    info["_mask"] = lm
    info["_align"] = align
    return info


# =============================================================================
# 検証
# =============================================================================
def validate_case(case: Case, geo: Dict, th_nom: Optional[Dict]) -> List[str]:
    errs = []
    for v in geo["_align"]:
        if abs(v / DX_M - round(v / DX_M)) > 1e-6 and not case.shape == "circle":
            errs.append(f"coordinate {v} is not a multiple of dx")
    if case.shape == "circle":
        xc, yc, d = geo["_align"]
        for v in (xc, yc):
            if abs(v / DX_M - round(v / DX_M)) > 1e-6:
                errs.append(f"circle centre {v} is not on a grid node")
        if abs(d / DX_M - round(d / DX_M)) > 1e-6:
            errs.append(f"circle diameter {d} is not a multiple of dx")
    (x0, x1), (y0, y1) = geo["x_extent_m"], geo["y_extent_m"]
    if y1 >= SURFACE_Y_M:
        errs.append("target top is at or above the surface")
    if y0 <= (PML_CELLS + 2) * DX_M:
        errs.append("target bottom is inside / too close to the bottom PML")
    if x0 <= (PML_CELLS + 2) * DX_M or x1 >= DOMAIN_X_M - (PML_CELLS + 2) * DX_M:
        errs.append("target is too close to the side PML")
    lm: LocalMask = geo["_mask"]
    if lm.n_cells == 0:
        errs.append("target has no cells after rasterization")
    if th_nom is not None:
        t_end = th_nom["bottom"]["t_bistatic_s"] + SOURCE_DELAY_EST_S + TIME_MARGIN_S
        if t_end > TIME_WINDOW_S:
            errs.append(f"time window too short: bottom echo + margins = {t_end*1e9:.1f} ns")
    return errs


# =============================================================================
# .in と JSON
# =============================================================================
def snapshot_block(enabled: bool) -> List[str]:
    c = "" if enabled else "#"
    s = SNAPSHOT
    return [
        "===== スナップショット =====",
        "既定では無効（for 文の 2 行が Python のコメントになっている）。" if not enabled else "有効。",
        "有効にするには、下の for 文 2 行の先頭の「#」を 1 文字ずつ外す。",
        "メモリを抑えるには t_start / t_end を対象のエコー付近に絞る（ケース JSON の theory を参照）。",
        "#python:",
        f"t_start = {s['t_start_s']:.6e}",
        f"t_end = {s['t_end_s']:.6e}",
        f"dt = {s['dt_s']:.6e}",
        f"d = {s['dxdy_m']}",
        "n0 = int(round(t_start / dt)) + 1",
        "n1 = int(round(t_end / dt))",
        f"{c}for i in range(n0, n1 + 1):",
        f"{c}    print('#snapshot: 0 0 0 {fm(DOMAIN_X_M)} {fm(DOMAIN_Y_M)} {fm(DX_M)} {{0}} {{0}} {fm(DX_M)} {{1}} snapshot{{2}}'.format(d, i * dt, i))",
        "#end_python:",
    ]


def render_in(case: Case, geo: Optional[Dict], snapshots: bool) -> str:
    L: List[str] = []
    bar = "=" * 76
    L += [bar, f"case_id: {case.case_id}"]
    if case.group == "reference":
        L.append("参照計算: " + ("背景レゴリスのみ（対象なし）" if case.ref_kind == "background"
                              else "自由空間のみ（レゴリスなし）。直接波から送信波形と波源の遅れを求める"))
    else:
        m = mat.MEDIA[case.target]
        L.append(f"対象: {case.target}（{m.label}）, 形状: {case.shape}（{geo['size_definition']}）, "
                 f"サイズ: {case.size_cm} cm, 上端の深さ: {fm(case.depth_m)} m")
    L += [f"生成: generate_ascan_inputs.py v{GENERATOR_VERSION}。詳細は同じフォルダの {case.case_id}.json",
          bar, "",
          f"#title: {case.case_id}",
          f"#domain: {fm(DOMAIN_X_M)} {fm(DOMAIN_Y_M)} {fm(DX_M)}",
          f"#dx_dy_dz: {fm(DX_M)} {fm(DX_M)} {fm(DX_M)}",
          f"#time_window: {TIME_WINDOW_S:.6e}",
          ""]

    # ---- 材料 ----
    used = [] if case.ref_kind == "freespace" else [mat.REGOLITH]
    if case.group != "reference":
        t = mat.MEDIA[case.target]
        if not t.builtin and t not in used:
            used.append(t)
    if used:
        L += ["===== 材料（2 極 Debye、帯域 "
              f"{mat.BAND_LO_HZ/1e6:.0f}-{mat.BAND_HI_HZ/1e6:.0f} MHz で tanδ 一定、最大平坦条件）=====",
              f"f0 = {mat.F0_HZ/1e6:.4f} MHz, tau1 = {mat.TAU1_S:.6e} s, tau2 = {mat.TAU2_S:.6e} s"]
        for m in used:
            extra = ""
            if m.density_g_cm3 is not None:
                extra = f", rho = {m.density_g_cm3} g/cm3, FeO+TiO2 = {m.feo_tio2_wt} wt%"
            L.append(f"{m.gprmax_name}: eps_r(f0) = {m.eps_r:.4f}, tan_delta = {m.tan_delta:.5f}{extra}")
            L += m.gprmax_lines()
        L.append("")
    if case.group != "reference" and mat.MEDIA[case.target].builtin:
        L += ["空隙は gprMax 組み込みの free_space（eps = 1、損失なし）", ""]

    # ---- 形状 ----
    L.append("===== 形状 =====")
    if case.ref_kind != "freespace":
        L += ["背景レゴリス（y = 0 - 13 m）",
              f"#box: 0 0 0 {fm(DOMAIN_X_M)} {fm(SURFACE_Y_M)} {fm(DX_M)} {mat.REGOLITH.gprmax_name} n"]
    if geo is not None:
        L.append(f"対象（誘電率の平滑化なし。{len(geo['gprmax_commands'])} 行）")
        L += geo["gprmax_commands"]
    L.append("")

    # ---- 形状の書き出し ----
    L += ["===== 形状の書き出し =====",
          f"#geometry_view: 0 0 0 {fm(DOMAIN_X_M)} {fm(DOMAIN_Y_M)} {fm(DX_M)} {fm(DX_M)} {fm(DX_M)} {fm(DX_M)} {case.case_id}_geometry n",
          "h5 は出力しない（必要なら make_h5_input.py で有効にした .in を作り、--geometry-only で実行）",
          f"DISABLED #geometry_objects_write: 0 0 0 {fm(DOMAIN_X_M)} {fm(DOMAIN_Y_M)} {fm(DX_M)} {case.case_id}_geometry",
          ""]

    # ---- 波源と受信点 ----
    w = WAVEFORM
    L += ["===== 波源と受信点 =====",
          f"送受信点の間隔 {fm(ANT_OFFSET_M)} m、地表からの高さ {fm(ANT_HEIGHT_M)} m、中点 x = {fm(ANT_MID_X_M)} m",
          f"#waveform: {w['type']} {fm(w['amplitude'])} {w['frequency_hz']/1e6:g}e6 {w['name']}",
          f"#hertzian_dipole: z {fm(TX_X_M)} {fm(ANT_Y_M)} 0 {w['name']}",
          f"#rx: {fm(RX_X_M)} {fm(ANT_Y_M)} 0",
          "#output_dir: A-scan",
          ""]
    for line in L:      # Python ブロックの外では、「#」で始まる行はすべて gprMax のコマンドでなければならない
        if line.startswith("#") and not line.split()[0].endswith(":"):
            raise RuntimeError(f"comment line must not start with '#': {line}")
    L += snapshot_block(snapshots)
    return "\n".join(L) + "\n"


def case_files(case: Case) -> Dict[str, str]:
    d = case.rel_dir
    c = case.case_id
    return dict(
        dir=d.as_posix(),
        input=(d / f"{c}.in").as_posix(),
        json=(d / f"{c}.json").as_posix(),
        output=(d / "A-scan" / f"{c}.out").as_posix(),
        geometry_vti=(d / f"{c}_geometry.vti").as_posix(),
        log=(d / f"{c}.log").as_posix(),
        done_marker=(d / f"{c}.done").as_posix(),
        snapshot_dir=(d / f"{c}_snaps").as_posix(),
    )


def physics_config() -> Dict:
    return dict(
        domain=dict(x_m=DOMAIN_X_M, y_m=DOMAIN_Y_M, dx_m=DX_M, nx=NX, ny=NY, mode="2D TMz",
                    surface_y_m=SURFACE_Y_M, pml_cells=PML_CELLS),
        time=dict(time_window_s=TIME_WINDOW_S, dt_s=DT_S, iterations=ITERATIONS),
        antenna=dict(mid_x_m=ANT_MID_X_M, height_m=ANT_HEIGHT_M, offset_m=ANT_OFFSET_M,
                     tx_xy_m=[TX_X_M, ANT_Y_M], rx_xy_m=[RX_X_M, ANT_Y_M], polarisation="z",
                     source="hertzian_dipole", waveform=dict(WAVEFORM),
                     transmitted_polarity_assumed=TRANSMITTED_POLARITY),
        materials={k: m.to_dict() for k, m in mat.MEDIA.items()},
        debye_band_hz=[mat.BAND_LO_HZ, mat.BAND_HI_HZ],
    )


def case_record(case: Case, geo: Optional[Dict], root: Path, snapshots: bool) -> Dict:
    files = case_files(case)
    ant = dict(antenna_height_m=ANT_HEIGHT_M, half_offset_m=ANT_OFFSET_M / 2.0)
    rec: Dict = dict(
        schema=CASE_SCHEMA,
        case_id=case.case_id,
        group=case.group,
        pair_key=case.pair_key,
        campaign=CAMPAIGN,
        campaign_root=str(root),
        generator=dict(script="generate_ascan_inputs.py", version=GENERATOR_VERSION),
        model=dict(domain_m=[DOMAIN_X_M, DOMAIN_Y_M], dx_m=DX_M, mode="2D TMz",
                   surface_y_m=SURFACE_Y_M, pml_cells=PML_CELLS,
                   time_window_s=TIME_WINDOW_S, dt_s=DT_S, iterations=ITERATIONS),
        antenna=dict(tx_xy_m=[TX_X_M, ANT_Y_M], rx_xy_m=[RX_X_M, ANT_Y_M], mid_x_m=ANT_MID_X_M,
                     offset_m=ANT_OFFSET_M, height_m=ANT_HEIGHT_M, polarisation="z",
                     waveform=dict(WAVEFORM)),
        snapshots_enabled=snapshots,
        files=files,
        references=dict(
            background=dict(case_id="ref_background",
                            output="reference/ref_background/A-scan/ref_background.out"),
            freespace=dict(case_id="ref_freespace",
                           output="reference/ref_freespace/A-scan/ref_freespace.out"),
        ),
    )
    rec["theory"] = dict(
        note=("Propagation times only (no source delay). Add the source delay measured from ref_freespace "
              "to compare with simulation time. Interfaces are treated as horizontal at the nadir."),
        transmitted_polarity_assumed=TRANSMITTED_POLARITY,
        attenuation_frequency_hz=ATTENUATION_FREQUENCY_HZ,
        fresnel_wavelength_m=FRESNEL_WAVELENGTH_M,
        fresnel_definition="r = sqrt(lambda * D / 2), D = antenna height + sum(thickness * sqrt(eps)) (EPS manuscript)",
        common=theory.common_arrivals(**ant),
    )

    if case.group == "reference":
        rec["reference_kind"] = case.ref_kind
        rec["materials"] = dict(background=None if case.ref_kind == "freespace" else mat.REGOLITH.to_dict())
        rec["target"] = None
        return rec

    m = mat.MEDIA[case.target]
    lm: LocalMask = geo["_mask"]
    as_built = as_built_summary(lm, ANT_MID_X_M, SURFACE_Y_M)
    rec["materials"] = dict(background=mat.REGOLITH.to_dict(), target=m.to_dict())
    rec["target"] = dict(
        kind="void" if case.target == "void" else "rock",
        material=case.target,
        gprmax_material=m.gprmax_name,
        shape=case.shape,
        size_cm=case.size_cm,
        size_m=geo["size_m"],
        size_definition=geo["size_definition"],
        depth_top_m=case.depth_m,
        center_xy_m=geo["center_xy_m"],
        x_extent_m=geo["x_extent_m"],
        y_extent_m=geo["y_extent_m"],
        dielectric_smoothing=geo["dielectric_smoothing"],
        gprmax_commands=geo["gprmax_commands"],
        irregular=geo.get("irregular"),
        nominal_nadir=geo["nominal_nadir"],
        as_built=as_built,
        as_built_note=("Predicted with the same rasterization rules as gprMax v3.1.6 (no averaging). "
                       "cells_*: solid/Material array (same as the .vti). ez nodes: material seen by Ez in 2D TMz; "
                       "effective_thickness_m = cell thickness + dx."),
    )
    th = {}
    if geo["nominal_nadir"] is not None:
        th["nominal"] = theory.target_theory(
            **ant, depth_top_m=geo["nominal_nadir"]["depth_top_m"],
            thickness_m=geo["nominal_nadir"]["thickness_m"],
            background=mat.REGOLITH, target=m, transmitted_polarity=TRANSMITTED_POLARITY,
            attenuation_frequency_hz=ATTENUATION_FREQUENCY_HZ, fresnel_wavelength_m=FRESNEL_WAVELENGTH_M)
    if as_built.get("nadir"):
        nb = as_built["nadir"]
        th["as_built_effective"] = theory.target_theory(
            **ant, depth_top_m=nb["depth_top_m"], thickness_m=nb["effective_thickness_m"],
            background=mat.REGOLITH, target=m, transmitted_polarity=TRANSMITTED_POLARITY,
            attenuation_frequency_hz=ATTENUATION_FREQUENCY_HZ, fresnel_wavelength_m=FRESNEL_WAVELENGTH_M)
    rec["theory"].update(th)
    return rec


# =============================================================================
# 一覧・実行スクリプト
# =============================================================================
MANIFEST_FIELDS = ["case_id", "group", "target", "kind", "shape", "size_cm", "depth_m", "variant",
                   "pair_key", "eps_r", "tan_delta", "input", "json", "output", "geometry_vti",
                   "t_top_s", "t_bottom_s", "dt_bottom_top_s"]


def scan_cases(camp: Path) -> List[Dict]:
    recs = []
    for p in camp.rglob("*.json"):
        if p.parent == camp or "scripts" in p.relative_to(camp).parts:
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("schema") == CASE_SCHEMA:
            recs.append(d)

    def key(d):
        g = {"reference": 0, "main": 1, "irregular": 2}[d["group"]]
        t = d.get("target") or {}
        return (g, t.get("depth_top_m", 0), TARGET_ORDER.get(t.get("material"), 99),
                t.get("shape", ""), t.get("size_cm", 0), d["case_id"])
    return sorted(recs, key=key)


def manifest_row(d: Dict) -> Dict:
    t = d.get("target") or {}
    nom = (d.get("theory") or {}).get("nominal") or {}
    tm = (d.get("materials") or {}).get("target") or {}
    f = d["files"]
    return dict(
        case_id=d["case_id"], group=d["group"], target=t.get("material", d.get("reference_kind")),
        kind=t.get("kind", "reference"), shape=t.get("shape"), size_cm=t.get("size_cm"),
        depth_m=t.get("depth_top_m"), variant=(t.get("irregular") or {}).get("variant"),
        pair_key=d.get("pair_key"), eps_r=tm.get("eps_r"), tan_delta=tm.get("tan_delta"),
        input=f["input"], json=f["json"], output=f["output"], geometry_vti=f["geometry_vti"],
        t_top_s=(nom.get("top") or {}).get("t_bistatic_s"),
        t_bottom_s=(nom.get("bottom") or {}).get("t_bistatic_s"),
        dt_bottom_top_s=nom.get("dt_bottom_minus_top_bistatic_s"),
    )


def write_lists(camp: Path) -> int:
    recs = scan_cases(camp)
    rows = [manifest_row(d) for d in recs]
    manifest = dict(campaign=CAMPAIGN, campaign_root=str(camp), updated=now_iso(),
                    n_cases=len(rows), path_note="paths are relative to campaign_root", cases=rows)
    (camp / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(camp / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (camp / "in_list.txt").write_text("".join(r["input"] + "\n" for r in rows), encoding="utf-8")
    (camp / "out_list.txt").write_text("".join(str(camp / r["output"]) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


RUN_ALL_SH = r"""#!/bin/bash
# run_all.sh -- in_list.txt の .in を上から順に計算する
#   完了したケースには <case_id>.done を作り、次回からは飛ばす（途中で止めても再開できる）。
#   使い方:   ./run_all.sh [リストファイル（既定: in_list.txt）]
#   環境変数: PYTHON      gprMax を入れた Python（既定: python）
#             GPRMAX_OPTS gprMax に渡す追加オプション（例: "-gpu"）
#   計算時間は run_history.tsv に追記する。
set -u
cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-python}"
GPRMAX_OPTS="${GPRMAX_OPTS:-}"
LIST="${1:-in_list.txt}"
HIST="run_history.tsv"
[ -f "$HIST" ] || printf "case_id\tstart\tend\tseconds\tstatus\n" > "$HIST"
total=$(grep -c . "$LIST")
n=0
while IFS= read -r inf || [ -n "$inf" ]; do
  [ -z "$inf" ] && continue
  case "$inf" in \#*) continue ;; esac
  n=$((n + 1))
  dir=$(dirname "$inf"); base=$(basename "$inf" .in)
  done_marker="$dir/$base.done"; out="$dir/A-scan/$base.out"; log="$dir/$base.log"
  if [ -f "$done_marker" ]; then
    echo "[$n/$total] skip (done)  $base"; continue
  fi
  if [ -f "$out" ] && [ -f "$log" ] && grep -q "Simulation completed" "$log"; then
    date '+%Y-%m-%dT%H:%M:%S' > "$done_marker"
    echo "[$n/$total] skip (output found)  $base"; continue
  fi
  s0=$(date '+%Y-%m-%dT%H:%M:%S'); t0=$(date +%s)
  echo "[$n/$total] $s0  start  $base"
  if "$PYTHON" -m gprMax "$inf" $GPRMAX_OPTS > "$log" 2>&1 < /dev/null && [ -f "$out" ]; then
    status=ok; date '+%Y-%m-%dT%H:%M:%S' > "$done_marker"
  else
    status=FAILED
  fi
  s1=$(date '+%Y-%m-%dT%H:%M:%S'); t1=$(date +%s)
  printf "%s\t%s\t%s\t%d\t%s\n" "$base" "$s0" "$s1" $((t1 - t0)) "$status" >> "$HIST"
  echo "            -> $status ($((t1 - t0)) s)"
done < "$LIST"
"""

CHECK_GEOMETRY_SH = r"""#!/bin/bash
# check_geometry.sh -- FDTD 計算をせず、形状（.vti）だけを作る（gprMax --geometry-only）
#   本計算の前に、全ケースの配置を確認するために使う。.done は作らない。
#   使い方:   ./check_geometry.sh [リストファイル（既定: in_list.txt）]
#   確認:     python scripts/verify_geometry.py <このフォルダ>
set -u
cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-python}"
LIST="${1:-in_list.txt}"
total=$(grep -c . "$LIST")
n=0; nfail=0
while IFS= read -r inf || [ -n "$inf" ]; do
  [ -z "$inf" ] && continue
  case "$inf" in \#*) continue ;; esac
  n=$((n + 1))
  dir=$(dirname "$inf"); base=$(basename "$inf" .in)
  if "$PYTHON" -m gprMax "$inf" --geometry-only > "$dir/$base.geometry.log" 2>&1 < /dev/null; then
    echo "[$n/$total] ok      $base"
  else
    echo "[$n/$total] FAILED  $base  (see $dir/$base.geometry.log)"; nfail=$((nfail + 1))
  fi
done < "$LIST"
echo "finished: $n cases, $nfail failed"
"""


# =============================================================================
# メイン
# =============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=ROOT, help="親ディレクトリ")
    p.add_argument("--campaign", default=CAMPAIGN, help="キャンペーン名（設定を変えたら新しい名前に）")
    p.add_argument("--dry-run", action="store_true", help="ファイルを作らず、件数・容量・検証結果だけ表示")
    p.add_argument("--force", action="store_true",
                   help="内容が変わった .in / .json を上書きする（計算済みのケースは上書きしない）")
    p.add_argument("--groups", nargs="+", choices=["reference", "main", "irregular"])
    p.add_argument("--targets", nargs="+")
    p.add_argument("--shapes", nargs="+")
    p.add_argument("--depths", nargs="+", type=float)
    p.add_argument("--sizes", nargs="+", type=int, help="サイズ [cm]")
    p.add_argument("--enable-snapshots", action="store_true", help="選んだケースのスナップショットを有効にする")
    p.add_argument("--sec-per-case", type=float, help="1 ケースの計算時間 [s]（合計時間の見積もりに使う）")
    return p.parse_args(argv)


def select(cases: List[Case], a) -> List[Case]:
    out = []
    for c in cases:
        if a.groups and c.group not in a.groups:
            continue
        if c.group != "reference":
            if a.targets and c.target not in a.targets:
                continue
            if a.shapes and c.shape not in a.shapes:
                continue
            if a.depths and not any(abs(c.depth_m - d) < 1e-9 for d in a.depths):
                continue
            if a.sizes and c.size_cm not in a.sizes:
                continue
        elif (a.targets or a.shapes or a.depths or a.sizes) and not a.groups:
            continue        # 絞り込み指定があるときは、明示しない限り参照計算は含めない
        out.append(c)
    return out


def clean_floats(obj, sig: int = 12):
    """JSON に書く前に浮動小数点の誤差（0.20000000000000018 など）を落とす"""
    if isinstance(obj, float):
        return float(f"{obj:.{sig}g}") if math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: clean_floats(v, sig) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_floats(v, sig) for v in obj]
    if isinstance(obj, np.generic):
        return clean_floats(obj.item(), sig)
    return obj


def comparable_json(d: Dict) -> Dict:
    d = dict(d)
    d.pop("generated_at", None)
    d.pop("generator", None)        # スクリプトのハッシュだけが変わった場合は「変更なし」とみなす
    return d


def main(argv=None) -> int:
    a = parse_args(argv)
    camp = (a.root / a.campaign).resolve() if a.root.exists() else (a.root / a.campaign)
    global CAMPAIGN
    CAMPAIGN = a.campaign

    ratio = mat.check_tau_vs_dt(DT_S)
    print(f"campaign: {camp}")
    print(f"domain {DOMAIN_X_M} x {DOMAIN_Y_M} m, dx {DX_M} m, dt {DT_S:.5e} s, "
          f"{ITERATIONS} iterations ({TIME_WINDOW_S*1e9:.0f} ns), tau1/dt = {ratio:.1f}")
    print(f"antenna: Tx x={fm(TX_X_M)}, Rx x={fm(RX_X_M)}, y={fm(ANT_Y_M)} (offset {ANT_OFFSET_M} m)")
    for m in mat.MEDIA.values():
        print(f"  {m.key:9s} eps_r(f0)={m.eps_r:.4f}  tanδ={m.tan_delta:.5f}")

    cases = select(planned_cases(), a)
    if not cases:
        print("no cases selected")
        return 1

    # ---- 形状・理論値・検証 ----
    prepared = []
    problems = []
    for c in cases:
        geo = None if c.group == "reference" else target_geometry(c)
        rec = clean_floats(case_record(c, geo, camp, a.enable_snapshots))
        errs = [] if geo is None else validate_case(c, geo, rec["theory"].get("nominal"))
        if errs:
            problems.append((c.case_id, errs))
        prepared.append((c, geo, rec))

    # ---- 集計 ----
    from collections import Counter
    cnt = Counter((c.group, c.target or c.ref_kind) for c in cases)
    print(f"\nselected cases: {len(cases)}")
    for (g, t), n in sorted(cnt.items()):
        print(f"  {g:10s} {t:12s} {n:4d}")
    vti_bytes = NX * NY * 6
    out_bytes = ITERATIONS * 6 * 4
    print(f"disk (approx.): vti {len(cases)*vti_bytes/1e9:.1f} GB, .out {len(cases)*out_bytes/1e9:.2f} GB"
          f" ({vti_bytes/1e6:.1f} MB + {out_bytes/1e6:.2f} MB per case)")
    if a.sec_per_case:
        tot = a.sec_per_case * len(cases)
        print(f"run time (approx.): {tot/3600:.1f} h = {tot/86400:.2f} days")
    if problems:
        print(f"\nVALIDATION FAILED for {len(problems)} cases:")
        for cid, errs in problems[:30]:
            print(f"  {cid}: " + "; ".join(errs))
        return 2
    print("validation: ok")
    if a.dry_run:
        print("\n(dry run: no files written)")
        return 0

    # ---- campaign.json ----
    camp.mkdir(parents=True, exist_ok=True)
    phys = physics_config()
    cj = camp / "campaign.json"
    if cj.exists():
        old = json.loads(cj.read_text(encoding="utf-8"))
        if json.dumps(old.get("physics"), sort_keys=True) != json.dumps(clean_floats(json.loads(json.dumps(phys))), sort_keys=True):
            if not a.force:
                print("\nERROR: campaign.json の物理設定が今の設定と異なります。"
                      "新しいキャンペーン名を使うか、--force で上書きしてください。")
                return 3
        created = old.get("created", now_iso())
    else:
        created = now_iso()
    sdir = camp / "scripts"
    sdir.mkdir(exist_ok=True)
    hashes = {}
    for name in ["generate_ascan_inputs.py", "materials.py", "theory.py", "grid_geometry.py",
                 "verify_geometry.py", "make_h5_input.py", "README.md"]:
        src = HERE / name
        if src.exists():
            shutil.copy2(src, sdir / name)
            hashes[name] = sha256_of(src)
    campaign = dict(
        campaign=a.campaign, campaign_root=str(camp), created=created, updated=now_iso(),
        generator=dict(version=GENERATOR_VERSION, sha256=hashes),
        environment=dict(python=sys.version.split()[0], numpy=np.__version__, platform=platform.platform(),
                         gprmax_expected="3.1.6"),
        physics=clean_floats(json.loads(json.dumps(phys))),
        plan=dict(targets=TARGETS, shapes=SHAPES, sizes_cm=SIZES_CM, depths_m=DEPTHS_M,
                  irregular=IRREGULAR, snapshot=SNAPSHOT),
    )
    cj.write_text(json.dumps(campaign, indent=1, ensure_ascii=False), encoding="utf-8")

    # ---- ケース ----
    n_new = n_same = n_upd = 0
    protected, conflicts = [], []
    for c, geo, rec in prepared:
        rec["generated_at"] = now_iso()
        rec["generator"]["sha256"] = hashes.get("generate_ascan_inputs.py")
        f = rec["files"]
        d = camp / f["dir"]
        p_in, p_js = camp / f["input"], camp / f["json"]
        text = render_in(c, geo, a.enable_snapshots)
        if p_in.exists() or p_js.exists():
            same_in = p_in.exists() and p_in.read_text(encoding="utf-8") == text
            try:
                old = json.loads(p_js.read_text(encoding="utf-8"))
                same_js = comparable_json(old) == comparable_json(json.loads(json.dumps(rec)))
            except Exception:
                same_js = False
            if same_in and same_js:
                n_same += 1
                continue
            has_output = (camp / f["done_marker"]).exists() or (camp / f["output"]).exists()
            if has_output:
                protected.append(c.case_id)
                continue
            if not a.force:
                conflicts.append(c.case_id)
                continue
            n_upd += 1
        else:
            n_new += 1
        d.mkdir(parents=True, exist_ok=True)
        p_in.write_text(text, encoding="utf-8")
        p_js.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")

    n_total = write_lists(camp)
    for name, body in [("run_all.sh", RUN_ALL_SH), ("check_geometry.sh", CHECK_GEOMETRY_SH)]:
        p = camp / name
        p.write_text(body, encoding="utf-8")
        p.chmod(0o755)

    print(f"\nwritten: new {n_new}, updated {n_upd}, unchanged {n_same}")
    if conflicts:
        print(f"NOT updated (content changed; use --force): {len(conflicts)} cases, e.g. {conflicts[:3]}")
    if protected:
        print(f"NOT updated (output already exists): {len(protected)} cases, e.g. {protected[:3]}")
    print(f"manifest: {n_total} cases in {camp/'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
