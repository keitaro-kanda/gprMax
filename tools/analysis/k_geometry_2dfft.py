#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geometry_2dfft.py
=================

gprMax の地形ファイル（geometry.h5 または geometry.vti）から地下の比誘電率 ε' の分布を作り、
2 次元 FFT で地下構造の波数特性を求める（単体で動く）。図の種類・軸・色・ファイル名は
LPR の解析（lpr_bscan_2dfft.py）と gprMax の B-scan の解析（gprmax_bscan_2dfft.py）と同じ。

B-scan の解析と揃えているもの
    * 解析する範囲：水平 WINDOW_X_M（既定は B-scan の測線と同じ x = 3-7 m）、深さは B-scan の
      解析窓と同じ（地表からの往復時間 30-100 ns ⇔ 深さ 2.6-8.7 m、ε_r = 3 で換算）
    * 区間長（4 m）、窓関数（Hann）、波数の定義、S_x を積分する k_y の帯域（250-750 MHz 相当）
    * 図の表示範囲（k_x, k_y の上限は B-scan のナイキスト波数 89.8 / 116.2 rad/m に合わせる）
    計算は 5 mm 格子のまま行う（補間しない）。地形のナイキスト波数（628 rad/m）は図の外。
    比誘電率 ε' はそのまま（平均を引かない）FFT する。平均（一様な背景）は k = 0 に集まり、
    窓関数のすそで k = 0 のごく近く（数ビン）に漏れる。その範囲は図の縦軸・色の範囲を
    決めるときに除いている（DC_EXCLUDE_BINS）。k_y は媒質の鉛直波数そのもの。

読み込み
    geometry.h5 : /data（材料番号）と、同じフォルダの <名前>_materials.txt（番号順の材料）
    geometry.vti: Material 配列と材料名（<gprMax> ブロック）と、<名前>_materials.txt
    ε' は Debye 分散を含めて EVAL_FREQ_HZ で計算する（CE-4 の帯域中心 433 MHz が既定）。
    地表は、各列で空気（free_space）でない最も上のセルの中央値。

出力（入力ファイルと同じフォルダに OUTPUT_DIRNAME を作る。構成は B-scan の解析と同じ）
    overview_windows.png / mean_<名前>_L<Ls>m.png / compare_lengths_<名前>.png
    spectra_mean.npz / run_info.json / L<Ls>m/<窓>_x<始点>-<終点>m.png
    eps_subsurface.txt（地下の ε'[深さ, x]、5 mm 格子。SAVE_PROCESSED_TXT = False で省略）
"""

import glob
import json
import os
import re
import time
import xml.etree.ElementTree as ET

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

# =============================================================================
# 設定  [EDIT HERE]
# =============================================================================
OUTPUT_DIRNAME = '2D_FFT_geometry'
EVAL_FREQ_HZ = 433e6        # ε' を計算する周波数（Debye 分散のある材料で効く）

# --- B-scan の解析と揃える範囲（gprmax_bscan_2dfft.py の設定と同じ値にすること）
EPS_R = 3.0                 # 深さ ↔ 往復時間の換算に使う比誘電率
WINDOW_X_M = (3.0, 7.0)     # 解析する水平範囲 [m]（領域の x 座標。既定は B-scan の測線の中点の範囲）
                            # 領域全体を使うなら例えば (0.2, 9.8)（4 m 区間が 2 つ取れる）
T_START_NS = 30.0           # 解析窓の上端（地表からの往復時間）[ns]（深さ 2.6 m）
TIME_LEN_NS = 70.0          # 解析窓の長さ [ns]（深さ 6.06 m）
SEG_LENGTHS_M = (4.0,)
SEG_OVERLAP = 0.0
BAND_MHZ = (250.0, 750.0)   # S_x を積分する帯域（B-scan の解析と同じ k_y の範囲になる）
WINDOW_FUNC = 'hann'
BACKGROUND = 'none'
INVALID_FRAC_MAX = 0.01
ZERO_RUN_MIN = 16
SUBTRACT_SEGMENT_MEAN = False   # False：ε' をそのまま FFT する（B-scan の解析では True 相当）

# --- 表示範囲（B-scan の格子のナイキスト波数に合わせる）
PLOT_DX_M = 0.035           # B-scan のトレース間隔 [m] -> k_x の表示上限 pi / 0.035 = 89.8 rad/m
PLOT_DT_NS = 0.3125         # B-scan のサンプリング間隔 [ns] -> k_y の表示上限 116.2 rad/m
DC_EXCLUDE_BINS = 3         # k = 0 から何ビン以内を縦軸・色の範囲の計算から除くか（平均の漏れ）

# --- 出力（B-scan の解析と同じ体裁）
SAVE_SEGMENT_FIGS = True
SAVE_PROCESSED_TXT = True
IMAGE_TITLE = "Relative permittivity ε'"   # 区間ごとの図の左上（B-scan の解析では 'B-scan'）
IMAGE_CMAP = 'viridis'      # ε' の図の色（B-scan の解析では赤青・灰色）
IMAGE_VLIM = None           # ε' の図の色の範囲 (下限, 上限)。None で地下の 0.5-99.5 % 点
KY_MAX_PLOT = None
DB_RANGE = 70.0
SX_YLIM_DB = None
SY_YLIM_DB = None
SPEC_DB_RANGE_MAX = 100.0
WL_TICKS_X = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50)
WL_TICKS_X_MAP = (0.075, 0.1, 0.15, 0.25, 0.5)
WL_TICKS_Y = (0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1)
OVERVIEW_STRIDE = 1
DPI = 130

C0 = 299792458.0
T0_NS = 0.0
DX_M = 0.005                # 実行時に地形の格子間隔で上書きする
DT_NS = 0.0577              # 実行時に「格子 1 つ分の深さの往復時間」で上書きする
WINDOWS = []                # main で設定する


# =============================================================================
# 地形ファイルの読み込みと ε' の分布
# =============================================================================
def parse_materials(path):
    """materials ファイル -> [(name, eps_inf, sigma, [(de, tau), ...]), ...]（ファイルの順）"""
    mats, index = [], {}
    with open(path, encoding='utf-8', errors='replace') as fh:
        for ln in fh:
            ln = ln.strip()
            if ln.startswith('#material:'):
                v = ln.split()
                name = v[5]
                index[name] = len(mats)
                mats.append([name, float(v[1]), float(v[2]), []])
            elif ln.startswith('#add_dispersion_debye:'):
                v = ln.split()
                npo = int(float(v[1]))
                poles = [(float(v[2 + 2 * p]), float(v[3 + 2 * p])) for p in range(npo)]
                for name in v[2 + 2 * npo:]:
                    if name in index:
                        mats[index[name]][3] = poles
    return mats


def eps_real(eps_inf, poles, f):
    w = 2 * np.pi * f
    return eps_inf + sum(de / (1 + (w * tau) ** 2) for de, tau in poles)


def read_vti(path):
    """gprMax の .vti から Material 配列（[x, y]）と {名前: 番号} を読む"""
    raw = open(path, 'rb').read()
    head_end = raw.index(b'<AppendedData')
    header = raw[:head_end].decode('utf-8', errors='replace')
    ext = [int(v) for v in re.search(r'WholeExtent="([^"]+)"', header).group(1).split()]
    nx, ny, nz = ext[1] - ext[0], ext[3] - ext[2], ext[5] - ext[4]
    spacing = [float(v) for v in re.search(r'Spacing="([^"]+)"', header).group(1).split()]
    htype = re.search(r'header_type="([^"]+)"', header)
    hdt = np.dtype('<u8') if (htype and htype.group(1) == 'UInt64') else np.dtype('<u4')
    m = re.search(r'<DataArray type="UInt32" Name="Material" format="appended" offset="(\d+)"', header)
    start = raw.index(b'_', raw.index(b'>', head_end)) + 1 + int(m.group(1))
    nbytes = int(np.frombuffer(raw[start:start + hdt.itemsize], hdt)[0])
    data = np.frombuffer(raw[start + hdt.itemsize:start + hdt.itemsize + nbytes], dtype='<u4')
    material = data.reshape((nz, ny, nx)).transpose(2, 1, 0)[:, :, 0]
    tail = raw[raw.index(b'</VTKFile>'):].decode('utf-8', errors='replace')
    gm = re.search(r'<gprMax>.*</gprMax>', tail, re.S)
    ids = {el.attrib['name']: int(el.text) for el in ET.fromstring(gm.group(0)).findall('Material')}
    return material, spacing[0], ids


def find_materials_file(path):
    stem = os.path.splitext(path)[0]
    cands = [stem + '_materials.txt'] + sorted(glob.glob(os.path.join(os.path.dirname(path),
                                                                     '*_materials.txt')))
    for c in cands:
        if os.path.exists(c) and not os.path.basename(c).startswith('ce4_voxels'):
            return c
    return None


def load_eps(path):
    """比誘電率 ε'[x, y]（y は上向き）、格子間隔、空気のマスク、材料の情報を返す"""
    mat_path = find_materials_file(path)
    if mat_path is None:
        raise SystemExit('材料ファイル（<名前>_materials.txt）が見つかりません')
    mats = parse_materials(mat_path)
    eps_list = np.array([eps_real(m[1], m[3], EVAL_FREQ_HZ) for m in mats])
    names = [m[0] for m in mats]
    air_ids = [i for i, n in enumerate(names) if re.sub(r'\{.*\}$', '', n) == 'free_space']
    if path.lower().endswith('.h5'):
        with h5py.File(path, 'r') as fh:
            idx = fh['/data'][:, :, 0].astype(np.int64)
            dx = float(fh.attrs['dx_dy_dz'][0])
        if idx.min() < 0 or idx.max() >= len(mats):
            raise SystemExit('/data の番号が材料ファイル {} の行数に合いません'.format(mat_path))
        eps = eps_list[idx]
        air = np.isin(idx, air_ids)
    elif path.lower().endswith('.vti'):
        material, dx, ids = read_vti(path)
        by_name = {n: i for i, n in enumerate(names)}
        lut = np.ones(max(ids.values()) + 1)
        air_vti = []
        for n, i in ids.items():
            if n in by_name:
                lut[i] = eps_list[by_name[n]]
            elif re.sub(r'\{.*\}$', '', n) in ('free_space', 'pec'):
                lut[i] = 1.0
            else:
                raise SystemExit('材料 {} の値が材料ファイルにありません'.format(n))
            if re.sub(r'\{.*\}$', '', n) == 'free_space':
                air_vti.append(i)
        eps = lut[material]
        air = np.isin(material, air_vti)
    else:
        raise SystemExit('.h5 か .vti を指定してください')
    return eps, dx, air, mat_path, len(mats)




# #############################################################################
# 以下は gprmax_bscan_2dfft.py（= lpr_bscan_2dfft.py）からの写し。地形用に変えた所には
# 「地形用」の印を付けている（区間の平均を引かない、表示範囲、ε' の図の色）。
# 変更したら 3 つとも直すこと。
# #############################################################################
# #############################################################################
# ここから lpr_bscan_2dfft.py と同じ（解析・描画。変更したら両方を直すこと）
# #############################################################################
# =============================================================================
# スペクトル
# =============================================================================
def window_1d(n, kind=None):
    kind = WINDOW_FUNC if kind is None else kind      # 呼び出し時の設定を使う
    if kind == 'none':
        return np.ones(n)
    if kind == 'hann':
        return np.hanning(n)
    if kind == 'blackmanharris':
        a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
        t = 2 * np.pi * np.arange(n) / (n - 1)
        return a0 - a1 * np.cos(t) + a2 * np.cos(2 * t) - a3 * np.cos(3 * t)
    raise ValueError('WINDOW_FUNC は hann / blackmanharris / none')


def velocity():
    return C0 / np.sqrt(EPS_R)


def dy_m():
    return velocity() * DT_NS * 1e-9 / 2.0      # 1 サンプルあたりの深さ [m]


def ky_to_mhz(ky):
    return np.asarray(ky) * velocity() / (4 * np.pi) / 1e6


def mhz_to_ky(f):
    return 4 * np.pi * np.asarray(f) * 1e6 / velocity()


class Grid:
    """区間（N_y x N_x）の波数軸。S は [k_y >= 0 の行, k_x の全列] だけを持つ"""

    def __init__(self, ny, nx):
        self.ny, self.nx = ny, nx
        self.dy, self.dx = dy_m(), DX_M
        self.kx = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(nx, d=self.dx))
        ky_all = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(ny, d=self.dy))
        self.ky_sel = ky_all >= 0
        self.ky = ky_all[self.ky_sel]
        self.dkx = 2 * np.pi / (nx * self.dx)
        self.dky = 2 * np.pi / (ny * self.dy)
        self.wy, self.wx = window_1d(ny), window_1d(nx)
        self.w2 = np.outer(self.wy, self.wx)
        self.w2mean = float((self.w2 ** 2).mean())
        lo, hi = mhz_to_ky(BAND_MHZ[0]), mhz_to_ky(BAND_MHZ[1])
        self.band = (self.ky >= lo) & (self.ky <= hi)
        self.kx_pos = self.kx > 0
        self.ky_pos = self.ky > 0

    def psd(self, block):
        """block[y, x] -> S[k_y >= 0, k_x]（fftshift 済み）"""
        b = block - block.mean() if SUBTRACT_SEGMENT_MEAN else block.astype(float)   # 地形用
        X = np.fft.fftshift(np.fft.fft2(b * self.w2))
        S = np.abs(X) ** 2 * self.dx * self.dy / ((2 * np.pi) ** 2 * self.ny * self.nx * self.w2mean)
        return S[self.ky_sel]

    def marginals(self, S):
        """片側の 1 次元スペクトル。S_x は帯域内の k_y だけで積分、S_y は全 k_x で積分"""
        # S(-kx, -ky) = S(kx, ky) なので、k_y >= 0 側の和を 2 倍すると k_y 全体の和になる
        Sx_full = 2.0 * S[self.band].sum(axis=0) * self.dky        # k_x 全体（両側）
        Sx = _fold(Sx_full, self.kx)
        Sy = 2.0 * S.sum(axis=1) * self.dkx                          # k_y > 0 の片側
        return Sx, Sy[self.ky_pos]


def _fold(s_full, k):
    """両側スペクトル s_full(k)（fftshift 済み）を片側 s(k) + s(-k)（k > 0）にまとめる"""
    i0 = int(np.argmin(np.abs(k)))                 # k = 0 の位置
    m = np.arange(1, len(k) - i0)                  # k > 0 の番号
    return s_full[i0 + m] + s_full[i0 - m]         # fftshift 後は -k が必ずある


# =============================================================================
# 区間の切り出し
# =============================================================================
def window_block(data, w):
    nt, ntr = data.shape
    it0 = int(round((w['t0_ns'] - T0_NS) / DT_NS))
    ny = int(round(TIME_LEN_NS / DT_NS))
    ix0 = int(round(w['x_m'][0] / DX_M))
    ix1 = int(round(w['x_m'][1] / DX_M))
    if it0 < 0 or it0 + ny > nt or ix0 < 0 or ix1 > ntr or ix1 <= ix0:
        raise SystemExit('解析窓 {} がデータの範囲外です（時刻 {}-{} ns、距離 {}-{} m、データは '
                         '{:.1f}-{:.1f} ns, 0-{:.1f} m）'.format(
                             w['name'], w['t0_ns'], w['t0_ns'] + TIME_LEN_NS, *w['x_m'],
                             T0_NS, T0_NS + nt * DT_NS, ntr * DX_M))
    blk = data[it0:it0 + ny, ix0:ix1].astype(np.float64)
    if BACKGROUND == 'window':
        good = ~invalid_mask(blk)
        mean_tr = np.where(good, blk, 0).sum(axis=1) / np.maximum(good.sum(axis=1), 1)
        blk = np.where(good, blk - mean_tr[:, None], blk)
    return blk, it0, ix0


def invalid_mask(blk, n=None):
    """無効なサンプル：NaN / inf と、時間方向に 0 が n 点以上続く所（地形補正の詰め物や
    欠けたトレース）。単独の 0 は有効な値として扱う。"""
    n = ZERO_RUN_MIN if n is None else n
    bad = ~np.isfinite(blk)
    z = (blk == 0).astype(np.int32)
    if n <= 1:
        return bad | z.astype(bool)
    ny = z.shape[0]
    if ny < n:
        return bad
    c = np.concatenate([np.zeros((1, z.shape[1]), np.int32), np.cumsum(z, axis=0)], axis=0)
    full = (c[n:] - c[:-n]) == n                  # [ny - n + 1, nx]：i から n 点がすべて 0
    cov = np.zeros((ny + 1, z.shape[1]), np.int32)  # 長さ n の窓が覆うサンプルに印を付ける
    cov[:ny - n + 1] += full
    cov[n:ny + 1] -= full
    run = np.cumsum(cov, axis=0)[:ny] > 0
    return bad | run


def segments(w, blk, ix0, Ls, bad_all):
    nx = int(round(Ls / DX_M))
    hop = max(1, int(round(nx * (1.0 - SEG_OVERLAP))))
    out = []
    for s in range(0, blk.shape[1] - nx + 1, hop):
        frac = float(bad_all[:, s:s + nx].mean())
        x0 = (ix0 + s) * DX_M
        out.append(dict(win=w['name'], x0=x0, x1=x0 + nx * DX_M, col=s, nx=nx,
                        invalid=frac, ok=frac <= INVALID_FRAC_MAX))
    return out


# =============================================================================
# 図
# =============================================================================
def _db(a):
    return 10 * np.log10(np.maximum(a, 1e-300))


# 線の色は名前で呼べる色にする（論文で「赤線」「青線」と言えるように）
WIN_COLORS = ['r', 'b', 'g', 'm', 'c', 'y']                    # 解析窓（A, B, C, ... の順）
COLOR_NAMES = {'r': 'red', 'b': 'blue', 'g': 'green', 'm': 'magenta', 'c': 'cyan',
               'y': 'yellow', 'k': 'black', 'orange': 'orange', 'gray': 'gray'}
LENGTH_COLORS = ['r', 'g', 'b', 'm', 'c', 'y']                 # 区間長（短い順）
MEAN_COLOR = 'k'                                               # 平均は黒
GROUP_STYLES = ['-', '--', ':', '-.']                          # グループは線種で区別
DK_STYLE = dict(color='gray', ls='-.', lw=1.0)                 # 分解能 Δk
KN_STYLE = dict(color='gray', ls=(0, (1, 1.5)), lw=1.6)        # ナイキスト波数 k_N
BAND_FILL = dict(color='0.85', alpha=0.6)                      # LPR 帯域（薄い灰色）
BAND_LINE = dict(color='w', ls='--', lw=0.9)                   # 2 次元図の帯域の線（白）

ALL_NAME = 'all'          # 全解析窓をまとめた平均の名前
LIM = {}                  # 全図に共通の軸の範囲（main で決める）


def _group(w):
    return w.get('group', 'default')


def groups():
    out = []
    for w in WINDOWS:
        if _group(w) not in out:
            out.append(_group(w))
    return out


def _win_color(name):
    i = [w['name'] for w in WINDOWS].index(name)
    return WIN_COLORS[i % len(WIN_COLORS)]


def _group_style(gname):
    gl = groups()
    return GROUP_STYLES[gl.index(gname) % len(GROUP_STYLES)] if gname in gl else '-'


def ky_max():
    kn = np.pi / (C0 / np.sqrt(EPS_R) * PLOT_DT_NS * 1e-9 / 2) if PLOT_DT_NS else np.pi / dy_m()  # 地形用
    return kn if KY_MAX_PLOT is None else min(KY_MAX_PLOT, kn)


def _lw(ls, lw):
    return lw * 1.4 if ls == ':' else lw            # 点線は細く見えるので少し太く


def _wl_axis(ax, which, wl_ticks, label, symmetric=False):
    """波長 λ = 2 pi / k の第 2 軸（上または右）。λ は k の逆数で単調でない（±k）場合も
    あるので、k の軸と同じ目盛り位置に λ の値を文字で付ける。軸の範囲を決めてから呼ぶ。"""
    ident = (lambda v: v, lambda v: v)
    sec = ax.secondary_xaxis('top', functions=ident) if which == 'x' else \
        ax.secondary_yaxis('right', functions=ident)
    lo, hi = sorted(ax.get_xlim() if which == 'x' else ax.get_ylim())
    pos, lab = [], []
    for wl in wl_ticks:
        k = 2 * np.pi / wl
        for kk in ((k, -k) if symmetric else (k,)):
            if lo <= kk <= hi:
                pos.append(kk)
                lab.append(f'{wl:g}')
    if which == 'x':
        sec.set_xticks(pos, labels=lab)
        sec.set_xlabel(label)
    else:
        sec.set_yticks(pos, labels=lab)
        sec.set_ylabel(label)
    sec.tick_params(which='minor', length=0)
    return sec


def draw_map(ax, g, S):
    m = g.ky <= ky_max() + 1e-9
    im = ax.imshow(_db(S[m]), origin='lower', aspect='auto', cmap='magma',
                   extent=[g.kx[0] - g.dkx / 2, g.kx[-1] + g.dkx / 2,
                           g.ky[m][0] - g.dky / 2, g.ky[m][-1] + g.dky / 2],
                   vmin=LIM['map'][0], vmax=LIM['map'][1])
    for f in BAND_MHZ:
        ax.axhline(mhz_to_ky(f), **BAND_LINE)
    ax.set_xlim(*LIM['kx_map'])
    ax.set_ylim(*LIM['ky_map'])
    ax.set_xlabel('k_x [rad/m]')
    ax.set_ylabel('k_y [rad/m]')
    _wl_axis(ax, 'x', WL_TICKS_X_MAP, 'λ_x [m]', symmetric=True)
    _wl_axis(ax, 'y', WL_TICKS_Y, f'λ_y [m] (ε_r = {EPS_R:g})')
    ax.set_title('S(k_x, k_y)')
    ax.legend(handles=[Line2D([], [], label=f'{BAND_MHZ[0]:g}–{BAND_MHZ[1]:g} MHz', **BAND_LINE)],
              loc='upper right', fontsize=7.5, facecolor='0.4', labelcolor='w', framealpha=0.7)
    return im


def _res_lines(ax, dk, kn, axis, show_dk=True):
    """分解能 Δk（灰色の一点鎖線）とナイキスト波数 k_N（灰色の点線）の縦線と凡例"""
    hs = []
    if show_dk:
        ax.axvline(dk, **DK_STYLE)
        hs.append(Line2D([], [], label=f'Δk_{axis} = {dk:.3f} rad/m', **DK_STYLE))
    ax.axvline(kn, **KN_STYLE)
    off = kn > (LIM['kx'][1] if axis == 'x' else LIM['ky'][1])                 # 地形用
    hs.append(Line2D([], [], label=f'k_N,{axis} = {kn:.1f} rad/m' + (' (off scale)' if off else ''),
                     **KN_STYLE))
    return hs


def _curve_handles(curves):
    return [Line2D([], [], color=c['color'], lw=_lw(c.get('ls', '-'), c.get('lw', 1.3)),
                   ls=c.get('ls', '-'), label=c['label']) for c in curves if c.get('label')]


def draw_sx(ax, g, curves, dk_lines=None, plot=True):
    """S_x の図。plot=False なら線は描かず（呼び出し側で描いた前提）、軸・凡例・縦線だけ整える"""
    k = g.kx[g.kx_pos]
    ax.set_xscale('log')
    for c in curves if plot else []:
        ls = c.get('ls', '-')
        ax.plot(k, _db(c['s']), color=c['color'], lw=_lw(ls, c.get('lw', 1.3)), ls=ls)
    hs = []
    if dk_lines is None:
        hs = _res_lines(ax, g.dkx, np.pi / DX_M, 'x')
    else:                                           # 区間長ごとの Δk_x（色 = 区間長）
        for dk, col in dk_lines:
            ax.axvline(dk, color=col, ls='-.', lw=1.0)
        hs = _res_lines(ax, None, np.pi / DX_M, 'x', show_dk=False)
        hs.insert(0, Line2D([], [], color='k', ls='-.', lw=1.0, label='Δk_x (colour = L)'))
    ax.set_xlim(*LIM['kx'])
    ax.set_ylim(*LIM['sx'])
    _wl_axis(ax, 'x', WL_TICKS_X, 'λ_x [m]')
    ax.set_xlabel('k_x [rad/m]')
    ax.set_ylabel('S_x [dB]')
    ax.set_title(f'S_x(k_x), k_y in {BAND_MHZ[0]:g}–{BAND_MHZ[1]:g} MHz')
    ax.grid(alpha=0.3, which='both')
    ax.legend(handles=_curve_handles(curves) + hs, fontsize=7.5, loc='lower left')


def draw_sy(ax, g, curves, plot=True):
    k = g.ky[g.ky_pos]
    m = k <= ky_max() + 1e-9
    for c in curves if plot else []:
        ls = c.get('ls', '-')
        ax.plot(k[m], _db(c['s'][m]), color=c['color'], lw=_lw(ls, c.get('lw', 1.3)), ls=ls)
    ax.axvspan(mhz_to_ky(BAND_MHZ[0]), mhz_to_ky(BAND_MHZ[1]), **BAND_FILL)
    hs = _res_lines(ax, g.dky, np.pi / dy_m(), 'y')
    ax.set_xlim(*LIM['ky'])
    ax.set_ylim(*LIM['sy'])
    ax.set_xlabel('k_y [rad/m]')
    ax.set_ylabel('S_y [dB]')
    ax.set_title('S_y(k_y)')
    ax.grid(alpha=0.3)
    _wl_axis(ax, 'x', WL_TICKS_Y, f'λ_y [m] (ε_r = {EPS_R:g})')
    h0 = _curve_handles(curves)
    h0.append(Patch(label=f'{BAND_MHZ[0]:g}–{BAND_MHZ[1]:g} MHz', **BAND_FILL))
    ax.legend(handles=h0 + hs, fontsize=7.5, loc='lower center')


def plot_overview(data, outdir, amp):
    nt, ntr = data.shape
    st = max(1, OVERVIEW_STRIDE)
    img = np.nan_to_num(data[:, ::st])
    fig, ax = plt.subplots(figsize=(16, 5.5), layout='constrained')
    # 解析窓の枠の色（r, b, g, ...）が見分けられるよう、全体図の B-scan は灰色で描く
    vlim = IMAGE_VLIM or (-amp, amp)                                           # 地形用
    imo = ax.imshow(img, aspect='auto', cmap=IMAGE_CMAP or 'gray', vmin=vlim[0], vmax=vlim[1],
                    extent=[0, ntr * DX_M, T0_NS + nt * DT_NS, T0_NS])
    if IMAGE_CMAP:
        fig.colorbar(imo, ax=ax, pad=0.06, label="ε'")
    for w in WINDOWS:
        col = _win_color(w['name'])
        ax.add_patch(Rectangle((w['x_m'][0], w['t0_ns']), w['x_m'][1] - w['x_m'][0], TIME_LEN_NS,
                               fill=False, ec=col, lw=2, ls=_group_style(_group(w))))
        ax.text(w['x_m'][0], w['t0_ns'] - 4, w['name'], color=col, fontsize=12, va='bottom',
                fontweight='bold')
    ax.legend(handles=[Line2D([], [], color='k', lw=2, ls=_group_style(gn), label=gn)
                       for gn in groups()], loc='upper right', fontsize=9, framealpha=0.85)
    ax.set_xlabel('Moving distance [m]')
    ax.set_ylabel('Time [ns]')
    sec = ax.secondary_yaxis('right', functions=(lambda t: t * 1e-9 * velocity() / 2,
                                                 lambda d: d * 2 / velocity() * 1e9))
    sec.set_ylabel(f'Depth [m] (ε_r = {EPS_R:g})')
    ax.set_title(f'Analysis windows ({TIME_LEN_NS:g} ns each)')
    fig.savefig(os.path.join(outdir, 'overview_windows.png'), dpi=DPI)
    plt.close(fig)


def _lname(Ls):
    return f'{Ls:03.0f}' if float(Ls).is_integer() else f'{Ls:05.1f}'.replace('.', 'p')


def _win_curves(res, show_group):
    out = []
    for name, r in res['by_win'].items():
        gw = _group(next(w for w in WINDOWS if w['name'] == name))
        lab = f'{name}, {gw} ({r["n"]})' if show_group else f'{name} ({r["n"]})'
        out.append((name, gw, lab, r))
    return out


def plot_mean(outdir, gname, Ls, g, res):
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.8), layout='constrained',
                            gridspec_kw={'width_ratios': [1.2, 1, 1]})
    im = draw_map(axs[0], g, res['S'])
    fig.colorbar(im, ax=axs[0], label='dB')
    ls0 = '-' if gname == ALL_NAME else _group_style(gname)
    cx = [dict(s=res['Sx'], color=MEAN_COLOR, lw=2.2, ls=ls0,
               label=f'{gname} mean ({res["n"]} segments)')]
    cy = [dict(s=res['Sy'], color=MEAN_COLOR, lw=2.2, ls=ls0,
               label=f'{gname} mean ({res["n"]} segments)')]
    for name, gw, lab, r in _win_curves(res, gname == ALL_NAME):
        col, ls = _win_color(name), _group_style(gw)
        cx.append(dict(s=r['Sx'], color=col, lw=1.0, ls=ls, label=lab))
        cy.append(dict(s=r['Sy'], color=col, lw=1.0, ls=ls, label=lab))
    draw_sx(axs[1], g, cx)
    draw_sy(axs[2], g, cy)
    names = ', '.join(res['by_win'])
    fig.suptitle(f'Mean spectrum, {gname}, L = {Ls:g} m (windows {names}, {TIME_LEN_NS:g} ns, '
                 f'window: {WINDOW_FUNC})', fontsize=13)
    fig.savefig(os.path.join(outdir, f'mean_{gname}_L{_lname(Ls)}m.png'), dpi=DPI)
    plt.close(fig)


def plot_segment(path, seg, blk, g, S, Sx, Sy, amp, t0_ns):
    fig = plt.figure(figsize=(16, 10.5), layout='constrained')
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, 0])
    vlim = IMAGE_VLIM or (-amp, amp)                                           # 地形用
    imb = ax.imshow(blk, aspect='auto', cmap=IMAGE_CMAP or 'seismic', vmin=vlim[0], vmax=vlim[1],
                    extent=[seg['x0'], seg['x1'], t0_ns + TIME_LEN_NS, t0_ns])
    if IMAGE_CMAP:
        fig.colorbar(imb, ax=ax, pad=0.12, label="ε'")
    ax.set_xlabel('Moving distance [m]')
    ax.set_ylabel('Time [ns]')
    ax.set_title(IMAGE_TITLE)                                                  # 地形用（元は 'B-scan'）
    sec = ax.secondary_yaxis('right', functions=(lambda t: t * 1e-9 * velocity() / 2,
                                                 lambda d: d * 2 / velocity() * 1e9))
    sec.set_ylabel(f'Depth [m] (ε_r = {EPS_R:g})')
    ax = fig.add_subplot(gs[0, 1])
    im = draw_map(ax, g, S)
    fig.colorbar(im, ax=ax, label='dB')
    col = _win_color(seg['win'])
    draw_sx(fig.add_subplot(gs[1, 0]), g, [dict(s=Sx, color=col, label=f'{seg["win"]}')])
    draw_sy(fig.add_subplot(gs[1, 1]), g, [dict(s=Sy, color=col, label=f'{seg["win"]}')])
    fig.suptitle(f'Window {seg["win"]} ({seg["group"]}), x = {seg["x0"]:.2f}–{seg["x1"]:.2f} m '
                 f'(L = {seg["x1"] - seg["x0"]:.2f} m), t = {t0_ns:g}–{t0_ns + TIME_LEN_NS:g} ns',
                 fontsize=13)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_compare_lengths(outdir, gname, res_by_L):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.0), layout='constrained')
    cx, cy, dks = [], [], []
    for i, (Ls, (g, res)) in enumerate(res_by_L.items()):
        col = LENGTH_COLORS[i % len(LENGTH_COLORS)]
        cx.append((g, dict(s=res['Sx'], color=col, lw=1.5,
                           label=f'L = {Ls:g} m ({res["n"]} seg.), Δk_x = {g.dkx:.3f}')))
        cy.append((g, dict(s=res['Sy'], color=col, lw=1.5, label=f'L = {Ls:g} m')))
        dks.append((g.dkx, col))
    # 区間長ごとに k 軸が違うので 1 本ずつ描き、凡例と範囲は最後にまとめる
    for g, c in cx:
        axs[0].semilogx(g.kx[g.kx_pos], _db(c['s']), color=c['color'], lw=c['lw'])
    draw_sx(axs[0], cx[0][0], [c for _, c in cx], dk_lines=dks, plot=False)
    for g, c in cy:
        k = g.ky[g.ky_pos]
        m = k <= ky_max() + 1e-9
        axs[1].plot(k[m], _db(c['s'][m]), color=c['color'], lw=c['lw'])
    draw_sy(axs[1], cy[0][0], [c for _, c in cy], plot=False)
    fig.suptitle(f'Mean spectra for different segment lengths ({gname})', fontsize=13)
    fig.savefig(os.path.join(outdir, f'compare_lengths_{gname}.png'), dpi=DPI)
    plt.close(fig)


def plot_compare_groups(outdir, Ls, res_by_group):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.0), layout='constrained')
    curves_x, curves_y = [], []
    g0 = None
    for gname, (g, res) in res_by_group.items():
        g0 = g
        ls = _group_style(gname)
        curves_x.append(dict(s=res['Sx'], color=MEAN_COLOR, lw=1.8, ls=ls,
                             label=f'{gname} mean ({res["n"]})'))
        curves_y.append(dict(s=res['Sy'], color=MEAN_COLOR, lw=1.8, ls=ls,
                             label=f'{gname} mean ({res["n"]})'))
        for name, gw, lab, r in _win_curves(res, False):
            curves_x.append(dict(s=r['Sx'], color=_win_color(name), lw=1.0, ls=ls, label=lab))
            curves_y.append(dict(s=r['Sy'], color=_win_color(name), lw=1.0, ls=ls, label=lab))
    draw_sx(axs[0], g0, curves_x)
    draw_sy(axs[1], g0, curves_y)
    fig.suptitle(f'Mean spectra by group, L = {Ls:g} m', fontsize=13)
    fig.savefig(os.path.join(outdir, f'compare_groups_L{_lname(Ls)}m.png'), dpi=DPI)
    plt.close(fig)


def set_common_limits(all_L):
    """全図に共通の軸の範囲を決める（横軸・縦軸・2 次元図の色）"""
    kn_x = np.pi / (PLOT_DX_M or DX_M)                                        # 地形用：表示上限
    dkx_min = min(d['g'].dkx for d in all_L.values())
    LIM['kx'] = (0.8 * dkx_min, 1.25 * kn_x)
    LIM['kx_map'] = (-kn_x, kn_x)
    LIM['ky'] = (0.0, ky_max() * 1.02)
    LIM['ky_map'] = (0.0, ky_max())
    tops_x, tops_y, bots_x, bots_y, map_max = [], [], [], [], []
    for d in all_L.values():
        g = d['g']
        mky = (g.ky[g.ky_pos] <= ky_max() + 1e-9) & (g.ky[g.ky_pos] >= DC_EXCLUDE_BINS * g.dky)  # 地形用
        mkx = (g.kx[g.kx_pos] <= LIM['kx'][1]) & (g.kx[g.kx_pos] >= DC_EXCLUDE_BINS * g.dkx)       # 地形用
        dc = (g.ky[:, None] < DC_EXCLUDE_BINS * g.dky) & (np.abs(g.kx)[None, :] < DC_EXCLUDE_BINS * g.dkx)
        for sx, sy in d['seg_marg']:                       # 区間ごとの線（上端を決める）
            tops_x.append(_db(sx[mkx]).max())
            tops_y.append(_db(sy[mky]).max())
        for byL in d['res'].values():                      # 平均の線（下端を決める）
            curves = [(byL['Sx'], byL['Sy'])] + [(r['Sx'], r['Sy']) for r in byL['by_win'].values()]
            for sx, sy in curves:
                bots_x.append(_db(sx[mkx]).min())
                bots_y.append(_db(sy[mky]).min())
            Sm = np.where(dc, 0.0, byL['S'])                                   # 地形用：k = 0 付近を除く
            map_max.append(_db(Sm[g.ky <= ky_max() + 1e-9]).max())
    for key, user, tops, bots in (('sx', SX_YLIM_DB, tops_x, bots_x), ('sy', SY_YLIM_DB, tops_y, bots_y)):
        if user is not None:
            LIM[key] = tuple(user)
        else:
            top = float(max(tops)) + 3.0
            bot = max(float(min(bots)) - 3.0, top - SPEC_DB_RANGE_MAX)
            LIM[key] = (bot, top)
    vmax = float(max(map_max)) + 5.0
    LIM['map'] = (vmax - DB_RANGE, vmax)
    print('共通の軸: k_x {:.3f}-{:.1f}、k_y 0-{:.1f} rad/m、S_x {:.1f}-{:.1f} dB、S_y {:.1f}-{:.1f} dB、'
          '2 次元図 {:.1f}-{:.1f} dB'.format(LIM['kx'][0], LIM['kx'][1], LIM['ky'][1], *LIM['sx'],
                                          *LIM['sy'], *LIM['map']))


def run_analysis(data, outdir, source, extra_info=None):
    """前処理した B-scan（data[時間, トレース]）を解析して outdir に図と数値を出す"""
    t_all = time.time()
    os.makedirs(outdir, exist_ok=True)
    nt, ntr = data.shape
    n_lead0 = int(np.argmax(np.any(np.nan_to_num(data) != 0, axis=1)))
    print('データ: {} 行（時間）x {} 列（トレース）'.format(nt, ntr))
    print('  時間 {:.2f}-{:.2f} ns（DT_NS = {} ns、T0_NS = {} ns）'.format(
        T0_NS, T0_NS + nt * DT_NS, DT_NS, T0_NS))
    print('  距離 0-{:.1f} m（DX_M = {} m）'.format(ntr * DX_M, DX_M))
    print('  NaN {} 個 / 先頭から全トレースで 0 の行 {} 行'.format(
        int(np.isnan(data).sum()), n_lead0))
    print('  深さ換算 ε_r = {}、1 サンプル {:.4f} m、v = {:.4f} m/ns'.format(
        EPS_R, dy_m(), velocity() * 1e-9))
    print('  ナイキスト波数 k_N,x = {:.2f} rad/m、k_N,y = {:.2f} rad/m（{:.0f} MHz）'.format(
        np.pi / DX_M, np.pi / dy_m(), float(ky_to_mhz(np.pi / dy_m()))))
    if nt > 5 * ntr:
        print('  注意: 行数が列数よりかなり多い。各列が A-scan になっているか確認すること')

    finite = data[np.isfinite(data) & (data != 0)]
    amp = float(np.percentile(np.abs(finite[::max(1, finite.size // 2_000_000)]), 99))
    del finite
    plot_overview(data, outdir, amp)
    print('-> {}'.format(os.path.join(outdir, 'overview_windows.png')))

    blocks = {}
    for w in WINDOWS:
        blk, it0, ix0 = window_block(data, w)
        bad = invalid_mask(blk)
        blocks[w['name']] = (w, blk, ix0, bad)
        print('解析窓 {}（{}）: x = {:g}-{:g} m（{} トレース）、t = {:g}-{:g} ns（{} 点）、'
              '無効なサンプル {:.2%}（単独の 0 は {:.2%}）'.format(
                  w['name'], _group(w), *w['x_m'], blk.shape[1], w['t0_ns'],
                  w['t0_ns'] + TIME_LEN_NS, blk.shape[0], float(bad.mean()),
                  float(((blk == 0) & ~bad).mean())))
    del data

    gl = groups()
    # 平均をとる単位：各グループと、グループが 2 つ以上あれば全解析窓をまとめたもの（all）
    avg_units = gl + ([ALL_NAME] if len(gl) > 1 else [])
    info = dict(settings={k: v for k, v in globals().items()
                          if k.isupper() and isinstance(v, (int, float, str, tuple, list,
                                                            dict, type(None)))},
                data_path=source, shape=[nt, ntr], extra=extra_info or {},
                nyquist=dict(kx=np.pi / DX_M, ky=np.pi / dy_m()), lengths={})

    # ---- 1. すべての区間長でスペクトルを計算する（軸の範囲を全図で揃えるため、描く前に全部）
    all_L = {}
    for Ls in SEG_LENGTHS_M:
        ny = int(round(TIME_LEN_NS / DT_NS))
        nx = int(round(Ls / DX_M))
        g = Grid(ny, nx)
        segs, spectra = [], []
        for name, (w, blk, ix0, bad) in blocks.items():
            for s in segments(w, blk, ix0, Ls, bad):
                s['group'] = _group(w)
                segs.append(s)
                spectra.append(g.psd(np.nan_to_num(blk[:, s['col']:s['col'] + nx]))
                               if s['ok'] else None)
        good = [i for i, sp in enumerate(spectra) if sp is not None]
        print('L = {:g} m: 区間 {}（除外 {}）、Δk_x = {:.3f} rad/m、Δk_y = {:.3f} rad/m'.format(
            Ls, len(good), len(segs) - len(good), g.dkx, g.dky))
        for w in WINDOWS:
            n_all = sum(1 for s in segs if s['win'] == w['name'])
            n_ok = sum(1 for i in good if segs[i]['win'] == w['name'])
            msg = '    {}: 使用 {} / {}'.format(w['name'], n_ok, n_all)
            if n_ok == 0:
                msg += '  ** 全区間が除外されたので、この窓はどの図にも入りません **'
            print(msg)
        res_g = {}
        for gname in avg_units:
            gi = [i for i in good if gname == ALL_NAME or segs[i]['group'] == gname]
            if not gi:
                continue
            Smean = np.mean([spectra[i] for i in gi], axis=0)
            Sx, Sy = g.marginals(Smean)
            res = dict(S=Smean, Sx=Sx, Sy=Sy, n=len(gi), by_win={})
            for w in WINDOWS:
                idx = [i for i in gi if segs[i]['win'] == w['name']]
                if idx:
                    sx, sy = g.marginals(np.mean([spectra[i] for i in idx], axis=0))
                    res['by_win'][w['name']] = dict(Sx=sx, Sy=sy, n=len(idx))
            res_g[gname] = res
        seg_marg = [g.marginals(spectra[i]) for i in good]
        all_L[Ls] = dict(g=g, nx=nx, segs=segs, spectra=spectra, good=good, res=res_g,
                         seg_marg=seg_marg)
        info['lengths'][str(Ls)] = dict(
            n_used=len(good), n_excluded=len(segs) - len(good), dkx=g.dkx, dky=g.dky, nx=nx, ny=ny,
            segments=[{k: v for k, v in s.items() if k != 'col'} for s in segs])
    if not any(d['good'] for d in all_L.values()):
        raise SystemExit('有効な区間がありません')

    # ---- 2. 全図に共通の軸の範囲
    set_common_limits(all_L)

    # ---- 3. 図
    for Ls, d in all_L.items():
        t = time.time()
        g = d['g']
        for gname, res in d['res'].items():
            plot_mean(outdir, gname, Ls, g, res)
            print('  L = {:g} m, {}: {} 区間 -> {}'.format(
                Ls, gname, res['n'], os.path.join(outdir, f'mean_{gname}_L{_lname(Ls)}m.png')))
        here = {gname: (g, d['res'][gname]) for gname in gl if gname in d['res']}   # all は除く
        if len(here) > 1:
            plot_compare_groups(outdir, Ls, here)
        if SAVE_SEGMENT_FIGS:
            sub = os.path.join(outdir, f'L{_lname(Ls)}m')
            os.makedirs(sub, exist_ok=True)
            for j, i in enumerate(d['good']):
                s = d['segs'][i]
                w, blk, ix0, _ = blocks[s['win']]
                sx, sy = d['seg_marg'][j]
                fn = '{}_x{:07.2f}-{:07.2f}m.png'.format(s['win'], s['x0'], s['x1'])
                plot_segment(os.path.join(sub, fn), s,
                             np.nan_to_num(blk[:, s['col']:s['col'] + d['nx']]),
                             g, d['spectra'][i], sx, sy, amp, w['t0_ns'])
            print('  -> {}/（{} 枚、{:.0f} s）'.format(sub, len(d['good']), time.time() - t))

    for gname in avg_units:
        byL = {Ls: (d['g'], d['res'][gname]) for Ls, d in all_L.items() if gname in d['res']}
        if byL:
            plot_compare_lengths(outdir, gname, byL)
    np.savez(os.path.join(outdir, 'spectra_mean.npz'),
             **{f'{gname}_L{_lname(Ls)}_{key}': val
                for Ls, d in all_L.items() for gname, r in d['res'].items()
                for key, val in (('kx', d['g'].kx), ('ky', d['g'].ky), ('S', r['S']),
                                 ('kx_pos', d['g'].kx[d['g'].kx_pos]), ('Sx', r['Sx']),
                                 ('ky_pos', d['g'].ky[d['g'].ky_pos]), ('Sy', r['Sy']),
                                 ('n', np.array(r['n'])))})
    info['common_limits'] = {k: list(v) for k, v in LIM.items()}
    info['colors'] = {w['name']: COLOR_NAMES.get(_win_color(w['name']), _win_color(w['name']))
                      for w in WINDOWS}
    with open(os.path.join(outdir, 'run_info.json'), 'w', encoding='utf-8') as fh:
        json.dump(info, fh, ensure_ascii=False, indent=1, default=str)
    print('線の色: ' + ', '.join('{} = {}'.format(k, v) for k, v in info['colors'].items()))
    print('完了（{:.0f} s）。出力先: {}'.format(time.time() - t_all, outdir))


# #############################################################################
# ここまで lpr_bscan_2dfft.py と同じ
# #############################################################################


# =============================================================================
def main():
    #* Get input parameters
    print('geometry.h5 または geometry.vti のパスを入力してください:')
    in_path = input().strip().strip("'\"")
    if not os.path.exists(in_path):
        print('エラー: 指定されたファイルが存在しません')
        exit(1)
    in_path = os.path.abspath(in_path)
    outdir = os.path.join(os.path.dirname(in_path), OUTPUT_DIRNAME)
    os.makedirs(outdir, exist_ok=True)

    eps, dx_g, air, mat_path, n_mat = load_eps(in_path)
    nx_g, ny_g = eps.shape
    print('読み込み: {}（{} x {} セル、格子 {} m、材料 {} 種、{}）'.format(
        os.path.basename(in_path), nx_g, ny_g, dx_g, n_mat, os.path.basename(mat_path)))

    # ---- 1. 地表（各列で空気でない最も上のセル。中央値）と、地下の ε'[x, 深さ]
    solid = ~air
    top = np.array([np.max(np.where(col)[0]) if col.any() else -1 for col in solid])
    top = top[top >= 0]
    if top.size == 0:
        raise SystemExit('地下（空気でないセル）が見つかりません')
    iy_surf = int(np.median(top)) + 1                          # 地表の y（セルの境界の番号）
    sub = eps[:, :iy_surf][:, ::-1]                            # [x, 深さ]、深さ 0 = 地表直下
    print('地表 y = {:.3f} m（地下 {:.3f} m）、ε\' {:.3f}-{:.3f}（平均 {:.4f}、標準偏差 {:.4f}）@ {:g} MHz'.format(
        iy_surf * dx_g, iy_surf * dx_g, sub.min(), sub.max(), sub.mean(), sub.std(),
        EVAL_FREQ_HZ / 1e6))

    # ---- 2. 5 mm 格子のまま、[深さ（往復時間）, x] の配列にする
    global DX_M, DT_NS, WINDOWS, IMAGE_VLIM
    v = C0 / np.sqrt(EPS_R)
    DX_M = dx_g
    DT_NS = 2.0 * dx_g / v * 1e9                               # 格子 1 つ分の深さの往復時間
    data = sub.T.astype(np.float64)                            # [深さ, x]、地表 = 0 ns
    if IMAGE_VLIM is None:
        IMAGE_VLIM = tuple(np.percentile(sub, [0.5, 99.5]))
    print('格子 {} m のまま（深さ 1 セル = 往復 {:.4f} ns）。地下 {} x {} セル'.format(
        dx_g, DT_NS, data.shape[1], data.shape[0]))
    if T_START_NS + TIME_LEN_NS > data.shape[0] * DT_NS:
        raise SystemExit('解析窓が地下の範囲を超えています')

    # ---- 3. 解析窓（列 j の左端が x = j * dx。図の Moving distance = 領域の x 座標）
    label = os.path.basename(os.path.dirname(in_path)) or 'geometry'
    WINDOWS = [dict(name='S', x_m=tuple(WINDOW_X_M), t0_ns=T_START_NS, group=label)]
    print('解析窓: x = {:.3f}-{:.3f} m、深さ {:.3f}-{:.3f} m（往復 {}-{} ns）'.format(
        *WINDOW_X_M, T_START_NS * 1e-9 * v / 2, (T_START_NS + TIME_LEN_NS) * 1e-9 * v / 2,
        T_START_NS, T_START_NS + TIME_LEN_NS))

    if SAVE_PROCESSED_TXT:
        p = os.path.join(outdir, 'eps_subsurface.txt')
        np.savetxt(p, data, fmt='%.6f')
        print("-> {}（地下の ε'[深さ, x]、{} m 格子）".format(p, dx_g))

    extra = dict(input_file=in_path, materials_file=mat_path, eval_freq_hz=EVAL_FREQ_HZ,
                 grid_m=dx_g, surface_y_m=iy_surf * dx_g, eps_mean=float(sub.mean()),
                 eps_std=float(sub.std()), window_x_m=list(WINDOW_X_M),
                 plot_nyquist=dict(dx_m=PLOT_DX_M, dt_ns=PLOT_DT_NS),
                 subtract_segment_mean=SUBTRACT_SEGMENT_MEAN)
    run_analysis(data, outdir, in_path, extra_info=extra)


if __name__ == '__main__':
    main()
