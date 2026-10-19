#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geometry_2dfft.py
=================

gprMax の地形ファイル（geometry.h5 または geometry.vti）から地下の比誘電率 ε' の分布を作り、
2 次元 FFT で地下構造そのものの波数特性を求める（単体で動く）。

B-scan の解析とは独立に、作った地下構造モデルだけを見る
    * 範囲：地下全体（地表直下から領域の底まで、x は領域全体）。X_RANGE_M / DEPTH_RANGE_M で狭められる
    * 区間：水平に SEG_LENGTHS_M（4, 10, 20, 50 m）の長さで切る。範囲に収まらない長さは飛ばす
      （区間が範囲より短いときは、余りが左右に等しく残るよう中央に並べる）。鉛直は範囲全体
    * 軸：x も深さ z も m のまま（時間 [ns] には直さない）。k_x, k_z [rad/m] は構造の波数そのもの
    * 計算は地形の格子（5 mm）のまま（地形のナイキスト波数 pi / dx = 628 rad/m）
    * 波数軸の範囲は B-scan の解析（gprmax_bscan_2dfft.py）と同じ固定値（K_LIM_1D、MAP_KX_LIM、
      MAP_KZ_LIM）。縦軸（dB）は見ている量が違うので揃えず、表示範囲の値から自動で決める
    * ε' はそのまま（平均を引かない）FFT する。平均は k = 0 に集まり、窓関数のすそで
      k = 0 のすぐ隣のビンに漏れる。k < DC_EXCLUDE_BINS * Δk は図で灰色に塗り、軸・色の範囲の
      計算から除く
    * S_x は全 k_z で、S_z は全 k_x で積分した片側スペクトル（周辺スペクトル）
    * 参考として CE-4 の帯域（250-750 MHz、ε_r = 3）に当たる k_z = 18.1-54.4 rad/m を薄く塗る
      （MARK_KZ_BAND = None で消える。計算には使わない）

読み込み
    geometry.h5 : /data（材料番号）と、同じフォルダの <名前>_materials.txt（番号順の材料）
    geometry.vti: Material 配列と材料名（<gprMax> ブロック）と、<名前>_materials.txt
    ε' は Debye 分散を含めて EVAL_FREQ_HZ で計算する（CE-4 の帯域中心 433 MHz が既定）。
    地表は、各列で空気（free_space）でない最も上のセルの中央値。

出力（入力ファイルと同じフォルダに OUTPUT_DIRNAME を作る）
    overview_segments.png        解析範囲の ε' と区間の枠
    mean_L<Ls>m.png              区間長ごとの平均スペクトル（2 次元図、S_x、S_z）
    compare_lengths.png          区間長による違い
    L<Ls>m/seg_x<始点>-<終点>m.png  区間ごとの図（ε'、2 次元図、S_x、S_z）
    spectra_mean.npz / run_info.json
    eps_subsurface.txt           解析範囲の ε'[深さ, x]（SAVE_PROCESSED_TXT = False で省略）
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

# --- 解析する範囲と区間
X_RANGE_M = None            # 水平範囲 (始点, 終点) [m]（領域の x 座標）。None で領域全体
DEPTH_RANGE_M = None        # 深さ範囲 (上端, 下端) [m]（地表から）。None で地下全体
SEG_LENGTHS_M = (4.0, 10.0, 20.0, 50.0)   # 水平の区間長 [m]。範囲に収まるものだけ解析する
SEG_OVERLAP = 0.0           # 区間の重なり（0 = 重ねない、0.5 = 半分重ねる）
WINDOW_FUNC = 'hann'        # hann / blackmanharris / none（x, z の両方向にかける）
DC_EXCLUDE_BINS = 2         # k < この数 x Δk を平均（k = 0）の漏れとして扱う（図で灰色、範囲の計算から除く）

# --- 表示
MARK_KZ_BAND = (18.1, 54.4) # 参考に塗る k_z の帯 [rad/m]（CE-4 の 250-750 MHz、ε_r = 3）。None で描かない
MARK_KZ_LABEL = 'CE-4 band (250–750 MHz, ε_r = 3)'
# --- 波数軸の範囲（gprmax_bscan_2dfft.py と同じ値にすること。B-scan と並べて比べるため）
K_LIM_1D = (0.5, 200.0)     # S_x, S_z の横軸 [rad/m]（対数軸）
                            #   下限：最小の分解能（10 m 区間の Δk = 0.63）より下
                            #   上限：B-scan のナイキスト波数（89.8 / 116.2）と、地下構造のスペクトルが
                            #         床に落ちる所（a = 0.05 m で約 150 rad/m）の両方より上
MAP_KX_LIM = 100.0          # 2 次元図の |k_x| の上限 [rad/m]
MAP_KZ_LIM = 150.0          # 2 次元図の k_z の上限 [rad/m]
WL_TICKS_MAP_X = (0.075, 0.1, 0.15, 0.25, 0.5)          # 2 次元図の λ_x の目盛り [m]
WL_TICKS_MAP_Z = (0.05, 0.075, 0.1, 0.15, 0.25, 0.5)    # 2 次元図の λ_z の目盛り [m]
DB_RANGE = 70.0             # 2 次元図の色の幅 [dB]
SX_YLIM_DB = None           # S_x の縦軸 (下限, 上限) [dB]。None で自動（全図共通）
SZ_YLIM_DB = None           # S_z の縦軸
SPEC_DB_RANGE_MAX = 100.0   # 自動のときの縦軸の幅の上限 [dB]
WL_TICKS = (0.03, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10)     # 1 次元図の波長の目盛り [m]
IMAGE_CMAP = 'viridis'      # ε' の図の色
IMAGE_VLIM = None           # ε' の図の色の範囲 (下限, 上限)。None で解析範囲の 0.5-99.5 % 点
SAVE_SEGMENT_FIGS = True
SAVE_PROCESSED_TXT = True
DPI = 130

# 線の色は名前で呼べる色にする（論文で「赤線」「青線」と言えるように）
SEG_COLORS = ['r', 'b', 'g', 'm', 'c', 'y']        # 区間（左から順）
LENGTH_COLORS = ['r', 'g', 'b', 'm', 'c', 'y']     # 区間長（短い順）
MEAN_COLOR = 'k'                                   # 平均は黒
COLOR_NAMES = {'r': 'red', 'b': 'blue', 'g': 'green', 'm': 'magenta', 'c': 'cyan',
               'y': 'yellow', 'k': 'black'}
DK_STYLE = dict(color='gray', ls='-.', lw=1.0)                 # 分解能 Δk
KN_STYLE = dict(color='gray', ls=(0, (1, 1.5)), lw=1.6)        # ナイキスト波数 k_N
DC_FILL = dict(color='0.80', alpha=0.5, hatch='//', lw=0)      # 平均の漏れ
BAND_FILL = dict(color='0.90', alpha=0.6)                      # 参考の k_z 帯
BAND_LINE = dict(color='w', ls='--', lw=0.9)                   # 2 次元図の参考の帯（白）

LIM = {}                    # 全図に共通の軸の範囲（main で決める）


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


# =============================================================================
# スペクトル
# =============================================================================
def window_1d(n, kind=None):
    kind = WINDOW_FUNC if kind is None else kind
    if kind == 'none':
        return np.ones(n)
    if kind == 'hann':
        return np.hanning(n)
    if kind == 'blackmanharris':
        a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
        t = 2 * np.pi * np.arange(n) / (n - 1)
        return a0 - a1 * np.cos(t) + a2 * np.cos(2 * t) - a3 * np.cos(3 * t)
    raise ValueError('WINDOW_FUNC は hann / blackmanharris / none')


def _fold(s_full, k):
    """両側スペクトル s_full(k)（fftshift 済み）を片側 s(k) + s(-k)（k > 0）にまとめる"""
    i0 = int(np.argmin(np.abs(k)))                 # k = 0 の位置
    m = np.arange(1, len(k) - i0)                  # k > 0 の番号
    return s_full[i0 + m] + s_full[i0 - m]         # fftshift 後は -k が必ずある


class Grid:
    """区間（N_z x N_x、格子 dx）の波数軸"""

    def __init__(self, nz, nx, dx):
        self.nz, self.nx, self.dx = nz, nx, dx
        self.kx = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(nx, d=dx))
        self.kz_all = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(nz, d=dx))
        self.kz_sel = self.kz_all >= 0
        self.kz = self.kz_all[self.kz_sel]             # 2 次元図の縦軸（k_z >= 0）
        self.dkx = 2 * np.pi / (nx * dx)
        self.dkz = 2 * np.pi / (nz * dx)
        self.kn = np.pi / dx                           # ナイキスト波数（x, z 共通）
        self.kx_pos = self.kx[self.kx > 0]             # 1 次元スペクトルの横軸
        self.kz_pos = self.kz_all[self.kz_all > 0]
        self.w2 = np.outer(window_1d(nz), window_1d(nx))
        self.w2mean = float((self.w2 ** 2).mean())

    def psd(self, block):
        """block[z, x] -> S[k_z, k_x]（両側、fftshift 済み）。平均は引かない"""
        X = np.fft.fftshift(np.fft.fft2(block.astype(np.float64) * self.w2))
        return np.abs(X) ** 2 * self.dx * self.dx / ((2 * np.pi) ** 2 * self.nz * self.nx * self.w2mean)

    def marginals(self, S):
        """片側の 1 次元スペクトル。S_x は全 k_z で、S_z は全 k_x で積分"""
        Sx = _fold(S.sum(axis=0) * self.dkz, self.kx)
        Sz = _fold(S.sum(axis=1) * self.dkx, self.kz_all)
        return Sx, Sz

    def half(self, S):
        return S[self.kz_sel]                          # 2 次元図用（k_z >= 0）


def segment_cols(ncols, nx):
    """幅 ncols の範囲に長さ nx の区間を並べる（余りは左右に等しく）。始点の列番号のリスト"""
    if nx > ncols:
        return []
    hop = max(1, int(round(nx * (1.0 - SEG_OVERLAP))))
    n = 1 + (ncols - nx) // hop
    off = (ncols - (nx + (n - 1) * hop)) // 2
    return [off + i * hop for i in range(n)]


# =============================================================================
# 図
# =============================================================================
def _db(a):
    return 10 * np.log10(np.maximum(a, 1e-300))


def _lname(Ls):
    return f'{Ls:03.0f}' if float(Ls).is_integer() else f'{Ls:05.1f}'.replace('.', 'p')


def _wl_axis(ax, which, wl_ticks, label, symmetric=False, min_frac=0.0):
    """波長 λ = 2 pi / k の第 2 軸（上または右）。k の軸と同じ目盛り位置に λ の値を付ける。
    軸の範囲を決めてから呼ぶ。min_frac > 0 なら |k| < min_frac x 上限 の目盛りは付けない（線形軸用）"""
    ident = (lambda v: v, lambda v: v)
    sec = ax.secondary_xaxis('top', functions=ident) if which == 'x' else \
        ax.secondary_yaxis('right', functions=ident)
    lo, hi = sorted(ax.get_xlim() if which == 'x' else ax.get_ylim())
    pos, lab = [], []
    for wl in wl_ticks:
        k = 2 * np.pi / wl
        if k < min_frac * max(abs(lo), abs(hi)):
            continue
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


def draw_map(ax, g, Sh):
    m = g.kz <= LIM['kz_map'][1] + g.dkz
    mx = np.abs(g.kx) <= LIM['kx_map'][1] + g.dkx
    im = ax.imshow(_db(Sh[m][:, mx]), origin='lower', aspect='auto', cmap='magma',
                   extent=[g.kx[mx][0] - g.dkx / 2, g.kx[mx][-1] + g.dkx / 2,
                           g.kz[m][0] - g.dkz / 2, g.kz[m][-1] + g.dkz / 2],
                   vmin=LIM['map'][0], vmax=LIM['map'][1], interpolation='nearest')
    hs = []
    if MARK_KZ_BAND:
        for k in MARK_KZ_BAND:
            ax.axhline(k, **BAND_LINE)
        hs.append(Line2D([], [], label=MARK_KZ_LABEL, **BAND_LINE))
    ax.set_xlim(*LIM['kx_map'])
    ax.set_ylim(*LIM['kz_map'])
    ax.set_xlabel('k_x [rad/m]')
    ax.set_ylabel('k_z [rad/m]')
    _wl_axis(ax, 'x', WL_TICKS_MAP_X, 'λ_x [m]', symmetric=True)
    _wl_axis(ax, 'y', WL_TICKS_MAP_Z, 'λ_z [m]')
    ax.set_title('S(k_x, k_z)')
    if hs:
        ax.legend(handles=hs, loc='upper right', fontsize=7.5, facecolor='0.4', labelcolor='w',
                  framealpha=0.7)
    return im


def _curve_handles(curves):
    return [Line2D([], [], color=c['color'], lw=c.get('lw', 1.3), ls=c.get('ls', '-'),
                   label=c['label']) for c in curves if c.get('label')]


def _draw_1d(ax, axis, k_list, curves, dk_lines, kn, plot):
    """S_x / S_z 共通。k_list[i] は curves[i] の横軸。dk_lines = [(Δk, 色 or None), ...]"""
    ax.set_xscale('log')
    if plot:
        for k, c in zip(k_list, curves):
            ax.plot(k, _db(c['s']), color=c['color'], lw=c.get('lw', 1.3), ls=c.get('ls', '-'))
    hs = []
    dk_dc = max(dk for dk, _ in dk_lines) * DC_EXCLUDE_BINS
    if DC_EXCLUDE_BINS > 0:
        ax.axvspan(LIM['k' + axis][0], dk_dc, **DC_FILL)
        hs.append(Patch(label=f'k < {DC_EXCLUDE_BINS}Δk: mean (k = 0) leakage', **DC_FILL))
    if axis == 'z' and MARK_KZ_BAND:
        ax.axvspan(*MARK_KZ_BAND, **BAND_FILL)
        hs.append(Patch(label=MARK_KZ_LABEL, **BAND_FILL))
    if len(dk_lines) == 1 and dk_lines[0][1] is None:
        ax.axvline(dk_lines[0][0], **DK_STYLE)
        hs.append(Line2D([], [], label=f'Δk_{axis} = {dk_lines[0][0]:.3f} rad/m', **DK_STYLE))
    else:                                           # 区間長ごとの Δk（色 = 区間長）
        for dk, col in dk_lines:
            ax.axvline(dk, color=col, ls='-.', lw=1.0)
        hs.append(Line2D([], [], color='k', ls='-.', lw=1.0, label=f'Δk_{axis} (colour = L)'))
    ax.axvline(kn, **KN_STYLE)
    off = ' (off scale)' if kn > LIM['k' + axis][1] else ''
    hs.append(Line2D([], [], label=f'k_N = {kn:.1f} rad/m{off}', **KN_STYLE))
    ax.set_xlim(*LIM['k' + axis])
    ax.set_ylim(*LIM['s' + axis])
    _wl_axis(ax, 'x', WL_TICKS, f'λ_{axis} [m]')
    ax.set_xlabel(f'k_{axis} [rad/m]')
    ax.set_ylabel(f'S_{axis} [dB]')
    other = 'z' if axis == 'x' else 'x'
    ax.set_title(f'S_{axis}(k_{axis}), integrated over all k_{other}')
    ax.grid(alpha=0.3, which='both')
    ax.legend(handles=_curve_handles(curves) + hs, fontsize=7.5, loc='lower left')


def draw_sx(ax, g, curves, dk_lines=None, k_list=None, plot=True):
    _draw_1d(ax, 'x', k_list or [g.kx_pos] * len(curves), curves,
             dk_lines or [(g.dkx, None)], g.kn, plot)


def draw_sz(ax, g, curves, dk_lines=None, k_list=None, plot=True):
    _draw_1d(ax, 'z', k_list or [g.kz_pos] * len(curves), curves,
             dk_lines or [(g.dkz, None)], g.kn, plot)


def _eps_image(fig, ax, img, x0, x1, z0, z1, title):
    im = ax.imshow(img, aspect='auto', cmap=IMAGE_CMAP, vmin=IMAGE_VLIM[0], vmax=IMAGE_VLIM[1],
                   extent=[x0, x1, z1, z0], interpolation='nearest')
    fig.colorbar(im, ax=ax, label="ε'")
    ax.set_xlabel('x [m]')
    ax.set_ylabel('Depth [m]')
    ax.set_title(title)


def plot_overview(outdir, label, eps_sub, x0, z0, dx, all_L):
    nz, nc = eps_sub.shape
    x1, z1 = x0 + nc * dx, z0 + nz * dx
    w = 10.0 * nc / max(nz, nc) + 3
    fig, ax = plt.subplots(figsize=(max(w, 7), 10.0 * nz / max(nz, nc) + 1.5), layout='constrained')
    _eps_image(fig, ax, eps_sub, x0, x1, z0, z1, f"{label}: relative permittivity ε' (analysed area)")
    hs = []
    for i, (Ls, d) in enumerate(all_L.items()):
        col = LENGTH_COLORS[i % len(LENGTH_COLORS)]
        pad = 0.012 * (i + 1) * (z1 - z0)            # 区間長ごとに枠を少し内側に（重なって見えないように）
        px = 0.004 * (i + 1) * (x1 - x0)
        for s in d['segs']:
            ax.add_patch(Rectangle((s['x0'] + px, z0 + pad), s['x1'] - s['x0'] - 2 * px,
                                   (z1 - z0) - 2 * pad,
                                   fill=False, ec=col, lw=2, ls='--' if i % 2 else '-'))
        hs.append(Line2D([], [], color=col, lw=2, ls='--' if i % 2 else '-',
                         label=f'L = {Ls:g} m ({len(d["segs"])} segment' +
                               ('s)' if len(d['segs']) > 1 else ')')))
    ax.legend(handles=hs, loc='lower right', fontsize=9, framealpha=0.85)
    fig.savefig(os.path.join(outdir, 'overview_segments.png'), dpi=DPI)
    plt.close(fig)


def plot_mean(outdir, label, Ls, d):
    g, res = d['g'], d['mean']
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.8), layout='constrained',
                            gridspec_kw={'width_ratios': [1.2, 1, 1]})
    im = draw_map(axs[0], g, res['S'])
    fig.colorbar(im, ax=axs[0], label='dB')
    n = len(d['segs'])
    cx = [dict(s=res['Sx'], color=MEAN_COLOR, lw=2.2, label=f'mean ({n} segments)')] if n > 1 else []
    cz = [dict(s=res['Sz'], color=MEAN_COLOR, lw=2.2, label=f'mean ({n} segments)')] if n > 1 else []
    for j, s in enumerate(d['segs']):
        col = SEG_COLORS[j % len(SEG_COLORS)]
        lab = f'x = {s["x0"]:.2f}–{s["x1"]:.2f} m'
        sx, sz = d['seg_marg'][j]
        cx.append(dict(s=sx, color=col, lw=1.0, label=lab))
        cz.append(dict(s=sz, color=col, lw=1.0, label=lab))
    draw_sx(axs[1], g, cx)
    draw_sz(axs[2], g, cz)
    fig.suptitle(f'{label}: mean spectrum, L = {Ls:g} m ({n} segment' + ('s' if n > 1 else '') +
                 f', depth {d["z0"]:g}–{d["z1"]:g} m, window: {WINDOW_FUNC})', fontsize=13)
    fig.savefig(os.path.join(outdir, f'mean_L{_lname(Ls)}m.png'), dpi=DPI)
    plt.close(fig)


def plot_segment(path, label, Ls, d, j, blk):
    g, s = d['g'], d['segs'][j]
    S = d['spectra'][j]
    sx, sz = d['seg_marg'][j]
    col = SEG_COLORS[j % len(SEG_COLORS)]
    fig = plt.figure(figsize=(16, 10.5), layout='constrained')
    gs = fig.add_gridspec(2, 2)
    _eps_image(fig, fig.add_subplot(gs[0, 0]), blk, s['x0'], s['x1'], d['z0'], d['z1'],
               "Relative permittivity ε'")
    ax = fig.add_subplot(gs[0, 1])
    im = draw_map(ax, g, g.half(S))
    fig.colorbar(im, ax=ax, label='dB')
    lab = f'x = {s["x0"]:.2f}–{s["x1"]:.2f} m'
    draw_sx(fig.add_subplot(gs[1, 0]), g, [dict(s=sx, color=col, label=lab)])
    draw_sz(fig.add_subplot(gs[1, 1]), g, [dict(s=sz, color=col, label=lab)])
    fig.suptitle(f'{label}: segment x = {s["x0"]:.2f}–{s["x1"]:.2f} m (L = {Ls:g} m), '
                 f'depth {d["z0"]:g}–{d["z1"]:g} m', fontsize=13)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_compare_lengths(outdir, label, all_L):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.0), layout='constrained')
    cx, cz, kx, kz, dkx, dkz = [], [], [], [], [], []
    for i, (Ls, d) in enumerate(all_L.items()):
        g, r = d['g'], d['mean']
        col = LENGTH_COLORS[i % len(LENGTH_COLORS)]
        n = len(d['segs'])
        cx.append(dict(s=r['Sx'], color=col, lw=1.5, label=f'L = {Ls:g} m ({n} seg.), Δk_x = {g.dkx:.3f}'))
        cz.append(dict(s=r['Sz'], color=col, lw=1.5, label=f'L = {Ls:g} m ({n} seg.), Δk_z = {g.dkz:.3f}'))
        kx.append(g.kx_pos)
        kz.append(g.kz_pos)
        dkx.append((g.dkx, col))
        dkz.append((g.dkz, col))
    g0 = next(iter(all_L.values()))['g']
    # 区間長ごとに k 軸が違うので、横軸を曲線ごとに渡す。Δk_z は区間長によらないこともある
    draw_sx(axs[0], g0, cx, dk_lines=dkx, k_list=kx)
    draw_sz(axs[1], g0, cz, dk_lines=dkz if len(set(round(v, 9) for v, _ in dkz)) > 1
            else [(dkz[0][0], None)], k_list=kz)
    fig.suptitle(f'{label}: mean spectra for different segment lengths', fontsize=13)
    fig.savefig(os.path.join(outdir, 'compare_lengths.png'), dpi=DPI)
    plt.close(fig)


def set_common_limits(all_L):
    """全図に共通の軸の範囲を決める（横軸・縦軸・2 次元図の色）"""
    # 波数軸は設定の固定値（B-scan の解析 gprmax_bscan_2dfft.py と同じ）
    LIM['kx'] = tuple(K_LIM_1D)
    LIM['kz'] = tuple(K_LIM_1D)
    LIM['kx_map'] = (-MAP_KX_LIM, MAP_KX_LIM)
    LIM['kz_map'] = (0.0, MAP_KZ_LIM)
    # 縦軸・色は、表示する波数の範囲に入る値だけで決める（平均の漏れ k < DC_EXCLUDE_BINS x Δk は除く）
    tops = {'x': [], 'z': []}
    bots = {'x': [], 'z': []}
    map_max = []
    for d in all_L.values():
        g = d['g']
        mk = {a: (k >= max(DC_EXCLUDE_BINS * dk, LIM['k' + a][0])) & (k <= LIM['k' + a][1])
              for a, k, dk in (('x', g.kx_pos, g.dkx), ('z', g.kz_pos, g.dkz))}
        for sx, sz in d['seg_marg'] + [(d['mean']['Sx'], d['mean']['Sz'])]:
            for a, s in (('x', sx), ('z', sz)):
                tops[a].append(_db(s[mk[a]]).max())
        for a, s in (('x', d['mean']['Sx']), ('z', d['mean']['Sz'])):
            bots[a].append(_db(s[mk[a]]).min())
        dc = (np.abs(g.kz)[:, None] < DC_EXCLUDE_BINS * g.dkz) & \
             (np.abs(g.kx)[None, :] < DC_EXCLUDE_BINS * g.dkx)
        inside = (g.kz[:, None] <= MAP_KZ_LIM) & (np.abs(g.kx)[None, :] <= MAP_KX_LIM)
        map_max.append(_db(np.where(dc | ~inside, 0.0, d['mean']['S'])).max())
    for a, user in (('x', SX_YLIM_DB), ('z', SZ_YLIM_DB)):
        if user is not None:
            LIM['s' + a] = tuple(user)
        else:
            top = float(max(tops[a])) + 3.0
            LIM['s' + a] = (max(float(min(bots[a])) - 3.0, top - SPEC_DB_RANGE_MAX), top)
    vmax = float(max(map_max)) + 5.0
    LIM['map'] = (vmax - DB_RANGE, vmax)
    print('共通の軸: k_x, k_z {:g}-{:g} rad/m（2 次元図 |k_x| <= {:g}、k_z <= {:g}）、'
          'S_x {:.1f}-{:.1f} dB、S_z {:.1f}-{:.1f} dB、2 次元図 {:.1f}-{:.1f} dB'.format(
              *LIM['kx'], MAP_KX_LIM, MAP_KZ_LIM, *LIM['sx'], *LIM['sz'], *LIM['map']))


# =============================================================================
def main():
    #* Get input parameters
    print('geometry.h5 または geometry.vti のパスを入力してください:')
    in_path = input().strip().strip("'\"")
    if not os.path.exists(in_path):
        print('エラー: 指定されたファイルが存在しません')
        exit(1)
    t_all = time.time()
    in_path = os.path.abspath(in_path)
    outdir = os.path.join(os.path.dirname(in_path), OUTPUT_DIRNAME)
    os.makedirs(outdir, exist_ok=True)
    label = os.path.basename(os.path.dirname(in_path)) or 'geometry'

    eps, dx, air, mat_path, n_mat = load_eps(in_path)
    nx_g, ny_g = eps.shape
    print('読み込み: {}（{} x {} セル、格子 {} m、材料 {} 種、{}）'.format(
        os.path.basename(in_path), nx_g, ny_g, dx, n_mat, os.path.basename(mat_path)))

    # ---- 1. 地表（各列で空気でない最も上のセル。中央値）と、地下の ε'[深さ, x]
    solid = ~air
    top = np.array([np.max(np.where(col)[0]) if col.any() else -1 for col in solid])
    top = top[top >= 0]
    if top.size == 0:
        raise SystemExit('地下（空気でないセル）が見つかりません')
    iy_surf = int(np.median(top)) + 1                          # 地表の y（セルの境界の番号）
    sub = eps[:, :iy_surf][:, ::-1].T.astype(np.float64)       # [深さ, x]、深さ 0 = 地表直下
    print('地表 y = {:.3f} m、地下 {:.3f} m x {:.3f} m（{} x {} セル）'.format(
        iy_surf * dx, iy_surf * dx, nx_g * dx, sub.shape[0], sub.shape[1]))

    # ---- 2. 解析範囲
    def _rng(r, n, name):
        if r is None:
            return 0, n
        i0, i1 = int(round(r[0] / dx)), int(round(r[1] / dx))
        if i0 < 0 or i1 > n or i1 <= i0:
            raise SystemExit('{} = {} が範囲外です（0-{:.3f} m）'.format(name, r, n * dx))
        return i0, i1
    iz0, iz1 = _rng(DEPTH_RANGE_M, sub.shape[0], 'DEPTH_RANGE_M')
    ix0, ix1 = _rng(X_RANGE_M, sub.shape[1], 'X_RANGE_M')
    area = sub[iz0:iz1, ix0:ix1]
    x0, z0 = ix0 * dx, iz0 * dx
    nz, nc = area.shape
    print("解析範囲: x = {:.3f}-{:.3f} m、深さ {:.3f}-{:.3f} m（{} x {} セル）、"
          "ε' {:.3f}-{:.3f}（平均 {:.4f}、標準偏差 {:.4f}）@ {:g} MHz".format(
              x0, x0 + nc * dx, z0, z0 + nz * dx, nz, nc, area.min(), area.max(), area.mean(),
              area.std(), EVAL_FREQ_HZ / 1e6))
    global IMAGE_VLIM
    if IMAGE_VLIM is None:
        IMAGE_VLIM = tuple(np.percentile(area, [0.5, 99.5]))
    if SAVE_PROCESSED_TXT:
        p = os.path.join(outdir, 'eps_subsurface.txt')
        np.savetxt(p, area, fmt='%.6f')
        print("-> {}（解析範囲の ε'[深さ, x]、{} m 格子）".format(p, dx))

    # ---- 3. 区間長ごとのスペクトル（解析できる長さだけ）
    all_L, skipped = {}, []
    for Ls in SEG_LENGTHS_M:
        nx = int(round(Ls / dx))
        cols = segment_cols(nc, nx)
        if not cols:
            skipped.append(Ls)
            print('L = {:g} m: 範囲の幅 {:.3f} m より長いので飛ばす'.format(Ls, nc * dx))
            continue
        g = Grid(nz, nx, dx)
        segs, spectra, seg_marg = [], [], []
        for c in cols:
            S = g.psd(area[:, c:c + nx])
            segs.append(dict(x0=x0 + c * dx, x1=x0 + (c + nx) * dx, col=c))
            spectra.append(S)
            seg_marg.append(g.marginals(S))
        Smean = np.mean(spectra, axis=0)
        Sx, Sz = g.marginals(Smean)
        all_L[Ls] = dict(g=g, nx=nx, segs=segs, spectra=spectra, seg_marg=seg_marg,
                         mean=dict(S=g.half(Smean), Sx=Sx, Sz=Sz),
                         z0=z0, z1=z0 + nz * dx)
        print('L = {:g} m: 区間 {} 個（{}）、Δk_x = {:.3f}、Δk_z = {:.3f} rad/m'.format(
            Ls, len(segs), ', '.join('{:.2f}-{:.2f} m'.format(s['x0'], s['x1']) for s in segs),
            g.dkx, g.dkz))
    if not all_L:
        raise SystemExit('解析できる区間長がありません（範囲の幅 {:.3f} m）'.format(nc * dx))
    print('ナイキスト波数 k_N = pi / {} = {:.1f} rad/m（x, z 共通）'.format(dx, np.pi / dx))

    # ---- 4. 図
    set_common_limits(all_L)
    plot_overview(outdir, label, area, x0, z0, dx, all_L)
    print('-> {}'.format(os.path.join(outdir, 'overview_segments.png')))
    for Ls, d in all_L.items():
        t = time.time()
        plot_mean(outdir, label, Ls, d)
        print('  L = {:g} m -> {}'.format(Ls, os.path.join(outdir, f'mean_L{_lname(Ls)}m.png')))
        if SAVE_SEGMENT_FIGS:
            subdir = os.path.join(outdir, f'L{_lname(Ls)}m')
            os.makedirs(subdir, exist_ok=True)
            for j, s in enumerate(d['segs']):
                fn = 'seg_x{:07.2f}-{:07.2f}m.png'.format(s['x0'], s['x1'])
                plot_segment(os.path.join(subdir, fn), label, Ls, d, j,
                             area[:, s['col']:s['col'] + d['nx']])
            print('  -> {}/（{} 枚、{:.0f} s）'.format(subdir, len(d['segs']), time.time() - t))
    plot_compare_lengths(outdir, label, all_L)
    print('-> {}'.format(os.path.join(outdir, 'compare_lengths.png')))

    # ---- 5. 数値
    np.savez(os.path.join(outdir, 'spectra_mean.npz'),
             **{f'L{_lname(Ls)}_{key}': val for Ls, d in all_L.items()
                for key, val in (('kx', d['g'].kx), ('kz', d['g'].kz), ('S', d['mean']['S']),
                                 ('kx_pos', d['g'].kx_pos), ('Sx', d['mean']['Sx']),
                                 ('kz_pos', d['g'].kz_pos), ('Sz', d['mean']['Sz']),
                                 ('n', np.array(len(d['segs']))))})
    info = dict(settings={k: v for k, v in globals().items()
                          if k.isupper() and isinstance(v, (int, float, str, tuple, list,
                                                            dict, type(None)))},
                input_file=in_path, materials_file=mat_path, eval_freq_hz=EVAL_FREQ_HZ,
                grid_m=dx, surface_y_m=iy_surf * dx,
                area=dict(x_m=[x0, x0 + nc * dx], depth_m=[z0, z0 + nz * dx], shape=[nz, nc],
                          eps_mean=float(area.mean()), eps_std=float(area.std())),
                nyquist_rad_m=np.pi / dx, skipped_lengths=skipped,
                lengths={str(Ls): dict(dkx=d['g'].dkx, dkz=d['g'].dkz, nx=d['nx'], nz=nz,
                                       segments=[{k: v for k, v in s.items() if k != 'col'}
                                                 for s in d['segs']])
                         for Ls, d in all_L.items()},
                common_limits={k: list(v) for k, v in LIM.items()})
    with open(os.path.join(outdir, 'run_info.json'), 'w', encoding='utf-8') as fh:
        json.dump(info, fh, ensure_ascii=False, indent=1, default=str)
    print('完了（{:.0f} s）。出力先: {}'.format(time.time() - t_all, outdir))


if __name__ == '__main__':
    main()