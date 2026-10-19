#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gprmax_bscan_2dfft.py
=====================

gprMax で計算した B-scan（outputfiles_merge で作った *_merged.out）の 2 次元 FFT を、
CE-4/LPR の実データと同じ処理・同じ図で求める（単体で動く）。

解析・描画の部分（スペクトル、区間の切り出し、図）は lpr_bscan_2dfft.py からの写しで、
図の種類・軸・色・ファイル名は LPR の解析と同じ。どちらかを変えたときは、もう一方も
手で同じように直すこと（写した範囲は「ここから lpr_bscan_2dfft.py と同じ」の印の間）。
波数軸の範囲は地下構造の解析（geometry_2dfft.py）と揃えている（K_LIM_1D、MAP_KX_LIM、
MAP_KY_LIM。変えた所には「軸の統一」の印）。S_y も S_x と同じ対数軸。

前処理（実データの処理に合わせる）
    1. 受信点の Ez を読む（行 = 時間、列 = トレース）
    2. 背景除去：全トレースの平均トレースを各トレースから引く（直接波を除く）
    3. 時刻 0 を地表反射にそろえる（TIME_ZERO で方法を選ぶ）
    4. サンプリング間隔を LPR と同じ 0.3125 ns に直す（帯域制限した補間。時刻 0 から始まる格子）
    5. 解析窓（地表反射から T_START_NS、長さ TIME_LEN_NS）を区間長 SEG_LENGTHS_M で切って
       2 次元 FFT

トレース間隔とケース名
    merged.out には送受信点の位置が残らないので、同じフォルダか 1-2 つ上のフォルダにある
    Ascan の記録ファイル（*_ce4_record.json）から読む。見つからなければ設定の値を使う。

出力（.out と同じフォルダに OUTPUT_DIRNAME を作る。構成は LPR の解析と同じ）
    overview_windows.png / mean_<ケース>_L<Ls>m.png / compare_lengths_<ケース>.png
    spectra_mean.npz / run_info.json / L<Ls>m/<窓>_x<始点>-<終点>m.png
    processed_bscan.txt（前処理後の B-scan。SAVE_PROCESSED_TXT = False で省略）
"""

import glob
import json
import os
import time

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
OUTPUT_DIRNAME = '2D_FFT'
RX_COMPONENT = 'Ez'
RX_NUMBER = 1

# --- 前処理
BACKGROUND_REMOVAL = True   # 平均トレースを引く（実データの背景除去に合わせる）
TIME_ZERO = 'envelope'      # 'envelope'：平均トレースの包絡線の最大（直接波 + 地表反射）を 0 にする
                            # 'theory'  ：地表反射の理論時刻（波形の遅れ + 往復の伝搬時間）
                            # 'manual'  ：TIME_ZERO_NS を使う
TIME_ZERO_SEARCH_NS = 20.0  # 'envelope' で最大を探す範囲 [ns]
TIME_ZERO_NS = 0.0          # 'manual' のときの時刻 0 [ns]（シミュレーションの時刻）
RESAMPLE_DT_NS = 0.3125     # LPR CH2 と同じサンプリング間隔 [ns]。None でそのまま

# --- 記録ファイルが見つからないときの値（CE-4 ランダム媒質シミュレーションの既定）
DX_M_DEFAULT = 0.035        # トレース間隔 [m]
ANT_HEIGHT_DEFAULT = 0.30   # [m]
TXRX_SEP_DEFAULT = 0.32     # [m]
WAVEFORM_FREQ_DEFAULT = 400e6  # gaussiandot の周波数パラメータ [Hz]

# --- 解析（LPR の解析 lpr_bscan_2dfft.py と同じ値にすること）
T_START_NS = 30.0           # 解析窓の上端（地表反射からの時刻）[ns]
TIME_LEN_NS = 70.0          # 解析窓の長さ [ns]（ε_r = 3 で 6.06 m）
SEG_LENGTHS_M = (4.0,)      # 区間長 [m]（測線 3.99 m なら 4 m で 1 区間）
SEG_OVERLAP = 0.0           # 区間の重なりの割合
EPS_R = 3.0                 # 時間 → 深さの換算に使う比誘電率
BAND_MHZ = (250.0, 750.0)   # LPR CH2 の動作帯域 [MHz]
WINDOW_FUNC = 'hann'        # 'hann' / 'blackmanharris' / 'none'
BACKGROUND = 'none'         # 解析窓ごとの背景除去（このスクリプトでは前処理で行うので 'none'）
INVALID_FRAC_MAX = 0.01     # 区間内の無効なサンプルの割合がこれを超えたら除外
ZERO_RUN_MIN = 16           # 0 がこの点数以上続く所を無効とみなす

# --- 出力（LPR の解析と同じ体裁）
SAVE_SEGMENT_FIGS = True
SAVE_PROCESSED_TXT = True
KY_MAX_PLOT = None          # 図の k_y の上限 [rad/m]。None でナイキスト波数まで
DB_RANGE = 70.0             # 2 次元スペクトル図の色の幅 [dB]
SX_YLIM_DB = None           # S_x の図の縦軸 (下限, 上限) [dB]。None で自動（6 ケースで揃えるなら指定）
SY_YLIM_DB = None           # S_y の図の縦軸 (下限, 上限) [dB]。同上
SPEC_DB_RANGE_MAX = 100.0

# --- 波数軸の範囲（geometry_2dfft.py と同じ値にすること。B-scan と地下構造の図を並べて比べるため）
#     縦軸（S_x, S_y の dB）は見ている量が違うので揃えず、それぞれ自動で決める
K_LIM_1D = (0.5, 200.0)     # S_x, S_y の横軸 [rad/m]（対数軸）
                            #   下限：最小の分解能（地下構造の 10 m 区間の Δk = 0.63）より下
                            #   上限：B-scan のナイキスト波数（89.8 / 116.2）と、地下構造のスペクトルが
                            #         床に落ちる所（a = 0.05 m で約 150 rad/m）の両方より上
MAP_KX_LIM = 100.0          # 2 次元図の |k_x| の上限 [rad/m]（B-scan の k_N,x = 89.8 が入る）
MAP_KY_LIM = 150.0          # 2 次元図の k_y の上限 [rad/m]（B-scan の k_N,y = 116.2 が入る）
NO_DATA_COLOR = '0.55'      # 2 次元図でナイキスト波数より外（データなし）の色
WL_TICKS = (0.03, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10)     # 1 次元図の波長の目盛り [m]
WL_TICKS_MAP_X = (0.075, 0.1, 0.15, 0.25, 0.5)          # 2 次元図の λ_x の目盛り
WL_TICKS_MAP_Y = (0.05, 0.075, 0.1, 0.15, 0.25, 0.5)    # 2 次元図の λ_y の目盛り
OVERVIEW_STRIDE = 1
DPI = 130

C0 = 299792458.0

# 実行時に決まる値（main で設定する）
DX_M = DX_M_DEFAULT
DT_NS = RESAMPLE_DT_NS or 0.3125
T0_NS = 0.0
WINDOWS = []


# =============================================================================
# gprMax 出力の前処理
# =============================================================================
def find_record(out_path):
    """Ascan の記録ファイル（*_ce4_record.json）を .out のフォルダと 1-2 つ上から探す"""
    d = os.path.dirname(os.path.abspath(out_path))
    for _ in range(3):
        hits = sorted(glob.glob(os.path.join(d, '*_ce4_record.json')))
        if hits:
            with open(hits[0], encoding='utf-8') as fh:
                return hits[0], json.load(fh)
        d = os.path.dirname(d)
    return None, None


def read_out(path):
    with h5py.File(path, 'r') as fh:
        dt = float(fh.attrs['dt'])
        title = fh.attrs.get('Title', '')
        key = 'rxs/rx{}/{}'.format(RX_NUMBER, RX_COMPONENT)
        if key not in fh:
            raise SystemExit('{} が見つかりません（受信点・成分を確認）'.format(key))
        data = fh[key][()]
    if data.ndim == 1:
        raise SystemExit('トレースが 1 本しかありません。outputfiles_merge で作った merged.out を指定すること')
    if isinstance(title, bytes):
        title = title.decode('utf-8', 'replace')
    return data.astype(np.float64), dt, str(title)


def envelope(x, axis=0):
    """解析信号の絶対値（FFT で計算）"""
    n = x.shape[axis]
    X = np.fft.fft(x, axis=axis)
    h = np.zeros(n)
    h[0] = 1
    if n % 2 == 0:
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[1:(n + 1) // 2] = 2
    shape = [1] * x.ndim
    shape[axis] = n
    return np.abs(np.fft.ifft(X * h.reshape(shape), axis=axis))


def resample_time(data, dt_in, t0, dt_out, end_taper=2e-9):
    """時刻 t0 から間隔 dt_out の格子へ、帯域制限した補間で直す（時刻 0 も同時にそろう）。
    * 記録の終わりを end_taper かけて 0 に落とし、2 倍に 0 詰めする（端の不連続による
      リンギングと、周期の折り返しを避ける）
    * 新しいナイキスト周波数の 80-95 % で滑らかに 0 に落とす低域通過をかける
    * 低域通過した三角級数を、新しい格子の時刻でそのまま計算する（間隔がちょうど dt_out）"""
    n, ntr = data.shape
    x = data.copy()
    nt_tap = max(1, int(round(end_taper / dt_in)))
    x[-nt_tap:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, nt_tap)))[:, None]
    N = 2 * n
    X = np.fft.rfft(x, n=N, axis=0)
    f = np.fft.rfftfreq(N, d=dt_in)
    fn = 0.5 / dt_out
    f1, f2 = 0.80 * fn, 0.95 * fn
    H = np.where(f <= f1, 1.0, np.where(f >= f2, 0.0, 0.5 * (1 + np.cos(np.pi * (f - f1) / (f2 - f1)))))
    K = int(np.searchsorted(f, f2)) + 1
    t_end = (n - nt_tap) * dt_in                 # テーパーより前まで
    m = int(np.floor((t_end - t0) / dt_out)) + 1
    tq = t0 + np.arange(m) * dt_out
    w = np.full(K, 2.0)
    w[0] = 1.0
    E = np.exp(2j * np.pi * np.outer(tq, f[:K]))                    # [m, K]
    y = (E @ (X[:K] * (H[:K] * w)[:, None])).real / N
    return y, dt_out




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
        b = block - block.mean()
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
    kn = np.pi / dy_m()
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
    ax.set_facecolor(NO_DATA_COLOR)                                           # 軸の統一：k_N より外
    ax.set_xlim(*LIM['kx_map'])
    ax.set_ylim(*LIM['ky_map'])
    ax.set_xlabel('k_x [rad/m]')
    ax.set_ylabel('k_y [rad/m]')
    _wl_axis(ax, 'x', WL_TICKS_MAP_X, 'λ_x [m]', symmetric=True)
    _wl_axis(ax, 'y', WL_TICKS_MAP_Y, f'λ_y [m] (ε_r = {EPS_R:g})')
    ax.set_title('S(k_x, k_y)')
    hs = [Line2D([], [], label=f'{BAND_MHZ[0]:g}–{BAND_MHZ[1]:g} MHz', **BAND_LINE)]
    if LIM['kx_map'][1] > np.pi / DX_M or LIM['ky_map'][1] > ky_max():
        hs.append(Patch(color=NO_DATA_COLOR, label='beyond k_N (no data)'))
    ax.legend(handles=hs, loc='upper right', fontsize=7.5, facecolor='0.4', labelcolor='w',
              framealpha=0.7)
    return im


def _res_lines(ax, dk, kn, axis, show_dk=True):
    """分解能 Δk（灰色の一点鎖線）とナイキスト波数 k_N（灰色の点線）の縦線と凡例"""
    hs = []
    if show_dk:
        ax.axvline(dk, **DK_STYLE)
        hs.append(Line2D([], [], label=f'Δk_{axis} = {dk:.3f} rad/m', **DK_STYLE))
    ax.axvline(kn, **KN_STYLE)
    hs.append(Line2D([], [], label=f'k_N,{axis} = {kn:.1f} rad/m', **KN_STYLE))
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
    _wl_axis(ax, 'x', WL_TICKS, 'λ_x [m]')
    ax.set_xlabel('k_x [rad/m]')
    ax.set_ylabel('S_x [dB]')
    ax.set_title(f'S_x(k_x), k_y in {BAND_MHZ[0]:g}–{BAND_MHZ[1]:g} MHz')
    ax.grid(alpha=0.3, which='both')
    ax.legend(handles=_curve_handles(curves) + hs, fontsize=7.5, loc='lower left')


def draw_sy(ax, g, curves, plot=True):
    k = g.ky[g.ky_pos]
    m = k <= ky_max() + 1e-9
    ax.set_xscale('log')                                                       # 軸の統一：対数軸
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
    ax.grid(alpha=0.3, which='both')
    _wl_axis(ax, 'x', WL_TICKS, f'λ_y [m] (ε_r = {EPS_R:g})')
    h0 = _curve_handles(curves)
    h0.append(Patch(label=f'{BAND_MHZ[0]:g}–{BAND_MHZ[1]:g} MHz', **BAND_FILL))
    ax.legend(handles=h0 + hs, fontsize=7.5, loc='lower left')


def plot_overview(data, outdir, amp):
    nt, ntr = data.shape
    st = max(1, OVERVIEW_STRIDE)
    img = np.nan_to_num(data[:, ::st])
    fig, ax = plt.subplots(figsize=(16, 5.5), layout='constrained')
    # 解析窓の枠の色（r, b, g, ...）が見分けられるよう、全体図の B-scan は灰色で描く
    ax.imshow(img, aspect='auto', cmap='gray', vmin=-amp, vmax=amp,
              extent=[0, ntr * DX_M, T0_NS + nt * DT_NS, T0_NS])
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
    ax.imshow(blk, aspect='auto', cmap='seismic', vmin=-amp, vmax=amp,
              extent=[seg['x0'], seg['x1'], t0_ns + TIME_LEN_NS, t0_ns])
    ax.set_xlabel('Moving distance [m]')
    ax.set_ylabel('Time [ns]')
    ax.set_title('B-scan')
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
    # 波数軸は設定の固定値（地下構造の解析 geometry_2dfft.py と同じ）            # 軸の統一
    LIM['kx'] = tuple(K_LIM_1D)
    LIM['kx_map'] = (-MAP_KX_LIM, MAP_KX_LIM)
    LIM['ky'] = tuple(K_LIM_1D)
    LIM['ky_map'] = (0.0, MAP_KY_LIM)
    # 縦軸は、表示する波数の範囲に入る値だけで決める（主要部分が切れないように）
    tops_x, tops_y, bots_x, bots_y, map_max = [], [], [], [], []
    for d in all_L.values():
        g = d['g']
        kxp, kyp = g.kx[g.kx_pos], g.ky[g.ky_pos]
        mkx = (kxp >= LIM['kx'][0]) & (kxp <= LIM['kx'][1])
        mky = (kyp <= ky_max() + 1e-9) & (kyp >= max(2 * g.dky, LIM['ky'][0])) & (kyp <= LIM['ky'][1])
        for sx, sy in d['seg_marg']:                       # 区間ごとの線（上端を決める）
            tops_x.append(_db(sx[mkx]).max())
            tops_y.append(_db(sy[mky]).max())
        for byL in d['res'].values():                      # 平均の線（下端を決める）
            curves = [(byL['Sx'], byL['Sy'])] + [(r['Sx'], r['Sy']) for r in byL['by_win'].values()]
            for sx, sy in curves:
                bots_x.append(_db(sx[mkx]).min())
                bots_y.append(_db(sy[mky]).min())
            map_max.append(_db(byL['S'][g.ky <= ky_max() + 1e-9]).max())
    for key, user, tops, bots in (('sx', SX_YLIM_DB, tops_x, bots_x), ('sy', SY_YLIM_DB, tops_y, bots_y)):
        if user is not None:
            LIM[key] = tuple(user)
        else:
            top = float(max(tops)) + 3.0
            bot = max(float(min(bots)) - 3.0, top - SPEC_DB_RANGE_MAX)
            LIM[key] = (bot, top)
    vmax = float(max(map_max)) + 5.0
    LIM['map'] = (vmax - DB_RANGE, vmax)
    print('共通の軸: k_x, k_y {:g}-{:g} rad/m（2 次元図 |k_x| <= {:g}、k_y <= {:g}）、'
          'S_x {:.1f}-{:.1f} dB、S_y {:.1f}-{:.1f} dB、2 次元図 {:.1f}-{:.1f} dB'.format(
              *LIM['kx'], MAP_KX_LIM, MAP_KY_LIM, *LIM['sx'], *LIM['sy'], *LIM['map']))


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
    print('merged.out ファイルのパスを入力してください:')
    out_path = input().strip()
    if not os.path.exists(out_path):
        print('エラー: 指定されたファイルが存在しません')
        exit(1)

    out_path = os.path.abspath(out_path)
    outdir = os.path.join(os.path.dirname(out_path), OUTPUT_DIRNAME)
    os.makedirs(outdir, exist_ok=True)

    data, dt, title = read_out(out_path)
    nt, ntr = data.shape
    print('読み込み: {} 行（時間）x {} トレース、dt = {:.4f} ps、時間窓 {:.1f} ns'.format(
        nt, ntr, dt * 1e12, nt * dt * 1e9))

    rec_path, rec = find_record(out_path)
    if rec:
        dx = float(rec['bscan']['step_m'])
        h = float(rec['antenna']['height_m'])
        sep = float(rec['antenna']['txrx_sep_m'])
        fwav = float(rec['antenna']['waveform'][2])
        label = 'case{:02d}_{}'.format(int(rec['case_id']), rec['case_label'])
        if abs(float(rec['medium']['eps_bg']) - EPS_R) > 1e-6:
            print('  注意: 記録の ε_r = {} と解析の EPS_R = {} が違います'.format(
                rec['medium']['eps_bg'], EPS_R))
        if int(rec['bscan']['n']) != ntr:
            print('  注意: 記録のトレース数 {} と .out のトレース数 {} が違います'.format(
                rec['bscan']['n'], ntr))
        print('記録ファイル: {}'.format(rec_path))
        print('  ケース {}、a_x = {} m、a_y = {} m、r = {}、s_rho = {}'.format(
            label, rec['random']['corr_x_m'], rec['random']['corr_z_m'],
            rec['random']['roughness_r'], rec['random']['rel_std']))
    else:
        dx, h, sep, fwav = DX_M_DEFAULT, ANT_HEIGHT_DEFAULT, TXRX_SEP_DEFAULT, WAVEFORM_FREQ_DEFAULT
        label = 'sim'
        print('  注意: 記録ファイルが見つからないので既定値を使います（トレース間隔 {} m）'.format(dx))
    print('  トレース間隔 {} m、測線長 {:.3f} m'.format(dx, ntr * dx))

    # ---- 1. 背景除去（時刻 0 を決める前に、平均トレースは残しておく）
    mean_trace = data.mean(axis=1)
    if BACKGROUND_REMOVAL:
        data = data - mean_trace[:, None]
        print('背景除去: 平均トレースを引きました')

    # ---- 2. 時刻 0
    t = np.arange(nt) * dt
    if TIME_ZERO == 'envelope':
        m = t <= TIME_ZERO_SEARCH_NS * 1e-9
        t0 = float(t[m][np.argmax(envelope(mean_trace)[m])])
    elif TIME_ZERO == 'theory':
        # gprMax の gaussiandot はピークが 1/f の遅れ。地表反射は鏡像点までの往復
        t0 = 1.0 / fwav + 2.0 * np.sqrt(h ** 2 + (sep / 2) ** 2) / C0
    elif TIME_ZERO == 'manual':
        t0 = TIME_ZERO_NS * 1e-9
    else:
        raise SystemExit("TIME_ZERO は 'envelope' / 'theory' / 'manual'")
    print('時刻 0: {:.3f} ns（方法 {}）'.format(t0 * 1e9, TIME_ZERO))

    # ---- 3. サンプリング間隔（時刻 0 からの格子に直す）
    if RESAMPLE_DT_NS:
        data, dt_new = resample_time(data, dt, t0, RESAMPLE_DT_NS * 1e-9)
        print('サンプリング: {:.4f} ps → {:.4f} ns（時刻 0 から {} 点）'.format(
            dt * 1e12, dt_new * 1e9, data.shape[0]))
    else:
        data = data[int(round(t0 / dt)):]
        dt_new = dt
    if T_START_NS + TIME_LEN_NS > data.shape[0] * dt_new * 1e9:
        raise SystemExit('解析窓 {}-{} ns が時間窓（時刻 0 から {:.1f} ns）を超えています'.format(
            T_START_NS, T_START_NS + TIME_LEN_NS, data.shape[0] * dt_new * 1e9))

    # ---- 4. 解析（ここから先は lpr_bscan_2dfft.py と同じ処理）
    global DX_M, DT_NS, T0_NS, WINDOWS
    DX_M = dx
    DT_NS = dt_new * 1e9
    T0_NS = 0.0
    WINDOWS = [dict(name='S', x_m=(0.0, ntr * dx), t0_ns=T_START_NS, group=label)]

    if SAVE_PROCESSED_TXT:
        p = os.path.join(outdir, 'processed_bscan.txt')
        np.savetxt(p, data, fmt='%.6e')
        print('-> {}（前処理後。DT_NS = {:.4f}、DX_M = {}、T0_NS = 0）'.format(p, dt_new * 1e9, dx))

    extra = dict(out_file=out_path, record=rec_path, case=label, title=title,
                 dt_sim_s=dt, time_zero_ns=t0 * 1e9, time_zero_method=TIME_ZERO,
                 background_removal=BACKGROUND_REMOVAL, resample_dt_ns=RESAMPLE_DT_NS,
                 trace_step_m=dx, n_traces=ntr)
    run_analysis(data, outdir, out_path, extra_info=extra)


if __name__ == '__main__':
    main()