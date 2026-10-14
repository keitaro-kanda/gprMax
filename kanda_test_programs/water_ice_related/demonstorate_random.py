#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demonstrate_random.py
=====================

Level 6 のランダム場（Jiang et al. 2013 のスペクトル法）について、
  (1) 生成手順（ステップ 1-6）を 1 段ずつ図にする
  (2) シードの違いによる影響を見る
  (3) 水平相関長 a_x、鉛直相関長 a_y、粗さ係数 r を 1 つずつ変えたときの
      ランダム場と、その 2 次元フーリエ変換（空間スペクトル）を比べる
ための学習・パラメータ検討用スクリプト。図の文字はすべて英語。

既定の設定は Level 6 の Ascan.in と同じ（地下領域 12 x 3 m、dx 2.5 mm、
生成格子 19 x 7 m、RandomState）。同じ a_x, a_y, r, seed なら、ここで作る R は
Ascan.in が gprMax に渡す R とビット単位で同じになる。

用語
    a_x : 水平相関長 [m]（Jiang 式 9 の a。自己相関が 1/e になる距離）
    a_y : 鉛直（深さ方向）相関長 [m]（Jiang の b。gprMax の y 方向）
    r   : 粗さ係数（0 = ガウス型、1 = 指数型）

出力（BASE_DIR 以下に 4 つのサブディレクトリを作る）
    steps_and_seeds/
        step1_acf.png ... step6_normalize.png   ステップ 1-6 の詳細
        overview_seed<N>.png                    流れ図（OVERVIEW_SEEDS の各シード）
        R_demo.npy                              BASE の R（[x, z]、z は地表から下向き）
    vary_ax/   a_x だけを変える（a_y, r, seed は BASE のまま）
    vary_ay/   a_y だけを変える
    vary_r/    r だけを変える
        fields_vary_<p>.png        ランダム場 R を並べた図
        fft_vary_<p>.png           2 次元パワースペクトルと、水平・鉛直方向の
                                   1 次元スペクトル（実現値と理論値）
        reflectivity_vary_<p>.png  鉛直方向の反射率スペクトル k_z^2 S_z(k_z)
        summary_vary_<p>.png       ランダム場と水平・鉛直方向のスペクトルを 1 枚に
        spectra_vary_<p>.npz       1 次元スペクトルの数値（実データとの比較用）

スペクトルの単位
    分散 1 の場のパワースペクトル密度 S [m^2]（∫∫ S dkx dkz = 1）を dB で表す。
    1 次元スペクトルは S を他方の波数で積分したもの [m]（∫ S_x dkx = 1）。
    波数は rad/m（波長 λ = 2π / k）。

必要なもの: numpy, matplotlib（scipy は不要）
メモリ: 生成格子 19 x 7 m（7600 x 2800）で 1 回あたりピーク約 2 GB。
"""

import hashlib
import os
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, FancyArrowPatch

# =============================================================================
# 設定  [EDIT HERE]
# =============================================================================
BASE_DIR = '/Volumes/SSD_Kanda_BUFFALO/test_programs_output/demonstrate_random'

# --- 格子（Level 6 の Ascan.in と同じにしておくと、R が gprMax に渡るものと一致する）
DX = 0.0025                 # 格子間隔 [m]
DOMAIN_X_M = 12.0           # 地下領域の水平幅 [m]（PML を含む計算領域の全幅）
DOMAIN_Z_M = 3.0            # 地下領域の深さ [m]（地表から領域底まで）
GEN_SIZE_M = (19.0, 7.0)    # 生成格子 (Lx, Lz) [m]。全ての比較で共通・固定
WRAP_EPS = 1e-3             # 周期性の条件に使う自己相関のしきい値

# --- 基準のパラメータ（比較では、変えるもの以外はこの値に固定する）
BASE = dict(ax=0.75, ay=0.25, r=0.0, seed=0)

# --- (2) シードの比較：overview を描くシード（1 つ目が BASE のシードと同じだと効率的）
OVERVIEW_SEEDS = (0, 1)

# --- (3) パラメータの比較：変える値（3-4 個を推奨）
VARY = {
    'ax': (0.15, 0.35, 0.75, 1.5),     # [m]
    'ay': (0.07, 0.15, 0.25, 0.5),     # [m]
    'r':  (0.0, 0.25, 0.5, 1.0),
}

# --- 実行するもの
RUN_STEPS_AND_SEEDS = True
RUN_VARY = ('ax', 'ay', 'r')           # 空のタプルにするとパラメータ比較をしない
CHECK_AGAINST_FUNCTION = True          # ステップ版と一括関数がビット一致するか検算

# --- 表示
FFT_KX_MAX = 30.0           # 2 次元スペクトル図の表示範囲 |kx| [rad/m]
FFT_KZ_MAX = 160.0          # 同 |kz| [rad/m]（ブラッグ帯の上端 2k(2 GHz) ≈ 150）
FFT_DB_RANGE = 80.0         # 2 次元スペクトル図の色の幅 [dB]（最大値から下へ）
SPEC_DB_RANGE = 120.0       # 1 次元スペクトル図の縦軸の幅 [dB]
FFT_WINDOW = 'blackmanharris'  # 'blackmanharris'（漏れ -92 dB）/ 'hann'（-31 dB）/ 'none'
                               # 実データ（B-scan）と比べるときは同じ窓を使うこと
R_COMPARE = (0.0, 0.5, 1.0) # ステップ 1・3 で自己相関とスペクトルの形を比べる r
N_REFRACT = 1.79            # ブラッグ帯の計算に使う屈折率（氷層深さ）
BAND_GHZ = (0.5, 2.0)       # LUPEX 帯域 [GHz]
SHOW = False                # True で画面にも表示
DPI = 130

C0 = 2.998e8                # 光速 [m/s]

SUBDIRS = {'steps': 'steps_and_seeds', 'ax': 'vary_ax', 'ay': 'vary_ay', 'r': 'vary_r'}
PNAME = {'ax': 'a_x', 'ay': 'a_y', 'r': 'r'}                          # 図に出す名前
PNAME_JA = {'ax': 'a_x（水平相関長）', 'ay': 'a_y（鉛直相関長）', 'r': 'r（粗さ係数）'}  # 端末表示用
PUNIT = {'ax': ' m', 'ay': ' m', 'r': ''}
WINDOW_NAME = {'blackmanharris': 'Blackman-Harris', 'hann': 'Hann', 'none': 'none'}


# =============================================================================
# 共通：表示と保存
# =============================================================================
def _stride(shape, max_px=1200):
    """表示用の間引き幅（長辺を max_px 程度に落とす）"""
    return max(1, int(np.ceil(max(shape) / max_px)))


def _outdir(key):
    return os.path.join(BASE_DIR, SUBDIRS[key])


def _save(fig, key, name):
    path = os.path.join(_outdir(key), name)
    fig.savefig(path, dpi=DPI)
    print(f'    -> {path}')
    if SHOW:
        plt.show()
    plt.close(fig)


def _prepare_dirs():
    """/Volumes 以下でボリュームがマウントされていなければ、フォルダを作らずに止める。"""
    parts = os.path.abspath(BASE_DIR).split(os.sep)
    if len(parts) > 2 and parts[1] == 'Volumes':
        vol = os.sep + os.path.join(parts[1], parts[2])
        if not os.path.isdir(vol):
            raise SystemExit(f'出力先のボリューム {vol} が見つかりません。'
                             'SSD を接続するか、BASE_DIR を変更してください。')
    for key in SUBDIRS:
        os.makedirs(_outdir(key), exist_ok=True)


def _fmt(key, v):
    return f'{v:g}{PUNIT[key]}'


def _settings(p, keys=('ax', 'ay', 'r'), seed=True):
    """図のタイトル用の設定の文字列（例: a_x = 0.75 m, a_y = 0.25 m, r = 0, seed = 0）"""
    s = ', '.join(f'{PNAME[k]} = {_fmt(k, p[k])}' for k in keys)
    return s + (f', seed = {p["seed"]}' if seed else '')


# =============================================================================
# 共通：スペクトル法の中身（Ascan.in と同じ計算）
# =============================================================================
def acf(x, z, ax, ay, r):
    """Jiang 式 (9) の楕円体自己相関  C = exp[-((x/a_x)^2 + (z/a_y)^2)^(1/(1+r))]"""
    return np.exp(-((x / ax) ** 2 + (z / ay) ** 2) ** (1.0 / (1.0 + r)))


def wrap_length(corr_m, r, eps=WRAP_EPS):
    """自己相関 C が eps まで下がる距離 p [m]"""
    return corr_m * np.log(1.0 / eps) ** ((1.0 + r) / 2.0)


def grid_info():
    nx = int(round(DOMAIN_X_M / DX))
    nz = int(round(DOMAIN_Z_M / DX))
    Gx = int(round(GEN_SIZE_M[0] / DX))
    Gz = int(round(GEN_SIZE_M[1] / DX))
    return dict(nx=nx, nz=nz, Gx=Gx, Gz=Gz, Lx=Gx * DX, Lz=Gz * DX, Dx=nx * DX, Dz=nz * DX)


def grid_requirement(p):
    """パラメータ p に必要な生成格子の大きさ [m]（L >= D + p かつ L >= 2p）"""
    g = grid_info()
    px, pz = wrap_length(p['ax'], p['r']), wrap_length(p['ay'], p['r'])
    return max(g['Dx'] + px, 2 * px), max(g['Dz'] + pz, 2 * pz), px, pz


def check_all_grids():
    """全ての比較で使う生成格子が足りているかを、計算を始める前に確かめる。"""
    g = grid_info()
    cases = [dict(BASE)]
    for key in RUN_VARY:
        for v in VARY[key]:
            cases.append(dict(BASE, **{key: v}))
    worst_x = max(grid_requirement(c)[0] for c in cases)
    worst_z = max(grid_requirement(c)[1] for c in cases)
    if g['Lx'] < worst_x - 1e-9 or g['Lz'] < worst_z - 1e-9:
        raise SystemExit('GEN_SIZE_M = {} m では足りません。比較する全ての組み合わせに'
                         '必要な大きさは x >= {:.2f} m, z >= {:.2f} m です。'
                         .format(GEN_SIZE_M, worst_x, worst_z))
    print(f'生成格子 {g["Lx"]:.1f} x {g["Lz"]:.1f} m（必要な最大 {worst_x:.2f} x {worst_z:.2f} m）: OK')


def make_random_field_spectral(nx, nz, dx, a, b, r, seed, gen_size_m, wrap_eps=1e-3):
    """一括版（ステップ 1-6）。Ascan.in の同名関数と同じ計算。R[ix, iz]、iz は地表から下向き。"""
    Gx = int(round(gen_size_m[0] / dx))
    Gz = int(round(gen_size_m[1] / dx))
    Lx, Lz = Gx * dx, Gz * dx
    px = a * np.log(1.0 / wrap_eps) ** ((1.0 + r) / 2.0)
    pz = b * np.log(1.0 / wrap_eps) ** ((1.0 + r) / 2.0)
    if Lx < max(nx * dx + px, 2 * px) or Lz < max(nz * dx + pz, 2 * pz):
        raise ValueError('生成格子が小さすぎます')
    x = np.fft.fftfreq(Gx, d=1.0 / Gx) * dx
    z = np.fft.fftfreq(Gz, d=1.0 / Gz) * dx
    X, Z = np.meshgrid(x, z, indexing='ij')
    C = np.exp(-((X / a) ** 2 + (Z / b) ** 2) ** (1.0 / (1.0 + r)))
    del X, Z
    F = np.fft.fft2(C).real
    del C
    F[F < 0] = 0.0
    rng = np.random.RandomState(seed)               # 乱数列が凍結された旧来の系統
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(Gx, Gz))
    f = np.fft.ifft2(np.sqrt(F) * np.exp(1j * theta)).real
    del F, theta
    fc = f[:nx, :nz]
    return (fc - fc.mean()) / fc.std()


def field_for(p):
    g = grid_info()
    return make_random_field_spectral(g['nx'], g['nz'], DX, p['ax'], p['ay'], p['r'],
                                      p['seed'], GEN_SIZE_M, WRAP_EPS)


def acf_along(R, axis, maxlag):
    """R の 1 次元自己相関（もう一方の軸で平均）。ゼロ詰め FFT で計算し、重なり数で割る。"""
    N = R.shape[axis]
    maxlag = min(maxlag, N - 1)
    X = np.fft.rfft(R - R.mean(), n=2 * N, axis=axis)
    p = np.fft.irfft(np.abs(X) ** 2, n=2 * N, axis=axis)
    p = p.take(np.arange(maxlag + 1), axis=axis).mean(axis=1 - axis)
    p /= (N - np.arange(maxlag + 1))
    return p / p[0]


def realized_1e(R, axis, maxlag_m=4.0):
    lag = int(round(maxlag_m / DX))
    c = acf_along(R, axis, lag)
    below = np.where(c < np.exp(-1))[0]
    return below[0] * DX if below.size else float('nan')


def bragg_band():
    k_lo = 2 * np.pi * BAND_GHZ[0] * 1e9 * N_REFRACT / C0
    k_hi = 2 * np.pi * BAND_GHZ[1] * 1e9 * N_REFRACT / C0
    return 2 * k_lo, 2 * k_hi


# =============================================================================
# 共通：実現した場の 2 次元フーリエ変換（パワースペクトル）
# =============================================================================
def window_1d(n, kind=None):
    """窓関数。有限の領域を FFT すると端の不連続がスペクトルに漏れ（リーク）を作るので、
    端をなめらかに 0 に落とす。漏れの床は Hann で約 -31 dB 下から k^-3 で減衰、
    4 項 Blackman-Harris で約 -92 dB。r = 0 のように急に落ちるスペクトルを見るには後者が要る。"""
    kind = FFT_WINDOW if kind is None else kind
    if kind == 'none':
        return np.ones(n)
    if kind == 'hann':
        return np.hanning(n)
    if kind == 'blackmanharris':
        a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
        t = 2 * np.pi * np.arange(n) / (n - 1)
        return a0 - a1 * np.cos(t) + a2 * np.cos(2 * t) - a3 * np.cos(3 * t)
    raise ValueError('FFT_WINDOW は blackmanharris / hann / none')


def periodogram_2d(R):
    """R[ix, iz] の 2 次元パワースペクトル密度 S(kx, kz) [m^2]（窓は FFT_WINDOW）。
    正規化は ∫∫ S dkx dkz = 分散（≈ 1）になるようにとる。fftshift 済みで返す。"""
    nx, nz = R.shape
    w = np.outer(window_1d(nx), window_1d(nz))
    X = np.fft.fft2((R - R.mean()) * w)
    dkx = 2 * np.pi / (nx * DX)
    dkz = 2 * np.pi / (nz * DX)
    S = np.abs(X) ** 2 / (nx * nz * (w ** 2).mean() * nx * nz * dkx * dkz)
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=DX)
    kz = 2 * np.pi * np.fft.fftfreq(nz, d=DX)
    return np.fft.fftshift(S), np.fft.fftshift(kx), np.fft.fftshift(kz), dkx, dkz


def spectra_1d(S, kx, kz, dkx, dkz):
    """2 次元スペクトルを他方の波数で積分した 1 次元スペクトル（正の波数側、両側分を合算）"""
    Sx = S.sum(axis=1) * dkz            # S_x(kx)（両側）
    Sz = S.sum(axis=0) * dkx            # S_z(kz)（両側）
    mx, mz = kx > 0, kz > 0
    # 実数の場なので S(k) = S(-k)。正の波数側を 2 倍して片側スペクトルにする
    return kx[mx], 2 * Sx[mx], kz[mz], 2 * Sz[mz]


def theory_1d(p, k, axis):
    """理論の 1 次元スペクトル（片側、∫_0^∞ S dk = 1）：C の軸上の断面のフーリエ変換 × 2/(2π)"""
    L = 200.0 * max(p['ax'], p['ay']) * (1 + p['r'])      # 十分長い区間
    n = int(2 ** np.ceil(np.log2(L / (DX / 2))))
    d = DX / 2
    s = np.fft.fftfreq(n, d=1.0 / n) * d
    c = acf(s, 0 * s, p['ax'], 1.0, p['r']) if axis == 'x' else acf(0 * s, s, 1.0, p['ay'], p['r'])
    Fk = np.fft.fft(c).real * d / (2 * np.pi) * 2
    kk = 2 * np.pi * np.fft.fftfreq(n, d=d)
    m = kk > 0
    return np.interp(k, kk[m], np.maximum(Fk[m], 1e-300))


# =============================================================================
# (1) ステップ 1-6
# =============================================================================
def step1_define_acf(p):
    print('\n[ステップ 1] 自己相関関数 C(x, z) を決める')
    print(f'    a_x = {p["ax"]} m（水平）, a_y = {p["ay"]} m（鉛直）, r = {p["r"]}')
    return p


def draw_acf_map(ax, p):
    x = np.linspace(-2.0, 2.0, 401)
    z = np.linspace(-1.0, 1.0, 201)
    X, Z = np.meshgrid(x, z, indexing='xy')
    C = acf(X, Z, p['ax'], p['ay'], p['r'])
    im = ax.imshow(C, extent=[x[0], x[-1], z[-1], z[0]], cmap='viridis', vmin=0, vmax=1,
                   aspect='equal')
    ax.contour(X, Z, C, levels=[np.exp(-1)], colors='w', linewidths=1.2)
    ax.annotate('', xy=(p['ax'], 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='w'))
    ax.annotate('', xy=(0, p['ay']), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='w'))
    ax.text(p['ax'] / 2, -0.06, 'a_x', color='w', ha='center', va='bottom')
    ax.text(0.06, p['ay'] / 2, 'a_y', color='w', ha='left', va='center')
    ax.set_xlabel('Δx [m]')
    ax.set_ylabel('Δz [m]')
    ax.set_title('(a) C(Δx, Δz)')
    ax.legend(handles=[Line2D([], [], color='w', lw=1.2, label='C = 1/e')], loc='lower right',
              fontsize=8, facecolor='0.3', labelcolor='w', framealpha=0.6)
    return im


def plot_step1(p):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained')
    im = draw_acf_map(axs[0], p)
    fig.colorbar(im, ax=axs[0], fraction=0.046, label='C')
    d = np.linspace(0, 2.0, 801)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(R_COMPARE)))
    for ax, logy in [(axs[1], False), (axs[2], True)]:
        for col, r in zip(colors, R_COMPARE):
            lw = 2.4 if np.isclose(r, p['r']) else 1.2
            ax.plot(d, acf(d, 0 * d, p['ax'], p['ay'], r), color=col, lw=lw,
                    label=f'C(Δx, 0), r = {r:g}')
            ax.plot(d, acf(0 * d, d, p['ax'], p['ay'], r), color=col, lw=lw, ls='--',
                    label=f'C(0, Δz), r = {r:g}')
        ax.axhline(np.exp(-1), color='gray', lw=0.8, ls=':')
        ax.set_xlabel('Lag [m]')
        ax.set_ylabel('C')
        if logy:
            ax.set_yscale('log')
            ax.set_ylim(1e-8, 1.5)
            ax.axhline(WRAP_EPS, color='crimson', lw=0.8, ls=':')
            ax.text(2.0, WRAP_EPS * 1.6, f'WRAP_EPS = {WRAP_EPS:g}', color='crimson',
                    ha='right', fontsize=8)
            ax.set_title('(c) Cross-sections (log scale)')
        else:
            ax.set_ylim(0, 1.05)
            ax.text(1.95, np.exp(-1) + 0.02, '1/e', color='gray', ha='right')
            ax.set_title('(b) Cross-sections')
            ax.legend(fontsize=7.5)
    fig.suptitle(f'Step 1: autocorrelation function ({_settings(p, seed=False)})', fontsize=13)
    _save(fig, 'steps', 'step1_acf.png')


def step2_make_grid(p):
    print('\n[ステップ 2] 生成格子を用意する')
    g = grid_info()
    need_x, need_z, px, pz = grid_requirement(p)
    g.update(px=px, pz=pz)
    print(f'    地下領域 : {g["nx"]} x {g["nz"]} セル（{g["Dx"]:.2f} x {g["Dz"]:.2f} m）')
    print(f'    生成格子 : {g["Gx"]} x {g["Gz"]} セル（{g["Lx"]:.2f} x {g["Lz"]:.2f} m）')
    print(f'    余白 p   : x {px:.2f} m, z {pz:.2f} m / 必要 x >= {need_x:.2f}, z >= {need_z:.2f} m')
    return g


def draw_grid_schematic(ax, g, annotate=True):
    Lx, Lz, Dx, Dz = g['Lx'], g['Lz'], g['Dx'], g['Dz']
    for i in range(2):
        for j in range(2):
            main = (i == 0 and j == 0)
            ax.add_patch(Rectangle((i * Lx, j * Lz), Lx, Lz, fill=main,
                                   fc='#eef3fb' if main else 'none',
                                   ec='k' if main else 'gray', lw=1.5 if main else 0.8,
                                   ls='-' if main else '--'))
            ax.add_patch(Rectangle((i * Lx, j * Lz), Dx, Dz, fill=True,
                                   fc='#d62728' if main else '#f4b6b6',
                                   ec='#d62728', alpha=0.85 if main else 0.6))
    ax.text(Lx * 0.5, Lz * 0.62, 'generation grid', ha='center', va='center', fontsize=9)
    ax.text(Lx * 1.5, Lz * 0.55, 'periodic copies', ha='center', va='center',
            fontsize=9, color='gray')
    ax.text(Dx / 2, Dz / 2, 'domain', ha='center', va='center', color='w', fontsize=8)
    if annotate:
        _annotate_margins(ax, g)
    ax.set_xlim(-0.3, 2 * Lx + 0.3)
    ax.set_ylim(2 * Lz + 0.3, -0.3)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')
    ax.set_title('(a) Generation grid and domain')


def _annotate_margins(ax, g):
    Lx, Lz, Dx, Dz = g['Lx'], g['Lz'], g['Dx'], g['Dz']
    yy = Dz / 2
    ax.add_patch(FancyArrowPatch((Dx, yy), (Lx, yy), arrowstyle='<->', mutation_scale=10,
                                 color='#1f4e9c'))
    ax.text((Dx + Lx) / 2, yy + 0.4, f'L_x − D_x = {Lx - Dx:.1f} m\np_x = {g["px"]:.2f} m',
            ha='center', va='top', fontsize=8, color='#1f4e9c')
    xx = Dx / 4
    ax.add_patch(FancyArrowPatch((xx, Dz), (xx, Lz), arrowstyle='<->', mutation_scale=10,
                                 color='#1f4e9c'))
    ax.text(xx + 0.3, (Dz + Lz) / 2, f'L_z − D_z = {Lz - Dz:.1f} m, p_z = {g["pz"]:.2f} m',
            ha='left', va='center', fontsize=8, color='#1f4e9c')


def _draw_wrap_profile(ax, g, p, axis, label):
    if axis == 'x':
        G, L, D, corr = g['Gx'], g['Lx'], g['Dx'], p['ax']
    else:
        G, L, D, corr = g['Gz'], g['Lz'], g['Dz'], p['ay']
    pos = np.arange(G) * DX
    dist = np.fft.fftfreq(G, d=1.0 / G) * DX
    c = acf(dist, 0 * dist, corr, 1.0, p['r'])
    ax.semilogy(pos, np.maximum(c, 1e-300), color='k', lw=1.2, label='C on grid')
    ax.axvspan(0, D, color='#d62728', alpha=0.12, label=f'0–D ({D:.1f} m)')
    ax.axvline(L - D, color='#1f4e9c', ls='--', lw=1, label=f'L − D ({L - D:.1f} m)')
    ax.axhline(WRAP_EPS, color='crimson', ls=':', lw=0.8, label=f'WRAP_EPS = {WRAP_EPS:g}')
    ax.set_ylim(1e-8, 2)
    ax.set_xlim(0, L)
    ax.set_xlabel(f'{axis} on grid [m]')
    ax.set_ylabel('C')
    ax.set_title(f'{label} C along {axis} (L = {L:.1f} m)')
    ax.legend(fontsize=8, loc='lower center')


def plot_step2(g, p):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.0), gridspec_kw={'width_ratios': [1.6, 1, 1]},
                            layout='constrained')
    draw_grid_schematic(axs[0], g)
    _draw_wrap_profile(axs[1], g, p, 'x', '(b)')
    _draw_wrap_profile(axs[2], g, p, 'z', '(c)')
    fig.suptitle(f'Step 2: generation grid ({g["Lx"]:g} × {g["Lz"]:g} m, '
                 f'domain {g["Dx"]:g} × {g["Dz"]:g} m, dx = {DX * 1e3:g} mm)', fontsize=13)
    _save(fig, 'steps', 'step2_grid.png')


def sample_acf_on_grid(g, p):
    x = np.fft.fftfreq(g['Gx'], d=1.0 / g['Gx']) * DX
    z = np.fft.fftfreq(g['Gz'], d=1.0 / g['Gz']) * DX
    X, Z = np.meshgrid(x, z, indexing='ij')
    return acf(X, Z, p['ax'], p['ay'], p['r'])


def wavenumbers(g):
    kx = 2 * np.pi * np.fft.fftfreq(g['Gx'], d=DX)
    kz = 2 * np.pi * np.fft.fftfreq(g['Gz'], d=DX)
    return kx, kz, (kx[1] - kx[0]) * (kz[1] - kz[0])


def to_psd_density(F, g, dk):
    return F / (g['Gx'] * g['Gz']) / dk


def step3_psd(g, p):
    print('\n[ステップ 3] パワースペクトル F を求める')
    C = sample_acf_on_grid(g, p)
    Fc = np.fft.fft2(C)
    imag_rel = np.abs(Fc.imag).max() / np.abs(Fc.real).max()
    F = Fc.real.copy()
    del Fc
    neg_frac = -F[F < 0].sum() / F.sum()
    F[F < 0] = 0.0
    print(f'    負の値の割合: {neg_frac:.2e} / 虚部・実部の最大比 {imag_rel:.1e}')
    kx, kz, dk = wavenumbers(g)
    st = _stride(C.shape)
    out = dict(F=F, kx=kx, kz=kz, dk=dk, C_show=C[::st, ::st].copy())
    del C
    sec = {}
    x = np.fft.fftfreq(g['Gx'], d=1.0 / g['Gx']) * DX
    z = np.fft.fftfreq(g['Gz'], d=1.0 / g['Gz']) * DX
    for r in sorted(set(R_COMPARE) | {p['r']}):
        csum = np.zeros(g['Gz'])
        for i0 in range(0, g['Gx'], 512):
            X, Z = np.meshgrid(x[i0:i0 + 512], z, indexing='ij')
            csum += acf(X, Z, p['ax'], p['ay'], r).sum(axis=0)
        sec[r] = to_psd_density(np.fft.fft(csum).real, g, dk)
    out['section'] = sec
    out['bragg'] = bragg_band()
    return out


def _shifted_window(arr2d, kx, kz, kx_max, kz_max, shifted=False):
    A = arr2d if shifted else np.fft.fftshift(arr2d)
    kxs, kzs = (kx, kz) if shifted else (np.fft.fftshift(kx), np.fft.fftshift(kz))
    ix = np.where(np.abs(kxs) <= kx_max)[0]
    iz = np.where(np.abs(kzs) <= kz_max)[0]
    return A[ix[0]:ix[-1] + 1, iz[0]:iz[-1] + 1], kxs[ix], kzs[iz]


def _bragg_lines(ax, bragg, lw=0.8):
    for kb in bragg:
        for sgn in (1, -1):
            ax.axhline(sgn * kb, color='gold', lw=lw, ls='--')


def _bragg_handle():
    return Line2D([], [], color='gold', ls='--', lw=1, label='Bragg band (2k)')


def draw_psd_map(ax, g, s3, title, kx_max=FFT_KX_MAX, kz_max=FFT_KZ_MAX):
    S = to_psd_density(s3['F'], g, s3['dk'])
    W, kxw, kzw = _shifted_window(S, s3['kx'], s3['kz'], kx_max, kz_max)
    WdB = 10 * np.log10(np.maximum(W, 1e-300))
    im = ax.imshow(WdB.T, extent=[kxw[0], kxw[-1], kzw[0], kzw[-1]], cmap='magma',
                   vmin=-160, vmax=WdB.max(), aspect='auto', origin='lower')
    _bragg_lines(ax, s3['bragg'])
    ax.legend(handles=[_bragg_handle()], loc='upper right', fontsize=7.5, framealpha=0.7)
    ax.set_xlabel('k_x [rad/m]')
    ax.set_ylabel('k_z [rad/m]')
    ax.set_title(title)
    return im


def plot_step3(g, p, s3):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.0), gridspec_kw={'width_ratios': [1.5, 1, 1]},
                            layout='constrained')
    im0 = axs[0].imshow(s3['C_show'].T, extent=[0, g['Lx'], g['Lz'], 0], cmap='viridis',
                        vmin=0, vmax=1, aspect='equal')
    axs[0].set_xlabel('x on grid [m]')
    axs[0].set_ylabel('z on grid [m]')
    axs[0].set_title('(a) C on the generation grid')
    fig.colorbar(im0, ax=axs[0], fraction=0.03, label='C')
    im1 = draw_psd_map(axs[1], g, s3, '(b) F(k_x, k_z)')
    fig.colorbar(im1, ax=axs[1], fraction=0.046, label='F [dB re 1 m²]')
    kz = s3['kz']
    m = kz > 0
    o = np.argsort(kz[m])
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(s3['section'])))
    for col, (r, s) in zip(colors, sorted(s3['section'].items())):
        lw = 2.4 if np.isclose(r, p['r']) else 1.2
        axs[2].semilogx(kz[m][o], 10 * np.log10(np.maximum(s[m][o], 1e-300)), color=col,
                        lw=lw, label=f'r = {r:g}')
    axs[2].axvspan(*s3['bragg'], color='gold', alpha=0.25, label='Bragg band (2k)')
    axs[2].set_xlim(1, np.pi / DX)
    axs[2].set_ylim(-160, 0)
    axs[2].set_xlabel('k_z [rad/m]')
    axs[2].set_ylabel('F(0, k_z) [dB re 1 m²]')
    axs[2].set_title('(c) F(0, k_z)')
    axs[2].legend(fontsize=8, loc='lower left')
    fig.suptitle(f'Step 3: power spectrum F = FFT[C] ({_settings(p, seed=False)})', fontsize=13)
    _save(fig, 'steps', 'step3_psd.png')


def step4_random_phase(g, seed):
    """位相 θ を RandomState(seed) の一様乱数で引く（乱数はここだけ。NEP 19 で乱数列が凍結）"""
    print(f'\n[ステップ 4] 一様乱数の位相 θ を引く（seed = {seed}）')
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(g['Gx'], g['Gz']))
    print(f'    θ の先頭 3 値（指紋）: {theta[0, 0]:.15f}, {theta[0, 1]:.15f}, {theta[0, 2]:.15f}')
    return dict(theta=theta, seed=seed)


def draw_theta_patch(ax, s4, title, n=64):
    im = ax.imshow(s4['theta'][:n, :n].T, cmap='twilight', vmin=0, vmax=2 * np.pi, origin='lower')
    ax.set_xlabel('k_x index')
    ax.set_ylabel('k_z index')
    ax.set_title(title)
    return im


def plot_step4(g, s3, s4):
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.8), layout='constrained')
    im = draw_theta_patch(axs[0], s4, '(a) θ (64 × 64 subset)')
    cb = fig.colorbar(im, ax=axs[0], fraction=0.046)
    cb.set_ticks([0, np.pi, 2 * np.pi])
    cb.set_ticklabels(['0', 'π', '2π'])
    axs[1].hist(s4['theta'].ravel()[::7], bins=48, range=(0, 2 * np.pi), density=True,
                color='#7b6fb0')
    axs[1].axhline(1 / (2 * np.pi), color='k', ls='--', lw=1, label='uniform, 1/(2π)')
    axs[1].set_xticks([0, np.pi, 2 * np.pi])
    axs[1].set_xticklabels(['0', 'π', '2π'])
    axs[1].set_xlabel('θ')
    axs[1].set_ylabel('Probability density')
    axs[1].set_title('(b) Histogram of θ')
    axs[1].legend(fontsize=8)
    kxm, kzm = 12.0, 40.0
    A, kxw, kzw = _shifted_window(np.sqrt(s3['F']), s3['kx'], s3['kz'], kxm, kzm)
    ext = [kxw[0], kxw[-1], kzw[0], kzw[-1]]
    im2 = axs[2].imshow(A.T, extent=ext, cmap='viridis', aspect='auto', origin='lower')
    axs[2].set_title('(c) √F')
    fig.colorbar(im2, ax=axs[2], fraction=0.046)
    spec = np.sqrt(s3['F']) * np.exp(1j * s4['theta'])
    Re, _, _ = _shifted_window(spec.real, s3['kx'], s3['kz'], kxm, kzm)
    del spec
    v = np.abs(Re).max()
    im3 = axs[3].imshow(Re.T, extent=ext, cmap='RdBu_r', vmin=-v, vmax=v, aspect='auto',
                        origin='lower')
    axs[3].set_title('(d) Re[√F·exp(iθ)]')
    fig.colorbar(im3, ax=axs[3], fraction=0.046)
    for ax in axs[2:]:
        ax.set_xlabel('k_x [rad/m]')
        ax.set_ylabel('k_z [rad/m]')
    fig.suptitle(f'Step 4: random phase θ (RandomState, seed = {s4["seed"]})', fontsize=13)
    _save(fig, 'steps', 'step4_phase.png')


def step5_inverse_fft(s3, s4):
    print('\n[ステップ 5] 逆 FFT して実部をとる')
    fc = np.fft.ifft2(np.sqrt(s3['F']) * np.exp(1j * s4['theta']))
    f, fi = fc.real, fc.imag
    del fc
    st = _stride(f.shape)
    return dict(f=f, fi_show=fi[::st, ::st].copy(), stride=st)


def draw_field_full(ax, g, arr_show, title, crop=True):
    v = np.percentile(np.abs(arr_show), 99.5)
    im = ax.imshow(arr_show.T, extent=[0, g['Lx'], g['Lz'], 0], cmap='RdBu_r', vmin=-v,
                   vmax=v, aspect='equal')
    if crop:
        ax.add_patch(Rectangle((0, 0), g['Dx'], g['Dz'], fill=False, ec='k', lw=2))
        ax.legend(handles=[Patch(fc='none', ec='k', lw=2, label='domain')], loc='lower right',
                  fontsize=8, framealpha=0.85)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')
    ax.set_title(title)
    return im


def plot_step5(g, s5, seed):
    st = s5['stride']
    fig, axs = plt.subplots(1, 3, figsize=(19, 4.8), gridspec_kw={'width_ratios': [1.4, 1.4, 0.7]},
                            layout='constrained')
    im = draw_field_full(axs[0], g, s5['f'][::st, ::st], '(a) Real part f')
    fig.colorbar(im, ax=axs[0], fraction=0.03)
    im = draw_field_full(axs[1], g, s5['fi_show'], '(b) Imaginary part (discarded)', crop=False)
    fig.colorbar(im, ax=axs[1], fraction=0.03)
    vals = s5['f'][::5, ::5].ravel()
    axs[2].hist(vals, bins=80, density=True, color='#4a7ab5', alpha=0.8, label='f')
    xs = np.linspace(vals.min(), vals.max(), 300)
    sd = vals.std()
    axs[2].plot(xs, np.exp(-xs ** 2 / (2 * sd ** 2)) / (np.sqrt(2 * np.pi) * sd), 'k--', lw=1,
                label='Gaussian (same variance)')
    axs[2].set_xlabel('f [arb.]')
    axs[2].set_ylabel('Probability density')
    axs[2].set_title('(c) Distribution of f')
    axs[2].legend(fontsize=8)
    fig.suptitle(f'Step 5: inverse FFT of √F·exp(iθ) (seed = {seed})', fontsize=13)
    _save(fig, 'steps', 'step5_ifft.png')


def step6_crop_normalize(g, s5):
    print('\n[ステップ 6] 切り出して平均 0・分散 1 に正規化する')
    fc = s5['f'][:g['nx'], :g['nz']]
    R = (fc - fc.mean()) / fc.std()
    print(f'    R: 最小 {R.min():.2f}, 最大 {R.max():.2f}, |R| > 4 のセル {int((np.abs(R) > 4).sum())}')
    return dict(R=R)


def draw_R(ax, R, title):
    st = _stride(R.shape)
    im = ax.imshow(R[::st, ::st].T, extent=[0, R.shape[0] * DX, R.shape[1] * DX, 0],
                   cmap='RdBu_r', vmin=-3, vmax=3, aspect='equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('Depth [m]')
    ax.set_title(title)
    return im


def plot_step6(p, s6):
    R = s6['R']
    fig = plt.figure(figsize=(17, 8.5), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax0 = fig.add_subplot(gs[0, :])
    im = draw_R(ax0, R, '(a) R')
    fig.colorbar(im, ax=ax0, fraction=0.02, label='R')
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.hist(R[::3, ::3].ravel(), bins=80, range=(-4.5, 4.5), density=True, color='#4a7ab5',
             alpha=0.8, label='R')
    xs = np.linspace(-4.5, 4.5, 300)
    ax1.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), 'k--', lw=1, label='N(0, 1)')
    ax1.set_xlabel('R')
    ax1.set_ylabel('Probability density')
    ax1.set_title('(b) Distribution of R')
    ax1.legend(fontsize=8)
    ax2 = fig.add_subplot(gs[1, 1])
    maxlag = int(round(min(4 * max(p['ax'], p['ay']), DOMAIN_Z_M / 2) / DX))
    lag = np.arange(maxlag + 1) * DX
    cx, cz = acf_along(R, 0, maxlag), acf_along(R, 1, maxlag)
    ax2.plot(lag, acf(lag, 0 * lag, p['ax'], p['ay'], p['r']), color='#1f77b4', lw=1, ls='--',
             label='x, target C(Δx, 0)')
    ax2.plot(lag, cx[:lag.size], color='#1f77b4', lw=2, label='x, realized')
    ax2.plot(lag, acf(0 * lag, lag, p['ax'], p['ay'], p['r']), color='#d62728', lw=1, ls='--',
             label='z, target C(0, Δz)')
    ax2.plot(lag, cz[:lag.size], color='#d62728', lw=2, label='z, realized')
    ax2.axhline(np.exp(-1), color='gray', ls=':', lw=0.8, label='1/e')
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set_xlabel('Lag [m]')
    ax2.set_ylabel('Autocorrelation')
    ax2.set_title('(c) Autocorrelation of R')
    ax2.legend(fontsize=8)
    fig.suptitle(f'Step 6: crop and normalize ({_settings(p)})', fontsize=13)
    _save(fig, 'steps', 'step6_normalize.png')


def plot_overview(g, p, s3, s4, s5, s6):
    fig = plt.figure(figsize=(19, 13), layout='constrained')
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9])
    ax = fig.add_subplot(gs[0, 0])
    draw_acf_map(ax, p)
    ax.set_title('Step 1: autocorrelation C')
    ax = fig.add_subplot(gs[0, 1])
    draw_grid_schematic(ax, g, annotate=False)
    ax.set_title('Step 2: generation grid')
    ax = fig.add_subplot(gs[0, 2])
    im = draw_psd_map(ax, g, s3, 'Step 3: power spectrum F')
    fig.colorbar(im, ax=ax, fraction=0.046, label='F [dB re 1 m²]')
    ax = fig.add_subplot(gs[1, 0])
    draw_theta_patch(ax, s4, 'Step 4: random phase θ')
    ax = fig.add_subplot(gs[1, 1:])
    st = s5['stride']
    draw_field_full(ax, g, s5['f'][::st, ::st], 'Step 5: real part of inverse FFT, f')
    ax = fig.add_subplot(gs[2, :])
    im = draw_R(ax, s6['R'], 'Step 6: normalized field R')
    fig.colorbar(im, ax=ax, fraction=0.02, label='R')
    ps = dict(p, seed=s4['seed'])
    fig.suptitle(f'Spectral method ({_settings(ps)}, grid {g["Lx"]:g} × {g["Lz"]:g} m)',
                 fontsize=14)
    _save(fig, 'steps', f'overview_seed{s4["seed"]}.png')


def run_steps_and_seeds():
    print('\n========== (1)(2) ステップ 1-6 とシードの比較 ==========')
    p = dict(BASE)
    step1_define_acf(p)
    plot_step1(p)
    g = step2_make_grid(p)
    plot_step2(g, p)
    s3 = step3_psd(g, p)
    plot_step3(g, p, s3)
    s4 = step4_random_phase(g, p['seed'])
    plot_step4(g, s3, s4)
    s5 = step5_inverse_fft(s3, s4)
    plot_step5(g, s5, p['seed'])
    s6 = step6_crop_normalize(g, s5)
    plot_step6(p, s6)
    if p['seed'] in OVERVIEW_SEEDS:
        plot_overview(g, p, s3, s4, s5, s6)
    R = s6['R'].copy()
    np.save(os.path.join(_outdir('steps'), 'R_demo.npy'), R)
    print(f'    R の sha256: {hashlib.sha256(np.ascontiguousarray(R).tobytes()).hexdigest()}'
          f'（numpy {np.__version__}）')
    print(f'    実現した 1/e 距離: 水平 {realized_1e(R, 0):.3f} m, 鉛直 {realized_1e(R, 1):.3f} m')
    del s4, s5, s6
    R_by_seed = {p['seed']: R}
    for sd in OVERVIEW_SEEDS:
        if sd == p['seed']:
            continue
        s4b = step4_random_phase(g, sd)
        s5b = step5_inverse_fft(s3, s4b)
        s6b = step6_crop_normalize(g, s5b)
        plot_overview(g, p, s3, s4b, s5b, s6b)
        R_by_seed[sd] = s6b['R'].copy()
        del s4b, s5b, s6b
    seeds = list(R_by_seed)
    for sd in seeds[1:]:
        cc = np.corrcoef(R_by_seed[seeds[0]].ravel(), R_by_seed[sd].ravel())[0, 1]
        print(f'    R の相関係数 seed {seeds[0]} vs {sd}: {cc:+.3f}（別の実現なら 0 付近）')
    del s3, R_by_seed
    if CHECK_AGAINST_FUNCTION:
        R2 = field_for(p)
        print(f'    [検算] ステップ版と一括関数のビット一致: {np.array_equal(R, R2)}')


# =============================================================================
# (3) パラメータを 1 つずつ変えた比較
# =============================================================================
def _vary_settings(key):
    """変えないパラメータの設定（例: a_y = 0.25 m, r = 0, seed = 0）"""
    return _settings(BASE, keys=[k for k in ('ax', 'ay', 'r') if k != key])


def _vary_colors(n):
    return plt.cm.viridis(np.linspace(0.05, 0.85, n))


def _db(a):
    return 10 * np.log10(np.maximum(a, 1e-300))


def _spec_arrays(sp, kind):
    """kind: 'x' = S_x, 'z' = S_z, 'zr' = k_z^2 S_z。(k, 実現値, 理論値) を返す"""
    if kind == 'x':
        return sp['kx1'], sp['Sx1'], sp['Sx1_th']
    k, s, th = sp['kz1'], sp['Sz1'], sp['Sz1_th']
    if kind == 'zr':
        return k, s * k ** 2, th * k ** 2
    return k, s, th


SPEC_LABEL = {
    'x': ('k_x [rad/m]', 'S_x(k_x) [dB re 1 m]', 'Horizontal spectrum S_x(k_x)'),
    'z': ('k_z [rad/m]', 'S_z(k_z) [dB re 1 m]', 'Vertical spectrum S_z(k_z)'),
    'zr': ('k_z [rad/m]', 'k_z² S_z(k_z) [dB re 1 m⁻¹]', 'Vertical reflectivity spectrum k_z² S_z(k_z)'),
}


def draw_spec_1d(ax, key, values, specs, kind, legend=True, title=None):
    """1 次元スペクトルを全ての値について重ねる（実線 = 実現値、破線 = 理論値）"""
    colors = _vary_colors(len(values))
    handles = []
    for v, sp, col in zip(values, specs, colors):
        k, s, th = _spec_arrays(sp, kind)
        ax.semilogx(k, _db(s), color=col, lw=1.3)
        ax.semilogx(k, _db(th), color=col, lw=1, ls='--')
        handles.append(Line2D([], [], color=col, lw=2, label=f'{PNAME[key]} = {_fmt(key, v)}'))
    handles += [Line2D([], [], color='k', lw=1.3, label='realized'),
                Line2D([], [], color='k', lw=1, ls='--', label='theory')]
    if kind != 'x':
        ax.axvspan(*bragg_band(), color='gold', alpha=0.2)
        handles.append(Patch(fc='gold', alpha=0.35, label='Bragg band (2k)'))
    top = max(_db(_spec_arrays(sp, kind)[1]).max() for sp in specs)
    ax.set_ylim(top - SPEC_DB_RANGE, top + 5)
    ax.set_xlim(0.3, np.pi / DX)
    ax.grid(alpha=0.3, which='both')
    xl, yl, tt = SPEC_LABEL[kind]
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(tt if title is None else title)
    if legend:
        ax.legend(handles=handles, fontsize=8, loc='lower left')
    return handles


def plot_vary_fields(key, values, fields):
    n = len(values)
    # 12 m × 3 m の縦横比を保つので、幅は 1 段の高さ × 4 + 余白程度にする
    fig, axs = plt.subplots(n, 1, figsize=(12.0, 3.0 * n + 0.6), squeeze=False,
                            layout='constrained')
    for i, (v, R) in enumerate(zip(values, fields)):
        im = draw_R(axs[i, 0], R, f'{PNAME[key]} = {_fmt(key, v)}')
        if i < n - 1:
            axs[i, 0].set_xlabel('')
    fig.colorbar(im, ax=axs[:, 0].tolist(), shrink=0.6, pad=0.01, label='R')
    fig.suptitle(f'Random field R, varying {PNAME[key]} ({_vary_settings(key)})', fontsize=13)
    _save(fig, key, f'fields_vary_{key}.png')


def plot_vary_fft(key, values, specs):
    n = len(values)
    fig = plt.figure(figsize=(4.4 * n + 1, 12), layout='constrained')
    gs = fig.add_gridspec(2, 2 * n, height_ratios=[1.35, 1])
    bragg = bragg_band()
    vmax = max(_db(sp['S2'].max()) for sp in specs)
    for i, (v, sp) in enumerate(zip(values, specs)):
        ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
        W, kxw, kzw = _shifted_window(sp['S2'], sp['kx2'], sp['kz2'], FFT_KX_MAX, FFT_KZ_MAX,
                                      shifted=True)
        im = ax.imshow(_db(W).T, extent=[kxw[0], kxw[-1], kzw[0], kzw[-1]], origin='lower',
                       aspect='auto', cmap='magma', vmin=vmax - FFT_DB_RANGE, vmax=vmax)
        _bragg_lines(ax, bragg, lw=0.7)
        ax.set_title(f'{PNAME[key]} = {_fmt(key, v)}')
        ax.set_xlabel('k_x [rad/m]')
        if i == 0:
            ax.set_ylabel('k_z [rad/m]')
            ax.legend(handles=[_bragg_handle()], loc='upper right', fontsize=8, framealpha=0.7)
    fig.colorbar(im, ax=fig.axes[:n], shrink=0.8, pad=0.01, label='S(k_x, k_z) [dB re 1 m²]')
    draw_spec_1d(fig.add_subplot(gs[1, :n]), key, values, specs, 'x')
    draw_spec_1d(fig.add_subplot(gs[1, n:]), key, values, specs, 'z')
    fig.suptitle(f'Power spectrum, varying {PNAME[key]} ({_vary_settings(key)}, '
                 f'window: {WINDOW_NAME[FFT_WINDOW]})', fontsize=13)
    _save(fig, key, f'fft_vary_{key}.png')


def plot_vary_reflectivity(key, values, specs):
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    draw_spec_1d(ax, key, values, specs, 'zr', title='')
    fig.suptitle(f'Vertical reflectivity spectrum k_z² S_z(k_z), varying {PNAME[key]} '
                 f'({_vary_settings(key)}, window: {WINDOW_NAME[FFT_WINDOW]})', fontsize=12)
    _save(fig, key, f'reflectivity_vary_{key}.png')


def plot_vary_summary(key, values, fields, specs):
    """左にランダム場を縦に並べ、右に水平・鉛直方向の 1 次元スペクトルを重ねて描く"""
    n = len(values)
    colors = _vary_colors(n)
    fig = plt.figure(figsize=(17, 2.55 * n + 1.4), layout='constrained')
    sf_l, sf_r = fig.subfigures(1, 2, width_ratios=[1.75, 1], wspace=0.01)
    axs = sf_l.subplots(n, 1, squeeze=False)[:, 0]
    for i, (v, R, col) in enumerate(zip(values, fields, colors)):
        im = draw_R(axs[i], R, f'{PNAME[key]} = {_fmt(key, v)}')
        for sp_name in ('left', 'right', 'top', 'bottom'):
            axs[i].spines[sp_name].set_edgecolor(col)
            axs[i].spines[sp_name].set_linewidth(3)
        if i < n - 1:
            axs[i].set_xlabel('')
    sf_l.colorbar(im, ax=axs.tolist(), shrink=0.5, pad=0.01, label='R')
    ax_x, ax_z = sf_r.subplots(2, 1)
    draw_spec_1d(ax_x, key, values, specs, 'x')
    draw_spec_1d(ax_z, key, values, specs, 'z')
    fig.suptitle(f'Random field and 1D spectra, varying {PNAME[key]} ({_vary_settings(key)}, '
                 f'window: {WINDOW_NAME[FFT_WINDOW]})', fontsize=13)
    _save(fig, key, f'summary_vary_{key}.png')


def run_vary(key):
    print(f'\n========== (3) {PNAME_JA[key]} を変える：{VARY[key]} ==========')
    values = list(VARY[key])
    fields, stats, specs = [], [], []
    for v in values:
        p = dict(BASE, **{key: v})
        t0 = time.time()
        R = field_for(p)
        lx, lz = realized_1e(R, 0), realized_1e(R, 1)
        S2, kx2, kz2, dkx, dkz = periodogram_2d(R)
        kx1, Sx1, kz1, Sz1 = spectra_1d(S2, kx2, kz2, dkx, dkz)
        specs.append(dict(S2=S2, kx2=kx2, kz2=kz2, kx1=kx1, Sx1=Sx1, kz1=kz1, Sz1=Sz1,
                          Sx1_th=theory_1d(p, kx1, 'x'), Sz1_th=theory_1d(p, kz1, 'z')))
        fields.append(R)
        stats.append(dict(lx=lx, lz=lz))
        print(f'    {key} = {v:g}: 実現 1/e 水平 {lx:.3f} m, 鉛直 {lz:.3f} m, '
              f'R {R.min():.2f}..{R.max():.2f}（{time.time() - t0:.1f} s）')
    plot_vary_fields(key, values, fields)
    plot_vary_fft(key, values, specs)
    plot_vary_reflectivity(key, values, specs)
    plot_vary_summary(key, values, fields, specs)
    np.savez(os.path.join(_outdir(key), f'spectra_vary_{key}.npz'),
             param=key, values=np.array(values, dtype=float),
             base=np.array([BASE['ax'], BASE['ay'], BASE['r'], BASE['seed']], dtype=float),
             window=FFT_WINDOW,
             kx=specs[0]['kx1'], kz=specs[0]['kz1'],
             Sx=np.array([s['Sx1'] for s in specs]), Sz=np.array([s['Sz1'] for s in specs]),
             Sx_theory=np.array([s['Sx1_th'] for s in specs]),
             Sz_theory=np.array([s['Sz1_th'] for s in specs]),
             realized_1e=np.array([[s['lx'], s['lz']] for s in stats]))
    print(f'    -> {os.path.join(_outdir(key), f"spectra_vary_{key}.npz")}')


# =============================================================================
def main():
    plt.rcParams['axes.unicode_minus'] = True
    _prepare_dirs()
    t_all = time.time()
    check_all_grids()
    if RUN_STEPS_AND_SEEDS:
        run_steps_and_seeds()
    for key in RUN_VARY:
        run_vary(key)
    print(f'\n完了（{time.time() - t_all:.0f} s）。出力先: {BASE_DIR}/')


if __name__ == '__main__':
    main()