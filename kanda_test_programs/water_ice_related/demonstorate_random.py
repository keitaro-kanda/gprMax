#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
level6_spectral_steps.py
========================

Level 6 のランダム場（Jiang et al. 2013 のスペクトル法）を、設計ドキュメントの
ステップ 1〜6 の順に 1 段ずつ計算し、各段の生成物を図にする学習用スクリプト。

    ステップ 1  自己相関関数 C(x, z) を決める            （Jiang 式 9）
    ステップ 2  生成格子を用意し、周期性の条件を確かめる   （追加）
    ステップ 3  C を格子に標本化して FFT → パワースペクトル F（式 10）
    ステップ 4  一様乱数の位相 θ を引く（乱数はここだけ）   （式 11 の θ）
                np.random.RandomState(SEED) を使う。NEP 19 で乱数列が凍結されているので、
                numpy の版が変わっても同じ整数シードから同じ θ が出る
    ステップ 5  √F·e^{iθ} を逆 FFT して実部をとる          （式 11, 12）
    ステップ 6  地下領域を切り出して平均 0・分散 1 に正規化 （式 13–15）

出力（OUT_DIR 以下）
    step1_acf.png        自己相関の形（2 次元マップ、断面、裾の対数表示）
    step2_grid.png       生成格子と切り出し領域、周期性の条件
    step3_psd.png        格子上の C（折り返し配置）、F の 2 次元マップ、鉛直断面
    step4_phase.png      位相 θ、そのヒストグラム、振幅 √F とランダムスペクトル
    step5_ifft.png       逆 FFT の実部（使う）と虚部（捨てる）、分布
    step6_normalize.png  正規化後の R、分布、実現した自己相関と目標 C の比較
    overview_seed<N>.png 各ステップの代表的な生成物を 1 枚に並べた流れ図。
                         OVERVIEW_SEEDS の各シードについて 1 枚ずつ（既定は 2 種類）。
                         ステップ 1–3 はシードに依存しないので共通、4–6 だけが変わる

最後に、段階的に計算した R がドキュメントの一括関数 make_random_field_spectral()
の結果とビット単位で一致することを確かめる（CHECK_AGAINST_FUNCTION）。
また、別の計算機と照合するための指紋として、θ の先頭 3 値と R の sha256 を表示する。
θ の先頭 3 値は numpy の版や計算機によらず一致するはず。R の sha256 は FFT の丸めの
違いで変わりうる（その場合も値の差は 1e-12 程度で、物理的には同じ場）。

必要なもの: numpy, matplotlib（scipy は不要）
メモリ: 既定設定（dx = 2.5 mm、生成格子 14 × 7 m = 5600 × 2800）でピーク 1.5 GB 程度。
        軽く試すだけなら DX を 0.005 などに上げる（ただし場は別の実現になる）。
"""

import hashlib
import os
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch

# =============================================================================
# 設定（ドキュメントの LEVEL6_RANDOM に対応）
# =============================================================================
DX = 0.0025                 # 格子間隔 [m]（gprMax の dx と同じ）
DOMAIN_X_M = 3.0            # 切り出す地下領域の水平幅 [m]（PML を含む計算領域の全幅）
DOMAIN_Z_M = 3.0            # 切り出す地下領域の深さ [m]（地表 y = 3.0 m から領域底まで）

A_M = 0.70                  # 水平相関長 a [m]（自己相関が 1/e になる距離）
B_M = 0.28                  # 鉛直相関長 b [m]
R_ROUGH = 0.0               # 粗さ係数 r（0 = ガウス型、1 = 指数型）
SEED = 20261009             # 乱数シード（ステップ 4 だけで使う）。0 <= SEED < 2**32 の整数
OVERVIEW_SEEDS = (20261009, 20261010)  # overview を描くシード（ステップ別の図は SEED だけ）
GEN_SIZE_M = (14.0, 7.0)    # 生成格子 (Lx, Lz) [m]。相関長を変えても固定する
WRAP_EPS = 1e-3             # 周期性の条件に使う自己相関のしきい値

R_COMPARE = (0.0, 0.5, 1.0)  # ステップ 1・3 で形を比べる r の値
N_REFRACT = 1.79             # ブラッグ帯の計算に使う屈折率（氷層深さ）
BAND_GHZ = (0.5, 2.0)        # LUPEX 帯域 [GHz]

OUT_DIR = '/Volumes/SSD_Kanda_BUFFALO/test_programs_output/demonstrate_random'
SHOW = False                 # True にすると各図を画面にも表示する
CHECK_AGAINST_FUNCTION = True
DPI = 130

C0 = 2.998e8                 # 光速 [m/s]


# =============================================================================
# 表示の準備
# =============================================================================
def _setup_japanese_font():
    """インストール済みの日本語フォントを探して使う（見つからなければ既定のまま）。"""
    candidates = ['Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'Yu Gothic', 'Meiryo',
                  'Noto Sans CJK JP', 'Noto Sans JP', 'IPAexGothic', 'IPAGothic',
                  'TakaoGothic']
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in have:
            plt.rcParams['font.family'] = name
            break
    plt.rcParams['axes.unicode_minus'] = False


def _stride(shape, max_px=900):
    """表示用の間引き幅。配列が大きすぎると imshow が重いので、長辺を max_px 程度に落とす。"""
    return max(1, int(np.ceil(max(shape) / max_px)))


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI)
    print(f'    -> {path}')
    if SHOW:
        plt.show()
    plt.close(fig)


# =============================================================================
# ステップ 1：自己相関関数を決める
# =============================================================================
def acf(x, z, a, b, r):
    """Jiang 式 (9) の楕円体自己相関  C = exp[-((x/a)^2 + (z/b)^2)^(1/(1+r))]"""
    return np.exp(-((x / a) ** 2 + (z / b) ** 2) ** (1.0 / (1.0 + r)))


def step1_define_acf():
    """C(x, z) は解析式なので、ここでは「どんな形か」を見るだけで配列はまだ作らない。
    実際に格子へ標本化するのはステップ 3（格子はステップ 2 で決まる）。"""
    print('\n[ステップ 1] 自己相関関数 C(x, z) を決める')
    print(f'    a = {A_M} m（水平）, b = {B_M} m（鉛直）, r = {R_ROUGH}')
    print('    r = 0: ガウス型（原点で滑らか）/ r = 1: 指数型（原点で尖る）')
    return {'a': A_M, 'b': B_M, 'r': R_ROUGH}


def draw_acf_map(ax, s1):
    """C(x, z) の 2 次元マップと 1/e の等値線（楕円）。"""
    x = np.linspace(-2.0, 2.0, 401)
    z = np.linspace(-1.0, 1.0, 201)
    X, Z = np.meshgrid(x, z, indexing='xy')
    C = acf(X, Z, s1['a'], s1['b'], s1['r'])
    im = ax.imshow(C, extent=[x[0], x[-1], z[-1], z[0]], cmap='viridis', vmin=0, vmax=1,
                   aspect='equal')
    ax.contour(X, Z, C, levels=[np.exp(-1)], colors='w', linewidths=1.2)
    ax.annotate('', xy=(s1['a'], 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='w'))
    ax.annotate('', xy=(0, s1['b']), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='w'))
    ax.text(s1['a'] / 2, -0.06, 'a', color='w', ha='center', va='bottom')
    ax.text(0.06, s1['b'] / 2, 'b', color='w', ha='left', va='center')
    ax.set_xlabel('水平のずれ Δx [m]')
    ax.set_ylabel('鉛直のずれ Δz [m]')
    ax.set_title(f'① C(Δx, Δz)（r = {s1["r"]}）\n白線 = 1/e の楕円')
    return im


def plot_step1(s1):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    im = draw_acf_map(axs[0], s1)
    fig.colorbar(im, ax=axs[0], fraction=0.046, label='C')

    d = np.linspace(0, 2.0, 801)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(R_COMPARE)))
    for ax, logy in [(axs[1], False), (axs[2], True)]:
        for col, r in zip(colors, R_COMPARE):
            lw = 2.4 if np.isclose(r, s1['r']) else 1.2
            ax.plot(d, acf(d, 0 * d, s1['a'], s1['b'], r), color=col, lw=lw,
                    label=f'水平 C(Δx, 0), r = {r}')
            ax.plot(d, acf(0 * d, d, s1['a'], s1['b'], r), color=col, lw=lw, ls='--',
                    label=f'鉛直 C(0, Δz), r = {r}')
        ax.axhline(np.exp(-1), color='gray', lw=0.8, ls=':')
        ax.set_xlabel('距離 [m]')
        ax.set_ylabel('C')
        if logy:
            ax.set_yscale('log')
            ax.set_ylim(1e-8, 1.5)
            ax.axhline(WRAP_EPS, color='crimson', lw=0.8, ls=':')
            ax.text(2.0, WRAP_EPS * 1.6, f'WRAP_EPS = {WRAP_EPS:g}', color='crimson',
                    ha='right', fontsize=8)
            ax.set_title('裾の対数表示：r が大きいほど裾が長い\n（ステップ 2 の余白 p を決める）')
        else:
            ax.set_ylim(0, 1.05)
            ax.text(1.95, np.exp(-1) + 0.02, '1/e', color='gray', ha='right')
            ax.set_title('断面：どの r でも a, b で 1/e になる\nr > 0 は原点で尖る（短波長が多い）')
            ax.legend(fontsize=7.5, ncol=1)
    fig.suptitle('ステップ 1：自己相関関数を決める（Jiang 式 9）', fontsize=13)
    fig.tight_layout()
    _save(fig, 'step1_acf.png')


# =============================================================================
# ステップ 2：生成格子を用意する
# =============================================================================
def wrap_length(corr_m, r, eps=WRAP_EPS):
    """自己相関 C が eps まで下がる距離 p [m]。"""
    return corr_m * np.log(1.0 / eps) ** ((1.0 + r) / 2.0)


def step2_make_grid(s1):
    """FFT は格子を周期的とみなす。端どうしの偽の相関を防ぐため、地下領域より大きい
    生成格子を用意し、次の 2 条件を確かめる。
        L - D >= p : 切り出した領域の外側に、相関が消えるだけの余白がある
        L >= 2p    : 周期格子に置ける自己相関（±L/2）の範囲で C が減衰しきる
    """
    print('\n[ステップ 2] 生成格子を用意する')
    nx = int(round(DOMAIN_X_M / DX))
    nz = int(round(DOMAIN_Z_M / DX))
    Gx = int(round(GEN_SIZE_M[0] / DX))
    Gz = int(round(GEN_SIZE_M[1] / DX))
    Lx, Lz = Gx * DX, Gz * DX
    Dx, Dz = nx * DX, nz * DX
    px = wrap_length(s1['a'], s1['r'])
    pz = wrap_length(s1['b'], s1['r'])
    need_x = max(Dx + px, 2 * px)
    need_z = max(Dz + pz, 2 * pz)
    print(f'    地下領域 : {nx} x {nz} セル（{Dx:.2f} x {Dz:.2f} m）')
    print(f'    生成格子 : {Gx} x {Gz} セル（{Lx:.2f} x {Lz:.2f} m）= {Gx * Gz / 1e6:.1f} M セル')
    print(f'    余白 p   : x {px:.2f} m, z {pz:.2f} m（C = {WRAP_EPS:g} になる距離）')
    print(f'    条件     : Lx {Lx:.2f} >= {need_x:.2f} m, Lz {Lz:.2f} >= {need_z:.2f} m')
    if Lx < need_x or Lz < need_z:
        raise ValueError('生成格子が小さすぎます。GEN_SIZE_M を大きくしてください。')
    print('    -> OK')
    return dict(nx=nx, nz=nz, Gx=Gx, Gz=Gz, Lx=Lx, Lz=Lz, Dx=Dx, Dz=Dz, px=px, pz=pz)


def draw_grid_schematic(ax, g):
    """生成格子の周期的な並び（2 x 2 枚）と、切り出す地下領域。"""
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
    ax.text(Lx * 0.6, Lz * 0.42, '生成格子\n（実際に計算する範囲）', ha='center', va='center',
            fontsize=9)
    ax.text(Lx * 1.5, Lz * 0.55, 'FFT が暗黙に仮定する\n周期的なコピー', ha='center',
            va='center', fontsize=9, color='gray')
    ax.text(Dx / 2, Dz / 2, '地下\n領域', ha='center', va='center', color='w', fontsize=8)
    # 右隣のコピーまでの距離 L - D
    yy = Dz / 2
    ax.add_patch(FancyArrowPatch((Dx, yy), (Lx, yy), arrowstyle='<->', mutation_scale=10,
                                 color='#1f4e9c'))
    ax.text((Dx + Lx) / 2, yy - 0.15, f'Lx − D = {Lx - Dx:.1f} m ≥ p_x = {g["px"]:.2f} m',
            ha='center', va='bottom', fontsize=8.5, color='#1f4e9c')
    xx = Dx / 2
    ax.add_patch(FancyArrowPatch((xx, Dz), (xx, Lz), arrowstyle='<->', mutation_scale=10,
                                 color='#1f4e9c'))
    ax.text(xx + 0.15, (Dz + Lz) / 2, f'Lz − D = {Lz - Dz:.1f} m\n≥ p_z = {g["pz"]:.2f} m',
            ha='left', va='center', fontsize=8.5, color='#1f4e9c')
    ax.set_xlim(-0.3, 2 * Lx + 0.3)
    ax.set_ylim(2 * Lz + 0.3, -0.3)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')
    ax.set_title('② 生成格子と切り出し領域\n（隣のコピーまで相関が届かないこと）')


def _draw_wrap_profile(ax, g, s1, axis):
    """格子上の C を、配列の並び順（0 → L）で描く。L の手前で値が戻るのが周期性。"""
    if axis == 'x':
        G, L, D, p, corr = g['Gx'], g['Lx'], g['Dx'], g['px'], s1['a']
    else:
        G, L, D, p, corr = g['Gz'], g['Lz'], g['Dz'], g['pz'], s1['b']
    pos = np.arange(G) * DX                       # 配列上の位置
    dist = np.fft.fftfreq(G, d=1.0 / G) * DX      # 折り返した距離 0, dx, ..., -dx
    c = acf(dist, 0 * dist, corr, 1.0, s1['r'])   # 1 次元断面（もう一方の相関長は無関係）
    ax.semilogy(pos, np.maximum(c, 1e-300), color='k', lw=1.2)
    ax.axvspan(0, D, color='#d62728', alpha=0.12, label=f'領域内の距離 0–D（{D:.1f} m）')
    ax.axvline(L / 2, color='gray', ls=':', lw=1)
    ax.axvline(L - D, color='#1f4e9c', ls='--', lw=1)
    ax.axhline(WRAP_EPS, color='crimson', ls=':', lw=0.8)
    cLD = acf(L - D, 0.0, corr, 1.0, s1['r'])
    cL2 = acf(L / 2, 0.0, corr, 1.0, s1['r'])
    ax.text(L - D, 1e-2, f' 折り返し距離 L−D\n C = {cLD:.1e}', color='#1f4e9c', fontsize=8,
            ha='right')
    ax.text(L / 2, 3e-6, f' L/2\n C = {cL2:.1e}', color='gray', fontsize=8)
    ax.set_ylim(1e-8, 2)
    ax.set_xlim(0, L)
    ax.set_xlabel(f'配列上の位置 {axis} [m]（0 → L）')
    ax.set_ylabel('C（格子上の値）')
    ax.set_title(f'{axis} 方向：C は 0 と L の両端で山になる\n'
                 f'（L = {L:.1f} m, p = {p:.2f} m）')
    ax.legend(fontsize=8, loc='upper center')


def plot_step2(g, s1):
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.0),
                            gridspec_kw={'width_ratios': [1.5, 1, 1]})
    draw_grid_schematic(axs[0], g)
    _draw_wrap_profile(axs[1], g, s1, 'x')
    _draw_wrap_profile(axs[2], g, s1, 'z')
    fig.suptitle('ステップ 2：生成格子を用意する（FFT の周期性への対処）', fontsize=13)
    fig.tight_layout()
    _save(fig, 'step2_grid.png')


# =============================================================================
# ステップ 3：パワースペクトルを求める
# =============================================================================
def sample_acf_on_grid(g, a, b, r):
    """C を「折り返した距離」で標本化する。配列の先頭が距離 0、後半が負の距離。
    こうすると FFT の結果が実数で、F(k) = F(-k) の対称性を持つ。"""
    x = np.fft.fftfreq(g['Gx'], d=1.0 / g['Gx']) * DX
    z = np.fft.fftfreq(g['Gz'], d=1.0 / g['Gz']) * DX
    X, Z = np.meshgrid(x, z, indexing='ij')       # 配列の向きは [x, z]
    return acf(X, Z, a, b, r)


def wavenumbers(g):
    kx = 2 * np.pi * np.fft.fftfreq(g['Gx'], d=DX)   # [rad/m]
    kz = 2 * np.pi * np.fft.fftfreq(g['Gz'], d=DX)
    dk = (kx[1] - kx[0]) * (kz[1] - kz[0])          # 波数格子 1 マスの面積
    return kx, kz, dk


def to_psd_density(F, g, dk):
    """FFT の値 F を、分散 1 の場のパワースペクトル密度 S [m^2] に換算する。
    sum(F) = Gx*Gz*C(0,0) = Gx*Gz なので、S = F / (Gx*Gz) / dk で ∫∫S dk = 1。"""
    return F / (g['Gx'] * g['Gz']) / dk


def step3_psd(g, s1):
    """C を格子に置いて 2 次元 FFT し、パワースペクトル F を得る（Jiang 式 10）。"""
    print('\n[ステップ 3] パワースペクトル F を求める')
    t0 = time.time()
    C = sample_acf_on_grid(g, s1['a'], s1['b'], s1['r'])
    Fc = np.fft.fft2(C)
    imag_rel = np.abs(Fc.imag).max() / np.abs(Fc.real).max()  # 対称に置いたので虚部はほぼ 0
    F = Fc.real.copy()
    del Fc
    neg_frac = -F[F < 0].sum() / F.sum()
    F[F < 0] = 0.0                       # sqrt をとるので負の値（丸め誤差）は 0 にする
    print(f'    FFT にかかった時間 {time.time() - t0:.1f} s')
    print(f'    負の値の割合（0 にした量）: {neg_frac:.2e}（1e-6 を超えたら格子が小さい）')
    print(f'    FFT の虚部 / 実部の最大比 {imag_rel:.1e}（折り返し配置なので F は実数）')

    kx, kz, dk = wavenumbers(g)
    st = _stride(C.shape)
    out = dict(F=F, neg_frac=neg_frac, kx=kx, kz=kz, dk=dk,
               C_show=C[::st, ::st].copy(), C_stride=st)
    del C

    # 鉛直断面 S(kx = 0, kz) を r ごとに比べる。
    # kx = 0 の断面は「C を x 方向に足して z 方向に 1 次元 FFT」したものに等しいので、
    # 2 次元 FFT を r ごとにやり直さなくてよい。
    sec = {}
    x = np.fft.fftfreq(g['Gx'], d=1.0 / g['Gx']) * DX
    z = np.fft.fftfreq(g['Gz'], d=1.0 / g['Gz']) * DX
    for r in sorted(set(R_COMPARE) | {s1['r']}):
        csum = np.zeros(g['Gz'])
        for i0 in range(0, g['Gx'], 512):             # メモリ節約のため x を分割して足す
            X, Z = np.meshgrid(x[i0:i0 + 512], z, indexing='ij')
            csum += acf(X, Z, s1['a'], s1['b'], r).sum(axis=0)
        sec[r] = to_psd_density(np.fft.fft(csum).real, g, dk)
    out['section'] = sec

    k_lo = 2 * np.pi * BAND_GHZ[0] * 1e9 * N_REFRACT / C0
    k_hi = 2 * np.pi * BAND_GHZ[1] * 1e9 * N_REFRACT / C0
    out['bragg'] = (2 * k_lo, 2 * k_hi)
    i = np.argmin(np.abs(kz - 2 * 2 * np.pi * 1.25e9 * N_REFRACT / C0))
    for r, s in sec.items():
        val = 10 * np.log10(max(abs(s[i]), 1e-300))
        note = '（倍精度の丸め誤差の床。解析値はさらに小さい）' if val < -150 else ''
        print(f'    r = {r:<4}: S(0, 2k) at 1.25 GHz = {val:7.1f} dB re 1 m^2{note}')
    return out


def draw_acf_on_grid(ax, g, s3):
    st = s3['C_stride']
    im = ax.imshow(s3['C_show'].T, extent=[0, g['Lx'], g['Lz'], 0], cmap='viridis',
                   vmin=0, vmax=1, aspect='equal')
    ax.set_xlabel('配列上の位置 x [m]')
    ax.set_ylabel('配列上の位置 z [m]')
    ax.set_title('③-a 格子に置いた C（折り返し配置）\n山が 4 隅に分かれて見える')
    return im


def _shifted_window(arr2d, kx, kz, kx_max, kz_max):
    """fftshift して |kx| <= kx_max, |kz| <= kz_max の範囲を切り出す（表示用）。"""
    A = np.fft.fftshift(arr2d)
    kxs, kzs = np.fft.fftshift(kx), np.fft.fftshift(kz)
    ix = np.where(np.abs(kxs) <= kx_max)[0]
    iz = np.where(np.abs(kzs) <= kz_max)[0]
    return A[ix[0]:ix[-1] + 1, iz[0]:iz[-1] + 1], kxs[ix], kzs[iz]


def draw_psd_map(ax, g, s3, kx_max=40.0, kz_max=180.0):
    S = to_psd_density(s3['F'], g, s3['dk'])
    W, kxw, kzw = _shifted_window(S, s3['kx'], s3['kz'], kx_max, kz_max)
    WdB = 10 * np.log10(np.maximum(W, 1e-300))
    im = ax.imshow(WdB.T, extent=[kxw[0], kxw[-1], kzw[0], kzw[-1]], cmap='magma',
                   vmin=-160, vmax=WdB.max(), aspect='auto', origin='lower')
    for kb in s3['bragg']:
        for sgn in (1, -1):
            ax.axhline(sgn * kb, color='gold', lw=0.8, ls='--')
    ax.text(kx_max * 0.95, s3['bragg'][1] * 0.95, 'ブラッグ帯', color='gold', ha='right',
            va='top', fontsize=8)
    ax.set_xlabel('k_x [rad/m]')
    ax.set_ylabel('k_z [rad/m]')
    ax.set_title('③-b パワースペクトル F（dB re 1 m²）\n楕円の向きが C と 90° 入れ替わる')
    return im


def draw_psd_section(ax, g, s3, s1):
    kz = s3['kz']
    m = kz > 0
    o = np.argsort(kz[m])
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(s3['section'])))
    for col, (r, s) in zip(colors, sorted(s3['section'].items())):
        lw = 2.4 if np.isclose(r, s1['r']) else 1.2
        ax.semilogx(kz[m][o], 10 * np.log10(np.maximum(s[m][o], 1e-300)), color=col, lw=lw,
                    label=f'r = {r}')
    ax.axvspan(*s3['bragg'], color='gold', alpha=0.25,
               label=f'ブラッグ帯 2k（{BAND_GHZ[0]}–{BAND_GHZ[1]} GHz）')
    ax.set_xlim(1, np.pi / DX)
    ax.set_ylim(-160, 0)
    ax.set_xlabel('鉛直波数 k_z [rad/m]')
    ax.set_ylabel('S(0, k_z) [dB re 1 m²]')
    ax.set_title('③-c 鉛直断面：r = 0 はブラッグ帯の手前で急落\n（後方散乱の強さを決める）')
    ax.legend(fontsize=8, loc='lower left')


def plot_step3(g, s1, s3):
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.0),
                            gridspec_kw={'width_ratios': [1.4, 1, 1]})
    im0 = draw_acf_on_grid(axs[0], g, s3)
    fig.colorbar(im0, ax=axs[0], fraction=0.03, label='C')
    im1 = draw_psd_map(axs[1], g, s3)
    fig.colorbar(im1, ax=axs[1], fraction=0.046, label='dB re 1 m²')
    draw_psd_section(axs[2], g, s3, s1)
    fig.suptitle('ステップ 3：C を格子に置いて FFT → パワースペクトル F（Jiang 式 10）',
                 fontsize=13)
    fig.tight_layout()
    _save(fig, 'step3_psd.png')


# =============================================================================
# ステップ 4：ランダム位相を与える
# =============================================================================
def step4_random_phase(g, s3, seed=SEED):
    """位相 θ を [0, 2π) の一様乱数で引く。乱数を使うのはこのステップだけ。
    振幅は √F に固定するので、統計（どの波数にどれだけパワーがあるか）は変わらず、
    「どこに粒があるか」だけが θ で決まる。

    乱数生成器は np.random.RandomState(SEED)（旧来の Mersenne Twister 系統）。
      - NEP 19 で乱数列が凍結されており、numpy の版が変わっても同じ整数から同じ θ が出る
        （新しい Generator の uniform にはこの保証がない）
      - np.random.seed() で全体の状態を設定する代わりに、手元のインスタンスで持つ。
        途中で他のコードが np.random を呼んでも θ がずれない
    """
    print('\n[ステップ 4] 一様乱数の位相 θ を引く')
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(g['Gx'], g['Gz']))
    print(f'    RandomState(seed = {seed}), 配列の形 = {theta.shape}（生成格子と同じ形）')
    print(f'    θ の先頭 3 値（指紋。版・計算機によらず一致するはず）: '
          f'{theta[0, 0]:.15f}, {theta[0, 1]:.15f}, {theta[0, 2]:.15f}')
    print(f'    θ の平均 {theta.mean():.4f}（一様なら π = {np.pi:.4f}）')
    return dict(theta=theta)


def draw_theta_patch(ax, s4, n=64):
    im = ax.imshow(s4['theta'][:n, :n].T, cmap='twilight', vmin=0, vmax=2 * np.pi,
                   origin='lower')
    ax.set_xlabel('k_x の番号')
    ax.set_ylabel('k_z の番号')
    ax.set_title(f'④-a 位相 θ（{n}×{n} の一部）\n各波数で独立な一様乱数')
    return im


def plot_step4(g, s3, s4):
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.8))
    im = draw_theta_patch(axs[0], s4)
    cb = fig.colorbar(im, ax=axs[0], fraction=0.046)
    cb.set_ticks([0, np.pi, 2 * np.pi])
    cb.set_ticklabels(['0', 'π', '2π'])

    axs[1].hist(s4['theta'].ravel()[::7], bins=48, range=(0, 2 * np.pi), density=True,
                color='#7b6fb0')
    axs[1].axhline(1 / (2 * np.pi), color='k', ls='--', lw=1, label='一様分布 1/(2π)')
    axs[1].set_xticks([0, np.pi, 2 * np.pi])
    axs[1].set_xticklabels(['0', 'π', '2π'])
    axs[1].set_xlabel('θ')
    axs[1].set_ylabel('確率密度')
    axs[1].set_title('④-b θ のヒストグラム\n偏りがない → 粒の位置に好みがない')
    axs[1].legend(fontsize=8)

    # 振幅 √F（決定的）と、ランダムスペクトル Re[√F e^{iθ}]
    kxm, kzm = 12.0, 40.0
    A, kxw, kzw = _shifted_window(np.sqrt(s3['F']), s3['kx'], s3['kz'], kxm, kzm)
    ext = [kxw[0], kxw[-1], kzw[0], kzw[-1]]
    im2 = axs[2].imshow(A.T, extent=ext, cmap='viridis', aspect='auto', origin='lower')
    axs[2].set_title('④-c 振幅 √F（乱数なし）\nどの波数にどれだけの強さを配るか')
    fig.colorbar(im2, ax=axs[2], fraction=0.046)
    spec = np.sqrt(s3['F']) * np.exp(1j * s4['theta'])
    Re, _, _ = _shifted_window(spec.real, s3['kx'], s3['kz'], kxm, kzm)
    del spec
    v = np.abs(Re).max()
    im3 = axs[3].imshow(Re.T, extent=ext, cmap='RdBu_r', vmin=-v, vmax=v, aspect='auto',
                        origin='lower')
    axs[3].set_title('④-d ランダムスペクトル Re[√F·e^{iθ}]\n包絡は √F、符号と大きさは θ 次第')
    fig.colorbar(im3, ax=axs[3], fraction=0.046)
    for ax in axs[2:]:
        ax.set_xlabel('k_x [rad/m]')
        ax.set_ylabel('k_z [rad/m]')
    fig.suptitle('ステップ 4：一様乱数の位相 θ を与える（Jiang 式 11 の θ）', fontsize=13)
    fig.tight_layout()
    _save(fig, 'step4_phase.png')


# =============================================================================
# ステップ 5：逆 FFT で空間の場に戻す
# =============================================================================
def step5_inverse_fft(s3, s4):
    """√F·e^{iθ} を逆 FFT する（Jiang 式 11, 12）。θ は ±k で対称でないので結果は複素数。
    実部を使い、虚部は捨てる。実部の自己相関は Σ F(k) cos(k·Δ) ∝ C(Δ) なので形は崩れない。"""
    print('\n[ステップ 5] 逆 FFT して実部をとる')
    t0 = time.time()
    fc = np.fft.ifft2(np.sqrt(s3['F']) * np.exp(1j * s4['theta']))
    f = fc.real
    fi = fc.imag
    del fc
    print(f'    逆 FFT にかかった時間 {time.time() - t0:.1f} s')
    print(f'    実部の分散 {f.var():.3e}, 虚部の分散 {fi.var():.3e}（ほぼ同じ大きさ）')
    print(f'    実部と虚部の相関係数 {np.corrcoef(f[::4, ::4].ravel(), fi[::4, ::4].ravel())[0, 1]:+.3f}'
          '（≈ 0：虚部は別の場）')
    st = _stride(f.shape)
    return dict(f=f, fi_show=fi[::st, ::st].copy(), stride=st)


def draw_field_full(ax, g, arr_show, title, crop=True):
    v = np.percentile(np.abs(arr_show), 99.5)
    im = ax.imshow(arr_show.T, extent=[0, g['Lx'], g['Lz'], 0], cmap='RdBu_r', vmin=-v,
                   vmax=v, aspect='equal')
    if crop:
        ax.add_patch(Rectangle((0, 0), g['Dx'], g['Dz'], fill=False, ec='k', lw=2))
        ax.text(g['Dx'] + 0.15, g['Dz'] - 0.1, '切り出す地下領域', fontsize=8.5, va='bottom',
                bbox=dict(fc='white', ec='none', alpha=0.85))
    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')
    ax.set_title(title)
    return im


def plot_step5(g, s5):
    st = s5['stride']
    f_show = s5['f'][::st, ::st]
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.8),
                            gridspec_kw={'width_ratios': [1.3, 1.3, 0.8]})
    im = draw_field_full(axs[0], g, f_show, '⑤-a 実部 f（使う）\n生成格子全体')
    fig.colorbar(im, ax=axs[0], fraction=0.03)
    im = draw_field_full(axs[1], g, s5['fi_show'], '⑤-b 虚部（捨てる）\n同じ統計の別の場',
                         crop=False)
    fig.colorbar(im, ax=axs[1], fraction=0.03)
    vals = s5['f'][::5, ::5].ravel()
    axs[2].hist(vals, bins=80, density=True, color='#4a7ab5', alpha=0.8)
    xs = np.linspace(vals.min(), vals.max(), 300)
    sd = vals.std()
    axs[2].plot(xs, np.exp(-xs ** 2 / (2 * sd ** 2)) / (np.sqrt(2 * np.pi) * sd), 'k--', lw=1,
                label='同じ分散の正規分布')
    axs[2].set_xlabel('f（任意単位）')
    axs[2].set_ylabel('確率密度')
    axs[2].set_title('⑤-c f の分布\n多数の余弦波の和 → ほぼ正規分布')
    axs[2].legend(fontsize=8)
    fig.suptitle('ステップ 5：√F·e^{iθ} を逆 FFT して実部をとる（Jiang 式 11, 12）',
                 fontsize=13)
    fig.tight_layout()
    _save(fig, 'step5_ifft.png')


# =============================================================================
# ステップ 6：切り出して正規化する
# =============================================================================
def step6_crop_normalize(g, s5):
    """地下領域を切り出し、その範囲の平均と標準偏差で正規化する（Jiang 式 13–15）。
    Jiang の φ = (υ/d)(f − μ) の υ（= s_ρ）は、媒質に付与するステップ 7 で掛ける。"""
    print('\n[ステップ 6] 切り出して平均 0・分散 1 に正規化する')
    fc = s5['f'][:g['nx'], :g['nz']]
    mu, d = fc.mean(), fc.std()
    R = (fc - mu) / d
    print(f'    切り出し前: μ = {mu:.3e}, d = {d:.3e}')
    print(f'    切り出し後: 平均 {R.mean():.1e}, 標準偏差 {R.std():.4f}, '
          f'最小 {R.min():.2f}, 最大 {R.max():.2f}, |R| > 4 のセル {int((np.abs(R) > 4).sum())}')
    return dict(R=R)


def acf_along(R, axis, maxlag):
    """R の 1 次元自己相関（もう一方の軸で平均）。ゼロ詰め FFT で計算し、重なり数で割る。"""
    N = R.shape[axis]
    X = np.fft.rfft(R - R.mean(), n=2 * N, axis=axis)
    p = np.fft.irfft(np.abs(X) ** 2, n=2 * N, axis=axis)
    p = p.take(np.arange(maxlag + 1), axis=axis).mean(axis=1 - axis)
    p /= (N - np.arange(maxlag + 1))
    return p / p[0]


def draw_R(ax, g, R, title):
    st = _stride(R.shape)
    im = ax.imshow(R[::st, ::st].T, extent=[0, g['Dx'], g['Dz'], 0], cmap='RdBu_r',
                   vmin=-3, vmax=3, aspect='equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('深さ z [m]')
    ax.set_title(title)
    return im


def plot_step6(g, s1, s6):
    R = s6['R']
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.0))
    im = draw_R(axs[0], g, R, '⑥-a 正規化した R（地下領域）\nこれが Level 6 の R.npy')
    fig.colorbar(im, ax=axs[0], fraction=0.046, label='R（標準偏差単位）')

    axs[1].hist(R[::3, ::3].ravel(), bins=80, range=(-4.5, 4.5), density=True,
                color='#4a7ab5', alpha=0.8, label='R')
    xs = np.linspace(-4.5, 4.5, 300)
    axs[1].plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), 'k--', lw=1, label='N(0, 1)')
    axs[1].set_xlabel('R')
    axs[1].set_ylabel('確率密度')
    axs[1].set_title('⑥-b R の分布\n平均 0・分散 1（模様が少ないので凸凹）。σ での打ち切りはしない')
    axs[1].legend(fontsize=8)

    maxlag = int(round(1.5 / DX))
    lag = np.arange(maxlag + 1) * DX
    ax_ = acf_along(R, 0, maxlag)
    az_ = acf_along(R, 1, maxlag)
    lx = lag[np.argmax(ax_ < np.exp(-1))]
    lz = lag[np.argmax(az_ < np.exp(-1))]
    axs[2].plot(lag, acf(lag, 0 * lag, s1['a'], s1['b'], s1['r']), color='#1f77b4', lw=1,
                ls='--', label=f'目標 C(Δx, 0)（a = {s1["a"]} m）')
    axs[2].plot(lag, ax_, color='#1f77b4', lw=2, label=f'実現（水平）1/e = {lx:.3f} m')
    axs[2].plot(lag, acf(0 * lag, lag, s1['a'], s1['b'], s1['r']), color='#d62728', lw=1,
                ls='--', label=f'目標 C(0, Δz)（b = {s1["b"]} m）')
    axs[2].plot(lag, az_, color='#d62728', lw=2, label=f'実現（鉛直）1/e = {lz:.3f} m')
    axs[2].axhline(np.exp(-1), color='gray', ls=':', lw=0.8)
    axs[2].axhline(0, color='k', lw=0.5)
    axs[2].set_xlabel('ずれ [m]')
    axs[2].set_ylabel('自己相関')
    axs[2].set_title('⑥-c 実現した自己相関と目標 C\n'
                     '領域が相関長の数倍しかないので、1 つの実現では目標からずれる')
    axs[2].legend(fontsize=8)
    print(f'    実現した 1/e 距離: 水平 {lx:.3f} m（目標 {s1["a"]}）, '
          f'鉛直 {lz:.3f} m（目標 {s1["b"]}）')
    fig.suptitle('ステップ 6：地下領域を切り出して正規化する（Jiang 式 13–15）',
                 fontsize=13)
    fig.tight_layout()
    _save(fig, 'step6_normalize.png')


# =============================================================================
# 全体の流れ（各ステップの代表的な生成物を 1 枚に）
# =============================================================================
def plot_overview(g, s1, s3, s4, s5, s6, seed=SEED):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10.5))
    draw_acf_map(axs[0, 0], s1)
    axs[0, 0].set_title('ステップ 1：自己相関 C を決める\n（解析式。乱数なし）')
    draw_grid_schematic(axs[0, 1], g)
    axs[0, 1].set_title('ステップ 2：生成格子を用意する\n（周期的なコピーまで相関が届かない大きさ）')
    im = draw_psd_map(axs[0, 2], g, s3)
    fig.colorbar(im, ax=axs[0, 2], fraction=0.046, label='dB re 1 m²')
    axs[0, 2].set_title('ステップ 3：C を FFT → パワースペクトル F\n（各波数に配る分散）')
    im = draw_theta_patch(axs[1, 0], s4)
    axs[1, 0].set_title('ステップ 4：位相 θ を一様乱数で引く\n（乱数はここだけ。seed で決まる）')
    st = s5['stride']
    draw_field_full(axs[1, 1], g, s5['f'][::st, ::st],
                    'ステップ 5：√F·e^{iθ} を逆 FFT、実部\n（生成格子全体の場 f）')
    im = draw_R(axs[1, 2], g, s6['R'], 'ステップ 6：切り出して正規化 → R\n（平均 0・分散 1）')
    fig.colorbar(im, ax=axs[1, 2], fraction=0.046, label='R')
    fig.suptitle(f'スペクトル法の流れ（a = {s1["a"]} m, b = {s1["b"]} m, r = {s1["r"]}, '
                 f'seed = {seed}, 生成格子 {g["Lx"]:.0f} × {g["Lz"]:.0f} m）', fontsize=14)
    fig.tight_layout()
    _save(fig, f'overview_seed{seed}.png')


# =============================================================================
# 検算：ドキュメントの一括関数と一致するか
# =============================================================================
def make_random_field_spectral(nx, nz, dx, a, b, r, seed, gen_size_m, wrap_eps=1e-3):
    """設計ドキュメントに載せた一括版（ステップ 1–6 を 1 関数にまとめたもの）。"""
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


# =============================================================================
def _prepare_out_dir():
    """出力先を用意する。/Volumes 以下を指定していてボリュームがマウントされていなければ、
    勝手にフォルダを作らずに止める（macOS では /Volumes 直下に作ろうとして権限エラーになる）。"""
    parts = os.path.abspath(OUT_DIR).split(os.sep)
    if len(parts) > 2 and parts[1] == 'Volumes':
        vol = os.sep + os.path.join(parts[1], parts[2])
        if not os.path.isdir(vol):
            raise SystemExit(f'出力先のボリューム {vol} が見つかりません。'
                             'SSD を接続するか、OUT_DIR を変更してください。')
    os.makedirs(OUT_DIR, exist_ok=True)


def main():
    _setup_japanese_font()
    _prepare_out_dir()
    t_all = time.time()

    s1 = step1_define_acf()
    plot_step1(s1)

    g = step2_make_grid(s1)
    plot_step2(g, s1)

    s3 = step3_psd(g, s1)
    plot_step3(g, s1, s3)

    s4 = step4_random_phase(g, s3)
    plot_step4(g, s3, s4)

    s5 = step5_inverse_fft(s3, s4)
    plot_step5(g, s5)

    s6 = step6_crop_normalize(g, s5)
    plot_step6(g, s1, s6)

    if SEED in OVERVIEW_SEEDS:
        plot_overview(g, s1, s3, s4, s5, s6, seed=SEED)

    R = s6['R'].copy()
    np.save(os.path.join(OUT_DIR, 'R_demo.npy'), R)
    print(f'\n    R を保存: {os.path.join(OUT_DIR, "R_demo.npy")}  形 {R.shape}（[x, z]）')
    print(f'    R の sha256: {hashlib.sha256(np.ascontiguousarray(R).tobytes()).hexdigest()}')
    print(f'    numpy {np.__version__}')

    # overview だけ、別のシードでも描く（ステップ 1–3 の結果 s1, g, s3 はそのまま使い回す）
    del s4, s5, s6                            # 先にメモリを空ける
    R_by_seed = {SEED: R}
    for sd in OVERVIEW_SEEDS:
        if sd == SEED:
            continue
        print(f'\n[overview] seed = {sd} でステップ 4–6 をやり直す')
        s4b = step4_random_phase(g, s3, seed=sd)
        s5b = step5_inverse_fft(s3, s4b)
        s6b = step6_crop_normalize(g, s5b)
        plot_overview(g, s1, s3, s4b, s5b, s6b, seed=sd)
        R_by_seed[sd] = s6b['R'].copy()
        del s4b, s5b, s6b
    if len(R_by_seed) > 1:
        seeds = list(R_by_seed)
        for sd in seeds[1:]:
            cc = np.corrcoef(R_by_seed[seeds[0]].ravel(), R_by_seed[sd].ravel())[0, 1]
            print(f'    R の相関係数 seed {seeds[0]} vs {sd}: {cc:+.3f}（別の実現なら 0 付近）')
    del R_by_seed

    if CHECK_AGAINST_FUNCTION:
        print('\n[検算] ドキュメントの一括関数と比べる')
        del s3                                # メモリを空けてから一括版を回す
        R2 = make_random_field_spectral(g['nx'], g['nz'], DX, A_M, B_M, R_ROUGH, SEED,
                                        GEN_SIZE_M, WRAP_EPS)
        same = np.array_equal(R, R2)
        print(f'    ビット単位で一致: {same}' +
              ('' if same else f'（最大差 {np.abs(R - R2).max():.2e}）'))

    print(f'\n完了（{time.time() - t_all:.0f} s）。図は {OUT_DIR}/ にあります。')


if __name__ == '__main__':
    main()