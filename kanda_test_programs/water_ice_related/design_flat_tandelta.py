"""
Flat eps'' design: analytic two-pole Debye parameters
=====================================================
月南極のレゴリス（イルメナイト <1 wt%）は、GPR 帯域内で **eps'' が一定**である。

    eps'' = 一定      ->      alpha ∝ f

これを gprMax で実現したいが、gprMax の材料モデルは

    #material: eps_r sigma mu_r sigma*     ->  eps'' = sigma/(w eps0)  ∝ 1/f

しか与えず、sigma 一定では eps'' が 1/f で落ちてしまう（これは Level 2）。
Debye/Lorentz/Drude しか使えないため、**複数の Debye 極を重ね合わせて
eps'' を帯域内で平坦にする**しかない。

    Debye 1 極の eps''(w) = De * (w tau) / (1 + (w tau)^2)
        -> w = 1/tau にピークを持つ山型

    tau を帯域の上下に振り分けて 2 つ重ねると、山どうしが重なって平坦部ができる

--- 用語について ------------------------------------------------------------
本スクリプトでは「非分散」という語を使わない。eps'' != 0 の媒質は
Kramers-Kronig 則により eps' が必ず対数的に変化するため、厳密な意味で
非分散な損失媒質は存在しない。正しい記述は
    eps'' = 一定（帯域内）、eps' は KK により約 0.4% 変化
である。

--- このスクリプトの位置づけ ------------------------------------------------
実際の Debye パラメータ設定は Level_N.in の中で完結している。本スクリプトは
**その設計ロジックの理論的な説明と図の生成**のために存在する。したがって
.in と同一の式・同一の定数を使うこと（値を 2 か所で持たない）。

    f0      = sqrt(f_lo * f_hi)                  帯域の幾何平均 = 1.0 GHz
    s       = arcsinh(1) = ln(1 + sqrt(2))       最大平坦条件
    tau_1,2 = 1/(2 pi f0 (1+sqrt2)), (1+sqrt2)/(2 pi f0)
    De      = sqrt(2) * eps''_target             （各極）
    eps_inf = eps_r - De                         （f0 で eps' = eps_r）

--- 数値最適化を使わない理由 ------------------------------------------------
旧版は least_squares で等リップル解を求めていた（eps'' 誤差 RMS 0.19%、
解析解 0.89%）。精度は高いが
  * 論文で式を一行で書けない
  * ソルバの初期値・収束条件が結果に影響する
  * .in の中で scipy を使うことになる
ため採用しない。解析解の誤差が振幅に与える影響は深さ 2.75 m で約 -0.06 dB
であり、合否判定幅 ±0.5 dB に対して十分小さい。

Requirements: numpy, matplotlib（scipy は使わない）
"""

import os
import io
import datetime

import numpy as np
import matplotlib.pyplot as plt

# NumPy 2.0 で np.trapz が np.trapezoid に改名された。どちらの版でも動くようにする。
_TRAPZ = getattr(np, 'trapezoid', None) or np.trapz


# ---------------------------------------------------------------------------
# [EDIT HERE] 設計条件   ※ Level_N.in と一致させること
# ---------------------------------------------------------------------------
EPS_R = 3.0                    # 比誘電率実部（帯域の幾何平均で一定にしたい値）

# --- Carrier 経験式 --------------------------------------------------------
# 出典: Carrier, Olhoeft & Mendell (1991), "Physical Properties of the Lunar
#       Surface", in Lunar Sourcebook, Cambridge Univ. Press, pp.475-594.
#       Fig. 9.53 (SOILS = 土壌試料のみの回帰) の図中式:
#           eps'      = 1.871^rho
#           tan_delta = 10^(0.027*(%TiO2 + %FeO) + 0.273*rho - 3.058)
#
# Fig. 9.53 を選ぶ理由:
#   (a) 本研究の対象はレゴリス（土壌）であり、岩石片を含む Fig. 9.52
#       (ALL DATA) や Fig. 9.54 (450 MHz DATA) より母集団が適切。
#   (b) tan_delta の周波数依存を無視する立場をとる以上、周波数で切った
#       サブセット（Fig. 9.54 = 450 MHz）を選ぶのは仮定と矛盾する。
#       選ぶべき軸は周波数ではなく試料種別。
#   (c) 土壌データは rho ~ 1.0-2.1 に分布し、rho = 1.753647 はその中心付近。
#
# 【重要】eps' の式と tan_delta の式は同一図・同一サブセットから取ること。
# 参考（本スクリプトでは使わない）:
#   Fig. 9.52 ALL DATA     : 1.919^rho, 10^(0.038 S + 0.312 rho - 3.260)
#   Fig. 9.54 450 MHz DATA : 1.843^rho, 10^(0.033 S + 0.231 rho - 3.061)
#   Fig. 9.55 APOLLO 15-17 : 1.908^rho, 10^(0.028 S + 0.167 rho - 2.975)
CARRIER_EPS_BASE = 1.871
CARRIER_TAND_A, CARRIER_TAND_B, CARRIER_TAND_C = 0.027, 0.273, 3.058

# 設計対象の組成 [wt%]。月南極域は 5-11 wt% に収束するため 3 点、
# 高Tiバサルト（月の海）の参考値として 20 wt% を併記する。
COMPOSITIONS = {'feo5': 5.0, 'feo7p5': 7.5, 'feo10': 10.0, 'feo20': 20.0}
PRIMARY_KEY = 'feo7p5'         # テキストで詳細を出す組成（図は全組成で出力）

BAND_LO, BAND_HI = 0.5e9, 2.0e9    # 平坦化する帯域（LUPEX GPR）
BAND_F0 = np.sqrt(BAND_LO * BAND_HI)      # 帯域の幾何平均 = 1.0 GHz
                                          # 解析解の対称中心かつ eps_r の基準
BAND_CENTRE_ARITH = 1.25e9     # 算術中心。報告時の参考値としてのみ使う

# gprMax の時間刻み制約
DX = 0.0025                    # [m] グリッドサイズ
C0 = 299792458.0               # [m/s]
DT_GPRMAX = DX / (np.sqrt(2.0) * C0)      # 2D クーラン条件
TAU_MIN_SAFE = 5.0             # tau/dt がこれを下回ったら警告

N_EVAL = 601                   # 評価・作図に使う帯域内の周波数点数
F_PLOT_LO, F_PLOT_HI = 1e8, 1e10          # 図の周波数範囲

OUTPUT_DIR = '/Volumes/SSD_Kanda_BUFFALO/test_programs_output/flat_tandelta_design'
RESULTS_FILENAME = 'flat_tandelta_design.txt'


# ---------------------------------------------------------------------------
# 1. 目標値の決定
# ---------------------------------------------------------------------------

def density_for_eps(eps_r):
    """eps' = 1.871^rho を rho について解いた密度 [g/cm^3]。

    eps_r = 3.0 とすると rho = 1.753647。これは Carrier の密度プロファイル
    rho(z) = 1.92 (z+12.2)/(z+18) [z: cm] の深さ約 49 cm に相当するので、
    「深さ 50 cm 付近のレゴリスを一様に敷き詰めた媒質」と解釈できる。
    """
    return np.log(eps_r) / np.log(CARRIER_EPS_BASE)


def carrier_tandelta(feotio2_wt, rho):
    """Carrier 経験式（Fig. 9.53, SOILS）の tan_delta。

    この式は周波数を説明変数に持たないため、周波数に依らない量として扱う。
    その扱いの根拠は Boivin et al. (2022)：単一装置で P/L/S/X 帯を通して
    測定した純バイトウナイト（イルメナイトなし）で eps'' が完全に一定。
    """
    return 10.0 ** (CARRIER_TAND_A * feotio2_wt
                    + CARRIER_TAND_B * rho - CARRIER_TAND_C)


# ---------------------------------------------------------------------------
# 2. 多極 Debye モデル
# ---------------------------------------------------------------------------

def debye_eps_imag(f, poles):
    """eps''(f) = sum_i De_i (w tau_i) / (1 + (w tau_i)^2)。"""
    w = 2.0 * np.pi * np.asarray(f, dtype=float)
    return sum(de * (w * tau) / (1.0 + (w * tau) ** 2) for de, tau in poles)


def debye_eps_real(f, eps_inf, poles):
    """eps'(f) = eps_inf + sum_i De_i / (1 + (w tau_i)^2)。"""
    w = 2.0 * np.pi * np.asarray(f, dtype=float)
    return eps_inf + sum(de / (1.0 + (w * tau) ** 2) for de, tau in poles)


def attenuation(f, eps_re, tand):
    """減衰係数 alpha [Np/m]（厳密式）。

        alpha = (w/c) sqrt(eps'/2) sqrt( sqrt(1 + tan_delta^2) - 1 )

    低損失極限では alpha -> pi f sqrt(eps') tan_delta / c となり、
    eps'' 一定なら alpha ∝ f になる（帯域 0.5-2.0 GHz で 4.0 倍）。
    これが重心周波数シフト法が機能するための必要条件である
    （df_c/dt = -2 pi tan_delta sigma_f^2 は alpha ∝ f を前提とする）。
    """
    w = 2.0 * np.pi * np.asarray(f, dtype=float)
    return (w / C0) * np.sqrt(eps_re / 2.0) * np.sqrt(np.sqrt(1.0 + tand ** 2) - 1.0)


# ---------------------------------------------------------------------------
# 3. 解析解
# ---------------------------------------------------------------------------
# Level_N.in の debye_flat_eps_imag() と同一の式。値を 2 か所で持たないため、
# .in を変更したらここも必ず合わせること。
# ---------------------------------------------------------------------------
S_FLAT = float(np.arcsinh(1.0))        # = ln(1 + sqrt(2)) = 0.881374
TAU_RATIO = float(np.exp(S_FLAT))      # = 1 + sqrt(2) = 2.414214


def two_pole_analytic(eps_r, eps_imag_target):
    """最大平坦 2 極 Debye の解析解（採用する設計）。

    u = ln(w/w0) と置き、緩和周波数を w0 e^{±s} に対称配置して振幅を
    等しく De とすると、Debye の和が双曲線関数で書ける:

        eps''(u) / De = (1/2) [ sech(u - s) + sech(u + s) ]

    中心 u=0 での 2 階微分をゼロにする（最大平坦条件）と

        sinh s = 1   ->   s = arcsinh(1) = ln(1 + sqrt(2)) = 0.881374

    となり、緩和周波数比は e^{2s} = 3 + 2 sqrt(2) = 5.828427 に決まる。
    中心での値は sech(s) = 1/sqrt(2) なので、目標に合わせるには

        De = sqrt(2) * eps''_target

    さらに、対称中心を帯域の幾何平均 f0 に取ると

        1/(1 + (w0 tau_1)^2) + 1/(1 + (w0 tau_2)^2) = 1     （厳密に 1）

    が成り立つ。実際 w0 tau_1 = sqrt(2) - 1、w0 tau_2 = sqrt(2) + 1 なので
    1/(4 - 2 sqrt2) + 1/(4 + 2 sqrt2) = 1。したがって f0 で eps' = eps_r に
    するための eps_inf も閉形式になる:

        eps_inf = eps_r - De

    数値最適化は不要。
    """
    w0 = 2.0 * np.pi * BAND_F0
    tau = [1.0 / (w0 * TAU_RATIO), TAU_RATIO / w0]
    de = np.sqrt(2.0) * eps_imag_target
    poles = [(de, t) for t in tau]
    eps_inf = eps_r - de
    return eps_inf, poles


def one_pole_analytic(eps_r, eps_imag_target):
    """1 極を帯域中心に置いた場合（比較用。平坦にはならない）。

    ピーク（w tau = 1）で eps'' = De/2 なので De = 2 * eps''_target。
    このとき eps'(f0) = eps_inf + De/2 なので eps_inf = eps_r - eps''_target。
    """
    w0 = 2.0 * np.pi * BAND_F0
    de = 2.0 * eps_imag_target
    poles = [(de, 1.0 / w0)]
    return eps_r - eps_imag_target, poles


def continuum_flatness():
    """連続極限の確認: ln(tau) に一様な密度 D で分布させると eps'' は厳密に一定。

        eps''(w) = int D (w tau)/(1 + (w tau)^2) d(ln tau)
                 = D int (1/2) sech(u) du = D pi/2

    これが「対数等間隔に極を置けば平坦になる」ことの原理であり、
    有限個の極はこの連続分布を離散サンプルしていることになる。
    """
    u = np.linspace(-40, 40, 400001)
    return float(_TRAPZ(0.5 / np.cosh(u), u)), np.pi / 2


def ripple_for_s(s):
    """対称配置の 2 極を u = ±s に置いたときの帯域内リップル p-p [%]。

    最大平坦条件 s = arcsinh(1) が有限帯域でも良い選択であることを示すため、
    s を振ってリップルを評価する（作図に使う）。
    振幅は中心で 1 になるよう規格化するので、リップルは s だけで決まる。
    """
    u_band = np.log(np.array([BAND_LO, BAND_HI]) / BAND_F0)
    u = np.linspace(u_band[0], u_band[1], N_EVAL)
    g = 0.5 * (1.0 / np.cosh(u - s) + 1.0 / np.cosh(u + s))
    g0 = 1.0 / np.cosh(s)                     # u = 0 の値
    r = g / g0
    return 100.0 * (r.max() - r.min())


def evaluate(eps_inf, poles, target_eps_imag):
    """設計した極の性能を評価する。"""
    f = np.geomspace(BAND_LO, BAND_HI, N_EVAL)
    ei = debye_eps_imag(f, poles)
    er = debye_eps_real(f, eps_inf, poles)
    td = ei / er
    alpha = attenuation(f, er, td)
    dev = ei / target_eps_imag - 1.0
    return dict(
        rms_eps_imag=100 * np.sqrt(np.mean(dev ** 2)),
        ptp_eps_imag=100 * (dev.max() - dev.min()),
        edge_eps_imag=100 * dev[0],
        ptp_tand=100 * (td.max() - td.min()) / td.mean(),
        ptp_eps_real=100 * (er.max() - er.min()) / er.mean(),
        alpha_ratio=alpha[-1] / alpha[0],
        tau_min_ratio=min(t for _, t in poles) / DT_GPRMAX,
        f=f, eps_re=er, eps_im=ei, tand=td, alpha=alpha,
        eps_inf=eps_inf, poles=poles,
    )


def kk_limit(eps_imag):
    """eps'' 一定の媒質が Kramers-Kronig で必ず持つ eps' の変化量。

        Delta eps' = -(2/pi) * eps'' * ln(f_hi/f_lo)

    2 極解析解の eps' 変動がこの極限に近いことを確認するために使う。
    「eps' が完全に一定にならないのは近似の粗さではなく物理」という説明の根拠。
    """
    return -(2.0 / np.pi) * eps_imag * np.log(BAND_HI / BAND_LO)


# ---------------------------------------------------------------------------
# 4. 作図
# ---------------------------------------------------------------------------

def _band_span(ax):
    ax.axvspan(BAND_LO / 1e9, BAND_HI / 1e9, color='0.9', zorder=0)
    ax.axvline(BAND_F0 / 1e9, color='0.5', ls=':', lw=1.0, zorder=1)


def save_fig(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.join(OUTPUT_DIR, name)
    fig.savefig(base + '.png', dpi=200, bbox_inches='tight')
    fig.savefig(base + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  Saved:', base + '.{png,pdf}')


def plot_design(key, wt, target, ev, ev1):
    """設計結果を 4 パネルで説明する図（組成ごと）。

    (a) 2 極の重ね合わせで eps'' が平坦になる様子（1 極との対比つき）
    (b) tan_delta の帯域内変動と、その内訳
    (c) alpha ∝ f の確認
    (d) eps' の KK 変化と、その理論極限
    """
    f = ev['f'] / 1e9
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    fig.suptitle("Flat eps'' design  |  FeO+TiO2 = {:.1f} wt%  [{}]  |  "
                 "target eps'' = {:.6f}".format(wt, key, target),
                 fontsize=14, y=1.00)

    # (a) 極の重ね合わせ
    f_wide = np.geomspace(F_PLOT_LO, F_PLOT_HI, 2000)
    for i, (de, tau) in enumerate(ev['poles']):
        ax_a.semilogx(f_wide / 1e9, debye_eps_imag(f_wide, [(de, tau)]),
                      ls='--', lw=1.2,
                      label='pole {} ({:.3f} GHz)'.format(
                          i + 1, 1.0 / (2 * np.pi * tau) / 1e9))
    ax_a.semilogx(f_wide / 1e9, debye_eps_imag(f_wide, ev['poles']),
                  color='k', lw=2.0, label='2 poles (adopted)')
    ax_a.semilogx(f_wide / 1e9, debye_eps_imag(f_wide, ev1['poles']),
                  color='tab:red', lw=1.2, ls='-.', label='1 pole (for contrast)')
    ax_a.axhline(target, color='b', ls=':', lw=1.5, label='target')
    _band_span(ax_a)
    ax_a.set_xlabel('Frequency [GHz]')
    ax_a.set_ylabel(r"$\varepsilon_r''$")
    ax_a.set_title(r"(a) Superposition of Debye poles flattens $\varepsilon_r''$")
    ax_a.legend(fontsize=9)

    # (b) tan_delta の内訳
    ax_b.plot(f, ev['tand'], color='k', lw=1.8, label='2 poles (adopted)')
    ax_b.plot(f, ev1['tand'], color='tab:red', lw=1.2, ls='-.', label='1 pole')
    ax_b.axhline(target / EPS_R, color='b', ls=':', lw=1.5,
                 label='target {:.6f}'.format(target / EPS_R))
    ax_b.set_xlim(BAND_LO / 1e9, BAND_HI / 1e9)
    ax_b.set_xlabel('Frequency [GHz]')
    ax_b.set_ylabel(r'$\tan\delta$')
    ax_b.set_title(r'(b) $\tan\delta$ in band  (p-p {:.2f}% = '
                   r"$\varepsilon''$ {:.2f}% + $\varepsilon'$ {:.2f}%)"
                   .format(ev['ptp_tand'], ev['ptp_eps_imag'],
                           ev['ptp_eps_real']))
    ax_b.legend(fontsize=9)

    # (c) alpha ∝ f
    ideal = ev['alpha'][0] * (ev['f'] / ev['f'][0])
    ax_c.plot(f, ev['alpha'], color='k', lw=1.8, label='2 poles (adopted)')
    ax_c.plot(f, ideal, color='b', ls=':', lw=1.5, label=r'ideal $\alpha \propto f$')
    ax_c.plot(f, ev1['alpha'], color='tab:red', lw=1.2, ls='-.', label='1 pole')
    ax_c.set_xlim(BAND_LO / 1e9, BAND_HI / 1e9)
    ax_c.set_xlabel('Frequency [GHz]')
    ax_c.set_ylabel(r'$\alpha$ [Np/m]')
    ax_c.set_title(r'(c) $\alpha \propto f$  (in-band ratio {:.3f}, ideal 4.000)'
                   .format(ev['alpha_ratio']))
    ax_c.legend(fontsize=9)

    # (d) eps' の KK 変化
    ax_d.plot(f, ev['eps_re'], color='k', lw=1.8,
              label="2 poles: $\\varepsilon'(f)$")
    ax_d.axhline(EPS_R, color='b', ls=':', lw=1.5,
                 label=r"target $\varepsilon'$ = {:.1f} at $f_0$".format(EPS_R))
    d_kk = abs(kk_limit(target))
    ax_d.plot([BAND_LO / 1e9, BAND_HI / 1e9], [EPS_R + d_kk / 2, EPS_R - d_kk / 2],
              color='tab:green', ls='--', lw=1.2,
              label='KK limit ({:.5f} over band)'.format(d_kk))
    ax_d.set_xlim(BAND_LO / 1e9, BAND_HI / 1e9)
    ax_d.set_xlabel('Frequency [GHz]')
    ax_d.set_ylabel(r"$\varepsilon_r'$")
    ax_d.set_title(r"(d) $\varepsilon'$ must vary (Kramers-Kronig): "
                   'p-p {:.3f}%'.format(ev['ptp_eps_real']))
    ax_d.legend(fontsize=9)

    for ax in axes.ravel():
        ax.grid(alpha=0.4)
        ax.minorticks_on()
    plt.tight_layout()
    save_fig(fig, 'design_{}'.format(key))


def plot_theory():
    """解析解そのものを説明する図（組成に依らないので 1 枚だけ）。

    (a) sech の重ね合わせ（対数周波数 u 軸）
    (b) 極間隔 s を振ったときの帯域内リップル -> arcsinh(1) の位置
        （最大平坦条件はリップル最小とは一致しないことも示す）
    (c) 連続極限（対数一様分布なら厳密に平坦）
    (d) gprMax の制約 tau > dt に対する余裕
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    fig.suptitle('Analytic maximally-flat two-pole design '
                 '(composition-independent)', fontsize=14, y=1.00)

    u_band = np.log(np.array([BAND_LO, BAND_HI]) / BAND_F0)

    # (a) sech の重ね合わせ
    u = np.linspace(-4, 4, 2000)
    for s, ls, lab in ((0.0, ':', 's = 0 (single peak)'),
                       (S_FLAT, '-', 's = arcsinh(1) (max flat)'),
                       (1.6, '--', 's = 1.6 (too wide)')):
        g = 0.5 * (1.0 / np.cosh(u - s) + 1.0 / np.cosh(u + s))
        ax_a.plot(u, g / (1.0 / np.cosh(s)), ls=ls,
                  lw=1.8 if ls == '-' else 1.2, label=lab)
    ax_a.axvspan(u_band[0], u_band[1], color='0.9', zorder=0)
    ax_a.axhline(1.0, color='0.5', ls=':', lw=1.0)
    ax_a.set_xlabel(r'$u = \ln(f/f_0)$')
    ax_a.set_ylabel(r"$\varepsilon''(u)$ (normalised at $u=0$)")
    ax_a.set_title(r'(a) Two sech peaks at $u = \pm s$  (shaded = band)')
    ax_a.legend(fontsize=9)

    # (b) s を振ったときのリップル
    s_grid = np.linspace(0.0, 2.0, 401)
    rip = np.array([ripple_for_s(s) for s in s_grid])
    ax_b.plot(s_grid, rip, color='k', lw=1.8)
    ax_b.axvline(S_FLAT, color='tab:red', ls='--', lw=1.5,
                 label='arcsinh(1) = {:.6f}'.format(S_FLAT))
    ax_b.plot([S_FLAT], [ripple_for_s(S_FLAT)], 'o', color='tab:red', ms=8)
    i_min = int(np.argmin(rip))
    ax_b.plot([s_grid[i_min]], [rip[i_min]], 'x', color='tab:blue', ms=10,
              mew=2, label='numerical minimum s = {:.4f}'.format(s_grid[i_min]))
    ax_b.set_xlabel('pole spacing $s$')
    ax_b.set_ylabel('in-band ripple p-p [%]')
    ax_b.set_title('(b) Max-flat is a closed form, not the ripple minimum')
    ax_b.legend(fontsize=9)

    # (c) 連続極限への収束
    u2 = np.linspace(-6, 6, 2000)
    for n in (1, 2, 3, 5, 9):
        if n == 1:
            s_i = np.array([0.0])
        else:
            s_i = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * (2 * S_FLAT)
        g = sum(0.5 / np.cosh(u2 - si) for si in s_i)
        ax_c.plot(u2, g / g[len(u2) // 2], lw=1.2, label='{} poles'.format(n))
    ax_c.axhline(1.0, color='0.5', ls=':', lw=1.0)
    ax_c.axvspan(u_band[0], u_band[1], color='0.9', zorder=0)
    ax_c.set_xlabel(r'$u = \ln(f/f_0)$')
    ax_c.set_ylabel(r"$\varepsilon''(u)$ (normalised at $u=0$)")
    ax_c.set_title(r'(c) Continuum limit: uniform in $\ln\tau$ is exactly flat')
    ax_c.legend(fontsize=9)

    # (d) gprMax の制約
    rho = density_for_eps(EPS_R)
    keys, ratios = [], []
    for key, wt in COMPOSITIONS.items():
        target = EPS_R * carrier_tandelta(wt, rho)
        _, poles = two_pole_analytic(EPS_R, target)
        keys.append('{}\n{:.1f} wt%'.format(key, wt))
        ratios.append(min(t for _, t in poles) / DT_GPRMAX)
    ax_d.bar(keys, ratios, color='tab:blue', alpha=0.8)
    ax_d.axhline(TAU_MIN_SAFE, color='tab:red', ls='--', lw=1.5,
                 label='safe limit {:.0f}'.format(TAU_MIN_SAFE))
    ax_d.axhline(1.0, color='k', ls=':', lw=1.0,
                 label=r'gprMax limit $\tau > dt$')
    ax_d.set_ylabel(r'min $\tau$ / $dt$')
    ax_d.set_title(r'(d) gprMax constraint: $\tau$ is set by the band only'
                   '\n(identical for every composition)')
    ax_d.legend(fontsize=9)

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.grid(alpha=0.4)
        ax.minorticks_on()
    plt.tight_layout()
    save_fig(fig, 'design_theory')


# ---------------------------------------------------------------------------
# 5. 実行
# ---------------------------------------------------------------------------

def main():
    buf = io.StringIO()

    def _p(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=buf)

    rho = density_for_eps(EPS_R)
    _p('=' * 78)
    _p("帯域内で eps'' を一定に保つ 2 極 Debye パラメータの設計（解析解）")
    _p('=' * 78)
    _p(f'帯域          : {BAND_LO / 1e9:.2f} - {BAND_HI / 1e9:.2f} GHz')
    _p(f'幾何平均 f0   : {BAND_F0 / 1e9:.4f} GHz  '
       f'(解析解の対称中心。eps_r の基準周波数)')
    _p(f'eps_r         : {EPS_R}  (f0 における値)')
    _p(f'密度          : rho = {rho:.6f} g/cm^3   '
       f'(eps_r = {CARRIER_EPS_BASE}^rho の解)')
    _p(f'経験式        : Carrier+1991 Lunar Sourcebook Fig. 9.53 (SOILS)')
    _p(f'                tan_delta = 10^({CARRIER_TAND_A}*S '
       f'+ {CARRIER_TAND_B}*rho - {CARRIER_TAND_C})')
    _p(f'gprMax dt     : {DT_GPRMAX * 1e12:.3f} ps  '
       f'(dx = {DX} m, 2D クーラン条件)')
    _p('')
    _p('注: eps_r は組成に依らないので、全組成で密度・走時・幾何減衰が共通になる。')
    _p('    違いは吸収だけなので、組成の効果だけを切り出して比較できる。')

    _p('\n' + '=' * 78)
    _p('解析解')
    _p('=' * 78)
    num_int, exact = continuum_flatness()
    _p(f'  連続極限の確認: int (1/2) sech(u) du = {num_int:.9f}  '
       f'(= pi/2 = {exact:.9f})')
    _p('    -> ln(tau) に一様分布させると eps_imag は厳密に一定になる。')
    _p('       これが対数等間隔で平坦化できることの原理。')
    _p(f'\n  最大平坦条件: s = arcsinh(1) = ln(1+sqrt2) = {S_FLAT:.6f}')
    _p(f'    緩和周波数比 = e^(2s) = {np.exp(2 * S_FLAT):.6f}  '
       f'(= 3 + 2*sqrt(2) = {3 + 2 * np.sqrt(2):.6f})')
    _p(f'    緩和周波数 = f0/(1+sqrt2), f0*(1+sqrt2) = '
       f'{BAND_F0 / TAU_RATIO / 1e9:.4f}, {BAND_F0 * TAU_RATIO / 1e9:.4f} GHz')
    _p('    De      = sqrt(2) * eps_imag_target   （各極）')
    _p('    eps_inf = eps_r - De                  （f0 で eps_r になる）')
    _p('')
    _p('  恒等式の確認（f0 における実部の寄与の和が厳密に 1 になること）:')
    w0 = 2.0 * np.pi * BAND_F0
    tau_a = [1.0 / (w0 * TAU_RATIO), TAU_RATIO / w0]
    ssum = sum(1.0 / (1.0 + (w0 * t) ** 2) for t in tau_a)
    _p(f'    sum 1/(1+(w0 tau_i)^2) = {ssum:.15f}')
    s_grid = np.linspace(0.0, 2.0, 2001)
    rip = np.array([ripple_for_s(s) for s in s_grid])
    _p(f'\n  有限帯域でのリップル: s = arcsinh(1) で {ripple_for_s(S_FLAT):.3f}%, '
       f'数値的な最小は s = {s_grid[int(np.argmin(rip))]:.4f} で {rip.min():.3f}%')
    _p('    -> 最大平坦条件は「中心で最も平ら」であって「帯域全体のリップル最小」')
    _p('       ではないので、リップルは約 4 倍になる。ただし振幅への影響は深さ')
    _p('       2.75 m で -0.06 dB 対 -0.015 dB であり、合否判定幅 ±0.5 dB に')
    _p('       対してどちらも十分小さい。閉形式で書けることを優先して')
    _p('       s = arcsinh(1) を採用する。')
    _p('       （精度を優先するなら s を上の数値に置き換えればよいが、')
    _p('         その値は帯域幅に依存し閉形式では書けない。）')

    all_results = {}
    for key, wt in COMPOSITIONS.items():
        tand_target = carrier_tandelta(wt, rho)
        target = EPS_R * tand_target

        eps_inf, poles = two_pole_analytic(EPS_R, target)
        ev = evaluate(eps_inf, poles, target)
        eps_inf1, poles1 = one_pole_analytic(EPS_R, target)
        ev1 = evaluate(eps_inf1, poles1, target)

        _p('\n' + '=' * 78)
        _p(f'FeO+TiO2 = {wt} wt%  [{key}]')
        _p('=' * 78)
        _p(f'  Carrier 経験式 tan_delta = {tand_target:.6f}   '
           f'-> 目標 eps_imag = {target:.6f}')
        # 注: alpha 比は 1 極でも 4.0 になるため判別に使えない。帯域端が f0 に
        # 対して対称なので、対称配置なら eps'' は両端で必ず等しくなるため。
        # 平坦さの判定には p-p を見ること。
        _p(f'\n  {"":>16}{"eps_imag RMS":>15}{"eps_imag p-p":>15}'
           f'{"tand p-p":>11}{"alpha 比":>10}{"min tau/dt":>13}')
        for lab, e in (('1 極 (比較用)', ev1), ('2 極 (採用)', ev)):
            _p(f'  {lab:>16}{e["rms_eps_imag"]:>14.3f}%{e["ptp_eps_imag"]:>14.3f}%'
               f'{e["ptp_tand"]:>10.3f}%{e["alpha_ratio"]:>10.3f}'
               f'{e["tau_min_ratio"]:>13.1f}')

        _p('\n  --- 採用: 2 極（解析解）---')
        for i, (de, tau) in enumerate(poles):
            _p(f'    極{i + 1}: De = {de:.6f}, tau = {tau * 1e12:.4f} ps  '
               f'(緩和ピーク {1.0 / (2 * np.pi * tau) / 1e9:.4f} GHz, '
               f'tau/dt = {tau / DT_GPRMAX:.1f})')
        if ev['tau_min_ratio'] < TAU_MIN_SAFE:
            _p(f'    ** WARNING: tau/dt = {ev["tau_min_ratio"]:.1f} は目安 '
               f'{TAU_MIN_SAFE} を下回る **')
        _p('\n  gprMax 記述:')
        _p(f'    #material: {eps_inf:.6f} 0 1 0 regolith')
        _p('    #add_dispersion_debye: {}{} regolith'.format(
            len(poles), ''.join(f' {de:.6f} {tau:.6e}' for de, tau in poles)))

        _p('\n  帯域内の検算:')
        for f0 in (BAND_LO, BAND_F0, BAND_CENTRE_ARITH, BAND_HI):
            i = int(np.argmin(abs(ev['f'] - f0)))
            note = '  <- f0' if abs(f0 - BAND_F0) < 1 else ''
            _p(f"    f = {f0 / 1e9:>4.2f} GHz: eps_re = {ev['eps_re'][i]:.6f}, "
               f"eps_imag = {ev['eps_im'][i]:.6f}, tand = {ev['tand'][i]:.6f}, "
               f"alpha = {ev['alpha'][i]:.4f} Np/m{note}")

        d_kk = abs(kk_limit(target))
        d_act = ev['eps_re'][0] - ev['eps_re'][-1]
        _p(f"\n  eps' の帯域内変化: 実測 {d_act:.6f} / KK 極限 {d_kk:.6f}  "
           f"(比 {d_act / d_kk:.3f})")
        _p("    -> eps' が一定にならないのは近似の粗さではなく Kramers-Kronig")
        _p('       が要求する物理。極数を増やしてもこの量は消えない。')

        all_results[key] = (wt, target, ev, ev1)

    _p('\n' + '=' * 78)
    _p('作図')
    _p('=' * 78)
    for key, (wt, target, ev, ev1) in all_results.items():
        plot_design(key, wt, target, ev, ev1)
    plot_theory()

    _p('\n' + '=' * 78)
    _p('組成間の比較')
    _p('=' * 78)
    _p(f"  {'組成':>8}{'wt%':>7}{'tan_delta':>12}{'eps_inf':>11}"
       f"{'De (各極)':>12}{'alpha@f0':>11}{'alpha 比':>10}")
    for key, (wt, target, ev, _) in all_results.items():
        i = int(np.argmin(abs(ev['f'] - BAND_F0)))
        _p(f"  {key:>8}{wt:>7.1f}{target / EPS_R:>12.6f}{ev['eps_inf']:>11.6f}"
           f"{ev['poles'][0][0]:>12.6f}{ev['alpha'][i]:>11.4f}"
           f"{ev['alpha_ratio']:>10.3f}")
    _p('\n  tau は全組成で共通、De だけが tan_delta に比例する。')
    _p('  -> 組成を変えても緩和の「形」は同じで、振幅だけが変わる。')
    _p('     tau は帯域だけで決まるので、gprMax の制約 tau/dt も組成に依らない。')

    buf.write(f'\n[Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}]\n')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, RESULTS_FILENAME)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(buf.getvalue())
    print(f'\nAll results saved to: {out_path}')


if __name__ == '__main__':
    main()