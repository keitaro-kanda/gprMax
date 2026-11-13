"""周波数分散のあるレゴリス＋水氷の誘電プロファイルと、LSR で見える tan_delta

calc_LLL_mixing_profile.py（分散なし）と calc_mixing_dispersion_profile.py（旧版、
分散あり）を合わせた版。説明と式は calc_LLL_dispersion_profile.md。

  乾燥レゴリス  Carrier et al. (1991) Fig. 9.54（450 MHz）の経験式を 450 MHz で合わせる
  周波数分散    450 MHz の eps'' を「平坦な部分」と「イルメナイトの緩和の部分」に分ける。
                緩和の形は Boivin et al. (2022) Table 4 の Cole-Cole（ホストの損失を除く）を
                3 極 Debye で当てはめたもの。平坦な部分は最大平坦 2 極 Debye。
                全体は 5 極 Debye になり、gprMax の #add_dispersion_debye にそのまま渡せる。
  S の配分      緩和の割合 w(S) = 1 - 10^(-A (S - S_BG))（S_BG 以下は 0）
  水氷の混合    LLL 増分形（pore / excess）。calc_LLL_mixing_profile.py と同じ
  その他        帯域 0.5-2 GHz、往復、アンテナ高 0.35 m、深さ 0-3 m、氷 tan_delta 2e-5

参考文献
  Carrier, W. D. III, Olhoeft, G. R., & Mendell, W. (1991). Physical properties of the
      lunar surface. In Lunar Sourcebook, Cambridge Univ. Press.
  Boivin, A. L. et al. (2022). Determination of broadband complex EM parameters of
      powdered materials: 2. Ilmenite-bearing lunar analogue materials.
      https://doi.org/10.1029/2022JE007200
"""

import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from scipy.optimize import least_squares

try:                      # numpy 2.x
    trapz = np.trapezoid
except AttributeError:    # numpy 1.x
    trapz = np.trapz

# 図の共通ルール
#   文字の大きさ：タイトル・軸ラベル 18、目盛り・凡例・図中の文字 14
#   グリッド線：特に意図がない限り全パネルに付ける（破線 '--'、alpha 0.3）
#   補助目盛り（minor ticks）：全パネルで表示する
plt.rcParams.update({'axes.titlesize': 18, 'axes.labelsize': 18, 'figure.titlesize': 18,
                     'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14,
                     'font.size': 14,
                     'axes.grid': True, 'axes.grid.which': 'major', 'axes.axisbelow': 'line',
                     'grid.linestyle': '--', 'grid.alpha': 0.3,
                     'xtick.minor.visible': True, 'ytick.minor.visible': True})

C0 = 299792458.0          # [m/s]
EPS0 = 8.8541878128e-12   # [F/m]
DB10E = 10.0 * np.log10(np.e)   # 10 log10(e)

# =============================================================================
# 0. 設定  [EDIT HERE]
# =============================================================================
OUTPUT_BASE = '/Volumes/gprMax_water_ice/test_programs_output/LLL_dispersion_profile'
FIGURE_DPI = 200

# --- 深さ ---------------------------------------------------------------------
Z_MAX = 3.0
DZ = 0.01
z = np.arange(0.0, Z_MAX + DZ / 2, DZ)

# --- 帯域・アンテナ -------------------------------------------------------------
BAND_LO, BAND_HI = 0.5e9, 2.0e9        # [Hz] LUPEX GPR 帯域
ANTENNA_HEIGHT = 0.35                  # [m] 往復の空中走時に使う
ASCAN_OUTFILE_PATH = ''                # gprMax の .out があれば指定。空なら合成スペクトル
SYNTH_TUKEY_ALPHA = 0.2
SYNTH_NFREQ = 3001

# --- 乾燥レゴリスの経験式（Carrier et al. 1991, Fig. 9.54, 450 MHz）-------------
F_ANCHOR = 450e6           # [Hz] 経験式を合わせる周波数
EMP_EPS_BASE = 1.843       # eps' = 1.843^rho
EMP_TAND_A = 0.033         # tan_delta = 10^(A S + B rho - C)
EMP_TAND_B = 0.231
EMP_TAND_C = 3.061

# --- 組成と分散の配分 -----------------------------------------------------------
FEOTIO2_WT = 20.0          # [wt%] 既定の FeO+TiO2（S）
S_BG = 7.5                 # [wt%] 緩和を起こさない FeO+TiO2（イルメナイト以外の分）
DISP_MODE = 'split'        # 'split' : w = 1 - 10^(-A (S - S_BG))（既定）
                           # 'all'   : w = 1（旧コードと同じ。分散の上限）
                           # 'none'  : w = 0（分散なし。calc_LLL_mixing_profile と同じ扱い）
DISP_MODES = ('none', 'split', 'all')       # 比較図で並べるもの
DISP_MODE_STYLES = {'none': ':', 'split': '-', 'all': '--'}
DISP_MODE_LABELS = {'none': 'no dispersion', 'split': r'split ($S_{\rm BG}$=' + f'{S_BG:g})',
                    'all': 'all dispersive'}

# --- Boivin et al. (2022) Table 4（1 極 Cole-Cole、Bayesian）--------------------
BOIVIN_CC = {   # eps_inf, delta_eps, tau [s], alpha, sigma_dc [S/m]
    '10wt_ilm': (3.554, 0.194, 4.370e-11, 0.717, 2e-5),
    '15wt_ilm': (3.659, 0.291, 5.201e-11, 0.746, 1e-5),
    '20wt_ilm': (3.792, 0.420, 5.036e-11, 0.756, 1e-5),
}
BOIVIN_ILM_WT = {'10wt_ilm': 10.0, '15wt_ilm': 15.0, '20wt_ilm': 20.0}
BOIVIN_COLORS = {'10wt_ilm': 'tab:blue', '15wt_ilm': 'tab:green', '20wt_ilm': 'tab:red'}
BOIVIN_HOST_EPS_IM = 0.006     # バイトウナイト（ホスト）の eps''。P/L/S/X 帯で一定（Table 7）
BOIVIN_FIT_SAMPLE = '20wt_ilm' # 緩和の形を当てはめる試料
BOIVIN_SUBTRACT_HOST = True    # True: ホストの平坦な損失を除いた緩和だけを当てはめる
                               # False: Cole-Cole の全体を当てはめる（旧コードの値になる）
BOIVIN_FIT_RANGE = (1e8, 8.5e9)   # [Hz] 測定範囲
BOIVIN_FIT_NF = 400
BOIVIN_FIT_BAND = (0.45e9, 2.0e9)  # この範囲の残差を BOIVIN_FIT_BAND_WEIGHT 倍に重み付け
BOIVIN_FIT_BAND_WEIGHT = 5.0
N_RELAX_POLES = 3              # 3 極で帯域の RMS 誤差 0.2 %（2 極では 1.1 %、旧コードは 2 極）
TAU_MIN = 1.2e-11              # [s] gprMax の下限の目安（2D、dx 2.5 mm の dt の 2 倍）

# --- 平坦な部分の実現（最大平坦 2 極 Debye）-------------------------------------
USE_DEBYE_REALIZATION = True   # False: 平坦な部分を周波数によらない理想値にする
FLAT_LO, FLAT_HI = 0.45e9, 2.0e9   # [Hz] 平坦にする範囲（450 MHz の合わせる点を含める）

# --- 水氷 -----------------------------------------------------------------------
ice_contents = [0, 1, 5, 10, 20]
ice_colors = ['k', 'r', 'g', 'b', 'c']
EPS_ICE = 3.15
TAND_ICE = 2.0e-5
RHO_ICE = 0.94
RHO_GRAIN = 2.645
ICE_MODELS = ('pore', 'excess')
ICE_MODEL_DIRS = {'pore': 'pore_ice', 'excess': 'excess_ice'}

# --- 図 -------------------------------------------------------------------------
PROFILE_FREQS = (0.5e9, 1.0e9, 2.0e9)       # 周波数の比較（組成 FEOTIO2_WT）
PROFILE_FIXED_FREQ = 1.0e9                  # 組成の比較で固定する周波数
PROFILE_WTS = (7.5, 15.0, 20.0)             # 組成の比較
STYLE_SET = ('-', '--', '-.')
DRY_PLOT_DEPTH = 1.5                        # [m] 乾燥レゴリスの周波数特性を描く深さ
DRY_PLOT_FRANGE = (0.3e9, 3.0e9)
DRY_PLOT_WTS = (7.5, 15.0, 20.0)
DRY_WT_COLORS = ('tab:blue', 'tab:orange', 'tab:red')

# --- LSR（bscan_detection と同じ推定式）----------------------------------------
LSR_REF_DEPTH = 0.25       # [m] 区間の上端
LSR_FLOOR_DB = 20.0        # 入射スペクトルの最大からこれ以上落ちる周波数は使わない
LSR_DEPTHS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)  # (a) に描く区間の下端
LSR_SEG = (0.25, 2.0)      # [m] 組成の比較で使う区間
LSR_ICE_FOR_COMP = 10      # [vol%] 組成の比較で使う氷量
MODEL_TAND_FREQ = 1.0e9    # bscan_detection が理論値に使う周波数
S_GRID = np.arange(0.0, 25.0 + 1e-9, 0.5)


# =============================================================================
# 1. 密度と経験式
# =============================================================================
def density_profile(depth_m):
    """深さ [m] -> バルク密度 [g/cm^3]（Carrier et al. 1991）。"""
    z_cm = np.asarray(depth_m, dtype=float) * 100.0
    return 1.92 * (z_cm + 12.2) / (z_cm + 18.0)


def emp_eps_real(rho):
    return EMP_EPS_BASE ** np.asarray(rho, dtype=float)


def emp_tandelta(rho, S):
    return 10.0 ** (EMP_TAND_A * S + EMP_TAND_B * np.asarray(rho, dtype=float) - EMP_TAND_C)


def porosity(rho):
    return 1.0 - np.asarray(rho, dtype=float) / RHO_GRAIN


def relax_fraction(S, mode=None):
    """450 MHz の eps'' のうち緩和の部分の割合 w。密度によらない。"""
    mode = DISP_MODE if mode is None else mode
    S = np.asarray(S, dtype=float)
    if mode == 'none':
        return np.zeros_like(S)
    if mode == 'all':
        return np.ones_like(S)
    return np.where(S > S_BG, 1.0 - 10.0 ** (-EMP_TAND_A * (S - S_BG)), 0.0)


# =============================================================================
# 2. Boivin の緩和を 2 極 Debye で当てはめる
# =============================================================================
def cole_cole_relax(f, key):
    """Cole-Cole の緩和の項だけ（eps_inf と sigma_dc を除く）。(re, im)、im > 0。"""
    _, de, tau, a, _ = BOIVIN_CC[key]
    c = de / (1.0 + (1j * 2 * np.pi * np.asarray(f) * tau) ** a)
    return c.real, -c.imag


def cole_cole_total(f, key):
    """Cole-Cole の全体（eps_inf と sigma_dc を含む）。(re, im)、im > 0。"""
    ei, de, tau, a, sdc = BOIVIN_CC[key]
    w = 2 * np.pi * np.asarray(f)
    c = ei + de / (1.0 + (1j * w * tau) ** a) - 1j * sdc / (w * EPS0)
    return c.real, -c.imag


def debye_sum(f, poles):
    """sum_k De_k / (1 + j w tau_k)。(re, im)、im > 0。poles: [(De, tau), ...]"""
    w = 2 * np.pi * np.atleast_1d(np.asarray(f, dtype=float))
    c = np.zeros(w.shape, dtype=complex)
    for de, tau in poles:
        c = c + de / (1.0 + 1j * w * tau)
    return c.real, -c.imag


def fit_relax_poles():
    """緩和の eps''（ホストの損失を除く）に N_RELAX_POLES 極の Debye を当てはめる。"""
    f = np.geomspace(*BOIVIN_FIT_RANGE, BOIVIN_FIT_NF)
    _, im = cole_cole_relax(f, BOIVIN_FIT_SAMPLE)
    target = im - (BOIVIN_HOST_EPS_IM if BOIVIN_SUBTRACT_HOST else 0.0)
    if np.any(target <= 0):
        raise ValueError('ホストの損失を除くと eps\'\' が負になる周波数がある')
    wt = np.ones_like(f)
    wt[(f >= BOIVIN_FIT_BAND[0]) & (f <= BOIVIN_FIT_BAND[1])] = BOIVIN_FIT_BAND_WEIGHT
    tau_cc = BOIVIN_CC[BOIVIN_FIT_SAMPLE][2]
    n = N_RELAX_POLES
    seeds = np.geomspace(tau_cc, tau_cc * 6.0, n) if n > 1 else np.array([tau_cc])
    de0 = BOIVIN_CC[BOIVIN_FIT_SAMPLE][1] / n
    x0 = np.r_[np.full(n, de0), np.log(seeds)]
    lo = np.r_[np.zeros(n), np.full(n, np.log(TAU_MIN))]
    hi = np.r_[np.full(n, 10.0), np.full(n, np.log(1e-8))]

    def resid(x):
        _, di = debye_sum(f, list(zip(x[:n], np.exp(x[n:]))))
        return (di - target) / target * wt

    sol = least_squares(resid, x0, bounds=(lo, hi), ftol=1e-12, xtol=1e-12, max_nfev=20000)
    poles = sorted(zip(sol.x[:n], np.exp(sol.x[n:])), key=lambda p: p[1])
    _, di = debye_sum(f, poles)
    rel = (di - target) / target
    band = (f >= BAND_LO) & (f <= BAND_HI)
    info = {'f': f, 'target': target, 'fit': di,
            'rms_band_pct': float(np.sqrt(np.mean(rel[band] ** 2)) * 100),
            'rms_all_pct': float(np.sqrt(np.mean(rel ** 2)) * 100)}
    return [(float(a), float(b)) for a, b in poles], info


RELAX_POLES, RELAX_FIT = fit_relax_poles()
_W_A = 2 * np.pi * F_ANCHOR
RELAX_IM_A = float(debye_sum(F_ANCHOR, RELAX_POLES)[1][0])   # 単位強さの緩和の 450 MHz の eps''

# 平坦な部分（最大平坦 2 極 Debye、解析解）
_F0_FLAT = np.sqrt(FLAT_LO * FLAT_HI)
_TAU_RATIO = 1.0 + np.sqrt(2.0)
FLAT_TAUS = (1.0 / (2 * np.pi * _F0_FLAT * _TAU_RATIO), _TAU_RATIO / (2 * np.pi * _F0_FLAT))
FLAT_SHAPE_A = float(debye_sum(F_ANCHOR, [(1.0, t) for t in FLAT_TAUS])[1][0])   # 単位強さの 450 MHz の値
FLAT_RIPPLE = float(debye_sum(_F0_FLAT, [(1.0, t) for t in FLAT_TAUS])[1][0]) / FLAT_SHAPE_A - 1.0


# =============================================================================
# 3. 乾燥レゴリスの誘電率（4 極 Debye）
# =============================================================================
def dry_poles(depth_m, S=None, mode=None):
    """乾燥レゴリスの eps_inf と Debye 極（gprMax に渡す形）。

    戻り値: eps_inf (Nz,), [(De_k (Nz,), tau_k), ...]
    450 MHz で eps' = 1.843^rho、eps'' = eps' tan_delta(9.54) になるように決める
    （平坦な部分の実現誤差を除く）。
    """
    S = FEOTIO2_WT if S is None else S
    rho = density_profile(depth_m)
    er_a = emp_eps_real(rho)
    ei_a = er_a * emp_tandelta(rho, S)
    w = relax_fraction(S, mode)
    poles = []
    # 緩和の部分：形は Boivin、強さは 450 MHz で w * eps''
    scale = w * ei_a / RELAX_IM_A
    for de, tau in RELAX_POLES:
        poles.append((scale * de, tau))
    # 平坦な部分：450 MHz でちょうど (1 - w) * eps'' になる最大平坦 2 極
    #   （帯域内は 0 〜 +FLAT_RIPPLE だけ大きくなる。最大平坦 2 極の実現誤差）
    if USE_DEBYE_REALIZATION:
        de_flat = (1.0 - w) * ei_a / FLAT_SHAPE_A
        for tau in FLAT_TAUS:
            poles.append((de_flat, tau))
    # eps_inf：450 MHz の eps' が経験式に一致するように
    eps_inf = er_a.copy()
    for de, tau in poles:
        eps_inf = eps_inf - de / (1.0 + (_W_A * tau) ** 2)
    return eps_inf, poles


def dry_eps(depth_m, freq_hz, S=None, mode=None):
    """乾燥レゴリスの (eps', eps'')。形は (Nz, Nf)。"""
    d = np.atleast_1d(np.asarray(depth_m, dtype=float))
    f = np.atleast_1d(np.asarray(freq_hz, dtype=float))
    eps_inf, poles = dry_poles(d, S, mode)
    w = 2 * np.pi * f[None, :]
    c = np.repeat(eps_inf[:, None], f.size, axis=1).astype(complex)
    for de, tau in poles:
        c = c + np.atleast_1d(de)[:, None] / (1.0 + 1j * w * tau)
    er, ei = c.real, -c.imag
    if not USE_DEBYE_REALIZATION:          # 平坦な部分を理想値で足す
        S_ = FEOTIO2_WT if S is None else S
        rho = density_profile(d)
        ei = ei + ((1.0 - relax_fraction(S_, mode)) * emp_eps_real(rho)
                   * emp_tandelta(rho, S_))[:, None]
    return er, ei


# =============================================================================
# 4. 水氷の混合（LLL 増分形、calc_LLL_mixing_profile.py と同じ）
# =============================================================================
_EPS_ICE_CBRT = EPS_ICE ** (1.0 / 3.0)


def mix_ice(er, ei, ice_volpct, model):
    v = float(ice_volpct) / 100.0
    if v == 0.0:
        return er, ei
    if model == 'excess':
        return (((1.0 - v) * er ** (1.0 / 3.0) + v * _EPS_ICE_CBRT) ** 3,
                (1.0 - v) * ei + v * EPS_ICE * TAND_ICE)
    return ((er ** (1.0 / 3.0) + v * (_EPS_ICE_CBRT - 1.0)) ** 3,
            ei + v * EPS_ICE * TAND_ICE)


def max_ice_volpct(depth_m, model):
    if model == 'excess':
        return np.full_like(np.asarray(depth_m, dtype=float), 100.0)
    return 100.0 * porosity(density_profile(depth_m))


def medium_eps(depth_m, freq_hz, ice_volpct, model, S=None, mode=None):
    er, ei = dry_eps(depth_m, freq_hz, S, mode)
    return mix_ice(er, ei, ice_volpct, model)


def alpha_velocity(er, ei, freq_hz):
    """減衰係数 alpha [Np/m] と位相速度 [m/s]。"""
    td = ei / er
    w = (2 * np.pi * np.atleast_1d(np.asarray(freq_hz, dtype=float)))[None, :]
    root = np.sqrt(1.0 + td ** 2)
    alpha = (w / C0) * np.sqrt(er / 2.0) * np.sqrt(root - 1.0)
    n_re = np.sqrt(er / 2.0) * np.sqrt(root + 1.0)
    return alpha, C0 / n_re


# =============================================================================
# 5. 入射スペクトルと伝搬
# =============================================================================
_incident = None


def get_incident_spectrum():
    """(f [Hz], |S0(f)|, 出典)。帯域内のみ。"""
    global _incident
    if _incident is not None:
        return _incident
    src = 'synthetic (Tukey {:.1f})'.format(SYNTH_TUKEY_ALPHA)
    f = amp = None
    if ASCAN_OUTFILE_PATH and os.path.exists(ASCAN_OUTFILE_PATH):
        try:
            import h5py
            with h5py.File(ASCAN_OUTFILE_PATH, 'r') as fh:
                dt = fh.attrs['dt']
                e = np.array(fh['rxs/rx1/Ez'])
            f_all = np.fft.rfftfreq(len(e), d=dt)
            amp_all = np.abs(np.fft.rfft(e))
            m = (f_all >= BAND_LO) & (f_all <= BAND_HI)
            f, amp, src = f_all[m], amp_all[m], ASCAN_OUTFILE_PATH
        except Exception as exc:            # noqa: BLE001
            print(f'  [warn] .out を読めないので合成スペクトルを使う: {exc}')
    if f is None:
        from scipy.signal.windows import tukey
        f = np.linspace(BAND_LO, BAND_HI, SYNTH_NFREQ)
        amp = tukey(f.size, alpha=SYNTH_TUKEY_ALPHA)
    _incident = (f, amp, src)
    return _incident


_prop_cache = {}


def propagation(ice_volpct, model, S=None, mode=None):
    """往復の累積減衰 [Np] と往復走時 [s]（空中を含む）。形は (Nz, Nf)。"""
    S = FEOTIO2_WT if S is None else S
    mode = DISP_MODE if mode is None else mode
    key = (ice_volpct, model, float(S), mode)
    if key not in _prop_cache:
        f, _, _ = get_incident_spectrum()
        er, ei = medium_eps(z, f, ice_volpct, model, S, mode)
        a, v = alpha_velocity(er, ei, f)
        ca = np.zeros_like(a)
        ct = np.zeros_like(a)
        ca[1:] = np.cumsum(0.5 * (a[1:] + a[:-1]) * DZ, axis=0)
        ct[1:] = np.cumsum(0.5 * (1 / v[1:] + 1 / v[:-1]) * DZ, axis=0)
        _prop_cache[key] = (2.0 * ca, 2.0 * ct + 2.0 * ANTENNA_HEIGHT / C0)
    return _prop_cache[key]


def incident_centroid():
    f, s0, _ = get_incident_spectrum()
    P = s0 ** 2
    return float(trapz(f * P, f) / trapz(P, f))


def lsr_mask():
    f, s0, _ = get_incident_spectrum()
    P = s0 ** 2
    return P >= P.max() * 10 ** (-LSR_FLOOR_DB / 10.0)


def lsr_curve(ice_volpct, model, z1, z2, S=None, mode=None):
    """区間 z1 -> z2 のパワーのスペクトル比 [dB] と往復走時差 [ns]。"""
    f, _, _ = get_incident_spectrum()
    ca, ct = propagation(ice_volpct, model, S, mode)
    i1, i2 = int(round(z1 / DZ)), int(round(z2 / DZ))
    y = -2.0 * DB10E * (ca[i2] - ca[i1])
    dt_ns = float(np.interp(incident_centroid(), f, ct[i2] - ct[i1])) * 1e9
    return y, dt_ns


def lsr_tandelta(ice_volpct, model, z1, z2, S=None, mode=None):
    """bscan_detection と同じ推定式：tan_delta = -slope / (10log10(e) 2 pi dt)。"""
    f, _, _ = get_incident_spectrum()
    m = lsr_mask()
    y, dt_ns = lsr_curve(ice_volpct, model, z1, z2, S, mode)
    slope = np.polyfit(f[m] / 1e9, y[m], 1)[0]
    return float(-slope / (DB10E * 2 * np.pi * dt_ns))


def model_tandelta(ice_volpct, model, z1, z2, freq_hz, S=None, mode=None):
    """区間の tan_delta の平均（bscan_detection の理論値と同じ取り方）。"""
    zz = np.linspace(z1, z2, 41)[1:-1]
    er, ei = medium_eps(zz, np.array([freq_hz]), ice_volpct, model, S, mode)
    return float(np.mean(ei[:, 0] / er[:, 0]))


# =============================================================================
# 6. 作図
# =============================================================================
def out_dir(model=None, *parts):
    p = OUTPUT_BASE if model is None else os.path.join(OUTPUT_BASE, ICE_MODEL_DIRS[model])
    p = os.path.join(p, *parts)
    os.makedirs(p, exist_ok=True)
    return p


def save_fig(fig, path_base):
    fig.savefig(path_base + '.png', dpi=FIGURE_DPI, bbox_inches='tight')
    os.makedirs(os.path.join(os.path.dirname(path_base), 'pdf'), exist_ok=True)
    fig.savefig(os.path.join(os.path.dirname(path_base), 'pdf',
                             os.path.basename(path_base) + '.pdf'), bbox_inches='tight')
    plt.close(fig)
    return path_base + '.png'


def depth_axis(ax, xlabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Depth [m]')
    ax.set_ylim(Z_MAX, 0.0)


def log_freq_axis(ax, ticks=(0.1, 0.2, 0.5, 1, 2, 5, 10)):
    """周波数の対数軸。目盛りを 0.5, 1, 2 のような普通の数字にする。"""
    ax.set_xscale('log')
    lo, hi = ax.get_xlim()
    ax.xaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())


def shade_band(ax):
    ax.axvspan(BAND_LO / 1e9, BAND_HI / 1e9, color='0.85', zorder=0)
    ax.axvline(F_ANCHOR / 1e9, color='k', ls='-.', lw=1.0)


def plot_boivin_check():
    """(a) Boivin の eps'' と当てはめ、(b) 450 MHz で割った形（2 成分の再現）。"""
    f = np.geomspace(*BOIVIN_FIT_RANGE, 400)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    ax = axes[0]
    for key, col in BOIVIN_COLORS.items():
        _, im = cole_cole_total(f, key)
        ax.plot(f / 1e9, im, color=col, lw=2.2)
    ax.axhline(BOIVIN_HOST_EPS_IM, color='0.4', ls=':', lw=2)
    ax.plot(RELAX_FIT['f'] / 1e9, RELAX_FIT['target'], color='k', lw=1.2)
    ax.plot(RELAX_FIT['f'] / 1e9, RELAX_FIT['fit'], color='k', ls='--', lw=2)
    shade_band(ax)
    ax.set_yscale('log')
    log_freq_axis(ax)
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel(r"$\varepsilon''$")
    ax.set_title('(a) Boivin+2022 and Debye fit')

    ax = axes[1]
    for key, col in BOIVIN_COLORS.items():
        _, im = cole_cole_total(f, key)
        _, im_a = cole_cole_total(F_ANCHOR, key)
        ax.plot(f / 1e9, im / im_a, color=col, lw=2.2)
        # 2 成分：ホストの平坦な損失 + 当てはめた緩和の形
        rel_a = im_a - BOIVIN_HOST_EPS_IM
        _, d = debye_sum(f, RELAX_POLES)
        pred = (BOIVIN_HOST_EPS_IM + rel_a * d / RELAX_IM_A) / im_a
        ax.plot(f / 1e9, pred, color=col, ls='--', lw=2)
    shade_band(ax)
    ax.set_xscale('log')
    ax.set_xlim(0.3, 3.0)
    log_freq_axis(ax, (0.3, 0.45, 1, 2, 3))
    ax.set_ylim(0.6, 2.2)
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel(r"$\varepsilon''(f)\,/\,\varepsilon''(450\,{\rm MHz})$")
    ax.set_title('(b) Shape normalised at 450 MHz')

    h = [Line2D([0], [0], color=c, lw=2.2, label=k.replace('wt_ilm', ' wt% ilm'))
         for k, c in BOIVIN_COLORS.items()]
    h += [Line2D([0], [0], color='0.4', ls=':', lw=2, label='host (bytownite)'),
          Line2D([0], [0], color='k', lw=1.2, label='relaxation part ({})'.format(
              BOIVIN_FIT_SAMPLE.replace('wt_ilm', ' wt%'))),
          Line2D([0], [0], color='k', ls='--', lw=2, label='{}-pole Debye / host + Debye'.format(
              N_RELAX_POLES)),
          Line2D([0], [0], color='k', ls='-.', lw=1, label='450 MHz')]
    fig.legend(handles=h, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    return save_fig(fig, os.path.join(out_dir(), 'boivin_dispersion_check'))


def plot_dry_dispersion():
    """深さ DRY_PLOT_DEPTH の乾燥レゴリスの (a) eps' (b) eps'' (c) tan_delta、(d) w(S)。"""
    f = np.geomspace(*DRY_PLOT_FRANGE, 300)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    keys = [("(a) Real part", r"$\varepsilon'$"), ("(b) Imaginary part", r"$\varepsilon''$"),
            ("(c) Loss tangent", r"$\tan\delta$")]
    for S, col in zip(DRY_PLOT_WTS, DRY_WT_COLORS):
        for mode in DISP_MODES:
            er, ei = dry_eps(np.array([DRY_PLOT_DEPTH]), f, S, mode)
            for ax, y in zip(axes.flat[:3], (er[0], ei[0], ei[0] / er[0])):
                ax.plot(f / 1e9, y, color=col, ls=DISP_MODE_STYLES[mode], lw=2)
        rho = density_profile(DRY_PLOT_DEPTH)
        axes[0, 0].plot(F_ANCHOR / 1e9, emp_eps_real(rho), 'o', color='k', ms=7)
        axes[0, 1].plot(F_ANCHOR / 1e9, emp_eps_real(rho) * emp_tandelta(rho, S), 'o',
                        color=col, mec='k', ms=8)
        axes[1, 0].plot(F_ANCHOR / 1e9, emp_tandelta(rho, S), 'o', color=col, mec='k', ms=8)
    for ax, (ttl, yl) in zip(axes.flat[:3], keys):
        shade_band(ax)
        ax.set_xscale('log')
        ax.set_xlim(DRY_PLOT_FRANGE[0] / 1e9, DRY_PLOT_FRANGE[1] / 1e9)
        log_freq_axis(ax, (0.3, 0.45, 1, 2, 3))
        ax.set_xlabel('Frequency [GHz]')
        ax.set_ylabel(yl)
        ax.set_title(ttl)
    ax = axes[1, 1]
    for mode in DISP_MODES:
        ax.plot(S_GRID, relax_fraction(S_GRID, mode), color='k', ls=DISP_MODE_STYLES[mode], lw=2)
    ax.set_xlabel(r'FeO+TiO$_2$ [wt%]')
    ax.set_ylabel(r'Relaxation fraction $w$')
    ax.set_title('(d) Dispersive fraction at 450 MHz')
    ax.set_ylim(-0.05, 1.05)
    h = [Line2D([0], [0], color=c, lw=2, label=f'{S:g} wt%')
         for S, c in zip(DRY_PLOT_WTS, DRY_WT_COLORS)]
    h += [Line2D([0], [0], color='k', ls=DISP_MODE_STYLES[m], lw=2, label=DISP_MODE_LABELS[m])
          for m in DISP_MODES]
    h += [Line2D([0], [0], marker='o', color='w', mec='k', mfc='0.6', ms=8,
                 label='Fig. 9.54 (450 MHz)')]
    fig.legend(handles=h, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    return save_fig(fig, os.path.join(out_dir(), 'dry_dispersion_{:.1f}m'.format(DRY_PLOT_DEPTH)))


def profile_set(model, pairs, mode=None):
    """[(S, f), ...] -> 各量 (n_ice, n_pair, Nz)。"""
    out = {k: np.zeros((len(ice_contents), len(pairs), z.size))
           for k in ('eps_re', 'eps_im', 'tand', 'alpha')}
    for j, (S, f) in enumerate(pairs):
        for i, c in enumerate(ice_contents):
            er, ei = medium_eps(z, np.array([f]), c, model, S, mode)
            a, _ = alpha_velocity(er, ei, np.array([f]))
            out['eps_re'][i, j] = er[:, 0]
            out['eps_im'][i, j] = ei[:, 0]
            out['tand'][i, j] = ei[:, 0] / er[:, 0]
            out['alpha'][i, j] = a[:, 0]
    return out


def plot_profiles(model, data, labels, fname, ref_S):
    """2x2：eps', eps'', tan_delta, alpha。色は氷量、線種は周波数か組成。"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    spec = [('eps_re', r"$\varepsilon'$", '(a) Real part'),
            ('eps_im', r"$\varepsilon''$", '(b) Imaginary part'),
            ('tand', r"$\tan\delta$", '(c) Loss tangent'),
            ('alpha', r'$\alpha$ [Np/m]', '(d) Attenuation')]
    rho = density_profile(z)
    for ax, (k, xl, ttl) in zip(axes.flat, spec):
        for i in range(len(ice_contents)):
            for j in range(len(labels)):
                ax.plot(data[k][i, j], z, color=ice_colors[i], ls=STYLE_SET[j], lw=1.6)
        if k == 'eps_re':
            ax.plot(emp_eps_real(rho), z, color='magenta', ls=':', lw=2)
        if k == 'tand':
            for S in np.atleast_1d(ref_S):
                ax.plot(emp_tandelta(rho, S), z, color='magenta', ls=':', lw=2)
        depth_axis(ax, xl)
        ax.set_title(ttl)
    h = [Line2D([0], [0], color='0.45', ls=STYLE_SET[j], lw=2, label=l) for j, l in enumerate(labels)]
    h += [Line2D([0], [0], color=ice_colors[i], lw=2, label=f'{c} vol% ice')
          for i, c in enumerate(ice_contents)]
    h += [Line2D([0], [0], color='magenta', ls=':', lw=2, label='Fig. 9.54 (dry, 450 MHz)')]
    fig.legend(handles=h, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    return save_fig(fig, os.path.join(out_dir(model), fname))


def plot_lsr_spectra(model):
    """(a) 乾燥レゴリスの区間ごとのスペクトル比と直線、(b) 直線からのずれ。"""
    f, _, _ = get_incident_spectrum()
    m = lsr_mask()
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(LSR_DEPTHS)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for d, col in zip(LSR_DEPTHS, cols):
        y, _ = lsr_curve(0, model, LSR_REF_DEPTH, d)
        p = np.polyfit(f[m] / 1e9, y[m], 1)
        line = np.polyval(p, f / 1e9)
        axes[0].plot(f / 1e9, y, color=col, lw=2.2)
        axes[0].plot(f / 1e9, line, color=col, ls=':', lw=1.5)
        axes[1].plot(f / 1e9, y - line, color=col, lw=2.2)
    axes[0].set_ylabel('Spectral ratio [dB]')
    axes[0].set_title(f'(a) Ratio ({FEOTIO2_WT:g} wt%, dry)')
    axes[1].set_ylabel('Ratio - linear fit [dB]')
    axes[1].set_title('(b) Curvature')
    for ax in axes:
        ax.set_xlabel('Frequency [GHz]')
        ax.set_xlim(BAND_LO / 1e9, BAND_HI / 1e9)
    h = [Line2D([0], [0], color=c, lw=2.2, label=f'{LSR_REF_DEPTH:g}–{d:g} m')
         for d, c in zip(LSR_DEPTHS, cols)]
    h += [Line2D([0], [0], color='k', ls=':', lw=1.5, label='linear fit')]
    fig.legend(handles=h, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    return save_fig(fig, os.path.join(out_dir(model), 'lsr_spectra'))


def lsr_profiles(model, S=None, mode=None):
    """区間 LSR_REF_DEPTH -> z の LSR の tan_delta と、1 GHz・450 MHz の平均 tan_delta。"""
    zs = z[z >= LSR_REF_DEPTH + 0.1 - 1e-9]
    out = {'z': zs}
    for c in ice_contents:
        out[('lsr', c)] = np.array([lsr_tandelta(c, model, LSR_REF_DEPTH, d, S, mode) for d in zs])
        out[('mod', c)] = np.array([model_tandelta(c, model, LSR_REF_DEPTH, d, MODEL_TAND_FREQ,
                                                   S, mode) for d in zs])
        out[('anc', c)] = np.array([model_tandelta(c, model, LSR_REF_DEPTH, d, F_ANCHOR,
                                                   S, mode) for d in zs])
    return out


def plot_lsr_profile(model, L):
    """(a) LSR の tan_delta、(b) 0 vol% からの差。実線 LSR、点線 1 GHz の平均。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))
    zs = L['z']
    for i, c in enumerate(ice_contents):
        axes[0].plot(L[('lsr', c)], zs, color=ice_colors[i], lw=2)
        axes[0].plot(L[('mod', c)], zs, color=ice_colors[i], ls=':', lw=2)
        if c == 0:
            axes[0].plot(L[('anc', c)], zs, color='0.5', ls='-.', lw=2)
            continue
        axes[1].plot(L[('lsr', c)] - L[('lsr', 0)], zs, color=ice_colors[i], lw=2)
        axes[1].plot(L[('mod', c)] - L[('mod', 0)], zs, color=ice_colors[i], ls=':', lw=2)
    depth_axis(axes[0], r'$\tan\delta$')
    depth_axis(axes[1], r'$\Delta\tan\delta$ (ice $-$ dry)')
    axes[0].set_title(f'(a) Segment {LSR_REF_DEPTH:g} m – depth')
    axes[1].set_title('(b) Change by ice')
    axes[1].axvline(0, color='k', lw=0.8)
    h = [Line2D([0], [0], color=ice_colors[i], lw=2, label=f'{c} vol% ice')
         for i, c in enumerate(ice_contents)]
    h += [Line2D([0], [0], color='0.45', lw=2, label='LSR estimate'),
          Line2D([0], [0], color='0.45', ls=':', lw=2,
                 label=f'mean at {MODEL_TAND_FREQ/1e9:g} GHz'),
          Line2D([0], [0], color='0.5', ls='-.', lw=2, label='mean at 450 MHz (dry)')]
    fig.legend(handles=h, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    return save_fig(fig, os.path.join(out_dir(model), 'lsr_tandelta_profile'))


def lsr_vs_composition(model):
    """S を振ったときの区間 LSR_SEG の tan_delta（乾燥）と、氷による差。"""
    z1, z2 = LSR_SEG
    c = LSR_ICE_FOR_COMP
    res = {}
    for mode in DISP_MODES:
        r = {k: np.zeros(S_GRID.size) for k in ('lsr0', 'lsrc', 'mod0', 'modc')}
        for j, S in enumerate(S_GRID):
            r['lsr0'][j] = lsr_tandelta(0, model, z1, z2, S, mode)
            r['lsrc'][j] = lsr_tandelta(c, model, z1, z2, S, mode)
            r['mod0'][j] = model_tandelta(0, model, z1, z2, MODEL_TAND_FREQ, S, mode)
            r['modc'][j] = model_tandelta(c, model, z1, z2, MODEL_TAND_FREQ, S, mode)
            _prop_cache.clear()              # 1 件 約 15 MB なので溜めない
        res[mode] = r
    _prop_cache.clear()
    return res


def plot_lsr_vs_composition(model, res):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    for mode in DISP_MODES:
        r = res[mode]
        ls = DISP_MODE_STYLES[mode]
        axes[0].plot(S_GRID, r['lsr0'], color='tab:red', ls=ls, lw=2)
        axes[0].plot(S_GRID, r['mod0'], color='tab:blue', ls=ls, lw=2)
        axes[1].plot(S_GRID, r['lsrc'] - r['lsr0'], color='tab:red', ls=ls, lw=2)
        axes[1].plot(S_GRID, r['modc'] - r['mod0'], color='tab:blue', ls=ls, lw=2)
        with np.errstate(divide='ignore', invalid='ignore'):
            axes[2].plot(S_GRID, r['lsr0'] / r['mod0'], color='tab:red', ls=ls, lw=2)
    axes[0].set_ylabel(r'$\tan\delta$ (dry)')
    axes[0].set_title('(a) Dry regolith')
    axes[1].set_ylabel(r'$\Delta\tan\delta$ (' + f'{LSR_ICE_FOR_COMP} vol% ice $-$ dry)')
    axes[1].set_title('(b) Change by ice')
    axes[1].axhline(0, color='k', lw=0.8)
    axes[2].set_ylabel('LSR / model (dry)')
    axes[2].set_title('(c) Apparent amplification')
    for ax in axes:
        ax.set_xlabel(r'FeO+TiO$_2$ [wt%]')
        ax.axvline(S_BG, color='0.5', ls='-.', lw=1)
    h = [Line2D([0], [0], color='tab:red', lw=2, label='LSR estimate'),
         Line2D([0], [0], color='tab:blue', lw=2, label=f'mean at {MODEL_TAND_FREQ/1e9:g} GHz')]
    h += [Line2D([0], [0], color='k', ls=DISP_MODE_STYLES[m], lw=2, label=DISP_MODE_LABELS[m])
          for m in DISP_MODES]
    h += [Line2D([0], [0], color='0.5', ls='-.', lw=1, label=r'$S_{\rm BG}$')]
    fig.legend(handles=h, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    return save_fig(fig, os.path.join(out_dir(model), 'lsr_vs_composition_{:g}-{:g}m'.format(
        *LSR_SEG)))


# =============================================================================
# 7. 数値出力と検算
# =============================================================================
def write_profile_csv(model, data, labels, fname):
    path = os.path.join(out_dir(model), fname)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        head = ['depth_m', 'rho']
        for c in ice_contents:
            for l in labels:
                t = l.replace(' ', '')
                head += [f'{k}_{c}vol_{t}' for k in ('eps_re', 'eps_im', 'tand', 'alpha')]
        w.writerow(head)
        rho = density_profile(z)
        for n in range(z.size):
            row = [round(float(z[n]), 4), float(rho[n])]
            for i in range(len(ice_contents)):
                for j in range(len(labels)):
                    row += [float(data[k][i, j, n]) for k in ('eps_re', 'eps_im', 'tand', 'alpha')]
            w.writerow(row)
    return path


def write_lsr_csv(model, L, res):
    p1 = os.path.join(out_dir(model), 'lsr_tandelta_profile.csv')
    with open(p1, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['depth_m'] + [f'{k}_{c}vol' for c in ice_contents
                                  for k in ('tand_lsr', 'tand_1GHz', 'tand_450MHz')])
        for n, d in enumerate(L['z']):
            w.writerow([round(float(d), 4)] + [float(L[(k, c)][n]) for c in ice_contents
                                                for k in ('lsr', 'mod', 'anc')])
    p2 = os.path.join(out_dir(model), 'lsr_vs_composition.csv')
    with open(p2, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        head = ['S_wt']
        for m in DISP_MODES:
            head += [f'{k}_{m}' for k in ('lsr_dry', f'lsr_{LSR_ICE_FOR_COMP}vol', 'model_dry',
                                          f'model_{LSR_ICE_FOR_COMP}vol')]
        w.writerow(head)
        for j, S in enumerate(S_GRID):
            row = [float(S)]
            for m in DISP_MODES:
                r = res[m]
                row += [r['lsr0'][j], r['lsrc'][j], r['mod0'][j], r['modc'][j]]
            w.writerow(row)
    return [p1, p2]


def write_gprmax_poles(depths=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)):
    """乾燥レゴリスの eps_inf と Debye 極を CSV と gprMax の書式で出す（S = FEOTIO2_WT）。"""
    path = os.path.join(out_dir(), 'dry_debye_poles_{:g}wt_{}.csv'.format(FEOTIO2_WT, DISP_MODE))
    d = np.asarray(depths, dtype=float)
    eps_inf, poles = dry_poles(d)
    lines = []
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        head = ['depth_m', 'rho', 'eps_inf']
        for k in range(len(poles)):
            head += [f'De{k+1}', f'tau{k+1}_s']
        w.writerow(head)
        for n, dd in enumerate(d):
            row = [dd, float(density_profile(dd)), float(eps_inf[n])]
            txt = []
            for de, tau in poles:
                row += [float(np.atleast_1d(de)[n]), tau]
                txt.append('{:.6g} {:.6e}'.format(float(np.atleast_1d(de)[n]), tau))
            w.writerow(row)
            name = 'dry_{:.2f}m'.format(dd)
            lines.append('#material: {:.6f} 0 1 0 {}'.format(float(eps_inf[n]), name))
            lines.append('#add_dispersion_debye: {} {} {}'.format(len(poles), ' '.join(txt), name))
    return path, lines


def run_checks(gprmax_lines):
    global USE_DEBYE_REALIZATION
    L = []
    add = L.append
    f, _, src = get_incident_spectrum()
    add('=' * 74)
    add('設定')
    add('=' * 74)
    add(f'  経験式          : Fig. 9.54  eps\' = {EMP_EPS_BASE}^rho, '
        f'tan_delta = 10^({EMP_TAND_A} S + {EMP_TAND_B} rho - {EMP_TAND_C})、{F_ANCHOR/1e6:.0f} MHz で合わせる')
    add(f'  FeO+TiO2 (S)    : {FEOTIO2_WT} wt%,  S_BG = {S_BG} wt%,  DISP_MODE = {DISP_MODE}')
    add(f'  緩和の形        : Boivin+2022 {BOIVIN_FIT_SAMPLE}、ホストの損失 '
        f'{"を除く" if BOIVIN_SUBTRACT_HOST else "を含む"} ({BOIVIN_HOST_EPS_IM})')
    add(f'  平坦な部分      : 最大平坦 2 極 Debye ({FLAT_LO/1e9:g}-{FLAT_HI/1e9:g} GHz)'
        if USE_DEBYE_REALIZATION else '  平坦な部分      : 理想値（周波数によらない）')
    add(f'  氷              : eps\' {EPS_ICE}, tan_delta {TAND_ICE:g}, rho {RHO_ICE}')
    add(f'  入射スペクトル  : {src}, f_c = {incident_centroid()/1e9:.4f} GHz')
    add(f'  LSR に使う周波数: {f[lsr_mask()].min()/1e9:.3f}-{f[lsr_mask()].max()/1e9:.3f} GHz')
    add('')
    add('=' * 74)
    add('検算 1: Boivin の緩和の当てはめ')
    add('=' * 74)
    for k, (de, tau) in enumerate(RELAX_POLES):
        add(f'  極 {k+1}: De = {de:.5f}, tau = {tau*1e12:.2f} ps (f = {1/(2*np.pi*tau)/1e9:.3f} GHz)')
    add(f'  eps\'\' の RMS 誤差: 帯域 {RELAX_FIT["rms_band_pct"]:.2f} %, '
        f'全範囲 {RELAX_FIT["rms_all_pct"]:.2f} %')
    add('  2 成分（ホスト + 緩和）で各試料の形を再現できるか（450 MHz で割った eps\'\'）')
    add('    試料        w(450MHz)    1.0 GHz            1.25 GHz           2.0 GHz')
    for key in BOIVIN_CC:
        _, im_a = cole_cole_total(F_ANCHOR, key)
        im_a = float(np.atleast_1d(im_a)[0])
        w_s = 1 - BOIVIN_HOST_EPS_IM / im_a
        cells = []
        for fq in (1.0e9, 1.25e9, 2.0e9):
            meas = float(np.atleast_1d(cole_cole_total(fq, key)[1])[0]) / im_a
            pred = (BOIVIN_HOST_EPS_IM + (im_a - BOIVIN_HOST_EPS_IM)
                    * float(debye_sum(fq, RELAX_POLES)[1][0]) / RELAX_IM_A) / im_a
            cells.append(f'{meas:.3f} vs {pred:.3f}')
        add(f'    {key:10s}  {w_s:6.3f}     ' + '   '.join(cells))
    add('    （左: Cole-Cole、右: 2 成分モデル）')
    add('')
    add('=' * 74)
    add('検算 2: 450 MHz で経験式に合っているか（0 vol%）')
    add('=' * 74)
    rho = density_profile(z)
    for S in DRY_PLOT_WTS:
        er, ei = dry_eps(z, np.array([F_ANCHOR]), S)
        e1 = np.max(np.abs(er[:, 0] / emp_eps_real(rho) - 1)) * 100
        e2 = np.max(np.abs(ei[:, 0] / er[:, 0] / emp_tandelta(rho, S) - 1)) * 100
        add(f'  S = {S:5.1f} wt% (w = {float(relax_fraction(S)):.3f}): '
            f'eps\' 最大誤差 {e1:.2e} %, tan_delta 最大誤差 {e2:.2f} %')
    add(f'  平坦な部分は 450 MHz で合わせているので、帯域内では最大 +{FLAT_RIPPLE*100:.1f} % × (1-w) 大きい'
      f'（{_F0_FLAT/1e9:.3f} GHz）')
    add('')
    add('=' * 74)
    add(f'検算 3: 深さ {DRY_PLOT_DEPTH} m、0 vol% の周波数特性（S = {FEOTIO2_WT} wt%）')
    add('=' * 74)
    fq = np.array([F_ANCHOR, 0.5e9, 1.0e9, 1.25e9, 2.0e9])
    for mode in DISP_MODES:
        er, ei = dry_eps(np.array([DRY_PLOT_DEPTH]), fq, FEOTIO2_WT, mode)
        td = ei[0] / er[0]
        add(f'  {mode:6s}: tan_delta ' + ', '.join(f'{x/1e9:.2f} GHz {t:.5f}'
                                                    for x, t in zip(fq, td)))
        add(f'          2.0/0.5 GHz 比 {td[4]/td[1]:.3f}, eps\' の変化 '
            f'{(er[0][4]-er[0][0])/er[0][0]*100:+.2f} % (450 MHz -> 2 GHz)')
    add('')
    add('=' * 74)
    add('検算 4: 分散なしなら LSR の推定値が平均の tan_delta と一致するか')
    add('=' * 74)
    keep = USE_DEBYE_REALIZATION
    for real in (True, False):
        USE_DEBYE_REALIZATION = real
        _prop_cache.clear()
        a = lsr_tandelta(0, 'pore', *LSR_SEG, FEOTIO2_WT, 'none')
        b = model_tandelta(0, 'pore', *LSR_SEG, MODEL_TAND_FREQ, FEOTIO2_WT, 'none')
        add(f'  平坦な部分 {"2 極 Debye" if real else "理想値   "}  {LSR_SEG[0]}-{LSR_SEG[1]} m: '
            f'LSR {a:.5f}, {MODEL_TAND_FREQ/1e9:g} GHz の平均 {b:.5f}, 比 {a/b:.4f}')
    USE_DEBYE_REALIZATION = keep
    _prop_cache.clear()
    add('  理想値で 1 に近ければ推定式は正しい。2 極 Debye での差は平坦な部分の実現誤差')
    add('')
    add('=' * 74)
    add('検算 5: 空隙に入る氷の上限（pore）')
    add('=' * 74)
    lim = max_ice_volpct(z, 'pore')
    for c in ice_contents:
        bad = np.where(c > lim)[0]
        if bad.size:
            add(f'  ** {c} vol% は深さ {z[bad[0]]:.2f} m 以深で空隙率を超える **')
    add(f'  空隙率 {porosity(rho).min()*100:.1f}-{porosity(rho).max()*100:.1f} %')
    add('')
    add('=' * 74)
    add(f'gprMax の書式（乾燥、S = {FEOTIO2_WT} wt%, {DISP_MODE}）')
    add('=' * 74)
    L.extend('  ' + s for s in gprmax_lines)
    text = '\n'.join(L)
    print(text)
    with open(os.path.join(out_dir(), 'summary.txt'), 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


def summarize_lsr(model, L, res):
    """描像ごとの LSR の要約（summary_<model>.txt）。"""
    z1, z2 = LSR_SEG
    i2 = int(np.argmin(np.abs(L['z'] - z2)))
    out = [f'描像 {model}、区間 {LSR_REF_DEPTH}-{z2} m、S = {FEOTIO2_WT} wt%, {DISP_MODE}']
    out.append('  氷 [vol%]   LSR tan_delta   1 GHz 平均     比      Δ(LSR)      Δ(1 GHz)')
    for c in ice_contents:
        a, b = L[('lsr', c)][i2], L[('mod', c)][i2]
        out.append('  {:>5}       {:.5f}        {:.5f}      {:.3f}   {:+.6f}   {:+.6f}'.format(
            c, a, b, a / b, a - L[('lsr', 0)][i2], b - L[('mod', 0)][i2]))
    out.append('')
    out.append(f'組成の比較（区間 {z1}-{z2} m、氷 {LSR_ICE_FOR_COMP} vol%）')
    out.append('  S [wt%]  ' + '  '.join(f'{m:>22s}' for m in DISP_MODES))
    out.append('           ' + '  '.join(f'{"dry LSR / ΔLSR":>22s}' for _ in DISP_MODES))
    for S in (7.5, 10.0, 15.0, 20.0, 25.0):
        j = int(np.argmin(np.abs(S_GRID - S)))
        cells = [f'{res[m]["lsr0"][j]:.5f} / {res[m]["lsrc"][j]-res[m]["lsr0"][j]:+.6f}'
                 for m in DISP_MODES]
        out.append(f'  {S:6.1f}   ' + '  '.join(f'{c:>22s}' for c in cells))
    text = '\n'.join(out)
    print(text)
    with open(os.path.join(out_dir(model), f'summary_{model}.txt'), 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


# =============================================================================
# 8. 実行
# =============================================================================
def main():
    made = [plot_boivin_check(), plot_dry_dispersion()]
    pole_csv, gl = write_gprmax_poles()
    made.append(pole_csv)
    run_checks(gl)
    freq_labels = [f'{f/1e9:g} GHz' for f in PROFILE_FREQS]
    wt_labels = [f'{S:g} wt%' for S in PROFILE_WTS]
    for model in ICE_MODELS:
        print('\n' + '=' * 74 + f'\n描像: {model}\n' + '=' * 74)
        d_f = profile_set(model, [(FEOTIO2_WT, f) for f in PROFILE_FREQS])
        made.append(plot_profiles(model, d_f, freq_labels, 'profiles_freq', FEOTIO2_WT))
        made.append(write_profile_csv(model, d_f, freq_labels, 'profiles_freq.csv'))
        d_c = profile_set(model, [(S, PROFILE_FIXED_FREQ) for S in PROFILE_WTS])
        made.append(plot_profiles(model, d_c, wt_labels, 'profiles_comp', PROFILE_WTS))
        made.append(write_profile_csv(model, d_c, wt_labels, 'profiles_comp.csv'))
        made.append(plot_lsr_spectra(model))
        L = lsr_profiles(model)
        made.append(plot_lsr_profile(model, L))
        res = lsr_vs_composition(model)
        made.append(plot_lsr_vs_composition(model, res))
        made += write_lsr_csv(model, L, res)
        summarize_lsr(model, L, res)
        _prop_cache.clear()
    print(f'\n出力先: {OUTPUT_BASE}（{len(made)} 件）')


if __name__ == '__main__':
    main()
