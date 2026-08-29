"""レゴリス＋水氷の誘電プロファイルとスペクトル変化の見積り

calc_mixing_dispersion_profile.py の後継。設計書 design_calc_profile_lll.md 参照。

旧版からの主な変更:
  * 混合則      Maxwell-Garnett -> LLL 増分形（3 相：粒子・真空・氷）
  * 分散の扱い  450 MHz アンカーの 2 極 Debye -> eps'' 一定（最大平坦解析解）
  * 経験式      Carrier Fig. 9.54 (450 MHz) -> Fig. 9.53 (SOILS)
  * 帯域        0.25-6.0 GHz -> 0.5-2.0 GHz
  * Hilbert / STFT / B-scan 経験値の各解析は移植していない

参考文献:
  [1] Carrier, Olhoeft & Mendell (1991), "Physical Properties of the Lunar
      Surface", in Lunar Sourcebook, Cambridge Univ. Press, pp.475-594.
      Fig. 9.53 (SOILS) の図中回帰式:
        eps'      = 1.871^rho
        tan_delta = 10^(0.027*(%TiO2+%FeO) + 0.273*rho - 3.058)
      密度プロファイル: rho(z) = 1.92(z+12.2)/(z+18)  [z: cm]
  [2] Boivin, A. et al. (2022)
      低イルメナイト試料が P/L/S/X 帯で eps'' 一定であることの実測根拠。
  [3] Looyenga, H. (1965) Physica 31, 401-406. / Landau & Lifshitz
      LLL 混合則 eps^(1/3) = sum_i v_i eps_i^(1/3)
  [4] Takekura, Miyamoto & Kobayashi (2025), Remote Sensing 17, 1050.
      水氷が空隙に凝結するという 3 相の描像（Fig. 2）と Eq.(6)-(9)。
  [5] Fujita et al. (2000) / Matsuoka et al. (1996)  氷の複素誘電率
  [6] Ghormley & Hochanadel (1971) Science 171, 62-64.  82-110 K の氷の密度
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:                      # numpy 2.x
    trapz = np.trapezoid
except AttributeError:    # numpy 1.x
    trapz = np.trapz

C0 = 299792458.0          # [m/s] 真空中の光速
EPS0 = 8.8541878128e-12   # [F/m] 真空の誘電率

# =============================================================================
# 0. 設定  [EDIT HERE]
# =============================================================================
OUTPUT_BASE = ('/Volumes/SSD_Kanda_BUFFALO/test_programs_output/'
               'LLL_mixing_profile')
DIR_FREQ = 'profile_diff_frequency'      # 周波数を振ったプロファイル
DIR_COMP = 'profile_diff_FeO+TiO2'       # 組成を振ったプロファイル
DIR_SPEC = 'spectrum'                    # スペクトル解析

# --- 深さ格子 ---------------------------------------------------------------
Z_MAX = 3.0               # [m] 計算領域の地下部分の厚さに合わせる
DZ    = 0.01              # [m] 深さ刻み
z     = np.arange(0.0, Z_MAX + DZ / 2, DZ)

# --- 組成と帯域 -------------------------------------------------------------
FEOTIO2_WT = 7.5          # [wt%] 既定の組成。スペクトル解析はこの値で行う。

BAND_LO, BAND_HI = 0.5e9, 2.0e9        # [Hz] LUPEX GPR 想定帯域
BAND_F0 = np.sqrt(BAND_LO * BAND_HI)   # [Hz] 帯域の幾何平均 = 1.0 GHz
                                       # 解析解の対称中心かつ eps' の基準周波数

# --- プロファイル図の 2 系統 -------------------------------------------------
# (A) 周波数を振る系統：組成を固定して 3 周波数を比較する。
#     目的は「周波数分散が出ていないこと」の確認。eps'' 一定モデルが
#     正しく効いていれば、eps'' の 3 本はほぼ重なり、alpha だけが f に
#     比例して開く（帯域内比 4.0）。
PROFILE_FREQS = np.array([0.5e9, 1.0e9, 2.0e9])
PROFILE_FREQ_LABELS = ['0.5 GHz', '1.0 GHz', '2.0 GHz']

# (B) 組成を振る系統：周波数を固定して 3 組成を比較する。
#     目的は FeO+TiO2 濃度が各量に与える影響の把握。
#     eps' は組成に依らないので、差が出るのは eps'' / sigma / tan_delta /
#     alpha の 4 つだけになるはず。
PROFILE_WTS = [5.0, 7.5, 10.0]
PROFILE_WT_LABELS = ['5.0 wt%', '7.5 wt%', '10.0 wt%']
PROFILE_FIXED_FREQ = 1.0e9        # [Hz] 組成系統で固定する周波数（帯域の幾何平均）

STYLE_SET = ['-', '--', '-.']     # 両系統で共通に使う線種

# --- 氷 ---------------------------------------------------------------------
ice_contents = [0, 1, 5, 10, 20]       # [vol%]
ice_colors   = ['k', 'r', 'g', 'b', 'c']
ice_labels   = [f'{c} vol% ice' for c in ice_contents]

EPS_ICE   = 3.15          # [5] GHz 帯の氷の eps'。低温での温度依存は小さい。
TAND_ICE  = 2.0e-4        # [要文献確認] 低温ほど小さくなるので保守側（検出しにくい側）
RHO_ICE   = 0.94          # [6] [g/cm^3] 82-110 K での氷の密度。wt% 換算に使う。
RHO_GRAIN = 2.645         # [g/cm^3] 斜長岩の粒子密度 [4]。
                          # 【用途は空隙率チェックのみ】混合則の式には現れない。

# --- モデルの切替 -----------------------------------------------------------
USE_DEBYE_REALIZATION = True
# True : Level_3.in と同じ最大平坦 2 極 Debye で eps'' 一定を実現する。
#        gprMax が実際に解く媒質と一致する。eps' は KK により 0.43% 変化。
# False: eps'・eps'' とも周波数に依らない理想モデル。差は alpha で最大 2.5%。

PROPAGATION_MODE = 'two_way'   # 'two_way' : 地表 tx/rx の往復（B 系統）
                               # 'one_way' : 埋設 rx への片道（A 系統）
ANTENNA_HEIGHT = 0.35          # [m] 地表面からの tx 高さ（空中走時のオフセット用）

# --- 入射スペクトル ---------------------------------------------------------
ASCAN_OUTFILE_PATH = ''   # gprMax の .out があれば指定。空なら合成波形を使う。
SYNTH_TUKEY_ALPHA = 0.2   # 合成スペクトルのテーパ。0.2 で sigma_f^2 = 0.1443 GHz^2
                          # となり実際の励振ファイルの実測値と一致する。
SYNTH_NFREQ = 3001        # 合成スペクトルの周波数点数

# --- スペクトル解析 ---------------------------------------------------------
SPECTRUM_TARGET_DEPTHS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]   # [m]
FLOHI_THRESHOLD_DB = -10.0    # 帯域端 f_lo / f_hi のしきい値
LSR_REF_DEPTH = 0.25          # [m] 相対 LSR の基準深さ

FIGURE_DPI = 200


# =============================================================================
# 1. 乾燥レゴリスの誘電モデル（Carrier et al. 1991, Fig. 9.53 SOILS）
# =============================================================================
CARRIER_EPS_BASE = 1.871      # Fig. 9.53 図中: eps' = 1.871^rho
CARRIER_TAND_A   = 0.027      # Fig. 9.53 図中の 3 次元回帰
CARRIER_TAND_B   = 0.273
CARRIER_TAND_C   = 3.058


def density_profile(depth_m):
    """深さ [m] -> バルク密度 [g/cm^3]。Carrier et al. 1991。"""
    z_cm = np.asarray(depth_m, dtype=float) * 100.0
    return 1.92 * (z_cm + 12.2) / (z_cm + 18.0)


def carrier_eps_real(rho_val):
    """乾燥レゴリスの eps'。eps' = 1.871^rho（Fig. 9.53）。"""
    return CARRIER_EPS_BASE ** np.asarray(rho_val, dtype=float)


def carrier_tandelta(rho_val, feotio2_wt=None):
    """乾燥レゴリスの tan_delta（Fig. 9.53）。周波数を説明変数に持たない。"""
    wt = FEOTIO2_WT if feotio2_wt is None else feotio2_wt
    return 10.0 ** (CARRIER_TAND_A * wt
                    + CARRIER_TAND_B * np.asarray(rho_val, dtype=float)
                    - CARRIER_TAND_C)


def porosity(rho_val):
    """空隙率。氷量の上限チェックにのみ使う。"""
    return 1.0 - np.asarray(rho_val, dtype=float) / RHO_GRAIN


# =============================================================================
# 2. eps'' 一定の実現（最大平坦 2 極 Debye の解析解）
# =============================================================================
# 1 極 Debye の eps'' は対数周波数 u = ln(w*tau) で sech 関数になる。
# u = ±s に等強度で 2 極置くと u=0 で最大平坦になる条件が
#     s = arcsinh(1) = ln(1+sqrt2)      -> tau 比 = (1+sqrt2)^2 = 5.8284
# と閉形式で決まり、対称中心を帯域の幾何平均 f0 に取ると
#     1/(1+(w0*tau1)^2) + 1/(1+(w0*tau2)^2) = 1   （厳密に 1）
# が成り立つので eps_inf も閉形式になる。数値最適化は不要。
# 詳細は Level_3.in の section 2 を参照。
# =============================================================================
S_FLAT    = np.arcsinh(1.0)      # = ln(1+sqrt2) = 0.881374
TAU_RATIO = np.exp(S_FLAT)       # = 1+sqrt2 = 2.414214
_W0 = 2.0 * np.pi * BAND_F0
TAU1 = 1.0 / (_W0 * TAU_RATIO)
TAU2 = TAU_RATIO / _W0


def dry_eps_complex(depth_m, freq_hz, feotio2_wt=None):
    """乾燥レゴリスの (eps', eps'') を返す。

    depth_m: (Nz,) / freq_hz: (Nf,)  -> それぞれ (Nz, Nf)
    """
    d = np.atleast_1d(np.asarray(depth_m, dtype=float))
    f = np.atleast_1d(np.asarray(freq_hz, dtype=float))
    rho_d = density_profile(d)
    eps_re_t = carrier_eps_real(rho_d)[:, None]           # (Nz,1) 目標 eps'
    eps_im_t = (eps_re_t
                * carrier_tandelta(rho_d, feotio2_wt)[:, None])   # (Nz,1)

    if not USE_DEBYE_REALIZATION:
        return (np.broadcast_to(eps_re_t, (d.size, f.size)).copy(),
                np.broadcast_to(eps_im_t, (d.size, f.size)).copy())

    de = np.sqrt(2.0) * eps_im_t                          # (Nz,1) 各極の De
    eps_inf = eps_re_t - de                               # (Nz,1)
    w = (2.0 * np.pi * f)[None, :]                        # (1,Nf)
    x1, x2 = w * TAU1, w * TAU2
    eps_re = eps_inf + de / (1.0 + x1 ** 2) + de / (1.0 + x2 ** 2)
    eps_im = de * x1 / (1.0 + x1 ** 2) + de * x2 / (1.0 + x2 ** 2)
    return eps_re, eps_im


# =============================================================================
# 3. 水氷の混合（LLL 増分形）
# =============================================================================
# 氷はレゴリス粒子を置き換えるのではなく、粒子間の空隙（真空）を埋める [4]。
# LLL を 3 相に適用すると
#     eps_wet^(1/3) = v_grain*eps_grain^(1/3) + v_ice*eps_ice^(1/3) + v_void
# 乾燥時は
#     eps_dry^(1/3) = v_grain*eps_grain^(1/3) + (v_void + v_ice)
# なので差を取ると粒子の項が消えて増分形になる。
# この形なら粒子密度も粒子誘電率も式に現れず、乾燥側に Carrier の経験式を
# そのまま使えるので経験式との接続が完全に保たれる。
#
# 損失は粒子にあり氷はほぼ無損失なので eps'' は保存し、氷自身の微小な損失
# だけを加える。虚部に混合則を適用してはならない（損失源が増えていないのに
# 減衰が増えるという非物理的な結果になる）。
# =============================================================================
_ICE_INCREMENT = EPS_ICE ** (1.0 / 3.0) - 1.0     # = 0.46573


def mix_ice(eps_re_dry, eps_im_dry, ice_volpct):
    """乾燥レゴリスに氷を加えた (eps', eps'') を返す。"""
    v = float(ice_volpct) / 100.0
    if v == 0.0:
        return eps_re_dry, eps_im_dry
    eps_re = (eps_re_dry ** (1.0 / 3.0) + v * _ICE_INCREMENT) ** 3
    eps_im = eps_im_dry + v * EPS_ICE * TAND_ICE
    return eps_re, eps_im


def medium_eps(depth_m, freq_hz, ice_volpct, feotio2_wt=None):
    """深さ・周波数・氷量に対する (eps', eps'')。形は (Nz, Nf)。"""
    er, ei = dry_eps_complex(depth_m, freq_hz, feotio2_wt)
    return mix_ice(er, ei, ice_volpct)


def alpha_velocity(depth_m, freq_hz, ice_volpct, feotio2_wt=None):
    """減衰係数 alpha [Np/m] と位相速度 v [m/s]。形は (Nz, Nf)。

        alpha = (omega/c) * sqrt(eps'/2) * sqrt(sqrt(1+tan_delta^2) - 1)
        n     = sqrt(eps'/2) * sqrt(sqrt(1+tan_delta^2) + 1)
    """
    er, ei = medium_eps(depth_m, freq_hz, ice_volpct, feotio2_wt)
    td = ei / er
    w = (2.0 * np.pi * np.atleast_1d(np.asarray(freq_hz, dtype=float)))[None, :]
    root = np.sqrt(1.0 + td ** 2)
    alpha = (w / C0) * np.sqrt(er / 2.0) * np.sqrt(root - 1.0)
    n_re = np.sqrt(er / 2.0) * np.sqrt(root + 1.0)
    return alpha, C0 / n_re


# =============================================================================
# 4. 入射スペクトル
# =============================================================================
_incident_cache = None


def _synthetic_spectrum():
    """0.5-2.0 GHz で平坦・Tukey テーパの合成振幅スペクトル。

    Tukey alpha=0.2 で sigma_f^2 = 0.1443 GHz^2、f_c = 1.2500 GHz となり、
    実際の励振ファイル（lupex_src.txt）の実測値と一致することを確認済み。
    """
    from scipy.signal.windows import tukey
    f = np.linspace(BAND_LO * 0.8, BAND_HI * 1.05, SYNTH_NFREQ)
    band = (f >= BAND_LO) & (f <= BAND_HI)
    amp = np.zeros_like(f)
    amp[band] = tukey(int(band.sum()), alpha=SYNTH_TUKEY_ALPHA)
    return f, amp


def get_incident_spectrum():
    """(freq_hz, |S0(f)|) を返す。帯域内のみ。"""
    global _incident_cache
    if _incident_cache is not None:
        return _incident_cache

    src = 'synthetic'
    if ASCAN_OUTFILE_PATH and os.path.exists(ASCAN_OUTFILE_PATH):
        try:
            from tools.core.outputfiles_merge import get_output_data
            data, dt = get_output_data(ASCAN_OUTFILE_PATH, 1, 'Ez')
            e = data if data.ndim == 1 else data[:, 0]
            f_all = np.fft.rfftfreq(len(e), d=dt)
            s_all = np.abs(np.fft.rfft(e))
            m = (f_all >= BAND_LO) & (f_all <= BAND_HI)
            f, amp = f_all[m], s_all[m]
            src = ASCAN_OUTFILE_PATH
        except Exception as exc:                       # noqa: BLE001
            print(f'  [warn] .out の読み込みに失敗したため合成波形を使う: {exc}')
            f, amp = _synthetic_spectrum()
    else:
        f, amp = _synthetic_spectrum()

    m = (f >= BAND_LO) & (f <= BAND_HI)
    _incident_cache = (f[m], amp[m], src)
    return _incident_cache


# =============================================================================
# 5. 伝搬（累積減衰・累積走時）
# =============================================================================
_prop_cache = {}


def propagation_table(ice_volpct):
    """累積減衰 A(f,d) [Np] と累積走時 T(f,d) [s] を返す。形は (Nz, Nf)。"""
    key = (ice_volpct, PROPAGATION_MODE, USE_DEBYE_REALIZATION)
    if key in _prop_cache:
        return _prop_cache[key]

    f, _, _ = get_incident_spectrum()
    k = 2.0 if PROPAGATION_MODE == 'two_way' else 1.0
    alpha, v = alpha_velocity(z, f, ice_volpct)          # (Nz, Nf)

    # 台形則で深さ方向に累積
    cum_a = np.zeros_like(alpha)
    cum_t = np.zeros_like(alpha)
    cum_a[1:] = np.cumsum(0.5 * (alpha[1:] + alpha[:-1]) * DZ, axis=0)
    cum_t[1:] = np.cumsum(0.5 * (1.0 / v[1:] + 1.0 / v[:-1]) * DZ, axis=0)
    t_air = k * ANTENNA_HEIGHT / C0

    _prop_cache[key] = (k * cum_a, k * cum_t + t_air)
    return _prop_cache[key]


def spectrum_at_depth(ice_volpct):
    """各深さでの振幅スペクトル |S(f,d)| を返す。形は (Nz, Nf)。"""
    f, s0, _ = get_incident_spectrum()
    cum_a, _ = propagation_table(ice_volpct)
    return s0[None, :] * np.exp(-cum_a)


# =============================================================================
# 6. スペクトル解析量
# =============================================================================
# 規約: モーメント（f_c, sigma_f）はパワースペクトル |S|^2 基準、
#       LSR は振幅基準 ln|S|。ascan_spectrum.py と揃えてある。
#       感度の恒等式 df_c/dt = -2*pi*tan_delta*sigma_f^2 はパワー基準・
#       往復走時で成立する。
# =============================================================================
_moments_cache = {}


def spectral_moments(ice_volpct):
    """(f_c, sigma_f, B_eff, f_lo, f_hi) を深さプロファイルで返す。単位 Hz。"""
    key = (ice_volpct, PROPAGATION_MODE, USE_DEBYE_REALIZATION)
    if key in _moments_cache:
        return _moments_cache[key]

    f, _, _ = get_incident_spectrum()
    P = spectrum_at_depth(ice_volpct) ** 2               # (Nz, Nf)

    I0 = trapz(P, f, axis=1)
    fc = trapz(f[None, :] * P, f, axis=1) / I0
    var = trapz((f[None, :] - fc[:, None]) ** 2 * P, f, axis=1) / I0
    sigma = np.sqrt(var)
    B_eff = I0 ** 2 / trapz(P ** 2, f, axis=1)

    # 帯域端: 帯域内最大に対する相対値がしきい値を横切る周波数
    thr = 10.0 ** (FLOHI_THRESHOLD_DB / 10.0)
    f_lo = np.full(len(z), np.nan)
    f_hi = np.full(len(z), np.nan)
    for i in range(len(z)):
        rel = P[i] / P[i].max()
        above = np.where(rel >= thr)[0]
        if above.size >= 2:
            f_lo[i], f_hi[i] = f[above[0]], f[above[-1]]

    _moments_cache[key] = (fc, sigma, B_eff, f_lo, f_hi)
    return _moments_cache[key]


def lsr_profile(ice_volpct, ref_depth=None):
    """LSR(f, d) = ln(|S(f,d)| / |S(f,d_ref)|)。形は (Nz, Nf)。

    振幅基準。幾何減衰を含まないので純粋な吸収項になり、
    理論の傾きは -k * d/df ∫alpha dz（alpha ∝ f なので直線）。
    """
    cum_a, _ = propagation_table(ice_volpct)
    if ref_depth is None:
        return -cum_a                       # 絶対 LSR（地表基準）
    i_ref = int(np.argmin(np.abs(z - ref_depth)))
    return -(cum_a - cum_a[i_ref][None, :])


def travel_time_at_fc(ice_volpct):
    """重心周波数における走時 [ns] の深さプロファイル。"""
    f, _, _ = get_incident_spectrum()
    _, cum_t = propagation_table(ice_volpct)
    fc, _, _, _, _ = spectral_moments(ice_volpct)
    return np.array([np.interp(fc[i], f, cum_t[i]) for i in range(len(z))]) * 1e9


# =============================================================================
# 7. プロファイル配列（2 系統ぶん構成する）
# =============================================================================
# 1 つの「系統」は (組成, 周波数) の組を n_style 個並べたもの。
#   系統 A（周波数を振る）: 組成を FEOTIO2_WT に固定し 3 周波数
#   系統 B（組成を振る）  : 周波数を PROFILE_FIXED_FREQ に固定し 3 組成
# どちらも氷量が色、系統内の並びが線種になる。
# =============================================================================
n_ice, Nz = len(ice_contents), len(z)
rho = density_profile(z)


def build_profile_set(pairs):
    """[(wt, freq), ...] に対する各量のプロファイルを返す。

    戻り値の各配列は (n_ice, n_style, Nz)。
    """
    n_style = len(pairs)
    out = {k: np.zeros((n_ice, n_style, Nz))
           for k in ('eps_re', 'eps_im', 'sigma', 'tand', 'alpha')}
    for si, (wt, f) in enumerate(pairs):
        fa = np.array([f])
        for ii, c in enumerate(ice_contents):
            er, ei = medium_eps(z, fa, c, wt)
            al, _ = alpha_velocity(z, fa, c, wt)
            out['eps_re'][ii, si] = er[:, 0]
            out['eps_im'][ii, si] = ei[:, 0]
            out['sigma'][ii, si] = ei[:, 0] * (2 * np.pi * f) * EPS0
            out['tand'][ii, si] = ei[:, 0] / er[:, 0]
            out['alpha'][ii, si] = al[:, 0]
    return out


# 系統 A: 周波数を振る（組成固定）
PAIRS_FREQ = [(FEOTIO2_WT, f) for f in PROFILE_FREQS]
SET_FREQ = build_profile_set(PAIRS_FREQ)
# 参照曲線（乾燥）。eps' も tan_delta も組成固定なので 1 本ずつ。
REF_FREQ = {
    'eps_re': carrier_eps_real(rho),
    'tand': carrier_tandelta(rho, FEOTIO2_WT),
}

# 系統 B: 組成を振る（周波数固定）
PAIRS_COMP = [(wt, PROFILE_FIXED_FREQ) for wt in PROFILE_WTS]
SET_COMP = build_profile_set(PAIRS_COMP)
# eps' は組成に依らないので 1 本。tan_delta は組成ごとに 3 本。
REF_COMP = {
    'eps_re': carrier_eps_real(rho),
    'tand': np.array([carrier_tandelta(rho, wt) for wt in PROFILE_WTS]),
}


# =============================================================================
# 8. 作図ヘルパ
# =============================================================================
def _ensure_dirs():
    for sub in (DIR_FREQ, DIR_COMP, DIR_SPEC):
        os.makedirs(os.path.join(OUTPUT_BASE, sub), exist_ok=True)


def save_fig(fig, base_path):
    """PNG と PDF の両方を保存する。"""
    fig.savefig(base_path + '.png', bbox_inches='tight', dpi=FIGURE_DPI)
    fig.savefig(base_path + '.pdf', bbox_inches='tight')
    plt.close(fig)
    return base_path


def draw_lines(ax, data, ref=None, ref_label='Carrier (dry)'):
    """data: (n_ice, n_style, Nz)。ref は (Nz,) か (n_style, Nz)。"""
    if ref is not None:
        ref_arr = np.atleast_2d(ref)
        for si in range(ref_arr.shape[0]):
            ls = STYLE_SET[si] if ref_arr.shape[0] > 1 else '--'
            ax.plot(ref_arr[si], z, color='gray', ls=ls, lw=2, zorder=1,
                    label=ref_label if si == 0 else None)
    n_style = data.shape[1]
    for ii in range(n_ice):
        for si in range(n_style):
            ax.plot(data[ii, si], z, color=ice_colors[ii],
                    linestyle=STYLE_SET[si], lw=1.6, zorder=3 + ii)
    ax.set_ylabel('Depth [m]', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.minorticks_on()
    ax.grid(True, alpha=0.4)
    ax.invert_yaxis()


ice_handles = [Line2D([0], [0], color=ice_colors[i], ls='-', lw=2,
                      label=ice_labels[i]) for i in range(n_ice)]
carrier_handle = [Line2D([0], [0], color='gray', ls='--', lw=2,
                         label='Carrier (dry)')]


def style_handles(labels):
    return [Line2D([0], [0], ls=STYLE_SET[i], color='k', lw=2, label=labels[i])
            for i in range(len(labels))]


def add_legend(fig, labels, with_ref=True):
    """凡例は図の下に置く。上はタイトル用に空けておく。"""
    handles = style_handles(labels) + ice_handles + (
        carrier_handle if with_ref else [])
    fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=12,
               frameon=True, bbox_to_anchor=(0.5, 0.0))


def style_depth_axis(ax, xlabel, logx=False):
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel('Depth [m]', fontsize=16)
    if logx:
        ax.set_xscale('log')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.minorticks_on()
    ax.grid(True, alpha=0.4)
    ax.invert_yaxis()


# =============================================================================
# 9. プロファイル図（両系統に対して同じものを作る）
# =============================================================================
def make_summary_2x2(dataset, ref, labels, subdir, title):
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    draw_lines(axes[0, 0], dataset['eps_re'], ref=ref.get('eps_re'))
    axes[0, 0].set_xlabel(r"$\varepsilon^{\prime}$", fontsize=16)
    draw_lines(axes[0, 1], dataset['eps_im'])
    axes[0, 1].set_xlabel(r"$\varepsilon^{\prime\prime}$", fontsize=16)
    draw_lines(axes[1, 0], dataset['sigma'])
    axes[1, 0].set_xlabel(r"$\sigma_{\rm eff}$ [S/m]", fontsize=16)
    draw_lines(axes[1, 1], dataset['tand'], ref=ref.get('tand'))
    axes[1, 1].set_xlabel(r"$\tan\delta$", fontsize=16)
    axes[1, 1].locator_params(axis='x', nbins=5)
    fig.suptitle(title, fontsize=15, y=1.0)
    add_legend(fig, labels)
    plt.tight_layout()
    return save_fig(fig, os.path.join(OUTPUT_BASE, subdir, 'summary_2x2'))


def make_profile_and_delta(data, label, fname, labels, subdir, title,
                           ref=None):
    """左: プロファイル本体、右: 0 vol% からの相対差 [%]。"""
    base0 = data[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    draw_lines(axes[0], data, ref=ref)
    axes[0].set_xlabel(label, fontsize=16)
    if fname == 'losstangent':
        axes[0].locator_params(axis='x', nbins=5)
    for ii, c in enumerate(ice_contents):
        if c == 0:
            continue
        for si in range(data.shape[1]):
            rel = np.abs(data[ii, si] - base0[si]) / base0[si] * 100.0
            axes[1].plot(rel, z, color=ice_colors[ii], ls=STYLE_SET[si],
                         lw=1.6, zorder=3 + ii)
    style_depth_axis(axes[1],
                     r'$|X_{0\%} - X| / X_{0\%} \times 100$ [%]', logx=True)
    fig.suptitle(title, fontsize=15, y=1.02)
    add_legend(fig, labels, with_ref=ref is not None)
    plt.tight_layout()
    return save_fig(fig, os.path.join(OUTPUT_BASE, subdir, fname))


def make_profile_family(dataset, ref, labels, subdir, title):
    """1 系統ぶんのプロファイル図をまとめて作る。"""
    made = [make_summary_2x2(dataset, ref, labels, subdir, title)]
    specs = [
        ('eps_re', r"$\varepsilon^{\prime}$", 'eps_real', ref.get('eps_re')),
        ('eps_im', r"$\varepsilon^{\prime\prime}$", 'eps_imag', None),
        ('sigma', r"$\sigma_{\rm eff}$ [S/m]", 'conductivity', None),
        ('tand', r"$\tan\delta$", 'losstangent', ref.get('tand')),
        ('alpha', r"$\alpha$ [Np/m]", 'attenuation', None),
    ]
    for key, lab, fname, r in specs:
        made.append(make_profile_and_delta(
            dataset[key], lab, fname, labels, subdir, title, ref=r))
    return made


def make_density_profile():
    """密度と空隙率。周波数にも組成にも依らないので基準ディレクトリに置く。"""
    fig, ax = plt.subplots(figsize=(5.5, 6))
    ax.plot(rho, z, color='k', lw=2)
    style_depth_axis(ax, r'$\rho$ [g/cm$^{3}$]')
    ax2 = ax.twiny()
    ax2.plot(porosity(rho) * 100.0, z, color='tab:orange', lw=2, ls='--')
    ax2.set_xlabel(r'Porosity [%]  ($\rho_{\rm grain}$ = '
                   + '{:.3f})'.format(RHO_GRAIN),
                   fontsize=14, color='tab:orange')
    ax2.tick_params(axis='x', colors='tab:orange', labelsize=12)
    plt.tight_layout()
    return save_fig(fig, os.path.join(OUTPUT_BASE, 'density_profile'))


def make_ice_wtpct_profile():
    """vol% 一定でも密度が深さ変化するので wt% は深さに依存する。"""
    fig, ax = plt.subplots(figsize=(5.5, 6))
    for ii, c in enumerate(ice_contents):
        if c == 0:
            continue
        v = c / 100.0
        wtpct = 100.0 * v * RHO_ICE / (v * RHO_ICE + rho)
        ax.plot(wtpct, z, color=ice_colors[ii], lw=2, label=f'{c} vol% ice')
    style_depth_axis(ax, 'Ice content [wt%]')
    ax.legend(loc='center right', fontsize=12)
    plt.tight_layout()
    return save_fig(fig, os.path.join(OUTPUT_BASE, 'ice_wtpct_profile'))


# =============================================================================
# 10. スペクトル解析図（組成は FEOTIO2_WT 固定）
# =============================================================================
def make_spectrum_evolution(ice_volpct):
    """深さごとのスペクトル形状の変化。"""
    f, s0, _ = get_incident_spectrum()
    S = spectrum_at_depth(ice_volpct)
    P0max = (s0 ** 2).max()
    fc, _, _, _, _ = spectral_moments(ice_volpct)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(SPECTRUM_TARGET_DEPTHS)))
    for i, d in enumerate(SPECTRUM_TARGET_DEPTHS):
        k = int(np.argmin(np.abs(z - d)))
        P_db = 10.0 * np.log10(S[k] ** 2 / P0max + 1e-30)
        ax.plot(f / 1e9, P_db, color=colors[i], lw=1.8,
                label=f'{d:.1f} m  ($f_c$={fc[k]/1e9:.3f} GHz)')
        ax.axvline(fc[k] / 1e9, color=colors[i], ls='--', alpha=0.6)
    ax.set_xlabel('Frequency [GHz]', fontsize=16)
    ax.set_ylabel('Normalized power [dB]', fontsize=16)
    ax.set_title(f'{ice_volpct} vol% ice, {FEOTIO2_WT} wt%, '
                 f'{PROPAGATION_MODE}', fontsize=14)
    ax.set_xlim(BAND_LO / 1e9, BAND_HI / 1e9)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.minorticks_on()
    ax.grid(True, alpha=0.4)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=11)
    plt.tight_layout()
    return save_fig(fig, os.path.join(OUTPUT_BASE, DIR_SPEC,
                                     f'spectrum_evolution_{ice_volpct}vol'))


def make_band_edges_profile():
    """f_lo / f_c / f_hi の深さプロファイル。"""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    st = {'f_lo': ':', 'f_c': '-', 'f_hi': '--'}
    for ii, c in enumerate(ice_contents):
        fc, _, _, f_lo, f_hi = spectral_moments(c)
        ax.plot(f_lo / 1e9, z, color=ice_colors[ii], ls=st['f_lo'], lw=1.6)
        ax.plot(fc / 1e9, z, color=ice_colors[ii], ls=st['f_c'], lw=2.0)
        ax.plot(f_hi / 1e9, z, color=ice_colors[ii], ls=st['f_hi'], lw=1.6)
    style_depth_axis(ax, 'Frequency [GHz]')
    ax.set_title(f'Band edges ({FLOHI_THRESHOLD_DB:.0f} dB) and centroid '
                 f'({FEOTIO2_WT} wt%)', fontsize=13)
    handles = ice_handles + [
        Line2D([0], [0], color='k', ls=st['f_lo'], lw=2, label=r'$f_{lo}$'),
        Line2D([0], [0], color='k', ls=st['f_c'], lw=2, label=r'$f_{c}$'),
        Line2D([0], [0], color='k', ls=st['f_hi'], lw=2, label=r'$f_{hi}$')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=12,
               frameon=True, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout()
    return save_fig(fig, os.path.join(OUTPUT_BASE, DIR_SPEC,
                                     'band_edges_profile'))


def make_centroid_width_profile():
    """左: f_c ± sigma_f、右: sigma_f 単独。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ii, c in enumerate(ice_contents):
        fc, sg, _, _, _ = spectral_moments(c)
        axes[0].plot(fc / 1e9, z, color=ice_colors[ii], lw=2)
        axes[0].fill_betweenx(z, (fc - sg) / 1e9, (fc + sg) / 1e9,
                              color=ice_colors[ii], alpha=0.12)
        axes[1].plot(sg / 1e9, z, color=ice_colors[ii], lw=2)
    style_depth_axis(axes[0], r'$f_c \pm \sigma_f$ [GHz]')
    style_depth_axis(axes[1], r'$\sigma_f$ [GHz]')
    fig.suptitle(f'{FEOTIO2_WT} wt%', fontsize=14, y=1.02)
    fig.legend(handles=ice_handles, loc='lower center', ncol=5, fontsize=12,
               frameon=True, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout()
    return save_fig(fig, os.path.join(OUTPUT_BASE, DIR_SPEC,
                                     'centroid_width_profile'))


def make_centroid_by_composition():
    """組成ごとの f_c と sigma_f（組成の影響をスペクトル側でも見る）。"""
    global FEOTIO2_WT
    keep = FEOTIO2_WT
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for si, wt in enumerate(PROFILE_WTS):
        FEOTIO2_WT = wt
        _prop_cache.clear()
        _moments_cache.clear()
        for ii, c in enumerate(ice_contents):
            fc, sg, _, _, _ = spectral_moments(c)
            axes[0].plot(fc / 1e9, z, color=ice_colors[ii],
                         ls=STYLE_SET[si], lw=1.8)
            axes[1].plot(sg / 1e9, z, color=ice_colors[ii],
                         ls=STYLE_SET[si], lw=1.8)
    FEOTIO2_WT = keep
    _prop_cache.clear()
    _moments_cache.clear()
    style_depth_axis(axes[0], r'$f_c$ [GHz]')
    style_depth_axis(axes[1], r'$\sigma_f$ [GHz]')
    fig.suptitle('Centroid and spectral width vs composition', fontsize=14,
                 y=1.02)
    fig.legend(handles=style_handles(PROFILE_WT_LABELS) + ice_handles,
               loc='lower center', ncol=4, fontsize=12, frameon=True,
               bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout()
    return save_fig(fig, os.path.join(OUTPUT_BASE, DIR_COMP,
                                     'centroid_width_profile'))


def make_lsr(ice_volpct):
    """左: 絶対 LSR、右: 相対 LSR（基準深さからの差）。"""
    f, _, _ = get_incident_spectrum()
    lsr_abs = lsr_profile(ice_volpct, None)
    lsr_rel = lsr_profile(ice_volpct, LSR_REF_DEPTH)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(SPECTRUM_TARGET_DEPTHS)))
    for i, d in enumerate(SPECTRUM_TARGET_DEPTHS):
        k = int(np.argmin(np.abs(z - d)))
        axes[0].plot(f / 1e9, lsr_abs[k], color=colors[i], lw=1.8,
                     label=f'{d:.1f} m')
        axes[1].plot(f / 1e9, lsr_rel[k], color=colors[i], lw=1.8)
    for ax, ttl in zip(axes, ['Absolute LSR (ref = surface)',
                              f'Relative LSR (ref = {LSR_REF_DEPTH:.2f} m)']):
        ax.set_xlabel('Frequency [GHz]', fontsize=16)
        ax.set_ylabel(r'$\ln(|S(f,d)| / |S(f,d_{ref})|)$', fontsize=14)
        ax.set_title(ttl, fontsize=13)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.minorticks_on()
        ax.grid(True, alpha=0.4)
    axes[0].legend(loc='lower left', fontsize=11,
                   title=f'{ice_volpct} vol% ice, {FEOTIO2_WT} wt%')
    plt.tight_layout()
    return save_fig(fig, os.path.join(OUTPUT_BASE, DIR_SPEC,
                                     f'lsr_{ice_volpct}vol'))


def make_lsr_slope_profile():
    """LSR の傾きから逆算した alpha と、直接計算した alpha の比較（検算図）。"""
    f, _, _ = get_incident_spectrum()
    k = 2.0 if PROPAGATION_MODE == 'two_way' else 1.0
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for ii, c in enumerate(ice_contents):
        lsr = lsr_profile(c, None)
        i_f = int(np.argmin(np.abs(f - BAND_F0)))
        with np.errstate(divide='ignore', invalid='ignore'):
            ax.plot(-lsr[:, i_f] / (k * z), z, color=ice_colors[ii], lw=2)
        a_direct, _ = alpha_velocity(z, np.array([BAND_F0]), c)
        cum = np.zeros(len(z))
        cum[1:] = np.cumsum(0.5 * (a_direct[1:, 0] + a_direct[:-1, 0]) * DZ)
        with np.errstate(divide='ignore', invalid='ignore'):
            ax.plot(cum / z, z, color=ice_colors[ii], ls=':', lw=2)
    style_depth_axis(ax, r'Path-averaged $\bar{\alpha}$ at 1.0 GHz [Np/m]')
    handles = ice_handles + [
        Line2D([0], [0], color='k', ls='-', lw=2, label='from LSR'),
        Line2D([0], [0], color='k', ls=':', lw=2, label='direct integral')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=12,
               frameon=True, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout()
    return save_fig(fig, os.path.join(OUTPUT_BASE, DIR_SPEC,
                                     'lsr_slope_profile'))


# =============================================================================
# 11. 検算とサマリ
# =============================================================================
def run_checks():
    lines = []
    add = lines.append
    i15 = int(1.5 / DZ)

    add('=' * 74)
    add('設定')
    add('=' * 74)
    _, _, src = get_incident_spectrum()
    add(f'  出力先          : {OUTPUT_BASE}')
    add(f'  帯域            : {BAND_LO/1e9:.2f} - {BAND_HI/1e9:.2f} GHz '
        f'(幾何平均 {BAND_F0/1e9:.3f} GHz)')
    add(f'  伝搬            : {PROPAGATION_MODE}')
    add(f'  Debye 実現      : {USE_DEBYE_REALIZATION}')
    add(f'  入射スペクトル  : {src}')
    add(f'  混合則          : LLL 増分形  eps_wet^(1/3) = eps_dry^(1/3) '
        f'+ v_ice*{_ICE_INCREMENT:.5f}')
    add(f'  系統A 周波数    : {[l for l in PROFILE_FREQ_LABELS]} '
        f'(組成 {FEOTIO2_WT} wt% 固定)')
    add(f'  系統B 組成      : {[l for l in PROFILE_WT_LABELS]} '
        f'(周波数 {PROFILE_FIXED_FREQ/1e9:.2f} GHz 固定)')
    add('')

    # -----------------------------------------------------------------
    add('=' * 74)
    add('検算 1: 周波数分散が出ていないか（系統 A の目的）')
    add('=' * 74)
    add('  eps\'\' 一定モデルが効いていれば、eps\'\' は 3 周波数でほぼ同じ値になり、')
    add('  eps\' は KK による僅かな変化のみ、alpha だけが f に比例して開く。')
    add('')
    add('  深さ 1.5 m, 0 vol% ice, {} wt%'.format(FEOTIO2_WT))
    add('    量          0.5 GHz     1.0 GHz     2.0 GHz    2.0/0.5 比')
    for key, name, ideal in (('eps_re', "eps'    ", '1.000 (一定)'),
                             ('eps_im', "eps''   ", '1.000 (一定)'),
                             ('tand', 'tan_delta', '1.000 (一定)'),
                             ('alpha', 'alpha   ', '4.000 (∝ f)')):
        v = SET_FREQ[key][0, :, i15]
        add('    {}  {:10.6f}  {:10.6f}  {:10.6f}   {:8.4f}   理想 {}'
            .format(name, v[0], v[1], v[2], v[2] / v[0], ideal))
    er = SET_FREQ['eps_re'][0, :, i15]
    ei = SET_FREQ['eps_im'][0, :, i15]
    add('')
    add("    -> eps'  の帯域内変動 (max-min)/mean = {:.3f} %".format(
        100 * (er.max() - er.min()) / er.mean()))
    add('       KK が要求する物理であり 0 にはできない。単調減少。')
    add("    -> eps'' の帯域内変動 (max-min)/mean = {:.3f} %".format(
        100 * (ei.max() - ei.min()) / ei.mean()))
    add('       2 極 Debye 解析解の設計誤差。帯域中心が最大、両端が等しく低い')
    add('       （対称なので 2.0/0.5 比はちょうど 1.000 になる）。')
    a = SET_FREQ['alpha'][0, :, i15]
    ok = abs(a[2] / a[0] - 4.0) < 0.05
    add('    -> alpha 帯域内比 {:.4f}  {}'.format(
        a[2] / a[0], 'OK（eps\'\' 一定が効いている）' if ok else '** 要確認 **'))
    add('')

    # -----------------------------------------------------------------
    add('=' * 74)
    add('検算 2: FeO+TiO2 濃度の影響（系統 B の目的）')
    add('=' * 74)
    add('  eps\' は組成に依らないので、差が出るのは eps\'\'/sigma/tan_delta/alpha のみ。')
    add('')
    add('  深さ 1.5 m, 0 vol% ice, {:.2f} GHz'.format(PROFILE_FIXED_FREQ / 1e9))
    add('    量          5.0 wt%     7.5 wt%    10.0 wt%   10.0/5.0 比')
    for key, name in (('eps_re', "eps'    "), ('eps_im', "eps''   "),
                      ('tand', 'tan_delta'), ('alpha', 'alpha   ')):
        v = SET_COMP[key][0, :, i15]
        add('    {}  {:10.6f}  {:10.6f}  {:10.6f}   {:8.4f}'
            .format(name, v[0], v[1], v[2], v[2] / v[0]))
    v = SET_COMP['eps_re'][0, :, i15]
    add('')
    add("    -> eps'  の組成による変動 {:.3e} %  (0 であるべき)"
        .format(100 * abs(v[2] / v[0] - 1)))
    v = SET_COMP['alpha'][0, :, i15]
    add('    -> alpha は 5.0 -> 10.0 wt% で {:.3f} 倍'.format(v[2] / v[0]))
    add('       これは tan_delta の比そのもの（alpha ∝ tan_delta）')
    add('    -> 往復 3 m の減衰差: {:.2f} dB (5.0 wt%) vs {:.2f} dB (10.0 wt%)'
        .format(-8.686 * 2 * np.trapezoid(SET_COMP['alpha'][0, 0], z),
                -8.686 * 2 * np.trapezoid(SET_COMP['alpha'][0, 2], z)))
    add('')

    # -----------------------------------------------------------------
    add('=' * 74)
    add('検算 3: その他')
    add('=' * 74)
    f, s0, _ = get_incident_spectrum()
    P = s0 ** 2
    I0 = trapz(P, f)
    fc0 = trapz(f * P, f) / I0
    var0 = trapz((f - fc0) ** 2 * P, f) / I0
    add(f'  入射スペクトル : f_c = {fc0/1e9:.4f} GHz, '
        f'sigma_f^2 = {var0/1e18:.4f} GHz^2  (励振ファイルの実測値 0.1443)')

    fc, sg, _, _, _ = spectral_moments(0)
    t_ns = travel_time_at_fc(0)
    i1, i2 = int(0.5 / DZ), int(2.5 / DZ)
    meas = (fc[i2] - fc[i1]) / (t_ns[i2] - t_ns[i1]) / 1e6
    td_mid = float(carrier_tandelta(density_profile(1.5)))
    sg_mid = float(sg[i15] / 1e9)
    pred = -2 * np.pi * td_mid * sg_mid ** 2 * 1e3
    add(f'  df_c/dt        : 実測 {meas:.3f} MHz/ns, 恒等式 {pred:.3f} MHz/ns')

    por = porosity(rho)
    add(f'  空隙率         : {por.min()*100:.1f} - {por.max()*100:.1f} % '
        f'(rho_grain = {RHO_GRAIN})')
    bad_any = False
    for c in ice_contents:
        bad = np.where(c / 100.0 > por)[0]
        if bad.size:
            bad_any = True
            add(f'    ** {c} vol% は深さ {z[bad[0]]:.2f} m 以深で空隙率超過 **')
    if not bad_any:
        add('    すべての氷量が全深さで空隙率以内 -> OK')

    lsr = lsr_profile(0, None)
    k = 2.0 if PROPAGATION_MODE == 'two_way' else 1.0
    i_f = int(np.argmin(np.abs(f - BAND_F0)))
    i_d = int(2.0 / DZ)
    a_lsr = -lsr[i_d, i_f] / (k * z[i_d])
    a_dir, _ = alpha_velocity(z, np.array([BAND_F0]), 0)
    cum = np.cumsum(0.5 * (a_dir[1:, 0] + a_dir[:-1, 0]) * DZ)[i_d - 1] / z[i_d]
    add(f'  alpha (2.0 m)  : LSR から {a_lsr:.6f}, 直接積分 {cum:.6f} Np/m '
        f'-> 相対差 {100*abs(a_lsr/cum-1):.3e} %')
    add('')

    # -----------------------------------------------------------------
    add('=' * 74)
    add('氷 1 vol% あたりの変化（深さ 1.5 m, {:.2f} GHz, {} wt%）'
        .format(PROFILE_FIXED_FREQ / 1e9, FEOTIO2_WT))
    add('=' * 74)
    si = PROFILE_WTS.index(FEOTIO2_WT) if FEOTIO2_WT in PROFILE_WTS else 1
    for ii, c in enumerate(ice_contents):
        if c == 0:
            continue
        d_er = (SET_COMP['eps_re'][ii, si, i15]
                / SET_COMP['eps_re'][0, si, i15] - 1) * 100 / c
        d_a = (SET_COMP['alpha'][ii, si, i15]
               / SET_COMP['alpha'][0, si, i15] - 1) * 100 / c
        n0 = np.sqrt(SET_COMP['eps_re'][0, si, i15])
        n1 = np.sqrt(SET_COMP['eps_re'][ii, si, i15])
        R = (n0 - n1) / (n0 + n1)
        add("  {:>3} vol%: d(eps')/vol% = {:+.3f} %, d(alpha)/vol% = {:+.3f} %, "
            '急峻界面 R = {:.1f} dB'.format(c, d_er, d_a, 20 * np.log10(abs(R))))

    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(OUTPUT_BASE, 'summary.txt'), 'w',
              encoding='utf-8') as fh:
        fh.write(text + '\n')


def write_csv():
    """数値の生データを CSV に書き出す（2 系統ぶん）。"""
    import csv
    paths = []
    for tag, dataset, labels, subdir in (
            ('freq', SET_FREQ, PROFILE_FREQ_LABELS, DIR_FREQ),
            ('comp', SET_COMP, PROFILE_WT_LABELS, DIR_COMP)):
        path = os.path.join(OUTPUT_BASE, subdir, f'profile_{tag}.csv')
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            head = ['depth_m', 'rho', 'porosity']
            for c in ice_contents:
                for lab in labels:
                    tl = lab.replace(' ', '')
                    head += [f'eps_re_{c}vol_{tl}', f'eps_im_{c}vol_{tl}',
                             f'sigma_{c}vol_{tl}', f'tand_{c}vol_{tl}',
                             f'alpha_{c}vol_{tl}']
            w.writerow(head)
            for i in range(Nz):
                row = [z[i], rho[i], porosity(rho[i])]
                for ii in range(n_ice):
                    for si in range(len(labels)):
                        row += [dataset['eps_re'][ii, si, i],
                                dataset['eps_im'][ii, si, i],
                                dataset['sigma'][ii, si, i],
                                dataset['tand'][ii, si, i],
                                dataset['alpha'][ii, si, i]]
                w.writerow(row)
        paths.append(path)

    # スペクトル量
    path = os.path.join(OUTPUT_BASE, DIR_SPEC, 'spectrum.csv')
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        head = ['depth_m']
        for c in ice_contents:
            head += [f'fc_GHz_{c}vol', f'sigma_f_GHz_{c}vol',
                     f'f_lo_GHz_{c}vol', f'f_hi_GHz_{c}vol', f't_ns_{c}vol']
        w.writerow(head)
        mom = {c: spectral_moments(c) for c in ice_contents}
        tt = {c: travel_time_at_fc(c) for c in ice_contents}
        for i in range(Nz):
            row = [z[i]]
            for c in ice_contents:
                fc, sg, _, flo, fhi = mom[c]
                row += [fc[i] / 1e9, sg[i] / 1e9, flo[i] / 1e9, fhi[i] / 1e9,
                        tt[c][i]]
            w.writerow(row)
    paths.append(path)
    return paths


# =============================================================================
# 12. 実行
# =============================================================================
def main():
    _ensure_dirs()
    made = []

    print('--- 系統 A: 周波数を振ったプロファイル ---')
    made += make_profile_family(
        SET_FREQ, REF_FREQ, PROFILE_FREQ_LABELS, DIR_FREQ,
        f'Frequency comparison (FeO+TiO2 = {FEOTIO2_WT} wt%)')

    print('--- 系統 B: FeO+TiO2 を振ったプロファイル ---')
    made += make_profile_family(
        SET_COMP, REF_COMP, PROFILE_WT_LABELS, DIR_COMP,
        f'Composition comparison ({PROFILE_FIXED_FREQ/1e9:.2f} GHz)')
    made.append(make_centroid_by_composition())

    print('--- 共通プロファイル ---')
    made.append(make_density_profile())
    made.append(make_ice_wtpct_profile())

    print('--- スペクトル解析図 ---')
    for c in ice_contents:
        made.append(make_spectrum_evolution(c))
        made.append(make_lsr(c))
    made.append(make_band_edges_profile())
    made.append(make_centroid_width_profile())
    made.append(make_lsr_slope_profile())

    print('--- 数値出力 ---')
    made += write_csv()

    print()
    run_checks()
    print(f'\n生成したファイル: {len(made)} 件（図は png / pdf 各 1）')


if __name__ == '__main__':
    main()