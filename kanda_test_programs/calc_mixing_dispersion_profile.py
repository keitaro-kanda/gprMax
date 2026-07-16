"""
Heiken基準の深さプロファイル + 2極Debye分散(Method A) + 水氷混合 + 解析的シフトレート
==================================================================
設計方針:
  - レゴリス母材: Method A (ANCHOR_FREQ でアンカー)
        eps'_static = HEIKEN_EPS_BASE^rho                (Heiken1991)
        tan_delta_H = 10^(A*FeOTiO2 + B*rho - C)         (Heiken1991)
        損失は2極Debye極が担う。各深さで Debye Delta_eps を
        「ANCHOR_FREQ で eps'' が Heiken 値に一致」するようスケール。
        sigma_ohmic = 0 (Boivin sigma_DC は誤差内でゼロのため不採用)。
  - 水氷混合: 各周波数で構成したレゴリス複素誘電率と、氷の複素誘電率を
        Maxwell-Garnett 則で混合(周波数ごとに評価 = 物理的に正しい順序)。
        氷パラメータ・混合式は非分散プロファイルコードから流用(Evans1965)。
  - 周波数=線の色, 水氷含有量=線の種類 で区別。

  ★重要: レゴリス誘電モデル(密度式 + Heiken経験式 + アンカー周波数)は
     このファイル冒頭の「単一の情報源」ブロックだけで定義する。
     プロファイル図・解析的見積もり(中心周波数/シフトレート/スペクトル)は
     すべてその共通関数を経由するため、式を変えるときはそこ1箇所を直せばよい。
出力:
  (1) Method A の 2x2 まとめ図 (eps', eps'', sigma_eff, tan delta)
  (2)-(5) 各物理量ごとに 左:深さプロファイル / 右:0vol%との相対差[%] の2列図
  (6) 密度プロファイル、水氷wt%プロファイル
  (7) 解析的周波数シフトレートプロファイル
  (8) 解析的中心周波数プロファイル
  (9) 各水氷含有量ごとの解析的スペクトル比較図 (規格化 dB)
 (10) ★追加: STFTパラメータ要求解析 (output_dir_centroid/STFT_parameter)
        (10-a) 中心周波数: 左=0vol%との差[GHz](絶対値) / 右=必要nperseg
        (10-b) シフトレート: 左=0vol%との差[GHz/ns](絶対値) / 右=必要nperseg
        (10-c) 横軸nperseg vs 周波数分解能 (Δf, Δḟ)
        (10-d) 横軸nperseg vs 深さ分解能 (Δz, Δż; eps_r = 2.4..3.2)
        (10-e) 要求サマリ txt
 (11) サマリ txt
==================================================================
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.constants as const  # 追加: 物理定数(光速)用

# gprMaxのルートディレクトリをパスに追加
gprmax_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, gprmax_root)

from gprMax.exceptions import CmdInputError
from tools.core.outputfiles_merge import get_output_data

# ============================================================
# 0. 出力先・基本設定
# ============================================================
output_base_dir = '/Volumes/SSD_Kanda_BUFFALO/test_programs_output/mixing_dispersion_profile'
os.makedirs(output_base_dir, exist_ok=True)
output_dir_profile = os.path.join(output_base_dir, 'profile')
os.makedirs(output_dir_profile, exist_ok=True)
output_dir_centroid = os.path.join(output_base_dir, 'centroid')
os.makedirs(output_dir_centroid, exist_ok=True)
output_dir_waveform = os.path.join(output_dir_centroid, 'waveform')
os.makedirs(output_dir_waveform, exist_ok=True)
# ★追加: STFTパラメータ検討用の出力先
output_dir_stft = os.path.join(output_dir_centroid, 'STFT_parameter')
os.makedirs(output_dir_stft, exist_ok=True)

eps0 = 8.8541878128e-12          # 真空の誘電率 [F/m]

# 深さ [m]
z   = np.arange(0, 3.01, 0.02)   # [m]

FeOTiO2 = 20.0                   # [wt%]

# 比較する周波数 (=線の色)
freqs = np.array([0.5e9, 1.25e9, 2.0e9])     # [Hz]
freq_labels = ['0.5 GHz', '1.25 GHz', '2.0 GHz']
freq_styles = ['-', '--', '-.']
#ANCHOR_FREQ = 1.25e9
ANCHOR_FREQ = 450E6 # Heiken1991 Fig 9.54の、450 MHz計測経験式を使う

# --- Heiken1991 Fig 9.54 の 450 MHz 計測経験式の係数 ---
#     (eps' と tan_d のモデル係数。これらとアンカーを変えれば全計算に反映される)
HEIKEN_EPS_BASE = 1.843          # eps' = HEIKEN_EPS_BASE ** rho
HEIKEN_TAND_A   = 0.033          # tan_d 経験式の (FeO+TiO2) 係数
HEIKEN_TAND_B   = 0.231          # tan_d 経験式の rho 係数
HEIKEN_TAND_C   = 3.061          # tan_d 経験式の定数項
# (参考: 旧 1.25 GHz 式は BASE=1.919, A=0.038, B=0.312, C=3.260, ANCHOR=1.25e9)

# 水氷含有量 (=線の種類)  [vol%]
ice_contents = [0, 1, 5, 10, 20]
ice_colors   = ['k', 'r', 'g', 'b', 'c']   # 0=実線, 以降で区別
ice_labels   = [f'{c} vol% ice' for c in ice_contents]

# 水氷パラメータ (非分散コードから流用; Evans1965)
EPS_ICE_RE = 3.17
EPS_ICE_IM = 3.17 * 6e-5         # = eps' * tan_d_ice
eps_ice_complex = EPS_ICE_RE - 1j * EPS_ICE_IM   # 周波数依存は無視(GPR帯で極小)
RHO_ICE = 0.934                  # 氷の密度 [g/cm^3] (月面温度 ~100K を想定;
                                  # Feistel & Wagner, 2006)。0℃での値0.917とは異なる点に注意

# ------------------------------------------------------------
# 解析的波形計算(中心周波数/シフトレート/スペクトル)の共通設定
#   ・入射波(Ascan.out)の格納パスと解析帯域
#   ・受信機深さ(rx_depth)と、スペクトル比較図の対象深さ
# ------------------------------------------------------------
ASCAN_OUTFILE_PATH = ("/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_3x4/"
                      "waveform_test/gaussiandot_1.25GHz_underground/result/Ascan.out")
FREQ_BAND_MIN = 0.25e9            # 解析帯域下限 [Hz]、帯域下限値の1/2
FREQ_BAND_MAX = 6.0e9            # 解析帯域上限 [Hz]、帯域上限値の2倍
RX_DEPTH      = 0.10             # 受信機(計算開始)深さ [m]
SPECTRUM_TARGET_DEPTHS = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # スペクトル比較の対象深さ [m]
#POWER_THRESHOLD_DB     = -30.0   # スペクトル比較図のパワーマスク基準 [dB]

# ------------------------------------------------------------
# ★追加: STFT パラメータ検討の設定 (シミュレーション設定に準拠して固定)
#   dt はシミュレーションの時間刻み(グリッドサイズ 0.005 m の場合 1.18e-11 s)。
#   f_s = 1/dt はサンプリング周波数で、nperseg 以外は固定値として扱う。
# ------------------------------------------------------------
STFT_DT_S     = 1.18e-11                    # シミュレーションの時間分解能 [s] (固定)
STFT_FS_HZ    = 1.0 / STFT_DT_S             # サンプリング周波数 [Hz] (固定; ≈84.75 GHz)
STFT_DT_NS    = STFT_DT_S * 1e9             # [ns]
STFT_FS_GHZ   = STFT_FS_HZ / 1e9            # [GHz]
STFT_NOVERLAP_RATIO = 0.75                  # noverlap = 3/4 * nperseg (解析コードの設定)
DETECT_MARGIN = 2.0                         # 差の 1/DETECT_MARGIN を分解できることを要求
                                            # (1.0: 差そのもの, 2.0: 差の半分まで分解)
EPSR_LIST_FOR_DZ = [2.4, 2.6, 2.8, 3.0, 3.2]   # 深さ分解能曲線用の比誘電率
EPSR_COLORS      = ['r', 'g', 'b', 'c', 'm']  # 深さ分解能曲線の色
NPERSEG_RANGE    = np.arange(16, 4097)      # 分解能曲線の横軸

# ============================================================
# レゴリス誘電モデルの「単一の情報源」(密度式・Heiken経験式)
#   ここだけを変更すれば、プロファイル図・解析的見積もりの
#   すべての計算に一貫して反映される(式の重複コピーを排除)。
# ============================================================
def density_profile(depth_m):
    """Heiken1991 の密度式。depth [m] -> rho [g/cm^3] (内部で[cm]換算)。
    スカラー・配列どちらでも動作する。"""
    z_cm = depth_m * 100.0
    return 1.92 * (z_cm + 12.2) / (z_cm + 18.0)

def heiken_eps_real(rho):
    """Heiken1991 経験式の静的実部 eps'。"""
    return HEIKEN_EPS_BASE ** rho

def heiken_tan_delta(rho):
    """Heiken1991 経験式の tan δ (FeO+TiO2 = FeOTiO2 wt%)。"""
    return 10 ** (HEIKEN_TAND_A * FeOTiO2 + HEIKEN_TAND_B * rho - HEIKEN_TAND_C)

# 深さ方向の密度プロファイル (以降の全計算はこの rho を参照)
rho = density_profile(z)                     # [g/cm^3]

# ============================================================
# 1. Heiken 基準量 (深さ依存, 周波数非依存)
# ============================================================
eps_re_Heiken = heiken_eps_real(rho)
tan_d_heiken  = heiken_tan_delta(rho)
eps_im_heiken = eps_re_Heiken * tan_d_heiken

# ============================================================
# 2. 2極Debye パラメータ (.in ファイルと同一)
# ============================================================
DEBYE_DE1  = 0.261
DEBYE_TAU1 = 4.6212e-11
DEBYE_DE2  = 0.088
DEBYE_TAU2 = 2.82195e-10

def debye_imag_shape(omega, tau):
    return omega * tau / (1.0 + (omega * tau) ** 2)

def debye_total_imag(omega, scale):
    de1 = DEBYE_DE1 * scale
    de2 = DEBYE_DE2 * scale
    return (de1 * debye_imag_shape(omega, DEBYE_TAU1)
            + de2 * debye_imag_shape(omega, DEBYE_TAU2))

def debye_total_real_drop(omega, scale):
    de1 = DEBYE_DE1 * scale
    de2 = DEBYE_DE2 * scale
    drop1 = de1 * (omega * DEBYE_TAU1) ** 2 / (1.0 + (omega * DEBYE_TAU1) ** 2)
    drop2 = de2 * (omega * DEBYE_TAU2) ** 2 / (1.0 + (omega * DEBYE_TAU2) ** 2)
    return drop1 + drop2

# ============================================================
# 3. Method A: 各深さの scale を ANCHOR_FREQ アンカーで決定
# ============================================================
w_anchor = 2 * np.pi * ANCHOR_FREQ
unit_imag_anchor = debye_total_imag(w_anchor, scale=1.0)
scale_A = eps_im_heiken / unit_imag_anchor          # (Nz,)

# ============================================================
# 4. Maxwell-Garnett 混合則 (複素対応; 非分散コードと同形)
#    epsilon1=母材(レゴリス), epsilon2=包有物(氷), f=体積分率[vol%]
# ============================================================
def maxwell_garnett(eps_host, eps_incl, f_volpct):
    f = f_volpct / 100.0
    return eps_host + 3.0 * f * eps_host * (eps_incl - eps_host) \
           / (eps_incl + 2.0 * eps_host - f * (eps_incl - eps_host))

# ============================================================
# 5. 各周波数 × 各氷含有量で複素誘電率を構成・混合
#    格納: shape (n_ice, n_freq, Nz)
# ============================================================
n_ice, n_freq, Nz = len(ice_contents), len(freqs), len(z)
EPS_RE = np.zeros((n_ice, n_freq, Nz))
EPS_IM = np.zeros((n_ice, n_freq, Nz))
SIGMA  = np.zeros((n_ice, n_freq, Nz))
TAND   = np.zeros((n_ice, n_freq, Nz))

for fi, f in enumerate(freqs):
    w = 2 * np.pi * f
    # --- レゴリス母材の複素誘電率 (Method A) ---
    reg_re = eps_re_Heiken - debye_total_real_drop(w, scale_A)   # 実部(分散低下込み)
    reg_im = debye_total_imag(w, scale_A)                        # 虚部(Debye損失)
    eps_reg_complex = reg_re - 1j * reg_im                       # (Nz,)
    for ii, c in enumerate(ice_contents):
        if c == 0:
            eps_mix = eps_reg_complex
        else:
            eps_mix = maxwell_garnett(eps_reg_complex, eps_ice_complex, c)
        re = np.real(eps_mix)
        im = -np.imag(eps_mix)          # eps'' は正の量として扱う
        EPS_RE[ii, fi] = re
        EPS_IM[ii, fi] = im
        SIGMA[ii, fi]  = im * w * eps0
        TAND[ii, fi]   = im / re

# ============================================================
# 6. 描画ユーティリティ
#    色=氷含有量, 線種=周波数
# ============================================================
def draw_lines(ax, data, ref=None):
    """data: shape (n_ice, n_freq, Nz) を全組み合わせ描画。"""
    if ref is not None:
        ax.plot(ref, z, color='gray', linestyle='--', lw=2, zorder=1, label='Heiken (ref)')
    for ii in range(n_ice):
        for fi in range(n_freq):
            ax.plot(data[ii, fi], z, color=ice_colors[ii],
                    linestyle=freq_styles[fi], lw=1.6, zorder=3 + ii)
    ax.set_ylabel('Depth (m)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.minorticks_on()
    ax.grid(True, alpha=0.4)
    ax.invert_yaxis()

# 凡例ハンドル: 周波数(色) と 氷含有量(線種) を分けて示す
freq_handles = [Line2D([0], [0], linestyle=freq_styles[i], color='k', lw=2, label=freq_labels[i])
                for i in range(n_freq)]
ice_handles = [Line2D([0], [0], color=ice_colors[i], linestyle='-', lw=2,
                      label=ice_labels[i]) for i in range(n_ice)]
heiken_handle = [Line2D([0], [0], color='gray', ls='--', lw=2, label=r'Heiken (for $\varepsilon_r$ and $\tan \delta$)')]

def add_legend(fig):
    fig.legend(handles=freq_handles + ice_handles + heiken_handle,
               loc='lower center', ncol=4, fontsize=14, frameon=True,
               bbox_to_anchor=(0.5, 1.0))

# ------------------------------------------------------------
# 6-A. 2x2 まとめ図
# ------------------------------------------------------------
def make_summary_2x2():
    fig, axes = plt.subplots(2, 2, figsize=(10, 11))
    draw_lines(axes[0, 0], EPS_RE, ref=eps_re_Heiken)
    axes[0, 0].set_xlabel(r"$\varepsilon^{\prime}$", fontsize=18)
    draw_lines(axes[0, 1], EPS_IM)
    axes[0, 1].set_xlabel(r"$\varepsilon^{\prime\prime}$", fontsize=18)
    draw_lines(axes[1, 0], SIGMA)
    axes[1, 0].set_xlabel(r"Conductivity $\sigma_{\rm eff}$ [S/m]", fontsize=18)
    draw_lines(axes[1, 1], TAND, ref=tan_d_heiken)
    axes[1, 1].set_xlabel(r"$\tan\delta$", fontsize=18)
    axes[1, 1].locator_params(axis='x', nbins=5)
    add_legend(fig)
    plt.tight_layout()
    base = os.path.join(output_dir_profile, 'summary_2x2')
    fig.savefig(base + '.png', bbox_inches='tight', dpi=200)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    plt.close(fig)
    return base + '.png'

# ------------------------------------------------------------
# 6-B. 各物理量ごとの [左:プロファイル / 右:0vol%との相対差[%]] 図
# ------------------------------------------------------------
def make_profile_and_delta(data, quantity_label, fname, ref=None):
    base0 = data[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    draw_lines(axes[0], data, ref=ref)
    axes[0].set_xlabel(quantity_label, fontsize=18)
    if fname == 'losstangent':
        axes[0].locator_params(axis='x', nbins=5)

    for ii in range(n_ice):
        if ice_contents[ii] == 0:
            continue
        for fi in range(n_freq):
            rel = np.abs(data[ii, fi] - base0[fi]) / base0[fi] * 100.0
            axes[1].plot(rel, z, color=ice_colors[ii],
                         linestyle=freq_styles[fi], lw=1.6, zorder=3 + ii)
    axes[1].set_xlabel(r'$|X_{0\%} - X|\,/\,X_{0\%}\times100$ [%]', fontsize=18)
    axes[1].set_ylabel('Depth (m)', fontsize=18)
    axes[1].set_xscale('log')
    axes[1].tick_params(axis='both', which='major', labelsize=14)
    axes[1].minorticks_on()
    axes[1].grid(True, alpha=0.4)
    axes[1].invert_yaxis()
    add_legend(fig)
    plt.tight_layout()
    base = os.path.join(output_dir_profile, fname)
    fig.savefig(base + '.png', bbox_inches='tight', dpi=200)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    plt.close(fig)
    return base + '.png'

# ------------------------------------------------------------
# 6-C. 密度プロファイル (横軸: 密度, 縦軸: 深さ)
# ------------------------------------------------------------
def make_density_profile():
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(rho, z, color='k', lw=2)
    ax.set_xlabel(r'$\rho$ [g/cm$^{3}$]', fontsize=18)
    ax.set_ylabel('Depth (m)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.minorticks_on()
    ax.grid(True, alpha=0.4)
    ax.invert_yaxis()
    plt.tight_layout()
    base = os.path.join(output_dir_profile, 'density_profile')
    fig.savefig(base + '.png', bbox_inches='tight', dpi=200)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    plt.close(fig)
    return base + '.png'

# ------------------------------------------------------------
# 6-D. 氷含有量 vol% -> wt% 変換プロファイル
# ------------------------------------------------------------
def make_ice_wtpct_profile():
    fig, ax = plt.subplots(figsize=(5, 6))
    for ii, c in enumerate(ice_contents):
        if c == 0:
            continue
        f_vol = c / 100.0
        wtpct = 100.0 * f_vol * RHO_ICE / (f_vol * RHO_ICE + (1.0 - f_vol) * rho)
        ax.plot(wtpct, z, color=ice_colors[ii], linestyle='-', lw=2,
                label=f'{c} vol% ice')
    ax.set_xlabel('Ice content [wt%]', fontsize=18)
    ax.set_ylabel('Depth (m)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.minorticks_on()
    ax.grid(True, alpha=0.4)
    ax.invert_yaxis()
    ax.legend(loc='center right', fontsize=14)
    plt.tight_layout()
    base = os.path.join(output_dir_profile, 'ice_wtpct_profile')
    fig.savefig(base + '.png', bbox_inches='tight', dpi=200)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    plt.close(fig)
    return base + '.png'

# ============================================================
# 6-E. 追加機能: 解析的プロファイル計算 (中心周波数・シフトレート・スペクトル)
# ============================================================

# --- 6-E-0. 共通ヘルパ (中心周波数/シフトレート/スペクトルで共有) ---
def get_eps_mix_spectrum(depth, omega, ice_volpct):
    """深さ depth [m], 角周波数配列 omega [rad/s] に対する
    Method A(レゴリス母材) + Maxwell-Garnett(氷混合) の複素誘電率スペクトルを返す。
    符号規約は eps = eps' - j*eps'' (eps'' >= 0)。ANCHOR_FREQ でアンカー。
    誘電モデルは冒頭の density_profile / heiken_* を経由する(単一の情報源)。"""
    rho_d = density_profile(depth)
    eps_re_H = heiken_eps_real(rho_d)
    tan_d_H = heiken_tan_delta(rho_d)
    eps_im_H = eps_re_H * tan_d_H

    unit_imag_anchor = debye_total_imag(2 * np.pi * ANCHOR_FREQ, 1.0)
    scale_A_val = eps_im_H / unit_imag_anchor

    reg_re = eps_re_H - debye_total_real_drop(omega, scale_A_val)
    reg_im = debye_total_imag(omega, scale_A_val)
    eps_reg = reg_re - 1j * reg_im

    if ice_volpct == 0:
        return eps_reg
    return maxwell_garnett(eps_reg, eps_ice_complex, ice_volpct)


def local_alpha_velocity(depth, omega, ice_volpct):
    """深さ depth [m] における場の振幅減衰係数 alpha [Np/m] と位相速度 v [m/s] を返す。
    減衰・速度の式をここに一元化し、中心周波数/シフトレート/スペクトルで共有する。"""
    sqrt_eps = np.sqrt(get_eps_mix_spectrum(depth, omega, ice_volpct))
    alpha = - (omega / const.c) * np.imag(sqrt_eps)
    v = const.c / np.real(sqrt_eps)
    return alpha, v


def spectral_centroid(power, f_calc):
    """パワースペクトルの重心周波数 [Hz]。中心周波数の定義を全図で統一する。"""
    return np.trapz(f_calc * power, f_calc) / np.trapz(power, f_calc)


_incident_spectrum_cache = None   # 入射波の「全帯域」スペクトルをキャッシュ (帯域制限は呼び出し毎に適用)
def load_incident_spectrum(freq_min=FREQ_BAND_MIN, freq_max=FREQ_BAND_MAX):
    """入射波(Ascan.out)を読み込み、指定帯域 [freq_min, freq_max] に限定した
    (f_calc [Hz], S0_calc [複素スペクトル], omega [rad/s]) を返す。
    Ascan.out が無い場合は合成ガウシアンパルスにフォールバックする。
    FFT結果(全帯域)をキャッシュし、帯域マスクのみを呼び出し毎に適用するため、
    帯域を変えても再FFTは不要。デフォルトは解析帯域 FREQ_BAND_MIN/MAX。"""
    global _incident_spectrum_cache
    if _incident_spectrum_cache is None:
        try:
            if os.path.exists(ASCAN_OUTFILE_PATH):
                ascan_data, dt_ascan = get_output_data(ASCAN_OUTFILE_PATH, 1, 'Ez')
                if ascan_data.ndim == 1:
                    e_incident = ascan_data
                else:
                    e_incident = ascan_data[:, 0]
                N = len(e_incident)
                freq_ascan = np.fft.rfftfreq(N, d=dt_ascan)
                S0_omega = np.fft.rfft(e_incident)
            else:
                raise FileNotFoundError

        except Exception as e:
            print(f"Warning: Could not load Ascan.out data. Using synthetic Gaussian pulse. Error: {e}")
            dt_ascan = 1e-10  # 0.1 ns
            t_ascan = np.arange(-5e-9, 5e-9, dt_ascan)
            e_incident = np.exp(-((t_ascan - 0) ** 2) / (2 * (1 / (2 * np.pi * ANCHOR_FREQ)) ** 2))
            N = len(e_incident)
            freq_ascan = np.fft.rfftfreq(N, d=dt_ascan)
            S0_omega = np.fft.rfft(e_incident)

        # 全帯域(rfft の全成分)をキャッシュ。帯域制限はここでは行わない。
        _incident_spectrum_cache = (freq_ascan, S0_omega)

    freq_ascan, S0_omega = _incident_spectrum_cache
    band_mask = (freq_ascan >= freq_min) & (freq_ascan <= freq_max)
    f_calc = freq_ascan[band_mask]
    S0_calc = S0_omega[band_mask]
    omega = 2 * np.pi * f_calc
    return f_calc, S0_calc, omega


def calc_analytical_centroid_and_shiftrate(ice_volpct):
    """水氷含有量に対する解析的な中心周波数とシフトレートの深さプロファイルを計算"""
    # 入射波スペクトル(解析帯域 FREQ_BAND_MIN/MAX に限定済み)を取得
    f_calc, S0_calc, omega = load_incident_spectrum()

    # --- 時間遅延（Time Offset）の計算 ---
    antenna_height = 0.35    # [m]
    system_lag_ns  = 0.837   # [ns]
    rx_depth       = RX_DEPTH  # [m]

    t_air_ns = (2.0 * antenna_height / const.c) * 1e9

    # 地表から受信機までのオフセット伝搬時間 (共通の密度式・Heiken式を利用)
    d_sub_offset = np.linspace(0, rx_depth, 50)
    eps_sub_offset = heiken_eps_real(density_profile(d_sub_offset))
    v_sub = const.c / np.sqrt(eps_sub_offset)
    dt_sub = d_sub_offset[1] - d_sub_offset[0]
    t_ground_start_ns = np.sum(2.0 * dt_sub / v_sub) * 1e9

    t_offset_ns = system_lag_ns + t_air_ns + t_ground_start_ns

    # 計算開始深さを受信機の深さに設定
    d_array = z[z >= rx_depth]
    if len(d_array) == 0:
        d_array = np.linspace(rx_depth, z[-1], 200)
    d_step = d_array[1] - d_array[0] if len(d_array) > 1 else 0.02

    f_peak_d = []
    t_delay_d = []
    cumulative_attenuation = np.zeros_like(omega)
    cumulative_time = np.zeros_like(omega)

    for i, d in enumerate(d_array):
        # 各深さの減衰係数 alpha と位相速度 v (共通ヘルパ)
        alpha_d, v_d = local_alpha_velocity(d, omega, ice_volpct)

        if i > 0:
            cumulative_attenuation += alpha_d * d_step
            cumulative_time += 2 * d_step / v_d

        # 深さ d の反射体から往復して戻るエコースペクトル (往復減衰 exp(-2*∫alpha dz))
        S_d_w = S0_calc * np.exp(-2 * cumulative_attenuation)
        power = np.abs(S_d_w)**2

        f_peak = spectral_centroid(power, f_calc)
        f_peak_d.append(f_peak)

        t_delay_ground = np.interp(f_peak, f_calc, cumulative_time)
        t_total_ns = t_offset_ns + (t_delay_ground * 1e9)
        t_delay_d.append(t_total_ns)

    f_peak_d_ghz = np.array(f_peak_d) / 1e9 # [GHz]
    t_delay_d = np.array(t_delay_d) # [ns]

    # 時間軸 dt_stft 上でのグラディエント算出（元のコードの仕様を踏襲）
    dt_stft = 0.1 # [ns]
    t_axis = np.arange(np.nanmin(t_delay_d), np.nanmax(t_delay_d) + dt_stft, dt_stft)
    if len(t_axis) > 1:
        analytical_f_peak_profile = np.interp(t_axis, t_delay_d, f_peak_d_ghz, left=np.nan, right=np.nan)
        analytical_shiftrate_profile = np.gradient(analytical_f_peak_profile, dt_stft)
        # 深さ軸にマップし直すための補間
        shiftrate_d = np.interp(t_delay_d, t_axis, analytical_shiftrate_profile, left=np.nan, right=np.nan)
    else:
        shiftrate_d = np.gradient(f_peak_d_ghz, t_delay_d)

    # 全体の深さ軸 z に展開 (rx_depthより浅い部分は NaN になる)
    shiftrate_z = np.interp(z, d_array, shiftrate_d, left=np.nan, right=np.nan)
    f_peak_z = np.interp(z, d_array, f_peak_d_ghz, left=np.nan, right=np.nan)

    return shiftrate_z, f_peak_z


# --- 6-E-1. 中心周波数/シフトレートのキャッシュ付きラッパ ---
#     (差分計算・要求 nperseg 計算・プロファイル図で同じ結果を使い回す)
_centroid_cache = {}
def get_centroid_shiftrate(ice_volpct):
    """calc_analytical_centroid_and_shiftrate のキャッシュ版。
    戻り値: (shiftrate_z [GHz/ns], f_peak_z [GHz])  ※ 深さ軸 z 上"""
    if ice_volpct not in _centroid_cache:
        _centroid_cache[ice_volpct] = calc_analytical_centroid_and_shiftrate(ice_volpct)
    return _centroid_cache[ice_volpct]


def make_shiftrate_profile():
    """解析的シフトレートプロファイルをプロット"""
    fig, ax = plt.subplots(figsize=(5, 6))

    for ii, c in enumerate(ice_contents):
        shiftrate, _ = get_centroid_shiftrate(c)
        ax.plot(shiftrate, z, color=ice_colors[ii], linestyle='-', lw=2,
                label=ice_labels[ii])

    ax.set_xlabel('Shift Rate [GHz/ns]', fontsize=18)
    ax.set_ylabel('Depth (m)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.minorticks_on()
    ax.grid(True, alpha=0.4)
    ax.invert_yaxis()

    ax.legend(loc='lower left', fontsize=14)

    plt.tight_layout()
    base = os.path.join(output_dir_centroid, 'shiftrate_profile')
    fig.savefig(base + '.png', bbox_inches='tight', dpi=200)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    plt.close(fig)
    return base + '.png'

def make_centroid_profile():
    """解析的中心周波数プロファイルをプロット"""
    fig, ax = plt.subplots(figsize=(5, 6))

    for ii, c in enumerate(ice_contents):
        _, f_peak = get_centroid_shiftrate(c)
        ax.plot(f_peak, z, color=ice_colors[ii], linestyle='-', lw=2,
                label=ice_labels[ii])

    ax.set_xlabel('Centroid Frequency [GHz]', fontsize=18)
    ax.set_ylabel('Depth (m)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.minorticks_on()
    ax.grid(True, alpha=0.4)
    ax.invert_yaxis()

    ax.legend(loc='upper left', fontsize=14)

    plt.tight_layout()
    base = os.path.join(output_dir_centroid, 'centroid_frequency_profile')
    fig.savefig(base + '.png', bbox_inches='tight', dpi=200)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    plt.close(fig)
    return base + '.png'

# ------------------------------------------------------------
# 6-F. 解析的スペクトル比較図 (規格化 dB スケール)
#      指定した水氷含有量について、複数の対象深さでの規格化パワースペクトルを
#      重ねてプロットする。入射波(rx_depth)のピークパワーを 0 dB 基準とし、
#      各深さの中心周波数を破線で示す。
#      ※ 中心周波数は calc_analytical_centroid_and_shiftrate と同一の
#        spectral_centroid / 同一帯域(FREQ_BAND)で計算するため、両図は整合する。
# ------------------------------------------------------------
def make_spectrum_comparison(ice_volpct):
    # 解析帯域 [FREQ_BAND_MIN, FREQ_BAND_MAX] に限定した入射波スペクトルを取得
    f_calc, S0_calc, omega = load_incident_spectrum(FREQ_BAND_MIN, FREQ_BAND_MAX)
    f_calc_ghz = f_calc / 1e9

    # 入射波(基準深さ)のパワー最大値を 0 dB の基準とする
    power_0 = np.abs(S0_calc) ** 2
    max_power_0 = np.max(power_0)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(SPECTRUM_TARGET_DEPTHS)))

    for i, d in enumerate(SPECTRUM_TARGET_DEPTHS):
        # 基準深さ(rx_depth)から目的の深さ(d)までの往復減衰を積分で計算
        if d <= RX_DEPTH:
            cum_alpha = np.zeros_like(omega)
        else:
            # 積分用の細かい刻み（rx_depth 〜 d まで）
            d_sub = np.linspace(RX_DEPTH, d, 200)
            dz = d_sub[1] - d_sub[0]
            cum_alpha = np.zeros_like(omega)
            for k, d_int in enumerate(d_sub):
                # 共通ヘルパで局所的な減衰率を取得
                alpha_int, _ = local_alpha_velocity(d_int, omega, ice_volpct)
                # 積分の累積 (最初の点は dz=0 と見なせるため k>0 で加算)
                if k > 0:
                    cum_alpha += alpha_int * dz

        # 積分された全減衰量を用いてエコースペクトルを計算 (往復 exp(-2*cum_alpha))
        S_d_w = S0_calc * np.exp(-2 * cum_alpha)
        power = np.abs(S_d_w) ** 2

        # 中心周波数（線形スケール。f_calc は既に解析帯域に限定済み）
        f_peak_ghz = spectral_centroid(power, f_calc) / 1e9

        # 入射波の最大パワーで規格化し、dB へ変換
        power_norm = power / max_power_0
        power_db = 10.0 * np.log10(power_norm + 1e-30)

        # スペクトル(dB)を描画
        ax.plot(f_calc_ghz, power_db, color=colors[i],
                label=f'Depth {d:.1f} m ($f_c$ = {f_peak_ghz:.2f} GHz)')
        # 中心周波数を破線で描画
        ax.axvline(f_peak_ghz, color=colors[i], linestyle='--', alpha=0.7)

    # パワーマスク基準を赤の点線で描画
    # ax.axhline(POWER_THRESHOLD_DB, color='red', linestyle=':', lw=2,
    #            label=f'Mask threshold ({POWER_THRESHOLD_DB:.0f} dB)')

    ax.set_xlabel('Frequency [GHz]', fontsize=18)
    ax.set_ylabel('Normalized Power [dB]', fontsize=18)
    ax.set_xlim(FREQ_BAND_MIN / 1e9, FREQ_BAND_MAX / 1e9)
    # Y軸の表示範囲を調整（閾値の少し下から 0 dB の少し上まで）
    # ax.set_ylim(bottom=POWER_THRESHOLD_DB - 15, top=5)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.minorticks_on()
    ax.grid(True, alpha=0.4)
    # 凡例を外側に配置してグラフと被らないようにする
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=14)

    plt.tight_layout()
    base = os.path.join(output_dir_waveform, f'spectrum_comparison_{ice_volpct}vol')
    fig.savefig(base + '.png', bbox_inches='tight', dpi=200)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    plt.close(fig)
    return base + '.png'

# ============================================================
# ★ 6-G. STFT パラメータ要求の解析 (追加機能)
#    「単一の情報源」方針に従い、STFT の分解能式はここで一元定義する。
#
#    前提 (STFT_parameter ノート準拠):
#       dt, f_s = 1/dt は シミュレーション設定に準拠した固定値
#       周波数分解能   : Δf     = f_s / nperseg                       [GHz]
#       空間分解能     : Δz     = nperseg * dt * v / 2                 [m]
#       シフトレート   : Δḟ    = 2√2 (f_s / nperseg)^2                [GHz/ns]
#                        (noverlap = 3/4, np.gradient の中心差分, 最悪ケース)
#       シフトレートの空間分解能: Δż = 1.5 * nperseg * dt * v / 2      [m]
#    ※ v は誘電率プロファイルから求めた「その場」の位相速度を使う。
# ============================================================

# --- 6-G-0. STFT 分解能の基本式 (nperseg -> 分解能) ---
def stft_delta_f_ghz(nperseg):
    """周波数分解能 Δf [GHz] = f_s / nperseg"""
    return STFT_FS_GHZ / np.asarray(nperseg, dtype=float)

def stft_delta_fdot_ghz_per_ns(nperseg):
    """シフトレート分解能 Δḟ [GHz/ns] = 2√2 (f_s/nperseg)^2 (最悪ケース)"""
    return 2.0 * np.sqrt(2.0) * (STFT_FS_GHZ / np.asarray(nperseg, dtype=float)) ** 2

def stft_delta_z(nperseg, v):
    """空間分解能 Δz [m] = nperseg * dt * v / 2   (v [m/s])"""
    return np.asarray(nperseg, dtype=float) * STFT_DT_S * np.asarray(v, dtype=float) / 2.0

def stft_delta_zdot(nperseg, v):
    """シフトレートの空間分解能 Δż [m] = 1.5 * nperseg * dt * v / 2"""
    return 1.5 * stft_delta_z(nperseg, v)

# --- 6-G-1. 分解能 -> 必要 nperseg (上の式の逆解き) ---
def nperseg_required_for_df(df_ghz):
    """Δf <= df_ghz を満たす最小 nperseg = f_s / df"""
    df = np.asarray(df_ghz, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(df > 0, STFT_FS_GHZ / df, np.nan)

def nperseg_required_for_dfdot(dfdot_ghz_per_ns):
    """Δḟ <= dfdot を満たす最小 nperseg = f_s * sqrt(2√2 / dfdot)"""
    dfd = np.asarray(dfdot_ghz_per_ns, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(dfd > 0,
                        STFT_FS_GHZ * np.sqrt(2.0 * np.sqrt(2.0) / dfd), np.nan)

# --- 6-G-2. その場の位相速度プロファイル ---
def local_velocity_profile(ice_volpct, f_peak_z):
    """各深さ z における「その場」の位相速度 v [m/s]。
    誘電率は Method A + Maxwell-Garnett のプロファイルを用い、
    評価周波数はその深さでの中心周波数 f_peak_z [GHz] とする
    (分散媒質なので、中心周波数での値を使うのが整合的)。"""
    v_z = np.full_like(z, np.nan, dtype=float)
    for j, d in enumerate(z):
        fc = f_peak_z[j]
        if not np.isfinite(fc) or fc <= 0:
            continue
        omega_j = np.array([2.0 * np.pi * fc * 1e9])
        _, v_j = local_alpha_velocity(d, omega_j, ice_volpct)
        v_z[j] = v_j[0]
    return v_z

# --- 6-G-3. 0 vol% との差 と 必要 nperseg の計算 ---
def calc_stft_requirements(ice_volpct):
    """水氷含有量 ice_volpct について、
       ・0 vol% との中心周波数の差   d_fc   [GHz]      (絶対値)
       ・0 vol% とのシフトレートの差 d_fdot [GHz/ns]   (絶対値)
       ・それぞれを分解するのに必要な nperseg
       ・その nperseg でのその場の空間分解能 Δz, Δż  [m]
       ・その場の位相速度 v [m/s]
    を深さ軸 z 上で返す (dict)。
    DETECT_MARGIN=2 とすれば「差の半分を分解する」要求になる。"""
    sr0, fc0 = get_centroid_shiftrate(0)
    sr, fc = get_centroid_shiftrate(ice_volpct)

    d_fc = np.abs(fc - fc0)          # [GHz]
    d_fdot = np.abs(sr - sr0)        # [GHz/ns]

    # 要求分解能 = 差 / DETECT_MARGIN
    req_df = d_fc / DETECT_MARGIN
    req_dfdot = d_fdot / DETECT_MARGIN

    n_req_fc = nperseg_required_for_df(req_df)
    n_req_fdot = nperseg_required_for_dfdot(req_dfdot)

    v_z = local_velocity_profile(ice_volpct, fc)

    dz_fc = stft_delta_z(n_req_fc, v_z)
    dz_fdot = stft_delta_zdot(n_req_fdot, v_z)

    return dict(d_fc=d_fc, d_fdot=d_fdot,
                n_req_fc=n_req_fc, n_req_fdot=n_req_fdot,
                dz_fc=dz_fc, dz_fdot=dz_fdot, v=v_z,
                fc=fc, sr=sr)

_stft_req_cache = {}
def get_stft_requirements(ice_volpct):
    if ice_volpct not in _stft_req_cache:
        _stft_req_cache[ice_volpct] = calc_stft_requirements(ice_volpct)
    return _stft_req_cache[ice_volpct]

# --- 6-G-4. [左:0vol%との差 / 右:必要nperseg] の2列図 ---
def make_stft_requirement_profile(kind):
    """kind = 'centroid' or 'shiftrate'
    左パネル: 0 vol% との差(絶対値)の深さプロファイル
    右パネル: その差を分解するために必要な nperseg の深さプロファイル
    色 = 水氷含有量 (0 vol% は差がゼロなので描画しない)"""
    if kind == 'centroid':
        key_d, key_n = 'd_fc', 'n_req_fc'
        xlabel_left = r'$|f_{c,0\%} - f_{c}|$ [GHz]'
        title = 'Centroid frequency'
        fname = 'stft_requirement_centroid'
    elif kind == 'shiftrate':
        key_d, key_n = 'd_fdot', 'n_req_fdot'
        xlabel_left = r'$|\dot{f}_{0\%} - \dot{f}|$ [GHz/ns]'
        title = 'Shift rate'
        fname = 'stft_requirement_shiftrate'
    else:
        raise CmdInputError(f'unknown kind: {kind}')

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))

    for ii, c in enumerate(ice_contents):
        if c == 0:
            continue                      # 差が 0 (基準) なので除外
        res = get_stft_requirements(c)
        axes[0].plot(res[key_d], z, color=ice_colors[ii], lw=2, label=ice_labels[ii])
        axes[1].plot(res[key_n], z, color=ice_colors[ii], lw=2, label=ice_labels[ii])

    axes[0].set_xlabel(xlabel_left, fontsize=18)
    axes[0].set_xscale('log')
    axes[1].set_xlabel('Required nperseg', fontsize=18)
    axes[1].set_xscale('log')

    for ax in axes:
        ax.set_ylabel('Depth (m)', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.minorticks_on()
        ax.grid(True, which='both', alpha=0.4)
        ax.invert_yaxis()
        ax.legend(loc='best', fontsize=13)

    fig.suptitle(f'{title}: difference from 0 vol% ice and required nperseg\n'
                 f'($f_s$ = {STFT_FS_GHZ:.2f} GHz, dt = {STFT_DT_S:.3e} s, '
                 f'margin = {DETECT_MARGIN:g})', fontsize=15)
    plt.tight_layout()
    base = os.path.join(output_dir_stft, fname)
    fig.savefig(base + '.png', bbox_inches='tight', dpi=200)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    plt.close(fig)
    return base + '.png'

# --- 6-G-5. 横軸 nperseg vs 周波数分解能 ---
def make_nperseg_vs_frequency_resolution():
    """横軸 nperseg, 縦軸 周波数分解能。
    Δf [GHz] (中心周波数用) と Δḟ [GHz/ns] (シフトレート用) を
    左右の縦軸に分けて表示する。"""
    n = NPERSEG_RANGE
    fig, ax = plt.subplots(figsize=(7, 6))

    l1, = ax.plot(n, stft_delta_f_ghz(n), color='k', lw=2,
                  label=r'$\Delta f = f_s/\mathrm{nperseg}$ [GHz]')
    ax.set_xlabel('nperseg', fontsize=18)
    ax.set_ylabel(r'$\Delta f$ [GHz]', fontsize=18)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, which='both', alpha=0.4)

    ax2 = ax.twinx()
    l2, = ax2.plot(n, stft_delta_fdot_ghz_per_ns(n), color='m', lw=2, ls='--',
                   label=r'$\Delta\dot{f} = 2\sqrt{2}(f_s/\mathrm{nperseg})^2$ [GHz/ns]')
    ax2.set_ylabel(r'$\Delta\dot{f}$ [GHz/ns]', fontsize=18, color='m')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', which='major', labelsize=14, colors='m')

    ax.legend(handles=[l1, l2], loc='upper right', fontsize=13)
    ax.set_title(f'$f_s$ = {STFT_FS_GHZ:.2f} GHz (dt = {STFT_DT_S:.3e} s)', fontsize=15)

    plt.tight_layout()
    base = os.path.join(output_dir_stft, 'nperseg_vs_frequency_resolution')
    fig.savefig(base + '.png', bbox_inches='tight', dpi=200)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    plt.close(fig)
    return base + '.png'

# --- 6-G-6. 横軸 nperseg vs 深さ分解能 (eps_r をパラメータ) ---
def make_nperseg_vs_depth_resolution():
    """横軸 nperseg, 縦軸 深さ分解能。
    比誘電率 EPSR_LIST_FOR_DZ ごとに色を変えて描画。
    実線: Δz  = nperseg*dt*v/2      (中心周波数の空間分解能)
    破線: Δż = 1.5*nperseg*dt*v/2  (シフトレートの空間分解能)"""
    n = NPERSEG_RANGE
    fig, ax = plt.subplots(figsize=(7, 6))

    for k, epsr in enumerate(EPSR_LIST_FOR_DZ):
        v = const.c / np.sqrt(epsr)          # [m/s]
        col = EPSR_COLORS[k % len(EPSR_COLORS)]
        ax.plot(n, stft_delta_z(n, v), color=col, lw=2,
                label=rf'$\varepsilon_r$ = {epsr:.1f} ($v$ = {v/1e9*1e0:.3f} m/ns)'.replace('m/ns', 'm/ns'))
        ax.plot(n, stft_delta_zdot(n, v), color=col, lw=1.6, ls='--')

    ax.set_xlabel('nperseg', fontsize=18)
    ax.set_ylabel(r'Depth resolution [m]', fontsize=18)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, which='both', alpha=0.4)

    style_handles = [Line2D([0], [0], color='gray', lw=2, ls='-',
                            label=r'$\Delta z$ (centroid)'),
                     Line2D([0], [0], color='gray', lw=1.6, ls='--',
                            label=r'$\Delta \dot{z}$ (shift rate, $\times 1.5$)')]
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles=h + style_handles, loc='upper left', fontsize=12)
    ax.set_title(f'dt = {STFT_DT_S:.3e} s (fixed)', fontsize=15)

    plt.tight_layout()
    base = os.path.join(output_dir_stft, 'nperseg_vs_depth_resolution')
    fig.savefig(base + '.png', bbox_inches='tight', dpi=200)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    plt.close(fig)
    return base + '.png'

# --- 6-G-7. STFT 要求のサマリ txt ---
def write_stft_summary():
    lines = []
    lines.append("===== STFT parameter requirement (difference from 0 vol% ice) =====")
    lines.append(f"dt = {STFT_DT_S:.4e} s (fixed),  f_s = 1/dt = {STFT_FS_GHZ:.4f} GHz (fixed)")
    lines.append(f"noverlap ratio = {STFT_NOVERLAP_RATIO}, detection margin = {DETECT_MARGIN:g}")
    lines.append("Delta_f      = f_s / nperseg                 [GHz]")
    lines.append("Delta_fdot   = 2*sqrt(2) * (f_s/nperseg)^2   [GHz/ns]  (worst case)")
    lines.append("Delta_z      = nperseg * dt * v / 2          [m]  (v: local phase velocity)")
    lines.append("Delta_zdot   = 1.5 * nperseg * dt * v / 2    [m]")
    lines.append("")

    for zt in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        j = int(np.argmin(np.abs(z - zt)))
        lines.append(f"--- depth = {z[j]:.2f} m ---")
        for c in ice_contents:
            if c == 0:
                continue
            r = get_stft_requirements(c)
            lines.append(
                f"  ice={c:>2d}vol%: |d_fc|={r['d_fc'][j]:8.5f} GHz -> nperseg>={r['n_req_fc'][j]:9.1f}"
                f" (dz={r['dz_fc'][j]:6.3f} m, v={r['v'][j]/1e9:.4f} m/ns)")
            lines.append(
                f"              |d_fdot|={r['d_fdot'][j]:9.6f} GHz/ns -> nperseg>={r['n_req_fdot'][j]:9.1f}"
                f" (dzdot={r['dz_fdot'][j]:6.3f} m)")
        lines.append("")

    text = "\n".join(lines) + "\n"
    fname = os.path.join(output_dir_stft, 'stft_requirement_summary.txt')
    with open(fname, 'w') as fh:
        fh.write(text)
    return fname

# プロット実行
png_sum = make_summary_2x2()
png_re  = make_profile_and_delta(EPS_RE, r"$\varepsilon^{\prime}$", 'permittivity_Re', ref=eps_re_Heiken)
png_im  = make_profile_and_delta(EPS_IM, r"$\varepsilon^{\prime\prime}$", 'permittivity_Im')
png_sig = make_profile_and_delta(SIGMA, r"Conductivity $\sigma_{\rm eff}$ [S/m]", 'conductivity')
png_tan = make_profile_and_delta(TAND, r"$\tan\delta$", 'losstangent', ref=tan_d_heiken)
png_rho = make_density_profile()
png_wtpct = make_ice_wtpct_profile()
png_shift = make_shiftrate_profile()
png_centroid = make_centroid_profile()
png_spectra = [make_spectrum_comparison(c) for c in ice_contents]  # 各氷含有量のスペクトル比較
# ★追加: STFT パラメータ検討
png_req_fc   = make_stft_requirement_profile('centroid')
png_req_fdot = make_stft_requirement_profile('shiftrate')
png_nps_freq = make_nperseg_vs_frequency_resolution()
png_nps_dz   = make_nperseg_vs_depth_resolution()
txt_stft     = write_stft_summary()

# ============================================================
# 7. サマリ txt
# ============================================================
def write_summary():
    lines = []
    lines.append("===== Method A + water-ice mixing =====")
    lines.append(f"FeOTiO2 = {FeOTiO2:.1f} wt%,  sigma_ohmic = 0 "
                 f"(loss carried by Debye poles)")
    lines.append(f"Heiken model: eps'=({HEIKEN_EPS_BASE})^rho, "
                 f"tan_d=10^({HEIKEN_TAND_A}*FeOTiO2 + {HEIKEN_TAND_B}*rho - {HEIKEN_TAND_C}), "
                 f"anchor={ANCHOR_FREQ/1e6:.0f} MHz")
    lines.append(f"2-pole Debye: DE1={DEBYE_DE1}, TAU1={DEBYE_TAU1:.4e}, "
                 f"DE2={DEBYE_DE2}, TAU2={DEBYE_TAU2:.4e}")
    lines.append(f"Ice (Evans1965): eps' = {EPS_ICE_RE}, "
                 f"eps'' = {EPS_ICE_IM:.3e}  (Maxwell-Garnett mixing)")
    lines.append(f"Ice contents [vol%]: {ice_contents}")
    lines.append(f"Frequencies: {freq_labels}")
    lines.append("")

    # 代表深さの要約
    lines.append("--- Representative depths ---")
    for zt in [0.0, 1.5, 3.0]:
        j = int(np.argmin(np.abs(z - zt)))
        lines.append(f"depth={z[j]:.2f} m, rho={rho[j]:.3f}, "
                     f"Heiken eps'={eps_re_Heiken[j]:.4f}, "
                     f"tand_H={tan_d_heiken[j]:.5f}")
        for ii, c in enumerate(ice_contents):
            for fi in range(n_freq):
                lines.append(f"   ice={c:>2d}vol% {freq_labels[fi]:>8s}: "
                             f"eps'={EPS_RE[ii,fi,j]:.4f}  "
                             f"eps''={EPS_IM[ii,fi,j]:.5f}  "
                             f"sigma_eff={SIGMA[ii,fi,j]:.4e}  "
                             f"tand={TAND[ii,fi,j]:.5f}")
        lines.append("")

    # f_ice [vol%] -> wt% 変換テーブル(代表深さ)
    lines.append(f"--- f_ice [vol%] to wt% conversion (rho_ice={RHO_ICE}) ---")
    for zt in [0.0, 1.5, 3.0]:
        j = int(np.argmin(np.abs(z - zt)))
        lines.append(f"depth={z[j]:.2f} m, rho_reg={rho[j]:.3f}:")
        for c in ice_contents:
            if c == 0:
                continue
            f_vol = c / 100.0
            wtpct = 100.0 * f_vol * RHO_ICE / (f_vol * RHO_ICE + (1.0 - f_vol) * rho[j])
            lines.append(f"   ice={c:>2d}vol% -> {wtpct:6.3f} wt%")
        lines.append("")

    # 0vol%との相対差(代表深さ, 1.25GHz)
    lines.append("--- Relative difference vs 0 vol% ice "
                 "(at 1.25 GHz, representative depths) ---")
    fi_ref = 1   # 1.25 GHz
    for zt in [0.0, 1.5, 3.0]:
        j = int(np.argmin(np.abs(z - zt)))
        lines.append(f"depth={z[j]:.2f} m:")
        for ii, c in enumerate(ice_contents):
            if c == 0:
                continue
            d_re = abs(EPS_RE[ii,fi_ref,j]-EPS_RE[0,fi_ref,j])/EPS_RE[0,fi_ref,j]*100
            d_im = abs(EPS_IM[ii,fi_ref,j]-EPS_IM[0,fi_ref,j])/EPS_IM[0,fi_ref,j]*100
            d_td = abs(TAND[ii,fi_ref,j]-TAND[0,fi_ref,j])/TAND[0,fi_ref,j]*100
            lines.append(f"   ice={c:>2d}vol%: d_eps'={d_re:6.3f}%  "
                         f"d_eps''={d_im:6.3f}%  d_tand={d_td:6.3f}%")
        lines.append("")

    # 全深さ完全テーブル(1.25GHz, 全氷含有量)
    lines.append("--- Full depth table at 1.25 GHz ---")
    header = f"{'depth[m]':>9s} {'rho':>7s}"
    for c in ice_contents:
        header += (f" {'eps_'+str(c):>9s} {'epsIm_'+str(c):>11s}"
                   f" {'sig_'+str(c):>11s} {'tand_'+str(c):>10s}")
    lines.append(header)
    fi_ref = 1
    for j in range(Nz):
        row = f"{z[j]:9.3f} {rho[j]:7.3f}"
        for ii in range(n_ice):
            row += (f" {EPS_RE[ii,fi_ref,j]:9.4f} {EPS_IM[ii,fi_ref,j]:11.5f}"
                    f" {SIGMA[ii,fi_ref,j]:11.4e} {TAND[ii,fi_ref,j]:10.5f}")
        lines.append(row)

    text = "\n".join(lines) + "\n"
    fname = os.path.join(output_base_dir, 'summary.txt')
    with open(fname, 'w') as fh:
        fh.write(text)
    # コンソールには代表部分のみ
    print("\n".join(lines[:40]))
    return fname

txt = write_summary()

print("\nsaved figures:")
# 出力リストに png_centroid とスペクトル比較図、STFT パラメータ図を追加
for p in [png_sum, png_re, png_im, png_sig, png_tan, png_rho, png_wtpct,
          png_shift, png_centroid] + png_spectra + [
          png_req_fc, png_req_fdot, png_nps_freq, png_nps_dz]:
    print("  ", p)
print("saved summary:", txt)
print("saved STFT summary:", txt_stft)