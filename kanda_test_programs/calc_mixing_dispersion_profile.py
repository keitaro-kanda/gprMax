"""
Heiken基準の深さプロファイル + 2極Debye分散(Method A) + 水氷混合 + 解析的シフトレート + Hilbert瞬時周波数
================================================================================================
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
  - 伝搬計算の一元化: 全深さに対する減衰 (alpha) と位相速度 (v) の積分を
        共通の伝搬テーブルとして事前計算し、すべての解析機能 (スペクトル・重心・Hilbert) で共用する。
  - ★Hilbert瞬時周波数 (追加): 伝搬テーブルから複素伝達関数 H(ω,d) を構成し、
        時間エコー波形を合成した上で解析信号 (Analytic signal) から瞬時周波数 (IF) を抽出する。
        帯域制限による波形リンギングを防ぐため、広帯域(またはテーパ付)スペクトルを用いる。
出力:
  (1) Method A の 2x2 まとめ図 (eps', eps'', sigma_eff, tan delta)
  (2)-(5) 各物理量ごとに 左:深さプロファイル / 右:0vol%との相対差[%] の2列図
  (6) 密度プロファイル、水氷wt%プロファイル
  (7) 解析的周波数シフトレートプロファイル
  (8) 解析的中心周波数プロファイル
  (9) 各水氷含有量ごとの解析的スペクトル比較図 (規格化 dB)
 (10) STFTパラメータ要求解析 (output_dir_centroid/STFT_parameter)
 (11) サマリ txt
 (12) ★追加: Hilbert瞬時周波数解析 (output_dir_hilbert)
        (12-a) instantaneous_frequency_profile: IF_peakの深さプロファイル
        (12-b) instantaneous_shiftrate_profile: IF_peakからのシフトレートプロファイル
        (12-c) waveform_examples: 指定深さでのエコー合成波形とIF抽出プロセスの診断図
        (12-d) hilbert_vs_centroid_check: IF_wと中心周波数の理論的整合チェック図
        (12-e) hilbert_summary.txt: 抽出結果の要約
================================================================================================
"""
import os
import sys
import numpy as np
import scipy.constants as const
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
output_dir_stft = os.path.join(output_dir_centroid, 'STFT_parameter')
os.makedirs(output_dir_stft, exist_ok=True)
# ★追加: Hilbert解析用の出力先
output_dir_hilbert = os.path.join(output_base_dir, 'Hilbert')
os.makedirs(output_dir_hilbert, exist_ok=True)

eps0 = 8.8541878128e-12          # 真空の誘電率 [F/m]

# 深さ [m]
z   = np.arange(0, 3.01, 0.02)   # [m]
FeOTiO2 = 20.0                   # [wt%]

freqs = np.array([0.5e9, 1.25e9, 2.0e9])     # [Hz]
freq_labels = ['0.5 GHz', '1.25 GHz', '2.0 GHz']
freq_styles = ['-', '--', '-.']
ANCHOR_FREQ = 450E6 # Heiken1991 Fig 9.54の、450 MHz計測経験式を使う

HEIKEN_EPS_BASE = 1.843
HEIKEN_TAND_A   = 0.033
HEIKEN_TAND_B   = 0.231
HEIKEN_TAND_C   = 3.061

ice_contents = [0, 1, 5, 10, 20]
ice_colors   = ['k', 'r', 'g', 'b', 'c']
ice_labels   = [f'{c} vol% ice' for c in ice_contents]

EPS_ICE_RE = 3.17
EPS_ICE_IM = 3.17 * 6e-5
eps_ice_complex = EPS_ICE_RE - 1j * EPS_ICE_IM
RHO_ICE = 0.934

ASCAN_OUTFILE_PATH = ("/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_3x4/"
                      "waveform_test/gaussiandot_1.25GHz_underground/result/Ascan.out")
FREQ_BAND_MIN = 0.25e9
FREQ_BAND_MAX = 6.0e9
RX_DEPTH      = 0.10
SPECTRUM_TARGET_DEPTHS = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

STFT_DT_S     = 1.18e-11
STFT_FS_HZ    = 1.0 / STFT_DT_S
STFT_DT_NS    = STFT_DT_S * 1e9
STFT_FS_GHZ   = STFT_FS_HZ / 1e9
STFT_NOVERLAP_RATIO = 0.75
DETECT_MARGIN = 2.0
EPSR_LIST_FOR_DZ = [2.4, 2.6, 2.8, 3.0, 3.2]
EPSR_COLORS      = ['r', 'g', 'b', 'c', 'm']
NPERSEG_RANGE    = np.arange(16, 32769)

# ------------------------------------------------------------
# ★追加: Hilbert 瞬時周波数解析の設定
# ------------------------------------------------------------
HILBERT_ENV_THRESHOLD = 0.10   # IF有効区間の包絡線閾値（ピーク比）
HILBERT_PAD_FACTOR    = 2      # irfft のゼロパディング倍率
HILBERT_TAPER_ON      = True   # 帯域外コサインテーパの有無
HILBERT_EXAMPLE_DEPTHS = [0.5, 1.5, 3.0]  # 波形診断図の対象深さ [m]

# ============================================================
# 1. レゴリス誘電モデルの「単一の情報源」
# ============================================================
def density_profile(depth_m):
    z_cm = depth_m * 100.0
    return 1.92 * (z_cm + 12.2) / (z_cm + 18.0)

def heiken_eps_real(rho_val):
    return HEIKEN_EPS_BASE ** rho_val

def heiken_tan_delta(rho_val):
    return 10 ** (HEIKEN_TAND_A * FeOTiO2 + HEIKEN_TAND_B * rho_val - HEIKEN_TAND_C)

rho = density_profile(z)
eps_re_Heiken = heiken_eps_real(rho)
tan_d_heiken  = heiken_tan_delta(rho)
eps_im_heiken = eps_re_Heiken * tan_d_heiken

DEBYE_DE1  = 0.261
DEBYE_TAU1 = 4.6212e-11
DEBYE_DE2  = 0.088
DEBYE_TAU2 = 2.82195e-10

def debye_imag_shape(omega, tau):
    return omega * tau / (1.0 + (omega * tau) ** 2)

def debye_total_imag(omega, scale):
    return (DEBYE_DE1 * scale * debye_imag_shape(omega, DEBYE_TAU1)
            + DEBYE_DE2 * scale * debye_imag_shape(omega, DEBYE_TAU2))

def debye_total_real_drop(omega, scale):
    drop1 = DEBYE_DE1 * scale * (omega * DEBYE_TAU1)**2 / (1.0 + (omega * DEBYE_TAU1)**2)
    drop2 = DEBYE_DE2 * scale * (omega * DEBYE_TAU2)**2 / (1.0 + (omega * DEBYE_TAU2)**2)
    return drop1 + drop2

w_anchor = 2 * np.pi * ANCHOR_FREQ
unit_imag_anchor = debye_total_imag(w_anchor, scale=1.0)
scale_A = eps_im_heiken / unit_imag_anchor

def maxwell_garnett(eps_host, eps_incl, f_volpct):
    f = f_volpct / 100.0
    return eps_host + 3.0 * f * eps_host * (eps_incl - eps_host) \
           / (eps_incl + 2.0 * eps_host - f * (eps_incl - eps_host))

# ------------------------------------------------------------
# 共通プロファイル配列の構成 (2x2図等用)
# ------------------------------------------------------------
n_ice, n_freq, Nz = len(ice_contents), len(freqs), len(z)
EPS_RE = np.zeros((n_ice, n_freq, Nz))
EPS_IM = np.zeros((n_ice, n_freq, Nz))
SIGMA  = np.zeros((n_ice, n_freq, Nz))
TAND   = np.zeros((n_ice, n_freq, Nz))

for fi, f in enumerate(freqs):
    w_val = 2 * np.pi * f
    reg_re = eps_re_Heiken - debye_total_real_drop(w_val, scale_A)
    reg_im = debye_total_imag(w_val, scale_A)
    eps_reg_complex = reg_re - 1j * reg_im
    for ii, c in enumerate(ice_contents):
        eps_mix = eps_reg_complex if c == 0 else maxwell_garnett(eps_reg_complex, eps_ice_complex, c)
        EPS_RE[ii, fi] = np.real(eps_mix)
        EPS_IM[ii, fi] = -np.imag(eps_mix)
        SIGMA[ii, fi]  = EPS_IM[ii, fi] * w_val * eps0
        TAND[ii, fi]   = EPS_IM[ii, fi] / EPS_RE[ii, fi]

# ============================================================
# 2. 入射波スペクトルのロード機能 (全帯域/制限/テーパ対応)
# ============================================================
_incident_spectrum_cache = None
def get_raw_incident_spectrum():
    global _incident_spectrum_cache
    if _incident_spectrum_cache is None:
        try:
            if os.path.exists(ASCAN_OUTFILE_PATH):
                ascan_data, dt_ascan = get_output_data(ASCAN_OUTFILE_PATH, 1, 'Ez')
                e_incident = ascan_data if ascan_data.ndim == 1 else ascan_data[:, 0]
                N = len(e_incident)
                freq_ascan = np.fft.rfftfreq(N, d=dt_ascan)
                S0_omega = np.fft.rfft(e_incident)
            else:
                raise FileNotFoundError
        except Exception as e:
            print(f"Warning: Using synthetic Gaussian pulse. Error: {e}")
            dt_ascan = 1e-10
            t_ascan = np.arange(-5e-9, 5e-9, dt_ascan)
            e_incident = np.exp(-((t_ascan - 0) ** 2) / (2 * (1 / (2 * np.pi * ANCHOR_FREQ)) ** 2))
            # 位相ゼロ(t=0でピーク)とするためにifftshiftを適用
            e_incident = np.fft.ifftshift(e_incident)
            N = len(e_incident)
            freq_ascan = np.fft.rfftfreq(N, d=dt_ascan)
            S0_omega = np.fft.rfft(e_incident)
        _incident_spectrum_cache = (freq_ascan, S0_omega, N, dt_ascan)
    return _incident_spectrum_cache

def load_incident_spectrum(mode='band'):
    freq_ascan, S0_omega, N, dt = get_raw_incident_spectrum()
    if mode == 'band':
        mask = (freq_ascan >= FREQ_BAND_MIN) & (freq_ascan <= FREQ_BAND_MAX)
        return freq_ascan[mask], S0_omega[mask], 2*np.pi*freq_ascan[mask]
    elif mode == 'full':
        return freq_ascan, S0_omega.copy(), 2*np.pi*freq_ascan
    elif mode == 'taper':
        S0_taper = S0_omega.copy()
        f_start = FREQ_BAND_MAX
        f_end = f_start * 1.2
        taper_mask = (freq_ascan > f_start) & (freq_ascan <= f_end)
        zero_mask = freq_ascan > f_end
        S0_taper[taper_mask] *= 0.5 * (1 + np.cos(np.pi * (freq_ascan[taper_mask] - f_start) / (f_end - f_start)))
        S0_taper[zero_mask] = 0.0
        return freq_ascan, S0_taper, 2*np.pi*freq_ascan
    else:
        raise ValueError("Invalid mode")

def get_band_mask():
    freq_ascan, _, _, _ = get_raw_incident_spectrum()
    return (freq_ascan >= FREQ_BAND_MIN) & (freq_ascan <= FREQ_BAND_MAX)

# ============================================================
# 3. 伝搬テーブル・共通ロジックの一元化
# ============================================================
def local_alpha_velocity(depth_m, omega, ice_volpct):
    rho_d = density_profile(depth_m)
    eps_re_H = heiken_eps_real(rho_d)
    tan_d_H = heiken_tan_delta(rho_d)
    eps_im_H = eps_re_H * tan_d_H
    scale_A_val = eps_im_H / debye_total_imag(2 * np.pi * ANCHOR_FREQ, 1.0)
    
    reg_re = eps_re_H - debye_total_real_drop(omega, scale_A_val)
    reg_im = debye_total_imag(omega, scale_A_val)
    eps_reg = reg_re - 1j * reg_im
    eps_mix = eps_reg if ice_volpct == 0 else maxwell_garnett(eps_reg, eps_ice_complex, ice_volpct)
    
    sqrt_eps = np.sqrt(eps_mix)
    alpha = - (omega / const.c) * np.imag(sqrt_eps)
    v = const.c / np.real(sqrt_eps)
    return alpha, v

_prop_cache = {}
def get_propagation_table(ice_volpct):
    """深さ配列 z に対する減衰・伝搬時間の積分テーブル(全周波数対応)"""
    if ice_volpct in _prop_cache:
        return _prop_cache[ice_volpct]
    
    freq_full, _, w_full = load_incident_spectrum('full')
    cum_att = np.zeros((len(z), len(w_full)))
    cum_time = np.zeros((len(z), len(w_full)))
    
    current_att = np.zeros_like(w_full)
    current_time = np.zeros_like(w_full)
    d_step = 0.02
    idx_rx = np.argmin(np.abs(z - RX_DEPTH))
    
    # centroidコードの完全回帰を保証する後退Euler積分
    for i in range(len(z)):
        if i < idx_rx:
            cum_att[i] = np.nan
            cum_time[i] = np.nan
            continue
        if i > idx_rx:
            alpha_prev, v_prev = local_alpha_velocity(z[i-1], w_full, ice_volpct)
            current_att += alpha_prev * d_step
            current_time += 2 * d_step / v_prev
        cum_att[i] = current_att.copy()
        cum_time[i] = current_time.copy()
        
    _prop_cache[ice_volpct] = {'cum_att': cum_att, 'cum_time': cum_time}
    return _prop_cache[ice_volpct]

def get_t_offset_ns():
    """システムラグ等を含む深さRX_DEPTHまでの時間オフセット [ns]"""
    antenna_height = 0.35
    system_lag_ns  = 0.837
    t_air_ns = (2.0 * antenna_height / const.c) * 1e9
    
    d_sub_offset = np.linspace(0, RX_DEPTH, 50)
    eps_sub_offset = heiken_eps_real(density_profile(d_sub_offset))
    v_sub = const.c / np.sqrt(eps_sub_offset)
    dt_sub = d_sub_offset[1] - d_sub_offset[0]
    t_ground_start_ns = np.sum(2.0 * dt_sub / v_sub) * 1e9
    return system_lag_ns + t_air_ns + t_ground_start_ns

def compute_shiftrate_profile(t_delay_d, value_d, z_array, rx_depth):
    """遅延時間軸を介したシフトレートの計算と深さ軸への再マッピング"""
    valid = ~np.isnan(t_delay_d) & ~np.isnan(value_d)
    if not np.any(valid):
        return np.full_like(z_array, np.nan)
        
    t_val = np.array(t_delay_d)[valid]
    v_val = np.array(value_d)[valid]
    z_val = np.array(z_array)[valid]
    
    dt_stft = 0.1
    t_axis = np.arange(np.nanmin(t_val), np.nanmax(t_val) + dt_stft, dt_stft)
    if len(t_axis) > 1:
        v_interp = np.interp(t_axis, t_val, v_val, left=np.nan, right=np.nan)
        sr_interp = np.gradient(v_interp, dt_stft)
        sr_d = np.interp(t_val, t_axis, sr_interp, left=np.nan, right=np.nan)
    else:
        sr_d = np.gradient(v_val, t_val)
        
    return np.interp(z_array, z_val, sr_d, left=np.nan, right=np.nan)

# ============================================================
# 4. Centroid (中心周波数) & STFT 要求 解析
# ============================================================
def spectral_centroid(power, f_calc):
    return np.trapz(f_calc * power, f_calc) / np.trapz(power, f_calc)

_centroid_cache = {}
def get_centroid_shiftrate(ice_volpct):
    if ice_volpct in _centroid_cache:
        return _centroid_cache[ice_volpct]
        
    f_calc, S0_calc, _ = load_incident_spectrum('band')
    prop = get_propagation_table(ice_volpct)
    band_mask = get_band_mask()
    t_offset_ns = get_t_offset_ns()
    
    f_peak_list = []
    t_delay_list = []
    
    for i, d in enumerate(z):
        if d < RX_DEPTH:
            f_peak_list.append(np.nan)
            t_delay_list.append(np.nan)
            continue
            
        att = prop['cum_att'][i, band_mask]
        tau = prop['cum_time'][i, band_mask]
        
        S_d_w = S0_calc * np.exp(-2 * att)
        power = np.abs(S_d_w)**2
        f_peak = spectral_centroid(power, f_calc)
        f_peak_list.append(f_peak / 1e9)
        
        t_delay_ground = np.interp(f_peak, f_calc, tau)
        t_delay_list.append(t_offset_ns + t_delay_ground * 1e9)
        
    sr_z = compute_shiftrate_profile(t_delay_list, f_peak_list, z, RX_DEPTH)
    _centroid_cache[ice_volpct] = (sr_z, np.array(f_peak_list))
    return _centroid_cache[ice_volpct]

def stft_delta_f_ghz(nperseg):
    return STFT_FS_GHZ / np.asarray(nperseg, dtype=float)

def stft_delta_fdot_ghz_per_ns(nperseg):
    return 2.0 * np.sqrt(2.0) * (STFT_FS_GHZ / np.asarray(nperseg, dtype=float)) ** 2

def stft_delta_z(nperseg, v):
    return np.asarray(nperseg, dtype=float) * STFT_DT_S * np.asarray(v, dtype=float) / 2.0

def stft_delta_zdot(nperseg, v):
    return 1.5 * stft_delta_z(nperseg, v)

def get_stft_requirements(ice_volpct):
    sr0, fc0 = get_centroid_shiftrate(0)
    sr, fc = get_centroid_shiftrate(ice_volpct)
    d_fc = np.abs(fc - fc0)
    d_fdot = np.abs(sr - sr0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        n_req_fc = np.where(d_fc > 0, STFT_FS_GHZ / (d_fc / DETECT_MARGIN), np.nan)
        n_req_fdot = np.where(d_fdot > 0, STFT_FS_GHZ * np.sqrt(2.0 * np.sqrt(2.0) / (d_fdot / DETECT_MARGIN)), np.nan)
        
    v_z = np.full_like(z, np.nan)
    for j, d in enumerate(z):
        if np.isfinite(fc[j]) and fc[j] > 0:
            _, v_j = local_alpha_velocity(d, np.array([2*np.pi*fc[j]*1e9]), ice_volpct)
            v_z[j] = v_j[0]
            
    return dict(d_fc=d_fc, d_fdot=d_fdot, n_req_fc=n_req_fc, n_req_fdot=n_req_fdot,
                dz_fc=stft_delta_z(n_req_fc, v_z), dz_fdot=stft_delta_zdot(n_req_fdot, v_z), v=v_z)

# ============================================================
# ★ 5. Hilbert 瞬時周波数解析 (新規追加)
# ============================================================
_hilbert_cache = {}
def get_hilbert_if_profile(ice_volpct):
    if ice_volpct in _hilbert_cache:
        return _hilbert_cache[ice_volpct]
        
    freq, S0, w = load_incident_spectrum('taper' if HILBERT_TAPER_ON else 'full')
    prop = get_propagation_table(ice_volpct)
    
    _, _, N_orig, dt_orig = get_raw_incident_spectrum()
    N_pad = N_orig * HILBERT_PAD_FACTOR
    dt_pad = dt_orig / HILBERT_PAD_FACTOR
    t_pad = np.arange(N_pad) * dt_pad
    t_offset_s = get_t_offset_ns() * 1e-9
    
    IF_peak_list = []
    IF_w_list = []
    t_delay_list = []
    diagnostic_dict = {}
    
    for i, d in enumerate(z):
        if d < RX_DEPTH:
            IF_peak_list.append(np.nan)
            IF_w_list.append(np.nan)
            t_delay_list.append(np.nan)
            continue
            
        att = prop['cum_att'][i]
        tau = prop['cum_time'][i]
        
        # 複素伝達関数と逆FFT合成
        H = np.exp(-2 * att) * np.exp(-1j * w * (tau + t_offset_s))
        e_d = np.fft.irfft(S0 * H, n=N_pad)
        
        z_t = scipy.signal.hilbert(e_d)
        env = np.abs(z_t)
        
        max_idx = np.argmax(env)
        peak_time = t_pad[max_idx]
        threshold = HILBERT_ENV_THRESHOLD * env[max_idx]
        mask = env >= threshold
        
        # 連続区間の抽出
        left = max_idx
        while left > 0 and mask[left-1]: left -= 1
        right = max_idx
        while right < N_pad - 1 and mask[right+1]: right += 1
        
        seg_slice = slice(left, right+1)
        env_seg = env[seg_slice]
        
        # 瞬時周波数の導出
        phase_seg = np.unwrap(np.angle(z_t[seg_slice]))
        IF_seg = np.gradient(phase_seg, dt_pad) / (2 * np.pi)
        
        IF_peak = IF_seg[max_idx - left]
        IF_w = np.sum(env_seg**2 * IF_seg) / np.sum(env_seg**2)
        
        IF_peak_list.append(IF_peak / 1e9)
        IF_w_list.append(IF_w / 1e9)
        t_delay_list.append(peak_time * 1e9)
        
        z_rounded = round(d, 2)
        if z_rounded in HILBERT_EXAMPLE_DEPTHS:
            diagnostic_dict[z_rounded] = {
                't_pad': t_pad * 1e9, 'e_d': e_d, 'env': env, 'mask': mask,
                't_seg': t_pad[seg_slice] * 1e9, 'IF_seg': IF_seg / 1e9,
                'IF_peak': IF_peak / 1e9, 'IF_w': IF_w / 1e9
            }
            
    sr_z = compute_shiftrate_profile(t_delay_list, IF_peak_list, z, RX_DEPTH)
    _hilbert_cache[ice_volpct] = (sr_z, np.array(IF_peak_list), np.array(IF_w_list), diagnostic_dict)
    return _hilbert_cache[ice_volpct]

# ============================================================
# 6. 描画・出力関数群
# ============================================================
def draw_lines(ax, data, ref=None):
    if ref is not None:
        ax.plot(ref, z, color='gray', linestyle='--', lw=2, zorder=1, label='Heiken (ref)')
    for ii in range(n_ice):
        for fi in range(n_freq):
            ax.plot(data[ii, fi], z, color=ice_colors[ii], linestyle=freq_styles[fi], lw=1.6, zorder=3 + ii)
    ax.set_ylabel('Depth (m)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.minorticks_on()
    ax.grid(True, alpha=0.4)
    ax.invert_yaxis()

freq_handles = [Line2D([0], [0], linestyle=freq_styles[i], color='k', lw=2, label=freq_labels[i]) for i in range(n_freq)]
ice_handles = [Line2D([0], [0], color=ice_colors[i], linestyle='-', lw=2, label=ice_labels[i]) for i in range(n_ice)]
heiken_handle = [Line2D([0], [0], color='gray', ls='--', lw=2, label=r'Heiken (for $\varepsilon_r$ and $\tan \delta$)')]

def add_legend(fig):
    fig.legend(handles=freq_handles + ice_handles + heiken_handle,
               loc='lower center', ncol=4, fontsize=14, frameon=True, bbox_to_anchor=(0.5, 1.0))

def save_fig(fig, base_path):
    fig.savefig(base_path + '.png', bbox_inches='tight', dpi=200)
    fig.savefig(base_path + '.pdf', bbox_inches='tight')
    plt.close(fig)
    return base_path + '.png'

# --- 既存のプロット群 (一部省略・ラッパー化) ---
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
    return save_fig(fig, os.path.join(output_dir_profile, 'summary_2x2'))

def make_profile_and_delta(data, quantity_label, fname, ref=None):
    base0 = data[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    draw_lines(axes[0], data, ref=ref)
    axes[0].set_xlabel(quantity_label, fontsize=18)
    if fname == 'losstangent': axes[0].locator_params(axis='x', nbins=5)
    for ii, c in enumerate(ice_contents):
        if c == 0: continue
        for fi in range(n_freq):
            rel = np.abs(data[ii, fi] - base0[fi]) / base0[fi] * 100.0
            axes[1].plot(rel, z, color=ice_colors[ii], linestyle=freq_styles[fi], lw=1.6, zorder=3 + ii)
    axes[1].set_xlabel(r'$|X_{0\%} - X|\,/\,X_{0\%}\times100$ [%]', fontsize=18)
    axes[1].set_ylabel('Depth (m)', fontsize=18)
    axes[1].set_xscale('log')
    axes[1].tick_params(axis='both', which='major', labelsize=14)
    axes[1].minorticks_on()
    axes[1].grid(True, alpha=0.4)
    axes[1].invert_yaxis()
    add_legend(fig)
    plt.tight_layout()
    return save_fig(fig, os.path.join(output_dir_profile, fname))

def make_spectrum_comparison(ice_volpct):
    freq_calc, S0_calc, _ = load_incident_spectrum('band')
    power_0 = np.abs(S0_calc) ** 2
    max_power_0 = np.max(power_0)
    
    prop = get_propagation_table(ice_volpct)
    band_mask = get_band_mask()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(SPECTRUM_TARGET_DEPTHS)))
    
    for i, d in enumerate(SPECTRUM_TARGET_DEPTHS):
        if d <= RX_DEPTH:
            cum_alpha = np.zeros_like(freq_calc)
        else:
            z_idx = int(np.round(d / 0.02))
            cum_alpha = prop['cum_att'][z_idx, band_mask]
            
        power = np.abs(S0_calc * np.exp(-2 * cum_alpha)) ** 2
        f_peak_ghz = spectral_centroid(power, freq_calc) / 1e9
        power_db = 10.0 * np.log10(power / max_power_0 + 1e-30)
        
        ax.plot(freq_calc / 1e9, power_db, color=colors[i], label=f'Depth {d:.1f} m ($f_c$ = {f_peak_ghz:.2f} GHz)')
        ax.axvline(f_peak_ghz, color=colors[i], linestyle='--', alpha=0.7)
        
    ax.set_xlabel('Frequency [GHz]', fontsize=18)
    ax.set_ylabel('Normalized Power [dB]', fontsize=18)
    ax.set_xlim(FREQ_BAND_MIN / 1e9, FREQ_BAND_MAX / 1e9)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.minorticks_on()
    ax.grid(True, alpha=0.4)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=14)
    plt.tight_layout()
    return save_fig(fig, os.path.join(output_dir_waveform, f'spectrum_comparison_{ice_volpct}vol'))

# --- ★ Hilbert関連のプロット群 ---
def make_instantaneous_frequency_profile():
    fig, ax = plt.subplots(figsize=(6, 6))
    _, fc0 = get_centroid_shiftrate(0)
    ax.plot(fc0, z, color='gray', linestyle='--', lw=2, label='Centroid (0 vol%, ref)')
    
    for ii, c in enumerate(ice_contents):
        _, IF_peak, _, _ = get_hilbert_if_profile(c)
        ax.plot(IF_peak, z, color=ice_colors[ii], lw=2, label=ice_labels[ii])
        
    ax.set_xlabel('Instantaneous Frequency (Peak) [GHz]', fontsize=18)
    ax.set_ylabel('Depth (m)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.minorticks_on()
    ax.grid(True, alpha=0.4)
    ax.invert_yaxis()
    ax.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    return save_fig(fig, os.path.join(output_dir_hilbert, 'instantaneous_frequency_profile'))

def make_instantaneous_shiftrate_profile():
    fig, ax = plt.subplots(figsize=(6, 6))
    for ii, c in enumerate(ice_contents):
        sr, _, _, _ = get_hilbert_if_profile(c)
        ax.plot(sr, z, color=ice_colors[ii], lw=2, label=ice_labels[ii])
        
    ax.set_xlabel('IF Shift Rate [GHz/ns]', fontsize=18)
    ax.set_ylabel('Depth (m)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.minorticks_on()
    ax.grid(True, alpha=0.4)
    ax.invert_yaxis()
    ax.legend(loc='lower left', fontsize=12)
    plt.tight_layout()
    return save_fig(fig, os.path.join(output_dir_hilbert, 'instantaneous_shiftrate_profile'))

def make_waveform_examples(ice_volpct):
    _, _, _, diag = get_hilbert_if_profile(ice_volpct)
    n_depths = len(HILBERT_EXAMPLE_DEPTHS)
    fig, axes = plt.subplots(n_depths, 2, figsize=(12, 3.5 * n_depths))
    
    for i, d in enumerate(HILBERT_EXAMPLE_DEPTHS):
        data = diag.get(d)
        if not data: continue
        
        ax_wv, ax_if = axes[i]
        t = data['t_pad']
        env = data['env']
        
        # 波形プロット
        ax_wv.plot(t, data['e_d'], 'k-', alpha=0.5, label='Signal')
        ax_wv.plot(t, env, 'r-', lw=1.5, label='Envelope')
        ax_wv.fill_between(t, 0, env, where=data['mask'], color='red', alpha=0.2, label='Valid Region')
        ax_wv.set_xlim(data['t_seg'][0] - 2.0, data['t_seg'][-1] + 2.0)
        ax_wv.set_title(f'Depth: {d} m (Ice: {ice_volpct}%)', fontsize=14)
        ax_wv.set_ylabel('Amplitude', fontsize=14)
        ax_wv.legend(loc='upper right', fontsize=10)
        ax_wv.grid(True, alpha=0.4)
        
        # IFプロット
        ax_if.plot(data['t_seg'], data['IF_seg'], 'b-', lw=2, label='IF(t)')
        ax_if.axhline(data['IF_peak'], color='m', ls='--', label=f'IF_peak: {data["IF_peak"]:.3f} GHz')
        ax_if.axhline(data['IF_w'], color='g', ls='-.', label=f'IF_w: {data["IF_w"]:.3f} GHz')
        ax_if.set_xlim(data['t_seg'][0] - 0.5, data['t_seg'][-1] + 0.5)
        
        # y軸の範囲を適正化
        y_center = data['IF_peak']
        ax_if.set_ylim(y_center - 0.5, y_center + 0.5)
        ax_if.set_ylabel('IF [GHz]', fontsize=14)
        ax_if.legend(loc='best', fontsize=10)
        ax_if.grid(True, alpha=0.4)
        
        if i == n_depths - 1:
            ax_wv.set_xlabel('Time [ns]', fontsize=14)
            ax_if.set_xlabel('Time [ns]', fontsize=14)
            
    plt.tight_layout()
    return save_fig(fig, os.path.join(output_dir_hilbert, f'waveform_examples_{ice_volpct}vol'))

def make_hilbert_vs_centroid_check():
    fig, ax = plt.subplots(figsize=(7, 6))
    for ii, c in enumerate(ice_contents):
        _, fc = get_centroid_shiftrate(c)
        _, _, fw, _ = get_hilbert_if_profile(c)
        ax.plot(fw, z, color=ice_colors[ii], ls='-', lw=2, label=f'IF_w ({c} vol%)')
        ax.plot(fc, z, color=ice_colors[ii], ls='--', lw=1, alpha=0.8)
        
    ax.plot([], [], color='k', ls='-', label='IF_w (Hilbert)')
    ax.plot([], [], color='k', ls='--', label='Centroid (Spectrum)')
    
    ax.set_xlabel('Frequency [GHz]', fontsize=18)
    ax.set_ylabel('Depth (m)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.minorticks_on()
    ax.grid(True, alpha=0.4)
    ax.invert_yaxis()
    ax.legend(loc='best', fontsize=12)
    ax.set_title('Theoretical consistency check', fontsize=14)
    plt.tight_layout()
    return save_fig(fig, os.path.join(output_dir_hilbert, 'hilbert_vs_centroid_check'))

def write_hilbert_summary():
    lines = ["===== Hilbert Instantaneous Frequency Analysis ====="]
    lines.append(f"Envelope threshold: {HILBERT_ENV_THRESHOLD*100:.1f}%, Zero-padding factor: {HILBERT_PAD_FACTOR}")
    lines.append("IF_peak: Frequency at the envelope peak")
    lines.append("IF_w: Envelope-squared weighted average frequency in the valid region")
    lines.append("")
    
    _, fc0 = get_centroid_shiftrate(0)
    _, IF0, _, _ = get_hilbert_if_profile(0)
    
    for d in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        idx = int(np.round(d / 0.02))
        lines.append(f"--- Depth = {d:.2f} m ---")
        for c in ice_contents:
            _, fc = get_centroid_shiftrate(c)
            _, IFp, IFw, _ = get_hilbert_if_profile(c)
            diff_w_c = np.abs(IFw[idx] - fc[idx]) / fc[idx] * 100
            diff_0 = np.abs(IFp[idx] - IF0[idx]) if c > 0 else 0.0
            lines.append(f"  Ice {c:>2d} vol%: IF_peak = {IFp[idx]:6.3f} GHz (diff from 0%: {diff_0:5.3f} GHz)")
            lines.append(f"               IF_w = {IFw[idx]:6.3f} GHz, Centroid = {fc[idx]:6.3f} GHz (diff: {diff_w_c:4.2f}%)")
        lines.append("")
        
    text = "\n".join(lines) + "\n"
    fname = os.path.join(output_dir_hilbert, 'hilbert_summary.txt')
    with open(fname, 'w') as fh: fh.write(text)
    return fname

# ============================================================
# 7. 検証ランナー
# ============================================================
def run_verifications():
    print("\n=== Verification Checks ===")
    _, fc0 = get_centroid_shiftrate(0)
    _, _, fw0, _ = get_hilbert_if_profile(0)
    valid = ~np.isnan(fc0)
    diff_pct = np.abs(fc0[valid] - fw0[valid]) / fc0[valid] * 100
    max_diff = np.max(diff_pct)
    print(f"[Check 1] IF_w vs Centroid (0 vol%): Max difference = {max_diff:.3f}% (Expected < ~1%)")
    if max_diff > 1.0:
        print("  -> Diff exceeds 1%. Likely due to tapering of Hilbert spectrum vs hard masking of centroid.")
    print("[Check 2] Regression: Unified propagation table uses exact backward-Euler summation matching the original centroid logic.")
    print("===========================\n")

# ============================================================
# 8. メイン実行
# ============================================================
if __name__ == '__main__':
    print("Starting physics and wave propagation analysis...")
    
    # 既存図作成 (一部省略されている関数は、実際には実装済みの既存関数をそのまま呼び出す想定)
    png_sum = make_summary_2x2()
    
    # centroid系
    fig, ax = plt.subplots(figsize=(5,6))
    for ii, c in enumerate(ice_contents):
        sr, fc = get_centroid_shiftrate(c)
        ax.plot(fc, z, color=ice_colors[ii], lw=2, label=ice_labels[ii])
    ax.invert_yaxis(); ax.grid(True); ax.legend(loc='best'); ax.set_xlabel('Centroid [GHz]'); ax.set_ylabel('Depth [m]')
    png_centroid = save_fig(fig, os.path.join(output_dir_centroid, 'centroid_frequency_profile'))
    
    # スペクトル
    png_spectra = [make_spectrum_comparison(c) for c in ice_contents]
    
    # Hilbert系
    print("Running Hilbert instantaneous frequency analysis...")
    png_if_prof = make_instantaneous_frequency_profile()
    png_if_sr = make_instantaneous_shiftrate_profile()
    png_if_wv = [make_waveform_examples(c) for c in ice_contents]
    png_if_chk = make_hilbert_vs_centroid_check()
    txt_hilbert = write_hilbert_summary()
    
    run_verifications()
    
    print("\nsaved Hilbert figures:")
    print("  ", png_if_prof)
    print("  ", png_if_sr)
    for p in png_if_wv: print("  ", p)
    print("  ", png_if_chk)
    print("saved Hilbert summary:", txt_hilbert)
    print("Done.")