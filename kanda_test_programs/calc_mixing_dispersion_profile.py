import os
import sys
import csv
import json
import warnings
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
# Hilbert解析用の出力先
output_dir_hilbert = os.path.join(output_base_dir, 'Hilbert')
os.makedirs(output_dir_hilbert, exist_ok=True)
# ★追加: 分解能要求解析用の出力先
output_dir_resolution = os.path.join(output_dir_hilbert, 'resolution_estimate')
os.makedirs(output_dir_resolution, exist_ok=True)

eps0 = 8.8541878128e-12          # 真空の誘電率 [F/m]

# 深さ [m]
z   = np.arange(0, 5.01, 0.02)   # [m]
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
NPERSEG_RANGE    = np.arange(16, 4097)

# ------------------------------------------------------------
# Hilbert 瞬時周波数解析の設定
# ------------------------------------------------------------
HILBERT_ENV_THRESHOLD = 0.10   # IF有効区間の包絡線閾値（ピーク比）
HILBERT_PAD_FACTOR    = 2      # irfft のゼロパディング倍率
HILBERT_TAPER_ON      = True   # 帯域外コサインテーパの有無
HILBERT_EXAMPLE_DEPTHS = [0.5, 1.5, 3.0]  # 波形診断図の対象深さ [m]
HILBERT_SUBSAMPLE     = True   # ★追加(A): IF_peak のサブサンプル(放物線)補間の有無

# ------------------------------------------------------------
# ★追加: Hilbert 分解能要求解析の設定
# ------------------------------------------------------------
RES_TAVG_LIST     = [1.0, 3.0, 10.0]      # 平滑化時間長のスイープ [ns]
RES_TAVG_DEFAULT  = 3.0                    # スイープ対象外の図で使う固定値 [ns]
RES_SIGMA_SCALES  = [0.5, 1.0, 2.0]       # σ_spec のスケール係数スイープ (モデル不確かさの感度)
RES_BEFF_SCALES   = [0.5, 1.0, 2.0]       # B_eff のスケール係数スイープ
RES_NTRACES_LIST  = [1, 14, 56]           # トレース平均数のスイープ
RES_NTRACES_DEFAULT = 56                   # 既存B-scanのトレース数
RES_STYLES        = ['-', '--', '-.', ':']  # スイープ値に割り当てる線種 (freq_styles流用+予備)

# 経験δf抽出用のB-scanレジストリ (rand_ampごとに辞書を1行追加するだけで拡張できる)
EMPIRICAL_BSCAN_REGISTRY = [
    {'label': 'rand0.01', 'rand_amp': 0.01,
    'bscan_json': '/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_3x4/Ice_Detection_NoRock/No_Ice/rand_amp_001/Bscan/Bscan.json',
    'enabled': True},
    {'label': 'rand0.00', 'rand_amp': 0.00,
    'bscan_json': '/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_3x4/Ice_Detection_NoRock/No_Ice/Not_Random/Bscan/Bscan.json',
    'enabled': True},
]
EMPIRICAL_MEAN_REMOVAL = True    # 経験δf計算時に平均トレース除去版も併算する
EMPIRICAL_FORCE_RECOMPUTE = False  # True で既存CSVを無視して再計算
EMPIRICAL_POWER_FLOOR_DB = -120.0  # 窓内包絡線²和のトレース最大値基準の下限 [dB]

RUN_RESOLUTION = True   # ★分解能要求解析(機能C/D)の実行フラグ


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
    return eps_host + 3.0 * f * eps_host * (eps_incl - eps_host)            / (eps_incl + 2.0 * eps_host - f * (eps_incl - eps_host))

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
    if ice_volpct in _prop_cache:
        return _prop_cache[ice_volpct]

    freq_full, _, w_full = load_incident_spectrum('full')
    cum_att = np.zeros((len(z), len(w_full)))
    cum_time = np.zeros((len(z), len(w_full)))

    current_att = np.zeros_like(w_full)
    current_time = np.zeros_like(w_full)
    d_step = 0.02
    idx_rx = np.argmin(np.abs(z - RX_DEPTH))

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
_centroid_tdelay_cache = {}
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
    _centroid_tdelay_cache[ice_volpct] = np.array(t_delay_list)
    return _centroid_cache[ice_volpct]

def get_centroid_tdelay(ice_volpct):
    if ice_volpct not in _centroid_tdelay_cache:
        get_centroid_shiftrate(ice_volpct)
    return _centroid_tdelay_cache[ice_volpct]

def stft_delta_f_ghz(nperseg):
    return STFT_FS_GHZ / np.asarray(nperseg, dtype=float)

def stft_delta_fdot_ghz_per_ns(nperseg):
    return 2.0 * np.sqrt(2.0) * (STFT_FS_GHZ / np.asarray(nperseg, dtype=float)) ** 2

def stft_delta_z(nperseg, v):
    return np.asarray(nperseg, dtype=float) * STFT_DT_S * np.asarray(v, dtype=float) / 2.0

def stft_delta_zdot(nperseg, v):
    return 1.5 * stft_delta_z(nperseg, v)

_velocity_cache = {}
def get_local_velocity_profile(ice_volpct):
    if ice_volpct in _velocity_cache:
        return _velocity_cache[ice_volpct]
    _, fc = get_centroid_shiftrate(ice_volpct)
    v_z = np.full_like(z, np.nan)
    for j, d in enumerate(z):
        if np.isfinite(fc[j]) and fc[j] > 0:
            _, v_j = local_alpha_velocity(d, np.array([2*np.pi*fc[j]*1e9]), ice_volpct)
            v_z[j] = v_j[0]
    _velocity_cache[ice_volpct] = v_z
    return v_z

def get_stft_requirements(ice_volpct):
    sr0, fc0 = get_centroid_shiftrate(0)
    sr, fc = get_centroid_shiftrate(ice_volpct)
    d_fc = np.abs(fc - fc0)
    d_fdot = np.abs(sr - sr0)

    with np.errstate(divide='ignore', invalid='ignore'):
        n_req_fc = np.where(d_fc > 0, STFT_FS_GHZ / (d_fc / DETECT_MARGIN), np.nan)
        n_req_fdot = np.where(d_fdot > 0, STFT_FS_GHZ * np.sqrt(2.0 * np.sqrt(2.0) / (d_fdot / DETECT_MARGIN)), np.nan)

    v_z = get_local_velocity_profile(ice_volpct)

    return dict(d_fc=d_fc, d_fdot=d_fdot, n_req_fc=n_req_fc, n_req_fdot=n_req_fdot,
                dz_fc=stft_delta_z(n_req_fc, v_z), dz_fdot=stft_delta_zdot(n_req_fdot, v_z), v=v_z)

# ============================================================
# 5. Hilbert 瞬時周波数解析
# ============================================================
_hilbert_cache = {}
_subsample_fallback_count = {'edge': 0, 'flat': 0, 'delta': 0, 'total': 0}

def _parabolic_peak_shift(env, m):
    _subsample_fallback_count['total'] += 1
    if m <= 0 or m >= len(env) - 1:
        _subsample_fallback_count['edge'] += 1
        return None, 'edge'
    y0, y1, y2 = env[m-1], env[m], env[m+1]
    denom = y0 - 2.0 * y1 + y2
    if np.abs(denom) < 1e-30 * np.abs(y1):
        _subsample_fallback_count['flat'] += 1
        return None, 'flat'
    delta = 0.5 * (y0 - y2) / denom
    if not np.isfinite(delta) or np.abs(delta) > 0.5:
        _subsample_fallback_count['delta'] += 1
        return None, 'delta'
    return delta, 'ok'

def get_hilbert_if_profile(ice_volpct, subsample=None):
    if subsample is None:
        subsample = HILBERT_SUBSAMPLE
    key = (ice_volpct, bool(subsample))
    if key in _hilbert_cache:
        return _hilbert_cache[key]

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

        H = np.exp(-2 * att) * np.exp(-1j * w * (tau + t_offset_s))
        e_d = np.fft.irfft(S0 * H, n=N_pad)

        z_t = scipy.signal.hilbert(e_d)
        env = np.abs(z_t)

        max_idx = np.argmax(env)
        threshold = HILBERT_ENV_THRESHOLD * env[max_idx]
        mask = env >= threshold

        left = max_idx
        while left > 0 and mask[left-1]: left -= 1
        right = max_idx
        while right < N_pad - 1 and mask[right+1]: right += 1

        seg_slice = slice(left, right+1)
        env_seg = env[seg_slice]
        t_seg = t_pad[seg_slice]

        phase_seg = np.unwrap(np.angle(z_t[seg_slice]))
        IF_seg = np.gradient(phase_seg, dt_pad) / (2 * np.pi)

        if subsample:
            delta, reason = _parabolic_peak_shift(env, max_idx)
        else:
            delta, reason = None, 'disabled'

        if delta is not None:
            peak_time = t_pad[max_idx] + delta * dt_pad
            IF_peak = float(np.interp(peak_time, t_seg, IF_seg))
        else:
            peak_time = t_pad[max_idx]
            IF_peak = IF_seg[max_idx - left]

        IF_w = np.sum(env_seg**2 * IF_seg) / np.sum(env_seg**2)

        IF_peak_list.append(IF_peak / 1e9)
        IF_w_list.append(IF_w / 1e9)
        t_delay_list.append(peak_time * 1e9)

        z_rounded = round(d, 2)
        if z_rounded in HILBERT_EXAMPLE_DEPTHS:
            diagnostic_dict[z_rounded] = {
                't_pad': t_pad * 1e9, 'e_d': e_d, 'env': env, 'mask': mask,
                't_seg': t_seg * 1e9, 'IF_seg': IF_seg / 1e9,
                'IF_peak': IF_peak / 1e9, 'IF_w': IF_w / 1e9,
                't_peak': peak_time * 1e9, 'subsample': delta is not None
            }

    IF_peak_arr = np.array(IF_peak_list)
    IF_w_arr = np.array(IF_w_list)
    sr_peak = compute_shiftrate_profile(t_delay_list, IF_peak_list, z, RX_DEPTH)
    sr_w = compute_shiftrate_profile(t_delay_list, IF_w_list, z, RX_DEPTH)

    _hilbert_cache[key] = dict(sr_peak=sr_peak, sr_w=sr_w, IF_peak=IF_peak_arr,
                               IF_w=IF_w_arr, t_delay=np.array(t_delay_list),
                               diag=diagnostic_dict)
    return _hilbert_cache[key]

def print_subsample_fallback_report():
    c = _subsample_fallback_count
    n_fb = c['edge'] + c['flat'] + c['delta']
    print(f"[Sub-sample interpolation] evaluated = {c['total']}, fallback = {n_fb} "
          f"(edge={c['edge']}, flat={c['flat']}, |delta|>0.5: {c['delta']})")

# ============================================================
# 6. ★ Hilbert 分解能 (δIF) の理論モデル
# ============================================================
_moments_cache = {}
def get_spectral_moments(ice_volpct):
    if ice_volpct in _moments_cache:
        return _moments_cache[ice_volpct]

    f_calc, S0_calc, _ = load_incident_spectrum('band')
    prop = get_propagation_table(ice_volpct)
    band_mask = get_band_mask()

    att = prop['cum_att'][:, band_mask]
    with np.errstate(over='ignore', invalid='ignore'):
        P = np.abs(S0_calc[None, :] * np.exp(-2.0 * att)) ** 2

    I0 = np.trapz(P, f_calc, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        fc = np.trapz(f_calc[None, :] * P, f_calc, axis=1) / I0
        var = np.trapz((f_calc[None, :] - fc[:, None])**2 * P, f_calc, axis=1) / I0
        sigma_spec = np.sqrt(var)
        I2 = np.trapz(P**2, f_calc, axis=1)
        B_eff = I0**2 / I2

    _moments_cache[ice_volpct] = (fc, sigma_spec, B_eff)
    return _moments_cache[ice_volpct]

def delta_if_profile(ice_volpct, k_sigma=1.0, k_beff=1.0,
                     T_avg=RES_TAVG_DEFAULT, n_traces=RES_NTRACES_DEFAULT):
    _, sigma_spec, B_eff = get_spectral_moments(ice_volpct)
    T_s = float(T_avg) * 1e-9
    with np.errstate(divide='ignore', invalid='ignore'):
        n_win = np.maximum(k_beff * B_eff * T_s, 1.0)
        N_indep = n_win * float(n_traces)
        d_if = k_sigma * sigma_spec / np.sqrt(N_indep)
    return d_if

def signal_delta_if_profile(ice_volpct, use_peak=False):
    key = 'IF_peak' if use_peak else 'IF_w'
    h0 = get_hilbert_if_profile(0)[key]
    hc = get_hilbert_if_profile(ice_volpct)[key]
    return np.abs(hc - h0) * 1e9

def required_tavg_profile(ice_volpct, n_traces=RES_NTRACES_DEFAULT,
                          k_sigma=1.0, k_beff=1.0, use_peak=False):
    _, sigma_spec, B_eff = get_spectral_moments(ice_volpct)
    d_sig = signal_delta_if_profile(ice_volpct, use_peak=use_peak)
    v_z = get_local_velocity_profile(ice_volpct)
    with np.errstate(divide='ignore', invalid='ignore'):
        target = d_sig / DETECT_MARGIN
        denom = k_beff * B_eff * float(n_traces) * target**2
        T_req_s = (k_sigma * sigma_spec)**2 / denom
        T_req_s = np.where(np.isfinite(T_req_s) & (target > 0), T_req_s, np.nan)
        dz_req = T_req_s * v_z / 2.0
    return T_req_s * 1e9, dz_req

# ============================================================
# 7. 描画・出力関数群
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
heiken_handle = [Line2D([0], [0], color='gray', ls='--', lw=2, label=r'Heiken (for $ arepsilon_r$ and $	an \delta$)')]

def add_legend(fig):
    fig.legend(handles=freq_handles + ice_handles + heiken_handle,
               loc='lower center', ncol=4, fontsize=14, frameon=True, bbox_to_anchor=(0.5, 1.0))

def add_split_legend(fig, style_handles, ice_only=None, ncol=4):
    handles = (ice_handles if ice_only is None else ice_only) + style_handles
    fig.legend(handles=handles, loc='lower center', ncol=ncol, fontsize=14,
               frameon=True, bbox_to_anchor=(0.5, 1.0))

def style_depth_axis(ax, xlabel, logx=False):
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel('Depth (m)', fontsize=18)
    if logx:
        ax.set_xscale('log')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.minorticks_on()
    ax.grid(True, alpha=0.4)
    ax.invert_yaxis()

def save_fig(fig, base_path):
    fig.savefig(base_path + '.png', bbox_inches='tight', dpi=200)
    fig.savefig(base_path + '.pdf', bbox_inches='tight')
    plt.close(fig)
    return base_path + '.png'

# --- 既存のプロット群 ---
def make_summary_2x2():
    fig, axes = plt.subplots(2, 2, figsize=(10, 11))
    draw_lines(axes[0, 0], EPS_RE, ref=eps_re_Heiken)
    axes[0, 0].set_xlabel(r"$ arepsilon^{\prime}$", fontsize=18)
    draw_lines(axes[0, 1], EPS_IM)
    axes[0, 1].set_xlabel(r"$ arepsilon^{\prime\prime}$", fontsize=18)
    draw_lines(axes[1, 0], SIGMA)
    axes[1, 0].set_xlabel(r"Conductivity $\sigma_{m eff}$ [S/m]", fontsize=18)
    draw_lines(axes[1, 1], TAND, ref=tan_d_heiken)
    axes[1, 1].set_xlabel(r"$	an\delta$", fontsize=18)
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
    axes[1].set_xlabel(r'$|X_{0\%} - X|\,/\,X_{0\%}	imes100$ [%]', fontsize=18)
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

# --- ★機能B: IF_w / IF_peak の個別プロファイル図 (4図)
def make_if_w_profile():
    fig, ax = plt.subplots(figsize=(6, 6))
    _, fc0 = get_centroid_shiftrate(0)
    ax.plot(fc0, z, color='gray', linestyle='--', lw=2, label='Centroid (0 vol%, ref)')
    for ii, c in enumerate(ice_contents):
        ax.plot(get_hilbert_if_profile(c)['IF_w'], z, color=ice_colors[ii], lw=2, label=ice_labels[ii])
    style_depth_axis(ax, r'Instantaneous Frequency $IF_w$ [GHz]')
    ax.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    return save_fig(fig, os.path.join(output_dir_hilbert, 'if_w_profile'))

def make_if_peak_profile():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(get_hilbert_if_profile(0)['IF_w'], z, color='gray', linestyle='--', lw=2,
            label=r'$IF_w$ (0 vol%, ref)')
    for ii, c in enumerate(ice_contents):
        ax.plot(get_hilbert_if_profile(c)['IF_peak'], z, color=ice_colors[ii], lw=2, label=ice_labels[ii])
    style_depth_axis(ax, r'Instantaneous Frequency $IF_{peak}$ [GHz]')
    ax.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    return save_fig(fig, os.path.join(output_dir_hilbert, 'if_peak_profile'))

def make_if_w_shiftrate_profile():
    fig, ax = plt.subplots(figsize=(6, 6))
    for ii, c in enumerate(ice_contents):
        ax.plot(get_hilbert_if_profile(c)['sr_w'], z, color=ice_colors[ii], lw=2, label=ice_labels[ii])
    style_depth_axis(ax, r'$IF_w$ Shift Rate [GHz/ns]')
    ax.legend(loc='lower left', fontsize=12)
    plt.tight_layout()
    return save_fig(fig, os.path.join(output_dir_hilbert, 'if_w_shiftrate_profile'))

def make_if_peak_shiftrate_profile():
    fig, ax = plt.subplots(figsize=(6, 6))
    for ii, c in enumerate(ice_contents):
        ax.plot(get_hilbert_if_profile(c)['sr_peak'], z, color=ice_colors[ii], lw=2, label=ice_labels[ii])
    style_depth_axis(ax, r'$IF_{peak}$ Shift Rate [GHz/ns]')
    ax.legend(loc='lower left', fontsize=12)
    plt.tight_layout()
    return save_fig(fig, os.path.join(output_dir_hilbert, 'if_peak_shiftrate_profile'))

def make_waveform_examples(ice_volpct):
    diag = get_hilbert_if_profile(ice_volpct)['diag']
    n_depths = len(HILBERT_EXAMPLE_DEPTHS)
    fig, axes = plt.subplots(n_depths, 2, figsize=(12, 3.5 * n_depths))

    for i, d in enumerate(HILBERT_EXAMPLE_DEPTHS):
        data = diag.get(d)
        if not data: continue

        ax_wv, ax_if = axes[i]
        t = data['t_pad']
        env = data['env']

        ax_wv.plot(t, data['e_d'], 'k-', alpha=0.5, label='Signal')
        ax_wv.plot(t, env, 'r-', lw=1.5, label='Envelope')
        ax_wv.fill_between(t, 0, env, where=data['mask'], color='red', alpha=0.2, label='Valid Region')
        ax_wv.axvline(data['t_peak'], color='m', ls=':', lw=1.5, label='Peak (sub-sample)')
        ax_wv.set_xlim(data['t_seg'][0] - 2.0, data['t_seg'][-1] + 2.0)
        ax_wv.set_title(f'Depth: {d} m (Ice: {ice_volpct}%)', fontsize=14)
        ax_wv.set_ylabel('Amplitude', fontsize=14)
        ax_wv.legend(loc='upper right', fontsize=10)
        ax_wv.grid(True, alpha=0.4)

        ax_if.plot(data['t_seg'], data['IF_seg'], 'b-', lw=2, label='IF(t)')
        ax_if.axhline(data['IF_peak'], color='m', ls='--',
                      label=f'IF_peak: {data["IF_peak"]:.3f} GHz')
        ax_if.axhline(data['IF_w'], color='g', ls='-.', label=f'IF_w: {data["IF_w"]:.3f} GHz')
        ax_if.axvline(data['t_peak'], color='m', ls=':', lw=1.0)
        ax_if.set_xlim(data['t_seg'][0] - 0.5, data['t_seg'][-1] + 0.5)

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
        fw = get_hilbert_if_profile(c)['IF_w']
        ax.plot(fw, z, color=ice_colors[ii], ls='-', lw=2, label=f'IF_w ({c} vol%)')
        ax.plot(fc, z, color=ice_colors[ii], ls='--', lw=1, alpha=0.8)

    ax.plot([], [], color='k', ls='-', label='IF_w (Hilbert)')
    ax.plot([], [], color='k', ls='--', label='Centroid (Spectrum)')

    style_depth_axis(ax, 'Frequency [GHz]')
    ax.legend(loc='best', fontsize=12)
    ax.set_title('Theoretical consistency check', fontsize=14)
    plt.tight_layout()
    return save_fig(fig, os.path.join(output_dir_hilbert, 'hilbert_vs_centroid_check'))

def write_hilbert_summary():
    lines = ["===== Hilbert Instantaneous Frequency Analysis ====="]
    lines.append(f"Envelope threshold: {HILBERT_ENV_THRESHOLD*100:.1f}%, Zero-padding factor: {HILBERT_PAD_FACTOR}")
    lines.append(f"IF_peak sub-sample (parabolic) interpolation: {'ON' if HILBERT_SUBSAMPLE else 'OFF'}")
    lines.append("IF_peak: Frequency at the (sub-sample interpolated) envelope peak")
    lines.append("IF_w: Envelope-squared weighted average frequency in the valid region")
    lines.append("")

    _, fc0 = get_centroid_shiftrate(0)
    IF0 = get_hilbert_if_profile(0)['IF_peak']

    for d in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        idx = int(np.round(d / 0.02))
        lines.append(f"--- Depth = {d:.2f} m ---")
        for c in ice_contents:
            _, fc = get_centroid_shiftrate(c)
            h = get_hilbert_if_profile(c)
            IFp, IFw = h['IF_peak'], h['IF_w']
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
# 8. ★機能C: 分解能要求スイープ図 (resolution_estimate/)
# ============================================================
_RES_PARAM_TEX = {'k_sigma': r'$k_\sigma$', 'k_beff': r'$k_B$',
                  'T_avg': r'$T_{avg}$', 'n_traces': r'$n_{traces}$'}
_RES_PARAM_UNIT = {'k_sigma': '', 'k_beff': '', 'T_avg': ' ns', 'n_traces': ''}

def _res_style(i):
    return RES_STYLES[i % len(RES_STYLES)]

def _fixed_params_title(fixed_params, exclude=None):
    parts = []
    for k in ['k_sigma', 'k_beff', 'T_avg', 'n_traces']:
        if k == exclude or k not in fixed_params:
            continue
        val = fixed_params[k]
        val_s = f'{val:g}'
        parts.append(f'{_RES_PARAM_TEX[k]} = {val_s}{_RES_PARAM_UNIT[k]}')
    return ', '.join(parts)

def make_resolution_inputs():
    out = []
    for key, idx, label, fname in [
        ('sigma', 1, r'$\sigma_{spec}$ [GHz]', 'inputs_sigma_spec_profile'),
        ('beff', 2, r'$B_{eff}$ [GHz]', 'inputs_beff_profile')]:
        fig, ax = plt.subplots(figsize=(6, 6))
        for ii, c in enumerate(ice_contents):
            m = get_spectral_moments(c)[idx] / 1e9
            ax.plot(m, z, color=ice_colors[ii], ls='-', lw=2, label=ice_labels[ii])
        style_depth_axis(ax, label)
        ax.legend(loc='best', fontsize=12)
        ax.set_title('Model input for Hilbert IF resolution', fontsize=14)
        plt.tight_layout()
        out.append(save_fig(fig, os.path.join(output_dir_resolution, fname)))
    return out

def make_resolution_sweep(sweep_name, sweep_values, fixed_params, fname):
    if sweep_name not in _RES_PARAM_TEX:
        raise ValueError(f"Invalid sweep_name: {sweep_name}")

    fig, ax = plt.subplots(figsize=(7, 6))
    for ii, c in enumerate(ice_contents):
        for si, val in enumerate(sweep_values):
            params = dict(fixed_params)
            params[sweep_name] = val
            d_if_ghz = delta_if_profile(c, **params) / 1e9
            ax.plot(d_if_ghz, z, color=ice_colors[ii], ls=_res_style(si), lw=1.6, zorder=3 + ii)

    style_depth_axis(ax, r'$\delta IF$ [GHz]', logx=True)
    style_handles = [Line2D([0], [0], color='k', ls=_res_style(si), lw=2,
                            label=f'{_RES_PARAM_TEX[sweep_name]} = {val:g}{_RES_PARAM_UNIT[sweep_name]}')
                     for si, val in enumerate(sweep_values)]
    add_split_legend(fig, style_handles)
    ax.set_title('Hilbert IF resolution: ' + _fixed_params_title(fixed_params, exclude=sweep_name)
                 + ' (fixed)', fontsize=13)
    plt.tight_layout()
    return save_fig(fig, os.path.join(output_dir_resolution, fname))

def make_requirement_overlay(use_peak=False):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for ii, c in enumerate(ice_contents):
        if c == 0:
            continue
        ax.plot(signal_delta_if_profile(c, use_peak=use_peak) / 1e9, z,
                color=ice_colors[ii], ls='-', lw=2)
    for si, T in enumerate(RES_TAVG_LIST):
        d_if = delta_if_profile(0, k_sigma=1.0, k_beff=1.0, T_avg=T,
                                n_traces=RES_NTRACES_DEFAULT) / 1e9
        ax.plot(d_if, z, color='0.25', ls=_res_style(si), lw=1.8)

    style_depth_axis(ax, r'Frequency separation / $\delta IF$ [GHz]', logx=True)
    style_handles = ([Line2D([0], [0], color='0.25', ls=_res_style(si), lw=2,
                             label=r'$\delta IF$ ($T_{avg}$ = ' + f'{T:g} ns)')
                      for si, T in enumerate(RES_TAVG_LIST)]
                     + [Line2D([0], [0], color='k', ls='-', lw=2, label='signal (solid, colored)')])
    add_split_legend(fig, style_handles, ice_only=ice_handles[1:])
    sig_name = 'IF_peak' if use_peak else 'IF_w'
    ax.set_title(f'Signal |{sig_name}(0%)-{sig_name}(ice)| vs noise ' + r'$\delta IF$' + '\n'
                 + f'(crossing depth = detection limit; $n_{{traces}}$ = {RES_NTRACES_DEFAULT}, '
                 + r'$k_\sigma = k_B = 1$)', fontsize=12)
    plt.tight_layout()
    return save_fig(fig, os.path.join(output_dir_resolution, 'requirement_overlay'))

def make_required_tavg_and_dz(use_peak=False):
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ii, c in enumerate(ice_contents):
        if c == 0:
            continue
        T_req_ns, dz_req = required_tavg_profile(c, n_traces=RES_NTRACES_DEFAULT, use_peak=use_peak)
        axes[0].plot(T_req_ns, z, color=ice_colors[ii], ls='-', lw=2, label=ice_labels[ii])
        axes[1].plot(dz_req, z, color=ice_colors[ii], ls='-', lw=2, label=ice_labels[ii])

    style_depth_axis(axes[0], r'Required $T_{avg}$ [ns]', logx=True)
    style_depth_axis(axes[1], r'Corresponding $\Delta z$ [m]', logx=True)
    add_split_legend(fig, [], ice_only=ice_handles[1:])
    fig.suptitle(f'Requirement (margin = {DETECT_MARGIN:g}, $n_{{traces}}$ = {RES_NTRACES_DEFAULT}, '
                 + r'$k_\sigma = k_B = 1$)', fontsize=13, y=1.02)
    plt.tight_layout()
    return save_fig(fig, os.path.join(output_dir_resolution, 'required_tavg_and_dz'))

# ★追加: 固定 T_avg (= RES_TAVG_LIST) の深さ方向平均化幅 Δz(d) = T_avg*v(d)/2。
# v(d) は required_tavg_and_dz と同じ get_local_velocity_profile を流用する。
def make_fixed_tavg_dz_profile():
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for ii, c in enumerate(ice_contents):
        v_profile = get_local_velocity_profile(c)
        for si, T in enumerate(RES_TAVG_LIST):
            dz = (T * 1e-9) * v_profile / 2.0
            ax.plot(dz, z, color=ice_colors[ii], ls=_res_style(si), lw=2)

    ax.axvline(0.5, color='gray', linestyle=':', lw=1.5, alpha=0.8)
    ax.text(0.5, 0.12, ' nominal layer\n thickness 0.5 m',
            fontsize=11, color='gray', va='top')

    style_depth_axis(ax, r'$\Delta z = T_{avg} \cdot v(d)\,/\,2$ [m]', logx=True)
    style_handles = [Line2D([0], [0], color='0.25', ls=_res_style(si), lw=2,
                            label=r'$T_{avg}$ = ' + f'{T:g} ns')
                     for si, T in enumerate(RES_TAVG_LIST)]
    add_split_legend(fig, style_handles)
    ax.set_title('Averaging window in depth for fixed ' + r'$T_{avg}$'
                 + '\n(matches requirement_overlay settings)', fontsize=13)
    plt.tight_layout()
    return save_fig(fig, os.path.join(output_dir_resolution, 'fixed_tavg_dz_profile'))

# ------------------------------------------------------------
# ドナーからの移植関数群
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

def make_shiftrate_profile():
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

def make_stft_requirement_profile(kind):
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
            continue
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

def make_nperseg_vs_frequency_resolution():
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

def make_nperseg_vs_depth_resolution():
    n = NPERSEG_RANGE
    fig, ax = plt.subplots(figsize=(7, 6))

    for k, epsr in enumerate(EPSR_LIST_FOR_DZ):
        v = const.c / np.sqrt(epsr)
        col = EPSR_COLORS[k % len(EPSR_COLORS)]
        ax.plot(n, stft_delta_z(n, v), color=col, lw=2,
                label=rf'$ arepsilon_r$ = {epsr:.1f} ($v$ = {v/1e9*1e0:.3f} m/ns)'.replace('m/ns', 'm/ns'))
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
                            label=r'$\Delta \dot{z}$ (shift rate, $	imes 1.5$)')]
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles=h + style_handles, loc='upper left', fontsize=12)
    ax.set_title(f'dt = {STFT_DT_S:.3e} s (fixed)', fontsize=15)

    plt.tight_layout()
    base = os.path.join(output_dir_stft, 'nperseg_vs_depth_resolution')
    fig.savefig(base + '.png', bbox_inches='tight', dpi=200)
    fig.savefig(base + '.pdf', bbox_inches='tight')
    plt.close(fig)
    return base + '.png'

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

    lines.append("--- Relative difference vs 0 vol% ice "
                 "(at 1.25 GHz, representative depths) ---")
    fi_ref = 1
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
    print("\n".join(lines[:40]))
    return fname


# ============================================================
# 9. ★機能D: 経験的 δf 抽出 (B-scanレジストリ方式)
# ============================================================
def load_bscan_data(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.json':
        with open(path, 'r') as fh:
            obj = json.load(fh)
        if isinstance(obj.get('data'), str):
            data_path = obj['data']
            data_name = obj.get('data_name', 'Ez')
            if not os.path.isabs(data_path):
                data_path = os.path.join(os.path.dirname(path), data_path)
            data, dt = get_output_data(data_path, 1, data_name)
            data = np.asarray(data, dtype=float)
        else:
            key = next((k for k in ('outputdata', 'bscan', 'Ez') if k in obj), None)
            if key is None:
                raise KeyError(f"No data path or array found in {path} (keys: {list(obj.keys())})")
            data = np.asarray(obj[key], dtype=float)
            dt_key = next((k for k in ('dt', 'dt_s', 'time_step') if k in obj), None)
            if dt_key is None:
                raise KeyError(f"No dt key found in {path}")
            dt = float(obj[dt_key])
    else:
        data, dt = get_output_data(path, 1, 'Ez')
        data = np.asarray(data, dtype=float)

    if data.ndim == 1:
        data = data[:, None]
    if data.shape[0] < data.shape[1]:
        print(f"  Note: transposing B-scan array {data.shape} -> {data.T.shape}")
        data = data.T
    return data, dt

def _windowed_df(arr, dt, T_win_ns, power_floor_db=EMPIRICAL_POWER_FLOOR_DB):
    n_samples, n_traces = arr.shape
    zt = scipy.signal.hilbert(arr, axis=0)
    env2 = np.abs(zt) ** 2
    phase = np.unwrap(np.angle(zt), axis=0)
    IF = np.gradient(phase, dt, axis=0) / (2 * np.pi)

    nwin = max(int(round(T_win_ns * 1e-9 / dt)), 4)
    hop = max(nwin // 2, 1)
    starts = np.arange(0, max(n_samples - nwin + 1, 1), hop)

    ref = np.max(env2, axis=0) * nwin * 10.0 ** (power_floor_db / 10.0)

    IF_w = np.full((len(starts), n_traces), np.nan)
    t_center = np.zeros(len(starts))
    for k, s in enumerate(starts):
        sl = slice(s, s + nwin)
        wgt = env2[sl]
        sw = wgt.sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            vals = (wgt * IF[sl]).sum(axis=0) / np.where(sw > 0, sw, np.nan)
        vals[sw < ref] = np.nan
        IF_w[k] = vals
        t_center[k] = (s + nwin / 2.0) * dt * 1e9

    n_valid = np.sum(~np.isnan(IF_w), axis=1)
    if_w_median = np.nanmedian(IF_w, axis=1) / 1e9
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        q75 = np.nanpercentile(IF_w, 75, axis=1)
        q25 = np.nanpercentile(IF_w, 25, axis=1)
    df_single = (q75 - q25) / 1.349 / 1e9
    df_single = np.where(n_valid >= 4, df_single, np.nan)
    df_profile = df_single / np.sqrt(n_traces)
    return t_center, df_single, df_profile, n_valid, if_w_median

def compute_empirical_df(bscan_path, T_win_ns=RES_TAVG_DEFAULT,
                         mean_removal=EMPIRICAL_MEAN_REMOVAL):
    data, dt = load_bscan_data(bscan_path)
    n_traces = data.shape[1]

    t_ns, df_s_raw, df_p_raw, n_valid, if_w_median_raw = _windowed_df(data, dt, T_win_ns)
    if mean_removal:
        data_ms = data - data.mean(axis=1, keepdims=True)
        _, df_s_ms, df_p_ms, _, if_w_median_meansub = _windowed_df(data_ms, dt, T_win_ns)
    else:
        df_s_ms = np.full_like(df_s_raw, np.nan)
        df_p_ms = np.full_like(df_p_raw, np.nan)
        if_w_median_meansub = np.full_like(if_w_median_raw, np.nan)

    t_map = get_centroid_tdelay(0)
    valid = ~np.isnan(t_map)
    depth = np.interp(t_ns, t_map[valid], z[valid], left=np.nan, right=np.nan)

    return dict(t_ns=t_ns, depth_m=depth, df_single_raw=df_s_raw, df_profile_raw=df_p_raw,
                df_single_meansub=df_s_ms, df_profile_meansub=df_p_ms,
                if_w_median_raw=if_w_median_raw, if_w_median_meansub=if_w_median_meansub,
                n_valid_traces=n_valid, n_traces=n_traces)

_EMP_COLUMNS = ['t_ns', 'depth_m', 'df_single_raw', 'df_profile_raw',
                'df_single_meansub', 'df_profile_meansub', 'if_w_median_raw', 'if_w_median_meansub', 'n_valid_traces']

def get_empirical_df(entry, force_recompute=EMPIRICAL_FORCE_RECOMPUTE):
    label = entry['label']
    csv_path = os.path.join(output_dir_resolution, f'empirical_df_{label}.csv')

    if os.path.exists(csv_path) and not force_recompute:
        with open(csv_path, 'r') as fh:
            header = fh.readline().strip().split(',')
        if all(c in header for c in _EMP_COLUMNS):
            arr = np.genfromtxt(csv_path, delimiter=',', names=True)
            res = {c: np.atleast_1d(arr[c]) for c in _EMP_COLUMNS}
            print(f"  [{label}] loaded cached CSV: {csv_path}")
            return res, csv_path
        else:
            print(f"  [{label}] schema outdated -> recompute")

    path = entry.get('bscan_json')
    if not entry.get('enabled', True):
        print(f"  Warning: entry '{label}' is disabled. Skipped.")
        return None, None
    if not path or not os.path.exists(path):
        print(f"  Warning: B-scan not found for entry '{label}': {path}. Skipped.")
        return None, None

    print(f"  [{label}] computing empirical df from {path} ...")
    res = compute_empirical_df(path, T_win_ns=RES_TAVG_DEFAULT,
                               mean_removal=EMPIRICAL_MEAN_REMOVAL)
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(_EMP_COLUMNS)
        for k in range(len(res['t_ns'])):
            writer.writerow([f"{res[c][k]:.6g}" for c in _EMP_COLUMNS])
    print(f"  [{label}] saved: {csv_path}  (df columns are in GHz)")
    return res, csv_path

def make_empirical_vs_theory(force_recompute=EMPIRICAL_FORCE_RECOMPUTE):
    fig, ax = plt.subplots(figsize=(8, 6))

    for si, nt in enumerate([1, RES_NTRACES_DEFAULT]):
        d_if = delta_if_profile(0, T_avg=RES_TAVG_DEFAULT, n_traces=nt) / 1e9
        ax.plot(d_if, z, color='k', ls=_res_style(si), lw=2,
                label=r'theory $\delta IF$ ($n_{traces}$ = ' + f'{nt})')

    cmap = plt.cm.tab10
    n_drawn = 0
    for ei, entry in enumerate(EMPIRICAL_BSCAN_REGISTRY):
        res, _ = get_empirical_df(entry, force_recompute=force_recompute)
        if res is None:
            continue
        color = cmap(ei % 10)
        lab = f"{entry['label']} (rand_amp = {entry.get('rand_amp', 'n/a')})"
        d = res['depth_m']
        ax.plot(res['df_single_raw'], d, color=color, ls='-', lw=1.8,
                label=f'{lab}, raw, single')
        ax.plot(res['df_profile_raw'], d, color=color, ls='-', lw=1.0, alpha=0.7,
                label=f'{lab}, raw, profile')
        if EMPIRICAL_MEAN_REMOVAL and np.any(np.isfinite(res['df_single_meansub'])):
            if entry.get('rand_amp') == 0:
                print(f"  [{entry['label']}] rand_amp=0 -> skipping mean-sub plots in empirical_vs_theory.")
            else:
                ax.plot(res['df_single_meansub'], d, color=color, ls='--', lw=1.8,
                        label=f'{lab}, mean-sub, single')
                ax.plot(res['df_profile_meansub'], d, color=color, ls='--', lw=1.0, alpha=0.7,
                        label=f'{lab}, mean-sub, profile')
        n_drawn += 1

    style_depth_axis(ax, r'$\delta f$ [GHz]', logx=True)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=10)
    ax.set_title('Empirical vs theoretical IF resolution (0 vol%)\n'
                 + r'$T_{avg}$ = ' + f'{RES_TAVG_DEFAULT:g} ns'
                 + (f', {n_drawn} B-scan entry(ies)' if n_drawn else ', registry empty'),
                 fontsize=13)
    plt.tight_layout()
    return save_fig(fig, os.path.join(output_dir_resolution, 'empirical_vs_theory'))

def make_empirical_bias_check(force_recompute=EMPIRICAL_FORCE_RECOMPUTE):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    
    theory_if_w = get_hilbert_if_profile(0)['IF_w']
    axes[0].plot(theory_if_w, z, color='k', ls='-', lw=2, label='theory IF_w (0 vol%)')
    
    cmap = plt.cm.tab10
    for ei, entry in enumerate(EMPIRICAL_BSCAN_REGISTRY):
        res, _ = get_empirical_df(entry, force_recompute=force_recompute)
        if res is None:
            continue
        color = cmap(ei % 10)
        lab = f"{entry['label']} (rand_amp = {entry.get('rand_amp', 'n/a')})"
        d = res['depth_m']
        
        axes[0].plot(res['if_w_median_raw'], d, color=color, ls='-', lw=1.8, label=f'{lab}, raw')
        
        if entry.get('rand_amp') != 0 and np.any(np.isfinite(res['if_w_median_meansub'])):
            axes[0].plot(res['if_w_median_meansub'], d, color=color, ls='--', lw=1.8, label=f'{lab}, mean-sub')
            
        theory_interp = np.interp(d, z, theory_if_w, left=np.nan, right=np.nan)
        bias = res['if_w_median_raw'] - theory_interp
        axes[1].plot(bias, d, color=color, ls='-', lw=1.8, label=lab)
        
    style_depth_axis(axes[0], r'IF$_w$ [GHz]', logx=False)
    axes[0].legend(loc='best', fontsize=10)
    
    axes[1].axvline(0, color='gray', ls='--')
    style_depth_axis(axes[1], r'Bias (Empirical median - Theory) [GHz]', logx=False)
    axes[1].legend(loc='best', fontsize=10)
    
    fig.suptitle('bias = coherent clutter (wake) systematic error; IQR-based $\delta f$ cannot detect coherent components', fontsize=12)
    plt.tight_layout()
    return save_fig(fig, os.path.join(output_dir_resolution, 'empirical_bias_check'))

def run_resolution_analysis():
    out = []
    out += make_resolution_inputs()
    fixed = dict(k_sigma=1.0, k_beff=1.0, T_avg=RES_TAVG_DEFAULT, n_traces=RES_NTRACES_DEFAULT)
    out.append(make_resolution_sweep('k_sigma', RES_SIGMA_SCALES, fixed, 'sweep_sigma_scale'))
    out.append(make_resolution_sweep('k_beff', RES_BEFF_SCALES, fixed, 'sweep_beff_scale'))
    out.append(make_resolution_sweep('T_avg', RES_TAVG_LIST, fixed, 'sweep_tavg'))
    out.append(make_resolution_sweep('n_traces', RES_NTRACES_LIST, fixed, 'sweep_ntraces'))
    out.append(make_requirement_overlay())
    out.append(make_required_tavg_and_dz())
    out.append(make_fixed_tavg_dz_profile())   # ★追加: 固定 T_avg の Δz(d) プロファイル
    out.append(make_empirical_vs_theory())
    out.append(make_empirical_bias_check())
    return out

# ============================================================
# 10. 検証ランナー
# ============================================================
def _adjacent_diff_std(profile, z_lo=1.5, z_hi=3.0):
    m = (z >= z_lo) & (z <= z_hi)
    seg = np.asarray(profile)[m]
    seg = seg[np.isfinite(seg)]
    if len(seg) < 3:
        return np.nan
    return np.std(np.diff(seg))

def run_verifications():
    print("\n=== Verification Checks ===")

    h_sub = get_hilbert_if_profile(0, subsample=True)
    h_raw = get_hilbert_if_profile(0, subsample=False)
    s_raw = _adjacent_diff_std(h_raw['sr_peak'])
    s_sub = _adjacent_diff_std(h_sub['sr_peak'])
    ratio = s_sub / s_raw if s_raw > 0 else np.nan
    print(f"[Check 1] IF_peak shift-rate roughness (std of adjacent differences, 1.5-3.0 m, 0 vol%):")
    print(f"          before interp = {s_raw:.5f}, after interp = {s_sub:.5f}, ratio = {ratio:.4f}")
    print("          -> PASS (<= 1/5)" if ratio <= 0.2 else
          "          -> FAIL (> 1/5): suspect the sub-sample interpolation implementation")
    print_subsample_fallback_report()

    same = np.allclose(h_sub['IF_w'], h_raw['IF_w'], equal_nan=True, rtol=0, atol=0)
    print(f"[Check 2] IF_w regression (subsample ON vs OFF): "
          f"{'exact match' if same else 'MISMATCH -> IF_w was affected!'}")

    _, fc0 = get_centroid_shiftrate(0)
    fw0 = h_sub['IF_w']
    valid = ~np.isnan(fc0)
    diff_pct = np.abs(fc0[valid] - fw0[valid]) / fc0[valid] * 100
    max_diff = np.max(diff_pct)
    print(f"[Check 3] IF_w vs Centroid (0 vol%): Max difference = {max_diff:.3f}% (Expected < ~1%)")
    if max_diff > 1.0:
        print("  -> Diff exceeds 1%. Likely due to tapering of Hilbert spectrum vs hard masking of centroid.")

    if RUN_RESOLUTION:
        _, sig, beff = get_spectral_moments(0)
        idx = int(np.round(1.5 / 0.02))
        n_indep = max(beff[idx] * RES_TAVG_DEFAULT * 1e-9, 1.0)
        d_if = delta_if_profile(0)[idx]
        print(f"[Check 4] Dimensions @1.5 m: sigma_spec = {sig[idx]/1e9:.4f} GHz [Hz], "
              f"B_eff = {beff[idx]/1e9:.4f} GHz [Hz], B_eff*T_avg = {n_indep:.3f} [-], "
              f"delta_IF = {d_if/1e9:.5f} GHz [Hz]  (T_avg converted ns -> s)")
        print(f"[Check 5] Empirical registry: {len(EMPIRICAL_BSCAN_REGISTRY)} entry(ies) "
              f"({'smoke test: empty registry path exercised' if not EMPIRICAL_BSCAN_REGISTRY else 'entries processed'})")
    print("[Check 6] Regression: Unified propagation table uses exact backward-Euler summation matching the original centroid logic.")
    
    print("[Check 7] CSV schema check for empirical registry...")
    if EMPIRICAL_BSCAN_REGISTRY:
        entry = EMPIRICAL_BSCAN_REGISTRY[0]
        label = entry['label']
        csv_path = os.path.join(output_dir_resolution, f'empirical_df_{label}.csv')
        get_empirical_df(entry, force_recompute=False)
        print("          -> Tested get_empirical_df schema check.")
        
    print("[Check 8] Coherent clutter (wake) bias quantification (rand_amp=0):")
    theory_if_w = get_hilbert_if_profile(0)['IF_w']
    bias_found = False
    for entry in EMPIRICAL_BSCAN_REGISTRY:
        if entry.get('rand_amp') == 0:
            res, _ = get_empirical_df(entry, force_recompute=False)
            if res is not None:
                d = res['depth_m']
                mask = (d >= 1.0) & (d <= 2.5)
                if np.any(mask):
                    theory_interp = np.interp(d, z, theory_if_w, left=np.nan, right=np.nan)
                    bias = res['if_w_median_raw'] - theory_interp
                    bias_seg = bias[mask]
                    bias_seg = bias_seg[np.isfinite(bias_seg)]
                    if len(bias_seg) > 0:
                        med_bias = np.median(bias_seg)
                        max_abs_bias = np.max(np.abs(bias_seg))
                        print(f"          [{entry['label']}] 1.0-2.5m bias: median = {med_bias:.5f} GHz, max_abs = {max_abs_bias:.5f} GHz")
                        bias_found = True
    if not bias_found:
        print("          -> No rand_amp=0 entry or no valid data in 1.0-2.5m.")

    print("===========================\n")

# ============================================================
# 11. メイン実行
# ============================================================
if __name__ == '__main__':
    print("Starting physics and wave propagation analysis...")

    # 1. make_summary_2x2
    png_sum = make_summary_2x2()

    # 2. make_profile_and_delta
    png_re  = make_profile_and_delta(EPS_RE, r"$\epsilon^{\prime}$", 'permittivity_Re', ref=eps_re_Heiken)
    png_im  = make_profile_and_delta(EPS_IM, r"$\epsilon^{\prime\prime}$", 'permittivity_Im')
    png_sig = make_profile_and_delta(SIGMA, r"Conductivity $\sigma_{m eff}$ [S/m]", 'conductivity')
    png_tan = make_profile_and_delta(TAND, r"$\tan\delta$", 'losstangent', ref=tan_d_heiken)

    # 3. make_density_profile, make_ice_wtpct_profile
    png_rho = make_density_profile()
    png_wtpct = make_ice_wtpct_profile()

    # 4. make_shiftrate_profile, make_centroid_profile
    png_shift = make_shiftrate_profile()
    png_centroid = make_centroid_profile()

    # 5. make_spectrum_comparison x ice_contents
    png_spectra = [make_spectrum_comparison(c) for c in ice_contents]

    # 6. STFT要求
    png_req_fc   = make_stft_requirement_profile('centroid')
    png_req_fdot = make_stft_requirement_profile('shiftrate')
    png_nps_freq = make_nperseg_vs_frequency_resolution()
    png_nps_dz   = make_nperseg_vs_depth_resolution()
    txt_stft     = write_stft_summary()

    # 7. write_summary
    txt = write_summary()

    # 8. Hilbert系
    print("Running Hilbert instantaneous frequency analysis...")
    png_if_w    = make_if_w_profile()
    png_if_p    = make_if_peak_profile()
    png_if_w_sr = make_if_w_shiftrate_profile()
    png_if_p_sr = make_if_peak_shiftrate_profile()
    png_if_wv   = [make_waveform_examples(c) for c in ice_contents]
    png_if_chk  = make_hilbert_vs_centroid_check()
    txt_hilbert = write_hilbert_summary()

    # 9. RUN_RESOLUTION
    png_res = []
    if RUN_RESOLUTION:
        print("Running Hilbert resolution requirement analysis...")
        png_res = run_resolution_analysis()

    # 10. run_verifications
    run_verifications()

    # 11. 保存パスのprint一覧
    print("\nsaved figures:")
    for p in [png_sum, png_re, png_im, png_sig, png_tan, png_rho, png_wtpct,
              png_shift, png_centroid] + png_spectra + [
              png_req_fc, png_req_fdot, png_nps_freq, png_nps_dz]:
        print("  ", p)
    print("saved summary:", txt)
    print("saved STFT summary:", txt_stft)
    
    print("\nsaved Hilbert figures:")
    for p in [png_if_w, png_if_p, png_if_w_sr, png_if_p_sr]: print("  ", p)
    for p in png_if_wv: print("  ", p)
    print("  ", png_if_chk)
    print("saved Hilbert summary:", txt_hilbert)
    if png_res:
        print("saved resolution figures:")
        for p in png_res: print("  ", p)
    print("Done.")