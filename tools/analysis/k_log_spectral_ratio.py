"""
スペクトル比法（Log-Spectral Ratio）による tanδ プロファイル解析コード

gprMaxで計算した月レゴリスGPRシミュレーション（B-scan）に対して、
スペクトル比法を適用し、実効ロスタンジェント tanδ および Q値 の
遅延時間プロファイルを推定します。
得られる tanδ は「基準窓から各時刻までの経路平均値」となります。
（幾何減衰や反射係数、アンテナ利得などの周波数非依存項は回帰の切片に
吸収されるため、補正は不要です。）

[FIX-E] 区間方式（移動窓ペア [t, t+DT_INT_NS]）による「局所 tanδ(t)」の
推定を追加。固定基準方式の全出力に加えて tand_interval_profile.csv /
tand_interval_profile.png を出力する。局所的な低損失層（水氷層など）の
署名は固定基準方式では経路平均で希釈されるが、区間方式では希釈なしの
箱型署名（層区間で tanδ が約-10%/10vol%）として現れる。
"""

import os
import sys
import warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import h5py
import glob
import re
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants as const
from gprMax.exceptions import CmdInputError
from tools.core.outputfiles_merge import get_output_data

# =============================================================================
# Spectral-ratio parameters
# =============================================================================
win_len_samples = 256          # 解析窓長 [サンプル]（fs≈84.75 GHzで約3 ns）
hop_samples     = win_len_samples // 4   # 窓の送り幅
freq_min        = 0.5          # [GHz] 回帰帯域の下限  # [FIX-B]
freq_max        = 2.5          # [GHz] 回帰帯域の上限  # [FIX-B]
snr_margin_db   = 10.0         # ノイズ床からのマージン（帯域選択用）
n_min_bins      = 5            # 回帰に必要な最小ビン数  # [FIX-B]
ref_margin_ns   = 5.0          # 基準窓中心 = 地表反射時刻 + このマージン  # [FIX-C]
sigma_clip      = 3.0          # 回帰残差のシグマクリップ閾値（1回のみ）
MEAN_TRACE_REMOVAL = True   # [FIX-A] True: 平均トレース除去（コヒーレントwake除去）
MIN_DT_NS   = 3.0   # [FIX-D] 固定基準ペアの最小時間差 [ns]（Δt→0での 1/(πΔt) 発散を回避）
DT_INT_NS   = 6.0   # [FIX-E] 区間方式LSRの窓ペア間隔 [ns]（局所 tanδ(t) 推定用）

# =============================================================================
# User input & Analytical settings
# =============================================================================
json_file_path = input('Input Bscan.json file path: ').strip()
if not os.path.exists(json_file_path):
    raise CmdInputError('JSON file {} does not exist'.format(json_file_path))

# [EDIT HERE] 入射波スペクトル計算用のA-scan出力ファイルパス
ascan_outfile_path = "/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_3x4/waveform_test/gaussiandot_1.25GHz_underground/result/Ascan.out" 

# =============================================================================
# Load data
# =============================================================================
with open(json_file_path) as f:
    params = json.load(f)
outfile_path = params['data']
GPR_step = params['antenna_settings']['src_step']
print('GPR step [m]:', GPR_step)

fh = h5py.File(outfile_path, 'r')
nrx = fh.attrs['nrx'] if 'nrx' in fh.attrs else len(fh['rxs'].keys())
fh.close()

outputdata, dt = get_output_data(outfile_path, 1, 'Ez')
dt_ns = dt * 1e9        # [ns]
fs    = 1.0 / dt_ns    # [GHz]
n_samples, n_traces = outputdata.shape
data_proc = outputdata - outputdata.mean(axis=1, keepdims=True) if MEAN_TRACE_REMOVAL else outputdata  # [FIX-A]

print(f'dt = {dt*1e12:.4f} ps,  fs = {fs:.2f} GHz,  fs/2 = {fs/2:.2f} GHz')
print(f'B-scan shape (samples, traces): {outputdata.shape}')

# =============================================================================
# Extract Debye Parameters from .in file
# =============================================================================
debye_params = {'tau1': 4.6212e-11, 'tau2': 2.82195e-10, 'de_ratio': 0.261 / (0.261 + 0.088)}
geom_json_path = params.get('geometry_settings', {}).get('geometry_json', '')
in_dir = os.path.dirname(geom_json_path)
in_file_found = False

if in_dir and os.path.exists(in_dir):
    in_files = glob.glob(os.path.join(in_dir, '*.in'))
    for in_file in in_files:
        try:
            with open(in_file, 'r', encoding='utf-8') as fin:
                content = fin.read()
                m_tau1 = re.search(r'DEBYE_TAU1\s*=\s*([0-9\.eE\+\-]+)', content)
                m_tau2 = re.search(r'DEBYE_TAU2\s*=\s*([0-9\.eE\+\-]+)', content)
                m_ratio = re.search(r'DE_RATIO\s*=\s*(.+)', content)
                if m_ratio:
                    # コメント部分などを除外して計算
                    expr = m_ratio.group(1).split('#')[0].strip()
                    debye_params['de_ratio'] = eval(expr)
                m_disp = re.search(r'#add_dispersion_debye:\s*\d+\s+([0-9\.eE\+\-]+)\s+[0-9\.eE\+\-]+\s+([0-9\.eE\+\-]+)', content)
                
                if m_tau1: debye_params['tau1'] = float(m_tau1.group(1))
                if m_tau2: debye_params['tau2'] = float(m_tau2.group(1))
                
                if not m_ratio and m_disp:
                    de1 = float(m_disp.group(1))
                    de2 = float(m_disp.group(2))
                    if (de1 + de2) > 0:
                        debye_params['de_ratio'] = de1 / (de1 + de2)
                
                if m_tau1 or m_tau2 or m_ratio or m_disp:
                    in_file_found = True
        except Exception as e:
            print(f"Warning: Could not parse {in_file}: {e}")

if not in_file_found:
    print("Warning: Could not extract Debye parameters from .in file. Using default values.")
print("Debye Parameters used:", debye_params)

# =============================================================================
# Dielectric Model Definitions
# =============================================================================
def get_eps_static(z_m):
    """深さ z [m] から静的実部とロスタンジェントを計算
    (Heiken1991 Fig 9.54 の 450 MHz 計測経験式; イルメナイト20wt%考慮)"""
    z_cm = z_m * 100.0
    rho = 1.92 * (z_cm + 12.2) / (z_cm + 18.0)
    eps_static = 1.843 ** rho
    tan_d = 10 ** (0.033 * 20.0 + 0.231 * rho - 3.061)
    return eps_static, tan_d

def get_eps_regolith(z_m, omega, d_params, anchor_freq=450e6):
    """指定深さ z_m [m] と角周波数配列 omega に対するレゴリス母材の複素誘電率を返す。"""
    eps_static, tan_d = get_eps_static(z_m)

    tau1 = d_params['tau1']
    tau2 = d_params['tau2']
    de_ratio = d_params['de_ratio']

    w_a = 2.0 * np.pi * anchor_freq
    unit_im_wa = (de_ratio * (w_a * tau1) / (1.0 + (w_a * tau1)**2) +
                  (1.0 - de_ratio) * (w_a * tau2) / (1.0 + (w_a * tau2)**2))
    eps_im_target = eps_static * tan_d
    de_tot = eps_im_target / unit_im_wa

    eps_inf = max(eps_static - de_tot, 1.0)
    de_tot = eps_static - eps_inf
    de1 = de_tot * de_ratio
    de2 = de_tot * (1.0 - de_ratio)

    eps_regolith = (eps_inf
                    + de1 / (1.0 + 1j * omega * tau1)
                    + de2 / (1.0 + 1j * omega * tau2))

    return eps_regolith

def surface_delay_ns(antenna_height, system_lag_ns):
    """地表面反射の到達時刻 (プロット上の基準線 'Surface') を計算 [ns]。"""
    return antenna_height * 2 / 0.3 + system_lag_ns

# =============================================================================
# Analytical Frequency Shift Calculation (Depth + Debye + Time Offset for Buried Rx)
# =============================================================================
antenna_height = 0.35    # [m] 送信機高さ
system_lag_ns  = 0.837   # [ns] システムラグ
rx_depth       = 0.10    # [m] 受信機の埋設深さ
f_center = 450e6

# A-scanから入射波のスペクトル準備
try:
    if os.path.exists(ascan_outfile_path):
        ascan_data, dt_ascan = get_output_data(ascan_outfile_path, 1, 'Ez')
        if ascan_data.ndim == 1:
            e_incident = ascan_data
        else:
            e_incident = ascan_data[:, 0]
        
        N_ascan = len(e_incident)
        freq_ascan = np.fft.rfftfreq(N_ascan, d=dt_ascan)
        S0_omega = np.fft.rfft(e_incident)
        
        f_min_hz = freq_min * 1e9
        f_max_hz = freq_max * 1e9
        band_mask = (freq_ascan >= f_min_hz) & (freq_ascan <= f_max_hz)
        
        f_calc = freq_ascan[band_mask]
        S0_calc = S0_omega[band_mask]
        omega = 2 * np.pi * f_calc
        
        t_air_ns = (2.0 * antenna_height / const.c) * 1e9 
        
        d_sub_offset = np.linspace(0, rx_depth, 50)
        eps_sub_offset, _ = get_eps_static(d_sub_offset)
        v_sub = const.c / np.sqrt(eps_sub_offset)
        dt_sub = d_sub_offset[1] - d_sub_offset[0]
        t_ground_start_ns = np.sum(2.0 * dt_sub / v_sub) * 1e9
        
        t_offset_ns = system_lag_ns + t_air_ns + t_ground_start_ns
        
        print(f"Time offset (depth {rx_depth}m reflection): {t_offset_ns:.3f} ns "
              f"(Lag: {system_lag_ns} + Air: {t_air_ns:.3f} + Ground({rx_depth}m): {t_ground_start_ns:.3f} ns)")
    else:
        print(f"Warning: A-scan file not found at {ascan_outfile_path}. Analytical calculation skipped.")
        S0_calc, f_calc, omega = None, None, None
except Exception as e:
    print(f"Warning: Analytical calculation failed: {e}")
    S0_calc, f_calc, omega = None, None, None

# =============================================================================
# Spectral Ratio setup (Windowing & Aggregation)
# =============================================================================
n_windows = (n_samples - win_len_samples) // hop_samples + 1
t_win_ns = np.zeros(n_windows)
f_win = np.fft.rfftfreq(win_len_samples, d=dt)
f_win_hz = f_win
f_win_ghz = f_win / 1e9

# 中央値集約スペクトルの格納配列
S_med = np.zeros((n_windows, len(f_win)))
hann_win = np.hanning(win_len_samples)[:, None]
t_axis = np.arange(n_samples) * dt_ns

for i in range(n_windows):
    start = i * hop_samples
    end = start + win_len_samples
    t_win_ns[i] = t_axis[start + win_len_samples // 2]
    windowed_data = data_proc[start:end, :] * hann_win  # [FIX-A]
    spectra = np.abs(np.fft.rfft(windowed_data, axis=0))
    S_med[i, :] = np.median(spectra, axis=1)

lnS = np.log(S_med + 1e-30)

print(f"Window length: {win_len_samples} samples ({win_len_samples * dt_ns:.3f} ns)")
print(f"Number of windows: {n_windows}")

# =============================================================================
# Reference window & Noise Floor
# =============================================================================
surf_t = surface_delay_ns(antenna_height, system_lag_ns)

k_ref = -1
for i, t in enumerate(t_win_ns):
    if t >= surf_t + ref_margin_ns:
        k_ref = i
        break

if k_ref == -1:
    raise ValueError("Could not find a valid reference window.")

print(f"Reference window center: {t_win_ns[k_ref]:.3f} ns (Index {k_ref})")

noise_end_ns = surf_t - 1.0
N_noise = int(max(0, noise_end_ns * fs))

# FDTDシミュレーションのため、直達波の影響を避けてリファレンスピークからの相対値でノイズ床を固定
ref_peak = np.max(S_med[k_ref, :])
noise_floor_db = -100.0  # リファレンスピークに対するノイズ床の相対レベル
fallback_val = ref_peak * (10 ** (noise_floor_db / 20))
N_f = np.full_like(f_win, fallback_val)
print(f"Noise floor set to {noise_floor_db} dB relative to reference peak.")

# [FIX-B] 経験的ノイズ床: 記録末尾 NOISE_GATE_NS の窓の中央値スペクトル（ビンごと）
NOISE_GATE_NS = 5.0   # [FIX-B] 記録末尾のこの区間をノイズ床推定に使用
noise_rows = np.where(t_win_ns >= t_axis[-1] - NOISE_GATE_NS)[0]   # [FIX-B]
if len(noise_rows) >= 2:                                            # [FIX-B]
    N_f = np.median(S_med[noise_rows, :], axis=0)                   # [FIX-B]
    print(f"Empirical noise floor from {len(noise_rows)} windows "  # [FIX-B]
          f"(t >= {t_axis[-1] - NOISE_GATE_NS:.1f} ns)")            # [FIX-B]
else:                                                               # [FIX-B]
    print("Warning: not enough tail windows; using flat fallback noise floor.")  # [FIX-B]

# =============================================================================
# Regression Function
# =============================================================================
def fit_log_spectral_ratio(f_hz, S_tgt, S_ref, noise_f, snr_margin, min_bins, s_clip):
    """2つのスペクトル間のスペクトル比を線形回帰する"""
    snr_tgt = 10.0 * np.log10(S_tgt / (noise_f + 1e-30))
    snr_ref = 10.0 * np.log10(S_ref / (noise_f + 1e-30))
    
    valid = (f_hz >= freq_min * 1e9) & (f_hz <= freq_max * 1e9) & \
            (snr_tgt >= snr_margin) & (snr_ref >= snr_margin)
            
    if np.sum(valid) < min_bins:
        return np.nan, np.nan, np.sum(valid), None
        
    f_sel = f_hz[valid]
    LR_sel = np.log(S_tgt[valid] + 1e-30) - np.log(S_ref[valid] + 1e-30)
    
    w_sel = np.minimum(S_tgt[valid] / (noise_f[valid] + 1e-30), 
                       S_ref[valid] / (noise_f[valid] + 1e-30))
                       
    p, cov = np.polyfit(f_sel, LR_sel, 1, w=w_sel, cov=True)
    res = LR_sel - (p[0] * f_sel + p[1])
    std_res = np.std(res)
    
    valid2 = np.abs(res) <= s_clip * std_res
    if np.sum(valid2) < min_bins:
        return np.nan, np.nan, np.sum(valid2), None
        
    p2, cov2 = np.polyfit(f_sel[valid2], LR_sel[valid2], 1, w=w_sel[valid2], cov=True)
    
    fit_info = {
        'f_sel': f_sel, 'LR_sel': LR_sel, 
        'valid2': valid2, 'p': p2, 
        'LR_all': np.log(S_tgt + 1e-30) - np.log(S_ref + 1e-30),
        'valid_mask_all': valid
    }
    
    return p2[0], np.sqrt(cov2[0, 0]), np.sum(valid2), fit_info

# =============================================================================
# Execution: Spectral Ratio (Observation)
# =============================================================================
tand_eff = np.full(n_windows, np.nan)
sigma_tand = np.full(n_windows, np.nan)
n_bins_used = np.zeros(n_windows, dtype=int)
fits_info = {}

for k in range(k_ref + 1, n_windows):
    if (t_win_ns[k] - t_win_ns[k_ref]) < MIN_DT_NS:   # [FIX-D]
        continue                                       # [FIX-D]
    dt_pair_s = (t_win_ns[k] - t_win_ns[k_ref]) * 1e-9
    slope, slope_err, n_bins, info = fit_log_spectral_ratio(
        f_win_hz, S_med[k, :], S_med[k_ref, :], N_f, 
        snr_margin_db, n_min_bins, sigma_clip
    )
    
    n_bins_used[k] = n_bins
    if not np.isnan(slope):
        tand_eff[k] = -slope / (np.pi * dt_pair_s)
        sigma_tand[k] = slope_err / (np.pi * dt_pair_s)
        fits_info[k] = info

valid_windows = np.sum(~np.isnan(tand_eff))
print(f"Valid regression windows: {valid_windows} / {n_windows - k_ref - 1}")
if valid_windows > 0:
    min_b = np.min(n_bins_used[~np.isnan(tand_eff)])
    max_b = np.max(n_bins_used[~np.isnan(tand_eff)])
    print(f"Effective bins per window range: {min_b} - {max_b}")

# [FIX-B] フィット帯域の診断
for k_diag in [k_ref + 2, (k_ref + n_windows) // 2]:                      # [FIX-B]
    if k_diag in fits_info and fits_info[k_diag] is not None:             # [FIX-B]
        info_d = fits_info[k_diag]                                        # [FIX-B]
        f_used = info_d['f_sel'][info_d['valid2']] / 1e9                  # [FIX-B]
        print(f"  t={t_win_ns[k_diag]:.1f} ns: fit band "                 # [FIX-B]
              f"{f_used.min():.2f}-{f_used.max():.2f} GHz, "              # [FIX-B]
              f"{len(f_used)} bins")                                      # [FIX-B]

# Q calculation
with np.errstate(divide='ignore', invalid='ignore'):
    Q_eff = np.where(tand_eff > 0, 1.0 / tand_eff, np.nan)

# =============================================================================
# Execution: Analytical LSR (Theory)
# =============================================================================
tand_theory = np.full(n_windows, np.nan)
local_tand_model = np.full(n_windows, np.nan)

if S0_calc is not None:
    max_depth = (t_axis[-1] * 1e-9) * const.c / 2 
    d_array = np.linspace(rx_depth, max_depth, 400)
    d_step = d_array[1] - d_array[0]
    
    cumulative_attenuation = np.zeros_like(omega)
    cumulative_time = np.zeros_like(omega)
    
    t_model_array = []
    S_model_array = []
    local_tan_array = []
    
    for i, d in enumerate(d_array):
        eps_complex_w = get_eps_regolith(d, omega, debye_params, anchor_freq=f_center)
        alpha_d = - (omega / const.c) * np.imag(np.sqrt(eps_complex_w))
        v_d = const.c / np.real(np.sqrt(eps_complex_w))
        
        _, l_tand = get_eps_static(d)
        local_tan_array.append(l_tand)
        
        if i > 0:
            cumulative_attenuation += alpha_d * d_step
            cumulative_time += 2 * d_step / v_d
            
        S_d_w = np.abs(S0_calc) * np.exp(-2 * cumulative_attenuation)
        
        mean_cum_time = np.mean(cumulative_time)
        t_total_ns = t_offset_ns + (mean_cum_time * 1e9)
        
        t_model_array.append(t_total_ns)
        S_model_array.append(S_d_w)
    
    t_model_array = np.array(t_model_array)
    local_tan_array = np.array(local_tan_array)
    
    # 基準窓に最も近い理論スペクトルを抽出
    idx_ref_model = np.argmin(np.abs(t_model_array - t_win_ns[k_ref]))
    S_model_ref = S_model_array[idx_ref_model]
    
    # 理論側はノイズフリーとする
    N_f_theory = np.zeros_like(f_calc)
    
    for k in range(k_ref + 1, n_windows):
        if (t_win_ns[k] - t_win_ns[k_ref]) < MIN_DT_NS:   # [FIX-D] 観測ループと同一条件
            continue                                       # [FIX-D]
        idx_tgt = np.argmin(np.abs(t_model_array - t_win_ns[k]))
        if t_model_array[idx_tgt] <= t_model_array[idx_ref_model]:
            continue
            
        S_model_tgt = S_model_array[idx_tgt]
        dt_pair_s = (t_model_array[idx_tgt] - t_model_array[idx_ref_model]) * 1e-9
        
        slope, _, _, _ = fit_log_spectral_ratio(
            f_calc, S_model_tgt, S_model_ref, N_f_theory,
            snr_margin=-999.0, min_bins=n_min_bins, s_clip=sigma_clip
        )
        
        if not np.isnan(slope):
            tand_theory[k] = -slope / (np.pi * dt_pair_s)
            
        local_tand_model[k] = local_tan_array[idx_tgt]

# =============================================================================
# [FIX-E] Execution: Interval-mode Spectral Ratio (moving window pair)
# =============================================================================
# 固定基準方式は「基準窓から t までの経路平均 tanδ」を返すため、局所的な
# 低損失層（例: 水氷層）の署名は経路長との比で希釈される（10 vol%氷で
# Δtanδ ≈ -0.001〜-0.002）。区間方式は移動する窓ペア [t, t+DT_INT_NS] で
# 回帰して局所 tanδ(t) を推定し、層内で希釈なしの Δtanδ（≈-10%）が
# 箱型署名として現れる。層の上端・下端（深さ分布）の推定に対応する。
# 既存の固定基準方式の計算・出力には一切影響しない。
hop_ns = hop_samples * dt_ns                                        # [FIX-E]
n_int = max(1, int(round(DT_INT_NS / hop_ns)))                      # [FIX-E]
dt_int_actual_ns = n_int * hop_ns                                   # [FIX-E]
print(f"[Interval LSR] pair separation: {n_int} windows "           # [FIX-E]
      f"({dt_int_actual_ns:.2f} ns)")                               # [FIX-E]

tand_int             = np.full(n_windows, np.nan)                   # [FIX-E]
sigma_tand_int       = np.full(n_windows, np.nan)                   # [FIX-E]
n_bins_int           = np.zeros(n_windows, dtype=int)               # [FIX-E]
t_int_mid            = np.full(n_windows, np.nan)                   # [FIX-E]
tand_theory_int      = np.full(n_windows, np.nan)                   # [FIX-E]
local_tand_model_int = np.full(n_windows, np.nan)                   # [FIX-E]

for k in range(k_ref, n_windows - n_int):                           # [FIX-E]
    t_a = t_win_ns[k]                                               # [FIX-E]
    t_b = t_win_ns[k + n_int]                                       # [FIX-E]
    t_int_mid[k] = 0.5 * (t_a + t_b)                                # [FIX-E]
    dt_pair_s = (t_b - t_a) * 1e-9                                  # [FIX-E]
    slope, slope_err, n_bins, _ = fit_log_spectral_ratio(           # [FIX-E]
        f_win_hz, S_med[k + n_int, :], S_med[k, :], N_f,            # [FIX-E]
        snr_margin_db, n_min_bins, sigma_clip                       # [FIX-E]
    )                                                               # [FIX-E]
    n_bins_int[k] = n_bins                                          # [FIX-E]
    if not np.isnan(slope):                                         # [FIX-E]
        tand_int[k] = -slope / (np.pi * dt_pair_s)                  # [FIX-E]
        sigma_tand_int[k] = slope_err / (np.pi * dt_pair_s)         # [FIX-E]

valid_int = np.sum(~np.isnan(tand_int))                             # [FIX-E]
print(f"[Interval LSR] valid windows: {valid_int} / "               # [FIX-E]
      f"{max(0, n_windows - n_int - k_ref)}")                       # [FIX-E]

if S0_calc is not None:                                             # [FIX-E]
    for k in range(k_ref, n_windows - n_int):                       # [FIX-E]
        idx_a = np.argmin(np.abs(t_model_array - t_win_ns[k]))          # [FIX-E]
        idx_b = np.argmin(np.abs(t_model_array - t_win_ns[k + n_int]))  # [FIX-E]
        if t_model_array[idx_b] <= t_model_array[idx_a]:            # [FIX-E]
            continue                                                # [FIX-E]
        dt_pair_s = (t_model_array[idx_b] - t_model_array[idx_a]) * 1e-9  # [FIX-E]
        slope, _, _, _ = fit_log_spectral_ratio(                    # [FIX-E]
            f_calc, S_model_array[idx_b], S_model_array[idx_a],     # [FIX-E]
            N_f_theory, snr_margin=-999.0,                          # [FIX-E]
            min_bins=n_min_bins, s_clip=sigma_clip                  # [FIX-E]
        )                                                           # [FIX-E]
        if not np.isnan(slope):                                     # [FIX-E]
            tand_theory_int[k] = -slope / (np.pi * dt_pair_s)       # [FIX-E]
        idx_mid = np.argmin(np.abs(t_model_array - t_int_mid[k]))   # [FIX-E]
        local_tand_model_int[k] = local_tan_array[idx_mid]          # [FIX-E]

# =============================================================================
# Output Setup
# =============================================================================
output_base_name = 'spectral_ratio_analysis'
if MEAN_TRACE_REMOVAL:                      # [FIX-A]
    output_base_name += '_meansub'          # [FIX-A]
output_dir = os.path.join(os.path.dirname(os.path.abspath(json_file_path)), output_base_name)
os.makedirs(output_dir, exist_ok=True)

# CSV Export
csv_path = os.path.join(output_dir, 'tand_profile.csv')
with open(csv_path, 'w', newline='') as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(['t_ns', 'tand_eff', 'sigma_tand', 'Q_eff', 'n_bins', 'tand_theory'])
    for k in range(n_windows):
        writer.writerow([
            t_win_ns[k], tand_eff[k], sigma_tand[k], 
            Q_eff[k] if not np.isnan(Q_eff[k]) else '', 
            n_bins_used[k], tand_theory[k]
        ])
print(f"Saved: {csv_path}")

# [FIX-E] CSV Export (interval mode)
csv_int_path = os.path.join(output_dir, 'tand_interval_profile.csv')       # [FIX-E]
with open(csv_int_path, 'w', newline='') as f_csv:                         # [FIX-E]
    writer = csv.writer(f_csv)                                             # [FIX-E]
    writer.writerow(['t_mid_ns', 'tand_int', 'sigma_tand_int', 'n_bins',   # [FIX-E]
                     'tand_theory_int', 'local_tand_model_int'])           # [FIX-E]
    for k in range(n_windows):                                             # [FIX-E]
        if np.isnan(t_int_mid[k]):                                         # [FIX-E]
            continue                                                       # [FIX-E]
        writer.writerow([t_int_mid[k], tand_int[k], sigma_tand_int[k],     # [FIX-E]
                         n_bins_int[k], tand_theory_int[k],                # [FIX-E]
                         local_tand_model_int[k]])                         # [FIX-E]
print(f"Saved: {csv_int_path}")                                            # [FIX-E]

# =============================================================================
# Plotting
# =============================================================================
plt.rcParams['font.size'] = 14

# --- Figure 1: tand_profile.png ---
fig1, ax1 = plt.subplots(figsize=(8, 10))

ax1.plot(tand_eff, t_win_ns, color='k', linestyle='-', label='Observation (LSR)')
ax1.fill_betweenx(t_win_ns, tand_eff - sigma_tand, tand_eff + sigma_tand, color='gray', alpha=0.4, label=r'$\pm 1\sigma$')

if np.any(~np.isnan(tand_theory)):
    ax1.plot(tand_theory, t_win_ns, color='r', linestyle='--', label='Analytical (LSR)')
if np.any(~np.isnan(local_tand_model)):
    ax1.plot(local_tand_model, t_win_ns, color='g', linestyle=':', label=r'Model Local tan$\delta$')

ax1.axhline(surf_t, color='gray', linestyle='--', lw=2, label='Surface')

ax1.set_ylim(t_win_ns[-1], t_win_ns[0])
ax1.set_xlabel(r'Loss Tangent (tan$\delta$)', size=18)
ax1.set_ylabel('Delay time [ns]', size=18)
ax1.minorticks_on()
ax1.grid(True)
ax1.legend(loc='lower left', fontsize=14)

def safe_reciprocal(x):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x == 0, np.inf, 1.0 / x)

secax = ax1.secondary_xaxis('top', functions=(safe_reciprocal, safe_reciprocal))
secax.set_xlabel(r'Q-factor (1 / tan$\delta$)', size=18)
secax.set_ticks([10, 20, 50, 100, 200, 500])

fig1.tight_layout()
fig1_path = os.path.join(output_dir, 'tand_profile.png')
fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
print(f'Saved: {fig1_path}')
plt.close(fig1)

# --- Figure 2: logspectral_ratio_examples.png ---
valid_indices = [k for k in range(n_windows) if not np.isnan(tand_eff[k])]
if len(valid_indices) >= 5:
    selected_k = np.linspace(valid_indices[0], valid_indices[-1], 5, dtype=int)
else:
    selected_k = valid_indices

if len(selected_k) > 0:
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(selected_k)))
    
    for i, k in enumerate(selected_k):
        info = fits_info[k]
        t_val = t_win_ns[k]
        tan_val = tand_eff[k]
        
        # 全帯域のプロット（除外された点は薄く）
        ax2.scatter(f_win_ghz[info['valid_mask_all']], info['LR_all'][info['valid_mask_all']], 
                    color=colors[i], alpha=0.2, s=10)
                    
        # 回帰に使われた点のプロット
        f_sel_ghz = info['f_sel'][info['valid2']] / 1e9
        LR_sel = info['LR_sel'][info['valid2']]
        ax2.scatter(f_sel_ghz, LR_sel, color=colors[i], s=30)
        
        # 回帰直線のプロット
        f_line = np.array([freq_min, freq_max])
        LR_line = info['p'][0] * (f_line * 1e9) + info['p'][1]
        ax2.plot(f_line, LR_line, color=colors[i], linestyle='-', 
                 label=fr't = {t_val:.1f} ns (tan$\delta$ = {tan_val:.4f})')
                 
    ax2.set_xlabel('Frequency [GHz]', size=18)
    ax2.set_ylabel('Log-Spectral Ratio', size=18)
    ax2.minorticks_on()
    ax2.grid(True)
    ax2.legend(loc='upper right', fontsize=12)
    
    fig2.tight_layout()
    fig2_path = os.path.join(output_dir, 'logspectral_ratio_examples.png')
    fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {fig2_path}')
    plt.close(fig2)

# --- [FIX-E] Figure 3: tand_interval_profile.png (区間方式・局所 tanδ) ---
mask_int = ~np.isnan(t_int_mid)                                            # [FIX-E]
if np.any(mask_int):                                                       # [FIX-E]
    fig3, ax3 = plt.subplots(figsize=(8, 10))                              # [FIX-E]
    ax3.plot(tand_int[mask_int], t_int_mid[mask_int],                      # [FIX-E]
             color='k', linestyle='-',                                     # [FIX-E]
             label=fr'Interval LSR ($\Delta t$ = {dt_int_actual_ns:.1f} ns)')  # [FIX-E]
    ax3.fill_betweenx(t_int_mid[mask_int],                                 # [FIX-E]
                      (tand_int - sigma_tand_int)[mask_int],               # [FIX-E]
                      (tand_int + sigma_tand_int)[mask_int],               # [FIX-E]
                      color='gray', alpha=0.4, label=r'$\pm 1\sigma$')     # [FIX-E]
    if np.any(~np.isnan(tand_theory_int)):                                 # [FIX-E]
        ax3.plot(tand_theory_int[mask_int], t_int_mid[mask_int],           # [FIX-E]
                 color='r', linestyle='--', label='Analytical (interval)') # [FIX-E]
    if np.any(~np.isnan(local_tand_model_int)):                            # [FIX-E]
        ax3.plot(local_tand_model_int[mask_int], t_int_mid[mask_int],      # [FIX-E]
                 color='g', linestyle=':',                                 # [FIX-E]
                 label=r'Model Local tan$\delta$')                         # [FIX-E]
    ax3.axhline(surf_t, color='gray', linestyle='--', lw=2,                # [FIX-E]
                label='Surface')                                           # [FIX-E]
    ax3.set_ylim(np.nanmax(t_int_mid), np.nanmin(t_int_mid))               # [FIX-E]
    ax3.set_xlabel(r'Loss Tangent (tan$\delta$)', size=18)                 # [FIX-E]
    ax3.set_ylabel('Delay time [ns]', size=18)                             # [FIX-E]
    ax3.minorticks_on()                                                    # [FIX-E]
    ax3.grid(True)                                                         # [FIX-E]
    ax3.legend(loc='lower left', fontsize=14)                              # [FIX-E]
    secax3 = ax3.secondary_xaxis('top',                                    # [FIX-E]
                                 functions=(safe_reciprocal,               # [FIX-E]
                                            safe_reciprocal))              # [FIX-E]
    secax3.set_xlabel(r'Q-factor (1 / tan$\delta$)', size=18)              # [FIX-E]
    secax3.set_ticks([10, 20, 50, 100, 200, 500])                          # [FIX-E]
    fig3.tight_layout()                                                    # [FIX-E]
    fig3_path = os.path.join(output_dir, 'tand_interval_profile.png')      # [FIX-E]
    fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')                  # [FIX-E]
    print(f'Saved: {fig3_path}')                                           # [FIX-E]
    plt.close(fig3)                                                        # [FIX-E]

print(f'OUTPUT DIR: {output_dir}')  # [FIX-A]
print(f'\nAll results saved to: {output_dir}')


# =============================================================================
# 検証ブロック
# =============================================================================
if __debug__:
    # 推定器の自己検証: 合成スペクトルによるテスト
    # 傾き [1/Hz] × (π・Δt[s])^-1 が無次元になることを確認
    tan_true = 0.01
    t_diff_ns = 10.0
    t_diff_s = t_diff_ns * 1e-9
    
    S1 = np.ones_like(f_win)
    S2 = np.exp(-np.pi * f_win * tan_true * t_diff_s)
    N_test = np.zeros_like(f_win)
    
    slope_test, _, _, _ = fit_log_spectral_ratio(
        f_win_hz, S2, S1, N_test, 
        snr_margin=-999.0, min_bins=10, s_clip=3.0
    )
    
    if not np.isnan(slope_test):
        tan_est = -slope_test / (np.pi * t_diff_s)
        rel_error = abs(tan_est - tan_true) / tan_true
        if rel_error < 0.01:
            pass # 検証成功
        else:
            print(f"Debug Warning: Estimator self-test failed. Expected {tan_true}, got {tan_est}")