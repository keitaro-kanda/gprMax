"""
スペクトル比法（Log-Spectral Ratio）による tanδ 差分プロファイル解析コード (k_diff_lsr.py)

氷あり/氷なしのB-scanデータに対してLSRプロファイル（固定基準方式および区間方式）を計算し、
その差分を評価してプロットとCSVを出力します。
"""

import os
import sys
import warnings
import json
import h5py
import glob
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as const
from gprMax.exceptions import CmdInputError

# 必要に応じてプロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.core.outputfiles_merge import get_output_data

# =============================================================================
# User input & Analytical settings
# =============================================================================
# [ADD] 指定された氷なし参照の絶対パス
NOICE_JSON = {
        0.01: '/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_3x4/Ice_Detection_NoRock/No_Ice/rand_amp_001/Bscan/Bscan.json',
        # 0.05: '/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_3x4/Ice_Detection_NoRock/No_Ice/rand_amp_005/Bscan/Bscan.json',
        # 0.05: '/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_10x4/Ice_Detection_NoRock/No_Ice/rand_amp_005/Bscan/Bscan.json'
        0.05: '/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_5x4/No_Ice/rand_amp_005/seed_0/Bscan/Bscan.json'
    }

# 入射波スペクトル計算用のA-scan出力ファイルパス
ascan_outfile_path = "/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_3x4/waveform_test/gaussiandot_1.25GHz_underground/result/Ascan.out" 

# =============================================================================
# Spectral-ratio parameters (Do NOT change these parameters)
# =============================================================================
win_len_samples = 256          # 解析窓長 [サンプル]（fs≈84.75 GHzで約3 ns）
hop_samples     = win_len_samples // 4   # 窓の送り幅
freq_min        = 0.5          # [GHz] 回帰帯域の下限
freq_max        = 2.5          # [GHz] 回帰帯域の上限
snr_margin_db   = 10.0         # ノイズ床からのマージン（帯域選択用）
n_min_bins      = 5            # 回帰に必要な最小ビン数
ref_margin_ns   = 5.0          # 基準窓中心 = 地表反射時刻 + このマージン
sigma_clip      = 3.0          # 回帰残差のシグマクリップ閾値（1回のみ）
MEAN_TRACE_REMOVAL = True      # True: 平均トレース除去（コヒーレントwake除去）
MIN_DT_NS   = 3.0              # 固定基準ペアの最小時間差 [ns]
DT_INT_NS   = 6.0              # 区間方式LSRの窓ペア間隔 [ns]
NOISE_GATE_NS = 5.0            # 記録末尾のこの区間をノイズ床推定に使用

antenna_height = 0.35          # [m] 送信機高さ
system_lag_ns  = 0.837         # [ns] システムラグ
rx_depth       = 0.10          # [m] 受信機の埋設深さ
f_center       = 450e6         # [Hz] アンカー周波数

# =============================================================================
# Helper Functions
# =============================================================================
def get_eps_static(z_m):
    z_cm = z_m * 100.0
    rho = 1.92 * (z_cm + 12.2) / (z_cm + 18.0)
    eps_static = 1.843 ** rho
    tan_d = 10 ** (0.033 * 20.0 + 0.231 * rho - 3.061)
    return eps_static, tan_d

def get_eps_regolith(z_m, omega, d_params, anchor_freq=450e6):
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

def surface_delay_ns(ant_height, sys_lag_ns):
    return ant_height * 2 / 0.3 + sys_lag_ns

def load_bscan(json_path):
    with open(json_path) as f:
        params = json.load(f)
    outfile = params['data']
    gpr_step = params['antenna_settings']['src_step']
    outputdata, dt = get_output_data(outfile, 1, 'Ez')
    return outputdata, dt, gpr_step, params

def extract_debye_params(params):
    debye_params = {'tau1': 4.6212e-11, 'tau2': 2.82195e-10, 'de_ratio': 0.261 / (0.261 + 0.088)}
    geom_json_path = params.get('geometry_settings', {}).get('geometry_json', '')
    in_dir = os.path.dirname(geom_json_path) if geom_json_path else ''
    
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
            except:
                pass
    return debye_params

def fit_log_spectral_ratio(f_hz, S_tgt, S_ref, noise_f, snr_margin, min_bins, s_clip):
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

def region_stats(t, d, t0, t1, corr_len_ns):
    """区間 [t0, t1] の Δ について統計量を計算"""
    mask = (t >= t0) & (t <= t1) & ~np.isnan(d)
    if np.sum(mask) == 0:
        return np.nan, np.nan, 0, np.nan
        
    t_val = t[mask]
    d_val = d[mask]
    mean_val = np.mean(d_val)
    std_val = np.std(d_val, ddof=1) if len(d_val) > 1 else 0.0
    duration = np.max(t_val) - np.min(t_val) if len(t_val) > 1 else 0.0
    
    n_eff = max(1.0, (duration / corr_len_ns) + 1.0)
    sem = std_val / np.sqrt(n_eff) if n_eff > 0 else 0.0
    z_val = mean_val / sem if sem > 0 else np.nan
    
    return mean_val, sem, n_eff, z_val

# =============================================================================
# Analytical Calculation (Shared)
# =============================================================================
def calc_analytical_setup():
    S0_calc, f_calc, omega, t_offset_ns = None, None, None, 0.0
    try:
        if os.path.exists(ascan_outfile_path):
            ascan_data, dt_ascan = get_output_data(ascan_outfile_path, 1, 'Ez')
            e_incident = ascan_data if ascan_data.ndim == 1 else ascan_data[:, 0]
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
    except Exception as e:
        print(f"Warning: Analytical calculation failed: {e}")
    return S0_calc, f_calc, omega, t_offset_ns

# =============================================================================
# Profile Computation Logic (Isolated to be reused)
# =============================================================================
def compute_lsr_profiles(outputdata, dt, gpr_step, params):
    dt_ns = dt * 1e9
    fs = 1.0 / dt_ns
    n_samples, n_traces = outputdata.shape
    
    data_proc = outputdata - outputdata.mean(axis=1, keepdims=True) if MEAN_TRACE_REMOVAL else outputdata
    debye_params = extract_debye_params(params)
    
    # --- Spectral setup ---
    n_windows = (n_samples - win_len_samples) // hop_samples + 1
    t_win_ns = np.zeros(n_windows)
    f_win = np.fft.rfftfreq(win_len_samples, d=dt)
    f_win_hz = f_win
    
    S_med = np.zeros((n_windows, len(f_win)))
    hann_win = np.hanning(win_len_samples)[:, None]
    t_axis = np.arange(n_samples) * dt_ns

    for i in range(n_windows):
        start = i * hop_samples
        end = start + win_len_samples
        t_win_ns[i] = t_axis[start + win_len_samples // 2]
        windowed_data = data_proc[start:end, :] * hann_win
        spectra = np.abs(np.fft.rfft(windowed_data, axis=0))
        S_med[i, :] = np.median(spectra, axis=1)

    # --- Reference Window & Noise Floor ---
    surf_t = surface_delay_ns(antenna_height, system_lag_ns)
    k_ref = -1
    for i, t in enumerate(t_win_ns):
        if t >= surf_t + ref_margin_ns:
            k_ref = i
            break
    if k_ref == -1:
        raise ValueError("Could not find a valid reference window.")

    ref_peak = np.max(S_med[k_ref, :])
    noise_floor_db = -100.0
    fallback_val = ref_peak * (10 ** (noise_floor_db / 20))
    N_f = np.full_like(f_win, fallback_val)
    
    noise_rows = np.where(t_win_ns >= t_axis[-1] - NOISE_GATE_NS)[0]
    if len(noise_rows) >= 2:
        N_f = np.median(S_med[noise_rows, :], axis=0)

    # --- Analytical Models ---
    S0_calc, f_calc, omega, t_offset_ns = calc_analytical_setup()
    t_model_array, S_model_array, local_tan_array = [], [], []
    if S0_calc is not None:
        max_depth = (t_axis[-1] * 1e-9) * const.c / 2 
        d_array = np.linspace(rx_depth, max_depth, 400)
        d_step = d_array[1] - d_array[0]
        cumulative_attenuation = np.zeros_like(omega)
        cumulative_time = np.zeros_like(omega)
        
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

    # --- 1. Fixed Reference Mode ---
    tand_eff = np.full(n_windows, np.nan)
    sigma_tand = np.full(n_windows, np.nan)
    tand_theory = np.full(n_windows, np.nan)
    
    for k in range(k_ref + 1, n_windows):
        if (t_win_ns[k] - t_win_ns[k_ref]) < MIN_DT_NS: continue
        dt_pair_s = (t_win_ns[k] - t_win_ns[k_ref]) * 1e-9
        slope, slope_err, _, _ = fit_log_spectral_ratio(
            f_win_hz, S_med[k, :], S_med[k_ref, :], N_f, snr_margin_db, n_min_bins, sigma_clip
        )
        if not np.isnan(slope):
            tand_eff[k] = -slope / (np.pi * dt_pair_s)
            sigma_tand[k] = slope_err / (np.pi * dt_pair_s)

    if S0_calc is not None:
        idx_ref_model = np.argmin(np.abs(t_model_array - t_win_ns[k_ref]))
        S_model_ref = S_model_array[idx_ref_model]
        N_f_theory = np.zeros_like(f_calc)
        
        for k in range(k_ref + 1, n_windows):
            if (t_win_ns[k] - t_win_ns[k_ref]) < MIN_DT_NS: continue
            idx_tgt = np.argmin(np.abs(t_model_array - t_win_ns[k]))
            if t_model_array[idx_tgt] <= t_model_array[idx_ref_model]: continue
            dt_pair_s = (t_model_array[idx_tgt] - t_model_array[idx_ref_model]) * 1e-9
            
            slope, _, _, _ = fit_log_spectral_ratio(
                f_calc, S_model_array[idx_tgt], S_model_ref, N_f_theory, -999.0, n_min_bins, sigma_clip
            )
            if not np.isnan(slope):
                tand_theory[k] = -slope / (np.pi * dt_pair_s)

    # --- 2. Interval Mode (Moving Window Pair) ---
    hop_ns = hop_samples * dt_ns
    n_int = max(1, int(round(DT_INT_NS / hop_ns)))
    
    tand_int             = np.full(n_windows, np.nan)
    sigma_tand_int       = np.full(n_windows, np.nan)
    t_int_mid            = np.full(n_windows, np.nan)
    tand_theory_int      = np.full(n_windows, np.nan)
    local_tand_model_int = np.full(n_windows, np.nan)

    for k in range(k_ref, n_windows - n_int):
        t_a = t_win_ns[k]
        t_b = t_win_ns[k + n_int]
        t_int_mid[k] = 0.5 * (t_a + t_b)
        dt_pair_s = (t_b - t_a) * 1e-9
        slope, slope_err, _, _ = fit_log_spectral_ratio(
            f_win_hz, S_med[k + n_int, :], S_med[k, :], N_f, snr_margin_db, n_min_bins, sigma_clip
        )
        if not np.isnan(slope):
            tand_int[k] = -slope / (np.pi * dt_pair_s)
            sigma_tand_int[k] = slope_err / (np.pi * dt_pair_s)

    if S0_calc is not None:
        for k in range(k_ref, n_windows - n_int):
            idx_a = np.argmin(np.abs(t_model_array - t_win_ns[k]))
            idx_b = np.argmin(np.abs(t_model_array - t_win_ns[k + n_int]))
            if t_model_array[idx_b] <= t_model_array[idx_a]: continue
            dt_pair_s = (t_model_array[idx_b] - t_model_array[idx_a]) * 1e-9
            
            slope, _, _, _ = fit_log_spectral_ratio(
                f_calc, S_model_array[idx_b], S_model_array[idx_a], N_f_theory, -999.0, n_min_bins, sigma_clip
            )
            if not np.isnan(slope):
                tand_theory_int[k] = -slope / (np.pi * dt_pair_s)
                
            idx_mid = np.argmin(np.abs(t_model_array - t_int_mid[k]))
            local_tand_model_int[k] = local_tan_array[idx_mid]

    res_fixed = {
        't_fixed': t_win_ns, 'tand_eff': tand_eff, 
        'sigma_tand': sigma_tand, 'tand_theory': tand_theory
    }
    res_int = {
        't_int': t_int_mid, 'tand_int': tand_int, 
        'sigma_tand_int': sigma_tand_int, 'tand_theory_int': tand_theory_int, 
        'local_tand_model_int': local_tand_model_int
    }
    return res_fixed, res_int

# =============================================================================
# Main Execution
# =============================================================================
def main():
    ice_json_path = input('Input ICE Bscan.json file path: ').strip()
    if not os.path.exists(ice_json_path):
        raise CmdInputError(f'JSON file {ice_json_path} does not exist')

    _sel = input('Select rand_amp for no-ice reference [0.01 / 0.05] (default 0.05): ').strip()
    rand_amp = 0.01 if _sel == '0.01' else 0.05
    print(f'Using no-ice reference for rand_amp = {rand_amp}')
    
    noice_json_path = NOICE_JSON.get(rand_amp)
    if not os.path.exists(noice_json_path):
        raise FileNotFoundError(f'No-ice JSON file {noice_json_path} does not exist')

    # [ADD] 指示通りのディレクトリ作成
    output_base_name = 'spectral_ratio_analysis_diff'
    if MEAN_TRACE_REMOVAL:
        output_base_name = 'spectral_ratio_analysis_meansub_diff'
    out_dir = os.path.join(os.path.dirname(os.path.abspath(ice_json_path)), output_base_name)
    os.makedirs(out_dir, exist_ok=True)

    print("\n--- Processing ICE data ---")
    data_i, dt_i, gs_i, params_i = load_bscan(ice_json_path)
    res_fixed_i, res_int_i = compute_lsr_profiles(data_i, dt_i, gs_i, params_i)
    
    print("\n--- Processing NO-ICE reference ---")
    data_n, dt_n, gs_n, params_n = load_bscan(noice_json_path)
    res_fixed_n, res_int_n = compute_lsr_profiles(data_n, dt_n, gs_n, params_n)

    # --- Sanity check ---
    if not np.allclose(res_fixed_i['t_fixed'], res_fixed_n['t_fixed'], atol=1e-6):
        raise ValueError("Time grids (t_fixed) do not match between ice and no-ice profiles.")
    if not np.allclose(res_int_i['t_int'], res_int_n['t_int'], equal_nan=True, atol=1e-6):
        raise ValueError("Time grids (t_int) do not match between ice and no-ice profiles.")

    # --- Difference Calculation ---
    t_fixed = res_fixed_i['t_fixed']
    d_tand = res_fixed_i['tand_eff'] - res_fixed_n['tand_eff']
    sigma_diff = np.sqrt(res_fixed_i['sigma_tand']**2 + res_fixed_n['sigma_tand']**2)

    t_int = res_int_i['t_int']
    d_tand_int = res_int_i['tand_int'] - res_int_n['tand_int']
    sigma_diff_int = np.sqrt(res_int_i['sigma_tand_int']**2 + res_int_n['sigma_tand_int']**2)

    # --- Statistical Analysis ---
    print("\n================ Regional Statistics ================")
    # Interval mode
    mean_int_full, sem_int_full, neff_int_full, z_int_full = region_stats(t_int, d_tand_int, 17.4, 34.8, 9.0)
    mean_int_shal, sem_int_shal, neff_int_shal, z_int_shal = region_stats(t_int, d_tand_int, 0.0, 17.4, 9.0)
    print(f"[Interval Mode] Layer (17.4-34.8ns): {mean_int_full:.5f} +/- {sem_int_full:.5f} (n_eff={neff_int_full:.1f}, z={z_int_full:.2f}) | Theory: -0.0023")
    print(f"[Interval Mode] Shallow (<17.4ns)  : {mean_int_shal:.5f} +/- {sem_int_shal:.5f} (n_eff={neff_int_shal:.1f}, z={z_int_shal:.2f}) | Theory: 0.0")

    # Fixed mode
    mean_fix_full, sem_fix_full, neff_fix_full, z_fix_full = region_stats(t_fixed, d_tand, 14.4, 37.8, 6.0)
    mean_fix_shal, sem_fix_shal, neff_fix_shal, z_fix_shal = region_stats(t_fixed, d_tand, 0.0, 14.4, 6.0)
    print(f"[Fixed Mode] Layer (14.4-37.8ns)   : {mean_fix_full:.5f} +/- {sem_fix_full:.5f} (n_eff={neff_fix_full:.1f}, z={z_fix_full:.2f}) | Theory: -0.0019")
    print(f"[Fixed Mode] Shallow (<14.4ns)     : {mean_fix_shal:.5f} +/- {sem_fix_shal:.5f} (n_eff={neff_fix_shal:.1f}, z={z_fix_shal:.2f}) | Theory: 0.0")

    # --- Output CSV ---
    csv_int_path = os.path.join(out_dir, 'lsr_diff_interval.csv')
    with open(csv_int_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t_mid_ns', 'd_tand_int', 'sigma_diff'])
        for i in range(len(t_int)):
            if not np.isnan(t_int[i]):
                writer.writerow([t_int[i], d_tand_int[i], sigma_diff_int[i]])
                
    csv_fix_path = os.path.join(out_dir, 'lsr_diff_fixed.csv')
    with open(csv_fix_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t_ns', 'd_tand', 'sigma_diff'])
        for i in range(len(t_fixed)):
            writer.writerow([t_fixed[i], d_tand[i], sigma_diff[i]])

    # --- Plotting ---
    plt.rcParams['font.size'] = 14

    # 1. Interval Plot
    mask_int = ~np.isnan(t_int)
    if np.any(mask_int):
        fig1, ax1 = plt.subplots(figsize=(8, 10))
        ax1.plot(d_tand_int[mask_int], t_int[mask_int], color='k', label=r'$\Delta \tan\delta$ (Interval)')
        ax1.fill_betweenx(t_int[mask_int], 
                          d_tand_int[mask_int] - sigma_diff_int[mask_int], 
                          d_tand_int[mask_int] + sigma_diff_int[mask_int], 
                          color='gray', alpha=0.4)
        ax1.axhspan(17.4, 34.8, color='lightblue', alpha=0.3, label='Ice Layer')
        ax1.axvline(0, color='gray', linestyle='-', lw=1)
        
        # 予測シグネチャ (箱型)
        t_theory = np.linspace(0, 50, 200)
        sig_theory = np.where((t_theory >= 17.4) & (t_theory <= 34.8), -0.0023, 0.0)
        ax1.plot(sig_theory, t_theory, color='red', linestyle='--', label='Theory Signature')
        
        # エラーバー (統計量)
        if not np.isnan(mean_int_full):
            ax1.errorbar(mean_int_full, (17.4+34.8)/2, xerr=sem_int_full, color='blue', fmt='o', capsize=5, label='Layer Mean')
        if not np.isnan(mean_int_shal):
            ax1.errorbar(mean_int_shal, 17.4/2, xerr=sem_int_shal, color='green', fmt='o', capsize=5, label='Shallow Mean')

        ax1.set_ylim(np.nanmax(t_int), np.nanmin(t_int))
        ax1.set_xlabel(r'$\Delta \tan\delta$', size=18)
        ax1.set_ylabel('Delay time [ns]', size=18)
        ax1.set_title(f"LSR Interval Diff (rand={rand_amp})")
        ax1.grid(True)
        ax1.legend(loc='lower left', fontsize=12)
        fig1.tight_layout()
        fig1.savefig(os.path.join(out_dir, 'lsr_diff_interval.png'), dpi=300, bbox_inches='tight')
        plt.close(fig1)

    # 2. Fixed Plot
    fig2, ax2 = plt.subplots(figsize=(8, 10))
    ax2.plot(d_tand, t_fixed, color='k', label=r'$\Delta \tan\delta$ (Fixed)')
    ax2.fill_betweenx(t_fixed, d_tand - sigma_diff, d_tand + sigma_diff, color='gray', alpha=0.4)
    ax2.axhspan(14.4, 37.8, color='lightblue', alpha=0.3, label='Ice Layer')
    ax2.axvline(0, color='gray', linestyle='-', lw=1)

    # 予測シグネチャ (ランプ近似)
    t_theory_fix = np.linspace(0, 50, 200)
    sig_theory_fix = np.interp(t_theory_fix, [14.4, 37.8], [0.0, -0.0019])
    ax2.plot(sig_theory_fix, t_theory_fix, color='red', linestyle='--', label='Theory Signature')
    
    if not np.isnan(mean_fix_full):
        ax2.errorbar(mean_fix_full, (14.4+37.8)/2, xerr=sem_fix_full, color='blue', fmt='o', capsize=5, label='Layer Mean')
    if not np.isnan(mean_fix_shal):
        ax2.errorbar(mean_fix_shal, 14.4/2, xerr=sem_fix_shal, color='green', fmt='o', capsize=5, label='Shallow Mean')

    ax2.set_ylim(t_fixed[-1], t_fixed[0])
    ax2.set_xlabel(r'$\Delta \tan\delta$', size=18)
    ax2.set_ylabel('Delay time [ns]', size=18)
    ax2.set_title(f"LSR Fixed Diff (rand={rand_amp})")
    ax2.grid(True)
    ax2.legend(loc='lower left', fontsize=12)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, 'lsr_diff_fixed.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)

    print(f"\nAll diff results saved to: {out_dir}")

if __name__ == '__main__':
    main()