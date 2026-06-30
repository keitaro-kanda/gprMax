import os
import sys
import warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import h5py
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import constants as const
from scipy.ndimage import gaussian_filter
import mpl_toolkits.axes_grid1 as axgrid1
from gprMax.exceptions import CmdInputError
from tools.core.outputfiles_merge import get_output_data

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

print(f'dt = {dt*1e12:.4f} ps,  fs = {fs:.2f} GHz,  fs/2 = {fs/2:.2f} GHz')
print(f'B-scan shape (samples, traces): {outputdata.shape}')

# =============================================================================
# Extract Debye Parameters from .in file
# =============================================================================
debye_params = {'de1': 0.261, 'tau1': 4.6212e-11, 'de2': 0.088, 'tau2': 2.82195e-10}
geom_json_path = params.get('geometry_settings', {}).get('geometry_json', '')
in_dir = os.path.dirname(geom_json_path)
in_file_found = False

if in_dir and os.path.exists(in_dir):
    in_files = glob.glob(os.path.join(in_dir, '*.in'))
    for in_file in in_files:
        try:
            with open(in_file, 'r', encoding='utf-8') as fin:
                content = fin.read()
                m_de1 = re.search(r'DEBYE_DE1\s*=\s*([0-9\.eE\+\-]+)', content)
                m_tau1 = re.search(r'DEBYE_TAU1\s*=\s*([0-9\.eE\+\-]+)', content)
                m_de2 = re.search(r'DEBYE_DE2\s*=\s*([0-9\.eE\+\-]+)', content)
                m_tau2 = re.search(r'DEBYE_TAU2\s*=\s*([0-9\.eE\+\-]+)', content)
                
                if m_de1 and m_tau1:
                    debye_params['de1'] = float(m_de1.group(1))
                    debye_params['tau1'] = float(m_tau1.group(1))
                    in_file_found = True
                if m_de2 and m_tau2:
                    debye_params['de2'] = float(m_de2.group(1))
                    debye_params['tau2'] = float(m_tau2.group(1))
        except Exception as e:
            print(f"Warning: Could not parse {in_file}: {e}")

if not in_file_found:
    print("Warning: Could not extract Debye parameters from .in file. Using default values.")
print("Debye Parameters used:", debye_params)

# =============================================================================
# STFT parameters
# =============================================================================
nperseg  = 256
noverlap = nperseg * 3 // 4
window   = 'hann'

freq_min = 0.25    # [GHz]
freq_max = 6.0    # [GHz]

power_threshold_db = -125.0   # [dB]
sigma = (3, 3)

# =============================================================================
# STFT and frequency/time axes
# =============================================================================
f_axis, t_axis, _ = signal.stft(outputdata[:, 0], fs=fs, window=window,
                                 nperseg=nperseg, noverlap=noverlap)

freq_mask  = (f_axis >= freq_min) & (f_axis <= freq_max)
valid_freq = f_axis[freq_mask]
n_time     = t_axis.size

print(f'STFT: nperseg={nperseg}, df={fs/nperseg:.3f} GHz')
print(f'Frequency band: {valid_freq[0]:.3f} – {valid_freq[-1]:.3f} GHz '
      f'({freq_mask.sum()} bins)')

# =============================================================================
# Compute centroid map from STFT
# =============================================================================
eps = 1e-30

centroid_map = np.zeros((n_time, n_traces))
power_map    = np.zeros((n_time, n_traces))

for itrace in range(n_traces):
    _, _, Zxx = signal.stft(outputdata[:, itrace], fs=fs, window=window,
                            nperseg=nperseg, noverlap=noverlap)
    power = np.abs(Zxx[freq_mask, :]) ** 2          
    total = power.sum(axis=0)                        

    centroid_map[:, itrace] = (valid_freq[:, None] * power).sum(axis=0) / (total + eps)
    power_map[:, itrace] = total

# =============================================================================
# Power mask: low-SNR pixels -> NaN
# =============================================================================
trace_peak = power_map.max(axis=0, keepdims=True)
with np.errstate(divide='ignore', invalid='ignore'):
    power_rel_db = 10.0 * np.log10(
        np.where(trace_peak > 0, power_map / (trace_peak + eps), eps))

valid_mask = power_rel_db >= power_threshold_db

centroid_masked  = np.where(valid_mask, centroid_map,  np.nan)

# =============================================================================
# NaN-aware Gaussian smoothing helper
# =============================================================================
def smooth_masked(data, mask, sigma):
    filled = np.where(mask, data, 0.0)
    sm_data   = gaussian_filter(filled,              sigma=sigma)
    sm_weight = gaussian_filter(mask.astype(float),  sigma=sigma)
    out = np.full_like(sm_data, np.nan)
    np.divide(sm_data, sm_weight, out=out, where=(sm_weight > 1e-6))
    return out

centroid_smooth  = smooth_masked(centroid_map,  valid_mask, sigma)

# =============================================================================
# Frequency-shift-rate maps  [GHz/ns]
# =============================================================================
dt_stft = t_axis[1] - t_axis[0]   # [ns]

def shift_rate(freq_map):
    return np.gradient(freq_map, dt_stft, axis=0)

shiftrate_centroid_raw    = shift_rate(centroid_masked)
shiftrate_centroid_smooth = shift_rate(centroid_smooth)

# =============================================================================
# Analytical Frequency Shift Calculation (Depth + Debye + Time Offset for Buried Rx)
# =============================================================================
try:
    if os.path.exists(ascan_outfile_path):
        ascan_data, dt_ascan = get_output_data(ascan_outfile_path, 1, 'Ez')
        
        if ascan_data.ndim == 1:
            e_incident = ascan_data
        else:
            e_incident = ascan_data[:, 0]
        
        # 入射波のFFT (S_0)
        N = len(e_incident)
        freq_ascan = np.fft.rfftfreq(N, d=dt_ascan)
        S0_omega = np.fft.rfft(e_incident)
        
        # 物理的に意味のある帯域
        f_min_hz = freq_min * 1e9
        f_max_hz = freq_max * 1e9
        band_mask = (freq_ascan >= f_min_hz) & (freq_ascan <= f_max_hz)
        
        f_calc = freq_ascan[band_mask]
        S0_calc = S0_omega[band_mask]
        omega = 2 * np.pi * f_calc  # 周波数依存計算用の配列
        
        # --- gprMaxモデルに基づく物理定数・リファレンス値 ---
        f_center = 1.25e9
        omega0 = 2.0 * np.pi * f_center
        eps0 = 8.8541878128e-12
        
        ice_top, ice_bot = 0.50, 0.70
        f_ice_vol = 0.1
        eps_ice_comp = 3.17 - 1j * (3.17 * 6e-5)
        
        EPS_STATIC_CC = 4.212
        DEBYE_DE1  = 0.261
        DEBYE_TAU1 = 4.6212e-11
        DEBYE_DE2  = 0.088
        DEBYE_TAU2 = 2.82195e-10
        
        # --- 時間遅延（Time Offset）の計算 ---
        antenna_height = 0.35    # [m] 送信機高さ
        system_lag_ns  = 0.837   # [ns] システムラグ
        rx_depth       = 0.10    # [m] 受信機の埋設深さ
        
        # 1. 空中の往復伝搬時間 [ns]
        t_air_ns = (2.0 * antenna_height / const.c) * 1e9 
        
        # 2. 地表面(d=0)から受信機(d=0.1)までの往復伝搬時間 [ns] を計算
        # （深さ依存の誘電率を用いて細かく積分）
        d_sub = np.linspace(0, rx_depth, 50)
        rho_sub = 1.92 * (d_sub + 12.2) / (d_sub + 18.0)
        eps_sub = 1.919 ** rho_sub
        v_sub = const.c / np.sqrt(eps_sub)
        dt_sub = d_sub[1] - d_sub[0]
        t_ground_start_ns = np.sum(2.0 * dt_sub / v_sub) * 1e9
        
        # 3. 赤点線の開始時刻 = システムラグ + 空中往復 + 地中10cm往復
        t_offset_ns = system_lag_ns + t_air_ns + t_ground_start_ns
        
        print(f"Time offset (depth {rx_depth}m reflection): {t_offset_ns:.3f} ns "
              f"(Lag: {system_lag_ns} + Air: {t_air_ns:.3f} + Ground({rx_depth}m): {t_ground_start_ns:.3f} ns)")
        # ---------------------------------------------------
        
        # 解析する最大深さと刻み幅の設定
        max_depth = (t_axis[-1] * 1e-9) * const.c / 2 
        # 計算開始深さを受信機の深さに設定
        d_array = np.linspace(rx_depth, max_depth, 400)
        d_step = d_array[1] - d_array[0]
        
        f_peak_d = []
        t_delay_d = []
        
        cumulative_attenuation = np.zeros_like(omega)
        cumulative_time = np.zeros_like(omega)
        
        for i, d in enumerate(d_array):
            # 1. 深さ依存のベースライン（中心周波数での代表値）
            rho = 1.92 * (d + 12.2) / (d + 18.0)
            eps_reg_real = 1.919 ** rho
            tan_d_reg = 10 ** (0.312 * rho - 2.36)
            eps_reg_comp = eps_reg_real - 1j * (eps_reg_real * tan_d_reg)
            
            # 2. MG混合則による氷層の追加
            f_vol = f_ice_vol if (ice_top <= d <= ice_bot) else 0.0
            eps_e = eps_reg_comp
            eps_i = eps_ice_comp
            eps_eff_comp = (eps_e 
                            + 3.0 * f_vol * eps_e * (eps_i - eps_e) 
                            / (eps_i + 2.0 * eps_e - f_vol * (eps_i - eps_e)))
            
            eps_r_eff = np.real(eps_eff_comp)
            sigma_eff = -np.imag(eps_eff_comp) * omega0 * eps0
            
            # 3. Debyeパラメータのスケーリング
            cell_scale = eps_r_eff / EPS_STATIC_CC
            de1_eff = DEBYE_DE1 * cell_scale
            de2_eff = DEBYE_DE2 * cell_scale
            eps_inf_eff = max(eps_r_eff - de1_eff - de2_eff, 1.0)
            
            # 4. 全帯域に対する複素誘電率の計算
            eps_complex_w = (eps_inf_eff 
                             + de1_eff / (1 + 1j * omega * DEBYE_TAU1) 
                             + de2_eff / (1 + 1j * omega * DEBYE_TAU2) 
                             - 1j * sigma_eff / (omega * eps0))
            
            # 5. 各周波数における局所的な減衰率 alpha と速度 v
            alpha_d = - (omega / const.c) * np.imag(np.sqrt(eps_complex_w))
            v_d = const.c / np.real(np.sqrt(eps_complex_w))
            
            # 6. 積分（累積和）: i=0 (d=0.1) の時はゼロのまま
            if i > 0:
                cumulative_attenuation += alpha_d * d_step
                cumulative_time += 2 * d_step / v_d
                
            # 7. 減衰スペクトルの計算
            S_d_w = S0_calc * np.exp(-2 * cumulative_attenuation)
            power = np.abs(S_d_w)**2
            
            # 8. 中心周波数の計算
            f_peak = np.trapz(f_calc * power, f_calc) / np.trapz(power, f_calc)
            f_peak_d.append(f_peak)
            
            # 9. 遅れ時間換算（地中伝搬時間 + オフセット時間）
            t_delay_ground = np.interp(f_peak, f_calc, cumulative_time)
            t_total_ns = t_offset_ns + (t_delay_ground * 1e9)
            t_delay_d.append(t_total_ns)
        
        f_peak_d = np.array(f_peak_d) / 1e9 # [GHz]
        t_delay_d = np.array(t_delay_d)
        
        # t_axisに合わせて補間
        analytical_f_peak_profile = np.interp(t_axis, t_delay_d, f_peak_d, left=np.nan, right=np.nan)
        # 解析的なシフトレートの算出 (単純前進差分への変更がまだの場合はここも合わせて修正推奨です)
        analytical_shiftrate_profile = np.gradient(analytical_f_peak_profile, dt_stft)
        print("Analytical frequency shift successfully calculated with buried Rx offset.")
        
    else:
        print(f"Warning: A-scan file not found at {ascan_outfile_path}. Analytical calculation skipped.")
        analytical_f_peak_profile = None
        analytical_shiftrate_profile = None

except Exception as e:
    print(f"Warning: Analytical calculation failed: {e}")
    analytical_f_peak_profile = None
    analytical_shiftrate_profile = None

# =============================================================================
# Output directory
# =============================================================================
output_base_name = 'centroid_frequency_analysis'
output_dir = os.path.join(os.path.dirname(os.path.abspath(json_file_path)),
                          output_base_name)
os.makedirs(output_dir, exist_ok=True)

extent = [0, n_traces * GPR_step, t_axis[-1], t_axis[0]]

# =============================================================================
# 1D profiles: trace-averaged (median and 25-75% percentiles)
# =============================================================================
# 全てNaNの行に対するnp.nanpercentileのRuntimeWarningを抑制
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    
    # Centroid frequency (raw)
    prof_cen_raw_med = np.nanmedian(centroid_masked, axis=1)
    prof_cen_raw_p25 = np.nanpercentile(centroid_masked, 25, axis=1)
    prof_cen_raw_p75 = np.nanpercentile(centroid_masked, 75, axis=1)
    
    # Centroid frequency (smooth)
    prof_cen_sm_med = np.nanmedian(centroid_smooth, axis=1)
    prof_cen_sm_p25 = np.nanpercentile(centroid_smooth, 25, axis=1)
    prof_cen_sm_p75 = np.nanpercentile(centroid_smooth, 75, axis=1)
    
    # Shift rate (raw)
    prof_sr_raw_med = np.nanmedian(shiftrate_centroid_raw, axis=1)
    prof_sr_raw_p25 = np.nanpercentile(shiftrate_centroid_raw, 25, axis=1)
    prof_sr_raw_p75 = np.nanpercentile(shiftrate_centroid_raw, 75, axis=1)
    
    # Shift rate (smooth)
    prof_sr_sm_med = np.nanmedian(shiftrate_centroid_smooth, axis=1)
    prof_sr_sm_p25 = np.nanpercentile(shiftrate_centroid_smooth, 25, axis=1)
    prof_sr_sm_p75 = np.nanpercentile(shiftrate_centroid_smooth, 75, axis=1)

# Colour scale for frequency maps
all_freq_valid = np.concatenate([
    centroid_masked[np.isfinite(centroid_masked)]
])
if all_freq_valid.size > 0:
    vmin_f, vmax_f = 0.5, 2.0 # [GHz]
else:
    vmin_f, vmax_f = freq_min, freq_max
print(f'Frequency colour scale: {vmin_f:.3f} – {vmax_f:.3f} GHz')

# Colour scale for shift-rate maps
all_sr_valid = np.concatenate([
    shiftrate_centroid_raw[np.isfinite(shiftrate_centroid_raw)]
])
if all_sr_valid.size > 0:
    sr_abs = np.percentile(np.abs(all_sr_valid), 95)
else:
    sr_abs = 1.0
vmin_sr, vmax_sr = -sr_abs, sr_abs
print(f'Shift-rate colour scale: {vmin_sr:.4f} – {vmax_sr:.4f} GHz/ns')

# =============================================================================
# Plot helpers
# =============================================================================
def plot_freq_map(data, fname, prof_med, prof_p25, prof_p75, analytical_profile=None):
    # --- 時間遅延（Time Offset）の計算 ---
    antenna_height = 0.35    # [m] 送信機高さ
    system_lag_ns  = 0.837   # [ns] システムラグ
    initial_delay = antenna_height * 2 / 0.3 + system_lag_ns # [ns]

    fig, axes = plt.subplots(
        nrows=1, ncols=2,
        width_ratios=[3, 1],
        height_ratios=[1],
        figsize=(14, 6),
    )
    ax = axes[0]
    im = ax.imshow(data, extent=extent, aspect='auto',
                   cmap='jet', vmin=vmin_f, vmax=vmax_f)
    ax.axhline(initial_delay, color='gray', linestyle='--', lw=2)
    ax.set_xlabel('Distance [m]', size=18)
    ax.set_ylabel('Delay time [ns]', size=18)
    ax.tick_params(labelsize=14)
    ax.grid()
    divider = axgrid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Frequency [GHz]', size=18)
    cbar.ax.tick_params(labelsize=14)

    ax2 = axes[1]
    ax2.fill_betweenx(t_axis, prof_p25, prof_p75, color='gray', alpha=0.4, label='IQR (25-75%)')
    ax2.plot(prof_med, t_axis, color='k', linestyle='-', label='Median')
    
    if analytical_profile is not None:
        ax2.plot(analytical_profile, t_axis, color='r', linestyle='--', label='Analytical')
    
    ax2.axhline(initial_delay, color='gray', linestyle='--', lw=2, label='Suarface')

    ax2.legend(fontsize=14)
    ax2.set_xlabel('Frequency [GHz]', size=18)
    ax2.set_ylabel('Delay time [ns]', size=18)
    ax2.set_ylim(t_axis[-1], t_axis[0])
    ax2.tick_params(labelsize=14)
    ax2.grid()

    plt.tight_layout()
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('Saved:', path)
    plt.close(fig)


def plot_shiftrate_map(data, fname, prof_med, prof_p25, prof_p75, analytical_profile=None):
    # --- 時間遅延（Time Offset）の計算 ---
    antenna_height = 0.35    # [m] 送信機高さ
    system_lag_ns  = 0.837   # [ns] システムラグ
    initial_delay = antenna_height * 2 / 0.3 + system_lag_ns # [ns]

    fig, axes = plt.subplots(
        nrows=1, ncols=2,
        width_ratios=[3, 1],
        height_ratios=[1],
        figsize=(14, 6),
    )
    ax = axes[0]
    im = ax.imshow(data, extent=extent, aspect='auto',
                   cmap='RdBu_r', vmin=vmin_sr, vmax=vmax_sr)
    ax.axhline(initial_delay, color='gray', linestyle='--', lw=2)
    ax.set_xlabel('Distance [m]', size=18)
    ax.set_ylabel('Delay time [ns]', size=18)
    ax.tick_params(labelsize=14)
    ax.grid()
    divider = axgrid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Frequency shift rate [GHz/ns]', size=18)
    cbar.ax.tick_params(labelsize=14)

    ax2 = axes[1]
    ax2.fill_betweenx(t_axis, prof_p25, prof_p75, color='gray', alpha=0.4, label='IQR (25-75%)')
    ax2.plot(prof_med, t_axis, color='k', linestyle='-', label='Median')
    
    if analytical_profile is not None:
        ax2.plot(analytical_profile, t_axis, color='r', linestyle='--', label='Analytical')

    ax2.axhline(initial_delay, color='gray', linestyle='--', lw=2, label='Surface')

    ax2.legend(fontsize=14)
    ax2.set_xlabel('Shift rate [GHz/ns]', size=18)
    ax2.set_ylabel('Delay time [ns]', size=18)
    ax2.set_ylim(t_axis[-1], t_axis[0])
    ax2.tick_params(labelsize=14)
    ax2.grid()

    plt.tight_layout()
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('Saved:', path)
    plt.close(fig)

# =============================================================================
# Produce maps
# =============================================================================
plot_freq_map(
    centroid_masked,
    'centroid_raw.png',
    prof_cen_raw_med, prof_cen_raw_p25, prof_cen_raw_p75,
    analytical_f_peak_profile)

plot_freq_map(
    centroid_smooth,
    'centroid_smooth.png',
    prof_cen_sm_med, prof_cen_sm_p25, prof_cen_sm_p75,
    analytical_f_peak_profile)

plot_shiftrate_map(
    shiftrate_centroid_raw,
    'centroid_shiftrate_raw.png',
    prof_sr_raw_med, prof_sr_raw_p25, prof_sr_raw_p75,
    analytical_shiftrate_profile)

plot_shiftrate_map(
    shiftrate_centroid_smooth,
    'centroid_shiftrate_smooth.png',
    prof_sr_sm_med, prof_sr_sm_p25, prof_sr_sm_p75,
    analytical_shiftrate_profile)

# =============================================================================
# Analytical Spectrum Comparison Plot (Normalized dB scale & Mask Threshold)
# =============================================================================
if analytical_f_peak_profile is not None:
    # 基準となる受信機の深さ（rx_depth = 0.1 m）を開始点とし、0.0mは除外します
    target_depths = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    fig_spec, ax_spec = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(target_depths)))
    
    # 入射波(d=0.1)のパワーを計算し、その最大値を0 dBの基準とする
    power_0 = np.abs(S0_calc)**2
    max_power_0 = np.max(power_0)
    
    for i, d in enumerate(target_depths):
        # 基準深さ(rx_depth)から目的の深さ(d)までの減衰を積分で計算
        if d <= rx_depth:
            cum_alpha = np.zeros_like(omega)
        else:
            # 積分用の細かい刻み（rx_depth 〜 d まで）
            d_sub = np.linspace(rx_depth, d, 200)
            dz = d_sub[1] - d_sub[0]
            cum_alpha = np.zeros_like(omega)
            
            for z in d_sub:
                # 1. 深さ依存のベースライン
                rho = 1.92 * (z + 12.2) / (z + 18.0)
                eps_reg_real = 1.919 ** rho
                tan_d_reg = 10 ** (0.312 * rho - 2.36)
                eps_reg_comp = eps_reg_real - 1j * (eps_reg_real * tan_d_reg)
                
                # 2. 氷層のMG混合
                f_vol = f_ice_vol if (ice_top <= z <= ice_bot) else 0.0
                eps_e = eps_reg_comp
                eps_i = eps_ice_comp
                eps_eff_comp = (eps_e 
                                + 3.0 * f_vol * eps_e * (eps_i - eps_e) 
                                / (eps_i + 2.0 * eps_e - f_vol * (eps_i - eps_e)))
                
                eps_r_eff = np.real(eps_eff_comp)
                sigma_eff = -np.imag(eps_eff_comp) * omega0 * eps0
                
                # 3. Debyeパラメータスケーリング
                cell_scale = eps_r_eff / EPS_STATIC_CC
                de1_eff = DEBYE_DE1 * cell_scale
                de2_eff = DEBYE_DE2 * cell_scale
                eps_inf_eff = max(eps_r_eff - de1_eff - de2_eff, 1.0)
                
                # 4. 複素誘電率
                eps_complex_w = (eps_inf_eff 
                                 + de1_eff / (1 + 1j * omega * DEBYE_TAU1) 
                                 + de2_eff / (1 + 1j * omega * DEBYE_TAU2) 
                                 - 1j * sigma_eff / (omega * eps0))
                
                # 5. 局所的な減衰率
                alpha_z = - (omega / const.c) * np.imag(np.sqrt(eps_complex_w))
                
                # 6. 積分の累積 (最初の点は dz=0 と見なせるため z > rx_depth で加算)
                if z > rx_depth:
                    cum_alpha += alpha_z * dz
        
        # 積分された全減衰量を用いてスペクトルを計算
        S_d_w = S0_calc * np.exp(-2 * cum_alpha)
        power = np.abs(S_d_w)**2
        
        # Calculate center frequency (MUST be calculated in linear scale)
        f_peak = np.trapz(f_calc * power, f_calc) / np.trapz(power, f_calc)
        f_peak_ghz = f_peak / 1e9
        
        # 入射波の最大パワーで規格化し、dBへ変換
        power_norm = power / max_power_0
        power_db = 10.0 * np.log10(power_norm + 1e-30)
        f_calc_ghz = f_calc / 1e9
        
        # Plot spectrum in dB
        label_str = f'Depth {d:.1f} m (fc = {f_peak_ghz:.2f} GHz)'
        ax_spec.plot(f_calc_ghz, power_db, color=colors[i], label=label_str)
        
        # Plot center frequency as a vertical dashed line
        ax_spec.axvline(f_peak_ghz, color=colors[i], linestyle='--', alpha=0.7)
        
    # パワーマスクの基準値（スクリプト上部で定義した power_threshold_db）を赤の点線でプロット
    ax_spec.axhline(power_threshold_db, color='red', linestyle=':', linewidth=2, 
                    label=f'Mask Threshold ({power_threshold_db} dB)')
        
    ax_spec.set_xlabel('Frequency [GHz]', size=18)
    ax_spec.set_ylabel('Normalized Power [dB]', size=18)
    ax_spec.set_xlim(freq_min, freq_max)
    
    # Y軸の表示範囲を調整（閾値の少し下から0 dBの少し上まで）
    ax_spec.set_ylim(bottom=power_threshold_db - 15, top=5) 

    ax_spec.tick_params(labelsize=14)
    ax_spec.grid(True)
    
    # 凡例を外側に配置してグラフと被らないようにする
    ax_spec.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
    
    plt.tight_layout()
    spec_plot_path = os.path.join(output_dir, 'analytical_spectra_comparison_normalized_db.png')
    fig_spec.savefig(spec_plot_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {spec_plot_path}')


print(f'\nAll figures saved to: {output_dir}')

plt.show()