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
ascan_outfile_path = "/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_3x4/waveform_test/gaussiandot_1.25GHz/result/Ascan.out" 
# [EDIT HERE] 無限大周波数における比誘電率（ベースの誘電率）
eps_inf = 3.792 # Boivin+2022, Table 4, 20 wt% Ilmenite

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
nperseg  = 512
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
# Analytical Frequency Shift Calculation
# =============================================================================
try:
    if os.path.exists(ascan_outfile_path):
        ascan_data, dt_ascan = get_output_data(ascan_outfile_path, 1, 'Ez')
        
        # 配列の次元数に応じた処理（A-scanの単一ファイルの場合は1次元になることがあるため）
        if ascan_data.ndim == 1:
            e_incident = ascan_data
        else:
            e_incident = ascan_data[:, 0]
        
        # 入射波のFFT (S_0)
        N = len(e_incident)
        freq_ascan = np.fft.rfftfreq(N, d=dt_ascan)
        S0_omega = np.fft.rfft(e_incident)
        
        # 物理的に意味のある帯域（STFTと同じ）で計算
        f_min_hz = freq_min * 1e9
        f_max_hz = freq_max * 1e9
        band_mask = (freq_ascan >= f_min_hz) & (freq_ascan <= f_max_hz)
        
        f_calc = freq_ascan[band_mask]
        S0_calc = S0_omega[band_mask]
        omega = 2 * np.pi * f_calc
        
        def eps_r(w):
            e = eps_inf
            e += debye_params['de1'] / (1 + 1j * w * debye_params['tau1'])
            e += debye_params['de2'] / (1 + 1j * w * debye_params['tau2'])
            return e
        
        eps_r_w = eps_r(omega)
        # 減衰率: np.imag(np.sqrt(eps_r_w))は負になるため、- (w/c) * Im[...] は正になる
        alpha = - (omega / const.c) * np.imag(np.sqrt(eps_r_w))
        
        # STFTの最大時間から、計算すべき最大深さを見積もる
        v_bg = const.c / np.sqrt(eps_inf)
        max_depth = (t_axis[-1] * 1e-9) * v_bg / 2
        d_array = np.linspace(0, max_depth, 300)
        
        f_peak_d = []
        t_delay_d = []
        
        for d in d_array:
            # 減衰スペクトル S(d, w) = S0(w) * exp(-2 * alpha * d)
            S_d_w = S0_calc * np.exp(-2 * alpha * d)
            power = np.abs(S_d_w)**2
            
            # 中心周波数
            f_peak = np.trapz(f_calc * power, f_calc) / np.trapz(power, f_calc)
            f_peak_d.append(f_peak)
            
            # 遅れ時間換算
            eps_peak = eps_r(2 * np.pi * f_peak)
            v_peak = const.c / np.real(np.sqrt(eps_peak))
            t_delay = 2 * d / v_peak
            t_delay_d.append(t_delay * 1e9) # [ns]
        
        f_peak_d = np.array(f_peak_d) / 1e9 # [GHz]
        t_delay_d = np.array(t_delay_d)
        
        # t_axisに合わせて補間
        analytical_f_peak_profile = np.interp(t_axis, t_delay_d, f_peak_d, left=np.nan, right=np.nan)
        # 解析的なシフトレートの算出
        analytical_shiftrate_profile = np.gradient(analytical_f_peak_profile, dt_stft)
        print("Analytical frequency shift successfully calculated.")
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
# 1D profiles: trace-averaged (nanmean over all traces)
# =============================================================================
# 全てNaNの行に対するnp.nanmeanのRuntimeWarningを抑制
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    profile_centroid_raw     = np.nanmean(centroid_masked,          axis=1)
    profile_centroid_smooth  = np.nanmean(centroid_smooth,          axis=1)
    profile_sr_cen_raw       = np.nanmean(shiftrate_centroid_raw,    axis=1)
    profile_sr_cen_smooth    = np.nanmean(shiftrate_centroid_smooth, axis=1)

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
def plot_freq_map(data, title, fname, profile_1d, analytical_profile=None):
    fig, axes = plt.subplots(
        nrows=1, ncols=2,
        width_ratios=[3, 1],
        height_ratios=[1],
        figsize=(14, 6),
    )
    ax = axes[0]
    im = ax.imshow(data, extent=extent, aspect='auto',
                   cmap='jet', vmin=vmin_f, vmax=vmax_f)
    ax.set_xlabel('Distance [m]', size=14)
    ax.set_ylabel('Delay time [ns]', size=14)
    ax.set_title(title)
    ax.tick_params(labelsize=12)
    ax.grid()
    divider = axgrid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Frequency [GHz]', size=14)
    cbar.ax.tick_params(labelsize=12)

    ax2 = axes[1]
    ax2.plot(profile_1d, t_axis, color='k', linestyle='-', label='Simulation')
    if analytical_profile is not None:
        ax2.plot(analytical_profile, t_axis, color='r', linestyle='--', label='Analytical')
        ax2.legend(loc='upper right')
    
    ax2.set_xlabel('Frequency [GHz]', size=14)
    ax2.set_ylabel('Delay time [ns]', size=14)
    ax2.set_ylim(t_axis[-1], t_axis[0])
    ax2.tick_params(labelsize=12)
    ax2.grid()

    plt.tight_layout()
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('Saved:', path)
    plt.close(fig)


def plot_shiftrate_map(data, title, fname, profile_1d, analytical_profile=None):
    fig, axes = plt.subplots(
        nrows=1, ncols=2,
        width_ratios=[3, 1],
        height_ratios=[1],
        figsize=(14, 6),
    )
    ax = axes[0]
    im = ax.imshow(data, extent=extent, aspect='auto',
                   cmap='RdBu_r', vmin=vmin_sr, vmax=vmax_sr)
    ax.set_xlabel('Distance [m]', size=14)
    ax.set_ylabel('Delay time [ns]', size=14)
    ax.set_title(title)
    ax.tick_params(labelsize=12)
    ax.grid()
    divider = axgrid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Frequency shift rate [GHz/ns]', size=14)
    cbar.ax.tick_params(labelsize=12)

    ax2 = axes[1]
    ax2.plot(profile_1d, t_axis, color='k', linestyle='-', label='Simulation')
    if analytical_profile is not None:
        ax2.plot(analytical_profile, t_axis, color='r', linestyle='--', label='Analytical')
        ax2.legend(loc='upper right')

    ax2.set_xlabel('Shift rate [GHz/ns]', size=14)
    ax2.set_ylabel('Delay time [ns]', size=14)
    ax2.set_ylim(t_axis[-1], t_axis[0])
    ax2.tick_params(labelsize=12)
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
    f'Centroid frequency – raw  (mask: {power_threshold_db} dB)',
    'centroid_raw.png',
    profile_centroid_raw,
    analytical_f_peak_profile)

plot_freq_map(
    centroid_smooth,
    'Centroid frequency – Gaussian smoothed',
    'centroid_smooth.png',
    profile_centroid_smooth,
    analytical_f_peak_profile)

plot_shiftrate_map(
    shiftrate_centroid_raw,
    'Centroid frequency shift rate – raw  [GHz/ns]',
    'centroid_shiftrate_raw.png',
    profile_sr_cen_raw,
    analytical_shiftrate_profile)

plot_shiftrate_map(
    shiftrate_centroid_smooth,
    'Centroid frequency shift rate – Gaussian smoothed  [GHz/ns]',
    'centroid_shiftrate_smooth.png',
    profile_sr_cen_smooth,
    analytical_shiftrate_profile)


print(f'\nAll 8 figures saved to: {output_dir}')
plt.show()