import os
import json
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
import mpl_toolkits.axes_grid1 as axgrid1
from gprMax.exceptions import CmdInputError
from tools.core.outputfiles_merge import get_output_data

# =============================================================================
# User input
# =============================================================================
json_file_path = input('Input Bscan.json file path: ').strip()
if not os.path.exists(json_file_path):
    raise CmdInputError('JSON file {} does not exist'.format(json_file_path))

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
# STFT parameters
# =============================================================================
# df = fs / nperseg.  With fs ~ 85 GHz and fc = 1.25 GHz:
#   nperseg=512 -> df ~ 0.17 GHz, ~17 bins in 0.3-3.0 GHz  (recommended)
#   nperseg=256 -> df ~ 0.33 GHz,  ~9 bins in 0.3-3.0 GHz  (faster)
nperseg  = 512
noverlap = nperseg * 3 // 4
window   = 'hann'

# Physically meaningful band for 1.25 GHz gaussiandot waveform
freq_min = 0.25    # [GHz]  lower cut (exclude DC noise)
freq_max = 6.0    # [GHz]  upper cut (beyond ~2.5 * fc is noise)

# Power mask: pixels more than this many dB below the per-trace peak -> NaN
power_threshold_db = -125.0   # [dB]

# Gaussian smoothing kernel (time-axis bins, trace-axis bins)
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
# Compute centroid map and peak-frequency map from STFT
# =============================================================================
eps = 1e-30

centroid_map = np.zeros((n_time, n_traces))   # power-weighted mean frequency
peak_freq_map = np.zeros((n_time, n_traces))  # frequency of maximum amplitude
power_map    = np.zeros((n_time, n_traces))   # total in-band power (for mask)

for itrace in range(n_traces):
    _, _, Zxx = signal.stft(outputdata[:, itrace], fs=fs, window=window,
                            nperseg=nperseg, noverlap=noverlap)
    power = np.abs(Zxx[freq_mask, :]) ** 2          # (n_freq_valid, n_time)
    total = power.sum(axis=0)                        # (n_time,)

    # Spectral centroid (power-weighted mean frequency)
    centroid_map[:, itrace] = (valid_freq[:, None] * power).sum(axis=0) / (total + eps)

    # Peak frequency (argmax along frequency axis)
    peak_idx = np.argmax(power, axis=0)              # (n_time,)
    peak_freq_map[:, itrace] = valid_freq[peak_idx]

    power_map[:, itrace] = total

# =============================================================================
# Power mask: low-SNR pixels -> NaN
# =============================================================================
trace_peak = power_map.max(axis=0, keepdims=True)
with np.errstate(divide='ignore', invalid='ignore'):
    power_rel_db = 10.0 * np.log10(
        np.where(trace_peak > 0, power_map / (trace_peak + eps), eps))

valid_mask = power_rel_db >= power_threshold_db   # True = valid

centroid_masked  = np.where(valid_mask, centroid_map,  np.nan)
peak_freq_masked = np.where(valid_mask, peak_freq_map, np.nan)

# =============================================================================
# NaN-aware Gaussian smoothing helper
# =============================================================================
def smooth_masked(data, mask, sigma):
    """Gaussian smooth of masked data; masked pixels are excluded from average."""
    filled = np.where(mask, data, 0.0)
    sm_data   = gaussian_filter(filled,              sigma=sigma)
    sm_weight = gaussian_filter(mask.astype(float),  sigma=sigma)
    out = np.full_like(sm_data, np.nan)
    np.divide(sm_data, sm_weight, out=out, where=(sm_weight > 1e-6))
    return out

centroid_smooth  = smooth_masked(centroid_map,  valid_mask, sigma)
peak_freq_smooth = smooth_masked(peak_freq_map, valid_mask, sigma)

# =============================================================================
# Frequency-shift-rate maps  [GHz/ns]
# gradient along the time axis (axis=0); t_axis is in [ns]
# =============================================================================
dt_stft = t_axis[1] - t_axis[0]   # [ns] STFT time step

def shift_rate(freq_map):
    """d(frequency)/d(time) along axis-0, in GHz/ns. NaN-safe."""
    # np.gradient handles NaN by propagating; acceptable for visualisation.
    return np.gradient(freq_map, dt_stft, axis=0)

shiftrate_centroid_raw    = shift_rate(centroid_masked)
shiftrate_centroid_smooth = shift_rate(centroid_smooth)
shiftrate_peak_raw        = shift_rate(peak_freq_masked)
shiftrate_peak_smooth     = shift_rate(peak_freq_smooth)

# =============================================================================
# Output directory
# =============================================================================
output_base_name = 'center_frequency_analysis'
output_dir = os.path.join(os.path.dirname(os.path.abspath(json_file_path)),
                          output_base_name)
os.makedirs(output_dir, exist_ok=True)

extent = [0, n_traces * GPR_step, t_axis[-1], t_axis[0]]

# Colour scale for frequency maps: percentile clip over all valid pixels
all_freq_valid = np.concatenate([
    centroid_masked[np.isfinite(centroid_masked)],
    peak_freq_masked[np.isfinite(peak_freq_masked)],
])
if all_freq_valid.size > 0:
    vmin_f, vmax_f = np.percentile(all_freq_valid, [5, 95])
else:
    vmin_f, vmax_f = freq_min, freq_max
print(f'Frequency colour scale: {vmin_f:.3f} – {vmax_f:.3f} GHz')

# Colour scale for shift-rate maps: symmetric around 0, percentile clip
all_sr_valid = np.concatenate([
    shiftrate_centroid_raw[np.isfinite(shiftrate_centroid_raw)],
    shiftrate_peak_raw[np.isfinite(shiftrate_peak_raw)],
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
def plot_freq_map(data, title, fname):
    """Plot a frequency map [GHz]."""
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, extent=extent, aspect='auto',
                   cmap='jet', vmin=vmin_f, vmax=vmax_f)
    ax.set_xlabel('Distance [m]')
    ax.set_ylabel('Delay time [ns]')
    ax.set_title(title)
    divider = axgrid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Frequency [GHz]')
    plt.tight_layout()
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('Saved:', path)
    plt.close(fig)


def plot_shiftrate_map(data, title, fname):
    """Plot a frequency shift-rate map [GHz/ns]."""
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, extent=extent, aspect='auto',
                   cmap='RdBu_r', vmin=vmin_sr, vmax=vmax_sr)
    ax.set_xlabel('Distance [m]')
    ax.set_ylabel('Delay time [ns]')
    ax.set_title(title)
    divider = axgrid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Frequency shift rate [GHz/ns]')
    plt.tight_layout()
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('Saved:', path)
    plt.close(fig)

# =============================================================================
# Produce 8 maps
# =============================================================================
# --- Group 1: Spectral centroid (power-weighted mean frequency) ---
plot_freq_map(
    centroid_masked,
    f'Centroid frequency – raw  (mask: {power_threshold_db} dB)',
    'centroid_raw.png')

plot_freq_map(
    centroid_smooth,
    'Centroid frequency – Gaussian smoothed',
    'centroid_smooth.png')

plot_shiftrate_map(
    shiftrate_centroid_raw,
    'Centroid frequency shift rate – raw  [GHz/ns]',
    'centroid_shiftrate_raw.png')

plot_shiftrate_map(
    shiftrate_centroid_smooth,
    'Centroid frequency shift rate – Gaussian smoothed  [GHz/ns]',
    'centroid_shiftrate_smooth.png')

# --- Group 2: Peak frequency (argmax amplitude) ---
plot_freq_map(
    peak_freq_masked,
    f'Peak frequency – raw  (mask: {power_threshold_db} dB)',
    'peak_freq_raw.png')

plot_freq_map(
    peak_freq_smooth,
    'Peak frequency – Gaussian smoothed',
    'peak_freq_smooth.png')

plot_shiftrate_map(
    shiftrate_peak_raw,
    'Peak frequency shift rate – raw  [GHz/ns]',
    'peak_freq_shiftrate_raw.png')

plot_shiftrate_map(
    shiftrate_peak_smooth,
    'Peak frequency shift rate – Gaussian smoothed  [GHz/ns]',
    'peak_freq_shiftrate_smooth.png')

print(f'\nAll 8 figures saved to: {output_dir}')
plt.show()