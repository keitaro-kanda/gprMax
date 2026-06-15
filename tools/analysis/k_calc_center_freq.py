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
# nperseg controls frequency resolution: df = fs / nperseg
# With fs ~ 85 GHz and GPR centre at 1.25 GHz, nperseg must be large
# enough to resolve the GPR band (0.3 – 3.0 GHz).
#   nperseg=512: df ~ 0.17 GHz -> ~17 bins in band  (recommended)
#   nperseg=256: df ~ 0.33 GHz ->  ~9 bins in band  (faster, less accurate)
# Trade-off: larger nperseg = finer freq resolution, coarser time resolution.
nperseg  = 512
noverlap = nperseg * 3 // 4
window   = 'hann'

# Physically meaningful frequency band for 1.25 GHz gaussiandot waveform
freq_min = 0.25    # [GHz]  exclude DC / near-DC noise
freq_max = 6.0    # [GHz]  exclude frequencies beyond ~2.5 * fc

# Power mask: pixels whose in-band power is more than this many dB below
# the per-trace peak are set to NaN (no valid signal).
power_threshold_db = -120.0   # [dB]  raise to mask more; lower to show deeper

# Smoothing kernel (time bins, trace bins)
sigma = (3, 3)

# =============================================================================
# Compute spectral-centroid map
# =============================================================================
f_axis, t_axis, _ = signal.stft(outputdata[:, 0], fs=fs, window=window,
                                 nperseg=nperseg, noverlap=noverlap)

freq_mask  = (f_axis >= freq_min) & (f_axis <= freq_max)
valid_freq = f_axis[freq_mask]
n_time     = t_axis.size

print(f'STFT: nperseg={nperseg}, df={fs/nperseg:.3f} GHz')
print(f'Frequency band: {valid_freq[0]:.3f} – {valid_freq[-1]:.3f} GHz '
      f'({freq_mask.sum()} bins)')

eps = 1e-30
centroid_map = np.zeros((n_time, n_traces))
power_map    = np.zeros((n_time, n_traces))

for itrace in range(n_traces):
    _, _, Zxx = signal.stft(outputdata[:, itrace], fs=fs, window=window,
                            nperseg=nperseg, noverlap=noverlap)
    power = np.abs(Zxx[freq_mask, :]) ** 2
    total = power.sum(axis=0)
    centroid_map[:, itrace] = (valid_freq[:, None] * power).sum(axis=0) / (total + eps)
    power_map[:, itrace]    = total

# =============================================================================
# Power mask: mark low-SNR pixels as NaN
# =============================================================================
trace_peak = power_map.max(axis=0, keepdims=True)   # (1, n_traces)
with np.errstate(divide='ignore', invalid='ignore'):
    power_rel_db = 10.0 * np.log10(
        np.where(trace_peak > 0, power_map / (trace_peak + eps), eps))

valid_mask = power_rel_db >= power_threshold_db     # True = valid signal

centroid_masked = np.where(valid_mask, centroid_map, np.nan)

# =============================================================================
# Smoothing (NaN-aware: fill with 0 before filter, divide by weight after)
# =============================================================================
data_filled   = np.where(valid_mask, centroid_map, 0.0)
weight_filled = valid_mask.astype(float)

# (a) Simple Gaussian smooth
sm_data   = gaussian_filter(data_filled,   sigma=sigma)
sm_weight = gaussian_filter(weight_filled, sigma=sigma)
# Safe division: use np.divide with 'out' and 'where' to avoid any FPE
centroid_smooth = np.full_like(sm_data, np.nan)
np.divide(sm_data, sm_weight, out=centroid_smooth, where=(sm_weight > 1e-6))

# (b) Power-weighted smooth
pw_data   = gaussian_filter(np.where(valid_mask, centroid_map * power_map, 0.0), sigma=sigma)
pw_weight = gaussian_filter(np.where(valid_mask, power_map,                0.0), sigma=sigma)
centroid_wsmooth = np.full_like(pw_data, np.nan)
np.divide(pw_data, pw_weight, out=centroid_wsmooth, where=(pw_weight > 0.0))

# =============================================================================
# Output directory
# =============================================================================
output_base_name = 'center_frequency_analysis'
output_dir = os.path.join(os.path.dirname(os.path.abspath(json_file_path)),
                          output_base_name)
os.makedirs(output_dir, exist_ok=True)

extent = [0, n_traces * GPR_step, t_axis[-1], t_axis[0]]

valid_vals = centroid_masked[np.isfinite(centroid_masked)]
if valid_vals.size > 0:
    vmin, vmax = np.percentile(valid_vals, [5, 95])
else:
    vmin, vmax = freq_min, freq_max
print(f'Colour scale: {vmin:.3f} – {vmax:.3f} GHz')

# =============================================================================
# Plot helper
# =============================================================================
def plot_centroid(data, title, fname):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, extent=extent, aspect='auto',
                   cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_xlabel('Distance [m]')
    ax.set_ylabel('Delay time [ns]')
    ax.set_title(title)
    divider = axgrid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Spectral centroid [GHz]')
    plt.tight_layout()
    save_path = os.path.join(output_dir, fname)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print('Saved:', save_path)

# =============================================================================
# Produce three maps
# =============================================================================
plot_centroid(centroid_masked,
              f'Spectral-centroid map  (raw, power mask {power_threshold_db} dB)',
              'spectral_centroid_map.png')

plot_centroid(centroid_smooth,
              'Spectral-centroid map  (Gaussian smoothed)',
              'spectral_centroid_map_smooth.png')

plot_centroid(centroid_wsmooth,
              'Spectral-centroid map  (power-weighted smoothed)',
              'spectral_centroid_map_wsmooth.png')

plt.show()