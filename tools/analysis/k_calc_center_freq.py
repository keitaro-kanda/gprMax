import os
import json
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from gprMax.exceptions import CmdInputError
from tools.core.outputfiles_merge import get_output_data
from scipy import signal
import mpl_toolkits.axes_grid1 as axgrid1



# =============================================================================
# ユーザーインプット：B-scan jsonファイルのパスを入力する
# =============================================================================
json_file_path = input('Input Bscan.json file path: ').strip()
if not os.path.exists(json_file_path):
    raise CmdInputError('JSON file {} does not exist'.format(json_file_path))


# =============================================================================
# データ読み込み
# =============================================================================
with open(json_file_path) as f:
    params = json.load(f)
outfile_path = params['data']

# トレース間隔の取得
GPR_step = params['antenna_settings']['src_step']
print('GPR step [m]: ', GPR_step)

# Open output file and read number of outputs (receivers)
f = h5py.File(outfile_path, 'r')
if 'nrx' in f.attrs:
    nrx = f.attrs['nrx']
elif 'rxs' in f:
    nrx = len(f['rxs'].keys())
else:
    f.close()
    raise CmdInputError('Invalid output file: {} - nrx attribute not found and no rxs group exists. The file may be empty or not a valid gprMax output file.'.format(outfile_path))
print('nrx: ', nrx)
f.close()

outputdata, dt = get_output_data(outfile_path, 1,'Ez') # いずれfull-pol用に変更の必要あり
dt = dt * 1e9 # Convert to [ns]


# =============================================================================
# STFT → 各 (トレース, 遅れ時間) のスペクトル重心 と 全帯域パワー
# =============================================================================
n_samples, n_traces = outputdata.shape
print('B-scan shape (samples, traces):', outputdata.shape)

fs = 1.0 / dt  # [GHz]（dt は ns なので 1/ns = GHz）

nperseg  = 64
noverlap = nperseg * 3 // 4
window   = 'hann'
freq_min = 0.5  # [GHz] DC近傍を除外

# 周波数軸・時間軸を先頭トレースで取得
f_axis, t_axis, _ = signal.stft(outputdata[:, 0], fs=fs, window=window,
                                nperseg=nperseg, noverlap=noverlap)
freq_mask  = f_axis > freq_min
valid_freq = f_axis[freq_mask]
n_time = t_axis.size

centroid_map = np.zeros((n_time, n_traces))  # スペクトル重心 [GHz]
power_map    = np.zeros((n_time, n_traces))  # 全帯域パワー（マスク用）

eps = 1e-12
for itrace in range(n_traces):
    _, _, Zxx = signal.stft(outputdata[:, itrace], fs=fs, window=window,
                            nperseg=nperseg, noverlap=noverlap)
    power = np.abs(Zxx) ** 2
    power = power[freq_mask, :]
    total = power.sum(axis=0)
    centroid_map[:, itrace] = (valid_freq[:, None] * power).sum(axis=0) / (total + eps)
    power_map[:, itrace]    = total

print('centroid_map shape:', centroid_map.shape, ' t_axis range [ns]:', t_axis[0], t_axis[-1])

from scipy.ndimage import gaussian_filter   # ファイル冒頭の import 群に移してもOK

# =============================================================================
# 共通の描画ヘルパ（横軸: 距離, 縦軸: 遅れ時間, カラー: スペクトル重心）
# =============================================================================
extent = [0, n_traces * GPR_step, t_axis[-1], t_axis[0]]
out_dir = os.path.dirname(os.path.abspath(json_file_path))

def plot_centroid(data, title, fname, vmin=None, vmax=None):
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
    save_path = os.path.join(out_dir, fname)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print('Saved figure to:', save_path)
    return im

# カラースケールは外れ値でつぶれないよう分位点でクリップ（全プロット共通）
vmin, vmax = np.percentile(centroid_map, [5, 95])

# =============================================================================
# (1) 生の centroid マップ
# =============================================================================
plot_centroid(centroid_map, 'Spectral-centroid map (raw)',
              'spectral_centroid_map.png', vmin, vmax)

# =============================================================================
# (2-a) 単純ガウシアン平滑化
#       sigma = (遅れ時間方向, トレース方向) のサンプル数。大きいほど滑らか
# =============================================================================
sigma = (3, 3)
centroid_smooth = gaussian_filter(centroid_map, sigma=sigma)
plot_centroid(centroid_smooth,
              'Spectral-centroid map (Gaussian smoothed)',
              'spectral_centroid_map_smooth.png', vmin, vmax)

# =============================================================================
# (2-b) パワー重み付き平滑化（推奨）
#       強い反射ほど重く扱う → 低パワー領域の乱れに引きずられない
# =============================================================================
num = gaussian_filter(centroid_map * power_map, sigma=sigma)
den = gaussian_filter(power_map, sigma=sigma)
centroid_wsmooth = num / (den + eps)
plot_centroid(centroid_wsmooth,
              'Spectral-centroid map (power-weighted smoothed)',
              'spectral_centroid_map_wsmooth.png', vmin, vmax)

plt.show()