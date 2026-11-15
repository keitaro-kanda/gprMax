import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import json
import h5py
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.axes_grid1 as axgrid1

from gprMax.exceptions import CmdInputError
from tools.core.outputfiles_merge import get_output_data

# 図の共通ルール
#   文字の大きさ：タイトル・軸ラベル 18、目盛り・凡例・図中の文字 14
#   グリッド線：特に意図がない限り全パネルに付ける（破線 '--'、alpha 0.3）
#   補助目盛り（minor ticks）：全パネルで表示する
plt.rcParams.update({'axes.titlesize': 18, 'axes.labelsize': 18, 'figure.titlesize': 18,
                     'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14,
                     'font.size': 14,
                     'axes.grid': True, 'axes.grid.which': 'major', 'axes.axisbelow': 'line',
                     'grid.linestyle': '--', 'grid.alpha': 0.3,
                     'xtick.minor.visible': True, 'ytick.minor.visible': True})

# Input
json_file_path = input('Input Bscan.json file path: ').strip()
if not os.path.exists(json_file_path):
    raise CmdInputError('JSON file {} does not exist'.format(json_file_path))

with open(json_file_path) as f:
    params = json.load(f)
file_name = params['data']
rx_step = params['antenna_settings']['src_step']
rx_start = params['antenna_settings']['rx_start']
print(f'src_step [m]: {rx_step}')
print(f'rx_start [m]: {rx_start}')

closeup = input("Do you want to closeup the plots? (y/n) [default: n]: ").strip()
if closeup == '' or closeup == 'n':
    closeup_option = False
elif closeup == 'y':
    closeup_option = True
    time_start = float(input("Enter closeup start time (ns): ").strip())
    time_end = float(input("Enter closeup end time (ns): ").strip())
else:
    print("Please enter 'y' or 'n'.")
    sys.exit(1)

# 目印の時刻（氷層の上端・下端など）。Enter で省略
marks_in = input('Marker times [ns], comma-separated (e.g. 18.7,31.2) [default: none]: ').strip()
mark_times = [float(v) for v in marks_in.split(',') if v.strip()] if marks_in else []

output_basename = 'background'
output_dir = os.path.join(os.path.dirname(file_name), output_basename)
os.makedirs(output_dir, exist_ok=True)


# .outファイルの読み込み
output_data = h5py.File(file_name, 'r')
nrx = output_data.attrs['nrx']
output_data.close()

for rx in range(1, nrx + 1):
    outputdata, dt = get_output_data(file_name, rx, 'Ez')
print(f"Loaded data from {file_name} with shape {outputdata.shape} and dt={dt:.2e} s")

time_ns = np.arange(outputdata.shape[0]) * dt * 1e9


# 全A-scanの平均を取る（コヒーレント平均。符号を残すので、反射の極性が見える）
outputdata_ave = np.mean(outputdata[:, 1:], axis=1) # 1列目はpml内なので除外
print(f"Calculated average background trace with shape {outputdata_ave.shape}")


# --- 版 1：パワー（dB）----------------------------------------------------------
# 平均トレースのパワー (mean Ez)^2 を、その最大値で割って dB にする（0 dB = 最大）。
# 10 log10[(mean Ez)^2 / max] = 20 log10[|mean Ez| / max|mean Ez|]。
# 包絡ではないので、波形が 0 を横切るたびに大きく落ち込む（1e-12 は log(0) よけ）。
outputdata_ave_db = 10 * np.log10(outputdata_ave ** 2 / np.amax(outputdata_ave ** 2) + 1e-24)

# 移動平均の計算
window_size = len(outputdata_ave_db)//20
outputdata_ave_db_moving = np.convolve(outputdata_ave_db, np.ones(window_size)/window_size, 'same') # 入力サイズと同じサイズの配列を返す

# --- 版 2：符号つきの平均 Ez（線形、V/m）--------------------------------------------
# 正負がそのまま残るので、反射の極性（地表や氷層の上端・下端で正負が反転するか）を読める。
# 横軸の範囲は B-scan のカラーバーと同じ ±(最大振幅 / BSCAN_CLIP_DIVISOR)。
# 直達波・地表の反射は範囲の外にはみ出し（枠で切れる）、地下の反射が見える。

# --- 軸ラベル（パネル 1 とパネル 2 で強度の単位が違うことを明示する）-------------
# パネル 1：生の電場 Ez（線形・符号つき、gprMax の出力単位 V/m）。
#           色の範囲は ±(最大振幅 / BSCAN_CLIP_DIVISOR) で切っており、これを超える値は端の色で飽和する。
# パネル 2：全トレースの平均（コヒーレント平均）のパワー [dB]、または符号つきの平均 Ez。
BSCAN_CLIP_DIVISOR = 1e3
BSCAN_CLIP_DB = 20 * np.log10(BSCAN_CLIP_DIVISOR)
BSCAN_CBAR_LABEL = ('$E_z$ [V/m] (clipped at ±max/{:g})'.format(BSCAN_CLIP_DIVISOR))
VCLIP = np.amax(np.abs(outputdata[:, 1:])) / BSCAN_CLIP_DIVISOR   # カラーバーと符号つき版の横軸の範囲
print('[INFO] カラーバーと符号つき版の横軸の範囲: ±{:.3e} V/m'.format(VCLIP))
SIGNED_EXP = int(np.floor(np.log10(VCLIP)))          # 横軸の桁（ラベルに書き、目盛りの右下の 1e-2 表示は出さない）
SIGNED_SCALE = 10.0 ** SIGNED_EXP
SIGNED_XLABEL = r'Mean $E_z$ [$10^{%d}$ V/m]' % SIGNED_EXP
POWER_XLABEL = 'Power of mean $E_z$ [dB]'

SUFFIX = '' if not closeup_option else f'_{time_start}_{time_end}'


def set_time_axis(ax):
    if closeup_option:
        ax.set_ylim(time_end, time_start)
    else:
        ax.set_ylim(time_ns[-1], 0)
    for t in mark_times:
        ax.axhline(t, color='0.4', ls=':', lw=1.2)


def draw_profile(ax, kind):
    """kind = 'power'（dB）または 'signed'（符号つきの平均 Ez）"""
    if kind == 'power':
        ax.plot(outputdata_ave_db, time_ns, color='k', linestyle='-')
        ax.plot(outputdata_ave_db_moving, time_ns, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel(POWER_XLABEL)
    else:
        ax.axvline(0.0, color='0.5', lw=0.8)
        y = outputdata_ave / SIGNED_SCALE
        ax.plot(y, time_ns, color='k', linestyle='-')
        ax.fill_betweenx(time_ns, 0, y, where=y > 0, color='tab:red', alpha=0.5, lw=0)
        ax.fill_betweenx(time_ns, 0, y, where=y < 0, color='tab:blue', alpha=0.5, lw=0)
        ax.set_xlim(-VCLIP / SIGNED_SCALE, VCLIP / SIGNED_SCALE)   # B-scan のカラーバーと同じ範囲
        ax.set_xlabel(SIGNED_XLABEL)
    ax.set_ylabel('Time [ns]')
    set_time_axis(ax)


PROFILE_NAMES = {'power': 'background_trace', 'signed': 'background_trace_signed'}
COMPARISON_NAMES = {'power': 'background_comparison', 'signed': 'background_comparison_signed'}
PROFILE_TITLES = {'power': 'Mean A-scan (power)', 'signed': 'Mean A-scan (signed)'}


def save(fig, name):
    fig.savefig(os.path.join(output_dir, name + SUFFIX + '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, name + SUFFIX + '.pdf'), format='pdf', dpi=300,
                bbox_inches='tight')


# plot background only（パワー版と符号つき版）
for kind in ('power', 'signed'):
    fig, ax = plt.subplots(figsize=(4.5, 8), facecolor='w', edgecolor='w')
    draw_profile(ax, kind)
    fig.tight_layout()
    save(fig, PROFILE_NAMES[kind])
plt.show()


# plot B-scan with background（パワー版と符号つき版）
for kind in ('power', 'signed'):
    # B-scanプロットの幅とbackgroundプロットの幅を3:1にする
    fig, ax = plt.subplots(nrows=1, ncols=2, width_ratios=[3, 1], height_ratios=[1],
                           figsize=(13, 8))
    im = ax[0].imshow(outputdata,
                      extent=[rx_start, rx_start + outputdata.shape[1] * rx_step,
                              outputdata.shape[0] * dt * 1e9, 0],
                      interpolation='nearest', aspect='auto', cmap='seismic',
                      vmin=-VCLIP, vmax=VCLIP)
    ax[0].set_title('(a) B-scan: raw $E_z$')
    ax[0].set_xlabel('Horizontal position [m]')
    ax[0].set_ylabel('Time [ns]')
    set_time_axis(ax[0])
    # colorbar
    divider = axgrid1.make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(BSCAN_CBAR_LABEL)
    cbar.formatter.set_powerlimits((-2, 2))   # 小さい値は ×10^n 表記にする
    cbar.update_ticks()

    draw_profile(ax[1], kind)
    ax[1].set_title('(b) ' + PROFILE_TITLES[kind])
    fig.tight_layout()
    save(fig, COMPARISON_NAMES[kind])
plt.show()