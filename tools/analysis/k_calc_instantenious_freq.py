import os
import sys
import warnings
import csv

# 既存コードの環境パス追加の流儀を踏襲
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import h5py
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, sosfiltfilt
from scipy import constants as const
import mpl_toolkits.axes_grid1 as axgrid1
from gprMax.exceptions import CmdInputError
from tools.core.outputfiles_merge import get_output_data

"""
k_calc_hilbert_if.py

[目的]
Hilbert変換を用いた包絡線二乗重み付き瞬時周波数 (IF_w) およびそのシフトレートの
2次元マップ・1次元プロファイルを作成する。STFTを用いず、解析信号に基づく
高分解能な周波数解析を実現する。

[T_avg (平均化窓長) の意味]
瞬時周波数の理論上の分散 (δIF) は 1/√(T_avg) に比例するため、T_avgを長くすると
ノイズが平滑化される。一方で、深さ方向の空間分解能 Δz は T_avg・v/2 に比例するため、
平滑化と追随性（空間分解能）のトレードオフが生じる。本スクリプトではこの特性を
確認するため、複数の T_avg について独立に解析を行う。

[出力一覧]
指定された output_dir 直下および T_avg ごとのサブディレクトリに以下を出力する:
- Tavg_{X}ns/if_w_map_Tavg{X}ns.png : IF_w の2次元マップおよび1次元プロファイル
- Tavg_{X}ns/if_w_shiftrate_map_Tavg{X}ns.png : シフトレートのマップおよびプロファイル
- Tavg_{X}ns/if_w_profile_Tavg{X}ns.csv : 中央値・IQRプロファイルと理論曲線の数値データ
- if_w_profile_comparison.png : 全 T_avg の IF_w 中央値プロファイルを比較する図
- if_w_shiftrate_comparison.png : 全 T_avg の シフトレート中央値プロファイルを比較する図
"""

# =============================================================================
# Hilbert IF parameters
# =============================================================================
T_AVG_LIST_NS = [1.0, 3.0, 10.0]   # 平均化窓長 [ns]（それぞれ別個に解析・出力）
HOP_RATIO     = 0.5                # ホップ = 窓長 × この係数
MEAN_TRACE_REMOVAL = False         # True: 平均トレース除去（コヒーレント背景の分離検証用）
USE_BRICKWALL = True   # False にすると従来の butter+hilbert (比較実験用)

freq_min = 0.5    # [GHz]
freq_max = 2.0     # [GHz]
power_threshold_db = -125.0   # [dB]
eps = 1e-30

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
fs    = 1.0 / dt_ns     # [GHz]
n_samples, n_traces = outputdata.shape

print(f'dt = {dt*1e12:.4f} ps,  fs = {fs:.2f} GHz,  fs/2 = {fs/2:.2f} GHz')
print(f'B-scan shape (samples, traces): {outputdata.shape}')

# =============================================================================
# Extract Debye Parameters from .in file (バグ修正済)
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
                m_disp = re.search(r'#add_dispersion_debye:\s*\d+\s+([0-9\.eE\+\-]+)\s+[0-9\.eE\+\-]+\s+([0-9\.eE\+\-]+)', content)
                
                if m_tau1: debye_params['tau1'] = float(m_tau1.group(1))
                if m_tau2: debye_params['tau2'] = float(m_tau2.group(1))
                
                if m_ratio:
                    # コメント部分などを除外して計算
                    expr = m_ratio.group(1).split('#')[0].strip()
                    debye_params['de_ratio'] = eval(expr)
                
                # m_ratioのfloatキャストによる上書きバグを修正
                if (not m_ratio) and m_disp:
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
# Dielectric Model Definitions (Method A) - 変更禁止
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
# k_calc_instantenious_freq.py 修正パッチ
# 目的: STFT重心とパイプラインを等価化し、IF_w の系統ズレの原因を切り分ける
#
# [変更点]
#  (1) Butterworth+hilbert を「ブリックウォール解析信号」に置換
#      → STFTと同一のハードマスク [freq_min, freq_max] で帯域を定義。
#        これにより IF_w ≡ 帯域制限スペクトル重心 の恒等式が構成的に成立し、
#        フィルタスカートを通る帯域外残留 (wake等) の混入経路を遮断する。
#  (2) 窓内平均をパルスペア推定 angle(Σ z[n+1] z*[n]) に置換
#      → 現行の Σ A²·IF / Σ A² と同じ A² 重みだが、「角度の平均」ではなく
#        「複素平均の角度」なので包絡線ヌル点の ±π 位相スリップに頑健。
#  (3) パワーマスクを「トレースごとのノイズ床基準」に変更
#      → 現行の -125 dB (ピーク基準) は実質マスク無しで、信号消失後の
#        IF≈0 窓が中央値を 0 に引きずる原因。
#  (4) 帯域外エネルギーの診断出力を追加 (原因の定量確認用)
#
# [使い方] Step 1 のトレースループ (sos = butter(...) 〜 IF_full 代入まで) を
#          以下の compute_analytic_if() 呼び出しに置き換え、
#          T_avg ループ内の IF_w 計算とマスクを (2)(3) に差し替える。
# =============================================================================
 
def compute_analytic_if(data_proc, dt_ns, freq_min, freq_max, use_brickwall=True):
    """全トレースの解析信号 z とサンプルごとの IF を返す。
 
    use_brickwall=True: 周波数領域で [freq_min, freq_max] 外を完全にゼロ化した
    片側スペクトルから解析信号を構成 (STFTのハードマスクと等価な帯域定義)。
    """
    n_samples, n_traces = data_proc.shape
    fs = 1.0 / dt_ns
    Z = np.zeros((n_samples, n_traces), dtype=complex)
 
    if use_brickwall:
        fr = np.fft.rfftfreq(n_samples, dt_ns)
        band = (fr >= freq_min) & (fr <= freq_max)
        for i in range(n_traces):
            X = np.fft.rfft(data_proc[:, i])
            Xfull = np.zeros(n_samples, dtype=complex)
            Xfull[:len(X)] = np.where(band, 2.0 * X, 0.0)
            Z[:, i] = np.fft.ifft(Xfull)
    else:
        sos = signal.butter(4, [freq_min/(fs/2), freq_max/(fs/2)],
                            btype='band', output='sos')
        for i in range(n_traces):
            tr = signal.sosfiltfilt(sos, data_proc[:, i])
            Z[:, i] = signal.hilbert(tr)
 
    A2 = np.abs(Z) ** 2
    dph = np.angle(Z[1:, :] * np.conj(Z[:-1, :]))       # -π〜π, unwrap不要
    IF = np.vstack([np.zeros((1, Z.shape[1])), dph / (2*np.pi*dt_ns)])
    return Z, A2, IF
 
 
def ifw_pulse_pair(Z, starts, L, dt_ns):
    """窓ごとのパルスペア推定 IF_w = angle(Σ z[n+1] z*[n]) / (2π dt)。
    Σ A²·IF / Σ A² と同じ A² 重みの一次モーメントだが、複素領域で平均して
    から角度を取るため、ヌル点の位相スリップ外れ値の影響を受けない。"""
    n_windows = len(starts)
    n_traces = Z.shape[1]
    IF_w = np.zeros((n_windows, n_traces))
    P_win = np.zeros((n_windows, n_traces))
    for k, st in enumerate(starts):
        R1 = np.sum(Z[st+1:st+L, :] * np.conj(Z[st:st+L-1, :]), axis=0)
        IF_w[k, :] = np.angle(R1) / (2*np.pi*dt_ns)
        P_win[k, :] = np.sum(np.abs(Z[st:st+L, :])**2, axis=0)
    return IF_w, P_win
 
 
def noise_floor_mask(P_win, L, A2, noise_gate_ns, dt_ns, snr_db=10.0):
    """記録末尾 noise_gate_ns 区間をノイズ床とみなし、窓パワーが
    ノイズ床 + snr_db を超える窓のみ有効とするマスク。"""
    n_gate = int(round(noise_gate_ns / dt_ns))
    noise_A2 = np.mean(A2[-n_gate:, :], axis=0)          # トレースごと [1/sample]
    P_noise = noise_A2[None, :] * L                       # 窓長ぶんのノイズパワー
    with np.errstate(divide='ignore'):
        snr = 10.0 * np.log10(P_win / (P_noise + eps))
    return snr >= snr_db
 
 
def band_leak_diagnostics(A2, IF, freq_min, freq_max, t, label=''):
    """帯域外 IF サンプルが A² 重みでどれだけ寄与しているかを時間帯別に出力。
    ブリックウォールでもここが大きければ真のマルチコンポーネント効果、
    butter 版でのみ大きければフィルタスカート残留 (wake等) が原因。"""
    print(f'--- band leak diagnostics {label} ---')
    for t0, t1 in [(5, 15), (15, 25), (25, 35), (35, 45)]:
        sel = (t >= t0) & (t < t1)
        a2 = A2[sel, :]; f_ = IF[sel, :]
        out = (f_ < freq_min) | (f_ > freq_max)
        w_out = (a2 * out).sum() / (a2.sum() + eps)
        bias = (a2 * np.where(out, f_, 0.0)).sum() / (a2.sum() + eps)
        print(f'  {t0:2d}-{t1:2d} ns: 帯域外A²重み {w_out*100:6.2f}% , '
              f'IF_wへのバイアス寄与 {bias:+.3f} GHz')

# =============================================================================
# Analytical Frequency Shift Calculation (Depth + Debye + Time Offset for Buried Rx)
# =============================================================================
analytical_calculated = False
t_max_ns = (n_samples - 1) * dt_ns

try:
    if os.path.exists(ascan_outfile_path):
        ascan_data, dt_ascan = get_output_data(ascan_outfile_path, 1, 'Ez')
        
        if ascan_data.ndim == 1:
            e_incident = ascan_data
        else:
            e_incident = ascan_data[:, 0]
        
        N = len(e_incident)
        freq_ascan = np.fft.rfftfreq(N, d=dt_ascan)
        S0_omega = np.fft.rfft(e_incident)
        
        f_min_hz = freq_min * 1e9
        f_max_hz = freq_max * 1e9
        band_mask = (freq_ascan >= f_min_hz) & (freq_ascan <= f_max_hz)
        
        f_calc = freq_ascan[band_mask]
        S0_calc = S0_omega[band_mask]
        omega = 2 * np.pi * f_calc
        f_center = 450e6
        
        antenna_height = 0.35
        system_lag_ns  = 0.837
        rx_depth       = 0.10
        
        t_air_ns = (2.0 * antenna_height / const.c) * 1e9 
        
        d_sub_offset = np.linspace(0, rx_depth, 50)
        eps_sub_offset, _ = get_eps_static(d_sub_offset)
        v_sub = const.c / np.sqrt(eps_sub_offset)
        dt_sub = d_sub_offset[1] - d_sub_offset[0]
        t_ground_start_ns = np.sum(2.0 * dt_sub / v_sub) * 1e9
        
        t_offset_ns = system_lag_ns + t_air_ns + t_ground_start_ns
        
        print(f"Time offset (depth {rx_depth}m reflection): {t_offset_ns:.3f} ns "
              f"(Lag: {system_lag_ns} + Air: {t_air_ns:.3f} + Ground({rx_depth}m): {t_ground_start_ns:.3f} ns)")
        
        # 補間処理はT_avgループ内で行うため、ここでは配列の準備まで
        max_depth = (t_max_ns * 1e-9) * const.c / 2 
        d_array = np.linspace(rx_depth, max_depth, 400)
        d_step = d_array[1] - d_array[0]
        
        f_peak_d = []
        t_delay_d = []
        
        cumulative_attenuation = np.zeros_like(omega)
        cumulative_time = np.zeros_like(omega)
        
        for i, d in enumerate(d_array):
            eps_complex_w = get_eps_regolith(d, omega, debye_params, anchor_freq=f_center)
            alpha_d = - (omega / const.c) * np.imag(np.sqrt(eps_complex_w))
            v_d = const.c / np.real(np.sqrt(eps_complex_w))
            
            if i > 0:
                cumulative_attenuation += alpha_d * d_step
                cumulative_time += 2 * d_step / v_d
                
            S_d_w = S0_calc * np.exp(-2 * cumulative_attenuation)
            power = np.abs(S_d_w)**2
            
            f_peak = np.trapz(f_calc * power, f_calc) / np.trapz(power, f_calc)
            f_peak_d.append(f_peak)
            
            t_delay_ground = np.interp(f_peak, f_calc, cumulative_time)
            t_total_ns = t_offset_ns + (t_delay_ground * 1e9)
            t_delay_d.append(t_total_ns)
        
        f_peak_d_ghz = np.array(f_peak_d) / 1e9 # [GHz]
        t_delay_d_ns = np.array(t_delay_d)
        analytical_calculated = True
        print("Analytical frequency shift successfully calculated with buried Rx offset (regolith only).")
        
    else:
        print(f"Warning: A-scan file not found at {ascan_outfile_path}. Analytical calculation skipped.")
except Exception as e:
    print(f"Warning: Analytical calculation failed: {e}")

# =============================================================================
# Output directory
# =============================================================================
output_base_name = 'hilbert_if_analysis'
if MEAN_TRACE_REMOVAL:
    output_base_name += '_meansub'
output_dir = os.path.join(os.path.dirname(os.path.abspath(json_file_path)), output_base_name)
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# Plot helpers (t_axis_h, extent_h を受け取るように一般化)
# =============================================================================
def plot_freq_map(data, fname, prof_med, prof_p25, prof_p75, t_axis_h, extent_h, analytical_profile=None):
    antenna_height = 0.35
    system_lag_ns  = 0.837
    initial_delay = surface_delay_ns(antenna_height, system_lag_ns)

    fig, axes = plt.subplots(nrows=1, ncols=2, width_ratios=[3, 1], height_ratios=[1], figsize=(12, 8))
    ax = axes[0]
    im = ax.imshow(data, extent=extent_h, aspect='auto', cmap='jet', vmin=0.5, vmax=2.0)
    ax.axhline(initial_delay, color='gray', linestyle='--', lw=2)
    ax.set_xlabel('Distance [m]', size=18)
    ax.set_ylabel('Delay time [ns]', size=18)
    ax.tick_params(labelsize=14)
    ax.minorticks_on()
    ax.grid()
    divider = axgrid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Frequency [GHz]', size=18)
    cbar.ax.tick_params(labelsize=14)

    ax2 = axes[1]
    ax2.fill_betweenx(t_axis_h, prof_p25, prof_p75, color='gray', alpha=0.4, label='IQR (25-75%)')
    ax2.plot(prof_med, t_axis_h, color='k', linestyle='-', label='Median')
    
    if analytical_profile is not None:
        ax2.plot(analytical_profile, t_axis_h, color='r', linestyle='--', label='Analytical')
    
    ax2.axhline(initial_delay, color='gray', linestyle='--', lw=2, label='Surface')
    ax2.legend(fontsize=14, loc='lower center')
    ax2.set_xlabel('Frequency [GHz]', size=18)
    ax2.set_ylabel('Delay time [ns]', size=18)
    ax2.set_ylim(t_axis_h[-1], t_axis_h[0])
    ax2.tick_params(labelsize=14)
    ax2.minorticks_on()
    ax2.grid()

    plt.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_shiftrate_map(data, fname, prof_med, prof_p25, prof_p75, t_axis_h, extent_h, vmin_sr, vmax_sr, analytical_profile=None):
    antenna_height = 0.35
    system_lag_ns  = 0.837
    initial_delay = surface_delay_ns(antenna_height, system_lag_ns)

    fig, axes = plt.subplots(nrows=1, ncols=2, width_ratios=[3, 1], height_ratios=[1], figsize=(12, 8))
    ax = axes[0]
    im = ax.imshow(data, extent=extent_h, aspect='auto', cmap='RdBu_r', vmin=vmin_sr, vmax=vmax_sr)
    ax.axhline(initial_delay, color='gray', linestyle='--', lw=2)
    ax.set_xlabel('Distance [m]', size=18)
    ax.set_ylabel('Delay time [ns]', size=18)
    ax.tick_params(labelsize=14)
    ax.minorticks_on()
    ax.grid()
    divider = axgrid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Frequency shift rate [GHz/ns]', size=18)
    cbar.ax.tick_params(labelsize=14)

    ax2 = axes[1]
    ax2.fill_betweenx(t_axis_h, prof_p25, prof_p75, color='gray', alpha=0.4, label='IQR (25-75%)')
    ax2.plot(prof_med, t_axis_h, color='k', linestyle='-', label='Median')
    
    if analytical_profile is not None:
        ax2.plot(analytical_profile, t_axis_h, color='r', linestyle='--', label='Analytical')

    ax2.axhline(initial_delay, color='gray', linestyle='--', lw=2, label='Surface')
    ax2.legend(fontsize=14, loc='lower center')
    ax2.set_xlabel('Shift rate [GHz/ns]', size=18)
    ax2.set_ylabel('Delay time [ns]', size=18)
    ax2.set_ylim(t_axis_h[-1], t_axis_h[0])
    ax2.tick_params(labelsize=14)
    ax2.minorticks_on()
    ax2.grid()

    plt.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)

# =============================================================================
# Step 1: Trace-wise Analytic Signal & IF (T_avg independent)
# =============================================================================
data_proc = outputdata - outputdata.mean(axis=1, keepdims=True) if MEAN_TRACE_REMOVAL else outputdata

# 修正パッチ実装
Z, A2_full, IF_full = compute_analytic_if(data_proc, dt_ns, freq_min, freq_max,
                                          use_brickwall=USE_BRICKWALL)
t_full = np.arange(A2_full.shape[0]) * dt_ns
band_leak_diagnostics(A2_full, IF_full, freq_min, freq_max, t_full,
                      label='brickwall' if USE_BRICKWALL else 'butter')
# 修正パッチ実装

# lo, hi = freq_min/(fs/2), freq_max/(fs/2)   # freq, fs は GHz
# b, a = butter(4, [lo, hi], btype='band')
# for itrace in range(n_traces):
#     sos = butter(4, [freq_min/(fs/2), freq_max/(fs/2)], btype='band', output='sos')
#     tr = sosfiltfilt(sos, data_proc[:, itrace])
#     # tr = filtfilt(b, a, data_proc[:, 20])
#     print("filtered std / raw std =", tr.std() / data_proc[:,20].std())
#     # フィルタ後のスペクトルを見る
#     F = np.abs(np.fft.rfft(tr)); f = np.fft.rfftfreq(len(tr), dt_ns)
#     print("spectral centroid of filtered =", np.sum(f*F**2)/np.sum(F**2), "GHz")

#     z = signal.hilbert(tr)
#     A2_full[:, itrace] = np.abs(z)**2
#     phase_raw = np.angle(z)  # アンラップしない生位相
#     # 各サンプルのIFを、隣接位相差から直接計算(アンラップ不要)
#     dphase = np.angle(z[1:] * np.conj(z[:-1]))  # -π〜πに収まる位相増分
#     IF_full[:, itrace] = np.concatenate([[0], dphase / (2*np.pi*dt_ns)])
#     # z = signal.hilbert(tr)
#     # A2_full[:, itrace] = np.abs(z) ** 2
#     # phase = np.unwrap(np.angle(z))
#     # IF_full[:, itrace] = np.gradient(phase, dt_ns) / (2.0 * np.pi)

# IF_full = np.clip(IF_full, -fs/2, fs/2)

# =============================================================================
# Validation Preparation
# =============================================================================
print("\n=== Validation ===")
# 1) Sine wave test
t_test = np.arange(n_samples) * dt_ns
test_trace = np.sin(2.0 * np.pi * 1.0 * t_test)
z_test = signal.hilbert(test_trace)
a2_test = np.abs(z_test)**2
ph_test = np.unwrap(np.angle(z_test))
if_test = np.gradient(ph_test, dt_ns) / (2.0 * np.pi)

L_test = int(round(3.0 / dt_ns))
H_test = max(1, int(round(L_test * HOP_RATIO)))
starts_test = np.arange(0, n_samples - L_test + 1, H_test)
if_w_test = []
for st in starts_test:
    num = np.sum(a2_test[st:st+L_test] * if_test[st:st+L_test])
    den = np.sum(a2_test[st:st+L_test])
    if_w_test.append(num / (den + eps))
max_err = np.max(np.abs(np.array(if_w_test) - 1.0))
print(f"1) Sine wave test (1.0 GHz, T_avg=3.0ns): Max Error = {max_err*100:.4f}% (< 0.1% expected)")

# 2) STFT Centroid for comparison (T_avg=3.0 ns ~ nperseg=256)
nperseg_val = 256
noverlap_val = nperseg_val * 3 // 4

# 軸情報の取得（1トレース目のみ）
f_val, t_stft_val, _ = signal.stft(data_proc[:, 0], fs=fs, window='hann', nperseg=nperseg_val, noverlap=noverlap_val)
mask_val = (f_val >= freq_min) & (f_val <= freq_max)
valid_f_val = f_val[mask_val]

centroid_map_val = np.zeros((len(t_stft_val), n_traces))

# 雛形コードと同じ流儀でトレースごとにSTFT重心を計算
for itrace in range(n_traces):
    _, _, Zxx = signal.stft(data_proc[:, itrace], fs=fs, window='hann', nperseg=nperseg_val, noverlap=noverlap_val)
    power = np.abs(Zxx[mask_val, :]) ** 2
    total = power.sum(axis=0)
    centroid_map_val[:, itrace] = (valid_f_val[:, None] * power).sum(axis=0) / (total + eps)

prof_stft_med = np.nanmedian(centroid_map_val, axis=1)
print("2) STFT Centroid vs Hilbert IF_w (T_avg=3.0ns) comparison setup done.")

# =============================================================================
# Step 2: T_avg window loop (Envelope^2 weighted average)
# =============================================================================
comp_data = {}

for T_avg in T_AVG_LIST_NS:
    L = int(round(T_avg / dt_ns))
    H = max(1, int(round(L * HOP_RATIO)))
    starts = np.arange(0, n_samples - L + 1, H)
    n_windows = len(starts)
    
    t_axis_h = (starts + (L - 1) / 2.0) * dt_ns
    dt_hop = H * dt_ns
    extent_h = [0, n_traces * GPR_step, t_axis_h[-1], t_axis_h[0]]
    
    # 累積和(cumsum)による情報落ちを防ぐため、窓ごとに直接スライスして和を計算
    P_win = np.zeros((n_windows, n_traces))
    A2IF_win = np.zeros((n_windows, n_traces))
    
    for k, st in enumerate(starts):
        P_win[k, :] = np.sum(A2_full[st:st+L, :], axis=0)
        A2IF_win[k, :] = np.sum(A2_full[st:st+L, :] * IF_full[st:st+L, :], axis=0)
    
    # 修正パッチ実装（旧実装のP_win/A2IF_winループと再代入行は削除）
    IF_w, P_win = ifw_pulse_pair(Z, starts, L, dt_ns)
    valid_mask = noise_floor_mask(P_win, L, A2_full, noise_gate_ns=5.0,
                                dt_ns=dt_ns, snr_db=10.0)
    IF_w_masked = np.where(valid_mask, IF_w, np.nan)
    
    # パワーマスク
    # P_peak = np.max(P_win, axis=0, keepdims=True)
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     P_rel_db = 10.0 * np.log10(np.where(P_peak > 0, P_win / (P_peak + eps), eps))
    # valid_mask = P_rel_db >= power_threshold_db
    # IF_w_masked = np.where(valid_mask, IF_w, np.nan)
    
    # シフトレート
    shiftrate = np.gradient(IF_w_masked, dt_hop, axis=0)
    
    # 1Dプロファイル
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        prof_if_med = np.nanmedian(IF_w_masked, axis=1)
        prof_if_p25 = np.nanpercentile(IF_w_masked, 25, axis=1)
        prof_if_p75 = np.nanpercentile(IF_w_masked, 75, axis=1)
        
        prof_sr_med = np.nanmedian(shiftrate, axis=1)
        prof_sr_p25 = np.nanpercentile(shiftrate, 25, axis=1)
        prof_sr_p75 = np.nanpercentile(shiftrate, 75, axis=1)
        
    # 理論線の補間
    if analytical_calculated:
        analytical_if = np.interp(t_axis_h, t_delay_d_ns, f_peak_d_ghz, left=np.nan, right=np.nan)
        analytical_sr = np.gradient(analytical_if, dt_hop)
    else:
        analytical_if = None
        analytical_sr = None
        
    # 比較図用データの保存
    comp_data[T_avg] = {
        't': t_axis_h, 'if_med': prof_if_med, 'sr_med': prof_sr_med,
        'ana_if': analytical_if, 'ana_sr': analytical_sr
    }
    
    # 出力
    sub_dir = os.path.join(output_dir, f'Tavg_{T_avg:g}ns')
    os.makedirs(sub_dir, exist_ok=True)
    
    valid_pixel_ratio = np.sum(valid_mask) / valid_mask.size * 100
    print(f"\n--- T_avg = {T_avg:g} ns ---")
    print(f"Window: L = {L} samples ({L*dt_ns:.3f} ns), H = {H} samples")
    print(f"Number of windows: {n_windows}")
    print(f"Valid pixel ratio: {valid_pixel_ratio:.1f}%")
    
    if abs(T_avg - 3.0) < 1e-3:
        stft_interp = np.interp(t_axis_h, t_stft_val, prof_stft_med, left=np.nan, right=np.nan)
        valid_idx = np.isfinite(stft_interp) & np.isfinite(prof_if_med)
        if np.any(valid_idx):
            mean_rel_err = np.nanmean(np.abs(prof_if_med[valid_idx] - stft_interp[valid_idx]) / stft_interp[valid_idx]) * 100
            print(f"   => Validation: STFT-Centroid vs IF_w Mean Rel. Error = {mean_rel_err:.2f}%")
            
    # 色スケール
    valid_sr = shiftrate[np.isfinite(shiftrate)]
    sr_abs = np.percentile(np.abs(valid_sr), 95) if valid_sr.size > 0 else 1.0
    vmin_sr, vmax_sr = -sr_abs, sr_abs

    plot_freq_map(IF_w_masked, os.path.join(sub_dir, f'if_w_map_Tavg{T_avg:g}ns.png'), 
                  prof_if_med, prof_if_p25, prof_if_p75, t_axis_h, extent_h, analytical_if)
    plot_shiftrate_map(shiftrate, os.path.join(sub_dir, f'if_w_shiftrate_map_Tavg{T_avg:g}ns.png'), 
                       prof_sr_med, prof_sr_p25, prof_sr_p75, t_axis_h, extent_h, vmin_sr, vmax_sr, analytical_sr)
                       
    csv_path = os.path.join(sub_dir, f'if_w_profile_Tavg{T_avg:g}ns.csv')
    with open(csv_path, mode='w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(['t_ns', 'if_w_med', 'if_w_p25', 'if_w_p75', 'sr_med', 'sr_p25', 'sr_p75', 'analytical_if', 'analytical_sr'])
        for i in range(len(t_axis_h)):
            writer.writerow([
                t_axis_h[i],
                prof_if_med[i], prof_if_p25[i], prof_if_p75[i],
                prof_sr_med[i], prof_sr_p25[i], prof_sr_p75[i],
                analytical_if[i] if analytical_if is not None else np.nan,
                analytical_sr[i] if analytical_sr is not None else np.nan
            ])

# =============================================================================
# Step 3: Comparison Plots
# =============================================================================
def plot_comparison(data_dict, key_med, key_ana, ylabel, xlabel, fname, ylim_max, ylim_min, initial_delay):
    fig, ax = plt.subplots(figsize=(8, 10))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(T_AVG_LIST_NS)))
    
    for i, T_avg in enumerate(T_AVG_LIST_NS):
        d = data_dict[T_avg]
        ax.plot(d[key_med], d['t'], color=colors[i], lw=2, label=f'T_avg = {T_avg:g} ns')
        
    if analytical_calculated:
        d_base = data_dict[T_AVG_LIST_NS[0]]
        if d_base[key_ana] is not None:
            ax.plot(d_base[key_ana], d_base['t'], color='r', linestyle='--', lw=2, label='Analytical')
            
    ax.axhline(initial_delay, color='gray', linestyle='--', lw=2, label='Surface')
    ax.set_ylim(ylim_max, ylim_min)
    ax.set_xlabel(xlabel, size=18)
    ax.set_ylabel(ylabel, size=18)
    ax.tick_params(labelsize=14)
    ax.minorticks_on()
    ax.grid(True)
    ax.legend(fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)

ylim_max = max(d['t'][-1] for d in comp_data.values())
ylim_min = min(d['t'][0] for d in comp_data.values())
initial_delay = surface_delay_ns(0.35, 0.837)

plot_comparison(comp_data, 'if_med', 'ana_if', 'Delay time [ns]', 'Frequency [GHz]', 'if_w_profile_comparison.png', ylim_max, ylim_min, initial_delay)
plot_comparison(comp_data, 'sr_med', 'ana_sr', 'Delay time [ns]', 'Shift rate [GHz/ns]', 'if_w_shiftrate_comparison.png', ylim_max, ylim_min, initial_delay)

print(f'\nAll maps, profiles, and comparison figures saved to: {output_dir}')