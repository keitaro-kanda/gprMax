import os
import sys
import warnings
import csv
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import constants as const
import mpl_toolkits.axes_grid1 as axgrid1
from gprMax.exceptions import CmdInputError

# 既存コードの環境パス追加の流儀を踏襲
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.core.outputfiles_merge import get_output_data

"""
k_diff_ifw.py

[目的]
氷あり/なしのB-scanデータを同一パイプラインでHilbert IF_w解析し、
T_avgごとの差分（IF_w およびシフトレート）と領域統計を計算・出力する。
"""

# =============================================================================
# Constants & Parameters
# =============================================================================
T_AVG_LIST_NS = [1.0, 3.0, 10.0]
HOP_RATIO = 0.5
MEAN_TRACE_REMOVAL = True  # 差分比較のためTrue推奨
USE_BRICKWALL = True
freq_min = 0.5    # [GHz]
freq_max = 2.0    # [GHz]
eps = 1e-30

NOICE_JSON = {
        0.01: '/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_3x4/Ice_Detection_NoRock/No_Ice/rand_amp_001/Bscan/Bscan.json',
        # 0.05: '/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_3x4/Ice_Detection_NoRock/No_Ice/rand_amp_005/Bscan/Bscan.json',
        0.05: '/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_10x4/Ice_Detection_NoRock/No_Ice/rand_amp_005/Bscan/Bscan.json'
    }

# 氷層の深さ範囲（デフォルト）
LAYER_T0 = 14.4
LAYER_T1 = 37.8

# =============================================================================
# Helper Functions (Original logic copied identically)
# =============================================================================
def surface_delay_ns(antenna_height, system_lag_ns):
    return antenna_height * 2 / 0.3 + system_lag_ns

def compute_analytic_if(data_proc, dt_ns, freq_min, freq_max, use_brickwall=True):
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
    dph = np.angle(Z[1:, :] * np.conj(Z[:-1, :]))
    IF = np.vstack([np.zeros((1, Z.shape[1])), dph / (2*np.pi*dt_ns)])
    return Z, A2, IF
 
def ifw_pulse_pair(Z, starts, L, dt_ns):
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
    n_gate = int(round(noise_gate_ns / dt_ns))
    noise_A2 = np.mean(A2[-n_gate:, :], axis=0)
    P_noise = noise_A2[None, :] * L
    with np.errstate(divide='ignore'):
        snr = 10.0 * np.log10(P_win / (P_noise + eps))
    return snr >= snr_db

def region_stats(t, d, t0, t1, corr_len_ns):
    """区間 [t0,t1] の Δ について 平均・SEM・n_eff・z を返す。"""
    valid_mask = (t >= t0) & (t <= t1) & np.isfinite(d)
    valid_d = d[valid_mask]
    
    if len(valid_d) == 0:
        return np.nan, np.nan, 0, np.nan
        
    mean_val = np.mean(valid_d)
    std_val = np.std(valid_d, ddof=1) if len(valid_d) > 1 else 0.0
    
    interval_len_ns = t1 - t0
    n_eff = max(1.0, (interval_len_ns / corr_len_ns) + 1.0)
    
    sem = std_val / np.sqrt(n_eff)
    z_score = mean_val / sem if sem > 0 else np.nan
    
    return mean_val, sem, n_eff, z_score

def load_bscan(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"B-scan JSON not found: {json_path}")
    with open(json_path) as f:
        params = json.load(f)
    outfile = params['data']
    gpr_step = params['antenna_settings']['src_step']
    outputdata, dt = get_output_data(outfile, 1, 'Ez')
    return outputdata, dt, gpr_step, params

def compute_ifw_profiles(outputdata, dt, gpr_step, params, T_avg_list):
    dt_ns = dt * 1e9
    n_samples, n_traces = outputdata.shape
    
    data_proc = outputdata - outputdata.mean(axis=1, keepdims=True) if MEAN_TRACE_REMOVAL else outputdata
    Z, A2_full, IF_full = compute_analytic_if(data_proc, dt_ns, freq_min, freq_max, use_brickwall=USE_BRICKWALL)
    
    results = {}
    for T_avg in T_avg_list:
        L = int(round(T_avg / dt_ns))
        H = max(1, int(round(L * HOP_RATIO)))
        starts = np.arange(0, n_samples - L + 1, H)
        t_axis_h = (starts + (L - 1) / 2.0) * dt_ns
        dt_hop = H * dt_ns
        
        IF_w, P_win = ifw_pulse_pair(Z, starts, L, dt_ns)
        valid_mask = noise_floor_mask(P_win, L, A2_full, noise_gate_ns=5.0, dt_ns=dt_ns, snr_db=10.0)
        IF_w_masked = np.where(valid_mask, IF_w, np.nan)
        shiftrate = np.gradient(IF_w_masked, dt_hop, axis=0)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            prof_if_med = np.nanmedian(IF_w_masked, axis=1)
            prof_if_p25 = np.nanpercentile(IF_w_masked, 25, axis=1)
            prof_if_p75 = np.nanpercentile(IF_w_masked, 75, axis=1)
            
            prof_sr_med = np.nanmedian(shiftrate, axis=1)
            prof_sr_p25 = np.nanpercentile(shiftrate, 25, axis=1)
            prof_sr_p75 = np.nanpercentile(shiftrate, 75, axis=1)
            
        results[T_avg] = {
            't': t_axis_h,
            'if_med': prof_if_med, 'if_p25': prof_if_p25, 'if_p75': prof_if_p75,
            'sr_med': prof_sr_med, 'sr_p25': prof_sr_p25, 'sr_p75': prof_sr_p75,
            'n_traces': n_traces
        }
    return results

def calc_comb_se(p25_ice, p75_ice, p25_noice, p75_noice, n_traces):
    sigma_ice = (p75_ice - p25_ice) / 1.349
    sigma_noice = (p75_noice - p25_noice) / 1.349
    return np.sqrt(sigma_ice**2 + sigma_noice**2) / np.sqrt(n_traces)

# =============================================================================
# Main Execution Flow
# =============================================================================
def main():
    ice_json_path = input('Input ICE Bscan.json file path: ').strip()
    if not os.path.exists(ice_json_path):
        raise CmdInputError('JSON file {} does not exist'.format(ice_json_path))
        
    _sel = input('Select rand_amp for no-ice reference [0.01 / 0.05] (default 0.05): ').strip()
    rand_amp = 0.01 if _sel == '0.01' else 0.05
    print(f'Using no-ice reference for rand_amp = {rand_amp}')
    
    noice_json_path = NOICE_JSON.get(rand_amp)
    
    # --- 出力ディレクトリの作成 ---
    base_dir = os.path.dirname(os.path.abspath(ice_json_path))
    dir_name = 'hilbert_if_analysis_diff_meansub' if MEAN_TRACE_REMOVAL else 'hilbert_if_analysis_diff'
    out_dir = os.path.join(base_dir, dir_name)
    os.makedirs(out_dir, exist_ok=True)
    
    ice_name = os.path.basename(base_dir)
    noice_name = os.path.basename(os.path.dirname(noice_json_path))
    
    print(f"\nOutput directory: {out_dir}")
    print("\nLoading data...")
    out_ice, dt_ice, gstep_ice, p_ice = load_bscan(ice_json_path)
    out_noice, dt_noice, gstep_noice, p_noice = load_bscan(noice_json_path)
    
    # 時間グリッド整合性チェック
    if abs(dt_ice - dt_noice) > 1e-15 or out_ice.shape[0] != out_noice.shape[0]:
        raise ValueError("Error: dt or sample size mismatch between ice and no-ice datasets.")
    
    # トレース数整合性チェック
    if abs(dt_ice - dt_noice) > 1e-15 or out_ice.shape != out_noice.shape:
        raise ValueError("Error: shape/dt mismatch between ice and no-ice datasets.")
        
    print("Computing profiles for ice dataset...")
    res_ice = compute_ifw_profiles(out_ice, dt_ice, gstep_ice, p_ice, T_AVG_LIST_NS)
    print("Computing profiles for no-ice dataset...")
    res_noice = compute_ifw_profiles(out_noice, dt_noice, gstep_noice, p_noice, T_AVG_LIST_NS)
    
    initial_delay = surface_delay_ns(0.35, 0.837)
    
    for T_avg in T_AVG_LIST_NS:
        print(f"\n=== T_avg = {T_avg} ns ===")
        r_ice = res_ice[T_avg]
        r_noice = res_noice[T_avg]
        t_axis = r_ice['t']
        
        # 差分計算（両方が有限な点のみ）
        valid = np.isfinite(r_ice['if_med']) & np.isfinite(r_noice['if_med'])
        d_if = np.where(valid, r_ice['if_med'] - r_noice['if_med'], np.nan)
        d_sr = np.where(valid, r_ice['sr_med'] - r_noice['sr_med'], np.nan)
        
        se_if = calc_comb_se(r_ice['if_p25'], r_ice['if_p75'], r_noice['if_p25'], r_noice['if_p75'], r_ice['n_traces'])
        se_sr = calc_comb_se(r_ice['sr_p25'], r_ice['sr_p75'], r_noice['sr_p25'], r_noice['sr_p75'], r_ice['n_traces'])
        
        # 領域統計
        corr_len_ns = max(1.0, T_avg)
        if T_avg == 1.0:
            print("  * Note: T_avg=1.0ns は相関長が短くノイズが大きくなります。")
            
        stats_if_layer = region_stats(t_axis, d_if, LAYER_T0, LAYER_T1, corr_len_ns)
        stats_if_shal = region_stats(t_axis, d_if, 0, LAYER_T0, corr_len_ns)
        stats_sr_layer = region_stats(t_axis, d_sr, LAYER_T0, LAYER_T1, corr_len_ns)
        stats_sr_shal = region_stats(t_axis, d_sr, 0, LAYER_T0, corr_len_ns)
        
        print(f"  [Layer {LAYER_T0}-{LAYER_T1} ns]")
        print(f"    ΔIF_w: {stats_if_layer[0]:.4f} ± {stats_if_layer[1]:.4f} GHz (n_eff={stats_if_layer[2]:.1f}, z={stats_if_layer[3]:.2f})")
        print(f"    ΔLSR:  {stats_sr_layer[0]:.4f} ± {stats_sr_layer[1]:.4f} GHz/ns (n_eff={stats_sr_layer[2]:.1f}, z={stats_sr_layer[3]:.2f})")
        print(f"  [Shallow < {LAYER_T0} ns]")
        print(f"    ΔIF_w: {stats_if_shal[0]:.4f} ± {stats_if_shal[1]:.4f} GHz")
        print(f"    ΔLSR:  {stats_sr_shal[0]:.4f} ± {stats_sr_shal[1]:.4f} GHz/ns")

        # --- CSV保存 ---
        csv_path = os.path.join(out_dir, f'ifw_diff_profile_Tavg{T_avg:g}ns.csv')
        with open(csv_path, 'w', newline='') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(['t_ns', 'd_if_GHz', 'se_if_GHz', 'd_sr_GHz_ns', 'se_sr_GHz_ns'])
            for i in range(len(t_axis)):
                writer.writerow([t_axis[i], d_if[i], se_if[i], d_sr[i], se_sr[i]])
                
        # --- 作図 ---
        fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
        fig.suptitle(f'IF_w Difference (Ice - NoIce, rand_amp={rand_amp}) | T_avg={T_avg}ns\n{ice_name} vs {noice_name}')
        
        # 共通設定
        for ax in axes:
            ax.axhspan(LAYER_T0, LAYER_T1, color='blue', alpha=0.1, label='Ice Layer')
            ax.axhline(initial_delay, color='gray', linestyle='--', lw=1, label='Surface')
            ax.axvline(0, color='k', linestyle='-', lw=1)
            ax.set_ylim(t_axis[-1], t_axis[0])
            ax.set_ylabel('Delay time [ns]')
            ax.grid(True, linestyle=':')
            
        # パネル1: IF_w
        axes[0].plot(d_if, t_axis, 'b-', lw=1.5, label='ΔIF_w')
        axes[0].fill_betweenx(t_axis, d_if - se_if, d_if + se_if, color='b', alpha=0.3, label='±1σ')
        axes[0].set_xlabel('ΔIF_w [GHz]')
        axes[0].legend(loc='lower left')
        
        # パネル2: Shift rate
        axes[1].plot(d_sr, t_axis, 'r-', lw=1.5, label='ΔShift rate')
        axes[1].fill_betweenx(t_axis, d_sr - se_sr, d_sr + se_sr, color='r', alpha=0.3, label='±1σ')
        axes[1].set_xlabel('ΔShift rate [GHz/ns]')
        axes[1].legend(loc='lower left')
        
        plt.tight_layout()
        fig_path = os.path.join(out_dir, f'ifw_diff_profile_Tavg{T_avg:g}ns.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    print(f"\nAll diff maps and profiles saved to: {out_dir}")

if __name__ == '__main__':
    main()