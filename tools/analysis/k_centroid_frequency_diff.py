import os
import sys
import json
import h5py
import glob
import re
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import constants as const
from scipy.ndimage import gaussian_filter

# Add parent directories to path if needed to find tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.core.outputfiles_merge import get_output_data

# =============================================================================
# Helper Functions (Exact ports from k_calc_centroid_freq.py)
# =============================================================================
def get_eps_static(z_m):
    """Calculate static real part and loss tangent from depth."""
    z_cm = z_m * 100.0
    rho = 1.92 * (z_cm + 12.2) / (z_cm + 18.0)
    eps_static = 1.843 ** rho
    tan_d = 10 ** (0.033 * 20.0 + 0.231 * rho - 3.061)
    return eps_static, tan_d

def get_eps_regolith(z_m, omega, d_params, anchor_freq=450e6):
    """Return complex permittivity of regolith base material."""
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
    return antenna_height * 2 / 0.3 + system_lag_ns

def smooth_masked(data, mask, sigma):
    filled = np.where(mask, data, 0.0)
    sm_data   = gaussian_filter(filled,              sigma=sigma)
    sm_weight = gaussian_filter(mask.astype(float),  sigma=sigma)
    out = np.full_like(sm_data, np.nan)
    np.divide(sm_data, sm_weight, out=out, where=(sm_weight > 1e-6))
    out[~mask] = np.nan
    return out

def load_bscan(json_path):
    """Loads B-scan outputdata, dt, step, and params."""
    with open(json_path) as f:
        params = json.load(f)
    outfile = params['data']
    gpr_step = params['antenna_settings']['src_step']
    outputdata, dt = get_output_data(outfile, 1, 'Ez')
    return outputdata, dt, gpr_step, params

def compute_centroid_profiles(outputdata, dt, gpr_step, params, ascan_outfile_path=""):
    """Computes STFT centroid profiles strictly adhering to original logic."""
    dt_ns = dt * 1e9
    fs = 1.0 / dt_ns
    n_samples, n_traces = outputdata.shape

    # Debye parameters extraction
    debye_params = {'tau1': 4.6212e-11, 'tau2': 2.82195e-10, 'de_ratio': 0.261 / (0.261 + 0.088)}
    geom_json_path = params.get('geometry_settings', {}).get('geometry_json', '')
    in_dir = os.path.dirname(geom_json_path)
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
                    if m_ratio: debye_params['de_ratio'] = float(m_ratio.group(1))
                    elif m_disp:
                        de1, de2 = float(m_disp.group(1)), float(m_disp.group(2))
                        if (de1 + de2) > 0: debye_params['de_ratio'] = de1 / (de1 + de2)
            except Exception:
                pass

    # Constants
    nperseg, noverlap, window = 256, 256 * 3 // 4, 'hann'
    freq_min, freq_max = 0.25, 6.0
    power_threshold_db = -125.0
    sigma = (3, 3)
    eps = 1e-30

    # STFT Setup
    f_axis, t_axis, _ = signal.stft(outputdata[:, 0], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    freq_mask = (f_axis >= freq_min) & (f_axis <= freq_max)
    valid_freq = f_axis[freq_mask]
    n_time = t_axis.size

    centroid_map = np.zeros((n_time, n_traces))
    power_map = np.zeros((n_time, n_traces))

    # STFT Map Generation
    for itrace in range(n_traces):
        _, _, Zxx = signal.stft(outputdata[:, itrace], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
        power = np.abs(Zxx[freq_mask, :]) ** 2          
        total = power.sum(axis=0)                        
        centroid_map[:, itrace] = (valid_freq[:, None] * power).sum(axis=0) / (total + eps)
        power_map[:, itrace] = total

    # Masking
    trace_peak = power_map.max(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        power_rel_db = 10.0 * np.log10(np.where(trace_peak > 0, power_map / (trace_peak + eps), eps))
    valid_mask = power_rel_db >= power_threshold_db
    centroid_masked = np.where(valid_mask, centroid_map, np.nan)
    centroid_smooth = smooth_masked(centroid_map, valid_mask, sigma)

    dt_stft = t_axis[1] - t_axis[0]
    def shift_rate(freq_map): return np.gradient(freq_map, dt_stft, axis=0)
    shiftrate_smooth = shift_rate(centroid_smooth)

    # 1D Medians & IQRs (Smoothed ONLY as requested)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cen_med = np.nanmedian(centroid_smooth, axis=1)
        cen_p25 = np.nanpercentile(centroid_smooth, 25, axis=1)
        cen_p75 = np.nanpercentile(centroid_smooth, 75, axis=1)
        sr_med = np.nanmedian(shiftrate_smooth, axis=1)
        sr_p25 = np.nanpercentile(shiftrate_smooth, 25, axis=1)
        sr_p75 = np.nanpercentile(shiftrate_smooth, 75, axis=1)

    # Analytical Profiles (Regolith Method A)
    analytical_f_peak_profile = np.full_like(t_axis, np.nan)
    analytical_shiftrate_profile = np.full_like(t_axis, np.nan)
    
    # We attempt an analytical profile calculation if an Ascan path is provided and exists
    if ascan_outfile_path and os.path.exists(ascan_outfile_path):
        try:
            ascan_data, dt_ascan = get_output_data(ascan_outfile_path, 1, 'Ez')
            e_incident = ascan_data if ascan_data.ndim == 1 else ascan_data[:, 0]
            freq_ascan = np.fft.rfftfreq(len(e_incident), d=dt_ascan)
            S0_omega = np.fft.rfft(e_incident)
            band_mask = (freq_ascan >= freq_min*1e9) & (freq_ascan <= freq_max*1e9)
            f_calc, S0_calc = freq_ascan[band_mask], S0_omega[band_mask]
            omega = 2 * np.pi * f_calc

            f_center = 450e6
            antenna_height, system_lag_ns, rx_depth = 0.35, 0.837, 0.10
            t_air_ns = (2.0 * antenna_height / const.c) * 1e9
            
            d_sub_offset = np.linspace(0, rx_depth, 50)
            eps_sub_offset, _ = get_eps_static(d_sub_offset)
            t_ground_start_ns = np.sum(2.0 * (d_sub_offset[1]-d_sub_offset[0]) / (const.c / np.sqrt(eps_sub_offset))) * 1e9
            t_offset_ns = system_lag_ns + t_air_ns + t_ground_start_ns

            max_depth = (t_axis[-1] * 1e-9) * const.c / 2 
            d_array = np.linspace(rx_depth, max_depth, 400)
            d_step = d_array[1] - d_array[0]
            
            f_peak_d, t_delay_d = [], []
            cumulative_attenuation = np.zeros_like(omega)
            cumulative_time = np.zeros_like(omega)

            for i, d in enumerate(d_array):
                eps_complex_w = get_eps_regolith(d, omega, debye_params, anchor_freq=f_center)
                alpha_d = - (omega / const.c) * np.imag(np.sqrt(eps_complex_w))
                v_d = const.c / np.real(np.sqrt(eps_complex_w))
                
                if i > 0:
                    cumulative_attenuation += alpha_d * d_step
                    cumulative_time += 2 * d_step / v_d
                    
                power = np.abs(S0_calc * np.exp(-2 * cumulative_attenuation))**2
                f_peak_d.append(np.trapz(f_calc * power, f_calc) / np.trapz(power, f_calc))
                
                t_delay_ground = np.interp(f_peak_d[-1], f_calc, cumulative_time)
                t_delay_d.append(t_offset_ns + (t_delay_ground * 1e9))

            analytical_f_peak_profile = np.interp(t_axis, t_delay_d, np.array(f_peak_d) / 1e9, left=np.nan, right=np.nan)
            analytical_shiftrate_profile = np.gradient(analytical_f_peak_profile, dt_stft)
        except Exception as e:
            print(f"Warning: Analytical calculation failed: {e}")

    surf_delay = surface_delay_ns(0.35, 0.837)
    
    return {
        't': t_axis, 'cen_med': cen_med, 'cen_p25': cen_p25, 'cen_p75': cen_p75,
        'sr_med': sr_med, 'sr_p25': sr_p25, 'sr_p75': sr_p75,
        'analytical_cen': analytical_f_peak_profile, 'analytical_sr': analytical_shiftrate_profile,
        'surface_delay': surf_delay, 'n_traces': n_traces
    }

def region_stats(t, d, t0, t1, corr_len_ns):
    """Calculates statistics for a region with standard error adjustment for correlation."""
    mask = (t >= t0) & (t <= t1) & np.isfinite(d)
    if not np.any(mask):
        return np.nan, np.nan, 0, np.nan
    d_val = d[mask]
    mean_d = np.mean(d_val)
    std_d = np.std(d_val, ddof=1) if len(d_val) > 1 else 0.0
    T_len = t1 - t0
    n_eff = max(1.0, (T_len / corr_len_ns) + 1.0)
    sem = std_d / np.sqrt(n_eff) if n_eff > 0 else np.nan
    z = mean_d / sem if sem > 0 and not np.isnan(sem) else np.nan
    return mean_d, sem, n_eff, z

# =============================================================================
# Main Execution Pipeline
# =============================================================================
if __name__ == "__main__":
    # 1. Inputs
    ice_json_path = input('Input ICE Bscan.json file path: ').strip()
    if not os.path.exists(ice_json_path):
        raise FileNotFoundError(f"ICE JSON file not found: {ice_json_path}")

    _sel = input('Select rand_amp for no-ice reference [0.01 / 0.05] (default 0.05): ').strip()
    rand_amp = 0.01 if _sel == '0.01' else 0.05
    print(f'Using no-ice reference for rand_amp = {rand_amp}')

    NOICE_JSON = {
        0.01: '/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_3x4/Ice_Detection_NoRock/No_Ice/rand_amp_001/Bscan/Bscan.json',
        0.05: '/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_3x4/Ice_Detection_NoRock/No_Ice/rand_amp_005/Bscan/Bscan.json',
    }
    noice_json_path = NOICE_JSON[rand_amp]

    if not os.path.exists(noice_json_path):
        raise FileNotFoundError(f"NO-ICE absolute path missing (Check NOICE_JSON placeholder): {noice_json_path}")

    # ---------------------------------------------------------
    # [変更点] 出力ディレクトリの構築ロジック
    # ---------------------------------------------------------
    # meansub の有無を判定するフラグ (必要に応じて対話的入力や引数に変更してください)
    use_meansub = False 
    
    # 手法に応じたディレクトリベース名（本コードは centroid 用）
    # ※IF法の場合は 'hilbert_if_analysis_diff' に変更してください
    dir_name = 'centroid_frequency_analysis_diff'
    if use_meansub:
        dir_name += '_meansub'
        
    out_dir = os.path.join(os.path.dirname(os.path.abspath(ice_json_path)), dir_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f'Output directory: {out_dir}')
    # ---------------------------------------------------------
    
    # Optional A-scan file passed to analytical derivation if needed
    ascan_path = "" # Point to an A-scan if analytical plotting is desired

    # 2. Data Loading & Profiling
    print("Computing ICE profiles...")
    out_ice, dt_ice, step_ice, params_ice = load_bscan(ice_json_path)
    prof_ice = compute_centroid_profiles(out_ice, dt_ice, step_ice, params_ice, ascan_path)

    print("Computing NO-ICE profiles...")
    out_noice, dt_noice, step_noice, params_noice = load_bscan(noice_json_path)
    prof_noice = compute_centroid_profiles(out_noice, dt_noice, step_noice, params_noice, ascan_path)

    # 3. Matching & Difference Calculation
    t_ice, t_noice = prof_ice['t'], prof_noice['t']
    if not (t_ice.size == t_noice.size and np.allclose(t_ice, t_noice, atol=1e-6)):
        raise ValueError("Time grids of ICE and NO-ICE do not match. Aborting.")
    t = t_ice

    # Difference Profile Calculation
    d_cen = prof_ice['cen_med'] - prof_noice['cen_med']
    d_sr  = prof_ice['sr_med'] - prof_noice['sr_med']

    # Combined Error Margin (1-sigma converted from IQR)
    sig_cen_ice = (prof_ice['cen_p75'] - prof_ice['cen_p25']) / 1.349
    sig_cen_noice = (prof_noice['cen_p75'] - prof_noice['cen_p25']) / 1.349
    syn_sigma_cen = np.sqrt(sig_cen_ice**2 + sig_cen_noice**2) / np.sqrt(prof_ice['n_traces'])

    sig_sr_ice = (prof_ice['sr_p75'] - prof_ice['sr_p25']) / 1.349
    sig_sr_noice = (prof_noice['sr_p75'] - prof_noice['sr_p25']) / 1.349
    syn_sigma_sr = np.sqrt(sig_sr_ice**2 + sig_sr_noice**2) / np.sqrt(prof_ice['n_traces'])

    # 4. Regional Stats
    print("\nNote: The ice signature in STFT Centroid mapping is inherently small (ΔIF ≈ -0.01 to -0.03 GHz).")
    corr_len_ns = 3.0 # Defined in prompt due to Gaussian(3,3) smoothing
    
    t_surface = prof_ice['surface_delay']
    t_layer_start = 14.4
    t_layer_end = 37.8

    stats_cen_shallow = region_stats(t, d_cen, t_surface, t_layer_start, corr_len_ns)
    stats_cen_layer = region_stats(t, d_cen, t_layer_start, t_layer_end, corr_len_ns)
    stats_sr_shallow = region_stats(t, d_sr, t_surface, t_layer_start, corr_len_ns)
    stats_sr_layer = region_stats(t, d_sr, t_layer_start, t_layer_end, corr_len_ns)

    print("\n=== Statistics (Shallow: Surface to 14.4ns | Layer: 14.4ns to 37.8ns) ===")
    print(f"Δ Centroid - Shallow : Mean={stats_cen_shallow[0]:.5f} GHz ± SEM={stats_cen_shallow[1]:.5f} (n_eff={stats_cen_shallow[2]:.1f}, z={stats_cen_shallow[3]:.2f})")
    print(f"Δ Centroid - Layer   : Mean={stats_cen_layer[0]:.5f} GHz ± SEM={stats_cen_layer[1]:.5f} (n_eff={stats_cen_layer[2]:.1f}, z={stats_cen_layer[3]:.2f})")
    print(f"Δ ShiftRate - Shallow: Mean={stats_sr_shallow[0]:.5f} GHz/ns ± SEM={stats_sr_shallow[1]:.5f} (n_eff={stats_sr_shallow[2]:.1f}, z={stats_sr_shallow[3]:.2f})")
    print(f"Δ ShiftRate - Layer  : Mean={stats_sr_layer[0]:.5f} GHz/ns ± SEM={stats_sr_layer[1]:.5f} (n_eff={stats_sr_layer[2]:.1f}, z={stats_sr_layer[3]:.2f})")

    # Sanity Check Check
    if ice_json_path == noice_json_path:
        is_zero_cen = np.allclose(np.nan_to_num(d_cen), 0, atol=1e-8)
        is_zero_sr = np.allclose(np.nan_to_num(d_sr), 0, atol=1e-8)
        print(f"\nSANITY CHECK (Self-diff == 0): Centroid={is_zero_cen}, ShiftRate={is_zero_sr}")

    # 5. Output CSV
    csv_path = os.path.join(out_dir, 'centroid_diff_profile.csv')
    np.savetxt(csv_path, np.column_stack((t, d_cen, d_sr)), delimiter=',', header='t_ns,d_cen,d_sr', comments='')
    print(f"\nSaved CSV to: {csv_path}")

    # 6. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    ice_base = os.path.basename(ice_json_path)
    noice_base = os.path.basename(noice_json_path)
    fig.suptitle(f"Diff: {ice_base} vs {noice_base} (rand={rand_amp})", fontsize=16)

    # Plot Settings Mapping
    plots = [
        (axes[0], d_cen, syn_sigma_cen, 'Δ Centroid [GHz]'),
        (axes[1], d_sr, syn_sigma_sr, 'Δ Shift Rate [GHz/ns]')
    ]

    for ax, data, sigma, title in plots:
        ax.plot(data, t, 'k-', lw=1.5)
        ax.fill_betweenx(t, data - sigma, data + sigma, color='gray', alpha=0.4, label='±1σ (synth)')
        ax.axhspan(t_layer_start, t_layer_end, color='blue', alpha=0.1, label='Ice Layer (14.4-37.8ns)')
        ax.axvline(0, color='r', linestyle='--', alpha=0.7)
        ax.set_ylim(t[-1], t[0])
        ax.set_xlabel(title, fontsize=14)
        ax.grid(True)
        ax.legend(loc='lower left')

    axes[0].set_ylabel('Delay time [ns]', fontsize=14)
    plt.tight_layout()
    
    img_path = os.path.join(out_dir, 'centroid_diff_profile.png')
    fig.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"Saved Image to: {img_path}")
    plt.close(fig)