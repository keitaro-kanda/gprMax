#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import importlib.util
from scipy.signal import hilbert
import json

# 絶対パスでoutputfiles_merge.pyを動的にロード
script_dir = os.path.dirname(os.path.abspath(__file__))
gprmax_root = os.path.dirname(os.path.dirname(script_dir))
outputfiles_merge_path = os.path.join(gprmax_root, 'tools', 'core', 'outputfiles_merge.py')

spec = importlib.util.spec_from_file_location("outputfiles_merge", outputfiles_merge_path)
outputfiles_merge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(outputfiles_merge)

# get_output_data関数を取得
get_output_data = outputfiles_merge.get_output_data

def detect_peaks(data, dt, FWHM_transmission=None):
    """
    Detects peaks in A-scan data. This is the core logic function.

    Args:
        data (np.ndarray): Amplitude data of the A-scan.
        dt (float): Time step of the simulation.
        FWHM_transmission (float, optional): Expected FWHM of transmission pulse [s]. If provided, calculates FWHM_difference.

    Returns:
        list: A list of dictionaries, where each dictionary contains information about a detected peak.
    """
    data_norm = data / np.amax(np.abs(data)) # Normalize the data
    time = np.arange(len(data_norm)) * dt  / 1e-9 # [ns]

    analytic_signal = hilbert(data_norm)
    envelope = np.abs(analytic_signal)
    evnvelope_moving_average = np.convolve(envelope, np.ones(10)/10, mode='same')

    peaks = []
    for i in range(1, len(data_norm) - 1):
        if evnvelope_moving_average[i - 1] < evnvelope_moving_average[i] > evnvelope_moving_average[i + 1] and evnvelope_moving_average[i] > 0.5e-3:
            peaks.append(i)

    pulse_info = []
    for i, peak_idx in enumerate(peaks):
        peak_amplitude = envelope[peak_idx]
        half_amplitude = peak_amplitude / 2

        left_idx = peak_idx
        while left_idx > 0 and envelope[left_idx] > half_amplitude:
            left_idx -= 1

        if left_idx == 0:
            left_half_time = time[0]
        else:
            left_slope = (envelope[left_idx + 1] - envelope[left_idx]) / (time[left_idx + 1] - time[left_idx])
            left_half_time = time[left_idx] + (half_amplitude - envelope[left_idx]) / left_slope

        right_idx = peak_idx
        while right_idx < len(envelope) - 1 and envelope[right_idx] > half_amplitude:
            right_idx += 1

        if right_idx == len(envelope) - 1:
            right_half_time = time[-1]
        else:
            right_slope = (envelope[right_idx] - envelope[right_idx - 1]) / (time[right_idx] - time[right_idx - 1])
            right_half_time = time[right_idx - 1] + (half_amplitude - envelope[right_idx - 1]) / right_slope

        fwhm = right_half_time - left_half_time
        hwhm = np.min([np.abs(time[peak_idx] - left_half_time), np.abs(time[peak_idx] - right_half_time)])

        # Calculate FWHM difference if FWHM_transmission is provided
        if FWHM_transmission is not None:
            fwhm_ns = fwhm  # FWHM is already in [ns]
            FWHM_transmission_ns = FWHM_transmission * 1e9  # Convert [s] to [ns]
            fwhm_error = np.abs(fwhm_ns - FWHM_transmission_ns) / FWHM_transmission_ns
            if fwhm_error < 0.1:
                fwhm_difference = 'N'
            else:
                fwhm_difference = 'Larger than 10%'
        else:
            fwhm_difference = 'N'  # Default value when FWHM_transmission is not provided

        separation_prev = time[peak_idx] - time[peaks[i - 1]] if i > 0 else None
        separation_next = time[peaks[i + 1]] - time[peak_idx] if i < len(peaks) - 1 else None

        # Check if the peaks are distinguishable
        distinguishable = True
        if separation_prev is not None and separation_prev < fwhm/2: # 試しに変更
            distinguishable = False
        if separation_next is not None and separation_next < fwhm/2:
            distinguishable = False

        # Find the maximum amplitude of the E-field within the FWHM range
        data_segment = np.abs(data_norm[int(left_half_time*1e-9/dt):int(right_half_time*1e-9/dt)])
        max_idx = peak_idx
        max_time = time[peak_idx]
        max_amplitude = data_norm[peak_idx]
        if len(data_segment) > 0:
            primary_max_idx = np.argmax(np.abs(data_segment))
            max_idx = int(left_half_time*1e-9/dt) + primary_max_idx
            max_time = time[max_idx]
            max_amplitude = data_norm[max_idx]
            distinguishable = True # We can find the peak within the FWHM range at least once.

        pulse_info.append({
            'peak_idx': peak_idx,
            'peak_time': time[peak_idx],
            'peak_amplitude': peak_amplitude,
            'width': fwhm,
            'width_half': hwhm,
            'left_half_time': left_half_time,
            'right_half_time': right_half_time,
            'FWHM': fwhm,
            'FWHM_difference': fwhm_difference,
            'separation': min(separation_prev or np.inf, separation_next or np.inf),
            'distinguishable': distinguishable,
            'max_idx': max_idx,
            'max_time': max_time,
            'max_amplitude': max_amplitude,
        })
    return pulse_info


def detect_two_peaks(data, dt, FWHM_transmission):
    """
    Detects the two largest peaks in A-scan data (first and second largest).
    This improved version uses better peak detection and masking strategies.

    Args:
        data (np.ndarray): Amplitude data of the A-scan.
        dt (float): Time step of the simulation.

    Returns:
        list: A list of dictionaries, where each dictionary contains information about the two largest detected peaks.
              Each entry has 'primary' and 'secondary' peak information.
    """
    from scipy.signal import find_peaks
    
    data_norm = data / np.amax(np.abs(data)) # Normalize the data
    time = np.arange(len(data_norm)) * dt  / 1e-9 # [ns]

    analytic_signal = hilbert(data_norm)
    envelope = np.abs(analytic_signal)
    evnvelope_moving_average = np.convolve(envelope, np.ones(10)/10, mode='same') # なんで移動平均を使おうとしたんだ？？

    # Find envelope peaks (same as original)
    peaks = []
    for i in range(1, len(data_norm) - 1):
        # if evnvelope_moving_average[i - 1] < evnvelope_moving_average[i] > evnvelope_moving_average[i + 1] and evnvelope_moving_average[i] > 0.5e-3:
        if envelope[i-1] <= envelope[i] and envelope[i+1] <= envelope[i] and envelope[i] > 0.5e-3:
            peaks.append(i)

    pulse_info = []
    for i, peak_idx in enumerate(peaks):
        peak_amplitude = envelope[peak_idx]
        half_amplitude = peak_amplitude / 2

        # Calculate FWHM boundaries (same as original)
        left_idx = peak_idx
        while left_idx > 0 and envelope[left_idx] > half_amplitude:
            left_idx -= 1

        if left_idx == 0:
            left_half_time = time[0]
        else:
            left_slope = (envelope[left_idx + 1] - envelope[left_idx]) / (time[left_idx + 1] - time[left_idx])
            left_half_time = time[left_idx] + (half_amplitude - envelope[left_idx]) / left_slope

        right_idx = peak_idx
        while right_idx < len(envelope) - 1 and envelope[right_idx] > half_amplitude:
            right_idx += 1

        if right_idx == len(envelope) - 1:
            right_half_time = time[-1]
        else:
            right_slope = (envelope[right_idx] - envelope[right_idx - 1]) / (time[right_idx] - time[right_idx - 1])
            right_half_time = time[right_idx - 1] + (half_amplitude - envelope[right_idx - 1]) / right_slope

        fwhm = right_half_time - left_half_time
        fwhm_effor = (np.abs(fwhm - FWHM_transmission) / FWHM_transmission)
        if fwhm_effor < 0.1:
            fwhm_difference = 'False'
        else:
            fwhm_difference = 'True'

        separation_prev = time[peak_idx] - time[peaks[i - 1]] if i > 0 else None
        separation_next = time[peaks[i + 1]] - time[peak_idx] if i < len(peaks) - 1 else None

        distinguishable = True
        if separation_prev is not None and separation_prev < fwhm/2:
            distinguishable = False
        if separation_next is not None and separation_next < fwhm/2:
            distinguishable = False

        # # Improved peak detection: Expand search range to 1.5x FWHM
        # if fwhm > 0 and dt > 0:
        #     fwhm_idx = max(1, int(fwhm * 1e-9 / dt))  # FWHM in index units, at least 1
        #     search_radius = max(int(1.5 * fwhm_idx), 20)  # At least 20 samples
        # else:
        #     search_radius = 50  # Default search radius
        
        data_segment_start = int(max(0, peak_idx - fwhm * 1e-9/dt/2))
        data_segment_end = int(min(len(data_norm), peak_idx + fwhm * 1e-9/dt/2))
        # print("peak_idx: ", peak_idx)
        # print("data_segment_start: ", data_segment_start)
        # print("data_segment_end: ", data_segment_end)

        # Find local minimum in the detected envelope peak \pm FWHM/2
        local_min_idxs = []
        for k in (data_segment_start, min(data_segment_end, len(data)-2)):
            if envelope[k] <= envelope[k+1] and envelope[k-1] <= envelope[k]:
                local_min_idxs.append(k)
        local_min_idxs = np.array(local_min_idxs) # この後のpeak_idxより大きい、小さい要素探索でarrayである必要がある。
        # print(local_min_idxs)
        # Redefine data segment based on detected local minimum points
        if len(local_min_idxs) > 0:
            if len(local_min_idxs[local_min_idxs < peak_idx]) > 0:
                local_min_idx_start = np.amax(local_min_idxs[local_min_idxs < peak_idx]) # local_min_idxのうち、peak_idx以下かつその中で最大のidx（＝peak_idxに最も近い極小idx）を探索
                data_segment_start = max(data_segment_start,local_min_idx_start) # peak_idx-FWHM/2と極小idxのうち、大きい方（peak_idxに近い方）を採用
            if len(local_min_idxs[local_min_idxs > peak_idx]) > 0:
                local_min_idx_end = np.amin(local_min_idxs[local_min_idxs > peak_idx]) # local_min_idxのうち、peak_idx以上かつその中で最小のidx（＝peak_idxに最も近い極小idx）を探索
                data_segment_end = min(data_segment_end, local_min_idx_end) #peak_idx-FWHM/2と極大idxのうち、大小さい方（peak_idxに近い方）を採用
        # if local_min_idx_start is not None:
            
        # if local_min_idx_end is not None:
            
        # print("data_segment_start: ", data_segment_start)
        
        data_segment = np.abs(data_norm[data_segment_start:data_segment_end]) # data_segment is an array of values of the envelpe


        # Initialize peak info structure
        peak_info = {
            'peak_idx': peak_idx,
            'peak_time': time[peak_idx],
            'peak_amplitude': peak_amplitude,
            'FWHM': fwhm,
            'FWHM_difference': fwhm_difference, # 送信波FWHMとの誤差が10%以内かどうか
            'separation': min(separation_prev or np.inf, separation_next or np.inf),
            'distinguishable': distinguishable, # 隣のエコーと分離できているか
            'primary': None,
            'secondary': None
        }
        
        if len(data_segment) > 5:  # Need minimum samples for reliable detection
            # Calculate noise level for adaptive threshold with safety checks
            max_val = np.max(data_segment)
            if max_val > 0:
                noise_data = data_segment[data_segment < 0.1 * max_val]
                if len(noise_data) > 1:
                    noise_level = np.std(noise_data)
                else:
                    noise_level = 0.01 * max_val  # Fallback noise level
            else:
                noise_level = 0.001  # Minimal noise level
                
            min_peak_height = max(0.05 * max_val, 3 * noise_level)
            
            # Calculate minimum distance between peaks with safety check
            if fwhm > 0 and dt > 0:
                fwhm_idx = max(1, int(fwhm * 1e-9 / dt))
                min_distance = max(2, int(0.1 * fwhm_idx))
            else:
                min_distance = 3  # Default minimum distance
            
            # Find all local maxima using scipy
            try:
                local_peaks, _ = find_peaks(data_segment, 
                                           height=min_peak_height,
                                           distance=min_distance)
            except Exception:
                # Fallback if find_peaks fails
                local_peaks = []
            
            if len(local_peaks) > 0:
                # Sort peaks by amplitude
                peak_amplitudes = data_segment[local_peaks]
                sorted_indices = np.argsort(peak_amplitudes)[::-1]  # Descending order
                
                # Primary peak (largest)
                primary_local_idx = local_peaks[sorted_indices[0]]
                primary_global_idx = data_segment_start + primary_local_idx
                primary_max_time = time[primary_global_idx]
                primary_max_amplitude = data_norm[primary_global_idx]
                
                peak_info['primary'] = {
                    'max_idx': primary_global_idx,
                    'max_time': primary_max_time,
                    'max_amplitude': primary_max_amplitude,
                }
                
                # Secondary peak (second largest) if exists
                if len(local_peaks) == 1:
                    peak_info['secondary'] = {
                            'max_idx': 'No secondary peak',
                            'max_time': 'No secondary peak',
                            'max_amplitude': 'No secondary peak',
                        }
                elif len(local_peaks) > 1:
                    secondary_local_idx = local_peaks[sorted_indices[1]]
                    secondary_global_idx = data_segment_start + secondary_local_idx
                    secondary_max_time = time[secondary_global_idx]
                    secondary_max_amplitude = data_norm[secondary_global_idx]
                    
                    # More flexible threshold: 10% of primary or 5x noise level
                    min_secondary_threshold = max(0.1 * abs(primary_max_amplitude), 
                                                 5 * noise_level)
                    
                    if abs(secondary_max_amplitude) >= min_secondary_threshold:
                        peak_info['secondary'] = {
                            'max_idx': secondary_global_idx,
                            'max_time': secondary_max_time,
                            'max_amplitude': secondary_max_amplitude,
                        }
                
                # For backwards compatibility
                # peak_info['max_idx'] = primary_global_idx
                # peak_info['max_time'] = primary_max_time
                # peak_info['max_amplitude'] = primary_max_amplitude
            else:
                # Fallback to original method if no peaks found
                primary_max_idx = np.argmax(data_segment)
                primary_global_idx = data_segment_start + primary_max_idx
                primary_max_time = time[primary_global_idx]
                primary_max_amplitude = data_norm[primary_global_idx]
                
                peak_info['primary'] = {
                    'max_idx': primary_global_idx,
                    'max_time': primary_max_time,
                    'max_amplitude': primary_max_amplitude,
                }
                
                # peak_info['max_idx'] = primary_global_idx
                # peak_info['max_time'] = primary_max_time
                # peak_info['max_amplitude'] = primary_max_amplitude

        pulse_info.append(peak_info)
    
    return pulse_info

#* Define the function to analyze the pulses
def detect_plot_peaks(data, dt, closeup, closeup_x_start, closeup_x_end, closeup_y_start, closeup_y_end, FWHM, output_dir, plt_show):
    pulse_info = detect_peaks(data, dt)
    data_norm = data / np.amax(np.abs(data))
    time = np.arange(len(data_norm)) * dt  / 1e-9 # [ns]
    analytic_signal = hilbert(data_norm)
    envelope = np.abs(analytic_signal)

    #* Plot A-scan
    fig, ax = plt.subplots(subplot_kw=dict(xlabel='Time [ns]', ylabel='Ez normalized field strength'),
                            figsize=(20, 10), facecolor='w', edgecolor='w', tight_layout=True)
    ax.plot(time, data_norm, label='A-scan', color='black')
    ax.plot(time, envelope, label='Envelope', color='blue', linestyle='-.')
    ax.plot(time, -envelope, color='blue', linestyle='-.')

    for i, info in enumerate(pulse_info):
        if info['distinguishable']==True:
            plt.plot(info['max_time'], info['max_amplitude'], 'ro', label='Primary Peak' if i == 0 else "", markersize=10)

        if FWHM:
            if info['distinguishable'] or len(pulse_info) == 1:
                plt.hlines(info['peak_amplitude'] / 2,
                        info['left_half_time'],
                        info['right_half_time'],
                        color='green', linestyle='--', label='FWHM' if i == 0 else "")

    plt.xlabel('Time [ns]', fontsize=24)
    plt.ylabel('Amplitude', fontsize=24)
    plt.legend(fontsize=20)
    plt.tick_params(labelsize=20)
    plt.grid(True)

    if FWHM or closeup:
        ax.set_xlim([closeup_x_start, closeup_x_end])
        ax.set_ylim([closeup_y_start, closeup_y_end])
    else:
        ax.set_xlim([0, np.amax(time)])

    if output_dir:
        if closeup:
            fig.savefig(os.path.join(output_dir, f'peak_detection_closeup_x{closeup_x_start}_{closeup_x_end}_y{closeup_y_start}_{closeup_y_end}.png'),
                        dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
        else:
            fig.savefig(os.path.join(output_dir, 'peak_detection.png'), dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

    if plt_show:
        plt.show()
    else:
        plt.close()

    return pulse_info


if __name__ == "__main__":
    print("=== Peak Detection Tool ===")
    print("Detect the peak from the A-scan data")
    print()
    
    # Get output file path
    while True:
        data_path = input("Enter path to the .out file: ").strip()
        if os.path.exists(data_path) and data_path.endswith('.out'):
            break
        else:
            print("Error: File does not exist or is not a .out file. Please try again.")
    
    # Get closeup option
    while True:
        closeup_input = input("Enable closeup zoom? (y/n): ").strip().lower()
        if closeup_input in ['y', 'yes']:
            closeup = True
            break
        elif closeup_input in ['n', 'no']:
            closeup = False
            break
        else:
            print("Please enter 'y' or 'n'.")
    
    # Get FWHM option
    while True:
        fwhm_input = input("Plot FWHM? (y/n): ").strip().lower()
        if fwhm_input in ['y', 'yes']:
            fwhm = True
            break
        elif fwhm_input in ['n', 'no']:
            fwhm = False
            break
        else:
            print("Please enter 'y' or 'n'.")
    
    # Get closeup parameters if enabled
    if closeup:
        print("\nEnter closeup parameters:")
        while True:
            try:
                closeup_x_start = float(input("X-axis start [ns] (default: 20): ") or "20")
                closeup_x_end = float(input("X-axis end [ns] (default: 40): ") or "40")
                if closeup_x_start >= closeup_x_end:
                    print("Error: X-axis start must be less than end. Please try again.")
                    continue
                break
            except ValueError:
                print("Error: Please enter valid numbers.")
        
        while True:
            try:
                closeup_y_range = float(input('Y-axis closeup range [ns] (default: 0.03): ' or '0.03'))
                closeup_y_start = - closeup_y_range
                closeup_y_end = closeup_y_range
                if closeup_y_start >= closeup_y_end:
                    print("Error: Y-axis start must be less than end. Please try again.")
                    continue
                break
            except ValueError:
                print("Error: Please enter valid numbers.")

    #* Define path
    output_dir = os.path.join(os.path.dirname(data_path), 'peak_detection')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing file: {data_path}")
    print(f"Output directory: {output_dir}")

    try:
        #* Load the A-scan data
        f = h5py.File(data_path, 'r')
        nrx = f.attrs['nrx']
        for rx in range(nrx):
            data, dt = get_output_data(data_path, (rx+1), 'Ez')

        time = np.arange(len(data)) * dt

        #* Run the pulse analysis
        pulse_info = detect_plot_peaks(data, dt, closeup, closeup_x_start, closeup_x_end, closeup_y_start, closeup_y_end, fwhm, output_dir, plt_show=True)

        # 結果の表示
        print("\n=== Peak Detection Results ===")
        for info in pulse_info:
            print(f"Peak at {info['peak_time']:.4f} ns: Width={info['width']:.4f} ns, Distinguishable={info['distinguishable']}")

        #* Save the pulse information
        filename = os.path.join(output_dir, 'peak_info.txt')
        try:
            # with open(peak_info_filename, "w") as fout:
            #     json.dump(pulse_info, fout, indent=2)
            # NumPy 型（np.generic）を Python の組み込み型に変換
            serializable_pulse = []
            for info in pulse_info:
                serializable_pulse.append({
                    key: (value.item() if isinstance(value, np.generic) else value)
                    for key, value in info.items()
                })
            with open(filename, "w") as fout:
                json.dump(serializable_pulse, fout, indent=2, ensure_ascii=False)
            print(f"Saved peak detection info: {filename}")
        except Exception as e:
            print(f"Error saving peak info for rx {rx+1}: {e}")
        # peak_info_save = []
        # for info in pulse_info:
        #     peak_info_save.append({
        #         'Peak time (envelope) [ns]': info['peak_time'],
        #         'Peak amplitude (envelope)': info['peak_amplitude'],
        #         'Distinguishable': info['distinguishable'],
        #         'Max amplitude': info['max_amplitude'],
        #         'Max time [ns]': info['max_time'],
        #         'FWHM [ns]': info['width'],
        #     })
        # np.savetxt(filename, peak_info_save, delimiter=' ', fmt='%s')
        
        print(f"\nPeak information saved to: {filename}")
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        print("Please check your input file and try again.")

