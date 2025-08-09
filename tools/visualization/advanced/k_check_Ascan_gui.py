import json
import sys
import os
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, CheckButtons, RadioButtons
from scipy.signal import hilbert

# Add project root to Python path for VS Code execution
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the refactored functions
from tools.analysis.k_detect_peak import detect_peaks
from tools.visualization.analysis.k_plot_TWT_estimation import calculate_TWT

def read_ascan_data(filepath):
    """
    Reads A-scan data from a gprMax .out (HDF5) file.
    """
    try:
        with h5py.File(filepath, 'r') as f:
            if f.attrs['nrx'] == 0:
                return None, None, None, None

            rx_path = '/rxs/rx1/'
            if not rx_path in f:
                return None, None, None, None

            available_outputs = list(f[rx_path].keys())
            component_to_plot = next((comp for comp in ['Ez', 'Ey', 'Ex'] if comp in available_outputs), None)

            if not component_to_plot:
                return None, None, None, None

            dt = f.attrs['dt']
            iterations = f.attrs['Iterations']
            time = np.linspace(0, (iterations - 1) * dt, num=iterations) / 1e-9  # in ns
            data = f[rx_path + component_to_plot][:]
            data_norm = data / np.max(np.abs(data))  # Normalize data
            return time, data_norm, component_to_plot, dt

    except Exception as e:
        print(f"Error reading file {filepath}: {e}", file=sys.stderr)
        return None, None, None, None

def load_zoom_settings(json_dir):
    """
    Load zoom_settings.json from the same directory as the input JSON file.
    """
    zoom_settings_path = os.path.join(json_dir, 'zoom_settings.json')
    print(f"[DEBUG] Looking for zoom_settings.json at: {zoom_settings_path}")
    
    try:
        if os.path.exists(zoom_settings_path):
            print(f"[DEBUG] zoom_settings.json found, loading...")
            with open(zoom_settings_path, 'r') as f:
                settings = json.load(f)
                print(settings)
                print(f"[DEBUG] Raw zoom_settings content: {settings}")
                
                zoom_config = {
                    'time_start': settings.get('x_min'),
                    'time_end': settings.get('x_max'),
                    'intensity_min': settings.get('y_min'),
                    'intensity_max': settings.get('y_max')
                }
                print(f"[DEBUG] Processed zoom_settings: {zoom_config}")
                return zoom_config
        else:
            print(f"[DEBUG] zoom_settings.json not found at {zoom_settings_path}")
    except Exception as e:
        print(f"Warning: Could not load zoom_settings.json: {e}", file=sys.stderr)
    return None

def load_config_json(out_filepath):
    """
    Load config.json corresponding to the .out file.
    """
    base_dir = os.path.dirname(out_filepath)
    base_name = os.path.splitext(os.path.basename(out_filepath))[0]
    config_path = os.path.join(base_dir, f"{base_name}_config.json")
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load config file {config_path}: {e}", file=sys.stderr)
    return None

def load_existing_labels(output_dir, label_type):
    """
    Load existing label JSON files from output directory.
    """
    label_file = os.path.join(output_dir, f'{label_type}_echo_labels.json')
    try:
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load existing labels from {label_file}: {e}", file=sys.stderr)
    return {}

def create_backup(file_path):
    """
    Create a backup of the existing file with .bak extension.
    """
    if os.path.exists(file_path):
        backup_path = f"{file_path}.bak"
        try:
            import shutil
            shutil.copy2(file_path, backup_path)
            return backup_path
        except Exception as e:
            print(f"Warning: Could not create backup for {file_path}: {e}", file=sys.stderr)
    return None

def calculate_envelope(data):
    """
    Calculate the envelope of A-scan data using Hilbert transform.
    """
    try:
        analytic_signal = hilbert(data)
        envelope = np.abs(analytic_signal)
        return envelope
    except Exception as e:
        print(f"Warning: Could not calculate envelope: {e}", file=sys.stderr)
        return None

class AscanViewer:
    def __init__(self, file_data, json_dir, output_base_dir, waveform_type):
        # file_data can be either a list or dict
        if isinstance(file_data, dict):
            self.file_keys = list(file_data.keys())
            self.file_list = list(file_data.values())
            self.has_keys = True
        else:
            self.file_keys = None
            self.file_list = file_data
            self.has_keys = False
        
        self.current_index = 0
        self.current_data = None
        self.current_dt = None
        self.current_config = None
        self.peak_scatter = None
        self.twt_lines = []
        
        # Current peaks and TWTs for persistence
        self.current_peaks = None
        self.current_twts = None
        
        # Load zoom settings
        self.zoom_settings = load_zoom_settings(json_dir)
        self.auto_zoom_enabled = False
        
        # Zoom state preservation
        self.saved_xlim = None
        self.saved_ylim = None
        self.preserve_zoom = False
        
        # Envelope display state
        self.show_envelope = False
        self.envelope_line = None
        
        # Display mode persistence
        self.show_peaks = False
        self.show_twts = False
        
        # Waveform type
        self.waveform_type = waveform_type
        
        # Output directories for labeling results
        self.output_peak_dir = os.path.join(output_base_dir, 'result_use_peak')
        self.output_twt_dir = os.path.join(output_base_dir, 'result_use_TWT')
        
        # Labeling state for current file
        self.top_echo_label = 1
        self.bottom_echo_label = 1
        
        # Load existing labeling results
        self.labeling_results = {
            'top': load_existing_labels(self.output_peak_dir, 'top'),
            'bottom': load_existing_labels(self.output_peak_dir, 'bottom')
        }
        
        # Merge with TWT directory labels if different
        if self.output_twt_dir != self.output_peak_dir:
            twt_top_labels = load_existing_labels(self.output_twt_dir, 'top')
            twt_bottom_labels = load_existing_labels(self.output_twt_dir, 'bottom')
            self.labeling_results['top'].update(twt_top_labels)
            self.labeling_results['bottom'].update(twt_bottom_labels)
        
        # Label statistics
        self.label_stats = self.calculate_label_stats()
        
        self.fig = plt.figure(figsize=(16, 10))
        plt.subplots_adjust(bottom=0.35, right=0.75)
        self.ax = self.fig.add_subplot(1, 1, 1)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.setup_widgets()
        self.update_plot()

    def setup_widgets(self):
        # Time zoom controls
        ax_start = plt.axes([0.05, 0.25, 0.08, 0.04])
        ax_end = plt.axes([0.15, 0.25, 0.08, 0.04])
        self.text_box_start = TextBox(ax_start, 'Start (ns)', initial='')
        self.text_box_end = TextBox(ax_end, 'End (ns)', initial='')
        
        # Intensity zoom controls
        ax_int_min = plt.axes([0.05, 0.18, 0.08, 0.04])
        ax_int_max = plt.axes([0.15, 0.18, 0.08, 0.04])
        self.text_box_int_min = TextBox(ax_int_min, 'Min Amp', initial='')
        self.text_box_int_max = TextBox(ax_int_max, 'Max Amp', initial='')
        
        # Auto-zoom checkbox
        ax_auto_zoom = plt.axes([0.25, 0.20, 0.12, 0.08])
        self.auto_zoom_check = CheckButtons(ax_auto_zoom, ['Auto Zoom'], [self.auto_zoom_enabled])
        self.auto_zoom_check.on_clicked(self.toggle_auto_zoom)
        
        # Envelope display checkbox
        ax_envelope = plt.axes([0.25, 0.10, 0.12, 0.08])
        self.envelope_check = CheckButtons(ax_envelope, ['Show Envelope'], [self.show_envelope])
        self.envelope_check.on_clicked(self.toggle_envelope)
        
        # Apply zoom button
        ax_zoom_btn = plt.axes([0.40, 0.22, 0.08, 0.04])
        self.zoom_button = Button(ax_zoom_btn, 'Apply Zoom')
        self.zoom_button.on_clicked(self.zoom)

        # Peak detection button
        ax_peak_btn = plt.axes([0.50, 0.22, 0.10, 0.04])
        self.peak_button = Button(ax_peak_btn, 'Detect Peaks')
        self.peak_button.on_clicked(self.run_peak_detection)

        # TWT estimation button
        ax_twt_btn = plt.axes([0.62, 0.22, 0.12, 0.04])
        self.twt_button = Button(ax_twt_btn, 'Show Estimated TWT')
        self.twt_button.on_clicked(self.run_twt_estimation)
        
        # Top Echo labeling
        ax_top_label = plt.axes([0.78, 0.22, 0.15, 0.12])
        self.top_radio = RadioButtons(ax_top_label, ['Top Label 1', 'Top Label 2', 'Top Label 3'])
        self.top_radio.on_clicked(self.set_top_label)
        
        # Bottom Echo labeling
        ax_bottom_label = plt.axes([0.78, 0.06, 0.15, 0.12])
        self.bottom_radio = RadioButtons(ax_bottom_label, ['Bottom Label 1', 'Bottom Label 2', 'Bottom Label 3'])
        self.bottom_radio.on_clicked(self.set_bottom_label)
        
        # Save labels button
        ax_save_btn = plt.axes([0.05, 0.06, 0.12, 0.04])
        self.save_button = Button(ax_save_btn, 'Save Labels')
        self.save_button.on_clicked(self.save_labels)
        
        # Reset zoom button
        ax_reset_btn = plt.axes([0.20, 0.06, 0.10, 0.04])
        self.reset_button = Button(ax_reset_btn, 'Reset Zoom')
        self.reset_button.on_clicked(self.reset_zoom)

    def clear_overlays(self):
        if self.peak_scatter:
            self.peak_scatter.remove()
            self.peak_scatter = None
        for line in self.twt_lines:
            line.remove()
        self.twt_lines = []
        if self.envelope_line:
            self.envelope_line.remove()
            self.envelope_line = None

    def save_current_zoom(self):
        """Save current zoom state"""
        if self.current_time is not None:
            self.saved_xlim = self.ax.get_xlim()
            self.saved_ylim = self.ax.get_ylim()
            # Update text boxes with current values
            self.text_box_start.set_val(str(self.saved_xlim[0]))
            self.text_box_end.set_val(str(self.saved_xlim[1]))
            self.text_box_int_min.set_val(str(self.saved_ylim[0]))
            self.text_box_int_max.set_val(str(self.saved_ylim[1]))
            self.preserve_zoom = True
    
    def restore_zoom(self):
        """Restore saved zoom state"""
        if self.preserve_zoom and self.saved_xlim and self.saved_ylim:
            self.ax.set_xlim(self.saved_xlim)
            self.ax.set_ylim(self.saved_ylim)
            # Update text boxes
            self.text_box_start.set_val(str(self.saved_xlim[0]))
            self.text_box_end.set_val(str(self.saved_xlim[1]))
            self.text_box_int_min.set_val(str(self.saved_ylim[0]))
            self.text_box_int_max.set_val(str(self.saved_ylim[1]))

    def update_plot(self, reset_zoom=True):
        self.clear_overlays()
        self.ax.clear()
        filepath = self.file_list[self.current_index]
        time, data, component, dt = read_ascan_data(filepath)
        
        # Load config for current file
        self.current_config = load_config_json(filepath)

        if time is not None and data is not None:
            self.current_time = time
            self.current_data = data
            self.current_dt = dt
            self.ax.plot(time, data, label='A-scan', color='black', linewidth=1.0)
            
            # Plot envelope if enabled
            if self.show_envelope:
                envelope = calculate_envelope(data)
                if envelope is not None:
                    self.envelope_line = self.ax.plot(time, envelope, label='Envelope', 
                                                    color='gray', linewidth=1.5, 
                                                    linestyle='--', alpha=0.8)[0]
            
            if reset_zoom:
                if self.auto_zoom_enabled and self.zoom_settings:
                    # Apply auto zoom settings
                    print(f"[DEBUG] update_plot: Applying auto zoom - time_start:{self.zoom_settings['time_start']}, time_end:{self.zoom_settings['time_end']}")
                    print(f"[DEBUG] update_plot: Applying auto zoom - intensity_min:{self.zoom_settings['intensity_min']}, intensity_max:{self.zoom_settings['intensity_max']}")
                    self.ax.set_xlim(self.zoom_settings['time_start'], self.zoom_settings['time_end'])
                    self.ax.set_ylim(self.zoom_settings['intensity_min'], self.zoom_settings['intensity_max'])
                    self.text_box_start.set_val(str(self.zoom_settings['time_start']))
                    self.text_box_end.set_val(str(self.zoom_settings['time_end']))
                    self.text_box_int_min.set_val(str(self.zoom_settings['intensity_min']))
                    self.text_box_int_max.set_val(str(self.zoom_settings['intensity_max']))
                    print(f"[DEBUG] update_plot: Auto zoom applied successfully")
                else:
                    print(f"[DEBUG] update_plot: Using default zoom (auto_zoom_enabled: {self.auto_zoom_enabled}, zoom_settings: {bool(self.zoom_settings)})")
                    self.ax.set_xlim(time[0], time[-1])
                    self.text_box_start.set_val('')
                    self.text_box_end.set_val('')
                    self.text_box_int_min.set_val('')
                    self.text_box_int_max.set_val('')
            else:
                # Restore saved zoom if available
                print(f"[DEBUG] update_plot: Restoring saved zoom")
                self.restore_zoom()
            
            # Apply persistent display modes after plotting data
            self.apply_persistent_displays()
        else:
            self.current_time = None
            self.current_data = None
            self.current_dt = None
            self.current_config = None
            self.current_peaks = None
            self.current_twts = None
            self.ax.text(0.5, 0.5, f"Could not load data from\n{filepath}", ha='center')

        # Update GUI with current file's label state
        self.update_label_gui()
        
        # Create title with key name if available
        if self.has_keys and self.file_keys:
            key_name = self.file_keys[self.current_index]
            title = f"File {self.current_index + 1}/{len(self.file_list)}: {key_name}\nPath: {os.path.basename(filepath)}\nComponent: {component or 'N/A'}"
        else:
            title = f"File {self.current_index + 1}/{len(self.file_list)}: {os.path.basename(filepath)}\nComponent: {component or 'N/A'}"
        
        # Add config info if available
        if self.current_config:
            title += f"\nConfig: Loaded"
        
        # Add label info if available
        current_key = self.get_current_file_key()
        if current_key in self.labeling_results['top'] or current_key in self.labeling_results['bottom']:
            top_label = self.labeling_results['top'].get(current_key, [0, 0, 0])[2]
            bottom_label = self.labeling_results['bottom'].get(current_key, [0, 0, 0])[2]
            title += f"\nLabels: Top={top_label}, Bottom={bottom_label}"
        
        self.ax.set_title(title)
        self.ax.set_xlabel("Time [ns]")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)
        self.ax.legend()
        plt.tight_layout(rect=[0, 0.35, 0.75, 1])
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'right':
            # Save current zoom before switching
            self.save_current_zoom()
            self.current_index = (self.current_index + 1) % len(self.file_list)
            self.update_plot(reset_zoom=False)
        elif event.key == 'left':
            # Save current zoom before switching
            self.save_current_zoom()
            self.current_index = (self.current_index - 1 + len(self.file_list)) % len(self.file_list)
            self.update_plot(reset_zoom=False)

    def toggle_auto_zoom(self, label):
        self.auto_zoom_enabled = not self.auto_zoom_enabled
        print(f"[DEBUG] Auto zoom toggled to: {self.auto_zoom_enabled}")
        print(f"[DEBUG] Current zoom_settings: {self.zoom_settings}")
        
        if self.auto_zoom_enabled and self.zoom_settings:
            print(f"[DEBUG] Applying auto zoom settings...")
            self.text_box_start.set_val(str(self.zoom_settings['time_start']))
            self.text_box_end.set_val(str(self.zoom_settings['time_end']))
            self.text_box_int_min.set_val(str(self.zoom_settings['intensity_min']))
            self.text_box_int_max.set_val(str(self.zoom_settings['intensity_max']))
            print(f"[DEBUG] Text boxes updated, calling zoom...")
            self.zoom(None)
        elif self.auto_zoom_enabled and not self.zoom_settings:
            print(f"[DEBUG] Auto zoom enabled but no zoom_settings available")
        else:
            print(f"[DEBUG] Auto zoom disabled")
    
    def toggle_envelope(self, label):
        """Toggle envelope display on/off"""
        self.show_envelope = not self.show_envelope
        
        # Clear overlays and redraw with all active displays
        self.clear_overlays()
        
        # Re-plot envelope if enabled
        if self.show_envelope and self.current_data is not None:
            envelope = calculate_envelope(self.current_data)
            if envelope is not None:
                self.envelope_line = self.ax.plot(self.current_time, envelope, label='Envelope', 
                                                color='gray', linewidth=1.5, 
                                                linestyle='--', alpha=0.8)[0]
        
        # Re-plot peaks if enabled
        if self.show_peaks and self.current_peaks is not None:
            peak_times, peak_amps = self.current_peaks
            if peak_times:
                self.peak_scatter = self.ax.scatter(peak_times, peak_amps, c='r', marker='x', 
                                                  label='Detected Peaks', zorder=5)
        
        # Re-plot TWTs if enabled
        if self.show_twts and self.current_twts is not None:
            tw_travel_time, boundary_names = self.current_twts
            if tw_travel_time:
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
                for i, (twt, name) in enumerate(zip(tw_travel_time, boundary_names)):
                    line = self.ax.axvline(twt, linestyle='--', color=colors[i % len(colors)], 
                                         label=f'{name}: {twt:.2f} ns')
                    self.twt_lines.append(line)
        
        # Update legend if any displays are active
        if self.show_peaks or self.show_twts or self.show_envelope:
            self.ax.legend()
        
        self.fig.canvas.draw()
    
    def calculate_label_stats(self):
        """Calculate statistics for labeled data"""
        stats = {
            'total_files': len(self.file_list),
            'labeled_files': set(),
            'top_labels': {1: 0, 2: 0, 3: 0},
            'bottom_labels': {1: 0, 2: 0, 3: 0}
        }
        
        for key, data in self.labeling_results['top'].items():
            stats['labeled_files'].add(key)
            if len(data) >= 3:
                label = data[2]
                if label in stats['top_labels']:
                    stats['top_labels'][label] += 1
        
        for key, data in self.labeling_results['bottom'].items():
            stats['labeled_files'].add(key)
            if len(data) >= 3:
                label = data[2]
                if label in stats['bottom_labels']:
                    stats['bottom_labels'][label] += 1
        
        stats['labeled_count'] = len(stats['labeled_files'])
        return stats
    
    def update_label_gui(self):
        """Update GUI radio buttons based on current file's existing labels"""
        current_key = self.get_current_file_key()
        
        # Update top label radio button
        if current_key in self.labeling_results['top']:
            top_data = self.labeling_results['top'][current_key]
            if len(top_data) >= 3:
                self.top_echo_label = top_data[2]
                # Update radio button selection (1-based to 0-based index)
                self.top_radio.set_active(self.top_echo_label - 1)
        
        # Update bottom label radio button
        if current_key in self.labeling_results['bottom']:
            bottom_data = self.labeling_results['bottom'][current_key]
            if len(bottom_data) >= 3:
                self.bottom_echo_label = bottom_data[2]
                # Update radio button selection (1-based to 0-based index)
                self.bottom_radio.set_active(self.bottom_echo_label - 1)
    
    def zoom(self, event):
        try:
            # Time zoom
            if self.text_box_start.text and self.text_box_end.text:
                start_time = float(self.text_box_start.text)
                end_time = float(self.text_box_end.text)
                if start_time < end_time:
                    self.ax.set_xlim(start_time, end_time)
            
            # Intensity zoom
            if self.text_box_int_min.text and self.text_box_int_max.text:
                min_intensity = float(self.text_box_int_min.text)
                max_intensity = float(self.text_box_int_max.text)
                if min_intensity < max_intensity:
                    self.ax.set_ylim(min_intensity, max_intensity)
            
            self.fig.canvas.draw()
        except ValueError:
            print("Invalid format. Please enter numbers only.", file=sys.stderr)
    
    def reset_zoom(self, event):
        """Reset zoom to show full data range"""
        if self.current_time is not None and self.current_data is not None:
            # Reset to full data range
            self.ax.set_xlim(self.current_time[0], self.current_time[-1])
            data_min = np.min(self.current_data)
            data_max = np.max(self.current_data)
            margin = (data_max - data_min) * 0.1
            self.ax.set_ylim(data_min - margin, data_max + margin)
            
            # Clear text boxes
            self.text_box_start.set_val('')
            self.text_box_end.set_val('')
            self.text_box_int_min.set_val('')
            self.text_box_int_max.set_val('')
            
            # Clear saved zoom state
            self.saved_xlim = None
            self.saved_ylim = None
            self.preserve_zoom = False
            
            self.fig.canvas.draw()
    
    def apply_persistent_displays(self):
        """Apply persistent display modes (peaks, TWTs) after plot update"""
        # Auto-calculate peaks if peak display mode is enabled
        if self.show_peaks and self.current_data is not None and self.current_dt is not None:
            try:
                pulse_info = detect_peaks(self.current_data, self.current_dt)
                peak_times = [info['max_time'] for info in pulse_info if info['distinguishable']]
                peak_amps = [info['max_amplitude'] for info in pulse_info if info['distinguishable']]
                self.current_peaks = (peak_times, peak_amps)
                
                if peak_times:
                    self.peak_scatter = self.ax.scatter(peak_times, peak_amps, c='r', marker='x', 
                                                      label='Detected Peaks', zorder=5)
                    print(f"Auto-detected {len(peak_times)} distinguishable peaks.")
                else:
                    print("No distinguishable peaks found in current data.")
            except Exception as e:
                print(f"Error in auto peak detection: {e}", file=sys.stderr)
                self.show_peaks = False
                self.current_peaks = None
        
        # Auto-calculate TWTs if TWT display mode is enabled
        if self.show_twts:
            try:
                current_file_path = self.file_list[self.current_index]
                #model_path = os.path.join(os.path.dirname(current_file_path), 'model.json')
                model_path = current_file_path.replace('.out', '_config.json')

                if os.path.exists(model_path):
                    tw_travel_time, boundary_names = calculate_TWT(model_path)
                    self.current_twts = (tw_travel_time, boundary_names)

                    if tw_travel_time:
                        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
                        for i, (twt, name) in enumerate(zip(tw_travel_time, boundary_names)):
                            # Calculate TWT start and end for each individual TWT value
                            if self.waveform_type == '1':
                                TWT_start = twt - 1.544
                                TWT_end = twt + 2.327
                            else:
                                TWT_start = twt - 1.255
                                TWT_end = twt + 1.530
                            
                            region = self.ax.axvspan(TWT_start, TWT_end, color=colors[i % len(colors)], alpha=0.3)
                            self.ax.add_patch(region)
                            line = self.ax.axvline(twt, linestyle='--', color=colors[i % len(colors)])
                            self.twt_lines.append(line)
                        print(f"Auto-calculated {len(tw_travel_time)} TWTs.")
                    else:
                        print("Could not calculate TWTs for current data.")
                else:
                    print(f"Warning: model.json not found for TWT calculation: {model_path}", file=sys.stderr)
                    self.show_twts = False
                    self.current_twts = None
            except Exception as e:
                print(f"Error in auto TWT calculation: {e}", file=sys.stderr)
                self.show_twts = False
                self.current_twts = None
        
        # Update legend if any displays are active
        if self.show_peaks or self.show_twts or self.show_envelope:
            self.ax.legend()
    
    def set_top_label(self, label):
        if 'Label 1' in label:
            self.top_echo_label = 1
        elif 'Label 2' in label:
            self.top_echo_label = 2
        elif 'Label 3' in label:
            self.top_echo_label = 3
    
    def set_bottom_label(self, label):
        if 'Label 1' in label:
            self.bottom_echo_label = 1
        elif 'Label 2' in label:
            self.bottom_echo_label = 2
        elif 'Label 3' in label:
            self.bottom_echo_label = 3
    
    def get_current_file_key(self):
        if self.has_keys and self.file_keys:
            return self.file_keys[self.current_index]
        else:
            filepath = self.file_list[self.current_index]
            return os.path.splitext(os.path.basename(filepath))[0]
    
    def get_geometry_from_filename(self):
        """Extract height and width from filename pattern like 'h0.3_w0.6'"""
        current_key = self.get_current_file_key()
        
        try:
            # Pattern to match h{number}_w{number} format
            pattern = r'h([0-9]*\.?[0-9]+)_w([0-9]*\.?[0-9]+)'
            match = re.search(pattern, current_key)
            
            if match:
                height = float(match.group(1))
                width = float(match.group(2))
                print(f"Extracted geometry from '{current_key}': height={height}, width={width}")
                return height, width
            else:
                print(f"Warning: Could not extract geometry from filename '{current_key}', using defaults")
                return 0.0, 0.0
        except (ValueError, AttributeError) as e:
            print(f"Error parsing geometry from filename '{current_key}': {e}", file=sys.stderr)
            return 0.0, 0.0
    
    def get_geometry_from_config(self):
        """Legacy method - now redirects to filename-based extraction"""
        return self.get_geometry_from_filename()
    
    def save_labels(self, event):
        current_key = self.get_current_file_key()
        height, width = self.get_geometry_from_filename()
        
        # Check if labels have changed
        old_top_label = None
        old_bottom_label = None
        if current_key in self.labeling_results['top']:
            old_top_label = self.labeling_results['top'][current_key][2]
        if current_key in self.labeling_results['bottom']:
            old_bottom_label = self.labeling_results['bottom'][current_key][2]
        
        # Update labeling results
        self.labeling_results['top'][current_key] = [height, width, self.top_echo_label]
        self.labeling_results['bottom'][current_key] = [height, width, self.bottom_echo_label]
        
        # Create output directories if they don't exist
        os.makedirs(self.output_peak_dir, exist_ok=True)
        os.makedirs(self.output_twt_dir, exist_ok=True)
        
        # Save to JSON files with backup and incremental update
        try:
            # Define file paths
            top_peak_path = os.path.join(self.output_peak_dir, 'top_echo_labels.json')
            top_twt_path = os.path.join(self.output_twt_dir, 'top_echo_labels.json')
            bottom_peak_path = os.path.join(self.output_peak_dir, 'bottom_echo_labels.json')
            bottom_twt_path = os.path.join(self.output_twt_dir, 'bottom_echo_labels.json')
            
            # Create backups
            create_backup(top_peak_path)
            create_backup(top_twt_path)
            create_backup(bottom_peak_path)
            create_backup(bottom_twt_path)
            
            # Load existing data and merge
            existing_top_peak = load_existing_labels(self.output_peak_dir, 'top')
            existing_top_twt = load_existing_labels(self.output_twt_dir, 'top')
            existing_bottom_peak = load_existing_labels(self.output_peak_dir, 'bottom')
            existing_bottom_twt = load_existing_labels(self.output_twt_dir, 'bottom')
            
            # Update only current file's labels
            existing_top_peak[current_key] = [height, width, self.top_echo_label]
            existing_top_twt[current_key] = [height, width, self.top_echo_label]
            existing_bottom_peak[current_key] = [height, width, self.bottom_echo_label]
            existing_bottom_twt[current_key] = [height, width, self.bottom_echo_label]
            
            # Save updated data
            with open(top_peak_path, 'w') as f:
                json.dump(existing_top_peak, f, indent=2)
            with open(top_twt_path, 'w') as f:
                json.dump(existing_top_twt, f, indent=2)
            with open(bottom_peak_path, 'w') as f:
                json.dump(existing_bottom_peak, f, indent=2)
            with open(bottom_twt_path, 'w') as f:
                json.dump(existing_bottom_twt, f, indent=2)
            
            # Update statistics
            self.label_stats = self.calculate_label_stats()
            
            # Show save status
            if old_top_label is not None or old_bottom_label is not None:
                print(f"Labels updated for {current_key}: Top={old_top_label}→{self.top_echo_label}, Bottom={old_bottom_label}→{self.bottom_echo_label}")
            else:
                print(f"Labels saved for {current_key}: Top={self.top_echo_label}, Bottom={self.bottom_echo_label}")
            
            # Show progress
            print(f"Progress: {self.label_stats['labeled_count']}/{self.label_stats['total_files']} files labeled")
            
        except Exception as e:
            print(f"Error saving labels: {e}", file=sys.stderr)

    def run_peak_detection(self, event):
        """Manually trigger peak detection and enable persistent display mode"""
        if self.current_data is None or self.current_dt is None:
            print("No data loaded to detect peaks on.", file=sys.stderr)
            return

        # Enable peak display mode
        self.show_peaks = True
        
        # Clear and redraw with all active displays (auto-calculation will occur)
        self.clear_overlays()
        self.apply_persistent_displays()
        
        # Re-plot envelope if enabled
        if self.show_envelope and self.current_data is not None:
            envelope = calculate_envelope(self.current_data)
            if envelope is not None:
                self.envelope_line = self.ax.plot(self.current_time, envelope, label='Envelope', 
                                                color='gray', linewidth=1.5, 
                                                linestyle='--', alpha=0.8)[0]
        
        # Update legend and redraw
        if self.show_peaks or self.show_twts or self.show_envelope:
            self.ax.legend()
        self.fig.canvas.draw()

    def run_twt_estimation(self, event):
        """Manually trigger TWT estimation and enable persistent display mode"""
        current_file_path = self.file_list[self.current_index]
        #model_path = os.path.join(os.path.dirname(current_file_path), 'model.json')
        model_path = current_file_path.replace('.out', '_config.json')

        if not os.path.exists(model_path):
            print(f"Error: model.json not found in the same directory as the .out file.", file=sys.stderr)
            return

        # Enable TWT display mode
        self.show_twts = True
        
        # Clear and redraw with all active displays (auto-calculation will occur)
        self.clear_overlays()
        self.apply_persistent_displays()
        
        # Re-plot envelope if enabled
        if self.show_envelope and self.current_data is not None:
            envelope = calculate_envelope(self.current_data)
            if envelope is not None:
                self.envelope_line = self.ax.plot(self.current_time, envelope, label='Envelope', 
                                                color='gray', linewidth=1.5, 
                                                linestyle='--', alpha=0.8)[0]
        
        # Update legend and redraw
        if self.show_peaks or self.show_twts or self.show_envelope:
            self.ax.legend()
        self.fig.canvas.draw()

def main():
    print("=== A-scan GUI Viewer ===")
    
    # Get JSON file path from user input
    while True:
        json_file = input("output_file_paths.jsonファイルのパスを入力してください (終了する場合は 'quit' を入力): ").strip()
        
        if json_file.lower() == 'quit':
            print("プログラムを終了します。")
            sys.exit(0)
            
        if not json_file:
            print("パスが入力されていません。再度入力してください。")
            continue
            
        # Convert relative path to absolute path if needed
        if not os.path.isabs(json_file):
            json_file = os.path.abspath(json_file)
            
        try:
            with open(json_file, 'r') as f:
                config = json.load(f)
                
                # Check for new format (direct key-value pairs)
                if "ascan_files" in config:
                    # Old format
                    file_data = config["ascan_files"]
                    if not file_data:
                        print("エラー: JSONファイル内の 'ascan_files' キーが空です。")
                        continue
                else:
                    # New format - assume all key-value pairs are file entries
                    file_data = config
                    if not file_data:
                        print("エラー: JSONファイルが空か、有効なファイルエントリがありません。")
                        continue
            break
        except FileNotFoundError:
            print(f"エラー: JSONファイルが見つかりません: {json_file}")
            print("正しいパスを入力してください。")
            continue
        except json.JSONDecodeError:
            print(f"エラー: JSONファイルの形式が正しくありません: {json_file}")
            continue
        except Exception as e:
            print(f"エラー: ファイル読み込み中にエラーが発生しました: {e}")
            continue
    
    # Get waveform type from user input
    waveform_type = input("波形タイプを選択してください (1: Bipolar, 2: Unipolar): ").strip()
    if waveform_type not in ['1', '2']:
        print("無効な波形タイプです。プログラムを終了します。")
        sys.exit(1)

    # Get directories for configuration and output
    json_dir = os.path.dirname(json_file)
    output_base_dir = json_dir
    
    # Get count for display
    file_count = len(file_data) if isinstance(file_data, dict) else len(file_data)
    print(f"読み込み完了: {file_count} 個のファイルが見つかりました。")
    
    if isinstance(file_data, dict):
        print("JSON形式: キー-ファイルパス形式")
        # Show first few keys as examples
        keys_sample = list(file_data.keys())[:3]
        print(f"例: {', '.join(keys_sample)}{'...' if len(file_data) > 3 else ''}")
    else:
        print("JSON形式: ファイルパス配列形式")
    print("操作方法:")
    print("- 左右矢印キー: ファイル切り替え")
    print("- Auto Zoom チェックボックス: zoom_settings.json使用の切り替え")
    print("- Show Envelope チェックボックス: A-scanのenvelope表示切り替え")
    print("- Apply Zoom ボタン: 時間・振幅範囲指定でズーム")
    print("- Reset Zoom ボタン: ズーム範囲をリセット")
    print("- Detect Peaks ボタン: ピーク検出")
    print("- Show Estimated TWT ボタン: 推定TWT表示")
    print("- Top/Bottom ラジオボタン: エコー特性ラベル選択")
    print("- Save Labels ボタン: ラベル結果をJSON保存")
    
    viewer = AscanViewer(file_data, json_dir, output_base_dir, waveform_type)
    
    print("\n既存ラベルの読み込み完了")
    if viewer.label_stats['labeled_count'] > 0:
        print(f"ラベル統計: {viewer.label_stats['labeled_count']}/{viewer.label_stats['total_files']} ファイルがラベル付け済み")
        print(f"Topラベル: {dict(viewer.label_stats['top_labels'])}")
        print(f"Bottomラベル: {dict(viewer.label_stats['bottom_labels'])}")
    else:
        print("ラベル付け済みファイルはありません")
    
    plt.show()

if __name__ == "__main__":
    main()
