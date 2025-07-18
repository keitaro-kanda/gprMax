import argparse
import json
import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button

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
            return time, data, component_to_plot, dt

    except Exception as e:
        print(f"Error reading file {filepath}: {e}", file=sys.stderr)
        return None, None, None, None

class AscanViewer:
    def __init__(self, file_list):
        self.file_list = file_list
        self.current_index = 0
        self.current_data = None
        self.current_dt = None
        self.peak_scatter = None
        self.twt_lines = []
        
        self.fig = plt.figure(figsize=(12, 9))
        plt.subplots_adjust(bottom=0.2)
        self.ax = self.fig.add_subplot(1, 1, 1)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.setup_widgets()
        self.update_plot()

    def setup_widgets(self):
        # Zoom controls
        ax_start = plt.axes([0.12, 0.05, 0.1, 0.04])
        ax_end = plt.axes([0.25, 0.05, 0.1, 0.04])
        ax_zoom_btn = plt.axes([0.37, 0.05, 0.1, 0.04])
        self.text_box_start = TextBox(ax_start, 'Start (ns)', initial='')
        self.text_box_end = TextBox(ax_end, 'End (ns)', initial='')
        self.zoom_button = Button(ax_zoom_btn, 'Apply Zoom')
        self.zoom_button.on_clicked(self.zoom)

        # Peak detection button
        ax_peak_btn = plt.axes([0.50, 0.05, 0.15, 0.04])
        self.peak_button = Button(ax_peak_btn, 'Detect Peaks')
        self.peak_button.on_clicked(self.run_peak_detection)

        # TWT estimation button
        ax_twt_btn = plt.axes([0.68, 0.05, 0.2, 0.04])
        self.twt_button = Button(ax_twt_btn, 'Show Estimated TWT')
        self.twt_button.on_clicked(self.run_twt_estimation)

    def clear_overlays(self):
        if self.peak_scatter:
            self.peak_scatter.remove()
            self.peak_scatter = None
        for line in self.twt_lines:
            line.remove()
        self.twt_lines = []

    def update_plot(self, reset_zoom=True):
        self.clear_overlays()
        self.ax.clear()
        filepath = self.file_list[self.current_index]
        time, data, component, dt = read_ascan_data(filepath)

        if time is not None and data is not None:
            self.current_time = time
            self.current_data = data
            self.current_dt = dt
            self.ax.plot(time, data, label='A-scan')
            if reset_zoom:
                self.ax.set_xlim(time[0], time[-1])
                self.text_box_start.set_val('')
                self.text_box_end.set_val('')
        else:
            self.current_time = None
            self.current_data = None
            self.current_dt = None
            self.ax.text(0.5, 0.5, f"Could not load data from\n{filepath}", ha='center')

        self.ax.set_title(f"File {self.current_index + 1}/{len(self.file_list)}: {os.path.basename(filepath)}\nComponent: {component or 'N/A'}")
        self.ax.set_xlabel("Time [ns]")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)
        self.ax.legend()
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'right':
            self.current_index = (self.current_index + 1) % len(self.file_list)
            self.update_plot()
        elif event.key == 'left':
            self.current_index = (self.current_index - 1 + len(self.file_list)) % len(self.file_list)
            self.update_plot()

    def zoom(self, event):
        try:
            start_time = float(self.text_box_start.text)
            end_time = float(self.text_box_end.text)
            if start_time < end_time:
                self.ax.set_xlim(start_time, end_time)
                self.fig.canvas.draw()
        except ValueError:
            print("Invalid time format. Please enter numbers only.", file=sys.stderr)

    def run_peak_detection(self, event):
        if self.current_data is None or self.current_dt is None:
            print("No data loaded to detect peaks on.", file=sys.stderr)
            return

        self.clear_overlays()
        pulse_info = detect_peaks(self.current_data, self.current_dt)
        peak_times = [info['max_time'] for info in pulse_info if info['distinguishable']]
        peak_amps = [info['max_amplitude'] for info in pulse_info if info['distinguishable']]

        if peak_times:
            self.peak_scatter = self.ax.scatter(peak_times, peak_amps, c='r', marker='x', label='Detected Peaks', zorder=5)
            self.ax.legend()
            self.fig.canvas.draw()
            print(f"Detected {len(peak_times)} distinguishable peaks.")
        else:
            print("No distinguishable peaks found.")

    def run_twt_estimation(self, event):
        self.clear_overlays()
        current_file_path = self.file_list[self.current_index]
        model_path = os.path.join(os.path.dirname(current_file_path), 'model.json')

        if not os.path.exists(model_path):
            print(f"Error: model.json not found in the same directory as the .out file.", file=sys.stderr)
            return

        tw_travel_time, boundary_names = calculate_TWT(model_path)

        if tw_travel_time:
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
            for i, (twt, name) in enumerate(zip(tw_travel_time, boundary_names)):
                line = self.ax.axvline(twt, linestyle='--', color=colors[i % len(colors)], label=f'{name}: {twt:.2f} ns')
                self.twt_lines.append(line)
            self.ax.legend()
            self.fig.canvas.draw()
            print(f"Displayed {len(tw_travel_time)} estimated TWTs.")
        else:
            print("Could not calculate TWTs.")

def main():
    parser = argparse.ArgumentParser(description="GUI tool to check and analyze A-scan files.")
    parser.add_argument("json_file", help="Path to the JSON file containing a list of .out file paths.")
    args = parser.parse_args()

    try:
        with open(args.json_file, 'r') as f:
            config = json.load(f)
            file_list = config.get("ascan_files", [])
    except FileNotFoundError:
        print(f"Error: JSON file not found at {args.json_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.json_file}", file=sys.stderr)
        sys.exit(1)

    if not file_list:
        print("Error: The 'ascan_files' key in the JSON is empty or missing.", file=sys.stderr)
        sys.exit(1)

    viewer = AscanViewer(file_list)
    plt.show()

if __name__ == "__main__":
    main()
