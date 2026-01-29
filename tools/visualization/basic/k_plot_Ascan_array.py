# Copyright (C) 2015-2023: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from gprMax.exceptions import CmdInputError


def mpl_plot(filename, use_closeup=False, closeup_x_start=None, closeup_x_end=None, closeup_y_range=None):
    """Plots Ez field from all receiver points in the given output file.
    All receivers are plotted on a single figure with different colors.
    This function is designed for files with multiple receivers.

    Args:
        filename (string): Filename (including path) of output file.
        use_closeup (boolean): Enable closeup view.
        closeup_x_start (float): Closeup x-axis start time [ns].
        closeup_x_end (float): Closeup x-axis end time [ns].
        closeup_y_range (float): Closeup y-axis range (normalized amplitude).

    Returns:
        plt (object): matplotlib plot object.
    """

    # Open output file and read some attributes
    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']
    dt = f.attrs['dt']
    iterations = f.attrs['Iterations']
    time = np.linspace(0, (iterations - 1) * dt, num=iterations) / 1e-9  # [ns]

    # Check there are any receivers
    if nrx == 0:
        raise CmdInputError('No receivers found in {}'.format(filename))

    # Check for multiple receivers
    if nrx == 1:
        print('Warning: Only 1 receiver found. This tool is designed for multiple receivers.')

    # First pass: read all data and find maximum amplitude
    all_data = []
    max_amplitude = 0.0
    for rx in range(1, nrx + 1):
        path = '/rxs/rx' + str(rx) + '/'
        availableoutputs = list(f[path].keys())

        # Check if Ez is available
        if 'Ez' not in availableoutputs:
            raise CmdInputError('Ez output not available for receiver {}. Available outputs: {}'.format(
                rx, ', '.join(availableoutputs)))

        # Get Ez data
        outputdata = f[path + 'Ez'][:]
        all_data.append(outputdata)

        # Update maximum amplitude
        current_max = np.max(np.abs(outputdata))
        if current_max > max_amplitude:
            max_amplitude = current_max

    # Normalize all data by maximum amplitude
    all_data_normalized = [data / max_amplitude for data in all_data]

    # Calculate offset based on closeup mode
    if use_closeup and closeup_y_range is not None:
        # Closeup mode: offset = closeup_y_range * 0.7
        offset = closeup_y_range * 0.7
    else:
        # Normal mode: offset = 0.7
        offset = 0.7

    # Create single figure for all receivers
    fig, ax = plt.subplots(figsize=(20, 10), facecolor='w', edgecolor='w', tight_layout=True)

    # Get default color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # Plot each receiver with offset
    for rx in range(1, nrx + 1):
        # Get pre-loaded normalized data
        outputdata = all_data_normalized[rx - 1]

        # Calculate offset for this receiver: rx1=0, rx2=-offset, rx3=-2*offset, ...
        rx_offset = -(rx - 1) * offset

        # Calculate envelope using Hilbert transform
        env = np.abs(signal.hilbert(outputdata))

        # Get color for this receiver (cycle through colors if more receivers than colors)
        color = colors[(rx - 1) % len(colors)]

        # Plot Ez data with offset
        ax.plot(time, outputdata + rx_offset, color=color, lw=2, label='rx{}'.format(rx))

        # Plot envelope with same color but dashed line, with offset
        ax.plot(time, env + rx_offset, color=color, lw=2, linestyle='--', alpha=0.5)

    # Configure plot
    ax.set_xlabel('Time [ns]', fontsize=28)
    ax.set_ylabel('Normalized Ez', fontsize=28)

    # Set axis limits based on closeup option
    if use_closeup and closeup_x_start is not None and closeup_x_end is not None and closeup_y_range is not None:
        ax.set_xlim([closeup_x_start, closeup_x_end])
        # y-axis range: upper = closeup_y_range, lower = -(nrx-1)*offset - closeup_y_range
        y_upper = closeup_y_range
        y_lower = -(nrx - 1) * offset - closeup_y_range
        ax.set_ylim([y_lower, y_upper])
    else:
        ax.set_xlim([0, np.amax(time)])

    ax.grid(which='both', axis='both', linestyle='-.')
    ax.minorticks_on()
    ax.tick_params(labelsize=24)
    ax.legend(fontsize=24, bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1)
    plt.tight_layout()

    # Save PNG file
    if use_closeup:
        output_filename = os.path.splitext(os.path.abspath(filename))[0] + \
            '_multi_rx_closeup_x{}_{}.png'.format(closeup_x_start, closeup_x_end)
    else:
        output_filename = os.path.splitext(os.path.abspath(filename))[0] + '_multi_rx.png'
    fig.savefig(output_filename, dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
    print('Saved plot to: {}'.format(output_filename))

    f.close()

    return plt


if __name__ == "__main__":

    print("Plots Ez field from all receiver points in the given output file.")
    print("All receivers are plotted on a single figure with different colors.")
    print("This tool is designed for files with multiple receivers.")

    # Get output file path through interactive input
    outputfile = input("Enter the path to the output file: ").strip()
    if not os.path.exists(outputfile):
        raise CmdInputError('Output file {} does not exist'.format(outputfile))

    # Get closeup option
    closeup_input = input("Plot close up of time domain signal? (y/n) [default: n]: ").strip().lower()
    use_closeup = closeup_input == 'y'

    if use_closeup:
        closeup_x_start = float(input("Enter close up x-axis start [ns]: ").strip())
        closeup_x_end = float(input("Enter close up x-axis end [ns]: ").strip())
        closeup_y_range = np.abs(float(input("Enter close up y-axis range: ").strip()))
    else:
        closeup_x_start = None
        closeup_x_end = None
        closeup_y_range = None

    plthandle = mpl_plot(outputfile, use_closeup=use_closeup,
                         closeup_x_start=closeup_x_start, closeup_x_end=closeup_x_end,
                         closeup_y_range=closeup_y_range)
    plthandle.show()
