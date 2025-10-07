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
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import markers
from numpy import size

from gprMax.exceptions import CmdInputError
from gprMax.receivers import Rx
from gprMax.utilities import fft_power
from scipy import signal


def mpl_plot(filename, outputs=Rx.defaultoutputs, fft=False):
    """Plots electric and magnetic fields and currents from all receiver points in the given output file. Each receiver point is plotted in a new figure window.

    Args:
        filename (string): Filename (including path) of output file.
        outputs (list): List of field/current components to plot.
        fft (boolean): Plot FFT switch.

    Returns:
        plt (object): matplotlib plot object.
    """

    # Open output file and read some attributes
    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']
    dt = f.attrs['dt']
    iterations = f.attrs['Iterations']
    time = np.linspace(0, (iterations - 1) * dt, num=iterations) / 1e-9 # [ns]

    # Check there are any receivers
    if nrx == 0:
        raise CmdInputError('No receivers found in {}'.format(filename))

    # Check for single output component when doing a FFT
    if fft:
        if not len(outputs) == 1:
            raise CmdInputError('A single output must be specified when using the -fft option')

    # New plot for each receiver
    for rx in range(1, nrx + 1):
        path = '/rxs/rx' + str(rx) + '/'
        availableoutputs = list(f[path].keys())

        # If only a single output is required, create one subplot
        if len(outputs) == 1:

            # Check for polarity of output and if requested output is in file
            if outputs[0][-1] == '-':
                polarity = -1
                outputtext = '-' + outputs[0][0:-1]
                output = outputs[0][0:-1]
            else:
                polarity = 1
                outputtext = outputs[0]
                output = outputs[0]

            if output not in availableoutputs:
                raise CmdInputError('{} output requested to plot, but the available output for receiver 1 is {}'.format(output, ', '.join(availableoutputs)))

            outputdata = f[path + output][:] * polarity
            outputdata_norm = outputdata / np.amax(np.abs(outputdata))
            env = np.abs(signal.hilbert(outputdata_norm))


            #* Calculate the background
            #background = np.mean(np.abs(outputdata[int(20e-9/dt):int(50e-9/dt)]))

            #* Detect the peak in the envelope
            """
            threshold = background * 1
            peak_idx = []
            peak_value = []

            i = 0
            while i < len(env):
                if env[i] > threshold:
                    start = i
                    while i < len(env) and env[i] > threshold:
                        i += 1
                    end = i
                    peak_idx.append(np.argmax(np.abs(outputdata[start:end])) + start)
                i += 1
            """

            # Plotting if FFT required
            if fft:
                # FFT
                freqs, power = fft_power(outputdata, dt)
                freqmaxpower = np.where(np.isclose(power, 0))[0][0]

                # Set plotting range to -60dB from maximum power or 4 times
                # frequency at maximum power
                try:
                    pltrange = np.where(power[freqmaxpower:] < -60)[0][0] + freqmaxpower + 1
                except:
                    pltrange = freqmaxpower * 4

                pltrange = np.s_[0:pltrange]

                # Calculate center frequency and bandwidth (-3dB)
                center_freq = freqs[freqmaxpower]
                lower_freq_3db = None
                upper_freq_3db = None
                bandwidth_3db = None

                # Find -3dB points
                try:
                    # Search for lower -3dB point (below center frequency)
                    lower_indices = np.where(power[:freqmaxpower] <= -3)[0]
                    if len(lower_indices) > 0:
                        lower_freq_3db = freqs[lower_indices[-1]]

                    # Search for upper -3dB point (above center frequency)
                    upper_indices = np.where(power[freqmaxpower:] <= -3)[0]
                    if len(upper_indices) > 0:
                        upper_freq_3db = freqs[upper_indices[0] + freqmaxpower]

                    # Calculate bandwidth if both points found
                    if lower_freq_3db is not None and upper_freq_3db is not None:
                        bandwidth_3db = upper_freq_3db - lower_freq_3db
                except:
                    pass

                # Plot time history of output component
                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, num='rx' + str(rx), figsize=(20, 10), facecolor='w', edgecolor='w', tight_layout=True)
                line1 = ax1.plot(time, outputdata_norm, 'k', lw=2, label=outputtext)
                #ax1.plot(time, env, 'b', lw=2, label='Envelope', linestyle='--', alpha=0.5)
                #ax1.scatter(time[peak_idx], outputdata[peak_idx], 'kx')
                #* Plot the peak
                #ax1.scatter(time[peak_idx], outputdata[peak_idx], color='r', marker='o', s=50, label='Peak')
                ax1.set_xlabel('Time [ns]', fontsize=28)
                ax1.set_ylabel(outputtext + ' field strength [V/m]', fontsize=28)
                #* Closeup otpion
                if use_closeup:
                    ax1.set_xlim([closeup_x_start, closeup_x_end])
                    ax1.set_ylim([closeup_y_start, closeup_y_end])
                else:
                    ax1.set_xlim([0, np.amax(time)])
                ax1.grid(which='both', axis='both', linestyle='-.')
                #ax1.legend(fontsize = 15)
                ax1.minorticks_on()
                ax1.tick_params(labelsize=24)

                # Plot frequency spectra
                line2 = ax2.plot(freqs[pltrange]/1e6, power[pltrange], 'k', lw=2)
                ax2.set_xlabel('Frequency [MHz]', fontsize=28)
                ax2.set_ylabel('Power [dB]', fontsize=28)
                ax2.grid(which='both', axis='both', linestyle='-.')
                ax2.tick_params(labelsize=24)
                ax2.set_ylim([-20, 1])

                # Change colours and labels for magnetic field components or currents
                if 'H' in outputs[0]:
                    plt.setp(line1, color='g')
                    plt.setp(line2, color='g')
                    plt.setp(ax1, ylabel=outputtext + ' field strength [A/m]')
                    plt.setp(stemlines, 'color', 'g')
                    plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
                elif 'I' in outputs[0]:
                    plt.setp(line1, color='b')
                    plt.setp(line2, color='b')
                    plt.setp(ax1, ylabel=outputtext + ' current [A]')
                    plt.setp(stemlines, 'color', 'b')
                    plt.setp(markerline, 'markerfacecolor', 'b', 'markeredgecolor', 'b')

                plt.show()

            #* Plotting if no FFT required
            else:
                fig, ax = plt.subplots(subplot_kw=dict(xlabel='Time [ns]', ylabel=outputtext + ' normalized field strength'), num='rx' + str(rx), figsize=(20, 10), facecolor='w', edgecolor='w', tight_layout=True)
                ax.plot(time, env, 'b', lw=2, label='Envelope', linestyle='--', alpha=0.5)
                line = ax.plot(time, outputdata_norm, 'k', lw=2, label=outputtext)
                #* Plot the peak
                #ax.scatter(time[peak_idx], outputdata[peak_idx], color='r', marker='o', s=50, label='Peak')
                #* Plot the background
                #ax.hlines(background, 0, np.amax(time), colors='gray', linestyles='--', label='Background')
                #ax.hlines(-background, 0, np.amax(time), colors='gray', linestyles='--')

                if use_closeup:
                    ax.set_xlim([closeup_x_start, closeup_x_end])
                    ax.set_ylim([closeup_y_start, closeup_y_end])
                else:
                    ax.set_xlim([0, np.amax(time)])
                ax.grid(which='both', axis='both', linestyle='-.')
                ax.minorticks_on()
                ax.set_xlabel('Time [ns]', fontsize=28)
                ax.set_ylabel(outputtext + ' field strength [V/m]', fontsize=28)
                ax.tick_params(labelsize=24)
                plt.tight_layout()
                ax.legend(fontsize = 24)

                if 'H' in output:
                    plt.setp(line, color='g')
                    plt.setp(ax, ylabel=outputtext + ', field strength [A/m]')
                elif 'I' in output:
                    plt.setp(line, color='b')
                    plt.setp(ax, ylabel=outputtext + ', current [A]')

        # If multiple outputs required, create all nine subplots and populate only the specified ones
        else:
            fig, ax = plt.subplots(subplot_kw=dict(xlabel='Time [ns]'), num='rx' + str(rx), figsize=(20, 10), facecolor='w', edgecolor='w')
            if len(outputs) == 9:
                gs = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.3)
            else:
                gs = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.3)

            for output in outputs:
                # Check for polarity of output and if requested output is in file
                if output[-1] == 'm':
                    polarity = -1
                    outputtext = '-' + output[0:-1]
                    output = output[0:-1]
                else:
                    polarity = 1
                    outputtext = output

                # Check if requested output is in file
                if output not in availableoutputs:
                    raise CmdInputError('Output(s) requested to plot: {}, but available output(s) for receiver {} in the file: {}'.format(', '.join(outputs), rx, ', '.join(availableoutputs)))

                outputdata = f[path + output][:] * polarity

                if output == 'Ex':
                    ax = plt.subplot(gs[0, 0])
                    ax.plot(time, outputdata, 'r', lw=2, label=outputtext)
                    ax.set_ylabel(outputtext + ', field strength [V/m]')
                # ax.set_ylim([-15, 20])
                elif output == 'Ey':
                    ax = plt.subplot(gs[1, 0])
                    ax.plot(time, outputdata, 'r', lw=2, label=outputtext)
                    ax.set_ylabel(outputtext + ', field strength [V/m]')
                # ax.set_ylim([-15, 20])

                # =====Ezのプロット=====
                elif output == 'Ez':
                    outputdata_norm = outputdata/np.amax(np.abs(outputdata)) * 100
                    ax = plt.subplot(gs[2, 0])
                    ax.plot(time, outputdata_norm, 'r', lw=2, label=outputtext)
                    #ax.set_ylim([-3, 3])
                    ax.set_xlim([5e-8, 5e-6])
                    #ax.set_xscale('log')
                    ax.set_ylabel(outputtext + ' strength [%]', size=18)
                    ax.tick_params(labelsize=18)


                elif output == 'Hx':
                    ax = plt.subplot(gs[0, 1])
                    ax.plot(time, outputdata, 'g', lw=2, label=outputtext)
                    ax.set_ylabel(outputtext + ', field strength [A/m]')
                # ax.set_ylim([-0.03, 0.03])
                elif output == 'Hy':
                    ax = plt.subplot(gs[1, 1])
                    ax.plot(time, outputdata, 'g', lw=2, label=outputtext)
                    ax.set_ylabel(outputtext + ', field strength [A/m]')
                # ax.set_ylim([-0.03, 0.03])
                elif output == 'Hz':
                    ax = plt.subplot(gs[2, 1])
                    ax.plot(time, outputdata, 'g', lw=2, label=outputtext)
                    ax.set_ylabel(outputtext + ', field strength [A/m]')
                # ax.set_ylim([-0.03, 0.03])
                elif output == 'Ix':
                    ax = plt.subplot(gs[0, 2])
                    ax.plot(time, outputdata, 'b', lw=2, label=outputtext)
                    ax.set_ylabel(outputtext + ', current [A]')
                elif output == 'Iy':
                    ax = plt.subplot(gs[1, 2])
                    ax.plot(time, outputdata, 'b', lw=2, label=outputtext)
                    ax.set_ylabel(outputtext + ', current [A]')
                elif output == 'Iz':
                    ax = plt.subplot(gs[2, 2])
                    ax.plot(time, outputdata, 'b', lw=2, label=outputtext)
                    ax.set_ylabel(outputtext + ', current [A]')
            for ax in fig.axes:
                if use_closeup:
                    ax.set_xlim([closeup_x_start*10**(-9), closeup_x_end*10**(-9)])
                else:
                    ax.set_xlim([0, np.amax(time)])
                ax.grid(which='both', axis='both', linestyle='--')
                ax.minorticks_on()

        # Save a PDF/PNG of the figure
        # fig.savefig(os.path.splitext(os.path.abspath(filename))[0] + '_rx' + str(rx) + '.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
        if use_fft:
            fig.savefig(os.path.splitext(os.path.abspath(filename))[0] + '_rx' + str(rx) + '_fft.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

            # Save FFT parameters to txt file
            if len(outputs) == 1 and fft:
                txt_filename = os.path.splitext(os.path.abspath(filename))[0] + '_rx' + str(rx) + '_fft_params.txt'
                if bandwidth_3db is not None:
                    with open(txt_filename, 'w') as txt_file:
                        txt_file.write(f'Center Frequency: {center_freq/1e6:.2f} MHz\n')
                        txt_file.write(f'Bandwidth (-3dB): {bandwidth_3db/1e6:.2f} MHz\n')
                        txt_file.write(f'Lower Frequency (-3dB): {lower_freq_3db/1e6:.2f} MHz\n')
                        txt_file.write(f'Upper Frequency (-3dB): {upper_freq_3db/1e6:.2f} MHz\n')
        elif use_closeup:
            fig.savefig(os.path.splitext(os.path.abspath(filename))[0] + '_rx' + str(rx) + '_closeup_x' + str(closeup_x_start) \
                            + '_' + str(closeup_x_end) + 'y' + str(closeup_y_end) +  '.png'
                            ,dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
        else:
            fig.savefig(os.path.splitext(os.path.abspath(filename))[0] + '_rx' + str(rx) + '.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

    f.close()

    return plt


if __name__ == "__main__":

    print("Plots electric and magnetic fields and currents from all receiver points in the given output file.")
    print("Each receiver point is plotted in a new figure window.")
    
    # Get output file path through interactive input
    outputfile = input("Enter the path to the output file: ").strip()
    if not os.path.exists(outputfile):
        raise CmdInputError('Output file {} does not exist'.format(outputfile))
    
    # Get output components
    print("\nAvailable output components: Ex, Ey, Ez, Hx, Hy, Hz, Ix, Iy, Iz")
    print("You can also use negative polarity (e.g., Ex-, Ey-, etc.)")
    print("Default outputs: {}".format(', '.join(Rx.defaultoutputs)))
    outputs_input = input("Enter output components to plot (space-separated) [default: {}]: ".format(' '.join(Rx.defaultoutputs))).strip()
    
    if outputs_input == "":
        outputs = Rx.defaultoutputs
    else:
        outputs = outputs_input.split()
        # Validate outputs
        valid_outputs = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 'Ix', 'Iy', 'Iz', 'Ex-', 'Ey-', 'Ez-', 'Hx-', 'Hy-', 'Hz-', 'Ix-', 'Iy-', 'Iz-']
        for output in outputs:
            if output not in valid_outputs:
                raise CmdInputError('Invalid output component: {}. Valid options: {}'.format(output, ', '.join(valid_outputs)))
    
    # Get FFT option
    fft_input = input("Plot FFT? (y/n) [default: n]: ").strip().lower()
    use_fft = fft_input == 'y'
    
    # Get closeup option
    closeup_input = input("Plot close up of time domain signal? (y/n) [default: n]: ").strip().lower()
    if closeup_input == 'y':
        closeup_x_start = float(input("Enter close up x-axis start [ns]: ").strip())
        closeup_x_end = float(input("Enter close up x-axis end [ns]: ").strip())
        closeup_y_range = np.abs(float(input("Enter close up y-axis range [ns]: ").strip()))
        closeup_y_start = - closeup_y_range
        closeup_y_end = closeup_y_range
    use_closeup = closeup_input == 'y'

    plthandle = mpl_plot(outputfile, outputs, fft=use_fft)
    plthandle.show()
