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

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy import size

from gprMax.exceptions import CmdInputError
from gprMax.utilities import fft_power, round_value
from gprMax.waveforms import Waveform


def check_timewindow(timewindow, dt):
    """Checks and sets time window and number of iterations.

    Args:
        timewindow (float): Time window.
        dt (float): Time discretisation.

    Returns:
        timewindow (float): Time window.
        iterations (int): Number of interations.
    """

    # Time window could be a string, float or int, so convert to string then check
    timewindow = str(timewindow)

    try:
        timewindow = int(timewindow)
        iterations = timewindow
        timewindow = (timewindow - 1) * dt

    except:
        timewindow = float(timewindow)
        if timewindow > 0:
            iterations = round_value((timewindow / dt)) + 1
        else:
            raise CmdInputError('Time window must have a value greater than zero')

    return timewindow, iterations


def mpl_plot(w, timewindow, dt, iterations, fft=False, power=False):
    """Plots waveform and prints useful information about its properties.

    Args:
        w (class): Waveform class instance.
        timewindow (float): Time window.
        dt (float): Time discretisation.
        iterations (int): Number of iterations.
        fft (boolean): Plot FFT switch.

    Returns:
        plt (object): matplotlib plot object.
    """

    time = np.linspace(0, (iterations - 1) * dt, num=iterations)
    waveform = np.zeros(len(time))
    timeiter = np.nditer(time, flags=['c_index'])

    while not timeiter.finished:
        waveform[timeiter.index] = w.calculate_value(timeiter[0], dt)
        timeiter.iternext()

    print('Waveform characteristics...')
    print('Type: {}'.format(w.type))
    print('Maximum (absolute) amplitude: {:g}'.format(np.max(np.abs(waveform))))

    if w.freq and not w.type == 'gaussian':
        print('Centre frequency: {:g} Hz'.format(w.freq))

    if w.type == 'gaussian' or w.type == 'gaussiandot' or w.type == 'gaussiandotnorm' or w.type == 'gaussianprime' or w.type == 'gaussiandoubleprime':
        delay = 1 / w.freq
        print('Time to centre of pulse: {:g} s'.format(delay))
    elif w.type == 'gaussiandotdot' or w.type == 'gaussiandotdotnorm' or w.type == 'ricker':
        delay = np.sqrt(2) / w.freq
        print('Time to centre of pulse: {:g} s'.format(delay))

    print('Time window: {:g} s ({} iterations)'.format(timewindow, iterations))
    print('Time step: {:g} s'.format(dt))

    if fft:
        # FFT
        freqs, power = fft_power(waveform, dt)

        # Set plotting range to 4 times frequency at max power of waveform or
        # 4 times the centre frequency
        freqmaxpower = np.where(np.isclose(power, 0))[0][0]
        if freqs[freqmaxpower] > w.freq:
            pltrange = np.where(freqs > 4 * freqs[freqmaxpower])[0][0]
        else:
            pltrange = np.where(freqs > 4 * w.freq)[0][0]
        pltrange = np.s_[0:pltrange]

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, num=w.type, figsize=(16, 8), facecolor='w', edgecolor='w', tight_layout=True)

        # Plot waveform
        ax1.plot(time * 1e9, waveform, 'k', lw=2)
        ax1.set_xlabel('Time [ns]', fontsize=28)
        ax1.set_ylabel('Amplitude', fontsize=28)
        ax1.set_xlim([0, 10])
        ax1.tick_params(labelsize=24)

        # Plot frequency spectra
        ax2.plot(freqs[pltrange]/1e6, power[pltrange], 'k', lw=2)
        ax2.set_xlabel('Frequency [MHz]', fontsize=28)
        ax2.set_ylabel('Power [dB]', fontsize=28)
        ax2.set_ylim([-20, 1])
        ax2.tick_params(labelsize=24)

    else:
        fig, ax1 = plt.subplots(num=w.type, figsize=(16, 8), facecolor='w', edgecolor='w')

        # Plot waveform
        ax1.plot(time * 1e9, waveform, 'r', lw=2)
        plt.supxtitle(args.type + ', ' + args.freq / 1e6 + ' MHz', fontsize=28)
        ax1.set_xlabel('Time [ns]', fontsize=28)
        ax1.set_ylabel('Amplitude', fontsize=28)
        ax1.tick_params(labelsize=24)

    [ax.grid(which='both', axis='both', linestyle='-.') for ax in fig.axes]  # Turn on grid

    output_dir = 'kanda/waveform'
    folder_name = str(args.type) + '_' + str(args.freq / 1e6) + 'MHz'
    output_dir = os.path.join(output_dir, folder_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    #* Save settings in txt file
    with open(output_dir + '/' + 'settings.txt', 'w') as f:
        f.write('Waveform characteristics...\n')
        f.write('Type: {}\n'.format(w.type))
        f.write('Maximum (absolute) amplitude: {:g}\n'.format(np.max(np.abs(waveform))))
        if w.freq and not w.type == 'gaussian':
            f.write('Centre frequency: {:g} Hz\n'.format(w.freq))
        if w.type == 'gaussian' or w.type == 'gaussiandot' or w.type == 'gaussiandotnorm' or w.type == 'gaussianprime' or w.type == 'gaussiandoubleprime':
            f.write('Time to centre of pulse: {:g} s\n'.format(delay))
        elif w.type == 'gaussiandotdot' or w.type == 'gaussiandotdotnorm' or w.type == 'ricker':
            f.write('Time to centre of pulse: {:g} s\n'.format(delay))
        f.write('Time window: {:g} s ({} iterations)\n'.format(timewindow, iterations))
        f.write('Time step: {:g} s\n'.format(dt))
    
    # Save a PNG/PDF of the figure
    plt.savefig(output_dir + '/' + 'waveform.png')
    plt.savefig(output_dir + '/' + 'waveform.pdf', format='pdf', dpi=300)

    # Save a PDF/PNG of the figure
    # fig.savefig(os.path.dirname(os.path.abspath(__file__)) + os.sep + w.type + '.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
    # fig.savefig(os.path.dirname(os.path.abspath(__file__)) + os.sep + w.type + '.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

    return plt


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='plot_source_wave.py',
        description='Plot built-in waveforms that can be used for sources.',
        epilog='End of help message',
        usage='python tools/plot_source_wave.py [type] [amp] [freq] [timewindow] [dt] [-fft] [-power]'
        )
    parser.add_argument('type', help='type of waveform', choices=Waveform.types)
    parser.add_argument('amp', type=float, help='amplitude of waveform')
    parser.add_argument('freq', type=float, help='centre frequency of waveform')
    parser.add_argument('timewindow', help='time window to view waveform')
    parser.add_argument('dt', type=float, help='time step to view waveform')
    parser.add_argument('-fft', action='store_true', help='plot FFT of waveform', default=False)
    parser.add_argument('-power', action='store_true', help='plot power of waveform', default=False)
    args = parser.parse_args()

    # Check waveform parameters
    if args.type.lower() not in Waveform.types:
        raise CmdInputError('The waveform must have one of the following types {}'.format(', '.join(Waveform.types)))
    if args.freq <= 0:
        raise CmdInputError('The waveform requires an excitation frequency value of greater than zero')

    # Create waveform instance
    w = Waveform()
    w.type = args.type
    w.amp = args.amp
    w.freq = args.freq

    timewindow, iterations = check_timewindow(args.timewindow, args.dt)
    plthandle = mpl_plot(w, timewindow, args.dt, iterations, args.fft)
    plthandle.show()
