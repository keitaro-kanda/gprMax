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

import h5py
import matplotlib.pyplot as plt
import numpy as np
from traitlets import default

from gprMax.exceptions import CmdInputError

from .outputfiles_merge import get_output_data


# プロットを作る関数の作成部分？
def mpl_plot(filename, outputdata, dt, rxnumber, rxcomponent, closeup=False):
    """Creates a plot (with matplotlib) of the B-scan.

    Args:
        filename (string): Filename (including path) of output file.
        outputdata (array): Array of A-scans, i.e. B-scan data.
        dt (float): Temporal resolution of the model.
        rxnumber (int): Receiver output number.
        rxcomponent (str): Receiver output field/current component.

    Returns:
        plt (object): matplotlib plot object.
    """

    (path, filename) = os.path.split(filename)
    outputdir = os.path.join(path, 'Bscan_plots')
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    fig = plt.figure(num=filename + ' - rx' + str(rxnumber), 
                     figsize=(15, 15), facecolor='w', edgecolor='w')
    

    # normalize
    outputdata_norm = outputdata / np.amax(np.abs(outputdata)) * 100


    plt.imshow(outputdata_norm, 
             extent=[0, outputdata_norm.shape[1], outputdata_norm.shape[0] * dt, 0], 
            interpolation='nearest', aspect='auto', cmap='seismic', vmin=-0.1, vmax=0.1)
    plt.xlabel('trace number', fontsize=20)
    plt.ylabel('Time [s]', fontsize=20)
    plt.tick_params(labelsize=18)

    # =====closeupオプション=====
    if closeup:

        closeup_start = 540 # [ns]
        closeup_end = 590 # [ns]

        # カラーバー範囲の設定

        plt.imshow(outputdata_norm, 
             extent=[0, outputdata_norm.shape[1], outputdata_norm.shape[0] * dt, 0], 
            interpolation='nearest', aspect='auto', cmap='seismic', vmin=-0.01, vmax=0.01)
        #plt.xlim(35, 55)
        plt.ylim(closeup_end*10**(-9), closeup_start*10**(-9))
        plt.minorticks_on( )


    if closeup:
        plt.title('rx' + str(rxnumber) + ' closeup: '+str(closeup_start)+'-'+str(closeup_end), fontsize=20)
    else:
        plt.title('rx' + str(rxnumber), fontsize=20)

    # Grid properties
    ax = fig.gca()
    ax.grid(which='both', axis='both', linestyle='-.')

    cb = plt.colorbar()
    if 'E' in rxcomponent:
        cb.set_label('Field strength percentage [%]', fontsize=20)
    elif 'H' in rxcomponent:
        cb.set_label('Field strength [A/m]')
    elif 'I' in rxcomponent:
        cb.set_label('Current [A]')
    cb.ax.tick_params(labelsize=18)

    # Save a PDF/PNG of the figure
    savefile = os.path.splitext(filename)[0]
    # fig.savefig(path + os.sep + savefile + '.pdf', dpi=None, format='pdf', 
    #             bbox_inches='tight', pad_inches=0.1)
    if closeup:
        fig.savefig(outputdir +os.sep + savefile + 'rx' + str(rxnumber) +
                    'closeup'+str(closeup_start)+'_'+str(closeup_end)+ '.png', dpi=150, format='png', 
                 bbox_inches='tight', pad_inches=0.1)
    else:
        fig.savefig(outputdir + os.sep + savefile + 'rx' + str(rxnumber) + '.png', dpi=150, format='png', 
                 bbox_inches='tight', pad_inches=0.1)

    return plt


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plots a B-scan image.', 
                                     usage='cd gprMax; python -m tools.plot_Bscan outputfile output -closeup --select-rx') 
    parser.add_argument('outputfile', help='name of output file including path')
    parser.add_argument('rx_component', help='name of output component to be plotted', 
                        choices=['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 'Ix', 'Iy', 'Iz'])
    # closeupのオプションを作る
    parser.add_argument('-closeup', action='store_true', help='closeup of the plot', default=False)
    parser.add_argument('--select-rx', action='store_true', help='select rx number', default=False)
    args = parser.parse_args()

    # Open output file and read number of outputs (receivers)
    f = h5py.File(args.outputfile, 'r') # outファイルの読み込み？
    nrx = f.attrs['nrx'] # Attribute(属性)の読み取り？、nrx:レシーバーの総数
    print(nrx)
    f.close()

    # Check there are any receivers
    if nrx == 0:
        raise CmdInputError('No receivers found in {}'.format(args.outputfile))

    # データの取得とプロットの作成を実行？
    if args.select_rx:
        rx = 1
        outputdata, dt = get_output_data(args.outputfile, rx, args.rx_component)
        plthandle = mpl_plot(args.outputfile, outputdata, dt, rx, args.rx_component, args.closeup)
    else:
        for rx in range(1, nrx + 1):
            outputdata, dt = get_output_data(args.outputfile, rx, args.rx_component)
            plthandle = mpl_plot(args.outputfile, outputdata, dt, rx, args.rx_component, args.closeup)
    plthandle.show()
