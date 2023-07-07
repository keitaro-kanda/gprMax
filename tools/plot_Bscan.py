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
from matplotlib.colors import LogNorm

from gprMax.exceptions import CmdInputError
from gprMax.input_cmd_funcs import src_steps

from .outputfiles_merge import get_output_data


# プロットを作る関数の作成部分？
def mpl_plot(filename, outputdata, dt, rxnumber, rxcomponent):
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

    fig = plt.figure(num=filename + ' - rx' + str(rxnumber), 
                     figsize=(20, 10), facecolor='w', edgecolor='w')
    
    """ 追加（logスケールにするやつ
    for i in range(len(outputdata)):
        for j in range(len(outputdata[i])):
            if outputdata[i][j] > 0:
                outputdata[i][j] = np.log10(outputdata[i][j])
            elif outputdata[i][j] < 0:
                outputdata[i][j] = -np.log10(-outputdata[i][j])
            #elif outputdata[i][j] == 0:
            #    outputdata[i][j] = 0
    追加 """

    """ 追加 （dBスケールっぽくするやつ）
    # Check the elements of "outputdata" one by one using a for statement, and if they are positive, rewrite the array elements to 1
    for i in range(len(outputdata)):
        for j in range(len(outputdata[i])):
            if outputdata[i][j] > 0:
                outputdata[i][j] = - 10 * np.log10(outputdata[i][j] / np.amax(outputdata))
            elif outputdata[i][j] < 0:
                outputdata[i][j] = 10 * np.log10(outputdata[i][j] / np.amin(outputdata))
            elif outputdata[i][j] == 0:
                outputdata[i][j] = 0
    追加 """

    # [%]に変換
    outputdata_norm = outputdata / np.amax(np.abs(outputdata)) * 100

    # 観測の方向
    radar_direction = 'horizontal' # horizontal or vertical
    # 観測の間隔
    src_step = 0.2 #[m]

    # プロット
    if radar_direction == 'horizontal':
        plt.imshow(outputdata_norm, 
                 extent=[0, outputdata_norm.shape[1] * src_step, outputdata_norm.shape[0] * dt, 0], 
                interpolation='nearest', aspect='auto', cmap='seismic', vmin=-0.1, vmax=0.1)
        plt.xlabel('Horizontal distance [m]')
        plt.ylabel('Time [s]')
        closeup = False # True or False
        if closeup:
            plt.ylim(2.5e-7, 0)
            plt.minorticks_on( )
    else:
    # Create a plot rotated 90 degrees and then reversed up and down.
        plt.imshow(outputdata_norm.T[::-1],
                extent=[0, outputdata_norm.shape[0] * dt, 0, outputdata_norm.shape[1]], 
                interpolation='nearest', aspect='auto', cmap='seismic', vmin=-10, vmax=10)
        plt.xlabel('Time [s]')
        plt.ylabel('Trace number')


    if closeup:
        plt.title('{}'.format(filename) + '_closeup')
    else:
        plt.title('{}'.format(filename))

    # Grid properties
    ax = fig.gca()
    ax.grid(which='both', axis='both', linestyle='-.')

    cb = plt.colorbar()
    if 'E' in rxcomponent:
        cb.set_label('Field strength percentage [%]')
    elif 'H' in rxcomponent:
        cb.set_label('Field strength [A/m]')
    elif 'I' in rxcomponent:
        cb.set_label('Current [A]')

    # Save a PDF/PNG of the figure
    savefile = os.path.splitext(filename)[0]
    # fig.savefig(path + os.sep + savefile + '.pdf', dpi=None, format='pdf', 
    #             bbox_inches='tight', pad_inches=0.1)
    if closeup:
        fig.savefig(path + os.sep + savefile + '_closeup.png', dpi=150, format='png', 
                 bbox_inches='tight', pad_inches=0.1)
    else:
        fig.savefig(path + os.sep + savefile + '.png', dpi=150, format='png', 
                 bbox_inches='tight', pad_inches=0.1)

    return plt


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plots a B-scan image.', 
                                     usage='cd gprMax; python -m tools.plot_Bscan outputfile output')
    parser.add_argument('outputfile', help='name of output file including path')
    parser.add_argument('rx_component', help='name of output component to be plotted', 
                        choices=['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 'Ix', 'Iy', 'Iz'])
    args = parser.parse_args()

    # Open output file and read number of outputs (receivers)
    f = h5py.File(args.outputfile, 'r') # outファイルの読み込み？
    nrx = f.attrs['nrx'] # Attribute(属性)の読み取り？、nrx:レシーバーの総数
    f.close()

    # Check there are any receivers
    if nrx == 0:
        raise CmdInputError('No receivers found in {}'.format(args.outputfile))

    # データの取得とプロットの作成を実行？
    for rx in range(1, nrx + 1):
        outputdata, dt = get_output_data(args.outputfile, rx, args.rx_component)
        plthandle = mpl_plot(args.outputfile, outputdata, dt, rx, args.rx_component)

    plthandle.show()
