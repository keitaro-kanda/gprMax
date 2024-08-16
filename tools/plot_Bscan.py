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
import json

import h5py
import matplotlib.pyplot as plt
import numpy as np

from gprMax.exceptions import CmdInputError

from outputfiles_merge import get_output_data
from scipy import signal
import mpl_toolkits.axes_grid1 as axgrid1


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


    #* Create figure
    fig = plt.figure(num=filename + ' - rx' + str(rxnumber),
                    figsize=(20, 15), facecolor='w', edgecolor='w')



    #* Load antenna settings
    src_step = params['antenna_settings']['src_step']
    rx_step = params['antenna_settings']['rx_step']
    src_start = params['antenna_settings']['src_start']
    rx_start = params['antenna_settings']['rx_start']
    if src_step == rx_step:
        antenna_step = src_step
        antenna_start = (src_start + rx_start) / 2
    elif np.abs(src_step) == np.abs(rx_step):
        antenna_step = np.abs(src_step) * 2
    else:
        antenna_step = 1


    #* Plot part
    #* Calculate and plot envelope
    if args.envelope:
        data = np.abs(signal.hilbert(outputdata, axis=0))
        plt.imshow(data,
             extent=[antenna_start, antenna_start + outputdata.shape[1] * antenna_step, outputdata.shape[0] * dt, 0],
            interpolation='nearest', aspect='auto', cmap='jet',
            vmin=0, vmax=10)
    #* Plot raw B-scan
    else:
        outputdata_norm = outputdata / np.amax(np.abs(outputdata)) * 100
        plt.imshow(outputdata_norm,
                extent=[antenna_start, antenna_start + outputdata_norm.shape[1] * antenna_step, outputdata_norm.shape[0] * dt, 0],
                interpolation='nearest', aspect='auto', cmap='seismic',
                vmin=-1, vmax=1)
    plt.xlabel('x [m]', fontsize=20)
    plt.ylabel('Time [ns]', fontsize=20)
    plt.tick_params(labelsize=18)



    #* Closeup settings
    if closeup:
        closeup_start = 20 # [ns]
        closeup_end = 70 # [ns]
        plt.ylim(closeup_end, closeup_start)
        #plt.minorticks_on( )


    plt.title('rx' + str(rxnumber), fontsize=20)

    # Grid properties
    ax = fig.gca()
    ax.grid(which='both', axis='both', linestyle='-.')


    #* Colorbar
    delvider = axgrid1.make_axes_locatable(ax)
    cax = delvider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(cax=cax)
    cbar.set_label('Normalized field amplitude [%]', size=20)
    cbar.ax.tick_params(labelsize=16)
    if args.envelope:
        cbar.set_label('Envelope amplitude', fontsize=20)
    else:
        cbar.set_label('Normalized field strength [%]', fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    #* Save plot
    savefile = os.path.splitext(filename)[0]
    # fig.savefig(path + os.sep + savefile + '.pdf', dpi=None, format='pdf', 
    #             bbox_inches='tight', pad_inches=0.1)

    #* Closeup of raw B-scan
    if closeup and args.envelope==False:
        outputdir = os.path.join(outputdir, 'closeup')
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
        fig.savefig(outputdir +os.sep + savefile + 'rx' + str(rxnumber) +
                    '_closeup'+str(closeup_start)+'_'+str(closeup_end)+ '.png', dpi=150, format='png', 
                bbox_inches='tight', pad_inches=0.1)
        fig.savefig(outputdir +os.sep + savefile + 'rx' + str(rxnumber) +
                    '_closeup'+str(closeup_start)+'_'+str(closeup_end)+ '.pdf', dpi=300, format='pdf',
                    bbox_inches='tight', pad_inches=0.1)
    #* Envelope
    elif args.envelope and closeup==False:
        outputdir = os.path.join(outputdir, 'envelope')
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
        fig.savefig(outputdir + os.sep + savefile + 'rx' + str(rxnumber) + '_envelope.png', dpi=150, format='png', 
                bbox_inches='tight', pad_inches=0.1)
        fig.savefig(outputdir + os.sep + savefile + 'rx' + str(rxnumber) + '_envelope.pdf', dpi=300, format='pdf',
                    bbox_inches='tight', pad_inches=0.1)
    #* Closeup of envelope
    elif args.envelope and closeup:
        outputdir = os.path.join(outputdir, 'envelope')
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
        fig.savefig(outputdir +os.sep + savefile + 'rx' + str(rxnumber) +
                    '_closeup'+str(closeup_start)+'_'+str(closeup_end)+ '_envelope.png', dpi=150, format='png', 
                bbox_inches='tight', pad_inches=0.1)
        fig.savefig(outputdir +os.sep + savefile + 'rx' + str(rxnumber) +
                    '_closeup'+str(closeup_start)+'_'+str(closeup_end)+ '_envelope.pdf', dpi=300, format='pdf',
                    bbox_inches='tight', pad_inches=0.1)
    #* Raw B-scan
    else:
        fig.savefig(outputdir + os.sep + savefile + 'rx' + str(rxnumber) + '.png', dpi=150, format='png', 
                bbox_inches='tight', pad_inches=0.1)
        fig.savefig(outputdir + os.sep + savefile + 'rx' + str(rxnumber) + '.pdf', dpi=300, format='pdf',
                    bbox_inches='tight', pad_inches=0.1)

    return plt



#* Main part of the script
if __name__ == "__main__":
    #* Parse command line arguments
    parser = argparse.ArgumentParser(
        prog = 'plot_Bscan.py',
        description = 'Plots a B-scan image.',
        epilog = 'End of help message',
        usage = 'python tools/plot_Bscan.py [jsonfile] [rx_component] [-closeup] [--select-rx] [-envelope]'
        )
    #parser.add_argument('outputfile', help='name of output file including path')
    parser.add_argument('jsonfile', help='name of json file including path')
    parser.add_argument('rx_component', help='name of output component to be plotted',
                        choices=['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 'Ix', 'Iy', 'Iz'])
    # closeupのオプションを作る
    parser.add_argument('-closeup', action='store_true', help='closeup of the plot', default=False)
    parser.add_argument('--select-rx', action='store_true', help='select rx number', default=False)
    parser.add_argument('-envelope', action='store_true', help='Apply envelope to the data', default=False)
    args = parser.parse_args()

    #* Load json file
    with open (args.jsonfile) as f:
        params = json.load(f)
    outfile_path = params['data']

    #* Open output file and read number of outputs (receivers)
    #f = h5py.File(args.outputfile, 'r') # outファイルの読み込み？
    f = h5py.File(outfile_path, 'r')
    nrx = f.attrs['nrx'] # Attribute(属性)の読み取り？、nrx:レシーバーの総数
    print('nrx: ', nrx)
    f.close()

    # Check there are any receivers
    if nrx == 0:
        raise CmdInputError('No receivers found in {}'.format(args.outputfile))

    # データの取得とプロットの作成を実行？
    if args.select_rx:
        rx = 1
        outputdata, dt = get_output_data(outfile_path, rx, args.rx_component)
        dt = dt * 1e9 # Convert to [ns]
        plthandle = mpl_plot(outfile_path, outputdata, dt, rx, args.rx_component, args.closeup)
    else:
        for rx in range(1, nrx + 1):
            outputdata, dt = get_output_data(outfile_path, rx, args.rx_component)
            dt = dt * 1e9 # Convert to [ns]
            plthandle = mpl_plot(outfile_path, outputdata, dt, rx, args.rx_component, args.closeup)
    plthandle.show()
