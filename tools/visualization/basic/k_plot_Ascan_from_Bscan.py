import json
import numpy as np
import matplotlib.pyplot as plt
import h5py
import mpl_toolkits.axes_grid1 as axgrid1
import os
import argparse
from outputfiles_merge import get_output_data
from tqdm import tqdm
from scipy import signal



#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='k_plot_Ascan_from_Bscan.py',
    description='Plot the cross-sectional A-scan from the B-scan',
    epilog='End of help message',
    usage='python -m tools.k_fk_migration [json_path] [x] [t_first] [t_last] [-auto]',
)
parser.add_argument('json_path', help='Path to the json file')
parser.add_argument('x', type=float, help='x position [m] of the A-scan')
parser.add_argument('t_first', type=float, help='Start time of the A-scan [ns]')
parser.add_argument('t_last', type=float, help='Last time of the A-scan [ns]')
parser.add_argument('-auto', action='store_true', help='Select plot area automatically from setting list')
args = parser.parse_args()



#* Load json file
with open(args.json_path) as f:
    params = json.load(f)
#* Load antenna settings
src_step = params['antenna_settings']['src_step']
rx_step = params['antenna_settings']['rx_step']
src_start = params['antenna_settings']['src_start']
rx_start = params['antenna_settings']['rx_start']
#* Check antenna step
if src_step == rx_step:
    antenna_step = src_step
    antenna_start = (src_start + rx_start) / 2



#* Load output file
data_path = params['data']
output_dir = os.path.join(os.path.dirname(data_path), 'Ascan_from_Bscan')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

f = h5py.File(data_path, 'r')
nrx = f.attrs['nrx']
for rx in range(nrx):
    Bscan, dt = get_output_data(data_path, (rx+1), params['data_name'])
print('data shape: ', Bscan.shape)





#* Extract A-scan data from B-scan data
def make_plot_data(Bscan_data, x, t_first, t_last):
    x_idx = int((x - antenna_start) / antenna_step)
    t_first_idx = int(t_first * 1e-9 / dt)
    t_last_idx = int(t_last * 1e-9 / dt)
    Ascan = Bscan_data[t_first_idx:t_last_idx, x_idx]

    t_array = np.arange(t_first_idx, t_last_idx) * dt / 1e-9

    #* Calculate the envelope
    envelope = np.abs(signal.hilbert(Ascan))

    #* Calculate the background
    background = np.mean(np.abs(Bscan[int(50e-9/dt):, x_idx]))


    #* Detect the peak in the envelope
    threshold = background * 3
    peak_idx = []
    peak_value = []

    i = 0
    while i < len(envelope):
        if envelope[i] > threshold:
            start = i
            while i < len(envelope) and envelope[i] > threshold:
                i += 1
            end = i
            peak_idx.append(np.argmax(np.abs(Ascan[start:end])) + start)
            peak_value.append(Ascan[peak_idx[-1]])
        i += 1


    #* Closeup B-scan around the A-scan
    x_first_idx = x_idx - int(5/antenna_step) # index
    x_last_idx = x_idx + int(5/antenna_step) # index
    if x_last_idx > Bscan.shape[1]:
        x_last_idx = Bscan.shape[1]
    Bscan_trim = Bscan[t_first_idx:t_last_idx, x_first_idx:x_last_idx]

    return Ascan, envelope, background, t_array, peak_idx, peak_value, Bscan_trim, x_first_idx, x_last_idx



#* Plot
def plot(Ascan_data, t_array, envelope_data, background_value, t_first, t_last, peak_idx, peak_value, Bscan_data, x, x_first_idx, x_last_idx):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8), sharey=True, tight_layout=True)

    #* Plot A-scan
    ax[0].plot(Ascan_data, t_array, color='black', label='Signal')
    ax[0].plot(envelope_data, t_array, color='blue', linestyle='-.', label='Envelope')
    ax[0].vlines(background_value, t_first, t_last, color='gray', linestyle='--', label='Background')
    ax[0].vlines(-background_value, t_first, t_last, color='gray', linestyle='--')
    ax[0].scatter(peak_value, t_array[peak_idx], color='r')

    ax[0].set_xlabel('Amplitude', fontsize=20)
    ax[0].grid(True)
    ax[0].tick_params(labelsize=18)
    ax[0].set_xlim(-np.amax(np.abs(Ascan_data))*1.2, np.amax(np.abs(Ascan_data))*1.2)
    ax[0].legend(fontsize=16)



    #* Plot B-scan
    im = ax[1].imshow(Bscan_data,
                    aspect='auto', cmap='seismic',
                    vmin=-np.amax(np.abs(Bscan))/5, vmax=np.amax(np.abs(Bscan))/5,
                    extent=[antenna_start + x_first_idx*antenna_step, antenna_start + x_last_idx*antenna_step, t_last, t_first])
    ax[1].set_xlabel('x [m]', fontsize=20)
    ax[1].tick_params(labelsize=18)

    ax[1].vlines(x, t_first, t_last, color='k', linestyle='-.')

    delvider = axgrid1.make_axes_locatable(ax[1])
    cax = delvider.append_axes('right', size='3%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Amplitude', fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    fig.supylabel('Time [ns]', fontsize=20)


    #* Save the plot
    plt.savefig(os.path.join(output_dir, f'{x}_t{t_first}_{t_last}.png'), dpi=120)
    plt.savefig(os.path.join(output_dir, f'{x}_t{t_first}_{t_last}.pdf'), dpi=300)

    if args.auto:
        plt.close()
    else:
        plt.show()


if args.auto:
    plot_list = np.loadtxt(os.path.join(output_dir, 'plot_list.txt'))
    for i in tqdm(range(plot_list.shape[0]), desc='Plot A-scan'):
        Ascan, envelope, background, t_array, peak_idx, peak_value, Bscan_trim, x_first_idx, x_last_idx = make_plot_data(Bscan, plot_list[i, 0], plot_list[i, 1], plot_list[i, 2])
        plot(Ascan, t_array, envelope, background, plot_list[i, 1], plot_list[i, 2], peak_idx, peak_value, Bscan_trim, plot_list[i, 0], x_first_idx, x_last_idx)

else:
    #* Save x, t_first, t_last to txt file
    plot_params = [args.x, args.t_first, args.t_last]
    plot_params = np.array(plot_params).reshape(1, 3)


    #* Make plot
    Ascan, envelope, background, t_array, peak_idx, peak_value, Bscan_trim, x_first_idx, x_last_idx = make_plot_data(Bscan, args.x, args.t_first, args.t_last)
    plot(Ascan, t_array, envelope, background, args.t_first, args.t_last, peak_idx, peak_value, Bscan_trim, args.x, x_first_idx, x_last_idx)


    #* Save the plot parameters
    if not os.path.exists(os.path.join(output_dir, 'plot_list.txt')):
        np.savetxt(os.path.join(output_dir, 'plot_list.txt'), plot_params)
        print('Plot list saved at', os.path.join(output_dir, 'plot_list.txt'))
    else:
        plot_list = np.loadtxt(os.path.join(output_dir, 'plot_list.txt'))
        #* もし同じパラメータがあればスキップ
        for i in range(plot_list.shape[0]):
            if np.allclose(plot_params, plot_list[i]):
                print('The same parameters are already in the list. Skip.')
                break
            else:
                plot_list = np.vstack([plot_list, plot_params])
                plot_list = plot_list[plot_list[:, 0].argsort()]
                np.savetxt(os.path.join(output_dir, 'plot_list.txt'), plot_list)
                print('Plot list saved at', os.path.join(output_dir, 'plot_list.txt'))