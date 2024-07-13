import json
import numpy as np
import matplotlib.pyplot as plt
from outputfiles_merge import get_output_data
import h5py
import cv2
import mpl_toolkits.axes_grid1 as axgrid1
import os


json_path = 'kanda/domain_6x10/subsurface_rock/5cm/calc/calc.json'
output_dir = os.path.dirname(json_path)


with open(json_path) as f:
    params = json.load(f)

data_path = params['out_file']
f = h5py.File(data_path, 'r')
nrx = f.attrs['nrx']

for rx in range(nrx):
    data, dt = get_output_data(data_path, (rx+1), 'Ez')

#* Apply sobel filter
def sobel():
    sobelx = cv2.Sobel(np.abs(data[int(10e-9/dt):]), cv2.CV_64F, 1, 0, ksize=5)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(np.abs(data[int(10e-9/dt):]), cv2.CV_64F, 0, 1, ksize=5)
    sobely = cv2.convertScaleAbs(sobely)

    sobel_combined = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    plot_list = [data, sobelx, sobely, sobel_combined]
    return plot_list
#list = sobel()


#* Calculate gradient of the data
def gradient():
    gradx = np.gradient(data[int(10e-9/dt):], axis=1)
    grady = np.gradient(data[int(10e-9/dt):], axis=0)
    grad = np.sqrt(gradx**2 + grady**2)

    plot_list = [data, gradx, grady, grad]
    return plot_list
list = gradient()


#* Plot the data
fig, ax = plt.subplots(2, 2, figsize=(15, 15))

for i, data in enumerate(list):
    if i == 0:
        im = ax[i//2, i%2].imshow(data/np.amax(data)*100, cmap='seismic', aspect='auto',
                                extent = [0, data.shape[1] * 0.04, data.shape[0] * dt * 1e9, 0],
                                vmin=-1, vmax=1)
        #ax[i//2, i%2].set_ylim(35, 25)
    else:
        im = ax[i//2, i%2].imshow(data, cmap='jet', aspect='auto',
                                extent = [0, data.shape[1] * 0.04, data.shape[0] * dt * 1e9, 10],
                                vmin=0, vmax=np.amax(data)/2)
        #ax[i//2, i%2].set_ylim(35, 25)
    ax[i//2, i%2].set_title(['Ez', 'Sobel x', 'Sobel y', 'Sobel combined'][i])

    delvider = axgrid1.make_axes_locatable(ax[i//2, i%2])
    cax = delvider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical')

plt.savefig(output_dir + '/gradient.png', bbox_inches='tight', dpi=300)
plt.show()