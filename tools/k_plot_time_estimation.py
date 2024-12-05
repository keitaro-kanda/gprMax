import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import argparse
from tqdm import tqdm
from outputfiles_merge import get_output_data
import json



#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='k_plot_time_estimation.py',
    description='Plot the estimated two-way travel time on the A-scan',
    epilog='End of help message',
    usage='python -m tools.k_plot_time_estimation [out_file] [model_json] [-closeup]',
)
parser.add_argument('out_file', help='Path to the .out file')
parser.add_argument('model_json', help='Path to the model json file')
parser.add_argument('-closeup', action='store_true', help='Zoom in the plot')
args = parser.parse_args()


#* Define path
data_path = args.out_file
output_dir = os.path.dirname(data_path)

#* Load the A-scan data
f = h5py.File(data_path, 'r')
nrx = f.attrs['nrx']
for rx in range(nrx):
    data, dt = get_output_data(data_path, (rx+1), 'Ez')

time = np.arange(len(data)) * dt  / 1e-9 # [ns]


#* Load the model json file
with open(args.model_json, 'r') as f:
    model = json.load(f)


#* Physical constants
c0 = 299792458  # Speed of light in vacuum [m/s]



#* Calculate the two-way travel path length
optical_path_length = []
boundary_names = []
boundaries = model['boundaries']
for boundary in boundaries:
    boundary_names.append(boundary['name'])
    if optical_path_length == []:
        optical_path_length.append(2 * boundary['length'] * np.sqrt(boundary['epsilon_r']))
    else:
        optical_path_length.append(optical_path_length[-1] +  2 * boundary['length'] * np.sqrt(boundary['epsilon_r']))

print(boundary_names)
print('Optical path length:')
print(optical_path_length)
print(' ')


#* Calculate the two-way travel time
two_way_travel_time = [length / c0 /1e-9 for length in optical_path_length] # [ns]
#* Add the initial pulse delay
delay = model['initial_pulse_delay']# [ns]
two_way_travel_time = [t + delay for t in two_way_travel_time]




#* Plot
fig, ax = plt.subplots(subplot_kw=dict(xlabel='Time [ns]', ylabel='Ez normalized field strength'), num='rx' + str(rx),
                            figsize=(20, 10), facecolor='w', edgecolor='w', tight_layout=True)

#* Plot A-scan
ax.plot(time, data, label='A-scan', color='black')

#* Plot the estimated two-way travel time
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i, t in enumerate(two_way_travel_time):
    ax.axvline(t, linestyle='--', label=boundary_names[i], color=colors[i])


plt.xlabel('Time [ns]', fontsize=24)
plt.ylabel('Amplitude', fontsize=24)
#plt.title('Pulse Analysis')
plt.legend(fontsize=20, loc='lower right')
plt.tick_params(labelsize=20)
plt.grid(True)


#* for closeup option
closeup_x_start = 0 #[ns]
closeup_x_end =100 #[ns]
closeup_y_start = -60
closeup_y_end = 60

if args.closeup:
        ax.set_xlim([closeup_x_start, closeup_x_end])
        ax.set_ylim([closeup_y_start, closeup_y_end])
else:
    ax.set_xlim([0, np.amax(time)])


#* Save the plot
if args.closeup:
        fig.savefig(output_dir + '/delay_time' + '_closeup_x' + str(closeup_x_start) \
                + '_' + str(closeup_x_end) + 'y' + str(closeup_y_end) +  '.png'
                ,dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
else:
    fig.savefig(output_dir + '/delay_time' + '.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

plt.show()
