import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json

#* Parse command line arguments
parser = argparse.ArgumentParser(description='Plots a error graph.',
                                 usage='cd gprMax; python -m tools.plot_error jsonfile')
parser.add_argument('jsonfile', help='json file path')
args = parser.parse_args()


#* load jason data
with open (args.jsonfile) as f:
    params = json.load(f)


#* set physical constants
c = 299792458 # [m/s], speed of light in vacuum


#* true value
t0_true = np.array(params['t0_theory']) * 10**(-9) # [s]
Vrms_true = np.array(params['V_RMS_theory']) * c # [m/s]
thickness_true = np.array(params['layer_thickness'][1:]) # [m], thickness of each layers, don't contains air layer
depth_true = np.cumsum(thickness_true) # [m], depth of each layers, don't contains air layer
epsilon_true = np.array(params['internal_permittivity'][1:]) # epsilon_r in each layers, don't contains air layer
Vn_true = c / np.sqrt(epsilon_true) # [m/s], propagation velocity in each layers



#* estimated value
t0_estimated = np.array(params['tau_ver']) * 10**(-9) # [s]
Vrms_estimated = np.array(params['V_RMS']) * c # [m/s]
depth_estimated = t0_estimated * Vrms_estimated / 2 - params['antenna_height'] # [m], depth of each layers`
thickness_estimated = np.diff(np.insert(depth_estimated, 0, 0)) # [m], thickness of each layers

# load subsurface_structure.txt
subsurface_structure = np.loadtxt(params['subsurface_structure'], delimiter=',')
epsilon_estimated = np.nan_to_num(subsurface_structure[:,2])
Vn_estimated = np.nan_to_num(subsurface_structure[:,1] * c) # [m/s]


#* calculate error
t0_error = (t0_estimated - t0_true) / t0_true * 100
Vrms_error = (Vrms_estimated - Vrms_true) / Vrms_true * 100
depth_error = (depth_estimated - depth_true) / depth_true * 100
thickness_error = (thickness_estimated - thickness_true) / thickness_true * 100
epsilon_error = (epsilon_estimated - epsilon_true) / epsilon_true * 100
Vn_error = (Vn_estimated - Vn_true) / Vn_true * 100



#* plot error
fig, ax = plt.subplots(1, 1, figsize=(15, 15), tight_layout=True)
ax.plot(depth_true, t0_error, marker='.', markersize=10, label=r'$t_0$')
ax.plot(depth_true, Vrms_error, marker='.', markersize=10, label=r'$V_{rms}$')
ax.plot(depth_true, depth_error, marker='.', markersize=10, label='depth')
ax.plot(depth_true, thickness_error, marker='.', markersize=10, label='thickness')
ax.plot(depth_true, Vn_error, marker='.', markersize=10, label=r'$V_n$')
ax.plot(depth_true, epsilon_error, marker='.', markersize=10, label=r'$\varepsilon_r$')

ax.set_xlabel('Depth of subsurface interface [m]', fontsize=20)
ax.set_ylabel('Error to model value [%]', fontsize=20)
ax.set_title('Error of estimated parameters', fontsize=20)
ax.grid(which='both', axis='both', linestyle='--')
ax.tick_params(labelsize=18)
ax.legend(fontsize=18)

ax.set_ylim(-105, 105)


#* save and show
json_dir = os.path.dirname(args.jsonfile)
plt.savefig(os.path.join(json_dir, 'error.png'))
plt.show()


#* save error values as csv
# estimated value
estimated_values = np.array([t0_estimated, Vrms_estimated, depth_estimated, thickness_estimated,
                            epsilon_estimated, Vn_estimated]).T
header = 't0 [s], Vrms [m/s], depth [m], thickness [m], epsilon_r, Vn [m/s]'
np.savetxt(os.path.join(json_dir, 'estimated.csv'), estimated_values, delimiter=',', header=header, comments='')
# error
error_values = np.array([t0_error, Vrms_error, depth_error, thickness_error,
                        epsilon_error, Vn_error]).T
header = 't0 [%], Vrms [%], depth [%], thickness [%], epsilon [%], Vn [%]'
np.savetxt(os.path.join(json_dir, 'error.csv'), error_values, delimiter=',', header=header, comments='')

