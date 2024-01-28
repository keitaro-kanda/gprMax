import numpy as np
import argparse
import json
import os


from tqdm import tqdm

#* Parse command line arguments
parser = argparse.ArgumentParser(description='Estimating internal velocity from t0 and Vrms',
                                 usage='cd gprMax; python -m tools.estimate_internal_velocity jsonfile')
parser.add_argument('jsonfile', help='json file path')
args = parser.parse_args()


#* load jason data
with open (args.jsonfile) as f:
    params = json.load(f)


#* set physical constants
c = 299792458 # [m/s], speed of light in vacuum


#* load parameters from json file
Vrms = np.array(params['V_RMS']) * c # [m/s], Vrms in each layers
t0 = np.array(params['tau_ver']) # [s], t0 in each layers


#* calculate t0 in vacuum
t0_vacuum = params['antenna_height'] * 2 / c

#* estimate internal velocity
internal_velocities = []

for i in tqdm(range (len(Vrms)), desc='calculating Vint'):
    if i ==0:
        Vint = np.sqrt( \
            (Vrms[i]**2 * t0[i] - c**2 * t0_vacuum) \
            / (t0[i] - t0_vacuum) \
        )
    if i >0:
        Vint = np.sqrt( \
            (Vrms[i]**2 * t0[i] -    Vrms[i-1]**2 * t0[i-1] ) \
            / (t0[i] - t0[i-1]) \
        )
    internal_velocities.append(Vint)

internal_velocities = np.array(internal_velocities)

#* estimate internal permittivity
epsilon_r = c**2 / internal_velocities**2

# normalize interna; velocity
internal_velocities = internal_velocities / c

#* combine t0 and Vrms
layer_num = np.arange(1, len(Vrms)+1)
subsurface_structure = np.column_stack((layer_num, internal_velocities, epsilon_r))


#* make output dir
output_dir_path = os.path.dirname(params['out_file']) + '/subsurface_structure'
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)

#* save data
fmt = ['%d'] + ['%0.15f'] * (subsurface_structure.shape[1] - 1)
np.savetxt(output_dir_path + '/subsurface_structure.txt', subsurface_structure,
        fmt=fmt, delimiter=',', header='layer number, internal velocity [/c], epsilon_r')