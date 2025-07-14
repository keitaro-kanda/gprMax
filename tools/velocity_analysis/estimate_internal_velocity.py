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
with open(params['geometry_settings']['geometry_json']) as f:
    geometry_params = json.load(f)


#* set physical constants
c = 299792458 # [m/s], speed of light in vacuum


#* load parameters from json file
Vrms = np.array(params['Vrms_estimation']['Vrms_results']) * c # [m/s], Vrms in each layers
t0 = np.array(params['Vrms_estimation']['t0_results']) * 10**(-9)# [s], t0 in each layers
print('Vrms: ', Vrms)
print('t0: ', t0)
epsilon_r_model = np.array(geometry_params['layering_structure_info']['internal_permittivity'][1:1+len(Vrms)]) # epsilon_r in each layers, contains air layer


#* calculate t0 in vacuum
vacuum_thickness = params['antenna_settings']['antenna_height'] # [m]
t0_vacuum = vacuum_thickness * 2 / c

#* estimate internal velocity by Dix formula
internal_velocities_Dix = []

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
    internal_velocities_Dix.append(Vint)

internal_velocities_Dix = np.array(internal_velocities_Dix)



#* estimate internal permittivity from layer thickness and layer 2-way travel time
internal_velocities_nonDix = []
for i in range(len(t0)):
    if i == 0:
        #! Vnって命名はヤバす
        Vn = (t0[i] * Vrms[i] - vacuum_thickness * 2) \
            / (t0[i] - t0_vacuum)
    else:
        Vn = (t0[i] * Vrms[i] - t0[i-1] * Vrms[i-1]) \
            / (t0[i] - t0[i-1])
    
    internal_velocities_nonDix.append(Vn) # [m/s]

internal_velocities_nonDix = np.array(internal_velocities_nonDix)



#* estimate internal permittivity
epsilon_r_Dix = c**2 / internal_velocities_Dix**2
epsilon_r_nonDix = c**2 / internal_velocities_nonDix**2

# calcculate relative error
error_Dix = (epsilon_r_Dix - epsilon_r_model) / epsilon_r_model * 100
error_nonDix = (epsilon_r_nonDix - epsilon_r_model) / epsilon_r_model * 100

# normalize interna; velocity
internal_velocities_Dix = internal_velocities_Dix / c # [/c]
internal_velocities_nonDix = internal_velocities_nonDix / c # [/c]
print('internal velocities Dix: ', internal_velocities_Dix)
print('internal epsilon_r Dix: ', epsilon_r_Dix)
print('error Dix: ', error_Dix)

#* combine t0 and Vrms
layer_num = np.arange(1, len(Vrms)+1)
subsurface_structure = np.column_stack((layer_num, internal_velocities_Dix, epsilon_r_Dix, error_Dix,
                                        internal_velocities_nonDix, epsilon_r_nonDix, error_nonDix))


#* make output dir
output_dir_path = os.path.dirname(args.jsonfile) + '/subsurface_structure'
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)

#* save data
fmt = ['%d'] + ['%0.15f'] * (subsurface_structure.shape[1] - 1)
np.savetxt(output_dir_path + '/subsurface_structure.csv', subsurface_structure,
        fmt=fmt, delimiter=',', header='layer number, internal velocity Dix[/c], epsilon_r, error [%], \
        internal velocity non Dix [/c], epsilon_r non Dix, error non Dix [%]')