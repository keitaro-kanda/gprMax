import numpy as np
import argparse
import json
import os


from tqdm import tqdm


#* Parse command line arguments
parser = argparse.ArgumentParser(description='Processing Su method',
                                 usage='cd gprMax; python -m tools.estimate_Vrms_theory jsonfile')
parser.add_argument('jsonfile', help='json file path')
args = parser.parse_args()


#* load jason data
with open (args.jsonfile) as f:
    params = json.load(f)


c = 299792458 # [m/s], speed of light in vacuum

internal_permittivity = np.array(params['internal_permittivity'])
internal_velocity = c / np.sqrt(internal_permittivity) # [m/s], interval velocity in each layers

layer_thickness = np.array(params['layer_thickness']) # [m], thickness of each layers

#* estimate theoretical value of t0
t0_theory = []
internal_2way_time = 2 * layer_thickness / internal_velocity # [s]
for i in range (len(internal_velocity)):
    #t0 = 2 / c + np.sum(internal_2way_time[:i+1])
    t0 = np.sum(internal_2way_time[:i+1])
    t0_theory.append(t0)


#* estimate theoretical value of Vrms
Vrms_theory = []
bunbo = 2 * layer_thickness * internal_velocity
for i in range (len(internal_velocity)):
    Vrms = np.sqrt(
        np.sum(bunbo[:i+1]) / t0_theory[i]
    )
    Vrms  = Vrms / c # normalize
    Vrms_theory.append(Vrms)


#* estimate mu4 in [castle, 1994] eq.(4)
mu4 = []
bunbo_mu4 = internal_2way_time * internal_velocity**4
for i in range (len(internal_velocity)):
    mu_4th_degree = \
        np.sum(bunbo_mu4[:i+1])\
        / t0_theory[i]
    mu4.append(mu_4th_degree)

t0_theory = np.array(t0_theory) + params['transmitting_delay']

#* combine t0 and Vrms
theoretical_estimation = np.column_stack((t0_theory, Vrms_theory, mu4, internal_permittivity))
print(theoretical_estimation)

#* make output dir
data_dir_path = os.path.dirname(params['data'])
output_dir_path = data_dir_path + '/Vrms'
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)

#* save as txt
np.savetxt(output_dir_path+'theoretical_estimation.txt', theoretical_estimation, delimiter=',')
