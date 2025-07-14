"""
This tool is used to extract observation data from the B-scan data already made and make new B-scan.
Input mergedout file must be B-scan observation data.
"""
import numpy as np
import json
import os
import h5py
from tools.outputfiles_merge import get_output_data
import argparse



#* Parse command line arguments
parser = argparse.ArgumentParser(usage='cd gprMax; python -m tools.extract_Bscan jsonfile')
parser.add_argument('jsonfile', help='json file path')
args = parser.parse_args()


#* load jason data
with open (args.jsonfile) as f:
    params = json.load(f)


#* load original B-scan data
original_data_path = params['original_info']['original_data']
data, dt = get_output_data(original_data_path, 1, 'Ez')
print(dt)
print(data.shape)

#* antenna parameters of original B-scan data
with open(params['original_info']['original_json_file']) as original_json:
    original_params = json.load(original_json)
original_src_step = original_params['antenna_settings']['src_step'] # [m]
original_rx_step = original_params['antenna_settings']['rx_step'] # [m]
original_src_move_times = original_params['antenna_settings']['src_move_times']


#* antenna_parameters after extraction
extracted_src_start = params['antenna_settings']['src_start'] # [m]
extracted_src_step = params['antenna_settings']['src_step'] # [m]
ectracted_rx_start = params['antenna_settings']['rx_start'] # [m]
extracted_rx_step = params['antenna_settings']['rx_step'] # [m]
extracted_src_move_times = params['antenna_settings']['src_move_times']


#* extraction antenna step ratio (must be integer and >= 1)
src_step_ratio = int(extracted_src_step / original_src_step) # if extracted_src_step = 2 [m], original_src_step = 1 [m], src_step_ratio = 2
rx_step_ratio = int(extracted_rx_step / original_rx_step)


#* extract B-scan data
class extract_Bscan:
    def __init__(self, data):
        self.data = data

    def extract_Bscan(self, src_move_times):
        self.Bscan = np.zeros((self.data.shape[0], src_move_times))
        for i in range(src_move_times):
            self.Bscan[:, i] = (self.data[:, src_step_ratio * i])
        return np.array(self.Bscan)

    def output_Bscan(self, outputfile):
        output_dir = os.path.dirname(args.jsonfile)
        with h5py.File(outputfile, 'w') as f:
            f.create_dataset(output_dir + '/extracted_Bscan', data=self.Bscan)
        return f
    

#* usage
extracted_Bscan = extract_Bscan(data)
extracted_Bscan = extracted_Bscan.extract_Bscan(extracted_src_move_times)
print('extracted_Bscan shape: ', extracted_Bscan.shape)


#* output extracted B-scan data
output_dir = os.path.dirname(args.jsonfile)
np.savetxt(output_dir + '/extracted_Bscan.txt', extracted_Bscan)