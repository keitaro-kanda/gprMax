"""
This tools is used to calculate the Vrms and t0 (2-way vertical travel time) from the geometry information.
The geometry information is stored in the json file. The json file should have the following structure:
{
    "layering_structure_info": {},
    "antenna_settings": {},
}
"""
import numpy as np
import json



class calc_Vrms:
    def __init__(self, jsonfile_path, c=3e8):
        self.jsonfile_path = jsonfile_path
        self.c = c

    def load_params_from_json(self):
        with open(self.jsonfile_path) as f:
            params = json.load(f)
        
        self.layer_thickness = np.array(params['layering_structure_info']['layer_thickness']) # [m]
        self.internal_permittivity = np.array(params['layering_structure_info']['internal_permittivity']) # []
        self.internal_velovity = np.array(self.c/ np.sqrt(self.internal_permittivity)) # [m/s]

        return self.layer_thickness, self.internal_permittivity, self.internal_velovity
    
    def calc_t0(self, layer_thickness, internal_velovity):
        #* calculate t0 [s]
        self.t0 = []
        for i in range(len(layer_thickness)):
            if i == 0:
                self.t0.append(2 *layer_thickness[i] / internal_velovity[i])
            else:
                self.t0.append(2 * layer_thickness[i] / internal_velovity[i] + self.t0[i-1])
        return np.array(self.t0)
    
    def calc_Vrms(self, layer_thickness, internal_velocity, t0):
        #* calculate Vrms [/c]
        self.Vrms = []
        Vrms_bunbo = []
        for i in range(len(layer_thickness)):
            if i == 0:
                Vrms_bunbo.append(2 * layer_thickness[i] * internal_velocity[i])
                self.Vrms.append(
                    np.sqrt(
                        (2 * layer_thickness[i] * internal_velocity[i]) / (t0[i])
                    ))
            else:
                Vrms_bunbo.append(2 * layer_thickness[i] * internal_velocity[i])
                self.Vrms.append(
                    np.sqrt(
                        np.sum(Vrms_bunbo) / t0[i]
                    ) / self.c)
        return self.Vrms



#* example usage
"""
jsonfile_path = 'kanda/domain_50x100/no_loss/geometry/geometry.json'
calc_Vrms = calc_Vrms_from_geometry(jsonfile_path)
layer_thickness, internal_permittivity, internal_velovity = calc_Vrms.load_params_from_json()
t0 = calc_Vrms.calc_t0(layer_thickness, internal_velovity)
Vrms = calc_Vrms.calc_Vrms(layer_thickness, internal_velovity, t0)

print('layer_thickness:', layer_thickness)
print('internal_permittivity:', internal_permittivity)
print('internal_velovity:', internal_velovity)
print('t0:', t0)
print('Vrms:', Vrms)
"""
