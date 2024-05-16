import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib import collections as mc


class make_plot_point_geometry:
    def __init__(self, json):
        self.json = json


    def load_params_from_json(self):
        #* load jason data
        with open (self.json) as f:
            params = json.load(f)

        self.thickness = np.array(params['layering_structure_info']['layer_thickness'][1:]) # [m], don't include vacuum layer
        self.permittivity = np.array(params['layering_structure_info']['internal_permittivity'][1:]) # don't include vacuum layer
        return self.thickness, self.permittivity

    #* make depth array
    def make_depth_array(self, thickness):
        self.thickness = thickness
        self.depth = np.cumsum(self.thickness) # [m]
        self.depth_plot = np.zeros(len(self.depth) * 2)
        self.depth_plot[1::2] = self.depth
        self.depth_plot[2::2] = self.depth[:-1]
        return self.depth_plot

    def make_permittivity_array(self, permittivity):
        self.permittivity = permittivity
        self.permittivity_plot = np.zeros(len(self.permittivity) * 2)
        self.permittivity_plot[::2] = self.permittivity
        self.permittivity_plot[1::2] = self.permittivity
        return self.permittivity_plot

    def make_plot_point_array(self, depth, permittivity):
        self.depth = depth
        self.permittivity = permittivity
        self.plot_points = np.array([[self.permittivity[i], self.depth[i]] for i in range(len(self.depth))]) # not collections.LineCollection
        return self.plot_points


class make_plot_point_estimation:
    def __init__(self, json):
        self.json = json

    def load_params_from_json(self):
        #* load jason data
        with open (self.json) as f:
            params = json.load(f)
        self.t0 = np.array(params['Vrms_estimation']['t0_results']) * 10**(-9) # [s]
        self.Vrms = np.array(params['Vrms_estimation']['Vrms_results']) * 3e8 # [m/s]
        self.epsilon_r = np.array(params['permittivity_structure_estimation']['epsilon_r_results']) # []
        return self.t0, self.Vrms, self.epsilon_r

    def make_depth_array(self, t0, Vrms):
        self.t0 = t0
        self.Vrms = Vrms
        self.depth = np.zeros(len(self.t0) * 2)
        self.depth[1::2] = self.t0 * self.Vrms / 2
        self.depth[2::2] = self.t0[:-1] * self.Vrms[:-1] / 2
        return self.depth

    def make_permittivity_array(self, epsilon):
        self.epsilon = epsilon
        self.epsilon_plot = np.zeros(len(self.epsilon) * 2)
        self.epsilon_plot[::2] = self.epsilon
        self.epsilon_plot[1::2] = self.epsilon
        return self.epsilon_plot

    def make_plot_point_array(self, depth, epsilon_r):
        self.depth = depth
        self.epsilon_r = epsilon_r
        self.plot_points = np.array([[self.epsilon_r[i], self.depth[i]] for i in range(len(self.depth))]) # not collections.LineCollection
        return self.plot_points

#* example usage
def run_geometry(json_path):
    call_class = make_plot_point_geometry(json_path)
    thickness, permittivity = call_class.load_params_from_json()
    depth = call_class.make_depth_array(thickness)
    permittivity = call_class.make_permittivity_array(permittivity)
    plot_points = call_class.make_plot_point_array(depth, permittivity)
    return mc.LineCollection([plot_points], linewidths=2)

def run_estimation(json_path):
    call_class = make_plot_point_estimation(json_path)
    t0, Vrms, epsilon_r = call_class.load_params_from_json()
    depth = call_class.make_depth_array(t0, Vrms)
    epsilon_r = call_class.make_permittivity_array(epsilon_r)
    plot_points = call_class.make_plot_point_array(depth, epsilon_r)
    return mc.LineCollection([plot_points], linewidths=2)


#* run the tool
collections_model = run_geometry('kanda/domain_50x100/no_loss/geometry/geometry.json')
collections_40m = run_estimation('kanda/domain_50x100/no_loss/multi_CMP_int1_40m/multi_int1.json')
collections_20m = run_estimation('kanda/domain_50x100/no_loss/multi_CMP_int1_20m/muti_CMP_int1_20m.json')
collections_10m = run_estimation('kanda/domain_50x100/no_loss/multi_CMP_int1_10m/muti_CMP_int1_10m.json')


arrays = [collections_model, collections_40m, collections_20m, collections_10m]
clors = ['k', 'c', 'm', 'y']
line_styles = ['-', '--', '-.', ':']
labels = ['Model', '40m', '20m', '10m']

#* plot
fig = plt.figure(figsize=(10, 8), tight_layout=True)
ax = fig.add_subplot(111)

fontsize_large = 20
fontsize_medium = 18
fontsize_small = 16

#ax.add_collection(collections_model, 'o-')
for i in range(len(arrays)):
    lines = arrays[i]
    ax.add_collection(lines)
    lines.set_color(clors[i])
    lines.set_linestyle(line_styles[i])
    lines.set_label(labels[i])

ax.set_xlabel('Dielectric constant', fontsize=fontsize_medium)
ax.set_ylabel('Depth [m]', fontsize=fontsize_medium)
ax.set_title('Dielectric structure', fontsize=fontsize_large)

ax.autoscale()
ax.set_xlim(1, 15)
ax.legend(fontsize=fontsize_medium, loc = 'lower right')
ax.tick_params(labelsize=fontsize_medium)
ax.grid()
plt.gca().invert_yaxis()

plt.show()