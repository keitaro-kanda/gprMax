import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib import collections as mc
from calc_Vrms_from_geometry import calc_Vrms


class make_plot_geometry:
    def __init__(self, json):
        self.json = json
    
    def load_params(self):
        #* load jason data
        with open (self.json) as f:
            params = json.load(f)
        self.t0 = np.array(params['Vrms_estimation']['t0_results']) # [ns]
        self.Vrms = np.array(params['Vrms_estimation']['Vrms_results']) # [/c]
        return self.t0, self.Vrms
    
    def make_t0_array(self, t0):
        self.t0 = t0
        self.t0_plot = np.zeros(len(self.t0) * 2)
        self.t0_plot[1::2] = self.t0
        self.t0_plot[2::2] = self.t0[:-1]
        return self.t0_plot # [ns]
    
    def make_Vrms_array(self, Vrms):
        self.Vrms = Vrms
        self.Vrms_plot = np.zeros(len(self.Vrms) * 2)
        self.Vrms_plot[::2] = self.Vrms
        self.Vrms_plot[1::2] = self.Vrms
        return self.Vrms_plot # [/c]
    
    def make_plot_points(self, t0, Vrms):
        self.t0 = t0
        self.Vrms = Vrms
        self.plot_points = np.array([[self.Vrms[i], self.t0[i]] for i in range(len(self.t0))])
        return self.plot_points


#* usage
def run(json_path):
    call_class = make_plot_geometry(json_path)
    t0, Vrms = call_class.load_params()
    t0_plot = call_class.make_t0_array(t0)
    Vrms_plot = call_class.make_Vrms_array(Vrms)
    plot_points = call_class.make_plot_points(t0_plot, Vrms_plot)
    return mc.LineCollection([plot_points], linewidths=2)

max_40m = run('kanda/domain_50x100/no_loss/multi_CMP_int1_40m/multi_int1.json')
max_20m = run('kanda/domain_50x100/no_loss/multi_CMP_int05_20m/multi_CMP_int05_20m.json')
max_10m = run('kanda/domain_50x100/no_loss/multi_CMP_int026_10.2m/multi_CMP_int026_10.2.json')


#* get theoretical value of Vrms and t0 from geometry
geometry_json = 'kanda/domain_50x100/no_loss/geometry/geometry.json'
def calc_Vrms_geometry(jsonpath):
    calc_Vrms_geometry = calc_Vrms(geometry_json)
    layer_thickness, internal_permittivity, internal_velovity = calc_Vrms_geometry.load_params_from_json()
    t0_theory = calc_Vrms_geometry.calc_t0(layer_thickness, internal_velovity) # [ns], remove t0 in vacuum
    Vrms_theory = calc_Vrms_geometry.calc_Vrms(layer_thickness, internal_velovity, t0_theory) # [/c], remove Vrms in vacuum

    t0_theory = t0_theory[1:]
    Vrms_theory = Vrms_theory[1:]
    t0_plot_geometry = np.zeros(len(t0_theory) * 2)
    t0_plot_geometry[1::2] = t0_theory
    t0_plot_geometry[2::2] = t0_theory[:-1]

    Vrms_plot_geometry = np.zeros(len(Vrms_theory) * 2)
    Vrms_plot_geometry[::2] = Vrms_theory
    Vrms_plot_geometry[1::2] = Vrms_theory

    plot_points_geometry = np.array([[Vrms_plot_geometry[i], t0_plot_geometry[i]*1e9] for i in range(len(t0_plot_geometry))])
    return t0_plot_geometry, Vrms_plot_geometry, mc.LineCollection([plot_points_geometry], linewidths=2)
t0_plot_geometry, Vrms_plot_geometry, geometry = calc_Vrms_geometry(geometry_json)
t0_plot_geometry = t0_plot_geometry * 1e9 # [ns]
print('t0: ', t0_plot_geometry)
print('Vrms: ', Vrms_plot_geometry)



arrays = [geometry, max_40m, max_20m, max_10m]
clors = ['k', 'c', 'm', 'y']
line_styles = ['-', '--', '-.', ':']
labels = ['model', '40m', '20m', '10.2m']


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

#* fill between
Vrms_upper = Vrms_plot_geometry * 1.05
Vrms_lower = Vrms_plot_geometry * 0.95
for i in range(0, len(Vrms_upper), 2):
    ax.fill_betweenx([t0_plot_geometry[i], t0_plot_geometry[i+1]], Vrms_lower[i], Vrms_upper[i], color='grey', alpha=0.3)

ax.set_xlabel('Vrms [/c]', fontsize=fontsize_medium)
ax.set_ylabel('t0 [ns]', fontsize=fontsize_medium)
ax.set_title('Average EM wave velocity structure', fontsize=fontsize_large)

ax.autoscale()
#ax.set_xlim(0, 1)
ax.legend(fontsize=fontsize_medium, loc = 'lower right')
ax.tick_params(labelsize=fontsize_medium)
ax.grid()
plt.gca().invert_yaxis()

plt.show()