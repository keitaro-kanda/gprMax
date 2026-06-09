#!/usr/bin/env python3
import os
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
from tqdm import tqdm
import sys


# Input file path of geometry JSON
json_file = input("Enter geometry JSON file path: ").strip()
if not os.path.exists(json_file):
    sys.exit("Error: Geometry JSON file not found.")
# Inpu frequency of GPR
freq = input("Enter GPR frequency (Hz): ").strip()
if freq == '':
    sys.exit("Error: Frequency is required.")
freq = float(freq)


# Output directory
output_basename = 'geometry_plot'
output_dir = os.path.join(os.path.dirname(json_file), output_basename)
os.makedirs(output_dir, exist_ok=True)


# Load geometry settings
with open(json_file) as f:
    params = json.load(f)


# Load parameters from JSON
spatial_grid = params['geometry_settings']['grid_size']


# Load material epsilon list
material_path = params['geometry_settings']['material_file']
epsilon_list = []
conductivity_list = []
with open(material_path, 'r') as mf:
    for line in mf:
        values = line.split()
        if len(values) >= 3:
            epsilon_list.append(float(values[1]))
            conductivity_list.append(float(values[2]))

# Read HDF5 data
h5_file = params['geometry_settings']['h5_file']
h5f = h5py.File(h5_file, 'r')
print(f"Opened HDF5 file: {h5f['data'].shape}")

# Extract and rotate data
geometry_data = h5f['data'][:, :, 0]
geometry_data = np.rot90(geometry_data)
print(f"Extracted geometry data with shape {geometry_data.shape}")

# Build map
z_num, x_num = geometry_data.shape
permittivity_map = np.zeros((z_num, x_num), dtype=float)
conductivity_map = np.zeros((z_num, x_num), dtype=float)
losstangent_map = np.zeros((z_num, x_num), dtype=float)
# Parameters for loss tangent calculation
epsilon_0 = 8.854187817e-12
omega = 2 * np.pi * freq
for i in tqdm(range(x_num), desc="Making geometry maps"):
    for j in range(z_num):
        idx_perm = int(geometry_data[j, i])
        idx_cond = int(geometry_data[j, i])
        permittivity_map[j, i] = epsilon_list[idx_perm]
        conductivity_map[j, i] = conductivity_list[idx_cond]
        losstangent_map[j, i] = conductivity_list[idx_cond] / (omega * epsilon_0 * epsilon_list[idx_perm])


# Calculate mean depth profile
permittivity_profile = np.mean(permittivity_map, axis=1)
conductivity_profile = np.mean(conductivity_map, axis=1)
losstangent_profile = np.mean(losstangent_map, axis=1)
print(f"Calculated depth profiles with length {permittivity_profile.shape[0]}")


# Define function to make map and profile plot
def plot_map_profile(map_data, profile_data, map_type, colors, names, output_names):
    """_summary_
    Args:
        map_data : 2D map data
        profile_data: 1D profile data
        map_type: permmittivity, conducitivity, or losstangent
    """
    idx = None
    if map_type == 'permittivity':
        idx = 0
    elif map_type == 'conductivity':
        idx = 1
    elif map_type == 'losstangent':
        idx = 2

    fig, ax = plt.subplots(
        nrows=1, # 縦
        ncols=2, # 横
        width_ratios=[3, 1],
        height_ratios=[1],
        figsize=(12, 8)
    )

    if idx == 0:
        im = ax[0].imshow(map_data,
                extent=[0, map_data.shape[1]*spatial_grid, map_data.shape[0]*spatial_grid, 0],
                interpolation='nearest', aspect='auto', cmap=colors[idx], vmin=1, vmax=6)
    else:
        im = ax[0].imshow(map_data,
                extent=[0, map_data.shape[1]*spatial_grid, map_data.shape[0]*spatial_grid, 0],
                interpolation='nearest', aspect='auto', cmap=colors[idx])
    ax[0].set_xlabel('X [m]', size=14)
    ax[0].set_ylabel('Y [m]', size=14)
    ax[0].tick_params(labelsize=12)
    ax[0].grid()
    # coloarbar
    delvider = axgrid1.make_axes_locatable(ax[0])
    cax = delvider.append_axes('right', size='5%', pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(names[idx], size=14)
    cbar.ax.tick_params(labelsize=12)

    ax[1].plot(profile_data, np.arange(profile_data.shape[0]) * spatial_grid) # time in ns
    ax[1].set_xlabel(names[idx], size=14)
    ax[1].set_ylabel('Depth (m)', size=14)
    ax[1].set_ylim(profile_data.shape[0] * spatial_grid, 0) # time in ns
    ax[1].tick_params(labelsize=12)
    ax[1].grid()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_names[idx] + 'map.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, output_names[idx] + '_map.pdf'), format='pdf', dpi=300)
    plt.show()


# Define function to plot profile only
def plot_profile(profile_data, map_type, names, output_names):
    """_summary_
    Args:
        map_data : 2D map data
        profile_data: 1D profile data
        map_type: permmittivity, conducitivity, or losstangent
    """
    idx = None
    if map_type == 'permittivity':
        idx = 0
    elif map_type == 'conductivity':
        idx = 1
    elif map_type == 'losstangent':
        idx = 2
    
    plt.figure(figsize=(4, 8), facecolor='w', edgecolor='w')
    plt.plot(profile_data, np.arange(profile_data.shape[0]) * spatial_grid) # time in ns
    plt.xlabel(names[idx], size=14)
    plt.ylabel('Depth (m)', size=14)
    plt.ylim(permittivity_profile.shape[0] * spatial_grid, 0) # time in ns
    plt.tick_params(labelsize=12)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_names[idx] + '_profile.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, output_names[idx] + 'permittivity_profile.pdf'), format='pdf', dpi=300)
    plt.close()


colors = ['jet', 'magma', 'viridis']
names = ['Relative permittivity', 'Conductivity', 'Loss tangent']
output_names = ['Permittivity', 'Conductivity', 'Losstangent']

# Plot
# Permittivity
plot_map_profile(permittivity_map, permittivity_profile, 'permittivity', colors, names, output_names)
plot_profile(permittivity_profile, 'permittivity', names, output_names)
# Conductivity
plot_map_profile(conductivity_map, conductivity_profile, 'conductivity', colors, names, output_names)
plot_profile(conductivity_profile, 'conductivity', names, output_names)
# Loss tangent
plot_map_profile(losstangent_map, losstangent_profile, 'losstangent', colors, names, output_names)
plot_profile(losstangent_profile, 'losstangent', names, output_names)