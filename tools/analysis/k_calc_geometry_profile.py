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
loss_tangent_profile = np.mean(losstangent_map, axis=1)
print(f"Calculated depth profiles with length {permittivity_profile.shape[0]}")


# Plotting
# Permittivity map and profile
fig, ax = plt.subplots(
    nrows=1, # 縦
    ncols=2, # 横
    width_ratios=[3, 1],
    height_ratios=[1],
    figsize=(12, 8)
)

im = ax[0].imshow(permittivity_map,
            extent=[0, permittivity_map.shape[1]*spatial_grid, permittivity_map.shape[0]*spatial_grid, 0],
            interpolation='nearest', aspect='auto', cmap='jet', vmin=1, vmax=6)
ax[0].set_xlabel('X [m]', size=14)
ax[0].set_ylabel('Y [m]', size=14)
ax[0].tick_params(labelsize=12)
ax[0].grid()
# coloarbar
delvider = axgrid1.make_axes_locatable(ax[0])
cax = delvider.append_axes('right', size='5%', pad=0.1)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Relative permittivity', size=14)
cbar.ax.tick_params(labelsize=12)

ax[1].plot(permittivity_profile, np.arange(permittivity_profile.shape[0]) * spatial_grid) # time in ns
ax[1].set_xlabel('Relative permittivity', size=14)
ax[1].set_ylabel('Depth (m)', size=14)
ax[1].set_ylim(permittivity_profile.shape[0] * spatial_grid, 0) # time in ns
ax[1].tick_params(labelsize=12)
ax[1].grid()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'permittivity_map.png'), dpi=300)
plt.savefig(os.path.join(output_dir, 'permittivity_map.pdf'), format='pdf', dpi=300)
plt.show()


# Permittivity profile only
plt.figure(figsize=(4, 8), facecolor='w', edgecolor='w')
plt.plot(permittivity_profile, np.arange(permittivity_profile.shape[0]) * spatial_grid) # time in ns
plt.xlabel('Relative permittivity', size=14)
plt.ylabel('Depth (m)', size=14)
plt.ylim(permittivity_profile.shape[0] * spatial_grid, 0) # time in ns
plt.tick_params(labelsize=12)
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'permittivity_profile.png'), dpi=300)
plt.savefig(os.path.join(output_dir, 'permittivity_profile.pdf'), format='pdf', dpi=300)
plt.close()


# Conductivity map and profile
fig, ax = plt.subplots(
    nrows=1, # 縦
    ncols=2, # 横
    width_ratios=[3, 1],
    height_ratios=[1],
    figsize=(12, 8)
)
im = ax[0].imshow(conductivity_map,
            extent=[0, conductivity_map.shape[1]*spatial_grid, conductivity_map.shape[0]*spatial_grid, 0],
            interpolation='nearest', aspect='auto', cmap='jet')
ax[0].set_xlabel('X [m]', size=14)
ax[0].set_ylabel('Y [m]', size=14)
ax[0].tick_params(labelsize=12)
ax[0].grid()
# coloarbar
delvider = axgrid1.make_axes_locatable(ax[0])
cax = delvider.append_axes('right', size='5%', pad=0.1)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Conductivity (S/m)', size=14)
cbar.ax.tick_params(labelsize=12)

ax[1].plot(conductivity_profile, np.arange(conductivity_profile.shape[0]) * spatial_grid) # time in ns
ax[1].set_xlabel('Conductivity (S/m)', size=14)
ax[1].set_ylabel('Depth (m)', size=14)
ax[1].set_ylim(conductivity_profile.shape[0] * spatial_grid, 0) # time in ns
ax[1].tick_params(labelsize=12)
ax[1].grid()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'conductivity_map.png'), dpi=300)
plt.savefig(os.path.join(output_dir, 'conductivity_map.pdf'), format='pdf', dpi=300)
plt.show()

# Conductivity profile only
plt.figure(figsize=(4, 8), facecolor='w', edgecolor='w')
plt.plot(conductivity_profile, np.arange(conductivity_profile.shape[0]) * spatial_grid) # time in ns
plt.xlabel('Conductivity (S/m)', size=14)
plt.ylabel('Depth (m)', size=14)
plt.ylim(conductivity_profile.shape[0] * spatial_grid, 0) # time in ns
plt.tick_params(labelsize=12)
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'conductivity_profile.png'), dpi=300)
plt.savefig(os.path.join(output_dir, 'conductivity_profile.pdf'), format='pdf', dpi=300)
plt.close()


# Loss tangent map and profile
fig, ax = plt.subplots(
    nrows=1, # 縦
    ncols=2, # 横
    width_ratios=[3, 1],
    height_ratios=[1],
    figsize=(12, 8)
)
im = ax[0].imshow(losstangent_map,
            extent=[0, losstangent_map.shape[1]*spatial_grid, losstangent_map.shape[0]*spatial_grid, 0],
            interpolation='nearest', aspect='auto', cmap='jet')
ax[0].set_xlabel('X [m]', size=14)
ax[0].set_ylabel('Y [m]', size=14)
ax[0].tick_params(labelsize=12)
ax[0].grid()
# coloarbar
delvider = axgrid1.make_axes_locatable(ax[0])
cax = delvider.append_axes('right', size='5%', pad=0.1)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Loss tangent', size=14)
cbar.ax.tick_params(labelsize=12)

ax[1].plot(loss_tangent_profile, np.arange(loss_tangent_profile.shape[0]) * spatial_grid) # time in ns
ax[1].set_xlabel('Loss tangent', size=14)
ax[1].set_ylabel('Depth (m)', size=14)
ax[1].set_ylim(loss_tangent_profile.shape[0] * spatial_grid, 0) # time in ns
ax[1].tick_params(labelsize=12)
ax[1].grid()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'losstangent_map.png'), dpi=300)
plt.savefig(os.path.join(output_dir, 'losstangent_map.pdf'), format='pdf', dpi=300)
plt.show()

# Loss tangent profile only
plt.figure(figsize=(4, 8), facecolor='w', edgecolor='w')
plt.plot(loss_tangent_profile, np.arange(loss_tangent_profile.shape[0]) * spatial_grid) # time in ns
plt.xlabel('Loss tangent', size=14)
plt.ylabel('Depth (m)', size=14)
plt.ylim(loss_tangent_profile.shape[0] * spatial_grid, 0) # time in ns
plt.tick_params(labelsize=12)
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'losstangent_profile.png'), dpi=300)
plt.savefig(os.path.join(output_dir, 'losstangent_profile.pdf'), format='pdf', dpi=300)
plt.close()


"""
def plot_geometry_from_json(json_file, use_closeup=False, x_start=None, x_end=None, y_start=None, y_end=None):
    # Load geometry settings
    with open(json_file) as f:
        params = json.load(f)

    # Load material epsilon list
    material_path = params['geometry_settings']['material_file']
    epsilon_list = []
    with open(material_path, 'r') as mf:
        for line in mf:
            values = line.split()
            if len(values) >= 2:
                epsilon_list.append(float(values[1]))

    # Read HDF5 data
    h5_file = params['geometry_settings']['h5_file']
    h5f = h5py.File(h5_file, 'r')
    data = h5f['data'][:, :, 0]
    data = np.rot90(data)
    z_num, x_num = data.shape

    # Build epsilon map
    epsilon_map = np.zeros((z_num, x_num), dtype=float)
    for i in tqdm(range(x_num), desc=f"Processing {os.path.basename(json_file)}"):
        for j in range(z_num):
            idx = int(data[j, i])
            epsilon_map[j, i] = epsilon_list[idx]

    # Plot
    spatial_grid = params['geometry_settings']['grid_size']
    figsize_ratio = z_num / x_num
    fig = plt.figure(figsize=(10, 10 * figsize_ratio), tight_layout=True)
    ax = fig.add_subplot(111)
    extent = [0, x_num * spatial_grid, z_num * spatial_grid, 0]
    im = ax.imshow(epsilon_map, extent=extent, cmap='jet', vmin=1, vmax=6) # jet or binary

    if use_closeup:
        ax.set_xlim(x_start, x_end)
        ax.set_ylim(y_end, y_start)

    ax.set_xlabel('x [m]', size=20)
    ax.set_ylabel('y [m]', size=20)
    ax.tick_params(labelsize=16)

    divider = axgrid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(r"$\varepsilon_r$", size=20)
    cbar.ax.tick_params(labelsize=16)

    # Save outputs
    output_dir = os.path.dirname(h5_file)
    if use_closeup:
        out_dir = os.path.join(output_dir, 'geometry_closeup')
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(json_file))[0]
        png_name = f"{base}_{x_start}-{x_end}_{y_start}-{y_end}.png"
        pdf_name = png_name.replace('.png', '.pdf')
        fig.savefig(os.path.join(out_dir, png_name), dpi=300)
        fig.savefig(os.path.join(out_dir, pdf_name), format='pdf', dpi=300)
    else:
        fig.savefig(os.path.join(output_dir, 'epsilon_map.png'), dpi=300)
        fig.savefig(os.path.join(output_dir, 'epsilon_map.pdf'), format='pdf', dpi=300)

    plt.close(fig)


def derive_geometry_json_from_out(out_path):
    # .out is in .../results_hX_wY/base.out
    parent = os.path.dirname(out_path)
    grand = os.path.dirname(parent)
    base = os.path.splitext(os.path.basename(out_path))[0]
    return os.path.join(grand, f"{base}.json")


def main():
    print("Select mode:")
    print("1: Single plot")
    print("2: Batch plot")
    mode = input("Enter choice [1/2]: ").strip()

    if mode == '2':
        mapping_path = input("Enter path to JSON mapping of .out files: ").strip()
        if not os.path.exists(mapping_path):
            print("Error: mapping JSON not found.")
            return
        with open(mapping_path) as mf:
            mapping = json.load(mf)

        zoom_input = input("Use closeup? (y/n) [n]: ").strip().lower()
        use_close = zoom_input == 'y'
        if use_close:
            x_start = float(input("Enter x-axis lower limit: "))
            x_end   = float(input("Enter x-axis upper limit: "))
            y_start = float(input("Enter y-axis lower limit: "))
            y_end   = float(input("Enter y-axis upper limit: "))
        else:
            x_start = x_end = y_start = y_end = None

        for key, out_path in mapping.items():
            geo_json = derive_geometry_json_from_out(out_path)
            if not os.path.exists(geo_json):
                print(f"Geometry JSON not found for {key}: {geo_json}")
                continue
            print(f"Processing {key} -> {geo_json}")
            plot_geometry_from_json(geo_json, use_close, x_start, x_end, y_start, y_end)

    else:
        json_file = input("Enter geometry JSON file path: ").strip()
        zoom_input = input("Use closeup? (y/n) [n]: ").strip().lower()
        use_close = zoom_input == 'y'
        if use_close:
            x_start = float(input("Enter x-axis lower limit: "))
            x_end   = float(input("Enter x-axis upper limit: "))
            y_start = float(input("Enter y-axis lower limit: "))
            y_end   = float(input("Enter y-axis upper limit: "))
            plot_geometry_from_json(json_file, True, x_start, x_end, y_start, y_end)
        else:
            plot_geometry_from_json(json_file)

if __name__ == "__main__":
    main()
"""