#!/usr/bin/env python3
"""
Plot relative permittivity, conductivity, and loss tangent from
gprMax geometry files (.h5 + materials.txt).

Axis convention:
  - Vertical axis: depth below surface [m]. Surface=0, subsurface>0, vacuum<0.
  - Surface position determined from JSON ground_depth
    (distance from box top to surface).
  - Vacuum region (depth < 0) is included; a dashed white line marks the surface.

Dispersive / non-dispersive handling (automatic):
  - Materials without #add_dispersion_debye have deps=tau=0
    and reduce to the static (non-dispersive) formula.
  - Multi-pole Debye is supported: poles are stored as a list
    and summed at the target frequency.
"""
import os
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
from tqdm import tqdm
import sys


# =============================================================================
# Input
# =============================================================================
json_file = input("Enter geometry JSON file path: ").strip()
if not os.path.exists(json_file):
    sys.exit("Error: Geometry JSON file not found.")

freq = input("Enter GPR frequency (Hz): ").strip()
if freq == '':
    sys.exit("Error: Frequency is required.")
freq = float(freq)

output_basename = 'geometry_plot'
output_dir = os.path.join(os.path.dirname(json_file), output_basename)
os.makedirs(output_dir, exist_ok=True)

with open(json_file) as f:
    params = json.load(f)

geo = params['geometry_settings']
spatial_grid  = geo['grid_size']    # [m]
domain_x      = geo['domain_x']     # [m]
domain_z      = geo['domain_z']     # [m]  (total height in Y direction)
ground_depth  = geo['ground_depth'] # [m]  distance from box top to surface

# Depth-axis limits (surface-referenced)
depth_top    = -ground_depth              # negative (vacuum region)
depth_bottom =  domain_z - ground_depth  # positive (deepest subsurface)

print(f"ground_depth = {ground_depth} m")
print(f"Depth axis: {depth_top:.3f} m (vacuum top) to {depth_bottom:.3f} m (bottom)")


# =============================================================================
# Load materials.txt (multi-pole Debye supported)
# =============================================================================
material_path = geo['material_file']

mat_names = []   # ordered list matching h5 integer indices
mat_props = {}   # name -> {eps, sigma, poles: [(deps,tau),...]}

with open(material_path, 'r') as mf:
    for line in mf:
        line = line.strip()
        if line.startswith('#material:'):
            v = line.split()
            name = v[5]
            eps_v = 1.0 if name == 'pec' else float(v[1])
            sig_v = 0.0 if name == 'pec' else float(v[2])
            mat_props[name] = {'eps': eps_v, 'sigma': sig_v, 'poles': []}
            mat_names.append(name)

        elif line.startswith('#add_dispersion_debye:'):
            # Syntax: #add_dispersion_debye: N  De1 tau1  [De2 tau2 ...]  name
            v = line.split()
            name  = v[-1]
            npoles = int(v[1])
            poles = []
            for p in range(npoles):
                de  = float(v[2 + p * 2])
                tau = float(v[3 + p * 2])
                poles.append((de, tau))
            if name in mat_props:
                mat_props[name]['poles'] = poles

epsilon_list      = np.array([mat_props[n]['eps']   for n in mat_names])
conductivity_list = np.array([mat_props[n]['sigma'] for n in mat_names])
poles_list        = [mat_props[n]['poles']           for n in mat_names]

has_dispersion = any(len(p) > 0 for p in poles_list)
print(f"Loaded {len(mat_names)} materials. Dispersion: {has_dispersion}")


# =============================================================================
# Load HDF5 geometry
# =============================================================================
h5_file = geo['h5_file']
with h5py.File(h5_file, 'r') as h5f:
    print(f"HDF5 data shape: {h5f['data'].shape}")
    geometry_data = h5f['data'][:, :, 0]

# h5 data[:,:,0] has shape (nx, ny): axis-0 = x, axis-1 = y (y=0 at box top).
# We need shape (ny, nx): axis-0 = y (row 0 = box top = vacuum top),
#                          axis-1 = x.
# rot90(k=-1) = clockwise 90 deg achieves this without flipping the y axis:
#   output[row, col] = input[col, row]  →  row=0 maps to y=0 (vacuum top)  ✓
# rot90(k=+1, default) would give output[row,col] = input[col, ny-1-row],
#   making row=0 map to y=ny-1 (box bottom) — the bug that caused y-inversion.
geometry_data = np.rot90(geometry_data, k=-1)
z_num, x_num = geometry_data.shape
print(f"Geometry map shape (rows=y/depth, cols=x): {geometry_data.shape}")


# =============================================================================
# Build property maps
# =============================================================================
epsilon_0 = 8.854187817e-12
omega     = 2 * np.pi * freq

permittivity_map = np.zeros((z_num, x_num), dtype=float)
conductivity_map = np.zeros((z_num, x_num), dtype=float)
losstangent_map  = np.zeros((z_num, x_num), dtype=float)

for i in tqdm(range(x_num), desc="Building property maps"):
    for j in range(z_num):
        idx    = int(geometry_data[j, i])
        eps_inf = epsilon_list[idx]
        sigma   = conductivity_list[idx]
        poles   = poles_list[idx]

        # Evaluate multi-pole Debye at target freq (no poles -> static / non-dispersive)
        eps_real      = eps_inf
        eps_imag_disp = 0.0
        for (de, tau) in poles:
            denom          = 1.0 + (omega * tau) ** 2
            eps_real      += de / denom
            eps_imag_disp += de * omega * tau / denom

        eps_imag_total = eps_imag_disp + sigma / (omega * epsilon_0)

        permittivity_map[j, i] = eps_real
        conductivity_map[j, i] = sigma
        losstangent_map[j, i]  = eps_imag_total / (eps_real + 1e-30)

permittivity_profile = np.mean(permittivity_map, axis=1)
conductivity_profile = np.mean(conductivity_map, axis=1)
losstangent_profile  = np.mean(losstangent_map,  axis=1)
print("Property maps and depth profiles computed.")


# =============================================================================
# Plot settings
# =============================================================================
colors       = ['jet', 'magma', 'viridis']
names        = ['Relative permittivity', 'Conductivity [S/m]', 'Loss tangent']
output_names = ['Permittivity', 'Conductivity', 'Losstangent']
disp_tag     = ' (dispersive)' if has_dispersion else ' (non-dispersive)'

# imshow extent: [x_left, x_right, depth_bottom, depth_top]
# imshow treats the 3rd value as "bottom" and 4th as "top", so placing
# depth_bottom (positive, deep) as 3rd gives a top-to-bottom increasing axis.
extent_map = [0, x_num * spatial_grid, depth_bottom, depth_top]

# Depth array for profile plots (cell centres)
depth_axis = depth_top + (np.arange(z_num) + 0.5) * spatial_grid


# =============================================================================
# Plot helpers
# =============================================================================
def _add_surface_line(ax):
    """Add a dashed white line at the surface (depth = 0)."""
    ax.axhline(0.0, color='white', linewidth=1.2, linestyle='--', label='Surface')


def plot_map_profile(map_data, profile_data, idx):
    """Plot 2D map (left) and depth profile (right) side by side, then save."""
    map_aspect = map_data.shape[0] / map_data.shape[1]

    fig, ax = plt.subplots(
        nrows=1, ncols=2,
        width_ratios=[3, 1],
        figsize=(12, 9 * map_aspect)
    )

    # ---- Left: 2D map ----
    imshow_kwargs = dict(
        extent=extent_map,
        interpolation='nearest',
        aspect='auto',
        cmap=colors[idx],
    )
    if idx == 0:
        imshow_kwargs.update(vmin=1, vmax=6)

    im = ax[0].imshow(map_data, **imshow_kwargs)
    _add_surface_line(ax[0])

    ax[0].set_title(names[idx] + disp_tag + f'  @ {freq:.3e} Hz', size=14)
    ax[0].set_xlabel('X [m]', size=14)
    ax[0].set_ylabel('Depth [m]', size=14)
    ax[0].set_ylim(depth_bottom, depth_top)   # deep at bottom (positive), vacuum at top (negative)
    ax[0].tick_params(labelsize=12)
    ax[0].grid(alpha=0.3)

    divider = axgrid1.make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(names[idx], size=14)
    cbar.ax.tick_params(labelsize=12)

    # ---- Right: depth profile ----
    ax[1].plot(profile_data, depth_axis)
    ax[1].axhline(0.0, color='gray', linewidth=1.0, linestyle='--')  # surface
    ax[1].set_xlabel(names[idx], size=12)
    ax[1].set_ylabel('Depth [m]', size=12)
    ax[1].set_ylim(depth_bottom, depth_top)
    ax[1].tick_params(labelsize=10)
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    base = output_names[idx]
    plt.savefig(os.path.join(output_dir, base + '_map.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, base + '_map.pdf'), dpi=300, bbox_inches='tight')
    print(f"Saved: {base}_map.png / .pdf")
    plt.show()


def plot_profile(profile_data, idx):
    """Plot depth profile only, then save."""
    fig, ax = plt.subplots(figsize=(4, 8))
    ax.plot(profile_data, depth_axis)
    ax.axhline(0.0, color='gray', linewidth=1.0, linestyle='--', label='Surface')
    ax.set_xlabel(names[idx], size=14)
    ax.set_ylabel('Depth [m]', size=14)
    ax.set_ylim(depth_bottom, depth_top)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    base = output_names[idx]
    plt.savefig(os.path.join(output_dir, base + '_profile.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, base + '_profile.pdf'), dpi=300, bbox_inches='tight')
    print(f"Saved: {base}_profile.png / .pdf")
    plt.close()


# =============================================================================
# Run
# =============================================================================
for idx, (map_data, profile_data) in enumerate([
    (permittivity_map, permittivity_profile),
    (conductivity_map, conductivity_profile),
    (losstangent_map,  losstangent_profile),
]):
    plot_map_profile(map_data, profile_data, idx)
    plot_profile(profile_data, idx)