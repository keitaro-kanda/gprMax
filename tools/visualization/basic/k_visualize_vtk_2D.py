#!/usr/bin/env python3
"""
Plot relative permittivity, conductivity, and loss tangent from
a gprMax geometry view file (.vti, written by #geometry_view with 'n').

Only the .vti path is required. Everything else is collected automatically:
  - Grid size, extent and origin       : read from the .vti header
  - Material ID -> material name        : read from the <gprMax> block at the end of the .vti
  - Material properties (eps, sigma,    : read from the .in file that wrote this .vti
    Debye poles)                          (#material / #add_dispersion_debye lines),
                                          and from <vti stem>_materials.txt if it exists
  - Surface position                    : detected from the geometry (top of the non-free-space
                                          material in each column, median over columns).
                                          Set SURFACE_Y_M below to override.

Frequency presets (chosen at start-up):
  [1] LPR   : CE-4/LPR CH2 simulations, 250 / 500 / 750 MHz
  [2] LUPEX : LUPEX GPR simulations, 0.5 / 1.25 / 2.0 GHz
  [3] Custom: frequencies entered one by one (MHz)
  Only the relative-permittivity "map + profile" figure at the middle frequency of the list
  is shown on screen; every figure is saved.

Output figures (for each frequency and each property):
  <vti dir>/geometry_plot/              <Prop>_<freq>_map.png/.pdf      2D map + depth profile
  <vti dir>/geometry_plot/maps/         <Prop>_<freq>_maponly.png/.pdf  2D map only
  <vti dir>/geometry_plot/profiles/     <Prop>_<freq>_profile.png/.pdf  depth profile only

Axis convention:
  - Vertical axis: depth below surface [m]. Surface=0, subsurface>0, vacuum<0.
  - Vacuum region (depth < 0) is included.

Dispersive / non-dispersive handling (automatic):
  - Materials without #add_dispersion_debye have deps=tau=0
    and reduce to the static (non-dispersive) formula.
  - Multi-pole Debye is supported: poles are stored as a list
    and summed at the target frequency.
"""
import os
import re
import glob
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
import sys


# =============================================================================
# Settings
# =============================================================================
FREQ_PRESETS = {                 # Hz
    '1': ('LPR',   [250e6, 500e6, 750e6]),     # CE-4/LPR CH2 (250-750 MHz)
    '2': ('LUPEX', [0.5e9, 1.25e9, 2.0e9]),    # LUPEX GPR (0.5-2.0 GHz)
}
SHOW_PLOT = True                 # Show the permittivity "map + profile" figure at the middle frequency
SURFACE_Y_M = None               # Surface height from the bottom of the domain [m]. None: detect automatically


def ask_frequencies():
    """Choose a preset or enter frequencies one by one. Returns (label, [Hz, ...])."""
    print("Select simulation type:")
    print("  [1] LPR   (250, 500, 750 MHz)")
    print("  [2] LUPEX (0.5, 1.25, 2.0 GHz)")
    print("  [3] Custom (enter frequencies one by one)")
    while True:
        ans = input("Enter 1, 2 or 3: ").strip()
        if ans in FREQ_PRESETS:
            return FREQ_PRESETS[ans]
        if ans == '3':
            break
        print("  Please enter 1, 2 or 3.")
    fs = []
    while True:
        a = input(f"Enter frequency #{len(fs) + 1} in MHz (blank to finish): ").strip()
        if not a:
            if fs:
                break
            print("  Enter at least one frequency.")
            continue
        try:
            f = float(a)
        except ValueError:
            print("  Not a number. Try again.")
            continue
        if f <= 0:
            print("  Frequency must be positive.")
            continue
        fs.append(f * 1e6)
    return 'Custom', sorted(set(fs))


def freq_label(f):
    """Human-readable frequency for titles (e.g. '500 MHz', '1.25 GHz')."""
    return f'{f / 1e9:g} GHz' if f >= 1e9 else f'{f / 1e6:g} MHz'


# =============================================================================
# Input
# =============================================================================
vti_file = input("Enter geometry .vti file path: ").strip().strip("'\"").replace("\\ ", " ")
if not os.path.exists(vti_file):
    sys.exit("Error: .vti file not found.")
vti_file = os.path.abspath(vti_file)

sim_label, freqs = ask_frequencies()
show_freq = freqs[len(freqs) // 2]          # middle frequency of the list
print(f"Frequencies ({sim_label}): {', '.join(freq_label(f) for f in freqs)}"
      f"  / shown on screen: {freq_label(show_freq)} (permittivity only)")

output_basename = 'geometry_plot'
output_dir = os.path.join(os.path.dirname(vti_file), output_basename)
output_dir_map = os.path.join(output_dir, 'maps')
output_dir_profile = os.path.join(output_dir, 'profiles')
for d in (output_dir, output_dir_map, output_dir_profile):
    os.makedirs(d, exist_ok=True)


# =============================================================================
# Load .vti (material IDs, grid, material names)
# =============================================================================
def read_vti(path):
    """Read the Material array (cells), spacing, origin and {name: ID} from a gprMax .vti."""
    raw = open(path, 'rb').read()
    head_end = raw.index(b'<AppendedData')
    header = raw[:head_end].decode('utf-8', errors='replace')
    ext = [int(v) for v in re.search(r'WholeExtent="([^"]+)"', header).group(1).split()]
    nx, ny, nz = ext[1] - ext[0], ext[3] - ext[2], ext[5] - ext[4]
    spacing = [float(v) for v in re.search(r'Spacing="([^"]+)"', header).group(1).split()]
    origin = [float(v) for v in re.search(r'Origin="([^"]+)"', header).group(1).split()]
    htype = re.search(r'header_type="([^"]+)"', header)
    hdt = np.dtype('<u8') if (htype and htype.group(1) == 'UInt64') else np.dtype('<u4')

    m = re.search(r'<DataArray type="UInt32" Name="Material" format="appended" offset="(\d+)"', header)
    start = raw.index(b'_', raw.index(b'>', head_end)) + 1 + int(m.group(1))
    nbytes = int(np.frombuffer(raw[start:start + hdt.itemsize], hdt)[0])
    data = np.frombuffer(raw[start + hdt.itemsize:start + hdt.itemsize + nbytes], dtype='<u4')
    material = data.reshape((nz, ny, nx)).transpose(2, 1, 0)      # (nx, ny, nz), VTK has x fastest

    ids = {}
    tail = raw[raw.index(b'</VTKFile>'):].decode('utf-8', errors='replace')
    gm = re.search(r'<gprMax>.*</gprMax>', tail, re.S)
    if gm:
        for el in ET.fromstring(gm.group(0)).findall('Material'):
            ids[el.attrib['name']] = int(el.text)
    if not ids:
        sys.exit("Error: material names not found in the .vti (<gprMax> block missing).")
    return material, spacing, origin, ids


material_3d, spacing, origin, name_to_id = read_vti(vti_file)
spatial_grid = spacing[0]
print(f"VTI: {vti_file}")
print(f"  cells {material_3d.shape[:2]}, grid {spatial_grid} m, origin ({origin[0]}, {origin[1]}) m")
print(f"  materials in vti: {', '.join(name_to_id)}")


# =============================================================================
# Find the .in that wrote this .vti, and load material properties
# =============================================================================
def strip_python_blocks(lines):
    """Drop lines inside #python: ... #end_python: (materials made by Python cannot be read here)."""
    out, inside = [], False
    for line in lines:
        s = line.strip()
        if s.startswith('#python:'):
            inside = True
        elif s.startswith('#end_python:'):
            inside = False
        elif not inside:
            out.append(s)
    return out


def find_input_file(vti_path):
    """Search the .vti folder and up to 2 parent folders for the .in whose #geometry_view wrote this file."""
    target = os.path.normcase(os.path.abspath(vti_path))
    d = os.path.dirname(vti_path)
    dirs = [d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))]
    for folder in dirs:
        for inf in sorted(glob.glob(os.path.join(folder, '*.in'))):
            try:
                lines = strip_python_blocks(open(inf, encoding='utf-8', errors='replace').read().splitlines())
            except OSError:
                continue
            for line in lines:
                if not line.startswith('#geometry_view:'):
                    continue
                v = line.split()
                if len(v) < 12:
                    continue
                # #geometry_view: xs ys zs xf yf zf dx dy dz filename type
                base = os.path.join(os.path.dirname(inf), v[10])
                # gprMax appends the model number when run with -n (e.g. geometry1.vti, geometry2.vti)
                pat = re.escape(os.path.normcase(os.path.abspath(base))) + r'\d*\.vti$'
                if re.match(pat, target):
                    return inf
    cands = glob.glob(os.path.join(d, '*.in'))
    if len(cands) == 1:
        return cands[0]
    return None


def parse_materials(lines, props):
    """#material / #add_dispersion_debye lines -> props[name] = {eps, sigma, poles}"""
    for line in lines:
        if line.startswith('#material:'):
            v = line.split()
            name = v[5]
            eps_v = 1.0 if name == 'pec' else float(v[1])
            sig_v = 0.0 if name == 'pec' else float(v[2])
            props[name] = {'eps': eps_v, 'sigma': sig_v, 'poles': []}
        elif line.startswith('#add_dispersion_debye:'):
            # Syntax: #add_dispersion_debye: N  De1 tau1  [De2 tau2 ...]  name(s)
            v = line.split()
            npoles = int(float(v[1]))
            poles = [(float(v[2 + p * 2]), float(v[3 + p * 2])) for p in range(npoles)]
            for name in v[2 + 2 * npoles:]:          # one or more material names
                if name in props:
                    props[name]['poles'] = poles


mat_props = {'pec': {'eps': 1.0, 'sigma': 0.0, 'poles': []},
             'free_space': {'eps': 1.0, 'sigma': 0.0, 'poles': []}}
sources = []

materials_txt = os.path.splitext(vti_file)[0] + '_materials.txt'
if os.path.exists(materials_txt):
    parse_materials([l.strip() for l in open(materials_txt)], mat_props)
    sources.append(materials_txt)

input_file = find_input_file(vti_file)
if input_file is None:
    try:
        ans = input("Could not find the .in file automatically. Enter .in file path (Enter to skip): ")
    except EOFError:
        ans = ''
    ans = ans.strip().strip("'\"").replace("\\ ", " ")
    input_file = ans if ans else None
if input_file:
    parse_materials(strip_python_blocks(open(input_file, encoding='utf-8', errors='replace').read().splitlines()),
                    mat_props)
    sources.append(input_file)
print(f"  material properties from: {', '.join(sources) if sources else '(built-in only)'}")

# Averaged materials made by gprMax (dielectric smoothing), e.g. 'regolith+basalt': mean of eps and sigma
for name in name_to_id:
    if name not in mat_props and '+' in name:
        parts = name.split('+')
        if all(p in mat_props for p in parts):
            mat_props[name] = {'eps': float(np.mean([mat_props[p]['eps'] for p in parts])),
                               'sigma': float(np.mean([mat_props[p]['sigma'] for p in parts])),
                               'poles': []}

missing = [n for n in name_to_id if n not in mat_props]
if missing:
    sys.exit(f"Error: properties not found for materials: {missing}\n"
             "  (materials defined inside #python blocks cannot be read from the .in; "
             "write <vti stem>_materials.txt with #geometry_objects_write, or define them with #material)")

# Arrays indexed by the material ID used in the .vti
n_ids = max(name_to_id.values()) + 1
mat_names = [None] * n_ids
for name, i in name_to_id.items():
    mat_names[i] = name
epsilon_list      = np.array([mat_props[n]['eps']   if n else 1.0 for n in mat_names])
conductivity_list = np.array([mat_props[n]['sigma'] if n else 0.0 for n in mat_names])
poles_list        = [mat_props[n]['poles'] if n else [] for n in mat_names]

has_dispersion = any(len(p) > 0 for p in poles_list)
print(f"Loaded {len(name_to_id)} materials. Dispersion: {has_dispersion}")
for name, i in sorted(name_to_id.items(), key=lambda t: t[1]):
    p = mat_props[name]
    print(f"  {i:3d} {name:20s} eps_inf={p['eps']:.6g} sigma={p['sigma']:.6g} poles={p['poles']}")


# =============================================================================
# Geometry map and depth axis
# =============================================================================
geometry_data = np.rot90(material_3d[:, :, 0], k=1)      # rows = y (top row = highest y), cols = x
z_num, x_num = geometry_data.shape
print(f"Geometry map shape (rows=depth, cols=x): {geometry_data.shape}")

y_bottom = origin[1]                                      # [m] bottom of the view
y_top = origin[1] + z_num * spatial_grid                  # [m] top of the view

if SURFACE_Y_M is None:
    # Surface = top of the non-free-space material in each column (median over columns)
    fs_id = name_to_id.get('free_space', -1)
    solid = geometry_data != fs_id                        # row 0 = top
    has_solid = solid.any(axis=0)
    if not has_solid.any():
        y_surf = y_bottom
    else:
        top_row = np.argmax(solid[:, has_solid], axis=0)  # first non-free-space row from the top
        y_surf = y_top - np.median(top_row) * spatial_grid
    print(f"Surface (detected): y = {y_surf:.4f} m from the bottom")
else:
    y_surf = SURFACE_Y_M
    print(f"Surface (SURFACE_Y_M): y = {y_surf:.4f} m from the bottom")

# Depth-axis limits (surface-referenced)
depth_top    = -(y_top - y_surf)      # negative (vacuum region)
depth_bottom =  y_surf - y_bottom     # positive (deepest subsurface)
print(f"Depth axis: {depth_top:.3f} m (vacuum top) to {depth_bottom:.3f} m (bottom)")


# =============================================================================
# Build property maps
# =============================================================================
epsilon_0 = 8.854187817e-12

def build_property_maps(freq):
    """Same formulas as before, evaluated once per material and mapped by ID (vectorized)."""
    omega = 2 * np.pi * freq

    eps_real_mat = epsilon_list.astype(float).copy()
    eps_imag_disp_mat = np.zeros_like(eps_real_mat)
    for k, poles in enumerate(poles_list):
        for (de, tau) in poles:
            denom = 1.0 + (omega * tau) ** 2
            eps_real_mat[k]      += de / denom
            eps_imag_disp_mat[k] += de * omega * tau / denom
    eps_imag_total_mat = eps_imag_disp_mat + conductivity_list / (omega * epsilon_0)
    losstangent_mat = eps_imag_total_mat / (eps_real_mat + 1e-30)

    idx = geometry_data.astype(np.intp)
    permittivity_map_at_freq = eps_real_mat[idx]
    conductivity_map_at_freq = conductivity_list.astype(float)[idx]
    losstangent_map_at_freq  = losstangent_mat[idx]

    permittivity_profile_at_freq = np.mean(permittivity_map_at_freq, axis=1)
    conductivity_profile_at_freq = np.mean(conductivity_map_at_freq, axis=1)
    losstangent_profile_at_freq  = np.mean(losstangent_map_at_freq,  axis=1)
    print("Property maps and depth profiles computed.")

    return permittivity_map_at_freq, conductivity_map_at_freq, losstangent_map_at_freq, \
            permittivity_profile_at_freq, conductivity_profile_at_freq, losstangent_profile_at_freq


# =============================================================================
# Plot settings
# =============================================================================
colors       = ['viridis', 'magma', 'plasma']
names        = ['Relative permittivity', 'Conductivity [S/m]', 'Loss tangent']
output_names = ['Permittivity', 'Conductivity', 'Losstangent']

# imshow extent: [x_left, x_right, depth_bottom, depth_top]
# imshow treats the 3rd value as "bottom" and 4th as "top", so placing
# depth_bottom (positive, deep) as 3rd gives a top-to-bottom increasing axis.
extent_map = [origin[0], origin[0] + x_num * spatial_grid, depth_bottom, depth_top]

# Depth array for profile plots (cell centres)
depth_axis = depth_top + (np.arange(z_num) + 0.5) * spatial_grid


# =============================================================================
# Plot helpers
# =============================================================================
def _add_surface_line(ax):
    """Add a dashed white line at the surface (depth = 0)."""
    ax.axhline(0.0, color='white', linewidth=1.2, linestyle='--', label='Surface')


def _imshow_kwargs(idx):
    kw = dict(extent=extent_map, interpolation='nearest', aspect='auto', cmap=colors[idx])
    if idx == 0:
        kw.update(vmin=1, vmax=4.0)
    elif idx == 2:
        kw.update(vmin=0, vmax=0.020)
    return kw


def draw_map(fig, ax, freq, map_data, idx):
    """2D map with colour bar (shared by the 'map + profile' and 'map only' figures)."""
    im = ax.imshow(map_data, **_imshow_kwargs(idx))
    # _add_surface_line(ax)
    ax.set_title(names[idx] + f'  @ {freq_label(freq)}', size=14)
    ax.set_xlabel('X [m]', size=18)
    ax.set_ylabel('Depth [m]', size=18)
    ax.set_ylim(depth_bottom, depth_top)   # deep at bottom (positive), vacuum at top (negative)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.3)
    divider = axgrid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(names[idx], size=14)
    cbar.ax.tick_params(labelsize=12)


def draw_profile(ax, profile_data, idx, title=None):
    """Depth profile (shared by the 'map + profile' and 'profile only' figures)."""
    ax.plot(profile_data, depth_axis)
    # ax.axhline(0.0, color='gray', linewidth=1.0, linestyle='--')  # surface
    ax.set_xlabel(names[idx], size=18)
    ax.set_ylabel('Depth [m]', size=18)
    if idx == 0:
        ax.set_xlim(0.95, 4.0)
    elif idx == 2:
        ax.set_xlim(-0.0005, 0.006)
    ax.set_ylim(depth_bottom, depth_top)
    ax.tick_params(labelsize=14)
    ax.minorticks_on()
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title, size=14)


def _save(fig, folder, stem):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(folder, f'{stem}.{ext}'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.relpath(os.path.join(folder, stem), output_dir)}.png / .pdf")


def plot_map_profile(freq, map_data, profile_data, idx):
    """2D map (left) and depth profile (right) side by side. Returns the figure (not closed)."""
    map_aspect = map_data.shape[0] / map_data.shape[1]
    fig, ax = plt.subplots(nrows=1, ncols=2, width_ratios=[3, 1], figsize=(12, 9 * map_aspect))
    draw_map(fig, ax[0], freq, map_data, idx)
    draw_profile(ax[1], profile_data, idx)
    fig.tight_layout()
    _save(fig, output_dir, f'{output_names[idx]}_{freq:.2e}_map')
    return fig


def plot_map(freq, map_data, idx):
    """2D map only."""
    map_aspect = map_data.shape[0] / map_data.shape[1]
    fig, ax = plt.subplots(figsize=(9.5, 9 * map_aspect + 0.6))
    draw_map(fig, ax, freq, map_data, idx)
    fig.tight_layout()
    _save(fig, output_dir_map, f'{output_names[idx]}_{freq:.2e}_maponly')
    plt.close(fig)


def plot_profile(freq, profile_data, idx):
    """Depth profile only."""
    fig, ax = plt.subplots(figsize=(4, 8))
    draw_profile(ax, profile_data, idx, title=f'@ {freq_label(freq)}')
    fig.tight_layout()
    _save(fig, output_dir_profile, f'{output_names[idx]}_{freq:.2e}_profile')
    plt.close(fig)


# =============================================================================
# Run
# =============================================================================
fig_show = None
for freq in freqs:
    print(f'===== Processing {freq_label(freq)} =====')
    permittivity_map, conductivity_map, losstangent_map, \
        permittivity_profile, conductivity_profile, losstangent_profile = build_property_maps(freq)

    for idx, (map_data, profile_data) in enumerate([
        (permittivity_map, permittivity_profile),
        (conductivity_map, conductivity_profile),
        (losstangent_map,  losstangent_profile),
    ]):
        fig = plot_map_profile(freq, map_data, profile_data, idx)
        if SHOW_PLOT and idx == 0 and freq == show_freq:
            fig_show = fig                 # keep only this one for the screen
        else:
            plt.close(fig)
        plot_map(freq, map_data, idx)
        plot_profile(freq, profile_data, idx)
    print(' ')

if fig_show is not None:
    print(f"Showing: {names[0]} (map + profile) @ {freq_label(show_freq)}")
    plt.show()