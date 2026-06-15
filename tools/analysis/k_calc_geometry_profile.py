#!/usr/bin/env python3
"""
gprMax の geometry (.h5) と materials.txt から、比誘電率・導電率・損失正接の
2D マップと深さプロファイルを描画するスクリプト。

分散あり / なしの両方のファイルに自動対応する:
  - #add_dispersion_debye 行が無い材料は deps=tau=0 となり、物性計算は
    自動的に非分散（静的）の式に縮退する（損失正接 = sigma/(omega*eps0*eps_r)）。
  - #add_dispersion_debye 行がある材料は、指定周波数での Debye 実効値を計算する。

主な修正点（旧版からの変更）:
  1. materials.txt は #material 行のみを材料登録順（= h5 の整数インデックス順）で
     読み込み、pec / free_space を除外。これにより h5 の整数と材料リストの
     インデックスずれを解消。
  2. #add_dispersion_debye 行を Δε・τ として正しく対応付け（材料リストには混入させない）。
  3. 損失誘電率・損失正接を Debye 分散込みで指定周波数において評価。
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
# 入力
# =============================================================================
# Input file path of geometry JSON
json_file = input("Enter geometry JSON file path: ").strip()
if not os.path.exists(json_file):
    sys.exit("Error: Geometry JSON file not found.")
# Input frequency of GPR
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


# =============================================================================
# materials.txt の読み込み（分散あり/なし両対応）
# =============================================================================
# gprMax の h5 整数は pec=0, free_space=1, mat_0000=2, ... と
# 「組み込み材料を含む全材料の登録順」で割り振られる。よって pec / free_space も
# 除外せずリストに含め、h5 の整数をそのままインデックスとして使う。
# #add_dispersion_debye 行があればその材料の Δε・τ を上書き。無ければ 0 のまま
# （→ 物性計算が自動的に非分散の式へ縮退する）。
material_path = params['geometry_settings']['material_file']

mat_names = []      # 材料名（登録順を保持 = h5 の整数順）
mat_props = {}      # name -> {'eps','sigma','deps','tau'}

with open(material_path, 'r') as mf:
    for line in mf:
        line = line.strip()
        if line.startswith('#material:'):
            v = line.split()
            # #material: eps_r sigma mu_r mag_sigma name
            name = v[5]
            # pec / free_space も含める（h5 の整数体系に合わせるため除外しない）
            # pec は sigma='inf' のため float 変換を避け、固定値で登録
            if name == 'pec':
                eps_v, sig_v = 1.0, 0.0
            else:
                eps_v, sig_v = float(v[1]), float(v[2])
            mat_props[name] = {
                'eps':   eps_v,    # ε∞（分散なしなら静的 εr に等しい）
                'sigma': sig_v,    # 導電率 σ
                'deps':  0.0,      # Debye Δε（デフォルト 0 = 非分散）
                'tau':   0.0,      # Debye τ（デフォルト 0）
            }
            mat_names.append(name)
        elif line.startswith('#add_dispersion_debye:'):
            # 単極 Debye のみ対応: "#add_dispersion_debye: i1 Δε τ name"
            v = line.split()
            name = v[-1]
            npoles = int(v[1])
            if name in mat_props and npoles == 1:
                mat_props[name]['deps'] = float(v[2])
                mat_props[name]['tau'] = float(v[3])
        # 他の分散コマンド（lorentz / drude）は対象外（非分散として扱う）

# h5 の整数インデックス順に並べたリストを構築
epsilon_list = []       # ε∞
conductivity_list = []  # σ
delta_eps_list = []     # Debye Δε
tau_list = []           # Debye τ
for name in mat_names:
    p = mat_props[name]
    epsilon_list.append(p['eps'])
    conductivity_list.append(p['sigma'])
    delta_eps_list.append(p['deps'])
    tau_list.append(p['tau'])

epsilon_list = np.array(epsilon_list)
conductivity_list = np.array(conductivity_list)
delta_eps_list = np.array(delta_eps_list)
tau_list = np.array(tau_list)

# このファイルが分散を含むかどうか（ログ・タイトル表示用）
has_dispersion = bool(np.any(delta_eps_list > 0))
print(f"Loaded {len(mat_names)} materials (incl. builtins). Dispersion in this file: {has_dispersion}")


# =============================================================================
# HDF5 データ読み込み
# =============================================================================
h5_file = params['geometry_settings']['h5_file']
h5f = h5py.File(h5_file, 'r')
print(f"Opened HDF5 file: {h5f['data'].shape}")

# Extract and rotate data
geometry_data = h5f['data'][:, :, 0]
geometry_data = np.rot90(geometry_data)
print(f"Extracted geometry data with shape {geometry_data.shape}")


# =============================================================================
# 物性マップ生成（分散あり/なし両対応）
# =============================================================================
# deps=tau=0 の材料では Debye 項が消え、以下が自動的に非分散の式へ縮退する:
#   eps_real      -> eps_inf
#   eps_imag_total-> sigma/(omega*eps0)
#   losstangent   -> sigma/(omega*eps0*eps_r)   （旧版の式と一致）
z_num, x_num = geometry_data.shape
permittivity_map = np.zeros((z_num, x_num), dtype=float)
conductivity_map = np.zeros((z_num, x_num), dtype=float)
losstangent_map = np.zeros((z_num, x_num), dtype=float)

epsilon_0 = 8.854187817e-12
omega = 2 * np.pi * freq

for i in tqdm(range(x_num), desc="Making geometry maps"):
    for j in range(z_num):
        idx = int(geometry_data[j, i])
        eps_inf = epsilon_list[idx]
        sigma = conductivity_list[idx]
        deps = delta_eps_list[idx]
        tau = tau_list[idx]

        # 単極 Debye を指定周波数で評価（非分散なら deps=tau=0 で自動縮退）
        denom = 1.0 + (omega * tau) ** 2
        eps_real = eps_inf + deps / denom                 # ε'(ω)
        eps_imag_debye = deps * omega * tau / denom        # ε''(ω) 分散ぶん
        eps_imag_total = eps_imag_debye + sigma / (omega * epsilon_0)

        permittivity_map[j, i] = eps_real
        conductivity_map[j, i] = sigma                     # σ は静的値を表示
        losstangent_map[j, i] = eps_imag_total / eps_real


# Calculate mean depth profile
permittivity_profile = np.mean(permittivity_map, axis=1)
conductivity_profile = np.mean(conductivity_map, axis=1)
losstangent_profile = np.mean(losstangent_map, axis=1)
print(f"Calculated depth profiles with length {permittivity_profile.shape[0]}")


# =============================================================================
# 描画関数：マップ＋プロファイル
# =============================================================================
def plot_map_profile(map_data, profile_data, map_type, colors, names, output_names):
    """マップ（左）と深さプロファイル（右）を並べて描画・保存する。
    Args:
        map_data     : 2D マップデータ
        profile_data : 1D 深さプロファイル
        map_type     : 'permittivity' / 'conductivity' / 'losstangent'
    """
    idx = None
    if map_type == 'permittivity':
        idx = 0
    elif map_type == 'conductivity':
        idx = 1
    elif map_type == 'losstangent':
        idx = 2

    map_aspect = map_data.shape[0] / map_data.shape[1]

    fig, ax = plt.subplots(
        nrows=1,          # 縦
        ncols=2,          # 横
        width_ratios=[3, 1],
        height_ratios=[1],
        figsize=(12, 9 * map_aspect)
    )

    if idx == 0:
        im = ax[0].imshow(
            map_data,
            extent=[0, map_data.shape[1] * spatial_grid, map_data.shape[0] * spatial_grid, 0],
            interpolation='nearest', aspect='auto', cmap=colors[idx], vmin=1, vmax=6)
    else:
        im = ax[0].imshow(
            map_data,
            extent=[0, map_data.shape[1] * spatial_grid, map_data.shape[0] * spatial_grid, 0],
            interpolation='nearest', aspect='auto', cmap=colors[idx])

    # 分散の有無をタイトルに明示（取り違え防止）
    disp_tag = ' (dispersive)' if has_dispersion else ' (non-dispersive)'
    ax[0].set_title(names[idx] + disp_tag + f'  @ {freq:.3e} Hz', size=14)
    ax[0].set_xlabel('X [m]', size=18)
    ax[0].set_ylabel('Y [m]', size=18)
    ax[0].tick_params(labelsize=14)
    ax[0].grid()

    # colorbar
    delvider = axgrid1.make_axes_locatable(ax[0])
    cax = delvider.append_axes('right', size='5%', pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(names[idx], size=18)
    cbar.ax.tick_params(labelsize=14)

    ax[1].plot(profile_data, np.arange(profile_data.shape[0]) * spatial_grid)
    ax[1].set_xlabel(names[idx], size=18)
    ax[1].set_ylabel('Depth (m)', size=18)
    ax[1].set_ylim(profile_data.shape[0] * spatial_grid, 0)
    ax[1].tick_params(labelsize=12)
    ax[1].grid()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_names[idx] + 'map.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, output_names[idx] + '_map.pdf'), format='pdf', dpi=300)
    plt.show()


# =============================================================================
# 描画関数：プロファイルのみ
# =============================================================================
def plot_profile(profile_data, map_type, names, output_names):
    """深さプロファイルのみを描画・保存する。"""
    idx = None
    if map_type == 'permittivity':
        idx = 0
    elif map_type == 'conductivity':
        idx = 1
    elif map_type == 'losstangent':
        idx = 2

    plt.figure(figsize=(4, 8), facecolor='w', edgecolor='w')
    plt.plot(profile_data, np.arange(profile_data.shape[0]) * spatial_grid)
    plt.xlabel(names[idx], size=18)
    plt.ylabel('Depth (m)', size=18)
    plt.ylim(permittivity_profile.shape[0] * spatial_grid, 0)
    plt.tick_params(labelsize=14)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_names[idx] + '_profile.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, output_names[idx] + 'permittivity_profile.pdf'), format='pdf', dpi=300)
    plt.close()


# =============================================================================
# 実行
# =============================================================================
colors = ['jet', 'magma', 'viridis']
names = ['Relative permittivity', 'Conductivity', 'Loss tangent']
output_names = ['Permittivity', 'Conductivity', 'Losstangent']

# Permittivity
plot_map_profile(permittivity_map, permittivity_profile, 'permittivity', colors, names, output_names)
plot_profile(permittivity_profile, 'permittivity', names, output_names)
# Conductivity
plot_map_profile(conductivity_map, conductivity_profile, 'conductivity', colors, names, output_names)
plot_profile(conductivity_profile, 'conductivity', names, output_names)
# Loss tangent
plot_map_profile(losstangent_map, losstangent_profile, 'losstangent', colors, names, output_names)
plot_profile(losstangent_profile, 'losstangent', names, output_names)