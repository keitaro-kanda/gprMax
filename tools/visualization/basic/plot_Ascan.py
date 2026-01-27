#!/usr/bin/env python3
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from vtkmodules.vtkIOXML import vtkXMLImageDataReader
from vtkmodules.vtkFiltersCore import vtkCellCenters
from vtkmodules.util.numpy_support import vtk_to_numpy

# 日本語フォント設定（必要に応じて環境に合わせてコメントアウトを外してください）
# plt.rcParams['font.family'] = 'sans-serif' 

def read_vti(filename):
    """VTIファイルを読み込み、vtkImageDataオブジェクトを返します"""
    reader = vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def extract_slice_for_geometry(img, field_name, target_z=0.0, tol=1e-6):
    """ジオメトリ表示用に特定Z座標のスライスデータを抽出する関数"""
    def choose_slice_z(z_coords):
        uniq = np.unique(z_coords)
        return uniq[np.argmin(np.abs(uniq - target_z))]

    point_data = img.GetPointData()
    array = point_data.GetArray(field_name)
    mesh = img
    
    if array is None:
        cell_data = img.GetCellData()
        array = cell_data.GetArray(field_name)
        if array is None:
            return None, None
        
        center = vtkCellCenters()
        center.SetInputData(img)
        center.Update()
        mesh = center.GetOutput()

    pts = vtk_to_numpy(mesh.GetPoints().GetData()).reshape(-1, 3)
    vals = vtk_to_numpy(array)
    
    slice_z = choose_slice_z(pts[:, 2])
    mask = np.isclose(pts[:, 2], slice_z, atol=tol)
    
    coords2d = pts[mask][:, :2]
    vals2d = vals[mask]
    
    if vals2d.ndim > 1 and vals2d.shape[1] >= 3:
        vals2d = vals2d[:, 2] 
        
    return coords2d, vals2d

def get_closest_point_id(img_data, target_pos):
    """指定座標に最も近いグリッド点のIDと実座標を返します"""
    point_id = img_data.FindPoint(target_pos)
    if point_id < 0:
        return None, None
    actual_coords = img_data.GetPoint(point_id)
    return point_id, actual_coords

def prepare_geometry_grid(geom_img, target_z):
    """ジオメトリプロット用のグリッドデータを準備して返します（再利用のため）"""
    pd_geom = geom_img.GetPointData()
    if pd_geom.GetNumberOfArrays() > 0:
        geom_field = pd_geom.GetArrayName(0)
    else:
        cd_geom = geom_img.GetCellData()
        geom_field = cd_geom.GetArrayName(0) if cd_geom.GetNumberOfArrays() > 0 else None
    
    if not geom_field:
        return None

    g_coords, g_vals = extract_slice_for_geometry(geom_img, geom_field, target_z=target_z)
    if g_coords is None:
        return None

    xs = np.unique(g_coords[:, 0])
    ys = np.unique(g_coords[:, 1])
    nx, ny = xs.size, ys.size
    
    geom_grid = np.zeros((ny, nx))
    ix = np.searchsorted(xs, g_coords[:, 0])
    iy = np.searchsorted(ys, g_coords[:, 1])
    geom_grid[iy, ix] = g_vals
    
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]
    return geom_grid, extent, g_vals

def plot_geometry_on_ax(ax, geom_grid, extent, g_vals, actual_coords, title_prefix=""):
    """指定されたAxesオブジェクトにジオメトリを描画します"""
    unique_ids = np.unique(g_vals)
    if len(unique_ids) <= 5:
        cmap_geom = ListedColormap(['gray', 'lightgray', 'white', 'lightblue', 'red'])
        cmap_geom = ListedColormap(cmap_geom.colors[:len(unique_ids)])
    else:
        cmap_geom = 'viridis'

    ax.imshow(geom_grid, extent=extent, origin='lower', cmap=cmap_geom, alpha=0.8, aspect='equal')
    ax.scatter(actual_coords[0], actual_coords[1], c='red', marker='x', s=150, linewidth=2, zorder=10)
    
    ax.set_title(f"{title_prefix}Loc (Z={actual_coords[2]:.2f})", fontsize=10)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True, linestyle=':', alpha=0.5)

def main():
    print("--- FDTD A-scan Extractor (Grouped & Combined) ---")
    
    # --- 1. Geometry & Snapshot Setup ---
    geometry_path = input("Enter path to geometry.vti: ").strip()
    geometry_path = os.path.normpath(geometry_path)
    
    if not os.path.isfile(geometry_path):
        raise FileNotFoundError(f"Geometry file not found: {geometry_path}")
    
    parent_dir = os.path.dirname(geometry_path)
    
    snap_candidates = [d for d in os.listdir(parent_dir) if d.endswith("_snaps") and os.path.isdir(os.path.join(parent_dir, d))]
    
    if len(snap_candidates) == 1:
        snap_dir = os.path.join(parent_dir, snap_candidates[0])
        print(f"[INFO] Auto-detected snapshot directory: {snap_dir}")
    else:
        print(f"[INFO] Found {len(snap_candidates)} snapshot directories.")
        snap_dir_input = input("Enter snapshot directory name or path: ").strip()
        if os.path.isabs(snap_dir_input):
            snap_dir = snap_dir_input
        else:
            snap_dir = os.path.join(parent_dir, snap_dir_input)

    if not os.path.isdir(snap_dir):
        raise FileNotFoundError(f"Snapshot directory not found: {snap_dir}")

    dt_ns = float(input("Enter time step [ns]: ").strip())
    
    print("\nSelect component to plot:")
    print("  0: Ex,  1: Ey,  2: Ez (default)")
    comp_input = input("Enter choice (0-2): ").strip()
    plot_component = int(comp_input) if comp_input in ['0', '1', '2'] else 2
    comp_labels = ['Ex', 'Ey', 'Ez']

    # --- 2. Input Mode Selection ---
    print("\nSelect Input Mode:")
    print("  1: Single Point (Manual Input)")
    print("  2: Batch Processing (from JSON file)")
    mode = input("Enter mode (1 or 2): ").strip()

    targets = [] # 全ターゲットのフラットなリスト
    
    # グループ管理用辞書 { "GroupName": [target_obj, ...], ... }
    groups = {} 

    if mode == '2':
        # --- JSON Mode ---
        json_path = input("Enter path to targets.json: ").strip()
        json_path = os.path.normpath(json_path)
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                
                # 辞書型（グループ化）かリスト型（旧仕様）かで分岐
                if isinstance(data, dict):
                    print("[INFO] Detected grouped JSON format.")
                    for group_name, items in data.items():
                        if not isinstance(items, list): continue
                        groups[group_name] = []
                        for item in items:
                            if all(k in item for k in ('x', 'y', 'z')):
                                t = {
                                    'group': group_name,
                                    'label': item.get('label', 'NoName'),
                                    'req_coords': (item['x'], item['y'], item['z'])
                                }
                                targets.append(t)
                                groups[group_name].append(t)
                
                elif isinstance(data, list):
                    print("[INFO] Detected simple list JSON format. Using 'Default' group.")
                    group_name = "Default"
                    groups[group_name] = []
                    for item in data:
                        if all(k in item for k in ('x', 'y', 'z')):
                            t = {
                                'group': group_name,
                                'label': item.get('label', 'NoName'),
                                'req_coords': (item['x'], item['y'], item['z'])
                            }
                            targets.append(t)
                            groups[group_name].append(t)
                else:
                    raise ValueError("JSON root must be a dict or a list.")

            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format.")
        
        print(f"[INFO] Loaded {len(targets)} targets in {len(groups)} groups.")
        
    else:
        # --- Single Mode ---
        print("\nEnter target coordinates [m]:")
        tx = float(input("  x: ").strip())
        ty = float(input("  y: ").strip())
        tz = float(input("  z: ").strip())
        
        group_name = "Manual_Extract"
        t = {
            'group': group_name,
            'label': 'Point',
            'req_coords': (tx, ty, tz)
        }
        targets.append(t)
        groups[group_name] = [t]

    if not targets:
        print("[ERROR] No targets specified. Exiting.")
        return

    # --- 3. Prepare Targets & Output Dirs ---
    print(f"\n[INFO] Initializing targets and checking geometry...")
    geom_img = read_vti(geometry_path)
    
    snap_dir_name = os.path.basename(os.path.normpath(snap_dir))
    output_base_folder = os.path.join(parent_dir, f"{snap_dir_name}_extract_Ascan")
    
    valid_targets = []
    
    # 有効なターゲットのみを抽出して再構築
    for t in targets:
        pid, act_coords = get_closest_point_id(geom_img, t['req_coords'])
        
        if pid is None:
            print(f"  [SKIP] Target '{t['label']}' (Group: {t['group']}) is out of bounds.")
            # グループリストからも削除が必要だが、後でgroupsを再生成する方が安全
            continue
            
        t['point_id'] = pid
        t['actual_coords'] = act_coords
        
        # フォルダ構造: Output / GroupName / Label_coords /
        coord_str = f"x{act_coords[0]:.3f}_y{act_coords[1]:.3f}_z{act_coords[2]:.3f}"
        
        # グループフォルダ
        t['group_dir'] = os.path.join(output_base_folder, t['group'])
        
        # 個別ターゲットフォルダ
        t['target_dir'] = os.path.join(t['group_dir'], f"{t['label']}_{coord_str}")
        os.makedirs(t['target_dir'], exist_ok=True)
        
        # データ格納用
        t['ex'], t['ey'], t['ez'] = [], [], []
        
        valid_targets.append(t)

    # グループ辞書の再構築（無効なターゲットを除外）
    groups = {}
    for t in valid_targets:
        if t['group'] not in groups:
            groups[t['group']] = []
        groups[t['group']].append(t)
        
    targets = valid_targets

    if not targets:
        raise RuntimeError("No valid targets found inside the simulation domain.")

    # --- 4. Main Extraction Loop ---
    snap_paths = sorted(
        glob.glob(os.path.join(snap_dir, "snapshot*.vti")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].replace("snapshot",""))
    )
    if not snap_paths:
        raise FileNotFoundError("No 'snapshot*.vti' files found.")

    print(f"\n[INFO] Extracting data from {len(snap_paths)} frames...")
    
    times = []
    field_name = "E-field"

    for i, path in enumerate(snap_paths):
        if i % 10 == 0: print(f"\rProcessing frame {i+1}/{len(snap_paths)}...", end="")
        
        reader = vtkXMLImageDataReader()
        reader.SetFileName(path)
        reader.Update()
        img = reader.GetOutput()
        
        pd_vtk = img.GetPointData()
        array = pd_vtk.GetArray(field_name)
        if array is None:
            cd_vtk = img.GetCellData()
            array = cd_vtk.GetArray(field_name)
        
        current_time = (i + 1) * dt_ns
        times.append(current_time)
        
        # 全ターゲット一括抽出
        for t in targets:
            if array:
                val = array.GetTuple(t['point_id'])
            else:
                val = (0.0, 0.0, 0.0)
            t['ex'].append(val[0])
            t['ey'].append(val[1])
            t['ez'].append(val[2])

    print("\n[INFO] Extraction complete.")

    # --- 5. Saving Results & Plotting ---
    print("[INFO] Generating plots and saving files...")
    target_label_comp = comp_labels[plot_component]

    # 個別ターゲットごとの処理
    for t in targets:
        # CSV保存
        df = pd.DataFrame({
            'Time [ns]': times,
            'Ex': t['ex'], 'Ey': t['ey'], 'Ez': t['ez']
        })
        df.to_csv(os.path.join(t['target_dir'], "ascan_data.csv"), index=False)
        
        # 波形データ
        vals_to_plot = [t['ex'], t['ey'], t['ez']][plot_component]
        
        # ジオメトリデータの準備（Combined Plot用）
        geom_res = prepare_geometry_grid(geom_img, t['actual_coords'][2])
        
        # ----------------------------------------------------
        # 1. ジオメトリ単体プロット (Location Check)
        # ----------------------------------------------------
        if geom_res:
            geom_grid, extent, g_vals = geom_res
            plt.figure(figsize=(6, 5))
            ax = plt.gca()
            plot_geometry_on_ax(ax, geom_grid, extent, g_vals, t['actual_coords'], title_prefix=f"{t['label']} ")
            plt.tight_layout()
            plt.savefig(os.path.join(t['target_dir'], "geometry_location_check.png"), dpi=200)
            plt.close()

        # ----------------------------------------------------
        # 2. A-scan単体プロット
        # ----------------------------------------------------
        plt.figure(figsize=(8, 4))
        plt.plot(times, vals_to_plot, color='blue', linewidth=1.2)
        plt.xlabel("Time [ns]")
        plt.ylabel("Electric Field [V/m]")
        plt.title(f"A-scan: {t['label']} ({target_label_comp})")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(t['target_dir'], f"ascan_plot_{target_label_comp}.png"), dpi=200)
        plt.close()

        # ----------------------------------------------------
        # 3. Combined Plot (左右配置: 左ジオメトリ, 右A-scan)
        # ----------------------------------------------------
        if geom_res:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1.5]})
            
            # Left: Geometry
            plot_geometry_on_ax(ax1, geom_grid, extent, g_vals, t['actual_coords'])
            
            # Right: A-scan
            ax2.plot(times, vals_to_plot, color='blue', linewidth=1.5)
            ax2.set_xlabel("Time [ns]")
            ax2.set_ylabel("Electric Field [V/m]")
            ax2.set_title(f"A-scan ({target_label_comp})")
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            plt.suptitle(f"Target Analysis: {t['label']}", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(t['target_dir'], "combined_view.png"), dpi=200)
            plt.close()

    # ----------------------------------------------------
    # 4. Group Comparison Plot (グループごとの重ね書き)
    # ----------------------------------------------------
    print("[INFO] Generating group comparison plots...")
    
    for group_name, group_targets in groups.items():
        if not group_targets: continue
        
        plt.figure(figsize=(12, 6))
        
        # カラーマップ生成 (ターゲット数分)
        colors = plt.cm.jet(np.linspace(0, 1, len(group_targets)))
        
        for idx, t in enumerate(group_targets):
            vals = [t['ex'], t['ey'], t['ez']][plot_component]
            # ラベルに座標情報を少し付加
            label_txt = f"{t['label']}"
            plt.plot(times, vals, label=label_txt, color=colors[idx], linewidth=1.2, alpha=0.8)
            
        plt.xlabel("Time [ns]")
        plt.ylabel("Electric Field [V/m]")
        plt.title(f"Group Comparison: {group_name} ({target_label_comp})")
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # グループフォルダ直下に保存
        save_path = os.path.join(group_targets[0]['group_dir'], f"group_comparison_{target_label_comp}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  Saved comparison: {save_path}")

    print("\n[INFO] All tasks finished successfully.")

if __name__ == "__main__":
    main()