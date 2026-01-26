#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from vtkmodules.vtkIOXML import vtkXMLImageDataReader
from vtkmodules.vtkFiltersCore import vtkCellCenters
from vtkmodules.util.numpy_support import vtk_to_numpy

def read_vti(filename):
    """VTIファイルを読み込み、vtkImageDataオブジェクトを返します"""
    reader = vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def extract_slice_for_geometry(img, field_name, target_z=0.0, tol=1e-6):
    """
    ジオメトリ表示用に特定Z座標のスライスデータを抽出する関数
    """
    def choose_slice_z(z_coords):
        uniq = np.unique(z_coords)
        return uniq[np.argmin(np.abs(uniq - target_z))]

    point_data = img.GetPointData()
    array = point_data.GetArray(field_name)
    mesh = img
    
    # PointDataにない場合CellDataを探す
    if array is None:
        cell_data = img.GetCellData()
        array = cell_data.GetArray(field_name)
        if array is None:
            # 配列が見つからない場合はNoneを返す
            return None, None
        
        # CellDataの場合はセル中心を取得
        center = vtkCellCenters()
        center.SetInputData(img)
        center.Update()
        mesh = center.GetOutput()

    pts = vtk_to_numpy(mesh.GetPoints().GetData()).reshape(-1, 3)
    vals = vtk_to_numpy(array)
    
    # ターゲットZに最も近いスライスを選択
    slice_z = choose_slice_z(pts[:, 2])
    mask = np.isclose(pts[:, 2], slice_z, atol=tol)
    
    coords2d = pts[mask][:, :2]
    vals2d = vals[mask]
    
    # ベクトルの場合は成分処理が必要だが、通常ジオメトリ(ID)はスカラ
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

def main():
    print("--- FDTD A-scan Extractor (with Geometry Check) ---")
    
    # --- 1. 設定入力 (Geometryパスから開始) ---
    geometry_path = input("Enter path to geometry.vti: ").strip()
    geometry_path = os.path.normpath(geometry_path)
    
    if not os.path.isfile(geometry_path):
        raise FileNotFoundError(f"Geometry file not found: {geometry_path}")
    
    parent_dir = os.path.dirname(geometry_path)
    print(f"[INFO] Parent directory: {parent_dir}")

    # スナップショットディレクトリの特定
    # 親ディレクトリ内の "_snaps" で終わるフォルダを探す
    snap_candidates = [d for d in os.listdir(parent_dir) if d.endswith("_snaps") and os.path.isdir(os.path.join(parent_dir, d))]
    
    if len(snap_candidates) == 1:
        snap_dir = os.path.join(parent_dir, snap_candidates[0])
        print(f"[INFO] Auto-detected snapshot directory: {snap_dir}")
    else:
        print(f"[INFO] Found {len(snap_candidates)} snapshot directories. Please specify manually.")
        snap_dir_input = input("Enter snapshot directory name or full path: ").strip()
        if os.path.isabs(snap_dir_input):
            snap_dir = snap_dir_input
        else:
            snap_dir = os.path.join(parent_dir, snap_dir_input)

    if not os.path.isdir(snap_dir):
        raise FileNotFoundError(f"Snapshot directory not found: {snap_dir}")

    dt_ns = float(input("Enter time step [ns]: ").strip())
    
    print("\nEnter target coordinates to extract A-scan [m]:")
    tx = float(input("  x: ").strip())
    ty = float(input("  y: ").strip())
    tz = float(input("  z: ").strip())
    target_pos = (tx, ty, tz)

    print("\nSelect component to plot:")
    print("  0: Ex,  1: Ey,  2: Ez (default)")
    comp_input = input("Enter choice (0-2): ").strip()
    plot_component = int(comp_input) if comp_input in ['0', '1', '2'] else 2
    comp_labels = ['Ex', 'Ey', 'Ez']

    # --- 2. ファイルリスト取得 ---
    snap_paths = sorted(
        glob.glob(os.path.join(snap_dir, "snapshot*.vti")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].replace("snapshot",""))
    )
    if not snap_paths:
        raise FileNotFoundError("No 'snapshot*.vti' files found.")

    # --- 3. ターゲット座標の特定 ---
    print(f"\n[INFO] Loading geometry to locate target grid point...")
    # geometryまたは最初のスナップショットを使って座標系を確認
    geom_img = read_vti(geometry_path)
    point_id, actual_coords = get_closest_point_id(geom_img, target_pos)
    
    if point_id is None:
        raise ValueError("Target coordinates are outside the simulation domain.")
    
    print(f"[INFO] Target: {target_pos}")
    print(f"[INFO] Actual Grid: ({actual_coords[0]:.4f}, {actual_coords[1]:.4f}, {actual_coords[2]:.4f})")

    # --- 4. 出力ディレクトリ準備 ---
    snap_dir_name = os.path.basename(os.path.normpath(snap_dir))
    new_base_folder = f"{snap_dir_name}_extract_Ascan"
    output_base = os.path.join(parent_dir, new_base_folder)
    
    coord_str = f"x{actual_coords[0]:.3f}_y{actual_coords[1]:.3f}_z{actual_coords[2]:.3f}"
    output_dir = os.path.join(output_base, coord_str)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory prepared: {output_dir}")

    # --- 5. ジオメトリ位置確認プロットの作成 ---
    print("\n[INFO] Generating geometry location check plot...")
    
    # ジオメトリデータの配列名取得 (通常はMaterial IDなど)
    pd_geom = geom_img.GetPointData()
    if pd_geom.GetNumberOfArrays() > 0:
        geom_field = pd_geom.GetArrayName(0)
    else:
        cd_geom = geom_img.GetCellData()
        geom_field = cd_geom.GetArrayName(0) if cd_geom.GetNumberOfArrays() > 0 else None
    
    if geom_field:
        # ターゲットのZ座標におけるスライスを取得
        g_coords, g_vals = extract_slice_for_geometry(geom_img, geom_field, target_z=actual_coords[2])
        
        if g_coords is not None:
            # グリッド化して表示
            xs = np.unique(g_coords[:, 0])
            ys = np.unique(g_coords[:, 1])
            nx, ny = xs.size, ys.size
            
            # グリッド再構成
            geom_grid = np.zeros((ny, nx))
            ix = np.searchsorted(xs, g_coords[:, 0])
            iy = np.searchsorted(ys, g_coords[:, 1])
            geom_grid[iy, ix] = g_vals
            
            # プロット作成
            plt.figure(figsize=(8, 6))
            extent = [xs.min(), xs.max(), ys.min(), ys.max()]
            
            # カラーマップ設定 (ID用)
            unique_ids = np.unique(g_vals)
            if len(unique_ids) <= 5:
                cmap_geom = ListedColormap(['gray', 'lightgray', 'white', 'lightblue', 'red'])
                # 色数が足りない場合は適当に切り詰める
                cmap_geom = ListedColormap(cmap_geom.colors[:len(unique_ids)])
            else:
                cmap_geom = 'viridis'

            plt.imshow(geom_grid, extent=extent, origin='lower', cmap=cmap_geom, alpha=0.8, aspect='equal')
            plt.colorbar(label="Material ID")
            
            # 解析位置を×印でプロット
            plt.scatter(actual_coords[0], actual_coords[1], c='red', marker='x', s=200, linewidth=2.5, label='A-scan Point')
            
            plt.title(f"A-scan Location Check (Z ≈ {actual_coords[2]:.3f} m)")
            plt.xlabel("X [m]")
            plt.ylabel("Y [m]")
            plt.legend(loc='upper right')
            plt.grid(True, linestyle=':', alpha=0.5)
            
            geom_plot_path = os.path.join(output_dir, "geometry_location_check.png")
            plt.savefig(geom_plot_path, dpi=300)
            plt.close()
            print(f"[INFO] Geometry check saved: {geom_plot_path}")
        else:
            print("[WARNING] Could not extract slice. Skipping geometry plot.")
    else:
        print("[WARNING] No data found in geometry.vti. Skipping geometry plot.")

    # --- 6. A-scan抽出ループ ---
    print("\n[INFO] Extracting time series data...")
    times, ex_vals, ey_vals, ez_vals = [], [], [], []
    field_name = "E-field"

    for i, path in enumerate(snap_paths):
        if i % 10 == 0: print(f"\rProcessing frame {i+1}/{len(snap_paths)}...", end="")
        
        reader = vtkXMLImageDataReader()
        reader.SetFileName(path)
        reader.Update()
        img = reader.GetOutput()
        
        # 変数名衝突回避: pd -> point_data
        point_data = img.GetPointData()
        array = point_data.GetArray(field_name)
        
        # 配列が見つからない場合のフォールバック
        if array is None:
            cell_data = img.GetCellData()
            array = cell_data.GetArray(field_name)
        
        if array:
            val = array.GetTuple(point_id)
        else:
            val = (0.0, 0.0, 0.0)
        
        times.append((i + 1) * dt_ns)
        ex_vals.append(val[0])
        ey_vals.append(val[1])
        ez_vals.append(val[2])

    print("\n[INFO] Extraction complete.")

    # --- 7. 結果保存 ---
    # CSV保存
    df = pd.DataFrame({'Time [ns]': times, 'Ex': ex_vals, 'Ey': ey_vals, 'Ez': ez_vals})
    csv_path = os.path.join(output_dir, "ascan_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] CSV saved: {csv_path}")

    # プロット保存
    target_vals = [ex_vals, ey_vals, ez_vals][plot_component]
    target_label = comp_labels[plot_component]
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, target_vals, label=f'{target_label} at {coord_str}')
    plt.xlabel("Time [ns]")
    plt.ylabel("Electric Field [V/m]")
    plt.title(f"A-scan Extraction ({target_label})")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plot_path = os.path.join(output_dir, f"ascan_plot_{target_label}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"[INFO] Plot saved: {plot_path}")

if __name__ == "__main__":
    main()