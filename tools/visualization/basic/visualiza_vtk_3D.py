#!/usr/bin/env python3
import os
import glob
import pyvista as pv
import numpy as np

def plot_and_save_vtk(vtk_file):
    vtk_dir = os.path.dirname(os.path.abspath(vtk_file))
    base_name = os.path.splitext(os.path.basename(vtk_file))[0]

    # 1. _materials.txt の自動検出と誘電率の取得
    mat_files = glob.glob(os.path.join(vtk_dir, '*_materials.txt'))
    if not mat_files:
        print(f"エラー: {vtk_dir} 内に '_materials.txt' を含むファイルが見つかりません。")
        return
    
    material_path = mat_files[0]
    print(f"マテリアルファイルを参照: {material_path}")

    epsilons = []
    with open(material_path, 'r') as mf:
        for line in mf:
            if line.startswith('#material:'):
                values = line.split()
                if len(values) >= 2:
                    epsilons.append(float(values[1]))

    # 2. VTKデータの読み込み
    print("VTKデータを読み込んでいます...")
    try:
        grid = pv.read(vtk_file)
    except Exception as e:
        print(f"エラー: VTKファイルの読み込みに失敗しました。({e})")
        return

    # 3. 誘電率 (Epsilon) を各セルにマッピング
    mat_array = grid.cell_data.get('Material')
    if mat_array is None:
        print("エラー: VTKファイルに 'Material' 配列が含まれていません。")
        return

    eps_array = np.zeros_like(mat_array, dtype=float)
    for mat_id, eps in enumerate(epsilons):
        eps_array[mat_array == mat_id] = eps
    
    grid.cell_data['Epsilon'] = eps_array

    # 4. 描画データの抽出 (空気・PMLを除外)
    print("3Dメッシュを構築中 (内部構造抽出)...")
    pml_array = grid.cell_data.get('Sources_PML', np.zeros_like(mat_array))
    rx_array = grid.cell_data.get('Receivers', np.zeros_like(mat_array))

    valid_mask = (eps_array > 1.01) & (pml_array != 1)
    source_mask = (pml_array == 2)
    rx_mask = (rx_array == 1)

    # まず抽出だけを行う
    geo_grid = grid.extract_cells(valid_mask)
    source_cells = grid.extract_cells(source_mask)
    rx_cells = grid.extract_cells(rx_mask)

    # 抽出した「実体モデル」の角が厳密に(0, 0, 0)になるよう平行移動
    if geo_grid.n_cells > 0:
        xmin, xmax, ymin, ymax, zmin, zmax = geo_grid.bounds
        shift_x, shift_y, shift_z = -xmin, -ymin, -zmin
        
        # モデルと送受信機の座標を一斉にシフト
        geo_grid.translate((shift_x, shift_y, shift_z), inplace=True)
        if source_cells.n_cells > 0:
            source_cells.translate((shift_x, shift_y, shift_z), inplace=True)
        if rx_cells.n_cells > 0:
            rx_cells.translate((shift_x, shift_y, shift_z), inplace=True)

    # 5. 送受信機の座標抽出
    print("送受信機をマッピング中...")
    source_pts = source_cells.cell_centers().points if source_cells.n_cells > 0 else []
    rx_pts = rx_cells.cell_centers().points if rx_cells.n_cells > 0 else []

    spacing = grid.spacing[0] if hasattr(grid, 'spacing') else 0.01
    sphere_radius = spacing * 3.0

    # 6. プロットのセットアップ
    plotter = pv.Plotter(off_screen=True, window_size=[1600, 1400])
    plotter.set_background('white')
    plotter.enable_depth_peeling() 

    # ★修正ポイント: VTKの自動タイトルを無効化し、use_2d=True でメモリの埋もれを防止
    if geo_grid.n_cells > 0:
        plotter.show_bounds(
            mesh=geo_grid,
            grid=True,
            location='origin',
            padding=0.0,
            xtitle='', # 自動生成のタイトルは一旦空にする
            ytitle='',
            ztitle='',
            color='black',
            font_size=16,
            show_xlabels=True,
            show_ylabels=True,
            show_zlabels=True,
            use_2d=True # メモリラベルを2Dオーバーレイとして描画し、モデルに隠れるのを防ぐ
        )

        # ★修正ポイント: 各軸の先端にX, Y, Zのラベルを手動で強制配置（常に最前面に表示）
        xmin, xmax, ymin, ymax, zmin, zmax = geo_grid.bounds
        offset_x = xmax * 0.05
        offset_y = ymax * 0.05
        offset_z = zmax * 0.05
        
        label_coords = np.array([
            [xmax + offset_x, 0, 0],
            [0, ymax + offset_y, 0],
            [0, 0, zmax + offset_z]
        ])
        
        plotter.add_point_labels(
            label_coords,
            ['X', 'Y', 'Z'],
            text_color='black',
            font_size=24,
            shape=None,          # 背景の枠線を消す
            show_points=False,   # ラベル位置のドットを消す
            always_visible=True  # モデルに隠れず常に表示する
        )

    # 抽出済み・移動済みの geo_grid から誘電率ごとにメッシュを描画
    unique_eps = np.unique(geo_grid.cell_data['Epsilon'])
    added_scalar_bar = False

    for eps_val in unique_eps:
        mask = geo_grid.cell_data['Epsilon'] == eps_val
        sub_grid = geo_grid.extract_cells(mask)
        
        if sub_grid.n_cells > 0:
            plotter.add_mesh(
                sub_grid,
                scalars='Epsilon',
                cmap='jet',
                clim=[1.0, 9.0],
                opacity=0.3,
                show_edges=False,
                show_scalar_bar=not added_scalar_bar,
                scalar_bar_args={
                    'title': 'Relative Permittivity', 
                    'color': 'black',
                    'vertical': True,
                    'position_x': 0.90,
                    'position_y': 0.05,
                    'height': 0.8,
                    'width': 0.05
                } if not added_scalar_bar else None
            )
            added_scalar_bar = True

    # 送信機の描画 (赤色)
    for pt in source_pts:
        plotter.add_mesh(pv.Sphere(radius=sphere_radius, center=pt), color='red')

    # 受信機の描画 (青色)
    for pt in rx_pts:
        plotter.add_mesh(pv.Sphere(radius=sphere_radius, center=pt), color='blue')

    # Tx, Rxのラベルサイズ・形状設定
    legend_entries = [
        ['Tx', 'red'],
        ['Rx', 'blue']
    ]
    plotter.add_legend(
        legend_entries, 
        loc='lower right', 
        bcolor='white', 
        border=True, 
        face='circle',
        size=(0.06, 0.06)
    )

    # 7. 複数視点からの画像出力設定
    out_dir = os.path.join(vtk_dir, 'gemetory')
    os.makedirs(out_dir, exist_ok=True)

    views = {
        'oblique_1': (25, 45),
        'oblique_2': (25, 135),
        'oblique_3': (25, 225),
        'oblique_4': (25, 315),
    }

    print(f"画像を {out_dir} に保存しています...")

    # 真上
    plotter.view_xy()
    plotter.screenshot(os.path.join(out_dir, f"{base_name}_top.png"))
    print("保存完了: top")

    # 真横
    plotter.view_xz()
    plotter.screenshot(os.path.join(out_dir, f"{base_name}_side_y.png"))
    print("保存完了: side_y")

    plotter.view_yz()
    plotter.screenshot(os.path.join(out_dir, f"{base_name}_side_x.png"))
    print("保存完了: side_x")

    # 斜め
    for view_name, (elev, azim) in views.items():
        elev_rad, azim_rad = np.radians(elev), np.radians(azim)
        cam_x = np.cos(elev_rad) * np.cos(azim_rad)
        cam_y = np.cos(elev_rad) * np.sin(azim_rad)
        cam_z = np.sin(elev_rad)
        
        plotter.view_vector((cam_x, cam_y, cam_z), viewup=(0, 0, 1))
        
        out_path = os.path.join(out_dir, f"{base_name}_{view_name}.png")
        plotter.screenshot(out_path)
        print(f"保存完了: {view_name}")

    plotter.close()
    print("すべての処理が完了しました。")


def main():
    vtk_file = input("VTKファイルのパスを入力してください: ").strip()
    
    vtk_file = vtk_file.replace("'", "").replace('"', '')

    if not os.path.exists(vtk_file):
        print("エラー: 指定されたファイルが見つかりません。")
        return
        
    plot_and_save_vtk(vtk_file)


if __name__ == "__main__":
    main()