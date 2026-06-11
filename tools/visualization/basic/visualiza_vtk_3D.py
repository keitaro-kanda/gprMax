#!/usr/bin/env python3
import os
import glob
import platform
import shutil
import subprocess
import pyvista as pv
import numpy as np


# ============================================================
# ★ 見え方の設定
# ============================================================
MODES = ['slices', 'clip', 'opaque']
#   'slices' : 直交3断面を不透明表示（地下構造の定番・おすすめ）
#   'clip'   : 手前の角を切り取って内部断面を露出
#   'opaque' : 箱を完全不透明（表面の分布がくっきり）

CMAP = 'turbo'           # 'viridis' も推奨
CLIM = (1.0, 6.0)        # 誘電率の表示範囲（将来 eps=6 を導入予定のため固定）

# 生成後に macOS の ._ ファイル(AppleDouble)を出力先から削除する（Macのみ有効）
CLEAN_APPLEDOUBLE = True
# ============================================================


def build_plotter(mode, geo_grid):
    """指定したモードで Plotter を構築して返す。"""
    plotter = pv.Plotter(off_screen=True, window_size=[1600, 1400])
    plotter.set_background('white')

    scalar_bar_args = {
        'title': 'Relative Permittivity',
        'color': 'black',
        'vertical': True,
        'position_x': 0.90,
        'position_y': 0.05,
        'height': 0.8,
        'width': 0.05,
        'n_labels': 6,
    }

    common_kwargs = dict(
        scalars='Epsilon',
        cmap=CMAP,
        clim=list(CLIM),
        show_edges=False,
        scalar_bar_args=scalar_bar_args,
    )

    # --- 表示モードごとの描画（不透明） ---
    if mode == 'opaque':
        plotter.add_mesh(geo_grid, opacity=1.0, **common_kwargs)

    elif mode == 'clip':
        gxmin, gxmax, gymin, gymax, gzmin, gzmax = geo_grid.bounds
        xm = (gxmin + gxmax) / 2.0
        ym = (gymin + gymax) / 2.0
        zm = (gzmin + gzmax) / 2.0
        # +x,+y,+z 側の角(1/8)を取り除いて内部断面を露出
        clipped = geo_grid.clip_box(
            bounds=[xm, gxmax, ym, gymax, zm, gzmax], invert=True
        )
        plotter.add_mesh(clipped, opacity=1.0, **common_kwargs)
        plotter.add_mesh(geo_grid.outline(), color='gray', line_width=1)

    else:  # 'slices'
        slices = geo_grid.slice_orthogonal()
        plotter.add_mesh(slices, opacity=1.0, **common_kwargs)
        plotter.add_mesh(geo_grid.outline(), color='gray', line_width=1)

    # --- 軸メモリ＆グリッド線を全方向に表示 ---
    #   grid='all'     : 全ての面にグリッド線
    #   location='all' : 座標メモリ(目盛)を全方向の面に表示
    #   （ごちゃつく場合は location を 'outer' にすると外周だけに整理されます）
    plotter.show_bounds(
        mesh=geo_grid,
        grid='all',
        location='all',
        ticks='both',
        padding=0.0,
        xtitle='X', ytitle='Y', ztitle='Z',
        color='black',
        font_size=12,
        show_xlabels=True,
        show_ylabels=True,
        show_zlabels=True,
    )

    return plotter


def save_all_views(plotter, out_dir, base_name, mode):
    """全視点の画像を保存する。ファイル名にモード名を含める。"""
    views = {
        'oblique_1': (25, 45),
        'oblique_2': (25, 135),
        'oblique_3': (25, 225),
        'oblique_4': (25, 315),
    }

    plotter.view_xy()
    plotter.screenshot(os.path.join(out_dir, f"{base_name}_{mode}_top.png"))
    print(f"  保存完了: {mode}_top")

    plotter.view_xz()
    plotter.screenshot(os.path.join(out_dir, f"{base_name}_{mode}_side_y.png"))
    print(f"  保存完了: {mode}_side_y")

    plotter.view_yz()
    plotter.screenshot(os.path.join(out_dir, f"{base_name}_{mode}_side_x.png"))
    print(f"  保存完了: {mode}_side_x")

    for view_name, (elev, azim) in views.items():
        elev_rad, azim_rad = np.radians(elev), np.radians(azim)
        cam_x = np.cos(elev_rad) * np.cos(azim_rad)
        cam_y = np.cos(elev_rad) * np.sin(azim_rad)
        cam_z = np.sin(elev_rad)
        plotter.view_vector((cam_x, cam_y, cam_z), viewup=(0, 0, 1))
        out_path = os.path.join(out_dir, f"{base_name}_{mode}_{view_name}.png")
        plotter.screenshot(out_path)
        print(f"  保存完了: {mode}_{view_name}")


def clean_appledouble(out_dir):
    """macOS が生成する ._ ファイルを出力先から掃除する。"""
    if platform.system() != 'Darwin':
        return
    # dot_clean があれば最優先（._ をマージ/削除する純正コマンド）
    if shutil.which('dot_clean'):
        try:
            subprocess.run(['dot_clean', '-m', out_dir], check=False)
        except Exception:
            pass
    # 残った ._ ファイルを念のため直接削除
    removed = 0
    for f in glob.glob(os.path.join(out_dir, '._*')):
        try:
            os.remove(f)
            removed += 1
        except OSError:
            pass
    if removed:
        print(f"._ ファイルを {removed} 件削除しました。")


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

    valid_mask = (eps_array > 1.01) & (pml_array != 1)
    geo_grid = grid.extract_cells(valid_mask)

    if geo_grid.n_cells == 0:
        print("エラー: 描画対象のセルが見つかりません。")
        return

    # 抽出した「実体モデル」の角が厳密に(0, 0, 0)になるよう平行移動
    xmin, xmax, ymin, ymax, zmin, zmax = geo_grid.bounds
    geo_grid.translate((-xmin, -ymin, -zmin), inplace=True)

    eps_vals = geo_grid.cell_data['Epsilon']
    print(f"カラースケール clim = {list(CLIM)}  "
          f"(実データ範囲: {eps_vals.min():.2f} 〜 {eps_vals.max():.2f})")

    # 5. 出力先
    out_dir = os.path.join(vtk_dir, 'gemetory')
    os.makedirs(out_dir, exist_ok=True)
    print(f"画像を {out_dir} に保存しています...")

    # 6. 各モードを自動で保存
    for mode in MODES:
        print(f"[モード: {mode}] を描画中...")
        plotter = build_plotter(mode, geo_grid)
        save_all_views(plotter, out_dir, base_name, mode)
        plotter.close()

    # 7. ._ ファイルの掃除
    if CLEAN_APPLEDOUBLE:
        clean_appledouble(out_dir)

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