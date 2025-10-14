import pyvista as pv
import numpy as np

# --- ユーザー設定項目 ---
# ユーザーがファイルパスを入力する方法を保持
VTK_FILE_PATH = input('Enter VTK file path: ').strip()

material_info = {
    0: {'name': 'pec', 'color': 'gray'},
    1: {'name': 'free_space', 'color': 'cyan'},
    2: {'name': 'regolith', 'color': 'tan'},
    3: {'name': 'basalt', 'color': 'darkred'},
    4: {'name': 'free_space+regolith+...', 'color': 'lightblue'},
    5: {'name': 'basalt+regolith+...', 'color': 'sienna'},
    6: {'name': 'basalt+regolith+...','color': 'chocolate'},
    7: {'name': 'basalt+basalt+...','color': 'maroon'},
}
# -------------------------

try:
    grid = pv.read(VTK_FILE_PATH)
    # スライス表示は元の'Material'データを使うため、ここではアクティブ設定しない
    plotter = pv.Plotter(shape=(1, 2), window_size=[1600, 800])

    # --- 左側のプロット：ボリュームレンダリング (最終修正版) ---
    plotter.subplot(0, 0)
    plotter.add_text("Volume Rendering (PML Corrected)", font_size=15)
    
    # ★★★ ここからが改善点 ★★★
    # 1. PMLをマスクするための新しいデータ配列を作成
    material_array = grid['Material'].copy() # 元のデータをコピー
    pml_array = grid['Sources_PML']
    
    # PML用の新しいIDを定義（既存のIDと衝突しないように大きな値を選ぶ）
    PML_ID = 99 
    
    # 'Sources_PML'が0より大きい場所（PML層）を新しいIDで上書き
    material_array[pml_array > 0] = PML_ID
    
    # 作成した配列を'Material_masked'という名前でグリッドに追加
    grid['Material_masked'] = material_array
    
    # ボリュームレンダリングでは、このマスク済みデータをアクティブにする
    grid.set_active_scalars('Material_masked')

    # 2. マテリアルID -> 不透明度 の対応リストを作成
    #    リストのインデックスがIDに対応する
    
    # マテリアルIDの最大値（PML_IDを含む）に合わせてリストを初期化
    max_id = max(int(grid['Material_masked'].max()), max(material_info.keys()))
    opacities = [0.5] * (max_id + 1) # デフォルトは少し透明

    # 各マテリアルの不透明度を設定
    opacities[1] = 0.0  # free_spaceは完全に透明
    opacities[3] = 1.0  # basaltは不透明
    opacities[PML_ID] = 0.0 # PML (ID=99) は完全に透明

    # 3. 色のリストもPML_IDに対応させる
    color_names = [material_info.get(mid, {}).get('color', 'white') for mid in range(max_id + 1)]
    # PMLの色は描画されないが、念のため設定
    color_names[PML_ID] = 'black' 
    
    # 4. 修正した設定でボリュームレンダリングを追加
    vol = plotter.add_volume(
        grid, # active_scalars ('Material_masked') が使われる
        cmap=color_names,
        opacity=opacities,
        show_scalar_bar=False,
    )
    # ★★★ 改善点ここまで ★★★
    
    vol.prop.shade = True
    
    material_ids_in_data = np.unique(grid['Material'])
    legend_entries = []
    for mid, info in material_info.items():
        if mid in material_ids_in_data:
            legend_entries.append([info['name'], info['color']])
    plotter.add_legend(legend_entries, bcolor=None, face=None)

    # --- 右側のプロット：スライス表示 ---
    plotter.subplot(0, 1)
    plotter.add_text("Interactive Slicer", font_size=15)
    
    # スライサーでは元のマテリアル情報を表示したいので、'scalars'を明示的に指定
    slicer_colors = [material_info.get(mid, {}).get('color', 'white') for mid in range(max(material_info.keys()) + 1)]
    plotter.add_mesh_slice_orthogonal(grid, scalars='Material', cmap=slicer_colors)


    # プロットウィンドウを表示
    plotter.link_views()
    plotter.view_isometric()
    plotter.show()

except FileNotFoundError:
    print(f"エラー: ファイル '{VTK_FILE_PATH}' が見つかりません。パスを確認してください。")
except KeyError as e:
    print(f"エラー: VTKファイル内に必要なデータ配列 '{e}' が見つかりません。")
except Exception as e:
    print(f"エラーが発生しました: {e}")