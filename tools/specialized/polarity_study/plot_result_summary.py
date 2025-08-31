import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# JSONファイルのパスをinput()で取得
file_path = input("jsonファイルのパスを入力してください: ")
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"指定されたファイルが存在しません: {file_path}")

# Max-peakかTWTかの選択
analysis_type = input("Select analysis type ('1: max-peak' or '2: TWT'): ").strip().lower()
if analysis_type not in ['1', '2']:
    raise ValueError("Invalid selection. Please choose '1: max-peak' or '2: TWT'.")
if analysis_type == '1':
    analysis_type = 'max-peak'
else:
    analysis_type = 'TWT'

# 上端か下端かの選択
top_or_bottom = input("Select '1: top' or '2: bottom': ").strip().lower()
if top_or_bottom not in ['1', '2']:
    raise ValueError("Invalid selection. Please choose '1: top' or '2: bottom'.")
if top_or_bottom == '1':
    top_or_bottom = 'top'
else:
    top_or_bottom = 'bottom'

# JSONファイルの読み込み
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

def create_grid_from_data(data):
    """データからグリッドを作成する共通関数"""
    # 各データを抽出し、ユニークなheightおよびwidthのリスト（m単位）を作成
    heights_all = []
    widths_all = []
    for key, values in data.items():
        # valuesの形式: [height, width, 計算結果]
        h, w, _ = values
        heights_all.append(h)
        widths_all.append(w)

    unique_heights = sorted(set(heights_all))
    unique_widths = sorted(set(widths_all))

    # グリッドの形状（行: height, 列: width）を確定し、各セルに計算結果を格納
    grid = np.empty((len(unique_heights), len(unique_widths)), dtype=int)
    for key, values in data.items():
        h, w, label = values
        row = unique_heights.index(h)
        col = unique_widths.index(w)
        grid[row, col] = label
    
    return grid, unique_heights, unique_widths

def calculate_fresnel_radius(top_or_bottom):
    """Fresnel半径を計算する共通関数"""
    # 各種パラメータの設定
    h_antenna = 30  # [cm]
    d_rock = np.ones(316) * 200  # [cm]
    h_rock = np.arange(0, 316, 1.0)  # [cm]
    er_regolith = 3.0
    er_rock = 9.0
    wavelength = 60  # [cm]
    
    # 光路長の計算
    L_top = h_antenna + d_rock * np.sqrt(er_regolith)
    L_bottom = h_antenna + d_rock * np.sqrt(er_regolith) + h_rock * np.sqrt(er_rock)  # [cm]
    
    # Fresnel半径の計算
    r_fresnel_top = np.sqrt(wavelength * L_top / 2)  # [cm]
    r_fresnel_bottom = np.sqrt(wavelength * L_bottom / 2)  # [cm]
    
    if top_or_bottom == 'top':
        return r_fresnel_top, h_rock
    else:
        return r_fresnel_bottom, h_rock

# TWT解析の場合、対応するMax-peak解析のデータを取得
max_peak_data = None
if analysis_type == 'TWT':
    max_peak_data_path = file_path.replace('result_use_TWT', 'result_use_peak')
    with open(max_peak_data_path, 'r', encoding='utf-8') as f:
        max_peak_data = json.load(f)

# グリッドの作成
grid, unique_heights, unique_widths = create_grid_from_data(data)

# 各ラベル番号が得られた数をカウントし、txt形式で出力
label_counts = {label: np.sum(grid == label) for label in np.unique(grid)}
output_dir = os.path.dirname(file_path)
output_path = os.path.join(output_dir, "label_counts.txt")
with open(output_path, 'w') as f:
    for label, count in label_counts.items():
        f.write(f"Label {label}: {count}\n")
print(f"[INFO] Label counts saved to {output_path}")


def get_colormap():
    """3色カラーマップを取得する"""
    cmap = colors.ListedColormap(["red", "blue", "green"])
    norm = colors.BoundaryNorm(boundaries=[0.5, 1.5, 2.5, 3.5], ncolors=3)
    return cmap, norm

def plot_max_peak_mode(grid, unique_heights, unique_widths, top_or_bottom, file_path):
    """Max-peakモードのプロット作成"""
    cmap, norm = get_colormap()
    r_fresnel, h_rock = calculate_fresnel_radius(top_or_bottom)
    
    # 軸の設定（cm単位に変換）
    width_cm = np.array(unique_widths) * 100
    height_cm = np.array(unique_heights) * 100
    width_step = width_cm[1] - width_cm[0] if len(width_cm) > 1 else 1
    height_step = height_cm[1] - height_cm[0] if len(height_cm) > 1 else 1
    extent = [width_cm[0] - width_step/2, width_cm[-1] + width_step/2,
              height_cm[0] - height_step/2, height_cm[-1] + height_step/2]
    
    # 基本プロット
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, origin="lower", interpolation="none", cmap=cmap, norm=norm, extent=extent)
    
    # 軸ラベルと設定
    ax.set_xlabel("Rock width (cm)", fontsize=20)
    ax.set_ylabel("Rock height (cm)", fontsize=20)
    ax.tick_params(labelsize=16)
    ax.set_xticks(width_cm)
    ax.set_xticklabels([f"{w:.0f}" for w in width_cm])
    ax.set_yticks(height_cm)
    ax.set_yticklabels([f"{h:.0f}" for h in height_cm])
    
    # カラーバー
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, ticks=[1, 2, 3])
    cbar.ax.set_yticklabels([f"Type {i}" for i in [1, 2, 3]], fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    
    # 保存
    directory = os.path.dirname(os.path.abspath(file_path))
    png_path = os.path.join(directory, f"result_summary_{top_or_bottom}.png")
    plt.savefig(png_path, dpi=150, format='png', bbox_inches='tight')
    plt.show()
    
    # Fresnel半径付きプロット
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, origin="lower", interpolation="none", cmap=cmap, norm=norm, extent=extent)
    ax.plot(r_fresnel, h_rock, color='w', linestyle='-.')
    
    # 軸ラベルと設定
    ax.set_xlabel("Rock width (cm)", fontsize=20)
    ax.set_ylabel("Rock height (cm)", fontsize=20)
    ax.tick_params(labelsize=16)
    ax.set_xticks(width_cm)
    ax.set_xticklabels([f"{w:.0f}" for w in width_cm])
    ax.set_yticks(height_cm)
    ax.set_yticklabels([f"{h:.0f}" for h in height_cm])
    ax.set_xlim([width_cm[0] - width_step/2, width_cm[-1] + width_step/2])
    ax.set_ylim([height_cm[0] - height_step/2, height_cm[-1] + height_step/2])
    
    # カラーバー
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, ticks=[1, 2, 3])
    cbar.ax.set_yticklabels([f"Type {i}" for i in [1, 2, 3]], fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    
    # 保存
    png_path = os.path.join(directory, f"result_summary_with_fresnel_{top_or_bottom}.png")
    plt.savefig(png_path, dpi=150, format='png', bbox_inches='tight')
    plt.show()

def plot_twt_mode(twt_grid, max_peak_grid, unique_heights, unique_widths, top_or_bottom, file_path):
    """TWTモードのプロット作成（3色カラーマップ + マーカー重ね合わせ）"""
    cmap, norm = get_colormap()
    r_fresnel, h_rock = calculate_fresnel_radius(top_or_bottom)
    
    # 軸の設定（cm単位に変換）
    width_cm = np.array(unique_widths) * 100
    height_cm = np.array(unique_heights) * 100
    width_step = width_cm[1] - width_cm[0] if len(width_cm) > 1 else 1
    height_step = height_cm[1] - height_cm[0] if len(height_cm) > 1 else 1
    extent = [width_cm[0] - width_step/2, width_cm[-1] + width_step/2,
              height_cm[0] - height_step/2, height_cm[-1] + height_step/2]
    
    # TWTマーカーの準備（1→○、2→×）
    marker_positions_o = []
    marker_positions_x = []
    
    for i in range(len(unique_heights)):
        for j in range(len(unique_widths)):
            if twt_grid[i, j] == 1:  # ○マーカー
                marker_positions_o.append((width_cm[j], height_cm[i]))
            elif twt_grid[i, j] == 2:  # ×マーカー
                marker_positions_x.append((width_cm[j], height_cm[i]))
    
    # 基本プロット（Max-peakの3色カラーマップ + TWTマーカー）
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(max_peak_grid, origin="lower", interpolation="none", cmap=cmap, norm=norm, extent=extent)
    
    # マーカーの追加
    if marker_positions_o:
        o_x, o_y = zip(*marker_positions_o)
        ax.scatter(o_x, o_y, marker='o', s=80, c='white', linewidth=2)
    if marker_positions_x:
        x_x, x_y = zip(*marker_positions_x)
        ax.scatter(x_x, x_y, marker='x', s=80, c='white', linewidth=3)
    
    # 軸ラベルと設定
    ax.set_xlabel("Rock width (cm)", fontsize=20)
    ax.set_ylabel("Rock height (cm)", fontsize=20)
    ax.tick_params(labelsize=16)
    ax.set_xticks(width_cm)
    ax.set_xticklabels([f"{w:.0f}" for w in width_cm])
    ax.set_yticks(height_cm)
    ax.set_yticklabels([f"{h:.0f}" for h in height_cm])
    
    # カラーバー
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, ticks=[1, 2, 3])
    cbar.ax.set_yticklabels([f"Type {i}" for i in [1, 2, 3]], fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    
    # 保存
    directory = os.path.dirname(os.path.abspath(file_path))
    png_path = os.path.join(directory, f"result_summary_twt_{top_or_bottom}.png")
    plt.savefig(png_path, dpi=150, format='png', bbox_inches='tight')
    plt.show()
    
    # Fresnel半径付きプロット
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(max_peak_grid, origin="lower", interpolation="none", cmap=cmap, norm=norm, extent=extent)
    ax.plot(r_fresnel, h_rock, color='w', linestyle='-.')
    
    # マーカーの追加
    if marker_positions_o:
        o_x, o_y = zip(*marker_positions_o)
        ax.scatter(o_x, o_y, marker='o', s=80, c='white', linewidth=2)
    if marker_positions_x:
        x_x, x_y = zip(*marker_positions_x)
        ax.scatter(x_x, x_y, marker='x', s=80, c='white', linewidth=3)

    # 軸ラベルと設定
    ax.set_xlabel("Rock width (cm)", fontsize=20)
    ax.set_ylabel("Rock height (cm)", fontsize=20)
    ax.tick_params(labelsize=16)
    ax.set_xticks(width_cm)
    ax.set_xticklabels([f"{w:.0f}" for w in width_cm])
    ax.set_yticks(height_cm)
    ax.set_yticklabels([f"{h:.0f}" for h in height_cm])
    ax.set_xlim([width_cm[0] - width_step/2, width_cm[-1] + width_step/2])
    ax.set_ylim([height_cm[0] - height_step/2, height_cm[-1] + height_step/2])
    
    # カラーバー
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, ticks=[1, 2, 3])
    cbar.ax.set_yticklabels([f"Type {i}" for i in [1, 2, 3]], fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    
    # 保存
    png_path = os.path.join(directory, f"result_summary_twt_with_fresnel_{top_or_bottom}.png")
    plt.savefig(png_path, dpi=150, format='png', bbox_inches='tight')
    plt.show()

### 旧バージョン：Type1-5
# # 離散カラーマップの定義（例：1→red, 2→blue, 3→green, 4→orange, 5→purple）
# cmap = colors.ListedColormap(["red", "blue", "green", "orange", "purple"])
# norm = colors.BoundaryNorm(boundaries=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], ncolors=5)


# プロット作成
if analysis_type == 'max-peak':
    plot_max_peak_mode(grid, unique_heights, unique_widths, top_or_bottom, file_path)
elif analysis_type == 'TWT':
    # TWTモードの場合、max-peakデータからもグリッドを作成
    max_peak_grid, _, _ = create_grid_from_data(max_peak_data)
    plot_twt_mode(grid, max_peak_grid, unique_heights, unique_widths, top_or_bottom, file_path)