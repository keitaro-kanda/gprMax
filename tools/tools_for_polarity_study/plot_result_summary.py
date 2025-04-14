import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# JSONファイルのパスをinput()で取得
file_path = input("jsonファイルのパスを入力してください: ")

# JSONファイルの読み込み
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

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

# 離散カラーマップの定義（例：1→red, 2→blue, 3→green, 4→orange, 5→purple）
cmap = colors.ListedColormap(["red", "blue", "green", "orange", "purple"])
norm = colors.BoundaryNorm(boundaries=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], ncolors=5)

# プロット作成（subplotsを利用）
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(grid, origin="lower", interpolation="none", cmap=cmap, norm=norm)

# 軸ラベルとタイトル（m単位の値をcm単位に変換して表示）
ax.set_xlabel("Rock width (cm)", fontsize=20)
ax.set_ylabel("Rock height (cm)", fontsize=20)
ax.tick_params(labelsize=16)

# 軸目盛りの設定（m単位の値をcmに変換して表示）
xtick_labels = [f"{w*100:.0f}" for w in unique_widths]
ytick_labels = [f"{h*100:.0f}" for h in unique_heights]
ax.set_xticks(np.arange(len(unique_widths)))
ax.set_xticklabels(xtick_labels)
ax.set_yticks(np.arange(len(unique_heights)))
ax.set_yticklabels(ytick_labels)

# カラーバーの作成：make_axes_locatableを利用してメインプロットの高さに合わせる
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, ticks=[1, 2, 3, 4, 5])
cbar.ax.set_yticklabels([f"Type {i}" for i in [1, 2, 3, 4, 5]], fontsize=20)
cbar.ax.tick_params(labelsize=16)

# JSONファイルと同じディレクトリにpngおよびpdfで保存
directory = os.path.dirname(os.path.abspath(file_path))
png_path = os.path.join(directory, "result_summary.png")
pdf_path = os.path.join(directory, "result_.pdf")
plt.savefig(png_path, dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
plt.savefig(pdf_path, dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)

# プロット表示
plt.show()
