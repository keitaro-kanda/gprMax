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

# 上端か下端かの選択
top_or_bottom = input("Select 'top' or 'bottom': ").strip().lower()
if top_or_bottom not in ['top', 'bottom']:
    raise ValueError("Invalid selection. Please choose 'top' or 'bottom'.")

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

# 各ラベル番号が得られた数をカウントし、txt形式で出力
label_counts = {label: np.sum(grid == label) for label in np.unique(grid)}
output_dir = os.path.dirname(file_path)
output_path = os.path.join(output_dir, "label_counts.txt")
with open(output_path, 'w') as f:
    for label, count in label_counts.items():
        f.write(f"Label {label}: {count}\n")
print(f"[INFO] Label counts saved to {output_path}")


# 離散カラーマップの定義（例：1→red, 2→blue, 3, 4, 5→green）
cmap = colors.ListedColormap(["red", "blue", "green", "green", "green"])
norm = colors.BoundaryNorm(boundaries=[0.5, 1.5, 2.5, 3.5], ncolors=3)

### 旧バージョン：Type1-5
# # 離散カラーマップの定義（例：1→red, 2→blue, 3→green, 4→orange, 5→purple）
# cmap = colors.ListedColormap(["red", "blue", "green", "orange", "purple"])
# norm = colors.BoundaryNorm(boundaries=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], ncolors=5)


### Fresnel半径の計算
# 各種パラメータの設定
h_antenna = 30 # [cm]
d_rock = np.ones(316) * 200 # [cm]
print(d_rock.shape)
h_rock = np.arange(0, 316, 1.0) # [cm]
print(h_rock.shape)
er_regolith = 3.0
er_rock = 9.0
wavelength = 60 # [cm]
# 光路長の計算
L_top = h_antenna  + d_rock * np.sqrt(er_regolith)
L_bottom = h_antenna + d_rock * np.sqrt(er_regolith) + h_rock * np.sqrt(er_rock) # [cm]
# Fresnel半径の計算
r_fresnel_top = np.sqrt(wavelength * L_top / 2) # [cm]
r_fresnel_bottom = np.sqrt(wavelength * L_bottom / 2) # [cm]

# r_fresnel_multi_medium = np.sqrt(wavelength * h_antenna * 2 + wavelength / np.sqrt(er_regolith) * d_rock * np.sqrt(er_regolith) * 2 + wavelength / np.sqrt(er_rock) * h_rock * np.sqrt(er_rock)) / 2


### プロット作成
fig, ax = plt.subplots(figsize=(8, 6))

# extentを設定してbinの中心が実際のcm値になるようにする
width_cm = np.array(unique_widths) * 100  # m to cm
height_cm = np.array(unique_heights) * 100  # m to cm
width_step = width_cm[1] - width_cm[0] if len(width_cm) > 1 else 1
height_step = height_cm[1] - height_cm[0] if len(height_cm) > 1 else 1

extent = [width_cm[0] - width_step/2, width_cm[-1] + width_step/2,
          height_cm[0] - height_step/2, height_cm[-1] + height_step/2]

im = ax.imshow(grid, origin="lower", interpolation="none", cmap=cmap, norm=norm, extent=extent)

# 軸ラベルとタイトル（m単位の値をcm単位に変換して表示）
ax.set_xlabel("Rock width (cm)", fontsize=20)
ax.set_ylabel("Rock height (cm)", fontsize=20)
ax.tick_params(labelsize=16)

# 軸目盛りの設定（実際のcm値をtick位置に設定）
ax.set_xticks(width_cm)
ax.set_xticklabels([f"{w:.0f}" for w in width_cm])
ax.set_yticks(height_cm)
ax.set_yticklabels([f"{h:.0f}" for h in height_cm])

# カラーバーの作成：make_axes_locatableを利用してメインプロットの高さに合わせる
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, ticks=[1, 2, 3, 4, 5])
cbar.ax.set_yticklabels([f"Type {i}" for i in [1, 2, 3, 4, 5]], fontsize=20)
cbar.ax.tick_params(labelsize=16)

# JSONファイルと同じディレクトリにpngおよびpdfで保存
directory = os.path.dirname(os.path.abspath(file_path))
if top_or_bottom == 'top':
    png_path = os.path.join(directory, "result_summary_top.png")
    pdf_path = os.path.join(directory, "result_summary_top.pdf")
elif top_or_bottom == 'bottom':
    png_path = os.path.join(directory, "result_summary_bottom.png")
    pdf_path = os.path.join(directory, "result_summary_bottom.pdf")
plt.savefig(png_path, dpi=150, format='png', bbox_inches='tight')
#plt.savefig(pdf_path, dpi=300, format='pdf', bbox_inches='tight')

# プロット表示
plt.show()


### Fresnel半径を含めたプロット
fig, ax = plt.subplots(figsize=(8, 6))

# 2番目のプロットでも同じextent設定を使用
im = ax.imshow(grid, origin="lower", interpolation="none", cmap=cmap, norm=norm,
                extent=extent)
# ax.vlines(r_fresnel_top, ymin=0, ymax=300,  color='w', linestyle='-')
if top_or_bottom == 'top':
    ax.plot(r_fresnel_top, h_rock, color='w', linestyle='-.')
elif top_or_bottom == 'bottom':
    ax.plot(r_fresnel_bottom, h_rock, color='w', linestyle='-.')
# ax.plot(r_fresnel_multi_medium, h_rock, color='k', linestyle='-.')

# 軸ラベルとタイトル（m単位の値をcm単位に変換して表示）
ax.set_xlabel("Rock width (cm)", fontsize=20)
ax.set_ylabel("Rock height (cm)", fontsize=20)
ax.tick_params(labelsize=16)

# 軸目盛りの設定（実際のcm値をtick位置に設定）
ax.set_xticks(width_cm)
ax.set_xticklabels([f"{w:.0f}" for w in width_cm])
ax.set_yticks(height_cm)
ax.set_yticklabels([f"{h:.0f}" for h in height_cm])

ax.set_xlim([width_cm[0] - width_step/2, width_cm[-1] + width_step/2])
ax.set_ylim([height_cm[0] - height_step/2, height_cm[-1] + height_step/2])

# カラーバーの作成：make_axes_locatableを利用してメインプロットの高さに合わせる
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, ticks=[1, 2, 3, 4, 5])
cbar.ax.set_yticklabels([f"Type {i}" for i in [1, 2, 3, 4, 5]], fontsize=20)
cbar.ax.tick_params(labelsize=16)

# JSONファイルと同じディレクトリにpngおよびpdfで保存
directory = os.path.dirname(os.path.abspath(file_path))
if top_or_bottom == 'top':
    png_path = os.path.join(directory, "result_summary_with_fresnel_top.png")
    pdf_path = os.path.join(directory, "result_summary_with_fresnel_top.pdf")
elif top_or_bottom == 'bottom':
    png_path = os.path.join(directory, "result_summary_with_fresnel_bottom.png")
    pdf_path = os.path.join(directory, "result_summary_with_fresnel_bottom.pdf")
plt.savefig(png_path, dpi=150, format='png', bbox_inches='tight')
#plt.savefig(pdf_path, dpi=300, format='pdf', bbox_inches='tight')

# プロット表示
plt.show()