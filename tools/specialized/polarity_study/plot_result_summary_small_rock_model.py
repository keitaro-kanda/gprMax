import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator


results_Ricker_circle = ['x', 's', 's', 's', 's', 's', 's', 'd', 'd', '2', '2', '2', '2', '2', '2']
results_Ricker_square = ['x', 'x', 's', 's', 's', 's', 'd', 'd', '2', '2', '2', '2', '2', '2', '2']
results_LPR_circle = ['x', 'x', 's', 's', 's', 's', 'd', 'd', '2', '2', '2', '2', '2', '2', '2']
results_LPR_square = ['x', 's', 's', 's', 's', 'd', '2', '2', '2', '2', '2', '2', '2', '2', '2']
results = [results_Ricker_circle, results_Ricker_square, results_LPR_circle, results_LPR_square]

sizes = np.arange(1, 16, 1)
model_names = ['Bipolar-circle', 'Bipolar-square', 'Unipolar-circle', 'Unipolar-square']


cat_to_int = {'x':0, 's':1, 'd':2, '2':3}
colors    = ['black','red','blue','green']
data = np.array([[cat_to_int[c] for c in row] for row in results])

# 整数配列に変換
data = np.array([[cat_to_int[c] for c in row] for row in results])

# 描画 - 縦に4つのsubplotを作成
fig, axes = plt.subplots(4, 1, figsize=(12, 6), sharex=True)

# 各subplotに1x15のデータを表示
for i, ax in enumerate(axes):
    # 1x15のデータを表示（1行のデータを2D配列として渡すために reshape）
    im = ax.imshow(data[i:i+1], aspect='auto', cmap=plt.matplotlib.colors.ListedColormap(colors),
                   vmin=0, vmax=3, origin='upper')

    # --- メジャー・ティックとラベルはセル中央に ---
    ax.set_xticks(np.arange(len(sizes)))
    
    # X軸ラベルは最下段のsubplotのみ表示
    if i == len(axes) - 1:
        ax.set_xticklabels(sizes, fontsize=16)
        ax.set_xlabel('Rock size / wavelength in the rock', fontsize=20)

        # 波長で規格化した岩石サイズを軸ラベルに設定
        wavelength_in_rock = 3e8 / np.sqrt(9) / 500e6 * 100  # cm
        ax.set_xticklabels([f"{s / wavelength_in_rock:.2f}" for s in sizes], fontsize=16)
    else:
        ax.set_xticklabels([])
    
    # Y軸は各subplotでモデル名をタイトルとして表示
    ax.set_yticks([0])
    ax.set_yticklabels([model_names[i]], fontsize=20)
    ax.set_ylabel('')

    # --- マイナー・ティックをビンの端に設定 --- 
    # x 軸
    N = len(sizes)
    x_edge = np.arange(N+1) - 0.5
    ax.xaxis.set_minor_locator(FixedLocator(x_edge))
    # y 軸（各subplotでは1行のみなので0.5と-0.5）
    y_edge = np.array([-0.5, 0.5])
    ax.yaxis.set_minor_locator(FixedLocator(y_edge))

    # --- グリッド設定 --- 
    # メジャー・グリッドはすべてオフ
    ax.grid(which='major', visible=False)
    # マイナー・グリッドのみ描画（ビン境界）
    ax.grid(which='minor', axis='both', color='white', linewidth=1, linestyle='-.')

# --- 凡例を全体の下部に配置 ---
labels = ['Not detected', 'One echo', 'One echo with dipression',
            'Two echoes']
legend_patches = [Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
fig.legend(
    handles=legend_patches,
    loc='upper center',            # 凡例の基準点を上中央に
    bbox_to_anchor=(0.5, 0.02),    # 全体図の下部中央
    ncol=4,                        # アイテムを横4列に
    frameon=False,                 # 枠線なし
    fontsize=16
)


plt.tight_layout()
plt.savefig('/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_5x5/result_summary_small_rock_model.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
plt.savefig('/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_5x5/result_summary_small_rock_model.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()