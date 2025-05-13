import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator


results_Ricker_circle = ['x', 'x', 's', 's', 's', 's', 's', 'd', 'd', '2', '2', '2', '2', '2', '2']
results_Ricker_square = ['x', 'x', 's', 's', 's', 's', 'd', 'd', '2', '2', '2', '2', '2', '2', '2']
results_LPR_circle = ['x', 'x', 's', 's', 's', 's', 'd', 'd', '2', '2', '2', '2', '2', '2', '2']
results_LPR_square = ['x', 'x', 's', 's', 's', 'd', '2', '2', '2', '2', '2', '2', '2', '2', '2']
results = [results_Ricker_circle, results_Ricker_square, results_LPR_circle, results_LPR_square]

sizes = np.arange(1, 16, 1)
model_names = ['Ricker-circle', 'Ricker-square', 'LPR-circle', 'LPR-square']


cat_to_int = {'x':0, 's':1, 'd':2, '2':3}
colors    = ['black','red','blue','green']
data = np.array([[cat_to_int[c] for c in row] for row in results])

# 整数配列に変換
data = np.array([[cat_to_int[c] for c in row] for row in results])

# 描画
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(data, aspect='auto', cmap=plt.matplotlib.colors.ListedColormap(colors),
                vmin=0, vmax=3, origin='upper')

# --- メジャー・ティックとラベルはセル中央に ---
ax.set_xticks(np.arange(len(sizes)))
ax.set_xticklabels(sizes, fontsize=16)
ax.set_xlabel('Rock size [cm]', fontsize=20)

ax.set_yticks(np.arange(len(model_names)))
ax.set_yticklabels(model_names, fontsize=20)

# --- マイナー・ティックをビンの端に設定 --- 
# x 軸
N = len(sizes)
x_edge = np.arange(N+1) - 0.5
ax.xaxis.set_minor_locator(FixedLocator(x_edge))
# y 軸
M = len(model_names)
y_edge = np.arange(M+1) - 0.5
ax.yaxis.set_minor_locator(FixedLocator(y_edge))

# --- グリッド設定 --- 
# メジャー・グリッドはすべてオフ
ax.grid(which='major', visible=False)
# マイナー・グリッドのみ描画（ビン境界）
ax.grid(which='minor', axis='both', color='white', linewidth=1, linestyle='-.')

# --- 凡例 ---
labels = ['Not detected', 'One echo', 'One echo with dipression',
            'Two echoes']
legend_patches = [Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
ax.legend(
    handles=legend_patches,
    loc='upper center',            # 凡例の基準点を上中央に
    bbox_to_anchor=(0.4, -0.12),   # プロット域の下中央 (x=0.5, y=-0.15)
    ncol=4,                        # アイテムを横4列に
    frameon=False,                  # 枠線なし、必要に応じて True に
    fontsize=16
)


plt.tight_layout()
plt.savefig('/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_5x5/result_summary_small_rock_model.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
plt.savefig('/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_5x5/result_summary_small_rock_model.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()