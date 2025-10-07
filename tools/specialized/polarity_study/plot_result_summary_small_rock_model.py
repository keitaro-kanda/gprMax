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

# 描画 - bipolarとunipolarで分けて上下に配置
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(12, 10))

# gridspecで5行（bipolar 2行 + 隙間 1行 + unipolar 2行）を定義
# height_ratios: [bipolar-1, bipolar-2, 隙間, unipolar-1, unipolar-2]
gs = GridSpec(5, 1, figure=fig, height_ratios=[1, 1, 0.8, 1, 1])

# 各subplotを配置
ax_bipolar_circle = fig.add_subplot(gs[0, 0])   # 1段目: Bipolar-circle
ax_bipolar_square = fig.add_subplot(gs[1, 0])   # 2段目: Bipolar-square
ax_unipolar_circle = fig.add_subplot(gs[3, 0])  # 3段目: Unipolar-circle
ax_unipolar_square = fig.add_subplot(gs[4, 0])  # 4段目: Unipolar-square

# bipolarペアでX軸を共有
ax_bipolar_circle.sharex(ax_bipolar_square)
# unipolarペアでX軸を共有
ax_unipolar_circle.sharex(ax_unipolar_square)

axes = [ax_bipolar_circle, ax_bipolar_square, ax_unipolar_circle, ax_unipolar_square]

# 各subplotに1x15のデータを表示
for i, ax in enumerate(axes):
    # 1x15のデータを表示（1行のデータを2D配列として渡すために reshape）
    im = ax.imshow(data[i:i+1], aspect='auto', cmap=plt.matplotlib.colors.ListedColormap(colors),
                   vmin=0, vmax=3, origin='upper')

    # --- メジャー・ティックとラベルはセル中央に ---
    ax.set_xticks(np.arange(len(sizes)))
    
    # X軸ラベル設定
    range_resolution_unipolar = 7.3 # cm
    range_resolution_bipolar = 7.8 # cm

    # 1段目・2段目：bipolarで規格化
    # 1段目：上横軸に岩石サイズ[cm]を追加
    if i == 0:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(np.arange(len(sizes)))
        ax2.set_xticklabels(sizes, fontsize=16)
        ax2.set_xlabel('Rock size [cm]', fontsize=20)
        ax.tick_params(labelbottom=False) # 1段目のX軸ラベルは非表示

    # 2段目：下横軸にラベル表示
    if i == 1:
        ax.set_xticklabels([f"{s / range_resolution_bipolar:.2f}" for s in sizes], fontsize=16)
        ax.set_xlabel('Rock size / Range resolution (bipolar)', fontsize=20)

    # 3段目・4段目：unipolarで規格化
    # 3段目：上横軸に岩石サイズ[cm]を追加
    if i == 2:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(np.arange(len(sizes)))
        ax2.set_xticklabels(sizes, fontsize=16)
        ax2.set_xlabel('Rock size [cm]', fontsize=20)
        ax.tick_params(labelbottom=False) # 1段目のX軸ラベルは非表示
    # 4段目：下横軸にラベル表示
    if i == 3:
        ax.set_xticklabels([f"{s / range_resolution_unipolar:.2f}" for s in sizes], fontsize=16)
        ax.set_xlabel('Rock size / Range resolution (unipolar)', fontsize=20)
    
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
    frameon=True,                 # 枠線なし
    fontsize=20
)


# tight_layoutは使用せず、GridSpecで制御
plt.savefig('/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_5x5/result_summary_small_rock_model.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
plt.savefig('/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_5x5/result_summary_small_rock_model.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()