import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator
from matplotlib.gridspec import GridSpec


# データ定義
results_Ricker_circle = ['x', 's', 's', 's', 's', 's', 's', 'd', 'd', '2', '2', '2', '2', '2', '2']
results_Ricker_square = ['x', 'x', 's', 's', 's', 's', 'd', 'd', '2', '2', '2', '2', '2', '2', '2']
results_LPR_circle = ['x', 's', 's', 's', 's', 's', 'd', 'd', '2', '2', '2', '2', '2', '2', '2']
results_LPR_square = ['x', 's', 's', 's', 's', 'd', '2', '2', '2', '2', '2', '2', '2', '2', '2']
results = [results_Ricker_circle, results_Ricker_square, results_LPR_circle, results_LPR_square]

sizes = np.arange(1, 16, 1)
model_names = ['Bipolar-circle', 'Bipolar-square', 'Unipolar-circle', 'Unipolar-square']

cat_to_int = {'x':0, 's':1, 'd':2, '2':3}
colors = ['black','red','blue','green']
data = np.array([[cat_to_int[c] for c in row] for row in results])

# 保存先ディレクトリ
save_dir = '/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_5x5/small_rock4journal'


def create_plot(data_indices, model_names_subset, plot_type, figsize=(12, 10)):
    """
    プロット作成関数

    Parameters:
    -----------
    data_indices : list
        dataから使用する行のインデックスリスト（例: [0, 1] でbipolarのみ）
    model_names_subset : list
        表示するモデル名のリスト
    plot_type : str
        'all', 'bipolar', 'unipolar' のいずれか
    figsize : tuple
        図のサイズ
    """
    range_resolution_unipolar = 7.3  # cm
    range_resolution_bipolar = 7.8  # cm

    n_models = len(data_indices)

    # GridSpecの設定
    if plot_type == 'all':
        # bipolar 2行 + 隙間 1行 + unipolar 2行
        gs = GridSpec(5, 1, figure=plt.figure(figsize=figsize), height_ratios=[1, 1, 0.8, 1, 1])
        ax_positions = [0, 1, 3, 4]
    elif plot_type == 'bipolar':
        # bipolar 2行のみ
        gs = GridSpec(2, 1, figure=plt.figure(figsize=(12, 5)), height_ratios=[1, 1])
        ax_positions = [0, 1]
    elif plot_type == 'unipolar':
        # unipolar 2行のみ
        gs = GridSpec(2, 1, figure=plt.figure(figsize=(12, 5)), height_ratios=[1, 1])
        ax_positions = [0, 1]

    fig = plt.gcf()
    axes = []

    # 各subplotを配置
    for idx, pos in enumerate(ax_positions):
        axes.append(fig.add_subplot(gs[pos, 0]))

    # X軸の共有設定
    if n_models >= 2:
        axes[0].sharex(axes[1])
    if n_models == 4:
        axes[2].sharex(axes[3])

    # 各subplotに1x15のデータを表示
    for idx, (data_idx, ax) in enumerate(zip(data_indices, axes)):
        # 1x15のデータを表示
        im = ax.imshow(data[data_idx:data_idx+1], aspect='auto',
                      cmap=plt.matplotlib.colors.ListedColormap(colors),
                      vmin=0, vmax=3, origin='upper')

        # メジャー・ティックとラベルはセル中央に
        ax.set_xticks(np.arange(len(sizes)))

        # プロットタイプに応じたX軸ラベル設定
        if plot_type == 'all':
            # 既存の全体プロットのロジック
            if idx == 0:  # Bipolar-circle
                ax2 = ax.twiny()
                ax2.set_xlim(ax.get_xlim())
                ax2.set_xticks(np.arange(len(sizes)))
                ax2.set_xticklabels(sizes, fontsize=16)
                ax2.set_xlabel('Rock size [cm]', fontsize=20)
                ax.tick_params(labelbottom=False)
            elif idx == 1:  # Bipolar-square
                ax.set_xticklabels([f"{s / range_resolution_bipolar:.2f}" for s in sizes], fontsize=16)
                ax.set_xlabel('Rock size / Range resolution (bipolar)', fontsize=20)
            elif idx == 2:  # Unipolar-circle
                ax2 = ax.twiny()
                ax2.set_xlim(ax.get_xlim())
                ax2.set_xticks(np.arange(len(sizes)))
                ax2.set_xticklabels(sizes, fontsize=16)
                ax2.set_xlabel('Rock size [cm]', fontsize=20)
                ax.tick_params(labelbottom=False)
            elif idx == 3:  # Unipolar-square
                ax.set_xticklabels([f"{s / range_resolution_unipolar:.2f}" for s in sizes], fontsize=16)
                ax.set_xlabel('Rock size / Range resolution (unipolar)', fontsize=20)

        elif plot_type == 'bipolar':
            # Bipolarのみのプロット
            if idx == 0:  # Bipolar-circle
                ax2 = ax.twiny()
                ax2.set_xlim(ax.get_xlim())
                ax2.set_xticks(np.arange(len(sizes)))
                ax2.set_xticklabels(sizes, fontsize=16)
                ax2.set_xlabel('Rock size [cm]', fontsize=20)
                ax.tick_params(labelbottom=False)
            elif idx == 1:  # Bipolar-square
                ax.set_xticklabels([f"{s / range_resolution_bipolar:.2f}" for s in sizes], fontsize=16)
                ax.set_xlabel('Rock size / Range resolution (bipolar)', fontsize=20)

        elif plot_type == 'unipolar':
            # Unipolarのみのプロット
            if idx == 0:  # Unipolar-circle
                ax2 = ax.twiny()
                ax2.set_xlim(ax.get_xlim())
                ax2.set_xticks(np.arange(len(sizes)))
                ax2.set_xticklabels(sizes, fontsize=16)
                ax2.set_xlabel('Rock size [cm]', fontsize=20)
                ax.tick_params(labelbottom=False)
            elif idx == 1:  # Unipolar-square
                ax.set_xticklabels([f"{s / range_resolution_unipolar:.2f}" for s in sizes], fontsize=16)
                ax.set_xlabel('Rock size / Range resolution (unipolar)', fontsize=20)

        # Y軸は各subplotでモデル名をタイトルとして表示
        ax.set_yticks([0])
        ax.set_yticklabels([model_names_subset[idx]], fontsize=20)
        ax.set_ylabel('')

        # マイナー・ティックをビンの端に設定
        N = len(sizes)
        x_edge = np.arange(N+1) - 0.5
        ax.xaxis.set_minor_locator(FixedLocator(x_edge))
        y_edge = np.array([-0.5, 0.5])
        ax.yaxis.set_minor_locator(FixedLocator(y_edge))

        # グリッド設定
        ax.grid(which='major', visible=False)
        ax.grid(which='minor', axis='both', color='white', linewidth=1, linestyle='-.')

    # 凡例を全体の下部に配置
    labels = ['Not detected', 'One echo', 'One echo with dipression', 'Two echoes']
    legend_patches = [Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
    fig.legend(
        handles=legend_patches,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=True,
        fontsize=20
    )

    return fig


# 1. 全体プロット（bipolar + unipolar）
fig_all = create_plot(
    data_indices=[0, 1, 2, 3],
    model_names_subset=model_names,
    plot_type='all',
    figsize=(12, 10)
)
fig_all.savefig(os.path.join(save_dir, 'result_summary_small_rock_model.png'),
                dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
fig_all.savefig(os.path.join(save_dir, 'result_summary_small_rock_model.pdf'),
                dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.close(fig_all)

# 2. Bipolarのみのプロット
fig_bipolar = create_plot(
    data_indices=[0, 1],
    model_names_subset=['Bipolar-circle', 'Bipolar-square'],
    plot_type='bipolar',
    figsize=(12, 5)
)
fig_bipolar.savefig(os.path.join(save_dir, 'result_summary_small_rock_model_bipolar.png'),
                    dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
fig_bipolar.savefig(os.path.join(save_dir, 'result_summary_small_rock_model_bipolar.pdf'),
                    dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.close(fig_bipolar)

# 3. Unipolarのみのプロット
fig_unipolar = create_plot(
    data_indices=[2, 3],
    model_names_subset=['Unipolar-circle', 'Unipolar-square'],
    plot_type='unipolar',
    figsize=(12, 5)
)
fig_unipolar.savefig(os.path.join(save_dir, 'result_summary_small_rock_model_unipolar.png'),
                     dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
fig_unipolar.savefig(os.path.join(save_dir, 'result_summary_small_rock_model_unipolar.pdf'),
                     dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.close(fig_unipolar)

print(f"Plots saved to {save_dir}")
print("- result_summary_small_rock_model.png/pdf (all models)")
print("- result_summary_small_rock_model_bipolar.png/pdf (bipolar only)")
print("- result_summary_small_rock_model_unipolar.png/pdf (unipolar only)")