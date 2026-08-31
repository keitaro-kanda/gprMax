import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys
import shutil
from datetime import datetime

# 現在のスクリプトの絶対パスを基準に、2つ上の階層（gprMaxルート）を取得
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))

# appendではなくinsert(0, ...)を使用し、モジュール検索の最優先に設定
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from tools.core.outputfiles_merge import get_output_data


COMPONENTS = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']


def load_all_components(filepath):
    data = {}
    dt = None
    for comp in COMPONENTS:
        d, dt = get_output_data(filepath, 1, comp)
        data[comp] = d
    return data, dt


def align_lengths(target_data, surface_data):
    n_target = len(target_data[COMPONENTS[0]])
    n_surface = len(surface_data[COMPONENTS[0]])
    if n_surface < n_target:
        print(f'ゼロパディング: 表面反射データを {n_surface} → {n_target} サンプルに拡張しました')
        for comp in COMPONENTS:
            surface_data[comp] = np.pad(surface_data[comp], (0, n_target - n_surface), 'constant')
    elif n_surface > n_target:
        print(f'トリミング: 表面反射データを {n_surface} → {n_target} サンプルに切り詰めました')
        for comp in COMPONENTS:
            surface_data[comp] = surface_data[comp][:n_target]
    return target_data, surface_data, n_target


def create_plots(time, target_data, surface_data, subtracted_data, output_dir, basename):
    plot_comps = ['Ex', 'Ey', 'Ez']
    row_labels = ['Original', 'Free Space Test', 'Subtracted']
    row_data = [target_data, surface_data, subtracted_data]
    row_colors = ['k', 'r', 'b']

    # Original と Surface Reflection 用の共通Y軸最大値を計算
    ymax_orig_surf = max(
        max(np.max(np.abs(target_data[comp]))*1.05 for comp in plot_comps),
        max(np.max(np.abs(surface_data[comp]))*1.05 for comp in plot_comps),
        1e-30
    )

    # Subtracted 用の共通Y軸最大値を計算（微小な振幅を拡大表示するため）
    ymax_sub = max(
        max(np.max(np.abs(subtracted_data[comp]))*1.05 for comp in plot_comps),
        1e-30
    )

    y_limits = [ymax_orig_surf, ymax_orig_surf, ymax_sub]

    # =========================================================
    # 1. 全成分をまとめた 3x3 プロットの作成
    # =========================================================
    fig, axes = plt.subplots(3, 3, figsize=(18, 12),
                             facecolor='w', edgecolor='w', tight_layout=True)

    for row, (label, data, color, ymax) in enumerate(zip(row_labels, row_data, row_colors, y_limits)):
        for col, comp in enumerate(plot_comps):
            ax = axes[row][col]
            ax.plot(time, data[comp], color=color, linewidth=0.8)
            ax.set_title(f'{label} - {comp}', fontsize=12)
            ax.set_xlabel('Time [ns]', fontsize=10)
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.grid(True)
            ax.set_xlim([0, time[-1]])
            ax.set_ylim([-ymax, ymax])

    all_plot_path = os.path.join(output_dir, basename + '_subtracted_all.png')
    fig.savefig(all_plot_path, dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)  # メモリ解放
    print(f'プロットを保存しました (All) : {all_plot_path}')

    # =========================================================
    # 2. 各成分 (Ex, Ey, Ez) 個別の 3x1 プロットの作成
    # =========================================================
    for comp in plot_comps:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8),
                                 facecolor='w', edgecolor='w', tight_layout=True)
        
        for row, (label, data, color, ymax) in enumerate(zip(row_labels, row_data, row_colors, y_limits)):
            ax = axes[row]
            ax.plot(time, data[comp], color=color, linewidth=0.8)
            ax.set_title(f'{label} - {comp}', fontsize=12)
            ax.set_xlabel('Time [ns]', fontsize=10)
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.grid(True)
            ax.set_xlim([0, time[-1]])
            ax.set_ylim([-ymax, ymax])
        
        single_plot_path = os.path.join(output_dir, f'{basename}_subtracted_{comp}.png')
        fig.savefig(single_plot_path, dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f'プロットを保存しました ({comp}):  {single_plot_path}')


def main():
    print('=' * 70)
    print('  k_subtract_Ascan.py: A-scan 差し引きツール')
    print('=' * 70)
    print()

    # パスを入力として受け取る
    target_file = input('差し引かれる対象のA-scanファイルパス (.out) を入力してください:\n> ').strip()
    surface_path = input('差し引く表面反射用のA-scanファイルパス (.out) を入力してください:\n> ').strip()
    print()

    # ファイルの存在チェック
    if not os.path.exists(target_file):
        print(f'エラー: 対象ファイルが見つかりません: {target_file}')
        sys.exit(1)
    if not os.path.exists(surface_path):
        print(f'エラー: 表面反射データファイルが見つかりません: {surface_path}')
        sys.exit(1)

    # 拡張子の簡易チェック
    if not target_file.endswith('.out'):
        print(f'警告: 対象ファイルの拡張子が .out ではありません。ファイルが間違っている可能性があります: {target_file}')
    if not surface_path.endswith('.out'):
        print(f'警告: 表面反射データの拡張子が .out ではありません。ファイルが間違っている可能性があります: {surface_path}')

    target_data, dt = load_all_components(target_file)
    surface_data, _ = load_all_components(surface_path)

    target_data, surface_data, n_target = align_lengths(target_data, surface_data)

    subtracted_data = {comp: target_data[comp] - surface_data[comp] for comp in COMPONENTS}

    output_dir = os.path.join(os.path.dirname(os.path.abspath(target_file)), 'subtracted')
    os.makedirs(output_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(target_file))[0]
    out_file_path = os.path.join(output_dir, basename + '_subtracted.out')

    shutil.copy2(target_file, out_file_path)
    with h5py.File(out_file_path, 'r+') as f:
        for comp in COMPONENTS:
            f[f'/rxs/rx1/{comp}'][:] = subtracted_data[comp]

    print(f'結果ファイルを保存しました: {out_file_path}')

    time = np.arange(n_target) * dt / 1e-9  # [ns]
    
    # 画像描画・保存処理を呼び出し（合計4枚出力されます）
    create_plots(time, target_data, surface_data, subtracted_data, output_dir, basename)

    log_path = os.path.join(output_dir, basename + '_subtracted_info.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f'実行日時: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'差し引かれる対象データ: {os.path.abspath(target_file)}\n')
        f.write(f'差し引くデータ（表面反射）: {os.path.abspath(surface_path)}\n')
    print(f'ログファイルを保存しました: {log_path}')
    print()
    print('完了')


if __name__ == '__main__':
    main()