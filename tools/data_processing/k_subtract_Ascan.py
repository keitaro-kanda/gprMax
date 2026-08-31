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


def create_plots(time, target_data, surface_data, subtracted_data, output_dir, basename, is_db=False):
    plot_comps = ['Ex', 'Ey', 'Ez']
    row_labels = ['Original', 'Free Space Test', 'Subtracted'] #[cite: 2]
    row_colors = ['k', 'r', 'b']

    if is_db:
        # dB変換関数 (0や微小値でのlogエラーを防ぐため1e-30でクリップ)
        def convert_to_db(d):
            return {c: 20 * np.log10(np.clip(np.abs(d[c]), 1e-30, None)) for c in plot_comps}
        
        row_data = [convert_to_db(target_data), convert_to_db(surface_data), convert_to_db(subtracted_data)]
        
        # dB用のY軸範囲計算 (ピークから120dBのダイナミックレンジを持たせる)
        db_max_orig_surf = max(
            max(np.max(row_data[0][comp]) for comp in plot_comps),
            max(np.max(row_data[1][comp]) for comp in plot_comps)
        )
        ylim_orig_surf = [db_max_orig_surf - 120, db_max_orig_surf + 5]
        
        # Subtractedパネル用の独立したY軸範囲
        db_max_sub = max(np.max(row_data[2][comp]) for comp in plot_comps)
        ylim_sub = [db_max_sub - 120, db_max_sub + 5]
        
        y_limits = [ylim_orig_surf, ylim_orig_surf, ylim_sub]
        suffix = '_db'
        ylabel = 'Amplitude [dB]'
    else:
        row_data = [target_data, surface_data, subtracted_data]
        
        # Original と Free Space Test 用の共通Y軸最大値を計算 (1.05倍のマージン)
        ymax_orig_surf = max(
            max(np.max(np.abs(target_data[comp])) * 1.05 for comp in plot_comps), #[cite: 2]
            max(np.max(np.abs(surface_data[comp])) * 1.05 for comp in plot_comps), #[cite: 2]
            1e-30
        )
        # Subtracted 用の共通Y軸最大値を計算
        ymax_sub = max(
            max(np.max(np.abs(subtracted_data[comp])) * 1.05 for comp in plot_comps), #[cite: 2]
            1e-30
        )
        
        y_limits = [[-ymax_orig_surf, ymax_orig_surf], 
                    [-ymax_orig_surf, ymax_orig_surf], 
                    [-ymax_sub, ymax_sub]]
        suffix = ''
        ylabel = 'Amplitude'

    # =========================================================
    # 1. 全成分をまとめた 3x3 プロットの作成
    # =========================================================
    fig, axes = plt.subplots(3, 3, figsize=(18, 12),
                             facecolor='w', edgecolor='w', tight_layout=True)

    for row, (label, data, color, ylim) in enumerate(zip(row_labels, row_data, row_colors, y_limits)):
        for col, comp in enumerate(plot_comps):
            ax = axes[row][col]
            ax.plot(time, data[comp], color=color, linewidth=0.8)
            ax.set_title(f'{label} - {comp}', fontsize=12)
            ax.set_xlabel('Time [ns]', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True)
            ax.set_xlim([0, time[-1]])
            ax.set_ylim(ylim)

    all_plot_path = os.path.join(output_dir, f'{basename}_subtracted_all{suffix}.png')
    fig.savefig(all_plot_path, dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'プロットを保存しました (All{suffix}) : {all_plot_path}')

    # =========================================================
    # 2. 各成分 (Ex, Ey, Ez) 個別の 3x1 プロットの作成
    # =========================================================
    for comp in plot_comps:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), #[cite: 2]
                                 facecolor='w', edgecolor='w', tight_layout=True)
        
        for row, (label, data, color, ylim) in enumerate(zip(row_labels, row_data, row_colors, y_limits)):
            ax = axes[row]
            ax.plot(time, data[comp], color=color, linewidth=0.8)
            ax.set_title(f'{label} - {comp}', fontsize=12)
            ax.set_xlabel('Time [ns]', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True)
            ax.set_xlim([0, time[-1]])
            ax.set_ylim(ylim)
        
        single_plot_path = os.path.join(output_dir, f'{basename}_subtracted_{comp}{suffix}.png')
        fig.savefig(single_plot_path, dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f'プロットを保存しました ({comp}{suffix}):  {single_plot_path}')


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
    
    # 振幅プロットとdBプロットの両方を出力
    create_plots(time, target_data, surface_data, subtracted_data, output_dir, basename, is_db=False)
    create_plots(time, target_data, surface_data, subtracted_data, output_dir, basename, is_db=True)

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