import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys
import json
import shutil
from datetime import datetime
from tools.core.outputfiles_merge import get_output_data


CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'k_subtract_Ascan_config.json')
COMPONENTS = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']


def load_config():
    if not os.path.exists(CONFIG_PATH):
        print(f'エラー: 設定ファイルが見つかりません: {CONFIG_PATH}')
        print('k_subtract_Ascan_config.json を作成し、surface_reflection_path を設定してください。')
        sys.exit(1)
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


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


def plot(time, target_data, surface_data, subtracted_data, output_path):
    plot_comps = ['Ex', 'Ey', 'Ez']
    row_labels = ['Original', 'Surface Reflection', 'Subtracted']
    row_data = [target_data, surface_data, subtracted_data]
    row_colors = ['k', 'r', 'b']

    ymax = max(
        max(np.max(np.abs(d[comp])) for d in row_data for comp in plot_comps),
        1e-30
    )

    fig, axes = plt.subplots(3, 3, figsize=(18, 12),
                             facecolor='w', edgecolor='w', tight_layout=True)

    for row, (label, data, color) in enumerate(zip(row_labels, row_data, row_colors)):
        for col, comp in enumerate(plot_comps):
            ax = axes[row][col]
            ax.plot(time, data[comp], color=color, linewidth=0.8)
            ax.set_title(f'{label} - {comp}', fontsize=12)
            ax.set_xlabel('Time [ns]', fontsize=10)
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.grid(True)
            ax.set_xlim([0, time[-1]])
            ax.set_ylim([-ymax, ymax])

    fig.savefig(output_path, dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close()


def main():
    config = load_config()
    surface_path = config['surface_reflection_path']

    print('=' * 70)
    print('  k_subtract_Ascan.py: A-scan 差し引きツール')
    print('=' * 70)
    print('  [差し引くデータ（表面反射のみ）]')
    print(f'    {surface_path}')
    print('=' * 70)
    print()

    target_file = input('差し引かれる対象のA-scanファイルパス (.out) を入力してください:\n> ').strip()

    if not os.path.exists(target_file):
        print(f'エラー: 対象ファイルが見つかりません: {target_file}')
        sys.exit(1)
    if not os.path.exists(surface_path):
        print(f'エラー: 表面反射データファイルが見つかりません: {surface_path}')
        sys.exit(1)

    target_data, dt = load_all_components(target_file)
    surface_data, _ = load_all_components(surface_path)

    target_data, surface_data, n_target = align_lengths(target_data, surface_data)

    subtracted_data = {comp: target_data[comp] - surface_data[comp] for comp in COMPONENTS}

    output_dir = os.path.join(os.path.dirname(os.path.abspath(target_file)), 'subtracted')
    os.makedirs(output_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(target_file))[0]
    out_file_path = os.path.join(output_dir, basename + '_subtracted.out')
    plot_path = os.path.join(output_dir, basename + '_subtracted.png')

    shutil.copy2(target_file, out_file_path)
    with h5py.File(out_file_path, 'r+') as f:
        for comp in COMPONENTS:
            f[f'/rxs/rx1/{comp}'][:] = subtracted_data[comp]

    print(f'結果ファイルを保存しました: {out_file_path}')

    time = np.arange(n_target) * dt / 1e-9  # [ns]
    plot(time, target_data, surface_data, subtracted_data, plot_path)
    print(f'プロットを保存しました:     {plot_path}')

    log_path = os.path.join(output_dir, basename + '_subtracted_info.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f'実行日時: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'差し引かれる対象データ: {os.path.abspath(target_file)}\n')
        f.write(f'差し引くデータ（表面反射）: {surface_path}\n')
    print(f'ログファイルを保存しました: {log_path}')
    print()
    print('完了')


if __name__ == '__main__':
    main()
