#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from outputfiles_merge import get_output_data
from scipy.signal import hilbert
import shutil

# 外部モジュール（既存の解析処理）
import k_detect_peak     # ピーク検出処理
import k_plot_TWT_estimation  # TWT推定処理
import k_subtract        # ※将来的に送信波形引き算・FWHM機能追加のためのモジュール（未実装）


def plot_Ascan(filename, data, time, use_zoom=False, x_min=None, x_max=None, y_min=None, y_max=None):
    """
    A-scanプロットを作成し、画像ファイルとして保存する。
    
    filename : データファイルのフルパス（出力ファイル名の元に利用）
    data     : A-scanデータ（1チャネル分）
    time     : 対応する時系列（ns単位）
    use_zoom : 拡大表示の有無（Trueならx,y軸の上限・下限を適用）
    x_min,x_max,y_min,y_max: 拡大表示用の軸パラメータ（use_zoom=Trueの場合に必須）
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(time, data, 'k', lw=2)
    if use_zoom:
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
    else:
        ax.set_xlim([0, np.amax(time)])
    ax.grid(which='both', axis='both', linestyle='-.')
    ax.minorticks_on()
    ax.set_xlabel('Time [ns]', fontsize=28)
    ax.set_ylabel('Ez normalized field strength [V/m]', fontsize=28)
    ax.tick_params(labelsize=24)
    plt.tight_layout()

    # 出力ファイル名：元のファイル名に_rx{番号}と拡大指定があれば付与
    base_filename = os.path.splitext(os.path.abspath(filename))[0]
    output_filename = base_filename + '_rx.png'
    if use_zoom:
        output_filename = base_filename + f'_closeup_x{x_min}_{x_max}_y{y_max}' + '_rx.png'
    fig.savefig(output_filename, dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved A-scan plot: {output_filename}")


def main():
    # 1. data pathをまとめたjsonファイルのパスの入力
    data_json_path = input("Enter the path to the JSON file containing data paths: ").strip()
    if not os.path.exists(data_json_path):
        print("Error: Specified JSON file does not exist.")
        sys.exit(1)
    try:
        with open(data_json_path, 'r') as f:
            path_group = json.load(f)
    except Exception as e:
        print("Error reading JSON file:", e)
        sys.exit(1)

    # 2. 使用する機能の選択
    print("\nSelect function mode:")
    print("1: A-scan plot (default)")
    print("2: Peak detection")
    print("3: TWT estimation")
    mode_input = input("Enter your choice [default 1]: ").strip()
    if mode_input == "" or mode_input == "1":
        mode = "A-scan"
    elif mode_input == "2":
        mode = "peak"
    elif mode_input == "3":
        mode = "TWT"
    else:
        print("Invalid selection. Defaulting to A-scan plot.")
        mode = "A-scan"
    print(f"Selected mode: {mode}")

    # TWTモードの場合、各データファイルの出力ディレクトリ内にmodel.jsonが存在するかを確認
    if mode == "TWT":
        for key, data_path in path_group.items():
            folder = os.path.dirname(data_path)
            model_path = data_path.replace('.out', '_config.json')
            if not os.path.exists(model_path):
                print(f"Error: simulation configuration file ({model_path}) not found in directory: {folder}.")
                print("TWT mode requires model.json to be present. Aborting.")
                sys.exit(1)

    # 3. 拡大表示の指定
    zoom_input = input("\nDo you want to use zoom view? (y/n) [default: n]: ").strip().lower()
    if zoom_input == "y":
        try:
            x_min = float(input("Enter x-axis lower limit: ").strip())
            x_max = float(input("Enter x-axis upper limit: ").strip())
            y_min = float(input("Enter y-axis lower limit: ").strip())
            y_max = float(input("Enter y-axis upper limit: ").strip())
        except Exception as e:
            print("Invalid numeric input for zoom parameters.", e)
            sys.exit(1)
        use_zoom = True
    else:
        use_zoom = False
        x_min = x_max = y_min = y_max = None

    # 4. 各データファイルごとにA-scanプロットと解析を実施
    print("\nProcessing data files...\n")
    # path_group はキー:データパス の辞書なので各ファイルに対してループする
    for key, data_path in tqdm(path_group.items(), desc="Processing files"):
        print(f"\nProcessing '{key}': {data_path}")
        try:
            f = h5py.File(data_path, 'r')
        except Exception as e:
            print(f"Error opening file {data_path}: {e}")
            continue

        # ファイル属性 'nrx' を取得
        try:
            nrx = f.attrs['nrx']
        except Exception as e:
            print(f"Error: 'nrx' attribute not found in file: {data_path}")
            f.close()
            continue

        # 各受信チャネル(rx)ごとに処理を実施
        for rx in range(nrx):
            try:
                data, dt = get_output_data(data_path, rx + 1, 'Ez')
                data_norm = data / np.max(np.abs(data))  # 正規化
            except Exception as e:
                print(f"Error reading data for rx {rx+1} in {data_path}: {e}")
                continue
            time = np.arange(len(data_norm)) * dt / 1e-9  # ns単位の時系列

            # --- A-scanプロットの作成 ---
            plot_Ascan(data_path, data_norm, time, use_zoom, x_min, x_max, y_min, y_max)

            # --- 追加機能の実行 ---
            if mode == "peak":
                # ピーク検出の実行（出力ディレクトリ："peak_detection"）
                output_dir_peak = os.path.join(os.path.dirname(data_path), "peak_detection")
                if not os.path.exists(output_dir_peak):
                    os.makedirs(output_dir_peak)
                # 関数引数は: data_norm, dt, use_zoom, x_min, x_max, y_min, y_max, (FWHM=False), output_dir, plt_show=False
                pulse_info = k_detect_peak.detect_plot_peaks(data_norm, dt, use_zoom, x_min, x_max, y_min, y_max, False, output_dir_peak, plt_show=False)
                peak_info_filename = os.path.join(output_dir_peak, "peak_info.txt")
                try:
                    with open(peak_info_filename, "w") as fout:
                        json.dump(pulse_info, fout, indent=2)
                    print(f"Saved peak detection info: {peak_info_filename}")
                except Exception as e:
                    print(f"Error saving peak info for rx {rx+1}: {e}")

            elif mode == "TWT":
                # TWT推定の実行（出力ディレクトリ："TWT_estimation"）
                output_dir_twt = os.path.join(os.path.dirname(data_path), "TWT_estimation")
                if not os.path.exists(output_dir_twt):
                    os.makedirs(output_dir_twt)
                model_path = data_path.replace('.out', '_config.json')
                if not os.path.exists(model_path):
                    print(f"Error: model.json not found in {os.path.dirname(data_path)}. Aborting TWT estimation.")
                    sys.exit(1)
                k_plot_TWT_estimation.calc_plot_TWT(data_norm, time, model_path, use_zoom, x_min, x_max, y_min, y_max, output_dir_twt, plt_show=False)

            # 以下、将来的に‐subtractionや‐FWHMの機能を追加する場合に備えたプレースホルダです。
            # elif mode == "subtraction":
            #     # ※送信波形引き算機能は未実装です。
            #     print("Subtraction mode is not implemented yet.")
            # elif mode == "FWHM":
            #     # ※FWHM機能は未実装です。
            #     print("FWHM mode is not implemented yet.")

        f.close()

    print("\nAll processing done!")

if __name__ == "__main__":
    main()
