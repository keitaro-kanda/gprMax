import numpy as np
import os
import decimal # 浮動小数点数の正確な表現のため
import json  # JSONライブラリ

# --- 設定 ---
heights_decimal = [decimal.Decimal(str(x)) for x in np.arange(0.01, 0.55, 0.01)]
widths_decimal = [decimal.Decimal(str(x)) for x in np.arange(0.3, 3.0 + 0.1, 0.3)]

# 出力ディレクトリ (絶対パス推奨)
output_base_dir = "/Volumes/SSD_Kanda_BUFFALO/gprMax/3D_domain_5x5x2/rock_test/circular_offset0.3"
# ローカルテスト用パス (必要に応じてコメントアウト解除)
# output_base_dir = "generated_inputs_hw_json_summary"

# 固定パラメータ
domain_x_val = 5.0
domain_y_val = 5.0
domain_z_gpr_val = 2.0
grid_size_val = 0.005
ground_y_max_val = 4.0
time_window_val = "50e-9"
ground_depth_val = domain_y_val - ground_y_max_val
# ----------------

# --- インプットファイルのテンプレート (変更なし) ---
template_part1 = f"""\
#domain: {domain_x_val} {domain_y_val} {domain_z_gpr_val}
#dx_dy_dz: {grid_size_val} {grid_size_val} {grid_size_val}
#time_window: {time_window_val}

#material: 3 0.0 1 0 ep3
#material: 9 0.0 1 0 ep9



#box: 0 0 0 {domain_x_val} {ground_y_max_val} {domain_z_gpr_val} ep3 n

#python:
import numpy as np
from gprMax.input_cmd_funcs import *

# --- Parameter Section Start ---
"""

template_part2 = f"""\
# --- Parameter Section End ---

# width is now directly defined above

top_y = 2.0 # [m]
bottom_y = top_y - height # [m]
center_y = top_y - height / 2 # [m]
radius = height / 2 # [m]

#* Main body of rock
# ---rectangle---
# box({domain_x_val / 2}-width/2, bottom_y, 0, {domain_x_val / 2}+width/2, top_y, {domain_z_gpr_val}, 'ep9', 'n')

# ---cylinder (2D)---
# cylinder({domain_x_val / 2}, center_y, 0, {domain_x_val / 2}, center_y, {domain_z_gpr_val}, radius, 'free_space', 'n')

# ---sphere (3D)---
sphere({domain_x_val / 2}, center_y, {domain_z_gpr_val / 2}, radius, 'ep9', 'n')

# ---ellipse---
def create_ellipse_with_boxes(x_center, y_center, z_center, a, b, height, material, box_size, c1='n'):
    boxes = []
    x_min = x_center - a
    x_max = x_center + a
    y_min = y_center - b
    y_max = y_center + b

    x_range = int(2 * a / box_size)
    y_range = int(2 * b / box_size)

    for i in range(x_range):
        for j in range(y_range):
            x = x_min + i * box_size + box_size / 2
            y = y_min + j * box_size + box_size / 2
            # 楕円の内部かどうかを判定
            if ((x - x_center) ** 2) / (a ** 2) + ((y - y_center) ** 2) / (b ** 2) <= 1:
                f1 = x - box_size / 2
                f2 = y - box_size / 2
                f4 = x + box_size / 2
                f5 = y + box_size / 2

                box(f1, f2, 0, f4, f5, 0.005, material, c1)

    return boxes

#ellipse = create_ellipse_with_boxes(2.5, center_y, 0, width/2, height/2, {domain_z_gpr_val}, 'ep9', {domain_z_gpr_val})

#end_python:


＜地形書き出し＞
#geometry_objects_write: 0 0 0 {domain_x_val} {domain_y_val} {domain_z_gpr_val} geometry


=====A-scan用=====
＜波源設定＞
#waveform: gaussiandot 1 500e6 my_src

#hertzian_dipole: z {domain_x_val / 2 + 0.15} {ground_y_max_val + 0.3} {domain_z_gpr_val / 2} my_src
#rx: {domain_x_val / 2 - 0.15} {ground_y_max_val + 0.3} {domain_z_gpr_val / 2}
"""

template_part3 = f"""\
==========


=====vti=====
＜地形vtiファイル作成＞
#geometry_view: 0 0 0 {domain_x_val} {domain_y_val} {domain_z_gpr_val} {grid_size_val} {grid_size_val} {grid_size_val} geometry n
==========
"""
# --- スクリプト本体 ---

# ベース出力ディレクトリを作成 (存在しない場合)
os.makedirs(output_base_dir, exist_ok=True)

# ★ 追加: .out ファイルパスを格納する辞書を初期化
output_paths_dict = {}

total_files = len(heights_decimal) * len(widths_decimal)
count = 0
print(f"Generating {total_files} pairs of (.in and .json) files and collecting output paths...")

# --- ループ開始 ---
for h_dec in heights_decimal:
    # --- height ごとの処理 ---
    h_float_str = format(h_dec, 'f')
    h_val_for_python = float(h_dec)
    h_str_foldername = format(h_dec, '.2f')
    height_folder_name = f"height_{h_str_foldername}"
    height_folder_path = os.path.join(output_base_dir, height_folder_name)
    os.makedirs(height_folder_path, exist_ok=True) # ディレクトリ作成

    # for w_dec in widths_decimal:
    #     # --- width ごとの処理 ---
    #     count += 1
    #     w_float_str = format(w_dec, 'f')
    #     w_val_for_python = float(w_dec)
    #     w_str_foldername = format(w_dec, '.1f')
    #     width_folder_name = f"width_{w_str_foldername}"

    #     # 保存先ディレクトリパス (例: .../height_0.3/width_0.3)
    #     target_dir = os.path.join(height_folder_path, width_folder_name)
    #     os.makedirs(target_dir, exist_ok=True) # ディレクトリ作成

    # ファイル名とディレクトリ名の生成
    h_str_filename = format(h_dec, '.2f')
    #    w_str_filename = format(w_dec, '.1f')
    # 入力ファイル名のベース (例: input_h0.3_w0.3)
    in_filename_base = f"h{h_str_filename}"
    # gprMax出力ディレクトリ名 (例: results_h0.3_w0.3)
    output_dir_name = f"results_h{h_str_filename}"
    # ★ 変更点: 出力ファイル名のベース (例: h0.3_w0.3 - ユーザー指定例に基づく)
    out_filename_base = f"h{h_str_filename}"

    # --- .in ファイル生成 ---
    in_filename = f"{in_filename_base}.in"
    in_filepath = os.path.join(height_folder_path, in_filename)
    python_vars_part = f"""
height = {h_val_for_python} # [m] (Generated value: {h_float_str})
width = {h_val_for_python}  # [m] (Generated value: {h_float_str})
"""
    #* ↑.inファイル内でインデントエラーが起きるので、インデントなしで記述すること
    output_dir_line = f"#output_dir: {output_dir_name}"
    file_content = (
        template_part1 + python_vars_part + template_part2 +
        "\n" + output_dir_line + "\n" + template_part3
    )
    try:
        with open(in_filepath, 'w', encoding='utf-8') as file:
            file.write(file_content)
    except IOError as e:
        print(f"Error writing .in file {in_filepath}: {e}")
        continue # エラー時はこのパラメータセットの処理をスキップ

    # --- JSON ファイル生成 (.in ごと) ---
    json_filename = f"{in_filename_base}.json" # JSONファイル名は入力ファイルに合わせる
    json_filepath = os.path.join(height_folder_path, json_filename)
    h5_file_path = os.path.abspath(os.path.join(height_folder_path, "geometry.h5")).replace(os.sep, '/')
    material_file_path = os.path.abspath(os.path.join(height_folder_path, "geometry_materials.txt")).replace(os.sep, '/')
    json_data = {
        "geometry_settings": {
            "h5_file": h5_file_path, "material_file": material_file_path,
            "comment": "unit: [m]", "domain_x": domain_x_val,
            "domain_z": domain_y_val, "ground_depth": ground_depth_val,
            "grid_size": grid_size_val
        }
    }
    try:
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
    except IOError as e:
        print(f"Error writing JSON file {json_filepath}: {e}")
        # .inファイルは生成されたので、処理は続行する場合が多い
    except TypeError as e:
            print(f"Error creating JSON data for {json_filepath}: {e}")

    # ★ 追加: .out ファイルパスの情報を収集
    # 識別キー (例: h3.0_w3.0) - 出力ファイル名のベースを使用
    sim_key = out_filename_base
    # .out ファイルのフルパスを計算 (ユーザー指定例に基づくファイル名)
    out_filename = f"{out_filename_base}.out"
    out_file_path = os.path.join(height_folder_path, output_dir_name, out_filename)
    # 絶対パスに変換し、パス区切り文字を '/' に統一 (オプション)
    out_file_path_abs = os.path.abspath(out_file_path).replace(os.sep, '/')
    # 辞書に追加
    output_paths_dict[sim_key] = out_file_path_abs

    # --- シミュレーション設定のJSONファイル生成 ---
    sim_config = {
        #! 使う波形によって変更が必要
        "initial_pulse_delay": 2.07, # [ns], 最大ピークの時刻, ricker: 2.57, LPR-like: 2.07
        "boundaries": [
            {"name": "vacuum-ground", "length": 0.30, "epsilon_r": 1.0},
            {"name": "ground-rock_top", "length": 2.0, "epsilon_r": 3.0},
            {"name": "rock_top-bottom", "length": h_val_for_python, "epsilon_r": 9.0}
        ]
    }
    # out_file_path_absは *.out のパスになっているので、その拡張子を _config.json に置換
    config_json_filepath = out_file_path_abs.replace('.out', '_config.json')
    # config_json_filepath のあるディレクトリは存在するはずですが念のため作成
    os.makedirs(os.path.dirname(config_json_filepath), exist_ok=True)
    try:
        with open(config_json_filepath, 'w', encoding='utf-8') as config_file:
            json.dump(sim_config, config_file, indent=4, ensure_ascii=False)
        print(f"Simulation config JSON file created: {config_json_filepath}")
    except IOError as e:
        print(f"Error writing simulation config JSON file {config_json_filepath}: {e}")



    # 進捗表示 (まとめて表示)
    relative_path_base = os.path.relpath(height_folder_path)
    print(f"({count}/{total_files}) Processed: {relative_path_base}/{in_filename_base} (Generated .in, .json; Collected .out path for '{sim_key}')")

# --- ループ終了 ---

# ★ 追加: 収集したパス情報をJSONファイルに書き込む
summary_json_filename = "output_file_paths.json"
summary_json_filepath = os.path.join(output_base_dir, summary_json_filename) # ベースディレクトリ直下に保存

print(f"\nWriting summary JSON file to: {summary_json_filepath}")
try:
    with open(summary_json_filepath, 'w', encoding='utf-8') as f:
        # indent=4 で整形、ensure_ascii=False で日本語等をそのまま出力
        # sort_keys=True でキーをソートして見やすくする
        json.dump(output_paths_dict, f, indent=4, ensure_ascii=False, sort_keys=True)
    print("Summary JSON file created successfully.")
except IOError as e:
    print(f"Error writing summary JSON file {summary_json_filepath}: {e}")
except TypeError as e:
    print(f"Error creating summary JSON data: {e}")


print("\nScript finished.")
print(f"Generated files are organized in subdirectories within '{output_base_dir}'.")
print(f"Structure: {output_base_dir}/height_X.X/width_Y.Y/{in_filename_base}[.in, .json]")
print(f"Summary of expected .out file paths (using key '{out_filename_base}') is saved in '{summary_json_filepath}'.")


# メモ：高さ・幅を両方変える仕様に戻す場合
# 'height_folder_path'をtarget_dir'に変更
# 'for w_dec in widths_decimal:'以降、'--- ループ終了 ---'までをインデントを戻して実行