import numpy as np
import os
import decimal # 浮動小数点数の正確な表現のため
import json  # JSONライブラリ

# --- 設定 ---
heights_decimal = [decimal.Decimal(str(x)) for x in np.arange(0.3, 3.0 + 0.1, 0.3)]
widths_decimal = [decimal.Decimal(str(x)) for x in np.arange(0.3, 3.0 + 0.1, 0.3)]

# 出力ディレクトリ (絶対パス推奨)
output_base_dir = "/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_5x5/polarity_obs4journal/ricker/rectangle"
# ローカルテスト用パス (必要に応じてコメントアウト解除)
# output_base_dir = "generated_inputs_hw_json_summary"

# 固定パラメータ
domain_x_val = 5.0
domain_y_val = 7.0
domain_z_gpr_val = 0.005
grid_size_val = 0.005
ground_y_max_val = 6.0
time_window_val = "100e-9"
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

top_y = 4.0 # [m]
bottom_y = top_y - height # [m]

#* Main body of rock
box(2.5-width/2, bottom_y, 0, 2.5+width/2, top_y, {domain_z_gpr_val}, 'ep9', 'n')

#end_python:


＜地形書き出し＞
#geometry_objects_write: 0 0 0 {domain_x_val} {domain_y_val} {domain_z_gpr_val} geometry


=====A-scan用=====
＜波源設定＞
#waveform: ricker 1 500e6 my_src

#hertzian_dipole: z 2.5 6.3 0 my_src
#rx: 2.5 6.3 0
"""

template_part3 = f"""\
==========


=====vti=====
＜地形vtiファイル作成＞
#geometry_view: 0 0 0 {domain_x_val} {domain_y_val} {domain_z_gpr_val} {grid_size_val} {grid_size_val} {grid_size_val} geometry n

<snapshot作成>
#python:
#* Snapshot
n = 200
time_step = {time_window_val} / n
domain_x_snap = {domain_x_val}
domain_y_snap = {domain_y_val}
domain_z_snap = {domain_z_gpr_val} # スナップショットのz範囲
grid_x_snap = {grid_size_val}
grid_y_snap = {grid_size_val}
grid_z_snap = {grid_size_val} # スナップショットのz解像度
for i in range(1, n):
    print(f'#snapshot: 0 0 0 {{domain_x_snap}} {{domain_y_snap}} {{domain_z_snap}} {{grid_x_snap}} {{grid_y_snap}} {{grid_z_snap}} {{i*time_step}} snapshot{{i}}')
#end_python:
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
    h_str_foldername = format(h_dec, '.1f')
    height_folder_name = f"height_{h_str_foldername}"
    height_folder_path = os.path.join(output_base_dir, height_folder_name)

    for w_dec in widths_decimal:
        # --- width ごとの処理 ---
        count += 1
        w_float_str = format(w_dec, 'f')
        w_val_for_python = float(w_dec)
        w_str_foldername = format(w_dec, '.1f')
        width_folder_name = f"width_{w_str_foldername}"

        # 保存先ディレクトリパス (例: .../height_0.3/width_0.3)
        target_dir = os.path.join(height_folder_path, width_folder_name)
        os.makedirs(target_dir, exist_ok=True) # ディレクトリ作成

        # ファイル名とディレクトリ名の生成
        h_str_filename = format(h_dec, '.1f')
        w_str_filename = format(w_dec, '.1f')
        # 入力ファイル名のベース (例: input_h0.3_w0.3)
        in_filename_base = f"input_h{h_str_filename}_w{w_str_filename}"
        # gprMax出力ディレクトリ名 (例: results_h0.3_w0.3)
        output_dir_name = f"results_h{h_str_filename}_w{w_str_filename}"
        # ★ 変更点: 出力ファイル名のベース (例: h0.3_w0.3 - ユーザー指定例に基づく)
        out_filename_base = f"h{h_str_filename}_w{w_str_filename}"

        # --- .in ファイル生成 ---
        in_filename = f"{in_filename_base}.in"
        in_filepath = os.path.join(target_dir, in_filename)
        python_vars_part = f"""
height = {h_val_for_python} # [m] (Generated value: {h_float_str})
width = {w_val_for_python}  # [m] (Generated value: {w_float_str})
"""
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
        json_filepath = os.path.join(target_dir, json_filename)
        h5_file_path = os.path.abspath(os.path.join(target_dir, "geometry.h5")).replace(os.sep, '/')
        material_file_path = os.path.abspath(os.path.join(target_dir, "geometry_materials.txt")).replace(os.sep, '/')
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
        out_file_path = os.path.join(target_dir, output_dir_name, out_filename)
        # 絶対パスに変換し、パス区切り文字を '/' に統一 (オプション)
        out_file_path_abs = os.path.abspath(out_file_path).replace(os.sep, '/')
        # 辞書に追加
        output_paths_dict[sim_key] = out_file_path_abs

        # 進捗表示 (まとめて表示)
        relative_path_base = os.path.relpath(target_dir)
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