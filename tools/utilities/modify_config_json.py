import json
import os
import numpy as np
import sys
from tqdm import tqdm
import h5py



output_file_paths = input("output_file_paths.jsonのパスを入力してください: ").strip()
if not os.path.isfile(output_file_paths):
    raise FileNotFoundError(f"指定されたファイルが存在しません: {output_file_paths}")
try:
        with open(output_file_paths, 'r') as f:
            path_group = json.load(f)
except Exception as e:
    print("Error reading JSON file:", e)
    sys.exit(1)


for key, data_path in tqdm(path_group.items(), desc="Processing files"):
        config_json_path = data_path.replace('.out', '_config.json')
        if not os.path.isfile(config_json_path):
            print(f"Config JSON file not found for {key}: {config_json_path}")
            continue
        try:
            with open(config_json_path, 'r') as f:
                config = json.load(f)
                if 'initial_pulse_delay' in config:
                    config['initial_pulse_delay'] = 2.57  # 初期パルス遅延を2.57に設定
                with open(config_json_path, 'w') as f:
                    json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Error reading config JSON file {config_json_path}: {e}")
            continue