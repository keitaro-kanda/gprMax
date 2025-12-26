import json
import shutil
from pathlib import Path


def main():
    # 対話的入力
    json_path = input("Enter JSON file path (output_file_paths.json): ").strip()
    output_dir = input("Enter output directory path: ").strip()

    # JSONファイルの読み込み
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            file_paths = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found: {json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {e}")
        return

    # 出力ディレクトリの作成（既存なら何もしない）
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ファイルのコピー
    copied_count = 0
    error_count = 0

    print(f"\nCopying files to: {output_dir}")
    print("-" * 60)

    for key, source_file in file_paths.items():
        source_path = Path(source_file)

        # ソースファイルの存在確認
        if not source_path.exists():
            print(f"Warning: Source file not found (skipping): {source_file}")
            error_count += 1
            continue

        # コピー先のファイルパス（ファイル名のみを使用）
        dest_file = output_path / source_path.name

        try:
            # ファイルをコピー（メタデータも保持）
            shutil.copy2(source_path, dest_file)
            print(f"Copied: {source_path.name}")
            copied_count += 1
        except Exception as e:
            print(f"Error copying {source_file}: {e}")
            error_count += 1

    # 結果報告
    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Total files in JSON: {len(file_paths)}")
    print(f"  Successfully copied: {copied_count}")
    print(f"  Errors/Skipped: {error_count}")
    print(f"\nAll files copied to: {output_dir}")


if __name__ == "__main__":
    main()
