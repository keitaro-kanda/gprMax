# Core Tools

このディレクトリには、gprMaxの基本的な機能を提供するコアツールが含まれています。

## 概要

コアツールは、gprMaxの基本的なワークフローに必要な機能を提供します：
- ファイル形式の変換
- 入力ファイルの処理
- 出力データの統合
- 基本的な可視化

## 含まれるツール

### ファイル変換・処理
- **`convert_png2h5.py`** - PNG画像をHDF5形式に変換してgprMaxで利用可能にする
- **`inputfile_old2new.py`** - 古いバージョンのgprMax入力ファイルを新しい形式に変換
- **`outputfiles_merge.py`** - 複数の出力ファイルを統合し、データ抽出ユーティリティを提供

### 基本可視化
- **`plot_source_wave.py`** - 励起源波形の特性をプロットして確認

## 使用方法

### ファイル変換
```bash
# PNG画像をHDF5に変換
python -m tools.core.convert_png2h5 input_image.png output_geometry.h5

# 古い入力ファイルを新形式に変換
python -m tools.core.inputfile_old2new old_input.in new_input.in

# 複数の出力ファイルを統合
python -m tools.core.outputfiles_merge file1.out file2.out merged_output.out
```

### 基本可視化
```bash
# 励起源波形をプロット
python -m tools.core.plot_source_wave input_file.in
```

## 関連ツール

- **visualization/**: より高度な可視化ツール
- **utilities/**: その他のユーティリティツール
- **data_processing/**: データ処理ツール

## 注意事項

- これらのツールは、gprMaxの基本機能に依存しているため、gprMaxが正しくインストールされている必要があります
- 大きなファイルの処理時は十分なメモリを確保してください
- 出力ファイルの統合時は、同じパラメータで生成されたファイルを使用してください