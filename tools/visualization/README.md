# Visualization Tools

このディレクトリには、GPRデータの可視化ツールが含まれています。基本的なプロットから高度な可視化、解析用の特殊プロットまで幅広く対応しています。

## 概要

可視化ツールは以下の3つのサブディレクトリに分類されています：
- **basic/**: A-scan、B-scanの基本的なプロット
- **advanced/**: 幾何構造、スナップショット、速度構造の高度な可視化
- **analysis/**: 解析結果の可視化（TWT推定など）

## サブディレクトリ構成

### basic/ - 基本プロット
- **`plot_Ascan.py`** - A-scan（単一トレース）データのプロット
- **`plot_Bscan.py`** - B-scan（複数トレース）データのプロット
- **`k_plot_Ascan_from_Bscan.py`** - B-scanデータからA-scanを抽出してプロット

### advanced/ - 高度な可視化
- **`k_plot_geometry.py`** - JSON設定から幾何構造をプロット
- **`k_plot_geometry_imaging.py`** - イメージング用の幾何構造可視化
- **`k_plot_snapshot.py`** - スナップショットデータの可視化
- **`k_plot_velocity_structure.py`** - 速度構造の可視化

### analysis/ - 解析用可視化
- **`k_plot_TWT_estimation.py`** - 往復走時（TWT）推定結果のプロット

## 使用方法

### 基本プロット
```bash
# A-scanをプロット
python -m tools.visualization.basic.plot_Ascan output_file.out

# B-scanをプロット
python -m tools.visualization.basic.plot_Bscan output_file.out

# B-scanから特定のA-scanを抽出してプロット
python -m tools.visualization.basic.k_plot_Ascan_from_Bscan config.json
```

### 高度な可視化
```bash
# 幾何構造をプロット
python -m tools.visualization.advanced.k_plot_geometry config.json

# スナップショットをプロット
python -m tools.visualization.advanced.k_plot_snapshot config.json

# 速度構造をプロット
python -m tools.visualization.advanced.k_plot_velocity_structure config.json
```

### 解析用可視化
```bash
# TWT推定結果をプロット
python -m tools.visualization.analysis.k_plot_TWT_estimation config.json
```

## 設定ファイル

多くのk_プレフィックスツールは、JSON設定ファイルを使用してパラメータを管理します。

典型的な設定ファイルの例：
```json
{
    "input_file": "simulation_result.out",
    "output_dir": "plots/",
    "plot_params": {
        "title": "GPR Survey Results",
        "xlabel": "Distance (m)",
        "ylabel": "Time (ns)"
    }
}
```

## 関連ツール

- **core/**: 基本的なファイル処理ツール
- **signal_processing/**: 信号処理後のデータ可視化
- **velocity_analysis/**: 速度解析結果の可視化
- **migration_imaging/**: マイグレーション結果の可視化

## 注意事項

- 大きなデータセットの可視化時は、メモリ使用量に注意してください
- 高解像度の画像を生成する場合は、十分なディスク容量を確保してください
- JSON設定ファイルのパスは絶対パスまたは実行ディレクトリからの相対パスで指定してください