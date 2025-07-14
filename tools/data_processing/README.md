# Data Processing Tools

このディレクトリには、GPRデータの基本的な処理・操作ツールが含まれています。データの前処理、抽出、変換、品質向上などの基本的な処理を提供します。

## 概要

データ処理ツールは、GPRデータ解析の前処理段階で使用される基本的な操作を提供します。これらのツールは、より高度な解析（速度解析、マイグレーション等）の前段階として重要な役割を果たします。

### 主な機能
- データの抽出・切り出し
- 信号の平滑化・平均化
- ノイズ付加・除去
- 基準信号の減算
- A-scanデータの生成

## 含まれるツール

### データ抽出・変換
- **`extract_Bscan.py`** - B-scanデータの抽出
- **`k_make_Ascans.py`** - A-scanデータの生成
- **`k_trimming.py`** - データの切り出し・トリミング

### 信号処理・改善
- **`averaging.py`** - B-scanデータの移動平均処理
- **`k_subtract.py`** - 基準信号の減算処理
- **`k_add_noise.py`** - ノイズの付加（テスト・評価用）
- **`gain_function.py`** - ゲイン関数の処理

## 使用方法

### データ抽出・変換
```bash
# B-scanデータを抽出
python -m tools.data_processing.extract_Bscan config.json

# A-scanデータを生成
python -m tools.data_processing.k_make_Ascans config.json

# データをトリミング
python -m tools.data_processing.k_trimming config.json
```

### 信号処理・改善
```bash
# 移動平均を適用
python -m tools.data_processing.averaging input_file.out

# 基準信号を減算
python -m tools.data_processing.k_subtract config.json

# ノイズを付加
python -m tools.data_processing.k_add_noise config.json

# ゲイン関数を処理
python -m tools.data_processing.gain_function config.json
```

## 典型的な処理フロー

### 基本的な前処理
1. **データ読み込み**: 生データの読み込み
2. **品質チェック**: データの整合性確認
3. **トリミング**: 不要な部分の除去
4. **平滑化**: ノイズ除去のための平均化
5. **基準信号減算**: 直達波等の除去
6. **データ保存**: 処理結果の保存

### A-scan生成
1. **B-scanデータ準備**: 複数トレースデータの確認
2. **位置指定**: 抽出位置の指定
3. **A-scan抽出**: 単一トレースの抽出
4. **品質確認**: 抽出結果の確認
5. **保存**: A-scanデータの保存

## 設定ファイル例

```json
{
    "input_file": "raw_data.out",
    "output_dir": "processed/",
    "processing_params": {
        "time_window": [0, 100],
        "spatial_window": [0, 50],
        "averaging_window": 3,
        "noise_level": 0.01,
        "gain_factor": 2.0
    },
    "output_format": "numpy",
    "save_plots": true
}
```

## 各ツールの詳細

### averaging.py
B-scanデータに移動平均フィルタを適用し、空間的なノイズを除去します。

### k_subtract.py
基準信号（背景信号、直達波等）を減算して、反射信号を強調します。

### k_trimming.py
時間窓・空間窓を指定してデータを切り出し、解析対象領域を限定します。

### k_add_noise.py
テスト・評価用にデータにノイズを付加します。アルゴリズムの頑健性評価に使用。

### extract_Bscan.py
複数の出力ファイルからB-scanデータを抽出・統合します。

### k_make_Ascans.py
B-scanデータから指定位置のA-scanを生成します。

### gain_function.py
時間減衰補正等のゲイン関数を適用します。

## 関連ツール

- **core/**: 基本的なファイル処理
- **signal_processing/**: より高度な信号処理
- **visualization/**: 処理結果の可視化
- **analysis/**: 処理後のデータ解析

## 注意事項

- データ処理は可逆性を考慮し、必要に応じて元データを保持してください
- 移動平均処理では、空間分解能の劣化に注意してください
- ノイズ付加は、実際のデータ特性に基づいて適切なレベルを設定してください
- 基準信号の減算では、信号の位相や振幅特性を考慮してください
- 大きなデータセットでは、メモリ使用量に注意してください
- 処理パラメータの記録・管理を適切に行い、再現性を確保してください