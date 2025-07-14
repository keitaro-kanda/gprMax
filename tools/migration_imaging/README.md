# Migration and Imaging Tools

このディレクトリには、GPRデータのマイグレーション処理とイメージング再構成ツールが含まれています。散乱体の位置を正確に再構成し、地下構造の真の形状を復元するための各種手法を提供します。

## 概要

マイグレーション・イメージング処理は、GPRデータに含まれる散乱・回折パターンを時間領域や周波数領域で処理し、地下構造の真の位置と形状を復元する重要な技術です。

### 主な機能
- 時間領域マイグレーション
- 周波数-波数（f-k）領域マイグレーション
- イメージング再構成
- 結果の統合・可視化

## 含まれるツール

### マイグレーション処理
- **`k_migration.py`** - 時間領域マイグレーション処理
- **`k_fk_migration.py`** - f-k領域マイグレーション処理
- **`migration_merge.py`** - マイグレーション結果の統合
- **`migration_plot.py`** - マイグレーション結果の可視化

### イメージング再構成
- **`imaging.py`** - 基本的なイメージング・再構成アルゴリズム
- **`imaging_mono.py`** - 単色波イメージング処理

## 使用方法

### 時間領域マイグレーション
```bash
# 基本的な時間領域マイグレーション
python -m tools.migration_imaging.k_migration config.json

# マイグレーション結果の統合
python -m tools.migration_imaging.migration_merge config.json

# 結果の可視化
python -m tools.migration_imaging.migration_plot config.json
```

### 周波数領域マイグレーション
```bash
# f-k領域マイグレーション
python -m tools.migration_imaging.k_fk_migration config.json

# イメージング再構成
python -m tools.migration_imaging.imaging config.json

# 単色波イメージング
python -m tools.migration_imaging.imaging_mono config.json
```

## マイグレーション理論

### 時間領域マイグレーション
時間領域マイグレーションは、各時間サンプルにおける散乱体の可能な位置を計算し、波の到達時間に基づいて真の位置を推定します：

```
t = sqrt((x-xs)² + (z-zs)²)/v + sqrt((x-xr)² + (z-zr)²)/v
```

### f-k領域マイグレーション
周波数-波数領域では、分散関係を利用して効率的にマイグレーション処理を行います：

```
kz = sqrt(ω²/v² - kx²)
```

### イメージング条件
散乱体の位置でのイメージング条件：

```
I(x,z) = Σ U(x,z,t) * δ(t - t_migration)
```

## 設定ファイル例

```json
{
    "input_file": "bscan_data.out",
    "output_dir": "migration_results/",
    "migration_params": {
        "velocity_model": "constant",
        "velocity_value": 0.1,
        "aperture_angle": 45,
        "imaging_depth": 5.0,
        "method": "kirchhoff"
    },
    "processing_params": {
        "time_window": [0, 100],
        "spatial_sampling": 0.01,
        "frequency_range": [50, 1000]
    }
}
```

## 典型的な処理フロー

1. **データ準備**: B-scanデータの読み込みと前処理
2. **速度モデル**: 速度解析結果の適用
3. **マイグレーション**: 時間領域またはf-k領域での処理
4. **イメージング**: 散乱体位置の再構成
5. **後処理**: 結果の統合と品質向上
6. **可視化**: 最終結果の表示と保存

## 速度モデル

マイグレーション処理では、正確な速度モデルが重要です：

### 定速度モデル
```python
velocity = constant_value
```

### 多層モデル
```python
velocity = f(depth)
```

### 不均質モデル
```python
velocity = f(x, z)
```

## 関連ツール

- **velocity_analysis/**: 速度モデルの構築
- **signal_processing/**: 前処理・信号強化
- **visualization/**: 結果の可視化
- **analysis/**: 散乱体検出・解析

## 注意事項

- マイグレーション処理は計算量が大きいため、大きなデータセットでは処理時間に注意してください
- 速度モデルの精度がマイグレーション結果の品質に直接影響します
- アパーチャー角度の設定により、分解能と計算時間のトレードオフがあります
- f-k領域マイグレーションでは、エイリアシングに注意してください
- 結果の品質評価には、既知の散乱体位置との比較が有効です
- 複雑な地下構造では、より高度な速度モデルが必要になります