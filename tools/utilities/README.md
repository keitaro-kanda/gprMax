# Utilities

このディレクトリには、GPRデータ解析をサポートするユーティリティツールが含まれています。入力ファイル生成、可視化補助、エラー解析などの補助的な機能を提供します。

## 概要

ユーティリティツールは、GPRデータ解析の効率化と自動化を支援する補助的なツールです。これらのツールは、他のカテゴリのツールと組み合わせて使用されることが多く、ワークフローの自動化や結果の評価に重要な役割を果たします。

### 主な機能
- 入力ファイルの自動生成
- アンテナパラメータの解析
- 誘電体構造の可視化
- エラー・品質評価
- CMPデータの可視化

## 含まれるツール

### 入力ファイル生成
- **`k_make_input_file.py`** - gprMax入力ファイルの自動生成

### アンテナ・パラメータ解析
- **`plot_antenna_params.py`** - アンテナパラメータの解析・可視化

### 構造可視化
- **`plot_dielectric_structure.py`** - 誘電体構造の可視化
- **`plot_Bscan_CMP.py`** - CMPデータのB-scan可視化

### エラー・品質評価
- **`plot_error.py`** - エラー解析・品質評価の可視化

## 使用方法

### 入力ファイル生成
```bash
# gprMax入力ファイルを自動生成
python -m tools.utilities.k_make_input_file config.json
```

### アンテナ・パラメータ解析
```bash
# アンテナパラメータを解析
python -m tools.utilities.plot_antenna_params antenna_data.out
```

### 構造可視化
```bash
# 誘電体構造を可視化
python -m tools.utilities.plot_dielectric_structure config.json

# CMPデータを可視化
python -m tools.utilities.plot_Bscan_CMP config.json
```

### エラー・品質評価
```bash
# エラー解析を実行
python -m tools.utilities.plot_error config.json
```

## 各ツールの詳細

### k_make_input_file.py
gprMaxのシミュレーション用入力ファイルを自動生成します：

- **幾何構造**: 地下構造の自動設定
- **材料特性**: 誘電率・導電率の設定
- **アンテナ配置**: 送受信アンテナの自動配置
- **パラメータ最適化**: 計算効率の最適化

### plot_antenna_params.py
アンテナの特性を解析・可視化します：

- **周波数特性**: 利得・指向性の周波数依存性
- **時間波形**: 励起波形の解析
- **空間パターン**: 放射パターンの可視化
- **効率評価**: アンテナ効率の計算

### plot_dielectric_structure.py
誘電体構造を可視化します：

- **誘電率分布**: 2D/3D誘電率マップ
- **層構造**: 多層構造の可視化
- **不均質性**: 不均質媒質の表示
- **材料境界**: 境界面の明示

### plot_Bscan_CMP.py
CMP（Common Midpoint）データを可視化します：

- **CMP集録**: 共通中点集録データの表示
- **速度解析**: 速度解析用の可視化
- **品質評価**: データ品質の評価
- **フィッティング**: 双曲線フィッティング結果の表示

### plot_error.py
エラー解析と品質評価を行います：

- **統計的エラー**: 平均・分散・標準偏差
- **系統的エラー**: バイアス・トレンド解析
- **品質指標**: S/N比・相関係数
- **比較解析**: 理論値との比較

## 設定ファイル例

### 入力ファイル生成用
```json
{
    "geometry": {
        "domain_size": [100, 100, 10],
        "discretization": [0.01, 0.01, 0.01],
        "layers": [
            {"thickness": 2, "material": "vacuum"},
            {"thickness": 3, "material": "regolith"},
            {"thickness": 5, "material": "basalt"}
        ]
    },
    "antennas": {
        "type": "dipole",
        "frequency": 150e6,
        "separation": 0.5,
        "height": 0.05
    },
    "simulation": {
        "time_window": 100e-9,
        "pml_thickness": 10
    }
}
```

### 可視化用
```json
{
    "input_file": "simulation_result.out",
    "output_dir": "visualizations/",
    "plot_params": {
        "colormap": "jet",
        "aspect_ratio": "equal",
        "interpolation": "bilinear",
        "save_format": "png"
    }
}
```

## 典型的な使用例

### シミュレーション準備
1. **構造設計**: 地下構造の設計
2. **入力ファイル生成**: 自動入力ファイル作成
3. **パラメータ確認**: 設定値の確認
4. **シミュレーション実行**: gprMaxの実行

### 結果評価
1. **品質チェック**: データ品質の評価
2. **エラー解析**: 誤差の定量化
3. **構造可視化**: 結果の可視化
4. **アンテナ特性**: アンテナ性能の評価

## 関連ツール

- **core/**: 基本的なファイル処理
- **visualization/**: 高度な可視化
- **analysis/**: 詳細な解析
- **velocity_analysis/**: 速度解析での利用

## 注意事項

- 入力ファイル生成時は、物理的制約（CFL条件等）を考慮してください
- アンテナパラメータ解析では、周波数帯域を適切に設定してください
- 誘電体構造の可視化では、材料境界の精度に注意してください
- エラー解析では、統計的に有意な結果を得るため十分なサンプル数を確保してください
- 大きなデータセットでは、メモリ使用量と処理時間に注意してください
- 設定ファイルのバージョン管理を行い、再現性を確保してください