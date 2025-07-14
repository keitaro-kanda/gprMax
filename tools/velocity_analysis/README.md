# Velocity Analysis Tools

このディレクトリには、GPRデータから地下構造の速度情報を推定するツールが含まれています。RMS速度、内部速度、理論的速度推定の各種手法を提供します。

## 概要

速度解析は、GPRデータから地下の電磁波伝播速度を推定し、地下構造の把握や深度変換に利用する重要な解析手法です。このディレクトリには以下の機能が含まれています：

- RMS速度の推定
- 内部速度の計算
- 理論的速度推定
- CMP（Common Midpoint）解析
- 幾何構造からの速度計算

## 含まれるツール

### RMS速度推定
- **`estimate_Vrms.py`** - Suの手法を使用したRMS速度推定
- **`estimate_Vrms_CMP.py`** - CMPデータからのRMS速度推定
- **`estimate_Vrms_DePue.py`** - DePue手法によるRMS速度推定（レガシー）
- **`estimate_Vrms_theory.py`** - 理論的RMS速度推定（レガシー）

### 内部速度・理論計算
- **`estimate_internal_velocity.py`** - Dixの公式を使用した内部速度推定
- **`DePue_eq9.py`** - DePue方程式による速度推定
- **`calc_Vrms_from_geometry.py`** - 幾何パラメータからのRMS速度計算

## 使用方法

### RMS速度推定
```bash
# 基本的なRMS速度推定
python -m tools.velocity_analysis.estimate_Vrms config.json

# CMPデータからのRMS速度推定
python -m tools.velocity_analysis.estimate_Vrms_CMP config.json

# DePue手法による推定
python -m tools.velocity_analysis.estimate_Vrms_DePue config.json
```

### 内部速度推定
```bash
# Dixの公式による内部速度推定
python -m tools.velocity_analysis.estimate_internal_velocity config.json

# 幾何パラメータからの速度計算
python -m tools.velocity_analysis.calc_Vrms_from_geometry config.json
```

### 理論的速度推定
```bash
# DePue方程式による推定
python -m tools.velocity_analysis.DePue_eq9 config.json

# 理論的RMS速度推定
python -m tools.velocity_analysis.estimate_Vrms_theory config.json
```

## 速度解析の理論

### RMS速度
RMS（Root Mean Square）速度は、多層構造における実効的な速度を表現します：

```
V_rms = sqrt(Σ(V_i² * t_i) / Σt_i)
```

### 内部速度
Dixの公式により、RMS速度から各層の内部速度を計算できます：

```
V_int = sqrt((V_rms² * t - V_rms_prev² * t_prev) / (t - t_prev))
```

### CMP解析
Common Midpoint解析では、双曲線フィッティングにより速度を推定します：

```
t² = t₀² + x²/V_rms²
```

## 設定ファイル例

```json
{
    "input_file": "cmp_data.out",
    "output_dir": "velocity_analysis/",
    "analysis_params": {
        "time_window": [0, 200],
        "offset_range": [0, 50],
        "velocity_range": [0.05, 0.2],
        "fitting_method": "hyperbola",
        "threshold": 0.5e-3
    },
    "geometry": {
        "antenna_spacing": 0.1,
        "profile_length": 100,
        "time_step": 0.1
    }
}
```

## 典型的な解析フロー

1. **データ準備**: CMPデータまたはB-scanデータの準備
2. **前処理**: ノイズ除去、ゲイン補正
3. **ピーク検出**: 反射波の自動検出
4. **双曲線フィッティング**: 速度パラメータの推定
5. **品質評価**: 相関係数やフィッティング誤差の評価
6. **結果出力**: 速度構造の可視化と保存

## 関連ツール

- **analysis/**: フィッティング・ピーク検出ツール
- **visualization/**: 速度構造の可視化
- **migration_imaging/**: マイグレーション処理での速度利用
- **data_processing/**: データ前処理

## 注意事項

- 速度解析は反射波の品質に大きく依存するため、事前の信号処理が重要です
- 多層構造では、各層の速度が適切に分離されているか確認してください
- CMPデータの品質（オフセット範囲、S/N比）が解析精度に影響します
- 理論値と実測値を比較して、推定結果の妥当性を検証してください
- 地下構造の複雑さに応じて、適切な解析手法を選択してください