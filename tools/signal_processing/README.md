# Signal Processing Tools

このディレクトリには、GPRデータの信号処理ツールが含まれています。周波数域処理、時間域処理、信号強化の各種手法を提供します。

## 概要

信号処理ツールは以下の3つのサブディレクトリに分類されています：
- **frequency_domain/**: 周波数域での信号処理
- **time_domain/**: 時間域での信号処理
- **enhancement/**: 信号強化・改善手法

## サブディレクトリ構成

### frequency_domain/ - 周波数域処理
- **`fourier.py`** - フーリエ変換ユーティリティ
- **`k_spectrogram.py`** - スペクトログラム生成・解析
- **`k_wavelet.py`** - ウェーブレット変換による時間-周波数解析

### time_domain/ - 時間域処理
- **`k_acorr.py`** - 自己相関関数の計算
- **`k_autocorrelation.py`** - 自己相関解析の処理
- **`k_envelope.py`** - エンベロープ・瞬時周波数の計算

### enhancement/ - 信号強化
- **`k_gain.py`** - ゲイン関数の適用
- **`k_matched_filter.py`** - マッチドフィルタリング
- **`k_pulse_compression.py`** - パルス圧縮処理

## 使用方法

### 周波数域処理
```bash
# スペクトログラムを生成
python -m tools.signal_processing.frequency_domain.k_spectrogram config.json

# ウェーブレット変換を実行
python -m tools.signal_processing.frequency_domain.k_wavelet config.json

# フーリエ変換ユーティリティを使用
python -m tools.signal_processing.frequency_domain.fourier input_data.out
```

### 時間域処理
```bash
# 自己相関を計算
python -m tools.signal_processing.time_domain.k_acorr config.json

# エンベロープを計算
python -m tools.signal_processing.time_domain.k_envelope config.json

# 自己相関解析を実行
python -m tools.signal_processing.time_domain.k_autocorrelation config.json
```

### 信号強化
```bash
# ゲイン関数を適用
python -m tools.signal_processing.enhancement.k_gain config.json

# マッチドフィルタリングを実行
python -m tools.signal_processing.enhancement.k_matched_filter config.json

# パルス圧縮を実行
python -m tools.signal_processing.enhancement.k_pulse_compression config.json
```

## 典型的な処理フロー

1. **前処理**: データの読み込みと基本的な前処理
2. **周波数解析**: スペクトログラムやウェーブレット変換による周波数特性の把握
3. **時間域処理**: 自己相関やエンベロープ計算による時間特性の抽出
4. **信号強化**: ゲイン補正やフィルタリングによる信号品質の改善
5. **後処理**: 結果の可視化と保存

## 設定ファイル例

```json
{
    "input_file": "gpr_data.out",
    "output_dir": "processed/",
    "processing_params": {
        "time_window": [0, 100],
        "frequency_range": [50, 1000],
        "gain_factor": 2.0,
        "filter_type": "bandpass"
    }
}
```

## 関連ツール

- **visualization/**: 処理結果の可視化
- **analysis/**: 処理後のデータ解析
- **data_processing/**: 基本的なデータ処理
- **velocity_analysis/**: 速度解析への応用

## 注意事項

- 信号処理は計算量が大きいため、大きなデータセットでは処理時間に注意してください
- 周波数域処理では、サンプリング定理を考慮してナイキスト周波数に注意してください
- フィルタリング処理では、位相歪みの影響を考慮してください
- 各処理の前後でデータの整合性を確認してください