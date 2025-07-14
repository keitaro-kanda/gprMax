# Frequency Domain Signal Processing Tools

周波数域での信号処理機能を提供します。

## 含まれるツール

- **`fourier.py`** - フーリエ変換ユーティリティ
- **`k_spectrogram.py`** - スペクトログラム生成・解析
- **`k_wavelet.py`** - ウェーブレット変換による時間-周波数解析

## 使用方法

```bash
# スペクトログラムを生成
python -m tools.signal_processing.frequency_domain.k_spectrogram config.json

# ウェーブレット変換を実行
python -m tools.signal_processing.frequency_domain.k_wavelet config.json

# フーリエ変換ユーティリティを使用
python -m tools.signal_processing.frequency_domain.fourier input_data.out
```

これらのツールは、信号の周波数特性解析と時間-周波数解析に使用されます。