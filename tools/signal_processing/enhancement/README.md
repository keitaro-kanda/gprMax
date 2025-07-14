# Signal Enhancement Tools

信号強化・改善機能を提供します。

## 含まれるツール

- **`k_gain.py`** - ゲイン関数の適用
- **`k_matched_filter.py`** - マッチドフィルタリング
- **`k_pulse_compression.py`** - パルス圧縮処理

## 使用方法

```bash
# ゲイン関数を適用
python -m tools.signal_processing.enhancement.k_gain config.json

# マッチドフィルタリングを実行
python -m tools.signal_processing.enhancement.k_matched_filter config.json

# パルス圧縮を実行
python -m tools.signal_processing.enhancement.k_pulse_compression config.json
```

これらのツールは、信号品質の向上と特定信号の抽出に使用されます。