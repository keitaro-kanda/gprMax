# Time Domain Signal Processing Tools

時間域での信号処理機能を提供します。

## 含まれるツール

- **`k_acorr.py`** - 自己相関関数の計算
- **`k_autocorrelation.py`** - 自己相関解析の処理
- **`k_envelope.py`** - エンベロープ・瞬時周波数の計算

## 使用方法

```bash
# 自己相関を計算
python -m tools.signal_processing.time_domain.k_acorr config.json

# 自己相関解析を実行
python -m tools.signal_processing.time_domain.k_autocorrelation config.json

# エンベロープを計算
python -m tools.signal_processing.time_domain.k_envelope config.json
```

これらのツールは、信号の時間特性解析と統計的処理に使用されます。