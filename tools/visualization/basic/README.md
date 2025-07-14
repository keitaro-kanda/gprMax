# Basic Visualization Tools

A-scan、B-scanの基本的なプロット機能を提供します。

## 含まれるツール

- **`plot_Ascan.py`** - A-scan（単一トレース）データの基本プロット
- **`plot_Bscan.py`** - B-scan（複数トレース）データの基本プロット  
- **`k_plot_Ascan_from_Bscan.py`** - B-scanデータから指定A-scanを抽出・プロット

## 使用方法

```bash
# A-scanをプロット
python -m tools.visualization.basic.plot_Ascan output_file.out

# B-scanをプロット  
python -m tools.visualization.basic.plot_Bscan output_file.out

# B-scanから特定のA-scanを抽出
python -m tools.visualization.basic.k_plot_Ascan_from_Bscan config.json
```

これらのツールは、GPRデータの最も基本的な可視化を提供し、データの品質確認や概要把握に使用されます。