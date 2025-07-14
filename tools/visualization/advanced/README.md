# Advanced Visualization Tools

幾何構造、スナップショット、速度構造などの高度な可視化機能を提供します。

## 含まれるツール

- **`k_plot_geometry.py`** - JSON設定からの幾何構造可視化
- **`k_plot_geometry_imaging.py`** - イメージング用幾何構造可視化
- **`k_plot_snapshot.py`** - スナップショットデータの可視化
- **`k_plot_velocity_structure.py`** - 速度構造の可視化

## 使用方法

```bash
# 幾何構造をプロット
python -m tools.visualization.advanced.k_plot_geometry config.json

# イメージング用幾何構造をプロット
python -m tools.visualization.advanced.k_plot_geometry_imaging config.json

# スナップショットをプロット
python -m tools.visualization.advanced.k_plot_snapshot config.json

# 速度構造をプロット
python -m tools.visualization.advanced.k_plot_velocity_structure config.json
```

これらのツールは、研究・解析における詳細な可視化と結果の解釈に使用されます。