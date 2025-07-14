# ModuleNotFoundError エラー解決レポート

**日付**: 2025年7月14日  
**エラーファイル**: `tools/visualization/basic/k_make_Ascans.py`  
**発生エラー**: `ModuleNotFoundError: No module named 'tools.core'`

## 問題の概要

`k_make_Ascans.py`を実行した際に、以下の2つの異なる実行方法で結果が異なる現象が発生した：

- ✅ **正常動作**: `python -m tools.visualization.basic.k_make_Ascans`
- ❌ **エラー発生**: `python /path/to/k_make_Ascans.py` (VSCodeの実行ボタン等)

## 根本原因の分析

### 1. Pythonパッケージ構造の不備

**問題**: toolsディレクトリのサブディレクトリに`__init__.py`ファイルが存在しなかった

```
tools/
├── core/                     # ❌ __init__.py なし
├── analysis/                 # ❌ __init__.py なし  
├── visualization/            # ❌ __init__.py なし
│   └── analysis/            # ❌ __init__.py なし
└── __init__.py              # ✅ 存在
```

**結果**: Pythonがサブディレクトリをパッケージとして認識できない

### 2. 不正なインポート文の散在

**問題**: 16個のファイルで不正なインポート文が使用されていた

```python
# ❌ 不正なインポート
from outputfiles_merge import get_output_data

# ✅ 正しいインポート  
from tools.core.outputfiles_merge import get_output_data
```

### 3. パス計算の誤り

**問題**: `k_make_Ascans.py`内のgprMaxルートディレクトリ計算が間違っていた

```python
# ❌ 修正前: 3回のdirname() → /Users/keitarokanda/gprMax/tools
gprmax_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ✅ 修正後: 4回のdirname() → /Users/keitarokanda/gprMax  
gprmax_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
```

### 4. 実行方法による動作の違い

| 実行方法 | Pythonパスの動作 | 結果 |
|---------|----------------|------|
| `python -m module.name` | カレントディレクトリが自動でsys.pathに追加 | ✅ 動作 |
| `python /path/to/script.py` | スクリプト内で明示的にパス設定が必要 | ❌ エラー |

## 対処方法

### 1. __init__.pyファイルの作成

以下のディレクトリに空の`__init__.py`ファイルを作成：

```bash
touch tools/core/__init__.py
touch tools/analysis/__init__.py  
touch tools/visualization/__init__.py
touch tools/visualization/analysis/__init__.py
```

### 2. インポート文の一括修正

16個のファイルで以下の置換を実行：

```bash
# sedコマンドを使用した一括置換
sed -i '' 's/from outputfiles_merge import get_output_data/from tools.core.outputfiles_merge import get_output_data/g' [対象ファイル群]
```

**修正対象ファイル**:
- `tools/visualization/analysis/k_plot_TWT_estimation.py`
- `tools/visualization/basic/plot_Bscan.py`
- `tools/data_processing/k_subtract.py`
- `tools/analysis/k_fitting.py`
- `tools/analysis/k_detect_peak.py`
- `tools/analysis/k_gradient.py`
- その他10ファイル

### 3. パス計算の修正

`k_make_Ascans.py`のgprmax_root計算を修正：

```python
# gprMaxルートディレクトリの正しい計算
gprmax_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
```

## 検証結果

修正後、両方の実行方法で正常に動作することを確認：

```bash
# ✅ モジュール実行
python -m tools.visualization.basic.k_make_Ascans

# ✅ 直接実行 (VSCode等)
python /Users/keitarokanda/gprMax/tools/visualization/basic/k_make_Ascans.py
```

## 予防策と推奨事項

### 1. プロジェクト構造の標準化

- 全てのPythonパッケージディレクトリに`__init__.py`を配置
- 相対インポートではなく絶対インポートを使用
- パッケージ構造を明確に文書化

### 2. インポート文の統一

```python
# 推奨: 絶対インポート
from tools.core.outputfiles_merge import get_output_data

# 非推奨: 相対インポートや不正なインポート
from outputfiles_merge import get_output_data
```

### 3. パス設定の堅牢性

- `__file__`を基準とした相対パス計算の確認
- 実行環境に依存しないパス設定の実装
- パス設定ロジックのテスト追加

### 4. 開発環境の統一

- VS Codeの設定ファイル（`.vscode/settings.json`）でPythonパスを明示
- 開発チーム内での実行方法の統一
- CI/CDでの複数実行方法のテスト

## 技術的教訓

1. **Pythonモジュールシステムの理解**: `__init__.py`の重要性と絶対インポートの優位性
2. **実行方法の違い**: `-m`フラグと直接実行の動作差異
3. **パス計算の重要性**: ファイル階層構造を正確に反映した計算の必要性
4. **一括修正の効率性**: 同様の問題が複数ファイルに散在する場合の対処法

## 関連ファイル

- **主要修正ファイル**: `tools/visualization/basic/k_make_Ascans.py`
- **追加された__init__.pyファイル**: 4個
- **インポート修正ファイル**: 16個
- **プロジェクトルート**: `/Users/keitarokanda/gprMax`

このエラーは、Pythonプロジェクトの構造設計と実行環境の違いに起因する典型的な問題であり、適切なパッケージ構造の重要性を示す事例となった。