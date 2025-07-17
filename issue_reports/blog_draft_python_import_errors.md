# 【Python初心者必見】VSCodeで動くのにコマンドラインで動かない？ModuleNotFoundErrorの正体と解決法

## はじめに

Pythonでプログラミングをしていて、こんな経験はありませんか？

- VSCodeの実行ボタンでは正常に動作する
- でもコマンドラインから実行すると`ModuleNotFoundError`が発生
- 「なぜ実行方法で結果が変わるの？」と困惑

今回は、実際に遭遇したこの問題を通して、Pythonのモジュールシステムの基本から実践的な解決策まで、分かりやすく解説します。

## 実際に起きた問題

### 症状

以下のような違いが発生しました：

```bash
# ✅ これは動く
python -m tools.visualization.basic.k_make_Ascans

# ❌ これはエラー
python /path/to/tools/visualization/basic/k_make_Ascans.py
```

エラーメッセージ：
```
ModuleNotFoundError: No module named 'tools.core'
```

「同じスクリプトなのに、なぜ実行方法で結果が変わるの？」これが今回解決した謎です。

## 原因を徹底解剖

### 1. `__init__.py`ファイルの役割を理解しよう

Pythonでは、ディレクトリをパッケージとして認識させるために`__init__.py`ファイルが必要です。

**問題があった構造：**
```
tools/
├── __init__.py              # ✅ 存在
├── core/                    # ❌ __init__.py なし
│   └── outputfiles_merge.py
├── analysis/                # ❌ __init__.py なし
│   └── k_detect_peak.py
└── visualization/           # ❌ __init__.py なし
    ├── basic/
    │   └── k_make_Ascans.py
    └── analysis/            # ❌ __init__.py なし
        └── k_plot_TWT_estimation.py
```

この状態では、Pythonは`tools.core`や`tools.analysis`をパッケージとして認識できません。

**正しい構造：**
```
tools/
├── __init__.py              # ✅ 
├── core/
│   ├── __init__.py          # ✅ 追加
│   └── outputfiles_merge.py
├── analysis/
│   ├── __init__.py          # ✅ 追加
│   └── k_detect_peak.py
└── visualization/
    ├── __init__.py          # ✅ 追加
    ├── basic/
    │   └── k_make_Ascans.py
    └── analysis/
        ├── __init__.py      # ✅ 追加
        └── k_plot_TWT_estimation.py
```

### 2. 実行方法の違いが引き起こす問題

**`python -m module.name`の場合：**
- Pythonが自動的にカレントディレクトリを`sys.path`に追加
- モジュール検索パスが適切に設定される

**`python /path/to/script.py`の場合：**
- スクリプトのディレクトリのみが`sys.path`に追加
- プロジェクトルートは自動で追加されない

### 3. インポート文の問題

多くのファイルで、以下のような不適切なインポートが使われていました：

```python
# ❌ 問題のあるインポート
from outputfiles_merge import get_output_data

# ✅ 正しいインポート
from tools.core.outputfiles_merge import get_output_data
```

## 解決方法を実装してみよう

### ステップ1: `__init__.py`ファイルを作成

必要なディレクトリに空の`__init__.py`ファイルを作成します：

```bash
# Linux/Mac の場合
touch tools/core/__init__.py
touch tools/analysis/__init__.py
touch tools/visualization/__init__.py
touch tools/visualization/analysis/__init__.py

# Windows の場合
type nul > tools\core\__init__.py
type nul > tools\analysis\__init__.py
type nul > tools\visualization\__init__.py
type nul > tools\visualization\analysis\__init__.py
```

### ステップ2: インポート文を修正

**手動で修正する場合：**
```python
# 修正前
from outputfiles_merge import get_output_data

# 修正後  
from tools.core.outputfiles_merge import get_output_data
```

**一括で修正する場合（Linux/Mac）：**
```bash
# sedコマンドを使用
find . -name "*.py" -exec sed -i 's/from outputfiles_merge import/from tools.core.outputfiles_merge import/g' {} \;
```

### ステップ3: パス設定を堅牢にする

スクリプト内でプロジェクトルートを動的に設定する方法：

```python
import os
import sys
from pathlib import Path

# プロジェクトルートの取得
script_path = Path(__file__).resolve()
project_root = script_path.parents[3]  # ファイル階層に応じて調整

# sys.pathに追加
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# これで安全にインポートできる
from tools.core.outputfiles_merge import get_output_data
```

## より良い解決策：現代的なアプローチ

### pyproject.tomlを使った管理

```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "your-project"
version = "0.1.0"
dependencies = [
    "numpy",
    "matplotlib",
    "h5py"
]

[tool.setuptools.packages.find]
where = ["."]
include = ["tools*"]
```

### 開発用インストール

```bash
# プロジェクトルートで実行
pip install -e .
```

これにより、どこからでも`import tools`が可能になります。

## 予防策とベストプラクティス

### 1. プロジェクト構造の設計

```
your_project/
├── pyproject.toml
├── README.md
├── src/                    # ソースコードは src/ 以下に
│   └── your_package/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   └── module.py
│       └── utils/
│           ├── __init__.py
│           └── helpers.py
├── tests/
└── docs/
```

### 2. インポートのルール

```python
# ✅ 絶対インポートを使用
from your_package.core.module import function

# ❌ 相対インポートは避ける（特殊な場合を除く）
from ..core.module import function

# ❌ 不明瞭なインポート
from module import function
```

### 3. 開発環境の設定

**VS Code設定例（.vscode/settings.json）：**
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.analysis.extraPaths": [
        "."
    ]
}
```

### 4. テストの追加

```python
# tests/test_imports.py
def test_all_imports():
    """全ての重要なモジュールがインポートできることを確認"""
    try:
        from tools.core.outputfiles_merge import get_output_data
        from tools.analysis.k_detect_peak import detect_plot_peaks
        # 他の重要なインポート...
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")
```

## まとめ

今回の問題は、以下の要因が組み合わさって発生しました：

1. **`__init__.py`の不備** → パッケージが認識されない
2. **不適切なインポート文** → モジュールが見つからない  
3. **実行方法の違い** → パス設定の差異
4. **パス計算の誤り** → プロジェクトルートの取得失敗

**解決のポイント：**
- 適切なパッケージ構造の構築
- 絶対インポートの使用
- 実行環境に依存しないパス設定
- 現代的な開発手法の採用

## おわりに

Pythonのモジュールシステムは最初は複雑に感じるかもしれませんが、基本原則を理解すれば、より保守性の高いコードが書けるようになります。

特に大きなプロジェクトでは、今回紹介したような問題は頻繁に発生します。この記事が同じような問題に遭遇した方の助けになれば幸いです。

**次のステップ：**
- 自分のプロジェクトのパッケージ構造を見直してみる
- `pyproject.toml`を使った現代的な管理方法を試してみる
- 自動テストでインポートエラーを早期発見する仕組みを作る

Happy coding! 🐍✨

---

*この記事は実際のデバッグ体験をもとに作成されています。質問や改善提案がございましたら、お気軽にコメントください。*