# A-scan 計算用 .in 生成スクリプト（CE-4 LPR 岩石サイズ・波形調査）

CE-4 LPR CH2 を模擬した 2D gprMax 計算で、岩石・空隙のサイズ、深さ、形状を変えた A-scan を
一括で計算するための入力一式を作る。

## ファイル

| ファイル | 役割 |
|---|---|
| `generate_ascan_inputs.py` | 本体。ケースの一覧、`.in`、ケース JSON、一覧ファイル、実行スクリプトを作る。設定はファイル先頭の `[EDIT HERE]` |
| `materials.py` | 媒質（レゴリス、海の玄武岩、高地の岩石、空隙）の誘電率・tanδ と 2 極 Debye パラメータ。**媒質の値はここだけが持つ** |
| `theory.py` | 理論値（往復時間、反射係数、予想される極性、フレネル半径、減衰量）。解析側でも import して使える |
| `grid_geometry.py` | 形状のラスタ化（gprMax と同じ規則）と、gprMax の `.vti` の読み込み |
| `verify_geometry.py` | gprMax が作った `.vti` を読み、ケース JSON の予測と照合する |
| `make_h5_input.py` | h5 が必要になったケースだけ、h5 を書き出す `.in` を作る |

依存は numpy のみ（`verify_geometry.py` も numpy のみ）。Python 3.8 以上。

## 使い方

```bash
# 1. 件数・容量・検証結果だけ確認（ファイルは作らない）
python generate_ascan_inputs.py --dry-run --sec-per-case 1800

# 2. 生成（ROOT / CAMPAIGN の下に作る）
python generate_ascan_inputs.py

# 3. 形状だけ作って確認（FDTD なし。1 ケース数秒）
cd /Volumes/SSD_Kanda_BUFFALO/gprMax/domain_5x14/size_waveform_investigation/ascan_v01
PYTHON=python ./check_geometry.sh            # gprMax を入れた Python を PYTHON に
python scripts/verify_geometry.py .          # ok / NG を表示

# 4. 本計算（.done があるケースは飛ばす。止めても再実行で続きから）
PYTHON=python ./run_all.sh
```

一部だけ生成・実行したいとき:

```bash
python generate_ascan_inputs.py --depths 1 --targets basalt void --shapes circle
python generate_ascan_inputs.py --groups reference
./run_all.sh my_list.txt        # in_list.txt から行を抜き出したリストを渡す
```

* 絞り込みを指定すると、`--groups reference` を付けない限り参照計算は含まれない。
* `manifest.json` / `in_list.txt` / `out_list.txt` は、フォルダ内のケース JSON を走査して毎回作り直すので、
  一部ずつ生成しても常に全ケースが載る。
* 同じ内容なら再生成しても何も変わらない（`unchanged`）。内容が変わった `.in` / `.json` は `--force` で上書きする。
  **計算済み（`.done` か `.out` がある）のケースは `--force` でも上書きしない。**
* `campaign.json` の物理設定（材料、計算領域、アンテナ、波形、時間窓）が変わったら停止する。
  設定を変えるときは `CAMPAIGN` を新しい名前（`ascan_v02` など）にする。

## 計算パターン（既定）

| | 値 |
|---|---|
| 対象 | `basalt`（海の玄武岩）、`highland`（高地の岩石）、`void`（空隙、gprMax 組み込みの `free_space`） |
| 形状 | `circle`（サイズ = 直径）、`square`（サイズ = 一辺） |
| サイズ | 1–15 cm（1 cm 刻み）、20, 30, 40, 50 cm |
| 深さ（地表から上端） | 1, 6, 12 m |
| 不規則な形状（様子見） | `irreg01`, `irreg02` × 6, 10, 30 cm × 深さ 1 m × basalt, void（サイズ = 面積相当直径） |
| 参照計算 | `ref_background`（レゴリスのみ）、`ref_freespace`（自由空間のみ） |

合計 356 ケース（主計算 342、不規則な形状 12、参照 2）。容量は .vti が約 24 GB、.out が約 0.25 GB。

### 媒質（`materials.py`）

| | ε (f0) | tanδ | 根拠 |
|---|---|---|---|
| regolith | 3.0 | 0.004 | Dong+2020; Chen+2022; Feng+2022 |
| basalt | 7.7927 | 0.03039 | ρ = 3.15 g/cm³（Kiefer+2012）、FeO+TiO₂ = 20 wt%、Carrier+1991 Fig. 9.52 |
| highland | 5.7739 | 0.00568 | ρ = 2.69 g/cm³（Kiefer+2012）、FeO+TiO₂ = 4.6 wt%（Korotev+2003）、同上 |
| void | 1 | 0 | 真空 |

損失は 2 極 Debye で、250–750 MHz の帯域内で tanδ を一定にしている（最大平坦条件、ε″ の変動 p-p 約 1%）。
ε は帯域の幾何平均 f0 = 433.01 MHz での値。帯域内の ε′ は Kramers-Kronig 則により変化する（玄武岩で約 2.5%）。
`python materials.py` で一覧と検算を表示できる。

## ディレクトリ構成

```
ascan_v01/
├── campaign.json        共通設定（物理設定、計画、スクリプトのハッシュ、環境）
├── manifest.json        全ケースの一覧（パスはキャンペーンフォルダからの相対パス）
├── manifest.csv         同じ内容の表（pandas で絞り込む用）
├── in_list.txt          .in の一覧（相対パス、実行順：参照 → 深さの浅い順）
├── out_list.txt         .out の一覧（絶対パス）
├── run_all.sh / check_geometry.sh
├── run_history.tsv      run_all.sh が追記する計算時間の記録
├── scripts/             生成に使ったスクリプトのコピー
├── reference/ref_background/ , reference/ref_freespace/
└── <target>/<shape>/depthXXm/DXXXcm/
    ├── <case_id>.in
    ├── <case_id>.json
    ├── <case_id>_geometry.vti
    ├── <case_id>.log  /  <case_id>.done
    └── A-scan/<case_id>.out
```

ケース ID は `{target}_{shape}_depth{深さ 2 桁}m_D{サイズ 3 桁}cm`（例: `basalt_circle_depth01m_D020cm`）。
`pair_key`（`circle_depth01m_D020cm` など）は対象の種類を除いたキーで、同じ形状・深さ・サイズの岩石と空隙を対応づける。

## .in の内容

* 計算領域 5 × 14 m、dx = 2.5 mm、2D TMz、時間窓 170 ns（28,831 ステップ）
* レゴリスは y = 0–13 m（地表 y = 13 m）
* 送信点 x = 2.34 m、受信点 x = 2.66 m（間隔 32 cm）、高さ y = 13.3 m（地表から 30 cm）。対象は中点 x = 2.5 m の真下
* 波形 `gaussiandot`、400 MHz（受信点での中心周波数は約 500 MHz）
* 対象は誘電率の平滑化なし（`n`）。分散性材料は gprMax が自動で平滑化を切るので、空隙も揃えて切っている
* `.vti` は全領域で出力。h5（`#geometry_objects_write`）は `DISABLED` を付けて無効にしてある
* スナップショットは Python ブロックの中でコメントアウトしてある（for 文の 2 行の先頭の `#` を外すと有効）。
  全時間・0.5 ns 間隔だと約 25 GB のメモリが必要。`t_start` / `t_end` を対象のエコー付近に絞ると減らせる。
  生成時に `--enable-snapshots` を付けると、選んだケースだけ有効にできる
* gprMax が読むのは `#` で始まる行だけ。説明の行は `#` で始めない

本計算に必要なメモリは約 2.5 GB（スナップショットなし、gprMax の表示値）。

## ケース JSON の主な項目

| キー | 内容 |
|---|---|
| `case_id`, `group`, `pair_key` | 識別。`group` は `reference` / `main` / `irregular` |
| `model`, `antenna` | 計算領域、時間窓、dt、送受信点の位置、波形 |
| `materials.background`, `materials.target` | ε、tanδ、ε∞、Δε、τ、密度、組成、出典、gprMax の行 |
| `target.shape`, `size_m`, `size_definition`, `depth_top_m` | 公称の形状とサイズ |
| `target.gprmax_commands` | 対象を作る gprMax の行 |
| `target.irregular` | 不規則な形状の頂点座標と乱数シード |
| `target.nominal_nadir` | 連続形状での、中点の真下の上端深さと厚さ |
| `target.as_built` | 格子化後の形状（gprMax と同じ規則で予測。`verify_geometry.py` で .vti と一致を確認） |
| `target.as_built.nadir.effective_thickness_m` | Ez が感じる実効的な厚さ（= セルの厚さ + dx、下の注意を参照） |
| `theory.common` | 直接波と地表反射の伝搬時間 |
| `theory.nominal` | 公称形状での理論値（上端・下端の往復時間、時間差、反射係数、予想される極性、フレネル半径、減衰量） |
| `theory.as_built_effective` | 実効的な厚さでの理論値 |
| `files` | `.in`、`.out`、`.vti`、ログ、`.done` の相対パス |
| `references` | 差し引く参照計算の `.out` |

### 注意

* **時間は伝搬時間のみ。** 波源のピークまでの遅れ（約 2.6 ns）は含まない。`ref_freespace` の直接波のピーク時刻から
  `theory.common.direct_wave_s` を引いた値が波源の遅れ。
* 往復時間は送受信点の間隔（32 cm）と空気・レゴリス・対象の屈折を考慮して計算している（境界は水平面で近似）。
  `t_nadir_s` は間隔 0 の値。
* **実効的な厚さ。** 平滑化なしの 2D TMz では、対象のセルの 4 隅の Ez 節点に対象の材料が入るため、
  Ez が感じる対象はセルの範囲より 1 節点広い。実効的な厚さはセルの厚さ + dx（2.5 mm）。
  1 cm の岩石では 25%、6 cm では 4% の差になるので、小さいサイズの推定誤差を議論するときは
  `nominal` と `as_built_effective` の両方と比べること。上端の深さは地表も同じ規則で変わるので変わらない。
* 予想される極性は、送信波を `NPN` と仮定している（`TRANSMITTED_POLARITY`）。`ref_freespace` で確認すること。

## 解析での読み込み例

```python
import json, h5py, numpy as np
from pathlib import Path

camp = Path("/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_5x14/size_waveform_investigation/ascan_v01")
man = json.loads((camp / "manifest.json").read_text())

def read_ez(path):
    with h5py.File(path, "r") as f:
        return np.array(f["rxs/rx1/Ez"]), f.attrs["dt"]

bg, dt = read_ez(camp / "reference/ref_background/A-scan/ref_background.out")
for row in man["cases"]:
    if row["group"] != "main" or row["shape"] != "circle":
        continue
    case = json.loads((camp / row["json"]).read_text())
    ez, _ = read_ez(camp / row["output"])
    echo = ez - bg                                  # 直接波と地表反射を除いた対象のエコー
    th = case["theory"]["nominal"]
    print(row["case_id"], th["dt_bottom_minus_top_bistatic_s"])
```

## h5 が必要になったとき

```bash
python scripts/make_h5_input.py basalt/circle/depth01m/D020cm/basalt_circle_depth01m_D020cm.in
python -m gprMax basalt/circle/depth01m/D020cm/basalt_circle_depth01m_D020cm_h5.in --geometry-only
```

`<case_id>_geometry.h5`（約 560 MB）と `<case_id>_geometry_materials.txt` ができる。FDTD の計算はしない。

## 変更するとき

* 媒質の値は `materials.py` の `[EDIT HERE]` だけを変える
* 計算パターン・領域・アンテナは `generate_ascan_inputs.py` の `[EDIT HERE]`
* 物理設定を変えたら `CAMPAIGN` を新しい名前にする
