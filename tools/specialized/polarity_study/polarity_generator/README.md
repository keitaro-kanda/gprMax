# 予想外極性の研究（EPS 再投稿）の .in 生成スクリプト（polarity_v01）

原稿「Reliability of GPR echo polarity: insight into lunar subsurface void/rock observation」の
2 章（表 1、表 2）の設定で、gprMax の .in を作る。
別研究の A-scan 生成ツール（ascan_v02）の構成を引き継ぎ、本研究で使わない機能（円形・多角形、
送受信点の間隔、岩石の 2 種類の損失）を外したもの。フォルダ内だけで動く。

## ファイル

| ファイル | 役割 |
|---|---|
| `generate_polarity_inputs.py` | 本体。設定はファイル先頭の `[EDIT HERE]` |
| `materials.py` | 媒質（regolith、rock、void）と 2 極 Debye の設計。媒質の値はここだけが持つ |
| `theory.py` | 直下のエコーの到達時刻、反射係数、期待される極性、フレネル半径（原稿の式）、減衰量 |
| `grid_geometry.py` | gprMax と同じ規則での #box のラスタ化と、.vti の読み込み |
| `verify_geometry.py` | 形状の .vti とケース JSON の予測を照合する |
| `make_h5_input.py` | 必要なケースだけ h5 を書き出す .in を作る（ascan_v02 と同じ） |

## 使い方

```bash
python generate_polarity_inputs.py --dry-run --sec-per-case 600   # 件数・容量・時間の見積もりだけ
python generate_polarity_inputs.py                                # 生成（ROOT / polarity_v01）
python materials.py                                               # 媒質と Debye パラメータの確認
```

生成後（キャンペーンフォルダで実行）:

```bash
cd /Volumes/SSD_Kanda_BUFFALO/gprMax/domain_10x10/Polarity_EPS/polarity_v01
PYTHON=python ./check_geometry.sh                  # 形状だけ作る（全ケース）
python scripts/verify_geometry.py .                # 予測と照合
PYTHON=python ./run_subset.sh reference slab       # 参照計算と平行平板計算を先に
PYTHON=python ./run_subset.sh rock/depth02m        # 岩片のノミナル計算
PYTHON=python ./run_all.sh                         # 残りを全部（.done があるケースは飛ばす）
```

`run_subset.sh --list <階層>` で、実行せずに対象のケースだけを確認できる。

## 計算設定（原稿の表 1）

| 項目 | 値 |
|---|---|
| 計算領域 | 10 × 10 m（2D TMz）。y = 0–9 m がレゴリス、9–10 m が真空 |
| 格子 / 時間ステップ / 計算時間 | 0.005 m / 11.79 ps / 100 ns（8481 ステップ） |
| PML | gprMax の既定値（10 セル、.in には書かない） |
| 送信源 | 微小ダイポール（z）、電流波形 gaussiandot、f = 400 MHz（放射電場のピーク 500 MHz） |
| アンテナ | 送信点と受信点は同じ位置（間隔 0 m）、x = 5 m、地表から 0.3 m |

| 媒質 | ε′（433 MHz） | tanδ |
|---|---|---|
| regolith | 3.0 | 0.004 |
| rock | 6.0 | 0.01 |
| void | 1.0 | 0（gprMax の free_space） |

損失は 2 極 Debye（250–750 MHz で tanδ 一定、最大平坦条件。ascan_v02 と同じ設計）。

## 計算ケース（原稿の表 2）

| パターン | 対象 | 上端の深さ | 高さ | 幅 | 件数 |
|---|---|---|---|---|---|
| reference | 背景（レゴリスのみ）、自由空間 | | | | 2 |
| slab | 岩片 / 空洞 | 2 m | 0.5 m / 1.2 m | 無限（左右の PML まで） | 2 |
| nominal | 岩片 | 2 m | 0.125, 0.25–2.50 m（0.25） | 0.2, 0.4–4.0 m（0.4） | 121 |
| nominal | 空洞 | 2 m | 0.3, 0.6–6.0 m（0.6） | 0.3, 0.6–6.0 m（0.6） | 121 |
| depth | 岩片 | 1, 4 m | 0.5, 1.25 m | ノミナルと同じ | 44 |
| depth | 空洞 | 1, 4 m | 1.2, 3.0 m | ノミナルと同じ | 44 |

合計 334 ケース。ref_freespace の受信点は送信点の 1 m 下（送信波形と主ピークの遅れ δt を求める）。
それ以外のケースは送受信点が同じセルにあるので、受信波形には送信源の場が含まれる。
解析では必ず ref_background を差し引くこと（原稿 2.5 節の手順 1）。

## フォルダ構成（パターン別 > 深さ別 > サイズ別）

```
polarity_v01/
├── campaign.json, manifest.json / .csv, in_list.txt, out_list.txt
├── run_all.sh, run_subset.sh, check_geometry.sh, scripts/
├── reference/ref_background/, reference/ref_freespace/
├── slab/rock/depth02m/H0500mm/, slab/void/depth02m/H1200mm/
├── rock/depth01m/, rock/depth02m/, rock/depth04m/
│   └── H0125mm_W0200mm/
│       ├── rock_depth02m_H0125mm_W0200mm.in / .json
│       └── A-scan/rock_depth02m_H0125mm_W0200mm.out
└── void/depth01m/, void/depth02m/, void/depth04m/
```

nominal（上端の深さ 2 m）と depth（1 m、4 m）は、上端の深さのフォルダで分かれる。
ケース JSON の `group` は `reference` / `slab` / `nominal` / `depth`。
`pair_key` が同じケースは、岩片と空洞で同じ形状（幅 1.2、2.4、3.6 m など）。

## ケース JSON の理論値（`theory.nominal`）

* `top` / `bottom`: 直下のエコーの往復伝搬時間 `t_nadir_s`（δt を含まない）、反射係数、期待される極性、
  等価距離 D、フレネル半径 r_F、その直径、幅とフレネル帯の直径の比 w / 2r_F、往復の減衰量
* `as_built_effective`: 格子上の実効的な厚さ（セルの厚さ + dx）で計算した同じ値

## 注意

* 設定を変えるときは `CAMPAIGN` を新しい名前にする（同じ名前で物理設定が変わると停止する）。
* スナップショットは既定で無効（.in の中でコメントアウト）。代表ケースだけ
  `--enable-snapshots` を付けて生成するか、.in の for 文 2 行の「#」を外す。全時間・全領域だと 1 ケース数 GB になる。
* レゴリスの tanδ = 0.004 の出典は、ascan_v02 では Feng et al. (2022)、原稿では Li et al. (2020)。原稿側で統一すること。
