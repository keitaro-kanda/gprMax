# A-scan 計算用 .in 生成スクリプト v2（ascan_v02）

v1（`ascan_generator/`、キャンペーン `ascan_v01`）とは別のフォルダ・別のファイル。v1 のファイルは変更していない。
フォルダ内だけで動く（v1 のモジュールは import しない）。

## v1 からの変更点

| 項目 | v1 | v2 |
|---|---|---|
| 形状 | 円形、正方形（+ 不規則形状 12 ケース） | 正方形、ランダムな多角形 2 通り（cut4 / cut8） |
| 岩石 | basalt（ε 7.79, tanδ 0.0304）、highland（ε 5.77, tanδ 0.0057）。密度・組成から経験式で計算 | 比誘電率 6.0 で統一。tanδ = 0.03（rock_hiloss）と 0.005（rock_loloss） |
| 空隙 | void | void（変更なし） |
| 正方形のサイズ | 1–15 cm（1 cm 刻み）、20、30、40、50 cm | 同じ |
| 多角形のサイズ | 6、10、30 cm（深さ 1 m のみ） | 5、10、20、30、40、50 cm（深さ 1、6、12 m） |
| キャンペーン名 | ascan_v01 | ascan_v02 |

計算領域、格子、時間窓、アンテナ、波形、レゴリス、Debye モデルの作り方、出力の構成（ケース JSON、manifest、
run_all.sh / run_subset.sh / check_geometry.sh）は v1 と同じ。

## ファイル

| ファイル | 役割 |
|---|---|
| `generate_ascan_inputs_v2.py` | 本体。設定はファイル先頭の `[EDIT HERE]` |
| `materials.py` | v2 の媒質（regolith、rock_hiloss、rock_loloss、void）。**v1 の materials.py とは中身が違う** |
| `theory.py`, `grid_geometry.py`, `verify_geometry.py`, `make_h5_input.py` | v1 と同じ内容のコピー |

## 使い方

```bash
python generate_ascan_inputs_v2.py --dry-run --sec-per-case 1800   # 件数・容量・時間の見積もりだけ
python generate_ascan_inputs_v2.py                                 # 生成（ROOT / ascan_v02）
python materials.py                                                # 媒質と Debye パラメータの確認
```

その後の手順（形状の確認、本計算、部分実行）は v1 と同じ。

```bash
cd /Volumes/SSD_Kanda_BUFFALO/gprMax/domain_5x14/size_waveform_investigation/ascan_v02
PYTHON=python ./check_geometry.sh                  # 形状だけ作る
python scripts/verify_geometry.py .                # 予測と照合
PYTHON=python ./run_subset.sh rock_hiloss/square   # 部分実行
```

## 計算ケース（既定）

| | 媒質 | 形状 | サイズ (cm) | 深さ (m) | 件数 |
|---|---|---|---|---|---|
| main | rock_hiloss, rock_loloss, void | square | 1–15, 20, 30, 40, 50 | 1, 6, 12 | 171 |
| irregular | rock_hiloss, rock_loloss, void | cut4v1, cut4v2, cut8v1, cut8v2 | 5, 10, 20, 30, 40, 50 | 1, 6, 12 | 216 |
| reference | ref_background, ref_freespace | | | | 2 |

合計 389 ケース。多角形を絞るときは `POLYGON` の `targets`、`depths_m`、`variants` を変える
（例: `variants=[1]` で 108 ケース減る）。

## ランダムな多角形

一辺 D の正方形の 4 つの角を切り落とした形。

* **cut4**：各角を 1 本の直線で切る（切断面 4 面、8 角形）
* **cut8**：各角を折れ線（2 本の線分）で切る（切断面 8 面、12 角形）
* 元の正方形の各辺に**残す長さは固定**：上面・底面は `keep_top_bottom` × D、左右の側面は `keep_sides` × D（既定はどちらも 0.30）
* **乱数で決めるのは位置だけ**：各辺の残す区間の位置（= 両端の角から切る長さの配分）と、cut8 の折れ点（位置と、角の側／内側へのずらし量）
  * 上面・底面の残す区間は、必ずアンテナの中点の真下を含む（区間の端から真下まで 0.05 D 以上）。そのため真下の厚さは D のまま
  * 1 つの角で各辺に沿って切る長さは 0.10 D 以上
  * 折れ点のずらし量は、角までの距離の −0.30〜+0.60 倍（+ は角の側に膨らむ）。|ずらし量| < 0.15 は引き直す（2 つの面がほぼ一直線になり、実質 1 面になるのを防ぐ）
* 同じ variant では、切り口の位置を cut4 と cut8 で共通にする。cut8 は cut4 の切り口に折れ点を 1 つ加えた形で、両者の違いは切断面の数だけ
* 形は (方法, variant) ごとに乱数シード（`base_seed + variant`）で固定し、サイズで相似に拡大縮小する
* サイズ = 高さ（鉛直方向の全長）= D
* shape 名は `cut4v1`、`cut8v1`、`cut4v2`、`cut8v2`。ケース JSON の `target.polygon` に頂点・切り方・シード・残す長さを記録

多角形のケースは manifest で `group = "irregular"` になる（v1 の解析スクリプトがそのまま読めるように）。

## 媒質

| 媒質 | ε′（433 MHz） | tanδ | 根拠 |
|---|---|---|---|
| regolith | 3.0 | 0.004 | Dong et al. (2020); Chen et al. (2022); Feng et al. (2022) |
| rock_hiloss | 6.0 | 0.03 | tanδ: 月の岩石サンプルの範囲 0.005–0.03 の上端（Strangway et al., 1977, p. 417 要旨, p. 418） |
| rock_loloss | 6.0 | 0.005 | tanδ: 同じ範囲の下端 |
| void | 1.0 | 0 | 真空（gprMax の free_space） |

岩石の ε′ = 6.0 の根拠：
* Apollo 14 の岩石サンプル（1 MHz）は 4.8–7。結晶質の 14310 が 6–7（Table 3）/ 6.4（Table 4）、角礫岩の 14321 が 5.9–6
  （Chung, Westphal & Olhoeft, 1972, Proc. Lunar Sci. Conf. 3rd, pp. 3161–3172, Table 3 p. 3165, Table 4 p. 3170）
* 固体サンプル（密度 2.2–3.3 g/cm³）はおおむね 4–8（Strangway et al., 1977, NASA SP-370, Fig. 1 p. 418）。
  典型値は約 7.7（同 p. 418, p. 421）

値を変えるときは materials.py の `ROCK_EPS_R`、`ROCK_HILOSS` / `ROCK_LOLOSS` の `tan_delta` を変える（キャンペーン名も変える）。

## 解析スクリプトとの関係

ケース JSON の構成は v1 と同じなので、`analyze_size_estimation.py` はそのまま使える（main と irregular を読む）。
集計図で多角形を区別するには、解析スクリプトの `SHAPE_STYLES` に次を追加する（なければ全部 ◇ になる）。

```python
    "cut4v1": dict(ls=":",  marker="^"),
    "cut4v2": dict(ls=":",  marker="v"),
    "cut8v1": dict(ls="-.", marker="<"),
    "cut8v2": dict(ls="-.", marker=">"),
```
