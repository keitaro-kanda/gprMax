# 極性判定と結果のまとめ（polarity_v01 の解析）

原稿 2.5 節の手順で、全ケースの上面・底面のエコーの極性を判定し、まとめの図を描く。
`generate_polarity_inputs.py` が作ったキャンペーンフォルダ（`polarity_v01`）を読む。
必要なもの: Python 3、numpy、h5py、matplotlib（gprMax を入れた環境でよい）。

## ファイル

| ファイル | 役割 |
|---|---|
| `analyze_polarity.py` | 段階 1。.out を読んで判定し、csv とケースごとの A-scan 図を出す。設定はファイル先頭の `[EDIT HERE]` |
| `plot_polarity_maps.py` | 段階 2。csv だけを読み、まとめの図を描く（数秒） |
| `polarity_core.py` | 判定の計算（送信波形の切り出し、規格化相互相関、分類） |

## 使い方

```bash
# 段階 1（全 334 ケース。A-scan 図込みで数分。ターミナルに 1 ケース 1 行で進捗と残り時間を表示）
python analyze_polarity.py /Volumes/SSD_Kanda_BUFFALO/gprMax/domain_10x10/Polarity_EPS/polarity_v01

# 段階 2
python plot_polarity_maps.py /Volumes/SSD_Kanda_BUFFALO/gprMax/domain_10x10/Polarity_EPS/polarity_v01/analysis/ncc_v01
python plot_polarity_maps.py .../analysis/ncc_v01 --rho-min 0.7      # 閾値を変えて描き直す（別フォルダに出る）
```

パスを省略すると、`analyze_polarity.py` 先頭の `CAMPAIGN_DIR` と `ANALYSIS_NAME` を使う。

段階 1 のオプション: `--groups slab nominal depth`、`--targets rock void`、`--cases ID ...`（一部だけ判定し直す。
結果は前回の csv に上書きで合流する）、`--no-ascan`（図を描かない）、`--name`（解析名）、`--force`。
判定の設定（T_w など）を変えたときは、`--name` で別の解析名にする（同じ名前だと停止する）。

## 判定の設定（原稿 2.5 節）

| 項目 | 値 |
|---|---|
| 背景の除去 | x(t) = 各ケース − ref_background |
| 送信波形 s(t) | ref_freespace（送信点から 1 m）の波形のうち、振幅が主ピークの 3% 以上の範囲（主ピークの −1.49 〜 +1.82 ns） |
| 主ピークの遅れ δt | 主ピークの時刻 − 1 m / c（約 2.68 ns。実行時に求めて表示する） |
| 到達時刻 τ | 表 2 の寸法による直下の往復時間（ケース JSON の `theory.nominal`）+ δt |
| 時間窓 | τ ± T_w、T_w = 1.5 ns |
| t′ の範囲 | \|t′ − τ\| ≤ 0.45 ns（送信波形の自己相関が最初に 0 になる時間差 0.47 ns より狭い） |
| 極性 | \|ρ\| が最大となる t′ での ρ の符号（ρ > 0 = 送信波形と同じ NPN） |
| Type 1 / 2 | 期待される極性と一致 / 逆 |
| Type 3 | 上面と底面の時間窓が重なる（τ_bottom − τ_top < 2T_w。岩片 h = 0.125 m、空洞 h = 0.3 m）、または \|ρ\| < 0.5 |

## 出力（`<キャンペーン>/analysis/ncc_v01/`）

```
ncc_v01/
├── polarity_results.csv       1 ケース・1 面につき 1 行（下の列の説明）
├── slab_check.csv             平行平板計算の判定（Type 1、|ρ| ≈ 1 になるはず）
├── analysis_settings.json     設定、δt、送信波形の切り出し範囲、自己相関のゼロ点（原稿に書く値の控え）
├── traces.npz                 背景を除いた全波形 x(t)（case_ids, x, dt）と送信波形
├── figs/transmitted_waveform.png   送信波形と切り出し範囲、自己相関と t′ の範囲
├── ascan/rock/depth02m/<case_id>.png など   ケースごとの A-scan と判定（原稿の図 4 の元）
└── maps_rho0.50/              段階 2 の図（閾値ごとのフォルダ）
    ├── type_maps/type_map_rock_depth02m.png/.pdf など    まとめ図（横軸 幅、縦軸 高さ、色 Type、破線 w = 2r_F）
    ├── score_maps/score_map_rock_depth02m.png/.pdf など  補助図 (a)：ρ × 期待される符号（+1 期待どおり、−1 逆）
    ├── fresnel_normalized/fresnel_normalized_rock.png/.pdf など  補助図 (b)：横軸 w / 2r_F、深さを並べて比較
    └── type_counts.csv        パターン・面ごとの Type 1/2/3 の件数
```

### polarity_results.csv の主な列

| 列 | 内容 |
|---|---|
| `pattern` | rock/depth02m、void/depth04m、slab/rock など |
| `face` | top / bottom |
| `expected_polarity` / `observed_polarity` | 期待される極性 / 判定した極性 |
| `type`, `type_reason` | 1, 2, 3 と、Type 3 の理由（`overlap`, `low_corr`） |
| `rho_best`, `lag_best_ns` | \|ρ\| が最大のときの ρ と、そのときの t′ − τ |
| `rho_at_tau` | t′ = τ のときの ρ（ずらさない場合の値。参考） |
| `rho_pos_max`, `rho_neg_min` | t′ の範囲内での ρ の最大値と最小値 |
| `tau_ns`, `window_start_ns`, `window_end_ns` | 到達時刻と時間窓（δt を含む） |
| `tau_as_built_ns` | 格子上の実効寸法による到達時刻（参考） |
| `fresnel_radius_m`, `width_over_fresnel_diameter` | その面の r_F と w / 2r_F |
| `amplitude_ls`, `window_peak_abs` | 送信波形の倍率（最小二乗）と窓内の最大振幅 |

段階 2 は `type_reason` が `overlap` のものを Type 3 のまま残し、それ以外は `--rho-min` で分類し直す。
