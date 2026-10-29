# A-scan サイズ推定解析（`analyze_size_estimation.py`）

計算済みの A-scan（gprMax の `.out`）から、対象（岩石・空隙）の上端と下端のエコーを自動で探し、
2 つのエコーの時間差から対象の高さを推定する。結果は表（CSV）1 つ、設定（JSON）1 つ、
集計図（対象ごとに 2 種類）、ケースごとの確認用 A-scan 図にまとめる。

サイズ推定の下限（どのサイズから推定できると言えるか）は**自動では判定しない**。集計図と確認用図を見て決める。

## 使い方

VS Code でこのファイルを開き、右上の実行ボタン（▷ Run Python File）を押す。ターミナルからなら引数なしで
`python analyze_size_estimation.py`。実行すると次の順に尋ねるので、番号を入力する（Enter で既定値）。

```
=== A-scan サイズ推定解析 ===
キャンペーンフォルダ（manifest.json がある場所） [/Volumes/.../ascan_v01]:      ← Enter で既定値
対象（target）
   1. basalt     計算済み 114 / 114
   2. highland   計算済み   0 / 114
   3. void       計算済み   0 / 120
  番号（カンマかスペース区切り、Enter = すべて）: 1
形状（shape）
   1. circle ...
  番号（カンマかスペース区切り、Enter = すべて）:
深さ（対象上端、m）
   ...
解析するケース: 114（未計算で飛ばすケース: 0）
ケースごとの確認用 A-scan 図を作りますか (Y/n):
パラメータ: 閾値 -90 dB, 許容差 1 ns, FWHM 自動, 図の余白 5 ns（変えるときはスクリプト先頭の [EDIT HERE]）
実行しますか (Y/n):
```

* キャンペーンフォルダは、パスを貼り付けるか、Finder からフォルダをターミナルにドラッグしてもよい
* 番号の代わりに名前（`basalt`、`circle` など）でも選べる。形状と深さの一覧は、前に選んだ対象・形状の中だけを出す
* 計算が終わっていないケースは飛ばす（件数だけ表示）。計算済みが増えたら、もう一度実行する。出力はすべて上書きされる
* 確認用図を作らない（n）と、CSV と集計図だけを数秒で作り直せる。パラメータを試すときに使う
* キャンペーンフォルダの `scripts/` にあるコピーを実行したときは、そのキャンペーンが既定値になる

依存：numpy、scipy（`signal.hilbert`、`signal.find_peaks`）、h5py、matplotlib。単独のファイルで動く
（生成スクリプトのほかのモジュールは import しない）ので、好きな場所に置いてよい。

### 設定（スクリプト先頭の `[EDIT HERE]`）

| 名前 | 既定値 | 意味 |
|---|---|---|
| `DEFAULT_CAMPAIGN` | `.../ascan_v01` | キャンペーンフォルダの既定値 |
| `THRESHOLD_DB` | −90 | 検出の閾値。地表反射のエンベロープの最大値に対する dB（振幅比） |
| `TOLERANCE_NS` | 1.0 | 予測時刻とエンベロープ極大点を対応づける許容差 [ns] |
| `FWHM_NS` | `None` | 上端と下端の分離の基準 [ns]。`None` なら `ref_freespace` の直接波から測る |
| `PLOT_MARGIN_NS` | 5 | 確認用図の表示範囲（t_top − 5 ns から t_bot + 5 ns） |
| `FONT_SIZE` | 12 | 図の文字の大きさ（matplotlib の既定は 10） |
| `DEPTH_COLORS`, `SHAPE_STYLES` | – | 集計図の深さの色と、形状の線種・記号 |

使った値、選んだ対象・形状・深さ、参照計算から測った値（FWHM、波源の遅れ、A_surf など）は
`analysis_config.json` に残る。

## 処理の流れ

時刻はすべて計算開始からの時間（サンプル番号 × dt）で扱う。

1. **ケースの選択**：`manifest.json` の main と irregular から、`<case_id>.done` があるケースだけ。
   未計算の件数は表示するだけ。
2. **参照計算**（1 回だけ）
   * `ref_freespace` の直接波で |Ez| が最大のピークの時刻 t_direct を求める。
     そのピークが正で、前後 1 FWHM 以内に負のピーク（ピークの 10% 以上）があれば送信波は **NPN**。
     NPN でなければ停止する（予想される符号の前提が崩れるため）
   * 波源の遅れ t_src = t_direct − （送受信点間隔 / c）。理論値はケース JSON の `theory.common.direct_wave_s`
   * 直接波のエンベロープの半値全幅を **FWHM** とする
   * `ref_background − ref_freespace` のエンベロープの最大値を地表反射の強さ **A_surf** とする
   * 検出の閾値 A_th = A_surf × 10^(THRESHOLD_DB / 20)
3. **前処理**：s(t) = Ez（ケース）− Ez（`ref_background`）、エンベロープ e(t) = |hilbert(s)|
4. **予測時刻**：t_top = `theory.nominal.top.t_bistatic_s` + t_src、t_bot = `theory.nominal.bottom.t_bistatic_s` + t_src
5. **段階 1（エンベロープ）**：e(t) の極大点のうち A_th 以上のものを取り出し、t_top に最も近いものを上端、
   t_bot に最も近いものを下端とする。どちらも予測時刻の ±tol 以内に限る
6. **エコーの範囲**：極大点 ± FWHM/2。上端と下端の極大点の間隔が FWHM より短いときは、
   その間の e(t) の極小点で区切る
7. **段階 2（電場のピーク）**：各範囲の電場のピーク（極値）のうち、予想される符号で絶対値が最大のものの時刻を
   τ_top、τ_bot とする。予想される符号はケース JSON の `theory.nominal.*.expected_polarity` の中央の文字
   （岩石：上端は負、下端は正。空隙：上端は正、下端は負）。
   範囲の端の値はピークとみなさない。範囲の中に予想される符号のピークがないとき（エコーが肩になっているだけ等）は
   τ を空欄にし、判定は single になる
8. **判定**（下の表）と **サイズ推定**

```
D_est = c (τ_bot − τ_top) / (2 √ε_true)        誤差 = (D_est − D) / D × 100 %
```

* ε_true はその対象の真の値（basalt 7.7927、highland 5.7739、void 1.0。ケース JSON の `materials.target.eps_r`）
* D は設定サイズ（高さ。円は直径、正方形は一辺、不規則な形状は高さ）。格子による実効的な厚さ（+2.5 mm）は
  `size_effective_cm` に記録するだけで、誤差の基準には使わない
* 推定に使うのは時間差だけなので、t_src の誤差は推定値に影響しない（探す位置と図への記入にだけ使う）
* 2D 計算では距離による減衰が 3D より緩やかなので、−90 dB での検出の判定は実際の LPR より検出しやすい側に偏る

### 判定

上から順に当てはめる。サイズを推定するのは partial と separated だけ。

| 判定 | 条件 | 集計図 |
|---|---|---|
| `not_detected` | 予測時刻の ±tol 以内に、上端・下端とも A_th 以上の極大点がない | 描かない |
| `single` | 上端と下端が同じ極大点に対応する、どちらか一方しか見つからない、または τ_bot ≤ τ_top | 描かない |
| `partial` | 別々の極大点に対応し、τ_bot − τ_top < FWHM | 白抜きの記号 |
| `separated` | 別々の極大点に対応し、τ_bot − τ_top ≥ FWHM | 塗りつぶしの記号 |

**極性の警告**（`polarity_flag_top` / `polarity_flag_bot`）：エコーの範囲で絶対値が最大のピークの符号が予想と逆。
判定は変えず、CSV と確認用図に記入するだけ。

## 出力

```
<キャンペーン>/analysis/size_estimation/
    results.csv                    全ケースの検出結果と推定サイズ（1 ケース 1 行）
    analysis_config.json           パラメータ、参照計算から測った値、件数、実行日時、スクリプトの sha256
    size_estimate_<target>.png/.pdf  設定サイズ vs 推定サイズ
    size_error_<target>.png/.pdf     設定サイズ vs 推定誤差
<ケースのフォルダ>/A-scan/size_estimation/<case_id>.png   確認用 A-scan 図
```

絞り込んで実行すると、`results.csv` と集計図はそのケースだけで作り直される（ほかの対象の集計図は前回のまま残る）。
論文用の図は、全ケースを対象に実行したものを使う。

### results.csv の列

時刻は ns、サイズは cm。推定しないケースや見つからなかったエコーの列は空欄。

| 列 | 内容 |
|---|---|
| `case_id`, `target`, `kind`, `shape`, `depth_m` | 識別（kind は rock か void） |
| `size_cm`, `size_effective_cm`, `eps_true` | 設定サイズ（高さ）、格子による実効的な厚さ（中点の真下）、推定に使った誘電率 |
| `t_top_pred_ns`, `t_bot_pred_ns` | 予測時刻（t_src を足したもの） |
| `env_top_ns`, `env_bot_ns` | 対応づけたエンベロープ極大点の時刻 |
| `env_top_db`, `env_bot_db` | その極大点の強さ（A_surf に対する dB） |
| `env_top_offset_ns`, `env_bot_offset_ns` | 極大点 − 予測時刻（時刻のずれの確認用） |
| `tau_top_ns`, `tau_bot_ns`, `dtau_ns` | 選んだ電場ピークの時刻と時間差（single でも求まれば記録する） |
| `category` | `not_detected` / `single` / `partial` / `separated` |
| `size_est_cm`, `error_pct` | 推定サイズと誤差 |
| `polarity_flag_top`, `polarity_flag_bot` | 極性の警告（1 = 予想と逆、0 = 予想どおり、空欄 = エコーなし） |

### 集計図

* 横軸：設定サイズ（cm、リニア）。縦軸：推定サイズ（cm）または推定誤差（%）。補助線は 1:1 と 0%（黒の点線）
* 色 = 深さ（matplotlib の既定の色の順：1 m は C0 青、6 m は C1 橙、12 m は C2 緑。深さに固定で割り当てるので、絞り込んでも色は変わらない）
* 線種と記号 = 形状（circle 実線 ○、square 破線 □、irreg01 △、irreg02 ▽。不規則な形状は線なし）
* 塗りつぶし = separated、白抜き = partial。線は推定できた点をサイズ順に結ぶ
* 集計図は `results.csv` だけから作る（`plot_summary()`）。体裁を変えるときはこの関数を直す

### 確認用 A-scan 図

1 ケース 1 枚（png、150 dpi）。

* 黒の実線（k）：背景を差し引いた Ez（A_surf で規格化）。緑の破線（g）：エンベロープ（±両側）
* 黒の縦の点線：予測時刻（上端と下端）。薄い帯：エコーの範囲（青 b = 上端、赤 r = 下端。重なると紫に見える）
* 丸印：選んだ上端（青）と下端（赤）のピーク
* 右側の文字：case_id、設定サイズ、真の ε、予測の時間差（dt pred）と検出した時間差（dtau）、推定サイズ、誤差、判定、極性の警告
* single でも、上端と下端が同じ極大点に対応したときは同じ範囲から選んだ 2 つのピークと dtau を表示・記録する
  （サイズは推定しない。判定の境目を確かめる材料）

## 確認のしかた

1. 最初に表示される参照計算の値を見る：送信波が NPN、FWHM が約 1.4 ns、t_src が 1 ns 前後
2. `results.csv` をサイズ順に見て、single → partial → separated と移ることを確かめる
3. 極性の警告が立ったケース、誤差が大きいケース、判定の境目のケースは確認用図を開いて、
   選んだピークが上端・下端のエコーの主ピークになっているかを見る
4. 対応づけがずれているときは `env_*_offset_ns` を見て `TOLERANCE_NS` を調整する