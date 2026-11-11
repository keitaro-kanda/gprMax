# random_field_profile.py — ランダム場（密度のゆらぎ）の深さプロファイル

no_ice のクラッタが 1.9〜2.6 m（往復走時 30〜38 ns）で弱まらなくなる理由を切り分けるためのツール。その深さで密度のゆらぎがたまたま強ければ「ランダム場の偶然」、深さによらず一定なら「媒質の性質（遅れて届く散乱波など）」の可能性が高い。

新しい計算は不要で、gprMax の地形ファイルを読むだけ。

---

## 1. 使い方

```
python random_field_profile.py
Enter geometry .h5 or .vti file path:  （パスを貼る）
```

| 渡せるファイル | 中身 | 材料の読み方 |
|---|---|---|
| `level6_voxels.h5`（Ascan.in が書く） | 地下だけ | 同じフォルダの `level6_voxels_materials.txt` |
| `geometry.h5`（`#geometry_objects_write`） | 空気を含む全域 | 同じフォルダの `geometry_materials.txt` |
| `geometry.vti`（`#geometry_view`、`n`） | 空気を含む全域 | 材料名は .vti の末尾の `<gprMax>` ブロック。`<名前>_materials.txt` があれば ε′ もそこから |

**no_ice の計算のファイルを使う。** 氷層の材料が入っていても密度は読めるが、(c) の反射係数に氷層の境界が入る。

出力は入力ファイルと同じフォルダの `random_field_profile/`：`fig_random_field_profile.png`（と `pdf/`）、`random_field_profile.csv`、`run_info.txt`。

| 設定 | 既定 | 意味 |
|---|---|---|
| `RHO_STEP` | 0.001 | 密度の量子化刻み [g/cm³]（Ascan.in と同じ） |
| `F_HZ` | 1.25 GHz | 材料ファイルの Debye 極から ε′ を求める周波数 |
| `X_RANGE_M` | None | 水平方向に使う範囲 [m]。None なら全部 |
| `SMOOTH_HW_M` | 0.10 m | 太線の移動平均の半幅（パルスの長さ程度） |
| `HIGHLIGHT_DEPTH_M` | (1.9, 2.55) | 灰色の帯（クラッタが平らになる深さ） |
| `BAND_DEPTHS_M` | 0.5–1.9 / 1.9–2.55 / 2.55–2.95 m | 画面と run_info に出す区間（最初が基準） |
| `TX_HEIGHT_M`, `SOURCE_DELAY_NS` | 0.35 m, 5.008 ns | 右の縦軸（往復走時の目安）に使う |

---

## 2. 記号と計算

| 記号 | 意味 |
|---|---|
| $\rho(x, z)$ | セル $(x, z)$ の密度 [g/cm³]。材料名 `r{Q}_dry` の $Q$ から $\rho = Q \times$ `RHO_STEP` |
| $\varepsilon'(x, z)$ | 比誘電率の実部。材料ファイルの Debye 極から $f$ = `F_HZ` で計算（無ければ $1.871^{\rho}$） |
| $N_x$ | 水平方向のセルの数 |
| $\langle \cdot \rangle_x$ | 同じ深さ $z$ での水平平均 $\frac{1}{N_x}\sum_x$ |
| $\Delta z$ | セルの大きさ（2.5 mm） |

**(a) 密度の水平平均**

$$
\bar\rho(z) = \langle \rho(x, z) \rangle_x
$$

**(b) 密度の相対ゆらぎ**（各深さで、水平方向の標準偏差 ÷ 平均）

$$
s(z) = \frac{\sqrt{\bigl\langle (\rho - \bar\rho)^2 \bigr\rangle_x}}{\bar\rho(z)}
$$

Ascan.in は $\rho = \rho_C(z)\,(1 + s_\rho R)$（$R$ は平均 0・分散 1 のランダム場、$s_\rho$ = 0.05）で作っているので、場が深さによらず同じ統計なら $s(z) \approx 5\,\%$ で一定になる。1 回の実現なので、深さごとに偶然上下する。

**(c) 隣り合うセルの境界の反射係数の RMS**

深さ $z$ と $z + \Delta z$ のセルの境界の反射係数（垂直入射）を

$$
r(x, z) = \frac{\sqrt{\varepsilon'(x, z + \Delta z)} - \sqrt{\varepsilon'(x, z)}}{\sqrt{\varepsilon'(x, z + \Delta z)} + \sqrt{\varepsilon'(x, z)}}
$$

とし、水平方向に 2 乗平均の平方根をとる。

$$
r_{\mathrm{RMS}}(z) = \sqrt{\langle r^2 \rangle_x}
$$

クラッタのもとになる「散乱の強さ」の目安。(b) と同じ形になるはずだが、深いほど $\bar\rho$ と $\varepsilon'$ が大きくなる分、同じ $s$ でも少し大きくなる。

**太線**：(b) は深さ $\pm$`SMOOTH_HW_M` の移動平均、(c) は $r_{\mathrm{RMS}}^2$ の移動平均の平方根。

**右の縦軸（往復走時の目安）**

$$
t(z) = t_{\mathrm{src}} + \frac{2 h}{c} + \frac{2}{c}\sum_{z' \le z} \sqrt{\langle \varepsilon' \rangle_x(z')}\,\Delta z
$$

$t_{\mathrm{src}}$ は波源遅延、$h$ はアンテナの高さ、$c$ は光速。位相速度で計算した目安で、bscan_detection の群速度の走時とは 0.5 ns 程度ずれることがある。

---

## 3. 読み方

| 結果 | 解釈 |
|---|---|
| 1.9–2.55 m の (b)(c) が、前後の区間より 2 dB 程度大きい | クラッタの平らな部分は、この実現でゆらぎがたまたま強いため（ランダム場の偶然）。v_min の上昇も偶然の可能性が高い |
| (b) は深さによらずほぼ一定 | 平らな部分は場のゆらぎの強さでは説明できない。媒質の性質（浅い場所で散乱された波が遅れて届く、など）を疑う |

目安：クラッタの平らな部分で、信号に対する差は約 2.2 dB（30〜38 ns）。(b) か (c) で同じ程度の差があれば、それで説明できる。

---

## 4. 限界

- 密度の場そのものの統計で、波の伝わり方（散乱の方向・多重散乱・2 次元の広がり）は入っていない。
- (c) はセル間の差なので、パルスの帯域で見える構造より細かいものも含む。相関長（鉛直 0.06 m）より十分細かいセルなので、深さ方向の形は (b) とほぼ同じになる。
- 合成データ（1.9〜2.55 m だけゆらぎを 1.3 倍にした場）で、その区間の (b) が +1.5 dB、(c) が +2.1 dB と出ることを確かめた。.vti の読み込みは k_visualize_vtk_2D.py と同じ方法で、実ファイルでは未確認。
