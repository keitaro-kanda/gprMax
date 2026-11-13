# calc_LLL_dispersion_profile.py — 周波数分散のあるレゴリスの理論プロファイル

FeO+TiO$_2$ が多く（20 wt% 程度）、イルメナイトによって周波数分散が出るレゴリスについて、水氷を含めた誘電率・減衰の深さプロファイルと、LSR（スペクトル比法）で推定される $\tan\delta$ を計算する。月（本研究）と火星（Meng+2026）で $\tan\delta$ の違いの検出しやすさが違う理由を考える材料に使う。

計算は 1 分程度で、入力ファイルは不要（`python calc_LLL_dispersion_profile.py`）。

---

## 1. 旧コード・現在のコードとの関係

| 項目 | 本ツール | calc_LLL_mixing_profile.py（現在） | calc_mixing_dispersion_profile.py（旧） |
|---|---|---|---|
| 乾燥レゴリスの経験式 | Fig. 9.54（450 MHz） | Fig. 9.53（SOILS） | Fig. 9.54 |
| 経験式を合わせる周波数 | 450 MHz（$\varepsilon'$ と $\varepsilon''$ の両方） | 1 GHz | $\varepsilon''$ は 450 MHz、$\varepsilon'$ は周波数 0 |
| 分散 | 平坦な部分 ＋ Boivin の緩和（§3） | なし（$\varepsilon''$ 一定） | Boivin 20 wt% の 2 極 Debye を全体に適用 |
| 混合則 | LLL（pore / excess） | LLL（pore / excess） | Maxwell-Garnett（実質 excess） |
| 氷 | 3.15、$\tan\delta$ 2×10⁻⁵、0.94 g/cm³ | 同じ | 3.17、6×10⁻⁵、0.934 |
| 帯域・幾何・深さ | 0.5–2 GHz、往復、0.35 m、0–3 m | 同じ | 0.25–6 GHz、埋設 rx、0–5 m |

旧コードの Debye の値（$\Delta\varepsilon$ = 0.261 と 0.088、$\tau$ = 46.2 ps と 282 ps）は、Boivin+2022 Table 4 の 20 wt% 試料の Cole-Cole を 2 極 Debye で当てはめた値と一致する（`compare_ColeCole_Debye.py`）。

---

## 2. 記号

| 記号 | 意味 |
|---|---|
| $z$ | 深さ [m] |
| $\rho(z)$ | バルク密度 [g/cm³]。$\rho = 1.92\,(z_{\rm cm}+12.2)/(z_{\rm cm}+18)$ |
| $S$ | FeO+TiO$_2$ [wt%]（`FEOTIO2_WT`） |
| $f$, $\omega = 2\pi f$ | 周波数、角周波数 |
| $f_A$ | 経験式を合わせる周波数 450 MHz（`F_ANCHOR`） |
| $\varepsilon = \varepsilon' - j\varepsilon''$ | 複素比誘電率 |
| $\tan\delta = \varepsilon''/\varepsilon'$ | 損失正接 |
| $\varepsilon'_A,\ \tan\delta_A,\ \varepsilon''_A$ | 経験式の 450 MHz の値 |
| $w$ | 450 MHz の $\varepsilon''$ のうち、緩和（分散する部分）の割合 |
| $S_{\rm BG}$ | 緩和を起こさない FeO+TiO$_2$（`S_BG`、既定 7.5 wt%） |
| $A$ | 経験式の $S$ の係数 0.033 |
| $\Delta\varepsilon_k,\ \tau_k$ | Debye 極の強さと緩和時間 |
| $\varepsilon_\infty$ | 高周波極限の $\varepsilon'$ |
| $v$ | 氷の体積分率 |

---

## 3. 乾燥レゴリスのモデル

### 3.1 経験式（450 MHz）

$$
\varepsilon'_A = 1.843^{\rho}, \qquad
\tan\delta_A = 10^{\,0.033 S + 0.231\rho - 3.061}, \qquad
\varepsilon''_A = \varepsilon'_A \tan\delta_A
$$

### 3.2 Boivin+2022 から分かること

Boivin+2022 は、バイトウナイト（斜長石）にイルメナイトを 1〜20 wt% 混ぜた試料を 0.1〜8.5 GHz で測っている。

| 試料 | Cole-Cole の $\tau$ | $\alpha$ | $\Delta\varepsilon$ / ilm wt% | 1.25 GHz の $\varepsilon''$ ÷ 450 MHz の値 |
|---|---|---|---|---|
| 10 wt% | 43.7 ps | 0.717 | 0.0194 | 1.57 |
| 15 wt% | 52.0 ps | 0.746 | 0.0194 | 1.59 |
| 20 wt% | 50.4 ps | 0.756 | 0.0210 | 1.63 |
| バイトウナイトのみ | 分散なし（$\varepsilon''$ = 0.006 で一定） | | | 1.00 |

1. **緩和の形はイルメナイトの量によらない。** $\tau$ と $\alpha$ はほぼ同じで、強さ $\Delta\varepsilon$ だけがイルメナイトの量に比例する。
2. **イルメナイトが少ないと分散しない。** 1 wt% は分散がなく、Cole-Cole の当てはめが収束しない（論文 §4.2）。
3. 10〜20 wt% で全体の形がわずかに違うのは、ホスト（バイトウナイト）の平坦な損失 0.006 の割合が違うためと考えられる。450 MHz の $\varepsilon''$ に占めるホストの分は 19 % / 12 % / 8 %。

### 3.3 2 成分モデル

そこで、450 MHz の $\varepsilon''$ を「平坦な部分」と「緩和の部分」に分ける。

$$
\varepsilon''(f) = (1-w)\,\varepsilon''_A\,\frac{h(f)}{h(f_A)} + w\,\varepsilon''_A\,\frac{g(f)}{g(f_A)}
$$

- $g(f)$：緩和の形。Boivin 20 wt% の Cole-Cole からホストの 0.006 を引き、3 極 Debye で当てはめたもの（`BOIVIN_SUBTRACT_HOST = True`、`N_RELAX_POLES = 3`）。帯域内の RMS 誤差は 0.2 %。2 極では 1.1 % で、450 MHz で割った形に 3 % 程度の誤差が出るため 3 極にした。
- $h(f)$：平坦な部分。最大平坦 2 極 Debye（0.45〜2 GHz）。理想的には $h$ = 一定。

**この 2 成分で Boivin の試料ごとの違いを再現できるかを確かめた**（`summary.txt` の検算 1、図 `boivin_dispersion_check` (b)）。20 wt% から求めた $g$ にホストの割合を変えて足すだけで、10 wt% と 15 wt% の形が 1.0〜2.0 GHz で ±1.7 % 以内で再現できる。

### 3.4 $S$ と $w$ の関係

Fig. 9.54 の式で $\tan\delta$ は $S$ について指数関数なので、$S_{\rm BG}$ を超える分をイルメナイト（緩和）によるものとすると

$$
w(S) = 1 - \frac{\tan\delta_A(S_{\rm BG}, \rho)}{\tan\delta_A(S, \rho)} = 1 - 10^{-A (S - S_{\rm BG})} \qquad (S > S_{\rm BG}), \qquad w = 0 \quad (S \le S_{\rm BG})
$$

$\rho$ の項は打ち消し合うので、$w$ は $S$ だけで決まる。

| `DISP_MODE` | $w$ | 意味 |
|---|---|---|
| `split`（既定） | 上の式 | $S_{\rm BG}$ = 7.5 wt% では現在のモデル（分散なし）と同じになる。$S$ = 20 で $w$ = 0.61 |
| `all` | 1 | 旧コードと同じ。分散の上限 |
| `none` | 0 | 分散なし |

**$S_{\rm BG}$ の決め方：** $S$（FeO+TiO$_2$）にはイルメナイト以外の FeO（輝石・ガラス）も入る。Boivin が変えたのはイルメナイトの量だけなので、緩和を起こすのはイルメナイトの分と考える。イルメナイト（FeTiO$_3$）の FeO+TiO$_2$ は約 100 %（Boivin の試料は Mg を含むので 87 %）。TiO$_2$ がほぼすべてイルメナイトにあるとすると、イルメナイトの量 ≈ TiO$_2$ / 0.526、したがって

$$
S_{\rm BG} \approx S - \frac{\mathrm{TiO_2}}{0.526}
$$

組成（TiO$_2$）が分かる場合はこれで決める。既定の 7.5 wt% は、現在の LUPEX 想定（分散なしとしている組成）と連続にするための値。

### 3.5 Debye 極と $\varepsilon'$

全体は 5 極（緩和 3 ＋ 平坦 2）の Debye になる。

$$
\varepsilon(f) = \varepsilon_\infty + \sum_{k=1}^{5} \frac{\Delta\varepsilon_k}{1 + j\omega\tau_k}
$$

$\varepsilon_\infty$ は、450 MHz で $\varepsilon' = \varepsilon'_A$ になるように決める。

$$
\varepsilon_\infty = \varepsilon'_A - \sum_k \frac{\Delta\varepsilon_k}{1 + (\omega_A \tau_k)^2}
$$

したがって 450 MHz では $\varepsilon'$ と $\varepsilon''$ の両方が経験式に一致する（検算 2 で誤差 0 %）。$\varepsilon'$ の周波数変化は Debye（Kramers-Kronig）が決める。$S$ = 20、深さ 1.5 m では 450 MHz → 2 GHz で −1.5 %。

平坦な部分は 450 MHz で合わせるので、帯域内では最大 +3.3 % × $(1-w)$ 大きくなる（0.95 GHz）。これは最大平坦 2 極で実現するときの誤差で、gprMax で計算しても同じになる。`USE_DEBYE_REALIZATION = False` にすると理想値（周波数によらない一定値）にできる。

`dry_debye_poles_<S>wt_<mode>.csv` と `summary.txt` の最後に、gprMax の書式（`#material` と `#add_dispersion_debye`）で深さごとの値を出す。

---

## 4. 水氷の混合

calc_LLL_mixing_profile.py と同じ LLL の増分形を、周波数ごとに適用する。

| 描像 | $\varepsilon'$ | $\varepsilon''$ |
|---|---|---|
| pore（空隙を埋める） | $\left[(\varepsilon'_{\rm dry})^{1/3} + v\,(\varepsilon_{\rm ice}^{1/3} - 1)\right]^3$ | $\varepsilon''_{\rm dry} + v\,\varepsilon_{\rm ice}\tan\delta_{\rm ice}$ |
| excess（レゴリスごと押しのける） | $\left[(1-v)(\varepsilon'_{\rm dry})^{1/3} + v\,\varepsilon_{\rm ice}^{1/3}\right]^3$ | $(1-v)\,\varepsilon''_{\rm dry} + v\,\varepsilon_{\rm ice}\tan\delta_{\rm ice}$ |

$\varepsilon_{\rm ice}$ = 3.15、$\tan\delta_{\rm ice}$ = 2×10⁻⁵。

---

## 5. LSR で推定される $\tan\delta$

### 5.1 推定式（bscan_detection と同じ）

区間 $z_1 \to z_2$ の往復のパワーの比 $R(f)$ [dB] を周波数 $f$ [GHz] の直線で当てはめ、傾き $s$ [dB/GHz] から

$$
\widehat{\tan\delta} = \frac{-s}{10\log_{10}e \cdot 2\pi\,\Delta t}
$$

$\Delta t$ [ns] は区間の往復走時（入射スペクトルの重心周波数 1.25 GHz での位相走時）。使う周波数は、入射スペクトルが最大から 20 dB 以内の範囲（`LSR_FLOOR_DB`）。

$$
R(f) = -20\log_{10}e \int_{z_1}^{z_2} 2\,\alpha(f, z)\,dz,
\qquad
\alpha = \frac{\omega}{c}\sqrt{\frac{\varepsilon'}{2}}\sqrt{\sqrt{1+\tan^2\delta}-1}
$$

### 5.2 分散があると何が変わるか

$\tan\delta \ll 1$ では $R(f) \approx -10\log_{10}e \cdot 2\pi f\,\tan\delta(f)\,\Delta t$ なので、直線で当てはめた傾きが表すのは $\tan\delta$ そのものではなく

$$
\widehat{\tan\delta} \approx \left\langle \frac{d}{df}\bigl[f\,\tan\delta(f)\bigr] \right\rangle_{\rm band}
= \bigl\langle \tan\delta + f\,\frac{d\tan\delta}{df} \bigr\rangle_{\rm band}
$$

になる。$\tan\delta$ が周波数とともに増えると、推定値は帯域中心の $\tan\delta$ より大きくなる。bscan_detection が比べる理論値（1 GHz の区間平均）とも合わなくなる。

- 分散なし（平坦な部分が理想値）では、推定値と 1 GHz の平均が 0.03 % で一致する（検算 4）。推定式と計算は正しい。
- $S$ = 20、`split` では、推定値は 1 GHz の平均の **1.33 倍**、`all` では 1.47 倍（図 `lsr_vs_composition` (c)）。
- 氷による差 $\Delta\tan\delta$ も同じ割合で大きく見える（図 `lsr_tandelta_profile` (b)）。

---

## 6. 出力

`OUTPUT_BASE` の下に出す。

| ファイル | 中身 |
|---|---|
| `boivin_dispersion_check.png` | (a) Boivin の $\varepsilon''$、ホストの損失、緩和の部分と 3 極 Debye。(b) 450 MHz で割った形と 2 成分モデル（破線） |
| `dry_dispersion_1.5m.png` | 深さ 1.5 m の乾燥レゴリスの (a) $\varepsilon'$ (b) $\varepsilon''$ (c) $\tan\delta$ の周波数特性（$S$ = 7.5 / 15 / 20、3 つの配分）、(d) $w(S)$。丸は経験式の 450 MHz の値 |
| `dry_debye_poles_*.csv`、`summary.txt` | 深さごとの $\varepsilon_\infty$ と Debye 極、検算、gprMax の書式 |
| `pore_ice/`、`excess_ice/` の `profiles_freq.png` | $S$ = 20 の (a) $\varepsilon'$ (b) $\varepsilon''$ (c) $\tan\delta$ (d) $\alpha$ の深さプロファイル（0.5 / 1 / 2 GHz） |
| 同 `profiles_comp.png` | 1 GHz で $S$ = 7.5 / 15 / 20 を比べたもの |
| 同 `lsr_spectra.png` | (a) 乾燥レゴリスのスペクトル比と直線、(b) 直線からのずれ（分散による曲がり） |
| 同 `lsr_tandelta_profile.png` | (a) 区間 0.25 m → 深さの LSR 推定値（実線）と 1 GHz の平均（点線）、(b) 氷による差 |
| 同 `lsr_vs_composition_0.25-2m.png` | $S$ を 0〜25 wt% に振ったときの (a) 乾燥の $\tan\delta$、(b) 氷 10 vol% による差、(c) LSR ÷ 1 GHz の平均 |
| 同 `summary_<描像>.txt`、各 CSV | 数値 |

---

## 7. 主な設定

| 設定 | 既定 | 意味 |
|---|---|---|
| `FEOTIO2_WT` | 20.0 | $S$ [wt%] |
| `S_BG` | 7.5 | 緩和を起こさない FeO+TiO$_2$ [wt%] |
| `DISP_MODE` | `'split'` | 配分（`split` / `all` / `none`） |
| `BOIVIN_FIT_SAMPLE` | `'20wt_ilm'` | 緩和の形を求める試料 |
| `BOIVIN_SUBTRACT_HOST` | True | ホストの損失 0.006 を引いてから当てはめる。False で旧コードと同じ |
| `N_RELAX_POLES` | 3 | 緩和の Debye 極の数 |
| `USE_DEBYE_REALIZATION` | True | 平坦な部分を 2 極 Debye で実現する（gprMax と同じ） |
| `LSR_SEG`, `LSR_ICE_FOR_COMP` | (0.25, 2.0) m、10 vol% | 組成の比較に使う区間と氷量 |
| `MODEL_TAND_FREQ` | 1 GHz | bscan_detection が理論値に使う周波数 |

---

## 8. 限界

- Boivin の測定は室温（約 22 °C）で、バイトウナイト＋イルメナイトの模擬試料。極域の低温（40〜100 K）では緩和の周波数と強さが変わる可能性があるが、データがないので考慮していない。
- 緩和の形はイルメナイト 10〜20 wt% の試料で確かめた範囲。$S$ = 20 は、ilmenite に換算すると 20〜23 wt% 程度で、少し外挿になる。
- 経験式（Fig. 9.54）はアポロ試料の回帰で、ばらつきが大きい。同じ $S$ でも、Boivin の測定の $\tan\delta$ は経験式の約 2 倍（ρ ≈ 1.76）。本ツールは「強さは経験式、周波数の形は Boivin」としている。
- 散乱、層構造、クラッタは入っていない（一様な氷量の理論プロファイル）。
- LSR の走時 $\Delta t$ は位相走時で、データの包絡のピークの時刻差（群遅延）とは 1 % 程度ずれうる。

## 9. 参考文献

- Carrier, W. D. III, Olhoeft, G. R., & Mendell, W. (1991). Physical properties of the lunar surface. In *Lunar Sourcebook*, Cambridge University Press.
- Boivin, A. L. et al. (2022). *Determination of Broadband Complex EM Parameters of Powdered Materials: 2. Ilmenite‐Bearing Lunar Analogue Materials*. https://doi.org/10.1029/2022JE007200
