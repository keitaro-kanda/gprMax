# プロファイル見積りコード 改訂設計書

**対象**：`calc_mixing_dispersion_profile.py` の後継
**新ファイル名**：`calc_profile_lll.py`
**作成日**：2026-08-29

---

## 0. この改訂の目的

既存コードは 2 点で現在のモデル設定と食い違っている。

| | 既存 | 改訂後 |
|---|---|---|
| **混合則** | Maxwell-Garnett（母材＝レゴリス ε′≈3、介在物＝氷） | **LLL 増分形**（3 相：粒子・真空・氷。氷は空隙を埋める） |
| **分散の扱い** | 2 極 Debye（450 MHz アンカー、高Ti想定の Cole-Cole フィット） | **ε″ 一定**（`Level_3.in` と同じ最大平坦 2 極 Debye の解析解） |
| 経験式 | Carrier Fig. 9.54（450 MHz DATA） | **Carrier Fig. 9.53（SOILS）** |
| 組成 | FeO+TiO2 = 20 wt% | **7.5 wt%**（南極想定。5/10 も切替可） |
| 帯域 | 0.25–6.0 GHz | **0.5–2.0 GHz**（LUPEX） |

この 2 点の変更で **ε′ の氷依存性が 20 倍以上変わり、結論が変わる**。既存の見積りはそのまま使えない。

---

## 1. 物理モデルの仕様

### 1.1 乾燥レゴリス

密度プロファイル（Carrier et al. 1991）:

```
rho(z) = 1.92 * (z_cm + 12.2) / (z_cm + 18.0)     [g/cm^3]
```

誘電率（Carrier et al. 1991, Lunar Sourcebook Ch.9, **Fig. 9.53 SOILS**）:

```
eps'      = 1.871 ^ rho
tan_delta = 10 ^ (0.027*(%TiO2+%FeO) + 0.273*rho - 3.058)
eps''     = eps' * tan_delta
```

> **Fig. 9.53 を使う理由**：本研究の対象は土壌であり、tanδ の周波数依存を無視する立場をとる以上、周波数で切ったサブセット（Fig. 9.54 = 450 MHz）を選ぶのは仮定と矛盾する。ε′ の式と tanδ の式は必ず同一図から取る。

### 1.2 周波数依存の扱い

**ε″ は帯域内で一定**とする（Boivin+2022）。ただし Kramers-Kronig 則により ε′ は必ず変化するので、`Level_3.in` と同一の **最大平坦 2 極 Debye の解析解**で実現する。

```
f0   = sqrt(f_lo * f_hi) = 1.0 GHz            帯域の幾何平均
tau1 = 1 / (2*pi*f0*(1+sqrt2))
tau2 = (1+sqrt2) / (2*pi*f0)
De   = sqrt(2) * eps''_target                 （各極）
eps_inf = eps'_target - De                    （f0 で eps' = eps'_target）
```

深さごとに `eps'_target = 1.871^rho(z)`、`eps''_target = eps'_target * tan_delta(z)` を与えて上式を適用する。

> **フラグ** `USE_DEBYE_REALIZATION`
> `True`（既定）：上の 2 極 Debye。gprMax が実際に解く媒質と一致する。
> `False`：ε′・ε″ とも周波数に依らない理想モデル。差は ε′ で 0.43%、α で 2.5%（帯域端）。
> 両者を切り替えて差を見られるようにしておく。

### 1.3 水氷の混合（LLL 増分形）

氷は**粒子間の空隙（真空）を埋める**（Takekura+2025 Fig. 2 と同じ描像）。LLL を 3 相に適用し、粒子の項が相殺する形にする。

```
eps'_wet ^(1/3) = eps'_dry ^(1/3) + v_ice * (eps_ice^(1/3) - 1)
eps''_wet       = eps''_dry + v_ice * eps_ice * tan_delta_ice
```

**この形なら粒子密度も粒子誘電率も式に現れない。**乾燥側には Carrier の経験式をそのまま使えるので、経験式との接続が完全に保たれる。

空隙率の上限チェックのみ粒子密度を使う（斜長岩 2.645 g/cm³）:

```
v_ice <= 1 - rho(z) / rho_grain
```

### 1.4 減衰率

```
alpha(f, z) = (omega/c) * sqrt(eps'/2) * sqrt( sqrt(1 + tan_delta^2) - 1 )     [Np/m]
v(f, z)     = c / Re(sqrt(eps_complex))
```

ε″ 一定なので **α ∝ f**（帯域内比 4.000）になるのが Level 3 の特徴。

---

## 2. 入射スペクトル

優先順位:

1. `ASCAN_OUTFILE_PATH` が存在すればそこから FFT で取得（従来どおり）
2. 無ければ **合成スペクトル**にフォールバック
   - 0.5–2.0 GHz で平坦、**Tukey α=0.2** のテーパ
   - 検証済み：この設定で σ_f² = **0.1443 GHz²**、f_c = 1.2500 GHz となり、実際の励振ファイルの実測値と一致する

フォールバックがあるので、SSD が繋がっていない環境でも全図が生成できる。

---

## 3. 伝搬

深さ方向に積分して累積量を作る。

```
A(f, d) = k * ∫_0^d alpha(f, z) dz          k = 2（往復）または 1（片道）
T(f, d) = k * ∫_0^d dz / v(f, z)  + t_air
S(f, d) = S0(f) * exp(-A(f, d))
```

- `PROPAGATION_MODE = 'two_way'`（既定）：地表 tx/rx の反射配置（B 系統）に対応
- `'one_way'`：埋設 rx（A 系統）に対応
- `t_air = k * antenna_height / c`

---

## 4. スペクトル解析量

すべてパワースペクトル `P = |S|²` に対して計算する（`ascan_spectrum.py` と規約を揃える）。

| 量 | 定義 |
|---|---|
| 重心 `f_c` | ∫f·P df / ∫P df |
| 幅 `sigma_f` | sqrt( ∫(f−f_c)²·P df / ∫P df ) |
| 帯域端 `f_lo`, `f_hi` | 帯域内最大に対して −10 dB を横切る周波数 |
| LSR | ln( |S(f,d)| / |S(f,d_ref)| ) = −(A(f,d) − A(f,d_ref)) |

LSR は幾何減衰を含まない純粋な吸収項になる。理論の傾き `dLSR/df = -k*∫(dalpha/df)dz` が α ∝ f を反映して直線になることが確認できる。

> **規約の注意**：LSR は振幅基準、モーメントはパワー基準。感度の恒等式
> `df_c/dt = -2*pi*tan_delta*sigma_f^2` はパワー基準・往復走時で成立する。

---

## 5. 出力する図（すべて PNG と PDF）

### 5.1 プロファイル図（`profile/`）

| ファイル | 内容 |
|---|---|
| `eps_real` | ε′（左）＋ 0 vol% からの相対差（右） |
| `eps_imag` | ε″ 同上 |
| `conductivity` | σ = ε″·ω·ε₀ 同上 |
| `losstangent` | tanδ 同上 |
| **`attenuation`** | **α [Np/m] 同上（新規）** |
| `summary_2x2` | ε′ / ε″ / σ / tanδ の 4 枚まとめ |
| `density_profile` | ρ(z) |
| `ice_wtpct_profile` | 氷の wt%(z)（vol% 一定でも密度が変わるので深さ依存する） |

線種は周波数（0.5 / 1.25 / 2.0 GHz）、色は氷量（0 / 1 / 5 / 10 / 20 vol%）。

### 5.2 スペクトル解析図（`spectrum/`）

| ファイル | 内容 |
|---|---|
| `spectrum_evolution_{N}vol` | 深さごとのスペクトル形状変化（氷量ごとに 1 枚） |
| `band_edges_profile` | f_lo / f_c / f_hi の深さプロファイル |
| `centroid_width_profile` | f_c ± σ_f の深さプロファイル |
| `lsr_{N}vol` | 深さごとの LSR(f)（氷量ごとに 1 枚） |
| `lsr_slope_profile` | LSR の傾きから逆算した α の深さプロファイル（検算用） |

### 5.3 テキスト出力

- `summary.txt`：設定値、各深さの代表値、検算結果
- `profile.csv`：数値の生データ

---

## 6. 既存コードから引き継がないもの

| 機能 | 扱い |
|---|---|
| Hilbert 瞬時周波数解析 | **今回は移植しない**（指示による） |
| STFT 分解能要求解析 | 同上 |
| B-scan からの経験 δf 抽出 | 同上 |
| Maxwell-Garnett | **削除**（LLL に置換） |
| 450 MHz アンカーの Debye | **削除**（ε″ 一定の解析解に置換） |

将来 Hilbert 系を戻す場合に備え、`get_spectral_moments()` は既存と同じ戻り値（f_c, sigma_f, B_eff）を保つ。

---

## 7. 検算項目（実行時に自動で表示）

1. **α の帯域内比が 4.000 になるか**（ε″ 一定の必要条件）
2. **合成スペクトルの σ_f² = 0.1443 GHz²** になるか
3. **f_c の勾配が恒等式 `df_c/dt = -2*pi*tan_delta*sigma_f^2` と一致するか**
4. **v_ice が空隙率を超えていないか**（超えた深さを警告）
5. **LSR の傾きから逆算した α が直接計算した α と一致するか**