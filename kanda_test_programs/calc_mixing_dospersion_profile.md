# GPR Regolith Mixing & Frequency Shift Analysis

本ドキュメントは、月面レゴリスと水氷の混合媒質中における GPR 信号の伝搬・周波数シフト（分散・減衰特性）の解析的見積もりと、Hilbert 瞬時周波数解析、および分解能評価（理論・経験）を行うシミュレーション解析コードの解説書である。実装されている物理モデルと信号処理理論の数学的背景、全出力プロットの読み方、検証機能までを詳述する。

---

## 1. 物理・誘電率モデルの理論

本コードの根幹は、月面浅層のレゴリス密度勾配と水氷混合による複素誘電率 $\epsilon = \epsilon' - i\epsilon''$（時間規約 $e^{+i\omega t}$、損失は虚部が負）の定式化にある。物理モデルはコード前半の「単一の情報源」セクション（`density_profile`, `heiken_eps_real`, `heiken_tan_delta`, `debye_*`, `maxwell_garnett`）に集約されており、シミュレーション（gprMax の .in 生成側）と**同一の式・同一の定数**であることが解析の前提となる。

### 1.1 密度プロファイル（Carrier / Heiken の経験則）

深さ方向の密度 $\rho(z)$ [g/cm³] はアポロ試料に基づく経験式を仮定する（$z$ は cm）。

$$\rho(z) = 1.92\,\frac{z + 12.2}{z + 18.0}$$

表層で約 1.3 g/cm³、深部で 1.92 g/cm³ に漸近する。コードではメートル入力を内部で cm に換算する（**単位を混同すると誘電率・損失プロファイルがほぼ一定になる**という既知の不具合モードがあるため、改修時は必ず換算を確認すること）。

### 1.2 ベースライン誘電特性（Heiken の経験則）

密度から、アンカー周波数におけるレゴリス単体の実効誘電率と損失正接を求める。

$$\epsilon'_{reg}(z) = a^{\rho(z)}, \qquad \tan\delta_{reg}(z) = 10^{\,A\cdot C_{\mathrm{TiO_2+FeO}} + B\cdot\rho(z) - C}$$

$C_{\mathrm{TiO_2+FeO}}$ はチタン鉄鉱等の質量パーセント（デフォルト 20 wt%）。底 $a$ と係数 $A, B, C$ の実装値は `heiken_eps_real` / `heiken_tan_delta` を単一の情報源とする。**注意**：Heiken 系の経験式には複数の変種が文献に存在するため、gprMax の .in 生成スクリプト側と本コード側で同一の定数を使っていることを、モデル改修のたびに照合すること（過去に両者で別変種が混在していた事例あり）。

### 1.3 周波数分散（2極 Debye 緩和モデル、450 MHz アンカー）

GPR 帯域におけるレゴリスの周波数分散は 2 極 Debye モデルで表現する。

$$\epsilon''_{Debye}(\omega) = \sum_{k=1}^{2} \Delta\epsilon_k\,\frac{\omega\tau_k}{1 + (\omega\tau_k)^2}, \qquad
\Delta\epsilon'_{drop}(\omega) = \sum_{k=1}^{2} \Delta\epsilon_k\,\frac{(\omega\tau_k)^2}{1 + (\omega\tau_k)^2}$$

実装値：$\tau_1 = 46.212$ ps（緩和周波数 $\approx 3.44$ GHz）、$\tau_2 = 282.195$ ps（$\approx 564$ MHz）、$\Delta\epsilon_1 : \Delta\epsilon_2 \approx 0.748 : 0.252$。総量 $\sum\Delta\epsilon_k$ は、**アンカー周波数 $f_{anchor} = 450$ MHz において $\tan\delta$ が §1.2 の Heiken 値と一致するようにスケーリング**される。

**重要な帰結**：緩和極の一つが 3.44 GHz にあるため、$\tan\delta(f)$ はアンカーから解析帯域（実効 1〜2 GHz）に向かって増加し、**帯域内の実効 $\tan\delta$ はアンカー値の約 1.5〜2 倍**になる。「450 MHz でアンカーした値」と「帯域内実効値」を混同すると、観測との比較で見かけ上 2 倍の食い違いが生じるので注意（スペクトル比法・LSR での比較時に実際に確認された事項）。

なお、一定導電率 $\sigma$ のみの非分散媒質では減衰係数が周波数に依存せず（$\alpha \approx (\sigma/2)\sqrt{\mu_0/\epsilon_0\epsilon'}$）、**周波数シフトは原理的に発生しない**。周波数シフトを生むのは「損失があること」ではなく「損失が周波数とともに増えること」であり、それを因果的に（Kramers–Kronig を満たして）実現する手段がこの Debye 分散である。

### 1.4 水氷の混合（Maxwell-Garnett 則）

混合媒質の有効複素誘電率は、レゴリスを母材、水氷（$\tan\delta \sim 10^{-5}$ でほぼ無損失）を介在物とする 2 相 Maxwell-Garnett 近似で計算する。体積含有率 $f$ に対して

$$\epsilon_{mix} = \epsilon_{reg} + 3f\,\epsilon_{reg}\,
\frac{\epsilon_{ice} - \epsilon_{reg}}{\epsilon_{ice} + 2\epsilon_{reg} - f(\epsilon_{ice} - \epsilon_{reg})}$$

これは「母材の一部体積を氷介在物で置き換えた」2 相モデルであり、空隙充填（レゴリス＋空隙＋氷の 3 相）モデルではない点に注意。氷の効果は主に (i) $\epsilon''$ の希釈（損失低下）、(ii) $\epsilon'$ のわずかな変化、として現れる。

---

## 2. 電磁波伝搬とスペクトル重心

### 2.1 減衰定数と位相速度

複素比誘電率 $\epsilon_{mix}(z, \omega)$ の媒質中の平面波（波数 $k = (\omega/c)\sqrt{\epsilon_{mix}}$）について、

$$\alpha(\omega, z) = -\frac{\omega}{c}\,\mathrm{Im}\!\left(\sqrt{\epsilon_{mix}}\right) \;[\mathrm{Np/m}], \qquad
v(\omega, z) = \frac{c}{\mathrm{Re}\!\left(\sqrt{\epsilon_{mix}}\right)}$$

（$\epsilon = \epsilon' - i\epsilon''$ の規約では $\mathrm{Im}\sqrt{\epsilon} < 0$ なので先頭の負号で $\alpha > 0$。この符号規約はコード全体で統一されており変更禁止。）深さ方向の累積量（往復）

$$\mathrm{cum\_att}(\omega, d) = \int_0^d \alpha(\omega, z)\,dz, \qquad
\tau(\omega, d) = \int_0^d \frac{2\,dz}{v(\omega, z)}$$

は氷含有量ごとに 1 回だけ計算され、伝搬テーブル（`get_propagation_table`）としてキャッシュされる。重心計算・スペクトル比較・Hilbert 波形合成・分解能評価はすべてこのテーブルを共用する。

### 2.2 時刻オフセット

シミュレーション時刻と「深さ 0.1 m（受信点深度）からの遅延時間」の対応付けとして、
$$t_{offset} = t_{lag} + t_{air} + t_{ground}(0.1\,\mathrm{m})$$
（システムラグ 0.837 ns、アンテナ高 0.35 m の空中往復、rx 深度までの地中往復）を用いる。

### 2.3 スペクトル重心とシフトレート

減衰後のパワースペクトル $P(f, d) = |S_0(f)\,e^{-2\,\mathrm{cum\_att}(f,d)}|^2$（$S_0$ は A-scan 実測の入射スペクトル、帯域 0.25–6 GHz）から、重心を

$$f_c(d) = \frac{\int f\,P(f, d)\,df}{\int P(f, d)\,df}$$

で定義する。誘電損失の周波数依存性（$\alpha \propto f\tan\delta$ 型）により高周波が選択的に減衰し、$f_c$ は深さとともに単調に低下する。シフトレートは**遅延時間に対する微分** $\dot f_c = df_c/dt$ [GHz/ns]（0.1 ns の一様時間格子上で `np.gradient`）として計算し、深さ軸にマップして表示する。ガウス形スペクトル・constant-Q 近似では $\dot f_c \approx -\pi\sigma_f^2\tan\delta$（Quan & Harris 1997 型）となり、深部でスペクトル分散 $\sigma_f^2$ が収縮するとシフトレートの氷含有量依存性も消失する（深部検出が難しい本質的理由）。

なお本解析モデルは誘電損失のみを含む。**散乱（ランダム媒質による周波数依存の後方散乱・散乱減衰）は FDTD シミュレーション側にのみ存在**し、理論と実データの残差を解釈する際はこの差を考慮する（§5.5、§7 参照）。

---

## 3. Hilbert 瞬時周波数解析

### 3.1 時間波形のフォワード合成（位相を含む伝達関数）

瞬時周波数は時間領域の量なので、まず各深さのエコー波形を合成する。減衰（振幅）だけでなく**分散による伝搬位相**を含む複素伝達関数を用いる点が本質である：

$$H(\omega, d) = \exp\!\big(-2\,\mathrm{cum\_att}(\omega, d)\big)\cdot
\exp\!\big(-i\,\omega\,[\tau(\omega, d) + t_{offset}]\big)$$

$$e_d(t) = \mathrm{irfft}\big(S_0^{full}(\omega)\,H(\omega, d)\big)$$

実装上の要点：

- **全帯域の複素入射スペクトル**を用いる（0.25–6 GHz のハードマスクは逆変換でリンギングを生むため使わない）。帯域外は緩いコサインテーパで抑制（`HILBERT_TAPER_ON`）。
- `irfft` は `HILBERT_PAD_FACTOR`（デフォルト 2）倍のゼロパディングで時間格子を細かくする（情報は増えないが、ピーク検出の量子化を軽減）。
- 位相符号は numpy 規約（forward が $e^{-2\pi i f t}$）に整合させており、遅延 $t_0$ はスペクトルに $e^{-2\pi i f t_0}$ を掛けることに対応する。

### 3.2 解析信号と瞬時周波数

$$z(t) = e_d(t) + i\,\mathcal{H}[e_d](t) = A(t)\,e^{i\phi(t)}, \qquad
IF(t) = \frac{1}{2\pi}\frac{d\phi}{dt}$$

$A(t)$ が包絡線、$\phi(t)$ はアンラップ位相。包絡線がピークの `HILBERT_ENV_THRESHOLD`（デフォルト 10%）未満の区間では IF は発散的に振動するため無効とし、アンラップは有効区間内で行う。

### 3.3 代表値：$IF_{peak}$ と $IF_w$

1. **$IF_{peak}$**：包絡線最大時刻での IF。離散 argmax の量子化（シフトレートに鋸歯状ノイズを生む）を避けるため、ピーク前後 3 点の放物線補間

$$\delta = \frac{y_0 - y_2}{2\,(y_0 - 2y_1 + y_2)}, \qquad t_{peak} = t_m + \delta\,\Delta t_{pad}$$

でサブサンプル位置を求め、IF もその時刻に線形補間して読む（端点・退化・$|\delta| > 0.5$ ではフォールバック）。遅延時刻もこの $t_{peak}$ で定義する。

2. **$IF_w$**：有効区間の包絡線 2 乗重み付き平均

$$IF_w = \frac{\int A^2(t)\,IF(t)\,dt}{\int A^2(t)\,dt}$$

**$IF_w$ とスペクトル重心 $f_c$ の一致は厳密な恒等式**である（解析信号の 1 次モーメントの性質：$\int A^2\,IF\,dt / \int A^2 dt = \int f\,|Z(f)|^2 df / \int |Z|^2 df$）。パルスの対称性は不要で、チャープした非対称パルスでも成立する。実装上のずれ（≲1%）は、(i) 10% 閾値による裾の切断、(ii) 帯域処理の差（テーパ付き全帯域 vs マスク帯域）のみに由来し、これが `hilbert_vs_centroid_check` による実装検証の原理になっている。

### 3.4 $IF_{peak}$ と $IF_w$ の系統差（パルス内チャープ）

エコーは内部チャープ構造を持つ：分散により高周波成分が波束前方に寄り、減衰スペクトルの非対称性が低周波の長い尾を作るため、**IF(t) はピーク付近で最高、裾で低下**する。その結果 $IF_{peak}$ は $IF_w$（＝重心）より系統的に約 9–11% 高い。両者はどちらも正しい代表周波数であり定義が異なるだけだが、**実データ解析と比較する際は同じ定義同士で比較する**こと（ピーク追跡型なら $IF_{peak}$、窓内エネルギー加重型なら $IF_w$ の理論線と比較。混同すると約 0.1 GHz の見かけの系統差が出る）。理論比較の主系列は、重心の理論枠組み（Quan & Harris 等）と直結する $IF_w$ を推奨する。

---

## 4. STFT の分解能要求

STFT で重心・シフトレートを観測する場合、窓長 $N_{perseg}$ は周波数ビン幅と深さ窓のトレードオフを支配する：

$$\Delta f = \frac{f_s}{N_{perseg}}, \qquad \Delta z = \frac{N_{perseg}\cdot\Delta t\cdot v}{2}, \qquad
\Delta f\cdot\Delta z = \frac{v}{2}\ (\text{不変量})$$

実装では、水氷 0% と各含有率の重心（およびシフトレート）差プロファイルに対し、検出マージン `DETECT_MARGIN`（= 2、差の 1/2 をビン幅が下回ること）を課して要求 $N_{perseg}$ を深さごとに逆算する。シフトレート側の分解能は誤差伝播 $\Delta\dot f = 2\sqrt{2}\,(f_s/N_{perseg})^2$（hop = $N_{perseg}/4$、中心差分）を用いる。

**解釈上の注意**：この要求はビン幅を測定限界と見なす保守的な枠組みである。重心はパワー加重平均であり**ビン幅に量子化されない**（サブビン精度を持つ）ため、実際の検出限界は §5 の統計的分解能で決まる。STFT 要求図は「ビン幅論法ではどれだけ厳しいか」を示す参照であり、検出可能性の最終判定には §5 を用いること。また $N_{perseg}$ は記録長（4241 サンプル）を超えられないため、探索範囲は 16–4096 に制限している。

---

## 5. Hilbert 法の分解能（統計的検出限界）

### 5.1 2 つのレジーム

Hilbert IF にはビン量子化が存在せず、分解能は統計で決まる。(a) **孤立エコー（SNR 律速）**：単一波束の周波数推定は Cramér–Rao 限界 $\delta f \propto 1/(\rho\,T^{3/2})$ に従い、シミュレーションの SNR では µHz〜MHz 級で実質無視できる。(b) **コーダ／スペックル（本解析の対象）**：分布散乱場では各時刻の IF がスペクトル幅程度ばらつき、平均化された独立標本数で精度が決まる。本コードの理論はレジーム (b) を定式化する。

### 5.2 理論 δIF

深さ $d$ の減衰パワースペクトル $P(f, d)$ から

$$\sigma_{spec}(d) = \sqrt{\frac{\int (f - f_c)^2 P\,df}{\int P\,df}}\ \ (\text{RMS スペクトル幅}), \qquad
B_{eff}(d) = \frac{\big(\int P\,df\big)^2}{\int P^2\,df}\ \ (\text{実効帯域幅})$$

を定義し、平滑化時間長 $T_{avg}$・トレース数 $n_{traces}$ に対して

$$\delta IF(d) = \frac{k_\sigma\,\sigma_{spec}(d)}{\sqrt{\max\!\big(k_B\,B_{eff}(d)\cdot T_{avg},\ 1\big)\cdot n_{traces}}}$$

（$k_\sigma, k_B$ はモデル不確かさの感度スイープ用スケール係数、デフォルト 1。下限クリップは窓内独立標本が 1 個未満になる非物理を防ぐ。$T_{avg}$ は秒に換算して掛けること。）

**STFT との本質的な違い**：空間分解能 $\Delta z = T_{avg}\cdot v/2$ との交換が $\delta IF \propto 1/\sqrt{T_{avg}}$ の**平方根則**であり、STFT のビン幅論法（$\Delta f\cdot\Delta z$ 固定の線形トレード）より緩い。この差が、同じ検出目標に対する要求空間分解能を約 1 桁改善する（STFT 要求 $\Delta z \sim$ 数 m に対し Hilbert 要求 $\Delta z \sim$ 0.5–1 m）。

### 5.3 検出要求の逆解き

シグナルを $\Delta IF(d) = |IF_w^{(0\%)}(d) - IF_w^{(f_{ice})}(d)|$ とし、検出条件 $\delta IF \le \Delta IF / \mathrm{margin}$ を $T_{avg}$ について解くと

$$T_{avg}^{req}(d) = \frac{\sigma_{spec}^2}{B_{eff}\cdot n_{traces}\cdot(\Delta IF/\mathrm{margin})^2}, \qquad
\Delta z^{req}(d) = \frac{T_{avg}^{req}\cdot v(d)}{2}$$

### 5.4 経験 δf（B-scan からの実測）

理論はスペックル統計の予測なので、実 B-scan からの経験値と比較検証する。`EMPIRICAL_BSCAN_REGISTRY` に登録された各 B-scan（rand_amp 別）について：

- 各トレースに Hilbert 変換を適用し、長さ $T_{avg}$（hop = 窓長/2）のスライディング窓で**窓付き $IF_w$** をトレース別に計算
- **`df_single`** = トレース間の頑健標準偏差 $\mathrm{IQR}/1.349$（理論の $n_{traces}=1$ 曲線と比較する量）
- **`df_profile`** = `df_single` $/\sqrt{n_{traces}}$（プロファイル中央値の精度、理論の $n_{traces}=$ 全数曲線と比較）
- 前処理 2 系統：raw と平均トレース除去（コヒーレント成分の分離用）
- 結果は `empirical_df_{label}.csv` にキャッシュされ、スキーマが古い場合は自動再計算される。レジストリへの辞書 1 行追加で rand_amp のケースを増やせる。

### 5.5 バイアス評価（IQR の盲点とコヒーレントクラッタ）

**トレース間 IQR は非コヒーレントなばらつきしか測れない**。2D 線源の wake のような全トレース共通のコヒーレント成分は、分散ではなく**バイアス**として効く。そこで「窓付き $IF_w$ のトレース中央値プロファイル」と「解析理論 $IF_w$ プロファイル」の**差**をバイアスとして別途測定する（`empirical_bias_check`）。rand_amp = 0 の B-scan はスペックルがゼロのネガティブコントロールであり、wake 単体の見かけの周波数挙動を分離できる。なお rand_amp = 0 に対する平均トレース除去残差は数値ノイズであり、経験 δf・バイアスとも解釈対象外とする（図からも自動的に除外される）。

---

## 6. 出力カタログと各プロットの読み方

出力はベースディレクトリ以下に生成される（png と pdf の両形式）。

```
output_base_dir/
 ├── summary_2x2 図・summary.txt        # 媒質定数・代表値の総覧
 ├── profile/                           # ε′, ε″, σ, tanδ の深度プロファイル（各: 全氷含有量の重ね描き＋0%との差）
 │                                      # 密度プロファイル・氷 wt% プロファイル
 ├── centroid/
 │    ├── 重心周波数プロファイル図      # f_c(z)。氷含有量で色分け
 │    ├── シフトレートプロファイル図    # df_c/dt [GHz/ns] vs 深さ
 │    ├── waveform/                     # 深度別の減衰後スペクトル比較（正規化 dB）
 │    └── STFT_parameter/               # STFT 要求解析一式
 │         ├── 要求 nperseg プロファイル（centroid / shiftrate）
 │         ├── nperseg vs Δf・Δz のトレードオフ図
 │         └── stft_summary.txt
 └── Hilbert/
      ├── if_w_profile                  # IF_w(z)（＝重心と恒等）。主系列
      ├── if_peak_profile               # IF_peak(z)（サブサンプル補間版）
      ├── if_w_shiftrate_profile        # IF_w のシフトレート
      ├── if_peak_shiftrate_profile     # IF_peak のシフトレート
      ├── hilbert_vs_centroid_check     # 恒等式検証図（実装の健全性確認）
      ├── waveform_examples_{c}vol      # 代表深度の合成波形・包絡線・IF(t)
      ├── hilbert_summary.txt
      └── resolution_estimate/
           ├── inputs_sigma_spec_profile / inputs_beff_profile   # モデル入力の確認
           ├── sweep_sigma_scale / sweep_beff_scale / sweep_tavg / sweep_ntraces
           ├── requirement_overlay        # シグナル vs δIF（交差深さ＝検出限界）
           ├── required_tavg_and_dz       # 要求 T_avg と対応 Δz
           ├── fixed_tavg_dz_profile      # 固定 T_avg (requirement_overlay と同じ3値) の Δz(d) プロファイル
           ├── empirical_vs_theory        # 理論 δIF と経験 δf の比較
           ├── empirical_bias_check       # IF_w バイアス（コヒーレントクラッタ定量）
           └── empirical_df_{label}.csv   # 経験値のキャッシュ
```

主要な図の読み方：

- **if_w_profile / if_peak_profile**：氷含有量が上がるほど曲線が右（高周波側）へずれる。これは氷による損失希釈で高周波の生存率が上がるため。0% と 20% の分離は 3 m で約 0.1 GHz。IF_peak は IF_w より全体に約 10% 高い（§3.4）。
- **hilbert_vs_centroid_check**：実線（IF_w）と破線（重心）が重なっていれば位相構成・逆 FFT・アンラップの実装が正しい。深部・高含氷での微小なずれは帯域処理差による想定内の挙動。
- **waveform_examples**：右パネルは 1 つのエコー内部の IF(t)（山形＝チャープ構造）。この内部チャープの傾き（〜0.4 GHz/ns）と、深さ方向のシフトレート（〜0.02–0.04 GHz/ns）は**別の量**であり混同しないこと。
- **inputs_sigma_spec / inputs_beff**：どちらも深さとともに減少（スペクトルの低周波化・狭帯域化）。σ_spec の収縮は重心・シフトレートの深部感度喪失の原因でもある。
- **sweep_***：色＝氷含有量、線種＝スイープ値。k_σ は δIF に比例、k_B・T_avg・n_traces は 1/√ で効くことが確認できる。
- **requirement_overlay**：色付き実線（シグナル）が灰色線（δIF）を上回る深さから検出可能。おおよそ 20 vol% は 0.5 m 以深、10 vol% は 0.9 m 以深、5 vol% は T_avg = 10 ns で 1 m 以深、1 vol% は全深度で不可（n_traces = 56 時）。
- **required_tavg_and_dz**：同じ結論の逆表現。10 vol% で要求 Δz ≈ 0.5–1 m と、STFT のビン幅論法（数 m〜10 m）より約 1 桁緩い。
- **fixed_tavg_dz_profile**：`requirement_overlay` と同じ $T_{avg}$ = 1, 3, 10 ns を選んだ場合の**空間コスト** $\Delta z = T_{avg}\cdot v(d)/2$ を深さごとに示す（`required_tavg_and_dz` と同一の局所位相速度 `get_local_velocity_profile` を使用）。色＝氷含有量（0, 1, 5, 10, 20 vol% すべて）、線種＝$T_{avg}$。$v(d)$ は深部ほどわずかに低下する（誘電率増加）ため各曲線は深部でわずかに左（小さい Δz 側）へ傾き、氷含有量が高いほど $v$ がわずかに速いため曲線は右寄りになる。灰色の点線は水氷層のノミナル厚 0.5 m の参照線。`requirement_overlay` が「その T_avg で検出できるか」を示すのに対し、本図は「その T_avg を選んだときに深さ方向でどれだけボケるか」を直接可視化する。
- **empirical_vs_theory**：rand_amp > 0 の raw/single 曲線が理論 δIF($n=1$) に一致していれば、スペックル統計モデルがスケール係数 ≈ 1 で妥当という検証になる（実測で確認済み）。
- **empirical_bias_check**：右パネルのバイアスが raw と mean-sub でほぼ同じなら、バイアスの主因はコヒーレント成分（wake）**ではない**ことを意味する（実測で確認済み。残差の有力候補は後方散乱断面積の周波数依存性 σ_s(f)）。rand_amp = 0 曲線は wake 単体が描く「見かけの周波数降下」を示すネガティブコントロール。浅部（地表反射が窓に入る領域）は解釈対象外。

---

## 7. 検証機能（run_verifications）

コードは実行時に以下の自己検証を行い、結果を print する。

1. **サブサンプル補間の効果**：0 vol% の IF_peak シフトレートについて、深さ 1.5–3.0 m の隣接差分 std が補間なし版の 1/5 以下に低減していること（量子化鋸歯の除去確認）。
2. **IF_w 不変性**：サブサンプル補間の ON/OFF で IF_w が完全一致すること。
3. **恒等式チェック**:各深さで IF_w と重心の相対差が ~1% 以内であること（§3.3）。
4. その他：δIF の次元整合、経験 CSV のスキーマ検査（列不足時の自動再計算）、rand_amp = 0 エントリのバイアス統計（深さ 1.0–2.5 m の中央値・最大値）の出力。

---

## 8. 主要パラメータ（コード冒頭の定数ブロック）

| 区分 | 定数 | デフォルト | 意味 |
|---|---|---|---|
| モデル | `ICE_CONTENTS` | 0, 1, 5, 10, 20 vol% | 氷含有量ケース |
| モデル | `f_anchor` | 450 MHz | Debye スケーリングのアンカー周波数（.in 側と一致必須） |
| 幾何 | アンテナ高 / ラグ / rx 深度 | 0.35 m / 0.837 ns / 0.1 m | 時刻オフセット構成要素 |
| 帯域 | freq_min–freq_max | 0.25–6 GHz | 重心・分解能評価の帯域 |
| Hilbert | `HILBERT_ENV_THRESHOLD` | 0.10 | IF 有効区間の包絡線閾値 |
| Hilbert | `HILBERT_PAD_FACTOR` | 2 | irfft ゼロパディング倍率 |
| STFT | `NPERSEG_RANGE` | 16–4096 | 記録長 4241 に制限した探索範囲 |
| STFT/分解能 | `DETECT_MARGIN` | 2 | 検出マージン（差の 1/2） |
| 分解能 | `RES_TAVG_LIST` / `RES_TAVG_DEFAULT` | 1, 3, 10 / 3 ns | 平滑化時間長 |
| 分解能 | `RES_SIGMA_SCALES` / `RES_BEFF_SCALES` | 0.5, 1, 2 | 感度スイープ係数 |
| 分解能 | `RES_NTRACES_LIST` / `RES_NTRACES_DEFAULT` | 1, 14, 56 / 56 | トレース平均数 |
| 経験 δf | `EMPIRICAL_BSCAN_REGISTRY` | — | label / rand_amp / Bscan.json パスの辞書リスト |
| 経験 δf | `EMPIRICAL_MEAN_REMOVAL` | True | 平均トレース除去系列の併算 |

---

## 9. 実行環境と実行方法

- Python 3.x、依存：`numpy`, `scipy`, `matplotlib`, `pandas`
- gprMax のユーティリティ（`tools.core.outputfiles_merge` の `get_output_data`）：経験 δf 機能で B-scan の .out を読むために必要
- 実行：`python calc_mixing_dispersion_profile.py`（スクリプト実体のファイル名に合わせること）
- 事前に `output_base_dir`、入射波 A-scan のパス（`ASCAN_OUTFILE_PATH`）、`EMPIRICAL_BSCAN_REGISTRY` の各パスを環境に合わせて編集する。レジストリが空でも解析本体は完走する（経験比較のみスキップ）。
- 物理モデルの改修は、コード前半の「単一の情報源」セクション（`density_profile`, `heiken_*`, `debye_*`, `maxwell_garnett`）のみで行い、**gprMax の .in 生成側と定数・式が一致していることを必ず照合する**こと。

---

## 10. 既知の知見メモ（2026-07 時点、データ更新で変わり得る）

- 帯域内実効 $\tan\delta$ はアンカー値の約 1.5–2 倍（§1.3）。理論比較は「同一推定器を理論スペクトルに通す」方式で行うこと。
- $IF_{peak} \approx 1.1 \times IF_w$（パルス内チャープ由来、§3.4）。
- スペックル統計モデル $\delta IF = \sigma_{spec}/\sqrt{B_{eff} T_{avg} n_{traces}}$ は rand_amp = 0.01 の B-scan でスケール係数 ≈ 1 で実証済み。
- IF_w バイアス（−0.05〜−0.15 GHz）の主因は wake ではない（平均トレース除去で不変）。有力候補は後方散乱断面積 σ_s(f) の低域通過性で、Born 近似による理論側補正が今後の課題。
- 検出限界（n_traces = 56）：20 vol% ≳ 0.5 m、10 vol% ≳ 0.9 m、5 vol% ≳ 1 m（T_avg = 10 ns）、1 vol% は不可。STFT ビン幅論法より要求空間分解能で約 1 桁有利。