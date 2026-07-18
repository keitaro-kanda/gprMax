# GPR Regolith Mixing & Frequency Shift Analysis

本ドキュメントは、月面レゴリスと水氷（Water Ice）の混合媒質中における地中レーダー（GPR）信号の伝搬・周波数シフト（分散・減衰特性）、およびそれに伴う瞬時周波数解析と分解能評価を行うシミュレーションコードの解説書です。実装されている物理モデルおよび信号処理理論の数学的背景を詳述します。

---

## 1. 物理・誘電率モデルの理論

本シミュレーションの根幹は、月面浅層のレゴリス密度勾配と、そこに含まれる水氷の混合による複素誘電率 $\epsilon = \epsilon' - i\epsilon''$ の定式化にあります。

### 1.1 月面レゴリスの密度・ベースライン誘電率 (Heikenの経験則)
月面レゴリスの深さ方向の密度プロファイル $\rho(z)$ [g/cm³] は、アポロ試料等に基づく典型的な関数として以下を仮定しています。
$$\rho(z) = 1.92 \frac{z + 12.2}{z + 18.0}$$
ここで、$z$ は深さ（cm換算）です。
この密度プロファイルに基づき、Heiken et al. (1991) などの経験則から、レゴリス単体のベースラインとなる実効誘電率 $\epsilon'_{reg}$ と損失正接 $\tan\delta_{reg}$ を求めます。
$$\epsilon'_{reg}(z) = 1.843^{\rho(z)}$$
$$\tan\delta_{reg}(z) = 10^{A \cdot C_{\text{FeO}} + B \cdot \rho(z) - C}$$
ここで、$C_{\text{FeO}}$ はイルメナイト等を含むFeOTiO2の質量パーセント（デフォルト: 20 wt%）、$A, B, C$ は経験定数です。

### 1.2 周波数分散 (Debye緩和モデル)
GPR帯域（数百MHz〜数GHz）におけるレゴリスの周波数分散特性を表現するため、2極Debyeモデル（2-pole Debye model）を導入しています。ベースラインの誘電率を基準（Anchor frequency: $450\text{ MHz}$）としてスケーリングし、複素誘電率に周波数依存性を与えます。
$$\epsilon''_{Debye}(\omega) \propto \sum_{k=1}^{2} \Delta\epsilon_k \frac{\omega\tau_k}{1 + (\omega\tau_k)^2}$$
$$\epsilon'_{drop}(\omega) \propto \sum_{k=1}^{2} \Delta\epsilon_k \frac{(\omega\tau_k)^2}{1 + (\omega\tau_k)^2}$$
これにより、高周波側で誘電率実部がわずかに低下し、虚部（損失）が変化する物理的挙動をFDTD等でのシミュレーションと整合する形で表現しています。

### 1.3 水氷の混合 (Maxwell-Garnett則)
水氷とレゴリスの混合媒質の有効複素誘電率 $\epsilon_{mix}$ は、Maxwell-Garnett (MG) 混合則を用いて計算されます。ホスト媒質をレゴリス ($\epsilon_{reg}$)、インクルージョン（包有物）を水氷 ($\epsilon_{ice}$) とし、氷の体積含有率を $f$ とすると、以下のように記述されます。
$$\epsilon_{mix} = \epsilon_{reg} + 3f\epsilon_{reg} \frac{\epsilon_{ice} - \epsilon_{reg}}{\epsilon_{ice} + 2\epsilon_{reg} - f(\epsilon_{ice} - \epsilon_{reg})}$$
この等価媒質近似により、レゴリスの空隙に氷が充填された際のマクロな電磁気的特性の変化を深度ごとに算出します。

---

## 2. 電磁波伝搬とスペクトル重心の推移

### 2.1 減衰定数と位相速度
複素誘電率 $\epsilon_{mix}$ を持つ媒質中を伝搬する平面波の波数は $k = \omega\sqrt{\mu_0\epsilon_{mix}\epsilon_0}$ となります。ここから、減衰定数 $\alpha$ と位相速度 $v$ を導出します。
$$\alpha(\omega, z) = -\frac{\omega}{c} \text{Im}\left(\sqrt{\epsilon_{mix}(z, \omega)}\right)$$
$$v(\omega, z) = \frac{c}{\text{Re}\left(\sqrt{\epsilon_{mix}(z, \omega)}\right)}$$
送信アンテナから深さ $z$ のターゲットで反射し、再び地表へ戻る（Two-way）際の累積減衰と遅延時間は、各深度レイヤーでの積分（プログラム上は離散和）として計算されます。

### 2.2 スペクトル重心周波数 ($f_c$)
高周波成分ほど散乱や誘電損失によって強く減衰するため、地中深くを伝搬した信号のスペクトルは低周波側へシフトします（周波数シフト）。
深さ $z$ でのパワースペクトル $P(f, z) = \vert{}S_0(f) \exp(-2 \int_0^z \alpha dz)\vert{}^2$ を用い、スペクトルの重心（Centroid）$f_c(z)$ を以下で定義します。
$$f_c(z) = \frac{\int f P(f, z) df}{\int P(f, z) df}$$
重心の推移（深度に対するシフトレート $\dot{f}_c$）は、地中の含氷率 $f$ を推定するための強力な指標となります。

---

## 3. Hilbert瞬時周波数解析

周波数領域の重心シフトだけでなく、時間領域の波形から直接周波数変移を読み取るため、解析信号（Analytic Signal）理論に基づく瞬時周波数解析を実装しています。

### 3.1 解析信号と瞬時周波数
実波形 $e(t)$ に対して、Hilbert変換 $\mathcal{H}[e(t)]$ を虚部に持つ解析信号 $z(t)$ を生成します。
$$z(t) = e(t) + i\mathcal{H}[e(t)] = A(t)e^{i\phi(t)}$$
ここで、$A(t)$ は包絡線（Envelope）、$\phi(t)$ は瞬時位相（Unwrapped Phase）です。瞬時周波数 $IF(t)$ は位相の時間微分として定義されます。
$$IF(t) = \frac{1}{2\pi} \frac{d\phi(t)}{dt}$$

### 3.2 特徴量の抽出 ($IF_{peak}$ と $IF_w$)
実装では、GPR反射波のパルス性に着目し、以下の2つの指標で瞬時周波数を代表させています。
1. **$IF_{peak}$**: 包絡線振幅 $A(t)$ が最大となる時刻での瞬時周波数。離散化誤差を抑えるため、包絡線のピーク近傍で放物線近似（サブサンプル補間）を行い、真のピーク位置における $IF$ を算出しています。
2. **$IF_w$ (包絡線二乗重み付き平均)**: パルス波形全体の周波数特性を安定して評価するため、振幅が一定閾値（デフォルト: ピークの10%）を超える区間において、包絡線のエネルギー（$A^2(t)$）で重み付けした平均周波数を計算します。
$$IF_w = \frac{\sum A^2(t) IF(t)}{\sum A^2(t)}$$
この $IF_w$ は、理論的にはスペクトル重心 $f_c$ と良い一致を示します（パルスの対称性が保たれている場合）。

---

## 4. 分解能要求と不確実性評価

周波数シフトを実測データから有意に検出するためのシステム要求・統計的限界を理論化しています。

### 4.1 STFT（短時間フーリエ変換）の限界
周波数シフトレート $\dot{f}$ や深度ごとの重心差 $\Delta f$ をSTFTで観測する際、窓長（サンプリング数 $N_{\text{perseg}}$）は時間分解能 $\Delta z$ と周波数分解能 $\Delta f$ のトレードオフを支配します。
$$\Delta f = \frac{f_s}{N_{\text{perseg}}}, \quad \Delta z = \frac{N_{\text{perseg}} \cdot \Delta t \cdot v}{2}$$
実装では、水氷0%の参照モデルとターゲットモデル（例: 水氷10%）との周波数差を識別するために必要な最小限の $N_{\text{perseg}}$ を理論計算しています。

### 4.2 周波数検出限界 ($\delta IF$)
限られた時間長（空間スケール）$T_{avg}$ における観測では、スペックルノイズやクラッターの影響により、推定される周波数に不確実性（分散）が生じます。信号の帯域幅（実効帯域幅 $B_{eff}$）とスペクトルの標準偏差 $\sigma_{spec}$ を用いて、独立なサンプリング数 $N_{indep} = B_{eff} \cdot T_{avg} \cdot n_{traces}$ を定義し、瞬時周波数の理論的な統計的ばらつき $\delta IF$ を評価します。
$$\delta IF = k_\sigma \frac{\sigma_{spec}}{\sqrt{N_{indep}}}$$
この理論式により、トレースのスタッキング数（$n_{traces}$）や平滑化窓（$T_{avg}$）が、検出限界に与える影響をスイープ解析（`make_resolution_sweep`）しています。また、gprMax等で取得されたB-scanデータ（`Bscan.json`等）を用いて、理論的な $\delta IF$ と、実データからIQR（四分位範囲）ベースで推定された経験的 $\delta IF$ を比較検証する機能も有しています。

---

## 5. ディレクトリ構成と実行環境

### 必須要件
* Python 3.x
* 依存パッケージ: `numpy`, `scipy`, `matplotlib`
* `gprMax` のユーティリティモジュール（`tools.core.outputfiles_merge`）

### 実行方法
ターミナル上でスクリプトを実行します。
```bash
python run_analysis.py
```

※スクリプト内の output_base_dir や ASCAN_OUTFILE_PATH、EMPIRICAL_BSCAN_REGISTRY のパスは、実行環境に合わせて事前に書き換えてください。

# 出力ディレクトリツリー
実行後、指定したベースディレクトリに以下のように結果が出力されます。
output_base_dir/
 ├── summary.txt                 # 全体の媒質定数・代表値のサマリー
 ├── profile/                    # 誘電率、密度、損失等の深度プロファイル (PNG/PDF)
 ├── centroid/                   # 重心周波数(fc)とシフトレートプロファイル
 │    ├── waveform/              # 減衰後スペクトルの深度別比較グラフ
 │    └── STFT_parameter/        # 分解能要求とSTFT窓長のトレードオフ評価グラフ
 ├── Hilbert/                    # 瞬時周波数解析結果
 │    ├── resolution_estimate/   # 理論的・経験的な分解能評価、クラッターバイアス評価
 │    ├── hilbert_summary.txt    # Hilbert IF解析の数値サマリー
 │    └── waveform_examples_*.png # 代表深度における波形と包絡線、IFの時間変化図

物理モデルの改修や、新たな含氷率プロファイルの検討を行う場合は、スクリプト前半の 1. レゴリス誘電モデルの「単一の情報源」 セクション関数（density_profile や heiken_eps_real 等）を修正してください。