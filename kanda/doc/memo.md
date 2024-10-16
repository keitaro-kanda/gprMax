# Spatial grid
- 空間グリッドサイズの基準について、特定のガイドラインはない
- 要求される精度・波源の周波数・ターゲットのサイズ次第
    - ターゲットが1 gridとか2 grid程度の大きさしかなかったらちょっとまずいよね
- 他に考慮する必要があるのは数値的に誘発される分散に関する誤差 (the errors associated with numerically induced dispersion：グリッド分散)
    - 現実世界では電磁波が方向や周波数に関係なく同じ速度で伝播する
    - 空間が離散化された空間ではそうもいかん
    - 詳細は[GIA1997]や[KUN1993]を参照
    - この誤差を避けるための基準：
$$
\Delta R 
\leq \frac{\lambda}{10}
= \frac{v}{10 f}
=\frac{c}{10 f \sqrt{\varepsilon_r}}
=\frac{300}{10 f [\mathrm{MHz}] \sqrt{\varepsilon_r}}
$$
- 最も高い誘電率の媒質でもグリッド分散が起きないように空間グリッドを設定する必要がある

参考：https://docs.gprmax.com/en/latest/gprmodelling.html#spatial-discretization


# Su methodの簡約化
Su+(2022)での相互相関イメージング手法（式７）：
$$
I (\vec{r}) = \sum_{i=1}^{N} \sum_{j \neq i}^{N} \sum_{\tau=0}^{\tau_m} \frac{ \omega_i(\vec{r}) \cdot S_i(t_i(\vec{r}) + \tau) \times \omega_j(\vec{r}) \cdot S_j^*(t_j(\vec{r}) + \tau) }{N(N-1) \tau_m}
$$


これを，減衰補正$\omega_k(\vec{r})$とパルス幅$\tau_m$を考慮しない形式で簡約化する．
$$
I (\vec{r}) = \sum_{i=1}^{N} \sum_{j \neq i}^{N} \frac{S_i(t_i(\vec{r})) \times \cdot S_j^*(t_j(\vec{r})) }{N(N-1)} 
$$



# Normal moveout theory
[Castle(1994)]より．

NMO方程式：
$$
t = \sqrt{t_0^2 + \frac{x^2}{V_{RMS}^2}}
$$
ただし，
- $t_0$: 送信新規から地下境界までの垂直往復時間
- $x$: オフセット距離
- $V_{RMS}$:
$$
V_{RMS} = \sqrt{
    \frac{\sum_{k=1}^{N} \Delta \tau_k V_K^2} {\sum_{k=1}^{N} \Delta \tau_k}
    }
$$
- $V_k$: k番目の層内での速度
- $\Delta \tau_k$: k番目の層内での垂直往復時間


# $V_{RMS}$の推定
相互相関関数 $F(V_{RMS}^2, \tau_{ver})$ から，２乗平均速度 $V_{RMS}^2$ とn番目の地下層までの鉛直方向遅れ時間 $\tau_{ver}$を求めることができる：

$$
F(V_{RMS}^2, \tau_{ver}) = \sum_{i \neq j} f_i \Big( \sqrt{\tau_{ver}^2 + \frac{L_i^2}{V_{RMS}^2}} \Big) \cdot f_j \Big( \sqrt{\tau_{ver}^2 + \frac{L_j^2}{V_{RMS}^2}} \Big)
$$

ここで，$f_i$ と $f_j$ は異なるそう受信点の組み合わせで得られたA-scan，$L_i, L_j$ は送受信点間の距離．


# Estimate internal velocity from $t_0$ and $V_{RMS}$
Internal velocity $v_i$ of i-th subsurface layer is estimated by

$$
v_i = \sqrt{
\frac{V_{RMS, i}^2 \cdot t_{0, i} - V_{RMS, i-1}^2 \cdot t_{0, i-1}
}
{t_{0, i} - t_{s0, i-1}}
}
$$

# domain_100x100の$t_0, V_{RMS}$推定

geometry：

- vacuum layer: thickness of 1 m, $\varepsilon_r = 1$
- legolith layer: thickness of 10 m, $\varepsilon_r = 4$
- first basalt layer: thicknes of 50 m, $\varepsilon_r = 6$

## $t_0$
$$
t_0 = \sum_{k=1}^N \frac{2 d_k}{V_k}
$$

where
- $d_k$: thickness of k-th layer
- $V_k$: interval velocity in the k-th layer

### first interface:
$$
t_0 = \frac{1 \cdot 2}{c/ \sqrt{1}} + \frac{10 \cdot 2}{c/ \sqrt{4}} \simeq 140 \ \mathrm{[ns]}
$$

###  second interface:
$$
t_0 = \frac{1 \cdot 2}{c/ \sqrt{1}} + \frac{10 \cdot 2}{c/ \sqrt{4}} + \frac{50 \cdot 2}{c/ \sqrt{6}} \simeq 790 \ \mathrm{[ns]}
$$

## $V_{RMS}$
[Castle, 1994]:

$$
V_{RMS} = \sqrt{
    \frac{\displaystyle \sum_{k=1}^{N} \Delta t_k V_k^2}
    {\displaystyle \sum_{k=1}^{N} \Delta t_k}
}
= \sqrt{
    \frac{\displaystyle \sum_{k=1}^{N} 2 d_k V_k}
    {\displaystyle \sum_{k=1}^{N} \Delta t_k}
}
$$

where
- $V_k$: intercal velocity in the k-th layer
- $\Delta t_k$: vertical travel time in the k-th layer
- $d_k$: thickness of k-th layer

### first interface:
$$
\frac{V_{RMS}}{c} = \sqrt{
     \frac{2 + 20 / \sqrt{4}}
     {140 \mathrm{[ns]} \cdot c}
     }
    \simeq 0.53 \ \mathrm{[/c]}
$$

###  second interface:
$$
\frac{V_{RMS}}{c} = \sqrt{
     \frac{2 + 20 / \sqrt{4} + 80 /{\sqrt{6}}}
     {790 \mathrm{[ns]} \cdot c}
     } \simeq 0.43 \ \mathrm{[/c]}
$$

# $V_{RMS}$と$\tau_{ver}$のオーダー
$\tau_{ver} = 2D / V_{RMS}$（$D$は境界面の深さ）なので，
$$
\begin{split}
\Delta t &= \sqrt{\tau_{ver}^2 + \frac{L_i^2}{V_{RMS}^2}} \\
&= \sqrt{\frac{D^2}{V_{RMS}^2} + \frac{L_i^2}{V_{RMS}^2}} \\
&= \frac{\sqrt{D^2 + L_i^2}}{V_{RMS}}
\end{split}
$$

- $V_{RMS}$：光速$c$の0%~100%→$10^8$くらい
- $\tau_{ver}$：（数十〜数百メートルの地下構造を考えるなら）数百ns→$10^{-7}$から$10^{-6}$くらい
- $L_i$：数メートル→$10^0$から$10^1$くらい

$$
\begin{split}
\Delta t &= \sqrt{\tau_{ver}^2 + \frac{L_i^2}{V_{RMS}^2}} \\
&\simeq \sqrt{ (10^{-7})^2 + \left(\frac{1}{10^{8}} \right)^2 } \\
&\simeq \sqrt{ 10^{-14} + 10^{-16}} \\
&\simeq 10^{-7}
\end{split}
$$

つまり，$L_i$が小さい場合，$\Delta t$はほとんど$\tau_{ver}$に支配されることになる．


# Source wave
ricker wave form:
$$
W(t) = - \Big( 2 \xi (t - \chi)^2 - 1 \Big) e^{-\xi (t - \chi)^2}
$$

where
- $\xi=\pi^2 f^2$
- $\chi=\sqrt{2}/f$
- $f$: frequency [Hz]

パルス幅を解析的に解こうと思ったけど，わからんかった☆
ただ，ピーク時刻は$t=\chi$になるらしいとわかった．

Desmosに突っ込んだ結果，$f=150$MHzの場合，
- start: $t=2.27 $ ns
- end: $t=16.6$ ns
- pulse width: $16.6 - 2.27 \simeq 14.3$ ns
らしい．
plot_source_waveの結果とも整合的な気がする．

