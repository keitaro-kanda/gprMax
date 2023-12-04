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