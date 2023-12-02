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