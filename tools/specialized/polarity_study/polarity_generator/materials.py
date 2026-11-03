"""
materials.py
============

予想外極性の研究（EPS 再投稿、キャンペーン polarity_v01）の媒質定義と、
帯域内で tanδ を一定にする 2 極 Debye パラメータの設計。

媒質（原稿 2.1 節、表 1）
------------------------
    regolith : eps_r = 3.0, tanδ = 0.004
    rock     : eps_r = 6.0, tanδ = 0.01
    void     : eps_r = 1.0, 損失なし（gprMax 組み込みの free_space）

**媒質の値と Debye パラメータは、このモジュールだけが持つ。**
.in 生成スクリプト・理論値計算・解析コードは、すべてここから import すること
（同じ式や定数を複数の場所に書かない）。

設計（最大平坦条件の 2 極 Debye。別研究 ascan_v02 の materials.py と同じ設計）
----------------------------------------------------------------------------
    f0      = sqrt(f_lo * f_hi)                       帯域の幾何平均
    tau_1   = 1 / (2 pi f0 (1 + sqrt2))
    tau_2   = (1 + sqrt2) / (2 pi f0)
    De      = sqrt(2) * eps_r * tan_delta             （各極）
    eps_inf = eps_r - De

f0 で eps' = eps_r、eps'' = eps_r * tan_delta が厳密に成り立つ。
帯域内では eps'' がほぼ一定。eps' は Kramers-Kronig 則により帯域内で必ず変化するので、
eps_r は「f0 での値」と明記すること。

複素誘電率の符号の約束: eps(w) = eps' - j eps''（時間因子 exp(+j w t)、gprMax と同じ）。

使い方
------
    python materials.py          # 媒質の一覧、Debye パラメータ、帯域内の検算を表示
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List

import numpy as np

# =============================================================================
# [EDIT HERE] 共通定数
# =============================================================================
C0 = 299792458.0                      # 真空中の光速 [m/s]

# tanδ を一定にする帯域（LPR CH2 の公称帯域, Fang et al., 2014）
BAND_LO_HZ = 250e6
BAND_HI_HZ = 750e6

# =============================================================================
# 設計に共通の量（帯域だけで決まる）
# =============================================================================
F0_HZ = math.sqrt(BAND_LO_HZ * BAND_HI_HZ)          # 433.0127 MHz
TAU1_S = 1.0 / (2.0 * math.pi * F0_HZ * (1.0 + math.sqrt(2.0)))
TAU2_S = (1.0 + math.sqrt(2.0)) / (2.0 * math.pi * F0_HZ)


@dataclass(frozen=True)
class Medium:
    """1 つの媒質。eps_r は f0 での比誘電率（実部）。"""
    key: str                         # スクリプト内で使うキー（regolith / rock / void）
    gprmax_name: str                 # gprMax の材料 ID
    eps_r: float                     # f0 での eps'
    tan_delta: float                 # 帯域内で一定にする tanδ
    label: str                       # 説明
    builtin: bool = False            # gprMax 組み込み材料（free_space）なら True
    sources: List[str] = field(default_factory=list)

    # ---- Debye ---------------------------------------------------------------
    @property
    def is_lossy(self) -> bool:
        return self.tan_delta > 0.0

    @property
    def delta_eps(self) -> float:
        """各極の Δε = sqrt(2) * eps'' (f0)"""
        return math.sqrt(2.0) * self.eps_r * self.tan_delta if self.is_lossy else 0.0

    @property
    def eps_inf(self) -> float:
        return self.eps_r - self.delta_eps

    def eps_complex(self, f_hz) -> np.ndarray:
        """eps(f) = eps' - j eps''（2 極 Debye。損失なしなら定数）"""
        f = np.asarray(f_hz, dtype=float)
        if not self.is_lossy:
            return np.full(f.shape, self.eps_r, dtype=complex)
        w = 2.0 * np.pi * f
        de = self.delta_eps
        return self.eps_inf + de / (1.0 + 1j * w * TAU1_S) + de / (1.0 + 1j * w * TAU2_S)

    def alpha_np_per_m(self, f_hz) -> np.ndarray:
        """電場の減衰係数 α [Np/m]（eps(f) から厳密に計算）"""
        f = np.asarray(f_hz, dtype=float)
        eps = self.eps_complex(f)
        kappa = -np.imag(np.sqrt(eps)) + 0.0    # sqrt(eps' - j eps'') = n - j kappa（+0.0 で -0 を消す）
        return 2.0 * np.pi * f * kappa / C0

    # ---- gprMax ----------------------------------------------------------------
    def gprmax_lines(self) -> List[str]:
        """この媒質を定義する gprMax のコマンド行（組み込み材料なら空）"""
        if self.builtin:
            return []
        lines = [f"#material: {self.eps_inf:.6f} 0 1 0 {self.gprmax_name}"]
        if self.is_lossy:
            de = self.delta_eps
            lines.append(
                f"#add_dispersion_debye: 2 {de:.6f} {TAU1_S:.6e} {de:.6f} {TAU2_S:.6e} {self.gprmax_name}"
            )
        return lines

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.update(
            eps_r_reference_frequency_hz=F0_HZ,
            debye=None if not self.is_lossy else dict(
                n_poles=2, eps_inf=self.eps_inf, delta_eps_each_pole=self.delta_eps,
                tau1_s=TAU1_S, tau2_s=TAU2_S,
                band_hz=[BAND_LO_HZ, BAND_HI_HZ], f0_hz=F0_HZ,
                design="two-pole Debye, maximally flat eps'' (s = asinh(1)), symmetric about f0",
            ),
            gprmax_lines=self.gprmax_lines(),
        )
        return d


# =============================================================================
# [EDIT HERE] 媒質の定義（原稿の表 1 と一致させること）
# =============================================================================
REGOLITH = Medium(
    key="regolith", gprmax_name="regolith", eps_r=3.0, tan_delta=0.004,
    label="Background regolith",
    sources=["eps_r = 3: Feng et al. (2022)",
             "tan_delta = 0.004: CE-4 regolith 0.003-0.007 (manuscript cites Li et al., 2020; "
             "ascan_v02 cites Feng et al., 2022) -- unify the citation in the manuscript"],
)

ROCK = Medium(
    key="rock", gprmax_name="rock", eps_r=6.0, tan_delta=0.01,
    label="Rock fragment",
    sources=[
        "eps_r = 6.0: Apollo 14 rocks at 1 MHz: 4.8-7 (Chung, Westphal & Olhoeft, 1972, Proc. LSC 3rd); "
        "K = 1.93^rho gives rho ~ 2.7 g/cm3 (Strangway, Pearce & Olhoeft, 1977, NASA SP-370)",
        "tan_delta = 0.01: within 0.005-0.03 for typical lunar rocks (Strangway et al., 1977)",
    ],
)

VOID = Medium(
    key="void", gprmax_name="free_space", eps_r=1.0, tan_delta=0.0,
    label="Void (vacuum)", builtin=True, sources=["vacuum"],
)

MEDIA: Dict[str, Medium] = {m.key: m for m in (REGOLITH, ROCK, VOID)}


# =============================================================================
# 検算
# =============================================================================
def check_tau_vs_dt(dt_s: float) -> float:
    """gprMax の制約 tau > dt の余裕（tau_1 / dt）を返す。1 以下なら例外。"""
    ratio = TAU1_S / dt_s
    if ratio <= 1.0:
        raise ValueError(f"tau_1 ({TAU1_S:.3e} s) <= dt ({dt_s:.3e} s)")
    return ratio


def band_report(m: Medium, n: int = 2001) -> Dict:
    """帯域内での eps'' と eps' の変化（設計の検算用）"""
    if not m.is_lossy:
        return dict(key=m.key, lossless=True)
    f = np.geomspace(BAND_LO_HZ, BAND_HI_HZ, n)
    eps = m.eps_complex(f)
    epp = -np.imag(eps)
    ep = np.real(eps)
    target = m.eps_r * m.tan_delta
    return dict(
        key=m.key,
        eps_imag_target=target,
        eps_imag_pp_percent=100.0 * (epp.max() - epp.min()) / target,
        eps_real_at_band_edges=[float(ep[0]), float(ep[-1])],
        eps_real_change_percent=100.0 * (ep[0] - ep[-1]) / m.eps_r,
    )


def main() -> None:
    dt_2d = 0.005 / (C0 * math.sqrt(2.0))
    print(f"Band: {BAND_LO_HZ/1e6:.0f}-{BAND_HI_HZ/1e6:.0f} MHz, f0 = {F0_HZ/1e6:.4f} MHz")
    print(f"tau1 = {TAU1_S:.6e} s, tau2 = {TAU2_S:.6e} s, tau1/dt (dx=5 mm, 2D) = {check_tau_vs_dt(dt_2d):.1f}")
    print()
    print(f"{'key':10s} {'eps_r':>7s} {'tanδ':>7s} {'eps_inf':>10s} {'Δε':>10s}  "
          f"{'eps″ p-p':>8s} {'Δeps′':>7s}  α@500MHz")
    for m in MEDIA.values():
        r = band_report(m)
        a = float(m.alpha_np_per_m(500e6))
        pp = f"{r['eps_imag_pp_percent']:6.2f}%" if m.is_lossy else "     -"
        de = f"{r['eps_real_change_percent']:5.2f}%" if m.is_lossy else "    -"
        print(f"{m.key:10s} {m.eps_r:7.3f} {m.tan_delta:7.4f} "
              f"{m.eps_inf:10.6f} {m.delta_eps:10.6f}  {pp:>8s} {de:>7s}  {a:.4f} Np/m")
    print()
    for m in MEDIA.values():
        for line in m.gprmax_lines():
            print(line)


if __name__ == "__main__":
    main()
