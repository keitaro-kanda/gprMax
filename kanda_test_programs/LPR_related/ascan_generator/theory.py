"""
theory.py
=========

A-scan の理論値（到達時間、反射係数、予想される極性、フレネル半径、減衰量）。

時間はすべて **伝搬時間のみ**（波源波形のピークまでの遅れを含まない）。
シミュレーションの時刻と比べるときは、自由空間の参照計算（ref_freespace）で
直接波のピーク時刻から求めた波源の遅れを足すこと。

到達時間の計算（bistatic）
-------------------------
送信点と受信点は地表から高さ h、間隔 2a（中点の真下に対象）。
水平な層（空気、レゴリス、対象内部）を通る反射波の経路を、スネルの法則
（水平スローネス p が一定）で解く。

    sin(theta_i) = p * v_i,   a = sum h_i tan(theta_i),   T = sum h_i / (v_i cos(theta_i))

往復時間は 2T。対象の上面・下面は水平面で近似する（円形の対象でも、
中点の真下の反射点付近を平面とみなす）。a = 0 のときは鉛直入射（nadir）の値になる。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from materials import C0, Medium   # noqa: E402

DB_PER_NEPER = 20.0 / math.log(10.0)


def layered_two_way_time(layers: Sequence[Tuple[float, float]], half_offset: float) -> float:
    """
    水平な層を通り、最下層の底面で反射して戻る波の往復時間 [s]。

    layers: [(厚さ [m], 比誘電率), ...]（上から順に）
    half_offset: 送受信点間隔の半分 [m]
    """
    hs = [h for h, _ in layers if h > 0]
    vs = [C0 / math.sqrt(e) for h, e in layers if h > 0]
    if not hs:
        return 0.0
    if half_offset <= 0.0:
        return 2.0 * sum(h / v for h, v in zip(hs, vs))

    vmax = max(vs)

    def offset(p: float) -> float:
        return sum(h * (p * v) / math.sqrt(1.0 - (p * v) ** 2) for h, v in zip(hs, vs))

    lo, hi = 0.0, (1.0 - 1e-15) / vmax
    for _ in range(200):                     # 二分法（offset(p) は p について単調増加）
        mid = 0.5 * (lo + hi)
        if offset(mid) < half_offset:
            lo = mid
        else:
            hi = mid
    p = 0.5 * (lo + hi)
    t = sum(h / (v * math.sqrt(1.0 - (p * v) ** 2)) for h, v in zip(hs, vs))
    return 2.0 * t


def reflection_coefficient(eps1: float, eps2: float) -> float:
    """鉛直入射の電場反射係数（媒質 1 から 2 へ）: (sqrt e1 - sqrt e2) / (sqrt e1 + sqrt e2)"""
    a, b = math.sqrt(eps1), math.sqrt(eps2)
    return (a - b) / (a + b)


def flip_polarity(label: str) -> str:
    return "".join({"N": "P", "P": "N"}[ch] for ch in label)


def expected_polarity(r: float, transmitted: str = "NPN") -> str:
    """反射係数の符号から予想される極性（R < 0 なら送信波と逆）"""
    return flip_polarity(transmitted) if r < 0 else transmitted


def fresnel_radius(optical_distance_m: float, wavelength_m: float) -> float:
    """フレネル半径 r = sqrt(lambda * D / 2)（EPS 論文と同じ定義。D はアンテナからの光学距離）"""
    return math.sqrt(wavelength_m * optical_distance_m / 2.0)


def target_theory(*, antenna_height_m: float, half_offset_m: float,
                  depth_top_m: float, thickness_m: float,
                  background: Medium, target: Medium,
                  transmitted_polarity: str = "NPN",
                  attenuation_frequency_hz: float = 500e6,
                  fresnel_wavelength_m: float = C0 / 500e6) -> Dict:
    """
    地表から depth_top_m の深さに上面があり、鉛直方向の厚さ thickness_m の対象に対する理論値。
    """
    e_b, e_t = background.eps_r, target.eps_r
    air = (antenna_height_m, 1.0)
    to_top = [air, (depth_top_m, e_b)]
    to_bottom = to_top + [(thickness_m, e_t)]

    t_top_bi = layered_two_way_time(to_top, half_offset_m)
    t_bot_bi = layered_two_way_time(to_bottom, half_offset_m)
    t_top_0 = layered_two_way_time(to_top, 0.0)
    t_bot_0 = layered_two_way_time(to_bottom, 0.0)

    r_top = reflection_coefficient(e_b, e_t)
    r_bot = reflection_coefficient(e_t, e_b)

    f = attenuation_frequency_hz
    a_b = float(background.alpha_np_per_m(f))
    a_t = float(target.alpha_np_per_m(f))
    att_top = 2.0 * depth_top_m * a_b * DB_PER_NEPER
    att_bot = att_top + 2.0 * thickness_m * a_t * DB_PER_NEPER

    d_top = antenna_height_m + depth_top_m * math.sqrt(e_b)
    d_bot = d_top + thickness_m * math.sqrt(e_t)

    return dict(
        depth_top_m=depth_top_m,
        thickness_m=thickness_m,
        top=dict(
            t_bistatic_s=t_top_bi, t_nadir_s=t_top_0,
            reflection_coefficient=r_top,
            expected_polarity=expected_polarity(r_top, transmitted_polarity),
            fresnel_radius_m=fresnel_radius(d_top, fresnel_wavelength_m),
            two_way_attenuation_db=att_top,
        ),
        bottom=dict(
            t_bistatic_s=t_bot_bi, t_nadir_s=t_bot_0,
            reflection_coefficient=r_bot,
            expected_polarity=expected_polarity(r_bot, transmitted_polarity),
            fresnel_radius_m=fresnel_radius(d_bot, fresnel_wavelength_m),
            two_way_attenuation_db=att_bot,
        ),
        dt_bottom_minus_top_bistatic_s=t_bot_bi - t_top_bi,
        dt_bottom_minus_top_nadir_s=t_bot_0 - t_top_0,
    )


def common_arrivals(*, antenna_height_m: float, half_offset_m: float) -> Dict:
    """直接波と地表反射の伝搬時間（対象によらない）"""
    return dict(
        direct_wave_s=2.0 * half_offset_m / C0,
        surface_reflection_s=layered_two_way_time([(antenna_height_m, 1.0)], half_offset_m),
    )


def size_from_dt(dt_s: float, eps_assumed: float) -> float:
    """上端・下端エコーの時間差からのサイズ推定 D = c dt / (2 sqrt(eps))（データ解析と同じ式）"""
    return C0 * dt_s / (2.0 * math.sqrt(eps_assumed))
