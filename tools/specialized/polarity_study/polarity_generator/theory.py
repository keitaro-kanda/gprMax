"""
theory.py
=========

A-scan の理論値（直下のエコーの到達時刻、反射係数、期待される極性、フレネル半径、減衰量）。
原稿の 2.3 節（フレネル半径）と 2.5 節（到達時刻）の式をそのまま実装する。

時間はすべて **伝搬時間のみ**（送信波形の主ピークまでの遅れ δt を含まない）。
シミュレーションの時刻と比べるときは、自由空間の参照計算（ref_freespace）で
求めた δt を足すこと（原稿の式 τ = L / c + δt）。

送受信点は同じ位置（アンテナ間隔 0 m）なので、到達時刻は鉛直入射（nadir）の値だけを扱う。

フレネル半径（原稿の式）
------------------------
    r_F      = sqrt(lambda_0 * D / 2)                        lambda_0: 真空中の波長（500 MHz で 0.6 m）
    D_top    = h_antenna + d / sqrt(eps_regolith)            各層の厚さを屈折率で割って足した等価距離
    D_bottom = h_antenna + d / sqrt(eps_regolith) + h / sqrt(eps_target)
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

from materials import C0, Medium

DB_PER_NEPER = 20.0 / math.log(10.0)


def two_way_time_nadir(layers: Sequence[Tuple[float, float]]) -> float:
    """
    水平な層を鉛直に往復する時間 [s]（原稿の式 τ の L / c の部分）。
    layers: [(厚さ [m], 比誘電率), ...]（上から順に）
    """
    return 2.0 * sum(h * math.sqrt(e) for h, e in layers if h > 0) / C0


def equivalent_distance(layers: Sequence[Tuple[float, float]]) -> float:
    """等価距離 D = sum(厚さ / 屈折率)（原稿の D_top, D_bottom）"""
    return sum(h / math.sqrt(e) for h, e in layers if h > 0)


def fresnel_radius(equivalent_distance_m: float, wavelength_vacuum_m: float) -> float:
    """フレネル半径 r_F = sqrt(lambda_0 * D / 2)（原稿の式）"""
    return math.sqrt(wavelength_vacuum_m * equivalent_distance_m / 2.0)


def reflection_coefficient(eps1: float, eps2: float) -> float:
    """鉛直入射の電場反射係数（媒質 1 から 2 へ）: (sqrt e1 - sqrt e2) / (sqrt e1 + sqrt e2)"""
    a, b = math.sqrt(eps1), math.sqrt(eps2)
    return (a - b) / (a + b)


def flip_polarity(label: str) -> str:
    return "".join({"N": "P", "P": "N"}[ch] for ch in label)


def expected_polarity(r: float, transmitted: str = "NPN") -> str:
    """反射係数の符号から期待される極性（R < 0 なら送信波と逆）"""
    return flip_polarity(transmitted) if r < 0 else transmitted


def target_theory(*, antenna_height_m: float, depth_top_m: float, height_m: float,
                  width_m: Optional[float], background: Medium, target: Medium,
                  transmitted_polarity: str = "NPN",
                  attenuation_frequency_hz: float = 500e6,
                  fresnel_wavelength_m: float = C0 / 500e6) -> Dict:
    """
    地表から depth_top_m の深さに上面があり、高さ height_m、幅 width_m の長方形の対象に対する理論値。
    width_m = None は水平方向に無限に広い平板（平行平板計算）。
    """
    e_b, e_t = background.eps_r, target.eps_r
    air = (antenna_height_m, 1.0)
    to_top = [air, (depth_top_m, e_b)]
    to_bottom = to_top + [(height_m, e_t)]

    t_top = two_way_time_nadir(to_top)
    t_bot = two_way_time_nadir(to_bottom)
    r_top = reflection_coefficient(e_b, e_t)
    r_bot = reflection_coefficient(e_t, e_b)

    f = attenuation_frequency_hz
    a_b = float(background.alpha_np_per_m(f))
    a_t = float(target.alpha_np_per_m(f))
    att_top = 2.0 * depth_top_m * a_b * DB_PER_NEPER
    att_bot = att_top + 2.0 * height_m * a_t * DB_PER_NEPER

    d_top = equivalent_distance(to_top)
    d_bot = equivalent_distance(to_bottom)
    rf_top = fresnel_radius(d_top, fresnel_wavelength_m)
    rf_bot = fresnel_radius(d_bot, fresnel_wavelength_m)

    def face(t, r, d_eq, rf, att):
        return dict(
            t_nadir_s=t,
            reflection_coefficient=r,
            expected_polarity=expected_polarity(r, transmitted_polarity),
            equivalent_distance_m=d_eq,
            fresnel_radius_m=rf,
            fresnel_diameter_m=2.0 * rf,
            width_over_fresnel_diameter=None if width_m is None else width_m / (2.0 * rf),
            two_way_attenuation_db=att,
        )

    return dict(
        depth_top_m=depth_top_m,
        height_m=height_m,
        width_m=width_m,
        top=face(t_top, r_top, d_top, rf_top, att_top),
        bottom=face(t_bot, r_bot, d_bot, rf_bot, att_bot),
        dt_bottom_minus_top_s=t_bot - t_top,
    )


def common_arrivals(*, antenna_height_m: float) -> Dict:
    """地表反射の伝搬時間（対象によらない）。送受信点が同じなので直接波は時刻 0 の送信源の場そのもの"""
    return dict(surface_reflection_s=two_way_time_nadir([(antenna_height_m, 1.0)]))
