"""
polarity_core.py
================

エコーの極性判定（原稿 2.5 節）の計算部分。analyze_polarity.py と plot_polarity_maps.py から使う。

手順（原稿 2.5 節と同じ）
------------------------
1. 背景の除去      x(t) = 各ケースの受信波形 - ref_background の受信波形
2. 送信波形        ref_freespace の受信波形（送信点から 1 m）。主ピークが t = 0 になるようにずらし、
                   振幅が最大値の TEMPLATE_LEVEL（3%）以上の範囲を切り出して s(t) とする。
                   主ピークの遅れ  delta_t = (主ピークの時刻) - (送受信点の距離) / c
3. 到達時刻        tau = (ケース JSON の伝搬時間 t_nadir_s。表 2 の寸法による) + delta_t
4. 規格化相互相関  rho(t') = sum_{窓} x(t) s(t - t') / sqrt( sum_{窓} x^2 * sum s^2 )
                   窓 = [tau - T_w, tau + T_w]、t' は |t' - tau| <= LAG_MAX の範囲で動かす。
                   |rho| が最大となる t' での rho の符号を極性とする（rho > 0 なら送信波形と同じ極性）。
5. 分類            期待される極性と一致 -> Type 1、逆 -> Type 2。
                   上面と底面の窓が重なる、または |rho| < RHO_MIN -> Type 3（判定不能）。

rho^2 は、窓の中のエネルギーのうち送信波形の定数倍（最小二乗近似）で説明できる割合に等しい
（送信波形が窓に収まっている場合）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

C0 = 299792458.0

# Type 3 の理由
REASON_OVERLAP = "overlap"          # 上面と底面の時間窓が重なる
REASON_LOW_CORR = "low_corr"        # |rho| の最大値が閾値未満
REASON_OUT_OF_RANGE = "out_of_range"  # 窓が計算時間の外にはみ出す
REASON_NO_SIGNAL = "no_signal"      # 窓の中の振幅がゼロ


def flip_polarity(label: str) -> str:
    return "".join({"N": "P", "P": "N"}[ch] for ch in label)


# =============================================================================
# gprMax の .out
# =============================================================================
@dataclass
class Trace:
    ez: np.ndarray       # 受信点の Ez [V/m]
    dt: float            # 時間ステップ [s]

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.ez.size) * self.dt


def read_out(path, component: str = "Ez", rx: str = "rx1") -> Trace:
    """gprMax v3 の .out（HDF5）から受信波形を読む"""
    import h5py
    with h5py.File(path, "r") as f:
        dt = float(f.attrs["dt"])
        ez = np.array(f[f"rxs/{rx}/{component}"][:], dtype=float)
    return Trace(ez=ez, dt=dt)


# =============================================================================
# 送信波形
# =============================================================================
@dataclass
class Template:
    s: np.ndarray            # 切り出した送信波形（主ピークの振幅で規格化。主ピーク = +1 または -1）
    peak_index: int          # s の中の主ピークの位置
    dt: float
    peak_time_s: float       # 自由空間の参照計算での主ピークの時刻（放物線補間）
    peak_amplitude: float    # 主ピークの振幅 [V/m]
    distance_m: float        # 送受信点の距離
    delta_t_s: float         # 主ピークの遅れ delta_t
    level: float             # 切り出しの振幅の割合
    polarity: str            # 主ピークと前後のピークの符号から読んだ極性（例 NPN）
    side_peaks: Dict         # 主ピークの前後の極値（相対振幅と時刻）

    @property
    def t_rel(self) -> np.ndarray:
        """主ピークを 0 とした時刻 [s]"""
        return (np.arange(self.s.size) - self.peak_index) * self.dt

    @property
    def energy(self) -> float:
        return float(np.sum(self.s ** 2))

    def autocorrelation(self) -> Tuple[np.ndarray, np.ndarray]:
        """規格化した自己相関（ずれ [s], 値）"""
        ac = np.correlate(self.s, self.s, mode="full") / self.energy
        lag = (np.arange(ac.size) - (self.s.size - 1)) * self.dt
        return lag, ac

    def first_zero_of_autocorrelation_s(self) -> float:
        """自己相関が最初に 0 になるずれ [s]（正負の小さい方）"""
        lag, ac = self.autocorrelation()
        c = self.s.size - 1
        out = []
        for direction in (1, -1):
            i = c
            while 0 <= i + direction < ac.size and ac[i + direction] > 0:
                i += direction
            j = i + direction
            if not (0 <= j < ac.size):
                out.append(math.inf)
                continue
            # 線形補間でゼロ点
            a0, a1 = ac[i], ac[j]
            frac = a0 / (a0 - a1)
            out.append(abs(lag[i] + frac * (lag[j] - lag[i])))
        return float(min(out))


def _parabolic_peak(y: np.ndarray, k: int) -> float:
    """3 点の放物線補間による極値の位置（標本単位、k からのずれを含む）"""
    if 0 < k < y.size - 1:
        a, b, c = y[k - 1], y[k], y[k + 1]
        den = a - 2 * b + c
        if den != 0:
            return k + 0.5 * (a - c) / den
    return float(k)


def build_template(ref: Trace, distance_m: float, level: float = 0.03,
                   search_s: float = 10e-9) -> Template:
    """
    自由空間の参照計算から送信波形 s(t) を作る。
    主ピーク = |Ez| の最大。主ピークの前後 search_s の範囲で、|s| >= level の最初と最後の標本までを切り出す。
    """
    e = ref.ez
    dt = ref.dt
    k = int(np.argmax(np.abs(e)))
    amp = float(e[k])
    s = e / abs(amp)                       # 主ピークの振幅で規格化（符号は残す）
    n = int(round(search_s / dt))
    lo, hi = max(k - n, 0), min(k + n, e.size - 1)
    seg = np.abs(s[lo:hi + 1])
    idx = np.flatnonzero(seg >= level) + lo
    i0, i1 = int(idx[0]), int(idx[-1])
    k_sub = _parabolic_peak(e if amp > 0 else -e, k)
    peak_time = k_sub * dt
    # 主ピークの前後の極値（極性の読み取り用）
    before = s[i0:k]
    after = s[k + 1:i1 + 1]
    sp = {}
    if before.size:
        jb = int(np.argmax(-np.sign(amp) * before))
        sp["before"] = dict(rel_amplitude=float(before[jb] * np.sign(amp)),
                            t_rel_s=float((i0 + jb - k) * dt))
    if after.size:
        ja = int(np.argmax(-np.sign(amp) * after))
        sp["after"] = dict(rel_amplitude=float(after[ja] * np.sign(amp)),
                           t_rel_s=float((k + 1 + ja - k) * dt))
    main = "P" if amp > 0 else "N"
    pol = flip_polarity(main) + main + flip_polarity(main)
    return Template(s=s[i0:i1 + 1].copy(), peak_index=k - i0, dt=dt, peak_time_s=peak_time,
                    peak_amplitude=amp, distance_m=distance_m,
                    delta_t_s=peak_time - distance_m / C0, level=level,
                    polarity=pol, side_peaks=sp)


# =============================================================================
# 規格化相互相関と分類
# =============================================================================
@dataclass
class FaceResult:
    tau_s: float
    window_s: Tuple[float, float]
    out_of_range: bool
    rho_best: float = float("nan")      # |rho| が最大のときの rho
    lag_best_s: float = float("nan")    # そのときの t' - tau
    rho_at_tau: float = float("nan")    # t' = tau のときの rho
    rho_pos_max: float = float("nan")   # 範囲内の rho の最大値
    rho_neg_min: float = float("nan")   # 範囲内の rho の最小値
    amplitude_ls: float = float("nan")  # 最も合った位置での送信波形の倍率（最小二乗）
    window_peak_abs: float = float("nan")
    window_rms: float = float("nan")


def ncc_face(x: np.ndarray, dt: float, tpl: Template, tau_s: float,
             t_w_s: float, lag_max_s: float) -> FaceResult:
    """原稿の式 rho(t') を、|t' - tau| <= lag_max_s の範囲の標本ごとに計算する"""
    if abs(dt - tpl.dt) > 1e-6 * dt:
        raise ValueError(f"dt of the trace ({dt}) and the template ({tpl.dt}) differ")
    n = x.size
    ia = int(math.ceil((tau_s - t_w_s) / dt - 1e-9))
    ib = int(math.floor((tau_s + t_w_s) / dt + 1e-9))
    res = FaceResult(tau_s=tau_s, window_s=(tau_s - t_w_s, tau_s + t_w_s),
                     out_of_range=(ia < 0 or ib > n - 1))
    if res.out_of_range:
        return res
    xw = x[ia:ib + 1]
    ex = float(np.sum(xw ** 2))
    res.window_peak_abs = float(np.max(np.abs(xw)))
    res.window_rms = float(np.sqrt(ex / xw.size))
    if ex == 0.0:
        return res
    es = tpl.energy
    norm = math.sqrt(ex * es)
    # t' は標本の時刻 c dt。|c dt - tau| <= lag_max_s を満たす c だけを使う
    c_lo = int(math.ceil((tau_s - lag_max_s) / dt - 1e-9))
    c_hi = int(math.floor((tau_s + lag_max_s) / dt + 1e-9))
    i_tau = int(round(tau_s / dt))
    if c_lo > c_hi:                       # LAG_MAX < dt/2（ずらさない）: tau に最も近い標本だけ
        c_lo = c_hi = i_tau
    L = tpl.s.size
    ms = list(range(c_lo - i_tau, c_hi - i_tau + 1))
    rhos = np.empty(len(ms))
    for q, m in enumerate(ms):
        c = i_tau + m                                  # t' の標本番号
        # 窓の標本 i について s(t_i - t') = s_crop[i - c + peak_index]
        j = np.arange(ia, ib + 1) - c + tpl.peak_index
        ok = (j >= 0) & (j < L)
        rhos[q] = float(np.sum(xw[ok] * tpl.s[j[ok]])) / norm
    # s は送信波形そのもの（符号を含む）なので、rho > 0 は送信波形と同じ極性を意味する
    qb = int(np.argmax(np.abs(rhos)))
    res.rho_best = float(rhos[qb])
    res.lag_best_s = float((i_tau + ms[qb]) * dt - tau_s)
    res.rho_at_tau = float(rhos[ms.index(0)])               # t' = tau に最も近い標本
    res.rho_pos_max = float(np.max(rhos))
    res.rho_neg_min = float(np.min(rhos))
    res.amplitude_ls = res.rho_best * math.sqrt(ex / es)     # x ~ amplitude_ls * s(t - t')
    return res


def observed_polarity(rho: float, transmitted: str) -> str:
    if not np.isfinite(rho) or rho == 0:
        return ""
    return transmitted if rho > 0 else flip_polarity(transmitted)


def classify(*, rho_best: float, observed: str, expected: str, overlap: bool,
             out_of_range: bool, rho_min: float) -> Tuple[int, str]:
    """(Type, 理由)。Type 3 の理由の優先順位: out_of_range > overlap > no_signal > low_corr"""
    if out_of_range:
        return 3, REASON_OUT_OF_RANGE
    if overlap:
        return 3, REASON_OVERLAP
    if not np.isfinite(rho_best):
        return 3, REASON_NO_SIGNAL
    if abs(rho_best) < rho_min:
        return 3, REASON_LOW_CORR
    return (1, "") if observed == expected else (2, "")


def windows_overlap(tau_top_s: float, tau_bottom_s: float, t_w_s: float) -> bool:
    """上面と底面の時間窓 [tau - T_w, tau + T_w] が重なるか"""
    return (tau_bottom_s - tau_top_s) < 2.0 * t_w_s


def fresnel_diameter_bottom(h_m, *, antenna_height_m: float, depth_top_m: float,
                            eps_background: float, eps_target: float, wavelength_m: float):
    """底面のフレネル帯の直径 2 r_F（原稿 2.3 節の式。h は配列でもよい）"""
    d_eq = antenna_height_m + depth_top_m / math.sqrt(eps_background) + np.asarray(h_m) / math.sqrt(eps_target)
    return 2.0 * np.sqrt(wavelength_m * d_eq / 2.0)


def fresnel_diameter_top(*, antenna_height_m: float, depth_top_m: float,
                         eps_background: float, wavelength_m: float) -> float:
    d_eq = antenna_height_m + depth_top_m / math.sqrt(eps_background)
    return 2.0 * math.sqrt(wavelength_m * d_eq / 2.0)
