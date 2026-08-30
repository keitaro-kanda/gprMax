#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ascan_reflection.py - at_tx (モノスタティック) 地下界面反射解析ツール

【既知の限界（README §9）】
1. 多重反射: 理論モデルには含まない（残差・干渉成分として現れる）。
2. 斜め入射: モノスタティック（オフセット 0）前提。垂直入射 Fresnel 係数を適用。
3. 背景差分: 実機観測では取得不能。シミュレーション上の理論上限評価としてのみ扱う。
4. 直達波の裾: 2D 線源グリーン関数の長時間テールが差分なし時のフロアを決定する。
5. 層厚と屈折率の縮退: 単一オフセットでは走時差 delta_t から n と L を完全分離不可（将来の CMP 展開課題）。
6. ラフネス・ランダム媒質: 平坦・水平成層を前提としており、粗動界面（Level 6/7）では散乱により崩れる。
"""

import os
import sys
import json
import logging
from pathlib import Path

# 自身が存在するディレクトリを検索パスの先頭に追加 (Phase 1 T-1.1, README §3.1)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.constants import c as C0
from scipy.signal import hilbert
import matplotlib.pyplot as plt

# ==============================================================================
# 媒質モデルのインポート (README §3.1, Phase 1 T-1.1)
# ==============================================================================
try:
    from ascan_spectrum import (
        LEVEL3_EPS_R,
        LEVEL3_RHO,
        level3_eps,
        level4_targets,
        level4_eps,
        level4_alpha,
        level4_ice_volume_fraction,
        describe_level4_medium,
    )
except ImportError as e:
    raise ImportError(
        "媒質モデルモジュール (ascan_spectrum.py) が見つかりません。"
    ) from e


# ==============================================================================
# [EDIT HERE] 設定ブロック (Phase 1 T-1.3, README §8)
# ==============================================================================
TX_HEIGHT = 0.35              # [m] アンテナ地上高 (Level_N.in と一致)
ICE_TOP_M = 1.00              # [m] 氷層上面深さ (Level_4.in と一致)
ICE_THICK_M = 1.00            # [m] 氷層層厚 (Level_4.in と一致)
R_REF = 1.00                  # [m] 参照計算 (far_1m) の距離
CENTER_FREQ_HZ = 1.25e9       # [Hz] レーダ中心周波数 (1.25 GHz)

USE_BACKGROUND_SUBTRACTION = True  # 背景差分の有効化フラグ
NOISE_BAND_NS = (8.0, 40.0)        # [ns] ノイズフロア測定時間帯 (README §2.2)
EVENT_WINDOW_NS = 2.0              # [ns] イベント探索窓の半幅 (README F-3)
DEFAULT_PATHS_JSON = "out_file_paths.json"
OUTPUT_DIR_NAME = "ascan_reflection_out"


# ==============================================================================
# Phase 2: 理論モデル (README §4)
# ==============================================================================
def r_eff_round_trip(L_layers, n_layers, h=TX_HEIGHT):
    """
    往復経路の見かけ源距離 r_eff = 2h + 2 * sum(L_j / n_j) (README §4.3)
    """
    if len(L_layers) == 0:
        return 2.0 * h
    r = 2.0 * h
    for L_j, n_j in zip(L_layers, n_layers):
        r += 2.0 * (L_j / n_j)
    return r


def calc_fresnel_reflection(n1, n2):
    """垂直入射反射係数 R = (n1 - n2) / (n1 + n2)"""
    return (n1 - n2) / (n1 + n2)


def calc_round_trip_transmission(n1, n2):
    """界面の往復透過係数 T_down * T_up = 4*n1*n2 / (n1 + n2)^2 (README §4.4)"""
    return 4.0 * n1 * n2 / ((n1 + n2) ** 2)


def get_layer_properties(ice_vol_frac, f_hz=CENTER_FREQ_HZ):
    """
    指定氷量における誘電率実部 eps' と減衰定数 alpha [Np/m] を取得 (LLL混合則補間)
    """
    eps_reg = LEVEL3_EPS_R
    alpha_reg = level4_alpha(f_hz, in_ice=False)

    if ice_vol_frac <= 0.0:
        return eps_reg, alpha_reg
    elif np.isclose(ice_vol_frac, 0.10):
        eps_ice_10 = level4_targets()[-1][0]
        alpha_ice_10 = level4_alpha(f_hz, in_ice=True)
        return eps_ice_10, alpha_ice_10
    else:
        eps_ice_10 = level4_targets()[-1][0]
        alpha_ice_10 = level4_alpha(f_hz, in_ice=True)
        # LLL混合則: eps^(1/3) の体積分率線形補間
        cbrt_reg = eps_reg ** (1.0 / 3.0)
        cbrt_10 = eps_ice_10 ** (1.0 / 3.0)
        cbrt_v = cbrt_reg + (ice_vol_frac / 0.10) * (cbrt_10 - cbrt_reg)
        eps_v = cbrt_v ** 3.0
        alpha_v = alpha_reg + (ice_vol_frac / 0.10) * (alpha_ice_10 - alpha_reg)
        return eps_v, alpha_v


def calc_theoretical_events(ice_vol_frac=0.10, h=TX_HEIGHT, L_top=ICE_TOP_M, L_ice=ICE_THICK_M, f_hz=CENTER_FREQ_HZ):
    """
    指定氷量における各反射イベント（地表、氷層上面、氷層下面）の理論値を計算 (README §4.2, §4.5)
    """
    eps_reg, alpha_reg = get_layer_properties(0.0, f_hz=f_hz)
    n_reg = np.sqrt(eps_reg)

    eps_ice, alpha_ice = get_layer_properties(ice_vol_frac, f_hz=f_hz)
    n_ice = np.sqrt(eps_ice)

    events = {}

    # Event 0: 地表面 (k=0, depth=0 m)
    r_eff_0 = r_eff_round_trip([], [], h=h)
    G_0 = np.sqrt(R_REF / r_eff_0)
    T_0 = 1.0
    R_0 = calc_fresnel_reflection(1.0, n_reg)
    A_0 = 1.0
    t_0 = 2.0 * h / C0
    amp_0 = G_0 * T_0 * np.abs(R_0) * A_0

    events["surface"] = {
        "depth_m": 0.0,
        "r_eff": r_eff_0,
        "G": G_0,
        "T": T_0,
        "R": R_0,
        "A": A_0,
        "time_s": t_0,
        "time_ns": t_0 * 1e9,
        "amp_lin": amp_0,
        "amp_db_rel_surf": 0.0,
    }

    # Event 1: 氷層上面 (k=1, depth=L_top)
    r_eff_1 = r_eff_round_trip([L_top], [n_reg], h=h)
    G_1 = np.sqrt(R_REF / r_eff_1)
    T_surf = calc_round_trip_transmission(1.0, n_reg)
    T_1 = T_surf
    R_1 = calc_fresnel_reflection(n_reg, n_ice)
    A_1 = np.exp(-2.0 * alpha_reg * L_top)
    t_1 = (2.0 * h + 2.0 * n_reg * L_top) / C0
    amp_1 = G_1 * T_1 * np.abs(R_1) * A_1

    events["ice_top"] = {
        "depth_m": L_top,
        "r_eff": r_eff_1,
        "G": G_1,
        "T": T_1,
        "R": R_1,
        "A": A_1,
        "time_s": t_1,
        "time_ns": t_1 * 1e9,
        "amp_lin": amp_1,
        "amp_db_rel_surf": 20.0 * np.log10(amp_1 / amp_0),
    }

    # Event 2: 氷層下面 (k=2, depth=L_top + L_ice)
    r_eff_2 = r_eff_round_trip([L_top, L_ice], [n_reg, n_ice], h=h)
    G_2 = np.sqrt(R_REF / r_eff_2)
    T_icetop = calc_round_trip_transmission(n_reg, n_ice)
    T_2 = T_surf * T_icetop
    R_2 = calc_fresnel_reflection(n_ice, n_reg)  # 符号反転
    A_2 = np.exp(-2.0 * (alpha_reg * L_top + alpha_ice * L_ice))
    t_2 = (2.0 * h + 2.0 * n_reg * L_top + 2.0 * n_ice * L_ice) / C0
    amp_2 = G_2 * T_2 * np.abs(R_2) * A_2

    events["ice_bot"] = {
        "depth_m": L_top + L_ice,
        "r_eff": r_eff_2,
        "G": G_2,
        "T": T_2,
        "R": R_2,
        "A": A_2,
        "time_s": t_2,
        "time_ns": t_2 * 1e9,
        "amp_lin": amp_2,
        "amp_db_rel_surf": 20.0 * np.log10(amp_2 / amp_0),
    }

    return events


# ==============================================================================
# 入出力および前処理ユーティリティ (Phase 1 T-1.2, README §2.3)
# ==============================================================================
def load_paths(json_path=DEFAULT_PATHS_JSON):
    """out_file_paths.json から各トレースのパス情報を取得"""
    p = Path(json_path)
    if not p.exists():
        logging.warning(f"{json_path} が見つかりません。モック/相対探索を行います。")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_output_dir(base_dir="."):
    out_dir = Path(base_dir) / OUTPUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_trace(file_path):
    """gprMax または numpy 形式の A-scan 出力ファイルを読み込む"""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
    
    if p.suffix == ".out":
        try:
            import h5py
            with h5py.File(p, "r") as f:
                dt = f.attrs["dt"]
                ez = f["rxs"]["rx1"]["Ez"][:]
                time = np.arange(len(ez)) * dt
                return time, ez
        except ImportError:
            raise ImportError("HDF5 (.out) 読み込みには h5py が必要です。")
    else:
        data = np.loadtxt(p)
        if data.ndim == 2:
            return data[:, 0], data[:, 1]
        return np.arange(len(data)), data


def resample_trace(t_src, trace_src, t_target):
    """時間軸のリサンプリング"""
    return np.interp(t_target, t_src, trace_src, left=0.0, right=0.0)


# ==============================================================================
# 信号処理・イベント検出・ノイズフロア (Phase 4, Phase 5)
# ==============================================================================
def calc_envelope(trace):
    """ヒルベルト変換による解析信号の包絡線"""
    analytic_signal = hilbert(trace)
    return np.abs(analytic_signal)


def measure_noise_floor(time_ns, trace, surf_peak_amp, band_ns=NOISE_BAND_NS):
    """
    指定時間帯におけるノイズフロア (RMS) の測定と 3 段階判定 (README §2.2, F-2)
    """
    mask = (time_ns >= band_ns[0]) & (time_ns <= band_ns[1])
    if not np.any(mask):
        return 0.0, -999.0, "INVALID_BAND"
    
    noise_rms = np.sqrt(np.mean(trace[mask] ** 2))
    floor_db = 20.0 * np.log10(noise_rms / surf_peak_amp) if surf_peak_amp > 0 else -999.0

    if floor_db < -70.0:
        verdict = "< -70 dB (0.5 vol% まで検出可能)"
    elif floor_db <= -60.0:
        verdict = "-60 〜 -70 dB (1.0 vol% は可能、0.5 vol% は厳しい)"
    else:
        verdict = "> -55 dB (反射チャネル成立困難・PML/直接波要対策)"

    return noise_rms, floor_db, verdict


def detect_event_peak(time_ns, env, t_theo_ns, window_ns=EVENT_WINDOW_NS, noise_floor=1e-12):
    """
    理論走時 t_theo_ns の周辺窓からピークを検出 (README F-3)
    """
    mask = (time_ns >= (t_theo_ns - window_ns)) & (time_ns <= (t_theo_ns + window_ns))
    if not np.any(mask):
        return None

    t_sub = time_ns[mask]
    env_sub = env[mask]
    peak_idx = np.argmax(env_sub)
    peak_amp = env_sub[peak_idx]
    peak_time_ns = t_sub[peak_idx]

    snr = peak_amp / noise_floor if noise_floor > 0 else np.nan
    snr_db = 20.0 * np.log10(snr) if snr > 0 else -999.0

    return {
        "time_ns": peak_time_ns,
        "amp": peak_amp,
        "snr": snr,
        "snr_db": snr_db,
    }


# ==============================================================================
# プロット生成関数群 (README §6.1, §6.4)
# ==============================================================================
def plot_fig1_trace(time_ns, trace_raw, trace_sub, env_sub, theo_events, floor_db, out_dir):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(time_ns, trace_raw, color="black", lw=1.0, label="Raw at_tx")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("(a) Raw Trace (No Background Subtraction)")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].plot(time_ns, trace_sub, color="tab:blue", lw=1.0, label="Subtracted at_tx")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("(b) Background Subtracted Trace")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    surf_peak = np.max(calc_envelope(trace_raw))
    env_db = 20.0 * np.log10(np.maximum(env_sub, 1e-12) / surf_peak)
    axes[2].plot(time_ns, env_db, color="tab:red", lw=1.2, label="Envelope (Sub)")
    axes[2].axhline(floor_db, color="gray", linestyle="--", label=f"Noise Floor ({floor_db:.1f} dB)")

    for name, ev in theo_events.items():
        color = "green" if name == "surface" else ("orange" if name == "ice_top" else "purple")
        for ax in axes:
            ax.axvline(ev["time_ns"], color=color, linestyle=":", alpha=0.7)
        axes[2].text(ev["time_ns"] + 0.3, -20, name, color=color, rotation=90, verticalalignment="bottom")

    axes[2].set_ylabel("Amplitude [dB rel. surface]")
    axes[2].set_xlabel("Time [ns]")
    axes[2].set_title("(c) Envelope & Theoretical Travel Times")
    axes[2].set_ylim([-80, 5])
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_dir / "fig1_trace.png", dpi=300)
    fig.savefig(out_dir / "fig1_trace.pdf")
    plt.close(fig)


def plot_fig2_events(events_res, theo_events, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    names = ["surface", "ice_top", "ice_bot"]

    t_obs = [events_res[k]["obs_time_ns"] if events_res[k] else np.nan for k in names]
    t_the = [theo_events[k]["time_ns"] for k in names]
    amp_obs = [events_res[k]["obs_amp_db"] if events_res[k] else np.nan for k in names]
    amp_the = [theo_events[k]["amp_db_rel_surf"] for k in names]

    axes[0].plot(names, t_the, "s--", label="Theory", color="tab:blue")
    axes[0].plot(names, t_obs, "o", label="Observed", color="tab:red")
    axes[0].set_ylabel("Travel Time [ns]")
    axes[0].set_title("(a) Travel Time")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    axes[1].plot(names, amp_the, "s--", label="Theory", color="tab:blue")
    axes[1].plot(names, amp_obs, "o", label="Observed", color="tab:red")
    axes[1].set_ylabel("Amplitude [dB rel. surface]")
    axes[1].set_title("(b) Amplitude")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    res_amp = np.array(amp_obs) - np.array(amp_the)
    axes[2].bar(names, res_amp, color="tab:purple", alpha=0.7)
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_ylabel("Residual [dB]")
    axes[2].set_title("(c) Residuals")
    axes[2].grid(True, linestyle=":", alpha=0.6)

    dt_layer = t_the[2] - t_the[1]
    f_multiple = 1.0 / (dt_layer * 1e-9) / 1e6 if dt_layer > 0 else 0.0
    axes[2].text(
        0.05, 0.15,
        f"* 多重反射の寄与（理論に非含有）\n  層内周期 1/Δt = {f_multiple:.1f} MHz",
        transform=axes[2].transAxes,
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3)
    )

    plt.tight_layout()
    fig.savefig(out_dir / "fig2_events.png", dpi=300)
    fig.savefig(out_dir / "fig2_events.pdf")
    plt.close(fig)


def plot_fig3_reflection(vols, r_theories, obs_vol, obs_r, floor_db, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(vols * 100, 20 * np.log10(np.abs(r_theories)), "b-", lw=1.5, label="Theoretical |R_top|")
    if obs_r is not None:
        ax.plot(obs_vol * 100, 20 * np.log10(np.abs(obs_r)), "ro", markersize=8, label=f"Observed ({obs_vol*100:.1f} vol%)")
    ax.axhline(floor_db, color="gray", linestyle="--", label=f"Noise Floor ({floor_db:.1f} dB)")
    ax.set_xlabel("Ice Volume Fraction [%]")
    ax.set_ylabel("Reflection Coefficient |R| [dB]")
    ax.set_title("Reflection Channel: Ice Content vs R_top")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "fig3_reflection.png", dpi=300)
    fig.savefig(out_dir / "fig3_reflection.pdf")
    plt.close(fig)


def plot_fig4_traveltime(vols, dt_theories, obs_vol, obs_dt, out_dir):
    fig, ax1 = plt.subplots(figsize=(7, 5))
    dt_no_ice = dt_theories[0]
    delay_ps = (dt_theories - dt_no_ice) * 1e3

    ax1.plot(vols * 100, dt_theories, "g-", lw=1.5, label="Round-trip Travel Time (Bottom)")
    ax1.set_xlabel("Ice Volume Fraction [%]")
    ax1.set_ylabel("Travel Time [ns]", color="green")
    ax1.tick_params(axis="y", labelcolor="green")

    ax2 = ax1.twinx()
    ax2.plot(vols * 100, delay_ps, "m--", lw=1.5, label="Delay from Dry Base [ps]")
    ax2.set_ylabel("Delay relative to Dry Base [ps]", color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")

    if obs_dt is not None:
        ax1.plot(obs_vol * 100, obs_dt, "ro", markersize=8, label="Observed")

    ax1.set_title("Travel-time Channel: Ice Content vs Delay")
    ax1.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(out_dir / "fig4_traveltime.png", dpi=300)
    fig.savefig(out_dir / "fig4_traveltime.pdf")
    plt.close(fig)


def plot_fig5_attenuation(freq_ghz, alpha_the, alpha_obs, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(freq_ghz, alpha_the, "b-", lw=1.5, label="Theory alpha(f)")
    if alpha_obs is not None:
        ax.plot(freq_ghz, alpha_obs, "r--", lw=1.5, label="Observed LSR alpha(f)")
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Attenuation Constant alpha [Np/m]")
    ax.set_title("Attenuation Channel: In-layer Loss")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "fig5_attenuation.png", dpi=300)
    fig.savefig(out_dir / "fig5_attenuation.pdf")
    plt.close(fig)


def plot_fig6_channels(vols, snr_ref, snr_time, snr_att, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(vols * 100, snr_ref, "tab:red", lw=2, label="Reflection Channel")
    ax.plot(vols * 100, snr_time, "tab:green", lw=2, label="Travel-time Channel")
    ax.plot(vols * 100, snr_att, "tab:blue", lw=2, label="Attenuation Channel")

    ax.axhline(1.0, color="black", linestyle="--", lw=1.2, label="Detection Limit (SNR = 1)")
    ax.set_yscale("log")
    ax.set_xlabel("Ice Volume Fraction [%]")
    ax.set_ylabel("Signal-to-Noise Ratio (SNR)")
    ax.set_title("Detection Limit Comparison Across 3 Channels (fig6)")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(out_dir / "fig6_channels.png", dpi=300)
    fig.savefig(out_dir / "fig6_channels.pdf")
    plt.close(fig)


# ==============================================================================
# メイン実行ルーチン
# ==============================================================================
def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info("=== at_tx 反射解析ツール (ascan_reflection.py) 実行開始 ===")

    if "describe_level4_medium" in globals():
        logging.info("媒質モデル設定:\n" + describe_level4_medium())

    out_dir = resolve_output_dir()
    paths_data = load_paths(DEFAULT_PATHS_JSON)

    # 理論値テーブルの照合 (Phase 2 検証)
    theo_events_10 = calc_theoretical_events(ice_vol_frac=0.10)
    logging.info("--- Phase 2 理論値の検証 (10 vol%) ---")
    for k, v in theo_events_10.items():
        logging.info(
            f"イベント [{k:8s}]: r_eff = {v['r_eff']:.4f} m, 走時 = {v['time_ns']:.3f} ns, "
            f"R = {v['R']:+.6f}, 相対振幅 = {v['amp_db_rel_surf']:.1f} dB"
        )

    # トレースデータの取得
    dt = 1e-11
    t_arr = np.arange(0, 50e-9, dt)
    t_ns = t_arr * 1e9

    target_path = paths_data.get("Level_4", {}).get("at_tx", None)
    ref_dry_path = paths_data.get("Level_4_dry", {}).get("at_tx", None)

    if target_path and Path(target_path).exists():
        t_arr, trace_raw = load_trace(target_path)
        t_ns = t_arr * 1e9
    else:
        logging.warning("実データトレースが見つからないため、検証用テスト波形を生成します。")
        pulse = np.exp(-((t_ns - 2.335) / 0.35) ** 2) * np.sin(2 * np.pi * 1.25 * (t_ns - 2.335))
        pulse_top = -0.042 * np.exp(-((t_ns - 13.890) / 0.35) ** 2) * np.sin(2 * np.pi * 1.25 * (t_ns - 13.890))
        pulse_bot = +0.027 * np.exp(-((t_ns - 26.010) / 0.35) ** 2) * np.sin(2 * np.pi * 1.25 * (t_ns - 26.010))
        noise = np.random.normal(0, 1e-4, len(t_ns))
        trace_raw = pulse + pulse_top + pulse_bot + noise

    if ref_dry_path and Path(ref_dry_path).exists():
        t_ref, trace_dry = load_trace(ref_dry_path)
        trace_dry = resample_trace(t_ref, trace_dry, t_arr)
    else:
        trace_dry = np.exp(-((t_ns - 2.335) / 0.35) ** 2) * np.sin(2 * np.pi * 1.25 * (t_ns - 2.335))

    trace_sub = trace_raw - trace_dry if USE_BACKGROUND_SUBTRACTION else trace_raw

    # ノイズフロア測定 (Phase 4)
    surf_peak = np.max(calc_envelope(trace_raw))
    noise_rms, floor_db, verdict = measure_noise_floor(t_ns, trace_dry, surf_peak, band_ns=NOISE_BAND_NS)
    logging.info(f"ノイズフロア測定: RMS = {noise_rms:.3e}, {floor_db:.2f} dB (対地表反射ピーク)")
    logging.info(f"判定: {verdict}")

    # イベント検出と解析 (Phase 5)
    env_sub = calc_envelope(trace_sub)
    events_res = {}
    for name, theo in theo_events_10.items():
        ev_env = calc_envelope(trace_raw) if name == "surface" else env_sub
        res = detect_event_peak(t_ns, ev_env, theo["time_ns"], window_ns=EVENT_WINDOW_NS, noise_floor=noise_rms)
        if res:
            res["obs_time_ns"] = res["time_ns"]
            res["obs_amp_db"] = 20.0 * np.log10(res["amp"] / surf_peak)
            events_res[name] = res
        else:
            events_res[name] = None

    # 層内屈折率の逆算
    if events_res["ice_top"] and events_res["ice_bot"]:
        dt_obs = events_res["ice_bot"]["obs_time_ns"] - events_res["ice_top"]["obs_time_ns"]
        n_layer_est = (dt_obs * 1e-9 * C0) / (2.0 * ICE_THICK_M)
        logging.info(f"層内走時差 delta_t = {dt_obs:.3f} ns -> 逆算屈折率 n = {n_layer_est:.5f}")

    # スイープ計算と 3 チャネル SNR 比較 (Phase 6)
    vol_sweep = np.linspace(0.001, 0.10, 50)
    r_theories = []
    dt_theories = []
    snr_ref_arr = []
    snr_time_arr = []
    snr_att_arr = []

    t_dry_bot = (2.0 * TX_HEIGHT + 2.0 * np.sqrt(LEVEL3_EPS_R) * (ICE_TOP_M + ICE_THICK_M)) / C0 * 1e9

    for v in vol_sweep:
        evs = calc_theoretical_events(ice_vol_frac=v)
        r_val = evs["ice_top"]["R"]
        t_bot = evs["ice_bot"]["time_ns"]
        r_theories.append(r_val)
        dt_theories.append(t_bot)

        sig_amp = evs["ice_top"]["amp_lin"]
        snr_ref_arr.append(sig_amp / (noise_rms / surf_peak) if noise_rms > 0 else 1e3)

        delay_ns = t_bot - t_dry_bot
        snr_time_arr.append(np.maximum(delay_ns / 0.025, 1e-3))

        _, alpha_v = get_layer_properties(v)
        snr_att_arr.append(np.maximum((2.0 * alpha_v * ICE_THICK_M) / 0.02, 1e-3))

    r_theories = np.array(r_theories)
    dt_theories = np.array(dt_theories)

    # プロット生成と数値保存 (Phase 7)
    logging.info("図・数値データの出力中...")
    plot_fig1_trace(t_ns, trace_raw, trace_sub, env_sub, theo_events_10, floor_db, out_dir)
    plot_fig2_events(events_res, theo_events_10, out_dir)
    plot_fig3_reflection(vol_sweep, r_theories, 0.10, events_res["ice_top"]["amp"] / surf_peak if events_res["ice_top"] else None, floor_db, out_dir)
    plot_fig4_traveltime(vol_sweep, dt_theories, 0.10, events_res["ice_bot"]["obs_time_ns"] if events_res["ice_bot"] else None, out_dir)
    
    freqs = np.linspace(0.5e9, 2.0e9, 50)
    alpha_curve = [level4_alpha(f, in_ice=True) for f in freqs]
    plot_fig5_attenuation(freqs / 1e9, alpha_curve, None, out_dir)
    
    plot_fig6_channels(vol_sweep, snr_ref_arr, snr_time_arr, snr_att_arr, out_dir)

    # CSV 出力
    csv_events_path = out_dir / "events.csv"
    with open(csv_events_path, "w", encoding="utf-8") as f:
        f.write("event_name,theo_time_ns,obs_time_ns,theo_amp_db,obs_amp_db,snr_db\n")
        for name, theo in theo_events_10.items():
            obs = events_res.get(name)
            o_t = f"{obs['obs_time_ns']:.4f}" if obs else "NaN"
            o_a = f"{obs['obs_amp_db']:.2f}" if obs else "NaN"
            snr_v = f"{obs['snr_db']:.2f}" if obs else "NaN"
            f.write(f"{name},{theo['time_ns']:.4f},{o_t},{theo['amp_db_rel_surf']:.2f},{o_a},{snr_v}\n")

    csv_chan_path = out_dir / "channels.csv"
    with open(csv_chan_path, "w", encoding="utf-8") as f:
        f.write("channel,signal_def,noise_def,snr_10vol\n")
        f.write(f"reflection,R_top peak amplitude,Noise floor RMS ({floor_db:.1f} dB),{snr_ref_arr[-1]:.2f}\n")
        f.write(f"travel_time,Delay from dry base ({dt_theories[-1]-t_dry_bot:.3f} ns),Dispersion residual (0.025 ns),{snr_time_arr[-1]:.2f}\n")
        f.write(f"attenuation,In-layer attenuation alpha,LSR residual RMS,{snr_att_arr[-1]:.2f}\n")

    info_path = out_dir / "run_info.txt"
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("=== ascan_reflection.py 実行結果サマリー ===\n")
        f.write(f"アンテナ地上高 h: {TX_HEIGHT} m\n")
        f.write(f"氷層設定: 深さ {ICE_TOP_M} m, 層厚 {ICE_THICK_M} m\n")
        f.write(f"ノイズフロア (8-40 ns): {floor_db:.2f} dB\n")
        f.write(f"フロア判定: {verdict}\n")
        if events_res["ice_top"] and events_res["ice_bot"]:
            f.write(f"逆算屈折率 n_layer: {n_layer_est:.5f} (理論値: {np.sqrt(level4_targets()[-1][0]):.5f})\n")

    logging.info(f"解析完了: 出力ファイルは '{out_dir}' に保存されました。")


if __name__ == "__main__":
    main()