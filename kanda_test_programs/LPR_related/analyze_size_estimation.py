#!/usr/bin/env python3
"""
analyze_size_estimation.py
==========================

計算済みの A-scan（gprMax の .out）から、対象（岩石・空隙）の上端と下端のエコーを探し、
時間差から高さを推定する。

実行方法
--------
VS Code の実行ボタン（または `python analyze_size_estimation.py`）で実行する。引数は不要。
実行すると、キャンペーンフォルダ、対象、形状、深さ、確認用図を作るかを順に尋ねる（Enter で既定値）。
検出の閾値などのパラメータは、下の [EDIT HERE] で変える。

処理（詳しくは README_analysis.md）
-----------------------------------
1. manifest.json の main と irregular から、.done があるケースを選ぶ
2. 参照計算（1 回だけ）
   * ref_freespace: 直接波の |Ez| 最大の時刻 t_direct、送信波の極性（NPN でなければ停止）、
     波源の遅れ t_src = t_direct - (送受信点間隔 / c)、直接波のエンベロープの半値全幅 FWHM
   * ref_background - ref_freespace: 地表反射のエンベロープの最大値 A_surf
   * 検出の閾値 A_th = A_surf * 10^(THRESHOLD_DB / 20)
3. 各ケース
   * s = Ez(ケース) - Ez(ref_background)、e = |hilbert(s)|
   * 予測時刻 t_top, t_bot = theory.nominal の bistatic 伝搬時間 + t_src
   * e の極大点（A_th 以上）のうち t_top, t_bot に最も近いもの（±tol 以内）を上端・下端とする
   * エコーの範囲 = 極大点 ± FWHM/2（2 つの極大点の間隔が FWHM 未満なら、間の e の極小点で区切る）
   * 範囲内の電場のピーク（極値）のうち、予想される符号で絶対値が最大のものの時刻を tau_top, tau_bot とする
   * 判定: not_detected / single / partial (dtau < FWHM) / separated (dtau >= FWHM)
   * partial と separated だけ D_est = c * dtau / (2 sqrt(eps_true))、誤差 = (D_est - D) / D

出力（再実行するとすべて上書き）
--------------------------------
  <キャンペーン>/analysis/size_estimation/results.csv
  <キャンペーン>/analysis/size_estimation/analysis_config.json
  <キャンペーン>/analysis/size_estimation/size_estimate_<target>.png / .pdf
  <キャンペーン>/analysis/size_estimation/size_error_<target>.png / .pdf
  <ケースのフォルダ>/A-scan/size_estimation/<case_id>.png      （作らないことも選べる）

集計図は results.csv だけから作る。サイズ推定の下限は自動では決めない（集計図を見て決める）。
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import find_peaks, hilbert

import matplotlib
matplotlib.use("Agg")                                 # 図はファイルに保存するだけ
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402

# =============================================================================
# [EDIT HERE] 設定
# =============================================================================
# キャンペーンフォルダの既定値（実行時に Enter だけ押すとこれを使う）。
# このスクリプトがキャンペーンの scripts/ にあるときは、その親フォルダが既定値になる。
DEFAULT_CAMPAIGN = Path("/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_5x14/size_waveform_investigation/ascan_v01")

THRESHOLD_DB = -90.0      # 検出の閾値（地表反射のエンベロープの最大値に対する dB、振幅比）
TOLERANCE_NS = 1.0        # 予測時刻とエンベロープ極大点を対応づける許容差 [ns]
FWHM_NS = None            # 上端と下端の分離の基準 [ns]。None なら ref_freespace の直接波から測る
PLOT_MARGIN_NS = 5.0      # 確認用図の表示範囲（t_top - 5 ns から t_bot + 5 ns）
FONT_SIZE = 12            # 図の文字の大きさ（matplotlib の既定は 10）

# 集計図：深さの色（matplotlib の既定の色の順 C0 青、C1 橙、C2 緑。深さに固定で割り当てる）
DEPTH_COLORS = {1.0: "C0", 6.0: "C1", 12.0: "C2"}
EXTRA_COLORS = ["C3", "C4", "C5"]                     # 計画にない深さ用（順に割り当て）
# 集計図：形状の線種と記号（不規則な形状は線なし）
SHAPE_STYLES = {
    "circle":  dict(ls="-",  marker="o"),
    "square":  dict(ls="--", marker="s"),
    "irreg01": dict(ls="",   marker="^"),
    "irreg02": dict(ls="",   marker="v"),
}
OTHER_SHAPE = dict(ls="", marker="D")

# =============================================================================
ANALYSIS_VERSION = "1.1.0"
C0 = 299792458.0
NS = 1e-9
plt.rcParams["font.size"] = FONT_SIZE

CATEGORIES = ["not_detected", "single", "partial", "separated"]
CSV_FIELDS = [
    "case_id", "target", "kind", "shape", "depth_m",
    "size_cm", "size_effective_cm", "eps_true",
    "t_top_pred_ns", "t_bot_pred_ns",
    "env_top_ns", "env_bot_ns", "env_top_db", "env_bot_db",
    "env_top_offset_ns", "env_bot_offset_ns",
    "tau_top_ns", "tau_bot_ns", "dtau_ns",
    "category", "size_est_cm", "error_pct",
    "polarity_flag_top", "polarity_flag_bot",
]


# =============================================================================
# 読み込み
# =============================================================================
def read_ez(path: Path):
    """gprMax .out の受信点 1 の Ez と dt"""
    with h5py.File(path, "r") as f:
        ez = np.asarray(f["rxs/rx1/Ez"][()], dtype=float)
        dt = float(f.attrs["dt"])
    return ez, dt


def envelope(x: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(x))


def fwhm_of(env: np.ndarray, dt: float):
    """エンベロープの最大値の半値全幅 [s]（半値を横切る点は線形補間）"""
    k = int(np.argmax(env))
    half = 0.5 * env[k]
    i = k
    while i > 0 and env[i] >= half:
        i -= 1
    j = k
    while j < len(env) - 1 and env[j] >= half:
        j += 1
    if env[i] >= half or env[j] >= half:
        raise RuntimeError("FWHM: 半値の位置が時間窓の外にあります")
    t_left = (i + (half - env[i]) / (env[i + 1] - env[i])) * dt
    t_right = (j - 1 + (env[j - 1] - half) / (env[j - 1] - env[j])) * dt
    return t_right - t_left, k * dt


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# =============================================================================
# 参照計算
# =============================================================================
def process_references(camp: Path, man_rows, threshold_db: float, fwhm_override_ns):
    refs = {r["case_id"]: r for r in man_rows if r["group"] == "reference"}
    for cid in ("ref_background", "ref_freespace"):
        if cid not in refs:
            raise SystemExit(f"manifest.json に {cid} がありません")
        done = camp / Path(refs[cid]["json"]).parent / f"{cid}.done"
        if not done.exists():
            raise SystemExit(f"{cid} の計算が終わっていません（{done} がない）")

    ez_bg, dt_bg = read_ez(camp / refs["ref_background"]["output"])
    ez_fs, dt_fs = read_ez(camp / refs["ref_freespace"]["output"])
    if dt_bg != dt_fs or len(ez_bg) != len(ez_fs):
        raise SystemExit("ref_background と ref_freespace の dt または長さが違います")
    dt = dt_fs
    fs_json = json.loads((camp / refs["ref_freespace"]["json"]).read_text(encoding="utf-8"))
    t_direct_theory = float(fs_json["theory"]["common"]["direct_wave_s"])   # = 送受信点間隔 / c

    # 直接波：|Ez| 最大のピークと極性
    k = int(np.argmax(np.abs(ez_fs)))
    t_direct = k * dt
    env_fs = envelope(ez_fs)
    fwhm_meas, _ = fwhm_of(env_fs, dt)
    w = max(1, int(round(fwhm_meas / dt)))                # 前後の負のピークを探す幅（1 FWHM）
    pre = ez_fs[max(0, k - w):k]
    post = ez_fs[k + 1:k + 1 + w]
    peak = ez_fs[k]
    lobe_pre = float(pre.min()) if pre.size else 0.0
    lobe_post = float(post.min()) if post.size else 0.0
    ok_npn = peak > 0 and lobe_pre < -0.1 * peak and lobe_post < -0.1 * peak
    polarity = "NPN" if ok_npn else "not NPN"
    if not ok_npn:
        raise SystemExit(f"送信波の極性が NPN ではありません（peak {peak:.3e}, 前 {lobe_pre:.3e}, 後 {lobe_post:.3e}）。"
                         "予想される符号の前提が崩れるので停止します。")
    t_src = t_direct - t_direct_theory

    fwhm = fwhm_override_ns * NS if fwhm_override_ns is not None else fwhm_meas

    # 地表反射
    surf = ez_bg - ez_fs
    env_surf = envelope(surf)
    k_s = int(np.argmax(env_surf))
    a_surf = float(env_surf[k_s])
    a_th = a_surf * 10.0 ** (threshold_db / 20.0)

    return dict(
        ez_background=ez_bg, dt=dt, n_samples=len(ez_bg),
        t_direct_s=t_direct, t_direct_theory_s=t_direct_theory, t_src_s=t_src,
        transmitted_polarity=polarity,
        direct_peak=dict(peak=float(peak), lobe_before=lobe_pre, lobe_after=lobe_post),
        fwhm_measured_s=fwhm_meas, fwhm_used_s=fwhm,
        surface_peak_time_s=k_s * dt, a_surf=a_surf, a_th=a_th,
    )


# =============================================================================
# 1 ケースの解析
# =============================================================================
def expected_sign(label: str) -> int:
    """'PNP' -> -1（中央の主ピークが負）、'NPN' -> +1"""
    return 1 if label[1] == "P" else -1


def pick_peak(s: np.ndarray, i0: int, i1: int, sign: int, dt: float):
    """
    範囲 [i0, i1] の電場のピーク（極値）のうち、予想される符号で絶対値が最大のものの時刻と、
    範囲内で絶対値が最大のピークの符号が予想と逆かどうか。
    範囲の端の値はピークとみなさない（エコーの裾を拾わないため）。予想される符号のピークがなければ時刻は None。
    """
    seg = s[i0:i1 + 1]
    pk_pos, _ = find_peaks(seg)
    pk_neg, _ = find_peaks(-seg)
    pk = np.r_[pk_pos, pk_neg]
    if pk.size == 0:
        return None, None
    k_abs = int(pk[np.argmax(np.abs(seg[pk]))])
    flag = bool(np.sign(seg[k_abs]) != sign)
    cand = pk_pos if sign > 0 else pk_neg
    if cand.size == 0:
        return None, flag
    k = int(cand[np.argmax(np.abs(seg[cand]))])
    return (i0 + k) * dt, flag


def analyze_case(camp: Path, d: dict, ref: dict, tol_s: float):
    dt = ref["dt"]
    ez, dt_c = read_ez(camp / d["files"]["output"])
    if dt_c != dt or len(ez) != ref["n_samples"]:
        raise RuntimeError("dt または長さが ref_background と違います")
    s = ez - ref["ez_background"]
    e = envelope(s)
    fwhm = ref["fwhm_used_s"]

    th = d["theory"]["nominal"]
    t_top = th["top"]["t_bistatic_s"] + ref["t_src_s"]
    t_bot = th["bottom"]["t_bistatic_s"] + ref["t_src_s"]
    sign_top = expected_sign(th["top"]["expected_polarity"])
    sign_bot = expected_sign(th["bottom"]["expected_polarity"])

    # 段階 1：エンベロープの極大点と対応づけ
    peaks, _ = find_peaks(e, height=ref["a_th"])
    tp = peaks * dt

    def nearest(t):
        if peaks.size == 0:
            return None
        k = int(np.argmin(np.abs(tp - t)))
        return int(peaks[k]) if abs(tp[k] - t) <= tol_s else None

    p_top, p_bot = nearest(t_top), nearest(t_bot)

    # エコーの範囲（サンプル番号）
    h = int(round(0.5 * fwhm / dt))
    n = len(s)
    win = {}
    if p_top is not None:
        win["top"] = [max(0, p_top - h), min(n - 1, p_top + h)]
    if p_bot is not None:
        win["bot"] = [max(0, p_bot - h), min(n - 1, p_bot + h)]
    if p_top is not None and p_bot is not None and p_bot > p_top and (p_bot - p_top) * dt < fwhm:
        k_min = p_top + int(np.argmin(e[p_top:p_bot + 1]))
        win["top"][1] = k_min
        win["bot"][0] = k_min

    # 段階 2：電場のピーク
    tau_top = tau_bot = None
    flag_top = flag_bot = None
    if "top" in win:
        tau_top, flag_top = pick_peak(s, *win["top"], sign_top, dt)
    if "bot" in win:
        tau_bot, flag_bot = pick_peak(s, *win["bot"], sign_bot, dt)

    # 判定
    if p_top is None and p_bot is None:
        cat = "not_detected"
    elif (p_top is None or p_bot is None or p_top == p_bot
          or tau_top is None or tau_bot is None or tau_bot <= tau_top):
        cat = "single"
    elif tau_bot - tau_top < fwhm:
        cat = "partial"
    else:
        cat = "separated"

    t = d["target"]
    eps = float(d["materials"]["target"]["eps_r"])
    size_m = float(t["size_m"])
    dtau = (tau_bot - tau_top) if (tau_top is not None and tau_bot is not None) else None
    est = err = None
    if cat in ("partial", "separated"):
        est = C0 * dtau / (2.0 * math.sqrt(eps))
        err = (est - size_m) / size_m * 100.0
    nadir = (t.get("as_built") or {}).get("nadir") or {}

    def ns(x):
        return None if x is None else x / NS

    def db(p):
        return None if p is None else 20.0 * math.log10(e[p] / ref["a_surf"])

    row = dict(
        case_id=d["case_id"], target=t["material"], kind=t["kind"], shape=t["shape"],
        depth_m=t["depth_top_m"], size_cm=t["size_cm"],
        size_effective_cm=(nadir["effective_thickness_m"] * 100.0 if "effective_thickness_m" in nadir else None),
        eps_true=eps,
        t_top_pred_ns=ns(t_top), t_bot_pred_ns=ns(t_bot),
        env_top_ns=ns(None if p_top is None else p_top * dt),
        env_bot_ns=ns(None if p_bot is None else p_bot * dt),
        env_top_db=db(p_top), env_bot_db=db(p_bot),
        env_top_offset_ns=ns(None if p_top is None else p_top * dt - t_top),
        env_bot_offset_ns=ns(None if p_bot is None else p_bot * dt - t_bot),
        tau_top_ns=ns(tau_top), tau_bot_ns=ns(tau_bot), dtau_ns=ns(dtau),
        category=cat,
        size_est_cm=None if est is None else est * 100.0,
        error_pct=err,
        polarity_flag_top=flag_top, polarity_flag_bot=flag_bot,
    )
    trace = dict(s=s, e=e, dt=dt, win=win, t_top=t_top, t_bot=t_bot, tau_top=tau_top, tau_bot=tau_bot)
    return row, trace


# =============================================================================
# 確認用 A-scan 図
# =============================================================================
def fmt(x, f="{:.2f}", none="-"):
    return none if x is None else f.format(x)


def plot_ascan(path: Path, row: dict, tr: dict, ref: dict, margin_s: float):
    """
    確認用 A-scan 図。黒 = 背景を差し引いた Ez、緑の破線 = エンベロープ、黒の点線 = 予測時刻、
    青 / 赤 = 上端 / 下端（エコーの範囲の帯と、選んだピーク）。右側に結果を書く。
    """
    dt, a = tr["dt"], ref["a_surf"]
    t0 = tr["t_top"] - margin_s
    t1 = tr["t_bot"] + margin_s
    i0, i1 = max(0, int(t0 / dt)), min(len(tr["s"]) - 1, int(math.ceil(t1 / dt)))
    t = np.arange(i0, i1 + 1) * dt / NS
    s = tr["s"][i0:i1 + 1] / a
    e = tr["e"][i0:i1 + 1] / a

    fig = plt.figure(figsize=(11.0, 5.0))
    ax = fig.add_axes([0.08, 0.13, 0.58, 0.80])           # 右側の余白に凡例と結果の文字を置く
    for key, col in (("top", "b"), ("bot", "r")):
        if key in tr["win"]:
            w0, w1 = tr["win"][key]
            ax.axvspan(w0 * dt / NS, w1 * dt / NS, color=col, alpha=0.1, lw=0)
    for tp in (tr["t_top"], tr["t_bot"]):
        ax.axvline(tp / NS, color="k", ls=":")
    ax.plot(t, s, "k-", label="Ez (background removed)")
    ax.plot(t, e, "g--", label="envelope")
    ax.plot(t, -e, "g--")
    for tau, col, lab in ((tr["tau_top"], "b", "top peak"), (tr["tau_bot"], "r", "bottom peak")):
        if tau is not None:
            k = int(round(tau / dt))
            ax.plot(tau / NS, tr["s"][k] / a, "o", color=col, ms=8, label=lab, zorder=5)
    ax.set_xlim(t0 / NS, t1 / NS)
    ax.set_xlabel("Time from simulation start (ns)")
    ax.set_ylabel("Ez / A_surf")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.legend(loc="lower left", bbox_to_anchor=(1.03, 0.0), borderaxespad=0)

    flags = [k for k, v in (("top", row["polarity_flag_top"]), ("bottom", row["polarity_flag_bot"])) if v]
    dt_pred = row["t_bot_pred_ns"] - row["t_top_pred_ns"]
    txt = "\n".join([
        row["case_id"],
        "",
        f"D          {row['size_cm']} cm",
        f"eps_true   {row['eps_true']:.4g}",
        f"dt pred    {dt_pred:.3f} ns",
        f"dtau       {fmt(row['dtau_ns'], '{:.3f}')} ns",
        f"D_est      {fmt(row['size_est_cm'])} cm",
        f"error      {fmt(row['error_pct'], '{:+.1f}')} %",
        f"category   {row['category']}",
        "polarity   " + ("WARNING " + ", ".join(flags) if flags else "ok"),
    ])
    ax.text(1.03, 1.0, txt, transform=ax.transAxes, va="top", ha="left", family="monospace",
            fontsize=FONT_SIZE - 1, linespacing=1.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# CSV と集計図
# =============================================================================
def write_csv(path: Path, rows):
    def cell(v):
        if v is None:
            return ""
        if isinstance(v, bool):
            return "1" if v else "0"
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_FIELDS)
        for r in rows:
            w.writerow([cell(r[k]) for k in CSV_FIELDS])


def read_csv(path: Path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def depth_color(depth: float, extra: dict) -> str:
    for k, c in DEPTH_COLORS.items():
        if abs(k - depth) < 1e-9:
            return c
    if depth not in extra:
        extra[depth] = EXTRA_COLORS[len(extra) % len(EXTRA_COLORS)]
    return extra[depth]


def plot_summary(csv_path: Path, out_dir: Path):
    """results.csv から、対象ごとに推定サイズの図と誤差の図を作る（色 = 深さ、線種と記号 = 形状）"""
    rows = read_csv(csv_path)
    written = []
    for target in sorted({r["target"] for r in rows}):
        est = [r for r in rows if r["target"] == target and r["category"] in ("partial", "separated")]
        if not est:
            print(f"  集計図: {target} は推定できたケースがないので作りません")
            continue
        eps = float(est[0]["eps_true"])
        depths = sorted({float(r["depth_m"]) for r in est})
        shapes = sorted({r["shape"] for r in est}, key=lambda s: (list(SHAPE_STYLES).index(s)
                                                                  if s in SHAPE_STYLES else 99, s))
        extra = {}
        for kind, ycol, ylab, fname in (("size", "size_est_cm", "Estimated size (cm)", "size_estimate"),
                                        ("error", "error_pct", "Estimation error (%)", "size_error")):
            fig, ax = plt.subplots(figsize=(8.0, 5.5))
            xmax = max(float(r["size_cm"]) for r in est)
            if kind == "size":
                ax.plot([0, xmax * 1.05], [0, xmax * 1.05], "k:", label="1:1")
            else:
                ax.axhline(0.0, color="k", ls=":")
            for dep in depths:
                col = depth_color(dep, extra)
                for shp in shapes:
                    st = SHAPE_STYLES.get(shp, OTHER_SHAPE)
                    pts = sorted(((float(r["size_cm"]), float(r[ycol]), r["category"]) for r in est
                                  if r["shape"] == shp and abs(float(r["depth_m"]) - dep) < 1e-9))
                    if not pts:
                        continue
                    if st["ls"]:
                        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, ls=st["ls"])
                    for xi, yi, c in pts:                  # 塗りつぶし = separated、白抜き = partial
                        ax.plot(xi, yi, st["marker"], color=col, ms=8,
                                mfc=col if c == "separated" else "w", zorder=3)
            ax.set_xlim(0, xmax * 1.05)
            if kind == "size":
                ax.set_ylim(bottom=0)
            ax.set_xlabel("True size (height, cm)")
            ax.set_ylabel(ylab)
            ax.set_title(f"{target} (eps = {eps:.4g})")
            ax.grid(alpha=0.3)

            h = [Line2D([], [], color=depth_color(dep, extra), label=f"depth {dep:g} m") for dep in depths]
            h += [Line2D([], [], color="k", ls=SHAPE_STYLES.get(s, OTHER_SHAPE)["ls"] or "none",
                         marker=SHAPE_STYLES.get(s, OTHER_SHAPE)["marker"], label=s) for s in shapes]
            h += [Line2D([], [], color="k", ls="none", marker="o", label="separated"),
                  Line2D([], [], color="k", ls="none", marker="o", mfc="w", label="partial")]
            ax.legend(handles=h, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                p = out_dir / f"{fname}_{target}.{ext}"
                fig.savefig(p, dpi=200, bbox_inches="tight")
                written.append(p)
            plt.close(fig)
    return written


# =============================================================================
# 対話での選択
# =============================================================================
def ask(prompt: str, default: str = "") -> str:
    """input() の薄いラッパー。Enter だけなら既定値"""
    try:
        ans = input(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    except EOFError:
        ans = ""
    return ans or default


def clean_path(text: str) -> Path:
    """Finder からドラッグしたパス（前後の引用符、空白の前の \\ ）も受け付ける"""
    text = text.strip().strip("'\"").replace("\\ ", " ")
    return Path(text).expanduser()


def choose(title: str, options):
    """
    options: [(値, 表示), ...]。番号または値をカンマ・スペース区切りで入力。Enter ならすべて。
    選んだ値のリストを返す。
    """
    print(f"\n{title}")
    for i, (_, label) in enumerate(options, 1):
        print(f"  {i:2d}. {label}")
    values = [v for v, _ in options]
    while True:
        ans = ask("  番号（カンマかスペース区切り、Enter = すべて）")
        if not ans:
            return values
        picked, bad = [], []
        for tok in ans.replace(",", " ").split():
            if tok.isdigit() and 1 <= int(tok) <= len(values):
                v = values[int(tok) - 1]
            elif tok in [str(x) for x in values]:
                v = values[[str(x) for x in values].index(tok)]
            else:
                bad.append(tok)
                continue
            if v not in picked:
                picked.append(v)
        if bad or not picked:
            print(f"  選べない入力です: {' '.join(bad) or ans}")
            continue
        return picked


def yes_no(prompt: str, default: bool = True) -> bool:
    while True:
        ans = ask(prompt + (" (Y/n)" if default else " (y/N)")).lower()
        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False


def count_label(rows, is_done) -> str:
    n_done = sum(1 for r in rows if is_done(r))
    return f"計算済み {n_done:3d} / {len(rows):3d}"


# =============================================================================
# main
# =============================================================================
def main() -> int:
    print("=== A-scan サイズ推定解析 ===")

    # キャンペーンフォルダ
    here = Path(__file__).resolve().parent
    default = here.parent if (here.name == "scripts" and (here.parent / "manifest.json").exists()) \
        else DEFAULT_CAMPAIGN
    while True:
        camp = clean_path(ask("キャンペーンフォルダ（manifest.json がある場所）", str(default))).resolve()
        if (camp / "manifest.json").exists():
            break
        print(f"  {camp / 'manifest.json'} がありません。もう一度入力してください。")
    man = json.loads((camp / "manifest.json").read_text(encoding="utf-8"))
    rows_all = man["cases"]
    cases = [r for r in rows_all if r["group"] in ("main", "irregular")]

    def is_done(r):
        return (camp / Path(r["json"]).parent / f"{r['case_id']}.done").exists()

    # 対象・形状・深さ（前の選択で絞った中から選ぶ）
    targets = list(dict.fromkeys(r["target"] for r in cases))
    targets = choose("対象（target）", [(t, f"{t:10s} {count_label([r for r in cases if r['target'] == t], is_done)}")
                                      for t in targets])
    cases = [r for r in cases if r["target"] in targets]
    shapes = list(dict.fromkeys(r["shape"] for r in cases))
    shapes = choose("形状（shape）", [(s, f"{s:10s} {count_label([r for r in cases if r['shape'] == s], is_done)}")
                                    for s in shapes])
    cases = [r for r in cases if r["shape"] in shapes]
    depths = sorted({float(r["depth_m"]) for r in cases})
    depths = choose("深さ（対象上端、m）",
                    [(d, f"{d:g} m{'':7s} {count_label([r for r in cases if float(r['depth_m']) == d], is_done)}")
                     for d in depths])
    cases = [r for r in cases if float(r["depth_m"]) in depths]

    sel = [r for r in cases if is_done(r)]
    pending = [r for r in cases if not is_done(r)]
    print(f"\n解析するケース: {len(sel)}（未計算で飛ばすケース: {len(pending)}）")
    if not sel:
        print("解析できるケースがありません。")
        return 1
    make_plots = yes_no("ケースごとの確認用 A-scan 図を作りますか")
    print(f"パラメータ: 閾値 {THRESHOLD_DB:g} dB, 許容差 {TOLERANCE_NS:g} ns, "
          f"FWHM {'自動' if FWHM_NS is None else f'{FWHM_NS:g} ns'}, 図の余白 {PLOT_MARGIN_NS:g} ns"
          "（変えるときはスクリプト先頭の [EDIT HERE]）")
    if not yes_no("実行しますか"):
        print("中止しました。")
        return 0

    selection = dict(targets=targets, shapes=shapes, depths_m=depths, ascan_plots=make_plots)
    return run_analysis(camp, rows_all, sel, pending, make_plots, selection)


def run_analysis(camp: Path, rows_all, sel, pending, make_plots: bool, selection: dict) -> int:
    # 2. 参照計算
    print()
    ref = process_references(camp, rows_all, THRESHOLD_DB, FWHM_NS)
    print(f"transmitted polarity: {ref['transmitted_polarity']}")
    print(f"t_direct = {ref['t_direct_s']/NS:.4f} ns, t_src = {ref['t_src_s']/NS:.4f} ns")
    print(f"FWHM = {ref['fwhm_measured_s']/NS:.4f} ns (measured)"
          + ("" if FWHM_NS is None else f", {FWHM_NS:.4f} ns (used, FWHM_NS)"))
    print(f"A_surf = {ref['a_surf']:.4e} at {ref['surface_peak_time_s']/NS:.3f} ns, "
          f"A_th = {ref['a_th']:.4e} ({THRESHOLD_DB:g} dB)\n")

    out_dir = camp / "analysis" / "size_estimation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. 各ケース
    results, failed = [], []
    for r in sel:
        d = json.loads((camp / r["json"]).read_text(encoding="utf-8"))
        try:
            row, tr = analyze_case(camp, d, ref, TOLERANCE_NS * NS)
        except Exception as ex:                                    # 1 ケースの失敗で全体を止めない
            failed.append((r["case_id"], str(ex)))
            print(f"  FAILED {r['case_id']}: {ex}")
            continue
        results.append(row)
        if make_plots:
            png = camp / Path(d["files"]["output"]).parent / "size_estimation" / f"{d['case_id']}.png"
            plot_ascan(png, row, tr, ref, PLOT_MARGIN_NS * NS)
        warn = [k for k in ("polarity_flag_top", "polarity_flag_bot") if row[k]]
        print(f"  {row['case_id']:34s} {row['category']:12s} "
              f"D_est {fmt(row['size_est_cm'], '{:7.2f}', '      -')} cm  "
              f"err {fmt(row['error_pct'], '{:+7.1f}', '      -')} %"
              + ("  [polarity: " + ",".join(w.split('_')[-1] for w in warn) + "]" if warn else ""))

    # 4. 出力
    csv_path = out_dir / "results.csv"
    write_csv(csv_path, results)
    counts = {c: sum(1 for x in results if x["category"] == c) for c in CATEGORIES}
    script = Path(__file__).resolve()
    cfg = dict(
        analysis="size_estimation", version=ANALYSIS_VERSION, script_sha256=sha256_of(script),
        run_at=_dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        campaign=str(camp),
        selection=dict(selection, n_analyzed=len(results), n_not_computed=len(pending),
                       failed=[dict(case_id=c, error=m) for c, m in failed]),
        parameters=dict(threshold_db=THRESHOLD_DB, tolerance_ns=TOLERANCE_NS,
                        fwhm_ns_setting=FWHM_NS, plot_margin_ns=PLOT_MARGIN_NS,
                        predicted_times="theory.nominal.{top,bottom}.t_bistatic_s + t_src",
                        size_formula="D_est = c * dtau / (2 sqrt(eps_true)); error = (D_est - D) / D * 100"),
        references=dict(
            transmitted_polarity=ref["transmitted_polarity"], direct_peak=ref["direct_peak"],
            t_direct_ns=ref["t_direct_s"] / NS, t_direct_theory_ns=ref["t_direct_theory_s"] / NS,
            t_src_ns=ref["t_src_s"] / NS,
            fwhm_measured_ns=ref["fwhm_measured_s"] / NS, fwhm_used_ns=ref["fwhm_used_s"] / NS,
            surface_peak_ns=ref["surface_peak_time_s"] / NS, a_surf=ref["a_surf"], a_th=ref["a_th"],
            dt_s=ref["dt"], n_samples=ref["n_samples"]),
        counts=counts,
    )
    (out_dir / "analysis_config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
    figs = plot_summary(csv_path, out_dir)

    print("\n" + ", ".join(f"{c} {n}" for c, n in counts.items()) + (f", FAILED {len(failed)}" if failed else ""))
    print(f"written: {csv_path}")
    print(f"         {out_dir / 'analysis_config.json'}")
    for f in figs:
        if f.suffix == ".png":
            print(f"         {f} (+ .pdf)")
    if make_plots:
        print("         <case>/A-scan/size_estimation/<case_id>.png")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())