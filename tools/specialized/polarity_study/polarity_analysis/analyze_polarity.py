#!/usr/bin/env python3
"""
analyze_polarity.py  --  段階 1: 極性判定
==========================================

キャンペーンフォルダ（generate_polarity_inputs.py が作ったもの）の .out を読み、
原稿 2.5 節の手順で上面と底面のエコーの極性を判定する。

    python analyze_polarity.py [キャンペーンフォルダ] [オプション]

出力（<キャンペーン>/analysis/<解析名>/）
  polarity_results.csv    1 ケース・1 面につき 1 行（判定に使った値と結果）
  slab_check.csv          平行平板計算の判定（判定法の確認）
  analysis_settings.json  判定の設定、delta_t、送信波形の切り出し範囲など（原稿に書く値の控え）
  traces.npz              背景を除いた全ケースの波形 x(t) と送信波形 s(t)
  figs/transmitted_waveform.png   送信波形、切り出し範囲、自己相関と t' の範囲
  ascan/<パターン>/<case_id>.png  ケースごとの A-scan と判定の図

まとめの図は plot_polarity_maps.py（段階 2）で描く。

オプション
  --name NAME          解析名（既定は ANALYSIS_NAME）
  --groups ...         slab nominal depth のうち対象にするもの
  --targets ...        rock void
  --cases ID ...       ケース ID を直接指定
  --no-ascan           A-scan の図を描かない（判定と csv だけ。速い）
  --force              設定が前回と違っても上書きする
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import polarity_core as pc   # noqa: E402

SCRIPT_VERSION = "1.0.0"

# =============================================================================
# [EDIT HERE] 解析の設定（原稿 2.5 節）
# =============================================================================
CAMPAIGN_DIR = Path("/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_10x10/Polarity_EPS/polarity_v01")
ANALYSIS_NAME = "ncc_v01"

T_W_S = 1.5e-9            # 時間窓の半幅 T_w
LAG_MAX_S = 0.45e-9       # t' の範囲 |t' - tau| <= LAG_MAX
RHO_MIN = 0.5             # これ未満の |rho| は Type 3（段階 2 で変えられる）
TEMPLATE_LEVEL = 0.03     # 送信波形の切り出し: 振幅が最大値の 3% 以上
ARRIVAL_DIMENSIONS = "nominal"   # 到達時刻に使う寸法: "nominal"（表 2）または "as_built_effective"

ASCAN_TAIL_S = 8e-9       # A-scan 図: 底面の到達時刻の後に描く長さ
ZOOM_HALF_S = 2.5e-9      # A-scan 図: 拡大図の半幅
# =============================================================================

FACES = ("top", "bottom")

CSV_FIELDS = [
    "case_id", "group", "target", "pattern", "depth_m", "height_m", "width_m", "pair_key", "face",
    "expected_polarity", "reflection_coefficient",
    "t_nadir_ns", "delta_t_ns", "tau_ns", "window_start_ns", "window_end_ns",
    "tau_as_built_ns", "windows_overlap",
    "fresnel_radius_m", "fresnel_diameter_m", "width_over_fresnel_diameter",
    "rho_best", "lag_best_ns", "rho_at_tau", "rho_pos_max", "rho_neg_min",
    "observed_polarity", "type", "type_reason",
    "amplitude_ls", "window_peak_abs", "window_rms",
]


# -----------------------------------------------------------------------------
def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def fmt_hms(sec: float) -> str:
    sec = int(round(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("campaign", nargs="?", type=Path, default=CAMPAIGN_DIR)
    p.add_argument("--name", default=ANALYSIS_NAME)
    p.add_argument("--groups", nargs="+", choices=["slab", "nominal", "depth"])
    p.add_argument("--targets", nargs="+", choices=["rock", "void"])
    p.add_argument("--cases", nargs="+")
    p.add_argument("--no-ascan", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pattern_of(d: Dict) -> str:
    """rock/depth02m、void/depth04m、slab/rock など（フォルダの上 2 階層）"""
    return "/".join(d["files"]["dir"].split("/")[:2])


# -----------------------------------------------------------------------------
# 図
# -----------------------------------------------------------------------------
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "figure.dpi": 100})
    return plt


TYPE_COLOR = {1: "#d62728", 2: "#1f77b4", 3: "0.45"}
TYPE_LABEL = {1: "Type 1", 2: "Type 2", 3: "Type 3"}


def plot_template(tpl: pc.Template, ref: pc.Trace, lag_max_s: float, out: Path) -> None:
    plt = _mpl()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    t = ref.t * 1e9
    a1.plot(t, ref.ez / abs(tpl.peak_amplitude), color="0.6", lw=1, label="ref_freespace (normalized)")
    t0 = tpl.peak_time_s * 1e9
    k0 = int(round(tpl.peak_time_s / tpl.dt)) - tpl.peak_index
    ts = (np.arange(tpl.s.size) + k0) * tpl.dt * 1e9
    a1.plot(ts, tpl.s, color="k", lw=1.4, label=f"s(t): |s| >= {tpl.level:.0%} of max")
    a1.axvline(t0, color="r", ls="--", lw=0.8)
    a1.set_xlim(t0 - 4, t0 + 6)
    a1.set_xlabel("Time (ns)")
    a1.set_ylabel("Normalized Ez")
    a1.set_title(f"Transmitted waveform ({tpl.polarity}); main peak {t0:.3f} ns, "
                 f"delta_t = {tpl.delta_t_s*1e9:.3f} ns")
    a1.legend(loc="upper right", fontsize=7)
    lag, ac = tpl.autocorrelation()
    a2.plot(lag * 1e9, ac, color="k", lw=1.2)
    a2.axvspan(-lag_max_s * 1e9, lag_max_s * 1e9, color="0.85", label=f"|t' - tau| <= {lag_max_s*1e9:.2f} ns")
    z = tpl.first_zero_of_autocorrelation_s() * 1e9
    a2.axvline(z, color="r", ls=":", lw=0.8)
    a2.axvline(-z, color="r", ls=":", lw=0.8, label=f"first zero {z:.2f} ns")
    a2.axhline(0, color="0.5", lw=0.5)
    a2.set_xlim(-2.5, 2.5)
    a2.set_xlabel("Lag (ns)")
    a2.set_ylabel("Normalized autocorrelation")
    a2.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_ascan(path: Path, d: Dict, t: np.ndarray, x: np.ndarray, tpl: pc.Template,
               faces: Dict[str, Dict], t_w_s: float, lag_max_s: float) -> None:
    plt = _mpl()
    fig = plt.figure(figsize=(9, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.42, wspace=0.22)
    ax = fig.add_subplot(gs[0, :])
    tn = t * 1e9
    t_end = min(t[-1], faces["bottom"]["res"].tau_s + ASCAN_TAIL_S)
    m = t <= t_end
    ax.plot(tn[m], x[m], color="k", lw=0.8)
    ymax = max(np.max(np.abs(x[m])), 1e-30)
    for face, col in (("top", "#9467bd"), ("bottom", "#2ca02c")):
        r = faces[face]["res"]
        ax.axvspan(r.window_s[0] * 1e9, r.window_s[1] * 1e9, color=col, alpha=0.15, lw=0)
        ax.axvline(r.tau_s * 1e9, color=col, ls="--", lw=0.8)
        ax.text(r.tau_s * 1e9, 1.02 * ymax, face, color=col, ha="center", va="bottom", fontsize=8)
    ax.set_xlim(0, t_end * 1e9)
    ax.set_ylim(-1.15 * ymax, 1.15 * ymax)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Ez - background (V/m)")
    t_info = d["target"]
    w = t_info.get("width_m")
    size = f"h = {t_info['height_m']:.3f} m, " + ("w = inf (slab)" if w is None else f"w = {w:.2f} m")
    ax.set_title(f"{d['case_id']}   ({d['target']['kind']}, d = {t_info['depth_top_m']:.1f} m, {size})")

    tt = tpl.t_rel
    for q, face in enumerate(FACES):
        a = fig.add_subplot(gs[1, q])
        f = faces[face]
        r = f["res"]
        lo, hi = r.tau_s - ZOOM_HALF_S, r.tau_s + ZOOM_HALF_S
        mm = (t >= lo) & (t <= hi)
        a.axvspan(r.window_s[0] * 1e9, r.window_s[1] * 1e9, color="0.9", lw=0)
        a.axvspan((r.tau_s - lag_max_s) * 1e9, (r.tau_s + lag_max_s) * 1e9, ymin=0, ymax=0.04, color="k", lw=0)
        a.plot(tn[mm], x[mm], color="k", lw=1.0, label="x(t)")
        a.axvline(r.tau_s * 1e9, color="0.3", ls="--", lw=0.8)
        if np.isfinite(r.rho_best):
            tp = r.tau_s + r.lag_best_s
            a.plot((tt + tp) * 1e9, r.amplitude_ls * tpl.s, color="r", lw=1.0, alpha=0.9,
                   label="fitted s(t - t')")
        loc_max = np.max(np.abs(x[mm])) if mm.any() else 1.0
        a.set_xlim(lo * 1e9, hi * 1e9)
        a.set_ylim(-1.3 * loc_max, 2.2 * loc_max)        # 上に数値を書く余白
        a.set_xlabel("Time (ns)")
        ty = f["type"]
        reason = f" ({f['reason']})" if f["reason"] else ""
        a.set_title(f"{face}: expected {f['expected']}, observed {f['observed'] or '-'}  ->  "
                    f"{TYPE_LABEL[ty]}{reason}", color=TYPE_COLOR[ty])
        txt = (f"rho = {r.rho_best:+.2f}  (t' - tau = {r.lag_best_s*1e9:+.2f} ns)\n"
               f"rho at tau = {r.rho_at_tau:+.2f}")
        if f.get("w_over_2rf") is not None:
            txt += f"\nw / 2r_F = {f['w_over_2rf']:.2f}"
        a.text(0.02, 0.97, txt, transform=a.transAxes, va="top", fontsize=7.5)
        if q == 0:
            a.set_ylabel("Ez - background (V/m)")
            a.legend(loc="upper right", fontsize=7, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
def main(argv=None) -> int:
    a = parse_args(argv)
    camp = a.campaign.resolve()
    if not (camp / "manifest.json").exists():
        print(f"manifest.json not found in {camp}")
        return 1
    out_dir = camp / "analysis" / a.name
    man = load_json(camp / "manifest.json")
    rows_man = man["cases"]

    # --- 参照計算 ------------------------------------------------------------
    by_id = {r["case_id"]: r for r in rows_man}
    for rid in ("ref_background", "ref_freespace"):
        if rid not in by_id or not (camp / by_id[rid]["output"]).exists():
            print(f"{rid}: .out not found. Run the reference cases first.")
            return 1
    fs_json = load_json(camp / by_id["ref_freespace"]["json"])
    tx, rx = np.array(fs_json["antenna"]["tx_xy_m"]), np.array(fs_json["antenna"]["rx_xy_m"])
    distance = float(np.hypot(*(rx - tx)))
    ref_fs = pc.read_out(camp / by_id["ref_freespace"]["output"])
    ref_bg = pc.read_out(camp / by_id["ref_background"]["output"])
    tpl = pc.build_template(ref_fs, distance, level=TEMPLATE_LEVEL)
    transmitted = tpl.polarity
    assumed = fs_json["theory"].get("transmitted_polarity_assumed", transmitted)
    if transmitted != assumed:
        print(f"WARNING: transmitted waveform is {transmitted}, but the campaign assumed {assumed}. "
              f"Expected polarities in the case JSON may be wrong.")
    z0 = tpl.first_zero_of_autocorrelation_s()
    if LAG_MAX_S >= z0:
        print(f"WARNING: LAG_MAX ({LAG_MAX_S*1e9:.2f} ns) >= first zero of autocorrelation ({z0*1e9:.2f} ns)")

    # Fresnel の線を描くための定数（段階 2 で使う）
    some = next((r for r in rows_man if r["group"] in ("nominal", "depth")), None)
    fres_consts = {}
    if some:
        dj = load_json(camp / some["json"])
        fres_consts = dict(antenna_height_m=dj["antenna"]["height_m"],
                           wavelength_m=dj["theory"]["fresnel_wavelength_m"],
                           eps_background=dj["materials"]["background"]["eps_r"])
    eps_targets = {}
    for r in rows_man:
        if r["group"] in ("nominal", "depth") and r["target"] not in eps_targets:
            eps_targets[r["target"]] = load_json(camp / r["json"])["materials"]["target"]["eps_r"]
    fres_consts["eps_target"] = eps_targets

    settings = dict(
        campaign=man.get("campaign"), campaign_dir=str(camp), analysis_name=a.name,
        script=dict(name=Path(__file__).name, version=SCRIPT_VERSION,
                    sha256={p.name: sha256_of(p) for p in (Path(__file__).resolve(), HERE / "polarity_core.py")}),
        method=dict(
            background_subtraction="x(t) = Ez(case) - Ez(ref_background)",
            t_w_s=T_W_S, lag_max_s=LAG_MAX_S, rho_min=RHO_MIN,
            template_level=TEMPLATE_LEVEL, arrival_dimensions=ARRIVAL_DIMENSIONS,
            overlap_rule="Type 3 for both faces if tau_bottom - tau_top < 2 T_w",
            polarity_rule="sign of rho at max |rho|; rho > 0 = same as transmitted waveform",
        ),
        transmitted_waveform=dict(
            polarity=transmitted, distance_m=distance, peak_time_s=tpl.peak_time_s,
            delta_t_s=tpl.delta_t_s, peak_amplitude_v_per_m=tpl.peak_amplitude,
            crop_t_rel_s=[float(tpl.t_rel[0]), float(tpl.t_rel[-1])],
            side_peaks=tpl.side_peaks, autocorrelation_first_zero_s=z0,
            dt_s=tpl.dt, n_samples=int(tpl.s.size),
        ),
        fresnel_constants=fres_consts,
    )

    # 前回の設定との比較
    old_settings = None
    if (out_dir / "analysis_settings.json").exists():
        old_settings = load_json(out_dir / "analysis_settings.json")
        same = (old_settings.get("method") == json.loads(json.dumps(settings["method"])) and
                abs(old_settings["transmitted_waveform"]["delta_t_s"] - tpl.delta_t_s) < 1e-15)
        if not same and not a.force:
            print(f"Settings differ from the previous analysis in {out_dir}.\n"
                  f"Use another --name, or --force to overwrite.")
            return 1
        if not same:
            old_settings = None          # 前回の行は引き継がない

    # --- 対象のケース --------------------------------------------------------
    todo = []
    for r in rows_man:
        if r["group"] == "reference":
            continue
        if a.groups and r["group"] not in a.groups:
            continue
        if a.targets and r["target"] not in a.targets:
            continue
        if a.cases and r["case_id"] not in a.cases:
            continue
        todo.append(r)

    print(f"campaign : {camp}")
    print(f"output   : {out_dir}")
    print(f"settings : T_w = {T_W_S*1e9:.2f} ns, |t' - tau| <= {LAG_MAX_S*1e9:.2f} ns, "
          f"|rho| >= {RHO_MIN}, template >= {TEMPLATE_LEVEL:.0%} of max, arrival = {ARRIVAL_DIMENSIONS}")
    print(f"waveform : {transmitted}, delta_t = {tpl.delta_t_s*1e9:.3f} ns, "
          f"crop {tpl.t_rel[0]*1e9:+.2f} .. {tpl.t_rel[-1]*1e9:+.2f} ns, autocorr. first zero {z0*1e9:.2f} ns")
    print(f"cases    : {len(todo)}" + ("" if not a.no_ascan else "  (no A-scan figures)"))
    print("-" * 100)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_template(tpl, ref_fs, LAG_MAX_S, out_dir / "figs" / "transmitted_waveform.png")

    new_rows: List[Dict] = []
    traces: Dict[str, np.ndarray] = {}
    missing: List[str] = []
    t_start = time.time()
    n = len(todo)
    for i, r in enumerate(todo, 1):
        cid = r["case_id"]
        d = load_json(camp / r["json"])
        p_out = camp / r["output"]
        if not p_out.exists():
            missing.append(cid)
            print(f"[{i:3d}/{n}] {cid:40s} no .out (skipped)")
            continue
        tr = pc.read_out(p_out)
        if tr.ez.size != ref_bg.ez.size or abs(tr.dt - ref_bg.dt) > 1e-6 * tr.dt:
            print(f"[{i:3d}/{n}] {cid:40s} length/dt differ from ref_background (skipped)")
            missing.append(cid)
            continue
        x = tr.ez - ref_bg.ez
        traces[cid] = x.astype(np.float32)

        th = d["theory"][ARRIVAL_DIMENSIONS]
        th_ab = d["theory"]["as_built_effective"]
        taus = {f: th[f]["t_nadir_s"] + tpl.delta_t_s for f in FACES}
        overlap = pc.windows_overlap(taus["top"], taus["bottom"], T_W_S)
        faces = {}
        for f in FACES:
            res = pc.ncc_face(x, tr.dt, tpl, taus[f], T_W_S, LAG_MAX_S)
            exp = th[f]["expected_polarity"]
            obs = pc.observed_polarity(res.rho_best, transmitted)
            ty, why = pc.classify(rho_best=res.rho_best, observed=obs, expected=exp, overlap=overlap,
                                  out_of_range=res.out_of_range, rho_min=RHO_MIN)
            faces[f] = dict(res=res, expected=exp, observed=obs, type=ty, reason=why,
                            w_over_2rf=th[f]["width_over_fresnel_diameter"])
            t_info = d["target"]
            new_rows.append(dict(
                case_id=cid, group=d["group"], target=t_info["kind"], pattern=pattern_of(d),
                depth_m=t_info["depth_top_m"], height_m=t_info["height_m"],
                width_m="" if t_info.get("width_m") is None else t_info["width_m"],
                pair_key=d.get("pair_key", ""), face=f,
                expected_polarity=exp, reflection_coefficient=th[f]["reflection_coefficient"],
                t_nadir_ns=th[f]["t_nadir_s"] * 1e9, delta_t_ns=tpl.delta_t_s * 1e9,
                tau_ns=res.tau_s * 1e9, window_start_ns=res.window_s[0] * 1e9,
                window_end_ns=res.window_s[1] * 1e9,
                tau_as_built_ns=(th_ab[f]["t_nadir_s"] + tpl.delta_t_s) * 1e9,
                windows_overlap=int(overlap),
                fresnel_radius_m=th[f]["fresnel_radius_m"], fresnel_diameter_m=th[f]["fresnel_diameter_m"],
                width_over_fresnel_diameter="" if th[f]["width_over_fresnel_diameter"] is None
                else th[f]["width_over_fresnel_diameter"],
                rho_best=res.rho_best, lag_best_ns=res.lag_best_s * 1e9, rho_at_tau=res.rho_at_tau,
                rho_pos_max=res.rho_pos_max, rho_neg_min=res.rho_neg_min,
                observed_polarity=obs, type=ty, type_reason=why,
                amplitude_ls=res.amplitude_ls, window_peak_abs=res.window_peak_abs, window_rms=res.window_rms,
            ))
        if not a.no_ascan:
            plot_ascan(out_dir / "ascan" / pattern_of(d) / f"{cid}.png", d, tr.t, x, tpl, faces,
                       T_W_S, LAG_MAX_S)

        el = time.time() - t_start
        eta = el / i * (n - i)
        def short(f):
            q = faces[f]
            return f"{f[0].upper()}:T{q['type']}({q['res'].rho_best:+.2f})"
        print(f"[{i:3d}/{n}] {100*i/n:5.1f}%  {fmt_hms(el)} elapsed, ETA {fmt_hms(eta)}  "
              f"{cid:38s} {short('top')} {short('bottom')}", flush=True)

    # --- 前回の結果と合わせて書き出す --------------------------------------
    done_ids = {r["case_id"] for r in new_rows}
    rows = []
    p_csv = out_dir / "polarity_results.csv"
    if old_settings is not None and p_csv.exists():
        with open(p_csv, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r["case_id"] not in done_ids]
    rows += new_rows
    order = {r["case_id"]: k for k, r in enumerate(rows_man)}
    rows.sort(key=lambda r: (order.get(r["case_id"], 10 ** 6), FACES.index(r["face"])))

    def fmtv(v):
        if isinstance(v, float):
            return "" if not np.isfinite(v) else f"{v:.6g}"
        return v
    with open(p_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmtv(r.get(k, "")) for k in CSV_FIELDS})

    p_npz = out_dir / "traces.npz"
    if old_settings is not None and p_npz.exists():
        with np.load(p_npz, allow_pickle=False) as z:
            old_ids, old_x = z["case_ids"], z["x"]          # 一度だけ読む
        for k, cid in enumerate(old_ids):
            if str(cid) not in traces:
                traces[str(cid)] = old_x[k]
    ids = sorted(traces, key=lambda c: order.get(c, 10 ** 6))
    np.savez_compressed(p_npz, case_ids=np.array(ids), x=np.stack([traces[c] for c in ids]) if ids else np.zeros((0, 0)),
                        dt=ref_bg.dt, template=tpl.s, template_t_rel=tpl.t_rel,
                        delta_t=tpl.delta_t_s)

    # 平行平板
    slab = [r for r in rows if r["group"] == "slab"]
    with open(out_dir / "slab_check.csv", "w", newline="", encoding="utf-8") as fh:
        keys = ["case_id", "target", "face", "expected_polarity", "observed_polarity",
                "rho_best", "lag_best_ns", "rho_at_tau", "type", "type_reason"]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in slab:
            w.writerow({k: fmtv(r[k]) for k in keys})

    settings["created"] = now_iso()
    settings["n_cases_in_csv"] = len({r["case_id"] for r in rows})
    settings["missing_outputs"] = missing
    (out_dir / "analysis_settings.json").write_text(json.dumps(settings, indent=1, ensure_ascii=False),
                                                     encoding="utf-8")

    # --- まとめ ----------------------------------------------------------
    print("-" * 100)
    print(f"done in {fmt_hms(time.time() - t_start)}; {len(done_ids)} cases analysed, {len(missing)} without .out")
    if slab:
        print("\nslab check (should be Type 1 with |rho| close to 1):")
        for r in slab:
            print(f"  {r['case_id']:28s} {r['face']:6s} expected {r['expected_polarity']} observed "
                  f"{r['observed_polarity']:3s} rho {float(r['rho_best']):+.3f}  Type {r['type']}")
    print("\ncounts (Type 1 / 2 / 3) at |rho| >= %.2f:" % RHO_MIN)
    pats = sorted({r["pattern"] for r in rows if r["group"] != "slab"})
    for p in pats:
        line = f"  {p:16s}"
        for f in FACES:
            sel = [int(r["type"]) for r in rows if r["pattern"] == p and r["face"] == f]
            line += f"  {f:6s} {sel.count(1):3d} / {sel.count(2):3d} / {sel.count(3):3d}"
        print(line)
    print(f"\nresults: {p_csv}")
    if missing:
        print(f"cases without .out: {len(missing)} (listed in analysis_settings.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
