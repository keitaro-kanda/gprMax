#!/usr/bin/env python3
"""
plot_polarity_maps.py  --  段階 2: 判定結果のまとめ図
=====================================================

analyze_polarity.py が書いた polarity_results.csv だけを読み、まとめの図を描く（.out は読まない）。
|rho| の閾値を変えて描き直すときは、この段階だけを実行すればよい。

    python plot_polarity_maps.py [解析フォルダ] [--rho-min 0.5] [--no-fresnel] [--formats png pdf]

解析フォルダの既定は analyze_polarity.py の CAMPAIGN_DIR / analysis / ANALYSIS_NAME。

出力（<解析フォルダ>/maps_rho0.50/ のように閾値ごとのフォルダ）
  type_maps/type_map_<対象>_depthXXm.png/.pdf
      パターン（岩片・空洞 x 上端の深さ）ごとの図。左が上面、右が底面。
      横軸 幅 w、縦軸 高さ h、色 Type（1 赤、2 青、3 灰）。破線は w = 2 r_F（フレネル帯の直径）。
  score_maps/score_map_<対象>_depthXXm.png/.pdf          補助図 (a)
      rho に期待される符号を掛けた値（+1 期待どおり、-1 逆）を連続的な色で示す。
      窓が重なる Type 3 は灰色、|rho| が閾値未満のマスには点を打つ。
  fresnel_normalized/fresnel_normalized_<対象>.png/.pdf    補助図 (b)
      横軸 w / 2 r_F。深さ 1, 2, 4 m で共通の高さについて、深さを並べて比べる。
  type_counts.csv   パターン・面ごとの Type 1/2/3 の件数
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import polarity_core as pc   # noqa: E402

try:
    from analyze_polarity import CAMPAIGN_DIR, ANALYSIS_NAME, RHO_MIN
except Exception:                      # pragma: no cover
    CAMPAIGN_DIR, ANALYSIS_NAME, RHO_MIN = Path("."), "ncc_v01", 0.5

import matplotlib                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
from matplotlib import patheffects     # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D    # noqa: E402
from matplotlib.patches import Patch   # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5})

FACES = ("top", "bottom")
TYPE_COLOR = {1: "#d62728", 2: "#1f77b4", 3: "#9a9a9a"}
TYPE_NAME = {1: "Type 1 (expected)", 2: "Type 2 (unexpected)", 3: "Type 3 (unclassified)"}
TARGET_NAME = {"rock": "Rock", "void": "Void"}
DEPTH_MARKER = {1.0: "o", 2.0: "s", 4.0: "^"}
FRESNEL_PE = [patheffects.withStroke(linewidth=3.2, foreground="w")]
FIXED_TYPE3 = (pc.REASON_OVERLAP, pc.REASON_OUT_OF_RANGE, pc.REASON_NO_SIGNAL)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("analysis_dir", nargs="?", type=Path, default=CAMPAIGN_DIR / "analysis" / ANALYSIS_NAME)
    p.add_argument("--rho-min", type=float, default=RHO_MIN)
    p.add_argument("--no-fresnel", action="store_true", help="w = 2 r_F の線を描かない")
    p.add_argument("--formats", nargs="+", default=["png", "pdf"])
    return p.parse_args(argv)


# -----------------------------------------------------------------------------
def load_rows(path: Path) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("depth_m", "height_m", "width_m", "rho_best", "fresnel_diameter_m",
                  "width_over_fresnel_diameter"):
            v = r.get(k, "")
            r[k] = float(v) if v not in ("", None) else None
    return rows


def reclassify(r: Dict, rho_min: float) -> (int, str):
    if r["type_reason"] in FIXED_TYPE3:
        return 3, r["type_reason"]
    rho = r["rho_best"]
    if rho is None or not np.isfinite(rho):
        return 3, pc.REASON_NO_SIGNAL
    if abs(rho) < rho_min:
        return 3, pc.REASON_LOW_CORR
    return (1, "") if r["observed_polarity"] == r["expected_polarity"] else (2, "")


def edges_from_centres(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    if v.size == 1:
        return np.array([v[0] * 0.5, v[0] * 1.5])
    mid = 0.5 * (v[1:] + v[:-1])
    return np.r_[v[0] - (mid[0] - v[0]), mid, v[-1] + (v[-1] - mid[-1])]


def fmt_size(v: float) -> str:
    """設定値の表記: 余分な 0 を省き、小数点以下は少なくとも 1 桁（1.0, 0.25, 0.125）"""
    t = f"{v:.3f}".rstrip("0")
    return t + "0" if t.endswith(".") else t


def set_value_ticks(ax, ws, hs) -> None:
    """目盛りを計算の設定値（各マスの中央）に置く"""
    ax.set_xticks(ws)
    ax.set_xticklabels([fmt_size(v) for v in ws], rotation=90, fontsize=7.5)
    ax.set_yticks(hs)
    ax.set_yticklabels([fmt_size(v) for v in hs], fontsize=7.5)
    ax.minorticks_off()


def panel_height(hs) -> float:
    """図の高さ: 高さが 2 通りしかない深さ依存性の図は低くする"""
    return 4.4 if len(hs) > 4 else 2.6


def save(fig, base: Path, formats) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(base.with_suffix("." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)


def depth_label(dm: float) -> str:
    return f"depth{int(round(dm)):02d}m"


def grid_of(rows: List[Dict], key: str):
    """(widths, heights, {face: 2D array [h, w]}) を返す"""
    ws = np.array(sorted({r["width_m"] for r in rows}))
    hs = np.array(sorted({r["height_m"] for r in rows}))
    out = {}
    for f in FACES:
        g = np.full((hs.size, ws.size), np.nan)
        for r in rows:
            if r["face"] != f:
                continue
            i = int(np.flatnonzero(np.isclose(hs, r["height_m"]))[0])
            j = int(np.flatnonzero(np.isclose(ws, r["width_m"]))[0])
            g[i, j] = r[key]
        out[f] = g
    return ws, hs, out


def fresnel_lines(ax, face: str, rows: List[Dict], consts: Dict, target: str, depth: float, hs_edges):
    pe = FRESNEL_PE
    if face == "top":
        x0 = rows[0]["fresnel_diameter_m"] if rows else None
        x0 = pc.fresnel_diameter_top(antenna_height_m=consts["antenna_height_m"], depth_top_m=depth,
                                     eps_background=consts["eps_background"],
                                     wavelength_m=consts["wavelength_m"]) if x0 is None else x0
        ax.plot([x0, x0], [hs_edges[0], hs_edges[-1]], color="k", ls="--", lw=1.5, path_effects=pe)
    else:
        h = np.linspace(hs_edges[0], hs_edges[-1], 200)
        w = pc.fresnel_diameter_bottom(h, antenna_height_m=consts["antenna_height_m"], depth_top_m=depth,
                                       eps_background=consts["eps_background"],
                                       eps_target=consts["eps_target"][target],
                                       wavelength_m=consts["wavelength_m"])
        ax.plot(w, h, color="k", ls="--", lw=1.5, path_effects=pe)


# -----------------------------------------------------------------------------
def type_map(rows, target, depth, consts, out: Path, fresnel: bool, formats, rho_min):
    ws, hs, g = grid_of(rows, "type_new")
    we, he = edges_from_centres(ws), edges_from_centres(hs)
    cmap = ListedColormap([TYPE_COLOR[1], TYPE_COLOR[2], TYPE_COLOR[3]])
    cmap.set_bad("white")
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, panel_height(hs)), sharey=True)
    for ax, f in zip(axes, FACES):
        ax.pcolormesh(we, he, np.ma.masked_invalid(g[f]), cmap=cmap, norm=norm,
                      edgecolors="w", linewidth=0.4)
        if fresnel:
            fresnel_lines(ax, f, [r for r in rows if r["face"] == f], consts, target, depth, he)
        ax.set_xlim(we[0], we[-1])
        ax.set_ylim(he[0], he[-1])
        ax.set_xlabel("Width w (m)")
        set_value_ticks(ax, ws, hs)
        ax.set_title(f"{TARGET_NAME[target]}, d = {depth:g} m: {f}")
        if np.isnan(g[f]).any():
            ii, jj = np.nonzero(np.isnan(g[f]))
            for i, j in zip(ii, jj):
                ax.text(ws[j], hs[i], "n/a", ha="center", va="center", fontsize=6, color="0.4")
    axes[0].set_ylabel("Height h (m)")
    handles = [Patch(color=TYPE_COLOR[k], label=TYPE_NAME[k]) for k in (1, 2, 3)]
    if fresnel:
        handles.append(Line2D([], [], color="k", ls="--", lw=1.5, label="w = 2 r_F (Fresnel zone diameter)"))
    fig.tight_layout(rect=(0, 0.4 / fig.get_figheight(), 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=8)
    save(fig, out, formats)


def score_map(rows, target, depth, consts, out: Path, fresnel: bool, formats, rho_min, transmitted):
    for r in rows:
        sgn = 1.0 if r["expected_polarity"] == transmitted else -1.0
        rho = r["rho_best"]
        r["score"] = np.nan if (r["type_reason"] in FIXED_TYPE3 or rho is None) else sgn * rho
        r["fixed3"] = 1.0 if r["type_reason"] in FIXED_TYPE3 else 0.0
    ws, hs, g = grid_of(rows, "score")
    _, _, fx = grid_of(rows, "fixed3")
    we, he = edges_from_centres(ws), edges_from_centres(hs)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#9a9a9a")
    H = panel_height(hs)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, H), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.95 / H, top=1 - 0.35 / H, wspace=0.08)
    for ax, f in zip(axes, FACES):
        pm = ax.pcolormesh(we, he, np.ma.masked_invalid(g[f]), cmap=cmap, vmin=-1, vmax=1,
                           edgecolors="w", linewidth=0.4)
        low = np.abs(g[f]) < rho_min
        ii, jj = np.nonzero(low & np.isfinite(g[f]))
        ax.plot(ws[jj], hs[ii], "k.", ms=3)
        ii, jj = np.nonzero(np.isnan(g[f]) & (fx[f] != 1.0))
        for i, j in zip(ii, jj):
            ax.text(ws[j], hs[i], "n/a", ha="center", va="center", fontsize=6, color="w")
        if fresnel:
            fresnel_lines(ax, f, [r for r in rows if r["face"] == f], consts, target, depth, he)
        ax.set_xlim(we[0], we[-1])
        ax.set_ylim(he[0], he[-1])
        ax.set_xlabel("Width w (m)")
        set_value_ticks(ax, ws, hs)
        ax.set_title(f"{TARGET_NAME[target]}, d = {depth:g} m: {f}")
    axes[0].set_ylabel("Height h (m)")
    cax = fig.add_axes([0.9, 0.95 / H, 0.015, 1 - 1.3 / H])
    cb = fig.colorbar(pm, cax=cax)
    cb.set_label("rho x sign(expected)\n(+1: expected, -1: opposite)")
    handles = [Patch(color="#9a9a9a", label="Type 3 (windows overlap)"),
               Line2D([], [], color="k", marker=".", ls="", label=f"|rho| < {rho_min:g}")]
    if fresnel:
        handles.append(Line2D([], [], color="k", ls="--", lw=1.5, label="w = 2 r_F (Fresnel zone diameter)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.47, 0.0), fontsize=8)
    save(fig, out, formats)


def fresnel_normalized(rows_t, target, out: Path, formats, rho_min):
    """深さ 1, 2, 4 m で共通の高さについて、横軸 w / 2 r_F で比べる"""
    depths = sorted({r["depth_m"] for r in rows_t})
    common = None
    for dm in depths:
        hs = {round(r["height_m"], 6) for r in rows_t if r["depth_m"] == dm}
        common = hs if common is None else common & hs
    common = sorted(common or [])
    if not common or len(depths) < 2:
        print(f"  {target}: no common heights across depths; skipped the normalized figure")
        return
    lines = [(h, dm) for h in common for dm in depths]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 0.42 * len(lines) + 1.6), sharey=True)
    for ax, f in zip(axes, FACES):
        for k, (h, dm) in enumerate(lines):
            sel = [r for r in rows_t if r["face"] == f and r["depth_m"] == dm
                   and np.isclose(r["height_m"], h)]
            sel.sort(key=lambda r: r["width_m"])
            x = np.array([r["width_over_fresnel_diameter"] for r in sel])
            ty = [r["type_new"] for r in sel]
            ax.plot(x, np.full(x.size, k), color="0.8", lw=0.8, zorder=1)
            ax.scatter(x, np.full(x.size, k), c=[TYPE_COLOR[t] for t in ty],
                       marker=DEPTH_MARKER.get(dm, "o"), s=30, edgecolors="k", linewidths=0.4, zorder=2)
        ax.axvline(1.0, color="k", ls="--", lw=0.9)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("w / 2 r_F")
        ax.set_title(f"{TARGET_NAME[target]}: {f}")
        for k in range(1, len(common)):
            ax.axhline(k * len(depths) - 0.5, color="0.6", lw=0.6)
    axes[0].set_yticks(range(len(lines)))
    axes[0].set_yticklabels([f"h = {h:.2f} m, d = {dm:g} m" for h, dm in lines])
    axes[0].set_ylim(len(lines) - 0.5, -0.5)
    handles = [Patch(color=TYPE_COLOR[k], label=TYPE_NAME[k]) for k in (1, 2, 3)]
    handles += [Line2D([], [], color="w", marker=DEPTH_MARKER.get(dm, "o"), markeredgecolor="k",
                       markerfacecolor="0.7", ls="", label=f"d = {dm:g} m") for dm in depths]
    handles.append(Line2D([], [], color="k", ls="--", label="w = 2 r_F"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0),
               fontsize=8)
    fig.tight_layout(rect=(0, 0.75 / fig.get_figheight(), 1, 1))
    save(fig, out, formats)


# -----------------------------------------------------------------------------
def main(argv=None) -> int:
    a = parse_args(argv)
    ad = a.analysis_dir.resolve()
    p_csv = ad / "polarity_results.csv"
    if not p_csv.exists():
        print(f"{p_csv} not found. Run analyze_polarity.py first.")
        return 1
    settings = json.loads((ad / "analysis_settings.json").read_text(encoding="utf-8"))
    consts = settings["fresnel_constants"]
    transmitted = settings["transmitted_waveform"]["polarity"]
    rows = load_rows(p_csv)
    for r in rows:
        r["type_new"], r["reason_new"] = reclassify(r, a.rho_min)

    out = ad / f"maps_rho{a.rho_min:.2f}"
    main_rows = [r for r in rows if r["group"] in ("nominal", "depth")]
    pats = sorted({(r["target"], r["depth_m"]) for r in main_rows},
                  key=lambda p: (["rock", "void"].index(p[0]) if p[0] in ("rock", "void") else 9, p[1]))
    print(f"analysis : {ad}")
    print(f"threshold: |rho| >= {a.rho_min:g}   output: {out}")
    counts = []
    for k, (tg, dm) in enumerate(pats, 1):
        sel = [dict(r) for r in main_rows if r["target"] == tg and r["depth_m"] == dm]
        tag = f"{tg}_{depth_label(dm)}"
        type_map(sel, tg, dm, consts, out / "type_maps" / f"type_map_{tag}", not a.no_fresnel,
                 a.formats, a.rho_min)
        score_map(sel, tg, dm, consts, out / "score_maps" / f"score_map_{tag}", not a.no_fresnel,
                  a.formats, a.rho_min, transmitted)
        for f in FACES:
            ty = [r["type_new"] for r in sel if r["face"] == f]
            rs = [r["reason_new"] for r in sel if r["face"] == f]
            counts.append(dict(pattern=f"{tg}/{depth_label(dm)}", face=f, n=len(ty),
                               type1=ty.count(1), type2=ty.count(2), type3=ty.count(3),
                               type3_overlap=rs.count(pc.REASON_OVERLAP),
                               type3_low_corr=rs.count(pc.REASON_LOW_CORR)))
        print(f"[{k}/{len(pats)}] maps for {tag}")
    for tg in sorted({p[0] for p in pats}):
        fresnel_normalized([r for r in main_rows if r["target"] == tg], tg,
                           out / "fresnel_normalized" / f"fresnel_normalized_{tg}", a.formats, a.rho_min)
        print(f"      normalized-width figure for {tg}")

    slab = [r for r in rows if r["group"] == "slab"]
    for r in slab:
        counts.append(dict(pattern=r["pattern"], face=r["face"], n=1,
                           type1=int(r["type_new"] == 1), type2=int(r["type_new"] == 2),
                           type3=int(r["type_new"] == 3),
                           type3_overlap=int(r["reason_new"] == pc.REASON_OVERLAP),
                           type3_low_corr=int(r["reason_new"] == pc.REASON_LOW_CORR)))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "type_counts.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(counts[0].keys()) if counts else ["pattern"])
        w.writeheader()
        w.writerows(counts)
    print("\npattern            face     n   T1   T2   T3 (overlap, low_corr)")
    for c in counts:
        print(f"{c['pattern']:18s} {c['face']:6s} {c['n']:4d} {c['type1']:4d} {c['type2']:4d} {c['type3']:4d}"
              f" ({c['type3_overlap']}, {c['type3_low_corr']})")
    print(f"\nfigures: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())