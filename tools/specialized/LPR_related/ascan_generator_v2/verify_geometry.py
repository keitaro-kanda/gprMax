#!/usr/bin/env python3
"""
verify_geometry.py
==================

gprMax が作った .vti（--geometry-only または本計算）を読み、ケース JSON の予測と照合する。

    python verify_geometry.py <キャンペーンフォルダ> [--cases ID ...] [--groups main ...]

確認する項目
------------
* 対象の材料がある（空隙は地表より下の free_space）
* 対象のセル数が JSON の as_built.n_cells と一致する
* 中点の真下の Ez 節点数と上端深さが JSON の as_built.nadir と一致する
* 地表（レゴリス上面）の高さ
* 送信点・受信点のセル位置
vti がまだないケースは「no vti」として数えるだけ。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from grid_geometry import as_built_summary, read_gprmax_vti, target_mask_from_vti   # noqa: E402


def check_case(camp: Path, d: dict) -> tuple:
    f = d["files"]
    vti = camp / f["geometry_vti"]
    if not vti.exists():
        return "no vti", []
    g = read_gprmax_vti(str(vti))
    dx = g.spacing[0]
    msgs = []
    surface = d["model"]["surface_y_m"]

    # 送受信点
    tx = np.argwhere(g.sources_pml[:, :, 0] == 2) if g.sources_pml is not None else np.empty((0, 2))
    rx = np.argwhere(g.receivers[:, :, 0] == 1) if g.receivers is not None else np.empty((0, 2))
    exp_tx = [int(round(v / dx)) for v in d["antenna"]["tx_xy_m"]]
    exp_rx = [int(round(v / dx)) for v in d["antenna"]["rx_xy_m"]]
    if len(tx) != 1 or list(tx[0]) != exp_tx:
        msgs.append(f"Tx cell {tx.tolist()} != {exp_tx}")
    if len(rx) != 1 or list(rx[0]) != exp_rx:
        msgs.append(f"Rx cell {rx.tolist()} != {exp_rx}")

    # 地表
    if d["group"] == "reference" and d.get("reference_kind") == "freespace":
        if "regolith" in g.material_ids and g.mask("regolith").any():
            msgs.append("regolith found in free-space reference")
        return ("ok" if not msgs else "NG"), msgs
    if "regolith" not in g.material_ids:
        return "NG", msgs + ["regolith material missing"]
    reg = g.mask("regolith")[:, :, 0]
    col = reg[int(round(d["antenna"]["mid_x_m"] / dx)) - 50]          # 対象から離れた列
    top_row = int(np.flatnonzero(col).max()) + 1
    if abs(top_row * dx - surface) > 1e-9:
        msgs.append(f"surface at {top_row*dx} m != {surface} m")

    if d["group"] == "reference":
        return ("ok" if not msgs else "NG"), msgs

    # 対象
    t = d["target"]
    name = t["gprmax_material"]
    if name not in g.material_ids:
        return "NG", msgs + [f"target material {name} missing"]
    lm = target_mask_from_vti(g, name, surface)
    got = as_built_summary(lm, d["antenna"]["mid_x_m"], surface)
    exp = t["as_built"]
    if got["n_cells"] != exp["n_cells"]:
        msgs.append(f"n_cells {got['n_cells']} != {exp['n_cells']}")
    gn, en = got.get("nadir"), exp.get("nadir")
    if (gn is None) != (en is None):
        msgs.append("nadir presence differs")
    elif gn is not None:
        for k in ("ez_node_count", "depth_top_m", "effective_thickness_m"):
            if abs(gn[k] - en[k]) > 1e-9:
                msgs.append(f"nadir {k} {gn[k]} != {en[k]}")
    return ("ok" if not msgs else "NG"), msgs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("campaign", type=Path)
    p.add_argument("--cases", nargs="+")
    p.add_argument("--groups", nargs="+")
    a = p.parse_args(argv)
    camp = a.campaign.resolve()
    man = json.loads((camp / "manifest.json").read_text(encoding="utf-8"))
    counts = {"ok": 0, "NG": 0, "no vti": 0}
    for row in man["cases"]:
        if a.cases and row["case_id"] not in a.cases:
            continue
        if a.groups and row["group"] not in a.groups:
            continue
        d = json.loads((camp / row["json"]).read_text(encoding="utf-8"))
        status, msgs = check_case(camp, d)
        counts[status] += 1
        if status != "no vti":
            print(f"{status:3s}  {row['case_id']}" + ("" if not msgs else "  <- " + "; ".join(msgs)))
    print(f"\nok {counts['ok']}, NG {counts['NG']}, no vti {counts['no vti']}")
    return 1 if counts["NG"] else 0


if __name__ == "__main__":
    sys.exit(main())
