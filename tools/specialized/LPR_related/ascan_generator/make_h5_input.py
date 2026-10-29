#!/usr/bin/env python3
"""
make_h5_input.py
================

本計算では h5（#geometry_objects_write）を出力しない。必要になったケースだけ、
h5 を書き出す .in（<case_id>_h5.in）を同じフォルダに作る。形状は .in から決まるので、
FDTD の計算をせずに作り直せる。

    python make_h5_input.py <case>.in [<case>.in ...]
    python -m gprMax <case>_h5.in --geometry-only

<case>_h5.in では
  * 「DISABLED #geometry_objects_write:」の行を有効にする
  * #geometry_view の行を無効にする（本計算の .vti を上書きしないため）
出力: <case_id>_geometry.h5 と <case_id>_geometry_materials.txt（.in と同じフォルダ）
"""
from __future__ import annotations

import sys
from pathlib import Path


def convert(path: Path) -> Path:
    lines = path.read_text(encoding="utf-8").splitlines()
    out, n_on = [], 0
    for line in lines:
        if line.startswith("DISABLED #geometry_objects_write:"):
            out.append(line[len("DISABLED "):])
            n_on += 1
        elif line.startswith("#geometry_view:"):
            out.append("DISABLED " + line)
        else:
            out.append(line)
    if n_on == 0:
        raise SystemExit(f"{path}: 'DISABLED #geometry_objects_write:' の行が見つかりません")
    dst = path.with_name(path.stem + "_h5.in")
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    return dst


def main(argv) -> int:
    if not argv:
        print(__doc__)
        return 1
    for a in argv:
        dst = convert(Path(a))
        print(f"written: {dst}\n  run:   python -m gprMax \"{dst}\" --geometry-only")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
