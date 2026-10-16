#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_cases.py
==============

CE-4 ランダム媒質シミュレーションのケースごとのフォルダ（case00〜case06）を作る。
このスクリプトと同じフォルダにある Ascan.in / Bscan.in を元に、Ascan.in の
CASE_ID の行だけを書き換えてコピーする。case00（均質、参照）は A-scan だけなので
Bscan.in は置かない。あわせて、実行コマンドを並べた run_all.sh を作る。

使い方
    python3 setup_cases.py
    （作成先のフォルダを聞かれる。空なら このスクリプトのフォルダ/cases）
"""
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASE_IDS = (0, 1, 2, 3, 4, 5, 6)
N_TRACES = 115


def main():
    print('ケースのフォルダを作る場所を入力してください（空なら {}）:'.format(
        os.path.join(HERE, 'cases')))
    base = input().strip() or os.path.join(HERE, 'cases')
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(HERE, 'Ascan.in'), encoding='utf-8') as fh:
        ascan = fh.read()
    if len(re.findall(r'(?m)^CASE_ID\s*=', ascan)) != 1:
        print('エラー: Ascan.in に CASE_ID の行が 1 つだけあること')
        sys.exit(1)
    lines = ['#!/bin/sh', '# 各ケースを順に計算する（gprMax の環境で実行）', 'set -e']
    for cid in CASE_IDS:
        d = os.path.join(base, 'case{:02d}'.format(cid))
        os.makedirs(d, exist_ok=True)
        txt = re.sub(r'(?m)^CASE_ID\s*=\s*\d+', 'CASE_ID = {}'.format(cid), ascan, count=1)
        with open(os.path.join(d, 'Ascan.in'), 'w', encoding='utf-8') as fh:
            fh.write(txt)
        lines.append('cd "{}"'.format(d))
        if cid == 0:
            lines.append('python -m gprMax Ascan.in            # 均質：A-scan 1 本（波形の参照）')
        else:
            shutil.copy(os.path.join(HERE, 'Bscan.in'), os.path.join(d, 'Bscan.in'))
            lines.append('python -m gprMax Ascan.in --geometry-only')
            lines.append('python -m gprMax Bscan.in -n {}'.format(N_TRACES))
            lines.append('python -m tools.outputfiles_merge result/Bscan')
        print('  {}'.format(d))
    sh = os.path.join(base, 'run_all.sh')
    with open(sh, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    os.chmod(sh, 0o755)
    print('-> {}'.format(sh))
    print('注意: tools.outputfiles_merge は gprMax のフォルダから import できる環境で実行すること')


if __name__ == '__main__':
    main()
