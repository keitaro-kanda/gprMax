"""
水氷の観測をまとめた図を作るためのスクリプト。
横軸：異なる論文、縦軸：水氷の濃度(wt%)、の形式の図を作る。
プロット点の色で観測手法を区別する。

データは辞書で管理する
- key：論文名（XXX et al. (YYYY)の形式）
- value：辞書で管理する。valueの項目は以下の通り
    - wt%：水氷の濃度(wt%)
    - wt%_error_low：水氷の濃度の下側誤差(wt%)（不明な場合はNone）
    - wt%_error_high：水氷の濃度の上側誤差(wt%)（不明な場合はNone）
    - method：観測手法（文字列）
    - loc：観測場所（文字列、任意）、複数地域の観測値がある場合は、それがわかるように示す（例：Highlands, Mareなど）
"""

import numpy as np
import matplotlib.pyplot as plt
import os

data = {
    "Colaprete et al. (2010)": {"wt%": 5.6, "wt%_error_low": 2.9, "wt%_error_high": 2.9, "method": "LCROSS", "loc": "Cabeus"},
    "Li et al. (2018)": {"wt%": 30, "wt%_error_low": None, "wt%_error_high": None, "method": "Chandrayaan-1/M3", "loc": "PSRs"},
    "Sanin et al. ( 2017)": {"wt%": 0.54, "wt%_error_low": 0.06, "wt%_error_high": 0.07, "method": "LRO/LEND", "loc": "Cabeus"},
    "Sanin et al. ( 2017)": {"wt%": 0.51, "wt%_error_low": 0.04, "wt%_error_high": 0.04, "method": "LRO/LEND", "loc": "Shoemaker"},
    "Miller et al. (2012)": {"wt%": 0.7, "wt%_error_low": None, "wt%_error_high": None, "method": "LP/GRS", "loc": "Shackleton"},
}


# プロット
plt.figure(figsize=(10, 6))
for paper, info in data.items():
    wt_percent = info["wt%"]
    error_low = info["wt%_error_low"]
    error_high = info["wt%_error_high"]
    method = info["method"]
    loc = info.get("loc", "")

    # エラーバーの設定
    if error_low is not None and error_high is not None:
        plt.errorbar(paper, wt_percent, yerr=[[error_low], [error_high]], fmt='o', label=f"{method}")
    else:
        plt.plot(paper, wt_percent, 'o', label=f"{method}")

    # 地域をテキストで表示
    if loc:
        plt.text(paper, wt_percent + 0.5, loc, ha='center', fontsize=14)

plt.xticks(rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Water Ice Concentration (wt%)", fontsize=18)
plt.grid()
plt.legend(fontsize=14)
plt.tight_layout()

plt.show()
# output_dir = "output"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# plt.savefig(os.path.join(output_dir, "water_ice_summary.png"))