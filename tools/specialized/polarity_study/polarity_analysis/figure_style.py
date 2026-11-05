"""
figure_style.py
===============

全ての図に共通するフォントサイズの規則（マスタールール）。
図を描くスクリプトは、最初に apply() を呼ぶ。個々の図で fontsize を直接指定しないこと。

  タイトル、軸ラベル、カラーバーのラベル : 16
  目盛り、凡例、図中の文字              : 12

文字が大きいので、タイトルと凡例には最低限の情報（線の意味や名称）だけを書く。
"""
FS_TITLE = 16
FS_LABEL = 16
FS_TICK = 12
FS_LEGEND = 12
FS_TEXT = 12


def apply() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": FS_TEXT,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "figure.titlesize": FS_TITLE,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "legend.title_fontsize": FS_LEGEND,
        "figure.dpi": 100,
    })
