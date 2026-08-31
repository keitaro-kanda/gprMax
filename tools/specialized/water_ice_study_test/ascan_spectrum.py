"""スペクトル解析コード

対象：減衰計測GPRによる月の水氷検出研究／複雑性のはしご Level 1 以降
目的：各深さの受信スペクトルを理論予測と比較し、減衰・分散に関わる量を
      段階的に検証する。

設計書: design_ascan_spectrum.md (v1, 2026-08-08) に準拠。
入出力仕様は ascan_amplitude.py (design_ascan_amplitude.md) を踏襲する。

現時点で完全実装しているのは Level_1（geom + surface_T, alpha=0）のみ。
Level_2 以降の吸収項（absorb_const / absorb_debye / density_profile）は、
設計書に具体的な物性値（tanδ 等）が未確定のため、骨格のみ用意し
NotImplementedError とする。
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import csv
import json
import re
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy import signal

from gprMax.exceptions import CmdInputError
from tools.core.outputfiles_merge import get_output_data

# =============================================================================
# 定数
# =============================================================================
# [EDIT HERE] パス JSON のハードコード（ascan_amplitude.py と同一ファイルを想定）
JSON_PATH = "/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_3x4/water_ice_study_test/out_file_paths.json"

C = 0.29979          # [m/ns] 光速
TX_HEIGHT = 0.35      # [m] 送信機高さ h
R_REF = 1.0           # [m] 参照計算（ref_freespace_1m）の距離

# Level_1 のレゴリス物性
EPS_R_REGOLITH = 3.0
N_REGOLITH = np.sqrt(EPS_R_REGOLITH)

# 解析帯域 (design_ascan_spectrum.md §3.1)
BAND_GHZ = (0.5, 2.0)

# LSR系マスク (§3.2)
MASK_REF_FLOOR_DB = -20.0    # |E_ref| が帯域内最大からこれ以上落ちる周波数を除外
MASK_SNR_MIN_DB = 20.0       # 各トレースのノイズフロアからこれ以上上にある周波数のみ使う

# 時間ゲート（既定は無効, §4.2）
GATE_ENABLED = False
GATE_HALFWIDTH_NS = 5.0
GATE_TAPER = 0.2             # Tukey

# 相対LSRの参照深さ (§6.2)
LSR_REF_DEPTH_M = 0.25

# 上限・下限周波数のしきい値 (§6.4)
FLOHI_THRESHOLDS_DB = [-3.0, -10.0, -20.0]
FLOHI_PRIMARY_DB = -10.0

# ノイズフロア評価帯域 (§6.7)
NOISE_BAND_GHZ = (3.0, 5.0)

# 合格基準 (design_ascan_spectrum.md §8)
LSR_TOL_DB = 0.5
LSR_FLATNESS_TOL_DB = 0.3
FC_TOL_MHZ = 10.0
TAUG_TOL_FRAC = 0.01
NOISE_FLOOR_DB = -60.0

# レベル定義 (design_ascan_amplitude.md §4.3 と共通)
LEVEL_EFFECTS = {
    'Level_1': ['geom', 'surface_T'],
    'Level_2': ['geom', 'surface_T', 'absorb_const'],
    'Level_3': ['geom', 'surface_T', 'absorb_tandelta'],
    'Level_3b': ['geom', 'surface_T', 'absorb_debye'],
    'Level_4': ['geom', 'surface_T', 'absorb_tandelta', 'ice_layer'],
    'Level_5': ['geom', 'surface_T', 'absorb_tandelta', 'ice_layer',
                'density_profile'],
}

# 【はしごの順序変更について】
# 本研究の主眼が水氷であることから、Level 4 と Level 5 の中身を入れ替えた。
#   旧: Level 4 = 密度プロファイル（氷なし） / Level 5 = 密度 + 水氷
#   新: Level 4 = 水氷層（均質背景）        / Level 5 = 密度 + 水氷
# 均質背景に氷層を 1 枚置くほうが実装が軽く（材料 2 個、理論は 2 層 Fresnel の
# 解析解）、反射・走時チャネルが成立するかを最短で確認できるため。
# Level 4 の「氷なし」に相当するのは Level 3 そのものなので、対照は別途
# 用意する必要がない（Level 5 以降は背景が変わるのでペア実行が必要になる）。
#
# 注意: 本ツールは「地表 tx + 埋設 rx」（片道透過, A 系統）専用である。
# at_tx（地表 tx/rx で地下反射を見る B 系統）の解析は別ツールで扱う。
# したがって Level 4 の理論は、氷層を「透過して」深さ d に届く波の
# 振幅・走時であり、氷層からの反射そのものは扱わない。
IMPLEMENTED_LEVELS = {'Level_1', 'Level_2', 'Level_3', 'Level_3b',
                      'Level_4'}

# JSON の下位選択階層につけるラベル（階層が深いほうまで使う）
SUBLEVEL_LABELS = ['波形種別', 'サブ条件', '組成 (FeO+TiO2)', 'サブ条件']

# 解析対象から除外する rx キー (design_ascan_amplitude.md §3 と共通)
# =============================================================================
# [EDIT HERE] Level 2 の媒質パラメータ（損失モデル）
# =============================================================================
# gprMax の `#material: er sigma mr sigma*` が与えるのは「導電率 sigma 一定」で
# あり、ロスタンジェント一定ではない。等価的に eps'' = sigma/(omega eps0) なので
#
#     tan_delta = eps'' / eps' = sigma / (omega eps0 eps_r)  ∝ 1/f
#
# となり、sigma が一定なら tan_delta は 1/f で落ちる。逆に tan_delta を一定に
# したければ eps'' を周波数によらず一定（= sigma ∝ f）にする必要があり、
# gprMax では #add_dispersion_debye による多極 Debye でしか実現できない。
# さらに Kramers-Kronig の関係から eps'' を一定にすると eps' も必ず分散する。
# したがって「tan_delta 一定」は Level 3 の領域であり、Level 2 で eps' を
# 厳密に一定に保てるのは sigma 一定のときだけである（DC 導電率の KK 対応項は
# eps' に寄与しないため、sigma 一定は eps' 一定と両立する数少ない損失モデル）。
LEVEL2_LOSS_MODEL = 'conductivity'   # 'conductivity' … gprMax の #material に対応（既定）
                                     # 'tan_delta'    … 参考用。Level 3 相当の理想化
LEVEL2_SIGMA = 0.0035                # [S/m] #material の第 2 引数と一致させること。
                                     #   プロファイル計算の 0 vol% ice / 1.25 GHz の値。
                                     #   tan_delta = 0.01678 @ 1.25 GHz に相当。
LEVEL2_TAN_DELTA = 0.0155            # LEVEL2_LOSS_MODEL='tan_delta' のときのみ使う

ETA0 = 376.730313668                 # [Ohm] 真空の波動インピーダンス
EPS0 = 8.8541878128e-12              # [F/m] 真空の誘電率

# =============================================================================
# [EDIT HERE] Level 3 の媒質パラメータ（eps'' 一定）
# =============================================================================
# 用語について:
#   本コードでは Level 3 を「非分散」と呼ばない。eps'' != 0 の媒質は
#   Kramers-Kronig 則により eps' が必ず対数的に変化するため、厳密な意味で
#   非分散な損失媒質は存在しない。Level 3 の正しい記述は
#       eps'' = 一定（帯域内）、eps' は KK により約 0.37% 変化
#   である。eps' が定数なのは Level 1（無損失）と Level 2（sigma 一定。
#   DC 導電率の KK 対応項は eps' に寄与しない）だけ。
#   【未対応】本コードの理論曲線は現状 eps' を定数として扱っている。
#   eps'(f) に置き換える作業が修正項目 A-3。深さ 2.75 m の群遅延で
#   -0.032 ns（数値分散 +0.051 ns の 62%、符号は逆）の効きがある。
#
# eps'' が帯域内でほぼ一定であることの実測根拠は Boivin+2022:
#   Table 7 の純バイトウナイトは P/L/S/X 帯で eps'=3.29, eps''=0.006 が
#   完全一定。1, 5 wt% でも有意な分散を検出できずフィット不能と報告。
# したがって Level 3 は「eps'' 一定」= alpha ∝ f をベースラインとする。
#
# 損失の振幅は Carrier 経験式から与える。
#   出典: Carrier, Olhoeft & Mendell (1991), Lunar Sourcebook Ch.9,
#         Fig. 9.53 (SOILS = 土壌試料のみの回帰) の図中式
#     eps'      = 1.871^rho
#     tan_delta = 10^(0.027*(%TiO2 + %FeO) + 0.273*rho - 3.058)
#
#   Fig. 9.53 を選ぶ理由:
#     (a) 本研究の対象はレゴリス（土壌）であり、岩石片を含む Fig. 9.52
#         (ALL DATA) や Fig. 9.54 (450 MHz DATA) より母集団が適切。
#     (b) tan_delta の周波数依存を無視する立場をとる以上、周波数で切った
#         サブセット（Fig. 9.54 = 450 MHz）を選ぶのは仮定と矛盾する。
#         選ぶべき軸は周波数ではなく試料種別。
#     (c) 土壌データは rho ~ 1.0-2.1 に分布し、下の rho = 1.753647 は
#         その中心付近にある（内挿であって外挿でない）。
#
#   【重要】eps' の式と tan_delta の式は同一図・同一サブセットから取ること。
#   片方だけ他図の式に差し替えると自己整合が崩れる。参考（本節では使わない）:
#     Fig. 9.52 ALL DATA     : 1.919^rho, 10^(0.038 S + 0.312 rho - 3.260)
#     Fig. 9.54 450 MHz DATA : 1.843^rho, 10^(0.033 S + 0.231 rho - 3.061)
#                              -> Level 3b（分散モデル）でこちらを使う
#     Fig. 9.55 APOLLO 15-17 : 1.908^rho, 10^(0.028 S + 0.167 rho - 2.975)
#   4 式の振れ幅は 5-10 wt% で 0.80-1.09 倍、N_eff で 0.9-1.5 倍。
#
# Boivin のバイトウナイト値（tan_delta=0.002）を使わないのは、バイトウナイトが
# 純長石で Fe をほぼ含まないのに対し、実際の南極レゴリスは olivine/pyroxene
# 由来の FeO を 5-6 wt% 含むため。Boivin から採るのは「eps'' が周波数に
# 依らない」という形状情報だけにする。
#
# eps' は組成に依らず 3.0（設計判断）とし、密度はそこから一意に決まる。
#   eps' = 1.871^rho = 3.0  ->  rho = 1.753647
# eps' が組成に依らないので、全組成で走時・幾何減衰が共通になり、
# 違いは吸収だけという比較しやすい構成になる。
# 参考: rho = 1.753647 は Carrier の密度プロファイル
#       rho(z) = 1.92(z+12.2)/(z+18) [z:cm] の深さ約 49 cm に相当する。
LEVEL3_EPS_R = 3.0                # 全組成共通。基準周波数は帯域の幾何平均
                                  # f0 = sqrt(0.5*2.0) = 1.0 GHz（.in と共通）。
                                  # 帯域中心 1.25 GHz では 2.998（-0.07%）。
LEVEL3_RHO   = 1.753647           # [g/cm^3] eps' = 1.871^rho = 3.0 の逆算値

LEVEL3_CARRIER_EPS_BASE = 1.871   # Fig. 9.53 図中: eps' = 1.871^rho
LEVEL3_CARRIER_TAND_A   = 0.027   # Fig. 9.53 図中の 3 次元回帰
LEVEL3_CARRIER_TAND_B   = 0.273
LEVEL3_CARRIER_TAND_C   = 3.058

# FeO+TiO2 [wt%] ごとの設定。JSON のサブ階層キーと対応させる。
#   月南極域   : 5 / 7.5 / 10 wt%（先行研究の収束域 6-11 wt% を挟む）
#   高Tiバサルト: 20 wt%（月の海。参考ケース。既存計算がこれに相当）
LEVEL3_COMPOSITIONS = {
    'feo5':    5.0,
    'feo7p5':  7.5,
    'feo10':  10.0,
    'feo20':  20.0,
}
LEVEL3_DEFAULT_COMPOSITION = 'feo7p5'   # サブ階層で指定がないときの既定値


# =============================================================================
# [EDIT HERE] Level 3 の 2 極 Debye 実現（Level_3.in と同一の解析解）
# =============================================================================
# gprMax は eps'' 一定の材料を直接持てないため、Level_3.in は最大平坦の
# 2 極 Debye でそれを実現している。理論側でも同じ式を使えるようにしておく。
#
#   1 極 Debye の eps'' は対数周波数 u = ln(w*tau) で sech 関数になる。
#   u = ±s に等強度で 2 極置くと u=0 で最大平坦になる条件が
#       s = arcsinh(1) = ln(1+sqrt2)      -> tau 比 = (1+sqrt2)^2 = 5.8284
#   と閉形式で決まり、対称中心を帯域の幾何平均 f0 に取ると
#       1/(1+(w0*tau1)^2) + 1/(1+(w0*tau2)^2) = 1   （厳密に 1）
#   が成り立つので eps_inf も閉形式になる。
#       De      = sqrt(2) * eps''_target   （各極）
#       eps_inf = eps'_target - De
#
# --- 2 つのモードを分けている理由 --------------------------------------------
# LEVEL3_EPS_REAL_MODE = 'debye'（既定, 修正項目 A-3）
#   eps'' != 0 の媒質は Kramers-Kronig 則により eps' が必ず変化する。これは
#   実装の副作用ではなく物理なので、理論側にも入れる。深さ 2.75 m の群遅延で
#   -0.032 ns（数値分散 +0.051 ns の 62%、符号は逆）効く。'ideal' にすると
#   従来どおり eps' = 定数として扱う。
#
# LEVEL3_EPS_IMAG_MODE = 'ideal'（既定）
#   eps'' の帯域内リップル（帯域端で -2.44%）は 2 極近似の設計誤差であって
#   物理ではない。理論は設計目標値（一定）のままにして、fig3(a) の alpha(f)
#   に -2.5% のずれが見えるようにしておく。解析コードがその大きさの系統誤差を
#   検出できていることの確認になる。'debye' にすると実装値に合わせられる。
LEVEL3_EPS_REAL_MODE = 'debye'    # 'debye' … KK 由来の eps'(f) を理論に入れる
                                  # 'ideal' … eps' = LEVEL3_EPS_R 一定（旧挙動）
LEVEL3_EPS_IMAG_MODE = 'ideal'    # 'ideal' … eps'' = 設計目標値 一定（既定）
                                  # 'debye' … 2 極 Debye の実装値に合わせる

LEVEL3_DEBYE_BAND_HZ = (0.5e9, 2.0e9)   # Level_3.in の BAND_LO / BAND_HI と一致させる
LEVEL3_DEBYE_F0 = float(np.sqrt(LEVEL3_DEBYE_BAND_HZ[0] * LEVEL3_DEBYE_BAND_HZ[1]))
_L3_S_FLAT = float(np.arcsinh(1.0))          # = ln(1+sqrt2) = 0.881374
_L3_TAU_RATIO = float(np.exp(_L3_S_FLAT))    # = 1+sqrt2 = 2.414214
_L3_W0 = 2.0 * np.pi * LEVEL3_DEBYE_F0
LEVEL3_DEBYE_TAU = (1.0 / (_L3_W0 * _L3_TAU_RATIO), _L3_TAU_RATIO / _L3_W0)


def debye_flat_eps(eps_r_target, eps_imag_target, f):
    """最大平坦 2 極 Debye の (eps'(f), eps''(f))。Level_3.in と同一の式。

    eps_r_target / eps_imag_target は帯域の幾何平均 f0 での目標値。
    氷層のように背景と eps' が違う材料にもそのまま使える。
    """
    f_arr = np.asarray(f, dtype=float)
    de = np.sqrt(2.0) * eps_imag_target
    eps_inf = eps_r_target - de
    w = 2.0 * np.pi * f_arr
    x1 = w * LEVEL3_DEBYE_TAU[0]
    x2 = w * LEVEL3_DEBYE_TAU[1]
    er = eps_inf + de / (1.0 + x1 ** 2) + de / (1.0 + x2 ** 2)
    ei = de * x1 / (1.0 + x1 ** 2) + de * x2 / (1.0 + x2 ** 2)
    return er, ei


def apply_eps_modes(eps_r_target, eps_imag_target, f):
    """モード設定に従って (eps', eps'') を返す共通ヘルパ。

    Level 3（背景レゴリス）と Level 4（氷層）の両方から使う。
    """
    f_arr = np.asarray(f, dtype=float)
    er_d, ei_d = debye_flat_eps(eps_r_target, eps_imag_target, f_arr)
    er = er_d if LEVEL3_EPS_REAL_MODE == 'debye' \
        else np.full_like(f_arr, float(eps_r_target))
    ei = ei_d if LEVEL3_EPS_IMAG_MODE == 'debye' \
        else np.full_like(f_arr, float(eps_imag_target))
    return er, ei


# =============================================================================
# [EDIT HERE] Level 4 の媒質パラメータ（水氷層）
# =============================================================================
# 背景は Level 3 と完全に同一（均質レゴリス、eps'' 一定）。そこに水氷を含む
# 層を 1 枚挟む。Level_4.in の設定と必ず一致させること。
#
# --- 混合則（LLL 増分形）------------------------------------------------------
# 水氷はレゴリス粒子を置き換えるのではなく、粒子間の空隙（真空）に凝結する
# （Takekura et al. 2025, Remote Sensing 17, 1050 の Fig.2 と同じ描像）。
# LLL 混合則 eps^(1/3) = sum_i v_i eps_i^(1/3) を 3 相（粒子・真空・氷）に
# 適用すると、乾燥時との差で粒子の項が相殺して
#       eps'_wet^(1/3) = eps'_dry^(1/3) + v_ice * (eps_ice^(1/3) - 1)
# となる。この形なら粒子密度も粒子誘電率も式に現れず、乾燥側に Carrier の
# 経験式をそのまま使えるので経験式との接続が保たれる。
#
# 損失は粒子にあり氷はほぼ無損失なので eps'' は保存し、氷自身の微小な損失
# だけを加える。虚部に混合則を適用してはならない（損失源が増えていないのに
# 減衰が増えるという非物理的な結果になる）。
#       eps''_wet = eps''_dry + v_ice * eps_ice * tan_delta_ice
#
# 参考: 混合則の選択（LLL か対数混合か）による差は 1 vol% の界面反射で約
# 1.5 dB。「氷が空隙を埋める」か「レゴリスを置き換える」かの差（約 26 dB）に
# 比べて桁違いに小さい。重要なのは規則ではなく描像。
LEVEL4_ICE_TOP_M   = 1.00     # [m] 地表面から氷層上面までの深さ
LEVEL4_ICE_THICK_M = 1.00     # [m] 氷層の厚さ

LEVEL4_ICE_SPEC = 'vol'       # 'vol' … 体積パーセントで指定
                              # 'wt'  … 質量パーセントで指定
# 【単位に注意】どちらもパーセント。分率ではない（10 vol% なら 10.0）。
LEVEL4_ICE_VOL_PCT = 10.0     # [vol%]
LEVEL4_ICE_WT_PCT  = 0.5      # [wt%]

LEVEL4_EPS_ICE  = 3.15        # 氷の eps'（GHz 帯。低温での温度依存は小さい）
LEVEL4_TAND_ICE = 2.0e-4      # [要文献確認] 氷の tan_delta。低温ほど小さいので保守側
LEVEL4_RHO_ICE  = 0.94        # [g/cm^3] 82-110 K での氷の密度（wt% 換算用）
LEVEL4_RHO_GRAIN = 2.645      # [g/cm^3] 斜長岩の粒子密度。空隙率チェックにのみ使う

_L4_ICE_INC = LEVEL4_EPS_ICE ** (1.0 / 3.0) - 1.0      # = 0.46590

# =============================================================================
# [EDIT HERE] Level 3b の媒質パラメータ（2 極 Debye 分散 = 高Tiバサルト想定）
# =============================================================================
# Boivin+2022 の 20 wt% イルメナイト試料（= FeO+TiO2 20 wt% に化学量論的に対応）
# の Cole-Cole を 2 極 Debye でフィットした形状を使う。月の海の高Tiバサルトを
# 想定した参考ケースであり、月南極（Level_3）には適用しない。
#
# Level_3 との違いは「周波数依存性の有無」だけで、下表のとおり。
#   Level_3  : tan_delta 一定    -> alpha ∝ f       （帯域内 4.0 倍）
#   Level_3b : 2 極 Debye 分散   -> alpha ∝ f^1.38  （帯域内 6.8 倍）
#
# rho は「1.25 GHz で eps' = 3.0」になる値。Level_3 の rho（1.753647）と
# 異なるのは、Level_3b では eps' 自体が大きく分散するため、どの周波数で 3.0 に
# 揃えるかを決める必要があるからである。
#
# 【経験式が Level 3 と違うことに注意】
#   Level 3b は分散モデルなので「経験式の値はどの周波数の値か」を決めないと
#   スケールが定まらない。そのためアンカー周波数が特定できる唯一のサブセット
#   である Fig. 9.54 (450 MHz DATA) を使う。
#   Level 3 は eps'' 一定を仮定するのでアンカーが不要であり、試料種別で切った
#   Fig. 9.53 (SOILS) を使う。両者で eps' の底も tan_delta の係数も異なる。
#   この不統一は意図的なもので、モデルの性質の違いに由来する。
LEVEL3B_RHO      = 1.820224       # [g/cm^3] 1.25 GHz で eps' = 3.0 になる密度
LEVEL3B_FEOTIO2  = 20.0           # [wt%] 高Tiバサルト想定
LEVEL3B_ANCHOR_FREQ = 450e6       # [Hz] Carrier+1991 Fig. 9.54 の 450 MHz 計測

LEVEL3B_CARRIER_EPS_BASE = 1.843  # Fig. 9.54 図中: eps' = 1.843^rho
LEVEL3B_CARRIER_TAND_A   = 0.033  # Fig. 9.54 図中の 3 次元回帰
LEVEL3B_CARRIER_TAND_B   = 0.231
LEVEL3B_CARRIER_TAND_C   = 3.061
LEVEL3B_DEBYE_DE1, LEVEL3B_DEBYE_TAU1 = 0.261, 4.6212e-11    # [s]
LEVEL3B_DEBYE_DE2, LEVEL3B_DEBYE_TAU2 = 0.088, 2.82195e-10   # [s]

# 走時・探索窓の基準に使う周波数（帯域中心）
BAND_CENTRE_HZ = 1.25e9

# 解析対象から除外する rx キー (design_ascan_amplitude.md §3 と共通)
#   depth_300 は y=0.0 で PML（gprMax デフォルト 10 層 = 0.05 m）の中にあるため、
#   物理的に意味のあるデータにならない。JSON に残っていても自動で除外する。
EXCLUDE_KEYS = {'depth_300'}

# 作図
# fig3 の縦軸範囲を決めるパーセンタイル。浅い rx の 1/d 増幅による外れ値を
# 落として傾向を見やすくする。(0, 100) にすると最小最大になる。
ATTEN_YLIM_PCT = (2.0, 98.0)

FIGURE_FORMATS = ('png', 'pdf')   # すべての図をこの形式すべてで保存する
FIGURE_DPI = 300

# 出力先 (レベル親ディレクトリ配下)
OUTPUT_PARENT_DIRNAME = 'analysis'
OUTPUT_SUBDIRNAME = 'ascan_spectrum'

# 自然対数 -> 20*log10 への変換係数 (ln(x) * LN_TO_DB20 == 20*log10(x))
LN_TO_DB20 = 20.0 / np.log(10.0)


# =============================================================================
# 入出力 (ascan_amplitude.py と同一仕様)
# =============================================================================
def _same_dt(dt_a, dt_b, rtol=1e-9):
    """時間刻みが同一かを相対誤差だけで判定する。

    np.isclose の既定 atol=1e-8 は dt（1e-11 秒オーダー）より遥かに大きいため、
    そのまま使うと 2 倍違う dt でも「一致」と判定されてしまう。必ず atol=0 にする。
    """
    return abs(dt_a - dt_b) <= rtol * abs(dt_a)


def _select(items, label):
    """番号入力で 1 つ選ばせる。候補が 1 つなら自動選択。"""
    if len(items) == 1:
        print('{}: {} (候補が 1 つのため自動選択)'.format(label, items[0]))
        return items[0]
    print('利用可能な{}:'.format(label))
    for i, name in enumerate(items, 1):
        print('  {}: {}'.format(i, name))
    while True:
        choice = input('{}番号を選択してください (1-{}) > '.format(label, len(items))).strip()
        if choice.isdigit() and 1 <= int(choice) <= len(items):
            picked = items[int(choice) - 1]
            print('選択された{}: {}'.format(label, picked))
            return picked
        print('1 から {} の数字を入力してください。'.format(len(items)))


def _kind_layer(node):
    """さらに下位の選択階層かどうかを判定する。

    値がすべて dict なら「選択キー -> 下位ノード」の階層、
    値が文字列なら「rx -> パス」の終端階層とみなす。
    """
    entries = {k: v for k, v in node.items() if not k.startswith('_')}
    if entries and all(isinstance(v, dict) for v in entries.values()):
        return entries
    return None


def _descend(node, labels):
    """rx の階層（値が文字列）に着くまで、番号選択で下位へ降りる。

    JSON の階層の深さは枝ごとに違ってよい。たとえば

        Level_1/gaussiandot/depth_025            … 2 段
        Level_1/excitation_waveform/dx_0005/...  … 3 段

    のように混在していても、値が文字列になった時点で終端と判断する。

    Returns
    -------
    (選択したキーのリスト, rx 辞書)
    """
    chosen = []
    while True:
        nested = _kind_layer(node)
        if nested is None:
            break
        # 階層の中身からラベルを決める。組成キーが並んでいれば専用のラベルを使う。
        # （階層の深さは枝ごとに違いうるので、位置ではなく内容で判定する）
        if _is_composition_layer(nested):
            label = '組成 (FeO+TiO2)'
        elif _is_ice_layer(nested):
            label = '水氷濃度 (vol%)'
        else:
            label = labels[len(chosen)] if len(chosen) < len(labels) else labels[-1]
        key = _select(sorted(nested), label)
        chosen.append(key)
        node = nested[key]
    rx_paths = {k: v for k, v in node.items() if not k.startswith('_')}
    return chosen, rx_paths


def _pick_reference(ref_node, chosen):
    """_reference から、選択したキー列に対応するエントリを取り出す。

    _reference の階層は Level 側より浅くてよい。たとえば Level 側で
    ['excitation_waveform', 'dx_0005'] を選んでも、_reference が
    波形種別までしか分かれていなければ 'excitation_waveform' の中身を返す。
    dx を変えても波源そのものは同じなので、これは正しい振る舞いである。
    """
    if not ref_node:
        return {}
    node = ref_node
    for key in chosen:
        nested = _kind_layer(node)
        if nested is None:
            break                                  # ここが rx の階層
        if key not in nested:
            if len(nested) == 1:
                node = next(iter(nested.values()))  # 候補が 1 つなら自動で降りる
                continue
            raise CmdInputError(
                '_reference に "{}" のエントリがありません（候補: {}）。'
                'JSON の _reference を確認してください。'.format(key, ', '.join(sorted(nested))))
        node = nested[key]
    if _kind_layer(node) is not None:
        raise CmdInputError(
            '_reference の階層が Level 側より深く、rx まで辿り着けません（残り: {}）。'.format(
                ', '.join(sorted(_kind_layer(node)))))
    return {k: v for k, v in node.items() if not k.startswith('_')}


def load_paths(json_path):
    """JSON からレベルと波形種別を番号で選択し、rx パスと _reference を分離する。

    仕様は ascan_amplitude.py の load_paths() と同一（design_ascan_spectrum.md §2）。
    """
    if not os.path.exists(json_path):
        raise CmdInputError('JSON file {} does not exist'.format(json_path))
    with open(json_path) as f:
        all_paths = json.load(f)

    levels = [k for k in all_paths if not k.startswith('_')]
    if not levels:
        raise CmdInputError('{} に解析可能なレベルがありません'.format(json_path))
    level = _select(levels, 'レベル')

    # rx の階層に着くまで降りる（階層の深さは枝ごとに違ってよい）
    chosen, rx_paths = _descend(all_paths[level], SUBLEVEL_LABELS)
    kind = ' / '.join(chosen) if chosen else None
    if chosen:
        print('選択された条件: {} / {}'.format(level, kind))

    # 参照計算：トップレベルをレベル内の設定で上書きする
    reference = _pick_reference(all_paths.get('_reference', {}), chosen)
    level_ref = all_paths[level].get('_reference', {})
    if level_ref:
        reference.update(_pick_reference(level_ref, chosen))

    if reference:
        print('使用する参照計算:')
        for k in sorted(reference):
            print('    {:<12} {}'.format(k, reference[k]))

    excluded = sorted(set(rx_paths) & EXCLUDE_KEYS)
    for key in excluded:
        del rx_paths[key]
    if excluded:
        print('除外した rx (PML 内などのため解析対象外): {}'.format(', '.join(excluded)))

    if not rx_paths:
        raise CmdInputError('Level {} に解析可能な rx がありません'.format(level))
    return level, kind, rx_paths, reference


def check_paths_exist(rx_paths, reference):
    """実行前に全 .out の存在を確認する（design_ascan_spectrum.md §2）。

    far_1m 欠落は即エラー、他は通知して該当エントリを取り除いた上で続行する。
    rx_paths / reference は破壊的に更新される。
    """
    if 'far_1m' not in reference:
        raise CmdInputError('_reference.far_1m が JSON にありません（E_ref(f) の校正に必要）')
    if not os.path.exists(reference['far_1m']):
        raise CmdInputError('_reference.far_1m のファイルが存在しません: {}'.format(reference['far_1m']))

    missing_rx = [k for k, p in rx_paths.items() if not os.path.exists(p)]
    for key in missing_rx:
        print('Notice: rx "{}" の .out が存在しないため解析対象から除外します: {}'.format(key, rx_paths[key]))
        del rx_paths[key]
    if not rx_paths:
        raise CmdInputError('解析可能な rx がありません（全て .out が欠落）')

    for key in ('at_tx', 'at_surface'):
        if key in reference and not os.path.exists(reference[key]):
            print('Notice: _reference.{} の .out が存在しないため、関連する解析をスキップします: {}'.format(
                key, reference[key]))
            del reference[key]


def resolve_output_dir(level, rx_paths):
    """レベル親ディレクトリ配下に出力ディレクトリのパスを組み立てる（ascan_amplitude.py と同一）。"""
    paths = [os.path.abspath(p) for p in rx_paths.values()]
    if len(paths) > 1:
        level_root = os.path.commonpath(paths)
    else:
        level_root = os.path.dirname(os.path.dirname(os.path.dirname(paths[0])))

    if level not in os.path.basename(level_root):
        print('Warning: 推定した親ディレクトリ "{}" に選択レベル "{}" が含まれていません。'
              'JSON のパス構成を確認してください。'.format(level_root, level))

    return os.path.join(level_root, OUTPUT_PARENT_DIRNAME, OUTPUT_SUBDIRNAME)


def load_trace(path):
    """.out から Ez の A-scan を読み込む（tools.core.outputfiles_merge 経由）。"""
    outputdata, dt = get_output_data(path, 1, 'Ez')
    outputdata = np.asarray(outputdata)
    if outputdata.ndim > 1:
        outputdata = outputdata[:, 0]
    return outputdata, dt


def rx_depth(key):
    """rx キー名から深さ[m]を求める。"at_surface" は 0.0、"at_tx" は None（別扱い）。"""
    if key == 'at_surface':
        return 0.0
    if key == 'at_tx':
        return None
    m = re.match(r'^depth_(\d+)$', key)
    if not m:
        raise CmdInputError('Unrecognised rx key: {}'.format(key))
    return int(m.group(1)) / 100.0


# =============================================================================
# 前処理
# =============================================================================
def apply_gate(trace, dt, t_center_ns):
    """時間ゲートをかける（GATE_ENABLED=True のときのみ有効）。

    参照・実測・理論すべてに同じ幅・同じ形状のゲートを、理論波形の
    包絡ピーク位置を中心にかける（design_ascan_spectrum.md §4.2）。
    """
    if not GATE_ENABLED:
        return trace
    n_samples = len(trace)
    dt_ns = dt * 1e9
    t_axis = np.arange(n_samples) * dt_ns
    lo, hi = t_center_ns - GATE_HALFWIDTH_NS, t_center_ns + GATE_HALFWIDTH_NS
    idx = np.where((t_axis >= lo) & (t_axis <= hi))[0]
    if len(idx) == 0:
        raise CmdInputError('ゲート窓 [{:.2f}, {:.2f}] ns 内にサンプルがありません'.format(lo, hi))
    window = np.zeros(n_samples)
    window[idx] = signal.windows.tukey(len(idx), alpha=GATE_TAPER)
    return trace * window


def spectrum(trace, dt):
    """FFT。実測・理論・参照すべてに同一処理を適用する（design_ascan_spectrum.md §4.1）。"""
    E = np.fft.rfft(trace)
    f = np.fft.rfftfreq(len(trace), d=dt)
    return f, E


def measure_peak(trace, dt):
    """包絡のピーク時刻[ns]を全区間から求める（波源遅延・ゲート中心の推定に使用）。"""
    dt_ns = dt * 1e9
    envelope = np.abs(signal.hilbert(trace))
    peak_idx = int(np.argmax(envelope))
    n_samples = len(trace)
    if 0 < peak_idx < n_samples - 1:
        y0, y1, y2 = envelope[peak_idx - 1], envelope[peak_idx], envelope[peak_idx + 1]
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        delta = float(np.clip(delta, -1.0, 1.0))
    else:
        delta = 0.0
    return {'t_peak': (peak_idx + delta) * dt_ns, 'amp_peak': envelope[peak_idx]}


def _interp_complex_to_grid(freq_from, E_from, freq_to):
    """複素スペクトルを別の周波数グリッドへ線形内挿する（実部・虚部を別々に）。"""
    real = np.interp(freq_to, freq_from, E_from.real)
    imag = np.interp(freq_to, freq_from, E_from.imag)
    return real + 1j * imag


# =============================================================================
# 理論：伝達関数 (ascan_amplitude.py と同一)
# =============================================================================
def transfer_geom(d, n):
    """幾何減衰項（2D遠方場、法線入射の見かけの源距離 r_eff を用いる）。"""
    r_eff = n * TX_HEIGHT + d
    return np.sqrt(n / r_eff) * np.sqrt(R_REF)


def transfer_surface_T(n):
    """地表面透過係数 T = 1 + R（Ez は界面に接線 → 連続）。深さに依存しない定数。"""
    return 2.0 / (1.0 + n)


def transfer_phase(f, d, n):
    """走時位相項。参照計算自身の伝搬遅延 (r_ref/c) を差し引いて基準を揃える。"""
    t_arr = TX_HEIGHT / C + n * d / C
    delay_s = (t_arr - R_REF / C) * 1e-9
    return np.exp(-2j * np.pi * f * delay_s), t_arr


def level2_tandelta(f, n):
    """Level 2 の tan_delta(f)。

    conductivity モデル : tan_delta = sigma / (omega eps0 eps_r)  ∝ 1/f
    tan_delta モデル     : 定数
    """
    f_arr = np.asarray(f, dtype=float)
    if LEVEL2_LOSS_MODEL == 'conductivity':
        eps_r = n ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(f_arr > 0,
                            LEVEL2_SIGMA / (2.0 * np.pi * np.maximum(f_arr, 1e-30)
                                            * EPS0 * eps_r),
                            np.nan)
    if LEVEL2_LOSS_MODEL == 'tan_delta':
        return np.full_like(f_arr, LEVEL2_TAN_DELTA)
    raise CmdInputError(
        "LEVEL2_LOSS_MODEL は 'conductivity' か 'tan_delta' にしてください: {}".format(
            LEVEL2_LOSS_MODEL))


def level2_alpha(f, n):
    """Level 2 の減衰係数 alpha(f) [Np/m]（厳密式）。

        alpha = (omega/c) * sqrt(eps_r / 2) * sqrt( sqrt(1 + tan_delta^2) - 1 )

    低損失極限 (tan_delta << 1) では
        conductivity モデル : alpha -> sigma * eta0 / (2 n)   ← 周波数に依存しない
        tan_delta モデル     : alpha -> pi f n tan_delta / c   ← f に比例
    に一致する。sigma=0.0035 S/m では両者の差は 0.02% 以下だが、
    近似を持ち込まないよう厳密式で実装している。
    """
    f_arr = np.asarray(f, dtype=float)
    td = level2_tandelta(f_arr, n)
    w_over_c = 2.0 * np.pi * f_arr / (C * 1e9)          # C は m/ns なので秒系に直す
    with np.errstate(invalid='ignore'):
        alpha = w_over_c * np.sqrt(n ** 2 / 2.0) * np.sqrt(np.sqrt(1.0 + td ** 2) - 1.0)
    return np.nan_to_num(alpha, nan=0.0)


def describe_level2_medium(n):
    """Level 2 の損失設定を人が読める形で返す（ログと run_info 用）。"""
    f0 = 1.25e9
    td0 = float(level2_tandelta(np.array([f0]), n)[0])
    a0 = float(level2_alpha(np.array([f0]), n)[0])
    if LEVEL2_LOSS_MODEL == 'conductivity':
        return ('loss model = conductivity, sigma = {:.6g} S/m  '
                '(tan_delta = {:.5f} @1.25 GHz, alpha = {:.4f} Np/m, '
                'alpha は帯域内でほぼ一定)'.format(LEVEL2_SIGMA, td0, a0))
    sigma_eq = 2.0 * np.pi * f0 * EPS0 * (n ** 2) * LEVEL2_TAN_DELTA
    return ('loss model = tan_delta, tan_delta = {:.5f}  '
            '(sigma = {:.6g} S/m @1.25 GHz 相当, alpha = {:.4f} Np/m @1.25 GHz, '
            'alpha は f に比例)'.format(LEVEL2_TAN_DELTA, sigma_eq, a0))


def transfer_absorb(f, d, n):
    """媒質の吸収項 exp(-alpha(f) * d)（片道透過）。Level 2 以降で使う。"""
    return np.exp(-level2_alpha(f, n) * d)


def level3_carrier_tandelta(feotio2_wt, rho=None):
    """Carrier 経験式（Lunar Sourcebook Fig. 9.53, SOILS）の tan_delta。

        tan_delta = 10^(0.027*(%TiO2 + %FeO) + 0.273*rho - 3.058)

    この式は周波数を説明変数に持たないため、周波数に依らない量として扱う。
    その扱いの根拠は Boivin+2022（section 冒頭のコメントを参照）。
    """
    rho = LEVEL3_RHO if rho is None else rho
    return 10.0 ** (LEVEL3_CARRIER_TAND_A * feotio2_wt
                    + LEVEL3_CARRIER_TAND_B * rho - LEVEL3_CARRIER_TAND_C)


def _parse_composition_key(key):
    """組成キーから FeO+TiO2 [wt%] を読み取る。表記ゆれに強くする。

    JSON のキー名とコードの定数名が食い違うと、既定値に落ちたまま
    気づかず誤った理論曲線で解析してしまう（実際に起きた）。
    そこで LEVEL3_COMPOSITIONS の完全一致に加えて、
    'FeO_100' / 'feo10' / 'FEO_7P5' のような表記からも数値を読み取る。

      FeO_050 -> 5.0     （3 桁ゼロ詰め = 10 倍表記とみなす）
      FeO_075 -> 7.5
      FeO_100 -> 10.0
      feo5    -> 5.0
      feo7p5  -> 7.5

    読み取れなければ None を返す（呼び出し側でエラーにする）。
    """
    if key in LEVEL3_COMPOSITIONS:
        return LEVEL3_COMPOSITIONS[key]

    m = re.fullmatch(r'(?i)feo[_-]?(\d+)(?:[p.](\d+))?', str(key))
    if not m:
        return None
    intpart, frac = m.group(1), m.group(2)
    if frac is not None:                       # feo7p5 形式
        return float('{}.{}'.format(intpart, frac))
    if len(intpart) >= 3:                      # FeO_050 形式（3 桁 = 10 倍表記）
        return int(intpart) / 10.0
    return float(intpart)                      # feo5, feo10 形式


def _is_composition_layer(keys):
    """その階層が組成の階層かどうかを判定する（ラベル表示用）。"""
    return bool(keys) and all(_parse_composition_key(k) is not None for k in keys)


def level3_feotio2(kind):
    """選択されたサブ階層キーから FeO+TiO2 [wt%] を取り出す。

    kind は load_paths が返す ' / ' 区切りのキー列
    （例 'excitation_waveform / dx_00025 / FeO_100'）。

    組成キーが 1 つも見つからない場合は、既定値に落とさずエラーにする。
    黙って既定値を使うと、誤った理論曲線のまま解析が完走してしまうため。
    """
    if kind:
        for token in str(kind).replace('/', ' ').split():
            wt = _parse_composition_key(token)
            if wt is not None:
                return wt, token
    raise CmdInputError(
        'Level 3 の組成を JSON のキーから判定できませんでした（選択: {}）。\n'
        'キー名を FeO_050 / FeO_075 / FeO_100 や feo5 / feo7p5 / feo10 の形式に'
        'するか、LEVEL3_COMPOSITIONS に追加してください。\n'
        '（既定値に落とすと誤った tan_delta で解析が完走してしまうため、'
        'あえてエラーにしています）'.format(kind))


# 解析対象の FeO+TiO2。main() が JSON の選択結果から設定する。
_LEVEL3_ACTIVE_WT = LEVEL3_COMPOSITIONS[LEVEL3_DEFAULT_COMPOSITION]
_LEVEL3_ACTIVE_KEY = LEVEL3_DEFAULT_COMPOSITION


def set_level3_composition(kind):
    """JSON の選択結果から Level 3 の組成を設定する（main から呼ぶ）。"""
    global _LEVEL3_ACTIVE_WT, _LEVEL3_ACTIVE_KEY
    _LEVEL3_ACTIVE_WT, _LEVEL3_ACTIVE_KEY = level3_feotio2(kind)
    return _LEVEL3_ACTIVE_WT, _LEVEL3_ACTIVE_KEY


# -----------------------------------------------------------------------------
# 水氷濃度の自動判定（組成の仕組みと同じ構造）
# -----------------------------------------------------------------------------
# JSON のサブ階層キー（例 'f_ice_10', 'f_ice_20'）から氷の体積パーセントを
# 読み取り、LEVEL4_ICE_VOL_PCT に反映する。
#
# 【単位の規約】キーの数字は体積パーセント。分率ではない。
#   f_ice_10   -> 10.0 vol%      （f_ice = 0.10 に対応）
#   f_ice_20   -> 20.0 vol%
#   f_ice_005  -> 0.5  vol%      （3 桁以上はゼロ詰めの 10 倍表記とみなす。
#                                  FeO_050 -> 5.0 と同じ規約）
#   f_ice_0p5  -> 0.5  vol%
#   f_ice_0    -> 0.0  vol%      （氷なし参照。Level 3 と同一になる）
# 明示的に指定したいキーは LEVEL4_ICE_KEYS に書けば優先される。
LEVEL4_ICE_KEYS = {}      # 例: {'f_ice_lowest': 0.5}


def _parse_ice_key(key):
    """キー名から氷の体積パーセントを読み取る。読めなければ None。"""
    if key in LEVEL4_ICE_KEYS:
        return float(LEVEL4_ICE_KEYS[key])

    m = re.fullmatch(r'(?i)(?:f[_-]?)?ice[_-]?(\d+)(?:[p.](\d+))?(?:[_-]?vol)?',
                     str(key))
    if not m:
        return None
    intpart, frac = m.group(1), m.group(2)
    if frac is not None:                       # f_ice_0p5 形式
        return float('{}.{}'.format(intpart, frac))
    if len(intpart) >= 3:                      # f_ice_005 形式（3 桁 = 10 倍表記）
        return int(intpart) / 10.0
    return float(intpart)                      # f_ice_10, f_ice_20 形式


def _is_ice_layer(keys):
    """その階層が氷濃度の階層かどうかを判定する（ラベル表示用）。"""
    return bool(keys) and all(_parse_ice_key(k) is not None for k in keys)


def level4_ice_volpct(kind):
    """選択されたサブ階層キーから氷の体積パーセントを取り出す。

    kind は load_paths が返す ' / ' 区切りのキー列
    （例 'excitation_waveform / dx_00025 / FeO_075 / f_ice_10'）。

    氷キーが 1 つも見つからない場合は、既定値に落とさずエラーにする。
    黙って既定値を使うと、誤った氷濃度の理論曲線のまま解析が完走してしまう
    ため（組成の判定と同じ方針）。
    """
    if kind:
        for token in str(kind).replace('/', ' ').split():
            vol = _parse_ice_key(token)
            if vol is not None:
                return vol, token
    raise CmdInputError(
        'Level 4 の水氷濃度を JSON のキーから判定できませんでした（選択: {}）。\n'
        'キー名を f_ice_10 / f_ice_20 / f_ice_005 の形式にするか、'
        'LEVEL4_ICE_KEYS に追加してください。\n'
        '（既定値に落とすと誤った氷濃度で解析が完走してしまうため、'
        'あえてエラーにしています）'.format(kind))


# 解析対象の氷濃度キー。main() が JSON の選択結果から設定する。
_LEVEL4_ACTIVE_ICE_KEY = None


def set_level4_ice(kind):
    """JSON の選択結果から Level 4 の水氷濃度を設定する（main から呼ぶ）。

    LEVEL4_ICE_VOL_PCT を上書きし、指定方法を 'vol' に固定する。
    ファイル冒頭で LEVEL4_ICE_SPEC='wt' にしていても、JSON のキーが
    体積パーセントである以上そちらを優先する。
    """
    global LEVEL4_ICE_VOL_PCT, LEVEL4_ICE_SPEC, _LEVEL4_ACTIVE_ICE_KEY
    vol, key = level4_ice_volpct(kind)
    LEVEL4_ICE_VOL_PCT = vol
    LEVEL4_ICE_SPEC = 'vol'
    _LEVEL4_ACTIVE_ICE_KEY = key
    return vol, key


def level3_targets(feotio2_wt=None):
    """Level 3 の設計目標値 (eps'_target, eps''_target) を返す。

    帯域の幾何平均 f0 = 1.0 GHz における値であり、Level_3.in が
    debye_flat_eps_imag() に渡す 2 つの数そのもの。
    """
    wt = _LEVEL3_ACTIVE_WT if feotio2_wt is None else feotio2_wt
    er_t = LEVEL3_EPS_R
    return er_t, er_t * level3_carrier_tandelta(wt)


def level3_eps(f, feotio2_wt=None):
    """Level 3 の複素比誘電率 (eps', eps'')。

    LEVEL3_EPS_REAL_MODE / LEVEL3_EPS_IMAG_MODE の設定に従う。
    既定では eps' は実装した 2 極 Debye の値（KK 由来の 0.43% の変化を含む。
    修正項目 A-3）、eps'' は設計目標値の一定値。

    f と同じ形の配列で返す（呼び出し側の一貫性のため）。
    """
    er_t, ei_t = level3_targets(feotio2_wt)
    return apply_eps_modes(er_t, ei_t, f)


def level3_tandelta(f, feotio2_wt=None):
    """Level 3 の tan_delta(f)。eps'' 一定なのでほぼ定数。"""
    er, ei = level3_eps(f, feotio2_wt)
    return ei / er


def level3_alpha(f, feotio2_wt=None):
    """Level 3 の減衰係数 alpha(f) [Np/m]（厳密式）。

        alpha = (omega/c) * sqrt(eps'/2) * sqrt( sqrt(1 + tan_delta^2) - 1 )

    tan_delta が一定なので alpha は f にほぼ比例する（帯域内で 4.0 倍）。
    Level 2（sigma 一定, alpha ∝ f^0）との違いがここに現れる。
    """
    f_arr = np.asarray(f, dtype=float)
    er, ei = level3_eps(f_arr, feotio2_wt)
    td = ei / er
    w_over_c = 2.0 * np.pi * f_arr / (C * 1e9)
    with np.errstate(invalid='ignore'):
        alpha = w_over_c * np.sqrt(er / 2.0) * np.sqrt(np.sqrt(1.0 + td ** 2) - 1.0)
    return np.nan_to_num(alpha, nan=0.0)


def _group_index_from_eps(eps_fn, f, rel_df=1e-3):
    """eps'(f) を返す関数から群屈折率 n_g = n + f dn/df を数値微分で求める。"""
    f_arr = np.atleast_1d(np.asarray(f, dtype=float))
    df = np.maximum(np.abs(f_arr) * rel_df, 1.0e3)
    n0 = np.sqrt(eps_fn(f_arr))
    np1 = np.sqrt(eps_fn(f_arr + df))
    nm1 = np.sqrt(eps_fn(f_arr - df))
    return n0 + f_arr * (np1 - nm1) / (2.0 * df)


def level3_group_index(f, feotio2_wt=None):
    """Level 3 の群屈折率。LEVEL3_EPS_REAL_MODE='ideal' なら n と一致する。"""
    return _group_index_from_eps(
        lambda ff: level3_eps(ff, feotio2_wt)[0], f)


# =============================================================================
# 理論：Level 4（水氷層を透過して深さ d に届く波）
# =============================================================================
def level4_ice_volume_fraction():
    """氷の体積分率（0-1 の分率）。入力はパーセントで受け取る。

        v_ice = (rho_bulk / rho_ice) * w / (1 - w)     Takekura+2025 Eq.(8)
    """
    if LEVEL4_ICE_SPEC == 'vol':
        v = LEVEL4_ICE_VOL_PCT / 100.0
    elif LEVEL4_ICE_SPEC == 'wt':
        w = LEVEL4_ICE_WT_PCT / 100.0
        v = (LEVEL3_RHO / LEVEL4_RHO_ICE) * w / (1.0 - w)
    else:
        raise CmdInputError("LEVEL4_ICE_SPEC は 'wt' か 'vol'")
    if v < 0.0:
        raise CmdInputError('氷の体積分率が負です')
    porosity = 1.0 - LEVEL3_RHO / LEVEL4_RHO_GRAIN
    if v > porosity:
        raise CmdInputError(
            '氷の体積分率 {:.4f} が空隙率 {:.4f} を超えています'.format(v, porosity))
    return v


def level4_targets(feotio2_wt=None):
    """(背景の目標値, 氷層の目標値) をそれぞれ (eps', eps'') の組で返す。

    Level_4.in の mix_ice() と同じ順序（目標値を混合してから Debye 化）。
    """
    er_dry, ei_dry = level3_targets(feotio2_wt)
    v = level4_ice_volume_fraction()
    er_ice = (er_dry ** (1.0 / 3.0) + v * _L4_ICE_INC) ** 3
    ei_ice = ei_dry + v * LEVEL4_EPS_ICE * LEVEL4_TAND_ICE
    return (er_dry, ei_dry), (er_ice, ei_ice)


def level4_eps(f, in_ice, feotio2_wt=None):
    """Level 4 の複素比誘電率。in_ice=True で氷層、False で背景レゴリス。"""
    dry, ice = level4_targets(feotio2_wt)
    er_t, ei_t = ice if in_ice else dry
    return apply_eps_modes(er_t, ei_t, f)


def level4_tandelta(f, in_ice=False, feotio2_wt=None):
    """Level 4 の tan_delta(f)。既定は背景レゴリス。"""
    er, ei = level4_eps(f, in_ice, feotio2_wt)
    return ei / er


def level4_alpha(f, in_ice=False, feotio2_wt=None):
    """Level 4 の減衰係数 alpha(f) [Np/m]（厳密式）。既定は背景レゴリス。"""
    f_arr = np.asarray(f, dtype=float)
    er, ei = level4_eps(f_arr, in_ice, feotio2_wt)
    td = ei / er
    w_over_c = 2.0 * np.pi * f_arr / (C * 1e9)
    with np.errstate(invalid='ignore'):
        alpha = w_over_c * np.sqrt(er / 2.0) * np.sqrt(np.sqrt(1.0 + td ** 2) - 1.0)
    return np.nan_to_num(alpha, nan=0.0)


def level4_segments(depth_m):
    """地表 (0) から深さ d までの経路を [(区間長, 氷層か), ...] に分ける。"""
    top = float(LEVEL4_ICE_TOP_M)
    bot = top + float(LEVEL4_ICE_THICK_M)
    d = float(depth_m)
    edges = sorted({0.0, d, min(max(top, 0.0), d), min(max(bot, 0.0), d)})
    segs = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a <= 0.0:
            continue
        mid = 0.5 * (a + b)
        segs.append((b - a, top <= mid < bot))
    return segs


def level4_interfaces_crossed(depth_m):
    """深さ d までに横切る氷層界面の数（0, 1, 2）を返す。"""
    top = float(LEVEL4_ICE_TOP_M)
    bot = top + float(LEVEL4_ICE_THICK_M)
    d = float(depth_m)
    return int(d > top) + int(d > bot)


def level4_alpha_path_avg(f, depth_m, feotio2_wt=None):
    """経路平均の減衰係数 (1/d) * ∫alpha dz。

    fig3 で LSR から逆算される alpha はこの量に対応する（単層なら
    level4_alpha と一致する）。
    """
    d = float(depth_m)
    if d <= 0.0:
        return level4_alpha(f, False, feotio2_wt)
    acc = np.zeros_like(np.asarray(f, dtype=float))
    for length, in_ice in level4_segments(d):
        acc = acc + level4_alpha(f, in_ice, feotio2_wt) * length
    return acc / d


def transfer_absorb_layered(f, d, n=None):
    """Level 4 の吸収項 exp(-∫alpha dz)（片道透過、層ごとに積分）。"""
    acc = np.zeros_like(np.asarray(f, dtype=float))
    for length, in_ice in level4_segments(d):
        acc = acc + level4_alpha(f, in_ice) * length
    return np.exp(-acc)


def transfer_ice_T(f, d, n=None):
    """氷層界面の透過係数の積（Ez は界面に接線なので T = 2 n1/(n1+n2)）。

    上面（レゴリス -> 氷）と下面（氷 -> レゴリス）を横切った分だけ掛ける。
    両方を横切ると積は 4 n1 n2/(n1+n2)^2 となりほぼ 1（10 vol% で -0.005 dB）
    だが、氷層の内部に rx がある場合は上面の 1 回分だけが効く（-0.21 dB）。
    多重反射は R^2 のオーダー（10 vol% で -64 dB）なので無視する。
    """
    n_reg = np.sqrt(level4_eps(f, False)[0])
    n_ice = np.sqrt(level4_eps(f, True)[0])
    crossed = level4_interfaces_crossed(d)
    T = np.ones_like(n_reg)
    if crossed >= 1:
        T = T * (2.0 * n_reg / (n_reg + n_ice))      # レゴリス -> 氷
    if crossed >= 2:
        T = T * (2.0 * n_ice / (n_reg + n_ice))      # 氷 -> レゴリス
    return T


def transfer_geom_layered(f, d):
    """層構造での幾何減衰項。

    見かけ源距離を界面ごとに更新する（近軸の屈折則 r -> r * n_new/n_old）。
    界面がなければ r_eff = n*TX_HEIGHT + d となり transfer_geom と一致する。
    """
    n_reg = np.sqrt(level4_eps(f, False)[0])
    n_ice = np.sqrt(level4_eps(f, True)[0])
    r = n_reg * TX_HEIGHT           # 真空 -> レゴリスの見かけ源距離
    n_prev = n_reg
    for length, in_ice in level4_segments(d):
        n_cur = n_ice if in_ice else n_reg
        r = r * (n_cur / n_prev) + length
        n_prev = n_cur
    return np.sqrt(n_prev / r) * np.sqrt(R_REF)


def transfer_phase_layered(f, d):
    """層構造での走時位相項。t_arr = h/c + Σ n_i L_i / c。"""
    n_reg = np.sqrt(level4_eps(f, False)[0])
    n_ice = np.sqrt(level4_eps(f, True)[0])
    t_arr_f = np.full_like(n_reg, TX_HEIGHT / C)
    for length, in_ice in level4_segments(d):
        t_arr_f = t_arr_f + (n_ice if in_ice else n_reg) * length / C
    delay_s = (t_arr_f - R_REF / C) * 1e-9
    return np.exp(-2j * np.pi * f * delay_s), t_arr_f


def level4_group_arrival(d, f0=None):
    """包絡ピークの位置に対応する群走時 [ns]（スカラー）。"""
    fc = BAND_CENTRE_HZ if f0 is None else f0
    ng_reg = float(_group_index_from_eps(
        lambda ff: level4_eps(ff, False)[0], np.array([fc]))[0])
    ng_ice = float(_group_index_from_eps(
        lambda ff: level4_eps(ff, True)[0], np.array([fc]))[0])
    t = TX_HEIGHT / C
    for length, in_ice in level4_segments(d):
        t += (ng_ice if in_ice else ng_reg) * length / C
    return t


def describe_level4_medium():
    """Level 4 の氷層設定を人が読める形で返す（ログと run_info 用）。"""
    v = level4_ice_volume_fraction()
    (er_d, ei_d), (er_i, ei_i) = level4_targets()
    wt_pct = 100.0 * v * LEVEL4_RHO_ICE / (LEVEL3_RHO + v * LEVEL4_RHO_ICE)
    n0, n1 = np.sqrt(er_d), np.sqrt(er_i)
    R = (n0 - n1) / (n0 + n1)
    a_d = float(level4_alpha(np.array([BAND_CENTRE_HZ]), False)[0])
    a_i = float(level4_alpha(np.array([BAND_CENTRE_HZ]), True)[0])
    key_note = ('' if _LEVEL4_ACTIVE_ICE_KEY is None
                else ' [{}]'.format(_LEVEL4_ACTIVE_ICE_KEY))
    return ('ice layer {:.3f} vol%{} ({:.3f} wt%) at {:.2f}-{:.2f} m, '
            'LLL mixing (pore filling), '
            "eps' {:.6f} -> {:.6f} ({:+.2f}%), eps'' {:.6f} -> {:.6f} ({:+.3f}%), "
            'alpha {:+.2f}% @{:.2f} GHz, interface R = {:.1f} dB'
            .format(100.0 * v, key_note, wt_pct, LEVEL4_ICE_TOP_M,
                    LEVEL4_ICE_TOP_M + LEVEL4_ICE_THICK_M,
                    er_d, er_i, 100.0 * (er_i / er_d - 1.0),
                    ei_d, ei_i, 100.0 * (ei_i / ei_d - 1.0),
                    100.0 * (a_i / a_d - 1.0), BAND_CENTRE_HZ / 1e9,
                    20.0 * np.log10(abs(R))))


def refractive_index(f, level):
    """レベルに応じた屈折率 n を返す。

    Level 1（無損失）と Level 2（sigma 一定）は eps' が厳密に定数。
    Level 3・4 は eps'' 一定なので KK により eps' が約 0.43% 変化する。
    LEVEL3_EPS_REAL_MODE='debye'（既定, 修正項目 A-3）ならその変化を含めた
    n(f) を返し、'ideal' なら従来どおり定数を返す。
    Level 4 でここが返すのは背景レゴリスの n（地表透過係数などに使う）で、
    氷層を含む経路の計算は transfer_*_layered が別途行う。
    f と同じ形の配列で返す。
    """
    f_arr = np.asarray(f, dtype=float)
    if 'absorb_tandelta' in LEVEL_EFFECTS[level]:
        er, _ = level3_eps(f_arr)
        return np.sqrt(er)
    if 'absorb_debye' in LEVEL_EFFECTS[level]:
        er, _ = level3b_eps(f_arr)
        return np.sqrt(er)
    return np.full_like(f_arr, N_REGOLITH)


def describe_level3_medium():
    """Level 3 の媒質設定を人が読める形で返す（ログと run_info 用）。"""
    wt = _LEVEL3_ACTIVE_WT
    td = level3_carrier_tandelta(wt)
    a_lo = float(level3_alpha(np.array([0.5e9]))[0])
    a_hi = float(level3_alpha(np.array([2.0e9]))[0])
    return ('constant eps_imag (tan_delta ~ const), FeO+TiO2 = {:.1f} wt% [{}], '
            'rho = {:.6f}, eps_r = {:.3f}, tan_delta = {:.6f}  '
            '(alpha = {:.4f} -> {:.4f} Np/m over 0.5-2.0 GHz, ratio {:.3f}) '
            "[eps' mode = {}, eps'' mode = {}]"
            .format(wt, _LEVEL3_ACTIVE_KEY, LEVEL3_RHO, LEVEL3_EPS_R, td,
                    a_lo, a_hi, a_hi / a_lo,
                    LEVEL3_EPS_REAL_MODE, LEVEL3_EPS_IMAG_MODE))


def transfer_absorb_tandelta(f, d, n=None):
    """Level 3 の吸収項 exp(-alpha(f) * d)（片道透過、eps'' 一定）。"""
    return np.exp(-level3_alpha(f) * d)


def level3b_carrier(rho=None):
    """密度 -> Carrier 経験式の (eps', eps'')。Level 3b 専用。

    Level 3 と違い Fig. 9.54 (450 MHz DATA) の式を使う。分散モデルでは
    「経験式の値がどの周波数の値か」を決める必要があり、周波数が特定できる
    サブセットはこれだけであるため。返す値は 450 MHz における値として扱い、
    level3b_debye_scale() で 2 極 Debye をこの値に合わせる。
    """
    rho = LEVEL3B_RHO if rho is None else rho
    eps_re = LEVEL3B_CARRIER_EPS_BASE ** rho
    tan_d = 10.0 ** (LEVEL3B_CARRIER_TAND_A * LEVEL3B_FEOTIO2
                     + LEVEL3B_CARRIER_TAND_B * rho - LEVEL3B_CARRIER_TAND_C)
    return eps_re, eps_re * tan_d


def level3b_debye_scale(rho=None):
    """eps''(450 MHz) が Heiken に一致するよう 2 極 Debye をスケールする係数。"""
    _, eps_im_h = level3b_carrier(rho)
    w = 2.0 * np.pi * LEVEL3B_ANCHOR_FREQ
    unit = (LEVEL3B_DEBYE_DE1 * w * LEVEL3B_DEBYE_TAU1 / (1.0 + (w * LEVEL3B_DEBYE_TAU1) ** 2)
            + LEVEL3B_DEBYE_DE2 * w * LEVEL3B_DEBYE_TAU2 / (1.0 + (w * LEVEL3B_DEBYE_TAU2) ** 2))
    return eps_im_h / unit


def level3b_eps(f):
    """Level 3b の複素比誘電率 (eps', eps'')。2 極 Debye により分散する。"""
    f_arr = np.asarray(f, dtype=float)
    eps_s, _ = level3b_carrier()
    s = level3b_debye_scale()
    w = 2.0 * np.pi * f_arr
    x1, x2 = w * LEVEL3B_DEBYE_TAU1, w * LEVEL3B_DEBYE_TAU2
    drop = (LEVEL3B_DEBYE_DE1 * s * x1 ** 2 / (1.0 + x1 ** 2)
            + LEVEL3B_DEBYE_DE2 * s * x2 ** 2 / (1.0 + x2 ** 2))
    imag = (LEVEL3B_DEBYE_DE1 * s * x1 / (1.0 + x1 ** 2)
            + LEVEL3B_DEBYE_DE2 * s * x2 / (1.0 + x2 ** 2))
    return eps_s - drop, imag


def level3b_tandelta(f):
    """Level 3b の tan_delta(f)。f とともに増加する。"""
    er, ei = level3b_eps(f)
    return ei / er


def level3b_alpha(f):
    """Level 3b の減衰係数 alpha(f) [Np/m]（厳密式）。alpha ∝ f^1.38 程度。"""
    f_arr = np.asarray(f, dtype=float)
    er, ei = level3b_eps(f_arr)
    td = ei / er
    w_over_c = 2.0 * np.pi * f_arr / (C * 1e9)
    with np.errstate(invalid='ignore'):
        alpha = w_over_c * np.sqrt(er / 2.0) * np.sqrt(np.sqrt(1.0 + td ** 2) - 1.0)
    return np.nan_to_num(alpha, nan=0.0)


def level3b_group_index(f):
    """Level 3b の群屈折率 n_g = n + f dn/df。包絡ピークの到達時刻を決める。

    分散性媒質では位相速度と群速度が分かれるため、走時位相に使う n(f) と
    包絡ピークが伝わる速さを決める n_g(f) は別物になる。
    """
    f_arr = np.atleast_1d(np.asarray(f, dtype=float))
    fs = np.linspace(0.5e9, 2.0e9, 601)
    er, _ = level3b_eps(fs)
    n_fs = np.sqrt(er)
    ng_fs = n_fs + fs * np.gradient(n_fs, fs)
    return np.interp(f_arr, fs, ng_fs)


def describe_level3b_medium():
    """Level 3b の媒質設定を人が読める形で返す。"""
    eps_s, _ = level3b_carrier()
    s = level3b_debye_scale()
    de1, de2 = s * LEVEL3B_DEBYE_DE1, s * LEVEL3B_DEBYE_DE2
    a_lo = float(level3b_alpha(np.array([0.5e9]))[0])
    a_hi = float(level3b_alpha(np.array([2.0e9]))[0])
    return ('2-pole Debye (high-Ti basalt case), FeO+TiO2 = {:.1f} wt%, '
            'rho = {:.6f}, eps_s = {:.6f}, eps_inf = {:.6f}, De1 = {:.6f}, De2 = {:.6f}  '
            '(alpha = {:.4f} -> {:.4f} Np/m over 0.5-2.0 GHz, ratio {:.3f})'
            .format(LEVEL3B_FEOTIO2, LEVEL3B_RHO, eps_s, eps_s - de1 - de2, de1, de2,
                    a_lo, a_hi, a_hi / a_lo))


def transfer_absorb_debye(f, d, n=None):
    """Level 3b の吸収項 exp(-alpha(f) * d)（片道透過）。"""
    return np.exp(-level3b_alpha(f) * d)


def transfer_density_profile(f, d, params):
    raise NotImplementedError('Level_5 (density_profile) は未実装です。')


def build_transfer(f, d, level, n=None):
    """LEVEL_EFFECTS に従って伝達関数 H_level(f,d) を効果の積で構成する。

    レベル依存はこの関数と LEVEL_EFFECTS 辞書にのみ現れる。

    n は refractive_index(f, level) から取得する。Level 1・2 では定数配列、
    Level 3 では n(f) = sqrt(eps'(f)) となり、幾何項・透過係数・走時位相の
    すべてが周波数依存になる。
    """
    effects = LEVEL_EFFECTS[level]
    n = refractive_index(f, level)
    # 'ice_layer' があるレベル（Level 4 以降）は経路が層構造になるので、
    # 幾何項・吸収項・走時位相を層ごとに積む版に差し替える。
    layered = 'ice_layer' in effects
    H = np.ones_like(f, dtype=complex)
    for effect in effects:
        if effect == 'geom':
            H = H * (transfer_geom_layered(f, d) if layered
                     else transfer_geom(d, n))
        elif effect == 'surface_T':
            H = H * transfer_surface_T(n)
        elif effect == 'absorb_const':
            H = H * transfer_absorb(f, d, n)
        elif effect == 'absorb_tandelta':
            H = H * (transfer_absorb_layered(f, d, n) if layered
                     else transfer_absorb_tandelta(f, d, n))
        elif effect == 'absorb_debye':
            H = H * transfer_absorb_debye(f, d, n)
        elif effect == 'ice_layer':
            H = H * transfer_ice_T(f, d, n)
        elif effect == 'density_profile':
            H = H * transfer_density_profile(f, d, None)
        else:
            raise CmdInputError('Unknown effect: {}'.format(effect))

    if layered:
        phase, t_arr_f = transfer_phase_layered(f, d)
    else:
        phase, t_arr_f = transfer_phase(f, d, n)
    H = H * phase

    # 探索窓の中心と CSV に載せる代表値としてスカラーの走時を作る。
    # Level 3 では包絡ピークが群速度で決まるため、群屈折率から算出する
    # （位相走時ではズレる）。Level 1・2 では両者は一致する。
    # Level 3b は分散性なので、包絡ピークの位置は群速度で決まる。
    # Level 1/2 は eps' が厳密に定数なので位相速度と群速度が一致する。
    # Level 3 も現状は定数扱い（A-3 で eps'(f) を入れると 0.2% ずれる）。
    # eps' が周波数依存のときは位相走時を代表値に使えない（f=0 の値になる）。
    # 包絡ピークの位置を決めるのは群速度なので、群屈折率から作る。
    if 'absorb_debye' in effects:
        ng = float(level3b_group_index(BAND_CENTRE_HZ)[0])
        t_arr = TX_HEIGHT / C + ng * d / C
    elif layered:
        t_arr = level4_group_arrival(d)
    elif 'absorb_tandelta' in effects:
        ng = float(level3_group_index(BAND_CENTRE_HZ)[0])
        t_arr = TX_HEIGHT / C + ng * d / C
    else:
        t_arr = float(np.atleast_1d(t_arr_f)[0]) if np.ndim(t_arr_f) else float(t_arr_f)
    return H, t_arr


def synth_theory_spectrum(E_ref_f, freq, d, level, n):
    """理論スペクトル E_theory(f,d) = E_ref(f)・H_level(f,d) を周波数領域で直接求める。

    A-scan ツールと異なり IFFT には戻さない（スペクトル解析では時間領域が不要なため）。
    """
    H, t_arr = build_transfer(freq, d, level, n)
    return E_ref_f * H, t_arr


# =============================================================================
# マスク
# =============================================================================
def noise_floor(freq_hz, E, band_ghz=NOISE_BAND_GHZ):
    """帯域外 (NOISE_BAND_GHZ) の |E| の RMS を数値ノイズフロアとする（design_ascan_spectrum.md §6.7）。"""
    freq_ghz = freq_hz * 1e-9
    mask = (freq_ghz >= band_ghz[0]) & (freq_ghz <= band_ghz[1])
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.mean(np.abs(E[mask]) ** 2)))


def valid_mask(freq_hz, E, E_ref, band_ghz=BAND_GHZ):
    """2 段マスク（design_ascan_spectrum.md §3.2）。LSR / α(f) / tanδ(f) / 群遅延に適用する。"""
    freq_ghz = freq_hz * 1e-9
    band_mask = (freq_ghz >= band_ghz[0]) & (freq_ghz <= band_ghz[1])

    ref_abs = np.abs(E_ref)
    ref_max_in_band = np.max(ref_abs[band_mask]) if np.any(band_mask) else np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        ref_db = 20.0 * np.log10(ref_abs / ref_max_in_band)
    mask1 = ref_db >= MASK_REF_FLOOR_DB

    nf = noise_floor(freq_hz, E)
    with np.errstate(divide='ignore', invalid='ignore'):
        snr_db = 20.0 * np.log10(np.abs(E) / nf)
    mask2 = snr_db >= MASK_SNR_MIN_DB

    return band_mask & mask1 & mask2


# =============================================================================
# 解析
# =============================================================================
def log_spectral_ratio(E_num, E_denom):
    """ln|E_num / E_denom|。絶対LSR・相対LSR共通（design_ascan_spectrum.md §6.1, §6.2）。"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(np.abs(E_num) / np.abs(E_denom))


def moments(freq_hz, E, band_ghz=BAND_GHZ):
    """f_c, sigma_f, skew（帯域 0.5-2.0 GHz、マスクは適用しない）（§6.3）。"""
    freq_ghz = freq_hz * 1e-9
    band_mask = (freq_ghz >= band_ghz[0]) & (freq_ghz <= band_ghz[1])
    f = freq_hz[band_mask]
    P = np.abs(E[band_mask]) ** 2
    P_sum = np.sum(P)
    f_c = np.sum(f * P) / P_sum
    sigma_f = np.sqrt(np.sum((f - f_c) ** 2 * P) / P_sum)
    skew = np.sum((f - f_c) ** 3 * P) / P_sum / sigma_f ** 3
    return {'f_c': f_c, 'sigma_f': sigma_f, 'skew': skew}


def _crossings(f, dB, threshold_db):
    """dB が threshold_db を横切る周波数を線形内挿ですべて求める。"""
    crossings = []
    diff = dB - threshold_db
    for i in range(len(f) - 1):
        if diff[i] == 0:
            crossings.append(f[i])
        elif diff[i] * diff[i + 1] < 0:
            frac = diff[i] / (diff[i] - diff[i + 1])
            crossings.append(f[i] + frac * (f[i + 1] - f[i]))
    return crossings


def lo_hi_freq(freq_hz, E, band_ghz=BAND_GHZ, thresholds_db=FLOHI_THRESHOLDS_DB):
    """帯域内最大値に対する -3/-10/-20 dB しきい値を横切る周波数（§6.4、マスクは適用しない）。

    複数回横切る場合は最も外側（f_lo は最小、f_hi は最大）を採用する。
    しきい値を一度も下回らない場合は帯域端をそのまま採用する。
    """
    freq_ghz = freq_hz * 1e-9
    band_mask = (freq_ghz >= band_ghz[0]) & (freq_ghz <= band_ghz[1])
    f = freq_hz[band_mask]
    mag = np.abs(E[band_mask])
    mag_max = np.max(mag)
    with np.errstate(divide='ignore', invalid='ignore'):
        dB = 20.0 * np.log10(mag / mag_max)

    result = {}
    for th in thresholds_db:
        crossings = _crossings(f, dB, th)
        if crossings:
            result[th] = {'f_lo': min(crossings), 'f_hi': max(crossings)}
        else:
            result[th] = {'f_lo': f[0], 'f_hi': f[-1]}
    return result


def alpha_from_abs_lsr(L_abs, d, n):
    """絶対LSRから α(f) を求める（design_ascan_spectrum.md §6.5）。"""
    log_geom_T = np.log(transfer_surface_T(n) * np.sqrt(n / (n * TX_HEIGHT + d)))
    return -(L_abs - log_geom_T) / d


def alpha_from_rel_lsr(L_rel, d, d0, n):
    """相対LSRから α(f) を求める（実機で計測可能な版）。"""
    r_eff = n * TX_HEIGHT + d
    r_eff0 = n * TX_HEIGHT + d0
    return -(L_rel - 0.5 * np.log(r_eff0 / r_eff)) / (d - d0)


def alpha_to_tandelta(alpha, freq_hz, n):
    """α(f) から tanδ(f) を求める。f は GHz に変換して使う（c が m/ns 単位のため）。"""
    freq_ghz = freq_hz * 1e-9
    with np.errstate(divide='ignore', invalid='ignore'):
        return alpha * C / (np.pi * freq_ghz * n)


def group_delay(freq_hz, E, E_ref):
    """参照に対する位相から群遅延[ns]を求める（design_ascan_spectrum.md §6.6）。"""
    with np.errstate(divide='ignore', invalid='ignore'):
        phase = np.unwrap(np.angle(E / E_ref))
    tau_g_s = -np.gradient(phase, freq_hz) / (2.0 * np.pi)
    return tau_g_s * 1e9 + R_REF / C


def _th_suffix(threshold_db):
    return 'm{}'.format(int(abs(threshold_db)))


# =============================================================================
# メイン処理：全 rx に対して実測・理論の比較を行う
# =============================================================================
def analyze_level(rx_paths, reference, level):
    e_ref, dt_ref = load_trace(reference['far_1m'])
    n = N_REGOLITH
    if 'absorb_const' in LEVEL_EFFECTS[level]:
        print('{} の媒質: {}'.format(level, describe_level2_medium(n)))
    if 'absorb_tandelta' in LEVEL_EFFECTS[level]:
        print('{} の媒質: {}'.format(level, describe_level3_medium()))
    if 'absorb_debye' in LEVEL_EFFECTS[level]:
        print('{} の媒質: {}'.format(level, describe_level3b_medium()))
    if 'ice_layer' in LEVEL_EFFECTS[level]:
        print('{} の氷層: {}'.format(level, describe_level4_medium()))

    ref_peak = measure_peak(e_ref, dt_ref)
    t_src_delay = ref_peak['t_peak'] - R_REF / C
    print('波源の内部遅延（参照波形から実測）: {:.3f} ns'.format(t_src_delay))

    e_ref_gated = apply_gate(e_ref, dt_ref, ref_peak['t_peak'])
    freq_ref, E_ref_raw = spectrum(e_ref_gated, dt_ref)

    depth_items = {k: v for k, v in rx_paths.items() if k != 'at_tx'}
    if 'at_tx' in rx_paths:
        print('Notice: at_tx はこのツールでは解析対象外です'
              '（design_ascan_spectrum.md §5 の理論モデルは透過経路のみを定義するため）。')
    if not depth_items:
        raise CmdInputError('解析可能な深さ依存 rx がありません')

    # 最初の rx で共通の周波数グリッド（dt・サンプル数）を確定し、以降すべて一致するか確認する
    freq_hz = None
    n_samples_common = None
    dt_common = None
    traces = {}
    for key, path in depth_items.items():
        trace, dt = load_trace(path)
        # 参照との dt 一致は不要（下で参照スペクトルを共通グリッドへ内挿するため）。
        # dx を変えると dt も変わるので、ここで弾いてはいけない。
        if n_samples_common is None:
            n_samples_common = len(trace)
            dt_common = dt
            freq_hz = np.fft.rfftfreq(n_samples_common, d=dt)
            if not _same_dt(dt, dt_ref, rtol=1e-6):
                print('')
                print('*** 警告: 参照計算と dt が異なります '
                      '(dt {:.4f} ps vs 参照 {:.4f} ps) ***'.format(dt * 1e12, dt_ref * 1e12))
                print('    dt の違いは dx の違いを意味します。周波数グリッドは内挿で揃えられますが、')
                print('    gprMax の #hertzian_dipole は電流密度 J = I*dl/(dx*dy*dz) と')
                print('    セル寸法で正規化されるため、2D では波源の絶対振幅が dx に依存します。')
                print('    このまま進めると絶対LSR が定数倍ずれます（dx 2 倍で約 6 dB）。')
                print('    → 解析対象と同じ dx で計算した参照計算を JSON に指定してください。')
                print('    （相対LSR・alpha(f)・群遅延には影響しません）')
                print('')
        elif len(trace) != n_samples_common or not _same_dt(dt, dt_common, rtol=1e-6):
            raise CmdInputError(
                'rx={} のサンプル数/dt が他の rx と一致しません'
                '（スペクトル比較には共通の周波数グリッドが必要です）'.format(key))
        traces[key] = (trace, dt)

    # 参照スペクトルは別ドメインの計算のため周波数グリッドが異なりうる → 共通グリッドへ内挿
    E_ref = _interp_complex_to_grid(freq_ref, E_ref_raw, freq_hz)

    entries = {}
    for key, (trace, dt) in traces.items():
        d = rx_depth(key)
        E_theory, t_arr = synth_theory_spectrum(E_ref, freq_hz, d, level, n)

        trace_gated = apply_gate(trace, dt, t_arr + t_src_delay)
        _, E_meas = spectrum(trace_gated, dt)

        nf = noise_floor(freq_hz, E_meas)
        mask = valid_mask(freq_hz, E_meas, E_ref)
        if not np.any(mask):
            print('Warning: rx={} で有効なマスク周波数がありません（マスク基準を確認してください）。'.format(key))

        entries[key] = {
            'depth_m': d, 'E_meas': E_meas, 'E_theory': E_theory,
            't_arr': t_arr, 'noise_floor': nf, 'mask': mask,
        }

    d0_key = min(entries, key=lambda k: abs(entries[k]['depth_m'] - LSR_REF_DEPTH_M))
    d0 = entries[d0_key]['depth_m']
    if abs(d0 - LSR_REF_DEPTH_M) > 1e-6:
        print('Notice: LSR_REF_DEPTH_M={} m に一致する rx がないため、'
              '最も近い depth={} m ({}) を相対LSRの基準にします。'.format(LSR_REF_DEPTH_M, d0, d0_key))
    E_meas_d0 = entries[d0_key]['E_meas']
    E_theory_d0 = entries[d0_key]['E_theory']

    results = []
    for key in sorted(entries, key=lambda k: entries[k]['depth_m']):
        e = entries[key]
        d = e['depth_m']
        E_meas = e['E_meas']
        E_theory = e['E_theory']
        mask = e['mask']
        t_arr = e['t_arr']

        # 幾何項を引くときの屈折率。Level 3 では n(f) = sqrt(eps'(f)) になるため、
        # 定数 n を使うと幾何項の見積もりがずれ、alpha に偽の周波数依存が乗る。
        n_f = refractive_index(freq_hz, level)

        L_abs_meas = log_spectral_ratio(E_meas, E_ref)
        L_abs_theory = log_spectral_ratio(E_theory, E_ref)
        L_rel_meas = log_spectral_ratio(E_meas, E_meas_d0)
        L_rel_theory = log_spectral_ratio(E_theory, E_theory_d0)

        mom_meas = moments(freq_hz, E_meas)
        mom_theory = moments(freq_hz, E_theory)
        flohi_meas = lo_hi_freq(freq_hz, E_meas)
        flohi_theory = lo_hi_freq(freq_hz, E_theory)

        if np.isclose(d, 0.0):
            # d=0 (at_surface) では伝搬距離がゼロのため alpha(f)=(残差)/d が未定義になる。
            # 物理的に意味のある量ではないので NaN として除外する。
            alpha_abs = np.full_like(freq_hz, np.nan)
            tandelta_abs = np.full_like(freq_hz, np.nan)
        else:
            alpha_abs = alpha_from_abs_lsr(L_abs_meas, d, n_f)
            tandelta_abs = alpha_to_tandelta(alpha_abs, freq_hz, n_f)
        if key == d0_key:
            alpha_rel = np.full_like(freq_hz, np.nan)
        else:
            alpha_rel = alpha_from_rel_lsr(L_rel_meas, d, d0, n_f)
        tandelta_rel = alpha_to_tandelta(alpha_rel, freq_hz, n_f)

        tau_g = group_delay(freq_hz, E_meas, E_ref)

        result = {
            'key': key, 'depth_m': d, 't_arr_ns': t_arr,
            'freq_hz': freq_hz, 'E_meas': E_meas, 'E_theory': E_theory,
            'L_abs_meas': L_abs_meas, 'L_abs_theory': L_abs_theory,
            'L_rel_meas': L_rel_meas, 'L_rel_theory': L_rel_theory,
            'alpha_abs': alpha_abs, 'alpha_rel': alpha_rel,
            'tandelta_abs': tandelta_abs, 'tandelta_rel': tandelta_rel,
            'tau_g': tau_g, 'mask': mask,
        }

        if np.any(mask):
            resid_db = (L_abs_meas - L_abs_theory) * LN_TO_DB20
            result['lsr_level_db'] = float(np.mean(L_abs_meas[mask]) * LN_TO_DB20)
            result['lsr_theory_db'] = float(np.mean(L_abs_theory[mask]) * LN_TO_DB20)
            result['lsr_resid_mean_db'] = float(np.mean(resid_db[mask]))
            detrended = L_abs_meas[mask] - np.mean(L_abs_meas[mask])
            result['lsr_flatness_db'] = float((np.max(detrended) - np.min(detrended)) * LN_TO_DB20)
            result['valid_band_lo_ghz'] = float(freq_hz[mask].min() * 1e-9)
            result['valid_band_hi_ghz'] = float(freq_hz[mask].max() * 1e-9)
            taug_resid_mean_ns = float(np.mean(tau_g[mask] - t_arr))
            result['taug_resid_mean_ns'] = taug_resid_mean_ns
            result['taug_resid_frac'] = abs(taug_resid_mean_ns) / t_arr
            result['noise_db'] = float(20.0 * np.log10(e['noise_floor'] / np.max(np.abs(E_meas)[mask]) + 1e-30))
        else:
            for k in ('lsr_level_db', 'lsr_theory_db', 'lsr_resid_mean_db', 'lsr_flatness_db',
                      'valid_band_lo_ghz', 'valid_band_hi_ghz', 'taug_resid_mean_ns',
                      'taug_resid_frac', 'noise_db'):
                result[k] = np.nan

        result['pass_lsr_level'] = (not np.isnan(result['lsr_resid_mean_db'])) and \
            abs(result['lsr_resid_mean_db']) < LSR_TOL_DB
        result['pass_lsr_flatness'] = (not np.isnan(result['lsr_flatness_db'])) and \
            result['lsr_flatness_db'] < LSR_FLATNESS_TOL_DB
        result['pass_taug'] = (not np.isnan(result['taug_resid_frac'])) and \
            result['taug_resid_frac'] < TAUG_TOL_FRAC
        result['pass_noise'] = (not np.isnan(result['noise_db'])) and result['noise_db'] < NOISE_FLOOR_DB

        result['fc_meas_ghz'] = mom_meas['f_c'] * 1e-9
        result['fc_theory_ghz'] = mom_theory['f_c'] * 1e-9
        result['fc_resid_mhz'] = (mom_meas['f_c'] - mom_theory['f_c']) * 1e-6
        result['pass_fc'] = abs(result['fc_resid_mhz']) < FC_TOL_MHZ
        result['sigma_f_meas_ghz'] = mom_meas['sigma_f'] * 1e-9
        result['sigma_f_theory_ghz'] = mom_theory['sigma_f'] * 1e-9
        result['skew_meas'] = mom_meas['skew']
        result['skew_theory'] = mom_theory['skew']

        for th in FLOHI_THRESHOLDS_DB:
            suffix = _th_suffix(th)
            result['flo_{}_meas_ghz'.format(suffix)] = flohi_meas[th]['f_lo'] * 1e-9
            result['fhi_{}_meas_ghz'.format(suffix)] = flohi_meas[th]['f_hi'] * 1e-9
            result['flo_{}_theory_ghz'.format(suffix)] = flohi_theory[th]['f_lo'] * 1e-9
            result['fhi_{}_theory_ghz'.format(suffix)] = flohi_theory[th]['f_hi'] * 1e-9

        result['bandwidth_m10_ghz'] = result['fhi_m10_meas_ghz'] - result['flo_m10_meas_ghz']
        result['ratio_m10'] = result['fhi_m10_meas_ghz'] / result['flo_m10_meas_ghz']

        results.append(result)

    return results, freq_hz, E_ref, d0_key, d0


# =============================================================================
# 作図
# =============================================================================
def _depth_norm_cmap(results):
    depths = [r['depth_m'] for r in results]
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=min(depths), vmax=max(depths))
    return cmap, norm


def save_figure(fig, output_dir, stem):
    """PNG と PDF の両方で保存する。"""
    for ext in FIGURE_FORMATS:
        path = os.path.join(output_dir, '{}.{}'.format(stem, ext))
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
        print('Saved:', path)
    plt.close(fig)


# -----------------------------------------------------------------------------
# fig1: スペクトルに関する集約
# -----------------------------------------------------------------------------
def plot_spectra(results, freq_hz, E_ref, output_dir):
    """(a) 生スペクトル比較、(b) 帯域端 f_lo/f_c/f_hi、(c) 重心と幅 f_c±sigma_f。

    3 段構成（nrows=3）。(b) と (c) を分けているのは、両者が性格の異なる量で
    あり同じ軸に重ねると読み違えるため。

    (b)(c) のエラーバーの意味:
      * 外側の細いキャップ  : f_lo - f_hi（しきい値を横切る周波数、既定 -10 dB）
      * 内側の太いバー      : f_c ± sigma_f（パワースペクトルの重心と標準偏差）
      * マーカー            : f_c

    f_lo/f_hi はしきい値を横切る「1 点」で決まる局所量、sigma_f はスペクトル
    全体をパワー重みで積分した大域量であり、性格が異なる。感度の恒等式
    df_c/dt = -2*pi*tan_delta*sigma_f^2 に現れるのは sigma_f のほう。
    """
    depth_results = sorted(results, key=lambda r: r['depth_m'])
    if not depth_results:
        return
    cmap, norm = _depth_norm_cmap(depth_results)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)

    freq_ghz = freq_hz * 1e-9
    band_lo, band_hi = BAND_GHZ
    band_mask = (freq_ghz >= band_lo) & (freq_ghz <= band_hi)
    ref_max = np.max(np.abs(E_ref)[band_mask])

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9, 15))

    # --- (a) 生スペクトル ---
    ax = axes[0]
    for r in depth_results:
        color = cmap(norm(r['depth_m']))
        with np.errstate(divide='ignore', invalid='ignore'):
            db = 20.0 * np.log10(np.abs(r['E_meas']) / ref_max)
        ax.plot(freq_ghz, db, color=color, lw=1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ref_db = 20.0 * np.log10(np.abs(E_ref) / ref_max)
    ax.plot(freq_ghz, ref_db, color='gray', lw=2.5, label='E_ref (far_1m)')
    ax.axvline(band_lo, color='k', ls='--', lw=0.8)
    ax.axvline(band_hi, color='k', ls='--', lw=0.8)
    ax.set_xlim(0, band_hi * 1.5)
    ax.set_ylim(-100, 5)
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel('|E(f)| [dB re. max(|E_ref|) in band]')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title('(a) Raw spectra')
    fig.colorbar(sm, ax=ax, label='rx depth [m]')

    # --- (b) 帯域端 f_lo / f_c / f_hi ---
    suffix = _th_suffix(FLOHI_PRIMARY_DB)
    depths = np.array([r['depth_m'] for r in depth_results])
    fc = np.array([r['fc_meas_ghz'] for r in depth_results])
    sd = np.array([r['sigma_f_meas_ghz'] for r in depth_results])
    flo = np.array([r['flo_{}_meas_ghz'.format(suffix)] for r in depth_results])
    fhi = np.array([r['fhi_{}_meas_ghz'.format(suffix)] for r in depth_results])
    fc_t = np.array([r['fc_theory_ghz'] for r in depth_results])
    sd_t = np.array([r['sigma_f_theory_ghz'] for r in depth_results])
    flo_t = np.array([r['flo_{}_theory_ghz'.format(suffix)] for r in depth_results])
    fhi_t = np.array([r['fhi_{}_theory_ghz'.format(suffix)] for r in depth_results])

    ax = axes[1]
    # 理論は 3 本の赤点線。凡例は 1 つにまとめる。
    # シミュレーション結果のプロット線を上にしたいので、先に理論曲線をプロットする
    ax.plot(flo_t, depths, 'r--', lw=1.2, label='theory')
    ax.plot(fc_t, depths, 'r--', lw=1.2)
    ax.plot(fhi_t, depths, 'r--', lw=1.2)
    ax.errorbar(fc, depths, xerr=[fc - flo, fhi - fc], fmt='o', color='k',
                    markersize=4, elinewidth=1.0, capsize=4,
                    label='measured: f_c with f_lo - f_hi ({:.0f} dB)'.format(FLOHI_PRIMARY_DB))
    ax.invert_yaxis()
    ax.set_xlim(flo.min() -  flo.min()*0.1, fhi.max() + flo.min()*0.1)
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel('rx depth [m]')
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)
    ax.set_title('(b) Band edges: f_lo / f_c / f_hi ({:.0f} dB)'.format(FLOHI_PRIMARY_DB))

    # --- (c) 重心とスペクトル幅 f_c, sigma_f ---
    ax = axes[2]
    # シミュレーション結果のプロット線を上にしたいので、先に理論曲線をプロットする
    ax.plot(fc_t, depths, 'r--', lw=1.2, label='theory')
    ax.plot(fc_t - sd_t, depths, 'r--', lw=1.2)
    ax.plot(fc_t + sd_t, depths, 'r--', lw=1.2)
    ax.errorbar(fc, depths, xerr=sd, fmt='o', color='k', markersize=4,
                    elinewidth=1.0, capsize=4,
                    label='measured: $f_c \\pm\\ \\sigma_f$')
    ax.set_xlim(flo.min() -  flo.min()*0.1, fhi.max() + flo.min()*0.1)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel('rx depth [m]')
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)
    ax.set_title('(c) Centroid and spectral width: $f_c$, $\\sigma_f$')

    fig.tight_layout()
    save_figure(fig, output_dir, 'fig1_spectra')


# -----------------------------------------------------------------------------
# fig2: LSR に関する集約
# -----------------------------------------------------------------------------
def _shared_ylim(data_arrays, pad_frac=0.08, pct=(1.0, 99.0)):
    """複数パネルで共有する縦軸範囲を、実データから頑健に決める。

    浅い rx では alpha = -(LSR残差)/d の 1/d 増幅で外れ値が出るため、
    最小最大をそのまま使うと範囲が広がりすぎて傾向が読めなくなる。
    そこでパーセンタイル（既定 1-99%）で外れ値を落としてから余白を付ける。

    data_arrays: 1 次元配列のリスト（NaN を含んでよい）
    戻り値: (lo, hi)。有効なデータがなければ None。
    """
    pooled = []
    for a in data_arrays:
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            pooled.append(a)
    if not pooled:
        return None
    pooled = np.concatenate(pooled)
    lo, hi = np.percentile(pooled, pct[0]), np.percentile(pooled, pct[1])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if np.isclose(lo, hi):
        span = max(abs(lo), 1e-12) * 0.1
        return lo - span, hi + span
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def plot_lsr(results, freq_hz, d0, output_dir):
    """絶対LSR・相対LSR とそれぞれの残差（2x2）。"""
    depth_results = sorted(results, key=lambda r: r['depth_m'])
    if not depth_results:
        return
    cmap, norm = _depth_norm_cmap(depth_results)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    freq_ghz = freq_hz * 1e-9
    band_lo, band_hi = BAND_GHZ

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    panels = [
        (axes[0, 0], 'L_abs_meas', 'L_abs_theory',
         '(a) Absolute LSR: measured (solid) vs theory (dashed)', 'Absolute LSR [dB]'),
        (axes[1, 0], 'L_rel_meas', 'L_rel_theory',
         '(c) Relative LSR (ref depth = {:.2f} m)'.format(d0),
         'Relative LSR [dB]'),
    ]
    # (a) 絶対LSR と (c) 相対LSR は同じ「LSR [dB]」なので縦軸を揃える。
    value_pool = []
    for _, key_m, key_t, _, _ in panels:
        for r in depth_results:
            mask = r['mask']
            value_pool.append(r[key_m][mask] * LN_TO_DB20)
            value_pool.append(r[key_t][mask] * LN_TO_DB20)
    value_ylim = _shared_ylim(value_pool, pct=(0.0, 100.0))

    for ax, key_m, key_t, title, ylabel in panels:
        for r in depth_results:
            color = cmap(norm(r['depth_m']))
            mask = r['mask']
            ax.plot(freq_ghz[mask], r[key_m][mask] * LN_TO_DB20, color=color, lw=1.2)
            ax.plot(freq_ghz[mask], r[key_t][mask] * LN_TO_DB20, color=color, lw=1.0, ls='--')
        if value_ylim:
            ax.set_ylim(*value_ylim)
        ax.set_xlim(band_lo, band_hi)
        ax.set_xlabel('Frequency [GHz]')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.set_title(title)
        fig.colorbar(sm, ax=ax, label='rx depth [m]')

    resid_panels = [
        (axes[0, 1], 'L_abs_meas', 'L_abs_theory', True,
         '(b) Absolute LSR residual (pass/fail)'),
        (axes[1, 1], 'L_rel_meas', 'L_rel_theory', False,
         '(d) Relative LSR residual'),
    ]
    # (b) 絶対LSR残差 と (d) 相対LSR残差 も縦軸を揃える。
    # 合否帯（±LSR_TOL_DB）が入る側に合わせると残差そのものが潰れるため、
    # 残差の実データ範囲で決めたうえで、合否帯が入りきる場合だけ広げる。
    resid_pool = []
    for _, key_m, key_t, _, _ in resid_panels:
        for r in depth_results:
            mask = r['mask']
            resid_pool.append((r[key_m] - r[key_t])[mask] * LN_TO_DB20)
    resid_ylim = _shared_ylim(resid_pool, pct=(0.0, 100.0))

    for ax, key_m, key_t, show_tol, title in resid_panels:
        for r in depth_results:
            color = cmap(norm(r['depth_m']))
            mask = r['mask']
            resid = (r[key_m] - r[key_t]) * LN_TO_DB20
            ax.plot(freq_ghz[mask], resid[mask], color=color, lw=1.2)
        if resid_ylim:
            ax.set_ylim(*resid_ylim)
        if show_tol:
            ax.axhspan(-LSR_TOL_DB, LSR_TOL_DB, color='green', alpha=0.15,
                       label='$\\pm${} dB'.format(LSR_TOL_DB))
            ax.legend()
        ax.axhline(0, color='gray', lw=1)
        ax.set_xlim(band_lo, band_hi)
        ax.set_xlabel('Frequency [GHz]')
        ax.set_ylabel('Residual [dB] (meas - theory)')
        ax.grid(alpha=0.3)
        ax.set_title(title)
        fig.colorbar(sm, ax=ax, label='rx depth [m]')

    fig.tight_layout()
    save_figure(fig, output_dir, 'fig2_lsr')


# -----------------------------------------------------------------------------
# fig3: 減衰率に関する集約
# -----------------------------------------------------------------------------
def plot_attenuation(results, freq_hz, level, n, output_dir):
    """alpha(f) と tan_delta(f) を、絶対LSR版・相対LSR版の両方で（2x2）。

    浅い rx ほど乖離が大きく見えるのは、alpha = -(LSR残差)/d という定義により
    同じ大きさの残差が 1/d で増幅されるため（README §2.4 参照）。
    """
    depth_results = sorted(results, key=lambda r: r['depth_m'])
    if not depth_results:
        return
    cmap, norm = _depth_norm_cmap(depth_results)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    freq_ghz = freq_hz * 1e-9
    band_lo, band_hi = BAND_GHZ
    band_mask = (freq_ghz >= band_lo) & (freq_ghz <= band_hi)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    # fig2 と同じ並びにする: 上段 (a)(b) が絶対LSR 由来、下段 (c)(d) が相対LSR 由来。
    # 左列 (a)(c) が alpha、右列 (b)(d) が tan_delta。
    # 同じ量どうし（左列・右列）で縦軸を揃える。
    specs = [
        (axes[0, 0], 'alpha_abs', r'$\alpha(f)$ [1/m]',
         '(a) Attenuation from absolute LSR (collapse check)'),
        (axes[0, 1], 'tandelta_abs', r'$\tan\delta(f)$',
         '(b) Loss tangent from absolute LSR'),
        (axes[1, 0], 'alpha_rel', r'$\alpha(f)$ [1/m]',
         '(c) Attenuation from relative LSR (field-measurable)'),
        (axes[1, 1], 'tandelta_rel', r'$\tan\delta(f)$',
         '(d) Loss tangent from relative LSR'),
    ]

    # 縦軸範囲は実データのパーセンタイルから決める。浅い rx は 1/d 増幅で
    # 外れ値が出るため、最小最大では範囲が広がりすぎて傾向が読めない。
    def theory_curve(key):
        if 'absorb_tandelta' in LEVEL_EFFECTS[level]:
            return level3_alpha(freq_hz) if key.startswith('alpha') \
                else level3_tandelta(freq_hz)
        # 注: Level 4 も 'absorb_tandelta' を持つので上の分岐に入り、
        #     背景レゴリスの理論値が返る。氷層側は別途重ね描きする。
        if 'absorb_debye' in LEVEL_EFFECTS[level]:
            return level3b_alpha(freq_hz) if key.startswith('alpha') \
                else level3b_tandelta(freq_hz)
        if 'absorb_const' in LEVEL_EFFECTS[level]:
            return level2_alpha(freq_hz, n) if key.startswith('alpha') \
                else level2_tandelta(freq_hz, n)
        return np.zeros_like(freq_hz)

    ylims = {}
    for group_keys in (('alpha_abs', 'alpha_rel'), ('tandelta_abs', 'tandelta_rel')):
        pool = []
        for key in group_keys:
            for r in depth_results:
                arr = r[key]
                if np.all(np.isnan(arr)):
                    continue
                # 必ず有効マスクを掛けてから集める。帯域端のマスク外は
                # LSR の分母が小さく alpha が発散するため、含めると範囲が
                # 数桁広がって傾向が読めなくなる（実際にそうなった）。
                pool.append(arr[r['mask']])
            pool.append(theory_curve(key)[band_mask])   # 理論曲線も枠内に入れる
        lim = _shared_ylim(pool, pct=ATTEN_YLIM_PCT)
        for key in group_keys:
            ylims[key] = lim

    for ax, key, ylabel, title in specs:
        plotted = False
        for r in depth_results:
            arr = r[key]
            if np.all(np.isnan(arr)):
                continue
            color = cmap(norm(r['depth_m']))
            mask = r['mask']
            ax.plot(freq_ghz[mask], arr[mask], color=color, lw=1.2)
            plotted = True
        # 理論曲線はレベルで変わる。Level 1 は alpha = 0、
        # Level 2 以降は媒質の損失モデルから計算した alpha(f) / tan_delta(f)。
        if 'absorb_tandelta' in LEVEL_EFFECTS[level]:
            th = (level3_alpha(freq_hz) if key.startswith('alpha')
                  else level3_tandelta(freq_hz))
            bg_label = ('Theory (regolith)' if 'ice_layer' in LEVEL_EFFECTS[level]
                        else 'Theory ({})'.format(level))
            ax.plot(freq_ghz, th, color='r', ls='--', lw=1.5, label=bg_label)
            # Level 4 では LSR から逆算される alpha は経路平均になるため、
            # 背景レゴリスと氷層の 2 本の間に測定値が入るのが期待される姿。
            if 'ice_layer' in LEVEL_EFFECTS[level]:
                th_ice = (level4_alpha(freq_hz, True) if key.startswith('alpha')
                          else level4_tandelta(freq_hz, True))
                ax.plot(freq_ghz, th_ice, color='m', ls=':', lw=1.5,
                        label='Theory (ice layer)')
        elif 'absorb_debye' in LEVEL_EFFECTS[level]:
            th = (level3b_alpha(freq_hz) if key.startswith('alpha')
                  else level3b_tandelta(freq_hz))
            ax.plot(freq_ghz, th, color='r', ls='--', lw=1.5,
                    label='Theory ({})'.format(level))
        elif 'absorb_const' in LEVEL_EFFECTS[level]:
            th = (level2_alpha(freq_hz, n) if key.startswith('alpha')
                  else level2_tandelta(freq_hz, n))
            ax.plot(freq_ghz, th, color='r', ls='--', lw=1.5,
                    label='Theory ({})'.format(level))
        else:
            ax.axhline(0.0, color='r', ls='--', lw=1.5,
                       label='Theory ({}: alpha=0)'.format(level))
        ax.set_xlim(band_lo, band_hi)
        if ylims.get(key):
            ax.set_ylim(*ylims[key])
        ax.set_xlabel('Frequency [GHz]')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.set_title(title)
        if plotted:
            ax.legend(fontsize=8)
        fig.colorbar(sm, ax=ax, label='rx depth [m]')

    fig.tight_layout()
    save_figure(fig, output_dir, 'fig3_attenuation')


# -----------------------------------------------------------------------------
# fig4: 位相に関する集約
# -----------------------------------------------------------------------------
def plot_phase(results, freq_hz, output_dir):
    """群遅延とその残差。残差は数値分散を直接測っている量。

    深い rx ほど残差が大きくなるのは、数値分散が伝搬距離に比例して蓄積するため。
    （alpha が浅いほど悪化するのとは逆の依存性になる。README §8.2 参照）
    """
    depth_results = sorted(results, key=lambda r: r['depth_m'])
    if not depth_results:
        return
    cmap, norm = _depth_norm_cmap(depth_results)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    freq_ghz = freq_hz * 1e-9
    band_lo, band_hi = BAND_GHZ

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 11))

    ax = axes[0]
    for r in depth_results:
        color = cmap(norm(r['depth_m']))
        mask = r['mask']
        ax.plot(freq_ghz[mask], r['tau_g'][mask], color=color, lw=1.2)
        ax.axhline(r['t_arr_ns'], color=color, ls='--', lw=0.9)
    ax.set_xlim(band_lo, band_hi)
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel('Group delay [ns]')
    ax.grid(alpha=0.3)
    ax.set_title(r'(a) Group delay: measured (solid) vs theory $t_{arr}$ (dashed)')
    fig.colorbar(sm, ax=ax, label='rx depth [m]')

    ax = axes[1]
    for r in depth_results:
        color = cmap(norm(r['depth_m']))
        mask = r['mask']
        ax.plot(freq_ghz[mask], r['tau_g'][mask] - r['t_arr_ns'], color=color, lw=1.2)
    ax.axhline(0, color='gray', lw=1)
    ax.set_xlim(band_lo, band_hi)
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel(r'Residual [ns] ($\tau_g - t_{arr}$)')
    ax.grid(alpha=0.3)
    ax.set_title('(b) Group delay residual (numerical dispersion)')
    fig.colorbar(sm, ax=ax, label='rx depth [m]')

    fig.tight_layout()
    save_figure(fig, output_dir, 'fig4_phase')


CSV_FIELDNAMES = [
    'key', 'depth_m', 't_arr_ns',
    'lsr_level_db', 'lsr_theory_db', 'lsr_resid_mean_db', 'lsr_flatness_db',
    'pass_lsr_level', 'pass_lsr_flatness',
    'fc_meas_ghz', 'fc_theory_ghz', 'fc_resid_mhz', 'pass_fc',
    'sigma_f_meas_ghz', 'sigma_f_theory_ghz',
    'skew_meas', 'skew_theory',
    'flo_m3_meas_ghz', 'fhi_m3_meas_ghz', 'flo_m3_theory_ghz', 'fhi_m3_theory_ghz',
    'flo_m10_meas_ghz', 'fhi_m10_meas_ghz', 'flo_m10_theory_ghz', 'fhi_m10_theory_ghz',
    'flo_m20_meas_ghz', 'fhi_m20_meas_ghz', 'flo_m20_theory_ghz', 'fhi_m20_theory_ghz',
    'bandwidth_m10_ghz', 'ratio_m10',
    'taug_resid_mean_ns', 'taug_resid_frac', 'pass_taug',
    'valid_band_lo_ghz', 'valid_band_hi_ghz',
    'noise_db', 'pass_noise',
]


def write_csv(results, output_dir):
    path = os.path.join(output_dir, 'results_spectrum.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction='ignore')
        writer.writeheader()
        for r in sorted(results, key=lambda r: r['depth_m']):
            writer.writerow(r)
    print('Saved:', path)


def write_npz(results, freq_hz, E_ref, output_dir):
    depth_results = sorted(results, key=lambda r: r['depth_m'])
    path = os.path.join(output_dir, 'spectra.npz')
    np.savez(
        path,
        freq_hz=freq_hz,
        depths_m=np.array([r['depth_m'] for r in depth_results]),
        keys=np.array([r['key'] for r in depth_results]),
        E_ref=E_ref,
        E_meas=np.stack([r['E_meas'] for r in depth_results]),
        E_theory=np.stack([r['E_theory'] for r in depth_results]),
        L_abs_meas=np.stack([r['L_abs_meas'] for r in depth_results]),
        L_abs_theory=np.stack([r['L_abs_theory'] for r in depth_results]),
        L_rel_meas=np.stack([r['L_rel_meas'] for r in depth_results]),
        L_rel_theory=np.stack([r['L_rel_theory'] for r in depth_results]),
        alpha_abs=np.stack([r['alpha_abs'] for r in depth_results]),
        alpha_rel=np.stack([r['alpha_rel'] for r in depth_results]),
        tandelta_abs=np.stack([r['tandelta_abs'] for r in depth_results]),
        tandelta_rel=np.stack([r['tandelta_rel'] for r in depth_results]),
        tau_g=np.stack([r['tau_g'] for r in depth_results]),
        valid_mask=np.stack([r['mask'] for r in depth_results]),
    )
    print('Saved:', path)


def write_run_info(level, kind, json_path, results, output_dir):
    path = os.path.join(output_dir, 'run_info.txt')
    with open(path, 'w') as f:
        f.write('ascan_spectrum.py run info\n')
        f.write('executed at: {}\n'.format(datetime.now().isoformat()))
        f.write('level: {}\n'.format(level))
        f.write('waveform: {}\n'.format(kind if kind else '(未指定)'))
        f.write('json_path: {}\n'.format(json_path))
        f.write('\nParameters:\n')
        f.write('  TX_HEIGHT = {} m\n'.format(TX_HEIGHT))
        f.write('  R_REF = {} m\n'.format(R_REF))
        f.write('  N_REGOLITH = {:.6f} (eps_r={})\n'.format(N_REGOLITH, EPS_R_REGOLITH))
        if 'absorb_const' in LEVEL_EFFECTS.get(level, []):
            f.write('  Level 2 medium: {}\n'.format(describe_level2_medium(N_REGOLITH)))
        if 'absorb_tandelta' in LEVEL_EFFECTS.get(level, []):
            f.write('  Level 3 medium: {}\n'.format(describe_level3_medium()))
        if 'ice_layer' in LEVEL_EFFECTS[level]:
            f.write('  Level 4 ice layer: {}\n'.format(describe_level4_medium()))
        if 'absorb_debye' in LEVEL_EFFECTS.get(level, []):
            f.write('  Level 3b medium: {}\n'.format(describe_level3b_medium()))
        f.write('  BAND_GHZ = {}\n'.format(BAND_GHZ))
        f.write('  MASK_REF_FLOOR_DB = {}\n'.format(MASK_REF_FLOOR_DB))
        f.write('  MASK_SNR_MIN_DB = {}\n'.format(MASK_SNR_MIN_DB))
        f.write('  GATE_ENABLED = {}\n'.format(GATE_ENABLED))
        f.write('  LSR_REF_DEPTH_M = {}\n'.format(LSR_REF_DEPTH_M))
        f.write('  FLOHI_THRESHOLDS_DB = {}\n'.format(FLOHI_THRESHOLDS_DB))
        f.write('  NOISE_BAND_GHZ = {}\n'.format(NOISE_BAND_GHZ))
        f.write('  LSR_TOL_DB = {}\n'.format(LSR_TOL_DB))
        f.write('  LSR_FLATNESS_TOL_DB = {}\n'.format(LSR_FLATNESS_TOL_DB))
        f.write('  FC_TOL_MHZ = {}\n'.format(FC_TOL_MHZ))
        f.write('  TAUG_TOL_FRAC = {}\n'.format(TAUG_TOL_FRAC))
        f.write('  NOISE_FLOOR_DB = {}\n'.format(NOISE_FLOOR_DB))
        f.write('\nPass/fail summary:\n')
        n = len(results)
        for pass_key, label in [
            ('pass_lsr_level', 'lsr_level'), ('pass_lsr_flatness', 'lsr_flatness'),
            ('pass_fc', 'fc'), ('pass_taug', 'taug'), ('pass_noise', 'noise'),
        ]:
            n_pass = sum(1 for r in results if r[pass_key])
            f.write('  {}: {}/{} pass\n'.format(label, n_pass, n))
    print('Saved:', path)


# =============================================================================
# main
# =============================================================================
def main():
    level, kind, rx_paths, reference = load_paths(JSON_PATH)

    # 背景レゴリスの組成をサブ階層キーから設定する。
    # Level 3 と Level 4 は背景が同一なので同じ経路を通る（他レベルでは無視）。
    if 'absorb_tandelta' in LEVEL_EFFECTS.get(level, []):
        wt, key = set_level3_composition(kind)
        print('背景レゴリスの組成: FeO+TiO2 = {:.1f} wt%  [{}]'.format(wt, key))

    # 水氷濃度もサブ階層キーから設定する（氷層を持つレベルのみ）。
    if 'ice_layer' in LEVEL_EFFECTS.get(level, []):
        vol, ice_key = set_level4_ice(kind)
        print('水氷濃度: {:.2f} vol%  [{}]'.format(vol, ice_key))

    if level not in IMPLEMENTED_LEVELS:
        raise NotImplementedError(
            '{} は未実装です（実装済み: {}）。Level_2 以降は吸収項の物性値確定後に '
            '追加してください。'.format(level, ', '.join(sorted(IMPLEMENTED_LEVELS))))

    check_paths_exist(rx_paths, reference)

    results, freq_hz, E_ref, d0_key, d0 = analyze_level(rx_paths, reference, level)

    output_dir = resolve_output_dir(level, rx_paths)
    os.makedirs(output_dir, exist_ok=True)

    plot_spectra(results, freq_hz, E_ref, output_dir)
    plot_lsr(results, freq_hz, d0, output_dir)
    plot_attenuation(results, freq_hz, level, N_REGOLITH, output_dir)
    plot_phase(results, freq_hz, output_dir)
    write_csv(results, output_dir)
    write_npz(results, freq_hz, E_ref, output_dir)
    write_run_info(level, kind, JSON_PATH, results, output_dir)

    print('\nAll outputs saved to:', output_dir)


if __name__ == '__main__':
    main()