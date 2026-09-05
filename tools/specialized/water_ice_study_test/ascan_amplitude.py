"""A-scan 強度検証コード

対象：減衰計測GPRによる月の水氷検出研究／複雑性のはしご Level 1 以降
目的：A-scan の振幅・到達時刻を理論予測と比較し、各 Level で理論からの
      乖離が生じる複雑性を特定する。

設計書: design_ascan_amplitude.md (v1, 2026-08-08) に準拠。

現時点で完全実装しているのは Level_1（geom + surface_T）のみ。
Level_2〜4 の吸収項（absorb_const / absorb_debye / density_profile）は、
設計書に具体的な物性値（tanδ 等）が未確定のため、骨格のみ用意し
NotImplementedError とする（設計書 §10「実装の進め方」のロードマップに従い、
Level_1 の合格を確認してから追加する）。
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import argparse
import csv
import json
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# 地下構造モデル（subsurface_model.py に集約）
# =============================================================================
# .in ファイルが定める物理（Carrier 経験式、最大平坦 2 極 Debye、水氷の混合、
# 減衰係数、屈折率、レベルごとの効果の定義、取得ジオメトリの定数）は
# すべて subsurface_model.py が持つ。このファイルには「解析の手順」だけを置く。
#
# レベルを増やすときに触るのは原則 subsurface_model.py だけになる。
# 以前は同じ物理が ascan_spectrum / ascan_amplitude / ascan_reflection* の
# 各所に散在していたが、修正項目 A-1 と同じ事故（定数の二重定義）を防ぐため
# 一本化した。
#
# 水氷の描像（pore = 空隙充填 / excess = バルク置換）も subsurface_model が
# 持ち、JSON のサブ階層キー（pore_ice / excess_ice）から自動判定する。
# 他のモジュールが階層選択のラベルを追加するためのフック。
# 例: _EXTRA_LAYER_LABELS.append((判定関数, 'ラベル'))
_EXTRA_LAYER_LABELS = []

import subsurface_model as sm
from subsurface_model import (  # noqa: F401  （再エクスポートを兼ねる）
    C, TX_HEIGHT, R_REF, EPS_R_REGOLITH, N_REGOLITH, BAND_GHZ,
    BAND_CENTRE_HZ, LEVEL_EFFECTS, IMPLEMENTED_LEVELS, LEVEL2_LOSS_MODEL,
    LEVEL2_SIGMA, LEVEL2_TAN_DELTA, ETA0, EPS0, LEVEL3_EPS_R, LEVEL3_RHO,
    LEVEL3_CARRIER_EPS_BASE, LEVEL3_CARRIER_TAND_A, LEVEL3_CARRIER_TAND_B,
    LEVEL3_CARRIER_TAND_C, LEVEL3_COMPOSITIONS, LEVEL3_DEFAULT_COMPOSITION,
    LEVEL3_EPS_REAL_MODE, LEVEL3_EPS_IMAG_MODE, LEVEL3_DEBYE_BAND_HZ,
    LEVEL3_DEBYE_F0, LEVEL3_DEBYE_TAU, LEVEL4_ICE_TOP_M, LEVEL4_ICE_THICK_M,
    LEVEL4_ICE_SPEC, LEVEL4_ICE_VOL_PCT, LEVEL4_ICE_WT_PCT, LEVEL4_EPS_ICE,
    LEVEL4_TAND_ICE, LEVEL4_RHO_ICE, LEVEL4_RHO_GRAIN, LEVEL4_ICE_KEYS,
    LEVEL4_ICE_MODEL, LEVEL4_ICE_MODELS, LEVEL4_ICE_MODEL_KEYS, LEVEL3B_RHO,
    LEVEL3B_FEOTIO2, LEVEL3B_ANCHOR_FREQ, LEVEL3B_CARRIER_EPS_BASE,
    LEVEL3B_CARRIER_TAND_A, LEVEL3B_CARRIER_TAND_B, LEVEL3B_CARRIER_TAND_C,
    LEVEL3B_DEBYE_DE1, LEVEL3B_DEBYE_TAU1, LEVEL3B_DEBYE_DE2,
    LEVEL3B_DEBYE_TAU2, debye_flat_eps, apply_eps_modes, level2_tandelta,
    level2_alpha, describe_level2_medium, level3_carrier_tandelta,
    _parse_composition_key, _is_composition_layer, level3_feotio2,
    set_level3_composition, _parse_ice_key, _is_ice_layer,
    level4_ice_volpct, set_level4_ice, _parse_ice_model_key,
    _is_ice_model_layer, level4_ice_model_from_kind, set_level4_ice_model,
    configure_from_kind, level3_targets, level3_eps, level3_tandelta,
    level3_alpha, _group_index_from_eps, level3_group_index,
    level4_porosity, level4_ice_weight_fraction, level4_ice_volume_fraction,
    level4_targets, level4_eps, level4_tandelta, level4_alpha,
    level4_segments, level4_interfaces_crossed, level4_alpha_path_avg,
    describe_level4_medium, refractive_index, level3b_carrier,
    level3b_debye_scale, level3b_eps, level3b_tandelta, level3b_alpha,
    level3b_group_index, describe_level3b_medium, describe_level3_medium,
)
from scipy import signal

from gprMax.exceptions import CmdInputError
from tools.core.outputfiles_merge import get_output_data

# =============================================================================
# 定数
# =============================================================================
# [EDIT HERE] パス JSON のハードコード（利用例は design_ascan_amplitude.md §2 参照）
JSON_PATH = "/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_3x4/water_ice_study_test/out_file_paths.json"


# Level_1 のレゴリス物性

# 測定関数パラメータ
SEARCH_HALFWIDTH_NS = 2.0     # [ns] 理論到達時刻からの探索窓半幅
NOISE_WINDOW_NS = (40.0, 50.0)  # [ns] ノイズフロア評価窓

# 合格基準 (design_ascan_amplitude.md §7)
AMP_TOL_DB = 0.5           # 振幅残差 |残差| < 0.5 dB
TIME_TOL_FRAC = 0.01       # 到達時刻残差 < 走時の1%
NOISE_FLOOR_DB = -60.0     # 直達波の -60 dB 以下
T_TOL_FRAC = 0.01          # 透過係数 ±1%

# レベル定義 (design_ascan_amplitude.md §4.3)

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
# 現時点で実装済みのレベル（それ以外は未実装のため実行不可）

# fig3_waveforms.png で重ね描きする代表深さ
REPRESENTATIVE_DEPTHS_M = [0.50, 1.50, 2.75]

# JSON の下位選択階層につけるラベル（階層が深いほうまで使う）
SUBLEVEL_LABELS = ['波形種別', 'サブ条件', '組成 (FeO+TiO2)', 'サブ条件']

# 解析対象から除外する rx キー (design_ascan_amplitude.md §3)
#   depth_300 は y=0.0 で PML（gprMax デフォルト 10 層 = 0.05 m）の中にあるため、
#   物理的に意味のあるデータにならない。JSON に残っていても自動で除外する。
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
                                     # 'tan_delta'    … 参考用。Level 3 相当の理想化
                                     #   プロファイル計算の 0 vol% ice / 1.25 GHz の値。
                                     #   tan_delta = 0.01678 @ 1.25 GHz に相当。


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
                                  # f0 = sqrt(0.5*2.0) = 1.0 GHz（.in と共通）。
                                  # 帯域中心 1.25 GHz では 2.998（-0.07%）。


# FeO+TiO2 [wt%] ごとの設定。JSON のサブ階層キーと対応させる。
#   月南極域   : 5 / 7.5 / 10 wt%（先行研究の収束域 6-11 wt% を挟む）
#   高Tiバサルト: 20 wt%（月の海。参考ケース。既存計算がこれに相当）


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
                                  # 'ideal' … eps' = LEVEL3_EPS_R 一定（旧挙動）
                                  # 'debye' … 2 極 Debye の実装値に合わせる

_L3_S_FLAT = float(np.arcsinh(1.0))          # = ln(1+sqrt2) = 0.881374
_L3_TAU_RATIO = float(np.exp(_L3_S_FLAT))    # = 1+sqrt2 = 2.414214
_L3_W0 = 2.0 * np.pi * LEVEL3_DEBYE_F0


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

                              # 'wt'  … 質量パーセントで指定
# 【単位に注意】どちらもパーセント。分率ではない（10 vol% なら 10.0）。


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

LEVEL3B_DEBYE_DE1, LEVEL3B_DEBYE_TAU1 = 0.261, 4.6212e-11    # [s]
LEVEL3B_DEBYE_DE2, LEVEL3B_DEBYE_TAU2 = 0.088, 2.82195e-10   # [s]

# 走時・探索窓の基準に使う周波数（帯域中心）

# 解析対象から除外する rx キー (design_ascan_amplitude.md §3)
#   depth_300 は y=0.0 で PML（gprMax デフォルト 10 層 = 0.05 m）の中にあるため、
#   物理的に意味のあるデータにならない。JSON に残っていても自動で除外する。
EXCLUDE_KEYS = {'depth_300'}

# 出力先 (レベル親ディレクトリ配下)
#   <Level_N>/OUTPUT_PARENT_DIRNAME/OUTPUT_SUBDIRNAME/ に解析結果を書き出す。
OUTPUT_PARENT_DIRNAME = 'analysis'
OUTPUT_SUBDIRNAME = 'ascan_amplitude'

# =============================================================================
# 入出力
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
        elif _is_ice_model_layer(nested):
            label = '水氷の描像 (pore / excess)'
        elif _EXTRA_LAYER_LABELS and any(
                fn(nested) for fn, _ in _EXTRA_LAYER_LABELS):
            label = next(lab for fn, lab in _EXTRA_LAYER_LABELS if fn(nested))
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

    想定する JSON 構成（"_" 始まりのキーはレベル／rx として扱わない）:

        {
          "_reference": {                     # 全レベル共通の参照計算
            "gaussiandot":         {"far_1m": ..., "at_tx": ..., ...},
            "excitation_waveform": {"far_1m": ..., "at_tx": ..., ...}
          },
          "Level_1": {
            "gaussiandot":         {"at_surface": ..., "depth_025": ...},
            "excitation_waveform": {                       # さらに下位階層があってもよい
              "dx_0005":  {"at_surface": ..., "depth_025": ...},
              "dx_00025": {"at_surface": ..., "depth_025": ...}
            }
          }
        }

    波形種別の階層を省いて、レベル直下に rx キーを並べた旧形式にも対応する。
    その場合は波形種別の選択を行わない。

    _reference はトップレベルにも各レベル内にも置ける（同名キーはレベル内が優先）。
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


def resolve_output_dir(level, rx_paths):
    """レベル親ディレクトリ配下に出力ディレクトリのパスを組み立てる。

    rx の .out パス（例 .../Level_1_gaussian_dot/depth_025/result/Ascan.out）の
    共通祖先をレベル親ディレクトリとみなし、その下に analysis/ascan_amplitude/ を作る。

    波形種別ごとにディレクトリが分かれている構成なら、出力も自動的に分かれる。
    """
    paths = [os.path.abspath(p) for p in rx_paths.values()]
    if len(paths) > 1:
        level_root = os.path.commonpath(paths)
    else:
        # rx が 1 つだけの場合は共通祖先が取れないので、
        # <親>/<rx名>/result/Ascan.out という構成を仮定して 3 段上る。
        level_root = os.path.dirname(os.path.dirname(os.path.dirname(paths[0])))

    # レベル名はパスのどこかに現れていればよい。次の 2 通りを許容する。
    #   .../Level_3/FeO_100/          組成ごとにサブディレクトリを切る構成
    #   .../Level_1_excitation_waveform/   ディレクトリ名にサフィックスが付く構成
    parts = level_root.split(os.sep)
    if not any(level in part for part in parts):
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
# 理論：伝達関数
# =============================================================================
def transfer_geom(d, n):
    """幾何減衰項（2D遠方場、法線入射の見かけの源距離 r_eff を用いる）。

    参照計算との比を取ると 2D Green 関数の 1/sqrt(k) が相殺するため、
    この項自体は「周波数に依存しない実数」になる。ただし n が周波数依存の
    Level 3 では n を通じて f 依存が復活する（n は配列で渡ってくる）。
    """
    r_eff = n * TX_HEIGHT + d
    return np.sqrt(n / r_eff) * np.sqrt(R_REF)


def transfer_surface_T(n):
    """地表面透過係数 T = 1 + R（Ez は界面に接線 → 連続）。深さに依存しない定数。"""
    return 2.0 / (1.0 + n)


def transfer_phase(f, d, n):
    """走時位相項。参照計算自身の伝搬遅延 (r_ref/c) を差し引いて基準を揃える。

    f は Hz（np.fft.rfftfreq に SI 秒の dt を渡した結果）、t_arr は ns 単位のため、
    位相計算では (t_arr - r_ref/c) を秒に変換してから f と掛け合わせる。

    位相には「位相速度」を決める n(f) を使う。Level 3 のように n が周波数依存だと
    位相走時も周波数依存になり、逆 FFT した波形の包絡ピークは自動的に
    群速度で決まる位置に現れる（群遅延を別途足す必要はない）。

    戻り値の t_arr はスカラー（探索窓の中心と CSV 用）で、n の代表値から作る。
    """
    t_arr_f = TX_HEIGHT / C + n * d / C
    delay_s = (t_arr_f - R_REF / C) * 1e-9
    phase = np.exp(-2j * np.pi * f * delay_s)
    return phase, t_arr_f


def transfer_absorb(f, d, n):
    """媒質の吸収項 exp(-alpha(f) * d)（片道透過）。Level 2 以降で使う。"""
    return np.exp(-level2_alpha(f, n) * d)


_LEVEL3_ACTIVE_WT = LEVEL3_COMPOSITIONS[LEVEL3_DEFAULT_COMPOSITION]
_LEVEL3_ACTIVE_KEY = LEVEL3_DEFAULT_COMPOSITION




_LEVEL4_ACTIVE_ICE_KEY = None


# =============================================================================
# 理論：Level 4（水氷層を透過して深さ d に届く波）
# =============================================================================
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


def transfer_absorb_tandelta(f, d, n=None):
    """Level 3 の吸収項 exp(-alpha(f) * d)（片道透過、eps'' 一定）。"""
    return np.exp(-level3_alpha(f) * d)


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


def resample_trace(trace, dt_src, dt_dst):
    """時間刻みの違う格子へリサンプルする。

    dx を変えると gprMax の dt も変わるため、参照計算（例 dx=5 mm）と
    解析対象（例 dx=2.5 mm）でサンプル数が食い違う。信号は 0.5-2.0 GHz に
    帯域制限されており Nyquist より遥かに低いので、FFT 補間が実質厳密になる。
    """
    if _same_dt(dt_src, dt_dst):
        return trace
    n_dst = int(round(len(trace) * dt_src / dt_dst))
    return signal.resample(trace, n_dst)


def align_length(trace, n):
    """長さ n に切り詰め、足りなければ 0 で埋める（差分・重ね描き用）。"""
    if len(trace) == n:
        return trace
    if len(trace) > n:
        return trace[:n]
    out = np.zeros(n, dtype=trace.dtype)
    out[:len(trace)] = trace
    return out


_RESAMPLE_NOTICED = set()


def reference_on_grid(e_ref, dt_ref, dt_target, n_target, label=''):
    """参照波形を対象トレースと同じ時間格子に載せ替える。

    通知は格子（dt, サンプル数）の組み合わせごとに 1 度だけ出す。
    """
    if _same_dt(dt_ref, dt_target) and len(e_ref) == n_target:
        return e_ref
    key = (round(dt_ref, 18), round(dt_target, 18), len(e_ref), n_target)
    if key not in _RESAMPLE_NOTICED:
        _RESAMPLE_NOTICED.add(key)
        print('')
        print('*** 警告: 参照計算と dt が異なります '
              '(dt {:.4f} ps vs 参照 {:.4f} ps) ***'.format(dt_target * 1e12, dt_ref * 1e12))
        print('    dt の違いは dx の違いを意味します。時間格子はリサンプルで揃えられますが、')
        print('    gprMax の #hertzian_dipole は電流密度 J = I*dl/(dx*dy*dz) と')
        print('    セル寸法で正規化されるため、2D では波源の絶対振幅が dx に依存します。')
        print('    このまま進めると絶対振幅が定数倍ずれます（dx 2 倍で約 6 dB）。')
        print('    → 解析対象と同じ dx で計算した参照計算を JSON に指定してください。')
        print('    （相対LSR や深さ間の比較には影響しません）')
        print('')
    return align_length(resample_trace(e_ref, dt_ref, dt_target), n_target)


def synth_theory(e_ref, dt_ref, d, level, n):
    """E_ref(f)・H_level(f,d) から理論波形を合成する（周波数領域でフォワードモデル→IFFT）。"""
    N = len(e_ref)
    freq = np.fft.rfftfreq(N, d=dt_ref)
    E_ref_f = np.fft.rfft(e_ref)
    H, t_arr = build_transfer(freq, d, level, n)
    e_theory = np.fft.irfft(E_ref_f * H, n=N)
    return e_theory, t_arr


def synth_theory_reflect(e_ref, dt_ref, n):
    """at_tx（反射測定）の理論波形。自己場を含まない片道透過ではなく往復反射式を使う。"""
    N = len(e_ref)
    freq = np.fft.rfftfreq(N, d=dt_ref)
    E_ref_f = np.fft.rfft(e_ref)

    R = (1.0 - n) / (1.0 + n)
    A = R * np.sqrt(R_REF / (2.0 * TX_HEIGHT))
    t_arr = 2.0 * TX_HEIGHT / C
    delay_s = (t_arr - R_REF / C) * 1e-9
    phase = np.exp(-2j * np.pi * freq * delay_s)

    e_theory = np.fft.irfft(E_ref_f * A * phase, n=N)
    return e_theory, t_arr


# =============================================================================
# 測定関数（実測・理論の両方に同一の関数を適用する）
# =============================================================================
def measure(trace, dt, t_predicted=None, search_halfwidth=SEARCH_HALFWIDTH_NS, label=''):
    """A-scan から直達波の特徴量を抽出する。

    Parameters
    ----------
    t_predicted : float or None
        探索窓の中心 [ns]。None なら全区間から包絡の最大値を探す
        （自由空間参照のように、対象パルスが 1 つしかない場合に使う）。

    Returns
    -------
    dict with keys:
        'amp_peak' : 包絡（|Hilbert|）のピーク値
        't_peak'   : 包絡ピークの時刻 [ns]（放物線内挿でサブサンプル精度）
        'noise_db' : 後方窓（40-50 ns）の包絡RMS / amp_peak [dB]

    fwhm は design_ascan_amplitude.md §9-5 の方針により、当面は amp_peak /
    t_peak のみで運用するため未実装。
    """
    n_samples = len(trace)
    dt_ns = dt * 1e9
    t_axis = np.arange(n_samples) * dt_ns
    envelope = np.abs(signal.hilbert(trace))

    if t_predicted is None:
        idx_in_window = np.arange(n_samples)
        lo, hi = t_axis[0], t_axis[-1]
    else:
        lo, hi = t_predicted - search_halfwidth, t_predicted + search_halfwidth
        window_mask = (t_axis >= lo) & (t_axis <= hi)
        if not np.any(window_mask):
            raise CmdInputError(
                '探索窓 [{:.3f}, {:.3f}] ns 内にサンプルがありません（trace 長: {:.3f} ns）'.format(
                    lo, hi, t_axis[-1]))
        idx_in_window = np.where(window_mask)[0]

    peak_idx = idx_in_window[np.argmax(envelope[idx_in_window])]

    # 探索窓の端でピークが見つかった場合、真のピークは窓の外にある可能性が高い。
    # 波源の内部遅延を取り違えるとこの状態になり、残差が全深さで一様にずれる。
    if t_predicted is not None and peak_idx in (idx_in_window[0], idx_in_window[-1]):
        print('Warning: {}探索窓 [{:.2f}, {:.2f}] ns の端でピークが見つかりました。'
              '真のピークが窓の外にある可能性があります。'.format(
                  '[{}] '.format(label) if label else '', lo, hi))

    if 0 < peak_idx < n_samples - 1:
        y0, y1, y2 = envelope[peak_idx - 1], envelope[peak_idx], envelope[peak_idx + 1]
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        delta = float(np.clip(delta, -1.0, 1.0))
    else:
        delta = 0.0

    t_peak = t_axis[peak_idx] + delta * dt_ns
    amp_peak = envelope[peak_idx]

    noise_mask = (t_axis >= NOISE_WINDOW_NS[0]) & (t_axis <= NOISE_WINDOW_NS[1])
    if np.any(noise_mask):
        noise_rms = np.sqrt(np.mean(envelope[noise_mask] ** 2))
        noise_db = 20.0 * np.log10(noise_rms / amp_peak + 1e-30)
    else:
        print('Warning: noise window {} ns が trace 長 {:.3f} ns を超えるため noise_db は NaN'.format(
            NOISE_WINDOW_NS, t_axis[-1]))
        noise_db = np.nan

    return {'amp_peak': amp_peak, 't_peak': t_peak, 'noise_db': noise_db}


# =============================================================================
# メイン処理：全 rx に対して実測・理論の比較を行う
# =============================================================================
def analyze_level(rx_paths, reference, level):
    if 'far_1m' not in reference:
        raise CmdInputError('_reference.far_1m が JSON にありません（E_ref(f) の校正に必要）')
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
    results = []

    # 参照波形（自由空間 1 m）の包絡ピークを全区間から求める。
    # ここから dB の基準振幅と、波源自身の内部遅延の 2 つを得る。
    ref_meas = measure(e_ref, dt_ref, None, label='reference far_1m')
    amp_ref = ref_meas['amp_peak']

    # 波源の内部遅延：波源波形のピークが t=0 からどれだけ後ろにあるか。
    #   gaussiandot           -> chi = 1/f = 0.8 ns 程度
    #   帯域制限 excitation   -> T_CENTER = 5.0 ns 程度
    # 理論波形のピークは t_arr + この遅延に現れるので、探索窓の中心に必ず加える。
    # 参照から実測するので、波源を変えても定数を書き換える必要がない。
    t_src_delay = ref_meas['t_peak'] - R_REF / C
    print('波源の内部遅延（参照波形から実測）: {:.3f} ns'.format(t_src_delay))
    if t_src_delay > SEARCH_HALFWIDTH_NS:
        print('  (SEARCH_HALFWIDTH_NS = {} ns より大きいため、'
              '探索中心の補正は必須)'.format(SEARCH_HALFWIDTH_NS))

    # --- 深さ依存の rx（at_surface, depth_XXX） ---
    for key, path in rx_paths.items():
        if key == 'at_tx':
            continue
        d = rx_depth(key)
        trace, dt = load_trace(path)
        # 参照を実測と同じ時間格子に載せ替えてから理論波形を合成する。
        # こうすると dt が違っても measure() も fig3 の重ね描きも成立する。
        e_ref_g = reference_on_grid(e_ref, dt_ref, dt, len(trace), label=key)
        e_theory, t_arr = synth_theory(e_ref_g, dt, d, level, n)

        # 探索窓の中心：
        #   理論波形 -> t_arr + 波源の内部遅延（ピークが実際に現れる位置）
        #   実測波形 -> その理論ピーク位置
        # measure() は同一関数のままで、中心の与え方だけを変えている。
        theo = measure(e_theory, dt, t_arr + t_src_delay, label=key + ' theory')
        meas = measure(trace, dt, theo['t_peak'], label=key + ' measured')

        amp_meas_db = 20.0 * np.log10(meas['amp_peak'] / amp_ref)
        amp_theory_db = 20.0 * np.log10(theo['amp_peak'] / amp_ref)
        amp_resid_db = amp_meas_db - amp_theory_db

        t_resid_ns = meas['t_peak'] - theo['t_peak']
        t_resid_frac = abs(t_resid_ns) / theo['t_peak']

        results.append({
            'key': key, 'depth_m': d, 'kind': 'depth',
            't_meas_ns': meas['t_peak'], 't_theory_ns': theo['t_peak'],
            't_resid_ns': t_resid_ns, 't_resid_frac': t_resid_frac,
            'amp_meas_db': amp_meas_db, 'amp_theory_db': amp_theory_db,
            'amp_resid_db': amp_resid_db, 'noise_db': meas['noise_db'],
            'pass_amp': abs(amp_resid_db) < AMP_TOL_DB,
            'pass_time': t_resid_frac < TIME_TOL_FRAC,
            'pass_noise': meas['noise_db'] < NOISE_FLOOR_DB,
            'trace_meas': trace, 'trace_theory': e_theory, 'dt': dt,
        })

    # --- at_tx（反射測定、自己場差分） ---
    if 'at_tx' in rx_paths and 'at_tx' not in reference:
        print('Notice: _reference.at_tx が JSON にないため at_tx の解析をスキップします。'
              'at_tx は自己場が地表面反射より 40-60 dB 大きく、同一ジオメトリの自由空間計算を'
              '差し引かないと比較できません (design_ascan_amplitude.md §4.4)。')
    elif 'at_tx' in rx_paths:
        trace_at_tx, dt_at_tx = load_trace(rx_paths['at_tx'])
        trace_at_tx_free, dt_free = load_trace(reference['at_tx'])
        # 自己場の差分はサンプルごとの引き算なので、格子を厳密に揃える必要がある。
        trace_free_g = reference_on_grid(trace_at_tx_free, dt_free, dt_at_tx,
                                         len(trace_at_tx), label='at_tx freespace')
        e_reflect = trace_at_tx - trace_free_g

        e_ref_g = reference_on_grid(e_ref, dt_ref, dt_at_tx, len(trace_at_tx), label='at_tx')
        e_theory_reflect, t_arr_reflect = synth_theory_reflect(e_ref_g, dt_at_tx, n)

        theo = measure(e_theory_reflect, dt_at_tx, t_arr_reflect + t_src_delay,
                       label='at_tx theory')
        meas = measure(e_reflect, dt_at_tx, theo['t_peak'], label='at_tx measured')

        amp_meas_db = 20.0 * np.log10(meas['amp_peak'] / amp_ref)
        amp_theory_db = 20.0 * np.log10(theo['amp_peak'] / amp_ref)
        amp_resid_db = amp_meas_db - amp_theory_db
        t_resid_ns = meas['t_peak'] - theo['t_peak']
        t_resid_frac = abs(t_resid_ns) / theo['t_peak']

        results.append({
            'key': 'at_tx', 'depth_m': np.nan, 'kind': 'reflect',
            't_meas_ns': meas['t_peak'], 't_theory_ns': theo['t_peak'],
            't_resid_ns': t_resid_ns, 't_resid_frac': t_resid_frac,
            'amp_meas_db': amp_meas_db, 'amp_theory_db': amp_theory_db,
            'amp_resid_db': amp_resid_db, 'noise_db': meas['noise_db'],
            'pass_amp': abs(amp_resid_db) < AMP_TOL_DB,
            'pass_time': t_resid_frac < TIME_TOL_FRAC,
            'pass_noise': meas['noise_db'] < NOISE_FLOOR_DB,
            'trace_meas': e_reflect, 'trace_theory': e_theory_reflect, 'dt': dt_at_tx,
        })

    # --- 透過係数チェック（at_surface / 自由空間surface） ---
    t_check = None
    if 'at_surface' in rx_paths and 'at_surface' not in reference:
        print('Notice: _reference.at_surface が JSON にないため透過係数チェックをスキップします。'
              '（レゴリスなしの同一位置計算との比が T = 2/(1+n) の直接検証になります）')
    elif 'at_surface' in rx_paths and 'at_surface' in reference:
        trace_surf, dt_surf = load_trace(rx_paths['at_surface'])
        trace_surf_free, dt_surf_free = load_trace(reference['at_surface'])
        # 自由空間側は dt が違いうるので、必ずそれぞれの dt で測る
        # （実測側の dt を流用すると時間軸がずれ、静かに誤った T が出る）
        amp_surf = measure(trace_surf, dt_surf, TX_HEIGHT / C + t_src_delay,
                           label='at_surface (T check)')['amp_peak']
        amp_surf_free = measure(trace_surf_free, dt_surf_free, TX_HEIGHT / C + t_src_delay,
                                label='at_surface freespace (T check)')['amp_peak']
        T_meas = amp_surf / amp_surf_free
        # Level 3 では n が周波数依存なので、帯域中心の値で代表させる
        n_check = float(np.atleast_1d(refractive_index(BAND_CENTRE_HZ, level))[0])
        T_theory = transfer_surface_T(n_check)
        t_check = {
            'T_meas': T_meas, 'T_theory': T_theory,
            'pass': abs(T_meas - T_theory) / T_theory < T_TOL_FRAC,
        }

    return results, t_check, e_ref, dt_ref


# =============================================================================
# 作図
# =============================================================================
def plot_overview(results, output_dir, true_amplitude=False):
    depth_results = sorted([r for r in results if r['kind'] == 'depth'], key=lambda r: r['depth_m'])
    if not depth_results:
        return

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 14), sharey=True)

    # --- (a) ショットギャザー（wiggle 表示） ---
    ax = axes[0]
    depths = np.array([r['depth_m'] for r in depth_results])
    depth_step = np.median(np.diff(np.unique(depths))) if len(depths) > 1 else 1.0
    scale = 0.4 * depth_step

    global_peak = max(np.max(np.abs(r['trace_meas'])) for r in depth_results)
    for r in depth_results:
        dt_ns = r['dt'] * 1e9
        t_axis = np.arange(len(r['trace_meas'])) * dt_ns
        norm = np.max(np.abs(r['trace_meas'])) if not true_amplitude else global_peak
        wiggle = r['trace_meas'] / norm * scale
        ax.plot(t_axis, r['depth_m'] - wiggle, color='k', lw=0.8)
        ax.fill_between(t_axis, r['depth_m'], r['depth_m'] - wiggle,
                         where=(wiggle > 0), color='k', alpha=0.5, interpolate=True)

    # 理論到達時刻の曲線は、解析式の t_arr ではなく理論波形から measure() で得た
    # ピーク時刻 t_theory_ns を使う。波源（gaussiandot, chi=1/f=0.8 ns）自身の
    # 立ち上がり遅延を含むため、こちらが実測波形と直接比較できる量になる。
    t_curve = [r['t_theory_ns'] for r in depth_results]
    ax.plot(t_curve, depths, 'r--', lw=1.5, label='Theoretical arrival time (envelope peak)')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('rx depth [m]')
    ax.invert_yaxis()
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title('(a) Shot gather' + (' (true amplitude)' if true_amplitude else ' (normalized)'))

    # --- (b) 包絡ピーク振幅 vs rx 深さ ---
    ax = axes[1]
    ax.plot([r['amp_theory_db'] for r in depth_results], depths, 'r-', label='Theory')
    ax.plot([r['amp_meas_db'] for r in depth_results], depths, 'ko', label='Measured')
    ax.set_xlabel('Envelope peak amplitude [dB re. E_ref]')
    ax.set_ylabel('rx depth [m]')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title('(b) Amplitude vs depth')

    # --- (c) 残差 [dB] vs rx 深さ ---
    ax = axes[2]
    resid = [r['amp_resid_db'] for r in depth_results]
    ax.axvspan(-AMP_TOL_DB, AMP_TOL_DB, color='green', alpha=0.15, label='±{} dB'.format(AMP_TOL_DB))
    ax.axvline(0, color='gray', lw=1)
    ax.plot(resid, depths, 'ko-')
    ax.set_xlabel('Residual [dB] (meas - theory)')
    ax.set_ylabel('rx depth [m]')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title('(c) Amplitude residual (pass/fail)')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig1_overview.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('Saved:', path)
    plt.close(fig)


def plot_timing(results, output_dir):
    depth_results = sorted([r for r in results if r['kind'] == 'depth'], key=lambda r: r['depth_m'])
    if not depth_results:
        return

    depths = np.array([r['depth_m'] for r in depth_results])
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharey=True)

    ax = axes[0]
    ax.plot([r['t_theory_ns'] for r in depth_results], depths, 'r-', label='Theory')
    ax.plot([r['t_meas_ns'] for r in depth_results], depths, 'ko', label='Measured')
    ax.set_xlabel('Arrival time [ns]')
    ax.set_ylabel('rx depth [m]')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("(b') Arrival time vs depth")

    ax = axes[1]
    ax.axvline(0, color='gray', lw=1)
    ax.plot([r['t_resid_ns'] for r in depth_results], depths, 'ko-')
    ax.set_xlabel('Residual [ns] (meas - theory)')
    ax.set_ylabel('rx depth [m]')
    ax.grid(alpha=0.3)
    ax.set_title("(c') Arrival time residual")

    axes[0].invert_yaxis()
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig2_timing.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('Saved:', path)
    plt.close(fig)


def plot_waveforms(results, output_dir):
    # 浮動小数の等値比較を避け、許容誤差つきで代表深さに最も近い rx を選ぶ
    depth_results = {}
    for target in REPRESENTATIVE_DEPTHS_M:
        for r in results:
            if r['kind'] == 'depth' and np.isclose(r['depth_m'], target, atol=1e-6):
                depth_results[target] = r
                break
    targets = [d for d in REPRESENTATIVE_DEPTHS_M if d in depth_results]
    if not targets:
        print('Warning: fig3 用の代表深さが results に見つかりません:', REPRESENTATIVE_DEPTHS_M)
        return

    fig, axes = plt.subplots(nrows=len(targets), ncols=1, figsize=(8, 3.5 * len(targets)))
    if len(targets) == 1:
        axes = [axes]

    for ax, d in zip(axes, targets):
        r = depth_results[d]
        dt_ns = r['dt'] * 1e9
        t_axis = np.arange(len(r['trace_meas'])) * dt_ns
        t_arr = r['t_theory_ns']
        lo, hi = max(0.0, t_arr - 10.0), t_arr + 10.0
        mask = (t_axis >= lo) & (t_axis <= hi)

        ax.plot(t_axis[mask], r['trace_meas'][mask], 'k-', label='Measured')
        ax.plot(t_axis[mask], r['trace_theory'][mask], 'r--', label='Theory')
        ax.axvline(t_arr, color='gray', ls=':', lw=1)
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Ez (linear)')
        ax.set_title('depth = {:.2f} m'.format(d))
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig3_waveforms.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('Saved:', path)
    plt.close(fig)


# =============================================================================
# 出力
# =============================================================================
def write_csv(results, t_check, output_dir):
    path = os.path.join(output_dir, 'results.csv')
    fieldnames = ['key', 'depth_m', 'kind', 't_meas_ns', 't_theory_ns', 't_resid_ns',
                  't_resid_frac', 'amp_meas_db', 'amp_theory_db', 'amp_resid_db',
                  'noise_db', 'pass_amp', 'pass_time', 'pass_noise']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in sorted(results, key=lambda r: (r['kind'], r['depth_m'] if r['kind'] == 'depth' else -1)):
            writer.writerow(r)
    # 透過係数チェックは run_info.txt に記録する（results.csv は pandas 等で
    # そのまま読める純粋な CSV に保つため、末尾のコメント行は書かない）
    print('Saved:', path)


def write_run_info(level, kind, json_path, results, t_check, output_dir):
    path = os.path.join(output_dir, 'run_info.txt')
    with open(path, 'w') as f:
        f.write('ascan_amplitude.py run info\n')
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
            f.write('  Level 4 ice model: {}\n'.format(sm.LEVEL4_ICE_MODEL))
            f.write('  Level 4 ice layer: {}\n'.format(describe_level4_medium()))
        if 'absorb_debye' in LEVEL_EFFECTS.get(level, []):
            f.write('  Level 3b medium: {}\n'.format(describe_level3b_medium()))
        f.write('  SEARCH_HALFWIDTH_NS = {}\n'.format(SEARCH_HALFWIDTH_NS))
        f.write('  NOISE_WINDOW_NS = {}\n'.format(NOISE_WINDOW_NS))
        f.write('  AMP_TOL_DB = {}\n'.format(AMP_TOL_DB))
        f.write('  TIME_TOL_FRAC = {}\n'.format(TIME_TOL_FRAC))
        f.write('  NOISE_FLOOR_DB = {}\n'.format(NOISE_FLOOR_DB))
        f.write('\nPass/fail summary:\n')
        n_pass_amp = sum(1 for r in results if r['pass_amp'])
        n_pass_time = sum(1 for r in results if r['pass_time'])
        n_pass_noise = sum(1 for r in results if r['pass_noise'])
        f.write('  amp:   {}/{} pass\n'.format(n_pass_amp, len(results)))
        f.write('  time:  {}/{} pass\n'.format(n_pass_time, len(results)))
        f.write('  noise: {}/{} pass\n'.format(n_pass_noise, len(results)))
        if t_check is not None:
            f.write('  T (surface transmission): meas={:.4f} theory={:.4f} pass={}\n'.format(
                t_check['T_meas'], t_check['T_theory'], t_check['pass']))
    print('Saved:', path)


# =============================================================================
# main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='A-scan 強度検証コード')
    parser.add_argument('--true-amplitude', action='store_true',
                         help='fig1 のショットギャザーを正規化せず真値振幅で表示する')
    args = parser.parse_args()

    level, kind, rx_paths, reference = load_paths(JSON_PATH)

    # 背景レゴリスの組成・水氷濃度・水氷の描像を、サブ階層キーから設定する。
    # 判定と設定は subsurface_model が持ち、見つからなければエラーで止まる
    # （既定値に落とすと誤ったモデルのまま解析が完走してしまうため）。
    for _note in configure_from_kind(kind, level):
        print(_note)

    if level not in IMPLEMENTED_LEVELS:
        raise NotImplementedError(
            '{} は未実装です（実装済み: {}）。Level_2 以降は吸収項の物性値確定後に '
            '追加してください。'.format(level, ', '.join(sorted(IMPLEMENTED_LEVELS))))

    results, t_check, e_ref, dt_ref = analyze_level(rx_paths, reference, level)

    output_dir = resolve_output_dir(level, rx_paths)
    os.makedirs(output_dir, exist_ok=True)

    plot_overview(results, output_dir, true_amplitude=args.true_amplitude)
    plot_timing(results, output_dir)
    plot_waveforms(results, output_dir)
    write_csv(results, t_check, output_dir)
    write_run_info(level, kind, JSON_PATH, results, t_check, output_dir)

    print('\nAll outputs saved to:', output_dir)


if __name__ == '__main__':
    main()