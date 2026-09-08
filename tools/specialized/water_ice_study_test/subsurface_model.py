"""subsurface_model.py — 解析コード共通の地下構造モデル

複雑性のはしご（Level 1-8）の各レベルで、シミュレーションの .in ファイルが
実装している媒質を、解析側で再現するためのモジュール。

--- なぜ独立させたか -------------------------------------------------------
これまでは媒質モデルが ascan_spectrum.py の中にあり、他の解析コード
（ascan_reflection.py / ascan_reflection_spectrum.py）はそこから import して
いた。そのためレベルを更新するたびに解析コード全体を読み直す必要があり、
どこが「媒質の定義」でどこが「解析の手順」なのかも分かりにくかった。

このファイルは **.in ファイルが定める物理だけ**を持つ。解析の手順
（JSON の読み込み、ゲート、作図、スペクトル量の定義）は各解析コードに残す。
レベルを増やすときに触るのは原則このファイルだけになる。

--- 何を持ち、何を持たないか -----------------------------------------------
持つもの:
    * 取得ジオメトリの定数（tx 高さ、参照距離、帯域）
    * レベルごとの効果の定義（LEVEL_EFFECTS）
    * 乾燥レゴリスの誘電モデル（Carrier Fig. 9.53 + 最大平坦 2 極 Debye）
    * 水氷の混合（pore / excess の 2 描像）
    * 減衰係数・屈折率・群屈折率
    * JSON のサブ階層キーからの自動判定（組成 / 氷量 / 氷の描像）
持たないもの:
    * JSON の読み込みとネスト選択、トレース読み込み、作図
    * スペクトルのモーメント、LSR、ゲート
    * 伝達関数（片道透過は ascan_spectrum、往復反射は ascan_reflection 側）

--- .in ファイルと揃えるべき定数 -------------------------------------------
    TX_HEIGHT / R_REF / BAND_GHZ
    LEVEL3_EPS_R / LEVEL3_CARRIER_* / LEVEL3_RHO
    LEVEL4_ICE_TOP_M / LEVEL4_ICE_THICK_M / LEVEL4_EPS_ICE / LEVEL4_TAND_ICE
    LEVEL4_RHO_ICE / LEVEL4_RHO_GRAIN
片方だけ変えると解析結果が黙ってずれるので、必ず両方を確認すること。

--- 水氷の描像（LEVEL4_ICE_MODEL）-----------------------------------------
    'pore'   空隙充填型（吸着水描像）。氷は粒子間の空隙だけを埋める。
             粒子の体積分率は不変なので eps'' が保存され、eps' が大きく
             増えて界面反射が立つ。上限は空隙率。
    'excess' 過剰氷型（バルク置換描像）。レゴリスの一部がまるごと純氷に
             置き換わる。粒子が (1-v_ice) 倍に減るので eps'' が希釈され
             減衰が緩和されるが、eps_ice ~ eps_dry のため eps' がほとんど
             変わらず界面反射がほぼ立たない。空隙率の上限がない。
JSON のサブ階層キー（pore_ice / excess_ice）から自動判定する。

【注意】現時点では ascan_spectrum.py も同じ媒質モデルを内部に持っている。
将来 ascan_spectrum.py 側をこのモジュールの import に切り替えて一本化する
こと。それまでは両者が一致していることを check_against_ascan_spectrum() で
確認できる。
"""

import contextlib
import re

import numpy as np

try:
    from gprMax.exceptions import CmdInputError
except Exception:                       # gprMax が無い環境でも読めるようにする
    class CmdInputError(Exception):
        pass


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
                      'Level_4', 'Level_5'}

# JSON の下位選択階層につけるラベル（階層が深いほうまで使う）
SUBLEVEL_LABELS = ['波形種別', 'サブ条件', '組成 (FeO+TiO2)', 'サブ条件']

# 解析対象から除外する rx キー (design_ascan_amplitude.md §3 と共通)

LEVEL2_LOSS_MODEL = 'conductivity'   # 'conductivity' … gprMax の #material に対応（既定）
                                     # 'tan_delta'    … 参考用。Level 3 相当の理想化
LEVEL2_SIGMA = 0.0035                # [S/m] #material の第 2 引数と一致させること。
                                     #   プロファイル計算の 0 vol% ice / 1.25 GHz の値。
                                     #   tan_delta = 0.01678 @ 1.25 GHz に相当。
LEVEL2_TAN_DELTA = 0.0155            # LEVEL2_LOSS_MODEL='tan_delta' のときのみ使う

ETA0 = 376.730313668                 # [Ohm] 真空の波動インピーダンス
EPS0 = 8.8541878128e-12              # [F/m] 真空の誘電率

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
_L4_EPS_ICE_CBRT = LEVEL4_EPS_ICE ** (1.0 / 3.0)       # = 1.46590

# --- 水氷の存在形態（描像）---------------------------------------------------
# Level_N.in の ICE_MODEL と対応させること。JSON のサブ階層キー
# （pore_ice / excess_ice）から自動判定するので、通常は手で設定しない。
#
#   'pore'   空隙充填型（吸着水描像）
#            氷は粒子間の空隙（真空）だけを埋める。粒子の体積分率は不変。
#              eps_wet^(1/3) = eps_dry^(1/3) + v*(eps_ice^(1/3) - 1)
#              eps''_wet     = eps''_dry + v*eps_ice*tan_delta_ice
#            粒子が減らないので減衰はほぼ緩和されないが、真空(1.0)が
#            氷(3.15)に置き換わるので eps' が大きく増え界面反射が立つ。
#            上限は空隙率（33.7 vol% = 14.9 wt%）。
#
#   'excess' 過剰氷型（バルク置換描像）
#            レゴリス（粒子＋空隙）の一部がまるごと純氷に置き換わる。
#              eps_wet^(1/3) = (1-v)*eps_dry^(1/3) + v*eps_ice^(1/3)
#              eps''_wet     = (1-v)*eps''_dry + v*eps_ice*tan_delta_ice
#            粒子が (1-v) 倍に減るので eps'' が希釈され減衰が緩和されるが、
#            eps_ice(3.15) と eps_dry(3.0) が近いので eps' がほとんど
#            変わらず界面反射がほぼ立たない。空隙率の上限がない。
#
# 2 つは減衰チャネルと反射チャネルで有利・不利がちょうど入れ替わるので、
# 両方を回して両チャネルを見れば描像そのものを判別できる可能性がある。
#   'none'   水氷なし（参照ケース）
#            氷層そのものを置かない。Level 4/5 の .in を ENABLE_ICE=False で
#            回したものに対応する。氷量の指定（f_ice_NN）があっても無視して
#            v_ice = 0 とし、界面も作らない。
#            氷ありの結果と比べる基準線になるので、各レベルで
#            pore / excess / none の 3 通りを回す。
LEVEL4_ICE_MODEL = 'pore'          # 'pore' / 'excess' / 'none'
LEVEL4_ICE_MODELS = ('pore', 'excess', 'none')

# JSON のサブ階層キーと描像の対応。別名を使うならここに足す。
LEVEL4_ICE_MODEL_KEYS = {
    'pore_ice': 'pore', 'pore': 'pore', 'adsorbed': 'pore',
    'excess_ice': 'excess', 'excess': 'excess', 'bulk_ice': 'excess',
    'no_ice': 'none', 'none': 'none', 'dry': 'none', 'ice_off': 'none',
}


# =============================================================================
# Level 5：密度プロファイル（深さ不均質）
# =============================================================================
# Level 4 までは密度を一定（eps_r = 3.0 に対応する rho = 1.753647）としていた。
# Level 5 では Carrier の密度プロファイルを入れ、深さ不均質を導入する。
#
#     rho(z) = 1.92 * (z_cm + 12.2) / (z_cm + 18.0)      [g/cm^3]
#
# 地表 1.301（eps' = 2.27）から深さ 3 m で 1.885（eps' = 3.26）まで連続的に
# 増える。Level 4 の均質モデル（rho = 1.753647）は深さ約 0.49 m に相当して
# いたので、Level 5 では地表付近が Level 4 より薄く、深部が濃くなる。
#
# 【Level 4 との違いで効くところ】
#   * 地表反射が弱くなる  R = -0.268 -> -0.202（-11.4 dB -> -13.9 dB）
#     地表直下の eps' が 3.0 から 2.27 に下がるため。
#   * 走時が短くなる      n が浅部で小さいので、同じ深さでも早く着く。
#   * 減衰が深さ依存      alpha が深いほど大きい。
#   * 氷層の反射係数は深さで変わる（背景 eps' が深さ依存になるため）。
#
# 【.in との対応】Level_5.in は密度プロファイルを薄い層に刻んで実装する。
# 刻み方（LAYER_DN_STEP / LAYER_DZ_MAX_M）で階段の人工反射が決まるので、
# .in の #title に出る「階段化による最大の段反射」を必ず確認すること
# （既定設定で -57.7 dB。氷 10 vol% の界面反射 -32.4 dB より十分小さい）。
LEVEL5_DENSITY_A = 1.92       # rho(z) = A * (z_cm + B) / (z_cm + C)
LEVEL5_DENSITY_B = 12.2
LEVEL5_DENSITY_C = 18.0

# 経路積分の刻み。走時・減衰の深さ積分に使う（.in の層分割とは独立）。
LEVEL5_INTEGRATION_DZ = 0.0025    # [m]


def density_profile(depth_m):
    """深さ [m] -> バルク密度 [g/cm^3]。Carrier et al. 1991, p493。"""
    z_cm = np.asarray(depth_m, dtype=float) * 100.0
    return (LEVEL5_DENSITY_A * (z_cm + LEVEL5_DENSITY_B)
            / (z_cm + LEVEL5_DENSITY_C))


def carrier_eps_real(rho):
    """Carrier 経験式（Fig. 9.53, SOILS）の eps'。eps' = 1.871^rho。"""
    return LEVEL3_CARRIER_EPS_BASE ** np.asarray(rho, dtype=float)


def level5_targets(depth_m, feotio2_wt=None, in_ice=False):
    """深さ z における設計目標値 (eps'_target, eps''_target)。

    帯域の幾何平均 f0 における値。Level 3/4 の level3_targets() の
    深さ依存版にあたる。in_ice=True なら氷を混ぜた値を返す。
    """
    wt = _LEVEL3_ACTIVE_WT if feotio2_wt is None else feotio2_wt
    rho = density_profile(depth_m)
    er = carrier_eps_real(rho)
    ei = er * level3_carrier_tandelta(wt, rho)
    if not in_ice:
        return er, ei
    v = level4_ice_volume_fraction()
    if LEVEL4_ICE_MODEL == 'excess':
        er_w = ((1.0 - v) * er ** (1.0 / 3.0) + v * _L4_EPS_ICE_CBRT) ** 3
        ei_w = (1.0 - v) * ei + v * LEVEL4_EPS_ICE * LEVEL4_TAND_ICE
    else:
        er_w = (er ** (1.0 / 3.0) + v * _L4_ICE_INC) ** 3
        ei_w = ei + v * LEVEL4_EPS_ICE * LEVEL4_TAND_ICE
    return er_w, ei_w


def level5_eps(f, depth_m, in_ice=False, feotio2_wt=None):
    """深さ z・周波数 f における (eps', eps'')。

    f はスカラーか (Nf,)、depth_m はスカラーか (Nz,)。
    両方が配列なら (Nz, Nf) を返す。
    """
    er_t, ei_t = level5_targets(depth_m, feotio2_wt, in_ice)
    er_t = np.atleast_1d(er_t)[:, None]
    ei_t = np.atleast_1d(ei_t)[:, None]
    f_arr = np.atleast_1d(np.asarray(f, dtype=float))[None, :]
    er_d, ei_d = debye_flat_eps(er_t, ei_t, f_arr)
    er = er_d if LEVEL3_EPS_REAL_MODE == 'debye' else np.broadcast_to(
        er_t, er_d.shape).copy()
    ei = ei_d if LEVEL3_EPS_IMAG_MODE == 'debye' else np.broadcast_to(
        ei_t, ei_d.shape).copy()
    return np.squeeze(er), np.squeeze(ei)


def level5_alpha(f, depth_m, in_ice=False, feotio2_wt=None):
    """深さ z・周波数 f における減衰係数 alpha [Np/m]（厳密式）。"""
    er, ei = level5_eps(f, depth_m, in_ice, feotio2_wt)
    td = ei / er
    w_over_c = 2.0 * np.pi * np.asarray(f, dtype=float) / (C * 1e9)
    with np.errstate(invalid='ignore'):
        a = w_over_c * np.sqrt(er / 2.0) * np.sqrt(np.sqrt(1.0 + td ** 2) - 1.0)
    return np.nan_to_num(a, nan=0.0)


def level5_index(f, depth_m, in_ice=False, feotio2_wt=None):
    """深さ z・周波数 f における屈折率 n = sqrt(eps')。"""
    return np.sqrt(level5_eps(f, depth_m, in_ice, feotio2_wt)[0])


def level5_in_ice(depth_m):
    """その深さが氷層の中かどうか。氷なし（'none'）なら常に False。"""
    z0 = np.asarray(depth_m, dtype=float)
    if LEVEL4_ICE_MODEL == 'none' or LEVEL4_ICE_THICK_M <= 0.0:
        return np.zeros(z0.shape, dtype=bool)
    top = float(LEVEL4_ICE_TOP_M)
    bot = top + float(LEVEL4_ICE_THICK_M)
    z = np.asarray(depth_m, dtype=float)
    return (z >= top) & (z < bot)


def level5_path_integrals(f, depth_m, with_ice=True, feotio2_wt=None,
                          dz=None):
    """地表から深さ d までの片道の経路積分を返す。

        att(f) = ∫ alpha(f, z) dz        [Np]（片道）
        opt(f) = ∫ n(f, z) dz            [m]（光学的距離。走時は opt/c）

    深さ方向に n も alpha も変わるので、閉形式では書けず数値積分になる。
    刻みは LEVEL5_INTEGRATION_DZ（既定 2.5 mm = セル幅）。
    往復にするときは呼び出し側で 2 倍すること。
    """
    d = float(depth_m)
    if d <= 0.0:
        z0 = np.zeros(1)
        return (np.zeros_like(np.atleast_1d(np.asarray(f, dtype=float))),
                np.zeros_like(np.atleast_1d(np.asarray(f, dtype=float))))
    step = LEVEL5_INTEGRATION_DZ if dz is None else float(dz)
    nz = max(2, int(np.ceil(d / step)))
    edges = np.linspace(0.0, d, nz + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    w = np.diff(edges)
    ice = level5_in_ice(mid) if with_ice else np.zeros(len(mid), dtype=bool)

    f_arr = np.atleast_1d(np.asarray(f, dtype=float))
    att = np.zeros_like(f_arr)
    opt = np.zeros_like(f_arr)
    for flag in (False, True):
        sel = ice == flag
        if not np.any(sel):
            continue
        nz_sel = int(np.count_nonzero(sel))
        shp = (nz_sel, f_arr.size)
        # level5_eps は squeeze して返すので、ここで (Nz, Nf) に戻す
        a = np.asarray(level5_alpha(f_arr, mid[sel], flag, feotio2_wt)
                       ).reshape(shp)
        n = np.asarray(level5_index(f_arr, mid[sel], flag, feotio2_wt)
                       ).reshape(shp)
        att = att + np.sum(a * w[sel][:, None], axis=0)
        opt = opt + np.sum(n * w[sel][:, None], axis=0)
    return att, opt


def describe_level5_medium(feotio2_wt=None):
    """Level 5 の密度プロファイル設定を人が読める形で返す。"""
    zs = (0.0, 0.5, 1.0, 2.0, 3.0)
    parts = []
    for z in zs:
        er, ei = level5_targets(z, feotio2_wt, False)
        parts.append('z={:.1f}m rho={:.4f} eps={:.4f} tand={:.6f}'.format(
            z, float(density_profile(z)), float(er), float(ei / er)))
    n_s = float(np.sqrt(level5_targets(0.0, feotio2_wt, False)[0]))
    return ('Carrier density profile rho(z)={}*(z+{})/(z+{}) [z:cm]; '
            'surface reflection R = {:.4f} ({:.1f} dB); '.format(
                LEVEL5_DENSITY_A, LEVEL5_DENSITY_B, LEVEL5_DENSITY_C,
                (1 - n_s) / (1 + n_s),
                20 * np.log10(abs((1 - n_s) / (1 + n_s))))
            + ' | '.join(parts))

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

LEVEL4_ICE_KEYS = {}      # 例: {'f_ice_lowest': 0.5}
_LEVEL3_ACTIVE_WT = LEVEL3_COMPOSITIONS[LEVEL3_DEFAULT_COMPOSITION]
_LEVEL3_ACTIVE_KEY = LEVEL3_DEFAULT_COMPOSITION
_LEVEL4_ACTIVE_ICE_KEY = None
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


# -----------------------------------------------------------------------------
# 水氷の描像の自動判定（組成・氷量の仕組みと同じ構造）
# -----------------------------------------------------------------------------
def _parse_ice_model_key(key):
    """キー名から氷の描像を読み取る。読めなければ None。"""
    return LEVEL4_ICE_MODEL_KEYS.get(str(key).strip().lower())


def _is_ice_model_layer(keys):
    """その階層が氷の描像の階層かどうかを判定する（ラベル表示用）。"""
    return bool(keys) and all(_parse_ice_model_key(k) is not None for k in keys)


def level4_ice_model_from_kind(kind):
    """選択されたサブ階層キーから氷の描像を取り出す。

    kind は load_paths が返す ' / ' 区切りのキー列
    （例 'excitation_waveform / dx_00025 / FeO_075 / pore_ice / f_ice_10'）。

    描像キーが 1 つも見つからない場合は、既定値に落とさずエラーにする。
    黙って既定値を使うと、誤った描像の理論曲線のまま解析が完走してしまう
    ため（組成・氷量の判定と同じ方針）。
    """
    if kind:
        for token in str(kind).replace('/', ' ').split():
            m = _parse_ice_model_key(token)
            if m is not None:
                return m, token
    raise CmdInputError(
        '水氷の描像を JSON のキーから判定できませんでした（選択: {}）。\n'
        'キー名を pore_ice / excess_ice にするか、LEVEL4_ICE_MODEL_KEYS に'
        '追加してください。\n'
        '（既定値に落とすと誤った描像で解析が完走してしまうため、'
        'あえてエラーにしています）'.format(kind))


_LEVEL4_ACTIVE_MODEL_KEY = None


def set_level4_ice_model(kind):
    """JSON の選択結果から氷の描像を設定する（main から呼ぶ）。"""
    global LEVEL4_ICE_MODEL, _LEVEL4_ACTIVE_MODEL_KEY
    LEVEL4_ICE_MODEL, _LEVEL4_ACTIVE_MODEL_KEY = level4_ice_model_from_kind(kind)
    return LEVEL4_ICE_MODEL, _LEVEL4_ACTIVE_MODEL_KEY


def configure_from_kind(kind, level):
    """組成・氷量・氷の描像をまとめて設定する。解析コードはこれ 1 つを呼ぶ。

    レベルに氷層がなければ氷まわりは設定しない（Level 1-3 でも呼べる）。
    戻り値は表示用の説明文のリスト。
    """
    notes = []
    effects = LEVEL_EFFECTS.get(level, [])
    if 'absorb_tandelta' in effects:
        wt, key = set_level3_composition(kind)
        notes.append('背景レゴリスの組成: FeO+TiO2 = {:.1f} wt%  [{}]'
                     .format(wt, key))
    if 'density_profile' in effects:
        notes.append('密度プロファイル: {}'.format(describe_level5_medium()))
    if 'ice_layer' in effects:
        model, mkey = set_level4_ice_model(kind)
        # 氷なしなら f_ice_NN キーが無くてもよい
        try:
            vol, vkey = set_level4_ice(kind)
        except CmdInputError:
            if model != 'none':
                raise
            vol, vkey = 0.0, '(none)'
        if model == 'none':
            notes.append('水氷: なし（参照ケース）  [{}]'.format(mkey))
        else:
            notes.append('水氷の描像: {}  [{}]'.format(model, mkey))
            notes.append('水氷濃度: {:.2f} vol% = {:.3f} wt%  [{}]'
                         .format(vol, 100 * level4_ice_weight_fraction(), vkey))
    return notes

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

def level4_porosity():
    """乾燥レゴリスの空隙率。pore 描像での氷量の上限になる。"""
    return 1.0 - LEVEL3_RHO / LEVEL4_RHO_GRAIN


def level4_ice_weight_fraction(v_ice=None):
    """体積分率 -> 質量分率。かさ密度の作られ方が描像で違う。

        'pore'   : かさ密度 = rho + v*rho_ice        （粒子はそのまま残る）
        'excess' : かさ密度 = (1-v)*rho + v*rho_ice  （レゴリスごと置き換わる）
    """
    v = level4_ice_volume_fraction() if v_ice is None else float(v_ice)
    m_ice = v * LEVEL4_RHO_ICE
    m_reg = ((1.0 - v) * LEVEL3_RHO if LEVEL4_ICE_MODEL == 'excess'
             else LEVEL3_RHO)
    return m_ice / (m_reg + m_ice)


def level4_ice_volume_fraction():
    """氷の体積分率（0-1 の分率）。入力はパーセントで受け取る。

    wt% -> vol% の換算式は描像で変わる（かさ密度の作られ方が違うため）。
        'pore'   : v = (rho/rho_ice) * w/(1-w)          Takekura+2025 Eq.(8)
        'excess' : v = w*rho / (rho_ice*(1-w) + w*rho)
    上限も描像で変わる。pore は空隙を埋めきったら終わりだが、excess は
    レゴリスごと置き換わるので空隙率は上限にならない。
    """
    if LEVEL4_ICE_MODEL not in LEVEL4_ICE_MODELS:
        raise CmdInputError(
            "LEVEL4_ICE_MODEL は {} のいずれか".format(LEVEL4_ICE_MODELS))
    if LEVEL4_ICE_MODEL == 'none':
        # 氷なしの参照ケース。JSON に f_ice_NN があっても無視する。
        return 0.0
    if LEVEL4_ICE_SPEC == 'vol':
        v = LEVEL4_ICE_VOL_PCT / 100.0
    elif LEVEL4_ICE_SPEC == 'wt':
        w = LEVEL4_ICE_WT_PCT / 100.0
        if LEVEL4_ICE_MODEL == 'excess':
            v = w * LEVEL3_RHO / (LEVEL4_RHO_ICE * (1.0 - w) + w * LEVEL3_RHO)
        else:
            v = (LEVEL3_RHO / LEVEL4_RHO_ICE) * w / (1.0 - w)
    else:
        raise CmdInputError("LEVEL4_ICE_SPEC は 'wt' か 'vol'")
    if not 0.0 <= v < 1.0:
        raise CmdInputError('氷の体積分率が範囲外です: {:.4f}'.format(v))
    if LEVEL4_ICE_MODEL == 'pore' and v > level4_porosity():
        raise CmdInputError(
            '氷の体積分率 {:.4f} が空隙率 {:.4f} を超えています。'
            "空隙充填では不可能な量です（excess 描像なら可能）"
            .format(v, level4_porosity()))
    return v

def level4_targets(feotio2_wt=None):
    """(背景の目標値, 氷層の目標値) をそれぞれ (eps', eps'') の組で返す。

    Level_4.in の mix_ice() と同じ順序（目標値を混合してから Debye 化）。
    """
    er_dry, ei_dry = level3_targets(feotio2_wt)
    v = level4_ice_volume_fraction()
    if LEVEL4_ICE_MODEL == 'excess':
        # レゴリス全体を置き換えるので eps^(1/3) 上の線形内挿になる。
        # 粒子が (1-v) 倍に減るので eps'' も同じ割合で希釈される。
        er_ice = ((1.0 - v) * er_dry ** (1.0 / 3.0)
                  + v * _L4_EPS_ICE_CBRT) ** 3
        ei_ice = (1.0 - v) * ei_dry + v * LEVEL4_EPS_ICE * LEVEL4_TAND_ICE
    else:
        # 真空を置き換えるので増分形。粒子は減らないので eps'' は保存。
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

def describe_level4_medium():
    """Level 4 の氷層設定を人が読める形で返す（ログと run_info 用）。"""
    v = level4_ice_volume_fraction()
    (er_d, ei_d), (er_i, ei_i) = level4_targets()
    wt_pct = 100.0 * level4_ice_weight_fraction(v)
    n0, n1 = np.sqrt(er_d), np.sqrt(er_i)
    R = (n0 - n1) / (n0 + n1)
    a_d = float(level4_alpha(np.array([BAND_CENTRE_HZ]), False)[0])
    a_i = float(level4_alpha(np.array([BAND_CENTRE_HZ]), True)[0])
    key_note = ('' if _LEVEL4_ACTIVE_ICE_KEY is None
                else ' [{}]'.format(_LEVEL4_ACTIVE_ICE_KEY))
    if LEVEL4_ICE_MODEL == 'none':
        return 'no ice (reference case: ice layer not present)'
    model_note = ('pore-filling (adsorbed water)' if LEVEL4_ICE_MODEL == 'pore'
                  else 'bulk replacement (excess ice)')
    return ('ice layer {:.3f} vol%{} ({:.3f} wt%) at {:.2f}-{:.2f} m, '
            'LLL mixing, {} model, '
            "eps' {:.6f} -> {:.6f} ({:+.2f}%), eps'' {:.6f} -> {:.6f} ({:+.3f}%), "
            'alpha {:+.2f}% @{:.2f} GHz, interface R = {:.1f} dB'
            .format(100.0 * v, key_note, wt_pct,
                    LEVEL4_ICE_TOP_M,
                    LEVEL4_ICE_TOP_M + LEVEL4_ICE_THICK_M,
                    model_note,
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
# 自己検証
# =============================================================================
def check_against_ascan_spectrum(verbose=True):
    """ascan_spectrum.py の媒質モデルと数値が一致するかを確認する。

    現時点では ascan_spectrum.py も同じモデルを内部に持っているため、
    両者がずれていないことを確認できるようにしておく。
    将来 ascan_spectrum.py をこのモジュールの import に切り替えたら不要になる。
    描像は 'pore' でのみ比較する（ascan_spectrum は pore しか持たないため）。
    """
    import ascan_spectrum as _asp
    f = np.linspace(0.4e9, 2.2e9, 401)
    keep = LEVEL4_ICE_MODEL
    globals()['LEVEL4_ICE_MODEL'] = 'pore'
    set_level3_composition('FeO_075'); _asp.set_level3_composition('FeO_075')
    set_level4_ice('f_ice_10');        _asp.set_level4_ice('f_ice_10')
    n_reg = np.full_like(f, np.sqrt(3.0))
    checks = [
        ("level3_eps'", level3_eps(f)[0], _asp.level3_eps(f)[0]),
        ('level3_eps"', level3_eps(f)[1], _asp.level3_eps(f)[1]),
        ('level3_alpha', level3_alpha(f), _asp.level3_alpha(f)),
        ('level3_group_index', level3_group_index(f), _asp.level3_group_index(f)),
        ('level4_eps(ice)', level4_eps(f, True)[0], _asp.level4_eps(f, True)[0]),
        ('level4_alpha(ice)', level4_alpha(f, True), _asp.level4_alpha(f, True)),
        ('level2_alpha', level2_alpha(f, n_reg), _asp.level2_alpha(f, n_reg)),
        ('level3b_eps', level3b_eps(f)[0], _asp.level3b_eps(f)[0]),
        ('level3b_alpha', level3b_alpha(f), _asp.level3b_alpha(f)),
    ]
    for lv in IMPLEMENTED_LEVELS:
        checks.append(('refractive_index ' + lv,
                       refractive_index(f, lv), _asp.refractive_index(f, lv)))
    worst = 0.0
    for name, a, b in checks:
        d = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        worst = max(worst, d)
        if verbose:
            print('  {:24s} 最大差 {:.3e}'.format(name, d))
    globals()['LEVEL4_ICE_MODEL'] = keep
    if verbose:
        print('  -> {} (最大差 {:.3e})'.format(
            '一致' if worst < 1e-12 else '** 不一致 **', worst))
    return worst < 1e-12


if __name__ == '__main__':
    print('subsurface_model 自己検証')
    check_against_ascan_spectrum()
    print()
    for _m in LEVEL4_ICE_MODELS:
        globals()['LEVEL4_ICE_MODEL'] = _m
        print(' ', describe_level4_medium())


# =============================================================================
# 経路積分（Level 3/4/5 を統一して扱う）
# =============================================================================
# Level 5 では n も alpha も深さで変わるので、伝達関数を閉形式で書けない。
# そこで「深さ 0 から d までの経路積分」を返す関数をここに集約し、
# 各解析コードはそれを呼ぶだけにする。
#
# Level 3（均質）と Level 4（均質＋氷層）は、密度が一定な特殊ケースとして
# 同じ関数で扱える。実際 Level 4 の結果は従来の閉形式と一致する。
#
# --- 見かけ源距離（幾何減衰）------------------------------------------------
# 界面ごとの近軸屈折則 r -> r*(n_new/n_old) を連続極限に取ると
#     dr/dz = 1 + r * d(ln n)/dz
# となり、解は
#     片道: r_eff(d) = n(d) * [ h + ∫_0^d dz/n(z) ]
#     往復: r_eff(d) = 2h + 2 ∫_0^d dz/n(z)
# 均質なら順に n*h + d、2h + 2d/n となり、これまでの式と一致する。
# =============================================================================

def has_density_profile(level):
    return 'density_profile' in LEVEL_EFFECTS.get(level, [])


def has_ice_layer(level):
    return 'ice_layer' in LEVEL_EFFECTS.get(level, [])


def eps_at_depth(f, depth_m, level, feotio2_wt=None):
    """レベルに応じた深さ z・周波数 f の (eps', eps'')。

    depth_m がスカラーなら (Nf,)、配列なら (Nz, Nf) を返す。
    氷層の内外、密度プロファイルの有無をここで吸収する。
    """
    z = np.atleast_1d(np.asarray(depth_m, dtype=float))
    f_arr = np.atleast_1d(np.asarray(f, dtype=float))
    in_ice = (level5_in_ice(z) if has_ice_layer(level)
              else np.zeros(z.shape, dtype=bool))
    er = np.empty((z.size, f_arr.size))
    ei = np.empty_like(er)
    for flag in (False, True):
        sel = in_ice == flag
        if not np.any(sel):
            continue
        if has_density_profile(level):
            a, b = level5_eps(f_arr, z[sel], flag, feotio2_wt)
        elif has_ice_layer(level):
            a, b = level4_eps(f_arr, flag, feotio2_wt)
            a = np.broadcast_to(a, (int(sel.sum()), f_arr.size))
            b = np.broadcast_to(b, (int(sel.sum()), f_arr.size))
        else:
            n = refractive_index(f_arr, level)
            a = np.broadcast_to(n ** 2, (int(sel.sum()), f_arr.size))
            td = (level2_tandelta(f_arr, n)
                  if 'absorb_const' in LEVEL_EFFECTS[level]
                  else np.zeros_like(f_arr))
            b = a * td
        er[sel] = np.asarray(a).reshape(int(sel.sum()), f_arr.size)
        ei[sel] = np.asarray(b).reshape(int(sel.sum()), f_arr.size)
    return (er[0], ei[0]) if np.ndim(depth_m) == 0 else (er, ei)


def alpha_at_depth(f, depth_m, level, feotio2_wt=None):
    """深さ z・周波数 f の減衰係数 alpha [Np/m]（厳密式）。"""
    er, ei = eps_at_depth(f, depth_m, level, feotio2_wt)
    td = ei / er
    w_over_c = 2.0 * np.pi * np.atleast_1d(np.asarray(f, dtype=float)) / (C * 1e9)
    with np.errstate(invalid='ignore'):
        a = w_over_c * np.sqrt(er / 2.0) * np.sqrt(np.sqrt(1.0 + td ** 2) - 1.0)
    return np.nan_to_num(a, nan=0.0)


def path_integrals(f, depth_m, level, feotio2_wt=None, dz=None, group=False):
    """深さ 0 から d までの片道の経路積分を返す。

        att(f) = ∫ alpha dz        [Np]   吸収
        opt(f) = ∫ n dz            [m]    光学的距離（走時は opt/c）
        inv(f) = ∫ dz/n            [m]    見かけ源距離に使う

    group=True なら opt に群屈折率を使う（包絡ピークの走時に対応）。
    刻みは既定 LEVEL5_INTEGRATION_DZ。氷層と密度プロファイルの境界は
    自動で刻みに含まれる（境界をまたぐ区間ができないよう分割する）。
    """
    f_arr = np.atleast_1d(np.asarray(f, dtype=float))
    d = float(depth_m)
    zero = np.zeros_like(f_arr)
    if d <= 0.0:
        return zero, zero.copy(), zero.copy()

    step = LEVEL5_INTEGRATION_DZ if dz is None else float(dz)
    if not has_density_profile(level):
        # 均質なら層の境界だけで十分（積分は解析的に厳密）
        edges = sorted({0.0, d} | {b for b in interface_depths(level)
                                   if 0.0 < b < d})
    else:
        edges = [0.0]
        for b in sorted(b for b in interface_depths(level) if 0.0 < b < d):
            edges.append(b)
        edges.append(d)
        fine = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = max(1, int(np.ceil((b - a) / step)))
            fine.extend(np.linspace(a, b, m + 1)[:-1])
        fine.append(d)
        edges = fine
    edges = np.asarray(edges, dtype=float)
    mid = 0.5 * (edges[:-1] + edges[1:])
    w = np.diff(edges)

    alpha = np.asarray(alpha_at_depth(f_arr, mid, level, feotio2_wt)
                       ).reshape(mid.size, f_arr.size)
    if group:
        n = np.empty((mid.size, f_arr.size))
        h = max(1e5, 1e-3 * float(np.min(np.abs(f_arr)) or 1e8))
        e0 = np.asarray(eps_at_depth(f_arr, mid, level, feotio2_wt)[0]
                        ).reshape(mid.size, f_arr.size)
        ep = np.asarray(eps_at_depth(f_arr + h, mid, level, feotio2_wt)[0]
                        ).reshape(mid.size, f_arr.size)
        em = np.asarray(eps_at_depth(f_arr - h, mid, level, feotio2_wt)[0]
                        ).reshape(mid.size, f_arr.size)
        n = np.sqrt(e0) + f_arr[None, :] * (np.sqrt(ep) - np.sqrt(em)) / (2 * h)
    else:
        n = np.sqrt(np.asarray(eps_at_depth(f_arr, mid, level, feotio2_wt)[0]
                               ).reshape(mid.size, f_arr.size))
    n_ph = np.sqrt(np.asarray(eps_at_depth(f_arr, mid, level, feotio2_wt)[0]
                              ).reshape(mid.size, f_arr.size))
    att = np.sum(alpha * w[:, None], axis=0)
    opt = np.sum(n * w[:, None], axis=0)
    inv = np.sum(w[:, None] / n_ph, axis=0)
    return att, opt, inv


def interface_depths(level):
    """レベルが持つ地下界面の深さ [m]（地表は含まない）。

    氷なし（LEVEL4_ICE_MODEL='none'）なら地下界面は存在しない。
    """
    if LEVEL4_ICE_MODEL == 'none':
        return []
    if has_ice_layer(level):
        top = float(LEVEL4_ICE_TOP_M)
        return [top, top + float(LEVEL4_ICE_THICK_M)]
    return []


def surface_index(f, level, feotio2_wt=None):
    """地表直下の屈折率。地表透過係数と幾何項の起点に使う。"""
    er, _ = eps_at_depth(f, 0.0, level, feotio2_wt)
    return np.sqrt(er)


def r_eff_oneway(f, depth_m, level, feotio2_wt=None):
    """片道の見かけ源距離 r_eff = n(d) * [ h + ∫dz/n ]。"""
    _, _, inv = path_integrals(f, depth_m, level, feotio2_wt)
    er, _ = eps_at_depth(f, float(depth_m), level, feotio2_wt)
    return np.sqrt(er) * (TX_HEIGHT + inv)


def r_eff_roundtrip(f, depth_m, level, feotio2_wt=None):
    """往復の見かけ源距離 r_eff = 2h + 2∫dz/n。"""
    _, _, inv = path_integrals(f, depth_m, level, feotio2_wt)
    return 2.0 * TX_HEIGHT + 2.0 * inv


def interface_reflection(f, depth_m, level, feotio2_wt=None, eps=1e-6):
    """深さ d の界面の反射係数 R = (n_above - n_below)/(n_above + n_below)。"""
    er_a, _ = eps_at_depth(f, float(depth_m) - eps, level, feotio2_wt)
    er_b, _ = eps_at_depth(f, float(depth_m) + eps, level, feotio2_wt)
    na, nb = np.sqrt(er_a), np.sqrt(er_b)
    return (na - nb) / (na + nb)


def transmission_product(f, depth_m, level, feotio2_wt=None, two_way=False,
                         include_surface=True, eps=1e-6):
    """深さ d までに横切る界面の透過係数の積。

    Ez は界面に接線なので片道 T = 2 n_above/(n_above + n_below)。
    two_way=True なら往復ぶん（下向き×上向き）を掛ける。
    """
    f_arr = np.atleast_1d(np.asarray(f, dtype=float))
    T = np.ones_like(f_arr)
    n_prev = np.ones_like(f_arr)
    if include_surface:
        n_s = surface_index(f_arr, level, feotio2_wt)
        T = T * (2.0 * n_prev / (n_prev + n_s))
        if two_way:
            T = T * (2.0 * n_s / (n_prev + n_s))
        n_prev = n_s
    for b in interface_depths(level):
        if b >= float(depth_m) - eps:
            continue
        er_a, _ = eps_at_depth(f_arr, b - eps, level, feotio2_wt)
        er_b, _ = eps_at_depth(f_arr, b + eps, level, feotio2_wt)
        na, nb = np.sqrt(er_a), np.sqrt(er_b)
        T = T * (2.0 * na / (na + nb))
        if two_way:
            T = T * (2.0 * nb / (na + nb))
    return T


# =============================================================================
# 氷なし理論（検出性能を見るための対照）
# =============================================================================
# 実測では氷の有無が未知なので、解析者はまず「氷がないと仮定した理論」を
# 当てはめる。そこで出る残差がそのまま検出信号になる。
#
# 【レベルは変えないこと】Level 5 の氷なし理論は Level 3 ではない。
# Level 5 は経験式で深さ方向の eps'/tan_delta 変化を入れているので、
# それを残したまま氷だけを取り除く必要がある。そこでレベルは据え置き、
# LEVEL4_ICE_MODEL を一時的に 'none' にする。
#     Level 4 の氷なし -> 均質背景（結果として Level 3 と同じ）
#     Level 5 の氷なし -> 密度プロファイルは残り、氷層だけが消える
# =============================================================================

def ice_is_present():
    """いま氷ありの設定になっているか（氷なしケースの解析では False）。"""
    return LEVEL4_ICE_MODEL != 'none' and LEVEL4_ICE_VOL_PCT > 0.0


@contextlib.contextmanager
def no_ice_theory():
    """氷なし理論を一時的に使う。レベルと密度プロファイルはそのまま。"""
    global LEVEL4_ICE_MODEL
    keep = LEVEL4_ICE_MODEL
    LEVEL4_ICE_MODEL = 'none'
    try:
        yield
    finally:
        LEVEL4_ICE_MODEL = keep


def interface_depths_for_plot():
    """作図で氷層の帯を描くための [上面, 下面]。氷なしなら空リスト。

    LEVEL4_ICE_MODEL が 'none' でも、設定上の氷層位置を返したい場面がある
    （氷なし理論の文脈で呼ばれても帯を描けるようにするため）。
    """
    if not has_ice_layer('Level_4') or LEVEL4_ICE_THICK_M <= 0:
        return []
    top = float(LEVEL4_ICE_TOP_M)
    return [top, top + float(LEVEL4_ICE_THICK_M)]