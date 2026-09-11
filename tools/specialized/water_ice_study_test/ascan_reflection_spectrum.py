"""at_tx（地表 tx/rx、モノスタティック）の A-scan スペクトル解析

ascan_spectrum.py（埋設 rx・片道透過）の反射版。要件は
README_ascan_reflection_spectrum.md を参照。

--- 本ツール固有なもの ------------------------------------------------------
  1. 反射の伝達関数（往復経路、反射係数、往復透過）
  2. 反射イベントの定義と時間ゲート
それ以外（JSON の読み取り、ネスト階層の選択、参照計算の校正、組成・水氷濃度
の自動判定、スペクトルのモーメントと LSR の規約、作図の体裁）はすべて
ascan_spectrum.py から import して使う。定数はコピーしない。

--- 走査軸の違い ------------------------------------------------------------
  ascan_spectrum.py : 1 深さ 1 トレース。深さ方向にイベントが並ぶ
  本ツール          : 1 トレースに複数イベント。時間方向に反射が並ぶ
したがって図の色分けは「rx 深さ」ではなく「反射イベント」になる。
"""

import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- 地下構造モデル（.in が定める物理）--------------------------------------
# レベルを増やすときに触るのは原則 subsurface_model.py だけ。
import subsurface_model as sm
from subsurface_model import (
    C, TX_HEIGHT, R_REF, BAND_GHZ, BAND_CENTRE_HZ,
    LEVEL_EFFECTS, IMPLEMENTED_LEVELS,
    LEVEL4_ICE_TOP_M, LEVEL4_ICE_THICK_M,
    configure_from_kind, _is_ice_model_layer,
    path_integrals, r_eff_roundtrip, transmission_product,
    interface_reflection, surface_index, eps_at_depth,
    has_density_profile, has_ice_layer, describe_level5_medium,
    describe_level2_medium, describe_level3_medium, describe_level3b_medium,
    describe_level4_medium,
    refractive_index, level4_eps, level4_alpha, level3_eps, level3_alpha,
    level2_alpha, _group_index_from_eps,
)

# --- 解析の手順（JSON の読み込み・スペクトル量・作図）-----------------------
import ascan_spectrum as asp
from ascan_spectrum import (
    FLOHI_PRIMARY_DB,
    load_paths, check_paths_exist, load_trace, spectrum, measure_peak,
    _interp_complex_to_grid, save_figure,
    moments, lo_hi_freq, log_spectral_ratio, alpha_to_tandelta,
    valid_mask, noise_floor, group_delay,
)
from gprMax.exceptions import CmdInputError

# 階層選択で「水氷の描像」のラベルを出せるようにする（ascan_spectrum のフック）。
if not any(lab == '水氷の描像' for _, lab in asp._EXTRA_LAYER_LABELS):
    asp._EXTRA_LAYER_LABELS.append((_is_ice_model_layer, '水氷の描像'))


# --- 氷なし理論との比較について ----------------------------------------------
# at_tx（反射法）では、氷なし理論に氷層のイベントがそもそも存在しない。
# したがって「残差」ではなく「氷なしモデルが予測しない到達があるか」という
# 形の検出になり、これは既存のノイズフロア判定（judge_snr）がそのまま該当する。
# そのため埋設 rx 側のような氷あり/なし比較図は用意していない。
# 氷なしのデータ（no_ice）を同じ設定で解析すれば、その参照線が得られる。

# =============================================================================
# 設定  [EDIT HERE]
# =============================================================================
JSON_PATH = asp.JSON_PATH          # 既存ツールと同じ JSON を使う
AT_TX_KEY = 'at_tx'                # 解析対象の rx キー
REF_KEY = 'far_1m'                 # 参照計算の rx キー

OUTPUT_SUBDIRNAME = 'ascan_reflection_spectrum'

# --- 時間ゲート（本ツールで最も結果に効くパラメータ。README §5）--------------
GATE_HALFWIDTH_NS = 2.0            # 半幅。パルス幅 約 0.7 ns、イベント間隔 11.6 ns
GATE_TAPER = 0.2                   # Tukey。既存コードと同じ
GATE_CENTER = 'theory'             # 'theory'   … 理論トレースの包絡ピークを中心にする
                                   #              （波源遅延を自動で含む）
                                   # 'measured' … さらに実測の窓内ピークに合わせ直す
GATE_SWEEP_NS = []                 # 例 [1.0, 2.0, 3.0]。空なら感度確認をしない

# --- ノイズフロア（README §4.3。修正 6 で全面的に見直した）------------------
# 【なぜ RMS をやめたか】
# 励振波形は平坦帯域＋Tukey テーパなので時間領域では sinc に近く、サイドローブが
# 長く尾を引く。実測では surface が 2-12 ns、ice_top が 14-24 ns、ice_bottom が
# 27-36 ns に広がっており、8-40 ns のほぼ全域がどれかのイベントのローブで埋まる。
#
# 【旧方式を削除した理由】以前は「窓内の包絡の下側 10% 分位点を、Rayleigh 分布を
# 仮定して RMS に換算したもの」をノイズフロアと呼んでいた。これは二重に誤って
# いた。
#   (1) FDTD には熱雑音が無い。あの水準を作っているのは地表反射のサイドローブと、
#       密度プロファイルを階段で近似したことによる段差反射で、どちらも数値誤差
#       ではなく「水氷に由来しない信号（クラッター）」である。実測 -81.2 dB は
#       深さ 1 m での段差反射の理論値 -81.5 dB とほぼ一致する。
#   (2) 換算係数 sqrt(-2 ln(1-p)) は包絡が Rayleigh 分布に従うとき、つまり下地が
#       ガウス雑音のときにしか成り立たない。決定論的なパルスの重なりでは包絡の
#       干渉零点を拾ってしまい、数値実験では換算が 30 dB ずれた。
#
# 【新方式】比較相手を「氷なし計算の同じ時間窓」にする。検出とは「氷を入れた
# ことで、入れなかった場合には無かった信号が現れた」ことなので、これが定義
# そのものになる。分位点も分布の仮定も換算係数も要らない。
#
#   検出 SNR = 20 log10( 氷ありの包絡ピーク / 氷なしの包絡ピーク )   ※同じゲート内
#
# 包絡は両方に同じ Hilbert 変換を使うので、分布の仮定は一切入らない。ピーク
# どうしの比較なので peak-to-RMS 換算も不要。深さごとに基準が変わるのは物理的に
# 正しい（段差反射は深さ 1 m で -81.5 dB、2 m で -92.1 dB と 10 dB 以上違う）。
NOICE_TRACE_PATH = ''              # 空でなければレベルによらずこれを使う
NOICE_TRACE_PATHS = {
    # Level 4 の氷なしは Level 3（同じ均質媒質で氷層だけが無い）
    'Level_4': '/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_3x4/water_ice_study_test'
               '/level_3/FeO075/at_tx/result/Ascan.out',
    # Level 5 以降は Level 5 の no_ice
    '_default': '/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_3x4/water_ice_study_test'
                '/Level_5/no_ice/at_tx/result/Ascan.out',
}
PLOT_WITHOUT_NOICE = True          # 重ね書きしない版の図も出すか

# --- 検出判定（修正 6）------------------------------------------------------
# 旧版はフロアの絶対値（-70/-60/-55 dB）で判定していたが、これは「地表反射比」
# という基準量に依存するため、直達波を差し引くと基準が 22 dB 変わって判定が
# ひっくり返った。判定は基準量に依らない SNR で行う。
SNR_DETECT_DB = 12.0               # これ以上なら検出可
SNR_MARGINAL_DB = 6.0              # これ以上なら限界

# --- 相対 LSR の基準イベント -------------------------------------------------
# 既定は最も浅い地下界面。地表反射を基準にすると地表の往復透過が残るため、
# 実測可能性の観点では地下界面どうしで取るほうが素直。
REL_LSR_REF_EVENT = 'surface'

# --- 直達波の除去（修正 2）---------------------------------------------------
# at_tx では rx が tx と同位置にあるため、直達波（波源の近傍場）が記録される。
# これは地表反射より 22 dB 大きく、しかも帯域制限波形は時間的に長い裾を引くため、
# 地表反射（直達波の 2.3 ns 後）を分離できず、ノイズフロアも押し上げる。
#
# 自由空間（地面なし）の at_tx を引けば直達波だけが消える。実機でもアンテナの
# 直達結合は事前較正できるので、氷なしトレースの差分と違ってこちらは
# 現実的に使える手法である。
SUBTRACT_FREESPACE = True          # 直達波の除去を行うか（メインは True）
FREESPACE_AT_TX_PATH = ''          # 空なら JSON の _reference / at_tx を使う

# 差分なしの結果もサブディレクトリに出力する。元データの姿を失わないため。
NO_SUBTRACTION_SUBDIR = 'no_subtraction'

# --- 背景差分（README §4.4）-------------------------------------------------
# 氷なしトレースの差分。実機では使えない（氷なしの観測が得られない）ので、
# 理論上限としてのみ扱う。SUBTRACT_FREESPACE とは別物。
BACKGROUND_TRACE_PATH = ''         # 氷なし at_tx の .out。空なら差分しない

# --- 修正 5：反射係数スペクトル（fig5）の位置づけ ----------------------------
# スカラーの反射係数 R は A-scan の振幅解析（ascan_reflection.py）の担当である。
# ここで R(f) を描くのは「帯域内で平坦になるはず」という性質を使った診断のため。
# 平坦でなければ、ゲート・吸収モデル・幾何項のいずれかが誤っていることになる。
PLOT_REFLECTION_SPECTRUM = False   # 既定 False。診断したいときだけ True

# --- fig6: 時間-周波数マップ（狙い撃ちをしない表示）--------------------------
# 【なぜ追加したか】
# fig1 は surface / ice_top / ice_bottom の理論走時をあらかじめ計算し、その時刻
# だけをゲートしてスペクトルを取っている。順方向モデルの検証には最短の方法だが、
# 「どこに何が来るか」を先に知っている前提なので、実際の観測ではできない。
# そこで、同じゲートを時間方向に一定間隔で滑らせ、全時刻のスペクトルを 2 次元
# マップとして描く。イベントの位置を仮定せずに重心周波数とスペクトル幅の
# 時間変化が読めるので、fig1 の「狙い撃ち」の結果が生データのどこに対応して
# いるのかを確認できる。fig1 は削らず、両方を出す。
#
# 【分解能のトレードオフ】窓長 2*halfwidth に対して周波数分解能は
# 約 1/(2*halfwidth) になる。既定の 2.0 ns なら 0.25 GHz で、帯域 1.5 GHz を
# 6 点でしか刻めない。重心とスペクトル幅は積分量なのでこの分解能でも意味を
# 持つが、スペクトルの細かい構造を読む用途には向かない。
PLOT_SPECTROGRAM = True
SPECTROGRAM_HALFWIDTH_NS = None    # None なら GATE_HALFWIDTH_NS と同じ（fig1 と揃う）
SPECTROGRAM_STEP_NS = 0.10         # ゲート中心を進める刻み
SPECTROGRAM_TIME_RANGE_NS = None   # None なら全時間範囲（トレース全長）。
                                   # 【なぜ全長を既定にするか】以前は
                                   # 「最後のイベント + 4*halfwidth」で切って
                                   # いたが、no ice はイベントが surface しか
                                   # 無いため 15 ns で打ち切られ、氷ありの図と
                                   # 軸が揃わなかった。この図は条件間の比較に
                                   # 使うものなので、範囲が中身に依存しては
                                   # いけない。狭めたいときだけ (t0, t1) を書く。
SPECTROGRAM_FREQ_RANGE_GHZ = (0.3, 2.3)   # 表示する周波数範囲（帯域より少し広く）
SPECTROGRAM_DB_RANGE = 90.0        # カラースケールの下限（最大値からの落差）。
                                   # 地表反射から 80 dB 下の excess の ice_top まで
                                   # 入れる必要があるので広めに取ってある。
                                   # 強いイベントだけを見たいなら 40-60 に狭める。
SPECTROGRAM_NORM = 'map_max'       # 'map_max' … マップ全体の最大で正規化
                                   # 'lsr'     … 参照計算 far_1m で割る（fig2 と同じ量）
SPECTROGRAM_SNR_MIN_DB = 6.0       # これ未満の時刻では重心の追跡線を灰色の点線に
                                   # 落とす。フロアを測っているだけの区間で重心が
                                   # 跳ねるのを、線種で区別できるようにする。
                                   # 既存の SNR_MARGINAL_DB に合わせてある。
SPECTROGRAM_SHOW_TRACKING_LIMIT = False
                                   # (b) にこの閾値の縦線を引くか。既定は引かない。
                                   # 閾値を跨いだことは (a) の線種の変化で分かるので
                                   # 線は冗長になり、図の要素が増えるだけになる。
                                   # 値は noise floor の凡例に併記する。
SPECTROGRAM_CMAP = 'viridis'
SPECTROGRAM_TRACK_SMOOTH_NS = 0.0  # 追跡線の移動平均の窓幅 [ns]。0 なら平滑化しない。
                                   # 【なぜ振動するか】励振が平坦帯域なので時間
                                   # 領域では sinc に近く、長いリンギングを持つ。
                                   # 窓を滑らせるとその山谷が窓の縁を出入りし、
                                   # 重心が窓の位置に応じて細かく振れる。これは
                                   # 数値誤差ではなく実際に起きていることなので、
                                   # 既定では平滑化しない。読みにくいときだけ
                                   # 0.5-1.0 ns 程度を入れる（表示のみ。CSV には
                                   # 常に平滑化前の値を書く）。

FIG_EVENT_COLORS = ['k', 'tab:red', 'tab:blue', 'tab:green', 'tab:purple']

# 実測と理論の描き分け（全図で共通）。o / 実線 = 実測、x / 破線 = 理論。
STYLE_MEAS = dict(ls='-', marker='o', ms=7)
STYLE_TH = dict(ls='--', marker='x', ms=8)


# =============================================================================
# 1. 反射イベントの定義
# =============================================================================
def build_events(level):
    """レベルの構造から反射イベントの一覧を作る。

    戻り値: [{'name', 'depth_m', 'above'}, ...]
      depth_m : 界面の深さ [m]
      above   : その界面より浅い層の [(厚さ, 氷層か), ...]（往復経路の計算用）
    """
    events = [{'name': 'surface', 'depth_m': 0.0, 'above': []}]
    # 氷なし（ICE_MODEL='none'）では interface_depths が空になるので、
    # 地表反射だけのイベント列になる。Level 1-3 と同じ扱い。
    bounds = sm.interface_depths(level)
    names = ['ice_top', 'ice_bottom']
    for i, b in enumerate(bounds):
        events.append({'name': names[i] if i < len(names) else 'iface{}'.format(i),
                       'depth_m': float(b), 'above': []})
    return events


def _n_of(f, level, in_ice, depth_m=None):
    """層の屈折率 n(f)。氷層かどうかで切り替える。

    Level 5 では深さでも変わるので、depth_m を渡せばその深さの値を返す。
    省略時は氷層の中央（in_ice=True）または氷層上面の直上（False）を使う。
    """
    if has_density_profile(level):
        z = depth_m
        if z is None:
            top = float(sm.LEVEL4_ICE_TOP_M)
            z = (top + 0.5 * float(sm.LEVEL4_ICE_THICK_M)) if in_ice else max(
                0.0, top - 1e-6)
        return np.sqrt(eps_at_depth(f, float(z), level)[0])
    if 'ice_layer' in LEVEL_EFFECTS[level]:
        return np.sqrt(level4_eps(f, in_ice)[0])
    return refractive_index(f, level)


def _alpha_of(f, level, in_ice, depth_m=None):
    """層の減衰係数 alpha(f) [Np/m]。Level 5 では深さ依存。"""
    if has_density_profile(level):
        z = depth_m
        if z is None:
            top = float(sm.LEVEL4_ICE_TOP_M)
            z = (top + 0.5 * float(sm.LEVEL4_ICE_THICK_M)) if in_ice else max(
                0.0, top - 1e-6)
        from subsurface_model import alpha_at_depth
        return np.atleast_1d(alpha_at_depth(f, float(z), level))
    if 'ice_layer' in LEVEL_EFFECTS[level]:
        return level4_alpha(f, in_ice)
    if 'absorb_tandelta' in LEVEL_EFFECTS[level]:
        return level3_alpha(f)
    if 'absorb_const' in LEVEL_EFFECTS[level]:
        return level2_alpha(f, refractive_index(f, level))
    return np.zeros_like(np.asarray(f, dtype=float))


# =============================================================================
# 2. 反射の伝達関数（README §3.1）
# =============================================================================
def event_terms(f, event, level):
    """イベントの伝達関数の各項を個別に返す。

    Level 5（密度プロファイル）では n も alpha も深さで変わるので、
    閉形式では書けない。subsurface_model の経路積分に委ねる。
    Level 3/4（均質）は密度一定の特殊ケースとして同じ関数で扱え、
    従来の閉形式と厳密に一致する。

    戻り値の辞書:
      'G'     幾何      sqrt(R_REF / r_eff)、r_eff = 2h + 2∫dz/n
      'T'     往復透過  地表と、界面より浅い界面すべて
      'R'     反射      その界面の (n_above - n_below)/(n_above + n_below)
      'A'     吸収      exp(-2 ∫alpha dz)
      't_ns'  往復走時  2h/c + 2∫n dz / c（帯域中心での代表値）
      'r_eff' 見かけ源距離
    """
    f_arr = np.asarray(f, dtype=float)
    d = float(event['depth_m'])

    r_eff = r_eff_roundtrip(f_arr, d, level)
    G = np.sqrt(R_REF / r_eff)

    if d <= 0.0:
        # 地表反射そのもの。地表を透過しないので T = 1。
        n_s = surface_index(f_arr, level)
        T = np.ones_like(f_arr)
        R = (np.ones_like(f_arr) - n_s) / (np.ones_like(f_arr) + n_s)
    else:
        T = transmission_product(f_arr, d, level, two_way=True,
                                 include_surface=True)
        R = interface_reflection(f_arr, d, level)

    att, opt, _ = path_integrals(f_arr, d, level)
    A = np.exp(-2.0 * att)
    t_f = 2.0 * TX_HEIGHT / C + 2.0 * opt / C

    i_c = int(np.argmin(np.abs(f_arr - BAND_CENTRE_HZ)))
    return {'G': G, 'T': T, 'R': R, 'A': A, 'r_eff': r_eff,
            't_ns': float(t_f[i_c]), 't_f': t_f}


def event_arrival_ns(event, level):
    """包絡ピークに対応する往復の群走時 [ns]（スカラー）。"""
    _, opt_g, _ = path_integrals(np.array([BAND_CENTRE_HZ]),
                                 float(event['depth_m']), level, group=True)
    return float(2.0 * TX_HEIGHT / C + 2.0 * opt_g[0] / C)


def synth_theory(E_ref_f, freq, event, level):
    """イベントの理論スペクトル（参照に伝達関数を掛けたもの）。"""
    tm = event_terms(freq, event, level)
    H = tm['G'] * tm['T'] * tm['R'] * tm['A']
    delay_s = (tm['t_f'] - R_REF / C) * 1e-9
    return E_ref_f * H * np.exp(-2j * np.pi * freq * delay_s), tm


# =============================================================================
# 3. 時間ゲート
# =============================================================================
def gate_trace(trace, dt, t_center_ns, halfwidth_ns=None):
    """イベント抽出用の時間ゲート。

    ascan_spectrum.apply_gate と同じ Tukey 窓だが、幅と中心を引数で受け取る。
    at_tx は 1 トレースに複数イベントが並ぶため、イベントごとに窓が要る。
    """
    hw = GATE_HALFWIDTH_NS if halfwidth_ns is None else halfwidth_ns
    dt_ns = dt * 1e9
    t_axis = np.arange(len(trace)) * dt_ns
    idx = np.where((t_axis >= t_center_ns - hw) & (t_axis <= t_center_ns + hw))[0]
    if len(idx) < 8:
        raise CmdInputError(
            'ゲート窓 [{:.2f}, {:.2f}] ns にサンプルがほとんどありません'
            .format(t_center_ns - hw, t_center_ns + hw))
    win = np.zeros(len(trace))
    win[idx] = signal.windows.tukey(len(idx), alpha=GATE_TAPER)
    return trace * win, (t_axis[idx[0]], t_axis[idx[-1]])


def refine_center(trace, dt, t_center_ns, halfwidth_ns=None):
    """窓内の包絡ピーク位置を返す（GATE_CENTER='measured' 用）。"""
    hw = GATE_HALFWIDTH_NS if halfwidth_ns is None else halfwidth_ns
    dt_ns = dt * 1e9
    t_axis = np.arange(len(trace)) * dt_ns
    idx = np.where((t_axis >= t_center_ns - hw) & (t_axis <= t_center_ns + hw))[0]
    env = np.abs(signal.hilbert(trace))[idx]
    return float(t_axis[idx[int(np.argmax(env))]])


# =============================================================================
# 4. 氷なし計算を基準にした検出判定
# =============================================================================
def resolve_noice_path(level):
    """そのレベルに対応する氷なし計算のパスを返す。

    氷層を持たないレベル（Level 1-3 など）では比較相手が存在しないので空を返す。
    """
    if NOICE_TRACE_PATH:
        return NOICE_TRACE_PATH
    # 定義済みのレベルで氷層を持たないものだけを除外する。未定義のレベル
    # （Level 6 以降）は既定（Level 5 の no_ice）に落とす。「氷層が無いから
    # 比較相手が無い」と「まだ LEVEL_EFFECTS に登録していない」を取り違えて
    # 黙って基準なしで走るのを防ぐため。
    eff = LEVEL_EFFECTS.get(level)
    if eff is not None and 'ice_layer' not in eff:
        return ''
    return NOICE_TRACE_PATHS.get(level, NOICE_TRACE_PATHS['_default'])


def envelope(trace):
    """Hilbert 変換による包絡 |x + i H{x}| = sqrt(x^2 + H{x}^2)。"""
    return np.abs(signal.hilbert(trace))


def judge_snr(snr_db):
    """イベントごとの検出判定（修正 6）。

    フロアの絶対値ではなく SNR で判定する。フロアを「地表反射比」で表すと、
    直達波を差し引いたかどうかで基準が 22 dB 変わり、同じデータなのに判定が
    ひっくり返ってしまうため（SNR は基準量に依らない）。
    """
    if not np.isfinite(snr_db):
        return '判定不能'
    if snr_db >= SNR_DETECT_DB:
        return '検出可'
    if snr_db >= SNR_MARGINAL_DB:
        return '限界'
    return '不可'


# =============================================================================
# 5. 解析本体
# =============================================================================
def _load_and_align(path, dt_ref, n_ref, what):
    """差し引くトレースを読み、dt と長さを揃える。"""
    tr, dt_x = load_trace(path)
    if not np.isclose(dt_x, dt_ref, rtol=1e-9):
        raise CmdInputError('{} の dt が一致しません: {} vs {}'
                            .format(what, dt_x, dt_ref))
    if len(tr) < n_ref:
        tr = np.pad(tr, (0, n_ref - len(tr)))
    return tr[:n_ref]


def analyze(at_tx_path, ref_path, level, freespace_path='',
            background_path='', noice_path=''):
    """at_tx トレースを読み、イベントごとのスペクトルと理論を突き合わせる。

    freespace_path : 自由空間 at_tx。指定すると直達波を差し引く（修正 2）。
    background_path: 氷なし at_tx。指定すると背景差分を行う（実機では不可）。
    noice_path     : 氷なし計算の at_tx。検出判定の基準になる。各イベントと
                     まったく同じ時間窓で包絡ピークを測り、その比を検出 SNR と
                     する。背景差分（background_path）とは別物で、こちらは
                     トレースを引かずに比較だけを行う。
    """
    trace, dt = load_trace(at_tx_path)
    ref_trace, dt_ref = load_trace(ref_path)
    if not np.isclose(dt, dt_ref, rtol=1e-9):
        raise CmdInputError('at_tx と参照で dt が異なります: {} vs {}'
                            .format(dt, dt_ref))

    raw_trace = np.array(trace, dtype=float)
    fs_trace = None
    if freespace_path:
        fs_trace = _load_and_align(freespace_path, dt, len(trace),
                                   '自由空間 at_tx')
        trace = trace - fs_trace

    bg_trace = None
    if background_path:
        bg_trace = _load_and_align(background_path, dt, len(trace),
                                   '氷なし at_tx')

    freq_ref, E_ref_full = spectrum(ref_trace, dt)
    freq, _ = spectrum(trace, dt)
    E_ref = _interp_complex_to_grid(freq_ref, E_ref_full, freq)

    events = build_events(level)

    # -------------------------------------------------------------------
    # 各イベントの理論トレースを先に作り、その包絡ピークをゲート中心にする。
    #
    # 【なぜ幾何走時をそのまま使わないか】
    # 励振ファイル（帯域制限波形）は自身に時間遅延を持つため、実測トレースの
    # 到達時刻は「波源遅延 + 幾何走時」になる。event_arrival_ns() が返すのは
    # 幾何走時だけなので、そのままゲート中心に使うと窓がイベントを外す。
    # 理論トレースは参照トレース（波源遅延を含む）に伝達関数を掛けたものなので、
    # その包絡ピークを使えば波源遅延を仮定なしで取り込める。
    # ascan_spectrum.py が実測・理論の両方を同じ土俵で比べているのと同じ考え方。
    # -------------------------------------------------------------------
    prepared = []
    for ev in events:
        E_th_full, tm = synth_theory(E_ref, freq, ev, level)
        th_trace = np.fft.irfft(E_th_full, n=len(trace))
        t_geom = event_arrival_ns(ev, level)
        t_th = measure_peak(th_trace, dt)['t_peak']
        prepared.append({'ev': ev, 'terms': tm, 'th_trace': th_trace,
                         't_geom': t_geom, 't_theory': t_th})
    source_delay = prepared[0]['t_theory'] - prepared[0]['t_geom']

    # 地表反射（at_tx では直達波を含む）のピーク振幅。dB 表示の基準。
    g_surf, _ = gate_trace(trace, dt, prepared[0]['t_theory'])
    surf_amp = measure_peak(g_surf, dt)['amp_peak']

    # 氷なし計算を読み込む。直達波の差分は氷ありとまったく同じ処理をする
    # （でないと基準側だけ直達波が残り、比較が成立しない）。
    if noice_path:
        nz_raw = _load_and_align(noice_path, dt, len(raw_trace), '氷なし at_tx')
        noice_trace = nz_raw if fs_trace is None else (nz_raw - fs_trace)
        noice_work = (noice_trace if bg_trace is None
                      else (noice_trace - bg_trace))
    else:
        noice_trace = noice_work = None

    work = trace if bg_trace is None else (trace - bg_trace)

    results = []
    for i, pr in enumerate(prepared):
        ev, tm, th_trace = pr['ev'], pr['terms'], pr['th_trace']
        t_th = pr['t_theory']
        t_center = (refine_center(work, dt, t_th) if GATE_CENTER == 'measured'
                    else t_th)
        gated, window = gate_trace(work, dt, t_center)
        _, E_meas = spectrum(gated, dt)

        # 理論側にも同じゲートをかける（既存コードと同じ思想）
        th_gated, _ = gate_trace(th_trace, dt, t_center)
        _, E_th = spectrum(th_gated, dt)

        pk = measure_peak(gated, dt)
        mask = valid_mask(freq, E_meas, E_ref)

        # 氷なし計算に「まったく同じ窓」をかけ、その包絡ピークを基準にする。
        # 窓の中心は氷ありのイベント時刻に合わせる（「そのイベントが見えた
        # 時刻に、氷が無ければ何があったか」を問うため）。pore では氷層下端が
        # 0.26 ns 遅れるので、氷なし側の理論走時を使ってはいけない。
        if noice_work is not None:
            nz_gated, _ = gate_trace(noice_work, dt, t_center)
            nz_pk = measure_peak(nz_gated, dt)
            noice_amp = nz_pk['amp_peak']
            noice_t_peak = nz_pk['t_peak']
            _, E_noice = spectrum(nz_gated, dt)
        else:
            noice_amp, noice_t_peak, E_noice = np.nan, np.nan, None

        L_abs_meas = log_spectral_ratio(E_meas, E_ref)
        L_abs_th = log_spectral_ratio(E_th, E_ref)

        mom_m = moments(freq, E_meas)
        mom_t = moments(freq, E_th)
        flohi_m = lo_hi_freq(freq, E_meas)
        flohi_t = lo_hi_freq(freq, E_th)

        results.append({
            'name': ev['name'], 'depth_m': ev['depth_m'], 'event': ev,
            'freq': freq,
            'color': FIG_EVENT_COLORS[i % len(FIG_EVENT_COLORS)],
            't_geom': pr['t_geom'], 't_theory': t_th,
            't_center': t_center, 't_measured': pk['t_peak'],
            'amp_peak': pk['amp_peak'], 'window': window,
            'noice_amp': noice_amp, 'noice_t_peak': noice_t_peak,
            'E_noice': E_noice,
            'E_meas': E_meas, 'E_theory': E_th, 'terms': tm, 'mask': mask,
            'L_abs_meas': L_abs_meas, 'L_abs_theory': L_abs_th,
            'moments_meas': mom_m, 'moments_theory': mom_t,
            'flohi_meas': flohi_m, 'flohi_theory': flohi_t,
            'tau_g_meas': group_delay(freq, E_meas, E_ref),
            'tau_g_theory': group_delay(freq, E_th, E_ref),
        })

    # --- 相対 LSR（基準イベントからの比）------------------------------------
    ref_name = REL_LSR_REF_EVENT
    names = [r['name'] for r in results]
    if ref_name not in names:
        ref_name = names[0]
    r0 = results[names.index(ref_name)]
    for r in results:
        r['rel_ref'] = ref_name
        r['L_rel_meas'] = log_spectral_ratio(r['E_meas'], r0['E_meas'])
        r['L_rel_theory'] = log_spectral_ratio(r['E_theory'], r0['E_theory'])
        r['d_rel'] = r['depth_m'] - r0['depth_m']

    info = {'dt': dt, 'freq': freq, 'E_ref': E_ref, 'trace': trace,
            'raw_trace': raw_trace, 'freespace': fs_trace,
            'work': work, 'background': bg_trace,
            'noice_trace': noice_trace, 'noice_work': noice_work,
            'noice_src': noice_path or '(氷層のないレベルなので比較相手なし)',
            'surface_amp': surf_amp,
            'rel_ref': ref_name, 'source_delay': source_delay,
            'subtracted': fs_trace is not None}
    return results, info


# =============================================================================
# 6. 減衰率の逆算（README §3.3）
# =============================================================================
def _theory_path_alpha(r, level):
    """イベントまでの往復経路の平均 alpha（片道換算）。theory 側の基準値。"""
    d = r['depth_m']
    if d <= 0.0:
        return None
    freq = r['freq']
    acc = np.zeros_like(freq)
    for length, in_ice in r['event']['above']:
        acc = acc + _alpha_of(freq, level, in_ice) * length
    return acc / d


def alpha_from_absolute(r, level):
    """絶対 LSR から往復経路平均の alpha を逆算する。

        alpha = alpha_theory + (L_abs_theory - L_abs_meas) / (2 d)

    理論との差分で書くのは、**時間ゲートによる振幅損失を打ち消すため**。
    ゲートは実測と理論に同じものをかけているので、差分を取れば窓の効果が
    消える。解析式 -[L - ln(G T |R|)]/(2d) を直接使うとゲート損失がそのまま
    alpha の系統誤差になる（深さ 1 m で 20% 程度）。

    なお R を既知として使う点は変わらないので、順方向モデルの照合用であって
    実機での逆解析には使えない（README §3.3 の注意）。
    """
    d = r['depth_m']
    if d <= 0.0:
        return None
    a_th = _theory_path_alpha(r, level)
    with np.errstate(invalid='ignore', divide='ignore'):
        return a_th + (r['L_abs_theory'] - r['L_abs_meas']) / (2.0 * d)


def alpha_from_relative(r, r0, level):
    """相対 LSR から層間の alpha を逆算する（実測可能な形）。

        alpha = alpha_theory_layer + (L_rel_theory - L_rel_meas) / (2 (d_k - d_j))

    絶対版と同じく理論との差分で書き、ゲートの効果を打ち消す。
    氷層の上下面で取ると |R| の比がちょうど 1、往復透過の比も氷上面の
    0.99943 だけになるので、実質的に幾何項の補正だけで層内 alpha が求まる。
    こちらは参照計算を必要としないため、実機でも使える形になっている。
    """
    dd = r['depth_m'] - r0['depth_m']
    if abs(dd) < 1e-9:
        return None
    freq = r['freq']
    # 2 イベントの間にある層だけを取り出す。'above' は浅い順の前置リストなので、
    # 浅いほうの長さ以降が「2 つの界面に挟まれた層」になる。
    deep, shallow = ((r, r0) if r['depth_m'] > r0['depth_m'] else (r0, r))
    layers = deep['event']['above'][len(shallow['event']['above']):]
    acc = np.zeros_like(freq)
    for length, in_ice in layers:
        acc = acc + _alpha_of(freq, level, in_ice) * length
    a_th = acc / abs(dd)
    # dd の符号が LSR の符号を打ち消すので、dd はそのまま（絶対値にしない）
    with np.errstate(invalid='ignore', divide='ignore'):
        return a_th + (r['L_rel_theory'] - r['L_rel_meas']) / (2.0 * dd)


def reflection_spectrum(r):
    """絶対 LSR から幾何・透過・吸収を除いた |R(f)| を返す。"""
    tm = r['terms']
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.exp(r['L_abs_meas']) / (tm['G'] * tm['T'] * tm['A'])


# =============================================================================
# 7. 作図
# =============================================================================
def theory_amp(r, i_c):
    """帯域中心での理論振幅 |G T R A|（イベント間の比較に使う）。"""
    tm = r['terms']
    return float(abs(tm['G'][i_c] * tm['T'][i_c] * tm['R'][i_c] * tm['A'][i_c]))


def _band(freq_hz):
    g = freq_hz * 1e-9
    return (g >= BAND_GHZ[0]) & (g <= BAND_GHZ[1])


def _event_handles(results, with_style=True):
    """色 = イベント、線種 = 実測／理論。凡例に両方を出す。"""
    h = [Line2D([0], [0], color=r['color'], lw=2, label=r['name'])
         for r in results]
    if with_style:
        h += [Line2D([0], [0], color='0.3', lw=2, ls='-', label='measured'),
              Line2D([0], [0], color='0.3', lw=2, ls='--', label='theory')]
    return h


def _marker_handles():
    """fig1 の (b)(c) 用。o = 実測、x = 理論。"""
    return [Line2D([0], [0], color='0.3', lw=2, **STYLE_MEAS, label='measured'),
            Line2D([0], [0], color='0.3', lw=2, **STYLE_TH, label='theory')]


def plot_trace(results, info, output_dir, overlay_noice=True,
               stem='fig0_trace'):
    """fig0: (a) 全波形＋理論走時＋ゲート窓 (b) 包絡の dB とノイズフロア。"""
    dt_ns = info['dt'] * 1e9
    t = np.arange(len(info['work'])) * dt_ns
    env = np.abs(signal.hilbert(info['work']))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    if info['subtracted'] or info['background'] is not None:
        axes[0].plot(t, info['raw_trace'], color='0.75', lw=0.6, zorder=0,
                     label='raw (before subtraction)')
    axes[0].plot(t, info['work'], color='k', lw=0.8,
                 label='at_tx' + (' (subtracted)' if info['subtracted'] else ''))
    for r in results:
        axes[0].axvline(r['t_theory'], color=r['color'], ls='--', lw=1.2)
        axes[0].axvspan(r['window'][0], r['window'][1], color=r['color'],
                        alpha=0.12)
    axes[0].set_ylabel('Ez [linear]', fontsize=13)
    axes[0].set_title('(a) Trace with theoretical arrivals and gates',
                      fontsize=13)
    axes[0].legend(fontsize=10)

    # 氷なし計算の包絡を重ねる。水平線ではなく曲線にするのは、背景の水準が
    # 時間（深さ）で大きく変わるため。理論では段差反射が深さ 1 m で -81.5 dB、
    # 2 m で -92.1 dB と 10 dB 以上違い、水平線で代表させると深い側を損する。
    if overlay_noice and info['noice_work'] is not None:
        with np.errstate(divide='ignore'):
            nz_db = 20.0 * np.log10(
                envelope(info['noice_work']) / info['surface_amp'])
        axes[1].plot(t, nz_db, color='tab:green', lw=0.9, alpha=0.85,
                     label='no ice (same medium without the ice layer)')
        # 各ゲート内での氷なし包絡ピーク＝検出 SNR の分母。これは「窓の
        # 代表値」であって曲線上の 1 点ではない（最大が現れる時刻は窓の中の
        # どこでもよい）。以前はゲート中心に点マーカーで描いていたため、
        # 緑の曲線から浮いて見えて誤解を招いた。窓の幅いっぱいに伸びる
        # 水平線にして、窓全体を代表する量であることを見た目で示す。
        # 実際にピークが立っている時刻には細い縦線を添える。
        for r in results:
            if not np.isfinite(r['noice_amp']):
                continue
            lvl = 20.0 * np.log10(r['noice_amp'] / info['surface_amp'])
            axes[1].hlines(lvl, r['window'][0], r['window'][1],
                           color='tab:green', lw=2.5, alpha=0.95, zorder=5)
            if np.isfinite(r['noice_t_peak']):
                axes[1].vlines(r['noice_t_peak'], lvl - 3.0, lvl + 3.0,
                               color='tab:green', lw=1.2, alpha=0.8, zorder=5)

    with np.errstate(divide='ignore'):
        env_db = 20.0 * np.log10(env / info['surface_amp'])
    axes[1].plot(t, env_db, color='k', lw=0.9)
    i_c = int(np.argmin(np.abs(info['freq'] - BAND_CENTRE_HZ)))
    for r in results:
        axes[1].axvline(r['t_theory'], color=r['color'], ls='--', lw=1.2)
        # 理論振幅は「地表反射の理論振幅」で正規化する。
        # 以前は分母が |R_surface| だけになっており、地表反射の幾何項
        # G = sqrt(R_REF/2h) = 1.195 を落としていた（1.55 dB のずれ）。
        th_db = 20.0 * np.log10(theory_amp(r, i_c) / theory_amp(results[0], i_c))
        axes[1].plot(r['t_theory'], th_db, marker='o', ms=7,
                     color=r['color'], mfc='none', mew=2)
    # 重ね書きなしの版では (b) にラベル付きの線が無くなるので、凡例を出さない
    # （出すと matplotlib が「ラベル付きの要素がない」と警告する）。
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(fontsize=9, loc='upper right')
    axes[1].set_xlabel('Time [ns]', fontsize=13)
    axes[1].set_ylabel('Envelope [dB re. surface peak]', fontsize=13)
    axes[1].set_title('(b) Envelope, theory (circles)'
                      + (' and no-ice reference' if overlay_noice
                         and info['noice_work'] is not None else ''),
                      fontsize=13)
    axes[1].set_ylim(-90, 5)
    for ax in axes:
        ax.grid(alpha=0.4)
        ax.minorticks_on()
    fig.legend(handles=_event_handles(results), loc='upper center', ncol=4,
               fontsize=11, bbox_to_anchor=(0.5, 0.0), frameon=True)
    plt.tight_layout()
    save_figure(fig, output_dir, stem)


def plot_spectra(results, info, output_dir):
    """fig1: (a) 生スペクトル (b) 重心と幅 f_c ± sigma_f。

    【修正 7：帯域端パネルを廃止した理由】
    旧 (b) は各イベントの帯域内最大に対する -10 dB 交差（f_lo / f_hi）を
    描いていたが、この量は媒質に反応しない。吸収による帯域内の傾きは
        surface 0.00 dB / ice_top 1.99 dB / ice_bottom 3.90 dB
    しかなく、-10 dB のしきい値に原理的に届かない。したがって交差点は
    媒質ではなく励振波形の Tukey ロールオフの位置で決まってしまい、
    帯域端に張り付くか一点に潰れる。しきい値を下げても平坦部に入るだけで
    改善しない。

    一方 (b)（旧 (c)）の f_c ± sigma_f は帯域内の全パワーを積分した量なので、
    わずかな傾きでも確実に反映される。実測でも surface 1.25 -> ice_top 1.21
    -> ice_bottom 1.17 GHz と単調に下がり、alpha ∝ f による重心シフトが
    そのまま見える。

    f_lo / f_hi は events.csv には残してあるので、必要なら参照できる。
    """
    freq = info['freq']
    fg = freq * 1e-9
    band = _band(freq)
    norm = max(float(np.max(np.abs(r['E_meas'][band]))) for r in results)

    fig, axes = plt.subplots(2, 1, figsize=(10, 9))

    ax = axes[0]
    with np.errstate(divide='ignore'):
        ax.plot(fg, 20 * np.log10(np.abs(info['E_ref']) / norm),
                color='0.5', lw=1.2, ls=':', label='E_ref (far_1m, source shape)')
        for r in results:
            ax.plot(fg, 20 * np.log10(np.abs(r['E_meas']) / norm),
                    color=r['color'], lw=1.4, label=r['name'] + ' (measured)')
            ax.plot(fg, 20 * np.log10(np.abs(r['E_theory']) / norm),
                    color=r['color'], lw=1.0, ls='--', alpha=0.8,
                    label=r['name'] + ' (theory)')
        for r in results:
            if r['E_noice'] is not None:
                ax.plot(fg, 20 * np.log10(np.abs(r['E_noice']) / norm),
                        color=r['color'], lw=1.0, ls=':', alpha=0.9,
                        label=r['name'] + ' (no ice)')
    for x in BAND_GHZ:
        ax.axvline(x, color='k', ls=':', lw=1.0)
    ax.set_xlim(0, 3.0)
    ax.set_ylim(-100, 5)
    ax.set_xlabel('Frequency [GHz]', fontsize=13)
    ax.set_ylabel('|E(f)| [dB re. max of measured events]', fontsize=13)
    ax.set_title('(a) Raw spectra', fontsize=13)
    ax.legend(fontsize=9, ncol=2, loc='lower left')

    ax = axes[1]
    for i, r in enumerate(results):
        for src, st in ((r['moments_meas'], STYLE_MEAS),
                        (r['moments_theory'], STYLE_TH)):
            fc = src['f_c'] * 1e-9
            sg = src['sigma_f'] * 1e-9
            ax.plot([fc - sg, fc, fc + sg], [i, i, i], color=r['color'],
                    lw=1.4, **st)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([r['name'] for r in results])
    ax.set_xlabel('Frequency [GHz]', fontsize=13)
    ax.set_title(r'(b) Centroid and spectral width $f_c \pm \sigma_f$'
                 '   (middle marker = $f_c$)', fontsize=13)
    ax.invert_yaxis() # surface > ice_top > ice_bottom の順に上から並べるため追加（消すな！！）

    for ax in axes:
        ax.grid(alpha=0.4)
        ax.minorticks_on()
    fig.legend(handles=_marker_handles(), loc='upper center', ncol=2,
               fontsize=11, bbox_to_anchor=(0.5, 0.0), frameon=True)
    plt.tight_layout()
    save_figure(fig, output_dir, 'fig1_spectra')


def plot_lsr(results, info, output_dir):
    """fig2: (a) 絶対 LSR (b) 絶対の残差 / (c) 相対 LSR (d) 相対の残差。

    修正 4 でパネル配置を上段 a,b・下段 c,d に変更し、
    実線 = 実測 / 破線 = 理論 を凡例に明記した。
    """
    freq = info['freq']
    fg = freq * 1e-9
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    spec = [((0, 0), (0, 1), 'L_abs_meas', 'L_abs_theory',
             'LSR (ref = far_1m)', '(a)', '(b)'),
            ((1, 0), (1, 1), 'L_rel_meas', 'L_rel_theory',
             'LSR (ref = {})'.format(info['rel_ref']), '(c)', '(d)')]
    for pos_v, pos_r, key_m, key_t, ttl, tag_v, tag_r in spec:
        av, ar = axes[pos_v], axes[pos_r]
        for r in results:
            m = r['mask']
            av.plot(fg[m], 8.686 * r[key_m][m], color=r['color'], lw=1.4)
            av.plot(fg[m], 8.686 * r[key_t][m], color=r['color'], lw=1.0,
                    ls='--')
            ar.plot(fg[m], 8.686 * (r[key_m][m] - r[key_t][m]),
                    color=r['color'], lw=1.4)
        av.set_title('{} {}'.format(tag_v, ttl), fontsize=12)
        ar.set_title('{} residual (measured - theory)'.format(tag_r),
                     fontsize=12)
        ar.axhline(0, color='k', lw=0.8)
    for a in axes.ravel():
        a.set_xlabel('Frequency [GHz]', fontsize=12)
        a.set_ylabel('LSR [dB]', fontsize=12)
        a.set_xlim(BAND_GHZ)
        a.grid(alpha=0.4)
        a.minorticks_on()
    fig.legend(handles=_event_handles(results), loc='upper center', ncol=5,
               fontsize=11, bbox_to_anchor=(0.5, 0.0), frameon=True)
    plt.tight_layout()
    save_figure(fig, output_dir, 'fig2_lsr')


def plot_attenuation(results, info, level, output_dir):
    """fig3: 絶対／相対 LSR から逆算した alpha と tanδ。"""
    freq = info['freq']
    fg = freq * 1e-9
    names = [r['name'] for r in results]
    r0 = results[names.index(info['rel_ref'])]
    n_reg = _n_of(freq, level, False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for r in results:
        m = r['mask']
        a_abs = alpha_from_absolute(r, level)
        if a_abs is not None:
            axes[0, 0].plot(fg[m], a_abs[m], color=r['color'], lw=1.4)
            axes[0, 1].plot(fg[m], alpha_to_tandelta(a_abs, freq, n_reg)[m],
                            color=r['color'], lw=1.4)
        a_rel = alpha_from_relative(r, r0, level)
        if a_rel is not None:
            axes[1, 0].plot(fg[m], a_rel[m], color=r['color'], lw=1.4)
            axes[1, 1].plot(fg[m], alpha_to_tandelta(a_rel, freq, n_reg)[m],
                            color=r['color'], lw=1.4)

    # 理論曲線：背景レゴリスと（あれば）氷層
    th = [(_alpha_of(freq, level, False), 'r', '--', 'Theory (regolith)')]
    if 'ice_layer' in LEVEL_EFFECTS[level]:
        th.append((_alpha_of(freq, level, True), 'm', ':', 'Theory (ice layer)'))
    for a_th, col, ls, lab in th:
        axes[0, 0].plot(fg, a_th, color=col, ls=ls, lw=1.5, label=lab)
        axes[1, 0].plot(fg, a_th, color=col, ls=ls, lw=1.5, label=lab)
        axes[0, 1].plot(fg, alpha_to_tandelta(a_th, freq, n_reg), color=col,
                        ls=ls, lw=1.5, label=lab)
        axes[1, 1].plot(fg, alpha_to_tandelta(a_th, freq, n_reg), color=col,
                        ls=ls, lw=1.5, label=lab)

    # 色 = イベント（すべて実測値）。理論は赤破線／紫点線の 2 本。
    titles = [('(a) alpha from absolute LSR (round-trip average)',
               r'$\alpha(f)$ [1/m]'),
              ('(b) tan_delta from absolute LSR', r'tan$\delta(f)$'),
              ('(c) alpha from relative LSR (field-measurable)',
               r'$\alpha(f)$ [1/m]'),
              ('(d) tan_delta from relative LSR', r'tan$\delta(f)$')]
    for a, (ttl, ylab) in zip(axes.ravel(), titles):
        a.set_title(ttl, fontsize=12)
        a.set_xlabel('Frequency [GHz]', fontsize=12)
        a.set_ylabel(ylab, fontsize=12)
        a.set_xlim(BAND_GHZ)
        a.grid(alpha=0.4)
        a.minorticks_on()
        a.legend(fontsize=9)
    fig.legend(handles=_event_handles(results, with_style=False)
               + [Line2D([0], [0], color='r', ls='--', lw=2,
                         label='Theory (regolith)'),
                  Line2D([0], [0], color='m', ls=':', lw=2,
                         label='Theory (ice layer)')],
               loc='upper center', ncol=5, fontsize=11,
               bbox_to_anchor=(0.5, 0.0), frameon=True)
    plt.tight_layout()
    save_figure(fig, output_dir, 'fig3_attenuation')


def plot_phase(results, info, output_dir):
    """fig4: 群遅延と残差。"""
    freq = info['freq']
    fg = freq * 1e-9
    fig, axes = plt.subplots(2, 1, figsize=(10, 9))
    for r in results:
        m = r['mask']
        axes[0].plot(fg[m], r['tau_g_meas'][m], color=r['color'], lw=1.4)
        axes[0].plot(fg[m], r['tau_g_theory'][m], color=r['color'], lw=1.0,
                     ls='--')
        axes[1].plot(fg[m], r['tau_g_meas'][m] - r['tau_g_theory'][m],
                     color=r['color'], lw=1.4)
    axes[0].set_title('(a) Group delay: measured (solid) vs theory (dashed)',
                      fontsize=13)
    axes[1].set_title('(b) Group delay residual (numerical dispersion '
                      '+ multiples)', fontsize=13)
    axes[1].axhline(0, color='k', lw=0.8)
    for a in axes:
        a.set_xlabel('Frequency [GHz]', fontsize=12)
        a.set_ylabel('Group delay [ns]', fontsize=12)
        a.set_xlim(BAND_GHZ)
        a.grid(alpha=0.4)
        a.minorticks_on()
    fig.legend(handles=_event_handles(results), loc='upper center', ncol=4,
               fontsize=11, bbox_to_anchor=(0.5, 0.0), frameon=True)
    plt.tight_layout()
    save_figure(fig, output_dir, 'fig4_phase')


def plot_reflection(results, info, output_dir):
    """fig5: |R(f)| の実測と理論。"""
    freq = info['freq']
    fg = freq * 1e-9
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in results:
        m = r['mask']
        with np.errstate(divide='ignore', invalid='ignore'):
            ax.plot(fg[m], 20 * np.log10(reflection_spectrum(r)[m]),
                    color=r['color'], lw=1.4)
            ax.plot(fg, 20 * np.log10(np.abs(r['terms']['R'])),
                    color=r['color'], lw=1.0, ls='--')
    ax.set_xlabel('Frequency [GHz]', fontsize=13)
    ax.set_ylabel(r'$|R(f)|$ [dB]', fontsize=13)
    ax.set_title('Reflection coefficient  (solid: measured, dashed: theory)\n'
                 'diagnostic only: |R(f)| should be flat in band. '
                 'scalar R is handled by ascan_reflection.py', fontsize=11)
    ax.set_xlim(BAND_GHZ)
    ax.grid(alpha=0.4)
    ax.minorticks_on()
    fig.legend(handles=_event_handles(results), loc='upper center', ncol=4,
               fontsize=11, bbox_to_anchor=(0.5, 0.0), frameon=True)
    plt.tight_layout()
    save_figure(fig, output_dir, 'fig5_reflection')


# =============================================================================
# 8. 数値出力
# =============================================================================
def compute_spectrogram(results, info):
    """ゲートを時間方向に滑らせ、時間-周波数マップと重心の追跡を返す。

    fig1 と同じ gate_trace / spectrum / moments を使うので、追跡線が各イベントの
    理論走時を横切る位置での値は、fig1 のマーカーと一致するはずである
    （GATE_CENTER='theory' のとき厳密に、'measured' のときはゲート中心の
    決め方のぶんだけずれる）。両者が食い違うなら、どちらかのゲートが
    イベントを外している。

    返す dict:
        t_ns      (Nt,)      ゲート中心の時刻
        freq_ghz  (Nf,)      表示帯の周波数
        S_db      (Nt, Nf)   スペクトル強度 [dB]
        f_c, sigma_f         (Nt,) 重心とスペクトル幅 [GHz]（帯域 BAND_GHZ 内）
        peak_db   (Nt,)      その窓の包絡ピーク [dB re. surface peak]
        snr_db    (Nt,)      peak_db - ノイズフロア
        usable    (Nt,) bool snr_db >= SPECTROGRAM_SNR_MIN_DB
    """
    dt = info['dt']
    dt_ns = dt * 1e9
    work = info['work']
    freq = info['freq']
    hw = (GATE_HALFWIDTH_NS if SPECTROGRAM_HALFWIDTH_NS is None
          else float(SPECTROGRAM_HALFWIDTH_NS))
    t_end = (len(work) - 1) * dt_ns

    if SPECTROGRAM_TIME_RANGE_NS is not None:
        t0, t1 = SPECTROGRAM_TIME_RANGE_NS
        t0, t1 = max(0.0, float(t0)), min(t_end, float(t1))
    else:
        t0, t1 = 0.0, t_end
    # 端では窓がトレードからはみ出すぶんだけ切り詰められる（gate_trace は
    # 存在するサンプルにだけ Tukey をかける）。中身が無い区間なので実害は
    # ないが、最初と最後の halfwidth ぶんは窓が非対称であることに注意。
    if not (t1 > t0):
        raise CmdInputError(
            'スペクトログラムの時間範囲が空です（t0={:.2f}, t1={:.2f} ns）。'
            'SPECTROGRAM_TIME_RANGE_NS を見直してください。'.format(t0, t1))
    t_centers = np.arange(t0, t1 + 0.5 * SPECTROGRAM_STEP_NS,
                          SPECTROGRAM_STEP_NS)
    t_centers = t_centers[t_centers <= t_end]

    f_ghz_all = freq * 1e-9
    fsel = ((f_ghz_all >= SPECTROGRAM_FREQ_RANGE_GHZ[0])
            & (f_ghz_all <= SPECTROGRAM_FREQ_RANGE_GHZ[1]))
    if not np.any(fsel):
        raise CmdInputError('SPECTROGRAM_FREQ_RANGE_GHZ に周波数点がありません')

    mag = np.empty((t_centers.size, int(np.count_nonzero(fsel))))
    f_c = np.full(t_centers.size, np.nan)
    sig = np.full(t_centers.size, np.nan)
    pk_db = np.full(t_centers.size, np.nan)

    for i, tc in enumerate(t_centers):
        gated, _ = gate_trace(work, dt, float(tc), hw)
        _, E = spectrum(gated, dt)
        mag[i] = np.abs(E[fsel])
        pk = measure_peak(gated, dt)['amp_peak']
        # 【なぜゼロを弾くか】時間範囲をトレース全長にしたので、波源が到達する
        # 前の区間ではゲート内が厳密にゼロになる。そのまま log10 を取ると
        # -inf が入り、軸の下限やカラースケールの上限が -inf になって
        # matplotlib が落ちる。信号が無い時刻は NaN にして「値なし」として扱う。
        if pk > 0:
            pk_db[i] = 20.0 * np.log10(pk / info['surface_amp'])
        # 帯域内に電力が無い窓（トレース冒頭など）では重心が定義できない。
        with np.errstate(invalid='ignore', divide='ignore'):
            try:
                m = moments(freq, E)
                if np.isfinite(m['f_c']):
                    f_c[i] = m['f_c'] * 1e-9
                    sig[i] = m['sigma_f'] * 1e-9
            except (ZeroDivisionError, FloatingPointError, ValueError):
                pass

    if SPECTROGRAM_NORM == 'lsr':
        # 【帯域端の発散を防ぐ】far_1m は帯域外でほぼゼロなので、そのまま割ると
        # SPECTROGRAM_FREQ_RANGE_GHZ の端で LSR が +300 dB といった値になり、
        # カラースケールがそこに占領されて中身が見えなくなる。
        # valid_mask の第 1 段と同じ規則（帯域内最大から MASK_REF_FLOOR_DB 以上
        # 落ちる周波数は使わない）で切り、切った列は NaN にして描かない。
        ref = np.abs(info['E_ref'])
        in_band = ((f_ghz_all >= BAND_GHZ[0]) & (f_ghz_all <= BAND_GHZ[1]))
        ref_max = np.max(ref[in_band]) if np.any(in_band) else np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            ref_db = 20.0 * np.log10(ref / ref_max)
        good = (ref_db >= asp.MASK_REF_FLOOR_DB)[fsel]
        ref_sel = np.where(good, ref[fsel], np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            S_db = 20.0 * np.log10(mag / ref_sel[None, :])
        norm_note = 'dB re. far_1m (absolute LSR)'
    else:
        peak = np.nanmax(mag)
        with np.errstate(divide='ignore', invalid='ignore'):
            S_db = 20.0 * np.log10(mag / peak)
        norm_note = 'dB re. map max'

    # 振幅ゼロの列は log10 で -inf になる。カラースケールの計算に -inf が
    # 混じると vmax が壊れるので、非有限値はすべて NaN（＝描かない）に揃える。
    S_db = np.where(np.isfinite(S_db), S_db, np.nan)

    # 基準は氷なし計算に「同じゲート」を滑らせた包絡ピーク。時間分解した
    # 背景レベルになるので、水平線より正確に「氷で増えたぶん」を切り出せる。
    if info['noice_work'] is not None:
        nz_db = np.full(t_centers.size, np.nan)
        for i, tc in enumerate(t_centers):
            nzg, _ = gate_trace(info['noice_work'], dt, float(tc), hw)
            npk = measure_peak(nzg, dt)['amp_peak']
            if npk > 0:
                nz_db[i] = 20.0 * np.log10(npk / info['surface_amp'])
        snr_db = pk_db - nz_db
    else:
        nz_db = np.full(t_centers.size, np.nan)
        snr_db = np.full(t_centers.size, np.nan)
    usable = np.isfinite(snr_db) & (snr_db >= SPECTROGRAM_SNR_MIN_DB)
    return {'t_ns': t_centers, 'freq_ghz': f_ghz_all[fsel], 'S_db': S_db,
            'f_c': f_c, 'sigma_f': sig, 'peak_db': pk_db, 'snr_db': snr_db,
            'noice_db': nz_db, 'usable': usable,
            'halfwidth_ns': hw, 'norm_note': norm_note}


def _write_spectrogram_csv(sg, output_dir):
    """重心の追跡を CSV に出す（図から数値を読み取らずに済むように）。"""
    path = os.path.join(output_dir, 'spectrogram_track.csv')

    def _f(x, fmt='{:.6f}'):
        return '' if not np.isfinite(x) else fmt.format(x)

    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('t_ns,f_c_GHz,sigma_f_GHz,f_lo_GHz,f_hi_GHz,'
                 'peak_dB_re_surface,noice_dB_re_surface,'
                 'snr_vs_noice_dB,usable\n')
        for i, t in enumerate(sg['t_ns']):
            fc, sf = sg['f_c'][i], sg['sigma_f'][i]
            fh.write('{:.4f},{},{},{},{},{},{},{},{}\n'.format(
                t, _f(fc), _f(sf), _f(fc - sf), _f(fc + sf),
                _f(sg['peak_db'][i], '{:.4f}'), _f(sg['noice_db'][i], '{:.4f}'),
                _f(sg['snr_db'][i], '{:.4f}'), int(bool(sg['usable'][i]))))
    print('Saved:', path)


def plot_spectrogram(results, info, output_dir):
    """fig6: (a) 時間-周波数マップ＋重心の追跡 (b) 窓ごとの包絡ピーク。

    fig1 との違いは「イベントの位置を仮定しないこと」だけで、ゲートも
    スペクトルも重心の定義も fig1 と同一である。理論走時は水平線として
    重ねるが、これは答え合わせのための参考線であって、解析には使っていない。

    (b) を併置しているのは、重心の追跡線がどこで信用できるかを読み手が
    判断できるようにするため。反射がノイズフロアに埋もれている時刻でも
    重心そのものは必ず計算できてしまい、フロアの形を反映した無意味な値が
    出る。SNR がしきい値を下回る区間は追跡線を破線に落としてある。
    """
    sg = compute_spectrogram(results, info)
    t, f = sg['t_ns'], sg['freq_ghz']
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 9), sharey=True,
                             gridspec_kw={'width_ratios': [3, 1]})

    # --- (a) 時間-周波数マップ ---------------------------------------------
    if np.any(np.isfinite(sg['S_db'])):
        vmax = float(np.nanmax(sg['S_db']))
    else:
        raise CmdInputError(
            'スペクトログラムに有限の値がありません。トレースが空か、'
            'SPECTROGRAM_FREQ_RANGE_GHZ / SPECTROGRAM_NORM の設定を'
            '確認してください。')
    vmin = vmax - SPECTROGRAM_DB_RANGE
    mesh = axes[0].pcolormesh(f, t, sg['S_db'], cmap=SPECTROGRAM_CMAP,
                              vmin=vmin, vmax=vmax, shading='auto')
    cbar = fig.colorbar(mesh, ax=axes[0], pad=0.02, fraction=0.05)
    cbar.set_label('|E(f)| [{}]'.format(sg['norm_note']), fontsize=12)

    # 解析帯域の外は参考。境界を引いておく。
    for fb in BAND_GHZ:
        axes[0].axvline(fb, color='w', ls=':', lw=1.2, alpha=0.8)

    # 重心とスペクトル幅の追跡。信用できない区間は破線に落とす。
    ok = sg['usable'] & np.isfinite(sg['f_c'])
    faint = np.isfinite(sg['f_c'])

    def _smooth(a):
        """表示用の移動平均。NaN を保ったまま平均する。CSV には適用しない。"""
        if SPECTROGRAM_TRACK_SMOOTH_NS <= 0:
            return a
        w = max(1, int(round(SPECTROGRAM_TRACK_SMOOTH_NS
                             / SPECTROGRAM_STEP_NS)))
        if w < 2:
            return a
        good = np.isfinite(a).astype(float)
        filled = np.where(np.isfinite(a), a, 0.0)
        k = np.ones(w) / w
        num = np.convolve(filled, k, mode='same')
        den = np.convolve(good, k, mode='same')
        with np.errstate(invalid='ignore', divide='ignore'):
            out = num / den
        return np.where(den > 0, out, np.nan)

    fc_p, sg_p = _smooth(sg['f_c']), _smooth(sg['sigma_f'])
    for arr, style, lab in (
            (fc_p, dict(lw=2.2, color='w'), 'centroid $f_c$'),
            (fc_p - sg_p, dict(lw=1.2, color='w', ls='--', alpha=0.9),
             r'$f_c \pm \sigma_f$'),
            (fc_p + sg_p, dict(lw=1.2, color='w', ls='--', alpha=0.9), None)):
        axes[0].plot(np.where(faint, arr, np.nan), t, lw=style.get('lw', 1.5),
                     color='0.55', ls=':', alpha=0.9, zorder=3)
        axes[0].plot(np.where(ok, arr, np.nan), t, zorder=4,
                     label=lab, **style)

    # 理論走時（答え合わせ用の参考線。解析には使っていない）
    for r in results:
        axes[0].axhline(r['t_theory'], color=r['color'], ls='--', lw=1.3,
                        alpha=0.9)
        axes[0].annotate(r['name'], xy=(f[-1], r['t_theory']),
                         xytext=(-4, -4), textcoords='offset points',
                         ha='right', va='top', fontsize=10, color=r['color'],
                         bbox=dict(fc='w', ec='none', alpha=0.6, pad=1.0))
    axes[0].set_xlabel('Frequency [GHz]', fontsize=13)
    axes[0].set_ylabel('Delay time [ns]', fontsize=13)
    axes[0].set_title('(a) Time-frequency map (gate swept, no event picking)\n'
                      'gate halfwidth {:.2f} ns  ->  dt ~ {:.2f} ns, '
                      'df ~ {:.2f} GHz'
                      .format(sg['halfwidth_ns'], 2.0 * sg['halfwidth_ns'],
                              1.0 / (2.0 * sg['halfwidth_ns'])),
                      fontsize=13)
    axes[0].legend(fontsize=10, loc='upper right', framealpha=0.85)

    # --- (b) 窓ごとの包絡ピーク --------------------------------------------
    axes[1].plot(sg['peak_db'], t, color='k', lw=1.0, label='ice')
    if np.any(np.isfinite(sg['noice_db'])):
        axes[1].plot(sg['noice_db'], t, color='tab:green', lw=1.0, alpha=0.85,
                     label='no ice\n(track limit +{:.0f} dB)'
                     .format(SPECTROGRAM_SNR_MIN_DB))
        if SPECTROGRAM_SHOW_TRACKING_LIMIT:
            axes[1].plot(sg['noice_db'] + SPECTROGRAM_SNR_MIN_DB, t,
                         color='tab:green', ls='--', lw=1.0, alpha=0.6)
    for r in results:
        axes[1].axhline(r['t_theory'], color=r['color'], ls='--', lw=1.3,
                        alpha=0.9)
    axes[1].set_xlabel('Gated peak [dB re. surface peak]', fontsize=13)
    axes[1].set_title('(b) Amplitude in the\nsame gate', fontsize=13)
    # 有限値だけで下限を決める（信号が無い時刻は NaN になっているため）。
    both = np.concatenate([sg['peak_db'], sg['noice_db']])
    finite_pk = both[np.isfinite(both)]
    x_lo = min(-90.0, float(np.min(finite_pk))) if finite_pk.size else -90.0
    axes[1].set_xlim(x_lo, 5.0)
    axes[1].legend(fontsize=9, loc='lower left', framealpha=0.85)

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.minorticks_on()
        
    # 縦軸をスペクトログラムの計算範囲によらず、常に元データの全時間範囲へ固定する
    t_max_ns = (len(info['work']) - 1) * info['dt'] * 1e9
    axes[0].set_ylim(t_max_ns, 0.0)      # 時間は下向きに増やす（レーダグラム流儀）
    
    plt.tight_layout()
    save_figure(fig, output_dir, 'fig6_spectrogram')
    _write_spectrogram_csv(sg, output_dir)
    return sg


def write_outputs(results, info, level, kind, output_dir):
    import csv
    freq = info['freq']
    i_c = int(np.argmin(np.abs(freq - BAND_CENTRE_HZ)))
    names = [r['name'] for r in results]
    r0 = results[names.index(info['rel_ref'])]

    path = os.path.join(output_dir, 'events.csv')
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        # noice_* は氷なし計算を「同じ時間窓」で測った値。
        # snr_vs_noice_dB = amp_rel_surface_dB - noice_rel_surface_dB が検出量。
        w.writerow(['event', 'depth_m', 't_geom_ns', 't_theory_ns',
                    't_measured_ns', 'dt_ns', 'amp_peak', 'amp_rel_surface_dB',
                    'amp_theory_rel_surface_dB',
                    'noice_amp_peak', 'noice_rel_surface_dB',
                    'snr_vs_noice_dB',
                    'R_theory', 'R_measured', 'alpha_abs_1.25GHz',
                    'alpha_rel_1.25GHz', 'f_c_GHz', 'sigma_f_GHz'])
        for r in results:
            a_abs = alpha_from_absolute(r, level)
            a_rel = alpha_from_relative(r, r0, level)
            Rm = reflection_spectrum(r)[i_c]
            amp_db = 20 * np.log10(r['amp_peak'] / info['surface_amp'])
            if np.isfinite(r['noice_amp']) and r['noice_amp'] > 0:
                nz_db = 20 * np.log10(r['noice_amp'] / info['surface_amp'])
                nz_cell, snr_cell = '{:.6f}'.format(nz_db), '{:.6f}'.format(amp_db - nz_db)
                nz_amp = '{:.6e}'.format(r['noice_amp'])
            else:
                nz_amp = nz_cell = snr_cell = ''
            w.writerow([
                r['name'], r['depth_m'], r['t_geom'], r['t_theory'],
                r['t_measured'], r['t_measured'] - r['t_theory'],
                r['amp_peak'], amp_db,
                20 * np.log10(theory_amp(r, i_c) / theory_amp(results[0], i_c)),
                nz_amp, nz_cell, snr_cell,
                float(np.abs(r['terms']['R'][i_c])), float(Rm),
                '' if a_abs is None else float(a_abs[i_c]),
                '' if a_rel is None else float(a_rel[i_c]),
                r['moments_meas']['f_c'] / 1e9,
                r['moments_meas']['sigma_f'] / 1e9])
    print('Saved:', path)

    np.savez(os.path.join(output_dir, 'spectrum.npz'), freq_hz=freq,
             E_ref=info['E_ref'],
             **{'E_' + r['name']: r['E_meas'] for r in results},
             **{'Eth_' + r['name']: r['E_theory'] for r in results})

    path = os.path.join(output_dir, 'run_info.txt')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('at_tx reflection spectrum analysis\n')
        fh.write('  level: {}\n  kind: {}\n  json: {}\n'
                 .format(level, kind, JSON_PATH))
        fh.write('  gate: halfwidth {:.2f} ns, taper {}, center {}\n'
                 .format(GATE_HALFWIDTH_NS, GATE_TAPER, GATE_CENTER))
        fh.write('  background subtraction: {}\n'
                 .format(BACKGROUND_TRACE_PATH or 'off'))
        fh.write('  relative LSR reference event: {}\n'.format(info['rel_ref']))
        if 'ice_layer' in LEVEL_EFFECTS[level]:
            fh.write('  ice model: {}\n'.format(sm.LEVEL4_ICE_MODEL))
        fh.write('  source delay (from theory trace): {:.3f} ns\n'
                 .format(info['source_delay']))
        fh.write('  detection reference (no-ice run): {}\n'
                 .format(info['noice_src']))
        if 'absorb_const' in LEVEL_EFFECTS[level]:
            fh.write('  medium: {}\n'.format(
                describe_level2_medium(refractive_index(freq, level))))
        if 'absorb_tandelta' in LEVEL_EFFECTS[level]:
            fh.write('  medium: {}\n'.format(describe_level3_medium()))
        if 'absorb_debye' in LEVEL_EFFECTS[level]:
            fh.write('  medium: {}\n'.format(describe_level3b_medium()))
        if 'ice_layer' in LEVEL_EFFECTS[level]:
            fh.write('  ice layer: {}\n'.format(describe_level4_medium()))
        fh.write('\n  event      d[m]  t_geom[ns] t_th[ns]  t_meas[ns]  amp[dB]\n')
        for r in results:
            fh.write('  {:10s} {:5.2f} {:9.3f} {:8.3f}  {:9.3f}  {:+8.2f}\n'
                     .format(r['name'], r['depth_m'], r['t_geom'],
                             r['t_theory'], r['t_measured'],
                             20 * np.log10(r['amp_peak']
                                           / results[0]['amp_peak'])))
    print('Saved:', path)


def gate_sensitivity(at_tx_path, ref_path, level, freespace_path=''):
    """ゲート幅を振って alpha がどれだけ動くかを確認する（README §4.4）。"""
    if not GATE_SWEEP_NS:
        return
    global GATE_HALFWIDTH_NS
    keep = GATE_HALFWIDTH_NS
    print('\n--- ゲート幅の感度確認 ---')
    print('  halfwidth[ns]  event        alpha@1.25GHz (relative LSR)')
    for hw in GATE_SWEEP_NS:
        GATE_HALFWIDTH_NS = hw
        res, info = analyze(at_tx_path, ref_path, level, freespace_path,
                            BACKGROUND_TRACE_PATH,
                            resolve_noice_path(level))
        names = [r['name'] for r in res]
        r0 = res[names.index(info['rel_ref'])]
        i_c = int(np.argmin(np.abs(info['freq'] - BAND_CENTRE_HZ)))
        for r in res:
            a = alpha_from_relative(r, r0, level)
            if a is not None:
                print('  {:12.2f}  {:12s} {:.6f}'
                      .format(hw, r['name'], float(a[i_c])))
    GATE_HALFWIDTH_NS = keep


# =============================================================================
# 9. main
# =============================================================================
def run_once(at_tx_path, ref_path, level, kind, output_dir,
             freespace_path=''):
    """1 条件ぶんの解析と作図。差分あり／なしで 2 回呼ぶ。"""
    os.makedirs(output_dir, exist_ok=True)
    results, info = analyze(at_tx_path, ref_path, level, freespace_path,
                            BACKGROUND_TRACE_PATH,
                            resolve_noice_path(level))

    tag = '直達波の差分あり' if info['subtracted'] else '直達波の差分なし（生データ）'
    print('\n=== {} -> {}'.format(tag, output_dir))
    print('  検出の基準（氷なし計算）: {}'.format(info['noice_src']))
    print('  波源遅延（理論トレースの包絡ピークから）: {:.3f} ns'
          .format(info['source_delay']))
    i_c = int(np.argmin(np.abs(info['freq'] - BAND_CENTRE_HZ)))
    print('  event      d[m]  t_th[ns]  t_meas[ns]  amp_meas  amp_th   diff   SNR   判定')
    print('                                          [dB]      [dB]     [dB]   [dB]')
    for r in results:
        amp_db = 20 * np.log10(r['amp_peak'] / results[0]['amp_peak'])
        th_db = 20 * np.log10(theory_amp(r, i_c) / theory_amp(results[0], i_c))
        if np.isfinite(r['noice_amp']) and r['noice_amp'] > 0:
            snr = amp_db - 20 * np.log10(r['noice_amp'] / results[0]['amp_peak'])
        else:
            snr = np.nan
        print('  {:10s} {:5.2f} {:9.3f} {:11.3f} {:+9.2f} {:+8.2f} {:+7.2f} {:+6.1f}  {}'
              .format(r['name'], r['depth_m'], r['t_theory'], r['t_measured'],
                      amp_db, th_db, amp_db - th_db, snr,
                      '-' if not np.isfinite(snr) else judge_snr(snr)))
    # ゲートによる損失を切り分けるため、理論トレースを同じゲートに通した値も出す。
    print('  （参考）理論トレースを同じゲートに通したときの振幅:')
    for r in results:
        g_db = 20 * np.log10(
            measure_peak(np.fft.irfft(r['E_theory'], n=len(info['work'])),
                         info['dt'])['amp_peak']
            / measure_peak(np.fft.irfft(results[0]['E_theory'],
                                        n=len(info['work'])),
                           info['dt'])['amp_peak'])
        print('    {:10s} {:+8.2f} dB'.format(r['name'], g_db))

    plot_trace(results, info, output_dir, overlay_noice=True,
               stem='fig0_trace_noice')
    if PLOT_WITHOUT_NOICE:
        plot_trace(results, info, output_dir, overlay_noice=False,
                   stem='fig0_trace')
    plot_spectra(results, info, output_dir)
    plot_lsr(results, info, output_dir)
    plot_attenuation(results, info, level, output_dir)
    plot_phase(results, info, output_dir)
    if PLOT_REFLECTION_SPECTRUM:
        plot_reflection(results, info, output_dir)
    if PLOT_SPECTROGRAM:
        plot_spectrogram(results, info, output_dir)
    write_outputs(results, info, level, kind, output_dir)
    return results, info


def main():
    level, kind, rx_paths, reference = load_paths(JSON_PATH)

    # 組成・水氷濃度・水氷の描像（pore / excess）をまとめて設定する。
    # サブ階層キーから自動判定し、判定できなければエラーで止まる。
    for _note in configure_from_kind(kind, level):
        print(_note)

    if level not in IMPLEMENTED_LEVELS:
        raise NotImplementedError('{} は未実装です'.format(level))

    if AT_TX_KEY not in rx_paths:
        raise CmdInputError(
            '選択した階層に "{}" がありません（本ツールは at_tx 専用です）。\n'
            '利用可能な rx: {}'.format(AT_TX_KEY, ', '.join(sorted(rx_paths))))
    check_paths_exist({AT_TX_KEY: rx_paths[AT_TX_KEY]}, reference)

    at_tx_path = rx_paths[AT_TX_KEY]
    ref_path = reference if isinstance(reference, str) else reference[REF_KEY]

    # --- 自由空間 at_tx（直達波の除去用）を決める -------------------------
    fs_path = ''
    if SUBTRACT_FREESPACE:
        fs_path = FREESPACE_AT_TX_PATH
        if not fs_path and isinstance(reference, dict):
            fs_path = reference.get(AT_TX_KEY, '')
        if not fs_path:
            raise CmdInputError(
                '直達波の除去に使う自由空間 at_tx が見つかりません。\n'
                'JSON の _reference に at_tx を用意するか、'
                'FREESPACE_AT_TX_PATH に .out のパスを設定してください。\n'
                '（除去しない場合は SUBTRACT_FREESPACE = False）')
        if not os.path.exists(fs_path):
            raise CmdInputError('自由空間 at_tx が存在しません: {}'.format(fs_path))
        print('直達波の除去に使う自由空間 at_tx: {}'.format(fs_path))

    asp.OUTPUT_SUBDIRNAME = OUTPUT_SUBDIRNAME     # 出力先だけ差し替える
    output_dir = asp.resolve_output_dir(level, rx_paths)

    # --- メイン：直達波を差し引いたもの -----------------------------------
    run_once(at_tx_path, ref_path, level, kind, output_dir, fs_path)

    # --- サブ：差分なし（元データの姿を残すため）--------------------------
    if fs_path:
        run_once(at_tx_path, ref_path, level, kind,
                 os.path.join(output_dir, NO_SUBTRACTION_SUBDIR), '')

    gate_sensitivity(at_tx_path, ref_path, level, fs_path)


if __name__ == '__main__':
    main()