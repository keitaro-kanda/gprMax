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

import ascan_spectrum as asp
from ascan_spectrum import (
    C, TX_HEIGHT, R_REF, BAND_GHZ, BAND_CENTRE_HZ, FLOHI_PRIMARY_DB,
    LEVEL_EFFECTS, IMPLEMENTED_LEVELS,
    load_paths, check_paths_exist, load_trace, spectrum, measure_peak,
    _interp_complex_to_grid, save_figure,
    set_level3_composition, set_level4_ice,
    describe_level2_medium, describe_level3_medium, describe_level3b_medium,
    describe_level4_medium,
    refractive_index, level4_eps, level4_alpha, level3_eps, level3_alpha,
    level2_alpha,
    moments, lo_hi_freq, log_spectral_ratio, alpha_to_tandelta,
    valid_mask, noise_floor, group_delay,
    LEVEL4_ICE_TOP_M, LEVEL4_ICE_THICK_M,
)
from gprMax.exceptions import CmdInputError

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
# そこで RMS を取ると「数値ノイズ」ではなく「励振波形のサイドローブレベル」を
# 測ってしまう（実測で -45 dB。静かな区間の実際の値は -80 dB 前後）。
#
# 対策は 3 つ。
#   (1) 推定量を分位点にする。ローブの山に引きずられない。
#   (2) 界面のないトレース（Level 3 の at_tx など）で測れるようにする。
#   (3) 時間分解したフロアを fig0(b) に重ね、どの時間帯が何に支配されているかを
#       目で見えるようにする。
NOISE_WINDOW_NS = (8.0, 40.0)      # 地表反射より後の、界面反射がない時間帯
NOISE_ESTIMATOR = 'percentile'     # 'percentile'（推奨）/ 'rms'（旧挙動）
NOISE_PERCENTILE = 10.0            # 分位点 [%]。包絡の下側を拾う
NOISE_PERCENTILE_TO_RMS = True     # 分位点を RMS 相当に換算するか（下記）
# 純雑音の包絡は Rayleigh 分布に従うので、p 分位点は sigma*sqrt(-2 ln(1-p)) に
# なる（p=10% で 0.459 sigma）。一方 RMS は sigma。換算しないと分位点のほうが
# 6.8 dB 低く出て、2 つの推定量を並べたときに比較できない。換算すると
# 「純雑音なら RMS と一致し、イベントのローブがあるときだけ低く出る」
# という素直な量になる。
NOISE_ROLL_NS = 2.0                # 時間分解フロアの移動窓幅 [ns]
NOISE_TRACE_PATH = ''              # 界面のないトレース（Level 3 の at_tx など）の
                                   # .out。指定するとこちらでフロアを測る。
                                   # 空なら解析対象トレース自身で測る（参考値）。

# --- 検出判定（修正 6）------------------------------------------------------
# 旧版はフロアの絶対値（-70/-60/-55 dB）で判定していたが、これは「地表反射比」
# という基準量に依存するため、直達波を差し引くと基準が 22 dB 変わって判定が
# ひっくり返った。判定は基準量に依らない SNR で行う。
SNR_DETECT_DB = 12.0               # これ以上なら検出可
SNR_MARGINAL_DB = 6.0              # これ以上なら限界

# --- 相対 LSR の基準イベント -------------------------------------------------
# 既定は最も浅い地下界面。地表反射を基準にすると地表の往復透過が残るため、
# 実測可能性の観点では地下界面どうしで取るほうが素直。
REL_LSR_REF_EVENT = 'ice_top'

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
    if 'ice_layer' in LEVEL_EFFECTS[level]:
        top = float(LEVEL4_ICE_TOP_M)
        bot = top + float(LEVEL4_ICE_THICK_M)
        events.append({'name': 'ice_top', 'depth_m': top,
                       'above': [(top, False)]})
        events.append({'name': 'ice_bottom', 'depth_m': bot,
                       'above': [(top, False), (bot - top, True)]})
    return events


def _n_of(f, level, in_ice):
    """層の屈折率 n(f)。氷層かどうかで切り替える。"""
    if 'ice_layer' in LEVEL_EFFECTS[level]:
        return np.sqrt(level4_eps(f, in_ice)[0])
    return refractive_index(f, level)


def _alpha_of(f, level, in_ice):
    """層の減衰係数 alpha(f) [Np/m]。"""
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

    戻り値の辞書:
      'G'     幾何      sqrt(R_REF / r_eff)
      'T'     往復透過  Π 4 n_a n_b/(n_a+n_b)^2（界面より浅い界面すべて）
      'R'     反射      (n_k - n_k+1)/(n_k + n_k+1)
      'A'     吸収      exp(-2 Σ alpha_j L_j)
      't_ns'  往復走時  2h/c + 2 Σ n_j L_j / c   （帯域中心での代表値）
      'r_eff' 見かけ源距離
    項を分けて返すのは、LSR から alpha を逆算するときに G・T・R を差し引く
    必要があるため（README §3.3）。
    """
    f_arr = np.asarray(f, dtype=float)
    n_vac = np.ones_like(f_arr)
    n_reg = _n_of(f_arr, level, False)
    n_ice = _n_of(f_arr, level, True)

    def n_layer(in_ice):
        return n_ice if in_ice else n_reg

    # --- 見かけ源距離：r_eff = 2h + 2 Σ L_j / n_j -----------------------------
    r_eff = 2.0 * TX_HEIGHT * np.ones_like(f_arr)
    for length, in_ice in event['above']:
        r_eff = r_eff + 2.0 * length / n_layer(in_ice)
    G = np.sqrt(R_REF / r_eff)

    # --- 往復透過：界面より浅い界面をすべて往復する ---------------------------
    T = np.ones_like(f_arr)
    n_prev = n_vac
    for _, in_ice in event['above']:
        n_cur = n_layer(in_ice)
        T = T * (4.0 * n_prev * n_cur / (n_prev + n_cur) ** 2)
        n_prev = n_cur

    # --- 反射係数：界面の直上／直下の屈折率から ------------------------------
    if event['name'] == 'surface':
        n_a, n_b = n_vac, n_reg
    elif event['name'] == 'ice_top':
        n_a, n_b = n_reg, n_ice
    elif event['name'] == 'ice_bottom':
        n_a, n_b = n_ice, n_reg
    else:
        raise CmdInputError('未知のイベント: {}'.format(event['name']))
    R = (n_a - n_b) / (n_a + n_b)

    # --- 吸収と走時：往復なので 2 倍 -----------------------------------------
    att = np.zeros_like(f_arr)
    t_f = 2.0 * TX_HEIGHT / C * np.ones_like(f_arr)
    for length, in_ice in event['above']:
        att = att + _alpha_of(f_arr, level, in_ice) * length
        t_f = t_f + 2.0 * n_layer(in_ice) * length / C
    A = np.exp(-2.0 * att)

    i_c = int(np.argmin(np.abs(f_arr - BAND_CENTRE_HZ)))
    return {'G': G, 'T': T, 'R': R, 'A': A, 'r_eff': r_eff,
            't_ns': float(t_f[i_c]), 't_f': t_f}


def event_arrival_ns(event, level):
    """包絡ピークに対応する群走時 [ns]（スカラー）。"""
    fc = np.array([BAND_CENTRE_HZ])
    t = 2.0 * TX_HEIGHT / C
    for length, in_ice in event['above']:
        ng = float(asp._group_index_from_eps(
            lambda ff: (level4_eps(ff, in_ice)[0]
                        if 'ice_layer' in LEVEL_EFFECTS[level]
                        else level3_eps(ff)[0]), fc)[0])
        t += 2.0 * ng * length / C
    return t


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
# 4. ノイズフロア（README §4.3）
# =============================================================================
def rolling_floor(trace, dt, width_ns=None):
    """包絡の移動分位点。時間分解したノイズフロアを返す（fig0(b) 用）。

    どの時間帯が数値ノイズに支配され、どこがイベントのサイドローブに
    支配されているかを目で見られるようにするための量。
    """
    w = NOISE_ROLL_NS if width_ns is None else width_ns
    env = np.abs(signal.hilbert(trace))
    n = max(3, int(round(w * 1e-9 / dt)))
    if len(env) <= n:
        return env
    from numpy.lib.stride_tricks import sliding_window_view
    val = np.percentile(sliding_window_view(env, n), NOISE_PERCENTILE, axis=1)
    if NOISE_PERCENTILE_TO_RMS:
        val = val / np.sqrt(-2.0 * np.log(1.0 - NOISE_PERCENTILE / 100.0))
    pad = len(env) - len(val)
    return np.pad(val, (pad // 2, pad - pad // 2), mode='edge')


def measure_noise_floor(trace, dt, surface_amp, exclude_ns=()):
    """ノイズフロアを (絶対値, 地表反射ピークに対する dB) で返す。

    NOISE_ESTIMATOR = 'percentile'（既定）
        窓内の包絡の下側分位点。イベントのサイドローブの山に引きずられない。
    NOISE_ESTIMATOR = 'rms'
        窓内の波形 RMS（旧挙動）。サイドローブを拾うので過大評価になる。

    exclude_ns に既知のイベント時刻を渡すと、その前後 ±(ゲート半幅 + 1 ns) を
    窓から除く。ただしサイドローブはこれよりずっと遠くまで伸びるので、
    除外だけでは足りない。正しくは界面のないトレース（NOISE_TRACE_PATH）で
    測ること（README §4.3）。
    """
    dt_ns = dt * 1e9
    t_axis = np.arange(len(trace)) * dt_ns
    lo, hi = NOISE_WINDOW_NS
    keep = (t_axis >= lo) & (t_axis <= hi)
    for t0 in exclude_ns:
        keep &= np.abs(t_axis - t0) > (GATE_HALFWIDTH_NS + 1.0)
    idx = np.where(keep)[0]
    if len(idx) == 0 or not surface_amp > 0:
        return np.nan, np.nan
    if NOISE_ESTIMATOR == 'rms':
        val = float(np.sqrt(np.mean(trace[idx] ** 2)))
    else:
        env = np.abs(signal.hilbert(trace))
        val = float(np.percentile(env[idx], NOISE_PERCENTILE))
        if NOISE_PERCENTILE_TO_RMS:
            val /= np.sqrt(-2.0 * np.log(1.0 - NOISE_PERCENTILE / 100.0))
    return val, 20.0 * np.log10(val / surface_amp)


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
            background_path='', noise_path=''):
    """at_tx トレースを読み、イベントごとのスペクトルと理論を突き合わせる。

    freespace_path : 自由空間 at_tx。指定すると直達波を差し引く（修正 2）。
    background_path: 氷なし at_tx。指定すると背景差分を行う（実機では不可）。
    noise_path     : 界面のない at_tx（Level 3 など）。指定するとフロアを
                     こちらで測る。イベントのサイドローブに汚染されないため、
                     こちらが正式な測り方（修正 6）。
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

    # 地表反射（at_tx では直達波を含む）のピーク振幅。ノイズフロアの基準。
    g_surf, _ = gate_trace(trace, dt, prepared[0]['t_theory'])
    surf_amp = measure_peak(g_surf, dt)['amp_peak']
    # フロアは界面のないトレースで測るのが正式（修正 6）。
    # 指定がなければ解析対象自身で測るが、その場合はイベントのサイドローブが
    # 混じるため過大評価になる（参考値）。
    if noise_path:
        nz = _load_and_align(noise_path, dt, len(raw_trace), 'ノイズ測定用 at_tx')
        if fs_trace is not None:
            nz = nz - fs_trace
        nf_val, nf_db = measure_noise_floor(nz, dt, surf_amp)
        nf_src = noise_path
        nf_trace = nz
    else:
        nf_val, nf_db = measure_noise_floor(
            trace, dt, surf_amp, exclude_ns=[p['t_theory'] for p in prepared])
        nf_src = '(解析対象自身。イベントのサイドローブを含む参考値)'
        nf_trace = trace

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
            'noise_val': nf_val, 'noise_db': nf_db, 'noise_src': nf_src,
            'noise_trace': nf_trace, 'surface_amp': surf_amp,
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


def plot_trace(results, info, output_dir):
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
    # 時間分解したフロア（包絡の移動分位点）。どの時間帯が数値ノイズに支配され、
    # どこがイベントのサイドローブに支配されているかが読み取れる（修正 6）。
    with np.errstate(divide='ignore'):
        roll_db = 20.0 * np.log10(
            rolling_floor(info['noise_trace'], info['dt']) / info['surface_amp'])
    axes[1].plot(t[:len(roll_db)], roll_db, color='m', lw=1.0, alpha=0.8,
                 label='rolling floor ({:.0f}th pct, {:.1f} ns)'
                 .format(NOISE_PERCENTILE, NOISE_ROLL_NS))
    if np.isfinite(info['noise_db']):
        axes[1].axhline(info['noise_db'], color='m', ls=':', lw=1.5,
                        label='noise floor {:.1f} dB'.format(info['noise_db']))
        axes[1].axvspan(NOISE_WINDOW_NS[0], NOISE_WINDOW_NS[1],
                        color='m', alpha=0.06)
    axes[1].legend(fontsize=9, loc='upper right')
    axes[1].set_xlabel('Time [ns]', fontsize=13)
    axes[1].set_ylabel('Envelope [dB re. surface peak]', fontsize=13)
    axes[1].set_title('(b) Envelope, theory (circles) and noise floor',
                      fontsize=13)
    axes[1].set_ylim(-90, 5)
    for ax in axes:
        ax.grid(alpha=0.4)
        ax.minorticks_on()
    fig.legend(handles=_event_handles(results), loc='upper center', ncol=4,
               fontsize=11, bbox_to_anchor=(0.5, 0.0), frameon=True)
    plt.tight_layout()
    save_figure(fig, output_dir, 'fig0_trace')


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
    if np.isfinite(info['noise_db']):
        ax.axhline(info['noise_db'], color='m', ls=':', lw=1.2,
                   label='noise floor')
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
             'Absolute LSR (ref = far_1m)', '(a)', '(b)'),
            ((1, 0), (1, 1), 'L_rel_meas', 'L_rel_theory',
             'Relative LSR (ref = {})'.format(info['rel_ref']), '(c)', '(d)')]
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
def write_outputs(results, info, level, kind, output_dir):
    import csv
    freq = info['freq']
    i_c = int(np.argmin(np.abs(freq - BAND_CENTRE_HZ)))
    names = [r['name'] for r in results]
    r0 = results[names.index(info['rel_ref'])]

    path = os.path.join(output_dir, 'events.csv')
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['event', 'depth_m', 't_geom_ns', 't_theory_ns',
                    't_measured_ns', 'dt_ns', 'amp_peak', 'amp_rel_surface_dB',
                    'amp_theory_rel_surface_dB',
                    'R_theory', 'R_measured', 'alpha_abs_1.25GHz',
                    'alpha_rel_1.25GHz', 'f_c_GHz', 'sigma_f_GHz'])
        for r in results:
            a_abs = alpha_from_absolute(r, level)
            a_rel = alpha_from_relative(r, r0, level)
            Rm = reflection_spectrum(r)[i_c]
            w.writerow([
                r['name'], r['depth_m'], r['t_geom'], r['t_theory'],
                r['t_measured'], r['t_measured'] - r['t_theory'],
                r['amp_peak'],
                20 * np.log10(r['amp_peak'] / results[0]['amp_peak']),
                20 * np.log10(theory_amp(r, i_c) / theory_amp(results[0], i_c)),
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
        fh.write('  source delay (from theory trace): {:.3f} ns\n'
                 .format(info['source_delay']))
        fh.write('  noise floor: {:.2f} dB re. surface peak ({}, {})\n'
                 .format(info['noise_db'], NOISE_ESTIMATOR, info['noise_src']))
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
                            BACKGROUND_TRACE_PATH, NOISE_TRACE_PATH)
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
                            BACKGROUND_TRACE_PATH, NOISE_TRACE_PATH)

    tag = '直達波の差分あり' if info['subtracted'] else '直達波の差分なし（生データ）'
    print('\n=== {} -> {}'.format(tag, output_dir))
    print('  ノイズフロア: {:.2f} dB re. surface peak  ({}, {})'
          .format(info['noise_db'], NOISE_ESTIMATOR, info['noise_src']))
    print('  波源遅延（理論トレースの包絡ピークから）: {:.3f} ns'
          .format(info['source_delay']))
    i_c = int(np.argmin(np.abs(info['freq'] - BAND_CENTRE_HZ)))
    print('  event      d[m]  t_th[ns]  t_meas[ns]  amp_meas  amp_th   diff   SNR   判定')
    print('                                          [dB]      [dB]     [dB]   [dB]')
    for r in results:
        amp_db = 20 * np.log10(r['amp_peak'] / results[0]['amp_peak'])
        th_db = 20 * np.log10(theory_amp(r, i_c) / theory_amp(results[0], i_c))
        snr = amp_db - info['noise_db']
        print('  {:10s} {:5.2f} {:9.3f} {:11.3f} {:+9.2f} {:+8.2f} {:+7.2f} {:+6.1f}  {}'
              .format(r['name'], r['depth_m'], r['t_theory'], r['t_measured'],
                      amp_db, th_db, amp_db - th_db, snr, judge_snr(snr)))
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

    plot_trace(results, info, output_dir)
    plot_spectra(results, info, output_dir)
    plot_lsr(results, info, output_dir)
    plot_attenuation(results, info, level, output_dir)
    plot_phase(results, info, output_dir)
    if PLOT_REFLECTION_SPECTRUM:
        plot_reflection(results, info, output_dir)
    write_outputs(results, info, level, kind, output_dir)
    return results, info


def main():
    level, kind, rx_paths, reference = load_paths(JSON_PATH)

    if 'absorb_tandelta' in LEVEL_EFFECTS.get(level, []):
        wt, key = set_level3_composition(kind)
        print('背景レゴリスの組成: FeO+TiO2 = {:.1f} wt%  [{}]'.format(wt, key))
    if 'ice_layer' in LEVEL_EFFECTS.get(level, []):
        vol, ice_key = set_level4_ice(kind)
        print('水氷濃度: {:.2f} vol%  [{}]'.format(vol, ice_key))

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