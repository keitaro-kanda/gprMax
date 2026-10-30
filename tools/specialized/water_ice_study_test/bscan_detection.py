"""at_tx B-scan の検出可否解析（Level 6 以降）

対象の B-scan（例 pore_ice / excess_ice）と、反射体のない基準の B-scan
（no_ice）を同じ測線・同じ処理で比べ、次の 3 つが「見えるか / 見えないか」を
判定する。設計書は bscan_detection.md。

  1. 反射        : 基準のクラッタより強い反射が現れるか（水氷と決め打ちしない）
  2. 減衰率の変化 : 帯域パワーの時間減衰率 d(dB)/dt が、基準のばらつきを超えて変わるか
  3. 重心周波数のシフトレートの変化 : df_c/dt が、基準のばらつきを超えて変わるか

--- 判定の約束 ----------------------------------------------------------------
* クラッタ基準は誤報率 P_FA = 5 %。
    反射      : 基準の包絡ピークの分布の 95 % 点をしきい値にする（片側）
    減衰率・シフトレート : 基準のブロック間のばらつきの両側 5 %（平均 ± 1.96σ）
* 検出確率 P_D = 対象のトレース（反射）またはブロック（レート）のうち、
  クラッタ基準の外に出た割合。P_D >= PD_VISIBLE（既定 0.5）なら「見える」。
* 基準と対象は同じランダム場（記録ファイルの rho_map_sha256 で確認）。
  判定には「対象 vs 基準の分布」（実機と同じ立場。同じ場所の氷なしデータは
  実機では得られない）を使い、「同じ場所どうしの差」は変化の大きさの参考
  （理想的な上限）としてだけ出す。

--- 既存コードとの共有 ---------------------------------------------------------
JSON の選択・参照計算・トレースの読み込み・作図の保存は ascan_spectrum.py から、
媒質の物理（界面の走時、tan_delta）は subsurface_model.py から import する。
定数はコピーしない。理論は「対象の地図に印を付ける」ためだけに使い、判定は
データ（対象と基準）だけで行う。
"""

import csv
import glob
import json
import os
import warnings
from datetime import datetime

import numpy as np
import h5py
from scipy import signal, ndimage, stats
from scipy import fft as sp_fft
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import subsurface_model as sm
import ascan_spectrum as asp
from ascan_spectrum import load_trace, measure_peak
from gprMax.exceptions import CmdInputError

# =============================================================================
# 設定  [EDIT HERE]
# =============================================================================
JSON_PATH = '/Volumes/gprMax_water_ice/2D_12x4/water_ice_study_test/out_file_paths.json'
BSCAN_KEY = 'bscan'                # 各条件の B-scan（merged.out）のキー
REF_KEY = 'far_1m'                 # 参照計算（波源遅延・dB の基準の確認に使う）
FREESPACE_KEY = 'at_tx'            # 自由空間 at_tx（直達波の除去用）
REFERENCE_CONDITION = 'no_ice'     # 反射体のない基準の条件キー
TARGET_CONDITIONS = None           # None なら基準以外のすべての条件を順に解析
OUTPUT_SUBDIRNAME = 'bscan_detection'
# 'per_bscan'：対象の merged.out と同じフォルダに <OUTPUT_SUBDIRNAME>/ を作る
# 'level_root'：レベル親ディレクトリの analysis/<OUTPUT_SUBDIRNAME>/<対象>/（A-scan ツールと同じ）
OUTPUT_LOCATION = 'per_bscan'
# その下に 2 種類の出力を分けて置く。PNG 以外の形式（PDF）は、それぞれの下の <形式名>/ に入る
#   vs_<基準>/    ：基準（no_ice）との比較。判定つき
#   target_only/ ：対象（非 no_ice）だけの図と数値。判定なし
SINGLE_SUBDIR = 'target_only'
FIGURE_FORMATS = ('png', 'pdf')
FIGURE_DPI = 300
RECORD_NAME = 'Ascan_level6_record.json'   # Ascan.in --geometry-only の記録ファイル
THEORY_LEVEL = 'Level_5'           # 平均媒質（ランダム性を除いた媒質）のレベル

SUBTRACT_FREESPACE = True          # 直達波を除去する（既定 ON。既存ツールと同じ）
FREESPACE_AT_TX_PATH = ''          # 空なら JSON の _reference の at_tx を使う
TRACE_X_RANGE_M = None             # 解析に使うトレースの x 範囲 (始点, 終点) [m]。None で全部

# --- 判定 ---------------------------------------------------------------------
P_FA = 0.05                        # クラッタ基準（誤報率）
PD_VISIBLE = 0.5                   # この検出確率以上なら「見える」

# --- 反射チャネル ---------------------------------------------------------------
REFL_GATE_HW_NS = 1.0              # 包絡ピークを探すゲートの半幅（パルス幅 約 0.7 ns）
REFL_STEP_NS = 0.05                # ゲート中心の刻み
REFL_POOL_NS = 0.5                 # しきい値を決めるとき、前後この範囲の基準も含める

# --- 減衰率・シフトレート（スペクトログラム）-------------------------------------
SPEC_HW_NS = 2.0                   # 窓の半幅（既存 at_tx ツールのゲートと同じ）
SPEC_TAPER = 0.5                   # Tukey 窓のテーパー
SPEC_STEP_NS = 0.2                 # 窓中心の刻み
SPEC_NFFT_MIN = 4096
BLOCK_TRACES = 9                   # 1 ブロック（1 測定点）にまとめるトレース数（0.45 m）
RATE_WIN_NS = 4.0                  # 時間分解したレートの回帰窓
GUARD_EXTRA_NS = 0.5               # 界面・反射の前後で窓から除く幅 = SPEC_HW_NS + これ
SURFACE_GUARD_NS = 5.0             # 地表反射の後ろはさらに長く除く（強い地表反射の窓の裾）
BAND_METHOD = 'normal'             # 'normal'：平均 ± z σ / 'percentile'：経験分位点

# --- 作図 ---------------------------------------------------------------------
IMG_DB_RANGE = 80.0                # B-scan 図の色の幅 [dB]
DIFF_CMAP = 'RdBu_r'               # fig1(c)（包絡の差 [dB]）の色。0 dB が白、赤が対象のほうが強い
DIFF_DB_RANGE = 20.0               # fig1(c) の色の範囲 ±[dB]
PLOT_BSCAN_ASCAN = True            # B-scan と平均 A-scan を縦軸（two-way time）共有で並べた図も出す
COL_REF = '0.35'                   # 基準（no_ice）
COL_TGT = 'tab:red'                # 対象
COL_EVENT = 'tab:blue'             # 平均媒質の界面の理論走時
DB20 = 20.0 / np.log(10.0)



def envelope(D):
    """包絡（解析信号の絶対値）を、時間軸（axis 0）方向にゼロ詰めしてから求める。

    scipy.signal.hilbert は FFT で計算するので、そのままだとトレースの末尾と先頭が
    つながった「円環」として扱われ、末尾（深い時間）の内容が先頭（0 ns 付近）の包絡に
    漏れ込む。トレース長以上のゼロを足してから計算し、元の長さに切り戻す。
    """
    n = D.shape[0]
    return np.abs(signal.hilbert(D, N=sp_fft.next_fast_len(2 * n), axis=0))[:n]

# =============================================================================
# 1. 入力
# =============================================================================
def select_conditions(json_path):
    """JSON からレベルと下位の階層を選び、条件（pore_ice / no_ice ...）の辞書を返す。

    条件の階層は「子がすべて {キー: パス} の辞書」になっている階層とみなす。
    戻り値: level, 選んだキーのリスト, {条件: {キー: パス}}, 参照計算の辞書
    """
    if not os.path.exists(json_path):
        raise CmdInputError('JSON file {} does not exist'.format(json_path))
    with open(json_path) as fh:
        all_paths = json.load(fh)
    print('パス JSON: {}'.format(json_path))
    bad = [k for k, v in all_paths.items() if not k.startswith('_') and not isinstance(v, dict)]
    if bad or not all_paths:
        raise CmdInputError(
            '{} はパス JSON の形になっていません（値が辞書でないキー: {}）。'
            'JSON_PATH が out_file_paths.json を指しているか確認してください。'
            .format(json_path, ', '.join(bad) or '(空)'))
    levels = [k for k in all_paths if not k.startswith('_')]
    level = asp._select(levels, 'レベル')
    node, chosen = all_paths[level], []
    while True:
        entries = asp._kind_layer(node)
        if entries is None:
            raise CmdInputError('{} の下に条件の階層（{} など）が見つかりません'
                                .format(level, REFERENCE_CONDITION))
        if all(asp._kind_layer(v) is None for v in entries.values()):
            break                                         # ここが条件の階層
        key = asp._select(sorted(entries), 'サブ条件')
        chosen.append(key)
        node = entries[key]
    conds = {k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')}
             for k, v in entries.items()}
    reference = asp._pick_reference(all_paths.get('_reference', {}), chosen)
    lref = all_paths[level].get('_reference', {})
    if lref:
        reference.update(asp._pick_reference(lref, chosen))
    return level, chosen, conds, reference


def has_ice(rec):
    """記録ファイルの条件に、実際に氷が入っているか。

    enable_ice = True でも ice_vol_frac = 0 なら、混合則（pore / excess）で
    誘電率は乾燥レゴリスと同じになり、物理的には氷なしと同じ。
    """
    return bool(rec.get('enable_ice')) and float(rec.get('ice_vol_frac', 0.0)) > 0.0


def find_record(path):
    """B-scan に対応する記録ファイルを探す。

    記録は A-scan の計算（モデル作成）のときに作られ、次のように B-scan と並ぶ:
        <条件>/<ランダムパラメータ>/Ascan/Ascan_level6_record.json
        <条件>/<ランダムパラメータ>/Bscan/Bscan_merged.out
    探す順:
      1. B-scan のフォルダ
      2. 1 つ上（<ランダムパラメータ>/）と、その直下のフォルダ（Ascan/ など）
      3. 2 つ上（<条件>/）。ここでは直下のフォルダは探さない
         （別のランダムパラメータ ax010_... / ax020_... の記録を拾わないため）
    同じ階層で複数見つかったら、取り違えを避けるため止める。
    """
    bdir = os.path.dirname(os.path.abspath(path))
    parent = os.path.dirname(bdir)
    steps = [[os.path.join(bdir, RECORD_NAME)],
             [os.path.join(parent, RECORD_NAME), os.path.join(parent, '*', RECORD_NAME)],
             [os.path.join(os.path.dirname(parent), RECORD_NAME)]]
    for patterns in steps:
        hits = sorted({h for p in patterns for h in glob.glob(p)})
        if len(hits) > 1:
            raise CmdInputError('{} に対応する記録ファイルが複数あります: {}'
                                .format(path, ', '.join(hits)))
        if hits:
            with open(hits[0], encoding='utf-8') as fh:
                return hits[0], json.load(fh)
    return None, None


def read_bscan(path):
    """merged.out から Ez の B-scan [時間, トレース] と dt [s] を読む。"""
    with h5py.File(path, 'r') as fh:
        dt = float(fh.attrs['dt'])
        data = np.asarray(fh['rxs/rx1/Ez'], dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise CmdInputError('{} は B-scan ではありません（outputfiles_merge で作った '
                            'merged.out を指定すること）'.format(path))
    return data, dt


def trace_positions(rec, ntr):
    """トレースの x [m]。記録が無ければトレース番号を返す。"""
    if rec and 'bscan' in rec:
        b = rec['bscan']
        x = float(b['x0_m']) + float(b['step_m']) * np.arange(ntr)
        if int(b.get('n', ntr)) != ntr:
            print('  注意: 記録の測線 n = {} と B-scan のトレース数 {} が違います'
                  '（x は x0 + step * 番号で付けます）'.format(b.get('n'), ntr))
        return x, True
    return np.arange(ntr, dtype=float), False


# =============================================================================
# 2. 平均媒質の理論（地図の印と、変化の目安だけに使う）
# =============================================================================
def _composition_token(wt):
    wt = float(wt)
    return 'feo{}'.format(int(wt)) if wt.is_integer() else 'feo' + str(wt).replace('.', 'p')


def configure_mean_medium(rec):
    """記録ファイルの値で subsurface_model を平均媒質（ランダム性なし）に設定する。"""
    sm.set_active_level(THEORY_LEVEL)
    sm.set_level3_composition(_composition_token(rec['feotio2_wt']))
    sm.LEVEL4_ICE_MODEL = rec['ice_model'] if has_ice(rec) else 'none'
    sm.LEVEL4_ICE_TOP_M = float(rec['ice_top_m'])
    sm.LEVEL4_ICE_THICK_M = float(rec['ice_thick_m'])
    sm.LEVEL4_ICE_SPEC = 'vol'
    sm.LEVEL4_ICE_VOL_PCT = 100.0 * float(rec['ice_vol_frac'])
    if abs(float(rec.get('tx_height_m', sm.TX_HEIGHT)) - sm.TX_HEIGHT) > 1e-9:
        raise CmdInputError('記録の tx 高さ {} と subsurface_model.TX_HEIGHT {} が違います'
                            .format(rec.get('tx_height_m'), sm.TX_HEIGHT))


def model_events(rec, source_delay_ns):
    """平均媒質の界面（地表を含む）の往復群走時 [ns]（波源遅延を含む）。"""
    configure_mean_medium(rec)
    names = ['surface', 'ice_top', 'ice_bottom']
    depths = [0.0] + list(sm.interface_depths(THEORY_LEVEL))
    f0 = np.array([sm.BAND_CENTRE_HZ])
    out = []
    for i, d in enumerate(depths):
        _, opt_g, _ = sm.path_integrals(f0, d, THEORY_LEVEL, group=True)
        t = 2.0 * sm.TX_HEIGHT / sm.C + 2.0 * float(opt_g[0]) / sm.C + source_delay_ns
        out.append({'name': names[i] if i < len(names) else 'iface{}'.format(i),
                    'depth_m': float(d), 't_ns': float(t)})
    return out


def domain_bottom_time(rec, source_delay_ns, n_pml=10):
    """計算領域の底（PML の上端）からの反射が来る往復群走時 [ns]。これより後は使わない。"""
    configure_mean_medium(rec)
    d = float(rec['surface_y_m']) - n_pml * float(rec['dx_m'])
    _, opt_g, _ = sm.path_integrals(np.array([sm.BAND_CENTRE_HZ]), d, THEORY_LEVEL, group=True)
    return 2.0 * sm.TX_HEIGHT / sm.C + 2.0 * float(opt_g[0]) / sm.C + source_delay_ns


def layer_tandelta(rec):
    """氷層の中央での tan_delta（氷あり / 氷なし）。変化の目安に使う。"""
    if not has_ice(rec):
        return None
    z = float(rec['ice_top_m']) + 0.5 * float(rec['ice_thick_m'])
    f0 = np.array([1.0e9])
    configure_mean_medium(rec)
    er, ei = sm.eps_at_depth(f0, z, THEORY_LEVEL)
    keep = sm.LEVEL4_ICE_MODEL
    sm.LEVEL4_ICE_MODEL = 'none'
    er0, ei0 = sm.eps_at_depth(f0, z, THEORY_LEVEL)
    sm.LEVEL4_ICE_MODEL = keep
    return {'tand_ice': float(ei[0] / er[0]), 'tand_dry': float(ei0[0] / er0[0]),
            'eps_ice': float(er[0]), 'eps_dry': float(er0[0]), 'z_m': z}


# =============================================================================
# 3. 反射チャネル
# =============================================================================
def reflection_scan(T, R, dt):
    """ゲートを滑らせ、トレースごとの包絡ピークを基準の 95 % 点と比べる。"""
    dt_ns = dt * 1e9
    envT = envelope(T)
    envR = envelope(R)
    n_gate = max(3, int(round(2 * REFL_GATE_HW_NS / dt_ns)) | 1)
    pkT = ndimage.maximum_filter1d(envT, size=n_gate, axis=0, mode='nearest')
    pkR = ndimage.maximum_filter1d(envR, size=n_gate, axis=0, mode='nearest')
    step = max(1, int(round(REFL_STEP_NS / dt_ns)))
    half = n_gate // 2
    idx = np.arange(half, T.shape[0] - half, step)
    PT, PR = pkT[idx], pkR[idx]
    m = int(round(REFL_POOL_NS / REFL_STEP_NS))
    th = np.empty(len(idx))
    for i in range(len(idx)):
        th[i] = np.quantile(PR[max(0, i - m):i + m + 1].ravel(), 1.0 - P_FA)
    above = PT > th[:, None]
    return {'t': idx * dt_ns, 'PT': PT, 'PR': PR, 'th': th,
            'pd': above.mean(axis=1), 'above': above,
            'excess_db': np.median(DB20 * np.log(PT / th[:, None]), axis=1)}


def visible_regions(t, pd, merge_ns):
    """P_D >= PD_VISIBLE が続く時間帯を返す（merge_ns 以下の切れ目はつなぐ）。"""
    vis = pd >= PD_VISIBLE
    regs, i = [], 0
    while i < len(t):
        if vis[i]:
            j = i
            while j + 1 < len(t) and vis[j + 1]:
                j += 1
            regs.append([i, j])
            i = j + 1
        else:
            i += 1
    merged = []
    for r in regs:
        if merged and t[r[0]] - t[merged[-1][1]] <= merge_ns:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return merged


# =============================================================================
# 4. 減衰率・シフトレート
# =============================================================================
def block_index(ntr):
    nb = ntr // BLOCK_TRACES
    if nb < 3:
        raise CmdInputError('ブロックが {} 個しか作れません（トレース {} 本、'
                            'BLOCK_TRACES = {}）'.format(nb, ntr, BLOCK_TRACES))
    return [np.arange(b * BLOCK_TRACES, (b + 1) * BLOCK_TRACES) for b in range(nb)]


def block_spectrogram(data, dt, blocks):
    """トレースごとの短時間パワースペクトルをブロック内で平均する（位相は使わない）。

    戻り値: t [ns], f [GHz]（帯域内）, P [ブロック, 時刻, 周波数]
    """
    dt_ns = dt * 1e9
    L = int(round(2 * SPEC_HW_NS / dt_ns)) | 1
    nfft = max(SPEC_NFFT_MIN, int(2 ** np.ceil(np.log2(4 * L))))
    win = signal.windows.tukey(L, alpha=SPEC_TAPER)
    f = np.fft.rfftfreq(nfft, d=dt) / 1e9
    band = (f >= sm.BAND_GHZ[0]) & (f <= sm.BAND_GHZ[1])
    step = max(1, int(round(SPEC_STEP_NS / dt_ns)))
    starts = np.arange(0, data.shape[0] - L, step)
    P = np.empty((len(blocks), len(starts), int(band.sum())))
    for k, s in enumerate(starts):
        X = np.fft.rfft(data[s:s + L] * win[:, None], n=nfft, axis=0)[band]
        pw = np.abs(X) ** 2                                  # [周波数, トレース]
        for b, idx in enumerate(blocks):
            P[b, k] = pw[:, idx].mean(axis=1)
    t = (starts + L // 2) * dt_ns
    return t, f[band], P


def spectral_measures(f, P):
    """帯域パワー [dB]、重心 f_c [GHz]、幅 sigma_f [GHz]（パワー基準）。"""
    df = f[1] - f[0]
    tot = P.sum(axis=-1) * df
    fc = (P * f).sum(axis=-1) / P.sum(axis=-1)
    sf = np.sqrt((P * (f - fc[..., None]) ** 2).sum(axis=-1) / P.sum(axis=-1))
    return 10 * np.log10(tot), fc, sf


def local_slope(y, t, win_ns):
    """等間隔の時系列 y[..., 時刻] の移動回帰の傾き（端は NaN）。"""
    dtt = t[1] - t[0]
    n = max(3, int(round(win_ns / dtt)) | 1)
    j = np.arange(n) - (n - 1) / 2.0
    k = j / (np.sum(j ** 2) * dtt)
    out = np.full(y.shape, np.nan)
    h = n // 2
    for idx in np.ndindex(y.shape[:-1]):
        c = np.convolve(y[idx], k[::-1], mode='same')
        c[:h] = np.nan
        c[-h:] = np.nan
        out[idx] = c
    return out


def clutter_band(ref):
    """基準のブロック値 [ブロック, ...] から両側 P_FA の範囲 (lo, hi) を返す。"""
    with warnings.catch_warnings():                    # 端の NaN（回帰窓の外）は静かに通す
        warnings.simplefilter('ignore', RuntimeWarning)
        return _clutter_band(ref)


def _clutter_band(ref):
    if BAND_METHOD == 'percentile':
        lo, hi = np.nanquantile(ref, [P_FA / 2, 1 - P_FA / 2], axis=0)
        return lo, hi
    z = stats.norm.ppf(1.0 - P_FA / 2.0)
    mu, sd = np.nanmean(ref, axis=0), np.nanstd(ref, axis=0, ddof=1)
    return mu - z * sd, mu + z * sd


def test_against(ref, tgt):
    """対象のブロックが基準の範囲の外に出た割合 P_D と、同じ場所どうしの差。"""
    lo, hi = clutter_band(ref)
    with warnings.catch_warnings(), np.errstate(invalid='ignore', divide='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)
        out = (tgt < lo) | (tgt > hi)
        half = 0.5 * (hi - lo)
        paired = np.nanmean(tgt - ref, axis=0)
        ratio = np.abs(paired) / half
        mu = np.nanmean(ref, axis=0)
    pd = np.where(np.isfinite(half), np.mean(out, axis=0), np.nan)
    return {'lo': lo, 'hi': hi, 'mu': mu, 'half': half,
            'pd': pd, 'paired': paired, 'ratio': ratio}


def window_mask(t, a, b, excl):
    """窓 [a, b] から除外区間 excl = [(lo, hi), ...] を除いた時刻のマスク。"""
    m = (t >= a) & (t <= b)
    for lo, hi in excl:
        m &= ~((t >= lo) & (t <= hi))
    return m


def window_fit(t, y, m):
    """マスク m の時刻で y[ブロック, 時刻] を直線回帰した傾き（ブロックごと）。"""
    if m.sum() < 4:
        return None
    return np.array([np.polyfit(t[m], yy[m], 1)[0] for yy in y])


# =============================================================================
# 5. 1 条件の解析
# =============================================================================
VERDICT_EN = {'見える': 'visible', '見えない': 'not visible', '判定不能': 'n/a'}


def verdict(pd):
    if not np.isfinite(pd):
        return '判定不能'
    return '見える' if pd >= PD_VISIBLE else '見えない'


def _argmax_centre(pd, idx, t=None, prefer_t=None):
    """idx の中で P_D が最大になる時刻の番号。最大値が平らに続くときは、
    prefer_t（理論の界面の走時）があればそれに最も近いものを、無ければ平らな部分の
    中央を選ぶ（先頭を選ぶとゲートが反射の手前に寄り、ピークがゲートの端で切れやすい）。"""
    v = pd[idx]
    top = idx[v >= np.nanmax(v) - 1e-12]
    if prefer_t is not None:
        return int(top[np.argmin(np.abs(t[top] - prefer_t))])
    return int(top[len(top) // 2])


def analyze_target(name, T, R, dt, x, events, tand, info):
    dt_ns = dt * 1e9
    # --- dB の基準：基準 B-scan の地表反射（全トレースの中央値）---------------
    envR = envelope(R)
    t_axis = np.arange(R.shape[0]) * dt_ns
    t_s = info['t_surface_data']
    ms = (t_axis >= t_s - REFL_GATE_HW_NS) & (t_axis <= t_s + REFL_GATE_HW_NS)
    a_surf = float(np.median(envR[ms].max(axis=0)))
    # 対象だけの図では、対象自身の地表反射で規格化する（基準に依らない）
    envT = envelope(T)
    a_surf_T = float(np.median(envT[ms].max(axis=0)))

    res = {'name': name, 'a_surf': a_surf, 'a_surf_T': a_surf_T, 'x': x, 'events': events,
           'x_label': info.get('x_label', 'x [m]')}

    # --- 1. 反射 ------------------------------------------------------------
    rs = reflection_scan(T, R, dt)
    res['refl'] = rs
    regs = visible_regions(rs['t'], rs['pd'], merge_ns=REFL_GATE_HW_NS)

    def trace_detail(k):
        """fig2(c) 用：ゲート中心 t_k（全トレース共通）と、各トレースでゲート内の
        包絡が最大になった時刻（＝ピークの値を取った時刻）。"""
        tk = float(rs['t'][k])
        m = np.abs(t_axis - tk) <= REFL_GATE_HW_NS
        idx = np.where(m)[0]
        return {'t_k': tk, 'gate': (float(t_axis[idx[0]]), float(t_axis[idx[-1]])),
                'peak_t_ns': t_axis[idx[np.argmax(envT[idx], axis=0)]]}

    reg_rows = []
    for i0, i1 in regs:
        k = _argmax_centre(rs['pd'], np.arange(i0, i1 + 1))
        near = (min(events, key=lambda e: abs(e['t_ns'] - rs['t'][k]))
                if events else None)
        reg_rows.append({'t_start_ns': rs['t'][i0], 't_end_ns': rs['t'][i1],
                         't_peak_ns': rs['t'][k], 'pd_max': rs['pd'][k],
                         'excess_db': rs['excess_db'][k],
                         'nearest_event': near['name'] if near else '',
                         'nearest_event_dt_ns': (rs['t'][k] - near['t_ns']) if near else np.nan,
                         # fig2(c) の代わりの表示用（記録ファイルが無く理論の界面が無いとき）
                         'label': 'detected {:.1f}–{:.1f} ns'.format(rs['t'][i0], rs['t'][i1]),
                         'pd': rs['pd'][k], 'verdict': verdict(rs['pd'][k]),
                         'peaks_rel_th_db': DB20 * np.log(rs['PT'][k] / rs['th'][k]),
                         **trace_detail(k)})
    res['regions'] = reg_rows
    ev_rows = []
    for e in events:
        m = np.abs(rs['t'] - e['t_ns']) <= REFL_GATE_HW_NS
        if not m.any():
            continue
        k = _argmax_centre(rs['pd'], np.where(m)[0], rs['t'], e['t_ns'])
        ev_rows.append({'event': e['name'], 'depth_m': e['depth_m'], 't_model_ns': e['t_ns'],
                        't_at_max_ns': rs['t'][k], 'pd': rs['pd'][k],
                        'excess_db': rs['excess_db'][k],
                        'peaks_rel_th_db': DB20 * np.log(rs['PT'][k] / rs['th'][k]),
                        'verdict': verdict(rs['pd'][k]), **trace_detail(k)})
    res['event_rows'] = ev_rows

    # 同じ場であれば、氷層より浅い時間では対象と基準が同一なので P_D ≈ P_FA になる
    if info['same_field'] and len(events) > 1:
        m = (rs['t'] > t_s + 2 * REFL_GATE_HW_NS) & (rs['t'] < events[1]['t_ns'] - 2 * REFL_GATE_HW_NS)
        res['false_alarm_check'] = float(np.mean(rs['pd'][m])) if m.any() else np.nan
    else:
        res['false_alarm_check'] = np.nan

    # --- 2・3. 減衰率とシフトレート -------------------------------------------
    blocks = block_index(T.shape[1])
    ts, f, PT = block_spectrogram(T, dt, blocks)
    _, _, PR = block_spectrogram(R, dt, blocks)
    dbT, fcT, sfT = spectral_measures(f, PT)
    dbR, fcR, sfR = spectral_measures(f, PR)
    rT, rR = local_slope(dbT, ts, RATE_WIN_NS), local_slope(dbR, ts, RATE_WIN_NS)
    sT = local_slope(fcT * 1e3, ts, RATE_WIN_NS)                 # MHz/ns
    sR = local_slope(fcR * 1e3, ts, RATE_WIN_NS)
    res['spec'] = {'t': ts, 'f': f, 'PT': PT.mean(axis=0), 'PR': PR.mean(axis=0),
                   'dbT': dbT - DB20 * np.log(a_surf), 'dbR': dbR - DB20 * np.log(a_surf),
                   'fcT': fcT, 'fcR': fcR, 'sfT': sfT, 'sfR': sfR,
                   'rT': rT, 'rR': rR, 'sT': sT, 'sR': sR,
                   'nb': len(blocks)}
    res['decay'] = test_against(rR, rT)
    res['shift'] = test_against(sR, sT)

    # --- 反射の影響を受ける時間帯（理論の界面 ＋ データで見えた反射）-------------
    # 反射のエネルギーが窓に入るとレートが跳ねるので、窓の検定から除く。
    # 平均媒質の理論走時がずれていても外れないよう、データで見えた反射も除く。
    g = SPEC_HW_NS + GUARD_EXTRA_NS
    excl = [(t_s - g, t_s + SURFACE_GUARD_NS)]
    excl += [(e['t_ns'] - g, e['t_ns'] + g) for e in events if e['depth_m'] > 0]
    excl += [(r['t_start_ns'] - g, r['t_end_ns'] + g) for r in reg_rows]
    if np.isfinite(info['t_bottom']):
        excl.append((info['t_bottom'] - g, np.inf))
    h = 0.5 * RATE_WIN_NS                                  # 局所レートは回帰窓の分だけ広がる
    res['guard_zones'] = [(lo - h, min(hi + h, ts[-1])) for lo, hi in excl]

    # --- 層ごとの窓 -----------------------------------------------------------
    t_end = ts[-1] - h
    t0 = t_s + SURFACE_GUARD_NS
    if len(events) >= 3:
        wins = [('above', t0, events[1]['t_ns']),
                ('layer', events[1]['t_ns'], events[2]['t_ns']),
                ('below', events[2]['t_ns'], t_end)]
    else:
        wins = [('all', t0, t_end)]
    win_rows = []
    for wname, a, b in wins:
        m = window_mask(ts, a, b, excl)
        if m.sum() < 4:
            continue
        fc_ref = float(np.nanmean(fcR[:, m]))
        sf2_ref = float(np.nanmean(sfR[:, m] ** 2))
        for ch, yT, yR, unit in (('decay_rate', dbT, dbR, 'dB/ns'),
                                 ('fc_shift_rate', fcT * 1e3, fcR * 1e3, 'MHz/ns')):
            vT, vR = window_fit(ts, yT, m), window_fit(ts, yR, m)
            tt = test_against(vR[:, None], vT[:, None])
            pd = float(tt['pd'][0])
            half = float(tt['half'][0])
            # Delta tan_delta の換算（P(f,t) ∝ exp(-2 pi tan_delta f t)、t は往復走時）
            if ch == 'decay_rate':
                k_conv = -10 * np.log10(np.e) * 2 * np.pi * fc_ref          # dB/ns per tan_delta
            else:
                k_conv = -2 * np.pi * sf2_ref * 1e3                         # MHz/ns per tan_delta
            row = {'window': wname, 't_start_ns': a, 't_end_ns': b, 'channel': ch,
                   'n_times': int(m.sum()), 'fc_ref_ghz': fc_ref, 'sf2_ref_ghz2': sf2_ref,
                   'unit': unit, 'ref_mean': float(tt['mu'][0]), 'ref_half_width': half,
                   'target_median': float(np.median(vT)), 'pd': pd, 'verdict': verdict(pd),
                   'paired_diff': float(tt['paired'][0]),
                   'paired_over_clutter': float(tt['ratio'][0]),
                   'paired_dtand_equiv': float(tt['paired'][0]) / k_conv,
                   'min_detectable_dtand': half / abs(k_conv),
                   'vals_T': vT, 'vals_R': vR}
            if tand is not None and wname == 'layer':
                d_t = tand['tand_ice'] - tand['tand_dry']
                row['model_dtand'] = d_t
                row['model_change'] = k_conv * d_t
            win_rows.append(row)
    res['window_rows'] = win_rows
    return res


# =============================================================================
# 6. 作図（意味の近いパネルだけを 1 枚にまとめる）
# =============================================================================
def save_figure(fig, out_dir, stem):
    """PNG は out_dir に、それ以外（PDF）は out_dir/<形式名>/ に保存する。"""
    for ext in FIGURE_FORMATS:
        d = out_dir if ext == 'png' else os.path.join(out_dir, ext)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, '{}.{}'.format(stem, ext))
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
        print('Saved:', path)
    plt.close(fig)


def _event_lines(ax, events, orient='v', label=True):
    for i, e in enumerate(events):
        kw = dict(color=COL_EVENT, ls=':', lw=1.0)
        if orient == 'v':
            ax.axvline(e['t_ns'], **kw)
        else:
            ax.axhline(e['t_ns'], **kw)
    return Line2D([], [], color=COL_EVENT, ls=':', lw=1.0,
                  label='model interfaces (mean medium)') if label and events else None


def _shade_guards(ax, zones):
    for a, b in zones:
        ax.axvspan(a, b, color='0.88', zorder=0)


def plot_bscan(res, T, R, dt, out_dir, tgt_name):
    """fig1：データそのもの（基準・対象・差）。"""
    t_max = T.shape[0] * dt * 1e9
    x = res['x']
    ext = [x[0], x[-1], t_max, 0]
    a = res['a_surf']
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.5), sharey=True, layout='constrained')

    # (a)(b) 基準・対象：包絡を基準（no_ice）の地表反射で規格化した dB
    for ax, D, title, lab in (
            (axs[0], R, '(a) Reference ({})'.format(REFERENCE_CONDITION),
             'Envelope of {} [dB]\nre {} surface reflection'.format(REFERENCE_CONDITION, REFERENCE_CONDITION)),
            (axs[1], T, '(b) Target ({})'.format(tgt_name),
             'Envelope of {} [dB]\nre {} surface reflection'.format(tgt_name, REFERENCE_CONDITION))):
        env = envelope(D)
        im = ax.imshow(DB20 * np.log(np.maximum(env, 1e-300) / a), aspect='auto', extent=ext,
                       cmap='viridis', vmin=-IMG_DB_RANGE, vmax=0, interpolation='nearest')
        fig.colorbar(im, ax=ax, label=lab, shrink=0.9, pad=0.01)
        ax.set_title(title)

    # (c) 差：包絡どうしの差を dB で。20 log10(env_T / env_R) = (b) の dB − (a) の dB。
    #     規格化の基準（地表反射）は打ち消し合うので、どこで規格化しても同じ値になる。
    #     正：対象のほうが強い、負：基準のほうが強い。
    env_T = envelope(T)
    env_R = envelope(R)
    d_db = DB20 * np.log(np.maximum(env_T, 1e-300) / np.maximum(env_R, 1e-300))
    # 両方とも (a)(b) の色の範囲より弱い（−IMG_DB_RANGE dB 未満）ところは、比が意味を持たないので灰色にする
    floor = a * 10 ** (-IMG_DB_RANGE / 20.0)
    d_db = np.ma.masked_where((env_T < floor) & (env_R < floor), d_db)
    cmap_d = plt.get_cmap(DIFF_CMAP).copy()
    cmap_d.set_bad('0.75')
    im = axs[2].imshow(d_db, aspect='auto', extent=ext, cmap=cmap_d,
                       vmin=-DIFF_DB_RANGE, vmax=DIFF_DB_RANGE, interpolation='nearest')
    fig.colorbar(im, ax=axs[2], shrink=0.9, pad=0.01,
                 label='Envelope difference [dB]\n= (b) − (a) = 20 log10(env {} / env {})'
                 .format(tgt_name, REFERENCE_CONDITION))
    axs[2].set_title('(c) Envelope difference (grey: both below −{:.0f} dB)'.format(IMG_DB_RANGE))

    for ax, col in zip(axs, ('w', 'w', 'k')):
        for e in res['events']:
            ax.axhline(e['t_ns'], color=col, ls=':', lw=0.9)
        ax.set_xlabel(res['x_label'])
    axs[0].set_ylabel('Two-way time [ns]')
    save_figure(fig, out_dir, 'fig1_bscan')


def _mean_ascan_db(D, a):
    """平均 A-scan [dB re a]：コヒーレント平均（トレースを足してから包絡）と、
    インコヒーレント平均（包絡を足す）。前者はランダム場のクラッタが打ち消し合い、
    水平な層の反射が残る。後者はクラッタも含めた平均の強さ。"""
    coh = envelope(D.mean(axis=1))
    inc = envelope(D).mean(axis=1)
    to_db = lambda v: DB20 * np.log(np.maximum(v, 1e-300) / a)
    return to_db(coh), to_db(inc)


def plot_bscan_ascan(res, T, dt, out_dir, tgt_name, stem, a_norm, R=None):
    """B-scan（左）と平均 A-scan（右）を、縦軸 two-way time を共有して並べる。

    R を渡すと（比較の図）、右のパネルに基準の平均 A-scan も重ねる。
    """
    t = np.arange(T.shape[0]) * dt * 1e9
    t_max = T.shape[0] * dt * 1e9
    x = res['x']
    fig = plt.figure(figsize=(14, 7.5), layout='constrained')
    gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    env = envelope(T)
    im = ax1.imshow(DB20 * np.log(np.maximum(env, 1e-300) / a_norm), aspect='auto',
                    extent=[x[0], x[-1], t_max, 0], cmap='viridis',
                    vmin=-IMG_DB_RANGE, vmax=0, interpolation='nearest')
    for e in res['events']:
        ax1.axhline(e['t_ns'], color='w', ls=':', lw=0.9)
    ax1.set_xlabel(res['x_label'])
    ax1.set_ylabel('Two-way time [ns]')
    ax1.set_title('(a) B-scan envelope ({})'.format(tgt_name))
    fig.colorbar(im, ax=ax1, label='Envelope [dB re surface reflection]', pad=0.01)

    curves = [(T, COL_TGT, tgt_name)]
    if R is not None:
        curves.insert(0, (R, COL_REF, REFERENCE_CONDITION))
    for D, col, lab in curves:
        coh, inc = _mean_ascan_db(D, a_norm)
        ax2.plot(coh, t, color=col, lw=1.3, label='{}: coherent mean'.format(lab))
        ax2.plot(inc, t, color=col, lw=1.0, ls='--', label='{}: mean of envelopes'.format(lab))
    h = _event_lines(ax2, res['events'], orient='h')
    ax2.set_xlim(-IMG_DB_RANGE - 10, 5)
    ax2.set_ylim(t_max, 0)
    ax2.set_xlabel('Envelope [dB re surface reflection]')
    ax2.tick_params(labelleft=False)
    ax2.grid(alpha=0.3)
    ax2.legend(handles=ax2.get_legend_handles_labels()[0] + ([h] if h else []),
               fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.09))
    ax2.set_title('(b) Mean A-scan ({} traces)'.format(T.shape[1]))
    ref_txt = ' vs {}'.format(REFERENCE_CONDITION) if R is not None else ' only'
    fig.suptitle('B-scan and mean A-scan [{}{}] (dotted: model interfaces of the mean medium)'
                 .format(tgt_name, ref_txt))
    save_figure(fig, out_dir, stem)


ROW_COLORS = ('tab:purple', 'tab:orange', 'tab:green', 'tab:brown', 'tab:cyan', 'tab:olive')  # 赤は対象の色なので使わない


def reflection_rows(res):
    """fig2(c)(d) に並べる行：理論の界面があればそれ、無ければデータで見えた反射。"""
    if res['event_rows']:
        return [dict(r, label=r['event']) for r in res['event_rows']], 'model interfaces'
    return res['regions'], 'detected reflections (no record file)'


def _plot_reflection_ab(res, ax1, ax2, tgt_name, rows=(), cols=()):
    """fig2 の (a)(b)。rows を渡すと、(b) にだけ (c)(d) で使ったゲートの枠と中心 t_k を描く。"""
    rs, a = res['refl'], res['a_surf']
    t = rs['t']
    to_db = lambda v: DB20 * np.log(np.maximum(v, 1e-300) / a)

    # --- (a) -------------------------------------------------------------------
    for P, col, lab in ((rs['PR'], COL_REF, REFERENCE_CONDITION), (rs['PT'], COL_TGT, tgt_name)):
        lo, med, hi = np.quantile(P, [0.05, 0.5, 0.95], axis=1)
        ax1.fill_between(t, to_db(lo), to_db(hi), color=col, alpha=0.25, lw=0)
        ax1.plot(t, to_db(med), color=col, lw=1.2, label='{}: median (band 5–95 %)'.format(lab))
    ax1.plot(t, to_db(rs['th']), color='k', lw=1.4, ls='--',
             label='clutter threshold ({:.0f} % false alarm)'.format(100 * P_FA))
    h = _event_lines(ax1, res['events'])
    ax1.set_ylabel('Gated envelope peak\n[dB re surface]')
    ax1.set_ylim(-IMG_DB_RANGE - 10, 5)
    ax1.legend(handles=ax1.get_legend_handles_labels()[0] + ([h] if h else []),
               fontsize=8, loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.set_title('(a) Reflection strength vs clutter')

    # --- (b) -------------------------------------------------------------------
    ax2.plot(t, rs['pd'], color=COL_TGT, lw=1.3)
    for r in res['regions']:
        ax2.axvspan(r['t_start_ns'], r['t_end_ns'], color='0.85', lw=0, zorder=0)
    ax2.axhline(P_FA, color='k', ls=':', lw=1, label='false alarm rate {:.0f} %'.format(100 * P_FA))
    ax2.axhline(PD_VISIBLE, color='k', ls='--', lw=1,
                label='visible if P_D ≥ {:.0f} %'.format(100 * PD_VISIBLE))
    _event_lines(ax2, res['events'], label=False)
    for r, col in zip(rows, cols):
        ax2.axvspan(r['gate'][0], r['gate'][1], facecolor='none', edgecolor=col, lw=1.4, zorder=3)
        ax2.axvline(r['t_k'], color=col, lw=1.0, zorder=3)
    handles = ax2.get_legend_handles_labels()[0]
    if len(rows):
        handles.append(Patch(facecolor='none', edgecolor='k', lw=1.2,
                             label='gate used in (c)(d) (line: centre t_k)'))
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel('Two-way time [ns]  (horizontal axis = gate centre)')
    ax2.set_ylabel('Detection probability P_D')
    ax2.legend(handles=handles, fontsize=8, loc='upper right')
    ax2.grid(alpha=0.3)
    ax2.set_title('(b) Fraction of target traces above the threshold (grey: visible)')


def plot_reflection_ab(res, out_dir, tgt_name):
    """fig2-1：反射チャネルの (a)(b) だけ（ゲートの表示なし）。"""
    fig = plt.figure(figsize=(12, 8.5), layout='constrained')
    gs = fig.add_gridspec(2, 1)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    _plot_reflection_ab(res, ax1, ax2, tgt_name)
    fig.suptitle('Reflection channel: is any reflection visible above the clutter? ({} vs {})'
                 .format(tgt_name, REFERENCE_CONDITION))
    save_figure(fig, out_dir, 'fig2-1_reflection')


def plot_reflection(res, out_dir, tgt_name):
    """fig2-2：反射チャネルの (a)-(d)。

    (a) ゲート内ピークの分布としきい値 (b) P_D(t)
    (c) 各行（界面 / 見えた反射）について、全トレース共通のゲート中心 t_k での
        トレースごとの「ピーク − しきい値」を、トレースの位置 x に対して描く
    (d) 同じトレースで、ゲート内の包絡が最大になった時刻（＝(c) の値を取った時刻）
    (b) の色つきの枠が、(c)(d) で使ったゲート（t_k ± REFL_GATE_HW_NS）。
    """
    x = res['x']
    rows, row_kind = reflection_rows(res)
    cols = [ROW_COLORS[i % len(ROW_COLORS)] for i in range(len(rows))]

    fig = plt.figure(figsize=(17, 9), layout='constrained')
    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1], sharex=ax3)
    _plot_reflection_ab(res, ax1, ax2, tgt_name, rows, cols)

    # --- (c)(d) ----------------------------------------------------------------
    if not rows:
        for ax in (ax3, ax4):
            ax.text(0.5, 0.5, 'no model interfaces\nand no detected reflection',
                    ha='center', va='center', transform=ax.transAxes)
    for r, col in zip(rows, cols):
        v = r['peaks_rel_th_db']
        lab = '{}: gate {:.2f}–{:.2f} ns (centre t_k = {:.2f} ns)\n  {}, P_D = {:.0f} %'.format(
            r['label'], r['gate'][0], r['gate'][1], r['t_k'], VERDICT_EN[r['verdict']], 100 * r['pd'])
        ax3.plot(x, v, color=col, lw=0.6, alpha=0.5)
        ax3.scatter(x, v, s=9, color=col, label=lab, zorder=3)
        ax4.axhspan(r['gate'][0], r['gate'][1], color=col, alpha=0.10, lw=0)
        ax4.axhline(r['t_k'], color=col, lw=0.8, ls='--')
        ax4.scatter(x, r['peak_t_ns'], s=9, color=col, zorder=3)
    ax3.axhline(0, color='k', ls='--', lw=1.2)
    ax3.set_ylabel('Peak − threshold [dB]\n(above 0: this trace exceeds the threshold)')
    ax3.set_title('(c) Each target trace at {}'.format(row_kind), fontsize=10)
    ax3.grid(alpha=0.3)
    if rows:
        ax3.legend(fontsize=7.5, loc='best')
    ax3.tick_params(labelbottom=False)
    ax4.set_xlabel(res['x_label'])
    ax4.set_ylabel('Time of the envelope peak\nwithin the gate [ns]')
    ax4.invert_yaxis()
    ax4.grid(alpha=0.3)
    ax4.set_title('(d) Where in the gate the value of (c) was taken\n'
                  '(band: gate, dashed: centre t_k)', fontsize=10)
    fig.suptitle('Reflection channel: is any reflection visible above the clutter? ({} vs {})'
                 .format(tgt_name, REFERENCE_CONDITION))
    save_figure(fig, out_dir, 'fig2-2_reflection_traces')


def _rate_figure(res, key, out_dir, tgt_name, stem, title, ylab_raw, raw_T, raw_R,
                 ylab_rate, unit, channel):
    sp = res['spec']
    t = sp['t']
    tt = res[key]
    fig = plt.figure(figsize=(15, 10), layout='constrained')
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax4 = fig.add_subplot(gs[:, 1])
    for ax in (ax1, ax2, ax3):
        _shade_guards(ax, res['guard_zones'])
    for yy in raw_R:
        ax1.plot(t, yy, color=COL_REF, lw=0.6, alpha=0.5)
    for yy in raw_T:
        ax1.plot(t, yy, color=COL_TGT, lw=0.6, alpha=0.5)
    ax1.plot(t, np.nanmean(raw_R, axis=0), color=COL_REF, lw=2, label=REFERENCE_CONDITION + ' (mean of blocks)')
    ax1.plot(t, np.nanmean(raw_T, axis=0), color=COL_TGT, lw=2, label=tgt_name + ' (mean of blocks)')
    h = _event_lines(ax1, res['events'])
    ax1.set_ylabel(ylab_raw)
    ax1.legend(handles=ax1.get_legend_handles_labels()[0] + ([h] if h else []), fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_title('(a) Raw observable (thin: each block of {} traces)'.format(BLOCK_TRACES))
    rT = sp['rT'] if key == 'decay' else sp['sT']
    ax2.fill_between(t, tt['lo'], tt['hi'], color=COL_REF, alpha=0.25, lw=0,
                     label='clutter range of {} ({:.0f} % two-sided)'.format(REFERENCE_CONDITION, 100 * P_FA))
    for yy in rT:
        ax2.plot(t, yy, color=COL_TGT, lw=0.6, alpha=0.6)
    ax2.plot(t, tt['mu'] + tt['paired'], color='k', lw=1.2,
             label='reference mean + paired difference (ideal upper bound)')
    _event_lines(ax2, res['events'], label=False)
    ax2.set_ylabel(ylab_rate)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_title('(b) Local rate (regression over {:g} ns), thin red: target blocks'.format(RATE_WIN_NS))
    ax3.plot(t, tt['pd'], color=COL_TGT, lw=1.3)
    ax3.axhline(P_FA, color='k', ls=':', lw=1, label='false alarm rate')
    ax3.axhline(PD_VISIBLE, color='k', ls='--', lw=1, label='visible threshold')
    _event_lines(ax3, res['events'], label=False)
    ax3.set_ylim(-0.02, 1.02)
    ax3.set_ylabel('P_D')
    ax3.set_xlabel('Two-way time [ns]  (grey: interface zones, excluded from the window test)')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    ax3.set_title('(c) Fraction of target blocks outside the clutter range')
    rows = [r for r in res['window_rows'] if r['channel'] == channel]
    for i, r in enumerate(rows):
        jr = (np.random.RandomState(10 + i).rand(len(r['vals_R'])) - 0.5) * 0.25
        jt = (np.random.RandomState(20 + i).rand(len(r['vals_T'])) - 0.5) * 0.25
        ax4.scatter(i - 0.18 + jr, r['vals_R'], s=14, color=COL_REF)
        ax4.scatter(i + 0.18 + jt, r['vals_T'], s=14, color=COL_TGT)
        ax4.errorbar(i - 0.18, r['ref_mean'], yerr=r['ref_half_width'], color='k', capsize=6, lw=1.2)
        ax4.text(i, max(np.max(r['vals_R']), np.max(r['vals_T'])),
                 '{}\nP_D={:.0f}%'.format(VERDICT_EN[r['verdict']], 100 * r['pd']),
                 ha='center', va='bottom', fontsize=9)
    ax4.set_xticks(range(len(rows)))
    ax4.set_xlim(-0.7, len(rows) - 0.3)
    ax4.set_xticklabels(['{}\n{:.1f}–{:.1f} ns'.format(r['window'], r['t_start_ns'], r['t_end_ns'])
                         for r in rows], fontsize=8)
    ax4.set_ylabel('{} fitted over the window [{}]'.format(ylab_rate.split(' [')[0], unit))
    ax4.legend(handles=[Line2D([], [], marker='o', ls='', color=COL_REF, label=REFERENCE_CONDITION + ' blocks'),
                        Line2D([], [], marker='o', ls='', color=COL_TGT, label=tgt_name + ' blocks'),
                        Line2D([], [], color='k', label='clutter range')], fontsize=8, loc='lower right')
    ax4.grid(alpha=0.3)
    ax4.set_title('(d) Window test (interfaces excluded)')
    if rows:
        lo = min(min(np.min(r['vals_R']), np.min(r['vals_T'])) for r in rows)
        hi = max(max(np.max(r['vals_R']), np.max(r['vals_T'])) for r in rows)
        pad = 0.25 * (hi - lo + 1e-12)
        ax4.set_ylim(lo - pad, hi + 2 * pad)
    fig.suptitle(title + ' ({} vs {})'.format(tgt_name, REFERENCE_CONDITION))
    save_figure(fig, out_dir, stem)


def plot_decay(res, out_dir, tgt_name):
    """fig3：減衰率チャネル。"""
    sp = res['spec']
    _rate_figure(res, 'decay', out_dir, tgt_name, 'fig3_decay_rate',
                 'Attenuation-rate channel: does the decay rate of band power change?',
                 'Band power {:.1f}–{:.1f} GHz\n[dB re surface]'.format(*sm.BAND_GHZ),
                 sp['dbT'], sp['dbR'], 'Decay rate [dB/ns]', 'dB/ns', 'decay_rate')


def plot_shift(res, out_dir, tgt_name):
    """fig4：重心周波数シフトレートのチャネル。"""
    sp = res['spec']
    _rate_figure(res, 'shift', out_dir, tgt_name, 'fig4_fc_shift_rate',
                 'Centroid-shift channel: does the shift rate of f_c change?',
                 'Centroid frequency f_c [GHz]', sp['fcT'], sp['fcR'],
                 'Shift rate df_c/dt [MHz/ns]', 'MHz/ns', 'fc_shift_rate')


def plot_spectrogram(res, out_dir, tgt_name):
    """fig5：時間-周波数の生データ（ブロック平均のスペクトログラム）。"""
    sp = res['spec']
    t, f = sp['t'], sp['f']
    a2 = res['a_surf'] ** 2
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True, layout='constrained')
    vmax = 10 * np.log10(max(sp['PR'].max(), sp['PT'].max()))
    for ax, P, fc, lab in ((axs[0], sp['PR'], sp['fcR'], REFERENCE_CONDITION),
                           (axs[1], sp['PT'], sp['fcT'], tgt_name)):
        im = ax.pcolormesh(t, f, 10 * np.log10(P.T), shading='auto', cmap='viridis',
                           vmin=vmax - IMG_DB_RANGE, vmax=vmax)
        ax.plot(t, np.nanmean(fc, axis=0), color='w', lw=1.5, label='f_c (mean of blocks)')
        for e in res['events']:
            ax.axvline(e['t_ns'], color='w', ls=':', lw=0.9)
        ax.set_title(lab)
        ax.set_xlabel('Two-way time [ns]')
        ax.legend(fontsize=8, loc='upper right')
    axs[0].set_ylabel('Frequency [GHz]')
    fig.colorbar(im, ax=axs, label='Power (mean over all traces) [dB, arbitrary]')
    fig.suptitle('Short-time power spectra (window ±{:g} ns)'.format(SPEC_HW_NS))
    save_figure(fig, out_dir, 'fig5_spectrogram')


# =============================================================================
# 6b. 作図：対象（非 no_ice）だけ（判定なし。基準は使わない）
# =============================================================================
# dB は対象自身の地表反射（全トレースの中央値）で規格化する。
# 窓（above / layer / below）と灰色の除外帯は比較の図と同じものを使い、両者を見比べられるようにする。
def _single_db_offset(res):
    """比較用の dB（基準の地表反射が 0 dB）を、対象の地表反射が 0 dB になるよう直す量。"""
    return DB20 * np.log(res['a_surf'] / res['a_surf_T'])


def _single_title(tgt_name):
    return '[{} only]'.format(tgt_name)


def plot_single_bscan(res, T, dt, out_dir, tgt_name):
    """single1：対象の B-scan。"""
    t_max = T.shape[0] * dt * 1e9
    x = res['x']
    env = envelope(T)
    fig, ax = plt.subplots(figsize=(8, 6.5), layout='constrained')
    im = ax.imshow(DB20 * np.log(np.maximum(env, 1e-300) / res['a_surf_T']), aspect='auto',
                   extent=[x[0], x[-1], t_max, 0], cmap='viridis',
                   vmin=-IMG_DB_RANGE, vmax=0, interpolation='nearest')
    for e in res['events']:
        ax.axhline(e['t_ns'], color='w', ls=':', lw=0.9)
    ax.set_xlabel(res['x_label'])
    ax.set_ylabel('Two-way time [ns]')
    fig.colorbar(im, ax=ax, label='Envelope [dB re own surface reflection]')
    ax.set_title('{} B-scan envelope (dotted: model interfaces)'.format(_single_title(tgt_name)))
    save_figure(fig, out_dir, 'single1_bscan')


def _single_event_peaks(res):
    """平均媒質の界面ごとに、ゲート内でトレース中央値が最大の時刻のピーク列 [dB re 自身の地表]。"""
    rs = res['refl']
    out = []
    for e in res['events']:
        m = np.abs(rs['t'] - e['t_ns']) <= REFL_GATE_HW_NS
        if not m.any():
            continue
        idx = np.where(m)[0]
        k = idx[int(np.argmax(np.median(rs['PT'][idx], axis=1)))]
        out.append({'event': e['name'], 't_model_ns': e['t_ns'], 't_at_max_ns': rs['t'][k],
                    'peaks_db': DB20 * np.log(np.maximum(rs['PT'][k], 1e-300) / res['a_surf_T'])})
    return out


def plot_single_reflection(res, out_dir, tgt_name):
    """single2：対象の反射の強さ（時間ごとの分布と、界面での値）。"""
    rs = res['refl']
    t = rs['t']
    to_db = lambda v: DB20 * np.log(np.maximum(v, 1e-300) / res['a_surf_T'])
    fig = plt.figure(figsize=(15, 5.5), layout='constrained')
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    lo, med, hi = np.quantile(rs['PT'], [0.05, 0.5, 0.95], axis=1)
    ax1.fill_between(t, to_db(lo), to_db(hi), color=COL_TGT, alpha=0.25, lw=0,
                     label='5–95 % of traces')
    ax1.plot(t, to_db(med), color=COL_TGT, lw=1.2, label='median of traces')
    h = _event_lines(ax1, res['events'])
    ax1.set_xlabel('Two-way time [ns]')
    ax1.set_ylabel('Gated envelope peak\n[dB re own surface]')
    ax1.set_ylim(-IMG_DB_RANGE - 10, 5)
    ax1.legend(handles=ax1.get_legend_handles_labels()[0] + ([h] if h else []),
               fontsize=8, loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.set_title('(a) Reflection strength (gate ±{:g} ns)'.format(REFL_GATE_HW_NS))
    rows = _single_event_peaks(res)
    for i, r in enumerate(rows):
        v = r['peaks_db']
        jit = (np.random.RandomState(i).rand(len(v)) - 0.5) * 0.3
        ax2.scatter(np.full(len(v), i) + jit, v, s=8, color=COL_TGT, alpha=0.6)
        ax2.text(i, np.nanmax(v) + 2, 'median\n{:.1f} dB'.format(np.median(v)),
                 ha='center', va='bottom', fontsize=9)
    ax2.set_xticks(range(len(rows)))
    ax2.set_xticklabels([r['event'] for r in rows], rotation=20)
    ax2.set_xlim(-0.7, len(rows) - 0.3)
    ax2.set_ylabel('Trace peak [dB re own surface]')
    ax2.set_title('(b) At model interfaces')
    ax2.grid(alpha=0.3)
    if rows:
        lo_ = min(np.nanmin(r['peaks_db']) for r in rows)
        hi_ = max(np.nanmax(r['peaks_db']) for r in rows)
        ax2.set_ylim(lo_ - 5, hi_ + 10)
    fig.suptitle('{} Reflection strength (no detection test)'.format(_single_title(tgt_name)))
    save_figure(fig, out_dir, 'single2_reflection')


def _single_rate_figure(res, out_dir, tgt_name, stem, title, ylab_raw, raw, rate,
                        ylab_rate, unit, channel):
    sp = res['spec']
    t = sp['t']
    z = stats.norm.ppf(1.0 - P_FA / 2.0)
    fig = plt.figure(figsize=(15, 8), layout='constrained')
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[:, 1])
    for ax in (ax1, ax2):
        _shade_guards(ax, res['guard_zones'])
    for yy in raw:
        ax1.plot(t, yy, color=COL_TGT, lw=0.6, alpha=0.5)
    ax1.plot(t, np.nanmean(raw, axis=0), color=COL_TGT, lw=2, label='mean of blocks')
    h = _event_lines(ax1, res['events'])
    ax1.set_ylabel(ylab_raw)
    ax1.legend(handles=ax1.get_legend_handles_labels()[0] + ([h] if h else []), fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_title('(a) Raw observable (thin: each block of {} traces)'.format(BLOCK_TRACES))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mu, sd = np.nanmean(rate, axis=0), np.nanstd(rate, axis=0, ddof=1)
    ax2.fill_between(t, mu - z * sd, mu + z * sd, color=COL_TGT, alpha=0.2, lw=0,
                     label='mean ± {:.2f} σ of blocks'.format(z))
    for yy in rate:
        ax2.plot(t, yy, color=COL_TGT, lw=0.5, alpha=0.4)
    ax2.plot(t, mu, color=COL_TGT, lw=1.8, label='mean of blocks')
    _event_lines(ax2, res['events'], label=False)
    ax2.set_ylabel(ylab_rate)
    ax2.set_xlabel('Two-way time [ns]  (grey: interface zones, excluded from the window fit)')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_title('(b) Local rate (regression over {:g} ns)'.format(RATE_WIN_NS))
    rows = [r for r in res['window_rows'] if r['channel'] == channel]
    for i, r in enumerate(rows):
        v = r['vals_T']
        jt = (np.random.RandomState(20 + i).rand(len(v)) - 0.5) * 0.25
        ax3.scatter(i + jt, v, s=14, color=COL_TGT)
        m, s = float(np.mean(v)), float(np.std(v, ddof=1))
        ax3.errorbar(i + 0.25, m, yerr=z * s, color='k', capsize=6, lw=1.2, marker='_', ms=14)
        ax3.text(i, max(np.max(v), m + z * s), '{:+.3g}\n± {:.2g}'.format(m, z * s),
                 ha='center', va='bottom', fontsize=9)
    ax3.set_xticks(range(len(rows)))
    ax3.set_xlim(-0.7, len(rows) - 0.3)
    ax3.set_xticklabels(['{}\n{:.1f}–{:.1f} ns'.format(r['window'], r['t_start_ns'], r['t_end_ns'])
                         for r in rows], fontsize=8)
    ax3.set_ylabel('{} fitted over the window [{}]'.format(ylab_rate.split(' [')[0], unit))
    ax3.legend(handles=[Line2D([], [], marker='o', ls='', color=COL_TGT, label='blocks'),
                        Line2D([], [], color='k', label='mean ± {:.2f} σ'.format(z))],
               fontsize=8, loc='lower right')
    ax3.grid(alpha=0.3)
    ax3.set_title('(c) Window fit (interfaces excluded)')
    if rows:
        ms = [(np.mean(r['vals_T']), z * np.std(r['vals_T'], ddof=1)) for r in rows]
        lo = min(min(np.min(r['vals_T']), m - s) for r, (m, s) in zip(rows, ms))
        hi = max(max(np.max(r['vals_T']), m + s) for r, (m, s) in zip(rows, ms))
        pad = 0.25 * (hi - lo + 1e-12)
        ax3.set_ylim(lo - pad, hi + 2 * pad)
    fig.suptitle('{} {}'.format(_single_title(tgt_name), title))
    save_figure(fig, out_dir, stem)


def plot_single_decay(res, out_dir, tgt_name):
    """single3：対象の帯域パワーと減衰率（幾何拡散・散乱を含む見かけの値）。"""
    sp = res['spec']
    _single_rate_figure(res, out_dir, tgt_name, 'single3_decay_rate',
                        'Decay of band power (apparent: includes spreading and scattering)',
                        'Band power {:.1f}–{:.1f} GHz\n[dB re own surface]'.format(*sm.BAND_GHZ),
                        sp['dbT'] + _single_db_offset(res), sp['rT'],
                        'Decay rate [dB/ns]', 'dB/ns', 'decay_rate')


def plot_single_shift(res, out_dir, tgt_name):
    """single4：対象の重心周波数とシフトレート。"""
    sp = res['spec']
    _single_rate_figure(res, out_dir, tgt_name, 'single4_fc_shift_rate',
                        'Centroid frequency and its shift rate',
                        'Centroid frequency f_c [GHz]', sp['fcT'], sp['sT'],
                        'Shift rate df_c/dt [MHz/ns]', 'MHz/ns', 'fc_shift_rate')


def plot_single_spectrogram(res, out_dir, tgt_name):
    """single5：対象のスペクトログラム（全トレース平均）。"""
    sp = res['spec']
    t, f = sp['t'], sp['f']
    P = sp['PT']
    vmax = 10 * np.log10(P.max())
    fig, ax = plt.subplots(figsize=(9, 5.5), layout='constrained')
    im = ax.pcolormesh(t, f, 10 * np.log10(P.T), shading='auto', cmap='viridis',
                       vmin=vmax - IMG_DB_RANGE, vmax=vmax)
    ax.plot(t, np.nanmean(sp['fcT'], axis=0), color='w', lw=1.5, label='f_c (mean of blocks)')
    for e in res['events']:
        ax.axvline(e['t_ns'], color='w', ls=':', lw=0.9)
    ax.set_xlabel('Two-way time [ns]')
    ax.set_ylabel('Frequency [GHz]')
    ax.legend(fontsize=8, loc='upper right')
    fig.colorbar(im, ax=ax, label='Power (mean over all traces) [dB, arbitrary]')
    ax.set_title('{} Short-time power spectra (window ±{:g} ns)'
                 .format(_single_title(tgt_name), SPEC_HW_NS))
    save_figure(fig, out_dir, 'single5_spectrogram')


def write_single_outputs(res, out_dir):
    """対象だけの数値：時間ごとの値と、窓ごとの回帰値。"""
    rs, sp = res['refl'], res['spec']
    db = lambda v: DB20 * np.log(np.maximum(v, 1e-300) / res['a_surf_T'])
    q = np.quantile(rs['PT'], [0.05, 0.5, 0.95], axis=1)
    _write_csv(os.path.join(out_dir, 'single_reflection_scan.csv'),
               [{'t_ns': rs['t'][i], 'peak_p05_db': db(q[0, i]), 'peak_median_db': db(q[1, i]),
                 'peak_p95_db': db(q[2, i])} for i in range(len(rs['t']))],
               ['t_ns', 'peak_p05_db', 'peak_median_db', 'peak_p95_db'])
    off = _single_db_offset(res)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        cols = {'band_power_db': np.nanmean(sp['dbT'] + off, axis=0),
                'decay_rate_mean': np.nanmean(sp['rT'], axis=0),
                'decay_rate_sd': np.nanstd(sp['rT'], axis=0, ddof=1),
                'fc_mean_ghz': np.nanmean(sp['fcT'], axis=0),
                'sigma_f_mean_ghz': np.nanmean(sp['sfT'], axis=0),
                'shift_rate_mean': np.nanmean(sp['sT'], axis=0),
                'shift_rate_sd': np.nanstd(sp['sT'], axis=0, ddof=1)}
    keys = ['t_ns'] + list(cols)
    _write_csv(os.path.join(out_dir, 'single_rate_scan.csv'),
               [dict({'t_ns': sp['t'][i]}, **{k: v[i] for k, v in cols.items()})
                for i in range(len(sp['t']))], keys)
    rows = []
    for r in res['window_rows']:
        v = r['vals_T']
        rows.append({'window': r['window'], 't_start_ns': r['t_start_ns'], 't_end_ns': r['t_end_ns'],
                     'channel': r['channel'], 'unit': r['unit'], 'n_blocks': len(v),
                     'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)),
                     'median': float(np.median(v))})
    _write_csv(os.path.join(out_dir, 'single_window_fits.csv'), rows,
               ['window', 't_start_ns', 't_end_ns', 'channel', 'unit', 'n_blocks',
                'mean', 'sd', 'median'])
    ev = _single_event_peaks(res)
    _write_csv(os.path.join(out_dir, 'single_reflection_events.csv'),
               [{'event': e['event'], 't_model_ns': e['t_model_ns'], 't_at_max_ns': e['t_at_max_ns'],
                 'peak_median_db': float(np.median(e['peaks_db'])),
                 'peak_p05_db': float(np.quantile(e['peaks_db'], 0.05)),
                 'peak_p95_db': float(np.quantile(e['peaks_db'], 0.95))} for e in ev],
               ['event', 't_model_ns', 't_at_max_ns', 'peak_median_db', 'peak_p05_db', 'peak_p95_db'])


# =============================================================================
# 7. 数値の出力
# =============================================================================
def _write_csv(path, rows, keys):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})
    print('Saved:', path)


def write_outputs(res, out_dir, tgt_name, info):
    rs, a = res['refl'], res['a_surf']
    db = lambda v: DB20 * np.log(np.maximum(v, 1e-300) / a)
    _write_csv(os.path.join(out_dir, 'reflection_scan.csv'),
               [{'t_ns': rs['t'][i], 'threshold_db': db(rs['th'][i]),
                 'target_median_db': db(np.median(rs['PT'][i])),
                 'ref_median_db': db(np.median(rs['PR'][i])),
                 'pd': rs['pd'][i], 'excess_median_db': rs['excess_db'][i]}
                for i in range(len(rs['t']))],
               ['t_ns', 'threshold_db', 'target_median_db', 'ref_median_db', 'pd', 'excess_median_db'])
    _write_csv(os.path.join(out_dir, 'reflection_regions.csv'), res['regions'],
               ['t_start_ns', 't_end_ns', 't_peak_ns', 'pd_max', 'excess_db',
                'nearest_event', 'nearest_event_dt_ns'])
    _write_csv(os.path.join(out_dir, 'reflection_events.csv'), res['event_rows'],
               ['event', 'depth_m', 't_model_ns', 't_at_max_ns', 'pd', 'excess_db', 'verdict'])
    rows, _ = reflection_rows(res)
    _write_csv(os.path.join(out_dir, 'reflection_traces.csv'),
               [{'row': r['label'], 'gate_centre_ns': r['t_k'], 'gate_start_ns': r['gate'][0],
                 'gate_end_ns': r['gate'][1], 'trace': j, 'x': res['x'][j],
                 'peak_time_ns': r['peak_t_ns'][j], 'peak_minus_threshold_db': r['peaks_rel_th_db'][j],
                 'above_threshold': bool(r['peaks_rel_th_db'][j] > 0)}
                for r in rows for j in range(len(res['x']))],
               ['row', 'gate_centre_ns', 'gate_start_ns', 'gate_end_ns', 'trace', 'x',
                'peak_time_ns', 'peak_minus_threshold_db', 'above_threshold'])
    sp, dc, sh = res['spec'], res['decay'], res['shift']
    _write_csv(os.path.join(out_dir, 'rate_scan.csv'),
               [{'t_ns': sp['t'][i],
                 'decay_ref_mean': dc['mu'][i], 'decay_ref_half': dc['half'][i],
                 'decay_paired': dc['paired'][i], 'decay_pd': dc['pd'][i],
                 'shift_ref_mean': sh['mu'][i], 'shift_ref_half': sh['half'][i],
                 'shift_paired': sh['paired'][i], 'shift_pd': sh['pd'][i],
                 'fc_ref_mean_ghz': np.nanmean(sp['fcR'][:, i]),
                 'fc_target_mean_ghz': np.nanmean(sp['fcT'][:, i]),
                 'sigma_f_ref_mean_ghz': np.nanmean(sp['sfR'][:, i])}
                for i in range(len(sp['t']))],
               ['t_ns', 'decay_ref_mean', 'decay_ref_half', 'decay_paired', 'decay_pd',
                'shift_ref_mean', 'shift_ref_half', 'shift_paired', 'shift_pd',
                'fc_ref_mean_ghz', 'fc_target_mean_ghz', 'sigma_f_ref_mean_ghz'])
    _write_csv(os.path.join(out_dir, 'window_tests.csv'), res['window_rows'],
               ['window', 't_start_ns', 't_end_ns', 'n_times', 'channel', 'unit',
                'fc_ref_ghz', 'sf2_ref_ghz2', 'ref_mean',
                'ref_half_width', 'target_median', 'pd', 'verdict', 'paired_diff',
                'paired_over_clutter', 'paired_dtand_equiv', 'min_detectable_dtand',
                'model_dtand', 'model_change'])
    np.savez(os.path.join(out_dir, 'bscan_detection.npz'),
             x=res['x'], refl_t=rs['t'], refl_PT=rs['PT'], refl_PR=rs['PR'], refl_th=rs['th'],
             refl_pd=rs['pd'], spec_t=sp['t'], spec_f=sp['f'], spec_PT=sp['PT'], spec_PR=sp['PR'],
             dbT=sp['dbT'], dbR=sp['dbR'], fcT=sp['fcT'], fcR=sp['fcR'], sfT=sp['sfT'], sfR=sp['sfR'],
             rT=sp['rT'], rR=sp['rR'], sT=sp['sT'], sR=sp['sR'], a_surf=a)
    print('Saved:', os.path.join(out_dir, 'bscan_detection.npz'))

    lines = ['bscan_detection.py  {}'.format(datetime.now().isoformat(timespec='seconds')),
             '', '[入力]']
    lines += ['  {:<22} {}'.format(k, v) for k, v in info['inputs'].items()]
    lines += ['', '[設定]']
    for k in ('P_FA', 'PD_VISIBLE', 'REFL_GATE_HW_NS', 'REFL_POOL_NS', 'SPEC_HW_NS',
              'SPEC_STEP_NS', 'BLOCK_TRACES', 'RATE_WIN_NS', 'BAND_METHOD',
              'SUBTRACT_FREESPACE', 'TRACE_X_RANGE_M'):
        lines.append('  {:<22} {}'.format(k, globals()[k]))
    lines += ['  {:<22} {}'.format('blocks', sp['nb']),
              '', '[確認]']
    lines += ['  ' + s for s in info['checks']]
    if np.isfinite(res['false_alarm_check']):
        lines.append('  氷層より浅い時間の P_D の平均 = {:.3f}（同じ場なので {:.2f} 前後になるはず）'
                     .format(res['false_alarm_check'], P_FA))
    lines += ['', '[判定] 見える = P_D >= {:.0f} %（クラッタ基準 {:.0f} %）'
              .format(100 * PD_VISIBLE, 100 * P_FA), '  反射（平均媒質の界面）:']
    for r in res['event_rows']:
        lines.append('    {:<11} t = {:6.2f} ns  P_D = {:5.1f} %  中央値の超過 {:+6.1f} dB  -> {}'
                     .format(r['event'], r['t_model_ns'], 100 * r['pd'], r['excess_db'], r['verdict']))
    lines.append('  反射（データから検出した時間帯）: {} 個'.format(len(res['regions'])))
    for r in res['regions']:
        lines.append('    {:6.2f}-{:6.2f} ns  P_D 最大 {:5.1f} %  近い界面 {} ({:+.2f} ns)'
                     .format(r['t_start_ns'], r['t_end_ns'], 100 * r['pd_max'],
                             r['nearest_event'] or '-', r['nearest_event_dt_ns']))
    lines.append('  減衰率・シフトレート（窓ごと）:')
    for r in res['window_rows']:
        extra = ''
        if 'model_dtand' in r:
            extra = '  理論の変化 {:+.4f} {}（Δtanδ = {:+.2e}）'.format(
                r['model_change'], r['unit'], r['model_dtand'])
        lines.append('    {:<6} {:<14} P_D = {:5.1f} % -> {:<4}  同じ場所の差 {:+.4f} {}'
                     '（クラッタ幅の {:.2f} 倍）  検出に要る Δtanδ ≈ {:.1e}{}'
                     .format(r['window'], r['channel'], 100 * r['pd'], r['verdict'],
                             r['paired_diff'], r['unit'], r['paired_over_clutter'],
                             r['min_detectable_dtand'], extra))
    lines.append('  （Δtanδ の換算は P(f,t) ∝ exp(-2π tanδ f t)。f_c と σ_f² は窓ごとの基準の値、'
                 '窓から反射の前後を除いた時間で回帰）')
    path = os.path.join(out_dir, 'run_info.txt')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('Saved:', path)
    print('\n'.join(lines[lines.index(next(l for l in lines if l.startswith('[判定]'))):]))


# =============================================================================
# 8. main
# =============================================================================
def run_target(tgt, conds, reference, R, rec_R, dt, x, fs, out_base, level, ctx):
    """1 つの対象条件を基準と比べて解析・作図する。ctx は基準側の共通情報。"""
    p_t = conds[tgt][BSCAN_KEY]
    print('\n=== 対象: {}  ({})'.format(tgt, p_t))
    T, dt_t = read_bscan(p_t)
    if abs(dt_t - dt) > 1e-9 * dt or T.shape != ctx['shape']:
        raise CmdInputError('{} と基準で dt または形が違います: {} {} vs {} {}'
                            .format(tgt, dt_t, T.shape, dt, ctx['shape']))
    if fs is not None:
        T = T - fs[:, None]
    T = T[:, ctx['sel']]
    rec_path, rec_T = find_record(p_t)
    checks = list(ctx['checks'])
    if rec_T:
        checks.append('対象の記録: {}（enable_ice = {}, ice_model = {}, ice_vol_frac = {}）'.format(
            rec_path, rec_T.get('enable_ice'), rec_T.get('ice_model'), rec_T.get('ice_vol_frac')))
        if not has_ice(rec_T):
            msg = ('注意: 対象 {} の記録では氷が入っていません（ice_vol_frac = {}）。'
                   '基準と同じ媒質のはずです'.format(tgt, rec_T.get('ice_vol_frac')))
            print('  ' + msg)
            checks.append(msg)
    same = False
    if rec_T and rec_R:
        h_t = rec_T.get('rho', {}).get('rho_map_sha256')
        h_r = rec_R.get('rho', {}).get('rho_map_sha256')
        same = bool(h_t) and h_t == h_r
        checks.append('ランダム場の同一性（rho_map_sha256）: {}'.format('一致' if same else '不一致'))
        if not same:
            print('  注意: ランダム場が基準と違います。同じ場所どうしの差（上限）は意味を持ちません。')
    else:
        checks.append('記録ファイルが無いのでランダム場の同一性を確認できません')

    # 波源遅延と平均媒質の界面
    far_tr, dt_far = load_trace(reference[REF_KEY])
    src_delay = measure_peak(far_tr, dt_far)['t_peak'] - sm.R_REF / sm.C
    events, tand, t_bottom = [], None, np.inf
    if rec_T:
        try:
            events = model_events(rec_T, src_delay)
            tand = layer_tandelta(rec_T)
            t_bottom = domain_bottom_time(rec_T, src_delay)
        except Exception as exc:                          # 理論は印だけなので解析は続ける
            print('  注意: 平均媒質の理論を作れませんでした（印なしで続行）: {}'.format(exc))
            events, tand, t_bottom = [], None, np.inf
    t_axis = np.arange(R.shape[0]) * dt * 1e9
    t_s_model = 2.0 * sm.TX_HEIGHT / sm.C + src_delay
    env_mean = envelope(R).mean(axis=1)
    m = t_axis <= t_s_model + 5.0
    t_s_data = float(t_axis[m][np.argmax(env_mean[m])])
    checks.append('波源遅延 {:.3f} ns、地表反射の走時：理論 {:.3f} ns / データ {:.3f} ns'
                  .format(src_delay, t_s_model, t_s_data))
    if tand:
        checks.append('氷層中央（{:.2f} m）の tanδ：氷あり {:.5f} / 氷なし {:.5f}、ε′ {:.4f} / {:.4f}'
                      .format(tand['z_m'], tand['tand_ice'], tand['tand_dry'],
                              tand['eps_ice'], tand['eps_dry']))
    if np.isfinite(t_bottom):
        checks.append('計算領域の底（PML 上端）の往復走時 {:.2f} ns。これより後は窓の検定に使わない'
                      .format(t_bottom))
    info = {'t_surface_data': t_s_data, 't_bottom': t_bottom, 'same_field': same, 'checks': checks,
            'x_label': ctx['x_label'],
            'inputs': {'level': level, 'target': '{} ({})'.format(tgt, p_t),
                       'reference': '{} ({})'.format(REFERENCE_CONDITION,
                                                     conds[REFERENCE_CONDITION][BSCAN_KEY]),
                       'record (target)': rec_path or '-',
                       'record (reference)': ctx.get('rec_path_R') or '-', 'far_1m': reference[REF_KEY],
                       'freespace at_tx': ctx['fs_path'] or '(差分なし)'}}
    res = analyze_target(tgt, T, R, dt, x, events, tand, info)
    if OUTPUT_LOCATION == 'per_bscan':
        base = os.path.join(os.path.dirname(os.path.abspath(p_t)), OUTPUT_SUBDIRNAME)
    else:
        base = os.path.join(out_base, tgt)

    # --- 対象だけ（判定なし）--------------------------------------------------
    single_dir = os.path.join(base, SINGLE_SUBDIR)
    os.makedirs(single_dir, exist_ok=True)
    plot_single_bscan(res, T, dt, single_dir, tgt)
    plot_single_reflection(res, single_dir, tgt)
    if PLOT_BSCAN_ASCAN:
        plot_bscan_ascan(res, T, dt, single_dir, tgt, 'single12_bscan_ascan', res['a_surf_T'])
    plot_single_decay(res, single_dir, tgt)
    plot_single_shift(res, single_dir, tgt)
    plot_single_spectrogram(res, single_dir, tgt)
    write_single_outputs(res, single_dir)

    # --- 基準（no_ice）との比較（判定つき）--------------------------------------
    out_dir = os.path.join(base, 'vs_' + REFERENCE_CONDITION)
    os.makedirs(out_dir, exist_ok=True)
    plot_bscan(res, T, R, dt, out_dir, tgt)
    plot_reflection_ab(res, out_dir, tgt)
    plot_reflection(res, out_dir, tgt)
    if PLOT_BSCAN_ASCAN:
        plot_bscan_ascan(res, T, dt, out_dir, tgt, 'fig12_bscan_ascan', res['a_surf'], R=R)
    plot_decay(res, out_dir, tgt)
    plot_shift(res, out_dir, tgt)
    plot_spectrogram(res, out_dir, tgt)
    write_outputs(res, out_dir, tgt, info)
    return res


def main():
    level, chosen, conds, reference = select_conditions(JSON_PATH)
    if REFERENCE_CONDITION not in conds:
        raise CmdInputError('基準の条件 "{}" がありません（候補: {}）'
                            .format(REFERENCE_CONDITION, ', '.join(sorted(conds))))
    targets = [c for c in sorted(conds) if c != REFERENCE_CONDITION]
    if TARGET_CONDITIONS:
        targets = [c for c in targets if c in TARGET_CONDITIONS]
    if not targets:
        raise CmdInputError('解析する対象の条件がありません')
    for c in [REFERENCE_CONDITION] + targets:
        p = conds[c].get(BSCAN_KEY)
        if not p or not os.path.exists(p):
            raise CmdInputError('条件 {} の {} がありません: {}'.format(c, BSCAN_KEY, p))
    if REF_KEY not in reference or not os.path.exists(reference[REF_KEY]):
        raise CmdInputError('_reference の {} がありません（波源遅延に必要）'.format(REF_KEY))

    p_r = conds[REFERENCE_CONDITION][BSCAN_KEY]
    R, dt = read_bscan(p_r)
    full_shape = R.shape
    fs, fs_path = None, ''
    if SUBTRACT_FREESPACE:
        fs_path = FREESPACE_AT_TX_PATH or reference.get(FREESPACE_KEY, '')
        if not fs_path or not os.path.exists(fs_path):
            raise CmdInputError('直達波の除去に使う自由空間 at_tx が見つかりません。'
                                'JSON の _reference に {} を置くか FREESPACE_AT_TX_PATH を設定してください'
                                '（除去しないなら SUBTRACT_FREESPACE = False）'.format(FREESPACE_KEY))
        fs, dt_fs = load_trace(fs_path)
        if abs(dt_fs - dt) > 1e-9 * dt:
            raise CmdInputError('自由空間 at_tx の dt が違います: {} vs {}'.format(dt_fs, dt))
        fs = np.pad(fs, (0, max(0, R.shape[0] - len(fs))))[:R.shape[0]]
        R = R - fs[:, None]
    rec_path_R, rec_R = find_record(p_r)
    x_all, has_x = trace_positions(rec_R, R.shape[1])
    sel = np.ones(R.shape[1], dtype=bool)
    if TRACE_X_RANGE_M is not None:
        sel = (x_all >= TRACE_X_RANGE_M[0] - 1e-9) & (x_all <= TRACE_X_RANGE_M[1] + 1e-9)
    R, x = R[:, sel], x_all[sel]
    checks = ['基準: {}、{} 本のトレース（x = {:.2f}-{:.2f} {}）、dt = {:.4f} ps'
              .format(p_r, R.shape[1], x[0], x[-1], 'm' if has_x else '(番号)', dt * 1e12)]
    if rec_R:
        checks.append('基準の記録: {}（enable_ice = {}, ice_vol_frac = {}）'.format(
            rec_path_R, rec_R.get('enable_ice'), rec_R.get('ice_vol_frac')))
        if has_ice(rec_R):
            raise CmdInputError('基準 {} の記録ファイル {} では氷が入っています'
                                '（enable_ice = True, ice_vol_frac = {}）'
                                .format(p_r, rec_path_R, rec_R.get('ice_vol_frac')))
    ctx = {'shape': full_shape, 'sel': sel, 'checks': checks, 'fs_path': fs_path,
           'x_label': 'x [m]' if has_x else 'Trace number (no record file)', 'rec_path_R': rec_path_R}

    out_base = None
    if OUTPUT_LOCATION != 'per_bscan':
        asp.OUTPUT_SUBDIRNAME = OUTPUT_SUBDIRNAME
        out_base = asp.resolve_output_dir(level, {c: conds[c][BSCAN_KEY] for c in conds})
    for tgt in targets:
        run_target(tgt, conds, reference, R, rec_R, dt, x, fs, out_base, level, ctx)


if __name__ == '__main__':
    main()