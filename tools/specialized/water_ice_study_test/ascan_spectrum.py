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
EPS_R_LEVEL1 = 3.0
N_LEVEL1 = np.sqrt(EPS_R_LEVEL1)

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
    'Level_3': ['geom', 'surface_T', 'absorb_debye'],
    'Level_4': ['geom', 'surface_T', 'absorb_debye', 'density_profile'],
}
IMPLEMENTED_LEVELS = {'Level_1'}

# 解析対象から除外する rx キー (design_ascan_amplitude.md §3 と共通)
EXCLUDE_KEYS = {'depth_300'}

# 出力先 (レベル親ディレクトリ配下)
OUTPUT_PARENT_DIRNAME = 'analysis'
OUTPUT_SUBDIRNAME = 'ascan_spectrum'

# 自然対数 -> 20*log10 への変換係数 (ln(x) * LN_TO_DB20 == 20*log10(x))
LN_TO_DB20 = 20.0 / np.log(10.0)


# =============================================================================
# 入出力 (ascan_amplitude.py と同一仕様)
# =============================================================================
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
    """波形種別の階層かどうかを判定する。

    値がすべて dict なら「波形種別 -> rx」の階層、
    値が文字列なら「rx -> パス」の階層（旧形式）とみなす。
    """
    entries = {k: v for k, v in node.items() if not k.startswith('_')}
    if entries and all(isinstance(v, dict) for v in entries.values()):
        return entries
    return None


def _pick_reference(ref_node, kind):
    """_reference から、選択した波形種別に対応するエントリを取り出す。"""
    if not ref_node:
        return {}
    nested = _kind_layer(ref_node)
    if nested is None:
        return dict(ref_node)                     # 直下に rx キー（旧形式）
    if kind is not None and kind in nested:
        return dict(nested[kind])
    if kind is not None:
        raise CmdInputError(
            '_reference に波形種別 "{}" のエントリがありません（候補: {}）。'
            'JSON の _reference を確認してください。'.format(kind, ', '.join(sorted(nested))))
    if len(nested) == 1:
        return dict(next(iter(nested.values())))
    return {}


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

    level_data = all_paths[level]

    kinds = _kind_layer(level_data)
    if kinds:
        kind = _select(sorted(kinds), '波形種別')
        rx_paths = dict(kinds[kind])
    else:
        kind = None
        rx_paths = {k: v for k, v in level_data.items() if not k.startswith('_')}

    reference = _pick_reference(all_paths.get('_reference', {}), kind)
    reference.update(_pick_reference(level_data.get('_reference', {}), kind))

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


def transfer_absorb(f, d, params):
    raise NotImplementedError(
        'Level_2 (absorb_const) は未実装です。tanδ 等の物性値が設計書で '
        '未確定のため、Level_1 の合格確認後に実装してください。')


def transfer_absorb_debye(f, d, params):
    raise NotImplementedError('Level_3 (absorb_debye) は未実装です。')


def transfer_density_profile(f, d, params):
    raise NotImplementedError('Level_4 (density_profile) は未実装です。')


def build_transfer(f, d, level, n):
    """LEVEL_EFFECTS に従って伝達関数 H_level(f,d) を効果の積で構成する。

    レベル依存はこの関数と LEVEL_EFFECTS 辞書にのみ現れる。
    """
    effects = LEVEL_EFFECTS[level]
    H = np.ones_like(f, dtype=complex)
    for effect in effects:
        if effect == 'geom':
            H = H * transfer_geom(d, n)
        elif effect == 'surface_T':
            H = H * transfer_surface_T(n)
        elif effect == 'absorb_const':
            H = H * transfer_absorb(f, d, None)
        elif effect == 'absorb_debye':
            H = H * transfer_absorb_debye(f, d, None)
        elif effect == 'density_profile':
            H = H * transfer_density_profile(f, d, None)
        else:
            raise CmdInputError('Unknown effect: {}'.format(effect))

    phase, t_arr = transfer_phase(f, d, n)
    H = H * phase
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
def analyze_level1(rx_paths, reference):
    e_ref, dt_ref = load_trace(reference['far_1m'])
    n = N_LEVEL1
    level = 'Level_1'

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
        if not np.isclose(dt, dt_ref, rtol=1e-6):
            raise CmdInputError('dt が rx={} と _reference.far_1m で一致しません'.format(key))
        if n_samples_common is None:
            n_samples_common = len(trace)
            dt_common = dt
            freq_hz = np.fft.rfftfreq(n_samples_common, d=dt)
        elif len(trace) != n_samples_common or not np.isclose(dt, dt_common, rtol=1e-6):
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
            alpha_abs = alpha_from_abs_lsr(L_abs_meas, d, n)
            tandelta_abs = alpha_to_tandelta(alpha_abs, freq_hz, n)
        if key == d0_key:
            alpha_rel = np.full_like(freq_hz, np.nan)
        else:
            alpha_rel = alpha_from_rel_lsr(L_rel_meas, d, d0, n)
        tandelta_rel = alpha_to_tandelta(alpha_rel, freq_hz, n)

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


def plot_spectra_overview(results, freq_hz, E_ref, output_dir):
    depth_results = sorted(results, key=lambda r: r['depth_m'])
    if not depth_results:
        return
    cmap, norm = _depth_norm_cmap(depth_results)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)

    freq_ghz = freq_hz * 1e-9
    band_lo, band_hi = BAND_GHZ
    band_mask = (freq_ghz >= band_lo) & (freq_ghz <= band_hi)
    ref_max = np.max(np.abs(E_ref)[band_mask])

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 14))

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

    ax = axes[1]
    for r in depth_results:
        color = cmap(norm(r['depth_m']))
        mask = r['mask']
        ax.plot(freq_ghz[mask], r['L_abs_meas'][mask] * LN_TO_DB20, color=color, lw=1.2)
        ax.plot(freq_ghz[mask], r['L_abs_theory'][mask] * LN_TO_DB20, color=color, lw=1.0, ls='--')
    ax.set_xlim(band_lo, band_hi)
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel('Absolute LSR [dB]')
    ax.grid(alpha=0.3)
    ax.set_title('(b) Absolute LSR: measured (solid) vs theory (dashed)')
    fig.colorbar(sm, ax=ax, label='rx depth [m]')

    ax = axes[2]
    for r in depth_results:
        color = cmap(norm(r['depth_m']))
        mask = r['mask']
        resid = (r['L_abs_meas'] - r['L_abs_theory']) * LN_TO_DB20
        ax.plot(freq_ghz[mask], resid[mask], color=color, lw=1.2)
    ax.axhspan(-LSR_TOL_DB, LSR_TOL_DB, color='green', alpha=0.15, label='±{} dB'.format(LSR_TOL_DB))
    ax.axhline(0, color='gray', lw=1)
    ax.set_xlim(band_lo, band_hi)
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel('Residual [dB] (meas - theory)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title('(c) Absolute LSR residual (pass/fail)')
    fig.colorbar(sm, ax=ax, label='rx depth [m]')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig1_spectra_overview.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('Saved:', path)
    plt.close(fig)


def plot_moments(results, output_dir):
    depth_results = sorted(results, key=lambda r: r['depth_m'])
    if not depth_results:
        return
    depths = np.array([r['depth_m'] for r in depth_results])

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 8))

    specs = [
        ('fc_meas_ghz', 'fc_theory_ghz', 'f_c [GHz]', FC_TOL_MHZ * 1e-3),
        ('sigma_f_meas_ghz', 'sigma_f_theory_ghz', 'sigma_f [GHz]', None),
        ('flo_m10_meas_ghz', 'flo_m10_theory_ghz', 'f_lo (-10dB) [GHz]', None),
        ('fhi_m10_meas_ghz', 'fhi_m10_theory_ghz', 'f_hi (-10dB) [GHz]', None),
    ]
    titles = ['f_c', 'sigma_f', 'f_lo (-10dB)', 'f_hi (-10dB)']
    for col, (meas_key, theory_key, label, tol) in enumerate(specs):
        ax = axes[0, col]
        meas = np.array([r[meas_key] for r in depth_results])
        theory = np.array([r[theory_key] for r in depth_results])
        ax.plot(theory, depths, 'r-', label='Theory')
        ax.plot(meas, depths, 'ko', label='Measured')
        ax.set_xlabel(label)
        ax.set_title(titles[col])
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        if col == 0:
            ax.set_ylabel('rx depth [m]')
            ax.legend()

        ax2 = axes[1, col]
        resid = meas - theory
        ax2.axvline(0, color='gray', lw=1)
        if tol is not None:
            ax2.axvspan(-tol, tol, color='green', alpha=0.15)
        ax2.plot(resid, depths, 'ko-')
        ax2.set_xlabel('Residual ({})'.format(label))
        ax2.invert_yaxis()
        ax2.grid(alpha=0.3)
        if col == 0:
            ax2.set_ylabel('rx depth [m]')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig2_moments.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('Saved:', path)
    plt.close(fig)


def plot_relative_lsr(results, freq_hz, d0, output_dir):
    depth_results = sorted(results, key=lambda r: r['depth_m'])
    if not depth_results:
        return
    cmap, norm = _depth_norm_cmap(depth_results)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    freq_ghz = freq_hz * 1e-9
    band_lo, band_hi = BAND_GHZ

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

    ax = axes[0]
    for r in depth_results:
        color = cmap(norm(r['depth_m']))
        mask = r['mask']
        ax.plot(freq_ghz[mask], r['L_rel_meas'][mask] * LN_TO_DB20, color=color, lw=1.2)
        ax.plot(freq_ghz[mask], r['L_rel_theory'][mask] * LN_TO_DB20, color=color, lw=1.0, ls='--')
    ax.set_xlim(band_lo, band_hi)
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel('Relative LSR [dB] (ref depth={:.2f} m)'.format(d0))
    ax.grid(alpha=0.3)
    ax.set_title('(a) Relative LSR: measured (solid) vs theory (dashed)')
    fig.colorbar(sm, ax=ax, label='rx depth [m]')

    ax = axes[1]
    for r in depth_results:
        color = cmap(norm(r['depth_m']))
        mask = r['mask']
        resid = (r['L_rel_meas'] - r['L_rel_theory']) * LN_TO_DB20
        ax.plot(freq_ghz[mask], resid[mask], color=color, lw=1.2)
    ax.axhline(0, color='gray', lw=1)
    ax.set_xlim(band_lo, band_hi)
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel('Residual [dB]')
    ax.grid(alpha=0.3)
    ax.set_title('(b) Relative LSR residual')
    fig.colorbar(sm, ax=ax, label='rx depth [m]')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig3_relative_lsr.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('Saved:', path)
    plt.close(fig)


def plot_attenuation(results, freq_hz, output_dir):
    depth_results = sorted(results, key=lambda r: r['depth_m'])
    if not depth_results:
        return
    cmap, norm = _depth_norm_cmap(depth_results)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    freq_ghz = freq_hz * 1e-9
    band_lo, band_hi = BAND_GHZ

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

    ax = axes[0]
    for r in depth_results:
        color = cmap(norm(r['depth_m']))
        mask = r['mask']
        ax.plot(freq_ghz[mask], r['alpha_abs'][mask], color=color, lw=1.2)
    ax.axhline(0.0, color='r', lw=1.5, ls='--', label='Theory (Level_1: alpha=0)')
    ax.set_xlim(band_lo, band_hi)
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel('alpha(f) [1/m]')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title('(a) Attenuation coefficient (collapse check)')
    fig.colorbar(sm, ax=ax, label='rx depth [m]')

    ax = axes[1]
    for r in depth_results:
        color = cmap(norm(r['depth_m']))
        mask = r['mask']
        ax.plot(freq_ghz[mask], r['tandelta_abs'][mask], color=color, lw=1.2)
    ax.axhline(0.0, color='r', lw=1.5, ls='--', label='Theory (Level_1: tan_delta=0)')
    ax.set_xlim(band_lo, band_hi)
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel('tan_delta(f)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title('(b) Loss tangent')
    fig.colorbar(sm, ax=ax, label='rx depth [m]')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig4_attenuation.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('Saved:', path)
    plt.close(fig)


def plot_phase(results, freq_hz, output_dir):
    depth_results = sorted(results, key=lambda r: r['depth_m'])
    if not depth_results:
        return
    cmap, norm = _depth_norm_cmap(depth_results)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    freq_ghz = freq_hz * 1e-9
    band_lo, band_hi = BAND_GHZ

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

    ax = axes[0]
    for r in depth_results:
        color = cmap(norm(r['depth_m']))
        mask = r['mask']
        ax.plot(freq_ghz[mask], r['tau_g'][mask], color=color, lw=1.2)
        ax.axhline(r['t_arr_ns'], color=color, lw=1.0, ls='--')
    ax.set_xlim(band_lo, band_hi)
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel('Group delay [ns]')
    ax.grid(alpha=0.3)
    ax.set_title('(a) Group delay: measured (solid) vs theory t_arr (dashed)')
    fig.colorbar(sm, ax=ax, label='rx depth [m]')

    ax = axes[1]
    for r in depth_results:
        color = cmap(norm(r['depth_m']))
        mask = r['mask']
        resid = r['tau_g'] - r['t_arr_ns']
        ax.plot(freq_ghz[mask], resid[mask], color=color, lw=1.2)
    ax.axhline(0, color='gray', lw=1)
    ax.set_xlim(band_lo, band_hi)
    ax.set_xlabel('Frequency [GHz]')
    ax.set_ylabel('Residual [ns] (tau_g - t_arr)')
    ax.grid(alpha=0.3)
    ax.set_title('(b) Group delay residual (numerical dispersion)')
    fig.colorbar(sm, ax=ax, label='rx depth [m]')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig5_phase.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('Saved:', path)
    plt.close(fig)


# =============================================================================
# 出力
# =============================================================================
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
        f.write('  N_LEVEL1 = {:.6f} (eps_r={})\n'.format(N_LEVEL1, EPS_R_LEVEL1))
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

    if level not in IMPLEMENTED_LEVELS:
        raise NotImplementedError(
            '{} は未実装です（実装済み: {}）。Level_2 以降は吸収項の物性値確定後に '
            '追加してください。'.format(level, ', '.join(sorted(IMPLEMENTED_LEVELS))))

    check_paths_exist(rx_paths, reference)

    results, freq_hz, E_ref, d0_key, d0 = analyze_level1(rx_paths, reference)

    output_dir = resolve_output_dir(level, rx_paths)
    os.makedirs(output_dir, exist_ok=True)

    plot_spectra_overview(results, freq_hz, E_ref, output_dir)
    plot_moments(results, output_dir)
    plot_relative_lsr(results, freq_hz, d0, output_dir)
    plot_attenuation(results, freq_hz, output_dir)
    plot_phase(results, freq_hz, output_dir)
    write_csv(results, output_dir)
    write_npz(results, freq_hz, E_ref, output_dir)
    write_run_info(level, kind, JSON_PATH, results, output_dir)

    print('\nAll outputs saved to:', output_dir)


if __name__ == '__main__':
    main()
