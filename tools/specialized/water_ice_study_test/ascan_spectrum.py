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
    'Level_3': ['geom', 'surface_T', 'absorb_debye'],
    'Level_4': ['geom', 'surface_T', 'absorb_debye', 'density_profile'],
}
IMPLEMENTED_LEVELS = {'Level_1', 'Level_2'}

# JSON の下位選択階層につけるラベル（階層が深いほうまで使う）
SUBLEVEL_LABELS = ['波形種別', 'サブ条件', 'サブ条件']

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
# したがって「tan_delta 一定」は Level 3（分散性）の領域であり、
# Level 2（非分散な損失媒質）の物理的に自己整合な姿は「sigma 一定」である。
LEVEL2_LOSS_MODEL = 'conductivity'   # 'conductivity' … gprMax の #material に対応（既定）
                                     # 'tan_delta'    … 参考用。Level 3 相当の理想化
LEVEL2_SIGMA = 0.0035                # [S/m] #material の第 2 引数と一致させること。
                                     #   プロファイル計算の 0 vol% ice / 1.25 GHz の値。
                                     #   tan_delta = 0.01678 @ 1.25 GHz に相当。
LEVEL2_TAN_DELTA = 0.0155            # LEVEL2_LOSS_MODEL='tan_delta' のときのみ使う

ETA0 = 376.730313668                 # [Ohm] 真空の波動インピーダンス
EPS0 = 8.8541878128e-12              # [F/m] 真空の誘電率

EXCLUDE_KEYS = {'depth_300'}

# 作図
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
            H = H * transfer_absorb(f, d, n)
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
def analyze_level(rx_paths, reference, level):
    e_ref, dt_ref = load_trace(reference['far_1m'])
    n = N_REGOLITH
    if 'absorb_const' in LEVEL_EFFECTS[level]:
        print('{} の媒質: {}'.format(level, describe_level2_medium(n)))

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
    """(a) 生スペクトル比較、(b) 帯域要約（f_lo / f_c / f_hi / sigma_f）。

    (b) のエラーバーの意味:
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
    for ax, key_m, key_t, title, ylabel in panels:
        for r in depth_results:
            color = cmap(norm(r['depth_m']))
            mask = r['mask']
            ax.plot(freq_ghz[mask], r[key_m][mask] * LN_TO_DB20, color=color, lw=1.2)
            ax.plot(freq_ghz[mask], r[key_t][mask] * LN_TO_DB20, color=color, lw=1.0, ls='--')
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
    for ax, key_m, key_t, show_tol, title in resid_panels:
        for r in depth_results:
            color = cmap(norm(r['depth_m']))
            mask = r['mask']
            resid = (r[key_m] - r[key_t]) * LN_TO_DB20
            ax.plot(freq_ghz[mask], resid[mask], color=color, lw=1.2)
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

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    specs = [
        (axes[0, 0], 'alpha_abs', r'$\alpha(f)$ [1/m]',
         '(a) Attenuation from absolute LSR (collapse check)'),
        (axes[0, 1], 'alpha_rel', r'$\alpha(f)$ [1/m]',
         '(b) Attenuation from relative LSR (field-measurable)'),
        (axes[1, 0], 'tandelta_abs', r'$\tan\delta(f)$',
         '(c) Loss tangent from absolute LSR'),
        (axes[1, 1], 'tandelta_rel', r'$\tan\delta(f)$',
         '(d) Loss tangent from relative LSR'),
    ]
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
        if 'absorb_const' in LEVEL_EFFECTS[level]:
            th = (level2_alpha(freq_hz, n) if key.startswith('alpha')
                  else level2_tandelta(freq_hz, n))
            ax.plot(freq_ghz, th, color='r', ls='--', lw=1.5,
                    label='Theory ({})'.format(level))
        else:
            ax.axhline(0.0, color='r', ls='--', lw=1.5,
                       label='Theory ({}: alpha=0)'.format(level))
        ax.set_xlim(band_lo, band_hi)
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