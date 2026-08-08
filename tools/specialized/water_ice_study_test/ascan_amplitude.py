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
from scipy import signal

from gprMax.exceptions import CmdInputError
from tools.core.outputfiles_merge import get_output_data

# =============================================================================
# 定数
# =============================================================================
# [EDIT HERE] パス JSON のハードコード（利用例は design_ascan_amplitude.md §2 参照）
JSON_PATH = "/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_3x4/water_ice_study_test/out_file_paths.json"

C = 0.29979          # [m/ns] 光速
TX_HEIGHT = 0.35      # [m] 送信機高さ h
R_REF = 1.0           # [m] 参照計算（ref_freespace_1m）の距離

# Level_1 のレゴリス物性
EPS_R_LEVEL1 = 3.0
N_LEVEL1 = np.sqrt(EPS_R_LEVEL1)

# 測定関数パラメータ
SEARCH_HALFWIDTH_NS = 2.0     # [ns] 理論到達時刻からの探索窓半幅
NOISE_WINDOW_NS = (40.0, 50.0)  # [ns] ノイズフロア評価窓

# 合格基準 (design_ascan_amplitude.md §7)
AMP_TOL_DB = 0.5           # 振幅残差 |残差| < 0.5 dB
TIME_TOL_FRAC = 0.01       # 到達時刻残差 < 走時の1%
NOISE_FLOOR_DB = -60.0     # 直達波の -60 dB 以下
T_TOL_FRAC = 0.01          # 透過係数 ±1%

# レベル定義 (design_ascan_amplitude.md §4.3)
LEVEL_EFFECTS = {
    'Level_1': ['geom', 'surface_T'],
    'Level_2': ['geom', 'surface_T', 'absorb_const'],
    'Level_3': ['geom', 'surface_T', 'absorb_debye'],
    'Level_4': ['geom', 'surface_T', 'absorb_debye', 'density_profile'],
}
# 現時点で実装済みのレベル（それ以外は未実装のため実行不可）
IMPLEMENTED_LEVELS = {'Level_1'}

# fig3_waveforms.png で重ね描きする代表深さ
REPRESENTATIVE_DEPTHS_M = [0.50, 1.50, 2.75]

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

    想定する JSON 構成（"_" 始まりのキーはレベル／rx として扱わない）:

        {
          "_reference": {                     # 全レベル共通の参照計算
            "gaussiandot":         {"far_1m": ..., "at_tx": ..., ...},
            "excitation_waveform": {"far_1m": ..., "at_tx": ..., ...}
          },
          "Level_1": {
            "gaussiandot":         {"at_surface": ..., "depth_025": ..., ...},
            "excitation_waveform": {"at_surface": ..., "depth_025": ..., ...}
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

    level_data = all_paths[level]

    # 波形種別の階層があれば選ばせる（gaussiandot / excitation_waveform など）
    kinds = _kind_layer(level_data)
    if kinds:
        kind = _select(sorted(kinds), '波形種別')
        rx_paths = dict(kinds[kind])
    else:
        kind = None
        rx_paths = {k: v for k, v in level_data.items() if not k.startswith('_')}

    # 参照計算：トップレベルをレベル内の設定で上書きする
    reference = _pick_reference(all_paths.get('_reference', {}), kind)
    reference.update(_pick_reference(level_data.get('_reference', {}), kind))

    # PML 内などの解析不能な rx を除外する (design_ascan_amplitude.md §3)
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

    # ディレクトリ名に波形種別のサフィックスが付く構成を許容するため部分一致で判定する
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
# 理論：伝達関数
# =============================================================================
def transfer_geom(d, n):
    """幾何減衰項（2D遠方場、法線入射の見かけの源距離 r_eff を用いる）。
    周波数依存の 1/sqrt(k) は参照計算との比を取ると相殺するため、f に依存しない。
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
    """
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
def analyze_level1(rx_paths, reference):
    if 'far_1m' not in reference:
        raise CmdInputError('_reference.far_1m が JSON にありません（E_ref(f) の校正に必要）')
    e_ref, dt_ref = load_trace(reference['far_1m'])

    n = N_LEVEL1
    level = 'Level_1'
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
        if not np.isclose(dt, dt_ref, rtol=1e-6):
            raise CmdInputError('dt が rx={} と _reference.far_1m で一致しません'.format(key))

        e_theory, t_arr = synth_theory(e_ref, dt_ref, d, level, n)

        # 探索窓の中心：
        #   理論波形 -> t_arr + 波源の内部遅延（ピークが実際に現れる位置）
        #   実測波形 -> その理論ピーク位置
        # measure() は同一関数のままで、中心の与え方だけを変えている。
        theo = measure(e_theory, dt_ref, t_arr + t_src_delay, label=key + ' theory')
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
        if not (np.isclose(dt_at_tx, dt_ref, rtol=1e-6) and np.isclose(dt_free, dt_ref, rtol=1e-6)):
            raise CmdInputError('dt が at_tx / at_tx(freespace) / far_1m で一致しません')

        e_reflect = trace_at_tx - trace_at_tx_free
        e_theory_reflect, t_arr_reflect = synth_theory_reflect(e_ref, dt_ref, n)

        theo = measure(e_theory_reflect, dt_ref, t_arr_reflect + t_src_delay,
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
        trace_surf_free, _ = load_trace(reference['at_surface'])
        amp_surf = measure(trace_surf, dt_surf, TX_HEIGHT / C + t_src_delay,
                           label='at_surface (T check)')['amp_peak']
        amp_surf_free = measure(trace_surf_free, dt_surf, TX_HEIGHT / C + t_src_delay,
                                label='at_surface freespace (T check)')['amp_peak']
        T_meas = amp_surf / amp_surf_free
        T_theory = transfer_surface_T(n)
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
        f.write('  N_LEVEL1 = {:.6f} (eps_r={})\n'.format(N_LEVEL1, EPS_R_LEVEL1))
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

    if level not in IMPLEMENTED_LEVELS:
        raise NotImplementedError(
            '{} は未実装です（実装済み: {}）。Level_2 以降は吸収項の物性値確定後に '
            '追加してください。'.format(level, ', '.join(sorted(IMPLEMENTED_LEVELS))))

    results, t_check, e_ref, dt_ref = analyze_level1(rx_paths, reference)

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