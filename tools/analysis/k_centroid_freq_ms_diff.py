#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
k_ms_centroid_diff.py
=====================
複数シード対応 STFT centroid 差分解析ツール (水氷あり - 水氷なし)

ice / No_Ice の B-scan をトレース方向に連結し、
2 つの pairing モード (same / cross) x nseed = 1..N について
Δcentroid / Δshift-rate のプロファイル・領域統計・SEM スケーリングを出力する。

アルゴリズム (STFT 設定・valid_mask・平滑 sigma・shift_rate・理論計算の各定数) は
すべて k_centroid_freq.py と同一。既存スクリプトは一切変更しない。

Usage:
    python k_ms_centroid_diff.py <path>
    python k_ms_centroid_diff.py <path> --noice_dir <no_ice_case_dir>
    python k_ms_centroid_diff.py <path> --modes same          # 健全性テスト用
    (引数なしで起動した場合は input() でパスを尋ねる)

<path> は
  (a) case ディレクトリ  例) .../Eval_thick/rand_amp_005/thick_10
  (c) 親ディレクトリ      例) .../Eval_thick/rand_amp_005   (配下 case を自動列挙)
のいずれか。
"""

import os
import sys
import json
import glob
import re
import argparse
import hashlib
import warnings
import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')          # 図はすべてファイル保存のみ (対話表示しない)
import matplotlib.pyplot as plt
from scipy import signal
from scipy import constants as const
from scipy.ndimage import gaussian_filter

# 既存スクリプトと同一の import 経路
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.core.outputfiles_merge import get_output_data

# NumPy 2.0 で np.trapz が np.trapezoid に改名されたことへの互換シム。
# 数値計算の中身は既存コード (np.trapz) と完全に同一。
_TRAPZ = getattr(np, 'trapezoid', None) or np.trapz

# 実行中のスクリプト名 (README の再現コマンド等に使う。リネームされても追従する)
SCRIPT_NAME = os.path.basename(os.path.abspath(__file__))


# =============================================================================
# [EDIT HERE] 実行環境に応じて変更する定数
# =============================================================================
# 入射波スペクトル計算用の A-scan 出力ファイル (全 case 共通の参照波形)。
# 注意: 各 case の Ascan.out は氷を含む実媒質を伝搬済みのため入射スペクトルには
#       使えない。必ず waveform_test の専用シミュレーション結果を指定する。
ASCAN_OUTFILE_PATH = "/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_3x4/waveform_test/gaussiandot_1.25GHz_underground/result/Ascan.out"

# 異シード差分のシード番号オフセット (ice seed k <-> noice seed k + OFFSET)。
# !!! 将来 No_Ice のシード数を変更する場合はこの値を必ず見直すこと !!!
#     現在は No_Ice が Seed_0..7 の 8 個あり、ice 側 4 個 (Seed_0..3) と
#     重複しないペアを作るために 4 としている。
#     例) ice を 6 seed に増やすなら No_Ice は 12 seed 必要で OFFSET=6 になる。
CROSS_SEED_OFFSET = 4

# 領域統計の系列相関長 [ns] (centroid は Gaussian 平滑により時間方向に相関を持つ)
CORR_LEN_NS = 3.0


# =============================================================================
# 解析パラメータ (k_centroid_freq.py と 1 文字も変えないこと)
# =============================================================================
NPERSEG   = 256
NOVERLAP  = NPERSEG * 3 // 4          # = 192
WINDOW    = 'hann'

FREQ_MIN  = 0.25                      # [GHz]
FREQ_MAX  = 6.0                       # [GHz]

POWER_THRESHOLD_DB = -125.0           # [dB]
SMOOTH_SIGMA = (3, 3)                 # Gaussian smoothing sigma for (time, trace)
EPS = 1e-30

# 理論計算の定数 (k_centroid_freq.py と同一)
ANTENNA_HEIGHT = 0.35                 # [m] 送信機高さ
SYSTEM_LAG_NS  = 0.837                # [ns] システムラグ
RX_DEPTH       = 0.10                 # [m] 受信機の埋設深さ
ANCHOR_FREQ    = 450e6                # [Hz] Method A の損失アンカー周波数

# .in から読めなかった場合の既定値 (k_centroid_freq_diff.py と同一)
DEFAULT_DEBYE = {'tau1': 4.6212e-11, 'tau2': 2.82195e-10,
                 'de_ratio': 0.261 / (0.261 + 0.088)}
DEFAULT_EPS_ICE = 3.17                # 純氷の実部
ICE_LOSS_TAN    = 6e-5                # 純氷の損失正接 (既存 diff コードの 3.17*(1-1j*6e-5) と同一)

PLATEAU_FRACTION = 0.25               # プラトー値: 層区間の下端側 1/4


# =============================================================================
# 物理モデル (k_centroid_freq.py からの完全移植)
# =============================================================================
def get_eps_static(z_m):
    """深さ z [m] から静的実部とロスタンジェントを計算
    (Heiken1991 Fig 9.54 の 450 MHz 計測経験式; イルメナイト20wt%考慮)"""
    z_cm = z_m * 100.0
    rho = 1.92 * (z_cm + 12.2) / (z_cm + 18.0)
    eps_static = 1.843 ** rho
    tan_d = 10 ** (0.033 * 20.0 + 0.231 * rho - 3.061)
    return eps_static, tan_d


def get_eps_regolith(z_m, omega, d_params, anchor_freq=ANCHOR_FREQ):
    """指定深さ z_m [m] と角周波数配列 omega に対するレゴリス母材の複素誘電率。
    2 極 Debye (Method A: 損失アンカー方式)。水氷層は含まない。"""
    eps_static, tan_d = get_eps_static(z_m)

    tau1 = d_params['tau1']
    tau2 = d_params['tau2']
    de_ratio = d_params['de_ratio']

    w_a = 2.0 * np.pi * anchor_freq
    unit_im_wa = (de_ratio * (w_a * tau1) / (1.0 + (w_a * tau1) ** 2) +
                  (1.0 - de_ratio) * (w_a * tau2) / (1.0 + (w_a * tau2) ** 2))
    eps_im_target = eps_static * tan_d
    de_tot = eps_im_target / unit_im_wa

    eps_inf = max(eps_static - de_tot, 1.0)
    de_tot = eps_static - eps_inf
    de1 = de_tot * de_ratio
    de2 = de_tot * (1.0 - de_ratio)

    eps_regolith = (eps_inf
                    + de1 / (1.0 + 1j * omega * tau1)
                    + de2 / (1.0 + 1j * omega * tau2))
    return eps_regolith


def maxwell_garnett(eps_host, eps_incl, f_incl):
    """Maxwell-Garnett 混合則 (既存 diff コードと同一式)。"""
    return eps_host + 3.0 * f_incl * eps_host * (eps_incl - eps_host) / \
        (eps_incl + 2.0 * eps_host - f_incl * (eps_incl - eps_host))


def surface_delay_ns(antenna_height=ANTENNA_HEIGHT, system_lag_ns=SYSTEM_LAG_NS):
    """地表面反射の到達時刻 (プロット上の基準線 'Surface') [ns]。"""
    return antenna_height * 2 / 0.3 + system_lag_ns


def smooth_masked(data, mask, sigma):
    """NaN を考慮した Gaussian 平滑 (k_centroid_freq.py と同一)。"""
    filled = np.where(mask, data, 0.0)
    sm_data = gaussian_filter(filled, sigma=sigma)
    sm_weight = gaussian_filter(mask.astype(float), sigma=sigma)
    out = np.full_like(sm_data, np.nan)
    np.divide(sm_data, sm_weight, out=out, where=(sm_weight > 1e-6))
    # スムージングで値が周囲へにじみ出るのを防ぐため、
    # 元々有効だったピクセル以外は NaN に戻す
    out[~mask] = np.nan
    return out


def shift_rate(freq_map, dt_stft):
    """[GHz/ns] (k_centroid_freq.py と同一)。"""
    return np.gradient(freq_map, dt_stft, axis=0)


# =============================================================================
# 入力とパス探索
# =============================================================================
# seed ディレクトリ名。大文字小文字を問わず `Seed_0` / `seed_0` の双方を受理する。
# glob は（大文字小文字を区別しないファイルシステム上でも）パターン照合自体は
# 区別するため、`Seed_*` では `seed_0` にマッチしない。listdir + 正規表現で判定し、
# ディレクトリ名を番号から組み立てる処理は書かない（実在する名前をそのまま使う）。
SEED_DIR_REGEX = re.compile(r'^seed_(\d+)$', re.IGNORECASE)


def resolve_ci(parent, name):
    """parent/name のパスを返す。完全一致が無ければ大文字小文字を無視して探す。

    `Bscan` / `bscan`、`Ascan` / `ascan` のような表記ゆれで解析全体が
    停止しないための保険。見つからなければ None。
    """
    if not parent:
        return None
    direct = os.path.join(parent, name)
    if os.path.exists(direct):
        return direct
    try:
        entries = os.listdir(parent)
    except OSError:
        return None
    low = name.lower()
    for e in entries:
        if e.lower() == low:
            return os.path.join(parent, e)
    return None


def find_seed_dirs(case_dir):
    """case_dir 直下の seed ディレクトリを [(番号, 実在パス), ...] 昇順で返す。"""
    try:
        entries = os.listdir(case_dir)
    except OSError:
        return []
    out = []
    for name in entries:
        m = SEED_DIR_REGEX.match(name)
        if m and os.path.isdir(os.path.join(case_dir, name)):
            out.append((int(m.group(1)), os.path.join(case_dir, name)))
    return sorted(out, key=lambda x: x[0])


def is_case_dir(path):
    return bool(find_seed_dirs(path))


def _subdirs(path):
    try:
        return sorted(d for d in os.listdir(path)
                      if os.path.isdir(os.path.join(path, d)))
    except OSError:
        return []


def discover_cases(input_path):
    """(a) case ディレクトリ / (c) 親ディレクトリ を判別して case 一覧を返す。

    判別方法: 指定パス直下に seed ディレクトリがあれば (a)、
              無ければ (c) として seed ディレクトリを持つサブディレクトリを case とする。
    """
    input_path = os.path.abspath(os.path.expanduser(input_path.strip()))
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"指定パスがディレクトリとして存在しません: {input_path}")

    if is_case_dir(input_path):
        print(f"[INPUT] case ディレクトリとして認識: {input_path}")
        return [input_path]

    cases = sorted(d for d in glob.glob(os.path.join(input_path, '*'))
                   if os.path.isdir(d) and find_seed_dirs(d))
    if not cases:
        raise FileNotFoundError(
            f"seed ディレクトリ (Seed_N / seed_N) も、それを持つサブディレクトリも"
            f"見つかりません: {input_path}\n"
            f"  直下のディレクトリ: {_subdirs(input_path)}\n"
            f"  -> case ディレクトリ (直下に Seed_N/) か、その親ディレクトリを指定してください。")
    print(f"[INPUT] 親ディレクトリとして認識: {input_path}")
    print(f"[INPUT] 検出した case ({len(cases)} 件): {[os.path.basename(c) for c in cases]}")
    return cases


def list_seeds(case_dir):
    """case_dir 配下の <seed_N>/Bscan/Bscan.json を列挙する。

    Returns
    -------
    jsons  : dict {seed 番号: Bscan.json の実在パス}
    dirs   : dict {seed 番号: seed ディレクトリの実在パス}
    no_json: [(seed 番号, seed ディレクトリ)] … ディレクトリはあるが json が無いもの
    """
    jsons, dirs, no_json = {}, {}, []
    for num, sdir in find_seed_dirs(case_dir):
        dirs[num] = sdir
        bdir = resolve_ci(sdir, 'Bscan')
        jpath = resolve_ci(bdir, 'Bscan.json') if bdir else None
        if jpath is None:
            no_json.append((num, sdir))
        else:
            jsons[num] = jpath
    return dict(sorted(jsons.items())), dict(sorted(dirs.items())), no_json


def validate_ice_seeds(case_dir, jsons, dirs, no_json):
    """ice 側 seed が 0 から連番で、全て Bscan.json を持つことを検証。"""
    if not jsons:
        raise FileNotFoundError(
            f"ice 側の seed ディレクトリ (Seed_N / seed_N) が見つかりません: {case_dir}\n"
            f"  直下のディレクトリ: {_subdirs(case_dir)}")
    if no_json:
        detail = '\n'.join(f"    seed {n}: {d}" for n, d in no_json)
        raise FileNotFoundError(
            f"Bscan/Bscan.json が見つからない seed があります: {case_dir}\n{detail}")

    idxs = sorted(jsons)
    expected = list(range(len(idxs)))
    if idxs != expected:
        missing = sorted(set(range(max(idxs) + 1)) - set(idxs))
        raise FileNotFoundError(
            f"ice 側の seed が 0 からの連番ではありません: {case_dir}\n"
            f"  検出: {[os.path.basename(dirs[i]) for i in idxs]}\n"
            f"  欠番: {['seed %d' % i for i in missing]}\n"
            f"  -> seed の欠損は解析結果を歪めるため処理を中止します。")
    return idxs


def extract_rand_amp(path):
    """パスから rand_amp_(\\d+) を抽出。見つからなければ None。"""
    m = re.search(r'rand_amp_(\d+)', path)
    return m.group(1) if m else None


def extract_eval_type(path):
    """パスから Eval_XXX の XXX を抽出。見つからなければ ''。"""
    m = re.search(r'Eval_([^/\\]+)', path)
    return m.group(1) if m else ''


def derive_noice_dir(case_dir):
    """パス中の Eval_*/rand_amp_XXX/<case> を No_Ice/rand_amp_XXX に置換して導出。"""
    case_dir = os.path.abspath(case_dir)
    pattern = re.compile(r'Eval_[^/\\]+[/\\]rand_amp_(\d+)[/\\][^/\\]+[/\\]?$')
    m = pattern.search(case_dir)
    if not m:
        raise ValueError(
            f"No_Ice ディレクトリを自動導出できません: {case_dir}\n"
            f"  期待するパス構造: .../Eval_<type>/rand_amp_<XXX>/<case>\n"
            f"  -> --noice_dir で明示指定してください。")
    noice_dir = case_dir[:m.start()] + os.path.join('No_Ice', f'rand_amp_{m.group(1)}')
    return noice_dir


def required_noice_seeds(mode, nseed_max):
    """当該モードで必要になる No_Ice seed 番号の集合。"""
    offset = 0 if mode == 'same' else CROSS_SEED_OFFSET
    return [k + offset for k in range(nseed_max)]


def pair_seed_indices(mode, nseed):
    """(ice seed 番号リスト, noice seed 番号リスト) を返す。"""
    ice_idx = list(range(nseed))
    offset = 0 if mode == 'same' else CROSS_SEED_OFFSET
    noice_idx = [k + offset for k in ice_idx]
    return ice_idx, noice_idx


def validate_noice_seeds(noice_dir, jsons, dirs, no_json, modes, nseed_max):
    """必要な No_Ice seed が揃っているか検証 (欠損はエラー停止)。"""
    if not jsons:
        raise FileNotFoundError(
            f"No_Ice 側の <seed_N>/Bscan/Bscan.json が 1 つも見つかりません: {noice_dir}\n"
            f"  直下のディレクトリ: {_subdirs(noice_dir)}")
    nojson_idx = {n for n, _ in no_json}
    for mode in modes:
        need = required_noice_seeds(mode, nseed_max)
        missing = [i for i in need if i not in jsons]
        if missing:
            detail = []
            for i in missing:
                if i in nojson_idx:
                    detail.append(f"seed {i} (ディレクトリはあるが Bscan/Bscan.json が無い)")
                else:
                    detail.append(f"seed {i} (ディレクトリ自体が無い)")
            raise FileNotFoundError(
                f"mode='{mode}' に必要な No_Ice seed が見つかりません。\n"
                f"  No_Ice dir : {noice_dir}\n"
                f"  必要な seed : {['seed %d' % i for i in need]}"
                f" (CROSS_SEED_OFFSET={CROSS_SEED_OFFSET})\n"
                f"  存在する seed: {[os.path.basename(dirs[i]) for i in sorted(dirs)]}\n"
                f"  不足: {detail}\n"
                f"  -> No_Ice のシード数を増やすか、CROSS_SEED_OFFSET を見直してください。")


# =============================================================================
# .in からのパラメータ読み取り (ice 側 Seed_0/Ascan/*.in)
# =============================================================================
def read_in_params(seed0_dir, seed0_json=None):
    """ice 側 seed 0 の Ascan/ 直下の *.in から氷層・Debye パラメータを読む。

    seed 連結時に参照先が曖昧になるのを避けるため seed 0 を明示的に使う。
    ディレクトリ名は番号から組み立てず、実在するパス (seed0_dir) を受け取る。
    `Ascan` が見つからない場合のみ、seed 0 の Bscan.json が指す
    geometry_json のディレクトリ (共通仕様 §3 が「同一箇所に到達する」と
    述べている場所) にフォールバックする。
    氷層定義 (ice_top / ice_bot) が読めない場合はエラー停止。
    """
    in_dir = resolve_ci(seed0_dir, 'Ascan')
    in_files = sorted(glob.glob(os.path.join(in_dir, '*.in'))) if in_dir else []

    if not in_files and seed0_json:
        try:
            with open(seed0_json) as f:
                _p = json.load(f)
            geom = _p.get('geometry_settings', {}).get('geometry_json', '')
            alt = os.path.dirname(geom) if geom else ''
            alt_files = sorted(glob.glob(os.path.join(alt, '*.in'))) if alt else []
        except Exception:
            alt, alt_files = '', []
        if alt_files:
            print(f"  Note: {seed0_dir}/Ascan に *.in が無いため "
                  f"geometry_json のディレクトリを使用します: {alt}")
            in_dir, in_files = alt, alt_files

    if not in_files:
        raise FileNotFoundError(
            f"*.in が見つかりません。\n"
            f"  探索先: {resolve_ci(seed0_dir, 'Ascan') or os.path.join(seed0_dir, 'Ascan')}\n"
            f"  seed 0 直下のディレクトリ: {_subdirs(seed0_dir)}\n"
            f"  -> ice 側 seed 0 の Ascan/ に .in ファイルが必要です (共通仕様 §3)。")

    p = {'tau1': DEFAULT_DEBYE['tau1'], 'tau2': DEFAULT_DEBYE['tau2'],
         'de_ratio': DEFAULT_DEBYE['de_ratio'],
         'f_ice': None, 'ice_top': None, 'ice_bot': None,
         'eps_ice': None, 'in_file': None}

    for in_file in in_files:
        try:
            with open(in_file, 'r', encoding='utf-8') as fin:
                content = fin.read()
        except Exception as e:
            print(f"Warning: Could not read {in_file}: {e}")
            continue

        m_tau1 = re.search(r'DEBYE_TAU1\s*=\s*([0-9\.eE\+\-]+)', content)
        m_tau2 = re.search(r'DEBYE_TAU2\s*=\s*([0-9\.eE\+\-]+)', content)
        m_ratio = re.search(r'DE_RATIO\s*=\s*(.+)', content)
        m_disp = re.search(r'#add_dispersion_debye:\s*\d+\s+([0-9\.eE\+\-]+)\s+'
                           r'[0-9\.eE\+\-]+\s+([0-9\.eE\+\-]+)', content)

        if m_tau1:
            p['tau1'] = float(m_tau1.group(1))
        if m_tau2:
            p['tau2'] = float(m_tau2.group(1))
        if m_ratio:
            # コメント部分などを除外して計算 (既存コードと同一の扱い)
            expr = m_ratio.group(1).split('#')[0].strip()
            p['de_ratio'] = float(eval(expr, {'__builtins__': {}}, {}))
        elif m_disp:
            de1, de2 = float(m_disp.group(1)), float(m_disp.group(2))
            if (de1 + de2) > 0:
                p['de_ratio'] = de1 / (de1 + de2)

        for key, pat in (('f_ice',   r'f_ice\s*=\s*([0-9\.eE\+\-]+)'),
                         ('ice_top', r'ice_top\s*=\s*([0-9\.eE\+\-]+)'),
                         ('ice_bot', r'ice_bot\s*=\s*([0-9\.eE\+\-]+)'),
                         ('eps_ice', r'eps_ice\s*=\s*([0-9\.eE\+\-]+)')):
            m = re.search(pat, content)
            if m:
                p[key] = float(m.group(1))

        # 値を上書きしたファイルを記録する (複数 .in がある場合は後勝ち)
        if any((m_tau1, m_tau2, m_ratio, m_disp)) or \
                any(re.search(pat, content) for pat in
                    (r'f_ice\s*=', r'ice_top\s*=', r'ice_bot\s*=')):
            p['in_file'] = in_file
    if p['in_file'] is None:
        p['in_file'] = in_files[0]

    # --- 氷層定義の検証 (読めなければ理論線が描けず解析の意味が失われる) ---
    if p['ice_top'] is None or p['ice_bot'] is None:
        raise ValueError(
            f"氷層定義 (ice_top / ice_bot) を .in から読み取れませんでした: {in_dir}\n"
            f"  読み取り結果: ice_top={p['ice_top']}, ice_bot={p['ice_bot']}\n"
            f"  -> 層区間・理論署名が定義できないため処理を中止します。")
    if not (p['ice_bot'] > p['ice_top']):
        raise ValueError(
            f"氷層の深さが不正です (ice_bot <= ice_top): "
            f"ice_top={p['ice_top']}, ice_bot={p['ice_bot']} ({in_dir})")
    if p['f_ice'] is None:
        raise ValueError(
            f"f_ice を .in から読み取れませんでした: {in_dir}\n"
            f"  -> 理論差分署名が計算できないため処理を中止します。")
    if p['f_ice'] <= 0:
        raise ValueError(
            f"f_ice = {p['f_ice']} です。氷ありの case を指定してください ({in_dir})。")

    if p['eps_ice'] is None:
        p['eps_ice'] = DEFAULT_EPS_ICE
        p['eps_ice_source'] = f'default ({DEFAULT_EPS_ICE})'
    else:
        p['eps_ice_source'] = '.in'

    print(' ')
    print(' === Parameters read from .in ===')
    print(f"  file      : {p['in_file']}")
    print(f"  f_ice     : {p['f_ice']}")
    print(f"  ice_top   : {p['ice_top']} m")
    print(f"  ice_bot   : {p['ice_bot']} m")
    print(f"  eps_ice   : {p['eps_ice']}  (source: {p['eps_ice_source']})")
    print(f"  DEBYE_TAU1: {p['tau1']}")
    print(f"  DEBYE_TAU2: {p['tau2']}")
    print(f"  DE_RATIO  : {p['de_ratio']}")
    print(' ================================')
    print(' ')
    return p


# =============================================================================
# B-scan 読み込みと 1 seed 分の centroid / power マップ
# =============================================================================
def load_bscan(json_path):
    """Bscan.json から B-scan データ・dt・gpr_step を読む。"""
    with open(json_path) as f:
        params = json.load(f)
    outfile = params['data']
    gpr_step = params['antenna_settings']['src_step']
    outputdata, dt = get_output_data(outfile, 1, 'Ez')
    return outputdata, dt, gpr_step, params


def stft_axes(trace, fs):
    """STFT の周波数軸・時間軸 (k_centroid_freq.py と同一設定)。"""
    f_axis, t_axis, _ = signal.stft(trace, fs=fs, window=WINDOW,
                                    nperseg=NPERSEG, noverlap=NOVERLAP)
    return f_axis, t_axis


def compute_seed_maps(outputdata, dt):
    """1 seed 分の centroid_map / power_map を計算して返す。

    ※ STFT もパワーマスクの規格化 (trace_peak) もトレースごとに独立なので、
       seed ごとにここまで計算してから hstack しても、
       「連結してから STFT」と数値的に完全に等価。
       トレース間を結合する処理 (Gaussian 平滑・中央値/IQR) は
       必ず連結後の配列に対して行う (concat_and_profile 参照)。
    """
    dt_ns = dt * 1e9
    fs = 1.0 / dt_ns
    n_samples, n_traces = outputdata.shape

    f_axis, t_axis = stft_axes(outputdata[:, 0], fs)
    freq_mask = (f_axis >= FREQ_MIN) & (f_axis <= FREQ_MAX)
    valid_freq = f_axis[freq_mask]
    n_time = t_axis.size

    centroid_map = np.zeros((n_time, n_traces))
    power_map = np.zeros((n_time, n_traces))

    for itrace in range(n_traces):
        _, _, Zxx = signal.stft(outputdata[:, itrace], fs=fs, window=WINDOW,
                                nperseg=NPERSEG, noverlap=NOVERLAP)
        power = np.abs(Zxx[freq_mask, :]) ** 2
        total = power.sum(axis=0)
        centroid_map[:, itrace] = (valid_freq[:, None] * power).sum(axis=0) / (total + EPS)
        power_map[:, itrace] = total

    return {'centroid_map': centroid_map, 'power_map': power_map,
            't_axis': t_axis, 'f_axis': f_axis, 'n_freq_bins': int(freq_mask.sum()),
            'valid_freq': valid_freq, 'fs': fs}


def data_fingerprint(centroid_map):
    """centroid マップの内容ハッシュ。別パスでも中身が同一なら一致する。"""
    cm = np.ascontiguousarray(centroid_map)
    h = hashlib.blake2b(digest_size=16)
    h.update(np.array(cm.shape, dtype=np.int64).tobytes())
    h.update(cm.view(np.uint8))
    return h.hexdigest()


def load_seed_set(seed_json, indices, label):
    """指定 seed 番号群の B-scan を読み込み、centroid/power マップを作る。

    Bscan.json の `data` は .out の **絶対パス**なので、テンプレの使い回しや
    コピー時の書き換え漏れで複数の seed が同じファイルを指してしまうことがある。
    後段で検出できるよう、解決した .out パスと内容ハッシュを必ず記録する。

    Returns: dict {idx: {maps..., 'dt':, 'gpr_step':, 'n_traces':, 'n_samples':}}
    """
    out = {}
    for idx in indices:
        jpath = seed_json[idx]
        data, dt, gpr_step, params = load_bscan(jpath)
        outfile = params.get('data', '')
        print(f"  [{label}] seed {idx}: {jpath}")
        print(f"           -> data: {outfile}")
        maps = compute_seed_maps(data, dt)
        maps.update({'dt': dt, 'gpr_step': gpr_step,
                     'n_samples': data.shape[0], 'n_traces': data.shape[1],
                     'json_path': jpath, 'outfile': outfile,
                     'fingerprint': data_fingerprint(maps['centroid_map'])})
        out[idx] = maps
    return out


def check_data_distinctness(ice_maps, ice_dirs, noice_maps, noice_dirs,
                            self_diff=False, allow_identical=False):
    """同一データの使い回しを検出する（差分解析の前提が壊れるため）。

    検出する事故:
      1. 複数 seed の Bscan.json が同じ .out を指している
         （テンプレの絶対パス焼き込み・コピー時の書き換え漏れ）
      2. パスは違うが centroid マップが完全一致（同じシミュレーション結果の複製）
      3. ice と No_Ice が同一データ  -> Δ が恒等的に 0 になる
      4. .out が seed ディレクトリの外を指している（別マシン/別 case の残骸の疑い）

    self_diff=True（健全性テストで noice_dir に ice の case_dir を指定した場合）は
    ice と No_Ice が同一なのが正しいので、3 の判定のみ飛ばす。
    """
    problems = []

    def dup_report(maps, dirs, label):
        by_path, by_fp = {}, {}
        for idx in sorted(maps):
            by_path.setdefault(maps[idx]['outfile'], []).append(idx)
            by_fp.setdefault(maps[idx]['fingerprint'], []).append(idx)
        for path, idxs in by_path.items():
            if len(idxs) > 1:
                problems.append(
                    f"[{label}] seed {idxs} の Bscan.json が同じ .out を指しています:\n"
                    f"      {path}")
        for fp, idxs in by_fp.items():
            if len(idxs) > 1 and len({maps[i]['outfile'] for i in idxs}) > 1:
                problems.append(
                    f"[{label}] seed {idxs} は .out パスは異なりますが中身が完全に同一です:\n"
                    + '\n'.join(f"      seed {i}: {maps[i]['outfile']}" for i in idxs))
        # .out が seed ディレクトリ配下に無い場合は警告のみ
        for idx in sorted(maps):
            of = os.path.abspath(maps[idx]['outfile'] or '')
            sd = os.path.abspath(dirs[idx]) + os.sep
            if of and not of.startswith(sd):
                print(f"  [WARN] [{label}] seed {idx} の .out が seed ディレクトリの外です。"
                      f"別 case の結果を読んでいないか確認してください。\n"
                      f"         seed dir: {dirs[idx]}\n"
                      f"         data    : {of}")

    dup_report(ice_maps, ice_dirs, 'ICE')
    dup_report(noice_maps, noice_dirs, 'NOICE')

    if not self_diff:
        ice_fp = {ice_maps[i]['fingerprint']: i for i in ice_maps}
        shared = [(ice_fp[noice_maps[j]['fingerprint']], j)
                  for j in sorted(noice_maps) if noice_maps[j]['fingerprint'] in ice_fp]
        if shared:
            detail = '\n'.join(
                f"      ice seed {i} == noice seed {j}\n"
                f"        ice  : {ice_maps[i]['outfile']}\n"
                f"        noice: {noice_maps[j]['outfile']}" for i, j in shared[:2])
            more = (f"\n      ... 他 {len(shared) - 2} 組" if len(shared) > 2 else '')
            problems.append(
                f"[ICE vs NOICE] ice と No_Ice が同一データです（{len(shared)} 組が一致）。"
                f"差分は恒等的に 0 になり解析が成立しません:\n" + detail + more)

    if problems:
        msg = ("入力データが重複しています（差分解析の前提が壊れます）。\n  "
               + '\n  '.join(problems)
               + "\n\n  確認すべき点:\n"
                 "    - 各 seed の Bscan/Bscan.json の \"data\" が"
                 " その seed 自身の .out を指しているか\n"
                 "    - No_Ice の各 seed が別々のシミュレーション結果か\n"
                 "  意図的に同一データを使う場合は --allow_identical_data を付けてください。")
        if allow_identical:
            print('[WARN] ' + msg)
        else:
            raise ValueError(msg)
    else:
        print('  データ重複チェック OK: ice / No_Ice の全 seed が相異なるデータです。')


def check_consistency(seed_maps, label):
    """全 seed で dt / n_samples / n_traces / gpr_step が一致することを検証。"""
    idxs = sorted(seed_maps)
    ref = seed_maps[idxs[0]]
    for idx in idxs[1:]:
        cur = seed_maps[idx]
        for key, tol in (('dt', 1e-18), ('gpr_step', 1e-12)):
            if abs(cur[key] - ref[key]) > tol:
                raise ValueError(
                    f"[{label}] seed 間で {key} が一致しません: "
                    f"Seed_{idxs[0]}={ref[key]} vs Seed_{idx}={cur[key]}\n"
                    f"  -> 連結すると時間軸/距離軸が破綻するため処理を中止します。")
        for key in ('n_samples', 'n_traces'):
            if cur[key] != ref[key]:
                raise ValueError(
                    f"[{label}] seed 間で {key} が一致しません: "
                    f"Seed_{idxs[0]}={ref[key]} vs Seed_{idx}={cur[key]}\n"
                    f"  -> 連結すると時間軸/距離軸が破綻するため処理を中止します。")
    return ref['dt'], ref['gpr_step'], ref['n_samples'], ref['n_traces']


def check_ice_noice_consistency(ice_ref, noice_ref):
    """ice 側と No_Ice 側の整合性を検証 (差分計算の前提)。"""
    names = ('dt', 'gpr_step', 'n_samples', 'n_traces')
    for key in names:
        a, b = ice_ref[key], noice_ref[key]
        ok = (abs(a - b) <= 1e-18) if key == 'dt' else \
             (abs(a - b) <= 1e-12) if key == 'gpr_step' else (a == b)
        if not ok:
            raise ValueError(
                f"ice 側と No_Ice 側で {key} が一致しません: ice={a} vs noice={b}\n"
                f"  -> 差分は同一の時間軸・トレース数を前提とするため処理を中止します。")


# =============================================================================
# 連結とプロファイル
# =============================================================================
def concat_and_profile(seed_maps, indices):
    """指定 seed をトレース方向に連結し、マスク・平滑・中央値/IQR を計算する。

    連結してから平滑・統計を行う (seed ごとに解析して後で平均しない)。
    """
    centroid_map = np.hstack([seed_maps[i]['centroid_map'] for i in indices])
    power_map = np.hstack([seed_maps[i]['power_map'] for i in indices])
    t_axis = seed_maps[indices[0]]['t_axis']
    dt_stft = t_axis[1] - t_axis[0]

    # --- Power mask: low-SNR pixels -> NaN (トレースごとのピークで規格化) ---
    trace_peak = power_map.max(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        power_rel_db = 10.0 * np.log10(
            np.where(trace_peak > 0, power_map / (trace_peak + EPS), EPS))
    valid_mask = power_rel_db >= POWER_THRESHOLD_DB

    centroid_smooth = smooth_masked(centroid_map, valid_mask, SMOOTH_SIGMA)
    shiftrate_smooth = shift_rate(centroid_smooth, dt_stft)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cen_med = np.nanmedian(centroid_smooth, axis=1)
        cen_p25 = np.nanpercentile(centroid_smooth, 25, axis=1)
        cen_p75 = np.nanpercentile(centroid_smooth, 75, axis=1)
        sr_med = np.nanmedian(shiftrate_smooth, axis=1)
        sr_p25 = np.nanpercentile(shiftrate_smooth, 25, axis=1)
        sr_p75 = np.nanpercentile(shiftrate_smooth, 75, axis=1)

    n_per_seed = seed_maps[indices[0]]['n_traces']
    gpr_step = seed_maps[indices[0]]['gpr_step']
    n_total = centroid_map.shape[1]

    return {
        't': t_axis, 'dt_stft': dt_stft,
        'cen_med': cen_med, 'cen_p25': cen_p25, 'cen_p75': cen_p75,
        'sr_med': sr_med, 'sr_p25': sr_p25, 'sr_p75': sr_p75,
        'n_traces': n_total,
        'x_axis': np.arange(n_total) * gpr_step,
        'seed_bounds_m': (np.cumsum([n_per_seed] * (len(indices) - 1)) * gpr_step
                          if len(indices) > 1 else np.array([])),
        'line_length_m': n_total * gpr_step,
        'seed_indices': list(indices),
    }


# =============================================================================
# 理論プロファイル (氷の有無・濃度は引数で切り替える。実装は 1 箇所のみ)
# =============================================================================
def load_incident_spectrum(ascan_path):
    """入射波 A-scan から帯域内スペクトル S0(omega) を取得。"""
    ascan_data, dt_ascan = get_output_data(ascan_path, 1, 'Ez')
    e_incident = ascan_data if ascan_data.ndim == 1 else ascan_data[:, 0]

    freq_ascan = np.fft.rfftfreq(len(e_incident), d=dt_ascan)
    S0_omega = np.fft.rfft(e_incident)
    band_mask = (freq_ascan >= FREQ_MIN * 1e9) & (freq_ascan <= FREQ_MAX * 1e9)

    f_calc = freq_ascan[band_mask]
    S0_calc = S0_omega[band_mask]
    return {'f_calc': f_calc, 'S0_calc': S0_calc, 'omega': 2 * np.pi * f_calc,
            'dt_ascan': dt_ascan, 'n_samples': len(e_incident), 'path': ascan_path}


def time_offset_ns():
    """赤線の開始時刻 = システムラグ + 空中往復 + 地中 rx_depth 往復 [ns]。"""
    t_air_ns = (2.0 * ANTENNA_HEIGHT / const.c) * 1e9
    d_sub_offset = np.linspace(0, RX_DEPTH, 50)
    eps_sub_offset, _ = get_eps_static(d_sub_offset)
    v_sub = const.c / np.sqrt(eps_sub_offset)
    dt_sub = d_sub_offset[1] - d_sub_offset[0]
    t_ground_start_ns = np.sum(2.0 * dt_sub / v_sub) * 1e9
    return SYSTEM_LAG_NS + t_air_ns + t_ground_start_ns, t_air_ns, t_ground_start_ns


def analytical_profile(t_axis, dt_stft, incident, debye_params,
                       f_ice, ice_top, ice_bot, eps_ice):
    """解析 centroid / shift-rate プロファイルを計算する共通関数。

    f_ice = 0.0 を渡せば「氷なしモデル」、case 実際の値を渡せば「氷ありモデル」。
    物理計算をここ 1 箇所にまとめ、氷の有無・濃度は引数で切り替える。

    Returns: dict('cen', 'sr', 'd_array', 't_delay_d')
    """
    f_calc = incident['f_calc']
    S0_calc = incident['S0_calc']
    omega = incident['omega']

    t_offset_ns, _, _ = time_offset_ns()

    max_depth = (t_axis[-1] * 1e-9) * const.c / 2
    d_array = np.linspace(RX_DEPTH, max_depth, 400)
    d_step = d_array[1] - d_array[0]

    eps_ice_complex = eps_ice * (1.0 - 1j * ICE_LOSS_TAN)

    f_peak_d, t_delay_d = [], []
    cumulative_attenuation = np.zeros_like(omega)
    cumulative_time = np.zeros_like(omega)

    for i, d in enumerate(d_array):
        eps_host = get_eps_regolith(d, omega, debye_params, anchor_freq=ANCHOR_FREQ)

        if f_ice > 0 and ice_top <= d <= ice_bot:
            eps_complex_w = maxwell_garnett(eps_host, eps_ice_complex, f_ice)
        else:
            eps_complex_w = eps_host

        alpha_d = - (omega / const.c) * np.imag(np.sqrt(eps_complex_w))
        v_d = const.c / np.real(np.sqrt(eps_complex_w))

        if i > 0:
            cumulative_attenuation += alpha_d * d_step
            cumulative_time += 2 * d_step / v_d

        power = np.abs(S0_calc * np.exp(-2 * cumulative_attenuation)) ** 2
        f_peak = _TRAPZ(f_calc * power, f_calc) / _TRAPZ(power, f_calc)
        f_peak_d.append(f_peak)

        t_delay_ground = np.interp(f_peak, f_calc, cumulative_time)
        t_delay_d.append(t_offset_ns + (t_delay_ground * 1e9))

    f_peak_d = np.array(f_peak_d) / 1e9      # [GHz]
    t_delay_d = np.array(t_delay_d)

    cen = np.interp(t_axis, t_delay_d, f_peak_d, left=np.nan, right=np.nan)
    sr = np.gradient(cen, dt_stft)
    return {'cen': cen, 'sr': sr, 'd_array': d_array, 't_delay_d': t_delay_d}


def theory_difference(t_axis, dt_stft, incident, debye_params, inp):
    """理論差分署名 d_cen_theory / d_sr_theory と層区間 [ns] を返す。"""
    prof_ice = analytical_profile(t_axis, dt_stft, incident, debye_params,
                                  inp['f_ice'], inp['ice_top'], inp['ice_bot'],
                                  inp['eps_ice'])
    prof_noice = analytical_profile(t_axis, dt_stft, incident, debye_params,
                                    0.0, inp['ice_top'], inp['ice_bot'],
                                    inp['eps_ice'])

    d_cen_theory = prof_ice['cen'] - prof_noice['cen']
    d_sr_theory = np.gradient(d_cen_theory, dt_stft)

    # 層区間: ice_top / ice_bot を往復遅延時間へ変換 (氷ありモデルの速度を使う)
    t_layer_top = float(np.interp(inp['ice_top'], prof_ice['d_array'], prof_ice['t_delay_d']))
    t_layer_bottom = float(np.interp(inp['ice_bot'], prof_ice['d_array'], prof_ice['t_delay_d']))

    return {'d_cen_theory': d_cen_theory, 'd_sr_theory': d_sr_theory,
            't_layer_top': t_layer_top, 't_layer_bottom': t_layer_bottom,
            'cen_ice': prof_ice['cen'], 'cen_noice': prof_noice['cen']}


# =============================================================================
# 領域統計
# =============================================================================
def region_stats(t, values, t0, t1, corr_len_ns):
    """区間 [t0, t1] の (平均, SEM, n_eff, z)。全領域統計で同一シグネチャ。"""
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return np.nan, np.nan, 0.0, np.nan
    mask = (t >= t0) & (t <= t1) & np.isfinite(values)
    if not np.any(mask):
        return np.nan, np.nan, 0.0, np.nan
    v = values[mask]
    mean_v = float(np.mean(v))
    std_v = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    n_eff = max(1.0, (t1 - t0) / corr_len_ns + 1.0)
    sem = std_v / np.sqrt(n_eff)
    z = mean_v / sem if sem > 0 else np.nan
    return mean_v, sem, n_eff, z


def effective_layer_window(t, values, t_layer_top, t_layer_bottom):
    """層区間 ∩ 有効データ区間 を返す。深部が記録終端/ノイズ床に達する case 対策。"""
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan, np.nan, False
    t_valid_max = float(t[finite].max())
    top = t_layer_top
    bot = min(t_layer_bottom, t_valid_max)
    clipped = bot < t_layer_bottom - 1e-9
    if not (bot > top):
        return np.nan, np.nan, clipped
    return top, bot, clipped


def plateau_window(t0, t1, fraction=PLATEAU_FRACTION):
    """層下端側 fraction の区間 (ランプの飽和値を見るため)。"""
    if not (np.isfinite(t0) and np.isfinite(t1) and t1 > t0):
        return np.nan, np.nan
    return t1 - (t1 - t0) * fraction, t1


# =============================================================================
# 作図
# =============================================================================
def _sym_xlim(data, sigma, theory):
    """外れ値で潰れないよう、ロバストに対称な x 範囲を決める。"""
    cands = []
    for arr in (data, theory):
        if arr is not None:
            fin = np.abs(arr[np.isfinite(arr)])
            if fin.size:
                cands.append(float(np.percentile(fin, 99)))
    if sigma is not None:
        fin = np.abs(sigma[np.isfinite(sigma)])
        if fin.size:
            cands.append(float(np.percentile(fin, 95)))
    lim = max(cands) if cands else 1.0
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0
    return -1.3 * lim, 1.3 * lim


def plot_diff_profile(out_path, t, d_cen, sig_cen, d_sr, sig_sr,
                      d_cen_theory, d_sr_theory,
                      t_layer_top, t_layer_bottom, t_surface,
                      stats_pack, title):
    """Δcentroid / Δshift-rate の 2 パネル横並び図。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

    panels = [
        (axes[0], d_cen, sig_cen, d_cen_theory, r'$\Delta$ Centroid [GHz]',
         stats_pack['cen']),
        (axes[1], d_sr, sig_sr, d_sr_theory, r'$\Delta$ Shift rate [GHz/ns]',
         stats_pack['sr']),
    ]

    for ax, data, sig, theory, xlabel, st in panels:
        # 0 線は灰色実線 (基準線)。赤は理論専用に確保する。
        ax.axvline(0, color='gray', linestyle='-', lw=1.2, zorder=1)
        ax.axhspan(t_layer_top, t_layer_bottom, color='blue', alpha=0.10, zorder=0,
                   label=f'Ice layer ({t_layer_top:.1f}-{t_layer_bottom:.1f} ns)')
        ax.fill_betweenx(t, data - sig, data + sig, color='gray', alpha=0.4,
                         label=r'$\pm 1\sigma$ (synth)', zorder=2)
        ax.plot(data, t, color='k', linestyle='-', lw=1.5, label='Measured', zorder=3)

        if theory is not None and np.any(np.isfinite(theory)):
            ax.plot(theory, t, color='r', linestyle='--', lw=2,
                    label='Theory signature', zorder=4)

        # 層内平均・浅部平均をエラーバー付きで重畳
        for key, color, marker, lab in (('layer', 'tab:blue', 'o', 'Layer mean'),
                                        ('shallow', 'tab:green', 's', 'Shallow mean')):
            mean, sem, t0, t1 = st[key]
            if np.isfinite(mean) and np.isfinite(t0) and np.isfinite(t1):
                ax.errorbar(mean, 0.5 * (t0 + t1), xerr=(sem if np.isfinite(sem) else 0.0),
                            fmt=marker, color=color, ecolor=color, capsize=5,
                            markersize=8, lw=2, zorder=5, label=lab)

        ax.axhline(t_surface, color='gray', linestyle='--', lw=2, zorder=1,
                   label='Surface')
        ax.set_ylim(t[-1], t[0])
        ax.set_xlim(*_sym_xlim(data, sig, theory))
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel('Delay time [ns]', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.minorticks_on()
        ax.grid(True)
        ax.legend(fontsize=10, loc='lower left')

    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def plot_sem_vs_ntraces(out_path, records, theory_layer_mean, case_name):
    """層内Δの SEM vs n_traces (本ツールの主要成果物)。"""
    fig, ax = plt.subplots(figsize=(9, 7))
    styles = {'same': ('tab:blue', 'o'), 'cross': ('tab:red', 's')}

    plotted = False
    for mode in ('same', 'cross'):
        rows = [r for r in records if r['mode'] == mode
                and np.isfinite(r['layer_sem']) and r['layer_sem'] > 0]
        if not rows:
            continue
        rows.sort(key=lambda r: r['n_traces'])
        n = np.array([r['n_traces'] for r in rows], dtype=float)
        sem = np.array([r['layer_sem'] for r in rows], dtype=float)
        color, marker = styles[mode]
        ax.plot(n, sem, marker=marker, color=color, lw=1.8, markersize=9,
                label=f'{mode} (measured)')
        plotted = True

        # sigma ∝ 1/sqrt(n) の参照線 (nseed=1 の点を通す)
        n_ref = np.logspace(np.log10(n[0]), np.log10(n[-1] * 4), 100)
        ax.plot(n_ref, sem[0] * np.sqrt(n[0] / n_ref), color=color,
                linestyle=':', lw=1.5, alpha=0.8,
                label=r'{} : $\propto 1/\sqrt{{n}}$'.format(mode))

    # 2σ 検出に必要な SEM 水準
    if theory_layer_mean is not None and np.isfinite(theory_layer_mean) and theory_layer_mean != 0:
        thr = abs(theory_layer_mean) / 2.0
        ax.axhline(thr, color='k', linestyle='--', lw=1.8,
                   label=r'$2\sigma$ detection: SEM = |theory|/2 = ' + f'{thr:.2e}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of traces', fontsize=17)
    ax.set_ylabel(r'SEM of layer-mean $\Delta$centroid [GHz]', fontsize=17)
    ax.set_title(f'{case_name}: SEM scaling vs trace count', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.grid(True, which='both', alpha=0.4)

    # 実測点の n_traces を目盛にして読み取りやすくする
    ticks = sorted({r['n_traces'] for r in records})
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(v) for v in ticks])
        ax.tick_params(axis='x', which='minor', labelbottom=False)

    if plotted:
        ax.legend(fontsize=11, loc='lower left')

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# =============================================================================
# 出力ファイル書き出し
# =============================================================================
def write_diff_profile_csv(path, t, d_cen, sig_cen, d_sr, sig_sr,
                           d_cen_theory, d_sr_theory):
    cols = np.column_stack([t, d_cen, sig_cen, d_sr, sig_sr, d_cen_theory, d_sr_theory])
    header = 't_ns,d_cen,sigma_cen,d_sr,sigma_sr,d_cen_theory,d_sr_theory'
    np.savetxt(path, cols, delimiter=',', header=header, comments='', fmt='%.10g')
    print(f'  Saved: {path}')


def write_stats_csv(path, rows):
    header = ('quantity,region,t0_ns,t1_ns,mean,sem,n_eff,z,theory_mean')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        for r in rows:
            f.write(','.join(
                [r['quantity'], r['region']] +
                ['%.10g' % r[k] for k in
                 ('t0_ns', 't1_ns', 'mean', 'sem', 'n_eff', 'z', 'theory_mean')]) + '\n')
    print(f'  Saved: {path}')


SUMMARY_COLUMNS = [
    'case_name', 'eval_type', 'rand_amp', 'f_ice', 'ice_top_m', 'ice_bot_m',
    'mode', 'nseed', 'line_length_m', 'n_traces',
    'layer_mean', 'layer_sem', 'layer_n_eff', 'layer_z',
    'plateau_mean', 'plateau_sem', 'plateau_z',
    'shallow_mean', 'shallow_sem', 'shallow_z',
    'theory_layer_mean', 't_layer_top_ns', 't_layer_bottom_ns',
]


def write_summary_csv(path, records):
    """1 行 = 1 条件。統計量は Δcentroid のもの (Δshift-rate は stats.csv 参照)。"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(','.join(SUMMARY_COLUMNS) + '\n')
        for r in records:
            vals = []
            for c in SUMMARY_COLUMNS:
                v = r[c]
                vals.append(v if isinstance(v, str) else '%.10g' % v)
            f.write(','.join(vals) + '\n')
    print(f'  Saved: {path}')


def write_run_info(path, info):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f'  Saved: {path}')


def write_readme(path, ctx):
    """README_diff.md を生成・上書きする。"""
    r = ctx
    lines = []
    A = lines.append

    A('# STFT centroid 差分解析 (複数シード) — README')
    A('')
    A(f"- ツール: `{SCRIPT_NAME}`")
    A(f"- 生成日時: {r['timestamp']}")
    A(f"- 対象 case: `{r['case_name']}`  (eval_type=`{r['eval_type']}`, rand_amp=`{r['rand_amp']}`)")
    A('')
    A('## 1. このディレクトリは何か')
    A('水氷あり (ice) と水氷なし (No_Ice) の B-scan を **トレース方向に連結** し、')
    A('STFT で求めた centroid 周波数プロファイルの **差分 Δ(t)** を、')
    A('2 つの pairing モード × nseed = 1..N について評価した結果です。')
    A('目的は「氷層による centroid シフトが、何トレース積めば検出できるか」を定量化することです。')
    A('')
    A('## 2. 入力')
    A(f"- ice case_dir : `{r['case_dir']}`")
    A(f"- noice_dir    : `{r['noice_dir']}`  ({r['noice_dir_source']})")
    A(f"- ice seed     : {r['ice_seeds']}")
    A(f"- noice seed   : {r['noice_seeds_used']}")
    A(f"- A-scan 参照波形: `{r['ascan_path']}`")
    A('')
    A('## 3. pairing モード')
    A('')
    A('| モード | ペアリング規則 | 意味 |')
    A('|---|---|---|')
    A('| `same` | ice seed k ↔ noice seed k | 同一シード。スペックル (ランダム媒質の実現) が'
      '点ごとに相殺するため、**統制実験として最高感度**。 |')
    A(f"| `cross` | ice seed k ↔ noice seed k + `CROSS_SEED_OFFSET`(={CROSS_SEED_OFFSET}) | "
      '異シード。同一実現の双子が存在しない**実際の月面探査に相当**する条件。 |')
    A('')
    A(f"nseed=2 の例: `same` = ice[0,1] − noice[0,1] / "
      f"`cross` = ice[0,1] − noice[{CROSS_SEED_OFFSET},{CROSS_SEED_OFFSET+1}]")
    A('')
    A('### `CROSS_SEED_OFFSET` について')
    A(f"現在の値は **{CROSS_SEED_OFFSET}** です。ice 側 {r['n_seed_max']} seed と重複しない No_Ice ペアを")
    A('作るためのオフセットで、No_Ice のシード数を変更する場合は**必ず見直してください**。')
    A('（例: ice を 6 seed に増やすなら No_Ice は 12 seed 必要で OFFSET=6）')
    A('スクリプト冒頭の `[EDIT HERE]` ブロックに同じ警告コメントがあります。')
    A('')
    A('## 4. 物理パラメータ (`.in` から読み取り)')
    A('')
    A('| 項目 | 値 |')
    A('|---|---|')
    A(f"| `.in` ファイル | `{r['in_file']}` |")
    A(f"| `f_ice` | {r['f_ice']} |")
    A(f"| `ice_top` | {r['ice_top']} m |")
    A(f"| `ice_bot` | {r['ice_bot']} m |")
    A(f"| `eps_ice` | {r['eps_ice']} ({r['eps_ice_source']}) |")
    A(f"| `DEBYE_TAU1` | {r['tau1']} |")
    A(f"| `DEBYE_TAU2` | {r['tau2']} |")
    A(f"| `DE_RATIO` | {r['de_ratio']} |")
    A('')
    A('導出した層区間 (往復遅延時間):')
    A('')
    A(f"- 氷層: **{r['t_layer_top']:.3f} – {r['t_layer_bottom']:.3f} ns**")
    A(f"- 実際に統計に用いた層区間 (層区間 ∩ 有効データ区間): "
      f"**{r['t_layer_top_eff']:.3f} – {r['t_layer_bottom_eff']:.3f} ns**"
      + ('  ← **記録終端/ノイズ床により下端をクリップしています**' if r['layer_clipped'] else ''))
    A(f"- 浅部コントロール区間 (地表反射直後〜層上端): {r['t_surface']:.3f} – {r['t_layer_top']:.3f} ns")
    A(f"- プラトー区間 (層下端側 {int(PLATEAU_FRACTION*100)}%): "
      f"{r['t_plateau_top']:.3f} – {r['t_layer_bottom_eff']:.3f} ns")
    A('')
    A('## 5. 解析条件')
    A('')
    A(f"- STFT: `nperseg={NPERSEG}`, `noverlap={NOVERLAP}`, `window='{WINDOW}'`, "
      f"帯域 {FREQ_MIN}–{FREQ_MAX} GHz ({r['n_freq_bins']} bins)")
    A(f"- パワーマスク閾値: {POWER_THRESHOLD_DB} dB (トレースごとのピーク基準)")
    A(f"- Gaussian 平滑 sigma: {SMOOTH_SIGMA} (時間軸, トレース軸)")
    A(f"- shift rate: `np.gradient(centroid, dt_stft={r['dt_stft']:.4f} ns, axis=0)`")
    A(f"- 1 seed あたり: {r['n_traces_per_seed']} トレース × "
      f"{r['gpr_step']} m = {r['line_length_per_seed']:.2f} m (実測値)")
    A(f"- 相関長 `CORR_LEN_NS` = {CORR_LEN_NS} ns")
    A('- **前処理は行っていません (平均トレース除去なし)。** 連結した生データを')
    A('  そのまま STFT にかけており、`k_centroid_freq.py` と完全に同一のアルゴリズムです。')
    A('- 連結してから STFT・統計を行っています (seed ごとに解析して後で平均していません)。')
    A('')
    A('## 6. 出力ファイル')
    A('')
    A('```')
    A('multi_seed_analysis/STFT_analysis/')
    A('├── diff_same/   nseed_NN_XX.Xm/  ... 同一シード差分')
    A('├── diff_cross/  nseed_NN_XX.Xm/  ... 異シード差分')
    A('├── summary/')
    A('│   ├── summary_diff.csv')
    A('│   ├── sem_vs_ntraces.png')
    A('│   └── run_info_diff.json')
    A('└── README_diff.md   (このファイル)')
    A('```')
    A('')
    A('### 各 `nseed_NN_XX.Xm/` の中身')
    A('')
    A('| ファイル | 内容 |')
    A('|---|---|')
    A('| `centroid_diff_profile.png` | 2 パネル横並び。左 Δcentroid [GHz]、右 Δshift rate [GHz/ns] |')
    A('| `diff_profile.csv` | 差分プロファイルの全点 |')
    A('| `stats.csv` | 層内平均・プラトー値・浅部平均の統計 |')
    A('')
    A('`diff_profile.csv` の列:')
    A('')
    A('| 列 | 意味 |')
    A('|---|---|')
    A('| `t_ns` | 往復遅延時間 [ns] (STFT 時間軸) |')
    A('| `d_cen` | Δcentroid = centroid(ice) − centroid(noice) [GHz]。中央値プロファイルの差 |')
    A('| `sigma_cen` | 合成σ = √(σ_ice² + σ_noice²) / √n_traces [GHz]、σ = IQR/1.349 |')
    A('| `d_sr` | Δshift rate = sr(ice) − sr(noice) [GHz/ns] |')
    A('| `sigma_sr` | Δshift rate の合成σ [GHz/ns] |')
    A('| `d_cen_theory` | 理論差分署名 = (氷ありモデル centroid) − (氷なしモデル centroid) [GHz] |')
    A('| `d_sr_theory` | `np.gradient(d_cen_theory, dt_stft)` [GHz/ns] |')
    A('')
    A('`stats.csv` の列: `quantity` (d_cen / d_sr)、`region` (layer / plateau / shallow)、')
    A('`t0_ns`,`t1_ns` (実際に使った区間)、`mean`,`sem`,`n_eff`,`z`、`theory_mean` (同区間の理論値平均)。')
    A('')
    A('`summary_diff.csv` は 1 行 = 1 条件 (mode × nseed)。統計量の列 (`layer_*`, `plateau_*`,')
    A('`shallow_*`, `theory_layer_mean`) は **Δcentroid** のものです (Δshift rate は各 `stats.csv` 参照)。')
    A('case 識別列を先頭に置いてあるので、複数 case の `summary_diff.csv` を単純連結すれば')
    A('厚さ依存性・濃度依存性の集約が可能です。')
    A('')
    A('## 7. 統計量の定義')
    A('')
    A('- σ (1 トレース分) = `(p75 − p25) / 1.349` … IQR から正規分布相当の標準偏差へ換算')
    A('- 中央値プロファイルの標準誤差 = σ / √n_traces')
    A('- 差分の合成σ = `√(σ_ice² + σ_noice²) / √n_traces`')
    A(f"- 領域統計の実効独立標本数 `n_eff = max(1, 区間長 / {CORR_LEN_NS} + 1)`")
    A('  (centroid は Gaussian 平滑により時間方向に相関を持つため、点数をそのまま使えない)')
    A('- 領域 SEM = 区間内の点間標準偏差 (ddof=1) / √n_eff、`z = 平均 / SEM`')
    A('- **層内統計は 2 通りを併記**しています:')
    A('  - **層内平均**: 層区間全体の平均。保守的で、他手法 (LSR 等) と比較可能。')
    A(f"  - **プラトー値**: 層下端側 {int(PLATEAU_FRACTION*100)}% 区間の平均。"
      'ランプの飽和値＝検出力の実力値。')
    A('')
    A('## 8. 解釈上の注意 (重要)')
    A('')
    A('1. **連結マップは異なるシードの並置**です。水平方向に地下構造が不連続であり、')
    A('   物理的に連続した 1 本の測線ではありません (統計量を稼ぐための連結です)。')
    A('2. **Δcentroid は層内でランプ状**になります (氷層を通過するほど累積効果が効くため)。')
    A('   したがって層内平均の SEM は区間内の傾きを分散として拾い**過大評価**、')
    A('   z は**過小評価**になります。プラトー値も必ず併せて見てください。')
    A('3. **`cross` モードでは点ごとの Δ(t) カーブに意味がありません。**')
    A('   独立な実現どうしの差なので、スペックルが点ごとに相殺しません。')
    A('   `cross` は必ず**領域統計 (層内平均・プラトー値・z)** で判断してください。')
    A('   点ごとのカーブが暴れていても、それはノイズであって物理ではありません。')
    A('4. 深部のマスク領域では Δ が **NaN のまま**残ります (0 埋めしていません)。')
    A('   統計は有限値のみで計算しています。')
    A('5. `same` と `cross` の感度差は、理論的にはノイズが √2 倍 (相殺が効かないため)')
    A('   になることが期待されます。`sem_vs_ntraces.png` で確認できます。')
    A('')
    A('## 9. `sem_vs_ntraces.png` の読み方')
    A('')
    A('- 横軸 = 連結後の総トレース数 (対数)、縦軸 = 層内 Δcentroid の SEM (対数)。')
    A('- 点線は **σ ∝ 1/√n の参照線** (各系列の nseed=1 の点を通す)。')
    A('  測定点がこの線に乗っていれば、ノイズが独立でトレース数を増やした分だけ')
    A('  素直に精度が上がっている (√N 則が成立している) ことを意味します。')
    A('- 黒破線は **2σ 検出に必要な SEM 水準** = |理論層内Δ| / 2。')
    A('  参照線とこの水平線の交点の横軸が「2σ 検出に必要なトレース数」の外挿値です。')
    A('- `same` と `cross` の縦方向の隔たりが、同一実現ペアが使えないことによる感度の損失です。')
    A('')
    A('## 10. 再現方法')
    A('')
    A('```bash')
    A(f"python {SCRIPT_NAME} {r['case_dir']}" +
      (f" --noice_dir {r['noice_dir']}" if r['noice_dir_source'] == '--noice_dir で明示指定' else '') +
      (f" --modes {','.join(r['modes'])}" if len(r['modes']) < 2 else ''))
    A('```')
    A('')
    A('健全性テスト (自己差分が 0 になることの確認):')
    A('')
    A('```bash')
    A(f"python {SCRIPT_NAME} {r['case_dir']} --noice_dir {r['case_dir']} --modes same")
    A('```')
    A('')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Saved: {path}')

# =============================================================================
# case 単位の実行
# =============================================================================
def run_case(case_dir, args):
    case_dir = os.path.abspath(case_dir)
    case_name = os.path.basename(case_dir)
    print('\n' + '=' * 78)
    print(f'CASE: {case_name}   ({case_dir})')
    print('=' * 78)

    # --- seed の列挙と検証 ---------------------------------------------------
    ice_seeds, ice_dirs, ice_nojson = list_seeds(case_dir)
    ice_idxs = validate_ice_seeds(case_dir, ice_seeds, ice_dirs, ice_nojson)
    n_seed_max = len(ice_idxs)
    print(f'[SEED] ice: {[os.path.basename(ice_dirs[i]) for i in ice_idxs]}  (N={n_seed_max})')

    if args.noice_dir:
        noice_dir = os.path.abspath(os.path.expanduser(args.noice_dir))
        noice_src = '--noice_dir で明示指定'
    else:
        noice_dir = derive_noice_dir(case_dir)
        noice_src = 'パス規約から自動導出'
    if not os.path.isdir(noice_dir):
        raise FileNotFoundError(f"No_Ice ディレクトリが存在しません: {noice_dir}")
    print(f'[SEED] No_Ice dir: {noice_dir}  ({noice_src})')

    noice_seeds, noice_dirs, noice_nojson = list_seeds(noice_dir)
    validate_noice_seeds(noice_dir, noice_seeds, noice_dirs, noice_nojson,
                         args.modes, n_seed_max)
    needed_noice = sorted({i for m in args.modes for i in required_noice_seeds(m, n_seed_max)})
    print(f'[SEED] No_Ice で使用: {[os.path.basename(noice_dirs[i]) for i in needed_noice]}')

    # --- .in パラメータ (ice 側 seed 0 の実在パスを渡す) ---------------------
    inp = read_in_params(ice_dirs[ice_idxs[0]], ice_seeds[ice_idxs[0]])
    rand_amp = extract_rand_amp(case_dir) or ''
    eval_type = extract_eval_type(case_dir)

    # --- B-scan 読み込みと seed ごとのマップ ---------------------------------
    print('[LOAD] ice B-scan ...')
    ice_maps = load_seed_set(ice_seeds, ice_idxs, 'ICE')
    print('[LOAD] No_Ice B-scan ...')
    noice_maps = load_seed_set(noice_seeds, needed_noice, 'NOICE')

    dt, gpr_step, n_samples, n_traces_per_seed = check_consistency(ice_maps, 'ICE')
    check_consistency(noice_maps, 'NOICE')
    check_ice_noice_consistency(ice_maps[ice_idxs[0]], noice_maps[needed_noice[0]])
    check_data_distinctness(ice_maps, ice_dirs, noice_maps, noice_dirs,
                            self_diff=(os.path.abspath(noice_dir) == case_dir),
                            allow_identical=getattr(args, 'allow_identical_data', False))
    print(f'[LOAD] dt = {dt*1e12:.4f} ps, n_samples = {n_samples}, '
          f'n_traces/seed = {n_traces_per_seed}, gpr_step = {gpr_step} m '
          f'-> 1 seed = {n_traces_per_seed * gpr_step:.2f} m')

    t_axis = ice_maps[ice_idxs[0]]['t_axis']
    if not (t_axis.size == noice_maps[needed_noice[0]]['t_axis'].size and
            np.allclose(t_axis, noice_maps[needed_noice[0]]['t_axis'], atol=1e-6)):
        raise ValueError('ice と No_Ice で STFT 時間軸が一致しません。処理を中止します。')
    dt_stft = t_axis[1] - t_axis[0]
    t_surface = surface_delay_ns()
    _fs = 1.0 / (dt * 1e9)
    print(f'[STFT] nperseg={NPERSEG}, noverlap={NOVERLAP}, df={_fs/NPERSEG:.3f} GHz, '
          f'band {FREQ_MIN}-{FREQ_MAX} GHz '
          f'({ice_maps[ice_idxs[0]]["n_freq_bins"]} bins), dt_stft={dt_stft:.4f} ns')

    # --- 理論差分署名 -------------------------------------------------------
    if not os.path.exists(ASCAN_OUTFILE_PATH):
        raise FileNotFoundError(
            f"入射波 A-scan が見つかりません: {ASCAN_OUTFILE_PATH}\n"
            f"  -> 理論署名も層区間も定義できないため処理を中止します。"
            f" スクリプト冒頭 [EDIT HERE] の ASCAN_OUTFILE_PATH を確認してください。")
    incident = load_incident_spectrum(ASCAN_OUTFILE_PATH)
    debye = {'tau1': inp['tau1'], 'tau2': inp['tau2'], 'de_ratio': inp['de_ratio']}
    th = theory_difference(t_axis, dt_stft, incident, debye, inp)
    t_layer_top, t_layer_bottom = th['t_layer_top'], th['t_layer_bottom']
    print(f'[THEORY] ice layer: {inp["ice_top"]} - {inp["ice_bot"]} m  ->  '
          f'{t_layer_top:.3f} - {t_layer_bottom:.3f} ns')

    # --- 出力ディレクトリ ---------------------------------------------------
    stft_root = os.path.join(case_dir, 'multi_seed_analysis', 'STFT_analysis')
    summary_dir = os.path.join(stft_root, 'summary')
    os.makedirs(summary_dir, exist_ok=True)

    self_diff = (os.path.abspath(noice_dir) == case_dir)
    records = []
    stats_ctx = None

    for mode in args.modes:
        for nseed in range(1, n_seed_max + 1):
            ice_idx, noice_idx = pair_seed_indices(mode, nseed)
            pair_str = (f'ice{ice_idx} - noice{noice_idx}').replace(' ', '')
            print(f'\n--- mode={mode}, nseed={nseed}  [{pair_str}] ---')

            p_ice = concat_and_profile(ice_maps, ice_idx)
            p_no = concat_and_profile(noice_maps, noice_idx)

            n_tr = p_ice['n_traces']
            if n_tr != p_no['n_traces']:
                raise ValueError(
                    f'ice と No_Ice で連結後の n_traces が一致しません '
                    f'({n_tr} vs {p_no["n_traces"]})。処理を中止します。')
            line_len = p_ice['line_length_m']

            # --- 差分 (両方が有限な点のみ。NaN は 0 埋めせず残す) ---
            both = np.isfinite(p_ice['cen_med']) & np.isfinite(p_no['cen_med'])
            d_cen = np.where(both, p_ice['cen_med'] - p_no['cen_med'], np.nan)
            both_sr = np.isfinite(p_ice['sr_med']) & np.isfinite(p_no['sr_med'])
            d_sr = np.where(both_sr, p_ice['sr_med'] - p_no['sr_med'], np.nan)

            # --- 合成σ ---
            sig_cen_i = (p_ice['cen_p75'] - p_ice['cen_p25']) / 1.349
            sig_cen_n = (p_no['cen_p75'] - p_no['cen_p25']) / 1.349
            sig_cen = np.sqrt(sig_cen_i ** 2 + sig_cen_n ** 2) / np.sqrt(n_tr)
            sig_sr_i = (p_ice['sr_p75'] - p_ice['sr_p25']) / 1.349
            sig_sr_n = (p_no['sr_p75'] - p_no['sr_p25']) / 1.349
            sig_sr = np.sqrt(sig_sr_i ** 2 + sig_sr_n ** 2) / np.sqrt(n_tr)

            # --- 統計区間 (層区間 ∩ 有効データ区間) ---
            lay_t0, lay_t1, clipped = effective_layer_window(
                t_axis, d_cen, t_layer_top, t_layer_bottom)
            pla_t0, pla_t1 = plateau_window(lay_t0, lay_t1)
            sha_t0, sha_t1 = t_surface, t_layer_top
            if clipped:
                print(f'  [WARN] 層区間の下端を有効データ終端でクリップ: '
                      f'{t_layer_bottom:.3f} -> {lay_t1:.3f} ns')
            print(f'  統計区間: layer {lay_t0:.3f}-{lay_t1:.3f} ns / '
                  f'plateau {pla_t0:.3f}-{pla_t1:.3f} ns / '
                  f'shallow {sha_t0:.3f}-{sha_t1:.3f} ns')

            regions = {'layer': (lay_t0, lay_t1),
                       'plateau': (pla_t0, pla_t1),
                       'shallow': (sha_t0, sha_t1)}

            stats_rows = []
            st = {'cen': {}, 'sr': {}}
            for qname, vals, theo in (('d_cen', d_cen, th['d_cen_theory']),
                                      ('d_sr', d_sr, th['d_sr_theory'])):
                key = 'cen' if qname == 'd_cen' else 'sr'
                for rname, (t0, t1) in regions.items():
                    mean, sem, n_eff, z = region_stats(t_axis, vals, t0, t1, CORR_LEN_NS)
                    th_mean = region_stats(t_axis, theo, t0, t1, CORR_LEN_NS)[0]
                    stats_rows.append({'quantity': qname, 'region': rname,
                                       't0_ns': t0, 't1_ns': t1, 'mean': mean,
                                       'sem': sem, 'n_eff': n_eff, 'z': z,
                                       'theory_mean': th_mean})
                    st[key][rname] = (mean, sem, t0, t1)
                    st[key][rname + '_full'] = (mean, sem, n_eff, z, th_mean)

            lay = st['cen']['layer_full']
            pla = st['cen']['plateau_full']
            sha = st['cen']['shallow_full']
            print(f'  Δcentroid layer  : mean={lay[0]:+.6f} GHz, SEM={lay[1]:.6f}, '
                  f'n_eff={lay[2]:.1f}, z={lay[3]:.2f}  (theory={lay[4]:+.6f})')
            print(f'  Δcentroid plateau: mean={pla[0]:+.6f} GHz, SEM={pla[1]:.6f}, z={pla[3]:.2f}')
            print(f'  Δcentroid shallow: mean={sha[0]:+.6f} GHz, SEM={sha[1]:.6f}, z={sha[3]:.2f}')
            n_nan = int(np.sum(~np.isfinite(d_cen)))
            fin = np.isfinite(d_cen)
            max_abs = float(np.max(np.abs(d_cen[fin]))) if np.any(fin) else np.nan
            n_zero = int(np.sum(d_cen[fin] == 0.0)) if np.any(fin) else 0
            print(f'  NaN (masked) points in d_cen: {n_nan} / {d_cen.size}')
            print(f'  |Δcentroid|: max={max_abs:.3e} GHz, '
                  f'厳密に 0 の点={n_zero}/{int(fin.sum())}')
            if np.any(fin) and max_abs == 0.0 and not self_diff:
                print('  [WARN] Δ が全点で厳密に 0 です。ice と No_Ice が'
                      '同一データの可能性が高いので入力を確認してください。')

            # --- 出力 ---
            out_dir = os.path.join(stft_root, f'diff_{mode}',
                                   f'nseed_{nseed:02d}_{line_len:.1f}m')
            os.makedirs(out_dir, exist_ok=True)

            write_diff_profile_csv(os.path.join(out_dir, 'diff_profile.csv'),
                                   t_axis, d_cen, sig_cen, d_sr, sig_sr,
                                   th['d_cen_theory'], th['d_sr_theory'])
            write_stats_csv(os.path.join(out_dir, 'stats.csv'), stats_rows)

            title = (f'{case_name} | mode={mode} | nseed={nseed} | '
                     f'{line_len:.1f} m | n_traces={n_tr} | {pair_str}')
            plot_diff_profile(os.path.join(out_dir, 'centroid_diff_profile.png'),
                              t_axis, d_cen, sig_cen, d_sr, sig_sr,
                              th['d_cen_theory'], th['d_sr_theory'],
                              t_layer_top, t_layer_bottom, t_surface,
                              st, title)

            records.append({
                'case_name': case_name, 'eval_type': eval_type, 'rand_amp': rand_amp,
                'f_ice': inp['f_ice'], 'ice_top_m': inp['ice_top'], 'ice_bot_m': inp['ice_bot'],
                'mode': mode, 'nseed': nseed, 'line_length_m': line_len, 'n_traces': n_tr,
                'layer_mean': lay[0], 'layer_sem': lay[1], 'layer_n_eff': lay[2], 'layer_z': lay[3],
                'plateau_mean': pla[0], 'plateau_sem': pla[1], 'plateau_z': pla[3],
                'shallow_mean': sha[0], 'shallow_sem': sha[1], 'shallow_z': sha[3],
                'theory_layer_mean': lay[4],
                't_layer_top_ns': t_layer_top, 't_layer_bottom_ns': t_layer_bottom,
                'seed_pair': pair_str, 'out_dir': out_dir,
            })

            if stats_ctx is None:
                stats_ctx = {'lay_t0': lay_t0, 'lay_t1': lay_t1, 'pla_t0': pla_t0,
                             'clipped': clipped,
                             'n_freq_bins': ice_maps[ice_idxs[0]]['n_freq_bins']}

    # --- 集約出力 -----------------------------------------------------------
    print('\n--- 集約出力 ---')
    write_summary_csv(os.path.join(summary_dir, 'summary_diff.csv'), records)

    theory_layer_mean = records[0]['theory_layer_mean'] if records else np.nan
    plot_sem_vs_ntraces(os.path.join(summary_dir, 'sem_vs_ntraces.png'),
                        records, theory_layer_mean, case_name)

    timestamp = datetime.datetime.now().isoformat(timespec='seconds')
    run_info = {
        'timestamp': timestamp,
        'tool': SCRIPT_NAME,
        'case_dir': case_dir, 'case_name': case_name,
        'eval_type': eval_type, 'rand_amp': rand_amp,
        'noice_dir': noice_dir, 'noice_dir_source': noice_src,
        'ice_seeds_used': [os.path.basename(ice_dirs[i]) for i in ice_idxs],
        'noice_seeds_used': [os.path.basename(noice_dirs[i]) for i in needed_noice],
        'modes': args.modes,
        'in_file': inp['in_file'],
        'in_params': {k: inp[k] for k in
                      ('f_ice', 'ice_top', 'ice_bot', 'eps_ice', 'eps_ice_source',
                       'tau1', 'tau2', 'de_ratio')},
        'ASCAN_OUTFILE_PATH': ASCAN_OUTFILE_PATH,
        'stft': {'nperseg': NPERSEG, 'noverlap': NOVERLAP, 'window': WINDOW,
                 'freq_min_GHz': FREQ_MIN, 'freq_max_GHz': FREQ_MAX,
                 'power_threshold_db': POWER_THRESHOLD_DB,
                 'smooth_sigma': list(SMOOTH_SIGMA),
                 'dt_stft_ns': float(dt_stft), 'n_freq_bins': stats_ctx['n_freq_bins']},
        'CORR_LEN_NS': CORR_LEN_NS,
        'CROSS_SEED_OFFSET': CROSS_SEED_OFFSET,
        'PLATEAU_FRACTION': PLATEAU_FRACTION,
        'dt_s': float(dt), 'n_samples': int(n_samples),
        'gpr_step_m': float(gpr_step),
        'n_traces_per_seed': int(n_traces_per_seed),
        'line_length_per_seed_m': float(n_traces_per_seed * gpr_step),
        'layer_window_ns': {'t_layer_top': t_layer_top,
                            't_layer_bottom': t_layer_bottom,
                            't_layer_bottom_used': stats_ctx['lay_t1'],
                            'clipped_by_valid_data': bool(stats_ctx['clipped']),
                            't_surface': t_surface},
        'conditions': [{k: r[k] for k in
                        ('mode', 'nseed', 'n_traces', 'line_length_m', 'seed_pair')}
                       for r in records],
    }
    write_run_info(os.path.join(summary_dir, 'run_info_diff.json'), run_info)

    ctx = {
        'timestamp': timestamp, 'case_name': case_name, 'case_dir': case_dir,
        'eval_type': eval_type, 'rand_amp': rand_amp,
        'noice_dir': noice_dir, 'noice_dir_source': noice_src,
        'ice_seeds': [os.path.basename(ice_dirs[i]) for i in ice_idxs],
        'noice_seeds_used': [os.path.basename(noice_dirs[i]) for i in needed_noice],
        'ascan_path': ASCAN_OUTFILE_PATH, 'n_seed_max': n_seed_max,
        'in_file': inp['in_file'], 'f_ice': inp['f_ice'],
        'ice_top': inp['ice_top'], 'ice_bot': inp['ice_bot'],
        'eps_ice': inp['eps_ice'], 'eps_ice_source': inp['eps_ice_source'],
        'tau1': inp['tau1'], 'tau2': inp['tau2'], 'de_ratio': inp['de_ratio'],
        't_layer_top': t_layer_top, 't_layer_bottom': t_layer_bottom,
        't_layer_top_eff': stats_ctx['lay_t0'], 't_layer_bottom_eff': stats_ctx['lay_t1'],
        't_plateau_top': stats_ctx['pla_t0'], 'layer_clipped': stats_ctx['clipped'],
        't_surface': t_surface, 'n_freq_bins': stats_ctx['n_freq_bins'],
        'dt_stft': dt_stft, 'n_traces_per_seed': n_traces_per_seed,
        'gpr_step': gpr_step, 'line_length_per_seed': n_traces_per_seed * gpr_step,
        'modes': args.modes,
    }
    write_readme(os.path.join(stft_root, 'README_diff.md'), ctx)

    # --- 健全性チェック (自己差分) -----------------------------------------
    if os.path.abspath(noice_dir) == case_dir:
        print('\n[SANITY CHECK] noice_dir == case_dir のため自己差分テストとして判定します。')
        ok_all = True
        for r in records:
            csv_path = os.path.join(r['out_dir'], 'diff_profile.csv')
            arr = np.genfromtxt(csv_path, delimiter=',', names=True)
            dc, ds = arr['d_cen'], arr['d_sr']
            ok = (np.nanmax(np.abs(dc)) if np.any(np.isfinite(dc)) else 0.0) < 1e-12 and \
                 (np.nanmax(np.abs(ds)) if np.any(np.isfinite(ds)) else 0.0) < 1e-12
            ok_all &= ok
            print(f'  mode={r["mode"]}, nseed={r["nseed"]}: '
                  f'max|d_cen|={np.nanmax(np.abs(dc)):.3e}, '
                  f'max|d_sr|={np.nanmax(np.abs(ds)):.3e}  -> {"OK" if ok else "NG"}')
        print(f'  SANITY CHECK RESULT: {"ALL ZERO (OK)" if ok_all else "NON-ZERO (NG)"}')

    # --- √N 則の数値報告 ----------------------------------------------------
    print('\n[SEM scaling] 層内Δcentroid の SEM')
    for mode in args.modes:
        rows = sorted([r for r in records if r['mode'] == mode], key=lambda r: r['nseed'])
        if not rows or not np.isfinite(rows[0]['layer_sem']) or rows[0]['layer_sem'] <= 0:
            continue
        base = rows[0]['layer_sem']
        print(f'  mode={mode}:')
        for r in rows:
            ratio = r['layer_sem'] / base if base > 0 else np.nan
            ideal = 1.0 / np.sqrt(r['nseed'])
            print(f'    nseed={r["nseed"]} (n_traces={r["n_traces"]:5d}): '
                  f'SEM={r["layer_sem"]:.6e}  SEM/SEM_1={ratio:.4f}  '
                  f'(ideal 1/sqrt(nseed)={ideal:.4f})')

    print(f'\n[OUTPUT] {stft_root}')
    return records, stft_root


# =============================================================================
# 関数インベントリの自己監査
# =============================================================================
EXPECTED_FUNCTIONS = [
    'resolve_ci', 'find_seed_dirs', 'is_case_dir',
    'discover_cases', 'list_seeds', 'validate_ice_seeds', 'validate_noice_seeds',
    'extract_rand_amp', 'extract_eval_type', 'derive_noice_dir',
    'required_noice_seeds', 'pair_seed_indices',
    'read_in_params', 'load_bscan', 'stft_axes', 'compute_seed_maps',
    'load_seed_set', 'data_fingerprint', 'check_data_distinctness',
    'check_consistency', 'check_ice_noice_consistency',
    'concat_and_profile',
    'get_eps_static', 'get_eps_regolith', 'maxwell_garnett', 'surface_delay_ns',
    'smooth_masked', 'shift_rate',
    'load_incident_spectrum', 'time_offset_ns', 'analytical_profile', 'theory_difference',
    'region_stats', 'effective_layer_window', 'plateau_window',
    '_sym_xlim', 'plot_diff_profile', 'plot_sem_vs_ntraces',
    'write_diff_profile_csv', 'write_stats_csv', 'write_summary_csv',
    'write_run_info', 'write_readme',
    'run_case', 'self_audit', 'main',
]

EXPECTED_OUTPUTS = [
    ('diff_<mode>/nseed_NN_XX.Xm/centroid_diff_profile.png', 'per condition'),
    ('diff_<mode>/nseed_NN_XX.Xm/diff_profile.csv', 'per condition'),
    ('diff_<mode>/nseed_NN_XX.Xm/stats.csv', 'per condition'),
    ('summary/summary_diff.csv', 'per case'),
    ('summary/sem_vs_ntraces.png', 'per case'),
    ('summary/run_info_diff.json', 'per case'),
    ('README_diff.md', 'per case'),
]


def self_audit(stft_roots):
    """全関数・全出力ファイルの存在を一覧で報告する。"""
    print('\n' + '=' * 78)
    print('SELF AUDIT: function inventory')
    print('=' * 78)
    g = globals()
    missing = []
    for name in EXPECTED_FUNCTIONS:
        ok = callable(g.get(name))
        if not ok:
            missing.append(name)
        print(f'  [{"OK" if ok else "MISSING"}] {name}')
    print(f'  -> {len(EXPECTED_FUNCTIONS) - len(missing)}/{len(EXPECTED_FUNCTIONS)} functions present'
          + (f', MISSING: {missing}' if missing else ''))

    print('\n' + '=' * 78)
    print('SELF AUDIT: output files')
    print('=' * 78)
    all_ok = True
    for root in stft_roots:
        print(f'  root: {root}')
        for pat, scope in EXPECTED_OUTPUTS:
            if scope == 'per case':
                paths = [os.path.join(root, pat)]
            else:
                paths = sorted(glob.glob(os.path.join(root, pat.replace('<mode>', '*')
                                                      .replace('NN', '*')
                                                      .replace('XX.X', '*'))))
            if not paths:
                print(f'    [MISSING] {pat}')
                all_ok = False
                continue
            n_ok = sum(1 for p in paths if os.path.exists(p) and os.path.getsize(p) > 0)
            flag = 'OK' if n_ok == len(paths) else 'MISSING'
            all_ok &= (n_ok == len(paths))
            print(f'    [{flag}] {pat}  ({n_ok}/{len(paths)} files)')
    print(f'  -> output audit: {"ALL PRESENT" if all_ok else "INCOMPLETE"}')
    return (not missing) and all_ok


# =============================================================================
# main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='複数シード STFT centroid 差分解析 (水氷あり - 水氷なし)')
    parser.add_argument('path', nargs='?', default=None,
                        help='case ディレクトリ、またはその親ディレクトリ')
    parser.add_argument('--noice_dir', default=None,
                        help='No_Ice の case_dir を明示指定 (未指定ならパス規約から自動導出)')
    parser.add_argument('--allow_identical_data', action='store_true',
                        help='ice と No_Ice が同一データでもエラーで止めない'
                             '（重複チェックを警告に格下げする）')
    parser.add_argument('--modes', default='same,cross',
                        help="解析する pairing モード (既定 'same,cross')。"
                             "健全性テストでは 'same' を指定する")
    args = parser.parse_args()

    if not args.path:
        args.path = input('Input case directory (or its parent) path: ').strip()

    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    for m in modes:
        if m not in ('same', 'cross'):
            raise ValueError(f"不正な mode: '{m}' (same / cross のみ)")
    args.modes = modes
    print(f'[CONFIG] modes = {modes}, CROSS_SEED_OFFSET = {CROSS_SEED_OFFSET}, '
          f'CORR_LEN_NS = {CORR_LEN_NS}')

    cases = discover_cases(args.path)
    if len(cases) > 1 and args.noice_dir:
        print('[WARN] 複数 case に対して同一の --noice_dir を適用します。')

    roots = []
    for case_dir in cases:
        _, root = run_case(case_dir, args)
        roots.append(root)

    self_audit(roots)

    print('\n' + '=' * 78)
    print('OUTPUT DIRECTORIES')
    for root in roots:
        print(f'  {root}')
    print('=' * 78)


if __name__ == '__main__':
    main()