#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
k_ms_centroid_single.py
================================================================================
複数シード STFT centroid 単体解析ツール（水氷ありのみ）

複数の Seed_* の B-scan をトレース方向に連結し、nseed = 1..N のそれぞれについて
STFT centroid / shift-rate のマップ・プロファイル・統計を出力する。

アルゴリズム（STFT・マスク・平滑・shift rate・解析プロファイル）は
`k_centroid_freq.py` からの逐語移植であり、パラメータは一切変更していない。
前処理（平均トレース除去など）は追加していない。したがって nseed=1 の結果は
同じ Seed_0 に対する `k_centroid_freq.py` の出力と数値誤差内で一致する。

仕様: prompt_ms_0_common.md（共通仕様） + prompt_ms_1_single.md（本ツール固有）

使い方:
    python k_ms_centroid_single.py <path>
    python k_ms_centroid_single.py <path> --noice_dir <No_Ice case dir>
    python k_ms_centroid_single.py            # 引数なしなら input() で受け取る

    <path> は次のいずれか:
      (a) case ディレクトリ  .../Eval_thick/rand_amp_005/thick_10
      (c) 親ディレクトリ      .../Eval_thick/rand_amp_005   （配下 case を一括処理）
================================================================================
"""

import os
import sys
import re
import glob
import json
import argparse
import datetime
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axgrid1
from scipy import signal
from scipy import constants as const
from scipy.ndimage import gaussian_filter

# gprMax の tools を探索パスに追加（既存スクリプトと同一の方法）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.core.outputfiles_merge import get_output_data

# NumPy 2.0 で np.trapz が np.trapezoid に改名されたことへの互換対応（数値は同一）
_TRAPZ = getattr(np, 'trapz', None) or np.trapezoid


# =============================================================================
# [EDIT HERE] 実行環境に応じて変更する定数
# =============================================================================
# 入射波スペクトル計算用の A-scan 出力ファイル (全 case 共通の参照波形)。
# 注意: 各 case の Ascan.out は氷を含む実媒質を伝搬済みのため入射スペクトルには
#       使えない。必ず waveform_test の専用シミュレーション結果を指定する。
ASCAN_OUTFILE_PATH = "/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_3x4/waveform_test/gaussiandot_1.25GHz_underground/result/Ascan.out"

# 理論プロファイルを描く氷体積分率のスイープ (単体ツールで使用)。
# 0.0 は氷なし。case 実際の f_ice は .in から読み取り、別途強調表示する。
THEORY_FICE_SWEEP = [0.0, 0.01, 0.05, 0.10, 0.20]

# 異シード差分のシード番号オフセット (ice seed k <-> noice seed k + OFFSET)。
# !!! 将来 No_Ice のシード数を変更する場合はこの値を必ず見直すこと !!!
#     現在は No_Ice が Seed_0..7 の 8 個あり、ice 側 4 個 (Seed_0..3) と
#     重複しないペアを作るために 4 としている。
#     例) ice を 6 seed に増やすなら No_Ice は 12 seed 必要で OFFSET=6 になる。
#     ※ 本ツール（単体解析）では未使用。差分ツール②と定義を共有するため記載し、
#        run_info_single.json にも記録する。
CROSS_SEED_OFFSET = 4

# 領域統計の系列相関長 [ns] (centroid は Gaussian 平滑により時間方向に相関を持つ)
CORR_LEN_NS = 3.0


# =============================================================================
# 固定パラメータ（k_centroid_freq.py と同一。変更禁止）
# =============================================================================
NPERSEG            = 256
NOVERLAP           = NPERSEG * 3 // 4      # = 192
WINDOW             = 'hann'
FREQ_MIN           = 0.25                  # [GHz]
FREQ_MAX           = 6.0                   # [GHz]
POWER_THRESHOLD_DB = -125.0                # [dB]
SIGMA              = (3, 3)                # Gaussian smoothing sigma (time, trace)
EPS                = 1e-30

ANTENNA_HEIGHT = 0.35     # [m] 送信機高さ
SYSTEM_LAG_NS  = 0.837    # [ns] システムラグ
RX_DEPTH       = 0.10     # [m] 受信機の埋設深さ
ANCHOR_FREQ    = 450e6    # [Hz] Method A の損失アンカー周波数

# 氷の複素誘電率。実部は .in から読めれば上書きする（§3）。
# 損失項 6e-5 は k_centroid_freq_diff.py の定数をそのまま踏襲する。
EPS_ICE_DEFAULT   = 3.17
ICE_LOSS_TANGENT  = 6e-5

# 描画のカラースケール（k_centroid_freq.py と同一）
VMIN_F, VMAX_F = 0.5, 2.0   # [GHz]

# Debye パラメータの既定値（.in から読めない場合のフォールバック）
DEBYE_DEFAULT = {'tau1': 4.6212e-11, 'tau2': 2.82195e-10,
                 'de_ratio': 0.261 / (0.261 + 0.088)}


# =============================================================================
# 1. 物理モデル（k_centroid_freq.py からの逐語移植）
# =============================================================================
def get_eps_static(z_m):
    """深さ z [m] から静的実部とロスタンジェントを計算
    (Heiken1991 Fig 9.54 の 450 MHz 計測経験式; イルメナイト20wt%考慮)"""
    z_cm = z_m * 100.0
    rho = 1.92 * (z_cm + 12.2) / (z_cm + 18.0)
    eps_static = 1.843 ** rho
    tan_d = 10 ** (0.033 * 20.0 + 0.231 * rho - 3.061)
    return eps_static, tan_d


def get_eps_regolith(z_m, omega, d_params, anchor_freq=450e6):
    """指定深さ z_m [m] と角周波数配列 omega に対するレゴリス母材の複素誘電率。
    2極Debye (Method A: 損失アンカー方式)。水氷層は考慮しない。"""
    eps_static, tan_d = get_eps_static(z_m)

    tau1 = d_params['tau1']
    tau2 = d_params['tau2']
    de_ratio = d_params['de_ratio']

    w_a = 2.0 * np.pi * anchor_freq
    unit_im_wa = (de_ratio * (w_a * tau1) / (1.0 + (w_a * tau1)**2) +
                  (1.0 - de_ratio) * (w_a * tau2) / (1.0 + (w_a * tau2)**2))
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


def surface_delay_ns(antenna_height, system_lag_ns):
    """地表面反射の到達時刻 (プロット上の基準線 'Surface') [ns]。"""
    return antenna_height * 2 / 0.3 + system_lag_ns


def smooth_masked(data, mask, sigma):
    """NaN を考慮した Gaussian 平滑（k_centroid_freq.py と同一）。"""
    filled = np.where(mask, data, 0.0)
    sm_data   = gaussian_filter(filled,              sigma=sigma)
    sm_weight = gaussian_filter(mask.astype(float),  sigma=sigma)
    out = np.full_like(sm_data, np.nan)
    np.divide(sm_data, sm_weight, out=out, where=(sm_weight > 1e-6))
    # スムージングで値が周囲へにじみ出るのを防ぐため、
    # 元々有効だったピクセル以外は NaN に戻す
    out[~mask] = np.nan
    return out


def mix_maxwell_garnett(eps_host, eps_incl, f_vol):
    """Maxwell-Garnett 混合則（k_centroid_freq_diff.py と同一式）。"""
    return eps_host + 3.0 * f_vol * eps_host * (eps_incl - eps_host) / (
        eps_incl + 2.0 * eps_host - f_vol * (eps_incl - eps_host))


# =============================================================================
# 2. `.in` からのパラメータ読み取り（§3）
# =============================================================================
def get_ice_params(in_dir):
    """ice 側 Seed_0/Ascan/ 直下の *.in から Debye/氷層パラメータを読み取る。

    k_centroid_freq_diff.py の抽出ロジックを流用。DE_RATIO は式表記
    (例 `0.261 / (0.261 + 0.088)`) を許容するため、コメントを除去してから
    eval する。これは k_centroid_freq.py が実効的に行っている処理と
    数値的に同一である。

    Returns
    -------
    dict: tau1, tau2, de_ratio, f_ice, ice_top, ice_bot, eps_ice,
          in_files（読んだファイル一覧）, found_*（各項目が読めたか）
    """
    p = {
        'tau1': DEBYE_DEFAULT['tau1'],
        'tau2': DEBYE_DEFAULT['tau2'],
        'de_ratio': DEBYE_DEFAULT['de_ratio'],
        'f_ice': None, 'ice_top': None, 'ice_bot': None,
        'eps_ice': None,
        'in_files': [], 'debye_found': False,
    }

    if not in_dir or not os.path.isdir(in_dir):
        raise FileNotFoundError(
            f".in 探索ディレクトリが存在しません: {in_dir}\n"
            f"  ice 側 Seed_0/Ascan/ に .in ファイルが必要です（共通仕様 §3）。")

    in_files = sorted(glob.glob(os.path.join(in_dir, '*.in')))
    if not in_files:
        raise FileNotFoundError(f".in ファイルが見つかりません: {in_dir}/*.in")

    for in_file in in_files:
        print(f"  Reading parameters from: {in_file}")
        p['in_files'].append(in_file)
        try:
            with open(in_file, 'r', encoding='utf-8') as fin:
                content = fin.read()

            m_tau1  = re.search(r'DEBYE_TAU1\s*=\s*([0-9\.eE\+\-]+)', content)
            m_tau2  = re.search(r'DEBYE_TAU2\s*=\s*([0-9\.eE\+\-]+)', content)
            m_ratio = re.search(r'DE_RATIO\s*=\s*(.+)', content)
            m_disp  = re.search(r'#add_dispersion_debye:\s*\d+\s+([0-9\.eE\+\-]+)'
                                r'\s+[0-9\.eE\+\-]+\s+([0-9\.eE\+\-]+)', content)

            if m_tau1:
                p['tau1'] = float(m_tau1.group(1))
                p['debye_found'] = True
            if m_tau2:
                p['tau2'] = float(m_tau2.group(1))
                p['debye_found'] = True

            if m_ratio:
                expr = m_ratio.group(1).split('#')[0].strip()
                p['de_ratio'] = float(eval(expr))
                p['debye_found'] = True
            elif m_disp:
                de1, de2 = float(m_disp.group(1)), float(m_disp.group(2))
                if (de1 + de2) > 0:
                    p['de_ratio'] = de1 / (de1 + de2)
                    p['debye_found'] = True

            m_fice = re.search(r'f_ice\s*=\s*([0-9\.eE\+\-]+)', content)
            m_top  = re.search(r'ice_top\s*=\s*([0-9\.eE\+\-]+)', content)
            m_bot  = re.search(r'ice_bot\s*=\s*([0-9\.eE\+\-]+)', content)
            m_eps  = re.search(r'eps_ice\s*=\s*([0-9\.eE\+\-]+)', content)

            if m_fice: p['f_ice']   = float(m_fice.group(1))
            if m_top:  p['ice_top'] = float(m_top.group(1))
            if m_bot:  p['ice_bot'] = float(m_bot.group(1))
            if m_eps:  p['eps_ice'] = float(m_eps.group(1))

        except Exception as e:
            print(f"  Warning: Failed to parse {in_file}: {e}")

    # --- 氷層定義が読めない場合はエラー停止（理論線が描けないため）---
    missing = [k for k in ('f_ice', 'ice_top', 'ice_bot') if p[k] is None]
    if missing:
        raise ValueError(
            f"氷層定義を .in から読み取れませんでした（不足: {', '.join(missing)}）。\n"
            f"  探索先: {in_dir}\n"
            f"  読んだファイル: {p['in_files']}\n"
            f"  理論線が描けず解析の意味が失われるため停止します（共通仕様 §3）。")

    if p['ice_bot'] <= p['ice_top']:
        raise ValueError(f"氷層定義が不正です: ice_top={p['ice_top']} >= ice_bot={p['ice_bot']}")

    if p['eps_ice'] is None:
        p['eps_ice'] = EPS_ICE_DEFAULT
        p['eps_ice_source'] = f'default ({EPS_ICE_DEFAULT})'
    else:
        p['eps_ice_source'] = '.in'

    if not p['debye_found']:
        print("  Warning: Debye パラメータを .in から抽出できませんでした。既定値を使用します。")

    print("  === Parameters read from .in ===")
    print(f"    f_ice   = {p['f_ice']}")
    print(f"    ice_top = {p['ice_top']} m")
    print(f"    ice_bot = {p['ice_bot']} m")
    print(f"    eps_ice = {p['eps_ice']}  (source: {p['eps_ice_source']})")
    print(f"    DEBYE_TAU1 = {p['tau1']}")
    print(f"    DEBYE_TAU2 = {p['tau2']}")
    print(f"    DE_RATIO   = {p['de_ratio']}")
    print("  ================================")
    return p


# =============================================================================
# 3. パス探索・seed 列挙（§1）
# =============================================================================
def _seed_num(path):
    m = re.search(r'Seed_(\d+)', os.path.basename(path.rstrip('/')))
    return int(m.group(1)) if m else -1


def list_seed_jsons(case_dir):
    """case_dir 配下の Seed_*/Bscan/Bscan.json を seed 番号昇順で返す。
    0 から連番でない場合はエラー停止。"""
    hits = glob.glob(os.path.join(case_dir, 'seed_*', 'Bscan', 'Bscan.json'))
    if not hits:
        raise FileNotFoundError(
            f"seed_*/Bscan/Bscan.json が見つかりません: {case_dir}")

    pairs = sorted(((_seed_num(os.path.dirname(os.path.dirname(h))), h) for h in hits),
                   key=lambda x: x[0])
    nums = [n for n, _ in pairs]

    expected = list(range(len(nums)))
    if nums != expected:
        missing = sorted(set(range(max(nums) + 1)) - set(nums))
        raise ValueError(
            f"seed が 0 からの連番になっていません: {case_dir}\n"
            f"  検出した seed : {nums}\n"
            f"  欠番の seed   : {missing}\n"
            f"  期待する seed : {expected}\n"
            f"  共通仕様 §1.3 によりエラー停止します。")

    return nums, [h for _, h in pairs]


def is_case_dir(path):
    return bool(glob.glob(os.path.join(path, 'Seed_*')))


def enumerate_cases(path):
    """(a) case ディレクトリ / (c) 親ディレクトリ を判別して case 一覧を返す。"""
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise NotADirectoryError(f"ディレクトリが存在しません: {path}")

    if is_case_dir(path):
        print(f"[入力判別] case ディレクトリとして処理します: {path}")
        return [path]

    subs = sorted(d for d in glob.glob(os.path.join(path, '*'))
                  if os.path.isdir(d) and glob.glob(os.path.join(d, 'seed_*')))
    if not subs:
        raise FileNotFoundError(
            f"seed_*/ も、seed_*/ を持つサブディレクトリも見つかりません: {path}")

    print(f"[入力判別] 親ディレクトリとして処理します: {path}")
    print(f"           対象 case ({len(subs)} 件): {[os.path.basename(s) for s in subs]}")
    return subs


def derive_rand_amp(case_dir):
    m = re.search(r'rand_amp_(\d+)', case_dir)
    return m.group(1) if m else ''


def derive_eval_type(case_dir):
    m = re.search(r'Eval_([^/\\]+)', case_dir)
    return m.group(1) if m else ''


def derive_noice_dir(case_dir):
    """パス中の Eval_XXX/rand_amp_YYY/ZZZ を No_Ice/rand_amp_YYY に置換。
    （本ツールでは使用しないが、共通仕様 §1.2 の導出規則として実装し記録する）"""
    m = re.search(r'(.*?)Eval_[^/\\]+[/\\]rand_amp_(\d+)[/\\][^/\\]+[/\\]?$',
                  case_dir.rstrip('/\\') + os.sep)
    if not m:
        return None
    return os.path.join(m.group(1), 'No_Ice', f'rand_amp_{m.group(2)}')


# =============================================================================
# 4. B-scan 読み込みとシード連結（§4）
# =============================================================================
def load_bscan(json_path):
    """Bscan.json から outputdata, dt, gpr_step, params を読む。"""
    with open(json_path) as f:
        params = json.load(f)
    outfile = params['data']
    gpr_step = params['antenna_settings']['src_step']
    outputdata, dt = get_output_data(outfile, 1, 'Ez')
    return outputdata, dt, gpr_step, params


def load_all_seeds(json_paths):
    """全 seed を読み込み、整合性を検証して返す（連結はまだしない）。

    検証（不一致はエラー停止）: dt / n_samples / n_traces / gpr_step
    """
    datas, metas = [], []
    for i, jp in enumerate(json_paths):
        outputdata, dt, gpr_step, params = load_bscan(jp)
        n_samples, n_traces = outputdata.shape
        datas.append(outputdata)
        metas.append({'seed': i, 'json': jp, 'dt': dt, 'gpr_step': gpr_step,
                      'n_samples': n_samples, 'n_traces': n_traces,
                      'params': params})
        print(f"  Seed_{i}: shape=(samples={n_samples}, traces={n_traces}), "
              f"dt={dt*1e12:.4f} ps, gpr_step={gpr_step} m")

    ref = metas[0]
    for m in metas[1:]:
        for key, tol in (('dt', 1e-18), ('gpr_step', 1e-12)):
            if abs(m[key] - ref[key]) > tol:
                raise ValueError(
                    f"seed 間で {key} が一致しません（共通仕様 §4.1-2）。\n"
                    f"  Seed_{ref['seed']}: {ref[key]}\n"
                    f"  Seed_{m['seed']}: {m[key]}")
        for key in ('n_samples', 'n_traces'):
            if m[key] != ref[key]:
                raise ValueError(
                    f"seed 間で {key} が一致しません（共通仕様 §4.1-2）。\n"
                    f"  Seed_{ref['seed']}: {ref[key]}\n"
                    f"  Seed_{m['seed']}: {m[key]}")

    print(f"  整合性検証 OK: dt / n_samples / n_traces / gpr_step が全 seed で一致")
    return datas, metas


def concat_seeds(datas, nseed, gpr_step):
    """seed 0..nseed-1 をトレース方向に連結し、軸と境界位置を返す。

    x 軸・測線長・境界位置はすべて実測値（n_traces × gpr_step）から算出する。
    前処理（平均トレース除去等）は一切行わない（共通仕様 §4.2）。
    """
    sel = datas[:nseed]
    data = np.hstack(sel)
    n_traces_per_seed = sel[0].shape[1]
    n_total = data.shape[1]

    x_axis = np.arange(n_total) * gpr_step
    bounds_idx = np.cumsum([n_traces_per_seed] * (nseed - 1)) if nseed > 1 else np.array([], dtype=int)
    bounds_x = bounds_idx * gpr_step
    line_length_m = n_total * gpr_step
    return data, x_axis, bounds_idx, bounds_x, line_length_m


# =============================================================================
# 5. STFT / centroid（k_centroid_freq.py からの逐語移植）
# =============================================================================
def compute_centroid_maps(outputdata, dt):
    """STFT → centroid map → valid_mask → smooth_masked → shift_rate。

    k_centroid_freq.py と同一ロジック・同一パラメータ。前処理は追加しない。
    """
    dt_ns = dt * 1e9          # [ns]
    fs = 1.0 / dt_ns          # [GHz]
    n_samples, n_traces = outputdata.shape

    f_axis, t_axis, _ = signal.stft(outputdata[:, 0], fs=fs, window=WINDOW,
                                    nperseg=NPERSEG, noverlap=NOVERLAP)
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

    trace_peak = power_map.max(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        power_rel_db = 10.0 * np.log10(
            np.where(trace_peak > 0, power_map / (trace_peak + EPS), EPS))
    valid_mask = power_rel_db >= POWER_THRESHOLD_DB

    centroid_masked = np.where(valid_mask, centroid_map, np.nan)
    centroid_smooth = smooth_masked(centroid_map, valid_mask, SIGMA)

    dt_stft = t_axis[1] - t_axis[0]   # [ns]

    def shift_rate(freq_map):
        return np.gradient(freq_map, dt_stft, axis=0)

    return {
        't_axis': t_axis, 'f_axis': f_axis, 'dt_stft': dt_stft,
        'fs': fs, 'n_traces': n_traces, 'n_samples': n_samples,
        'valid_freq': valid_freq,
        'centroid_masked': centroid_masked,
        'centroid_smooth': centroid_smooth,
        'sr_raw': shift_rate(centroid_masked),
        'sr_smooth': shift_rate(centroid_smooth),
        'valid_mask': valid_mask,
    }


def trace_profiles(maps):
    """トレース方向の中央値・25/75%（raw と smooth の両方）。"""
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for key, arr in (('cen_raw', maps['centroid_masked']),
                         ('cen_sm',  maps['centroid_smooth']),
                         ('sr_raw',  maps['sr_raw']),
                         ('sr_sm',   maps['sr_smooth'])):
            out[f'{key}_med'] = np.nanmedian(arr, axis=1)
            out[f'{key}_p25'] = np.nanpercentile(arr, 25, axis=1)
            out[f'{key}_p75'] = np.nanpercentile(arr, 75, axis=1)
    return out


# =============================================================================
# 6. 解析（理論）プロファイル — 物理計算はこの 1 関数のみ（共通仕様 §0-3）
# =============================================================================
def load_incident_spectrum(ascan_outfile_path):
    """入射波 A-scan から帯域内スペクトル (f_calc, S0_calc, omega) を得る。"""
    if not ascan_outfile_path or not os.path.exists(ascan_outfile_path):
        raise FileNotFoundError(
            f"A-scan 参照波形が見つかりません: {ascan_outfile_path}\n"
            f"  ASCAN_OUTFILE_PATH ([EDIT HERE] ブロック) を確認してください。")

    ascan_data, dt_ascan = get_output_data(ascan_outfile_path, 1, 'Ez')
    e_incident = ascan_data if ascan_data.ndim == 1 else ascan_data[:, 0]

    N = len(e_incident)
    freq_ascan = np.fft.rfftfreq(N, d=dt_ascan)
    S0_omega = np.fft.rfft(e_incident)

    band_mask = (freq_ascan >= FREQ_MIN * 1e9) & (freq_ascan <= FREQ_MAX * 1e9)
    f_calc = freq_ascan[band_mask]
    S0_calc = S0_omega[band_mask]
    omega = 2 * np.pi * f_calc
    return {'f_calc': f_calc, 'S0_calc': S0_calc, 'omega': omega,
            'dt_ascan': dt_ascan, 'n_samples': N}


def compute_time_offset_ns():
    """t_offset = システムラグ + 空中往復 + rx 埋設分の往復 [ns]。"""
    t_air_ns = (2.0 * ANTENNA_HEIGHT / const.c) * 1e9
    d_sub_offset = np.linspace(0, RX_DEPTH, 50)
    eps_sub_offset, _ = get_eps_static(d_sub_offset)
    v_sub = const.c / np.sqrt(eps_sub_offset)
    dt_sub = d_sub_offset[1] - d_sub_offset[0]
    t_ground_start_ns = np.sum(2.0 * dt_sub / v_sub) * 1e9
    t_offset_ns = SYSTEM_LAG_NS + t_air_ns + t_ground_start_ns
    return t_offset_ns, t_air_ns, t_ground_start_ns


def compute_analytical_profile(t_axis, dt_stft, incident, debye_params,
                               f_ice, ice_top, ice_bot, eps_ice_real):
    """解析 centroid / shift-rate プロファイルを計算する唯一の関数。

    氷の有無・濃度は引数 `f_ice` で切り替える（f_ice=0.0 なら氷なし＝レゴリスのみ
    となり、k_centroid_freq.py の解析計算と完全に一致する）。
    共通仕様 §0-3 に従い、物理計算をここ以外には書かない。

    Returns
    -------
    dict: cen [GHz], sr [GHz/ns]（いずれも t_axis 上に補間済み）,
          d_array, t_delay_d, t_offset_ns, t_layer_top, t_layer_bottom
    """
    f_calc = incident['f_calc']
    S0_calc = incident['S0_calc']
    omega = incident['omega']

    t_offset_ns, _, _ = compute_time_offset_ns()

    max_depth = (t_axis[-1] * 1e-9) * const.c / 2
    d_array = np.linspace(RX_DEPTH, max_depth, 400)
    d_step = d_array[1] - d_array[0]

    eps_ice_complex = eps_ice_real * (1.0 - 1j * ICE_LOSS_TANGENT)

    f_peak_d, t_delay_d = [], []
    cumulative_attenuation = np.zeros_like(omega)
    cumulative_time = np.zeros_like(omega)

    for i, d in enumerate(d_array):
        eps_host = get_eps_regolith(d, omega, debye_params, anchor_freq=ANCHOR_FREQ)

        if f_ice > 0 and ice_top <= d <= ice_bot:
            eps_complex_w = mix_maxwell_garnett(eps_host, eps_ice_complex, f_ice)
        else:
            eps_complex_w = eps_host

        alpha_d = - (omega / const.c) * np.imag(np.sqrt(eps_complex_w))
        v_d = const.c / np.real(np.sqrt(eps_complex_w))

        if i > 0:
            cumulative_attenuation += alpha_d * d_step
            cumulative_time += 2 * d_step / v_d

        S_d_w = S0_calc * np.exp(-2 * cumulative_attenuation)
        power = np.abs(S_d_w) ** 2

        f_peak = _TRAPZ(f_calc * power, f_calc) / _TRAPZ(power, f_calc)
        f_peak_d.append(f_peak)

        t_delay_ground = np.interp(f_peak, f_calc, cumulative_time)
        t_delay_d.append(t_offset_ns + (t_delay_ground * 1e9))

    f_peak_d = np.array(f_peak_d) / 1e9    # [GHz]
    t_delay_d = np.array(t_delay_d)

    cen = np.interp(t_axis, t_delay_d, f_peak_d, left=np.nan, right=np.nan)
    sr = np.gradient(cen, dt_stft)

    # 層区間（§5）: 理論の深さ→往復遅延マップをそのまま使う。
    # 速度モデルは上のループと同一（氷層内は case 実際の f_ice で MG 混合）。
    t_layer_top = float(np.interp(ice_top, d_array, t_delay_d))
    t_layer_bottom = float(np.interp(ice_bot, d_array, t_delay_d))

    return {'cen': cen, 'sr': sr, 'd_array': d_array, 't_delay_d': t_delay_d,
            't_offset_ns': t_offset_ns,
            't_layer_top': t_layer_top, 't_layer_bottom': t_layer_bottom,
            'f_ice': f_ice}


# =============================================================================
# 7. 統計（§6）
# =============================================================================
def region_stats(t, values, t0, t1, corr_len_ns):
    """区間 [t0, t1] の (mean, sem, n_eff, z)。全領域統計はこの同一シグネチャ。"""
    if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
        return np.nan, np.nan, 0.0, np.nan

    mask = (t >= t0) & (t <= t1) & np.isfinite(values)
    if not np.any(mask):
        return np.nan, np.nan, 0.0, np.nan

    v = np.asarray(values)[mask]
    mean_v = float(np.mean(v))
    std_v = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    n_eff = max(1.0, (t1 - t0) / corr_len_ns + 1.0)
    sem = std_v / np.sqrt(n_eff)
    z = mean_v / sem if sem > 0 else np.nan
    return mean_v, sem, n_eff, z


def valid_time_range(t_axis, prof_med):
    """有効データ区間 = 中央値プロファイルが有限値を持つ時間範囲。"""
    finite = np.isfinite(prof_med)
    if not finite.any():
        return np.nan, np.nan
    idx = np.where(finite)[0]
    return float(t_axis[idx[0]]), float(t_axis[idx[-1]])


def clip_region(t0, t1, tv0, tv1):
    """区間 ∩ 有効データ区間。空なら (nan, nan)。"""
    if not all(np.isfinite(x) for x in (t0, t1, tv0, tv1)):
        return np.nan, np.nan
    a, b = max(t0, tv0), min(t1, tv1)
    if b <= a:
        return np.nan, np.nan
    return float(a), float(b)


def plateau_region(t0, t1, t_axis=None, min_pts=2):
    """層下端側 1/4 区間（プラトー値の評価区間）。

    薄い層では 1/4 区間に STFT の時間サンプルが 1 点も入らず統計が NaN になる。
    （STFT のホップは (nperseg-noverlap)=64 サンプル分あり、層厚によっては
    1/4 区間幅がこれを下回るため。）そのため、区間内のサンプル数が `min_pts`
    未満の場合は層下端側から `min_pts` 点を含むところまで区間を広げ、広げたか
    どうかをフラグで返す。実際に使った区間は stats.csv の t0_ns / t1_ns に
    そのまま出力される（共通仕様 §5）。

    Returns
    -------
    (t_start, t_end, widened: bool)
    """
    if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
        return np.nan, np.nan, False

    a = float(t1 - (t1 - t0) / 4.0)
    if t_axis is None:
        return a, float(t1), False

    t_axis = np.asarray(t_axis, dtype=float)
    in_layer = t_axis[(t_axis >= t0) & (t_axis <= t1)]
    if in_layer.size == 0:
        return np.nan, np.nan, False

    n_in_plateau = int(np.sum(in_layer >= a))
    if n_in_plateau >= min_pts:
        return a, float(t1), False

    take = min(min_pts, in_layer.size)
    a_new = float(in_layer[-take])
    # 端点の浮動小数比較で取りこぼさないよう僅かに広げる
    a_new -= 1e-9 * max(1.0, abs(a_new))
    return a_new, float(t1), True


def pointwise_sem(p25, p75, n_traces):
    """中央値プロファイルの標準誤差 σ/√n, σ=(p75-p25)/1.349（§6）。"""
    sigma = (np.asarray(p75) - np.asarray(p25)) / 1.349
    return sigma / np.sqrt(n_traces)


# =============================================================================
# 8. 作図（§8。k_centroid_freq.py の体裁を踏襲）
# =============================================================================
def _theory_label(f_ice, is_case):
    if f_ice <= 0:
        base = '0 vol% (no ice)'
    else:
        base = f'{f_ice * 100:g} vol%'
    return base + (' (this case)' if is_case else '')


def plot_map_with_profile(data, t_axis, extent, prof_med, prof_p25, prof_p75,
                          theory_curves, out_path, title, kind,
                          vmin, vmax, bounds_x=(), layer_span=None,
                          concatenated=False):
    """左＝マップ(imshow)、右＝プロファイル(中央値+IQR+理論線+地表線)。

    kind: 'centroid' | 'shiftrate'
    theory_curves: [(f_ice, profile_array, is_case), ...]
    """
    initial_delay = surface_delay_ns(ANTENNA_HEIGHT, SYSTEM_LAG_NS)

    fig, axes = plt.subplots(
        nrows=1, ncols=2,
        width_ratios=[3, 1],
        height_ratios=[1],
        figsize=(12, 8),
    )

    cmap = 'jet' if kind == 'centroid' else 'RdBu_r'
    cbar_label = 'Frequency [GHz]' if kind == 'centroid' else 'Frequency shift rate [GHz/ns]'
    xlabel_prof = 'Frequency [GHz]' if kind == 'centroid' else 'Shift rate [GHz/ns]'

    ax = axes[0]
    im = ax.imshow(data, extent=extent, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axhline(initial_delay, color='gray', linestyle='--', lw=2)
    for xb in bounds_x:
        ax.axvline(xb, color='white', linestyle='--', lw=1.5)
    ax.set_xlabel('Distance [m] (concatenated)' if concatenated else 'Distance [m]', size=18)
    ax.set_ylabel('Delay time [ns]', size=18)
    ax.tick_params(labelsize=14)
    ax.minorticks_on()
    ax.grid()
    divider = axgrid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(cbar_label, size=18)
    cbar.ax.tick_params(labelsize=14)

    ax2 = axes[1]
    if layer_span is not None and all(np.isfinite(layer_span)):
        ax2.axhspan(layer_span[0], layer_span[1], color='blue', alpha=0.10,
                    label=f'Ice layer ({layer_span[0]:.1f}-{layer_span[1]:.1f} ns)')
    ax2.fill_betweenx(t_axis, prof_p25, prof_p75, color='gray', alpha=0.4,
                      label='IQR (25-75%)')
    ax2.plot(prof_med, t_axis, color='k', linestyle='-', label='Median')

    n_theory = max(1, len(theory_curves))
    colors = plt.cm.viridis(np.linspace(0, 0.85, n_theory))
    for i, (f_ice, prof, is_case) in enumerate(theory_curves):
        if prof is None or np.all(~np.isfinite(prof)):
            continue
        ax2.plot(prof, t_axis, color=colors[i], linestyle='--',
                 lw=3.0 if is_case else 1.5,
                 label=_theory_label(f_ice, is_case))

    ax2.axhline(initial_delay, color='gray', linestyle='--', lw=2, label='Surface')

    ax2.legend(fontsize=9, loc='lower center')
    ax2.set_xlabel(xlabel_prof, size=18)
    ax2.set_ylabel('Delay time [ns]', size=18)
    ax2.set_ylim(t_axis[-1], t_axis[0])
    ax2.tick_params(labelsize=14)
    ax2.minorticks_on()
    ax2.grid()

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'  Saved: {out_path}')
    plt.close(fig)


# =============================================================================
# 9. 出力ユーティリティ
# =============================================================================
def fice_tag(f_ice):
    """理論列名のタグ。0.10 -> 'fice010'（THEORY_FICE_SWEEP から動的生成）。"""
    pct = f_ice * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f'fice{int(round(pct)):03d}'
    return 'fice' + f'{f_ice:.4f}'.replace('.', 'p')


def write_profile_csv(path, t_axis, prof, theory_curves):
    """profile.csv（列名の理論部は THEORY_FICE_SWEEP から動的生成）。"""
    cols = [('t_ns', t_axis)]
    for key in ('cen_raw', 'cen_sm', 'sr_raw', 'sr_sm'):
        for stat in ('med', 'p25', 'p75'):
            cols.append((f'{key}_{stat}', prof[f'{key}_{stat}']))
    for f_ice, tc, _ in theory_curves:
        cols.append((f'theory_cen_{fice_tag(f_ice)}', tc['cen']))
    for f_ice, tc, _ in theory_curves:
        cols.append((f'theory_sr_{fice_tag(f_ice)}', tc['sr']))

    header = ','.join(name for name, _ in cols)
    stack = np.column_stack([np.asarray(v, dtype=float) for _, v in cols])
    np.savetxt(path, stack, delimiter=',', header=header, comments='', fmt='%.8g')
    print(f'  Saved: {path}')
    return [name for name, _ in cols]


STATS_HEADER = [
    'metric', 'region', 't0_ns', 't1_ns',
    'mean', 'sem', 'n_eff', 'z',
    'pt_sem_mean', 'region_widened',
    'theory_case_mean', 'theory_noice_mean',
    'diff_vs_noice_mean', 'diff_vs_noice_sem', 'diff_vs_noice_z',
]


def build_stats_rows(t_axis, prof, theory_case, theory_noice, regions, n_traces,
                     widened=None):
    """centroid / shift rate × 層内・プラトー・浅部 の統計行を作る。

    観測は smooth 版の中央値プロファイルを用いる（差分ツール②と整合）。
    「氷なし理論との差」= 観測 - 理論(f_ice=0) の同一区間統計。
    """
    widened = widened or {}
    rows = []
    metrics = (
        ('centroid',  'cen_sm', theory_case['cen'], theory_noice['cen']),
        ('shiftrate', 'sr_sm',  theory_case['sr'],  theory_noice['sr']),
    )
    for mname, key, th_case, th_noice in metrics:
        obs = prof[f'{key}_med']
        ptsem = pointwise_sem(prof[f'{key}_p25'], prof[f'{key}_p75'], n_traces)
        resid = obs - th_noice
        for rname, (t0, t1) in regions.items():
            mean, sem, n_eff, z = region_stats(t_axis, obs, t0, t1, CORR_LEN_NS)
            d_mean, d_sem, _, d_z = region_stats(t_axis, resid, t0, t1, CORR_LEN_NS)
            th_c, _, _, _ = region_stats(t_axis, th_case, t0, t1, CORR_LEN_NS)
            th_n, _, _, _ = region_stats(t_axis, th_noice, t0, t1, CORR_LEN_NS)
            ps, _, _, _ = region_stats(t_axis, ptsem, t0, t1, CORR_LEN_NS)
            rows.append({
                'metric': mname, 'region': rname, 't0_ns': t0, 't1_ns': t1,
                'mean': mean, 'sem': sem, 'n_eff': n_eff, 'z': z,
                'pt_sem_mean': ps,
                'region_widened': int(bool(widened.get(rname, False))),
                'theory_case_mean': th_c, 'theory_noice_mean': th_n,
                'diff_vs_noice_mean': d_mean, 'diff_vs_noice_sem': d_sem,
                'diff_vs_noice_z': d_z,
            })
    return rows


def _fmt(v):
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return 'nan'
    return f'{v:.8g}'


def write_csv_rows(path, header, rows):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')
        for r in rows:
            f.write(','.join(_fmt(r.get(k)) for k in header) + '\n')
    print(f'  Saved: {path}')


SUMMARY_HEADER = [
    'case_name', 'eval_type', 'rand_amp', 'f_ice', 'ice_top_m', 'ice_bot_m',
    'mode', 'metric', 'nseed', 'line_length_m', 'n_traces',
    'layer_mean', 'layer_sem', 'layer_n_eff', 'layer_z',
    'plateau_mean', 'plateau_sem', 'plateau_z',
    'shallow_mean', 'shallow_sem', 'shallow_z',
    'theory_layer_mean', 't_layer_top_ns', 't_layer_bottom_ns',
]


# =============================================================================
# 10. README 生成（§9）
# =============================================================================
def write_readme(path, info):
    sweep_txt = ', '.join(f'{f*100:g} vol%' for f in info['theory_sweep'])
    seeds_txt = ', '.join(f'Seed_{i}' for i in info['seeds'])
    nseed_rows = '\n'.join(
        f"| {d['nseed']} | {d['n_traces']} | {d['line_length_m']:.1f} | `{d['dirname']}/` |"
        for d in info['nseed_table'])
    prof_cols = '\n'.join(f'- `{c}`' for c in info['profile_columns'])

    txt = f"""# STFT centroid 単体解析（複数シード連結） — README

**生成日時**: {info['timestamp']}
**生成ツール**: `k_ms_centroid_single.py`（単体解析ツール①）
**対象 case**: `{info['case_name']}`

このディレクトリは、水氷ありデータのみを用いた STFT centroid 解析の出力です。
複数シードの B-scan をトレース方向に連結し、nseed = 1..{info['nseed_max']} のそれぞれについて
マップ・プロファイル・統計を出力しています。氷なしとの差分は取っていません
（検出判定は差分ツール② `diff_same/` `diff_cross/` が行います）。

---

## 1. 入力

| 項目 | 値 |
|---|---|
| case ディレクトリ | `{info['case_dir']}` |
| No_Ice ディレクトリ | `{info['noice_dir']}`（**本ツールでは未使用**。参考記録） |
| 使用した seed | {seeds_txt}（計 {len(info['seeds'])} 個） |
| A-scan 参照波形 | `{info['ascan_path']}` |
| 1 seed のトレース数 | {info['n_traces_per_seed']}（実測値） |
| GPR step | {info['gpr_step']} m（実測値） |
| 1 seed の測線長 | {info['seed_line_length_m']:.1f} m（実測値 = トレース数 × step） |

`.in` の探索先: `{info['in_dir']}`
読み取ったファイル: {info['in_files']}

## 2. 物理パラメータ（`.in` から読み取り）

| 項目 | 値 |
|---|---|
| `f_ice` | {info['f_ice']} |
| `ice_top` | {info['ice_top']} m |
| `ice_bot` | {info['ice_bot']} m |
| `eps_ice` | {info['eps_ice']}（source: {info['eps_ice_source']}、損失 tanδ={ICE_LOSS_TANGENT} は既存コード踏襲）|
| `DEBYE_TAU1` | {info['tau1']} |
| `DEBYE_TAU2` | {info['tau2']} |
| `DE_RATIO` | {info['de_ratio']} |

### 導出した層区間

深さ→往復遅延の換算は理論プロファイル計算と同一の速度モデル
（`v(z) = c / Re(√ε(z))`、氷層内は case 実際の `f_ice` で Maxwell-Garnett 混合）
を用いています。

| 区間 | 開始 [ns] | 終了 [ns] |
|---|---|---|
| 時間オフセット `t_offset` | — | {info['t_offset_ns']:.3f} |
| 地表反射（基準線） | — | {info['t_surface_ns']:.3f} |
| 氷層（理論） | {info['t_layer_top']:.3f} | {info['t_layer_bottom']:.3f} |
| 有効データ区間 | {info['t_valid0']:.3f} | {info['t_valid1']:.3f} |
| **層内統計に実際に使った区間** | {info['t_layer_used0']:.3f} | {info['t_layer_used1']:.3f} |
| **プラトー統計に実際に使った区間** | {info['t_plateau0']:.3f} | {info['t_plateau1']:.3f} |
| **浅部コントロールに実際に使った区間** | {info['t_shallow0']:.3f} | {info['t_shallow1']:.3f} |

{info['clip_note']}

## 3. 解析条件

| 項目 | 値 |
|---|---|
| STFT | `nperseg={NPERSEG}`, `noverlap={NOVERLAP}`, `window='{WINDOW}'` |
| 解析帯域 | {FREQ_MIN} – {FREQ_MAX} GHz |
| パワーマスク閾値 | {POWER_THRESHOLD_DB} dB（トレース内ピーク基準の相対値）|
| 平滑 sigma | {SIGMA}（時間軸, トレース軸）NaN 考慮の Gaussian |
| 系列相関長 `CORR_LEN_NS` | {CORR_LEN_NS} ns |
| 損失アンカー周波数 | {ANCHOR_FREQ/1e6:g} MHz |
| アンテナ高 / システムラグ / rx 埋設深 | {ANTENNA_HEIGHT} m / {SYSTEM_LAG_NS} ns / {RX_DEPTH} m |

**前処理は一切行っていません。** 平均トレース除去も、その他の B-scan 前処理も
適用せず、連結した生データをそのまま STFT にかけています。このため
**nseed=1 の結果は同じ Seed_0 に対する `k_centroid_freq.py` の出力と一致します**。

## 4. 出力ファイル

### nseed 別ディレクトリ

| nseed | n_traces | 測線長 [m] | ディレクトリ |
|---|---|---|---|
{nseed_rows}

各ディレクトリの中身:

| ファイル | 内容 |
|---|---|
| `centroid_map_smooth.png` | 平滑版 centroid マップ + プロファイル（理論線重畳）|
| `centroid_map_raw.png` | 生版 centroid |
| `shiftrate_map_smooth.png` | 平滑版 shift rate マップ + プロファイル（理論線重畳）|
| `shiftrate_map_raw.png` | 生版 shift rate |
| `profile.csv` | 時間方向 1D プロファイル（下記の列）|
| `stats.csv` | 層内・プラトー・浅部の統計 |

### `profile.csv` の列

{prof_cols}

- `t_ns`: 遅延時間 [ns]（STFT の時間軸）
- `cen_*`: centroid 周波数 [GHz]。`raw` = マスク後の生値、`sm` = Gaussian 平滑後。
- `sr_*`: shift rate [GHz/ns]（centroid の時間微分）。
- `med` / `p25` / `p75`: トレース方向の中央値と 25 / 75 パーセンタイル。
- `theory_cen_ficeNNN` / `theory_sr_ficeNNN`: 氷体積分率 NNN/1000 …ではなく
  **NNN は体積パーセントの 3 桁表記**（例 `fice010` = 10 vol%）。
  スイープ値は {sweep_txt}。層の幾何（`ice_top`/`ice_bot`）は case 固有の実値です。

### `stats.csv` の列

`metric`（centroid / shiftrate）× `region`（layer / plateau / shallow）ごとに 1 行。

- `t0_ns`, `t1_ns`: 実際に統計を取った区間（有効データ区間で切った後の値）。
- `mean`, `sem`, `n_eff`, `z`: 観測値（平滑版中央値プロファイル）の領域統計。
- `pt_sem_mean`: 区間内で平均した「中央値プロファイルの標準誤差」σ/√n_traces。
  トレース数を増やしたときの誤差の縮小を見る量です（後述の注意も参照）。
- `region_widened`: 1 なら、その区間は定義どおりの幅では STFT 時間サンプルが
  足りず、下端側に自動拡張されています（プラトー区間で薄い層のときに起こる）。
  実際に使った区間は `t0_ns` / `t1_ns` を見てください。
- `theory_case_mean`: case 実際の `f_ice` の理論線の区間平均。
- `theory_noice_mean`: `f_ice=0`（氷なし）理論線の区間平均。
- `diff_vs_noice_*`: **観測 − 氷なし理論** の区間統計。観測が 5 本の理論線の
  どのあたりに位置するかの指標です。

### `summary/` 配下

| ファイル | 内容 |
|---|---|
| `summary_single.csv` | 1 行 = 1 条件（nseed × metric）。case 識別列を先頭に配置 |
| `run_info_single.json` | 再現性のための実行記録 |

`summary_single.csv` は case をまたいで単純連結できるよう設計されています
（先頭に `case_name, eval_type, rand_amp, f_ice, ice_top_m, ice_bot_m`）。

## 5. 統計量の定義

- σ（1 トレース分）= `(p75 − p25) / 1.349`
- 中央値プロファイルの標準誤差 = σ / √n_traces
- 領域統計の実効独立標本数 `n_eff = max(1, 区間長 / {CORR_LEN_NS} + 1)`
  （centroid は Gaussian 平滑により時間方向に相関を持つため、点数をそのまま
  独立標本数にはできない）
- SEM = 区間内の点間標準偏差 (ddof=1) / √n_eff、 z = 平均 / SEM
- **層内統計は 2 通りを併記**:
  - **(a) 層内平均** (`region=layer`): 区間全体の平均。保守的で他手法（LSR 等）と
    比較可能。
  - **(b) プラトー値** (`region=plateau`): 層下端側 1/4 区間の平均。ランプの飽和値
    ＝検出力の実力値。
    ただし STFT のホップは `nperseg - noverlap = {NPERSEG - NOVERLAP}` サンプル分あり、
    層が薄いと 1/4 区間に時間サンプルが 1 点も入りません。その場合は層下端側から
    2 点を含むまで区間を自動拡張し、`stats.csv` の `region_widened=1` と
    `t0_ns`/`t1_ns` で明示しています。

## 6. 解釈上の注意（重要）

- **連結マップは異なるシードの並置です。** 水平方向に地下構造が不連続であり、
  seed 境界（マップ上の白破線）をまたぐ横方向の構造には物理的意味がありません。
  nseed ≥ 2 では x 軸ラベルを `Distance [m] (concatenated)` としています。
- **Δcentroid は層内でランプ状**（層を通過するにつれ効果が累積）になるため、
  層内平均の SEM は過大評価（z は過小評価）になります。プラトー値も併せて
  見てください。
- **異シード差分はスペックルの点ごとの相殺が働かない**ため、点ごとの Δ(t)
  カーブではなく領域統計で判断してください（差分ツール②に該当）。
- **`pt_sem_mean` は nseed に対して単純な 1/√N では縮まりません。** σ 自体が
  `(p75 − p25)/1.349` というトレース方向のばらつきであり、異なるシードを連結すると
  「別の地下構造の実現値」が混ざるため σ が増加します。√n_traces の効果と σ の
  増加が競合するので、誤差の縮小は 1/√N より緩やかになります（場合によっては
  nseed=1 → 2 で一旦増えます）。これはバグではなく、連結が独立試行の追加ではなく
  異なる実現値の並置であることの直接の帰結です。
- **本ツールは検出の判定を行いません。** 単体データでは氷シグナルが系統誤差に
  埋もれることが既に判明しています。ここでの統計は
  (1) 観測が理論線のどのあたりに位置するか、(2) トレース数による誤差の縮小
  を見るためのものです。検出判定は差分ツール② で行ってください。
- 理論線 5 本は `f_ice` のみのスイープです。**層の幾何は case 固有の実値**
  （`ice_top={info['ice_top']}` m, `ice_bot={info['ice_bot']}` m）を使っています。

## 7. 再現方法

```bash
python k_ms_centroid_single.py "{info['case_dir']}"
```

親ディレクトリを渡せば配下の case を一括処理します:

```bash
python k_ms_centroid_single.py "{os.path.dirname(info['case_dir'])}"
```

実行時の全パラメータは `summary/run_info_single.json` に記録されています。
"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(txt)
    print(f'  Saved: {path}')


def _format_region_notes(notes):
    """(nseed, 本文) のリストを本文ごとにまとめた README 用の文字列にする。"""
    if not notes:
        return ('層区間・プラトー区間はいずれも定義どおりに取れており、'
                '切り詰め・拡張は発生していません。')
    grouped = {}
    for nseed, body in notes:
        grouped.setdefault(body, []).append(nseed)
    lines = ['**区間の切り詰め・拡張**:', '']
    for body, ns in grouped.items():
        lines.append(f'- nseed = {", ".join(str(n) for n in ns)}: {body}')
    return '\n'.join(lines)


# =============================================================================
# 11. case 単位の処理
# =============================================================================
def process_case(case_dir, incident, noice_dir_override=None):
    case_name = os.path.basename(case_dir.rstrip('/\\'))
    print('\n' + '=' * 78)
    print(f'[CASE] {case_name}')
    print(f'       {case_dir}')
    print('=' * 78)

    # --- a. seed 一覧の取得・検証 ---
    seeds, seed_jsons = list_seed_jsons(case_dir)
    print(f'  検出した seed: {seeds}')

    # --- b. .in からパラメータ読み取り（ice 側 Seed_0/Ascan/）---
    in_dir = os.path.join(case_dir, f'Seed_{seeds[0]}', 'Ascan')
    p = get_ice_params(in_dir)
    debye_params = {'tau1': p['tau1'], 'tau2': p['tau2'], 'de_ratio': p['de_ratio']}

    rand_amp = derive_rand_amp(case_dir)
    eval_type = derive_eval_type(case_dir)
    noice_dir = noice_dir_override or derive_noice_dir(case_dir)
    print(f'  rand_amp={rand_amp}, eval_type={eval_type}')
    print(f'  No_Ice dir (未使用・記録のみ): {noice_dir}')

    # --- B-scan 読み込み（全 seed）と整合性検証 ---
    print('  B-scan を読み込み中...')
    datas, metas = load_all_seeds(seed_jsons)
    dt = metas[0]['dt']
    gpr_step = metas[0]['gpr_step']
    n_traces_per_seed = metas[0]['n_traces']
    nseed_max = len(datas)

    # --- c/d. 時間軸を確定し、理論プロファイルを計算（case ごとに 1 回）---
    fs_probe = 1.0 / (dt * 1e9)
    _, t_axis, _ = signal.stft(datas[0][:, 0], fs=fs_probe, window=WINDOW,
                               nperseg=NPERSEG, noverlap=NOVERLAP)
    dt_stft = t_axis[1] - t_axis[0]

    sweep = list(THEORY_FICE_SWEEP)
    case_f_ice = p['f_ice']
    if not any(abs(case_f_ice - s) < 1e-12 for s in sweep):
        sweep.append(case_f_ice)      # スイープに無ければ 6 本目として追加
        sweep.sort()
        print(f'  case の f_ice={case_f_ice} はスイープ外のため理論線を追加しました。')

    print(f'  理論プロファイルを計算中 (f_ice sweep = {sweep}) ...')
    theory_curves = []
    for f_val in sweep:
        tc = compute_analytical_profile(t_axis, dt_stft, incident, debye_params,
                                        f_val, p['ice_top'], p['ice_bot'], p['eps_ice'])
        is_case = abs(f_val - case_f_ice) < 1e-12
        theory_curves.append((f_val, tc, is_case))

    theory_case = next(tc for f_val, tc, is_c in theory_curves if is_c)
    theory_noice = next(tc for f_val, tc, _ in theory_curves if f_val == 0.0)

    t_offset_ns = theory_case['t_offset_ns']
    t_layer_top = theory_case['t_layer_top']
    t_layer_bottom = theory_case['t_layer_bottom']
    t_surface = surface_delay_ns(ANTENNA_HEIGHT, SYSTEM_LAG_NS)
    print(f'  t_offset      = {t_offset_ns:.3f} ns')
    print(f'  t_surface     = {t_surface:.3f} ns')
    print(f'  t_layer_top   = {t_layer_top:.3f} ns  (ice_top={p["ice_top"]} m)')
    print(f'  t_layer_bottom= {t_layer_bottom:.3f} ns  (ice_bot={p["ice_bot"]} m)')

    # --- 出力ディレクトリ ---
    ms_root = os.path.join(case_dir, 'multi_seed_analysis', 'STFT_analysis')
    single_root = os.path.join(ms_root, 'single')
    summary_dir = os.path.join(ms_root, 'summary')
    os.makedirs(single_root, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    summary_rows, nseed_table, profile_columns = [], [], []
    last_regions, clip_notes = None, []

    # --- e. nseed = 1..N ---
    for nseed in range(1, nseed_max + 1):
        data, x_axis, bounds_idx, bounds_x, line_length_m = concat_seeds(datas, nseed, gpr_step)
        n_total = data.shape[1]
        dirname = f'nseed_{nseed:02d}_{line_length_m:.1f}m'
        out_dir = os.path.join(single_root, dirname)
        os.makedirs(out_dir, exist_ok=True)

        print(f'\n  --- nseed={nseed}: n_traces={n_total}, line={line_length_m:.1f} m ---')

        maps = compute_centroid_maps(data, dt)
        if maps['t_axis'].size != t_axis.size or not np.allclose(maps['t_axis'], t_axis):
            raise ValueError('連結後の STFT 時間軸が probe と一致しません（想定外）。')

        prof = trace_profiles(maps)

        # 有効データ区間 ∩ 各区間（§5）
        tv0, tv1 = valid_time_range(t_axis, prof['cen_sm_med'])
        lay0, lay1 = clip_region(t_layer_top, t_layer_bottom, tv0, tv1)
        pla0, pla1, pla_widened = plateau_region(lay0, lay1, t_axis)
        sh0, sh1 = clip_region(t_surface, t_layer_top, tv0, tv1)
        regions = {'layer': (lay0, lay1), 'plateau': (pla0, pla1), 'shallow': (sh0, sh1)}
        widened = {'plateau': pla_widened}
        last_regions = regions

        if np.isfinite(lay1) and lay1 < t_layer_bottom - 1e-9:
            note = (f'層下端 {t_layer_bottom:.2f} ns が有効データ区間の終端 '
                    f'{tv1:.2f} ns を超えたため、層区間を {lay0:.2f}–{lay1:.2f} ns に'
                    f'切り詰めました。')
            print(f'    [注意] nseed={nseed}: {note}')
            clip_notes.append((nseed, note))
        print(f'    有効データ区間: {tv0:.2f} – {tv1:.2f} ns')
        print(f'    layer  : {lay0:.2f} – {lay1:.2f} ns')
        print(f'    plateau: {pla0:.2f} – {pla1:.2f} ns'
              + ('  [1/4 区間に STFT サンプルが不足のため下端側 2 点まで拡張]'
                 if pla_widened else ''))
        if pla_widened:
            note = (f'層が薄く 1/4 区間に STFT 時間サンプルが 2 点未満だったため、'
                    f'プラトー区間を {pla0:.2f}–{pla1:.2f} ns に拡張しました'
                    f'（stats.csv の region_widened=1）。')
            clip_notes.append((nseed, note))
        print(f'    shallow: {sh0:.2f} – {sh1:.2f} ns')

        # 作図
        extent = [0, n_total * gpr_step, t_axis[-1], t_axis[0]]
        all_sr_valid = maps['sr_raw'][np.isfinite(maps['sr_raw'])]
        sr_abs = np.percentile(np.abs(all_sr_valid), 95) if all_sr_valid.size > 0 else 1.0
        title_base = (f'{case_name} | single | nseed={nseed} | '
                      f'{line_length_m:.1f} m | {n_total} traces')
        concatenated = nseed >= 2
        cen_theory = [(f_val, tc['cen'], is_c) for f_val, tc, is_c in theory_curves]
        sr_theory = [(f_val, tc['sr'], is_c) for f_val, tc, is_c in theory_curves]

        for tag, arr, pkey, kind, th, vmin, vmax in (
            ('centroid_map_smooth.png', maps['centroid_smooth'], 'cen_sm', 'centroid', cen_theory, VMIN_F, VMAX_F),
            ('centroid_map_raw.png',    maps['centroid_masked'], 'cen_raw', 'centroid', cen_theory, VMIN_F, VMAX_F),
            ('shiftrate_map_smooth.png', maps['sr_smooth'], 'sr_sm', 'shiftrate', sr_theory, -sr_abs, sr_abs),
            ('shiftrate_map_raw.png',    maps['sr_raw'],    'sr_raw', 'shiftrate', sr_theory, -sr_abs, sr_abs),
        ):
            plot_map_with_profile(
                arr, t_axis, extent,
                prof[f'{pkey}_med'], prof[f'{pkey}_p25'], prof[f'{pkey}_p75'],
                th, os.path.join(out_dir, tag),
                title_base + (' | smooth' if 'smooth' in tag else ' | raw'),
                kind, vmin, vmax, bounds_x=bounds_x,
                layer_span=(lay0, lay1), concatenated=concatenated)

        # CSV
        profile_columns = write_profile_csv(
            os.path.join(out_dir, 'profile.csv'), t_axis, prof, theory_curves)
        stats_rows = build_stats_rows(t_axis, prof, theory_case, theory_noice,
                                      regions, n_total, widened)
        write_csv_rows(os.path.join(out_dir, 'stats.csv'), STATS_HEADER, stats_rows)

        # summary 行（1 行 = 1 条件 = nseed × metric）
        by = {(r['metric'], r['region']): r for r in stats_rows}
        for metric in ('centroid', 'shiftrate'):
            lay, pla, sha = by[(metric, 'layer')], by[(metric, 'plateau')], by[(metric, 'shallow')]
            summary_rows.append({
                'case_name': case_name, 'eval_type': eval_type, 'rand_amp': rand_amp,
                'f_ice': p['f_ice'], 'ice_top_m': p['ice_top'], 'ice_bot_m': p['ice_bot'],
                'mode': 'single', 'metric': metric, 'nseed': nseed,
                'line_length_m': line_length_m, 'n_traces': n_total,
                'layer_mean': lay['mean'], 'layer_sem': lay['sem'],
                'layer_n_eff': lay['n_eff'], 'layer_z': lay['z'],
                'plateau_mean': pla['mean'], 'plateau_sem': pla['sem'], 'plateau_z': pla['z'],
                'shallow_mean': sha['mean'], 'shallow_sem': sha['sem'], 'shallow_z': sha['z'],
                'theory_layer_mean': lay['theory_case_mean'],
                't_layer_top_ns': t_layer_top, 't_layer_bottom_ns': t_layer_bottom,
            })

        nseed_table.append({'nseed': nseed, 'n_traces': n_total,
                            'line_length_m': line_length_m, 'dirname': dirname})

    # --- f. summary / run_info / README ---
    write_csv_rows(os.path.join(summary_dir, 'summary_single.csv'),
                   SUMMARY_HEADER, summary_rows)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    run_info = {
        'tool': 'k_ms_centroid_single.py', 'mode': 'single',
        'timestamp': timestamp,
        'case_dir': case_dir, 'case_name': case_name,
        'noice_dir': noice_dir, 'noice_dir_used': False,
        'eval_type': eval_type, 'rand_amp': rand_amp,
        'seeds_used': seeds, 'seed_jsons': seed_jsons,
        'in_dir': in_dir, 'in_files': p['in_files'],
        'ice_params': {'f_ice': p['f_ice'], 'ice_top': p['ice_top'],
                       'ice_bot': p['ice_bot'], 'eps_ice': p['eps_ice'],
                       'eps_ice_source': p['eps_ice_source'],
                       'ice_loss_tangent': ICE_LOSS_TANGENT},
        'debye_params': debye_params,
        'ascan_outfile_path': ASCAN_OUTFILE_PATH,
        'stft_params': {'nperseg': NPERSEG, 'noverlap': NOVERLAP, 'window': WINDOW,
                        'freq_min_ghz': FREQ_MIN, 'freq_max_ghz': FREQ_MAX,
                        'power_threshold_db': POWER_THRESHOLD_DB,
                        'smoothing_sigma': list(SIGMA),
                        'dt_s': float(dt), 'dt_stft_ns': float(dt_stft),
                        'n_samples': int(metas[0]['n_samples'])},
        'preprocessing': 'none (no mean-trace subtraction)',
        'corr_len_ns': CORR_LEN_NS,
        'cross_seed_offset': CROSS_SEED_OFFSET,
        'theory_fice_sweep': sweep,
        'geometry': {'antenna_height_m': ANTENNA_HEIGHT, 'system_lag_ns': SYSTEM_LAG_NS,
                     'rx_depth_m': RX_DEPTH, 'anchor_freq_hz': ANCHOR_FREQ},
        'timing_ns': {'t_offset': t_offset_ns, 't_surface': t_surface,
                      't_layer_top': t_layer_top, 't_layer_bottom': t_layer_bottom},
        'regions_used_ns': {k: list(v) for k, v in last_regions.items()},
        'gpr_step_m': gpr_step, 'n_traces_per_seed': n_traces_per_seed,
        'per_nseed': nseed_table,
    }
    run_info_path = os.path.join(summary_dir, 'run_info_single.json')
    with open(run_info_path, 'w', encoding='utf-8') as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)
    print(f'  Saved: {run_info_path}')

    tv0, tv1 = valid_time_range(t_axis, prof['cen_sm_med'])
    readme_info = {
        'timestamp': timestamp, 'case_name': case_name, 'case_dir': case_dir,
        'noice_dir': noice_dir, 'seeds': seeds, 'ascan_path': ASCAN_OUTFILE_PATH,
        'n_traces_per_seed': n_traces_per_seed, 'gpr_step': gpr_step,
        'seed_line_length_m': n_traces_per_seed * gpr_step,
        'in_dir': in_dir, 'in_files': p['in_files'],
        'f_ice': p['f_ice'], 'ice_top': p['ice_top'], 'ice_bot': p['ice_bot'],
        'eps_ice': p['eps_ice'], 'eps_ice_source': p['eps_ice_source'],
        'tau1': p['tau1'], 'tau2': p['tau2'], 'de_ratio': p['de_ratio'],
        't_offset_ns': t_offset_ns, 't_surface_ns': t_surface,
        't_layer_top': t_layer_top, 't_layer_bottom': t_layer_bottom,
        't_valid0': tv0, 't_valid1': tv1,
        't_layer_used0': last_regions['layer'][0], 't_layer_used1': last_regions['layer'][1],
        't_plateau0': last_regions['plateau'][0], 't_plateau1': last_regions['plateau'][1],
        't_shallow0': last_regions['shallow'][0], 't_shallow1': last_regions['shallow'][1],
        'theory_sweep': sweep, 'nseed_max': nseed_max, 'nseed_table': nseed_table,
        'profile_columns': profile_columns,
        'clip_note': _format_region_notes(clip_notes),
    }
    write_readme(os.path.join(ms_root, 'README_single.md'), readme_info)

    print(f'\n  [CASE 完了] 出力先: {ms_root}')
    return ms_root


# =============================================================================
# 12. main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description='複数シード STFT centroid 単体解析ツール（水氷ありのみ）')
    ap.add_argument('path', nargs='?', default=None,
                    help='case ディレクトリ または その親ディレクトリ')
    ap.add_argument('--noice_dir', default=None,
                    help='No_Ice の case_dir を明示指定（本ツールでは記録のみ）')
    args = ap.parse_args()

    path = args.path or input('Input case dir (or parent dir): ').strip()
    path = os.path.abspath(os.path.expanduser(path.strip('"\' ')))

    cases = enumerate_cases(path)

    print(f'\nA-scan 参照波形を読み込み中: {ASCAN_OUTFILE_PATH}')
    incident = load_incident_spectrum(ASCAN_OUTFILE_PATH)
    print(f'  入射スペクトル: {incident["n_samples"]} samples, '
          f'帯域内 {incident["f_calc"].size} bins '
          f'({incident["f_calc"][0]/1e9:.3f} – {incident["f_calc"][-1]/1e9:.3f} GHz)')

    out_dirs = []
    for case_dir in cases:
        out_dirs.append(process_case(case_dir, incident, args.noice_dir))

    print('\n' + '=' * 78)
    print('全 case 完了。出力ディレクトリ:')
    for d in out_dirs:
        print(f'  {d}')
    print('=' * 78)


if __name__ == '__main__':
    main()