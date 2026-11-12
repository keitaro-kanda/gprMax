"""反射で検出できる水氷の最小濃度 v_min（案 C → 新案 A）

案 C（計算結果をそのまま濃度方向に伸ばす）
    氷の応答 ΔT = T − W[R] を界面の走時のまわりで切り出し、v / v0 倍して R に足す。
        T(v) = R + Σ_k (v / v0) · ΔT_k(t − δt_k(v))
    これを bscan_detection と同じ反射チャネルにかけて P_D(v) を出し、P_D = 0.5 の v を v_min とする。
    値が出るのは計算した 2 点（上端 1 m・下端 2 m）だけ。

新案 A（案 C を全部の深さで行う）
    基準点（計算した反射）の応答 ΔT_s を、振幅の倍率 g と走時のずれ Δt で別の濃度 v・深さ z に移し、
    その深さの水氷なしのクラッタ R に足して、同じ反射チャネルで P_D(v, z) を出す。
        T(v, z) = R + g · ΔT_s(t − Δt)
        g  = F_k(v, z) / F_s(v0, z0)，  F = |Γ| · G · Tr · L（帯域中心での反射係数・2D 拡がり・透過・吸収）
        Δt = t_k(v, z) − t_s(v0, z0)
    s は使う基準点（A_SOURCE = 'same'：上端の地図は上端 1 m、下端の地図は下端 2 m）。
    検証：上端 1 m の応答から下端 2 m を予測し、下端 2 m の案 C と比べる（逆も）。

案 A′（参考線）：旧来の「信号の中央値 vs しきい値 ＋ 較正 ΔM」。SHOW_A_PRIME_REF で図に重ねる。

式の説明は bscan_detectability_map.md。
出力：<対象の merged.out のフォルダ>/bscan_detection/detectability_map/（PDF は pdf/ の下）
"""

import copy
import csv
import os
import time
from datetime import datetime

import numpy as np
from scipy import fft as sp_fft
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import bscan_detection as bd               # 入力・反射チャネル・平均媒質
import bscan_superposition_check as bsc    # 時間伸縮 W[R]
import subsurface_model as sm
from ascan_spectrum import load_trace, measure_peak
from gprMax.exceptions import CmdInputError

# 図の共通ルール（このファイルの中でも明示する。bscan_detection の版に依存しないように）
#   文字の大きさ：タイトル・軸ラベル 18、目盛り・凡例・図中の文字 14
#   グリッド線：特に意図がない限り全パネルに付ける（破線 '--'、alpha 0.3）
#   補助目盛り（minor ticks）：全パネルで表示する
plt.rcParams.update({'axes.titlesize': 18, 'axes.labelsize': 18, 'figure.titlesize': 18,
                     'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14,
                     'font.size': 14,
                     'axes.grid': True, 'axes.grid.which': 'major', 'axes.axisbelow': 'line',
                     'grid.linestyle': '--', 'grid.alpha': 0.3,
                     'xtick.minor.visible': True, 'ytick.minor.visible': True})

# =============================================================================
# 設定  [EDIT HERE]
# =============================================================================
JSON_PATH = '/Volumes/gprMax_water_ice/2D_12x4/water_ice_study_test/out_file_paths.json'
TARGET_CONDITION = None            # 氷ありの条件名。None なら基準以外で最初のもの
OUTPUT_SUBDIR = 'detectability_map'
EVENTS = ('ice_top', 'ice_bottom')

# --- 案 C -----------------------------------------------------------------------
C_V_GRID = np.round(np.arange(0.0, 0.4001, 0.01), 4)   # 合成する濃度（体積分率）
C_GATE_FLAT_NS = 2.5               # 氷の応答を切り出す窓の平らな部分の半幅 a（反射ゲート ±1 ns と
                                   # |E| の最大値フィルタ ±1 ns を覆う幅）
C_GATE_TAPER_NS = 0.5              # その外側のコサインテーパの幅 b
C_WORK_MARGIN_NS = 10.0            # 反射チャネルを計算する時間範囲の余白（計算を軽くするため）
AMP_HW_NS = 1.5                    # A_sim・σ_a（トレースごとの |E| の最大値）を取るゲートの半幅

# --- 新案 A ---------------------------------------------------------------------
A_V_GRID = np.round(np.arange(0.0, 0.4001, 0.01), 4)    # 地図の横軸（体積分率）
A_Z_GRID = np.round(np.arange(0.20, 2.95 + 1e-9, 0.05), 3)  # 地図の縦軸：氷層の上端の深さ [m]
A_SOURCE = 'same'                  # 使う基準点：'same'（上端→①、下端→②）/ 'nearest'（深さが近いほう）
A_WORK_HW_NS = 6.0                 # 判定に使う時間窓の半幅（界面の走時 ± これ）
VMIN_SMOOTH_HW_M = 0.25            # v_min(z) の移動平均の半幅 [m]（前後この範囲の点を単純平均する）
CROSS_K_SIGMA = 2.0                # 検証の目安：差が「ばらつき σ の何倍以内」なら整合とみなすか
BIAS_CHECK = True                  # 全深さで基準点を入れ替えた地図も作り、系統的な偏りを調べる
                                   #（地図の計算時間が約 2 倍になる）
SURFACE_SKIP_NS = 5.0              # 地表反射の後ろこの範囲は使わない
EDGE_GUARD_NS = 1.0                # 計算領域の底からの反射の手前この範囲は使わない

# --- 案 A′（参考線）------------------------------------------------------------
SHOW_A_PRIME_REF = True            # 旧案 A′（中央値 vs しきい値 ＋ ΔM）の v_min(z) を figA2 に重ねる
A_FIXPOINT_ITERS = 8

# --- 作図 -----------------------------------------------------------------------
EV_COL = {'ice_top': 'tab:orange', 'ice_bottom': 'tab:green'}   # bscan_detection と同じ
EV_ANCHOR_NAME = {'ice_top': '①', 'ice_bottom': '②'}
DB20 = 20.0 / np.log(10.0)


# =============================================================================
# 1. 入力
# =============================================================================
def load_inputs():
    """B-scan（基準・対象）、記録、far_1m を読み、bscan_detection と同じ前処理をする。"""
    level, chosen, conds, reference = bd.select_conditions(JSON_PATH)
    ref = bd.REFERENCE_CONDITION
    if ref not in conds:
        raise CmdInputError('基準の条件 "{}" がありません'.format(ref))
    targets = [c for c in sorted(conds) if c != ref]
    tgt = TARGET_CONDITION or (targets[0] if targets else None)
    if tgt not in conds:
        raise CmdInputError('対象の条件 "{}" がありません（候補: {}）'.format(tgt, targets))
    p_r, p_t = conds[ref][bd.BSCAN_KEY], conds[tgt][bd.BSCAN_KEY]
    R, dt = bd.read_bscan(p_r)
    T, dt_t = bd.read_bscan(p_t)
    if abs(dt_t - dt) > 1e-9 * dt or T.shape != R.shape:
        raise CmdInputError('基準と対象で dt または B-scan の大きさが違います')
    if bd.SUBTRACT_FREESPACE:
        fs_path = bd.FREESPACE_AT_TX_PATH or reference.get(bd.FREESPACE_KEY, '')
        if not fs_path or not os.path.exists(fs_path):
            raise CmdInputError('直達波の除去に使う自由空間 at_tx が見つかりません')
        fs, _ = load_trace(fs_path)
        fs = np.pad(fs, (0, max(0, R.shape[0] - len(fs))))[:R.shape[0]]
        R, T = R - fs[:, None], T - fs[:, None]
    rp_R, rec_R = bd.find_record(p_r)
    rp_T, rec_T = bd.find_record(p_t)
    if not rec_R or bd.has_ice(rec_R):
        raise CmdInputError('基準の記録が無いか、氷が入っています: {}'.format(rp_R))
    if not rec_T or not bd.has_ice(rec_T):
        raise CmdInputError('対象の記録が無いか、氷が入っていません: {}'.format(rp_T))
    h_t, h_r = rec_T.get('rho', {}).get('rho_map_sha256'), rec_R.get('rho', {}).get('rho_map_sha256')
    if not h_t or h_t != h_r:
        print('注意: ランダム場が基準と一致しません（rho_map_sha256）。案 C は成り立ちません。')
    x_all, has_x = bd.trace_positions(rec_R, R.shape[1])
    sel = np.ones(R.shape[1], dtype=bool)
    if bd.TRACE_X_RANGE_M is not None:
        sel = (x_all >= bd.TRACE_X_RANGE_M[0] - 1e-9) & (x_all <= bd.TRACE_X_RANGE_M[1] + 1e-9)
    far, dt_far = load_trace(reference[bd.REF_KEY])
    src_delay = measure_peak(far, dt_far)['t_peak'] - sm.R_REF / sm.C
    return {'R': R[:, sel], 'T': T[:, sel], 'dt': dt, 'x': x_all[sel],
            'x_label': 'x [m]' if has_x else 'Trace number', 'rec_R': rec_R, 'rec_T': rec_T,
            'src_delay': src_delay, 't': np.arange(R.shape[0]) * dt * 1e9,
            'paths': {'reference': p_r, 'target': p_t, 'record_ref': rp_R, 'record_target': rp_T,
                      'far_1m': reference[bd.REF_KEY]}, 'target': tgt,
            'sha_match': bool(h_t) and h_t == h_r}


# =============================================================================
# 2. 平均媒質の式（振幅の 4 因子と走時）
# =============================================================================
def model_rec(rec, v, z_top):
    """記録を写して、濃度と上端の深さだけを変える（厚さ・モデル・組成はそのまま）。"""
    r = copy.deepcopy(rec)
    r['ice_vol_frac'] = float(max(v, 1e-9))
    r['ice_top_m'] = float(z_top)
    r['enable_ice'] = True
    return r


def iface_depth(rec, name):
    z = float(rec['ice_top_m'])
    return z if name == 'ice_top' else z + float(rec['ice_thick_m'])


def event_time(rec, name, src_delay):
    """界面の往復群走時 [ns]（bscan_detection.model_events と同じ）。"""
    for e in bd.model_events(rec, src_delay):
        if e['name'] == name:
            return e['t_ns']
    raise CmdInputError('界面 {} がありません'.format(name))


def factors(rec, name):
    """帯域中心での |Γ|、G、Tr（往復の透過）、L（往復の吸収）と、その積 F。"""
    bd.configure_mean_medium(rec)
    lvl = bd.THEORY_LEVEL
    f = np.array([sm.BAND_CENTRE_HZ])
    d = iface_depth(rec, name)
    att, _, inv = sm.path_integrals(f, d, lvl)
    G = float(np.sqrt(sm.R_REF / (2.0 * sm.TX_HEIGHT + 2.0 * inv[0])))
    Tr = float(np.abs(sm.transmission_product(f, d, lvl, two_way=True, include_surface=True)[0]))
    Gam = float(np.abs(sm.interface_reflection(f, d, lvl)[0]))
    L = float(np.exp(-2.0 * att[0]))
    return {'Gamma': Gam, 'G': G, 'Tr': Tr, 'L': L, 'F': Gam * G * Tr * L}


def pore_capacity(rec, z_top):
    """pore ice で空隙を満たせる上限 φ(z + D) = 1 − ρ(z + D)/ρ_g（excess には上限なし）。"""
    if rec['ice_model'] != 'pore':
        return np.inf
    zb = z_top + float(rec['ice_thick_m'])
    return float(1.0 - sm.density_profile(zb) / sm.LEVEL4_RHO_GRAIN)


# =============================================================================
# 3. 反射チャネル（bscan_detection と同じ定義）
# =============================================================================
def pd_at(rs, t_ev):
    """界面の走時 ± REFL_GATE_HW_NS での P_D の最大。"""
    m = np.abs(rs['t'] - t_ev) <= bd.REFL_GATE_HW_NS
    return float(rs['pd'][m].max()) if m.any() else np.nan


def reflection_scan_window(T, R, dt, t, t0, t1):
    """反射チャネルを t0–t1 の範囲だけで計算する（案 C 用）。"""
    i0 = max(0, int(np.floor(t0 / (dt * 1e9))))
    i1 = min(len(t), int(np.ceil(t1 / (dt * 1e9))))
    rs = bd.reflection_scan(T[i0:i1], R[i0:i1], dt)
    rs['t'] = rs['t'] + t[i0]
    return rs


def pd_local(ctx, Tw, i0, t_k):
    """時間窓 Tw（全体の i0 サンプル目から）について、界面 t_k での P_D を出す。
    しきい値は水氷なし全体から作った A_thr(t) を使う（bscan_detection.reflection_scan と同じ値）。"""
    dt_ns = ctx['dt'] * 1e9
    n_gate = max(3, int(round(2 * bd.REFL_GATE_HW_NS / dt_ns)) | 1)
    sel = np.abs(ctx['thr_t'] - t_k) <= bd.REFL_GATE_HW_NS
    loc = ctx['thr_idx'][sel] - i0
    ok = (loc >= 0) & (loc < Tw.shape[0])
    if not ok.any():
        return np.nan
    pk = ndimage.maximum_filter1d(np.abs(Tw), size=n_gate, axis=0, mode='nearest')   # bscan_detection と同じ量
    above = pk[loc[ok]] > ctx['thr'][sel][ok][:, None]
    return float(above.mean(axis=1).max())


def gate(t, tc):
    """氷の応答を切り出す窓 w_k：平らな部分 ±a、その外側に幅 b のコサインテーパ。"""
    u = np.abs(t - tc) - C_GATE_FLAT_NS
    w = np.where(u <= 0, 1.0, 0.0)
    m = (u > 0) & (u < C_GATE_TAPER_NS)
    w[m] = 0.5 * (1.0 + np.cos(np.pi * u[m] / C_GATE_TAPER_NS))
    return w


def shift_rows(X, delay_ns, dt):
    """X(t − delay)（周波数領域の位相回転、全トレース同じ遅れ）。"""
    if abs(delay_ns) < 1e-12:
        return X
    n = X.shape[0]
    nfft = sp_fft.next_fast_len(2 * n)
    fs = sp_fft.rfftfreq(nfft)
    ph = np.exp(-2j * np.pi * fs * delay_ns / (dt * 1e9))
    return sp_fft.irfft(sp_fft.rfft(X, nfft, axis=0) * ph[:, None], nfft, axis=0)[:n]


def crossing(v, pd, level=0.5):
    """P_D(v) が初めて level を下から越える v（線形内挿）。nan は飛ばす。越えなければ nan。"""
    ok = np.isfinite(pd)
    v, pd = np.asarray(v)[ok], np.asarray(pd)[ok]
    if v.size == 0:
        return np.nan
    if pd[0] >= level:
        return float(v[0])
    for i in range(1, len(v)):
        if pd[i - 1] < level <= pd[i]:
            return float(v[i - 1] + (level - pd[i - 1]) * (v[i] - v[i - 1]) / (pd[i] - pd[i - 1]))
    return np.nan


def time_ok(ctx, t_k):
    """走時が使える範囲か（地表反射の直後と計算領域の底の近くは使わない。窓がデータに収まる）。"""
    lo = ctx['t_surface'] + SURFACE_SKIP_NS
    hi = min(ctx['t_dom'] - EDGE_GUARD_NS, ctx['t'][-1] - A_WORK_HW_NS)
    return lo <= t_k <= hi


# =============================================================================
# 4. 案 C と基準点
# =============================================================================
def case_c(ctx):
    R, T, dt, t = ctx['R'], ctx['T'], ctx['dt'], ctx['t']
    rec_R, rec_T, src = ctx['rec_R'], ctx['rec_T'], ctx['src_delay']
    v0, z0 = float(rec_T['ice_vol_frac']), float(rec_T['ice_top_m'])

    # 氷の応答 ΔT = T − W[R]（W：氷層より深いクラッタの遅れを戻す）
    wm = bsc.warp_model(rec_R, rec_T, 2.0 * sm.TX_HEIGHT / sm.C + src,
                        float(rec_T.get('surface_y_m', 3.0)))
    Rw, _, _ = bsc.apply_warp(R, dt, wm)
    DW = T - Rw
    t_ev0 = {k: event_time(rec_T, k, src) for k in EVENTS}
    resp = {k: DW * gate(t, t_ev0[k])[:, None] for k in EVENTS}

    # 基準点 ①（上端 1 m）・②（下端 2 m）：切り出した応答、帯域中心の因子、振幅、ばらつき
    anchor = {}
    for k in EVENTS:
        w = gate(t, t_ev0[k])
        idx = np.nonzero(w > 0)[0]
        g = np.abs(t - t_ev0[k]) <= AMP_HW_NS
        pk = DB20 * np.log(np.maximum(np.abs(resp[k])[g].max(axis=0), 1e-300))   # |E| の最大値
        anchor[k] = {'t_ns': t_ev0[k], 'depth_m': iface_depth(rec_T, k),
                     'seg': resp[k][idx[0]:idx[-1] + 1].copy(), 't_seg0': float(t[idx[0]]),
                     'F0': factors(model_rec(rec_T, v0, z0), k),
                     'A_sim': float(10.0 ** (np.median(pk) / 20.0)), 'sigma_db': float(pk.std())}

    # 濃度を変えた合成と P_D（空隙を満たせる上限までの濃度だけ）
    vgrid = C_V_GRID[C_V_GRID <= pore_capacity(rec_T, z0) + 1e-12]
    t_bot_max = event_time(model_rec(rec_T, vgrid.max(), z0), 'ice_bottom', src)
    t0w, t1w = min(t_ev0.values()) - C_WORK_MARGIN_NS, t_bot_max + C_WORK_MARGIN_NS
    pd = {k: np.zeros(vgrid.size) for k in EVENTS}
    t_ev = {k: np.zeros(vgrid.size) for k in EVENTS}
    for i, v in enumerate(vgrid):
        rv = model_rec(rec_T, v, z0)
        Tv = R.copy()
        for k in EVENTS:
            t_ev[k][i] = event_time(rv, k, src)
            Tv += (v / v0) * shift_rows(resp[k], t_ev[k][i] - t_ev0[k], dt)
        rs = reflection_scan_window(Tv, R, dt, t, t0w, t1w)
        for k in EVENTS:
            pd[k][i] = pd_at(rs, t_ev[k][i])
    print('  案 C: 完了（{} 濃度）'.format(vgrid.size))

    # 整合の確認：v0 での合成の P_D と、gprMax の T そのものの P_D
    rsT = reflection_scan_window(T, R, dt, t, t0w, t1w)
    for k in EVENTS:
        anchor[k]['pd_T'] = pd_at(rsT, t_ev0[k])
        anchor[k]['pd_C_at_v0'] = float(np.interp(v0, vgrid, pd[k]))
        anchor[k]['vmin_C'] = crossing(vgrid, pd[k], 0.5)
    return {'v': vgrid, 'pd': pd, 't_ev': t_ev, 'anchor': anchor, 'v0': v0, 'z0': z0}


# =============================================================================
# 5. 新案 A：基準点の応答を別の濃度・深さに移して判定する
# =============================================================================
def source_for(ctx, k, z_top):
    """界面 k・上端の深さ z の反射を作るときに使う基準点。"""
    if A_SOURCE == 'same':
        return k
    d = z_top if k == 'ice_top' else z_top + float(ctx['rec_T']['ice_thick_m'])
    anc = ctx['C']['anchor']
    return min(EVENTS, key=lambda s: abs(anc[s]['depth_m'] - d))


def place(seg, t_seg0, delay_ns, t_w0, nwin, dt):
    """切り出した応答 seg（先頭の時刻 t_seg0）を delay_ns だけ遅らせ、時刻 t_w0 から始まる長さ nwin の窓に置く。"""
    dt_ns = dt * 1e9
    start = (t_seg0 + delay_ns - t_w0) / dt_ns
    n_int = int(np.floor(start))
    frac = start - n_int
    m = seg.shape[0]
    nfft = sp_fft.next_fast_len(m + 8)
    ph = np.exp(-2j * np.pi * sp_fft.rfftfreq(nfft) * frac)
    s = sp_fft.irfft(sp_fft.rfft(seg, nfft, axis=0) * ph[:, None], nfft, axis=0)[:m + 1]
    out = np.zeros((nwin, seg.shape[1]))
    a, b = max(0, n_int), min(nwin, n_int + s.shape[0])
    if b > a:
        out[a:b] = s[a - n_int:b - n_int]
    return out


def pd_point(ctx, k, v, z_top, source):
    """新案 A の 1 点：T(v, z) = R + g · ΔT_s(t − Δt) を作り、界面 k での P_D を返す。"""
    rec_T, src, R, dt = ctx['rec_T'], ctx['src_delay'], ctx['R'], ctx['dt']
    if v > pore_capacity(rec_T, z_top) + 1e-12:
        return np.nan, np.nan, np.nan
    r = model_rec(rec_T, v, z_top)
    t_k = event_time(r, k, src)
    if not time_ok(ctx, t_k):
        return np.nan, t_k, np.nan
    anc = ctx['C']['anchor'][source]
    gain = factors(r, k)['F'] / anc['F0']['F'] if v > 0 else 0.0
    dt_ns = dt * 1e9
    i0 = max(0, int(np.floor((t_k - A_WORK_HW_NS) / dt_ns)))
    i1 = min(R.shape[0], int(np.ceil((t_k + A_WORK_HW_NS) / dt_ns)))
    Tw = R[i0:i1].copy()
    if gain > 0:
        Tw += gain * place(anc['seg'], anc['t_seg0'], t_k - anc['t_ns'], ctx['t'][i0], i1 - i0, dt)
    return pd_local(ctx, Tw, i0, t_k), t_k, gain


def compute_map(ctx, k, choose):
    """界面 k の P_D(v, z) の地図。choose(z) が使う基準点を返す。"""
    P = np.full((A_Z_GRID.size, A_V_GRID.size), np.nan)
    Gm = np.full_like(P, np.nan)
    tk = np.full(A_Z_GRID.size, np.nan)
    srcs = []
    for i, z in enumerate(A_Z_GRID):
        s = choose(z)
        srcs.append(s)
        for j, v in enumerate(A_V_GRID):
            P[i, j], t_k, Gm[i, j] = pd_point(ctx, k, v, z, s)
            if j == 0:
                tk[i] = t_k
    vmin = np.array([crossing(A_V_GRID, P[i]) if np.isfinite(P[i]).any() else np.nan
                     for i in range(A_Z_GRID.size)])
    return P, Gm, tk, srcs, vmin


def smooth_vmin(z, v, hw):
    """v_min(z) の移動平均（前後 ±hw の単純平均。dB で平均する）。有限の点だけを使う。
    戻り値：平均した v_min、各点の平均線からのずれ [dB]、ずれの標準偏差 σ_pt [dB]。"""
    v = np.asarray(v, dtype=float)
    ok = np.isfinite(v) & (v > 0)
    x = np.full(v.shape, np.nan)
    x[ok] = DB20 * np.log(v[ok])
    xs = np.full(v.shape, np.nan)
    for i in np.nonzero(ok)[0]:
        xs[i] = np.mean(x[ok & (np.abs(z - z[i]) <= hw + 1e-9)])
    r = x - xs
    rr = r[np.isfinite(r)]
    sig = float(np.std(rr)) if rr.size >= 5 else np.nan
    return 10.0 ** (xs / 20.0), r, sig


def case_new_a(ctx):
    t0 = time.time()
    out = {'pd': {}, 'vmin': {}, 'source': {}, 'gain': {}, 't': {},
           'vmin_s': {}, 'resid_db': {}, 'sigma_pt': {}, 'bias': {}}
    for k in EVENTS:
        P, Gm, tk, srcs, vmin = compute_map(ctx, k, lambda z, k=k: source_for(ctx, k, z))
        out['pd'][k], out['gain'][k], out['t'][k], out['source'][k], out['vmin'][k] = \
            P, Gm, tk, srcs, vmin
        d_k = depth_axis(ctx, k)
        out['vmin_s'][k], out['resid_db'][k], out['sigma_pt'][k] = \
            smooth_vmin(d_k, vmin, VMIN_SMOOTH_HW_M)
        print('  新案 A: {} の地図を計算しました（{:.0f} 秒）'.format(k, time.time() - t0))
        if BIAS_CHECK:
            # 全深さで基準点を入れ替えた地図（偏りの検証用）
            other = {s: [e for e in EVENTS if e != s][0] for s in EVENTS}
            _, _, _, _, valt = compute_map(ctx, k, lambda z, k=k: other[source_for(ctx, k, z)])
            d = DB20 * np.log(valt / vmin)
            d = np.where(np.isfinite(d), d, np.nan)
            m = np.isfinite(d)
            out['bias'][k] = {'vmin_alt': valt, 'diff_db': d, 'n': int(m.sum()),
                              'mean_db': float(np.nanmean(d)) if m.any() else np.nan,
                              'std_db': float(np.nanstd(d)) if m.any() else np.nan}
            print('  新案 A: {} の入れ替え地図を計算しました（{:.0f} 秒）'.format(k, time.time() - t0))
    out['vcap'] = np.array([pore_capacity(ctx['rec_T'], z) for z in A_Z_GRID])

    # 検証：基準点どうしの相互予測（① → ② の位置、② → ① の位置）
    C, z0 = ctx['C'], ctx['C']['z0']
    out['cross'] = {}
    for k in EVENTS:
        s = [e for e in EVENTS if e != k][0]
        pdx = np.array([pd_point(ctx, k, v, z0, s)[0] for v in C['v']])
        vx = crossing(C['v'], pdx)
        vc = C['anchor'][k]['vmin_C']
        diff = DB20 * np.log(vx / vc) if np.isfinite(vx) and np.isfinite(vc) else np.nan
        # 差のばらつき σ_Δ：入れ替え地図があればその差の標準偏差（直接の測定）、
        # なければ点ごとのばらつき σ_pt の √2 倍（独立な 2 つの推定値の差）
        if BIAS_CHECK and np.isfinite(out['bias'][k]['std_db']):
            sig, how = out['bias'][k]['std_db'], '入れ替え地図の差の標準偏差'
        else:
            sig, how = np.sqrt(2.0) * out['sigma_pt'][k], '√2 × σ_pt'
        out['cross'][k] = {'source': s, 'pd': pdx, 'vmin': vx, 'diff_db': diff,
                           'sigma_db': float(sig), 'sigma_how': how}
    # 自己整合：同じ基準点で z = z0 に置いたもの（案 C と同じはず）
    out['self'] = {}
    for k in EVENTS:
        pds = np.array([pd_point(ctx, k, v, z0, k)[0] for v in C['v']])
        out['self'][k] = {'pd': pds, 'max_abs_diff': float(np.nanmax(np.abs(pds - C['pd'][k])))}
    return out


# =============================================================================
# 6. 案 A′（参考線）：信号の中央値 vs しきい値 ＋ 較正 ΔM
# =============================================================================
class PoresFull(Exception):
    """pore ice で空隙を満たせる上限を超えた。"""


def margin_db(ctx, k, v, z_top, dM):
    """M′(v, z) = 20 log10(A_sim · F(v, z)/F(v0, z0) / A_thr(t_k)) + ΔM と、その材料。"""
    rec_T, src, anc = ctx['rec_T'], ctx['src_delay'], ctx['C']['anchor'][k]
    if v > pore_capacity(rec_T, z_top) + 1e-12:
        raise PoresFull
    r = model_rec(rec_T, v, z_top)
    F = factors(r, k)
    t_k = event_time(r, k, src)
    A = anc['A_sim'] * F['F'] / anc['F0']['F']
    thr = float(np.interp(t_k, ctx['thr_t'], ctx['thr']))
    M = DB20 * np.log(A / thr) + dM if time_ok(ctx, t_k) else np.nan
    return M, t_k, A, thr, F


def vmin_fixpoint(ctx, k, z_top, dM):
    v = min(ctx['C']['v0'], 0.5 * pore_capacity(ctx['rec_T'], z_top))
    for _ in range(A_FIXPOINT_ITERS):
        try:
            M = margin_db(ctx, k, v, z_top, dM)[0]
        except PoresFull:
            return np.inf
        if not np.isfinite(M):
            return np.nan
        v = v * 10.0 ** (-M / 20.0)
        if v > pore_capacity(ctx['rec_T'], z_top):
            return np.inf
    return float(v)


def case_a_prime(ctx):
    """参考線用：ΔM と v_min′(z)、figA0 の材料（各因子の深さ変化）。"""
    C, rec_T = ctx['C'], ctx['rec_T']
    v0, z0 = C['v0'], C['z0']
    dMk = {}
    for k in EVENTS:
        va, vc = vmin_fixpoint(ctx, k, z0, 0.0), C['anchor'][k]['vmin_C']
        dMk[k] = DB20 * np.log(va / vc) if np.isfinite(va) and np.isfinite(vc) else np.nan
    dm = [d for d in dMk.values() if np.isfinite(d)]
    dM = float(np.mean(dm)) if dm else 0.0
    out = {'dM': dM, 'dM_each': dMk, 'vmin': {}, 'ing': {}}
    for k in EVENTS:
        out['vmin'][k] = (np.array([vmin_fixpoint(ctx, k, z, dM) for z in A_Z_GRID])
                          if SHOW_A_PRIME_REF else None)
        ing = {key: [] for key in ('A', 'thr', 't', 'ok', 'Gamma', 'G', 'Tr', 'L')}
        for z in A_Z_GRID:
            vv = min(v0, 0.5 * pore_capacity(rec_T, z))
            M, t_k, A, thr, F = margin_db(ctx, k, vv, z, 0.0)
            A = A * v0 / vv                      # 浅くて v0 が入らない場合は比例で戻す
            for key, val in (('A', A), ('thr', thr), ('t', t_k), ('ok', np.isfinite(M)),
                             ('Gamma', F['Gamma']), ('G', F['G']), ('Tr', F['Tr']), ('L', F['L'])):
                ing[key].append(val)
        out['ing'][k] = {key: np.array(val) for key, val in ing.items()}
    return out


def depth_axis(ctx, k):
    """界面 k の地表からの深さ d_k（上端は z、下端は z + D）を A_Z_GRID に対応させて返す。"""
    return A_Z_GRID + (0.0 if k == 'ice_top' else float(ctx['rec_T']['ice_thick_m']))


DEPTH_LABEL = 'Depth below surface [m]'


# =============================================================================
# 7. 作図（タイトル・軸ラベル 18、目盛り・凡例・図中の文字 14、グリッドは破線 alpha 0.3）
# =============================================================================
def plot_cross(ctx, out_dir):
    """figV1：基準点での P_D と濃度。実線＝案 C、破線＝もう一方の基準点からの予測、★＝gprMax。"""
    C, A = ctx['C'], ctx['A']
    fig, ax = plt.subplots(figsize=(11, 6), layout='constrained')
    for k in EVENTS:
        col, anc, cr = EV_COL[k], C['anchor'][k], A['cross'][k]
        ax.plot(100 * C['v'], C['pd'][k], color=col, lw=2.0, marker='o', ms=4,
                label='{} {:.1f} m: C'.format(k, anc['depth_m']))
        ax.plot(100 * C['v'], cr['pd'], color=col, lw=1.8, ls='--',
                label='{} {:.1f} m: from {}'.format(k, anc['depth_m'], cr['source']))
        ax.plot(100 * C['v0'], anc['pd_T'], marker='*', ms=16, color=col, mec='k', ls='none')
    ax.axhline(0.5, color='0.4', lw=1.0, ls='-.')
    ax.axhline(bd.P_FA, color='0.6', lw=0.8, ls=':')
    ax.set_xlim(0, 100 * C['v'].max())
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('Ice volume fraction [vol%]')
    ax.set_ylabel('P_D (reflection)')
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], marker='*', ms=14, color='0.5', mec='k', ls='none'))
    l.append('gprMax T')
    ax.legend(h, l, loc='lower right')
    bd.save_figure(fig, out_dir, 'figV1_cross_check')


def plot_ingredients(ctx, out_dir):
    """figA0：振幅の深さ変化の材料。(a) 10 vol% の信号としきい値、(b)(c) 因子ごとの寄与。"""
    Ap, a = ctx['Ap'], ctx['a_surf']
    fig, axs = plt.subplots(3, 1, figsize=(11, 12), sharex=True, layout='constrained')
    for k in EVENTS:
        ing, col, z = Ap['ing'][k], EV_COL[k], depth_axis(ctx, k)
        ok = ing['ok'].astype(bool)
        axs[0].plot(z[ok], DB20 * np.log(ing['A'][ok] / a), color=col, lw=2.0,
                    label='{}: signal'.format(k))
        axs[0].plot(z[ok], DB20 * np.log(ing['thr'][ok] / a), color=col, lw=1.6, ls='--',
                    label='{}: threshold'.format(k))
    axs[0].set_ylabel(bd.peak_label('dB re surface'))
    axs[0].set_title('(a) signal at {:.0f} vol% and threshold'.format(100 * ctx['C']['v0']))
    axs[0].legend(loc='upper right', ncol=2)
    styles = {'Gamma': '-', 'G': '--', 'Tr': ':', 'L': '-.'}
    names = {'Gamma': '|Γ|', 'G': 'G', 'Tr': 'Tr', 'L': 'L'}
    for ax, k, lab in ((axs[1], 'ice_top', '(b)'), (axs[2], 'ice_bottom', '(c)')):
        ing, F0, z = Ap['ing'][k], ctx['C']['anchor'][k]['F0'], depth_axis(ctx, k)
        ok = ing['ok'].astype(bool)
        for key, ls in styles.items():
            ax.plot(z[ok], DB20 * np.log(ing[key][ok] / F0[key]), color=EV_COL[k], ls=ls,
                    lw=1.8, label=names[key])
        tot = ing['A'][ok] / ctx['C']['anchor'][k]['A_sim']
        ax.plot(z[ok], DB20 * np.log(tot), color='k', lw=2.2, label='g (product)')
        ax.axhline(0.0, color='0.5', lw=0.8)
        ax.axvline(ctx['C']['anchor'][k]['depth_m'], color='0.5', lw=0.8, ls=':')
        ax.set_ylabel('dB re anchor')
        ax.set_title('{} {}'.format(lab, k))
        ax.legend(loc='lower left', ncol=5)
    axs[2].set_xlabel(DEPTH_LABEL)
    bd.save_figure(fig, out_dir, 'figA0_ingredients')


def plot_map(ctx, out_dir):
    """figA1：P_D(v, z) の地図。黒の細線＝P_D = 0.5（各深さの v_min）、黒の太線＝その移動平均、
    黒の点線＝帯の端（移動平均 ± CROSS_K_SIGMA · σ_pt。figA2 の帯と同じ）、灰破線＝空隙を満たせる上限。"""
    A, C = ctx['A'], ctx['C']
    v = 100 * A_V_GRID
    dv, dz = v[1] - v[0], A_Z_GRID[1] - A_Z_GRID[0]
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('0.85')
    # 縦軸の範囲：どちらかの界面で値がある深さ
    rows = [depth_axis(ctx, k)[np.isfinite(A['pd'][k]).any(axis=1)] for k in EVENTS]
    rows = np.concatenate(rows) if any(r.size for r in rows) else A_Z_GRID
    y0, y1 = float(rows.min()) - dz, float(rows.max()) + dz
    fig, axs = plt.subplots(1, 2, figsize=(16, 8.2), sharey=True, layout='constrained')
    for ax, k, lab in zip(axs, EVENTS, ('(a)', '(b)')):
        P = A['pd'][k]
        z = depth_axis(ctx, k)
        ax.set_facecolor('0.85')               # 格子の外（その界面が存在しない深さ）も灰色
        ext = [v[0] - dv / 2, v[-1] + dv / 2, z[-1] + dz / 2, z[0] - dz / 2]
        im = ax.imshow(np.ma.masked_invalid(P), aspect='auto', extent=ext, cmap=cmap,
                       vmin=0.0, vmax=1.0, interpolation='nearest')
        if np.isfinite(P).any():
            ax.contour(v, z, np.ma.masked_invalid(P), levels=[0.5], colors='k', linewidths=1.0)
        vs, sig = A['vmin_s'][k], A['sigma_pt'][k]
        ok = np.isfinite(vs)
        if ok.any():
            ax.plot(100 * vs, z, color='k', lw=2.6)
            if np.isfinite(sig):
                f = 10.0 ** (CROSS_K_SIGMA * sig / 20.0)
                for g in (1.0 / f, f):
                    ax.plot(100 * vs * g, z, color='k', lw=1.5, ls=':')
        ax.plot(100 * A['vcap'], z, color='0.35', lw=1.6, ls='--')
        anc = C['anchor'][k]
        if np.isfinite(anc['vmin_C']):
            ax.plot(100 * anc['vmin_C'], anc['depth_m'], marker='*', ms=16, color='gold', mec='k',
                    ls='none')
        ax.plot(100 * C['v0'], anc['depth_m'], marker='o', ms=11, color='white', mec='k', mew=1.8,
                ls='none')
        ax.set_xlim(ext[0], ext[1])
        ax.set_title('{} {}'.format(lab, k))
        ax.set_xlabel('Ice volume fraction [vol%]')
    axs[0].set_ylim(y1, y0)
    axs[0].set_ylabel(DEPTH_LABEL)
    fig.colorbar(im, ax=axs, label='P_D (reflection)', shrink=0.9, pad=0.01)
    fig.legend(handles=[Line2D([], [], color='k', lw=1.0, label='P_D = 0.5'),
                           Line2D([], [], color='k', lw=2.6, label='moving average'),
                           Line2D([], [], color='k', lw=1.5, ls=':',
                                  label='±{:.0f}σ'.format(CROSS_K_SIGMA)),
                           Line2D([], [], color='0.35', lw=1.6, ls='--', label='pores full'),
                           Line2D([], [], marker='*', ms=14, color='gold', mec='k', ls='none',
                                  label='case C: v_min'),
                           Line2D([], [], marker='o', ms=10, color='white', mec='k', mew=1.8,
                                  ls='none', label='gprMax ({:.0f} vol%)'.format(100 * C['v0']))],
                  loc='outside lower center', ncol=6)   # 凡例は地図の外（色と重ねない）
    bd.save_figure(fig, out_dir, 'figA1_pd_map')


def plot_vmin(ctx, out_dir):
    """figA2：検出できる最小濃度 v_min（P_D = 0.5）と界面の深さ。
    細線＋点＝各深さの値、太線＝移動平均、帯＝± CROSS_K_SIGMA · σ_pt（各深さの値のばらつき）。"""
    A, C, Ap = ctx['A'], ctx['C'], ctx['Ap']
    fig, ax = plt.subplots(figsize=(13, 8), layout='constrained')
    fin, ys = [], []
    for k in EVENTS:
        vm, z = A['vmin'][k], depth_axis(ctx, k)
        vs, sig = A['vmin_s'][k], A['sigma_pt'][k]
        ys.append(z[np.isfinite(vm)])
        ax.plot(100 * vm, z, color=EV_COL[k], lw=0.8, alpha=0.6, marker='.', ms=4)
        ok = np.isfinite(vs)
        if np.isfinite(sig) and ok.any():
            f = 10.0 ** (CROSS_K_SIGMA * sig / 20.0)
            ax.fill_betweenx(z[ok], 100 * vs[ok] / f, 100 * vs[ok] * f, color=EV_COL[k],
                             alpha=0.2, lw=0)
            fin.append(vs[ok] * f)
        ax.plot(100 * vs, z, color=EV_COL[k], lw=2.4,
                label='{}: A (±{:.0f}σ = ±{:.1f} dB)'.format(k, CROSS_K_SIGMA, CROSS_K_SIGMA * sig))
        fin.append(vm[np.isfinite(vm)])
        if SHOW_A_PRIME_REF and Ap['vmin'][k] is not None:
            vp = np.where(np.isfinite(Ap['vmin'][k]), Ap['vmin'][k], np.nan)
            ax.plot(100 * vp, z, color=EV_COL[k], lw=1.2, ls='--', label="{}: A' (ref)".format(k))
        anc = C['anchor'][k]
        if np.isfinite(anc['vmin_C']):
            ax.plot(100 * anc['vmin_C'], anc['depth_m'], marker='*', ms=16, color=EV_COL[k],
                    mec='k', ls='none', label='{}: C'.format(k))
            fin.append(np.array([anc['vmin_C']]))
        ax.plot(100 * C['v0'], anc['depth_m'], marker='o', ms=11, color='white', mec='k',
                mew=1.8, ls='none', label='gprMax ({:.0f} vol%)'.format(100 * C['v0'])
                if k == EVENTS[0] else None)
    ys = np.concatenate(ys) if any(y.size for y in ys) else A_Z_GRID
    dz = A_Z_GRID[1] - A_Z_GRID[0]
    ax.set_ylim(float(ys.max()) + dz, float(ys.min()) - dz)
    fin = np.concatenate(fin) if fin else np.array([])
    vcapmax = float(np.nanmax(A['vcap']))
    right = 100 * max(0.05, 1.4 * fin.max()) if fin.size else 100 * vcapmax
    right = min(right, 100 * vcapmax * 1.05)
    if 100 * float(np.nanmin(A['vcap'])) <= right:     # 空隙の上限は、図の範囲に入るときだけ描く
        for k in EVENTS:
            ax.plot(100 * A['vcap'], depth_axis(ctx, k), color=EV_COL[k], lw=1.2, ls=':',
                    label='pores full ({})'.format(k))
    ax.set_xlim(0, right)
    ax.set_xlabel('Minimum detectable ice [vol%]')
    ax.set_ylabel(DEPTH_LABEL)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))   # 凡例は枠の外（線と重ねない）
    bd.save_figure(fig, out_dir, 'figA2_vmin_curve')


# =============================================================================
# 8. 出力
# =============================================================================
def _write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: ('{:.6g}'.format(v) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print('Saved:', path)


def _fmt_v(v):
    return '{:.1f}%'.format(100 * v) if np.isfinite(v) else ('満杯でも不可' if np.isinf(v) else '—')


def write_outputs(ctx, out_dir):
    C, A, Ap = ctx['C'], ctx['A'], ctx['Ap']
    _write_csv(os.path.join(out_dir, 'cross_check.csv'),
               [dict(v=float(v),
                     **{'{}_pd_C'.format(k): float(C['pd'][k][i]) for k in EVENTS},
                     **{'{}_pd_from_{}'.format(k, A['cross'][k]['source']): float(A['cross'][k]['pd'][i])
                        for k in EVENTS},
                     **{'{}_pd_A_self'.format(k): float(A['self'][k]['pd'][i]) for k in EVENTS})
                for i, v in enumerate(C['v'])])
    _write_csv(os.path.join(out_dir, 'newA_vmin.csv'),
               [dict(ice_top_depth_m=float(z), pores_full=float(A['vcap'][i]),
                     **{'{}_surface_depth_m'.format(k): float(depth_axis(ctx, k)[i]) for k in EVENTS},
                     **{'{}_vmin'.format(k): float(A['vmin'][k][i]) for k in EVENTS},
                     **{'{}_vmin_smoothed'.format(k): float(A['vmin_s'][k][i]) for k in EVENTS},
                     **{'{}_source'.format(k): A['source'][k][i] for k in EVENTS},
                     **({'{}_vmin_other_anchor'.format(k): float(A['bias'][k]['vmin_alt'][i])
                         for k in EVENTS} if BIAS_CHECK else {}),
                     **{'{}_t_ns'.format(k): float(A['t'][k][i]) for k in EVENTS},
                     **({'{}_vmin_Aprime_ref'.format(k): float(Ap['vmin'][k][i]) for k in EVENTS}
                        if SHOW_A_PRIME_REF else {}))
                for i, z in enumerate(A_Z_GRID)])
    _write_csv(os.path.join(out_dir, 'newA_pd_map.csv'),
               [dict(ice_top_depth_m=float(z), v=float(v),
                     **{'{}_surface_depth_m'.format(k): float(depth_axis(ctx, k)[i]) for k in EVENTS},
                     **{'{}_pd'.format(k): float(A['pd'][k][i, j]) for k in EVENTS},
                     **{'{}_gain'.format(k): float(A['gain'][k][i, j]) for k in EVENTS})
                for i, z in enumerate(A_Z_GRID) for j, v in enumerate(A_V_GRID)])

    L = ['detectability_map（新案 A）  {}'.format(datetime.now().isoformat(timespec='seconds')), '']
    L += ['[入力]'] + ['  {}: {}'.format(k, v) for k, v in ctx['paths'].items()]
    L += ['  ランダム場の同一性: {}'.format('一致' if ctx['sha_match'] else '不一致（案 C は参考）'),
          '  波源遅延 {:.3f} ns、A_surf = {:.4g}'.format(ctx['src_delay'], ctx['a_surf']), '']
    rec = ctx['rec_T']
    L += ['[基準点]  {}、{:.1f} vol%、上端 {:.2f} m、厚さ {:.2f} m'.format(
        rec['ice_model'], 100 * C['v0'], C['z0'], float(rec['ice_thick_m']))]
    for k in EVENTS:
        an, F0 = C['anchor'][k], C['anchor'][k]['F0']
        L.append('  {} {:<10} 深さ {:.2f} m、走時 {:.2f} ns、σ_a {:.2f} dB、'
                 '因子 |Γ| {:.5f}・G {:.4f}・Tr {:.4f}・L {:+.2f} dB'.format(
                     EV_ANCHOR_NAME[k], k, an['depth_m'], an['t_ns'], an['sigma_db'],
                     F0['Gamma'], F0['G'], F0['Tr'], DB20 * np.log(F0['L'])))
    L += ['', '[案 C：計算結果を濃度方向に伸ばす]']
    for k in EVENTS:
        an = C['anchor'][k]
        L.append('  {:<10} P_D：gprMax の T {:.3f} / 合成（v0）{:.3f}　→　v_min,C = {}'.format(
            k, an['pd_T'], an['pd_C_at_v0'], _fmt_v(an['vmin_C'])))
    kk = CROSS_K_SIGMA
    L += ['', '[新案 A の検証]',
          '  v_min の推定値は、1 回分のランダム場のクラッタによって深さごとに偶然上下する。',
          '  そこで (1) 基準点による違い と (2) 相互予測の差 を、データから見積もった深さごとの',
          '  ばらつきと比べて判定する（目安：{:.0f}σ 以内なら整合）。'.format(kk)]
    L.append('  点ごとのばらつき σ_pt（v_min(z) の移動平均 ±{:.2f} m からのずれの標準偏差）：'.format(
        VMIN_SMOOTH_HW_M) + '、'.join('{} {:.2f} dB'.format(k, A['sigma_pt'][k]) for k in EVENTS))
    ok_all = True
    if BIAS_CHECK:
        L.append('  (1) 基準点による違い：全深さで基準点を入れ替えたときの v_min の差 d(z) = 20 log10(入れ替え後 / 元)')
        L.append('      平均 b は「どちらの基準点を使うか」で地図全体がずれる量。帯の半幅 {:.0f}σ_pt と比べる。'.format(kk))
        for k in EVENTS:
            b = A['bias'][k]
            ok = np.isfinite(b['mean_db']) and abs(b['mean_db']) <= kk * A['sigma_pt'][k]
            ok_all &= ok
            L.append('      {:<10} b = {:+.2f} dB（点数 {}）、d の標準偏差 {:.2f} dB'
                     '  → {}'.format(
                         k, b['mean_db'], b['n'], b['std_db'],
                         '帯の中（|b| ≤ {:.1f} dB）'.format(kk * A['sigma_pt'][k]) if ok else
                         '帯より大きい（|b| > {:.1f} dB）'.format(kk * A['sigma_pt'][k])))
        L.append('      ※ 基準点は 1 回分の計算なので、b が偶然（その基準点の応答のくせ）か、移し替えの式の'
                 '偏りかは、2 点だけでは分けられない。')
    else:
        L.append('  (1) 系統的な偏り：BIAS_CHECK = False のため未評価')
    L.append('  (2) 相互予測：もう一方の基準点から案 C の v_min を予測（差 Δ = 20 log10(予測 / 案 C)）')
    for k in EVENTS:
        cr = A['cross'][k]
        ok = np.isfinite(cr['diff_db']) and abs(cr['diff_db']) <= kk * cr['sigma_db']
        ok_all &= ok
        L.append('      {:<10} {} から予測 {} / 案 C {}：Δ = {:+.2f} dB、σ_Δ = {:.2f} dB（{}）'
                 '、|Δ| = {:.1f}σ  → {}'.format(
                     k, cr['source'], _fmt_v(cr['vmin']), _fmt_v(C['anchor'][k]['vmin_C']),
                     cr['diff_db'], cr['sigma_db'], cr['sigma_how'],
                     abs(cr['diff_db']) / cr['sigma_db'] if cr['sigma_db'] > 0 else np.nan,
                     '整合' if ok else '不整合'))
    if ok_all:
        L.append('  判定：整合（基準点による違いも相互予測の差も、深さごとのばらつきの範囲内。'
                 'v_min(z) は移動平均 ± {:.0f}σ_pt の帯で読む）'
                 .format(kk))
    else:
        L.append('  判定：帯を超える違いあり。基準点の選び方で v_min が帯より大きくずれる、または相互予測が外れる。')
        L.append('        v_min(z) は帯に加えて、b の大きさも不確かさとして読む（figA2 の帯には含めていない）。')
    L.append('  ※ (1)(2) は A_SOURCE の選び方によらない（どちらの基準点も両方の向きで試すため）。')
    L.append('  自己整合（同じ基準点を z0 に置いた P_D と案 C の差の最大）：' + '、'.join(
        '{} {:.3f}'.format(k, A['self'][k]['max_abs_diff']) for k in EVENTS))
    L += ['', '[新案 A：v_min（P_D = 0.5）と界面の地表からの深さ]  A_SOURCE = {}'.format(A_SOURCE)]
    for k in EVENTS:
        d = depth_axis(ctx, k)
        L.append('  {:<10} '.format(k) + '、'.join(
            '{:.1f} m: {}'.format(dd, _fmt_v(v)) for dd, v in zip(d[::10], A['vmin'][k][::10])))
        f = 10.0 ** (kk * A['sigma_pt'][k] / 20.0) if np.isfinite(A['sigma_pt'][k]) else np.nan
        L.append('  {:<10} 移動平均（±{:.0f}σ_pt = ×/÷ {:.2f}）：'.format('', kk, f) + '、'.join(
            '{:.1f} m: {}'.format(dd, _fmt_v(v)) for dd, v in zip(d[::10], A['vmin_s'][k][::10])))
        if SHOW_A_PRIME_REF:
            L.append("  {:<10} A' 参考（ΔM {:+.2f} dB）：".format('', Ap['dM']) + '、'.join(
                '{:.1f} m: {}'.format(dd, _fmt_v(v)) for dd, v in zip(d[::10], Ap['vmin'][k][::10])))
    L.append('  空隙満杯（氷層の上端の深さ）：' + '、'.join(
        '{:.1f} m: {}'.format(z, _fmt_v(v)) for z, v in zip(A_Z_GRID[::10], A['vcap'][::10])))
    L.append('  （— は走時が使えない深さ、または P_D が 0.5 に届かない）')
    path = os.path.join(out_dir, 'run_info.txt')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print('Saved:', path)
    print('\n'.join(L))


# =============================================================================
# 9. 本体
# =============================================================================
def main():
    ctx = load_inputs()
    R, dt, t, src = ctx['R'], ctx['dt'], ctx['t'], ctx['src_delay']
    # dB の基準（基準 B-scan の地表反射）と、使える走時の範囲
    t_s_model = 2.0 * sm.TX_HEIGHT / sm.C + src
    envR = bd.envelope(R)
    ms = np.abs(t - t_s_model) <= 5.0
    ctx['t_surface'] = float(t[ms][np.argmax(envR[ms].mean(axis=1))])
    mg = np.abs(t - ctx['t_surface']) <= bd.REFL_GATE_HW_NS
    ctx['a_surf'] = float(np.median(np.abs(R)[mg].max(axis=0)))   # 判定と同じ量（±1 ns の |E| の最大値）
    ctx['t_dom'] = bd.domain_bottom_time(ctx['rec_T'], src)
    rsR = bd.reflection_scan(R, R, dt)              # しきい値 A_thr(t)（水氷なし全体から）
    ctx['thr_t'], ctx['thr'] = rsR['t'], rsR['th']
    ctx['thr_idx'] = np.rint(rsR['t'] / (dt * 1e9)).astype(int)

    print('\n=== 案 C ===')
    ctx['C'] = case_c(ctx)
    print('\n=== 新案 A ===')
    ctx['A'] = case_new_a(ctx)
    print('\n=== 案 A′（参考線・材料） ===')
    ctx['Ap'] = case_a_prime(ctx)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(ctx['paths']['target'])),
                           bd.OUTPUT_SUBDIRNAME, OUTPUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    plot_cross(ctx, out_dir)
    plot_ingredients(ctx, out_dir)
    plot_map(ctx, out_dir)
    plot_vmin(ctx, out_dir)
    write_outputs(ctx, out_dir)


if __name__ == '__main__':
    main()