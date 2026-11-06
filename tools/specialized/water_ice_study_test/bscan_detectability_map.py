"""反射で検出できる水氷の最小濃度 v_min（案 C → 案 A）

案 C（計算結果をそのまま濃度方向に伸ばす）
    氷の応答 ΔT = T − W[R] を界面の走時のまわりで切り出し、v / v0 倍して R に足す。
        T(v) = R + Σ_k (v / v0) · ΔT_k(t − Δt_k(v))
    これを bscan_detection と同じ反射チャネルにかけて P_D(v) を出し、P_D = 0.5 の v を v_min とする。
    仮定：氷の応答（トレースごとのゆらぎも含む）は v に比例する。値が出るのは計算した深さだけ。

案 A（比例でずらす）
    案 C の基準点（v0, z0）での振幅 A_sim を、帯域中心での 3 つ（＋透過）の比で動かす。
        A(v, z) = A_sim · [ |R| G |T| L ](v, z) / [ |R| G |T| L ](v0, z0)
        M(v, z) = 20 log10( A(v, z) / A_thr(t(v, z)) )          しきい値に対する余裕
        v_min(z)：M = 0（信号の中央値 ＝ しきい値 ⇔ P_D ≈ 0.5）
    仮定：振幅は v に比例する、深さ方向の変化は平均媒質の伝搬の式で決まる、
          P_D = 0.5 の境界は信号の中央値 ＝ しきい値。

先に案 C を出し、1 m・2 m で案 A と比べてから案 A の曲線と地図を使う。
式の説明は bscan_detectability_map.md。

出力：<対象の merged.out のフォルダ>/bscan_detection/detectability_map/（PDF は pdf/ の下）
"""

import copy
import csv
import os
from datetime import datetime

import numpy as np
from scipy import fft as sp_fft
from scipy.special import ndtr, ndtri
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import bscan_detection as bd               # 入力・反射チャネル・平均媒質（rcParams もここで設定）
import bscan_superposition_check as bsc    # 時間伸縮 W[R]
import subsurface_model as sm
from ascan_spectrum import load_trace, measure_peak
from gprMax.exceptions import CmdInputError

# 図の共通ルール（このファイルの中でも明示する。bscan_detection の版に依存しないように）
#   文字の大きさ：タイトル・軸ラベル 16、目盛り・凡例・図中の文字 12
#   グリッド線：特に意図がない限り全パネルに付ける（破線 '--'、alpha 0.3）
plt.rcParams.update({'axes.titlesize': 16, 'axes.labelsize': 16, 'figure.titlesize': 16,
                     'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
                     'font.size': 12,
                     'axes.grid': True, 'axes.grid.which': 'major', 'axes.axisbelow': 'line',
                     'grid.linestyle': '--', 'grid.alpha': 0.3})

# =============================================================================
# 設定  [EDIT HERE]
# =============================================================================
JSON_PATH = '/Volumes/gprMax_water_ice/2D_12x4/water_ice_study_test/out_file_paths.json'
TARGET_CONDITION = None            # 氷ありの条件名。None なら基準以外で最初のもの
OUTPUT_SUBDIR = 'detectability_map'
EVENTS = ('ice_top', 'ice_bottom')

# --- 案 C -----------------------------------------------------------------------
C_V_GRID = np.round(np.arange(0.0, 0.4001, 0.01), 4)   # 合成する濃度（体積分率）
C_GATE_FLAT_NS = 2.5               # 氷の応答を切り出す範囲（界面の走時 ± これ）。反射チャネルの
                                   # ゲート ±1 ns と包絡の最大値フィルタ ±1 ns を覆う幅
C_GATE_TAPER_NS = 0.5              # その外側のコサインテーパの幅
C_WORK_MARGIN_NS = 10.0            # 反射チャネルを計算する時間範囲の余白（計算を軽くするため）
AMP_HW_NS = 1.5                    # A_sim（トレースごとの包絡ピーク）を取るゲートの半幅

# --- 案 A -----------------------------------------------------------------------
A_V_GRID = np.linspace(0.005, 0.40, 80)                 # 地図の横軸（体積分率）
A_Z_GRID = np.round(np.arange(0.20, 2.95 + 1e-9, 0.05), 3)  # 地図の縦軸：氷層の上端の深さ [m]
A_FIXPOINT_ITERS = 8               # v_min を求める反復の回数
A_CALIBRATE = True                 # 案 A′：余裕 M に定数 ΔM を足す（ΔM は基準点で案 C に合わせて決める）
A_PD_EXTRA = ()                    # 例 (0.9,)：P_D = 0.5 以外の等高線（σ_a の深さ内挿を使う）
SURFACE_SKIP_NS = 5.0              # 地表反射の後ろこの範囲は使わない（bscan_detection と同じ考え方）
EDGE_GUARD_NS = 1.0                # 計算領域の底からの反射の手前この範囲は使わない

# --- 作図 -----------------------------------------------------------------------
MAP_DB_RANGE = 20.0                # 地図の色の範囲 ±[dB]
EV_COL = {'ice_top': 'tab:orange', 'ice_bottom': 'tab:green'}   # bscan_detection と同じ
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
# 2. 平均媒質の式（案 A の 3 つの比と走時）
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
    """帯域中心での |R|、G、|T|、L（＝往復の吸収）と、その積 F。"""
    bd.configure_mean_medium(rec)
    lvl = bd.THEORY_LEVEL
    f = np.array([sm.BAND_CENTRE_HZ])
    d = iface_depth(rec, name)
    att, _, inv = sm.path_integrals(f, d, lvl)
    G = float(np.sqrt(sm.R_REF / (2.0 * sm.TX_HEIGHT + 2.0 * inv[0])))
    Tr = float(np.abs(sm.transmission_product(f, d, lvl, two_way=True, include_surface=True)[0]))
    Rc = float(np.abs(sm.interface_reflection(f, d, lvl)[0]))
    L = float(np.exp(-2.0 * att[0]))
    return {'R': Rc, 'G': G, 'T': Tr, 'L': L, 'F': Rc * G * Tr * L}


def pore_capacity(rec, z_top):
    """pore ice で空隙を満たせる上限 = 氷層の底での平均の空隙率 1 − ρ/ρ_grain
    （subsurface_model が pore の濃度を受け付ける上限と同じ。excess には上限なし）。"""
    if rec['ice_model'] != 'pore':
        return np.inf
    zb = z_top + float(rec['ice_thick_m'])
    return float(1.0 - sm.density_profile(zb) / sm.LEVEL4_RHO_GRAIN)


# =============================================================================
# 3. 反射チャネル（bscan_detection と同じ定義）
# =============================================================================
def pd_at(rs, t_ev):
    """界面の走時 ± REFL_GATE_HW_NS での P_D の最大（bscan_superposition_check と同じ）。"""
    m = np.abs(rs['t'] - t_ev) <= bd.REFL_GATE_HW_NS
    return float(rs['pd'][m].max()) if m.any() else np.nan


def reflection_scan_window(T, R, dt, t, t0, t1):
    """反射チャネルを t0–t1 の範囲だけで計算する（端の影響は余白で避ける）。"""
    i0 = max(0, int(np.floor(t0 / (dt * 1e9))))
    i1 = min(len(t), int(np.ceil(t1 / (dt * 1e9))))
    rs = bd.reflection_scan(T[i0:i1], R[i0:i1], dt)
    rs['t'] = rs['t'] + t[i0]
    return rs


def threshold_curve(R, dt):
    """しきい値 A_thr(t)：水氷なしのゲート内包絡ピークの 95 パーセンタイル。"""
    rs = bd.reflection_scan(R, R, dt)
    return rs['t'], rs['th']


def gate(t, tc):
    """平らな部分 ±C_GATE_FLAT_NS、その外側にコサインテーパ。"""
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
    """P_D(v) が初めて level を下から越える v（線形内挿）。越えなければ nan。"""
    for i in range(1, len(v)):
        if pd[i - 1] < level <= pd[i]:
            return float(v[i - 1] + (level - pd[i - 1]) * (v[i] - v[i - 1]) / (pd[i] - pd[i - 1]))
    return float(v[0]) if pd[0] >= level else np.nan


# =============================================================================
# 4. 案 C
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

    # 基準点の振幅 A_sim と、トレースごとのばらつき σ_a（ともに包絡ピーク）
    anchor = {}
    for k in EVENTS:
        g = np.abs(t - t_ev0[k]) <= AMP_HW_NS
        pk = DB20 * np.log(np.maximum(bd.envelope(resp[k])[g].max(axis=0), 1e-300))
        anchor[k] = {'t_ns': t_ev0[k], 'A_sim': float(10.0 ** (np.median(pk) / 20.0)),
                     'sigma_db': float(pk.std()), 'depth_m': iface_depth(rec_T, k)}

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
        print('  案 C: v = {:5.1f} vol%  P_D = {}'.format(
            100 * v, ', '.join('{} {:.3f}'.format(k, pd[k][i]) for k in EVENTS)))

    # 整合の確認：v0 での合成の P_D と、gprMax の T そのものの P_D
    rsT = reflection_scan_window(T, R, dt, t, t0w, t1w)
    for k in EVENTS:
        anchor[k]['pd_T'] = pd_at(rsT, t_ev0[k])
        anchor[k]['pd_C_at_v0'] = float(np.interp(v0, vgrid, pd[k]))
        anchor[k]['vmin_C'] = crossing(vgrid, pd[k], 0.5)
    return {'v': vgrid, 'pd': pd, 't_ev': t_ev, 'anchor': anchor, 'v0': v0, 'z0': z0}


# =============================================================================
# 5. 案 A
# =============================================================================
class PoresFull(Exception):
    """pore ice で空隙を満たせる上限を超えた。"""


def margin_db(ctx, k, v, z_top):
    """M(v, z) [dB]（＋較正の ΔM）、走時、各因子。走時が使えない範囲は M = nan。"""
    rec_T, src, anc = ctx['rec_T'], ctx['src_delay'], ctx['C']['anchor'][k]
    if v > pore_capacity(rec_T, z_top) + 1e-12:
        raise PoresFull
    r = model_rec(rec_T, v, z_top)
    F = factors(r, k)
    t_k = event_time(r, k, src)
    A = anc['A_sim'] * F['F'] / ctx['F0'][k]['F']
    thr = float(np.interp(t_k, ctx['thr_t'], ctx['thr']))
    ok = (ctx['t_surface'] + SURFACE_SKIP_NS <= t_k <= ctx['t_dom'] - EDGE_GUARD_NS)
    M = DB20 * np.log(A / thr) + ctx.get('dM', 0.0) if ok else np.nan
    return M, t_k, A, thr, F


def vmin_fixpoint(ctx, k, z_top):
    """v ← v · A_thr(t(v)) / A(v) を繰り返す（A がほぼ v に比例するので数回で収束する）。"""
    v = min(ctx['C']['v0'], 0.5 * pore_capacity(ctx['rec_T'], z_top))
    for _ in range(A_FIXPOINT_ITERS):
        try:
            M = margin_db(ctx, k, v, z_top)[0]
        except PoresFull:
            return np.inf             # 空隙を満たしても届かない
        if not np.isfinite(M):
            return np.nan             # 走時が使えない範囲
        v = v * 10.0 ** (-M / 20.0)
        if v > pore_capacity(ctx['rec_T'], z_top):
            return np.inf
    return float(v)


def sigma_at(ctx, depth):
    """σ_a(界面の深さ)：2 つの基準点（上端・下端）を通る直線で内挿・外挿（A_PD_EXTRA にだけ使う）。"""
    a = ctx['C']['anchor']
    z1, z2 = (a[k]['depth_m'] for k in EVENTS)
    s1, s2 = (a[k]['sigma_db'] for k in EVENTS)
    return float(max(0.1, s1 + (s2 - s1) * (depth - z1) / (z2 - z1)))


def case_a(ctx):
    rec_T = ctx['rec_T']
    v0, z0 = ctx['C']['v0'], ctx['C']['z0']
    ctx['F0'] = {k: factors(model_rec(rec_T, v0, z0), k) for k in EVENTS}
    out = {'M': {}, 'vmin': {}, 'ing': {}, 'pd_anchor': {}, 'vmin_anchor': {},
           'pd_anchor_raw': {}, 'vmin_anchor_raw': {}, 'dM_each': {}}

    # 案 A（較正なし）の基準点での v_min と P_D(v)、案 C との差 ΔM_k = 20 log10(v_min,A / v_min,C)
    ctx['dM'] = 0.0
    for k in EVENTS:
        s = ctx['C']['anchor'][k]['sigma_db']
        Mv = np.array([margin_db(ctx, k, max(v, 1e-6), z0)[0] for v in ctx['C']['v']])
        out['pd_anchor_raw'][k] = ndtr(Mv / s)
        out['vmin_anchor_raw'][k] = vmin_fixpoint(ctx, k, z0)
        vc = ctx['C']['anchor'][k]['vmin_C']
        out['dM_each'][k] = (DB20 * np.log(out['vmin_anchor_raw'][k] / vc)
                             if np.isfinite(out['vmin_anchor_raw'][k]) and np.isfinite(vc) else np.nan)
    dm = [d for d in out['dM_each'].values() if np.isfinite(d)]
    ctx['dM'] = float(np.mean(dm)) if (A_CALIBRATE and dm) else 0.0
    out['dM'] = ctx['dM']
    print('  案 A: ΔM = {:+.2f} dB（基準点ごと {}）'.format(
        ctx['dM'], ', '.join('{} {:+.2f}'.format(k, d) for k, d in out['dM_each'].items())))

    for k in EVENTS:
        M = np.full((A_Z_GRID.size, A_V_GRID.size), np.nan)
        for i, z in enumerate(A_Z_GRID):
            for j, v in enumerate(A_V_GRID):
                try:
                    M[i, j] = margin_db(ctx, k, v, z)[0]
                except PoresFull:
                    pass              # 空隙を満たせる上限より右は空白（灰色）
        out['M'][k] = M
        out['vmin'][k] = np.array([vmin_fixpoint(ctx, k, z) for z in A_Z_GRID])
        ing = {key: [] for key in ('A', 'thr', 't', 'R', 'G', 'T', 'L')}
        for z in A_Z_GRID:
            _, t_k, A, thr, F = margin_db(ctx, k, min(v0, 0.5 * pore_capacity(rec_T, z)), z)
            A = A * v0 / min(v0, 0.5 * pore_capacity(rec_T, z))   # 浅すぎて入らない場合は比例で戻す
            for key, val in (('A', A), ('thr', thr), ('t', t_k), ('R', F['R']), ('G', F['G']),
                             ('T', F['T']), ('L', F['L'])):
                ing[key].append(val)
        out['ing'][k] = {key: np.array(val) for key, val in ing.items()}
        # 基準点での案 A の P_D(v) = Φ(M / σ_a)（σ_a は基準点で測った値）
        s = ctx['C']['anchor'][k]['sigma_db']
        Mv = np.array([margin_db(ctx, k, max(v, 1e-6), z0)[0] for v in ctx['C']['v']])
        out['pd_anchor'][k] = ndtr(Mv / s)
        out['vmin_anchor'][k] = vmin_fixpoint(ctx, k, z0)
        print('  案 A: {} の地図と v_min(z) を計算しました'.format(k))
    out['vcap'] = np.array([pore_capacity(rec_T, z) for z in A_Z_GRID])
    return out


# =============================================================================
# 6. 作図（タイトル・軸ラベル 16、目盛り・凡例・図中の文字 12 は bscan_detection の rcParams）
# =============================================================================
def plot_c(ctx, out_dir):
    """figC1：基準点での P_D と濃度。実線＝案 C、破線＝案 A′（Φ((M + ΔM)/σ_a)）、点線＝較正なしの案 A。"""
    C, A = ctx['C'], ctx['A']
    fig, ax = plt.subplots(figsize=(11, 6), layout='constrained')
    for k in EVENTS:
        col = EV_COL[k]
        anc = C['anchor'][k]
        ax.plot(100 * C['v'], C['pd'][k], color=col, lw=2.0, marker='o', ms=4,
                label='{} {:.1f} m: C'.format(k, anc['depth_m']))
        ax.plot(100 * C['v'], A['pd_anchor'][k], color=col, lw=1.8, ls='--',
                label='{} {:.1f} m: {}'.format(k, anc['depth_m'], _a_name()))
        if A['dM'] != 0.0:
            ax.plot(100 * C['v'], A['pd_anchor_raw'][k], color=col, lw=1.0, ls=':')
        ax.plot(100 * C['v0'], anc['pd_T'], marker='*', ms=16, color=col, mec='k', ls='none')
    ax.axhline(0.5, color='0.4', lw=1.0, ls='-.')
    ax.axhline(bd.P_FA, color='0.6', lw=0.8, ls=':')
    ax.set_xlim(0, 100 * C['v'].max())
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('Ice volume fraction [vol%]')
    ax.set_ylabel('P_D (reflection)')
    h, l = ax.get_legend_handles_labels()
    if A['dM'] != 0.0:
        h.append(Line2D([], [], color='0.4', lw=1.0, ls=':'))
        l.append('A (no ΔM)')
    h.append(Line2D([], [], marker='*', ms=14, color='0.5', mec='k', ls='none'))
    l.append('gprMax T')
    ax.legend(h, l, loc='lower right')
    bd.save_figure(fig, out_dir, 'figC1_pd_vs_concentration')


def _a_name():
    return "A'" if A_CALIBRATE else 'A'


def plot_ingredients(ctx, out_dir):
    """figA0：案 A の材料。(a) 信号 A(v0, z) としきい値、(b)(c) 因子ごとの寄与（基準点との比）。"""
    A, a = ctx['A'], ctx['a_surf']
    z = A_Z_GRID
    fig, axs = plt.subplots(3, 1, figsize=(11, 12), sharex=True, layout='constrained')
    for k in EVENTS:
        ing, col = A['ing'][k], EV_COL[k]
        ok = np.isfinite(A['M'][k][:, 0])
        axs[0].plot(z[ok], DB20 * np.log(ing['A'][ok] / a), color=col, lw=2.0,
                    label='{}: signal'.format(k))
        axs[0].plot(z[ok], DB20 * np.log(ing['thr'][ok] / a), color=col, lw=1.6, ls='--',
                    label='{}: threshold'.format(k))
    axs[0].set_ylabel('dB re A_surf')
    axs[0].set_title('(a) signal at {:.0f} vol% and threshold'.format(100 * ctx['C']['v0']))
    axs[0].legend(loc='upper right', ncol=2)
    styles = {'R': '-', 'G': '--', 'T': ':', 'L': '-.'}
    for ax, k, lab in ((axs[1], 'ice_top', '(b)'), (axs[2], 'ice_bottom', '(c)')):
        ing, F0 = A['ing'][k], ctx['F0'][k]
        ok = np.isfinite(A['M'][k][:, 0])
        for key, ls in styles.items():
            ax.plot(z[ok], DB20 * np.log(ing[key][ok] / F0[key]), color=EV_COL[k], ls=ls,
                    lw=1.8, label={'R': '|R|', 'G': 'G', 'T': '|T|', 'L': 'L'}[key])
        tot = ing['A'][ok] / ctx['C']['anchor'][k]['A_sim']
        ax.plot(z[ok], DB20 * np.log(tot), color='k', lw=2.2, label='product')
        ax.axhline(0.0, color='0.5', lw=0.8)
        ax.axvline(ctx['C']['z0'], color='0.5', lw=0.8, ls=':')
        ax.set_ylabel('dB re anchor')
        ax.set_title('{} {}'.format(lab, k))
        ax.legend(loc='lower left', ncol=5)
    axs[2].set_xlabel('Ice top depth [m]')
    bd.save_figure(fig, out_dir, 'figA0_ingredients')


def plot_map(ctx, out_dir):
    """figA1：余裕 M(v, z) の地図。黒線＝M = 0（P_D ≈ 0.5）、灰破線＝空隙を満たせる上限。"""
    A, C = ctx['A'], ctx['C']
    v, z = 100 * A_V_GRID, A_Z_GRID
    dv, dz = v[1] - v[0], z[1] - z[0]
    ext = [v[0] - dv / 2, v[-1] + dv / 2, z[-1] + dz / 2, z[0] - dz / 2]
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('0.85')
    fig, axs = plt.subplots(1, 2, figsize=(16, 7.5), sharey=True, layout='constrained')
    for ax, k, lab in zip(axs, EVENTS, ('(a)', '(b)')):
        M = A['M'][k]
        im = ax.imshow(np.ma.masked_invalid(M), aspect='auto', extent=ext, cmap=cmap,
                       vmin=-MAP_DB_RANGE, vmax=MAP_DB_RANGE, interpolation='nearest')
        if np.isfinite(M).any():
            ax.contour(v, z, np.ma.masked_invalid(M), levels=[0.0], colors='k', linewidths=2.0)
            for p in A_PD_EXTRA:      # P_D = Φ(M / σ_a) ≥ p  ⇔  M ≥ σ_a(z) · Φ⁻¹(p)
                off = np.array([sigma_at(ctx, zz + (0.0 if k == 'ice_top'
                                                    else float(ctx['rec_T']['ice_thick_m'])))
                                for zz in z]) * float(ndtri(p))
                ax.contour(v, z, np.ma.masked_invalid(M - off[:, None]), levels=[0.0],
                           colors='k', linewidths=1.4, linestyles='--')
        ax.plot(100 * A['vcap'], z, color='0.35', lw=1.6, ls='--')
        anc = C['anchor'][k]
        if np.isfinite(anc['vmin_C']):
            ax.plot(100 * anc['vmin_C'], C['z0'], marker='*', ms=16, color='gold', mec='k',
                    ls='none')
        ax.plot(100 * C['v0'], C['z0'], marker='o', ms=11, color='white', mec='k', mew=1.8,
                ls='none')
        ax.set_xlim(ext[0], ext[1])
        ax.set_title('{} {}'.format(lab, k))
        ax.set_xlabel('Ice volume fraction [vol%]')
    axs[0].set_ylim(ext[2], ext[3])
    axs[0].set_ylabel('Ice top depth [m]')
    fig.colorbar(im, ax=axs, label='Margin {} [dB]'.format('M + ΔM' if A['dM'] else 'M'),
                 shrink=0.9, pad=0.01)
    axs[1].legend(handles=[Line2D([], [], color='k', lw=2.0, label='P_D = 0.5'),
                           Line2D([], [], color='0.35', lw=1.6, ls='--', label='pores full'),
                           Line2D([], [], marker='*', ms=14, color='gold', mec='k', ls='none',
                                  label='case C: v_min'),
                           Line2D([], [], marker='o', ms=10, color='white', mec='k', mew=1.8,
                                  ls='none', label='gprMax ({:.0f} vol%)'.format(100 * C['v0']))],
                  loc='lower right')
    bd.save_figure(fig, out_dir, 'figA1_margin_map')


def plot_vmin(ctx, out_dir):
    """figA2：検出できる最小濃度 v_min（P_D = 0.5）と氷層の上端の深さ。"""
    A, C = ctx['A'], ctx['C']
    z = A_Z_GRID
    fig, ax = plt.subplots(figsize=(9, 8), layout='constrained')
    for k in EVENTS:
        vm = np.where(np.isfinite(A['vmin'][k]), A['vmin'][k], np.nan)
        ax.plot(100 * vm, z, color=EV_COL[k], lw=2.2, label='{}: {}'.format(k, _a_name()))
        anc = C['anchor'][k]
        if np.isfinite(anc['vmin_C']):
            ax.plot(100 * anc['vmin_C'], C['z0'], marker='*', ms=16, color=EV_COL[k], mec='k',
                    ls='none', label='{}: C'.format(k))
    ax.plot(100 * C['v0'], C['z0'], marker='o', ms=11, color='white', mec='k', mew=1.8,
            ls='none', label='gprMax ({:.0f} vol%)'.format(100 * C['v0']))
    ax.plot(100 * A['vcap'], z, color='0.35', lw=1.6, ls='--', label='pores full')
    ax.set_ylim(z[-1], z[0])
    fin = np.concatenate([A['vmin'][k][np.isfinite(A['vmin'][k])] for k in EVENTS]
                         + [np.array([C['anchor'][k]['vmin_C'] for k in EVENTS])])
    fin = fin[np.isfinite(fin)]
    right = 100 * max(0.05, 1.4 * fin.max()) if fin.size else 100 * float(np.nanmax(A['vcap']))
    ax.set_xlim(0, min(right, 100 * float(np.nanmax(A['vcap'])) * 1.05))
    ax.set_xlabel('Minimum detectable ice [vol%]')
    ax.set_ylabel('Ice top depth [m]')
    ax.legend(loc='lower right')
    bd.save_figure(fig, out_dir, 'figA2_vmin_curve')


# =============================================================================
# 7. 出力
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


def write_outputs(ctx, out_dir):
    C, A = ctx['C'], ctx['A']
    _write_csv(os.path.join(out_dir, 'caseC_pd_vs_concentration.csv'),
               [dict(v=float(v), **{'{}_pd_C'.format(k): float(C['pd'][k][i]) for k in EVENTS},
                     **{'{}_pd_A'.format(k): float(A['pd_anchor'][k][i]) for k in EVENTS},
                     **{'{}_pd_A_raw'.format(k): float(A['pd_anchor_raw'][k][i]) for k in EVENTS},
                     **{'{}_t_ns'.format(k): float(C['t_ev'][k][i]) for k in EVENTS})
                for i, v in enumerate(C['v'])])
    _write_csv(os.path.join(out_dir, 'caseA_vmin.csv'),
               [dict(ice_top_depth_m=float(z), pores_full=float(A['vcap'][i]), delta_M_db=A['dM'],
                     **{'{}_vmin'.format(k): float(A['vmin'][k][i]) for k in EVENTS},
                     **{'{}_signal_db'.format(k): float(DB20 * np.log(A['ing'][k]['A'][i]
                                                                      / ctx['a_surf']))
                        for k in EVENTS},
                     **{'{}_threshold_db'.format(k): float(DB20 * np.log(A['ing'][k]['thr'][i]
                                                                         / ctx['a_surf']))
                        for k in EVENTS},
                     **{'{}_t_ns'.format(k): float(A['ing'][k]['t'][i]) for k in EVENTS})
                for i, z in enumerate(A_Z_GRID)])
    _write_csv(os.path.join(out_dir, 'caseA_margin_map.csv'),
               [dict(ice_top_depth_m=float(z), v=float(v),
                     **{'{}_margin_db'.format(k): float(A['M'][k][i, j]) for k in EVENTS})
                for i, z in enumerate(A_Z_GRID) for j, v in enumerate(A_V_GRID)])

    L = ['detectability_map  {}'.format(datetime.now().isoformat(timespec='seconds')), '']
    L += ['[入力]'] + ['  {}: {}'.format(k, v) for k, v in ctx['paths'].items()]
    L += ['  ランダム場の同一性: {}'.format('一致' if ctx['sha_match'] else '不一致（案 C は参考）'),
          '  波源遅延 {:.3f} ns、A_surf = {:.4g}'.format(ctx['src_delay'], ctx['a_surf']), '']
    rec = ctx['rec_T']
    L += ['[基準点]  {}、{:.1f} vol%、上端 {:.2f} m、厚さ {:.2f} m'.format(
        rec['ice_model'], 100 * C['v0'], C['z0'], float(rec['ice_thick_m']))]
    for k in EVENTS:
        an, F0 = C['anchor'][k], ctx['F0'][k]
        thr = float(np.interp(an['t_ns'], ctx['thr_t'], ctx['thr']))
        L.append('  {:<10} 深さ {:.2f} m、走時 {:.2f} ns：A_sim {:+.2f} dB、しきい値 {:+.2f} dB'
                 '（余裕 {:+.2f} dB）、σ_a {:.2f} dB'.format(
                     k, an['depth_m'], an['t_ns'], DB20 * np.log(an['A_sim'] / ctx['a_surf']),
                     DB20 * np.log(thr / ctx['a_surf']), DB20 * np.log(an['A_sim'] / thr),
                     an['sigma_db']))
        L.append('             帯域中心の因子：|R| {:.5f}、G {:.4f}、|T| {:.4f}、L {:+.2f} dB'
                 .format(F0['R'], F0['G'], F0['T'], DB20 * np.log(F0['L'])))
    L += ['', '[案 C：計算結果を濃度方向に伸ばす]']
    for k in EVENTS:
        an = C['anchor'][k]
        L.append('  {:<10} P_D：gprMax の T {:.3f} / 合成（v0）{:.3f}　→　v_min = {}'.format(
            k, an['pd_T'], an['pd_C_at_v0'],
            '{:.2f} vol%'.format(100 * an['vmin_C']) if np.isfinite(an['vmin_C']) else '範囲外'))
    L += ['', '[案 A と案 C の比較（基準点）]  ΔM_k = 20 log10(v_min,A / v_min,C)']
    for k in EVENTS:
        L.append('  {:<10} 較正なし：案 A {:.2f} vol% / 案 C {:.2f} vol%（ΔM_k {:+.2f} dB）'.format(
            k, 100 * A['vmin_anchor_raw'][k], 100 * C['anchor'][k]['vmin_C'], A['dM_each'][k]))
    L.append('  ΔM（平均、{}）= {:+.2f} dB'.format('使用' if A_CALIBRATE else '未使用', A['dM']))
    for k in EVENTS:
        an, va = C['anchor'][k], A['vmin_anchor'][k]
        vc = an['vmin_C']
        diff = DB20 * np.log(va / vc) if np.isfinite(va) and np.isfinite(vc) else np.nan
        L.append('  {:<10} 較正後：案 {} {:.2f} vol% / 案 C {:.2f} vol%（残差 {:+.2f} dB）{}'.format(
            k, _a_name(), 100 * va, 100 * vc, diff,
            '' if not np.isfinite(diff) else ('  ← 1 dB 以内' if abs(diff) <= 1.0
                                               else '  ← 1 dB を超える')))
    L += ['', '[案 {}：v_min(z)（P_D = 0.5）]  上端の深さ: '.format(_a_name()) + '、'.join(
        '{:.1f} m'.format(z) for z in A_Z_GRID[::10])]
    for k in EVENTS:
        L.append('  {:<10} '.format(k) + '、'.join(
            ('{:.1f}%'.format(100 * v) if np.isfinite(v) else ('満杯でも不可' if np.isinf(v) else '—'))
            for v in A['vmin'][k][::10]))
    L.append('  （— は走時が使えない深さ：地表反射の直後、または計算領域の底の近く）')
    L.append('  {:<10} '.format('空隙満杯') + '、'.join(
        '{:.1f}%'.format(100 * v) for v in A['vcap'][::10]))
    path = os.path.join(out_dir, 'run_info.txt')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print('Saved:', path)
    print('\n'.join(L))


# =============================================================================
# 8. 本体
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
    ctx['a_surf'] = float(np.median(envR[mg].max(axis=0)))
    ctx['t_dom'] = bd.domain_bottom_time(ctx['rec_T'], src)
    ctx['thr_t'], ctx['thr'] = threshold_curve(R, dt)

    print('\n=== 案 C ===')
    ctx['C'] = case_c(ctx)
    print('\n=== 案 A ===')
    ctx['A'] = case_a(ctx)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(ctx['paths']['target'])),
                           bd.OUTPUT_SUBDIRNAME, OUTPUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    plot_c(ctx, out_dir)
    plot_ingredients(ctx, out_dir)
    plot_map(ctx, out_dir)
    plot_vmin(ctx, out_dir)
    write_outputs(ctx, out_dir)


if __name__ == '__main__':
    main()