"""差分トレース（水氷あり − 水氷なし）と解析応答の比較（Level 6 以降）

目的
----
検出可能域のマップ（濃度 × 深さ × 厚み）を、条件ごとの gprMax 計算なしに作れるかを確かめる。
そのために次の「重ね合わせモデル」が、同じランダム場で計算した水氷ありの B-scan を
どこまで再現するかを調べる。

    M0（単純な足し算） : T ≈ R + S
    M1（時間伸縮つき） : T ≈ W[R] + S
    M2（ゆらぎつき）   : T ≈ W[R] + Σ_k a_jk S_k(t − δ_jk)

      T    : 水氷ありの B-scan（gprMax）
      R    : 水氷なしの B-scan（gprMax、同じランダム場）
      S    : 氷層の上端・下端の反射の解析応答（平均媒質、far_1m の実測波形に伝達関数を掛けたもの）
      W[R] : R の時間軸を、氷層による往復走時の遅れ τ(t) だけ伸ばしたもの
             （氷層より深い散乱体からのクラッタは、氷層を往復する分だけ遅れて届く）

      a_jk, δ_jk : トレース j・界面 k の反射の振幅比と走時のずれ。ランダムな上の層を
             通る伝搬で生じる。T − W[R] を S_k に当てはめて測る。M2 は 3 通りで作る：
               measured   … 測った a_jk, δ_jk をそのまま使う（当てはめの上限）
               shuffled   … トレースの間で入れ替える（分布だけを保つ）
               parametric … 振幅は対数正規、δ は正規分布で、平均と標準偏差だけから引く
             parametric で P_D が再現できれば、深さごとの統計（a の dB の平均・σ、δ の σ）
             だけを持っておけば、任意の濃度のマップを作れる。

M1 の残差がクラッタより十分小さければ、「水氷なしの計算 1 本 ＋ 解析応答」で任意の
濃度・深さ・厚みの B-scan を合成でき、bscan_detection.py と同じ処理で検出確率を出せる。

解析応答の式（ascan_reflection_spectrum.synth_theory と同じ）
------------------------------------------------------------
    S_k(f) = E_ref(f) · G_k · T_k · R_k · A_k · exp(−i 2π f (t_k − R_REF / c))
      G_k = sqrt(R_REF / r_eff)、r_eff = 2h + 2∫dz/n      2D（線波源）の拡がり
      T_k = 地表と、それより浅い界面の往復透過
      R_k = (n_上 − n_下) / (n_上 + n_下)
      A_k = exp(−2∫α dz)
      t_k = 2h/c + 2∫n dz/c                                位相の往復走時
媒質の物理はすべて subsurface_model.py、入力と反射チャネルは bscan_detection.py を使う。

出力：<対象の merged.out のフォルダ>/bscan_detection/superposition_check/（PDF は pdf/ の下）
"""

import csv
import os
from datetime import datetime

import numpy as np
from scipy import fft as sp_fft
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import bscan_detection as bd          # rcParams（フォントのルール）もここで設定される
import subsurface_model as sm
from ascan_spectrum import load_trace, measure_peak
from gprMax.exceptions import CmdInputError

# 図の共通ルール（このファイルの中でも明示する。bscan_detection の版に依存しないように）
#   文字の大きさ：タイトル・軸ラベル 16、目盛り・凡例・図中の文字 12
#   グリッド線：特に意図がない限り全パネルに付ける（破線 '--'、alpha 0.3）
#   補助目盛り（minor ticks）：全パネルで表示する
plt.rcParams.update({'axes.titlesize': 16, 'axes.labelsize': 16, 'figure.titlesize': 16,
                     'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
                     'font.size': 12,
                     'axes.grid': True, 'axes.grid.which': 'major', 'axes.axisbelow': 'line',
                     'grid.linestyle': '--', 'grid.alpha': 0.3,
                     'xtick.minor.visible': True, 'ytick.minor.visible': True})

# =============================================================================
# 設定  [EDIT HERE]
# =============================================================================
JSON_PATH = '/Volumes/gprMax_water_ice/2D_12x4/water_ice_study_test/out_file_paths.json'
OUTPUT_SUBDIR = 'superposition_check'      # <対象>/bscan_detection/ の下に作る

# --- 解析応答 -------------------------------------------------------------------
F_MIN_HZ = 0.05e9                  # 伝達関数を計算する周波数範囲（外側は 0。波源の帯域は 0.5-2 GHz）
F_MAX_HZ = 4.0e9

# --- 時間伸縮 W[R] -------------------------------------------------------------
WARP_F_HZ = sm.BAND_CENTRE_HZ      # 走時の遅れを計算する周波数（位相速度）
WARP_DZ_M = 0.001                  # 深さ方向の積分刻み
WARP_AMPLITUDE = True              # 氷層による往復吸収の変化（WARP_F_HZ での値）も掛ける

# --- 比較 -----------------------------------------------------------------------
EVENT_HW_NS = 1.5                  # 反射の波形を比べるゲートの半幅（パルス幅 約 0.7 ns）
ZOOM_HW_NS = 4.0                   # fig1 の表示範囲（反射の走時 ± これ）
RMS_SMOOTH_NS = 1.0                # fig3 の RMS を時間方向に平均する幅
WINDOW_GUARD_NS = 2.0              # 窓ごとの集計で、界面の前後この範囲を「界面の窓」にする
SURFACE_SKIP_NS = 5.0              # 地表反射の後ろこの範囲は集計に使わない
N_THIN_TRACES = 40                 # fig1 に細線で描くトレースの本数（全部描くと黒く潰れる）

# --- 走時の遅れの実測（相互相関）-------------------------------------------------
LAG_WIN_NS = 4.0                   # 相互相関の窓の幅
LAG_STEP_NS = 0.5                  # 窓をずらす刻み
LAG_MAX_NS = 1.5                   # 探す遅れの範囲 ±

# --- M2：反射のトレースごとのゆらぎ（振幅 a_j と走時のずれ δ_j）--------------------
FIT_LAG_MAX_NS = 0.5               # δ_j を探す範囲 ±
FIT_LAG_STEP_NS = 0.005            # δ_j の刻み（1.25 GHz で位相 2.25°）
N_STAT_REAL = 10                   # 統計だけで合成する M2（シャッフル・パラメトリック）の実現数
STAT_SEED = 0                      # その乱数シード

# --- 作図 -----------------------------------------------------------------------
IMG_DB_SPAN = 60.0                # fig2 の色の幅 [dB]（最大値から下へ）
COL_SIM = 'k'                      # gprMax の差分
COL_ANA = 'tab:red'                # 解析応答 S
COL_M0 = 'tab:orange'              # M0
COL_M1 = 'tab:blue'                # M1
COL_M2 = 'tab:green'               # M2（各トレースの a_j, δ_j をそのまま使う）
COL_M2S = 'tab:purple'             # M2（統計だけ：パラメトリック）
EVENT_COLORS = {'ice_top': 'tab:orange', 'ice_bottom': 'tab:green'}   # bscan_detection と同じ
COL_CLUT = '0.55'                  # 水氷なし（クラッタ）
DB20 = 20.0 / np.log(10.0)
DB10 = 10.0 / np.log(10.0)


# =============================================================================
# 1. 解析応答 S（氷層の上端・下端）
# =============================================================================
def analytic_response(rec_T, far, nt, dt):
    """平均媒質の上端・下端の反射を、far_1m の波形から合成する。

    戻り値: {'ice_top': {...}, 'ice_bottom': {...}}、各要素に 'trace'（長さ nt）と帯域中心での各項。
    """
    bd.configure_mean_medium(rec_T)
    lvl = bd.THEORY_LEVEL
    nfft = sp_fft.next_fast_len(2 * max(nt, len(far)))
    freq = sp_fft.rfftfreq(nfft, dt)
    E_ref = sp_fft.rfft(far, nfft)
    sel = (freq >= F_MIN_HZ) & (freq <= F_MAX_HZ)
    f = freq[sel]
    i_c = int(np.argmin(np.abs(f - sm.BAND_CENTRE_HZ)))
    out = {}
    names = ['ice_top', 'ice_bottom']
    for k, d in enumerate(sm.interface_depths(lvl)):
        att, opt, inv = sm.path_integrals(f, d, lvl)
        G = np.sqrt(sm.R_REF / (2.0 * sm.TX_HEIGHT + 2.0 * inv))
        Tr = sm.transmission_product(f, d, lvl, two_way=True, include_surface=True)
        Rc = sm.interface_reflection(f, d, lvl)
        A = np.exp(-2.0 * att)
        t_f = 2.0 * sm.TX_HEIGHT / sm.C + 2.0 * opt / sm.C            # [ns]
        H = np.zeros(freq.size, dtype=complex)
        H[sel] = G * Tr * Rc * A * np.exp(-2j * np.pi * f * (t_f - sm.R_REF / sm.C) * 1e-9)
        tr = sp_fft.irfft(E_ref * H, nfft)[:nt]
        out[names[k] if k < 2 else 'iface{}'.format(k)] = {
            'trace': tr, 'depth_m': float(d), 'R': float(Rc[i_c]), 'G': float(G[i_c]),
            'T': float(Tr[i_c]), 'A_db': float(DB20 * np.log(A[i_c])),
            't_phase_ns': float(t_f[i_c])}
    return out


# =============================================================================
# 2. 時間伸縮 W[R]
# =============================================================================
def _profile(rec, z_mid):
    """平均媒質の n(z) と alpha(z)（WARP_F_HZ での値）。"""
    bd.configure_mean_medium(rec)
    f0 = np.array([WARP_F_HZ])
    er, _ = sm.eps_at_depth(f0, z_mid, bd.THEORY_LEVEL)
    a = sm.alpha_at_depth(f0, z_mid, bd.THEORY_LEVEL)
    return np.sqrt(np.asarray(er).reshape(-1)), np.asarray(a).reshape(-1)


def warp_model(rec_R, rec_T, t0_ns, z_max):
    """深さ z から届く散乱の往復走時（基準・対象）と、対象の時刻 t での遅れ τ(t)・振幅比。"""
    z_e = np.arange(0.0, z_max + 0.5 * WARP_DZ_M, WARP_DZ_M)
    z_m = 0.5 * (z_e[:-1] + z_e[1:])
    w = np.diff(z_e)
    nR, aR = _profile(rec_R, z_m)
    nT, aT = _profile(rec_T, z_m)
    cum = lambda v: np.concatenate([[0.0], np.cumsum(v * w)])
    t_R = t0_ns + 2.0 * cum(nR) / sm.C
    t_T = t0_ns + 2.0 * cum(nT) / sm.C
    datt = 2.0 * cum(aT - aR)                       # 往復吸収の差 [Np]（正なら対象が弱くなる）
    return {'z': z_e, 't_R': t_R, 't_T': t_T, 'tau': t_T - t_R, 'datt': datt}


def apply_warp(R, dt, wm):
    """W[R](t) = R(t − τ(t)) · exp(−Δatt(t))。地表より前は τ = 0。"""
    t = np.arange(R.shape[0]) * dt * 1e9
    z_t = np.interp(t, wm['t_T'], wm['z'], left=0.0, right=wm['z'][-1])
    tau = np.interp(z_t, wm['z'], wm['tau'])
    amp = np.exp(-np.interp(z_t, wm['z'], wm['datt'])) if WARP_AMPLITUDE else np.ones_like(t)
    src = np.clip(t - tau, t[0], t[-1])
    Rw = CubicSpline(t, R, axis=0)(src) * amp[:, None]
    return Rw, tau, amp


# =============================================================================
# 2b. M2：反射のトレースごとのゆらぎ
# =============================================================================
def _spec(tr):
    nfft = sp_fft.next_fast_len(2 * len(tr))
    return sp_fft.rfft(tr, nfft), sp_fft.rfftfreq(nfft), nfft   # 周波数はサンプル単位


def shifted(tr, delays_ns, dt):
    """tr(t − δ) を δ ごとに並べた [時間, len(δ)]（周波数領域の位相回転で非整数サンプルも扱う）。"""
    F, fs, nfft = _spec(tr)
    d = np.atleast_1d(np.asarray(delays_ns, dtype=float)) / (dt * 1e9)        # [サンプル]
    return sp_fft.irfft(F[:, None] * np.exp(-2j * np.pi * fs[:, None] * d[None, :]),
                        nfft, axis=0)[:len(tr)]


def fit_event(X, Sk, t, t_ev, dt):
    """各トレースの X（= T − W[R]）を a · S_k(t − δ) に当てはめる。

    δ は相互相関が最大になる遅れ（FIT_LAG_STEP_NS 刻み）、a はその δ での最小 2 乗の振幅。
    戻り値: a [ntr]、δ [ntr]、当てはめ後の相関 [ntr]
    """
    g = np.abs(t - t_ev) <= EVENT_HW_NS
    lags = np.arange(-FIT_LAG_MAX_NS, FIT_LAG_MAX_NS + 0.5 * FIT_LAG_STEP_NS, FIT_LAG_STEP_NS)
    Ss = shifted(Sk, lags, dt)[g]                      # [ng, nlag]
    Xg = X[g]                                          # [ng, ntr]
    ss = (Ss ** 2).sum(axis=0)                         # [nlag]
    xs = Ss.T @ Xg                                     # [nlag, ntr]
    xx = (Xg ** 2).sum(axis=0)
    corr = xs / np.sqrt(ss[:, None] * xx[None, :])
    k = np.argmax(corr, axis=0)
    cols = np.arange(Xg.shape[1])
    a = xs[k, cols] / ss[k]
    return a, lags[k], corr[k, cols]


def synth_events(ana, params, dt, nt):
    """Σ_k a_jk S_k(t − δ_jk) を [時間, ntr] で返す。params = {event: (a, δ)}。"""
    out = 0.0
    for name, (a, d) in params.items():
        out = out + shifted(ana[name]['trace'], d, dt)[:nt] * a[None, :]
    return out


def fluctuation_stats(a, d):
    """振幅の dB（|a| の 20 log10）と δ の統計。"""
    adb = DB20 * np.log(np.maximum(np.abs(a), 1e-6))
    return {'a_db_mean': float(adb.mean()), 'a_db_std': float(adb.std()),
            'a_db_median': float(np.median(adb)),
            'a_db_robust_std': float(1.4826 * np.median(np.abs(adb - np.median(adb)))),
            'a_neg_frac': float((a <= 0).mean()),
            'd_mean_ns': float(d.mean()), 'd_std_ns': float(d.std()),
            'd_robust_std_ns': float(1.4826 * np.median(np.abs(d - np.median(d))))}


def coherent_loss_db(a, d, f_hz=None):
    """トレース平均で失われる振幅 [dB re S]：20 log10 |mean(a exp(−i2πfδ))|（f は帯域中心）。"""
    f = sm.BAND_CENTRE_HZ if f_hz is None else f_hz
    return float(DB20 * np.log(np.abs(np.mean(a * np.exp(-2j * np.pi * f * d * 1e-9)))))


def draw_params(fits, how, rng):
    """M2 の統計版の (a, δ)。shuffled：トレースの間で入れ替え、parametric：分布から引く。"""
    out = {}
    for name, f in fits.items():
        n = f['a'].size
        if how == 'shuffled':
            p = rng.permutation(n)
            out[name] = (f['a'][p], f['d'][p])
        else:
            st = f['stats']
            adb = rng.normal(st['a_db_mean'], st['a_db_std'], n)
            out[name] = (10.0 ** (adb / 20.0), rng.normal(st['d_mean_ns'], st['d_std_ns'], n))
    return out


# =============================================================================
# 3. 走時の遅れの実測（T と R の窓ごとの相互相関）
# =============================================================================
def lag_scan(T, R, dt, t_start, t_end):
    """窓ごと・トレースごとに、T が R に対して何 ns 遅れているかを相互相関のピークで求める。"""
    dt_ns = dt * 1e9
    L = int(round(LAG_WIN_NS / dt_ns))
    nfft = sp_fft.next_fast_len(2 * L)
    win = np.hanning(L)[:, None]
    mlag = int(round(LAG_MAX_NS / dt_ns))
    centres = np.arange(t_start + 0.5 * LAG_WIN_NS, t_end - 0.5 * LAG_WIN_NS + 1e-9, LAG_STEP_NS)
    lags = np.full((centres.size, T.shape[1]), np.nan)
    ccs = np.full_like(lags, np.nan)
    for i, tc in enumerate(centres):
        i0 = int(round((tc - 0.5 * LAG_WIN_NS) / dt_ns))
        a, b = T[i0:i0 + L] * win, R[i0:i0 + L] * win
        if a.shape[0] < L:
            continue
        cc = sp_fft.irfft(sp_fft.rfft(a, nfft, axis=0) * np.conj(sp_fft.rfft(b, nfft, axis=0)),
                          nfft, axis=0)
        cc = np.concatenate([cc[-mlag:], cc[:mlag + 1]], axis=0)       # 遅れ −mlag … +mlag
        norm = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
        cc = cc / np.where(norm > 0, norm, np.inf)
        k = np.argmax(cc, axis=0)
        kk = np.clip(k, 1, cc.shape[0] - 2)
        cols = np.arange(cc.shape[1])
        y0, y1, y2 = cc[kk - 1, cols], cc[kk, cols], cc[kk + 1, cols]
        den = y0 - 2.0 * y1 + y2
        delta = np.where(den != 0, 0.5 * (y0 - y2) / np.where(den != 0, den, 1.0), 0.0)
        lags[i] = (kk - mlag + np.clip(delta, -1, 1)) * dt_ns
        ccs[i] = y1
    return centres, lags, ccs


# =============================================================================
# 4. 集計
# =============================================================================
def smoothed_rms(D, dt):
    """トレース方向の RMS を、時間方向に RMS_SMOOTH_NS で平均したもの。"""
    p = (D ** 2).mean(axis=1)
    n = max(1, int(round(RMS_SMOOTH_NS / (dt * 1e9))))
    return np.sqrt(np.convolve(p, np.ones(n) / n, mode='same'))


def event_compare(name, t_ev, t, D, DW, S, a):
    """反射 1 つについて、gprMax の差分と解析応答を比べる。

    D  = T − R（そのままの差分）、DW = T − W[R]（時間伸縮で下の層のずれを除いた差分）。
    """
    g = (t >= t_ev - EVENT_HW_NS) & (t <= t_ev + EVENT_HW_NS)
    s = S[g]
    row = {'event': name, 't_model_ns': t_ev}
    env_s = bd.envelope(S)[g]
    row['ana_peak_db'] = DB20 * np.log(env_s.max() / a)
    row['ana_t_peak_ns'] = float(t[g][np.argmax(env_s)])
    for key, X in (('diff', D), ('diffW', DW)):
        m = X.mean(axis=1)
        env_m = bd.envelope(m)[g]
        env_k = bd.envelope(X)[g]
        row[key + '_mean_peak_db'] = DB20 * np.log(env_m.max() / a)
        row[key + '_mean_minus_ana_db'] = row[key + '_mean_peak_db'] - row['ana_peak_db']
        row[key + '_mean_dt_ns'] = float(t[g][np.argmax(env_m)]) - row['ana_t_peak_ns']
        row[key + '_mean_corr'] = float(np.corrcoef(m[g], s)[0, 1])
        pk = DB20 * np.log(env_k.max(axis=0) / a)
        row[key + '_trace_peak_db_median'] = float(np.median(pk))
        row[key + '_trace_peak_db_std'] = float(np.std(pk))
        ck = [np.corrcoef(X[g, j], s)[0, 1] for j in range(X.shape[1])]
        row[key + '_trace_corr_median'] = float(np.median(ck))
        row[key + '_trace_corr_p10'] = float(np.quantile(ck, 0.1))
        row[key + '_resid_over_ana_db'] = DB10 * np.log(
            ((X[g] - s[:, None]) ** 2).mean() / (s ** 2).mean())
    return row


def window_table(t, windows, R, D, resids, S):
    """時間窓ごとのエネルギー（トレース・時間平均した 2 乗）の比 [dB]。resids = {モデル名: 残差}。"""
    rows = []
    for name, (a, b) in windows:
        m = (t >= a) & (t < b)
        if m.sum() < 10:
            continue
        e = lambda X: float((X[m] ** 2).mean())
        eR, eD = e(R), e(D)
        eS = float((S[m] ** 2).mean())
        row = {'window': name, 't0_ns': a, 't1_ns': b,
               'diff_over_clutter_db': DB10 * np.log(eD / eR),
               'ana_over_clutter_db': DB10 * np.log(max(eS, 1e-300) / eR)}
        for key, X in resids.items():
            row[key + '_resid_over_clutter_db'] = DB10 * np.log(e(X) / eR)
        for key, X in resids.items():
            row[key + '_resid_over_diff_db'] = DB10 * np.log(e(X) / eD)
        rows.append(row)
    return rows


def pd_at(rs, t_ev):
    m = np.abs(rs['t'] - t_ev) <= bd.REFL_GATE_HW_NS
    return float(rs['pd'][m].max()) if m.any() else np.nan


# =============================================================================
# 5. 作図（タイトル・軸ラベル 16、目盛り・凡例・図中の文字 12 は bscan_detection の rcParams）
# =============================================================================
def _vlines(ax, events, orient='v'):
    for e in events:
        (ax.axvline if orient == 'v' else ax.axhline)(e['t_ns'], color='0.3', ls=':', lw=1.0)


def plot_event_waveforms(ctx, out_dir):
    """fig1：反射の波形。細線＝トレースごとの差分、太線＝トレース平均、破線＝解析応答。"""
    t, D, DW, S, a = ctx['t'], ctx['D'], ctx['DW'], ctx['S'], ctx['a']
    evs = [e for e in ctx['events'] if e['name'] != 'surface']
    fig, axs = plt.subplots(2, len(evs), figsize=(7.5 * len(evs), 9), squeeze=False,
                            layout='constrained')
    pick = np.unique(np.linspace(0, D.shape[1] - 1, min(N_THIN_TRACES, D.shape[1])).astype(int))
    lab = 'abcdefgh'
    for j, e in enumerate(evs):
        g = (t >= e['t_ns'] - ZOOM_HW_NS) & (t <= e['t_ns'] + ZOOM_HW_NS)
        for i, (X, rowname) in enumerate(((D, 'T − R'), (DW, 'T − W[R]'))):
            ax = axs[i, j]
            ax.plot(t[g], X[g][:, pick] / a, color='0.6', lw=0.5, alpha=0.6)
            ax.plot(t[g], X[g].mean(axis=1) / a, color=COL_SIM, lw=2.2)
            ax.plot(t[g], S[g] / a, color=COL_ANA, lw=1.8, ls='--')
            ax.axvspan(e['t_ns'] - EVENT_HW_NS, e['t_ns'] + EVENT_HW_NS, color='0.9', zorder=0)
            ax.set_title('({}) {}: {}'.format(lab[i * len(evs) + j], e['name'], rowname))
            ax.set_xlim(t[g][0], t[g][-1])
            ax.set_xlabel('Two-way time [ns]')
            if j == 0:
                ax.set_ylabel('Ez / A_surf')
    axs[0, 0].legend(handles=[Line2D([], [], color='0.6', lw=0.8, label='each trace'),
                              Line2D([], [], color=COL_SIM, lw=2.2, label='trace mean'),
                              Line2D([], [], color=COL_ANA, lw=1.8, ls='--', label='analytic S')],
                     loc='upper right')
    bd.save_figure(fig, out_dir, 'fig1_event_waveforms')


def plot_residual_bscans(ctx, out_dir):
    """fig2：差分と残差の B-scan（包絡 [dB re A_surf]、色の範囲は 3 枚で共通）。"""
    t, x, a = ctx['t'], ctx['x'], ctx['a']
    panels = ((ctx['D'], '(a) T − R'), (ctx['resid0'], '(b) T − (R + S)  [M0]'),
              (ctx['resid1'], '(c) T − (W[R] + S)  [M1]'))
    envs = [DB20 * np.log(np.maximum(bd.envelope(X), 1e-300) / a) for X, _ in panels]
    vmax = np.ceil(np.nanmax(envs[0]) / 5.0) * 5.0
    ext = [x[0], x[-1], t[-1], t[0]]
    fig, axs = plt.subplots(1, 3, figsize=(20, 7.5), sharey=True, layout='constrained')
    for ax, env, (_, title) in zip(axs, envs, panels):
        im = ax.imshow(env, aspect='auto', extent=ext, cmap='viridis', vmin=vmax - IMG_DB_SPAN,
                       vmax=vmax, interpolation='nearest')
        _vlines(ax, ctx['events'], orient='h')
        ax.set_title(title)
        ax.set_xlabel(ctx['x_label'])
    axs[0].set_ylabel('Two-way time [ns]')
    if np.isfinite(ctx['t_bottom']):
        axs[0].set_ylim(min(t[-1], ctx['t_bottom'] + 5.0), t[0])
    fig.colorbar(im, ax=axs, label='Envelope [dB re A_surf]', shrink=0.9, pad=0.01)
    bd.save_figure(fig, out_dir, 'fig2_residual_bscans')


def plot_rms(ctx, out_dir):
    """fig3：トレース方向の RMS の時間変化。残差がクラッタより下にあるかを見る。"""
    t, a, dt = ctx['t'], ctx['a'], ctx['dt']
    to_db = lambda v: DB20 * np.log(np.maximum(v, 1e-300) / a)
    fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
    ax.plot(t, to_db(smoothed_rms(ctx['R'], dt)), color=COL_CLUT, lw=2.0, label='R (clutter)')
    ax.plot(t, to_db(smoothed_rms(ctx['D'], dt)), color=COL_SIM, lw=1.4, label='T − R')
    ax.plot(t, to_db(smoothed_rms(ctx['S'][:, None], dt)), color=COL_ANA, lw=1.4, ls='--',
            label='S')
    ax.plot(t, to_db(smoothed_rms(ctx['resid0'], dt)), color=COL_M0, lw=1.4, label='residual M0')
    ax.plot(t, to_db(smoothed_rms(ctx['resid1'], dt)), color=COL_M1, lw=1.4, label='residual M1')
    ax.plot(t, to_db(smoothed_rms(ctx['resid2'], dt)), color=COL_M2, lw=1.4,
            label='residual M2 (measured)')
    _vlines(ax, ctx['events'])
    t1 = min(t[-1], ctx['t_bottom'] + 5.0) if np.isfinite(ctx['t_bottom']) else t[-1]
    ax.set_xlim(ctx['events'][0]['t_ns'] - 2.0 if ctx['events'] else t[0], t1)
    vis = (t >= ax.get_xlim()[0]) & (t <= t1)
    top = np.nanmax(to_db(smoothed_rms(ctx['R'], dt))[vis])
    ax.set_ylim(top - 90.0, top + 5.0)
    ax.set_xlabel('Two-way time [ns]')
    ax.set_ylabel('RMS over traces [dB re A_surf]')
    ax.legend(loc='upper right', ncol=2)
    bd.save_figure(fig, out_dir, 'fig3_rms_vs_time')


def plot_lag(ctx, out_dir):
    """fig4：T が R に対して何 ns 遅れているか（相互相関）と、モデルの τ(t)。"""
    tc, lags, ccs = ctx['lag']
    blocks = bd.block_index(lags.shape[1])
    fig, axs = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True, layout='constrained')
    for b in blocks:
        axs[0].plot(tc, np.nanmedian(lags[:, b], axis=1), color='0.6', lw=0.6, alpha=0.7)
        axs[1].plot(tc, np.nanmedian(ccs[:, b], axis=1), color='0.6', lw=0.6, alpha=0.7)
    axs[0].plot(tc, np.nanmedian(lags, axis=1), color=COL_SIM, lw=2.2)
    axs[1].plot(tc, np.nanmedian(ccs, axis=1), color=COL_SIM, lw=2.2)
    axs[0].plot(ctx['t'], ctx['tau'], color=COL_ANA, lw=1.8, ls='--')
    for ax in axs:
        _vlines(ax, ctx['events'])
    axs[0].set_ylabel('Delay of T vs R [ns]')
    axs[1].set_ylabel('Max correlation')
    axs[1].set_xlabel('Two-way time [ns] (window centre)')
    axs[0].set_title('(a) delay')
    axs[1].set_title('(b) correlation at the delay')
    axs[1].set_ylim(-0.05, 1.05)
    axs[0].legend(handles=[Line2D([], [], color='0.6', lw=0.8, label='block median'),
                           Line2D([], [], color=COL_SIM, lw=2.2, label='all traces'),
                           Line2D([], [], color=COL_ANA, lw=1.8, ls='--', label='model τ(t)')],
                  loc='upper left')
    axs[0].set_xlim(tc[0] - 0.5 * LAG_WIN_NS, tc[-1] + 0.5 * LAG_WIN_NS)
    bd.save_figure(fig, out_dir, 'fig4_delay')


def plot_pd(ctx, out_dir):
    """fig5：反射チャネルの検出確率。gprMax の T と、合成した T（M1 / M2）。
    M2 の統計版は N_STAT_REAL 回の実現の平均。M0 は M1 とほぼ重なるので描かない（数値は run_info）。"""
    fig, ax = plt.subplots(figsize=(12, 5.5), layout='constrained')
    for key, col, ls, lab in (('T', COL_SIM, '-', 'gprMax T'), ('M1', COL_M1, '--', 'M1'),
                              ('M2', COL_M2, '-', 'M2 measured'),
                              ('M2_shuffled', COL_M2, ':', 'M2 shuffled'),
                              ('M2_parametric', COL_M2S, '-.', 'M2 parametric')):
        rs = ctx['rs'][key]
        ax.plot(rs['t'], rs['pd'], color=col, ls=ls, lw=1.8, label=lab)
    ax.axhline(bd.PD_VISIBLE, color='0.4', lw=1.0, ls='-.')
    _vlines(ax, ctx['events'])
    t1 = ctx['t_bottom'] + 5.0 if np.isfinite(ctx['t_bottom']) else ctx['t'][-1]
    ax.set_xlim(ctx['events'][0]['t_ns'] - 2.0 if ctx['events'] else 0.0, t1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('Two-way time [ns]')
    ax.set_ylabel('P_D (reflection)')
    ax.legend(loc='upper right')
    bd.save_figure(fig, out_dir, 'fig5_pd_reflection')


def plot_fluctuation(ctx, out_dir):
    """fig6：反射のトレースごとのゆらぎ。振幅比 a（dB re S）と走時のずれ δ を、x に沿って・分布で。"""
    x, fits = ctx['x'], ctx['fits']
    fig, axs = plt.subplots(2, 2, figsize=(15, 9), layout='constrained',
                            gridspec_kw={'width_ratios': [3, 1]})
    for name, f in fits.items():
        col = EVENT_COLORS.get(name, 'k')
        adb = DB20 * np.log(np.maximum(np.abs(f['a']), 1e-6))
        st = f['stats']
        axs[0, 0].plot(x, adb, color=col, lw=1.2, label=name)
        axs[1, 0].plot(x, f['d'], color=col, lw=1.2, label=name)
        axs[0, 1].hist(adb, bins=25, color=col, alpha=0.5, orientation='horizontal',
                       label='σ = {:.1f} dB'.format(st['a_db_std']))
        axs[1, 1].hist(f['d'], bins=25, color=col, alpha=0.5, orientation='horizontal',
                       label='σ = {:.3f} ns'.format(st['d_std_ns']))
    for ax in axs[:, 0]:
        ax.set_xlabel(ctx['x_label'])
        ax.axhline(0.0, color='0.4', lw=0.8)
    for ax in axs[:, 1]:
        ax.set_xlabel('Count')
        ax.axhline(0.0, color='0.4', lw=0.8)
        ax.legend(loc='upper right')
    axs[0, 1].sharey(axs[0, 0])
    axs[1, 1].sharey(axs[1, 0])
    axs[0, 0].set_ylabel('Amplitude a [dB re S]')
    axs[1, 0].set_ylabel('Delay δ [ns]')
    axs[0, 0].set_title('(a) amplitude')
    axs[0, 1].set_title('(b)')
    axs[1, 0].set_title('(c) delay')
    axs[1, 1].set_title('(d)')
    axs[0, 0].legend(loc='lower left')
    bd.save_figure(fig, out_dir, 'fig6_reflection_fluctuation')


# =============================================================================
# 6. 出力
# =============================================================================
def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ('{:.6g}'.format(v) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print('Saved:', path)


def write_outputs(ctx, out_dir):
    _write_csv(os.path.join(out_dir, 'event_comparison.csv'), ctx['ev_rows'])
    _write_csv(os.path.join(out_dir, 'window_energy.csv'), ctx['win_rows'])
    rows = []
    for j in range(len(ctx['x'])):
        r = {'x': float(ctx['x'][j])}
        for name, f in ctx['fits'].items():
            r[name + '_a'] = float(f['a'][j])
            r[name + '_a_db'] = float(DB20 * np.log(max(abs(f['a'][j]), 1e-6)))
            r[name + '_delay_ns'] = float(f['d'][j])
            r[name + '_corr'] = float(f['corr'][j])
        rows.append(r)
    _write_csv(os.path.join(out_dir, 'reflection_fluctuation_per_trace.csv'), rows)
    _write_csv(os.path.join(out_dir, 'reflection_fluctuation_stats.csv'),
               [dict(event=n, **f['stats'], mean_loss_meas_db=f['mean_loss_meas_db'],
                     mean_loss_pred_db=f['mean_loss_pred_db']) for n, f in ctx['fits'].items()])
    tc, lags, ccs = ctx['lag']
    _write_csv(os.path.join(out_dir, 'delay_scan.csv'),
               [{'t_centre_ns': float(c), 'delay_median_ns': float(np.nanmedian(lags[i])),
                 'delay_p25_ns': float(np.nanquantile(lags[i], 0.25)),
                 'delay_p75_ns': float(np.nanquantile(lags[i], 0.75)),
                 'corr_median': float(np.nanmedian(ccs[i])),
                 'model_tau_ns': float(np.interp(c, ctx['t'], ctx['tau']))}
                for i, c in enumerate(tc)])
    L = ['bscan_superposition_check  {}'.format(datetime.now().isoformat(timespec='seconds')), '']
    L += ['[入力]'] + ['  {}: {}'.format(k, v) for k, v in ctx['inputs'].items()] + ['']
    L += ['[確認]'] + ['  ' + c for c in ctx['checks']] + ['']
    L += ['[解析応答（帯域中心 {:.2f} GHz の値）]'.format(sm.BAND_CENTRE_HZ / 1e9)]
    for k, v in ctx['ana'].items():
        L.append('  {:<10} 深さ {:.3f} m  R = {:+.5f}  G = {:.4f}  T = {:.4f}  A = {:.2f} dB  '
                 '位相走時 {:.3f} ns'.format(k, v['depth_m'], v['R'], v['G'], v['T'], v['A_db'],
                                           v['t_phase_ns']))
    L.append('  氷層の下での往復走時の遅れ（モデル）{:.3f} ns、往復吸収の差 {:+.3f} dB'.format(
        ctx['tau_total'], -DB20 * ctx['datt_total']))
    L += ['', '[反射の波形の比較]  diff = T − R、diffW = T − W[R]（dB は A_surf 基準）']
    for r in ctx['ev_rows']:
        L.append('  {}: 解析 {:.2f} dB / 差分の平均 {:.2f} dB（差 {:+.2f} dB、ずれ {:+.3f} ns、相関 {:.3f}）'
                 ' / W 補正後 {:.2f} dB（差 {:+.2f} dB、相関 {:.3f}）'.format(
                     r['event'], r['ana_peak_db'], r['diff_mean_peak_db'],
                     r['diff_mean_minus_ana_db'], r['diff_mean_dt_ns'], r['diff_mean_corr'],
                     r['diffW_mean_peak_db'], r['diffW_mean_minus_ana_db'], r['diffW_mean_corr']))
        L.append('        トレースごと：ピーク {:.2f} ± {:.2f} dB、解析との相関 中央値 {:.3f}'
                 '（W 補正後 {:.3f}）、残差/解析 {:+.1f} dB（W 補正後 {:+.1f} dB）'.format(
                     r['diff_trace_peak_db_median'], r['diff_trace_peak_db_std'],
                     r['diff_trace_corr_median'], r['diffW_trace_corr_median'],
                     r['diff_resid_over_ana_db'], r['diffW_resid_over_ana_db']))
    L += ['', '[反射のトレースごとのゆらぎ（T − W[R] を a · S_k(t − δ) に当てはめ）]']
    for name, f in ctx['fits'].items():
        st = f['stats']
        L.append('  {:<10} a：平均 {:+.2f} dB、σ {:.2f} dB（ロバスト σ {:.2f}）、a ≤ 0 の割合 {:.3f}'
                 .format(name, st['a_db_mean'], st['a_db_std'], st['a_db_robust_std'],
                         st['a_neg_frac']))
        L.append('             δ：平均 {:+.3f} ns、σ {:.3f} ns（ロバスト σ {:.3f}）、'
                 '当てはめ後の相関 中央値 {:.3f}'.format(st['d_mean_ns'], st['d_std_ns'],
                                                 st['d_robust_std_ns'], float(np.median(f['corr']))))
        L.append('             平均波形の目減り：実測 {:+.2f} dB / a と δ からの予測 {:+.2f} dB'
                 .format(f['mean_loss_meas_db'], f['mean_loss_pred_db']))
    if len(ctx['fits']) == 2:
        (n1, f1), (n2, f2) = list(ctx['fits'].items())
        L.append('  {} と {} の相関：a（dB）{:+.2f}、δ {:+.2f}'.format(
            n1, n2, float(np.corrcoef(DB20 * np.log(np.maximum(np.abs(f1['a']), 1e-6)),
                                      DB20 * np.log(np.maximum(np.abs(f2['a']), 1e-6)))[0, 1]),
            float(np.corrcoef(f1['d'], f2['d'])[0, 1])))
    models = list(ctx['resids'].keys())
    L += ['', '[時間窓ごとのエネルギー比 dB]  クラッタ = R、差分 = T − R、Mx = その残差']
    head = ['diff/clut', 'S/clut'] + ['{}/clut'.format(k) for k in models] + \
           ['{}/diff'.format(k) for k in models]
    L.append('  {:<8}'.format('window') + ''.join('{:>12}'.format(h) for h in head))
    for r in ctx['win_rows']:
        vals = [r['diff_over_clutter_db'], r['ana_over_clutter_db']] + \
               [r[k + '_resid_over_clutter_db'] for k in models] + \
               [r[k + '_resid_over_diff_db'] for k in models]
        L.append('  {:<8}'.format(r['window']) + ''.join('{:>12.1f}'.format(v) for v in vals))
    keys = ['T', 'M0', 'M1', 'M2', 'M2_shuffled', 'M2_parametric']
    L += ['', '[反射チャネルの P_D（界面の走時 ± {:.1f} ns の最大）]  統計版は {} 回の実現の平均'
          .format(bd.REFL_GATE_HW_NS, N_STAT_REAL)]
    L.append('  {:<10}'.format('') + ''.join('{:>15}'.format(k) for k in keys))
    for e in ctx['events']:
        if e['name'] == 'surface':
            continue
        L.append('  {:<10}'.format(e['name']) +
                 ''.join('{:>15.3f}'.format(pd_at(ctx['rs'][k], e['t_ns'])) for k in keys))
    m = ctx['pd_mask']
    for key in keys[1:]:
        dp = np.abs(ctx['rs'][key]['pd'] - ctx['rs']['T']['pd'])[m]
        L.append('  |P_D({}) − P_D(T)|（地表の後ろ〜領域の底）：平均 {:.3f}、最大 {:.3f}'
                 .format(key, float(dp.mean()), float(dp.max())))
    path = os.path.join(out_dir, 'run_info.txt')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print('Saved:', path)
    print('\n'.join(L))


# =============================================================================
# 7. 本体
# =============================================================================
def run_target(tgt, conds, reference, R, rec_R, dt, x, fs, sel, x_label, base_checks):
    p_t = conds[tgt][bd.BSCAN_KEY]
    print('\n=== 対象: {}  ({})'.format(tgt, p_t))
    T, dt_t = bd.read_bscan(p_t)
    if abs(dt_t - dt) > 1e-9 * dt or T.shape[0] != R.shape[0]:
        raise CmdInputError('{} と基準で dt または時間サンプル数が違います'.format(tgt))
    if fs is not None:
        T = T - fs[:, None]
    T = T[:, sel]
    if T.shape != R.shape:
        raise CmdInputError('{} と基準でトレース数が違います: {} vs {}'.format(tgt, T.shape, R.shape))
    rec_path, rec_T = bd.find_record(p_t)
    if not rec_T or not bd.has_ice(rec_T):
        raise CmdInputError('対象 {} の記録ファイルが無いか、氷が入っていません（解析応答を作れない）'
                            .format(tgt))
    checks = list(base_checks)
    checks.append('対象の記録: {}（{}、{:.1f} vol%、深さ {:.2f} m、厚さ {:.2f} m）'.format(
        rec_path, rec_T['ice_model'], 100 * float(rec_T['ice_vol_frac']),
        float(rec_T['ice_top_m']), float(rec_T['ice_thick_m'])))
    h_t = rec_T.get('rho', {}).get('rho_map_sha256')
    h_r = rec_R.get('rho', {}).get('rho_map_sha256')
    same = bool(h_t) and h_t == h_r
    checks.append('ランダム場の同一性（rho_map_sha256）: {}'.format('一致' if same else '不一致'))
    if not same:
        print('  注意: ランダム場が基準と違います。この比較は成り立ちません（結果は参考）。')
    for key in ('n_clip_lo', 'n_clip_hi', 'max'):
        v_t, v_r = rec_T.get('rho', {}).get(key), rec_R.get('rho', {}).get(key)
        if v_t is not None and v_r is not None and v_t != v_r:
            checks.append('注意: 密度の打ち切り {} が基準と違います（対象 {} / 基準 {}）'
                          .format(key, v_t, v_r))

    # 波源遅延・界面の走時（bscan_detection と同じ定義）
    far, dt_far = load_trace(reference[bd.REF_KEY])
    if abs(dt_far - dt) > 1e-6 * dt:
        raise CmdInputError('far_1m の dt {} が B-scan の dt {} と違います'.format(dt_far, dt))
    src_delay = measure_peak(far, dt_far)['t_peak'] - sm.R_REF / sm.C
    events = bd.model_events(rec_T, src_delay)
    t_bottom = bd.domain_bottom_time(rec_T, src_delay)
    t = np.arange(R.shape[0]) * dt * 1e9

    # 解析応答 S と合成 B-scan
    ana = analytic_response(rec_T, far, R.shape[0], dt)
    S = sum(v['trace'] for v in ana.values())
    t0 = 2.0 * sm.TX_HEIGHT / sm.C + src_delay
    wm = warp_model(rec_R, rec_T, t0, float(rec_T.get('surface_y_m', 3.0)))
    Rw, tau, amp = apply_warp(R, dt, wm)
    syn0 = R + S[:, None]
    syn1 = Rw + S[:, None]
    D, DW = T - R, T - Rw
    resid0, resid1 = T - syn0, T - syn1
    zb = float(rec_T['ice_top_m']) + float(rec_T['ice_thick_m'])
    tau_total = float(np.interp(zb, wm['z'], wm['tau']))
    datt_total = float(np.interp(zb, wm['z'], wm['datt']))

    # dB の基準：基準 B-scan の地表反射（bscan_detection と同じ）
    envR = bd.envelope(R)
    ms = np.abs(t - t0) <= 5.0
    t_s = float(t[ms][np.argmax(envR[ms].mean(axis=1))])
    mg = np.abs(t - t_s) <= bd.REFL_GATE_HW_NS
    a = float(np.median(envR[mg].max(axis=0)))
    checks.append('波源遅延 {:.3f} ns、地表反射 {:.3f} ns（データ）、A_surf = {:.4g}'
                  .format(src_delay, t_s, a))

    ev_rows = [event_compare(e['name'], e['t_ns'], t, D, DW, ana[e['name']]['trace'], a)
               for e in events if e['name'] in ana]
    t_top = [e['t_ns'] for e in events if e['name'] == 'ice_top'][0]
    t_bot = [e['t_ns'] for e in events if e['name'] == 'ice_bottom'][0]
    g = WINDOW_GUARD_NS
    t_end = min(t[-1], t_bottom)
    windows = [('above', (t_s + SURFACE_SKIP_NS, t_top - g)), ('top', (t_top - g, t_top + g)),
               ('layer', (t_top + g, t_bot - g)), ('bottom', (t_bot - g, t_bot + g)),
               ('below', (t_bot + g, t_end))]
    # M2：反射のトレースごとのゆらぎを測り、それを入れて合成する
    fits = {}
    for e in events:
        if e['name'] not in ana:
            continue
        a_j, d_j, c_j = fit_event(DW, ana[e['name']]['trace'], t, e['t_ns'], dt)
        row = [r for r in ev_rows if r['event'] == e['name']][0]
        fits[e['name']] = {'a': a_j, 'd': d_j, 'corr': c_j, 'stats': fluctuation_stats(a_j, d_j),
                           'mean_loss_meas_db': row['diffW_mean_minus_ana_db'],
                           'mean_loss_pred_db': coherent_loss_db(a_j, d_j)}
    nt = R.shape[0]
    syn2 = Rw + synth_events(ana, {k: (f['a'], f['d']) for k, f in fits.items()}, dt, nt)
    resid2 = T - syn2
    resids = {'M0': resid0, 'M1': resid1, 'M2': resid2}
    win_rows = window_table(t, windows, R, D, resids, S)
    lag = lag_scan(T, R, dt, t_s + SURFACE_SKIP_NS, t_end)
    rs = {'T': bd.reflection_scan(T, R, dt), 'M1': bd.reflection_scan(syn1, R, dt),
          'M0': bd.reflection_scan(syn0, R, dt), 'M2': bd.reflection_scan(syn2, R, dt)}
    rng = np.random.RandomState(STAT_SEED)
    for how in ('shuffled', 'parametric'):
        pds = []
        for _ in range(N_STAT_REAL):
            syn = Rw + synth_events(ana, draw_params(fits, how, rng), dt, nt)
            pds.append(bd.reflection_scan(syn, R, dt)['pd'])
        rs['M2_' + how] = {'t': rs['T']['t'], 'pd': np.mean(pds, axis=0),
                           'pd_min': np.min(pds, axis=0), 'pd_max': np.max(pds, axis=0)}
    pd_mask = (rs['T']['t'] >= t_s + SURFACE_SKIP_NS) & (rs['T']['t'] <= t_end)

    ctx = {'t': t, 'x': x, 'x_label': x_label, 'dt': dt, 'a': a, 'R': R, 'D': D, 'DW': DW,
           'S': S, 'resid0': resid0, 'resid1': resid1, 'resid2': resid2, 'resids': resids,
           'fits': fits, 'tau': tau, 'events': events,
           't_bottom': t_bottom, 'ana': ana, 'ev_rows': ev_rows, 'win_rows': win_rows,
           'lag': lag, 'rs': rs, 'pd_mask': pd_mask, 'tau_total': tau_total,
           'datt_total': datt_total, 'checks': checks,
           'inputs': {'target': '{} ({})'.format(tgt, p_t),
                      'reference': '{} ({})'.format(bd.REFERENCE_CONDITION,
                                                    conds[bd.REFERENCE_CONDITION][bd.BSCAN_KEY]),
                      'far_1m': reference[bd.REF_KEY]}}
    out_dir = os.path.join(os.path.dirname(os.path.abspath(p_t)), bd.OUTPUT_SUBDIRNAME,
                           OUTPUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    plot_event_waveforms(ctx, out_dir)
    plot_residual_bscans(ctx, out_dir)
    plot_rms(ctx, out_dir)
    plot_lag(ctx, out_dir)
    plot_pd(ctx, out_dir)
    plot_fluctuation(ctx, out_dir)
    write_outputs(ctx, out_dir)
    return ctx


def main():
    level, chosen, conds, reference = bd.select_conditions(JSON_PATH)
    if bd.REFERENCE_CONDITION not in conds:
        raise CmdInputError('基準の条件 "{}" がありません'.format(bd.REFERENCE_CONDITION))
    targets = [c for c in sorted(conds) if c != bd.REFERENCE_CONDITION]
    if bd.TARGET_CONDITIONS:
        targets = [c for c in targets if c in bd.TARGET_CONDITIONS]
    if bd.REF_KEY not in reference or not os.path.exists(reference[bd.REF_KEY]):
        raise CmdInputError('_reference の {} がありません（解析応答の波形に必要）'.format(bd.REF_KEY))
    p_r = conds[bd.REFERENCE_CONDITION][bd.BSCAN_KEY]
    R, dt = bd.read_bscan(p_r)
    fs = None
    if bd.SUBTRACT_FREESPACE:
        fs_path = bd.FREESPACE_AT_TX_PATH or reference.get(bd.FREESPACE_KEY, '')
        if not fs_path or not os.path.exists(fs_path):
            raise CmdInputError('直達波の除去に使う自由空間 at_tx が見つかりません')
        fs, _ = load_trace(fs_path)
        fs = np.pad(fs, (0, max(0, R.shape[0] - len(fs))))[:R.shape[0]]
        R = R - fs[:, None]
    rec_path_R, rec_R = bd.find_record(p_r)
    if not rec_R:
        raise CmdInputError('基準 {} の記録ファイルが見つかりません（時間伸縮のモデルに必要）'.format(p_r))
    if bd.has_ice(rec_R):
        raise CmdInputError('基準の記録 {} では氷が入っています'.format(rec_path_R))
    x_all, has_x = bd.trace_positions(rec_R, R.shape[1])
    sel = np.ones(R.shape[1], dtype=bool)
    if bd.TRACE_X_RANGE_M is not None:
        sel = (x_all >= bd.TRACE_X_RANGE_M[0] - 1e-9) & (x_all <= bd.TRACE_X_RANGE_M[1] + 1e-9)
    R, x = R[:, sel], x_all[sel]
    checks = ['基準: {}、{} 本、dt = {:.4f} ps、記録 {}'.format(p_r, R.shape[1], dt * 1e12, rec_path_R),
              '直達波の除去: {}'.format('あり' if fs is not None else 'なし')]
    for tgt in targets:
        run_target(tgt, conds, reference, R, rec_R, dt, x, fs, sel,
                   'x [m]' if has_x else 'Trace number', checks)


if __name__ == '__main__':
    main()