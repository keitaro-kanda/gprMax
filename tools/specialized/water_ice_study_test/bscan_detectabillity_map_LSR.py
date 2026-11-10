"""スペクトル比法（LSR）で水氷層を見分けられるかの地図

問い：氷層（上端〜下端）の tanδ が、上の層（地表〜上端）の tanδ と違うと言えるか。
      判定は bscan_detection のスペクトル比法と同じ（上の層を基準にする）。

考え方（反射の地図の新案 A と同じく「信号はモデル、ばらつきは計算から」）
    信号：平均媒質のモデルで、tanδ の差
              Δtanδ(v, z, S) = tanδ(氷層 z〜z+D) − tanδ(上の層 0〜z)
          v：氷の体積分率、z：氷層の上端の深さ、S：FeO+TiO2 [wt%]、D：氷層の厚さ（記録の値に固定）
    ばらつき：bscan_detection の LSR の結果（v0・z0・S0 の 1 回の計算）から測る
          ブロック：氷層の区間のブロックごとの tanδ の、平均からのずれ e_j
          測線全体：ブートストラップの差（氷層 − 上の層）の、中央値からのずれ e_b
    判定：
          ブロック  P_D = 「|Δtanδ + e_j| > 上の層の範囲の半幅」となるブロックの割合（P_D ≥ 0.5 で見える）
          測線全体  Δtanδ + e_b の 95 % 区間が 0 をまたがなければ見える
                    → 比 |Δtanδ| / 必要な差 が 1 以上で見える

地図は 2 種類
    (v, z)：組成は計算のまま（S0）。氷の濃度と深さを動かす
    (S, z)：氷の濃度は固定（S_MAP_V。None なら計算の値 v0）。組成と深さを動かす

入力（先に bscan_detection.py と detectability_map.py を実行しておく）
    <対象の merged.out のフォルダ>/bscan_detection/spectral_ratio/ の CSV
        spectral_ratio_segments.csv / spectral_ratio_blocks.csv / spectral_ratio_bootstrap_samples.csv
    <同>/bscan_detection/detectability_map/newA_vmin.csv（反射が見えない領域を灰色にする。無ければ省く）
式の説明は bscan_detectability_lsr.md。
出力：<対象の merged.out のフォルダ>/bscan_detection/detectability_lsr/（PDF は pdf/ の下）
"""

import copy
import csv
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import bscan_detection as bd               # 入力・平均媒質・LSR のモデル値
import subsurface_model as sm
from gprMax.exceptions import CmdInputError

# 図の共通ルール（このファイルの中でも明示する）
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
OUTPUT_SUBDIR = 'detectability_lsr'
LSR_SUBDIR = 'spectral_ratio'      # bscan_detection の LSR の出力（bscan_detection/ の下）
REFL_SUBDIR = 'detectability_map'  # 反射の地図の出力（bscan_detection/ の下）
USE_REFL_MASK = True               # 反射（上端・下端）が見えない領域を灰色にする

V_GRID = np.round(np.arange(0.0, 0.4001, 0.01), 4)          # (v, z) 地図の横軸（体積分率）
Z_GRID = np.round(np.arange(0.20, 2.95 + 1e-9, 0.05), 3)    # 縦軸：氷層の上端の深さ [m]
S_GRID = np.round(np.arange(0.0, 30.0001, 0.5), 3)          # (S, z) 地図の横軸：FeO+TiO2 [wt%]
S_MAP_V = None                     # (S, z) 地図の氷の体積分率。None なら計算の値 v0
S_MARKS = (20.0,)                  # (S, z) 地図に縦の点線で示す組成（例：海の玄武岩）[wt%]
GAP_DEPTHS = (0.5, 1.0, 2.0)       # figL2 (a) で線を引く上端の深さ [m]
CI_PCT = 95.0                      # 測線全体の判定に使う区間 [%]

DEPTH_LABEL = 'Ice top depth [m]'
EV_COL = {'ice_top': 'tab:orange', 'ice_bottom': 'tab:green'}   # 反射の地図と同じ色
DEPTH_LS = ('-', '--', ':', '-.')       # figL2 (a) の深さの線種（色は界面の色と混同しないよう黒）
FIG4_PANELS = ('b', 'b')           # figL4 に並べるパネル：(figL1 の, figL3 の)。'a' = ブロック、'b' = 測線全体。
                                   # None なら figL4 を作らない
MAP_GAP_FIGS = True                # figL5（figL1 (b) ＋ figL2 (a)）と figL6（figL3 (b) ＋ figL2 (b)）を作る
GAP_YMIN = 1e-5                    # figL2 の縦軸の下限（打ち消し合う濃度の谷が深くなりすぎないよう）


# =============================================================================
# 1. 入力
# =============================================================================
def _read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def load_inputs():
    level, chosen, conds, reference = bd.select_conditions(JSON_PATH)
    targets = [c for c in sorted(conds) if c != bd.REFERENCE_CONDITION]
    tgt = TARGET_CONDITION or (targets[0] if targets else None)
    if tgt not in conds:
        raise CmdInputError('対象の条件 "{}" がありません（候補: {}）'.format(tgt, targets))
    p_t = conds[tgt][bd.BSCAN_KEY]
    rp, rec = bd.find_record(p_t)
    if not rec or not bd.has_ice(rec):
        raise CmdInputError('対象の記録が無いか、氷が入っていません: {}'.format(rp))
    base = os.path.join(os.path.dirname(os.path.abspath(p_t)), bd.OUTPUT_SUBDIRNAME)
    d_lsr = os.path.join(base, LSR_SUBDIR)

    segs = _read_csv(os.path.join(d_lsr, 'spectral_ratio_segments.csv'))
    blocks = _read_csv(os.path.join(d_lsr, 'spectral_ratio_blocks.csv'))
    boot = _read_csv(os.path.join(d_lsr, 'spectral_ratio_bootstrap_samples.csv'))
    if not segs or not blocks:
        raise CmdInputError('LSR の結果が見つかりません（先に bscan_detection.py を実行）: {}'.format(d_lsr))
    ref = [s for s in segs if s['role'] == 'reference']
    test = [s for s in segs if s['role'] == 'test' and s['A'] == 'ice_top' and s['B'] == 'ice_bottom']
    if not ref or not test:
        raise CmdInputError('上の層（地表→上端）と氷層（上端→下端）の区間がそろっていません。'
                            '上端と下端の反射が両方「見える」計算で bscan_detection を実行してください')
    ref, test = ref[0], test[0]
    e_blk = np.array([_num(b['tand']) for b in blocks if b['segment'] == test['name']])
    e_blk = e_blk[np.isfinite(e_blk)]
    e_blk = e_blk - e_blk.mean()
    half = _num(ref['ref_half_width'])
    if boot:
        d = np.array([_num(r[test['name']]) - _num(r[ref['name']]) for r in boot])
        d = d[np.isfinite(d)]
    else:
        d = np.array([])
    e_boot = d - np.median(d) if d.size else d

    refl = None
    if USE_REFL_MASK:
        rows = _read_csv(os.path.join(base, REFL_SUBDIR, 'newA_vmin.csv'))
        if rows:
            col = lambda k: ('{}_vmin_smoothed'.format(k) if '{}_vmin_smoothed'.format(k) in rows[0]
                             else '{}_vmin'.format(k))
            refl = {'z': np.array([_num(r['ice_top_depth_m']) for r in rows]),
                    'top': np.array([_num(r[col('ice_top')]) for r in rows]),
                    'bottom': np.array([_num(r[col('ice_bottom')]) for r in rows]),
                    'col': col('ice_top').replace('ice_top_', '')}
    return {'rec': rec, 'target': p_t, 'base': base, 'ref': ref, 'test': test,
            'e_blk': e_blk, 'half': half, 'd_boot': d, 'e_boot': e_boot, 'refl': refl}


# =============================================================================
# 2. 信号（平均媒質のモデル）
# =============================================================================
def model_rec(rec, v, z, S):
    r = copy.deepcopy(rec)
    r['ice_vol_frac'] = float(v)
    r['ice_top_m'] = float(z)
    r['feotio2_wt'] = float(S)
    r['enable_ice'] = True
    return r


def delta_tand(rec, v, z, S):
    """Δtanδ = tanδ(氷層 z〜z+D) − tanδ(上の層 0〜z)。bscan_detection.lsr_model_tandelta と同じ計算。"""
    if v > pore_capacity(rec, z) + 1e-9:                 # pore ice では入りきらない濃度
        return np.nan
    r = model_rec(rec, v, z, S)
    D = float(rec['ice_thick_m'])
    _, over = bd.lsr_model_tandelta(r, 0.0, z)       # 上の層には氷がない（氷なしの値）
    lay, _ = bd.lsr_model_tandelta(r, z, z + D)      # 氷層（氷ありの値）
    return lay - over


def pore_capacity(rec, z):
    """pore ice で空隙を満たせる上限 φ(z + D) = 1 − ρ(z + D)/ρ_g（excess には上限なし）。"""
    if rec['ice_model'] != 'pore':
        return np.inf
    bd.configure_mean_medium(rec)
    return float(1.0 - sm.density_profile(z + float(rec['ice_thick_m'])) / sm.LEVEL4_RHO_GRAIN)


# =============================================================================
# 3. 判定
# =============================================================================
def thresholds(ctx):
    """測線全体で見えるのに必要な差：Δ > thr_pos または Δ < thr_neg。"""
    e = ctx['e_boot']
    if not e.size:
        return np.nan, np.nan
    lo, hi = np.percentile(e, [(100 - CI_PCT) / 2, 100 - (100 - CI_PCT) / 2])
    return float(-lo), float(-hi)


def judge(ctx, D):
    """Δtanδ の配列 → ブロックの P_D と、測線全体の比 |Δ| / 必要な差。"""
    e, half = ctx['e_blk'], ctx['half']
    pd = np.full(D.shape, np.nan)
    ok = np.isfinite(D)
    pd[ok] = np.mean(np.abs(D[ok][..., None] + e) > half, axis=-1)
    tp, tn = ctx['thr']
    need = np.where(D >= 0, tp, -tn)
    ratio = np.where(ok, np.abs(D) / need, np.nan)
    return pd, ratio


def refl_need(ctx, z):
    """反射の地図から、上端・下端の両方が見える最小濃度（無ければ NaN）。"""
    R = ctx['refl']
    if R is None:
        return np.zeros_like(z)
    need = np.maximum(np.interp(z, R['z'], R['top'], left=np.nan, right=np.nan),
                      np.interp(z, R['z'], R['bottom'], left=np.nan, right=np.nan))
    return need


def make_maps(ctx):
    rec = ctx['rec']
    v0, z0, S0 = float(rec['ice_vol_frac']), float(rec['ice_top_m']), float(rec['feotio2_wt'])
    vS = v0 if S_MAP_V is None else float(S_MAP_V)
    ctx.update(v0=v0, z0=z0, S0=S0, vS=vS)
    ctx['delta0'] = delta_tand(rec, v0, z0, S0)
    ctx['thr'] = thresholds(ctx)
    cap = np.array([pore_capacity(rec, z) for z in Z_GRID])
    need = refl_need(ctx, Z_GRID)
    ctx['cap'], ctx['refl_need'] = cap, need

    # (v, z)
    Dvz = np.array([[delta_tand(rec, v, z, S0) for v in V_GRID] for z in Z_GRID])
    invalid = (V_GRID[None, :] > cap[:, None] + 1e-9)
    if ctx['refl'] is not None:
        invalid |= ~np.isfinite(need[:, None]) | (V_GRID[None, :] < need[:, None] - 1e-9)
    Dvz_m = np.where(invalid, np.nan, Dvz)
    pd, ratio = judge(ctx, Dvz_m)
    ctx['vz'] = {'D': Dvz, 'pd': pd, 'ratio': ratio, 'invalid': invalid}

    # (S, z)
    DSz = np.array([[delta_tand(rec, vS, z, S) for S in S_GRID] for z in Z_GRID])
    invS = np.repeat((vS > cap + 1e-9)[:, None], S_GRID.size, axis=1)
    if ctx['refl'] is not None:
        invS |= (~np.isfinite(need) | (vS < need - 1e-9))[:, None]
    pdS, ratioS = judge(ctx, np.where(invS, np.nan, DSz))
    ctx['Sz'] = {'D': DSz, 'pd': pdS, 'ratio': ratioS, 'invalid': invS}

    # figL2 用：深さごとの v の線、z0 での S の線（v0 と空隙満杯）
    ctx['gap_v'] = {z: np.array([delta_tand(rec, v, z, S0) for v in V_GRID]) for z in GAP_DEPTHS}
    vfull = pore_capacity(rec, z0)
    vfull = vfull if np.isfinite(vfull) else float(V_GRID.max())
    ctx['vfull'] = vfull
    ctx['gap_S'] = {vS: np.array([delta_tand(rec, vS, z0, S) for S in S_GRID]),
                    vfull: np.array([delta_tand(rec, vfull, z0, S) for S in S_GRID])}


# =============================================================================
# 4. 図
# =============================================================================
MAP_SPEC = {   # 地図の種類：(横軸の格子, 表示の倍率, 横軸ラベル, 縦の点線, gprMax の点のキー)
    'vz': (V_GRID, 100.0, 'Ice volume fraction [vol%]', (), ('v0', 100.0)),
    'Sz': (S_GRID, 1.0, 'FeO+TiO2 [wt%]', tuple(S_MARKS), ('S0', 1.0)),
}


def _ratio_vmax(arrays):
    """比の地図の色の上限（1 未満なら 1。1 = 見える境が色の上端に来る）。"""
    return max([1.0] + [float(np.nanmax(r)) for r in arrays if np.isfinite(r).any()])


def _draw(ctx, ax, key, which, rmax=None):
    """地図を 1 枚描く。which = 'a'（ブロックの P_D）/ 'b'（測線全体の比）。戻り値は imshow。"""
    xg, xs, xlab, marks, (mk, mscale) = MAP_SPEC[key]
    M = ctx[key]
    dx, dz = (xg[1] - xg[0]) * xs, Z_GRID[1] - Z_GRID[0]
    x = xg * xs
    ext = [x[0] - dx / 2, x[-1] + dx / 2, Z_GRID[-1] + dz / 2, Z_GRID[0] - dz / 2]
    if which == 'a':
        cm = plt.get_cmap('RdBu_r').copy()
        cm.set_bad('0.85')
        im = ax.imshow(np.ma.masked_invalid(M['pd']), aspect='auto', extent=ext, cmap=cm,
                       vmin=0, vmax=1, interpolation='nearest')
        if np.nanmax(M['pd']) >= 0.5 > np.nanmin(M['pd']):
            ax.contour(x, Z_GRID, np.ma.masked_invalid(M['pd']), levels=[0.5], colors='k',
                       linewidths=2)
    else:
        r = M['ratio']
        rmax = _ratio_vmax([r]) if rmax is None else rmax
        cm = plt.get_cmap('viridis').copy()
        cm.set_bad('0.85')
        im = ax.imshow(np.ma.masked_invalid(np.maximum(r, 1e-3)), aspect='auto', extent=ext,
                       cmap=cm, norm=LogNorm(vmin=1e-2, vmax=rmax), interpolation='nearest')
        if np.nanmax(r) >= 1.0:
            ax.contour(x, Z_GRID, np.ma.masked_invalid(r), levels=[1.0], colors='w', linewidths=2)
    if key == 'vz':
        # 空隙満杯の上限（反射の地図と同じ線）と、反射で上端・下端が見える下限（反射の地図の移動平均）
        ax.plot(100 * ctx['cap'], Z_GRID, color='0.35', lw=1.6, ls='--')
        R = ctx['refl']
        if R is not None:
            for k in ('top', 'bottom'):
                ax.plot(100 * np.interp(Z_GRID, R['z'], R[k], left=np.nan, right=np.nan), Z_GRID,
                        color=EV_COL['ice_' + k], lw=2.2)
    ax.set_facecolor('0.85')
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel(xlab)
    for m in marks:
        ax.axvline(m, color='k', lw=1.2, ls=':')
    ax.plot(ctx[mk] * mscale, ctx['z0'], marker='o', ms=11, color='white', mec='k', mew=1.8,
            ls='none')
    return im


CBAR_LABEL = {'a': 'P_D (block)', 'b': '|Δtanδ| / required'}


def _legend(fig, marks, ctx=None):
    """図の下の凡例。ctx を渡すと (v, z) 地図の線（空隙満杯・反射の下限）も入れる。"""
    h = [Line2D([], [], marker='o', ms=10, color='white', mec='k', mew=1.8, ls='none',
                label='gprMax'),
         Patch(color='0.85', label='not applicable')]
    if ctx is not None:
        h.append(Line2D([], [], color='0.35', lw=1.6, ls='--', label='pores full'))
        if ctx['refl'] is not None:
            h += [Line2D([], [], color=EV_COL[k], lw=2.2, label='detection limit: {}'.format(k))
                  for k in ('ice_top', 'ice_bottom')]
    if marks:
        h.append(Line2D([], [], color='k', lw=1.2, ls=':', label='{:g} wt%'.format(marks[0])))
    fig.legend(handles=h, loc='outside lower center', ncol=len(h))


def _map_panels(ctx, key, stem):
    """figL1 / figL3：(a) ブロックの P_D、(b) 測線全体の比。"""
    fig, axs = plt.subplots(1, 2, figsize=(16, 8.2), sharey=True, layout='constrained')
    for ax, w, t in zip(axs, 'ab', ('(a) block', '(b) whole line')):
        im = _draw(ctx, ax, key, w)
        fig.colorbar(im, ax=ax, label=CBAR_LABEL[w], shrink=0.9, pad=0.01)
        ax.set_title(t)
    axs[0].set_ylabel(DEPTH_LABEL)
    _legend(fig, MAP_SPEC[key][3], ctx if key == 'vz' else None)
    bd.save_figure(fig, ctx['out_dir'], stem)


def plot_compare(ctx):
    """figL4：figL1 と figL3 のパネルを並べる（FIG4_PANELS で選ぶ）。
    同じ種類のパネルなら色の範囲をそろえ、カラーバーを 1 本にする。"""
    pa, pb = FIG4_PANELS
    fig, axs = plt.subplots(1, 2, figsize=(16, 8.2), sharey=True, layout='constrained')
    rmax = _ratio_vmax([ctx['vz']['ratio'], ctx['Sz']['ratio']]) if pa == pb == 'b' else None
    ims = []
    for ax, key, w, t in zip(axs, ('vz', 'Sz'), (pa, pb), ('(a) ice content', '(b) composition')):
        ims.append(_draw(ctx, ax, key, w, rmax=rmax))
        ax.set_title(t)
    if pa == pb:
        fig.colorbar(ims[1], ax=list(axs), label=CBAR_LABEL[pa], shrink=0.9, pad=0.01)
    else:
        for ax, im, w in zip(axs, ims, (pa, pb)):
            fig.colorbar(im, ax=ax, label=CBAR_LABEL[w], shrink=0.9, pad=0.01)
    axs[0].set_ylabel(DEPTH_LABEL)
    _legend(fig, MAP_SPEC['Sz'][3], ctx)              # 左は (v, z) 地図なので、その線も凡例に入れる
    bd.save_figure(fig, ctx['out_dir'], 'figL4_lsr_map_compare')


def plot_maps(ctx):
    _map_panels(ctx, 'vz', 'figL1_lsr_map_ice')
    _map_panels(ctx, 'Sz', 'figL3_lsr_map_composition')
    if FIG4_PANELS:
        plot_compare(ctx)


def _gap_panel(ctx, ax, which, block=True):
    """信号 |Δtanδ| と見えるのに必要な差を 1 枚描く。which = 'v'（氷の濃度）/ 'S'（組成）。
    block = False なら、必要な差は測線全体の線だけ描く。"""
    tp, tn = ctx['thr']
    need_line = 0.5 * (tp - tn)
    if which == 'v':
        for (z, D), ls in zip(ctx['gap_v'].items(), DEPTH_LS):
            ax.plot(100 * V_GRID, np.abs(D), color='k', lw=2, ls=ls, label='z = {:g} m'.format(z))
            last = np.nonzero(np.isfinite(D))[0]
            if last.size:                                # 空隙満杯の点
                i = last[-1]
                ax.plot(100 * V_GRID[i], abs(D[i]), marker='s', ms=9, color='white', mec='k',
                        ls='none')
        ax.set_xlabel('Ice volume fraction [vol%]')
    else:
        for (v, D), ls in zip(ctx['gap_S'].items(), ('-', '--')):
            ax.plot(S_GRID, np.abs(D), color='k', lw=2, ls=ls,
                    label='{:.0f} vol%{}'.format(100 * v, ' (pores full)' if v == ctx['vfull'] else ''))
        for m in S_MARKS:
            ax.axvline(m, color='k', lw=1.2, ls=':')
        ax.set_xlabel('FeO+TiO2 [wt%]')
    if block:
        ax.axhline(ctx['half'], color='tab:red', lw=2, label='required (block)')
    if np.isfinite(need_line):
        ax.axhline(need_line, color='tab:blue', lw=2, label='required (whole line)')
    ax.set_yscale('log')
    data = list(ctx['gap_v'].values()) if which == 'v' else list(ctx['gap_S'].values())
    top = max([float(np.nanmax(np.abs(D))) for D in data if np.isfinite(D).any()]
              + [need_line if np.isfinite(need_line) else 0.0] + ([ctx['half']] if block else []))
    ax.set_ylim(GAP_YMIN, 2.0 * top)                 # 一番上の線が枠に重ならないよう少し余白を取る
    ax.legend(loc='lower right')


def plot_gap(ctx):
    """figL2：信号 |Δtanδ| と、見えるのに必要な差。"""
    fig, axs = plt.subplots(1, 2, figsize=(16, 7), sharey=True, layout='constrained')
    _gap_panel(ctx, axs[0], 'v')
    axs[0].set_title('(a) ice content (FeO+TiO2 {:g} wt%)'.format(ctx['S0']))
    _gap_panel(ctx, axs[1], 'S')
    axs[1].set_title('(b) composition (z = {:g} m)'.format(ctx['z0']))
    axs[0].set_ylabel('|Δtanδ|')
    bd.save_figure(fig, ctx['out_dir'], 'figL2_lsr_gap')


def plot_map_gap(ctx):
    """figL5：figL1 (b) と figL2 (a)、figL6：figL3 (b) と figL2 (b)。
    必要な差は測線全体の線だけ（地図の (b) と同じ判定）。"""
    # タイトルは「何を見せるか」（地図か、深さを決めた断面か）だけにする。共通の条件
    # （測線全体の判定、組成 S0）はパネルごとに書かない。左右で条件が違うと読まれないように。
    for key, which, stem, t_map, t_gap in (
            ('vz', 'v', 'figL5_lsr_ice_map_gap', '(a) Map', '(b) Profiles'),
            ('Sz', 'S', 'figL6_lsr_composition_map_gap', '(a) Map',
             '(b) Profiles at z = {:g} m'.format(ctx['z0']))):
        fig, axs = plt.subplots(1, 2, figsize=(16, 8.2), layout='constrained')
        im = _draw(ctx, axs[0], key, 'b')
        fig.colorbar(im, ax=axs[0], label=CBAR_LABEL['b'], shrink=0.9, pad=0.01)
        axs[0].set_title(t_map)
        axs[0].set_ylabel(DEPTH_LABEL)
        _gap_panel(ctx, axs[1], which, block=False)
        axs[1].set_title(t_gap)
        axs[1].set_ylabel('|Δtanδ|')
        _legend(fig, MAP_SPEC[key][3], ctx if key == 'vz' else None)
        bd.save_figure(fig, ctx['out_dir'], stem)


# =============================================================================
# 5. 出力
# =============================================================================
def _write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: ('{:.6g}'.format(v) if isinstance(v, float) else v) for k, v in r.items()})
    print('Saved:', path)


def write_outputs(ctx):
    od = ctx['out_dir']
    for key, xg, xname in (('vz', V_GRID, 'ice_vol_frac'), ('Sz', S_GRID, 'feotio2_wt')):
        M = ctx[key]
        rows = []
        for i, z in enumerate(Z_GRID):
            for j, x in enumerate(xg):
                rows.append({'ice_top_depth_m': float(z), xname: float(x), 'delta_tand': float(M['D'][i, j]),
                             'pd_block': float(M['pd'][i, j]), 'ratio_whole_line': float(M['ratio'][i, j]),
                             'applicable': not bool(M['invalid'][i, j])})
        _write_csv(os.path.join(od, 'lsr_map_{}.csv'.format('ice' if key == 'vz' else 'composition')), rows)

    tp, tn = ctx['thr']
    need_line = 0.5 * (tp - tn)
    t, ref = ctx['test'], ctx['ref']
    d_pt = float(np.median(ctx['d_boot'])) if ctx['d_boot'].size else np.nan
    sd_b = float(np.std(ctx['d_boot'], ddof=1)) if ctx['d_boot'].size > 1 else np.nan
    L = ['detectability_lsr.py  {}'.format(datetime.now().isoformat(timespec='seconds')),
         '対象: {}'.format(ctx['target']),
         '計算の条件: v0 = {:.0f} vol%、上端 z0 = {:g} m、厚さ D = {:g} m、FeO+TiO2 S0 = {:g} wt%、{}'.format(
             100 * ctx['v0'], ctx['z0'], float(ctx['rec']['ice_thick_m']), ctx['S0'], ctx['rec']['ice_model']),
         '', '[ばらつき（計算から）]',
         '  ブロック：上の層の範囲の半幅 {:.2e}（= 1 ブロックで見えるのに必要な差）。'
         '氷層のブロック {} 個の標準偏差 {:.2e}'.format(ctx['half'], ctx['e_blk'].size,
                                                 float(np.std(ctx['e_blk'], ddof=1))),
         '  測線全体：ブートストラップの差（氷層 − 上の層）{} 回、標準偏差 {:.2e}。'
         '見えるのに必要な差 +{:.2e} / {:.2e}（平均の大きさ {:.2e}）'.format(
             ctx['d_boot'].size, sd_b, tp, tn, need_line),
         '', '[計算した点での確認]',
         '  モデルの差 Δtanδ(v0, z0, S0) = {:+.2e}'.format(ctx['delta0']),
         '  計算の差（ブートストラップの中央値）       = {:+.2e}  → モデルとの差は {:.1f}σ'.format(
             d_pt, abs(d_pt - ctx['delta0']) / sd_b if sd_b > 0 else np.nan),
         '  （地図は「信号 = モデル」「ばらつき = 計算」で作る。この差は地図に入れていない）']

    def best(M):
        r = np.where(M['invalid'], np.nan, M['ratio'])
        p = np.where(M['invalid'], np.nan, M['pd'])
        return (float(np.nanmax(r)) if np.isfinite(r).any() else np.nan,
                float(np.nanmax(p)) if np.isfinite(p).any() else np.nan)

    rv, pv = best(ctx['vz'])
    rS, pS = best(ctx['Sz'])
    L += ['', '[(v, z) 地図]  組成 S0 = {:g} wt%'.format(ctx['S0']),
          '  使える領域での最大：ブロックの P_D {:.2f}、測線全体の比 {:.2f}（1 以上で見える）'.format(pv, rv)]
    for z, D in ctx['gap_v'].items():
        ok = np.isfinite(D)
        if ok.any():
            i = np.nonzero(ok)[0][-1]
            L.append('  上端 {:g} m：Δtanδ は v = 0 で {:+.2e}、空隙満杯 {:.0f} vol% で {:+.2e}'.format(
                z, D[0], 100 * V_GRID[i], D[i]))
    L += ['', '[(S, z) 地図]  氷 {:.0f} vol%'.format(100 * ctx['vS']),
          '  使える領域での最大：ブロックの P_D {:.2f}、測線全体の比 {:.2f}'.format(pS, rS)]
    for v, D in ctx['gap_S'].items():
        j0 = int(np.argmin(np.abs(S_GRID - ctx['S0'])))
        L.append('  z0 = {:g} m・{:.0f} vol%：|Δtanδ| は S = {:g} で {:.2e}、S = {:g} で {:.2e}'.format(
            ctx['z0'], 100 * v, S_GRID[j0], abs(D[j0]), S_GRID[-1], abs(D[-1])))
        # tanδ ∝ 10^(0.027 S) なので |Δ| も同じ割合で増える。必要な S の目安（外挿）
        for nm, need in (('測線全体', need_line), ('ブロック', ctx['half'])):
            if np.isfinite(need) and abs(D[j0]) > 0:
                S_req = S_GRID[j0] + np.log10(need / abs(D[j0])) / sm.LEVEL3_CARRIER_TAND_A
                L.append('      {}で見えるのに必要な S（|Δtanδ| ∝ 10^(0.027 S) で外挿）：約 {:.0f} wt%'.format(
                    nm, S_req))
    if ctx['refl'] is None:
        L += ['', '注意：反射の地図（newA_vmin.csv）が無いため、反射が見えない領域を灰色にしていない']
    else:
        L += ['', '灰色：空隙満杯を超える濃度、または上端・下端の反射のどちらかが見えない'
              '（反射の地図の v_min の{}を使用）'.format('移動平均' if ctx['refl']['col'] == 'vmin_smoothed'
                                                    else '各深さの値')]
    path = os.path.join(od, 'run_info.txt')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print('Saved:', path)
    print('\n'.join(L))


def main():
    ctx = load_inputs()
    ctx['out_dir'] = os.path.join(ctx['base'], OUTPUT_SUBDIR)
    os.makedirs(ctx['out_dir'], exist_ok=True)
    make_maps(ctx)
    plot_maps(ctx)
    plot_gap(ctx)
    if MAP_GAP_FIGS:
        plot_map_gap(ctx)
    write_outputs(ctx)


if __name__ == '__main__':
    main()