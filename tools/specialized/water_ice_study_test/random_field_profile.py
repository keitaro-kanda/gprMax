#!/usr/bin/env python3
"""ランダム場（密度のゆらぎ）の深さプロファイル

目的：no_ice のクラッタが 1.9〜2.6 m（往復走時 30〜38 ns）で弱まらなくなる理由を切り分ける。
      その深さで密度のゆらぎがたまたま強ければ「ランダム場の偶然」、深さによらず一定なら
      「媒質の性質（遅れて届く散乱波など）」の可能性が高い。

入力：gprMax の地形ファイルのパスを input() で渡す（どちらでもよい）
    - .h5  ：Ascan.in が書く level6_voxels.h5（地下だけ）、または #geometry_objects_write の
              geometry.h5（空気を含む全域）。材料は同じフォルダの <名前>_materials.txt から読む
    - .vti ：#geometry_view（'n'）の geometry.vti。材料名は .vti の末尾の <gprMax> ブロックから読む
    no_ice の計算のファイルを使う（氷層の材料が入っていても密度は読めるが、反射は氷を含む）。

読み方の前提（Ascan.in の書き方）
    材料名 r{Q:04d}_dry（氷層は _pore / _excess）の Q が量子化した密度：ρ = Q × RHO_STEP。
    比誘電率の実部は、材料ファイルがあればその Debye 極から F_HZ で計算し、無ければ
    Carrier の ε′ = 1.871^ρ で求める。

出力（入力ファイルと同じフォルダの random_field_profile/）
    fig_random_field_profile.png（と pdf/）：深さごとに
        (a) 密度の水平平均 ρ̄(z)
        (b) 密度の相対ゆらぎ σ_x(ρ)/ρ̄ [%]（各深さで水平方向の標準偏差 ÷ 平均）
        (c) 隣り合うセルの境界の反射係数の RMS [dB]（各深さで水平方向に 2 乗平均の平方根）
        細線：セルごと、太線：深さ ±SMOOTH_HW_M の移動平均。灰色の帯：HIGHLIGHT_DEPTH_M。
        右の縦軸：往復走時の目安（水平平均の ε′ と TX_HEIGHT_M・SOURCE_DELAY_NS から）
    random_field_profile.csv：深さごとの値
    画面：深さの区間ごとの平均（基準の区間との比 [dB]）
式の説明は random_field_profile.md。
"""
import os
import re
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
RHO_STEP = 0.001                  # [g/cm^3] 密度の量子化刻み（Ascan.in の RHO_STEP と同じ）
F_HZ = 1.25e9                     # [Hz] 材料ファイルの Debye 極から ε′ を求める周波数
X_RANGE_M = None                  # 水平方向に使う範囲 (x0, x1) [m]。None なら全部
DEPTH_MAX_M = None                # 描く深さの下限 [m]。None なら地下の底まで
SMOOTH_HW_M = 0.10                # 太線の移動平均の半幅 [m]（パルスの長さ程度）
HIGHLIGHT_DEPTH_M = (1.9, 2.55)   # 灰色の帯（クラッタが平らになる深さ）。None なら描かない
BAND_DEPTHS_M = ((0.5, 1.9), (1.9, 2.55), (2.55, 2.95))   # 画面に出す区間（最初が基準）
TX_HEIGHT_M = 0.35                # [m] アンテナの高さ（右の縦軸の走時の目安に使う）
SOURCE_DELAY_NS = 5.008           # [ns] 波源遅延（bscan_detection の run_info の値）
C0 = 299792458.0

NAME_RE = re.compile(r'^r(\d+)_(dry|pore|excess)')


# =============================================================================
# 1. 入力
# =============================================================================
def ask_path():
    p = input('Enter geometry .h5 or .vti file path: ').strip().strip("'\"").replace('\\ ', ' ')
    if not os.path.exists(p):
        sys.exit('Error: file not found: {}'.format(p))
    p = os.path.abspath(p)
    if os.path.splitext(p)[1].lower() not in ('.h5', '.vti'):
        sys.exit('Error: .h5 か .vti を指定してください: {}'.format(p))
    return p


def clean_name(name):
    """gprMax が読み込み時に付ける {ファイル名} を外す（例 r1650_dry{level6_voxels_materials}）。"""
    return re.sub(r'\{.*\}$', '', name)


def parse_materials(path):
    """#material / #add_dispersion_debye を読む。戻り値：[(名前, ε∞, [(Δε, τ), ...]), ...]（ファイル順）。"""
    mats, idx = [], {}
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            s = line.strip()
            if s.startswith('#material:'):
                v = s.split()
                idx[v[5]] = len(mats)
                mats.append([v[5], float(v[1]), []])
            elif s.startswith('#add_dispersion_debye:'):
                v = s.split()
                n = int(float(v[1]))
                poles = [(float(v[2 + 2 * k]), float(v[3 + 2 * k])) for k in range(n)]
                for name in v[2 + 2 * n:]:
                    if name in idx:
                        mats[idx[name]][2] = poles
    return mats


def eps_real(eps_inf, poles, f):
    w = 2.0 * np.pi * f
    return eps_inf + sum(de / (1.0 + (w * tau) ** 2) for de, tau in poles)


def read_h5(path):
    """gprMax の地形 h5（data[x, y, z]、y は上向き）と、同じフォルダの <名前>_materials.txt。"""
    with h5py.File(path, 'r') as fh:
        data = np.asarray(fh['data'][:, :, 0])
        dx = float(np.atleast_1d(fh.attrs['dx_dy_dz'])[0])
    mat_path = os.path.splitext(path)[0] + '_materials.txt'
    if not os.path.exists(mat_path):
        sys.exit('Error: 材料ファイルが見つかりません: {}'.format(mat_path))
    mats = parse_materials(mat_path)
    names = [clean_name(m[0]) for m in mats]
    eps = [eps_real(m[1], m[2], F_HZ) for m in mats]
    return data, dx, 0.0, names, eps, mat_path


def read_vti(path):
    """gprMax の .vti（Material 配列、格子、材料名）。k_visualize_vtk_2D.py と同じ読み方。"""
    raw = open(path, 'rb').read()
    head_end = raw.index(b'<AppendedData')
    header = raw[:head_end].decode('utf-8', errors='replace')
    ext = [int(v) for v in re.search(r'WholeExtent="([^"]+)"', header).group(1).split()]
    nx, ny, nz = ext[1] - ext[0], ext[3] - ext[2], ext[5] - ext[4]
    spacing = [float(v) for v in re.search(r'Spacing="([^"]+)"', header).group(1).split()]
    origin = [float(v) for v in re.search(r'Origin="([^"]+)"', header).group(1).split()]
    htype = re.search(r'header_type="([^"]+)"', header)
    hdt = np.dtype('<u8') if (htype and htype.group(1) == 'UInt64') else np.dtype('<u4')
    m = re.search(r'<DataArray type="UInt32" Name="Material" format="appended" offset="(\d+)"', header)
    start = raw.index(b'_', raw.index(b'>', head_end)) + 1 + int(m.group(1))
    nbytes = int(np.frombuffer(raw[start:start + hdt.itemsize], hdt)[0])
    arr = np.frombuffer(raw[start + hdt.itemsize:start + hdt.itemsize + nbytes], dtype='<u4')
    material = arr.reshape((nz, ny, nx)).transpose(2, 1, 0)[:, :, 0]     # [x, y]
    ids = {}
    tail = raw[raw.index(b'</VTKFile>'):].decode('utf-8', errors='replace')
    gm = re.search(r'<gprMax>.*</gprMax>', tail, re.S)
    if gm:
        for el in ET.fromstring(gm.group(0)).findall('Material'):
            ids[el.attrib['name']] = int(el.text)
    if not ids:
        sys.exit('Error: .vti に材料名（<gprMax> ブロック）がありません')
    n_ids = max(ids.values()) + 1
    names = [''] * n_ids
    for nm, i in ids.items():
        names[i] = clean_name(nm)
    # ε′：同じ名前の材料ファイル（<名前>_materials.txt）があれば Debye 極から、無ければ密度から
    eps = [np.nan] * n_ids
    mat_path = os.path.splitext(path)[0] + '_materials.txt'
    if os.path.exists(mat_path):
        props = {clean_name(mm[0]): eps_real(mm[1], mm[2], F_HZ) for mm in parse_materials(mat_path)}
        eps = [props.get(nm, np.nan) for nm in names]
    else:
        mat_path = None
    return material, float(spacing[0]), float(origin[0]), names, eps, mat_path


# =============================================================================
# 2. 深さごとの値
# =============================================================================
def build_maps(data, dx, x0, names, eps):
    """材料番号の地図 → 密度と ε′ の地図（行 = 深さ、上から下）。空気は NaN。"""
    n = len(names)
    rho_of = np.full(n, np.nan)
    for i, nm in enumerate(names):
        mm = NAME_RE.match(nm)
        if mm:
            rho_of[i] = int(mm.group(1)) * RHO_STEP
    if not np.isfinite(rho_of).any():
        sys.exit('Error: 材料名 r####_dry などが見つかりません（密度を読めない）。例: {}'.format(names[:5]))
    eps_of = np.array(eps, dtype=float)
    miss = ~np.isfinite(eps_of) & np.isfinite(rho_of)
    eps_of[miss] = 1.871 ** rho_of[miss]                    # 材料ファイルが無いときは Carrier の式
    img = data[:, ::-1].T                                    # [深さ（上から）, x]
    rho = rho_of[img]
    ep = eps_of[img]
    solid = np.isfinite(rho).any(axis=1)
    top = int(np.argmax(solid))                              # 地表のすぐ下の行
    rho, ep = rho[top:], ep[top:]
    x = x0 + (np.arange(rho.shape[1]) + 0.5) * dx
    if X_RANGE_M is not None:
        keep = (x >= X_RANGE_M[0]) & (x <= X_RANGE_M[1])
        rho, ep, x = rho[:, keep], ep[:, keep], x[keep]
    z = (np.arange(rho.shape[0]) + 0.5) * dx
    if DEPTH_MAX_M is not None:
        keep = z <= DEPTH_MAX_M
        rho, ep, z = rho[keep], ep[keep], z[keep]
    return z, x, rho, ep, bool(miss.any())


def profiles(z, rho, ep, dx):
    """深さごとの水平平均・相対ゆらぎ・境界の反射係数の RMS。"""
    rho_m = np.nanmean(rho, axis=1)
    rel = np.nanstd(rho, axis=1) / rho_m
    s = np.sqrt(ep)
    r = (s[1:] - s[:-1]) / (s[1:] + s[:-1])                  # 上から下へ、隣り合うセルの境界
    r_rms = np.sqrt(np.nanmean(r * r, axis=1))
    z_b = 0.5 * (z[1:] + z[:-1])
    n = max(1, int(round(SMOOTH_HW_M / dx)))
    sm = lambda y: np.array([np.nanmean(y[max(0, i - n):i + n + 1]) for i in range(len(y))])
    eps_m = np.nanmean(ep, axis=1)
    t = SOURCE_DELAY_NS + 1e9 * (2.0 * TX_HEIGHT_M + 2.0 * np.cumsum(np.sqrt(eps_m)) * dx) / C0
    return {'rho': rho_m, 'rel': rel, 'rel_s': sm(rel), 'z_b': z_b,
            'r_rms': r_rms, 'r_rms_s': np.sqrt(sm(r_rms ** 2)), 'eps': eps_m, 't': t}


# =============================================================================
# 3. 図と出力
# =============================================================================
def plot(z, P, out_dir):
    fig, axs = plt.subplots(1, 3, figsize=(17, 9), sharey=True, layout='constrained')
    col = 'k'
    axs[0].plot(P['rho'], z, color=col, lw=1.8)
    axs[0].set_xlabel('Mean density [g/cm³]')
    axs[0].set_title('(a) mean density')
    axs[1].plot(100 * P['rel'], z, color='0.6', lw=0.7)
    axs[1].plot(100 * P['rel_s'], z, color=col, lw=2.2)
    axs[1].set_xlabel('σ_x(ρ) / mean ρ [%]')
    axs[1].set_title('(b) density fluctuation')
    db = lambda v: 20.0 * np.log10(v)
    axs[2].plot(db(P['r_rms']), P['z_b'], color='0.6', lw=0.7)
    axs[2].plot(db(P['r_rms_s']), P['z_b'], color=col, lw=2.2)
    axs[2].set_xlabel('RMS reflection coefficient [dB]')
    axs[2].set_title('(c) cell-to-cell reflection')
    for ax in axs:
        if HIGHLIGHT_DEPTH_M:
            ax.axhspan(*HIGHLIGHT_DEPTH_M, color='0.85', zorder=0)
    axs[0].set_ylim(z[-1], 0.0)
    axs[0].set_ylabel('Depth below surface [m]')
    # 右の縦軸：往復走時の目安（深さ → 走時は単調なので補間で行き来する）
    zt = np.concatenate([[0.0], z])
    tt = np.concatenate([[SOURCE_DELAY_NS + 1e9 * 2.0 * TX_HEIGHT_M / C0], P['t']])
    sec = axs[2].secondary_yaxis('right', functions=(lambda d: np.interp(d, zt, tt),
                                                     lambda t: np.interp(t, tt, zt)))
    sec.set_ylabel('Two-way time (approx.) [ns]')
    h = [Line2D([], [], color='0.6', lw=0.7, label='each cell'),
         Line2D([], [], color=col, lw=2.2, label='±{:g} m average'.format(SMOOTH_HW_M))]
    if HIGHLIGHT_DEPTH_M:
        h.append(Patch(color='0.85', label='{:g}–{:g} m'.format(*HIGHLIGHT_DEPTH_M)))
    fig.legend(handles=h, loc='outside lower center', ncol=len(h))
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        d = out_dir if ext == 'png' else os.path.join(out_dir, ext)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, 'fig_random_field_profile.{}'.format(ext))
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print('Saved:', p)
    plt.close(fig)


def summary(z, P):
    """区間ごとの平均。最初の区間を基準に dB で比べる。"""
    L = ['', '[区間ごとの平均]  （比は最初の区間に対する値）']
    base = None
    for a, b in BAND_DEPTHS_M:
        m = (z >= a) & (z < b)
        mb = (P['z_b'] >= a) & (P['z_b'] < b)
        if not m.any():
            continue
        rel = float(np.nanmean(P['rel'][m]))
        rr = float(np.sqrt(np.nanmean(P['r_rms'][mb] ** 2)))
        if base is None:
            base = (rel, rr)
        L.append('  {:.2f}–{:.2f} m（約 {:.1f}–{:.1f} ns）：密度の相対ゆらぎ {:.2f} %（{:+.2f} dB）、'
                 '反射係数の RMS {:.2f} dB（{:+.2f} dB）'.format(
                     a, b, float(np.interp(a, z, P['t'])), float(np.interp(b, z, P['t'])),
                     100 * rel, 20 * np.log10(rel / base[0]), 20 * np.log10(rr),
                     20 * np.log10(rr / base[1])))
    return L


def main():
    path = ask_path()
    if path.lower().endswith('.h5'):
        data, dx, x0, names, eps, mat_path = read_h5(path)
    else:
        data, dx, x0, names, eps, mat_path = read_vti(path)
    z, x, rho, ep, used_carrier = build_maps(data, dx, x0, names, eps)
    P = profiles(z, rho, ep, dx)
    out_dir = os.path.join(os.path.dirname(path), 'random_field_profile')
    L = ['入力: {}'.format(path),
         '材料ファイル: {}'.format(mat_path or '(なし)'),
         '格子 {:.4g} m、地下 {} 行 × {} 列（x = {:.2f}–{:.2f} m）、深さ 0–{:.3f} m'.format(
             dx, rho.shape[0], rho.shape[1], x[0], x[-1], z[-1] + 0.5 * dx),
         'ε′ の求め方: {}'.format('Carrier の 1.871^ρ（材料ファイルに無い材料）' if used_carrier
                                 else '材料ファイルの Debye 極（{:.3g} Hz）'.format(F_HZ)),
         '右の縦軸の走時：アンテナ高さ {:g} m、波源遅延 {:g} ns、水平平均の ε′ の位相速度（目安）'.format(
             TX_HEIGHT_M, SOURCE_DELAY_NS)]
    L += summary(z, P)
    print('\n'.join(L))
    plot(z, P, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'random_field_profile.csv'), 'w', encoding='utf-8') as fh:
        fh.write('depth_m,two_way_time_ns,mean_rho,mean_eps,rho_rel_std,rho_rel_std_smoothed,'
                 'refl_rms_below\n')
        rb = np.append(P['r_rms'], np.nan)
        for i in range(len(z)):
            fh.write('{:.5f},{:.4f},{:.6f},{:.6f},{:.6g},{:.6g},{:.6g}\n'.format(
                z[i], P['t'][i], P['rho'][i], P['eps'][i], P['rel'][i], P['rel_s'][i], rb[i]))
    with open(os.path.join(out_dir, 'run_info.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print('Saved:', os.path.join(out_dir, 'random_field_profile.csv'))


if __name__ == '__main__':
    main()
