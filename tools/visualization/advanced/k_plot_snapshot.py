#!/usr/bin/env python3
"""gprMax のスナップショット（.vti）を地形に重ねて動画にする。

入力は input() で与える：geometry.vti のパス、スナップショットの時間間隔、拡大範囲、色の範囲。
同じフォルダの *_snaps フォルダから snapshot<番号>.vti を読み、
<geometry.vti のフォルダ>/snapshot.../ にフレーム画像（PNG）と動画（MP4）を書く。

地形の表示
    材料を「空気 / レゴリス / 氷層 / 岩石」の 4 種類に分けて塗り、その境界だけを線で描く。
    材料名は geometry.vti の末尾の <gprMax> ブロックから読む。
      空気   : free_space
      氷層   : 名前が _pore / _excess で終わる材料
      岩石   : rock
      レゴリス: それ以外
    Level 6 は密度ごとに材料が分かれていて数千種類あるため、材料ごとの境界は描かない。

スナップショットの格子
    スナップショットの間隔（#snapshot の dx dy）は地形の格子と違ってよい。
    Ez はスナップショット自身の格子で描く。

時刻
    snapshot<n>.vti の時刻を n × (入力した時間間隔) とする（Ascan.in の書き方と同じ）。
"""
import glob
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FormatStrFormatter
import mpl_toolkits.axes_grid1 as axgrid1
from vtkmodules.vtkIOXML import vtkXMLImageDataReader
from vtkmodules.vtkFiltersCore import vtkCellCenters
from vtkmodules.util.numpy_support import vtk_to_numpy

# 図の共通ルール
#   文字の大きさ：タイトル・軸ラベル 18、目盛り・凡例・図中の文字 14
#   グリッド線：特に意図がない限り全パネルに付ける（破線 '--'、alpha 0.3）
#   補助目盛り（minor ticks）：全パネルで表示する
plt.rcParams.update({'axes.titlesize': 18, 'axes.labelsize': 18, 'figure.titlesize': 18,
                     'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14,
                     'font.size': 14,
                     'axes.grid': True, 'axes.grid.which': 'major', 'axes.axisbelow': 'line',
                     'grid.linestyle': '--', 'grid.alpha': 0.3,
                     'xtick.minor.visible': True, 'ytick.minor.visible': True})

DEFAULT_VRANGE = 0.03          # Ez の色の範囲（最大値で割った値の ±）
EZ_FIELD = 'E-field'
FPS = 10
FRAME_DPI = 300
FIG_HEIGHT_IN = 8.0

CATEGORIES = ('air', 'regolith', 'ice', 'rock')
CATEGORY_COLORS = ('white', '0.70', '0.50', '0.15')


# =============================================================================
# 読み込み
# =============================================================================
def read_vti_image(filename):
    """VTI を読み、vtkImageData を返す。"""
    reader = vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def read_vti_image_safe(filename):
    """読めたかどうか、vtkImageData、エラー文を返す。"""
    try:
        output = read_vti_image(filename)
        if output is None or output.GetNumberOfPoints() == 0:
            return False, None, 'VTK reader returned empty data'
        return True, output, None
    except Exception as e:                       # noqa: BLE001
        return False, None, str(e)


def extract_slice(img, field_name, target_z=0.0, tol=1e-6):
    """target_z に最も近い面の (x, y) 座標と値（3 成分なら z 成分）を返す。"""
    pd = img.GetPointData()
    array = pd.GetArray(field_name)
    if array is None:
        array = img.GetCellData().GetArray(field_name)
        if array is None:
            raise KeyError(f"Field '{field_name}' not found.")
        center = vtkCellCenters()
        center.SetInputData(img)
        center.Update()
        mesh = center.GetOutput()
    else:
        mesh = img
    pts = vtk_to_numpy(mesh.GetPoints().GetData()).reshape(-1, 3)
    vals = vtk_to_numpy(array)
    uniq = np.unique(pts[:, 2])
    slice_z = uniq[np.argmin(np.abs(uniq - target_z))]
    mask = np.isclose(pts[:, 2], slice_z, atol=tol)
    vals2d = vals[mask]
    if vals2d.ndim > 1 and vals2d.shape[1] >= 3:
        vals2d = vals2d[:, 2]
    return pts[mask][:, :2], vals2d


def material_names(path):
    """geometry.vti の末尾の <gprMax> ブロックから {番号: 名前} を読む（無ければ空）。"""
    with open(path, 'rb') as fh:
        raw = fh.read()
    i = raw.rfind(b'</VTKFile>')
    if i < 0:
        return {}
    m = re.search(r'<gprMax>.*</gprMax>', raw[i:].decode('utf-8', errors='replace'), re.S)
    if not m:
        return {}
    out = {}
    for el in ET.fromstring(m.group(0)).findall('Material'):
        out[int(el.text)] = re.sub(r'\{[^}]*\}$', '', el.attrib['name'])   # 読み込み時の {ファイル名} を外す
    return out


def categorize(ids, names):
    """材料番号の配列 → 0 空気 / 1 レゴリス / 2 氷層 / 3 岩石。"""
    cat = np.ones(ids.shape, dtype=np.int8)
    if not names:                                # 名前が無ければ gprMax の既定（1 = free_space）
        cat[ids == 1] = 0
        return cat
    for k, nm in names.items():
        if nm in ('free_space', 'pec'):
            c = 0
        elif nm.endswith('_pore') or nm.endswith('_excess'):
            c = 2
        elif nm == 'rock':
            c = 3
        else:
            continue
        cat[ids == k] = c
    return cat


def to_grid(coords, vals):
    """散らばった (x, y, 値) を規則格子に並べる。戻り値: 格子 [y, x]、x、y、セル幅 (dx, dy)。"""
    xs, ys = np.unique(coords[:, 0]), np.unique(coords[:, 1])
    grid = np.zeros((ys.size, xs.size))
    grid[np.searchsorted(ys, coords[:, 1]), np.searchsorted(xs, coords[:, 0])] = vals
    dx = float(np.median(np.diff(xs))) if xs.size > 1 else 1.0
    dy = float(np.median(np.diff(ys))) if ys.size > 1 else 1.0
    return grid, xs, ys, (dx, dy)


def extent_cm(xs, ys, d):
    """セルの外縁までの範囲 [cm]（imshow の extent）。"""
    return [100 * (xs.min() - d[0] / 2), 100 * (xs.max() + d[0] / 2),
            100 * (ys.min() - d[1] / 2), 100 * (ys.max() + d[1] / 2)]


def snap_number(path):
    return int(os.path.splitext(os.path.basename(path))[0].replace('snapshot', ''))


# =============================================================================
# 本体
# =============================================================================
def main():
    print('[INFO] Starting k_plot_snapshot...')
    geometry_path = input('Enter path to geometry.vti: ').strip()
    if not os.path.isfile(geometry_path):
        raise FileNotFoundError(f'Cannot find file: {geometry_path}')
    parent_dir = os.path.dirname(geometry_path)

    dt_ns = float(input('Enter time step between snapshots [ns]: ').strip())

    do_zoom = input('Generate zoomed plot? (y/n): ').strip().lower() == 'y'
    if do_zoom:
        x_min = float(input('Enter zoom x_min [m]: ').strip())
        x_max = float(input('Enter zoom x_max [m]: ').strip())
        y_min = float(input('Enter zoom y_min [m]: ').strip())
        y_max = float(input('Enter zoom y_max [m]: ').strip())

    ans = input(f'Change Ez color range? Default: {DEFAULT_VRANGE}. (y/n, Enter = n): ').strip().lower()
    if ans == 'y':
        vrange = float(input('Enter max absolute value for Ez color range: ').strip())
    else:
        if ans not in ('', 'n'):
            print('[WARNING] Invalid input. Using default color range.')
        vrange = DEFAULT_VRANGE
    print(f'[INFO] Ez color range: ±{vrange}')

    snap_dirs = [d for d in os.listdir(parent_dir) if d.endswith('_snaps')]
    if len(snap_dirs) == 1:
        snap_dir = os.path.join(parent_dir, snap_dirs[0])
    else:
        snap_dir = input('Enter snapshot directory: ').strip()
    if not os.path.isdir(snap_dir):
        raise FileNotFoundError(f'Snap dir not found: {snap_dir}')
    print(f'[INFO] Snapshot directory: {snap_dir}')

    output_dir = os.path.join(parent_dir, 'snapshot')
    if do_zoom:
        output_dir = os.path.join(output_dir, f'zoom{x_min}_{x_max}_{y_min}_{y_max}')
    output_dir = output_dir + f'_vrange{vrange}'
    frame_dir = os.path.join(output_dir, 'snapshot_frames')
    os.makedirs(frame_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, 'snapshot_animation.mp4')

    # --- 地形 ----------------------------------------------------------------
    print('[INFO] Reading geometry...')
    geom_img = read_vti_image(geometry_path)
    arrs = geom_img.GetCellData() if geom_img.GetCellData().GetNumberOfArrays() else geom_img.GetPointData()
    if arrs.GetNumberOfArrays() == 0:
        raise RuntimeError('No data in geometry.vti')
    geom_coords, geom_ids = extract_slice(geom_img, arrs.GetArrayName(0))
    names = material_names(geometry_path)
    cat_vals = categorize(geom_ids.astype(int), names)
    geom_grid, gxs, gys, gd = to_grid(geom_coords, cat_vals)
    present = [CATEGORIES[c] for c in np.unique(cat_vals)]
    print(f'[INFO] Geometry {gxs.size} x {gys.size} cells, {len(names)} materials -> {present}')

    # --- スナップショット ------------------------------------------------------
    snap_paths = sorted(glob.glob(os.path.join(snap_dir, 'snapshot*.vti')), key=snap_number)
    print(f'[INFO] Found {len(snap_paths)} snapshot files.')
    if not snap_paths:
        raise FileNotFoundError('No snapshot*.vti found.')

    sizes = np.array([os.path.getsize(p) for p in snap_paths], dtype=float)
    for p, s in zip(snap_paths, sizes):
        if s == 0 or s < sizes.mean() - 2 * sizes.std():
            print(f'  [WARNING] {os.path.basename(p)} has unusual size: {int(s)} bytes '
                  f'(avg {sizes.mean():.0f})')

    # 正規化の最大値（先頭 20 枚）
    max_abs, failed = 0.0, []
    for idx, path in enumerate(snap_paths[:20], start=1):
        ok, img, msg = read_vti_image_safe(path)
        if not ok:
            failed.append((idx, path, msg))
            continue
        try:
            _, vals = extract_slice(img, EZ_FIELD)
            if vals.size:
                max_abs = max(max_abs, float(np.abs(vals).max()))
        except Exception as e:                   # noqa: BLE001
            failed.append((idx, path, str(e)))

    if failed:
        print(f'\n[ERROR] {len(failed)} frame(s) failed to load:')
        for idx, path, msg in failed:
            print(f'  Frame {idx}: {os.path.basename(path)} - {msg}')
        print('Options:\n  1. Stop here (recommended)\n  2. Process only frames before the first error'
              '\n  3. Skip failed frames (time gaps)\n  4. Abort')
        choice = input('Enter choice (1-4): ').strip()
        if choice == '2':
            snap_paths = snap_paths[:min(f[0] for f in failed) - 1]
        elif choice == '3':
            bad = {f[1] for f in failed}
            snap_paths = [p for p in snap_paths if p not in bad]
        else:
            print('[INFO] Stopped.')
            return

    if max_abs == 0.0:
        raise RuntimeError('No Ez data found in first 20 frames.')
    print(f'[INFO] max_abs for normalization: {max_abs:.3e}')

    # Ez の格子はスナップショット自身のもの（地形の格子と違ってよい）
    ok, img0, msg = read_vti_image_safe(snap_paths[0])
    if not ok:
        raise RuntimeError(f'Failed to load {snap_paths[0]}: {msg}')
    c0, v0 = extract_slice(img0, EZ_FIELD)
    ez0, sxs, sys_, sd = to_grid(c0, v0 / max_abs)
    print(f'[INFO] Snapshot grid {sxs.size} x {sys_.size} cells, spacing {sd[0]*100:.2f} cm')

    # --- 図 ------------------------------------------------------------------
    width = FIG_HEIGHT_IN * (gxs.max() - gxs.min()) / (gys.max() - gys.min())
    if do_zoom:
        width = FIG_HEIGHT_IN * (x_max - x_min) / (y_max - y_min)
    fig, ax = plt.subplots(figsize=(width + 1.5, FIG_HEIGHT_IN), dpi=120)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')

    gext = extent_cm(gxs, gys, gd)
    ax.imshow(geom_grid, extent=gext, origin='lower', cmap=ListedColormap(CATEGORY_COLORS),
              norm=BoundaryNorm(np.arange(-0.5, len(CATEGORIES)), len(CATEGORIES)),
              interpolation='nearest', zorder=0)
    ax.contour(100 * gxs, 100 * gys, geom_grid, levels=[0.5, 1.5, 2.5],
               colors='k', linewidths=0.8, zorder=1)

    vmin, vmax = -vrange, vrange
    ez_im = ax.imshow(ez0, extent=extent_cm(sxs, sys_, sd), origin='lower', cmap='seismic',
                      vmin=vmin, vmax=vmax, alpha=0.6, interpolation='nearest', zorder=2)
    if do_zoom:
        ax.set_xlim(100 * x_min, 100 * x_max)
        ax.set_ylim(100 * y_min, 100 * y_max)
    cax = axgrid1.make_axes_locatable(ax).append_axes('right', size='3%', pad=0.1)
    cbar = plt.colorbar(ez_im, cax=cax, ticks=[vmin, 0.0, vmax])
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar.set_label('Normalized Ez')
    cbar.ax.grid(False)

    iy = np.searchsorted(sys_, c0[:, 1])
    ix = np.searchsorted(sxs, c0[:, 0])

    def update(i):
        path = snap_paths[i]
        ok, img, msg = read_vti_image_safe(path)
        if not ok:
            raise RuntimeError(f'Failed to load snapshot file {path}: {msg}')
        coords, vals = extract_slice(img, EZ_FIELD)
        grid = np.zeros_like(ez0)
        if coords.shape == c0.shape:
            grid[iy, ix] = vals / max_abs
        else:                                    # 格子が違うファイルが混ざっていたとき
            grid, _, _, _ = to_grid(coords, vals / max_abs)
        ez_im.set_data(grid)
        ax.set_title(f'Time = {snap_number(path) * dt_ns:.3f} ns')
        frame_path = os.path.join(frame_dir, f'frame_{i + 1:03d}.png')
        fig.savefig(frame_path, dpi=FRAME_DPI)
        print(f'[INFO] Saved frame {i + 1}/{len(snap_paths)}')
        return [ez_im]

    print('[INFO] Starting animation...')
    ani = animation.FuncAnimation(fig, update, frames=len(snap_paths), blit=True)
    writer = animation.writers['ffmpeg'](fps=FPS, metadata={'artist': 'gprMax'})
    ani.save(output_video_path, writer=writer)
    print(f'[INFO] Saved animation to {output_video_path}')


if __name__ == '__main__':
    main()