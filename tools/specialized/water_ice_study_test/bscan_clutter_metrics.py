"""クラッタの強さの指標の比較（bscan_detection の fig8）を、1 つの B-scan だけで作るツール

使い方：python bscan_clutter_metrics.py を実行し、merged.out のパスを貼る。
    - 想定は水氷なし（no_ice）の B-scan。シード（ランダム場）を変えた計算どうしを比べるために使う。
    - 前処理と図・CSV は bscan_detection.py と同じ（同じフォルダに置く）。
    - 参照計算を 2 つ使う（どちらも励振波形と格子が同じなら、どの計算でも共通）。
        at_tx（自由空間）：直達波の除去。地表反射は直達波と重なるので、除去しないと dB の基準
                          （地表反射の強さ）が変わり、他の結果と比べられなくなる
        far_1m           ：波源遅延。地表反射・計算領域の底の目印の走時に使う
      探す順：設定の AT_TX_PATH / FAR_1M_PATH → JSON_PATH の _reference（あれば）→ input() で聞く。
      at_tx が無ければ直達波を除去せずに続ける（注意を出す）。far_1m が無ければ SOURCE_DELAY_NS を使う。
    - 記録ファイル（Ascan_level6_record.json）は B-scan の近くから bscan_detection と同じ規則で探す。
      見つからなければ、計算領域の底の目印なしで続ける。

出力：<merged.out のフォルダ>/bscan_detection/clutter_metrics/
    fig8_clutter_metrics.png（と pdf/）、clutter_metrics.csv、run_info.txt（シード・ランダム場の指紋を含む）
図の見方は bscan_detection.md の fig8 を参照。
"""
import os
import sys
from datetime import datetime

import numpy as np

import bscan_detection as bd                    # 前処理・図・CSV を共有する
import subsurface_model as sm

# =============================================================================
# 設定  [EDIT HERE]
# =============================================================================
JSON_PATH = '/Volumes/gprMax_water_ice/2D_12x4/water_ice_study_test/out_file_paths.json'   # 無くてもよい
AT_TX_PATH = ''                                 # 自由空間 at_tx（直達波の除去）。空なら JSON → input()
FAR_1M_PATH = ''                                # far_1m（波源遅延）。空なら JSON → input()
SOURCE_DELAY_NS = 5.008                         # far_1m が無いときの波源遅延 [ns]（これまでの計算の値）
OUTPUT_SUBDIR = 'clutter_metrics'               # bscan_detection/ の下に作るフォルダ


def ask_path():
    p = input('Enter merged.out (B-scan) path: ').strip().strip("'\"").replace('\\ ', ' ')
    if not os.path.exists(p):
        sys.exit('Error: file not found: {}'.format(p))
    return os.path.abspath(p)


def _ask_optional(prompt):
    p = input(prompt).strip().strip("'\"").replace('\\ ', ' ')
    if p and not os.path.exists(p):
        print('  見つかりません（使わずに続けます）: {}'.format(p))
        return ''
    return os.path.abspath(p) if p else ''


def reference_paths():
    """at_tx（直達波の除去）と far_1m（波源遅延）のパス。設定 → JSON → input() の順に探す。"""
    fs, far = AT_TX_PATH, FAR_1M_PATH
    if (not fs or not far) and JSON_PATH and os.path.exists(JSON_PATH):
        try:
            _, _, _, reference = bd.select_conditions(JSON_PATH)
            fs = fs or bd.FREESPACE_AT_TX_PATH or reference.get(bd.FREESPACE_KEY, '')
            far = far or reference.get(bd.REF_KEY, '')
        except Exception as exc:
            print('注意: JSON から参照計算を読めませんでした: {}'.format(exc))
    if bd.SUBTRACT_FREESPACE and not (fs and os.path.exists(fs)):
        fs = _ask_optional('Enter free-space at_tx path for direct-wave removal (Enter to skip): ')
    if not (far and os.path.exists(far)):
        far = _ask_optional('Enter far_1m path for source delay (Enter to use {} ns): '.format(SOURCE_DELAY_NS))
    if not bd.SUBTRACT_FREESPACE:
        fs = ''
    return far, fs


def main():
    path = ask_path()
    far_path, fs_path = reference_paths()

    # --- 読み込みと前処理（bscan_detection と同じ）----------------------------
    R, dt = bd.read_bscan(path)
    if fs_path:
        fs, dt_fs = bd.load_trace(fs_path)
        if abs(dt_fs - dt) > 1e-9 * dt:
            sys.exit('Error: 自由空間 at_tx の dt が違います: {} vs {}'.format(dt_fs, dt))
        R = R - np.pad(fs, (0, max(0, R.shape[0] - len(fs))))[:R.shape[0], None]
    rec_path, rec = bd.find_record(path)
    x_all, has_x = bd.trace_positions(rec, R.shape[1])
    sel = np.ones(R.shape[1], dtype=bool)
    if bd.TRACE_X_RANGE_M is not None:
        sel = (x_all >= bd.TRACE_X_RANGE_M[0] - 1e-9) & (x_all <= bd.TRACE_X_RANGE_M[1] + 1e-9)
    R, x = R[:, sel], x_all[sel]

    # --- 走時の目印（モデル）と dB の基準 ---------------------------------------
    if far_path:
        far_tr, dt_far = bd.load_trace(far_path)
        src_delay = bd.measure_peak(far_tr, dt_far)['t_peak'] - sm.R_REF / sm.C
    else:
        src_delay = SOURCE_DELAY_NS
    t_bottom = np.inf
    if rec:
        try:
            bd.configure_mean_medium(rec)
            t_bottom = bd.domain_bottom_time(rec, src_delay)
        except Exception as exc:                      # 目印だけなので続ける
            print('注意: 計算領域の底の走時を計算できませんでした（目印なしで続行）: {}'.format(exc))
    t_axis = np.arange(R.shape[0]) * dt * 1e9
    t_s_model = 2.0 * sm.TX_HEIGHT / sm.C + src_delay
    m = t_axis <= t_s_model + 5.0
    t_s_data = float(t_axis[m][np.argmax(bd.envelope(R).mean(axis=1)[m])])
    ms = np.abs(t_axis - t_s_data) <= bd.REFL_GATE_HW_NS
    a_surf = float(np.median(np.abs(R)[ms].max(axis=0)))   # 判定と同じ量（±1 ns の |E| の最大値）
    res = {'a_surf': a_surf}
    info = {'t_surface_data': t_s_data, 't_surface_model': t_s_model, 't_bottom': t_bottom}

    # --- 図と CSV（bscan_detection と同じ関数）---------------------------------
    out_dir = os.path.join(os.path.dirname(path), bd.OUTPUT_SUBDIRNAME, OUTPUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    bd.plot_clutter_metrics(res, R, dt, out_dir, info)

    # --- run_info ---------------------------------------------------------------
    rnd = (rec or {}).get('random', {})
    L = ['{}  {}'.format(os.path.basename(__file__), datetime.now().isoformat(timespec='seconds')),
         'B-scan: {}'.format(path),
         '記録: {}'.format(rec_path or '(見つからない)'),
         'far_1m: {}'.format(far_path or '(なし。波源遅延は設定の {} ns)'.format(SOURCE_DELAY_NS)),
         'at_tx（直達波の除去）: {}'.format(fs_path or '(除去なし)'),
         'トレース {} 本（x = {:.2f}–{:.2f} {}）、dt = {:.4f} ps'.format(
             R.shape[1], x[0], x[-1], 'm' if has_x else '(番号)', dt * 1e12)]
    if rec:
        L += ['氷: enable_ice = {}, ice_vol_frac = {}'.format(rec.get('enable_ice'), rec.get('ice_vol_frac')),
              'ランダム場: seed = {}、相関長 x {} m / z {} m、相対ゆらぎ {}'.format(
                  rnd.get('seed'), rnd.get('corr_x_m'), rnd.get('corr_z_m'), rnd.get('rel_std')),
              '  R_sha256 = {}'.format(rnd.get('R_sha256')),
              '  rho_map_sha256 = {}'.format((rec.get('rho') or {}).get('rho_map_sha256'))]
        if bd.has_ice(rec):
            L.append('注意: 氷が入った計算です。fig8 は水氷なし（no_ice）での比較を想定しています')
    if not fs_path:
        L.append('注意: 直達波を除去していません。地表反射に直達波が重なるので、dB の基準（地表反射）が'
                 '除去した結果と違い、他の fig8 と縦軸の位置を直接は比べられません（形は比べられる）')
    L += ['波源遅延 {:.3f} ns、地表反射の走時：モデル {:.3f} ns / データ {:.3f} ns'.format(
              src_delay, t_s_model, t_s_data),
          '計算領域の底の走時（モデル）: {}'.format('{:.2f} ns'.format(t_bottom) if np.isfinite(t_bottom)
                                                 else '(なし)'),
          'dB の基準: 地表反射の ±{:g} ns の |E| の最大値（全トレースの中央値）= {:.4e}'.format(
              bd.REFL_GATE_HW_NS, a_surf)]
    p = os.path.join(out_dir, 'run_info.txt')
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('Saved:', p)


if __name__ == '__main__':
    bd.JSON_PATH = JSON_PATH
    main()