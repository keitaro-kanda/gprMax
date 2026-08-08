"""LUPEX GPR 帯域 (0.5-2.0 GHz) を模擬する gprMax 用 excitation file を生成する。

背景
----
gprMax の #waveform で gaussiandot 等を指定すると、

  * スペクトルのピークが指定した中心周波数からずれる
  * 帯域が連続的に広く伸び、2 GHz より上に無視できないパワーが残る

という問題がある。実際、gaussiandot@1.25 GHz では放射スペクトルの
27.5% のパワーが 2 GHz より上にあり、これが FDTD の数値分散
（PPW が小さい高周波ほど遅れる）を通じてパルスを広げ、
振幅ピークを押し下げていた。

このスクリプトは「放射電界」が指定帯域で平坦になるように
逆算した「電流」波形を作り、gprMax の #excitation_file 形式で書き出す。

放射電界と電流の関係
--------------------
#hertzian_dipole は電流密度源 J_s = I dl / (dx dy dz) であり、
2D では不変方向の線電流源として働く。放射電界は

    E(f) ∝ I(f) × f^P

  * f^1     : 電流 → 放射電界で時間微分が 1 回入る
  * f^-0.5  : 2D Green 関数 H0^(2)(kr) の遠方場 ~ 1/sqrt(kr)
  → P = 0.5

P は理論値だが、calibrate_exponent() で既存の自由空間計算から
実測して検証できる（推奨。§使い方 を参照）。

使い方
------
1. まず P を検証する（1 回だけ）:

     python make_excitation_file.py --calibrate /path/to/free_space_test/result/Ascan.out

   gaussiandot@1.25 GHz を入力としたときの |E_ref(f)/I(f)| を
   べき乗則でフィットし、P の実測値を表示する。0.5 付近になるはず。

2. 波形を生成する（出力ディレクトリは実行時に対話的に尋ねられる）:

     python make_excitation_file.py
         出力ディレクトリを入力してください [Enter で "."] > /path/to/water_ice_study_test/source

3. .in ファイルで使う:

     #excitation_file: /path/to/lupex_src.txt
     #hertzian_dipole: z 1.5 3.35 0 lupex_src

   （#waveform は不要）

注意
----
* 生成した波形を使ったら、自由空間参照計算 (free_space_test) を
  必ず取り直すこと。ascan_amplitude.py は E_ref(f) を実測から
  校正するため、波源を変えたらコード修正は不要だが参照は必要。
* time 列を含めているので、dx（したがって dt）を変えても
  同じファイルを使い回せる。
"""

import argparse
import os

import numpy as np

# NumPy 2.0 で np.trapz が np.trapezoid に改名された。どちらの版でも動くようにする。
# （数値計算の中身は同一。既存の k_centroid_freq_ms_diff.py と同じ方針。）
_TRAPZ = getattr(np, 'trapezoid', None) or np.trapz

# =============================================================================
# [EDIT HERE] 設計パラメータ
# =============================================================================
OUT_FILENAME = 'lupex_src.txt'          # 出力 ASCII ファイル名
PLOT_FILENAME = 'lupex_src_diagnostics.png'   # 診断プロットのファイル名
DEFAULT_OUTDIR = '.'            # 出力先の既定値（実行時に Enter だけ押すとこれを使う）
WAVEFORM_ID = 'lupex_src'       # #hertzian_dipole から参照する識別子（空白不可）

F_LO = 0.5e9                    # [Hz] 帯域下端
F_HI = 2.0e9                    # [Hz] 帯域上端
TUKEY_ALPHA = 0.2               # 帯域端のテーパー幅の割合 (0=矩形, 1=Hann)
                                #   大きくすると時間領域のリンギングは減るが
                                #   sigma_f^2 が下がり水氷検出感度も下がる。
                                #   0.1-0.2 推奨。

EXPONENT_P = 0.5                # 放射電界 E(f) ∝ I(f) * f^P の P
                                #   --calibrate で実測値を確認してから設定する

T_CENTER = 5.0e-9               # [s] パルス中心時刻
T_HALF_WIDTH = 4.5e-9           # [s] 時間窓の半幅。t=0 と t=2*T_CENTER で厳密に
                                #   ゼロになるよう波形を切り出す（不連続の注入を防ぐ）
TIME_TAPER = 0.3                # 時間窓の Tukey テーパー割合。大きいほど帯域外
                                #   漏れが減るが、帯域端がなまる
T_WINDOW = 50e-9                # [s] 書き出す時間長（モデルの #time_window 以上）
DT_FILE = 5e-12                 # [s] ファイル内の時間刻み。モデルの dt と
                                #   一致していなくてよい（gprMax が補間する）

PEAK_CURRENT = 1.0              # [A] 電流波形のピーク振幅

# 診断用
NFFT = 2 ** 18
LEAK_REF_LO, LEAK_REF_HI = 0.4e9, 2.2e9   # 帯域外漏れを評価する外側の境界


# =============================================================================
# 設計
# =============================================================================
def tukey_band(f, f_lo, f_hi, alpha):
    """[f_lo, f_hi] で 1、両端を Tukey テーパーで落とす帯域窓。"""
    w = np.zeros_like(f)
    inside = (f >= f_lo) & (f <= f_hi)
    if not np.any(inside):
        raise ValueError('帯域内に周波数点がありません。NFFT か DT_FILE を見直してください。')
    x = (f[inside] - f_lo) / (f_hi - f_lo)
    taper = np.ones_like(x)
    if alpha > 0:
        e = alpha / 2.0
        lo = x < e
        hi = x > 1.0 - e
        taper[lo] = 0.5 * (1.0 + np.cos(np.pi * (x[lo] / e - 1.0)))
        taper[hi] = 0.5 * (1.0 + np.cos(np.pi * ((x[hi] - (1.0 - e)) / e)))
    w[inside] = taper
    return w


def _time_window(t, t_center, half_width, alpha):
    """[t_center ± half_width] の外で厳密に 0 になる Tukey 時間窓。"""
    w = np.zeros_like(t)
    m = np.abs(t - t_center) <= half_width
    x = (t[m] - (t_center - half_width)) / (2.0 * half_width)
    taper = np.ones_like(x)
    if alpha > 0:
        e = alpha / 2.0
        lo, hi = x < e, x > 1.0 - e
        taper[lo] = 0.5 * (1.0 + np.cos(np.pi * (x[lo] / e - 1.0)))
        taper[hi] = 0.5 * (1.0 + np.cos(np.pi * ((x[hi] - (1.0 - e)) / e)))
    w[m] = taper
    return w


def design_current(f_lo=F_LO, f_hi=F_HI, alpha=TUKEY_ALPHA, p=EXPONENT_P,
                   t_center=T_CENTER, dt=DT_FILE, nfft=NFFT):
    """放射電界が帯域内で平坦になるような電流波形を設計する。

    E(f) ∝ I(f) * f^P としたいので、目標 E(f) = W(f) に対し
    I(f) = W(f) / f^P（ゼロ位相、t_center へ平行移動）とする。

    Returns
    -------
    t [s], current [A], f [Hz], E_pred(f) 予測放射スペクトル（振幅、任意単位）
    """
    f = np.fft.rfftfreq(nfft, d=dt)
    w_band = tukey_band(f, f_lo, f_hi, alpha)

    # f^-P による事前補償。帯域外は W=0 なので f=0 の発散は起きないが念のため保護
    with np.errstate(divide='ignore', invalid='ignore'):
        comp = np.where(f > 0, np.power(np.maximum(f, 1e-30), -p), 0.0)
    spec = w_band * comp

    # ゼロ位相の対称パルスを t_center に置く
    spec = spec * np.exp(-2j * np.pi * f * t_center)

    current = np.fft.irfft(spec, n=nfft)
    t = np.arange(nfft) * dt

    # 時間窓：帯域制限波形はサイドローブの減衰が遅く、t=0 で振幅が残ると
    # そこが不連続点になって広帯域の過渡応答を注入してしまう。
    # [t_center - T_HALF_WIDTH, t_center + T_HALF_WIDTH] で Tukey 窓をかけ、
    # 両端で厳密にゼロにする。帯域外漏れは report() で確認すること。
    current = current * _time_window(t, t_center, T_HALF_WIDTH, TIME_TAPER)

    current = current / np.max(np.abs(current)) * PEAK_CURRENT

    # 予測される放射スペクトル（振幅）
    e_pred = np.abs(np.fft.rfft(current)) * np.where(f > 0, np.power(np.maximum(f, 1e-30), p), 0.0)
    return t, current, f, e_pred


def spectral_moments(f, amp, f_lo=0.25e9, f_hi=6.0e9):
    """パワースペクトルの重心と分散。sigma_f^2 は水氷検出感度に直結する。"""
    m = (f >= f_lo) & (f <= f_hi)
    ff, p = f[m] / 1e9, amp[m] ** 2
    total = _TRAPZ(p, ff)
    fc = _TRAPZ(ff * p, ff) / total
    var = _TRAPZ((ff - fc) ** 2 * p, ff) / total
    return fc, var


def report(t, current, f, e_pred):
    """設計結果の診断を表示する。"""
    fc, var = spectral_moments(f, e_pred)
    # 平坦設計では単一のピークが定義できないため -3 dB 帯域で報告する
    ok = e_pred >= e_pred.max() / np.sqrt(2.0)
    f3lo, f3hi = f[ok].min() / 1e9, f[ok].max() / 1e9

    p_all = _TRAPZ(e_pred ** 2, f)
    out_hi = _TRAPZ(e_pred[f > LEAK_REF_HI] ** 2, f[f > LEAK_REF_HI])
    out_lo = _TRAPZ(e_pred[f < LEAK_REF_LO] ** 2, f[f < LEAK_REF_LO])
    leak = (out_hi + out_lo) / p_all

    pre = max(np.max(np.abs(current[t < 0.2e-9])),
              np.max(np.abs(current[(t > 2 * T_CENTER - 0.2e-9) & (t < 2 * T_CENTER)])))
    pk = np.max(np.abs(current))

    print('--- 設計結果 ---')
    print('  帯域            : {:.2f} - {:.2f} GHz (Tukey alpha={})'.format(
        F_LO / 1e9, F_HI / 1e9, TUKEY_ALPHA))
    print('  事前補償の指数 P: {:.2f}  (E ∝ I * f^P)'.format(EXPONENT_P))
    print('  放射スペクトル  : -3 dB 帯域 {:.3f} - {:.3f} GHz / 重心 {:.3f} GHz'.format(
        f3lo, f3hi, fc))
    print('  sigma_f^2       : {:.4f} GHz^2  (感度 ∝ この値)'.format(var))
    print('  帯域外漏れ      : {:.3e}  ({:.1f} dB)'.format(leak, 10 * np.log10(leak + 1e-30)))
    print('  時間窓端の残留   : {:.1f} dB (ピーク比)'.format(20 * np.log10(pre / pk + 1e-30)))
    if 20 * np.log10(pre / pk + 1e-30) > -60:
        print('  ** 警告: 端の残留が大きすぎます。T_HALF_WIDTH か TIME_TAPER を見直してください。')
    if 10 * np.log10(leak + 1e-30) > -40:
        print('  ** 警告: 帯域外漏れが大きすぎます。TIME_TAPER を大きくしてください。')


def write_excitation_file(path, waveform_id, t, current,
                          t_window=T_WINDOW, dt=DT_FILE):
    """gprMax の #excitation_file 形式（ASCII）で書き出す。

    1 行目 : 識別子（先頭列は time 固定）
    2 行目以降 : 時刻と振幅

    time 列を入れておくと gprMax が scipy.interpolate.interp1d で
    モデルの dt に補間するため、dx を変えても同じファイルが使える。
    """
    n = int(round(t_window / dt)) + 1
    if n > len(t):
        raise ValueError('NFFT が小さすぎます。T_WINDOW/DT_FILE = {} 点必要です。'.format(n))

    with open(path, 'w') as fh:
        fh.write('time {}\n'.format(waveform_id))
        for i in range(n):
            fh.write('{:.6e} {:.9e}\n'.format(t[i], current[i]))
    print('\n書き出し: {}  ({} 行, 0 - {:.1f} ns)'.format(path, n, t[n - 1] * 1e9))


# =============================================================================
# P の実測校正（既存の自由空間計算を使う）
# =============================================================================
def calibrate_exponent(ascan_path, src_freq=1.25e9, f_lo=0.4e9, f_hi=3.0e9):
    """既存の free_space 計算から E(f) ∝ I(f) * f^P の P を実測する。

    入力波形が #waveform: gaussiandot <amp> <src_freq> であることを前提とする。
    gprMax の gaussiandot: chi = 1/f, zeta = 2 pi^2 f^2 の Gaussian の 1 階微分。
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    from tools.core.outputfiles_merge import get_output_data

    e_ref, dt = get_output_data(ascan_path, 1, 'Ez')
    e_ref = np.asarray(e_ref)
    if e_ref.ndim > 1:
        e_ref = e_ref[:, 0]
    n = len(e_ref)

    t = np.arange(n) * dt
    chi = 1.0 / src_freq
    zeta = 2.0 * np.pi ** 2 * src_freq ** 2
    i_src = -2.0 * zeta * (t - chi) * np.exp(-zeta * (t - chi) ** 2)

    f = np.fft.rfftfreq(n, d=dt)
    g = np.abs(np.fft.rfft(e_ref)) / (np.abs(np.fft.rfft(i_src)) + 1e-30)

    m = (f >= f_lo) & (f <= f_hi) & (np.abs(np.fft.rfft(i_src)) >
                                     1e-3 * np.max(np.abs(np.fft.rfft(i_src))))
    p, logc = np.polyfit(np.log(f[m]), np.log(g[m]), 1)

    print('--- P の実測校正 ---')
    print('  参照ファイル : {}'.format(ascan_path))
    print('  フィット帯域 : {:.2f} - {:.2f} GHz, 有効点数 {}'.format(f_lo / 1e9, f_hi / 1e9, m.sum()))
    print('  実測 P       : {:.3f}   (理論値 0.5)'.format(p))
    resid = np.log(g[m]) - (p * np.log(f[m]) + logc)
    print('  フィット残差 : RMS {:.3f} (自然対数) = {:.2f} dB'.format(
        np.std(resid), 20 * np.std(resid) / np.log(10)))
    if abs(p - 0.5) > 0.15:
        print('  ** 実測値が理論値から離れています。EXPONENT_P を実測値に設定してください。')
    return p


# =============================================================================
# main
# =============================================================================
def ask_output_dir(default=DEFAULT_OUTDIR):
    """出力ディレクトリを対話的に受け取り、存在しなければ作成して返す。

    Enter だけ押した場合は default を使う。~ は展開する。
    """
    while True:
        raw = input('出力ディレクトリを入力してください [Enter で "{}"] > '.format(default)).strip()
        path = os.path.abspath(os.path.expanduser(raw if raw else default))
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as err:
            print('  ディレクトリを作成できません: {}'.format(err))
            continue
        if not os.access(path, os.W_OK):
            print('  書き込み権限がありません: {}'.format(path))
            continue
        print('出力先: {}'.format(path))
        return path


# =============================================================================
# main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description='gprMax 用の帯域制限 excitation file を生成する')
    ap.add_argument('--calibrate', metavar='ASCAN_OUT',
                    help='既存の自由空間 A-scan から指数 P を実測して終了する')
    ap.add_argument('--outdir', help='出力ディレクトリ（省略時は実行中に対話的に尋ねる）')
    ap.add_argument('--no-plot', action='store_true', help='診断プロットを作成しない')
    args = ap.parse_args()

    if args.calibrate:
        calibrate_exponent(args.calibrate)
        return

    outdir = (os.path.abspath(os.path.expanduser(args.outdir)) if args.outdir
              else ask_output_dir())
    if args.outdir:
        os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, OUT_FILENAME)

    t, current, f, e_pred = design_current()
    report(t, current, f, e_pred)
    write_excitation_file(out_path, WAVEFORM_ID, t, current)

    print('\n.in ファイルでの使い方:')
    print('  #excitation_file: {}'.format(out_path))
    print('  #hertzian_dipole: z 1.5 3.35 0 {}'.format(WAVEFORM_ID))
    print('  （#waveform は不要）')

    if not args.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(3, 1, figsize=(8, 10))
        m = t < 2 * T_CENTER
        ax[0].plot(t[m] * 1e9, current[m], 'k-')
        ax[0].set_xlabel('Time [ns]'); ax[0].set_ylabel('Current [A]')
        ax[0].set_title('Source current waveform'); ax[0].grid(alpha=0.3)

        fm = f < 5e9
        i_spec = np.abs(np.fft.rfft(current))
        ax[1].plot(f[fm] / 1e9, i_spec[fm] / i_spec.max(), 'b-', label='Current I(f)')
        ax[1].plot(f[fm] / 1e9, e_pred[fm] / e_pred.max(), 'r-', label='Radiated E(f) (predicted)')
        for x in (F_LO / 1e9, F_HI / 1e9):
            ax[1].axvline(x, color='gray', ls=':')
        ax[1].set_xlabel('Frequency [GHz]'); ax[1].set_ylabel('Normalised amplitude')
        ax[1].set_title('Spectra (linear)'); ax[1].legend(); ax[1].grid(alpha=0.3)

        ax[2].plot(f[fm] / 1e9, 20 * np.log10(e_pred[fm] / e_pred.max() + 1e-30), 'r-')
        for x in (F_LO / 1e9, F_HI / 1e9):
            ax[2].axvline(x, color='gray', ls=':')
        ax[2].set_ylim(-100, 5)
        ax[2].set_xlabel('Frequency [GHz]'); ax[2].set_ylabel('[dB]')
        ax[2].set_title('Radiated spectrum (dB)'); ax[2].grid(alpha=0.3)

        plt.tight_layout()
        png = os.path.join(outdir, PLOT_FILENAME)
        fig.savefig(png, dpi=150, bbox_inches='tight')
        print('診断プロット: {}'.format(png))


if __name__ == '__main__':
    main()