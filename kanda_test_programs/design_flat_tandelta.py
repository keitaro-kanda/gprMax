"""
Flat tan_delta design: multi-pole Debye parameters for a non-dispersive medium
==============================================================================
月南極のレゴリス（イルメナイト <1 wt%）は非分散である。すなわち帯域内で

    eps'  = 一定
    eps'' = 一定      ->      tan_delta = eps''/eps' = 一定

これを gprMax で実現したいが、gprMax の材料モデルは

    #material: eps_r sigma mu_r sigma*     ->  eps'' = sigma/(w eps0)  ∝ 1/f

しか与えず、sigma 一定では tan_delta が 1/f で落ちてしまう（これは Level 2）。
Debye/Lorentz/Drude しか使えないため、**複数の Debye 極を重ね合わせて
eps'' を帯域内で平坦にする**しかない。

    Debye 1 極の eps''(w) = De * (w tau) / (1 + (w tau)^2)
        -> w = 1/tau にピークを持つ山型

    tau を帯域の上下に振り分けて複数重ねると、山どうしが重なって平坦部ができる

このスクリプトは
  (1) 極数 1-4 について「eps'' が目標値一定」となる (De_i, tau_i) を最小二乗で決定
  (2) 極数ごとの平坦度・gprMax の tau > dt 制約・alpha の周波数依存を評価
  (3) 選んだ極数で tan_delta 一定が達成できていることを図で説明
を行う。

なぜ tan_delta 一定が重要か
---------------------------
減衰係数は低損失極限で

    alpha ≈ pi f n tan_delta / c

なので、tan_delta 一定なら alpha ∝ f（帯域 0.5-2.0 GHz で 4.0 倍）。
一方 sigma 一定なら alpha は周波数に依存しない（帯域内比 1.0）。

alpha が周波数依存を持つことは、重心周波数シフト法が機能するための必要条件
である（df_c/dt = -2 pi tan_delta sigma_f^2 は alpha ∝ f を前提とする）。
したがって「非分散」を正しく実装できるかどうかが、手法の成立可否を分ける。

Requirements: numpy, scipy, matplotlib
"""

import os
import io
import datetime

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# [EDIT HERE] 設計条件
# ---------------------------------------------------------------------------
EPS_R = 3.0                    # 比誘電率実部（帯域内で一定にしたい値）

# Heiken/Carrier 経験式（450 MHz 版）で tan_delta を決める
#   tan_delta = 10^(0.033*(FeO+TiO2) + 0.231*rho - 3.061)
HEIKEN_EPS_BASE = 1.843
HEIKEN_TAND_A, HEIKEN_TAND_B, HEIKEN_TAND_C = 0.033, 0.231, 3.061

# 設計対象の組成 [wt%]。月南極域は 5-11 wt% に収束するため 3 点、
# 高Tiバサルト（月の海）の参考値として 20 wt% を併記する。
COMPOSITIONS = {'feo5': 5.0, 'feo7p5': 7.5, 'feo10': 10.0, 'feo20': 20.0}
PRIMARY_KEY = 'feo7p5'         # 図の主対象

BAND_LO, BAND_HI = 0.5e9, 2.0e9    # 平坦化する帯域（LUPEX GPR）
BAND_CENTRE = 1.25e9

N_POLES_LIST = [1, 2, 3, 4]    # 比較する極数
N_POLES_CHOSEN = 2             # 採用する極数（結果を見て決める）

# gprMax の時間刻み制約
DX = 0.0025                    # [m] グリッドサイズ
C0 = 299792458.0               # [m/s]
DT_GPRMAX = DX / (np.sqrt(2.0) * C0)      # 2D クーラン条件
TAU_MIN_SAFE = 5.0             # tau/dt がこれを下回ったら警告

N_FIT = 300                    # フィットに使う帯域内の周波数点数
F_PLOT_LO, F_PLOT_HI = 1e8, 1e10          # 図の周波数範囲

OUTPUT_DIR = '/Volumes/SSD_Kanda_BUFFALO/test_programs_output/flat_tandelta_design'
RESULTS_FILENAME = 'flat_tandelta_design.txt'


# ---------------------------------------------------------------------------
# 1. 目標値の決定
# ---------------------------------------------------------------------------

def density_for_eps(eps_r):
    """eps' = 1.843^rho を eps_r について解いた密度 [g/cm^3]。

    非分散なので eps' は全周波数で同じ値になり、「どの周波数で」という
    但し書きが不要になる。分散性モデル（Level 3b）では eps' 自体が
    周波数依存になるため、基準周波数を決める必要があった。
    """
    return np.log(eps_r) / np.log(HEIKEN_EPS_BASE)


def heiken_tandelta(feotio2_wt, rho):
    """Heiken/Carrier 経験式の tan_delta。周波数に依存しない。"""
    return 10.0 ** (HEIKEN_TAND_A * feotio2_wt + HEIKEN_TAND_B * rho - HEIKEN_TAND_C)


# ---------------------------------------------------------------------------
# 2. 多極 Debye モデル
# ---------------------------------------------------------------------------

def debye_eps_imag(f, poles):
    """eps''(f) = sum_i De_i (w tau_i) / (1 + (w tau_i)^2)。"""
    w = 2.0 * np.pi * np.asarray(f, dtype=float)
    return sum(de * (w * tau) / (1.0 + (w * tau) ** 2) for de, tau in poles)


def debye_eps_real_drop(f, poles):
    """静的値からの eps' の低下量 sum_i De_i (w tau_i)^2 / (1 + (w tau_i)^2)。"""
    w = 2.0 * np.pi * np.asarray(f, dtype=float)
    return sum(de * (w * tau) ** 2 / (1.0 + (w * tau) ** 2) for de, tau in poles)


def debye_eps_real(f, eps_inf, poles):
    """eps'(f) = eps_inf + sum_i De_i / (1 + (w tau_i)^2)。"""
    w = 2.0 * np.pi * np.asarray(f, dtype=float)
    return eps_inf + sum(de / (1.0 + (w * tau) ** 2) for de, tau in poles)


# ---------------------------------------------------------------------------
# 3. eps'' を平坦にする極の決定
# ---------------------------------------------------------------------------

def design_flat_poles(target_eps_imag, n_poles):
    """eps'' が帯域内で target_eps_imag 一定になる n 極 Debye を決める。

    最適化変数は [De_1..De_n, log(tau_1)..log(tau_n)]。
    tau を対数で扱うのは、桁をまたぐ探索を安定させるため。
    下限は gprMax の制約 tau > dt に余裕を持たせて 2*dt とする。

    初期値の tau は帯域の少し外側（0.35-3 GHz 相当）に対数等間隔で置く。
    帯域端まで平坦にするには、ピークを帯域の外に出す必要があるため。
    """
    f = np.geomspace(BAND_LO, BAND_HI, N_FIT)

    def unpack(x):
        return list(zip(x[:n_poles], np.exp(x[n_poles:])))

    tau_seed = np.geomspace(1.0 / (2 * np.pi * 3e9),
                            1.0 / (2 * np.pi * 0.35e9), n_poles)
    if n_poles == 1:
        tau_seed = np.array([1.0 / (2 * np.pi * 1e9)])

    x0 = np.concatenate([np.full(n_poles, target_eps_imag * 1.5), np.log(tau_seed)])
    lo = np.concatenate([np.zeros(n_poles), np.full(n_poles, np.log(2 * DT_GPRMAX))])
    hi = np.concatenate([np.full(n_poles, 1.0), np.full(n_poles, np.log(1e-8))])

    r = least_squares(lambda x: debye_eps_imag(f, unpack(x)) / target_eps_imag - 1.0,
                      x0, bounds=(lo, hi), xtol=1e-15, ftol=1e-15, max_nfev=20000)

    poles = sorted(unpack(r.x), key=lambda p: p[1])
    # eps' が帯域中心で EPS_R になるよう eps_inf を決める
    eps_inf = EPS_R - float(np.interp(BAND_CENTRE, f,
                                      debye_eps_real_drop(f, poles)))
    return eps_inf, poles


def evaluate(eps_inf, poles, target_eps_imag):
    """設計した極の性能を評価する。"""
    f = np.geomspace(BAND_LO, BAND_HI, N_FIT)
    ei = debye_eps_imag(f, poles)
    er = debye_eps_real(f, eps_inf, poles)
    td = ei / er
    alpha = attenuation(f, er, td)
    return dict(
        rms_eps_imag=100 * np.sqrt(np.mean((ei / target_eps_imag - 1.0) ** 2)),
        ptp_tand=100 * (td.max() - td.min()) / td.mean(),
        ptp_eps_real=100 * (er.max() - er.min()) / er.mean(),
        alpha_ratio=alpha[-1] / alpha[0],
        tau_min_ratio=min(t for _, t in poles) / DT_GPRMAX,
        f=f, eps_re=er, eps_im=ei, tand=td, alpha=alpha,
    )


def attenuation(f, eps_re, tand):
    """減衰係数 alpha [Np/m]（厳密式）。

        alpha = (w/c) sqrt(eps'/2) sqrt( sqrt(1 + tan_delta^2) - 1 )

    低損失極限では alpha -> pi f sqrt(eps') tan_delta / c となり、
    tan_delta 一定なら alpha ∝ f になる。
    """
    w = 2.0 * np.pi * np.asarray(f, dtype=float)
    return (w / C0) * np.sqrt(eps_re / 2.0) * np.sqrt(np.sqrt(1.0 + tand ** 2) - 1.0)


# ---------------------------------------------------------------------------
# 4. 作図
# ---------------------------------------------------------------------------

def plot_design(key, feotio2_wt, target, eps_inf, poles, ev, all_ev):
    """設計結果を 4 パネルで説明する図。

    (a) 各極の eps'' と、その和が帯域内で平坦になる様子
    (b) tan_delta が一定になっていること（sigma 一定の場合と対比）
    (c) alpha が f に比例すること（sigma 一定の場合と対比）
    (d) 極数ごとの平坦度と gprMax 制約のトレードオフ
    """
    f = np.geomspace(F_PLOT_LO, F_PLOT_HI, 800)
    fg = f / 1e9

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f'Flat tan$\\delta$ design — FeO+TiO$_2$ = {feotio2_wt} wt% [{key}], '
        f'{len(poles)}-pole Debye\n'
        rf'target: $\varepsilon_r$ = {EPS_R}, $\tan\delta$ = {target / EPS_R:.6f} '
        rf'($\varepsilon_r^{{\prime\prime}}$ = {target:.6f})', fontsize=12)
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    def deco(ax, logx=True):
        if logx:
            ax.set_xscale('log')
            ax.set_xlim(F_PLOT_LO / 1e9, F_PLOT_HI / 1e9)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, _: f'{x:.1g}' if x < 1 else f'{int(x)}'))
        ax.axvspan(BAND_LO / 1e9, BAND_HI / 1e9, alpha=0.12, color='gray',
                   label='LUPEX GPR band')
        ax.axvline(BAND_CENTRE / 1e9, color='gray', lw=0.8, ls='-.', alpha=0.6)
        ax.set_xlabel('Frequency (GHz)')
        ax.grid(True, which='both', alpha=0.2)

    # --- (a) 極の重ね合わせ ---
    for i, (de, tau) in enumerate(poles):
        ax_a.plot(fg, debye_eps_imag(f, [(de, tau)]), ls='--', lw=1.4,
                  label=rf'pole {i + 1}: $\Delta\varepsilon$={de:.5f}, '
                        rf'$\tau$={tau * 1e12:.1f} ps')
        ax_a.axvline(1.0 / (2 * np.pi * tau) / 1e9, color='C%d' % i,
                     lw=0.7, ls=':', alpha=0.7)
    ax_a.plot(fg, debye_eps_imag(f, poles), 'k-', lw=2.5, label='sum (total)', zorder=5)
    ax_a.axhline(target, color='r', lw=1.5, ls='--', label=f'target = {target:.6f}')
    deco(ax_a)
    ax_a.set_ylabel(r"$\varepsilon_r''$")
    ax_a.set_title("(a) Superposition of Debye poles flattens $\\varepsilon_r''$")
    ax_a.set_ylim(0, target * 2.2)
    ax_a.legend(fontsize=8, loc='upper right')

    # --- (b) tan_delta ---
    er = debye_eps_real(f, eps_inf, poles)
    td = debye_eps_imag(f, poles) / er
    sigma_eq = 2 * np.pi * BAND_CENTRE * 8.8541878128e-12 * target   # 同じ eps'' を与える sigma
    td_sigma = sigma_eq / (2 * np.pi * f * 8.8541878128e-12 * EPS_R)
    ax_b.plot(fg, td, 'k-', lw=2.5, label=f'{len(poles)}-pole Debye (this design)')
    ax_b.axhline(target / EPS_R, color='r', lw=1.5, ls='--', label='target (constant)')
    ax_b.plot(fg, td_sigma, color='steelblue', lw=1.6, ls=':',
              label=r'constant $\sigma$ (Level 2) $\propto 1/f$')
    deco(ax_b)
    ax_b.set_yscale('log')
    ax_b.set_ylabel(r'$\tan\delta$')
    ax_b.set_title(r'(b) $\tan\delta$ stays constant in band')
    ax_b.legend(fontsize=8, loc='upper right')

    # --- (c) alpha ---
    al = attenuation(f, er, td)
    al_sigma = attenuation(f, np.full_like(f, EPS_R), td_sigma)
    ax_c.plot(fg, al, 'k-', lw=2.5, label=f'{len(poles)}-pole Debye ($\\propto f$)')
    ax_c.plot(fg, al_sigma, color='steelblue', lw=1.6, ls=':',
              label=r'constant $\sigma$ (flat)')
    ideal = al[np.argmin(abs(f - BAND_CENTRE))] * (f / BAND_CENTRE)
    ax_c.plot(fg, ideal, color='r', lw=1.2, ls='--', label=r'ideal $\alpha \propto f$')
    deco(ax_c)
    ax_c.set_xscale('log'); ax_c.set_yscale('log')
    ax_c.set_ylabel(r'$\alpha$ (Np/m)')
    ax_c.set_title(rf'(c) $\alpha \propto f$  (in-band ratio {ev["alpha_ratio"]:.3f}, ideal 4.000)')
    ax_c.legend(fontsize=8, loc='upper left')

    # --- (d) 極数のトレードオフ ---
    ns = sorted(all_ev)
    rms = [all_ev[n]['rms_eps_imag'] for n in ns]
    tau_r = [all_ev[n]['tau_min_ratio'] for n in ns]
    ax_d.set_xscale('linear')
    ln1 = ax_d.plot(ns, rms, 'o-', color='crimson', label=r"$\varepsilon_r''$ RMS error")
    ax_d.set_yscale('log')
    ax_d.set_xlabel('Number of Debye poles')
    ax_d.set_ylabel(r"$\varepsilon_r''$ RMS error (%)", color='crimson')
    ax_d.set_xticks(ns)
    ax_d.grid(True, alpha=0.2)
    ax2 = ax_d.twinx()
    ln2 = ax2.plot(ns, tau_r, 's--', color='navy', label=r'min $\tau/\Delta t$')
    ax2.axhline(TAU_MIN_SAFE, color='navy', lw=0.8, ls=':', alpha=0.7)
    ax2.set_ylabel(r'min $\tau / \Delta t$  (gprMax constraint)', color='navy')
    ax_d.axvline(N_POLES_CHOSEN, color='gray', lw=1.2, ls='-.', alpha=0.7)
    ax_d.set_title(f'(d) Trade-off: accuracy vs gprMax constraint '
                   f'(chosen: {N_POLES_CHOSEN} poles)')
    lns = ln1 + ln2
    ax_d.legend(lns, [l.get_label() for l in lns], fontsize=8, loc='center right')

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    name = f'flat_tandelta_{key}'
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(OUTPUT_DIR, f'{name}.{ext}'),
                    dpi=150 if ext == 'png' else 300, bbox_inches='tight')
    print(f'プロット保存: {OUTPUT_DIR}/{name}.png/.pdf')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. main
# ---------------------------------------------------------------------------

def main():
    buf = io.StringIO()

    def _p(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=buf)

    rho = density_for_eps(EPS_R)
    _p('=' * 78)
    _p('帯域内で tan_delta を一定に保つ多極 Debye パラメータの設計')
    _p('=' * 78)
    _p(f'帯域          : {BAND_LO / 1e9:.2f} - {BAND_HI / 1e9:.2f} GHz')
    _p(f'eps_r         : {EPS_R}  (非分散なので全周波数で一定)')
    _p(f'密度          : rho = {rho:.6f} g/cm^3   (eps_r = 1.843^rho の解)')
    _p(f'gprMax dt     : {DT_GPRMAX * 1e12:.3f} ps  (dx = {DX} m, 2D クーラン条件)')
    _p('')
    _p('注: eps_r は組成に依らないので、全組成で密度・走時・幾何減衰が共通になる。')
    _p('    違いは吸収だけなので、組成の効果だけを切り出して比較できる。')

    all_results = {}
    for key, wt in COMPOSITIONS.items():
        tand_target = heiken_tandelta(wt, rho)
        target = EPS_R * tand_target

        _p('\n' + '=' * 78)
        _p(f'FeO+TiO2 = {wt} wt%  [{key}]')
        _p('=' * 78)
        _p(f'  Heiken 経験式 tan_delta = {tand_target:.6f}   '
           f'-> 目標 eps_imag = {target:.6f}')

        evs = {}
        for n in N_POLES_LIST:
            eps_inf, poles = design_flat_poles(target, n)
            ev = evaluate(eps_inf, poles, target)
            ev['eps_inf'], ev['poles'] = eps_inf, poles
            evs[n] = ev

        _p(f"\n  {'極数':>4}{'eps_imag RMS':>15}{'tand p-p':>11}"
           f"{'eps_re p-p':>13}{'alpha 比':>10}{'min tau/dt':>13}{'判定':>8}")
        for n in N_POLES_LIST:
            ev = evs[n]
            ok = 'OK' if ev['tau_min_ratio'] >= TAU_MIN_SAFE else 'tau 小'
            _p(f"  {n:>4}{ev['rms_eps_imag']:>14.3f}%{ev['ptp_tand']:>10.3f}%"
               f"{ev['ptp_eps_real']:>12.3f}%{ev['alpha_ratio']:>10.3f}"
               f"{ev['tau_min_ratio']:>13.1f}{ok:>8}")

        ev = evs[N_POLES_CHOSEN]
        eps_inf, poles = ev['eps_inf'], ev['poles']
        _p(f'\n  --- 採用: {N_POLES_CHOSEN} 極 ---')
        for i, (de, tau) in enumerate(poles):
            _p(f'    極{i + 1}: De = {de:.6f}, tau = {tau * 1e12:.2f} ps  '
               f'(緩和ピーク {1.0 / (2 * np.pi * tau) / 1e9:.3f} GHz, '
               f'tau/dt = {tau / DT_GPRMAX:.1f})')
        _p(f'\n  gprMax 記述:')
        _p(f'    #material: {eps_inf:.6f} 0 1 0 regolith')
        _p('    #add_dispersion_debye: {}{} regolith'.format(
            len(poles), ''.join(f' {de:.6f} {tau:.6e}' for de, tau in poles)))

        _p(f'\n  帯域内の検算:')
        for f0 in (BAND_LO, BAND_CENTRE, BAND_HI):
            i = np.argmin(abs(ev['f'] - f0))
            _p(f"    f = {f0 / 1e9:>4.2f} GHz: eps_re = {ev['eps_re'][i]:.5f}, "
               f"eps_imag = {ev['eps_im'][i]:.6f}, tand = {ev['tand'][i]:.6f}, "
               f"alpha = {ev['alpha'][i]:.4f} Np/m")

        all_results[key] = (wt, target, eps_inf, poles, ev, evs)

    # 図は主対象のみ（他組成は tau が共通で De が比例するだけ）
    wt, target, eps_inf, poles, ev, evs = all_results[PRIMARY_KEY]
    plot_design(PRIMARY_KEY, wt, target, eps_inf, poles, ev, evs)

    _p('\n' + '=' * 78)
    _p('組成間の比較（採用した極数）')
    _p('=' * 78)
    _p(f"  {'組成':>8}{'wt%':>7}{'tan_delta':>12}{'eps_inf':>11}"
       f"{'De (各極)':>12}{'alpha@1.25':>12}{'alpha 比':>10}")
    for key, (wt, target, eps_inf, poles, ev, _) in all_results.items():
        i = np.argmin(abs(ev['f'] - BAND_CENTRE))
        _p(f"  {key:>8}{wt:>7.1f}{target / EPS_R:>12.6f}{eps_inf:>11.6f}"
           f"{poles[0][0]:>12.6f}{ev['alpha'][i]:>12.4f}{ev['alpha_ratio']:>10.3f}")
    _p('\n  tau は全組成で共通、De だけが tan_delta に比例する。')
    _p('  -> 組成を変えても緩和の「形」は同じで、振幅だけが変わる。')

    buf.write(f'\n[Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}]\n')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, RESULTS_FILENAME)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(buf.getvalue())
    print(f'\nAll results saved to: {out_path}')


if __name__ == '__main__':
    main()