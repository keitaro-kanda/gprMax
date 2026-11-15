#!/usr/bin/env python3
"""密度ゆらぎの反射係数と、それと同じ強度比を保つのに必要な水氷濃度の深さプロファイル

独立したスクリプト（ほかのツールを読み込まない）。

計算すること
    1. 密度ゆらぎ（1σ）による反射係数 Γ_fl(z) = tanh[ (1/4) ln(1.871) s ρ(z) ]
    2. 氷層（pore）の反射係数 Γ_ice(v, z) と Γ_fl(z) の比を、基準の深さ Z_REF と
       同じに保つのに必要な氷の体積分率 v(z) を厳密な式で逆算し、
       v(z) / v(Z_REF) を「基準の何倍の濃度が必要か」として出す。

式（記号と導出は vmin_permittivity_scaling.md、gamma_fluctuation_derivation.md）
    ρ(z)      = 1.92 (z_cm + 12.2) / (z_cm + 18)          Carrier et al. (1991)
    ε_d       = 1.871^ρ                                    Carrier et al. (1991) Fig. 9.53
    Γ_fl      = tanh[ (1/4) ln(1.871) s ρ ]
    Γ_ice     = tanh[ (3/4) ln(1 + a v / ε_d^(1/3)) ],  a = ε_ice^(1/3) - 1   （pore の LLL）
    C0        = Γ_ice(V_REF, Z_REF) / Γ_fl(Z_REF)
    v(z)      = ε_d^(1/3) / a · [ exp( (4/3) artanh( C0 Γ_fl(z) ) ) - 1 ]
    近似      v(z) / v(Z_REF) ≈ (ε_d(z) / ε_d(Z_REF))^(1/3) · ρ(z) / ρ(Z_REF)

出力（OUTPUT_DIR）
    fluctuation_reflection_profile.png（と pdf/）、fluctuation_reflection_profile.csv、
    run_info.txt
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# =============================================================================
# 設定  [EDIT HERE]
# =============================================================================
OUTPUT_DIR = '/Volumes/gprMax_water_ice/test_programs_output/fluctuation_reflection_profile'

Z_MIN, Z_MAX, DZ = 0.0, 3.0, 0.01     # [m] 深さの範囲と刻み
Z_REF = 0.5                           # [m] 基準の深さ（ここの濃度を 1 とする）
V_REF = 0.06                          # [-] 基準の深さでの氷の体積分率（厳密な式でだけ使う。
                                      #     比 v(z)/v(Z_REF) はこの値にほとんどよらない）
S_REL = 0.05                          # [-] 密度の相対ゆらぎ（Ascan.in の RHO_REL_STD）
CARRIER_EPS_BASE = 1.871              # ε_d = 1.871^ρ
EPS_ICE = 3.15                        # 氷の ε'

COLOR_CONTENT = 'k'                   # 必要な濃度（下の横軸）
COLOR_GAMMA = 'tab:blue'              # ゆらぎの反射係数（上の横軸）
FIGSIZE = (7.5, 8.0)
FIGURE_DPI = 200


# =============================================================================
# 計算
# =============================================================================
def density_profile(depth_m):
    """深さ [m] -> 乾燥バルク密度 [g/cm^3]（Carrier et al. 1991）"""
    z_cm = np.asarray(depth_m, dtype=float) * 100.0
    return 1.92 * (z_cm + 12.2) / (z_cm + 18.0)


def eps_dry(rho):
    return CARRIER_EPS_BASE ** np.asarray(rho, dtype=float)


A_ICE = EPS_ICE ** (1.0 / 3.0) - 1.0


def gamma_fluct(rho):
    """密度が 1σ（s ρ）違う 2 層の境界の反射係数"""
    return np.tanh(0.25 * np.log(CARRIER_EPS_BASE) * S_REL * np.asarray(rho, dtype=float))


def gamma_ice(v, rho):
    """乾燥層と氷層（pore、体積分率 v）の境界の反射係数"""
    return np.tanh(0.75 * np.log(1.0 + A_ICE * v / eps_dry(rho) ** (1.0 / 3.0)))


def v_for_gamma(gamma, rho):
    """gamma_ice(v, rho) = gamma となる v（厳密な逆関数）"""
    return eps_dry(rho) ** (1.0 / 3.0) / A_ICE * (np.exp(4.0 / 3.0 * np.arctanh(gamma)) - 1.0)


def compute():
    z = np.arange(Z_MIN, Z_MAX + DZ / 2, DZ)
    rho = density_profile(z)
    rho0 = float(density_profile(Z_REF))
    g_fl = gamma_fluct(rho)
    c0 = float(gamma_ice(V_REF, rho0) / gamma_fluct(rho0))
    v = v_for_gamma(c0 * g_fl, rho)
    ratio = v / V_REF
    ratio_approx = (eps_dry(rho) / eps_dry(rho0)) ** (1.0 / 3.0) * rho / rho0
    return {'z': z, 'rho': rho, 'eps': eps_dry(rho), 'g_fl': g_fl, 'v': v,
            'ratio': ratio, 'ratio_approx': ratio_approx, 'c0': c0, 'rho0': rho0}


# =============================================================================
# 図と出力
# =============================================================================
def plot(res, path_base):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    z = res['z']
    ax.plot(res['ratio'], z, color=COLOR_CONTENT, lw=2.2)
    ax.axhline(Z_REF, color='0.5', ls=':', lw=1.2)
    ax.plot(1.0, Z_REF, 'o', color=COLOR_CONTENT, ms=7)
    ax.set_ylim(Z_MAX, Z_MIN)
    ax.set_ylabel('Depth [m]')
    ax.set_xlabel(f'Required ice content (× value at {Z_REF:g} m)', color=COLOR_CONTENT)

    ax2 = ax.twiny()
    ax2.grid(False)
    ax2.plot(res['g_fl'], z, color=COLOR_GAMMA, lw=2.2)
    ax2.set_xlabel(r'Fluctuation reflection coefficient $\Gamma_{\rm fl}$', color=COLOR_GAMMA)
    ax2.tick_params(axis='x', which='both', colors=COLOR_GAMMA)
    ax2.spines['top'].set_color(COLOR_GAMMA)
    ax2.set_ylim(Z_MAX, Z_MIN)

    fig.tight_layout()
    fig.savefig(path_base + '.png', dpi=FIGURE_DPI, bbox_inches='tight')
    os.makedirs(os.path.join(os.path.dirname(path_base), 'pdf'), exist_ok=True)
    fig.savefig(os.path.join(os.path.dirname(path_base), 'pdf',
                             os.path.basename(path_base) + '.pdf'), bbox_inches='tight')
    plt.close(fig)
    return path_base + '.png'


def write_outputs(res):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'fluctuation_reflection_profile.csv'), 'w',
              newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['depth_m', 'rho_g_cm3', 'eps_dry', 'gamma_fluct', 'v_required',
                    'ratio_to_ref', 'ratio_to_ref_approx'])
        for i in range(res['z'].size):
            w.writerow([round(float(res['z'][i]), 4)] + [float(res[k][i]) for k in
                        ('rho', 'eps', 'g_fl', 'v', 'ratio', 'ratio_approx')])
    lines = ['密度ゆらぎの反射係数と、同じ強度比を保つのに必要な水氷濃度（pore）',
             f'  基準の深さ {Z_REF} m（ρ = {res["rho0"]:.4f}）、基準の濃度 {100 * V_REF:.1f} vol%',
             f'  密度の相対ゆらぎ s = {S_REL}、ε_ice = {EPS_ICE}、ε_d = {CARRIER_EPS_BASE}^ρ',
             f'  基準での比 Γ_ice / Γ_fl = {res["c0"]:.4f}', '',
             '  深さ [m]   ρ       ε_d     Γ_fl      必要な濃度（倍）  近似      [dB]']
    for zz in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        i = int(np.argmin(np.abs(res['z'] - zz)))
        lines.append('  {:6.2f}   {:.4f}  {:.4f}  {:.5f}   {:.4f}            {:.4f}    {:+.2f}'
                     .format(res['z'][i], res['rho'][i], res['eps'][i], res['g_fl'][i],
                             res['ratio'][i], res['ratio_approx'][i],
                             20 * np.log10(res['ratio'][i])))
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(OUTPUT_DIR, 'run_info.txt'), 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')


def main():
    res = compute()
    write_outputs(res)
    png = plot(res, os.path.join(OUTPUT_DIR, 'fluctuation_reflection_profile'))
    print(f'\n出力: {png}')


if __name__ == '__main__':
    main()