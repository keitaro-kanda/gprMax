"""
Cole-Cole (Boivin+2022, Table 4) vs multi-pole Debye approximation
====================================================================
Boivin et al. (2022) JGR Planets, Table 4 — Bayesian one-pole Cole-Cole
fit parameters for four bytownite-ilmenite mixture samples:

    sample        eps_inf   Delta_eps   tau (s)      alpha   sigma_dc (S/m)
    10 w% ilm      3.554     0.194      4.370e-11    0.717   2e-5
    15 w% ilm      3.659     0.291      5.201e-11    0.746   1e-5
    20 w% ilm      3.792     0.420      5.036e-11    0.756   1e-5
    pure ilmenite  7.048    17.437      3.742e-10    0.553   2e-5

gprMax supports only Debye/Lorentz/Drude dispersion models.
This script fits EACH of the four Boivin+2022 Cole-Cole spectra with
1-, 2-, and 3-pole Debye models and visualises the approximation error
across 0.1-10 GHz, with emphasis on the GPR band (0.5-1.25 GHz).

For every sample the script produces:
    - Frequency-dependent eps', eps'', loss tangent
    - Relative error of each Debye fit vs Cole-Cole
    - Fitted parameters printed in gprMax #add_dispersion_debye syntax

All four samples' results are appended to a single results text file,
and one comparison figure (PNG + PDF) is generated per sample.

Requirements: numpy, scipy, matplotlib
"""

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import io
import datetime
import os

# ---------------------------------------------------------------------------
# 0. Sample definitions (Boivin+2022, Table 4 — Bayesian 1-pole Cole-Cole fits)
# ---------------------------------------------------------------------------

SAMPLES = {
    "10wt_ilm": {
        "label"    : "10 w% ilmenite in bytownite",
        "eps_inf"  : 3.554,
        "delta_eps": 0.194,
        "tau"      : 4.370e-11,   # s
        "alpha"    : 0.717,
        "sigma_dc" : 2e-5,        # S/m
    },
    "15wt_ilm": {
        "label"    : "15 w% ilmenite in bytownite",
        "eps_inf"  : 3.659,
        "delta_eps": 0.291,
        "tau"      : 5.201e-11,
        "alpha"    : 0.746,
        "sigma_dc" : 1e-5,
    },
    "20wt_ilm": {
        "label"    : "20 w% ilmenite in bytownite",
        "eps_inf"  : 3.792,
        "delta_eps": 0.420,
        "tau"      : 5.036e-11,
        "alpha"    : 0.756,
        "sigma_dc" : 1e-5,
    },
    "pure_ilm": {
        "label"    : "Pure ilmenite",
        "eps_inf"  : 7.048,
        "delta_eps": 17.437,
        "tau"      : 3.742e-10,
        "alpha"    : 0.553,
        "sigma_dc" : 2e-5,
    },
}

# gprMax timestep constraint (dx = 0.002 m, 3D Courant limit)
DX        = 0.002       # m
C0        = 3e8         # m/s
DT_GPRMAX = DX / (np.sqrt(3) * C0)   # ~3.85 ps
TAU_MIN   = 2 * DT_GPRMAX            # gprMax lower limit ~7.7 ps

# Frequency grid: 100 MHz - 10 GHz, log-spaced
F_MIN, F_MAX, N_F = 1e8, 1e10, 500
freqs = np.geomspace(F_MIN, F_MAX, N_F)
EPS0  = 8.854187817e-12  # F/m

# GPR band
GPR_LO, GPR_HI = 5e8, 2.0e9
FC_GPR = 1.25e9          # centre frequency of gprMax waveform

# Output locations
OUTPUT_DIR      = '/Volumes/SSD_Kanda_BUFFALO/test_programs_output/compare_ColeCole_debye'
RESULTS_FILENAME = "debye_fit_results.txt"


# ---------------------------------------------------------------------------
# 1. Cole-Cole model (Boivin Eq. 1 sign convention)
# ---------------------------------------------------------------------------

def cole_cole(f, eps_inf, delta_eps, tau, alpha, sigma_dc=0.0):
    """
    1-pole Cole-Cole complex permittivity.
        eps*(w) = eps_inf - i*sigma_dc/(w*eps0)
                  + delta_eps / (1 + (i*w*tau)^alpha)
    Returns (eps_real, eps_imag) with eps_imag > 0 for lossy media.
    """
    w = 2 * np.pi * f
    jot = 1j * w * tau
    denom = 1 + jot ** alpha
    eps_star = eps_inf + delta_eps / denom - 1j * sigma_dc / (w * EPS0)
    return eps_star.real, -eps_star.imag


# ---------------------------------------------------------------------------
# 2. Multi-pole Debye model and least-squares fitting
# ---------------------------------------------------------------------------

def multi_debye(f, eps_inf, poles):
    """
    poles: list of (delta_eps_i, tau_i)
    eps*(w) = eps_inf + sum_i  delta_eps_i / (1 + i*w*tau_i)
    Returns (eps_real, eps_imag).
    """
    w = 2 * np.pi * f
    eps_star = np.full_like(w, eps_inf, dtype=complex)
    for de, tau in poles:
        eps_star += de / (1 + 1j * w * tau)
    return eps_star.real, -eps_star.imag


def fit_debye(n_poles, eps_inf, delta_eps, tau_cc, cc_re, cc_im, tau_seeds=None):
    """
    Fit n_poles Debye poles to the Cole-Cole spectrum of a single sample.
    Optimisation variables: [delta_eps_1,..,delta_eps_n, log(tau_1),..,log(tau_n)]
    Constraints: delta_eps_i >= 0, tau_i >= TAU_MIN.
    GPR band is up-weighted by factor 5 in the residual.
    """
    if tau_seeds is None:
        if n_poles == 1:
            tau_seeds = [tau_cc]
        elif n_poles == 2:
            tau_seeds = [tau_cc * 0.2, tau_cc * 5.0]
        else:
            tau_seeds = [tau_cc * 0.05, tau_cc * 1.0, tau_cc * 20.0]

    # Clip seeds to stay above TAU_MIN
    tau_seeds = [max(t, TAU_MIN * 1.1) for t in tau_seeds]
    x0 = np.array([delta_eps / n_poles] * n_poles
                  + [np.log(t) for t in tau_seeds])

    lo = [0.0] * n_poles + [np.log(TAU_MIN)] * n_poles
    hi = [delta_eps * 3] * n_poles + [np.log(tau_cc * 200)] * n_poles

    # GPR-band frequency weight
    w_freq = np.ones(N_F)
    w_freq[(freqs >= GPR_LO) & (freqs <= GPR_HI * 2)] *= 5.0

    def residuals(x):
        des  = x[:n_poles]
        taus = np.exp(x[n_poles:])
        d_re, d_im = multi_debye(freqs, eps_inf, list(zip(des, taus)))
        res_re = (d_re - cc_re) / cc_re * w_freq
        res_im = (d_im - cc_im) / np.maximum(cc_im, 1e-6) * w_freq
        return np.concatenate([res_re, res_im])

    result = least_squares(residuals, x0, bounds=(lo, hi),
                           method='trf', ftol=1e-12, xtol=1e-12, max_nfev=10000)

    des  = result.x[:n_poles]
    taus = np.exp(result.x[n_poles:])
    order = np.argsort(taus)
    return list(zip(des[order], taus[order])), result.cost


def rms(a, b, ref):
    return np.sqrt(np.mean(((a - b) / np.maximum(ref, 1e-9)) ** 2)) * 100


def run_sample(sample_key, params, buf):
    """
    Run the Cole-Cole vs. 1/2/3-pole Debye fitting + plotting pipeline for a
    single sample and append the printed results to the shared StringIO buf.
    """
    eps_inf   = params["eps_inf"]
    delta_eps = params["delta_eps"]
    tau_cc    = params["tau"]
    alpha_cc  = params["alpha"]
    sigma_dc  = params["sigma_dc"]
    label     = params["label"]

    def _p(*args, **kwargs):
        """Print to both stdout and the shared string buffer."""
        print(*args, **kwargs)
        print(*args, **kwargs, file=buf)

    # --- Cole-Cole reference spectrum ---
    cc_re, cc_im = cole_cole(freqs, eps_inf, delta_eps, tau_cc, alpha_cc, sigma_dc)
    cc_tand = cc_im / cc_re

    # --- Run fits for 1, 2, 3 poles ---
    fits = {}
    for n in [1, 2, 3]:
        poles, cost = fit_debye(n, eps_inf, delta_eps, tau_cc, cc_re, cc_im)
        d_re, d_im = multi_debye(freqs, eps_inf, poles)
        gpr_mask = (freqs >= GPR_LO) & (freqs <= GPR_HI)

        fits[n] = {
            'poles'       : poles,
            're'          : d_re,
            'im'          : d_im,
            'tand'        : d_im / d_re,
            'rms_re'      : rms(d_re, cc_re, cc_re),
            'rms_im'      : rms(d_im, cc_im, cc_im),
            'rms_re_gpr'  : rms(d_re[gpr_mask], cc_re[gpr_mask], cc_re[gpr_mask]),
            'rms_im_gpr'  : rms(d_im[gpr_mask], cc_im[gpr_mask], cc_im[gpr_mask]),
        }

    # --- Print / save results ---
    SEP = "=" * 70
    _p("\n" + SEP)
    _p(f"Sample: {label}  [{sample_key}]")
    _p("Cole-Cole parameters  (Boivin+2022 Table 4)")
    _p(f"  eps_inf={eps_inf},  Delta_eps={delta_eps},  "
       f"tau={tau_cc*1e12:.1f} ps,  alpha={alpha_cc},  sigma_dc={sigma_dc:.1e} S/m")
    _p(f"  gprMax tau_min = {TAU_MIN*1e12:.1f} ps  (dx={DX} m)")
    _p(SEP)

    for n in [1, 2, 3]:
        f = fits[n]
        _p(f"\n--- {n}-pole Debye fit " + "-" * 45)
        for i, (de, tau) in enumerate(f['poles']):
            ok = "OK" if tau >= TAU_MIN else "FAIL (< tau_min)"
            _p(f"  pole {i+1}:  Delta_eps = {de:.3f},  tau = {tau*1e12:.3f} ps  [{ok}]")
        _p(f"  sum(Delta_eps) = {sum(de for de, _ in f['poles']):.2f}"
           f"  (Cole-Cole Delta_eps = {delta_eps})")
        _p(f"  RMS error (full band):  eps' {f['rms_re']:.2f}%,  eps'' {f['rms_im']:.2f}%")
        _p(f"  RMS error (GPR band):   eps' {f['rms_re_gpr']:.2f}%,  eps'' {f['rms_im_gpr']:.2f}%")

        pole_str = "".join(f" {de:.6f} {tau:.6e}" for de, tau in f['poles'])
        mat_name = f"{sample_key}"
        _p(f"\n  gprMax syntax (material name '{mat_name}'):")
        _p(f"    #add_dispersion_debye: {n}{pole_str} {mat_name}")

    _p("\n" + SEP)
    best_n = min([1, 2, 3], key=lambda n: fits[n]['rms_re_gpr'] + fits[n]['rms_im_gpr'])
    _p(f"Recommendation for {label}: {best_n}-pole Debye gives the best "
       f"accuracy/complexity trade-off (lowest combined GPR-band RMS error).")
    _p(SEP)

    # --- Plot ---
    COLORS = {1: 'r', 2: 'g', 3: 'b'}
    STYLES = {1: '-', 2: '--', 3: ':'}
    LABELS = {1: '1-pole Debye', 2: '2-pole Debye', 3: '3-pole Debye'}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Cole-Cole (Boivin+2022, {label}) vs multi-pole Debye\n"
        rf"$\varepsilon_\infty$={eps_inf}, $\Delta\varepsilon$={delta_eps}, "
        rf"$\tau$={tau_cc*1e12:.1f} ps, $\alpha$={alpha_cc}",
        fontsize=12
    )

    ax_re, ax_im, ax_td, ax_err = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    def shade_gpr(ax):
        ax.axvspan(GPR_LO / 1e9, GPR_HI / 1e9, alpha=0.12, color='gray', label='LUPEX GPR band')
        ax.axvline(FC_GPR / 1e9, color='gray', lw=0.8, ls='-.', alpha=0.6)

    for ax in [ax_re, ax_im, ax_td, ax_err]:
        ax.set_xscale('log')
        ax.set_xlabel('Frequency (GHz)', fontsize=10)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f'{x:.1g}' if x < 1 else f'{int(x)}'
        ))
        ax.set_xlim(F_MIN / 1e9, F_MAX / 1e9)
        ax.grid(True, which='both', alpha=0.2)
        ax.axvline(0.1, color='lightgray', lw=1.0, ls='--')

    # Real part
    ax_re.plot(freqs / 1e9, cc_re, 'k-', lw=2.5, label='Cole-Cole', zorder=5)
    for n in [1, 2, 3]:
        ax_re.plot(freqs / 1e9, fits[n]['re'],
                   color=COLORS[n], ls=STYLES[n], lw=1.8, label=LABELS[n])
    shade_gpr(ax_re)
    ax_re.set_ylabel(r"$\varepsilon_r'$", fontsize=11)
    ax_re.set_title("Real permittivity", fontsize=11)
    ax_re.legend(fontsize=9, loc='upper right')
    ax_re.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    # Imaginary part
    ax_im.plot(freqs / 1e9, cc_im, 'k-', lw=2.5, label='Cole-Cole', zorder=5)
    for n in [1, 2, 3]:
        ax_im.plot(freqs / 1e9, fits[n]['im'],
                   color=COLORS[n], ls=STYLES[n], lw=1.8, label=LABELS[n])
    shade_gpr(ax_im)
    ax_im.set_ylabel(r"$\varepsilon_r''$", fontsize=11)
    ax_im.set_title("Imaginary permittivity", fontsize=11)
    ax_im.legend(fontsize=9, loc='upper left')
    ax_im.set_yscale('log')

    # Loss tangent
    ax_td.plot(freqs / 1e9, cc_tand, 'k-', lw=2.5, label='Cole-Cole', zorder=5)
    for n in [1, 2, 3]:
        ax_td.plot(freqs / 1e9, fits[n]['tand'],
                   color=COLORS[n], ls=STYLES[n], lw=1.8, label=LABELS[n])
    shade_gpr(ax_td)
    ax_td.set_ylabel(r"$\tan \delta$", fontsize=11)
    ax_td.set_title("Loss tangent", fontsize=11)
    ax_td.legend(fontsize=9, loc='upper left')

    # Relative error in real permittivity
    for n in [1, 2, 3]:
        rel_err = np.abs((fits[n]['re'] - cc_re) / cc_re) * 100
        ax_err.plot(freqs / 1e9, rel_err,
                    color=COLORS[n], ls=STYLES[n], lw=1.8, label=LABELS[n])
    shade_gpr(ax_err)
    ax_err.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.7)
    ax_err.set_ylabel(r"$|(\varepsilon_{r, Deb}' - \varepsilon_{r, CC}') / \varepsilon_{r, CC}'|$ (%)", fontsize=11)
    ax_err.set_title("Relative error in real permittivity", fontsize=11)
    ax_err.legend(fontsize=9, loc='center left')

    # Annotate GPR-band RMS values
    for i, n in enumerate([1, 2, 3]):
        ax_err.text(0.5, 0.62 - i * 0.1,
                    f"{LABELS[n]}: GPR RMS = {fits[n]['rms_re_gpr']:.2f}%",
                    transform=ax_err.transAxes, fontsize=8, color=COLORS[n])

    plt.tight_layout()

    fig_name = f'ColeCole_vs_Debye_{sample_key}'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, fig_name + '.png'), dpi=150, bbox_inches='tight', format='png')
    plt.savefig(os.path.join(OUTPUT_DIR, fig_name + '.pdf'), dpi=300, bbox_inches='tight', format='pdf')
    print(f"プロット保存 ({sample_key}): {OUTPUT_DIR}/{fig_name}.png/.pdf")
    plt.close(fig)

    return fits


# ---------------------------------------------------------------------------
# 3. Run all samples
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _buf = io.StringIO()

    all_fits = {}
    for sample_key, params in SAMPLES.items():
        all_fits[sample_key] = run_sample(sample_key, params, _buf)

    # Timestamp line appended only to the file (not echoed to console)
    _ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _buf.write(f"\n[Generated: {_ts}]\n")

    # Write results to disk — same directory for all samples, single file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, RESULTS_FILENAME)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_buf.getvalue())

    print(f"\nAll results saved to: {out_path}")
    print(f"Per-sample figures saved to: {OUTPUT_DIR}/ColeCole_vs_Debye_<sample_key>.png/.pdf")