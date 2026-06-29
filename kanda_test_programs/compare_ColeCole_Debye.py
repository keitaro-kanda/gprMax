"""
Cole-Cole (Boivin+2022, 20 w% ilmenite) vs multi-pole Debye approximation
==========================================================================
Boivin et al. (2022) JGR Planets, Table 4, 20 w% ilmenite in bytownite:
    eps_inf = 3.792, Delta_eps = 0.420, tau = 5.036e-11 s, alpha = 0.756

gprMax supports only Debye/Lorentz/Drude dispersion models.
This script fits the Boivin Cole-Cole spectrum with 1-, 2-, and 3-pole
Debye models and visualises the approximation error across 0.1-10 GHz,
with emphasis on the GPR band (0.5-1.25 GHz).

Output:
    - Frequency-dependent eps', eps'', loss tangent
    - Relative error of each Debye fit vs Cole-Cole
    - Fitted parameters printed in gprMax #add_dispersion_debye syntax

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
# 0. Parameters
# ---------------------------------------------------------------------------

# Boivin+2022 Table 4, 20 w% ilmenite in bytownite
EPS_INF   = 3.792
DELTA_EPS = 0.420
TAU_CC    = 5.036e-11   # s  (~50.4 ps)
ALPHA_CC  = 0.756       # Cole-Cole exponent (1 = Debye, <1 = broadened)
SIGMA_DC  = 1e-5        # S/m (~0; upper bound from Table 4)

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


cc_re, cc_im = cole_cole(freqs, EPS_INF, DELTA_EPS, TAU_CC, ALPHA_CC, SIGMA_DC)
cc_tand = cc_im / cc_re

# ---------------------------------------------------------------------------
# 2. Multi-pole Debye model and least-squares fitting
# ---------------------------------------------------------------------------

def multi_debye(f, poles):
    """
    poles: list of (delta_eps_i, tau_i)
    eps*(w) = eps_inf + sum_i  delta_eps_i / (1 + i*w*tau_i)
    Returns (eps_real, eps_imag).
    """
    w = 2 * np.pi * f
    eps_star = np.full_like(w, EPS_INF, dtype=complex)
    for de, tau in poles:
        eps_star += de / (1 + 1j * w * tau)
    return eps_star.real, -eps_star.imag


def fit_debye(n_poles, tau_seeds=None):
    """
    Fit n_poles Debye poles to the Cole-Cole spectrum.
    Optimisation variables: [delta_eps_1,..,delta_eps_n, log(tau_1),..,log(tau_n)]
    Constraints: delta_eps_i >= 0, tau_i >= TAU_MIN.
    GPR band is up-weighted by factor 5 in the residual.
    """
    if tau_seeds is None:
        if n_poles == 1:
            tau_seeds = [TAU_CC]
        elif n_poles == 2:
            tau_seeds = [TAU_CC * 0.2, TAU_CC * 5.0]
        else:
            tau_seeds = [TAU_CC * 0.05, TAU_CC * 1.0, TAU_CC * 20.0]

    # Clip seeds to stay above TAU_MIN
    tau_seeds = [max(t, TAU_MIN * 1.1) for t in tau_seeds]
    x0 = np.array([DELTA_EPS / n_poles] * n_poles
                  + [np.log(t) for t in tau_seeds])

    lo = [0.0] * n_poles + [np.log(TAU_MIN)] * n_poles
    hi = [DELTA_EPS * 3] * n_poles + [np.log(TAU_CC * 200)] * n_poles

    # GPR-band frequency weight
    w_freq = np.ones(N_F)
    w_freq[(freqs >= GPR_LO) & (freqs <= GPR_HI * 2)] *= 5.0

    def residuals(x):
        des  = x[:n_poles]
        taus = np.exp(x[n_poles:])
        d_re, d_im = multi_debye(freqs, list(zip(des, taus)))
        res_re = (d_re - cc_re) / cc_re * w_freq
        res_im = (d_im - cc_im) / np.maximum(cc_im, 1e-6) * w_freq
        return np.concatenate([res_re, res_im])

    result = least_squares(residuals, x0, bounds=(lo, hi),
                           method='trf', ftol=1e-12, xtol=1e-12, max_nfev=10000)

    des  = result.x[:n_poles]
    taus = np.exp(result.x[n_poles:])
    order = np.argsort(taus)
    return list(zip(des[order], taus[order])), result.cost


# Run fits for 1, 2, 3 poles
fits = {}
for n in [1, 2, 3]:
    poles, cost = fit_debye(n)
    d_re, d_im = multi_debye(freqs, poles)
    gpr_mask = (freqs >= GPR_LO) & (freqs <= GPR_HI)

    def rms(a, b, ref):
        return np.sqrt(np.mean(((a - b) / np.maximum(ref, 1e-9)) ** 2)) * 100

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

# ---------------------------------------------------------------------------
# 3. Print and save results
# ---------------------------------------------------------------------------

SEP = "=" * 70
print(SEP)
print("Cole-Cole parameters  (Boivin+2022 Table 4, 20 w% ilmenite)")
print(f"  eps_inf={EPS_INF},  Delta_eps={DELTA_EPS},  "
      f"tau={TAU_CC*1e12:.1f} ps,  alpha={ALPHA_CC}")
print(f"  gprMax tau_min = {TAU_MIN*1e12:.1f} ps  (dx={DX} m)")
print(SEP)

for n in [1, 2, 3]:
    f = fits[n]
    print(f"\n--- {n}-pole Debye fit " + "-" * 45)
    for i, (de, tau) in enumerate(f['poles']):
        ok = "OK" if tau >= TAU_MIN else "FAIL (< tau_min)"
        print(f"  pole {i+1}:  Delta_eps = {de:.3f},  tau = {tau*1e12:.3f} ps  [{ok}]")
    print(f"  sum(Delta_eps) = {sum(de for de,_ in f['poles']):.2f}"
          f"  (Cole-Cole Delta_eps = {DELTA_EPS})")
    print(f"  RMS error (full band):  eps' {f['rms_re']:.2f}%,  eps'' {f['rms_im']:.2f}%")
    print(f"  RMS error (GPR band):   eps' {f['rms_re_gpr']:.2f}%,  eps'' {f['rms_im_gpr']:.2f}%")

    pole_str = "".join(f" {de:.6f} {tau:.6e}" for de, tau in f['poles'])
    print(f"\n  gprMax syntax (replace 'my_mat' with material name):")
    print(f"    #add_dispersion_debye: {n}{pole_str} my_mat")

print("\n" + SEP)
print("Recommendation: 2-pole Debye gives the best accuracy/complexity trade-off.")
print(SEP)

# ---------------------------------------------------------------------------
# 4. Save results to text file (same format as the print output above)
# ---------------------------------------------------------------------------
# Build the result string by redirecting the same print calls to a StringIO buffer,
# then write it to disk.  This guarantees the file is byte-for-byte identical to
# the console output — no risk of formatting drift between the two.

_buf = io.StringIO()

def _p(*args, **kwargs):
    """Print to both stdout and the string buffer."""
    print(*args, **kwargs)                        # console (existing behaviour)
    print(*args, **kwargs, file=_buf)             # buffer  (for the file)

_p(SEP)
_p("Cole-Cole parameters  (Boivin+2022 Table 4, 20 w% ilmenite)")
_p(f"  eps_inf={EPS_INF},  Delta_eps={DELTA_EPS},  "
   f"tau={TAU_CC*1e12:.1f} ps,  alpha={ALPHA_CC}")
_p(f"  gprMax tau_min = {TAU_MIN*1e12:.1f} ps  (dx={DX} m)")
_p(SEP)

for n in [1, 2, 3]:
    f = fits[n]
    _p(f"\n--- {n}-pole Debye fit " + "-" * 45)
    for i, (de, tau) in enumerate(f['poles']):
        ok = "OK" if tau >= TAU_MIN else "FAIL (< tau_min)"
        _p(f"  pole {i+1}:  Delta_eps = {de:.3f},  tau = {tau*1e12:.3f} ps  [{ok}]")
    _p(f"  sum(Delta_eps) = {sum(de for de, _ in f['poles']):.2f}"
       f"  (Cole-Cole Delta_eps = {DELTA_EPS})")
    _p(f"  RMS error (full band):  eps' {f['rms_re']:.2f}%,  eps'' {f['rms_im']:.2f}%")
    _p(f"  RMS error (GPR band):   eps' {f['rms_re_gpr']:.2f}%,  eps'' {f['rms_im_gpr']:.2f}%")

    pole_str = "".join(f" {de:.6f} {tau:.6e}" for de, tau in f['poles'])
    _p(f"\n  gprMax syntax (replace 'my_mat' with material name):")
    _p(f"    #add_dispersion_debye: {n}{pole_str} my_mat")

_p("\n" + SEP)
_p("Recommendation: 2-pole Debye gives the best accuracy/complexity trade-off.")
_p(SEP)

# Timestamp line appended only to the file (not echoed to console)
_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
_buf.write(f"\n[Generated: {_ts}]\n")

# Write to disk — same directory as this script, with a fixed name
output_dir = '/Volumes/SSD_Kanda_BUFFALO/test_programs_output/compare_ColeCole_debye'
outpu_file_name = "debye_fit_results.txt"
out_path = os.path.join(output_dir, outpu_file_name)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(_buf.getvalue())

print(f"\nResults saved to: {out_path}")

# ---------------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------------

COLORS = {1: 'r', 2: 'g', 3: 'b'}
STYLES = {1: '-',       2: '--',      3: ':'}
LABELS = {1: '1-pole Debye', 2: '2-pole Debye', 3: '3-pole Debye'}

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(
    "Cole-Cole (Boivin+2022, 20 w% ilmenite) vs multi-pole Debye\n"
    r"$\varepsilon_\infty$=3.792, $\Delta\varepsilon$=0.420, "
    r"$\tau$=50.4 ps, $\alpha$=0.756",
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
ax_err.set_title("Relative error in real perittivity", fontsize=11)
ax_err.legend(fontsize=9, loc='center left')

# Annotate GPR-band RMS values
for i, n in enumerate([1, 2, 3]):
    ax_err.text(0.5, 0.62 - i * 0.1,
                f"{LABELS[n]}: GPR RMS = {fits[n]['rms_re_gpr']:.2f}%",
                transform=ax_err.transAxes, fontsize=8, color=COLORS[n])


plt.tight_layout()
output_dir = '/Volumes/SSD_Kanda_BUFFALO/test_programs_output/compare_ColeCole_debye'
fig_name = 'ColeCole_vs_Debye'
plt.savefig(output_dir + '/' + fig_name + '.png', dpi=150, bbox_inches='tight', format='png')
plt.savefig(output_dir + '/' + fig_name + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
print(f"\nプロット保存: {output_dir}")
plt.show()