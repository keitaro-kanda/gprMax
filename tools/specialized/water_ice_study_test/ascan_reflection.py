#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ascan_reflection.py — round-trip reflection analysis of an at_tx A-scan.

Extracts subsurface interface reflections from a monostatic surface (at_tx) GPR
A-scan and quantifies three channels to see which reaches its detection limit
first:
  1. reflection  : interface amplitude -> R -> permittivity contrast -> ice
  2. traveltime  : two-way time -> layer thickness / index -> ice
  3. attenuation : interface amplitude ratio -> layer attenuation alpha

The medium model (permittivity / attenuation) lives in subsurface_model.py and
is imported from there; no constant is duplicated here.  JSON/plot utilities
still come from ascan_spectrum.  The ice picture (pore-filling vs excess ice)
is taken from the JSON sub-layer key (pore_ice / excess_ice).

Known limitations: multiple reflections are not modelled (they appear in the
fig2 residual); monostatic normal incidence is assumed; background subtraction
is an upper bound (no ice-free observation exists in the field); the two-way
time difference cannot separate layer thickness from index; interface roughness
breaks the flat-layer assumption.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert
from scipy.fft import rfft, irfft, rfftfreq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 地下構造モデル（.in が定める物理）。レベル更新時はここだけを触る。
import subsurface_model as sm
# 解析の手順（JSON の読み込みなど）はこれまでどおり ascan_spectrum から。
import ascan_spectrum as asp


# ---------------------------------------------------------------------------
# [EDIT HERE] configuration
# ---------------------------------------------------------------------------
# Geometry / placement and the real-data ice fraction come from ascan_spectrum
# (single source of truth, kept in sync with the .in files; no duplication).
TX_HEIGHT_M = sm.TX_HEIGHT               # tx/rx height above surface [m]
ICE_TOP_M   = sm.LEVEL4_ICE_TOP_M        # depth of ice-layer top [m]
ICE_THICK_M = sm.LEVEL4_ICE_THICK_M      # ice-layer thickness [m]
R_REF_M     = sm.R_REF                   # far_1m reference distance [m]
# 実データの氷量は JSON のキーから設定されたあとに読む必要があるので、
# ここでは定数にせず関数で取る（描像 pore/excess でも値が変わるため）。


def detected_ice_vol():
    return sm.level4_ice_volume_fraction()

C = 299_792_458.0                # speed of light [m/s] (traveltime/geometry only)

NOISE_BAND_NS      = (8.0, 40.0) # window used to measure the noise floor [ns]
EVENT_WINDOW_NS    = 2.0         # half-width of the event search window [ns]
GATE_HALFWIDTH_NS  = 3.0         # half-width of the attenuation time gate [ns]
BELOW_FLOOR_FACTOR = 3.0         # peak below this * floor_rms is "below floor"
ICE_VOL_SWEEP      = (0.005, 0.01, 0.05, 0.10)   # swept ice fractions
CENTER_FREQ_HZ     = 1.25e9      # representative frequency [Hz]
FIG_DPI            = 150

# JSON input/output settings based on ascan_reflection_spectrum.py
JSON_PATH = asp.JSON_PATH
AT_TX_KEY = 'at_tx'
REF_KEY   = 'far_1m'

OUTPUT_SUBDIRNAME = 'ascan_reflection'

BACKGROUND_TRACE_PATH = '/Volumes/SSD_Kanda_BUFFALO/gprMax/domain_3x4/water_ice_study_test/free_space_test/excitation_LUPEX_dx00025/at_tx/result/Ascan.out'       # 氷なし at_tx の .out。空なら差分しない


# ---------------------------------------------------------------------------
# Medium adapter (the only code that touches ascan_spectrum)
# ---------------------------------------------------------------------------
@contextmanager
def _ice_fraction(v):
    """Temporarily set subsurface_model's ice volume fraction so its mixing
    model is evaluated at `v` (used for the detection-limit sweep). v=None keeps
    the module default (the real-data fraction). No constants are duplicated.
    The ice picture (pore / excess) is not touched here, so the sweep always
    uses the picture selected from the JSON key."""
    if v is None:
        yield
        return
    spec, pct = sm.LEVEL4_ICE_SPEC, sm.LEVEL4_ICE_VOL_PCT
    sm.LEVEL4_ICE_SPEC = "vol"
    sm.LEVEL4_ICE_VOL_PCT = v * 100.0
    try:
        yield
    finally:
        sm.LEVEL4_ICE_SPEC, sm.LEVEL4_ICE_VOL_PCT = spec, pct


def _index_from_eps(eps):
    """Refractive index n = Re sqrt(eps' - i|eps''|) from an (eps', eps'') pair."""
    er, ei = eps
    er = np.atleast_1d(np.asarray(er, dtype=float))
    ei = np.atleast_1d(np.asarray(ei, dtype=float))
    return np.sqrt(er - 1j * np.abs(ei)).real


@dataclass
class MediumModel:
    """Thin wrapper over ascan_spectrum. `ice_vol` selects the ice fraction for
    the sweep; None uses the module default (the real-data fraction)."""
    ice_vol: float | None = None

    def regolith_index(self, f):
        return _index_from_eps(sm.level3_eps(np.atleast_1d(f)))

    def regolith_alpha(self, f):
        return np.atleast_1d(sm.level3_alpha(np.atleast_1d(f)))

    def ice_index(self, f):
        with _ice_fraction(self.ice_vol):
            return _index_from_eps(sm.level4_eps(np.atleast_1d(f), True))

    def ice_alpha(self, f):
        with _ice_fraction(self.ice_vol):
            return np.atleast_1d(sm.level4_alpha(np.atleast_1d(f), True))

    def describe(self):
        with _ice_fraction(self.ice_vol):
            return sm.describe_level4_medium()


# ---------------------------------------------------------------------------
# Round-trip reflection theory (pure functions of refractive index n)
# ---------------------------------------------------------------------------
def r_eff_round_trip(depth_index, layer_L, layer_n):
    """Apparent source distance r_eff = 2h + 2 Sum L_j / n_j."""
    r = np.asarray(2.0 * TX_HEIGHT_M, dtype=float)
    for j in range(depth_index):
        r = r + 2.0 * layer_L[j] / np.asarray(layer_n[j], dtype=float)
    return r


def optical_path_round_trip(depth_index, layer_L, layer_n):
    """Two-way optical path P = 2h + 2 Sum n_j L_j [m] (traveltime = P/c)."""
    p = np.asarray(2.0 * TX_HEIGHT_M, dtype=float)
    for j in range(depth_index):
        p = p + 2.0 * np.asarray(layer_n[j], dtype=float) * layer_L[j]
    return p


def travel_time(depth_index, layer_L, layer_n):
    return optical_path_round_trip(depth_index, layer_L, layer_n) / C


def absorption_round_trip(depth_index, layer_L, layer_alpha):
    """Two-way absorption A = exp(-2 Sum alpha_j L_j)."""
    s = np.asarray(0.0, dtype=float)
    for j in range(depth_index):
        s = s + np.asarray(layer_alpha[j], dtype=float) * layer_L[j]
    return np.exp(-2.0 * s)


def reflection_coefficient(n_above, n_below):
    """Interface reflection R = (n_above - n_below)/(n_above + n_below)."""
    na = np.asarray(n_above, dtype=float)
    nb = np.asarray(n_below, dtype=float)
    return (na - nb) / (na + nb)


def roundtrip_transmission(index_pairs):
    """Product of two-way transmission across shallower interfaces:
    Prod 4 n_a n_b / (n_a + n_b)^2."""
    t = np.asarray(1.0, dtype=float)
    for na, nb in index_pairs:
        na = np.asarray(na, dtype=float)
        nb = np.asarray(nb, dtype=float)
        t = t * (4.0 * na * nb / (na + nb) ** 2)
    return t


def invert_layer_index(dt, L_layer):
    """Layer index from the two-way time difference: n = dt c / (2 L)."""
    return dt * C / (2.0 * L_layer)


def invert_layer_alpha(A_bot, A_top, R_top, R_bot, G_top, G_bot, T_top, L_layer):
    """Layer alpha from the top/bottom amplitude ratio:
    alpha = -ln( (A_bot/A_top)(|R_top|/|R_bot|)(G_top/G_bot)/T_top ) / (2 L)."""
    ratio = (np.asarray(A_bot) / np.asarray(A_top)) \
        * (np.abs(R_top) / np.abs(R_bot)) \
        * (np.asarray(G_top) / np.asarray(G_bot)) / np.asarray(T_top)
    return -np.log(ratio) / (2.0 * L_layer)


@dataclass
class Interface:
    name: str
    depth_index: int
    n_above: object
    n_below: object
    shallower_pairs: list
    layer_L: list
    layer_n: list
    layer_alpha: list


def build_interfaces(medium, f):
    """Three interfaces of the two-layer model:
    0 = surface (vacuum/regolith), 1 = ice-top (regolith/ice), 2 = ice-bottom.
    f may be a scalar (representative freq) or an array (full spectrum)."""
    n0 = np.ones_like(np.atleast_1d(np.asarray(f, dtype=float)))

    def _b(x):
        x = np.atleast_1d(np.asarray(x)).ravel()
        return x if x.shape == n0.shape else np.broadcast_to(x, n0.shape)

    n_reg = _b(medium.regolith_index(f))
    n_ice = _b(medium.ice_index(f))
    a_reg = _b(medium.regolith_alpha(f))
    a_ice = _b(medium.ice_alpha(f))
    n0 = _b(n0)

    return [
        Interface("surface", 0, n0, n_reg, [], [], [], []),
        Interface("ice_top", 1, n_reg, n_ice,
                  [(n0, n_reg)], [ICE_TOP_M], [n_reg], [a_reg]),
        Interface("ice_bot", 2, n_ice, n_reg,
                  [(n0, n_reg), (n_reg, n_ice)],
                  [ICE_TOP_M, ICE_THICK_M], [n_reg, n_ice], [a_reg, a_ice]),
    ]


def event_transfer(iface, f):
    """Complex transfer function H(f) = G T R A exp(-2 pi i f (P - R_ref)/c).

    The phase uses the frequency-dependent optical path (so dispersion is not
    double-counted) and subtracts the 1 m reference delay carried by far_1m, so
    that irfft(E_ref * H) places the event at its true arrival time."""
    f = np.asarray(f, dtype=float)
    G = np.sqrt(R_REF_M / r_eff_round_trip(iface.depth_index, iface.layer_L, iface.layer_n))
    T = roundtrip_transmission(iface.shallower_pairs)
    R = reflection_coefficient(iface.n_above, iface.n_below)
    A = absorption_round_trip(iface.depth_index, iface.layer_L, iface.layer_alpha)
    P = optical_path_round_trip(iface.depth_index, iface.layer_L, iface.layer_n)
    return G * T * R * A * np.exp(-2j * np.pi * f * (P - R_REF_M) / C)


@dataclass
class EventTheory:
    name: str
    t: float
    R: float
    G: float
    T: float
    A: float
    r_eff: float
    peak: float = 0.0
    peak_db: float = 0.0


def _scalar(x):
    return float(np.ravel(np.asarray(x))[0])


def theory_events(medium, freqs, E_ref, dt, t0=0.0):
    """Theory quantities per event. Arrival time = geometric two-way time + t0,
    where t0 is the source emission delay (from far_1m); this stays valid even
    when an interface reflection vanishes (R->0). The envelope peak amplitude
    comes from the reference pulse convolved with H(f)."""
    ifaces = build_interfaces(medium, freqs)
    ifc = build_interfaces(medium, CENTER_FREQ_HZ)   # representative R/G/T/A
    n_time = 2 * (len(freqs) - 1)
    out = []
    for iface, ic in zip(ifaces, ifc):
        sig = irfft(E_ref * event_transfer(iface, freqs), n=n_time)
        out.append(EventTheory(
            name=iface.name,
            t=_scalar(travel_time(ic.depth_index, ic.layer_L, ic.layer_n)) + t0,
            R=_scalar(reflection_coefficient(ic.n_above, ic.n_below)),
            G=_scalar(np.sqrt(R_REF_M / r_eff_round_trip(ic.depth_index, ic.layer_L, ic.layer_n))),
            T=_scalar(roundtrip_transmission(ic.shallower_pairs)),
            A=_scalar(absorption_round_trip(ic.depth_index, ic.layer_L, ic.layer_alpha)),
            r_eff=_scalar(r_eff_round_trip(ic.depth_index, ic.layer_L, ic.layer_n)),
            peak=float(np.abs(hilbert(sig)).max()),
        ))
    ref = out[0].peak
    for ev in out:
        ev.peak_db = 20.0 * np.log10(ev.peak / ref) if ref > 0 else np.nan
    return out


# ---------------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------------
def load_trace(path, component="Ez"):
    """Load a gprMax .out (HDF5) trace -> (time [s], amplitude)."""
    import h5py
    with h5py.File(path, "r") as f:
        dt = float(f.attrs["dt"])
        data = f[f"/rxs/rx1/{component}"][()]
    return np.arange(len(data)) * dt, np.asarray(data, dtype=float)


# ---------------------------------------------------------------------------
# Noise floor
# ---------------------------------------------------------------------------
@dataclass
class NoiseFloor:
    rms: float
    surface_peak: float
    db: float
    band_ns: tuple
    verdict: str


def measure_noise_floor(t, amp, surface_peak, band_ns=NOISE_BAND_NS):
    """RMS in the noise window, in dB relative to the surface peak, with a
    three-tier verdict."""
    lo, hi = band_ns
    m = (t >= lo * 1e-9) & (t <= hi * 1e-9)
    rms = float(np.sqrt(np.mean(amp[m] ** 2))) if np.any(m) else float("nan")
    db = 20.0 * np.log10(rms / surface_peak) if surface_peak > 0 else float("nan")
    if db < -70:
        verdict = "< -70 dB: reflection channel usable down to 0.5 vol%"
    elif db <= -60:
        verdict = "-70..-60 dB: 1 vol% ok, 0.5 vol% marginal"
    elif db <= -55:
        verdict = "-60..-55 dB: borderline (1 vol% only)"
    else:
        verdict = "> -55 dB: reflection channel not viable (fix PML/domain first)"
    return NoiseFloor(rms, surface_peak, db, band_ns, verdict)


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    name: str
    t_theory: float
    t_measured: float
    peak: float
    snr: float
    below_floor: bool
    signed_peak: float = 0.0


def envelope(amp):
    return np.abs(hilbert(amp))


def detect_event(t, amp, t_theory, floor_rms, half_ns=EVENT_WINDOW_NS):
    """Envelope peak in a window around the theory arrival. Records the signed
    sample at the peak (used to recover the reflection polarity)."""
    env = envelope(amp)
    m = (t >= t_theory - half_ns * 1e-9) & (t <= t_theory + half_ns * 1e-9)
    if not np.any(m):
        return Detection("", t_theory, np.nan, 0.0, 0.0, True, 0.0)
    idx = np.where(m)[0]
    k = idx[np.argmax(env[idx])]
    peak = float(env[k])
    snr = peak / floor_rms if floor_rms > 0 else np.inf
    return Detection("", t_theory, float(t[k]), peak, snr,
                     peak < BELOW_FLOOR_FACTOR * floor_rms, float(amp[k]))


def detect_all(t, amp, events, floor_rms):
    dets = []
    for ev in events:
        d = detect_event(t, amp, ev.t, floor_rms)
        d.name = ev.name
        dets.append(d)
    return dets


# ---------------------------------------------------------------------------
# Three channels
# ---------------------------------------------------------------------------
def channel_reflection(dets, events):
    """Invert R at each interface using the surface reflection as internal
    reference. Magnitude from the envelope-peak ratio, sign from the signed-peak
    polarity relative to the surface."""
    ev0, d0 = events[0], dets[0]
    base, s0 = d0.peak, d0.signed_peak
    out = {}
    for ev, d in zip(events, dets):
        if d.below_floor or base <= 0:
            R = np.nan
        else:
            corr = (ev0.G * ev0.T * ev0.A) / (ev.G * ev.T * ev.A)
            pol = float(np.sign(d.signed_peak * s0)) if (s0 and d.signed_peak) else 1.0
            R = (d.peak / base) * corr * ev0.R * pol
        out[ev.name] = {"R_theory": ev.R, "R_measured": float(R),
                        "residual": float(R - ev.R) if np.isfinite(R) else np.nan}
    return out


def channel_traveltime(dets, events, events_noice):
    """Match arrivals, invert the layer index from the top/bottom time
    difference, and report the shift versus the ice-free theory."""
    by = {e.name: e for e in events}
    byn = {e.name: e for e in events_noice}
    dby = {d.name: d for d in dets}
    per = {}
    for name in by:
        e, d = by[name], dby[name]
        per[name] = {"t_theory": e.t, "t_measured": d.t_measured,
                     "residual": float(d.t_measured - e.t) if np.isfinite(d.t_measured) else np.nan,
                     "delta_vs_noice": float(e.t - byn[name].t)}
    dt_meas = dby["ice_bot"].t_measured - dby["ice_top"].t_measured
    dt_theory = by["ice_bot"].t - by["ice_top"].t
    layer = {"dt_measured": float(dt_meas), "dt_theory": float(dt_theory),
             "n_layer_measured": float(invert_layer_index(dt_meas, ICE_THICK_M)),
             "n_layer_theory": float(invert_layer_index(dt_theory, ICE_THICK_M))}
    return {"per_event": per, "layer": layer}


def _gate(t, amp, t_center, half_ns=GATE_HALFWIDTH_NS):
    m = (t >= t_center - half_ns * 1e-9) & (t <= t_center + half_ns * 1e-9)
    g = np.zeros_like(amp)
    g[m] = amp[m]
    return g


def channel_attenuation(t, amp, dets, events, medium):
    """Gate the ice-top and ice-bottom events and invert the layer alpha(f)
    from their spectral amplitude ratio (README 4.7). T_top is the two-way
    transmission of the ice-top interface = (ice_bot cumulative T)/(ice_top)."""
    dby = {d.name: d for d in dets}
    eby = {e.name: e for e in events}
    n = len(t)
    freqs = rfftfreq(n, t[1] - t[0])
    S_top = rfft(_gate(t, amp, dby["ice_top"].t_measured))
    S_bot = rfft(_gate(t, amp, dby["ice_bot"].t_measured))
    band = (freqs > 0.6e9) & (freqs < 1.9e9)
    with np.errstate(divide="ignore", invalid="ignore"):
        A_ratio = np.abs(S_bot) / np.abs(S_top)
    et, eb = eby["ice_top"], eby["ice_bot"]
    T_top = (eb.T / et.T) if et.T else et.T
    alpha_f = invert_layer_alpha(A_ratio, 1.0, et.R, eb.R, et.G, eb.G, T_top, ICE_THICK_M)
    return {
        "freqs": freqs[band],
        "alpha_measured": alpha_f[band],
        "alpha_theory": np.atleast_1d(medium.ice_alpha(freqs[band])),
        "alpha_at_center": float(np.interp(CENTER_FREQ_HZ, freqs[band], alpha_f[band])),
        "alpha_theory_at_center": float(medium.ice_alpha(CENTER_FREQ_HZ)[0]),
    }


# ---------------------------------------------------------------------------
# Detection-limit sweep (fig6, the main deliverable)
# ---------------------------------------------------------------------------
def _cross(vols, snr):
    """Ice fraction where SNR crosses 1 (linear interpolation)."""
    snr = np.asarray(snr, dtype=float)
    for i in range(len(vols) - 1):
        a, b = snr[i], snr[i + 1]
        if (a - 1) * (b - 1) <= 0 and b != a:
            return float(vols[i] + (1 - a) / (b - a) * (vols[i + 1] - vols[i]))
    return np.nan


def sweep_detection_limits(freqs, E_ref, dt, t0, floor_db, traveltime_noise_ns,
                           alpha_noise, ice_vols=ICE_VOL_SWEEP):
    """SNR of each channel versus ice fraction, and the SNR=1 crossing.
    Signal/noise: reflection = ice-top peak / floor; traveltime = bottom-time
    shift vs ice-free / numerical-dispersion residual; attenuation = layer
    alpha shift vs ice-free / LSR residual."""
    floor_lin = 10 ** (floor_db / 20.0)
    vols = np.asarray(ice_vols, dtype=float)
    ev0 = theory_events(MediumModel(0.0), freqs, E_ref, dt, t0)
    alpha0 = float(MediumModel(0.0).ice_alpha(CENTER_FREQ_HZ)[0])
    snr_refl, snr_time, snr_atten = [], [], []
    for v in vols:
        med = MediumModel(float(v))
        evs = theory_events(med, freqs, E_ref, dt, t0)
        snr_refl.append((evs[1].peak / evs[0].peak) / floor_lin)
        snr_time.append(abs((evs[2].t - ev0[2].t) * 1e9) / traveltime_noise_ns)
        alpha_v = float(med.ice_alpha(CENTER_FREQ_HZ)[0])
        snr_atten.append(abs(alpha_v - alpha0) / alpha_noise if alpha_noise > 0 else np.nan)
    return {
        "ice_vols": vols,
        "snr_reflection": np.asarray(snr_refl),
        "snr_traveltime": np.asarray(snr_time),
        "snr_attenuation": np.asarray(snr_atten),
        "limit_reflection": _cross(vols, snr_refl),
        "limit_traveltime": _cross(vols, snr_time),
        "limit_attenuation": _cross(vols, snr_atten),
    }


# ---------------------------------------------------------------------------
# Figures (English labels)
# ---------------------------------------------------------------------------
def _save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def fig1_trace(out_dir, t, amp_raw, amp_sub, events, floor):
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    tn = t * 1e9
    ax[0].plot(tn, amp_raw, lw=0.8)
    ax[0].set_ylabel("Ez (raw)"); ax[0].set_title("fig1 (a) waveform (no bg-sub)")
    if amp_sub is not None:
        ax[1].plot(tn, amp_sub, lw=0.8, color="C1")
    ax[1].set_ylabel("Ez (bg-sub)"); ax[1].set_title("(b) waveform (bg-sub)")
    env_db = 20 * np.log10(np.maximum(envelope(amp_raw), 1e-30) / events[0].peak)
    ax[2].plot(tn, env_db, lw=0.8, color="C2")
    ax[2].axhline(floor.db, ls="--", color="k", label=f"noise floor {floor.db:.1f} dB")
    ax[2].set_ylabel("envelope [dB re surface]"); ax[2].set_xlabel("time [ns]")
    ax[2].set_ylim(min(floor.db - 10, -80), 5); ax[2].set_title("(c) envelope (dB)")
    for a in ax:
        for ev in events:
            a.axvline(ev.t * 1e9, ls=":", color="gray", lw=0.8)
    ax[2].legend(loc="upper right", fontsize=8)
    _save(fig, out_dir, "fig1_trace")


def fig2_events(out_dir, dets, events, dt):
    names = [e.name for e in events]
    t_th = np.array([e.t for e in events]) * 1e9
    t_ms = np.array([d.t_measured for d in dets]) * 1e9
    a_th = np.array([e.peak for e in events]); a_th = a_th / a_th[0]
    a_ms = np.array([d.peak for d in dets]); a_ms = a_ms / a_ms[0]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].plot(t_th, t_ms, "o"); ax[0].plot(t_th, t_th, "k--", lw=0.8)
    ax[0].set_xlabel("t theory [ns]"); ax[0].set_ylabel("t measured [ns]")
    ax[0].set_title("(a) arrival time")
    ax[1].loglog(a_th, a_ms, "o"); ax[1].loglog(a_th, a_th, "k--", lw=0.8)
    ax[1].set_xlabel("amplitude / surface (theory)")
    ax[1].set_ylabel("amplitude / surface (measured)")
    ax[1].set_title("(b) amplitude vs surface")
    ax[2].bar(range(len(names)), t_ms - t_th)
    ax[2].set_xticks(range(len(names))); ax[2].set_xticklabels(names, rotation=30)
    ax[2].set_ylabel("traveltime residual [ns]"); ax[2].set_title("(c) residual")
    ax[2].annotate("multiple reflections (not modelled)", xy=(0.02, 0.9),
                   xycoords="axes fraction", fontsize=8, color="crimson")
    _save(fig, out_dir, "fig2_events")


def fig3_reflection(out_dir, vols, R_theory_curve, meas_points, floor):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.asarray(vols) * 100, 20 * np.log10(np.abs(R_theory_curve)),
            "-", label="theory |R_top|")
    for v, R in meas_points:
        if np.isfinite(R):
            ax.plot(v * 100, 20 * np.log10(abs(R)), "o", color="C1", label="measured")
    ax.axhline(floor.db, ls="--", color="k", label=f"noise floor {floor.db:.1f} dB")
    ax.set_xlabel("ice content [vol%]"); ax.set_ylabel("|R_top| [dB]")
    ax.set_title("fig3 reflection channel"); ax.legend()
    _save(fig, out_dir, "fig3_reflection")


def fig4_traveltime(out_dir, vols, dt_bot_vs_noice_ns):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.asarray(vols) * 100, dt_bot_vs_noice_ns, "o-")
    ax.set_xlabel("ice content [vol%]")
    ax.set_ylabel("bottom traveltime shift vs ice-free [ns]")
    ax.set_title("fig4 traveltime channel")
    _save(fig, out_dir, "fig4_traveltime")


def fig5_attenuation(out_dir, atten):
    fig, ax = plt.subplots(figsize=(7, 5))
    f = atten["freqs"] / 1e9
    ax.plot(f, atten["alpha_measured"], ".", ms=3, label="measured")
    if len(np.atleast_1d(atten["alpha_theory"])) == len(f):
        ax.plot(f, atten["alpha_theory"], "-", label="theory")
    ax.set_xlabel("frequency [GHz]"); ax.set_ylabel(r"layer $\alpha$ [Np/m]")
    ax.set_title("fig5 attenuation channel"); ax.legend()
    _save(fig, out_dir, "fig5_attenuation")


def fig6_channels(out_dir, sweep):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    v = sweep["ice_vols"] * 100
    ax.loglog(v, np.maximum(sweep["snr_reflection"], 1e-3), "o-", label="reflection")
    ax.loglog(v, np.maximum(sweep["snr_traveltime"], 1e-3), "s-", label="traveltime")
    ax.loglog(v, np.maximum(sweep["snr_attenuation"], 1e-3), "^-", label="attenuation")
    ax.axhline(1.0, ls="--", color="k", label="detection limit SNR=1")
    ax.set_xlabel("ice content [vol%]"); ax.set_ylabel("SNR")
    ax.set_title("fig6 channel detection limits"); ax.legend()
    ax.text(0.02, 0.02, "main deliverable", transform=ax.transAxes,
            fontsize=8, color="gray")
    _save(fig, out_dir, "fig6_channels")


# ---------------------------------------------------------------------------
# Numeric outputs
# ---------------------------------------------------------------------------
def write_events_csv(path, dets, events, refl):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["event", "t_theory_ns", "t_measured_ns", "peak", "peak_db",
                    "R_theory", "R_measured", "R_residual", "below_floor"])
        for e, d in zip(events, dets):
            r = refl.get(e.name, {})
            w.writerow([e.name, f"{e.t*1e9:.4f}", f"{d.t_measured*1e9:.4f}",
                        f"{d.peak:.6e}", f"{e.peak_db:.2f}",
                        f"{r.get('R_theory', np.nan):.6f}",
                        f"{r.get('R_measured', np.nan):.6f}",
                        f"{r.get('residual', np.nan):.6f}", d.below_floor])


def write_channels_csv(path, sweep):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ice_vol", "snr_reflection", "snr_traveltime", "snr_attenuation"])
        for i, v in enumerate(sweep["ice_vols"]):
            w.writerow([f"{v:.4f}", f"{sweep['snr_reflection'][i]:.4f}",
                        f"{sweep['snr_traveltime'][i]:.4f}",
                        f"{sweep['snr_attenuation'][i]:.4f}"])
        w.writerow([])
        w.writerow(["limit_reflection_vol", f"{sweep['limit_reflection']:.4f}"])
        w.writerow(["limit_traveltime_vol", f"{sweep['limit_traveltime']:.4f}"])
        w.writerow(["limit_attenuation_vol", f"{sweep['limit_attenuation']:.4f}"])


def write_run_info(path, medium, floor, tt, atten):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("=== ascan_reflection run_info ===\n\n[config]\n")
        for k in ("TX_HEIGHT_M", "ICE_TOP_M", "ICE_THICK_M", "R_REF_M",
                  "NOISE_BAND_NS", "EVENT_WINDOW_NS", "CENTER_FREQ_HZ"):
            fh.write(f"  {k} = {globals()[k]}\n")
        fh.write(f"  ICE_MODEL = {sm.LEVEL4_ICE_MODEL}\n")
        fh.write(f"  ICE_VOL = {sm.LEVEL4_ICE_VOL_PCT / 100.0:.4f}\n")
        fh.write("\n[medium]\n  " + medium.describe().replace("\n", "\n  ") + "\n")
        fh.write("\n[noise floor]\n")
        fh.write(f"  band = {floor.band_ns} ns\n  rms = {floor.rms:.6e}\n")
        fh.write(f"  vs surface peak = {floor.db:.2f} dB\n  verdict = {floor.verdict}\n")
        fh.write("\n[traveltime channel]\n")
        fh.write(f"  dt_measured = {tt['layer']['dt_measured']*1e9:.4f} ns\n")
        fh.write(f"  n_layer measured = {tt['layer']['n_layer_measured']:.5f} "
                 f"(theory {tt['layer']['n_layer_theory']:.5f})\n")
        fh.write("\n[attenuation channel]\n")
        fh.write(f"  alpha(1.25GHz) measured = {atten['alpha_at_center']:.5f} Np/m "
                 f"(theory {atten['alpha_theory_at_center']:.5f})\n")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_analysis(at_tx_path, ref_path, ice_vol, bg_path="", use_bg_sub=False, out_dir=""):
    """Run one condition."""
    t_at, a_at = load_trace(at_tx_path)
    t_ref, a_ref = load_trace(ref_path)
    
    if bg_path and use_bg_sub:
        t_ni, a_ni = load_trace(bg_path)
    else:
        t_ni, a_ni = t_at, a_at

    dt = t_at[1] - t_at[0]
    n = len(a_at)
    a_ref = np.pad(a_ref, (0, n - len(a_ref))) if len(a_ref) < n else a_ref[:n]
    freqs = rfftfreq(n, dt)
    E_ref = rfft(a_ref)

    medium = MediumModel(ice_vol)
    medium0 = MediumModel(0.0)
    # Source emission delay from far_1m: its envelope peaks at t0 + R_ref/c.
    t0 = int(np.argmax(np.abs(hilbert(a_ref)))) * dt - R_REF_M / C
    events = theory_events(medium, freqs, E_ref, dt, t0)
    events_noice = theory_events(medium0, freqs, E_ref, dt, t0)

    surf = detect_event(t_ni, a_ni, events[0].t, floor_rms=1.0)
    floor = measure_noise_floor(t_ni, a_ni, surface_peak=surf.peak)

    a_sub = a_at - np.interp(t_at, t_ni, a_ni) if use_bg_sub else None
    a_use = a_sub if a_sub is not None else a_at

    dets = detect_all(t_at, a_use, events, floor.rms)
    refl = channel_reflection(dets, events)
    tt = channel_traveltime(dets, events, events_noice)
    atten = channel_attenuation(t_at, a_use, dets, events, medium)

    sweep = sweep_detection_limits(
        freqs, E_ref, dt, t0, floor_db=floor.db,
        traveltime_noise_ns=0.025,
        alpha_noise=max(atten["alpha_theory_at_center"] * 0.1, 1e-6))

    fig1_trace(out_dir, t_at, a_at, a_sub, events, floor)
    fig2_events(out_dir, dets, events, dt)
    R_curve = np.array([reflection_coefficient(
        MediumModel(v).regolith_index(CENTER_FREQ_HZ)[0],
        MediumModel(v).ice_index(CENTER_FREQ_HZ)[0]) for v in ICE_VOL_SWEEP])
    fig3_reflection(out_dir, ICE_VOL_SWEEP, R_curve,
                    [(ice_vol, refl.get("ice_top", {}).get("R_measured", np.nan))], floor)
    dt_bot = [abs((theory_events(MediumModel(v), freqs, E_ref, dt, t0)[2].t
                   - events_noice[2].t) * 1e9) for v in ICE_VOL_SWEEP]
    fig4_traveltime(out_dir, ICE_VOL_SWEEP, dt_bot)
    fig5_attenuation(out_dir, atten)
    fig6_channels(out_dir, sweep)

    write_events_csv(os.path.join(out_dir, "events.csv"), dets, events, refl)
    write_channels_csv(os.path.join(out_dir, "channels.csv"), sweep)
    write_run_info(os.path.join(out_dir, "run_info.txt"), medium, floor, tt, atten)

    return {"out_dir": out_dir, "events": events, "detections": dets,
            "floor": floor, "reflection": refl, "traveltime": tt,
            "attenuation": atten, "sweep": sweep}


def main(argv=None):
    print("=== ascan_reflection ===")
    
    # ascan_spectrum の load_paths を用いて対話的に解析対象を決定する
    level, kind, rx_paths, reference = asp.load_paths(JSON_PATH)
    
    if AT_TX_KEY not in rx_paths:
        print(f"Error: 選択した階層に '{AT_TX_KEY}' がありません。\n利用可能な rx: {', '.join(sorted(rx_paths))}")
        return 1
        
    at_tx_path = rx_paths[AT_TX_KEY]
    ref_path = reference if isinstance(reference, str) else reference.get(REF_KEY, "")
    
    if not ref_path:
        print(f"Error: 参照パス '{REF_KEY}' が見つかりません。")
        return 1

    # 組成・水氷濃度・水氷の描像（pore / excess）をまとめて設定する。
    # サブ階層キーから自動判定し、判定できなければエラーで止まる。
    for _note in sm.configure_from_kind(kind, level):
        print("  " + _note)

    if 'ice_layer' in sm.LEVEL_EFFECTS.get(level, []):
        ice_vol = detected_ice_vol()
        print(f"\n# {level} / {sm.LEVEL4_ICE_MODEL} "
              f"(ice = {ice_vol * 100:.2f} vol%)")
    else:
        ice_vol = 0.0
        print(f"\n# {level}")
        
    # JSON の階層に pore_ice / excess_ice が入っているので、出力先も
    # 自動的に描像ごとに分かれる（resolve_output_dir が .out のパスを使うため）。
    asp.OUTPUT_SUBDIRNAME = OUTPUT_SUBDIRNAME
    out_dir_base = asp.resolve_output_dir(level, rx_paths)

    for bg in (False, True):
        if bg and not BACKGROUND_TRACE_PATH:
            print("  [bg_sub=True] Skipped (BACKGROUND_TRACE_PATH が設定されていません)")
            continue
            
        out_dir = os.path.join(out_dir_base, "bg_sub" if bg else "no_bg_sub")
        os.makedirs(out_dir, exist_ok=True)
        
        res = run_analysis(at_tx_path, ref_path, ice_vol, BACKGROUND_TRACE_PATH, bg, out_dir)
        print(f"  [bg_sub={bg}] out_dir = {res['out_dir']}  "
              f"floor = {res['floor'].db:.1f} dB ({res['floor'].verdict})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())