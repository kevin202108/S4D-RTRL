#!/usr/bin/env python3
"""
drift_models.py -- shared PA-drift operators for the drift-tracking experiments.

Why this module exists
----------------------
Until 2026-07-10 all three drift experiments applied the SAME memoryless perturbation
on top of a frozen-memory PA:

    y_drift[n] = g(t) * ( y[n] + c3(t) * y[n] * |y[n]|^2 ),   y = pa(xpd)

`g` (gain/phase) and `c3` (3rd-order AM-AM/AM-PM) are scalars applied sample-by-sample,
so the ONLY memory in the loop is the frozen DGRU's hidden state.  Nothing in the drift
changes the PA's time constants, its frequency response, or its ISI.

That matters because the S4D pole `Abar` *is* the DPD's memory time constant.  Asking
"does adapting Abar earn its 32 eligibility-trace channels?" while the drift contains no
memory change is a rigged question -- Abar has nothing to track by construction, and the
answer can only come back "no".

Which drift actually stresses Abar?
----------------------------------
Write one S4D layer's transfer function.  With s[n] = Abar*s[n-1] + h0[n] and
y[n] = sum_n C_eff[h,n]*s[h,n]:

    Y_h(z) = sum_n C[h,n] / (1 - Abar[h,n] z^-1)
           = <numerator polynomial, linear in C> / prod_n (1 - Abar[h,n] z^-1)

So **Abar spans the poles and C_eff spans the zeros.**  Inverting a PA that grew a
*pole* needs a *zero* -> C_eff can do it alone.  Inverting a PA that grew a *zero*
(a notch) needs a *pole* -> only Abar can do it.  That asymmetry decides which drift
is a fair test, and it also explains why the historical memoryless drift (which moves
neither) always made Abar adaptation look worthless.

Operators (single-variable by construction -- none of them also drift g/c3 except 'both')
----------------------------------------------------------------------------------------
memoryless : the historical drift, bit-for-bit.  g/c3 drift, no filter.  Control arm.
pole       : output matching-network detuning.  Complex pole whose radius (Q) and angle
             (centre frequency) drift.  DPD must answer with a drifting zero -> tests
             C_eff, and is expected NOT to need Abar.
zero       : transmission notch that drifts (series/shunt resonance in the output match
             or the bias decoupling network).  DPD must answer with a drifting pole ->
             this is the arm where Abar adaptation should earn its 32 trace channels.
both       : memoryless + pole together.  Realistic, but confounded -- use it to confirm
             a conclusion, never to reach one.
thermal    : self-heating envelope memory (M2), NOT YET IMPLEMENTED.

Keeping these single-variable is not a stylistic choice.  The memoryless schedule alone
drives the frozen offline DPD from EVM -47.6 dB to -2.5 dB by the last block; any effect
of Abar adaptation would be buried under that.

Because |H| of (1-p)/(1-p z^-1) and of (1-q z^-1)/(1-q) are reciprocal, the two arms at
the same r_max inflict the SAME in-band tilt and near-identical damage on the frozen
offline DPD -- a controlled contrast.  Measured at blk = N-1, r_max = 0.5 (baseline
ACLR -50.79 / EVM -47.60 / NMSE ~ -41):

    arm    |H| tilt    ACLR      EVM      NMSE
    pole     2.70 dB   -53.75   -11.37   -10.40
    zero     2.70 dB   -49.20   -10.02    -8.83

Metric warning: a pure 'pole' drift is low-pass, so it *suppresses* out-of-band regrowth
and ACLR gets BETTER while EVM collapses.  Gate the linear-memory arms on NMSE/EVM,
never on ACLR.

Both filters are normalised so |H(1)| = 1: a pure memory change, decoupled from any gain
change.  At blk = 0 every operator is the identity perturbation
(A=1, phi=0, b3=ampm=0, p=q=0), so block 0 always reproduces the undrifted PA.
"""
from __future__ import annotations
import numpy as np
import torch

# Matching-network detuning defaults.  omega is in rad/sample; at the APA_200MHz
# sample rate (fs = 983.04 MHz) a 5 MHz centre-frequency shift is 2*pi*5/983.04.
FS_DEFAULT = 983.04e6
POLE_R_MAX = 0.50                      # final pole radius (Q); 0 at blk=0
POLE_DF_MAX = 5.0e6                    # final centre-frequency shift [Hz]

ZERO_R_MAX = 0.50                      # final notch radius; 0 at blk=0
KINDS = ("memoryless", "pole", "zero", "both", "thermal")
NOMINAL = (1.0, 0.0, 0.0, 0.0)         # (A, phi, b3, ampm) = no memoryless perturbation
MEMORY_KINDS = ("pole", "zero")        # gate these on NMSE/EVM, never ACLR


def drift_params(blk, n_blocks, step_blk):
    """(A, phi, b3, ampm) -- the historical memoryless schedule, verbatim.

    Linear gain droop + phase rotation, plus growing 3rd-order AM-AM (b3) and AM-PM
    (ampm); a mid-run bias step at `step_blk` jumps the nonlinearity."""
    dl = blk / (n_blocks - 1)
    A = 1.0 - 0.05 * dl
    phi = 0.10 * dl
    b3 = 0.14 * dl
    ampm = 0.18 * dl
    if blk >= step_blk:
        A *= 0.97
        phi += 0.05
        b3 += 0.05
        ampm += 0.07
    return A, phi, b3, ampm


def pole_params(blk, n_blocks, fs=FS_DEFAULT, r_max=POLE_R_MAX, df_max=POLE_DF_MAX):
    """Complex pole p = r * exp(j*omega) of the drifting output matching network.

    Both r (Q / damping) and omega (centre frequency) ramp linearly from 0, so blk=0
    gives p = 0 -> H(z) = 1 -> the undrifted PA."""
    dl = blk / (n_blocks - 1)
    r = r_max * dl
    omega = 2.0 * np.pi * (df_max * dl) / fs
    return r * np.exp(1j * omega)


def zero_params(blk, n_blocks, fs=FS_DEFAULT, r_max=ZERO_R_MAX, df_max=POLE_DF_MAX):
    """Complex transmission zero q = r * exp(j*omega) of a drifting notch.

    Same ramp as `pole_params`; q = 0 at blk=0 -> H(z) = 1."""
    dl = blk / (n_blocks - 1)
    r = r_max * dl
    omega = 2.0 * np.pi * (df_max * dl) / fs
    return r * np.exp(1j * omega)


def _apply_pole(yc, p):
    """y_d[n] = (1-p)*y[n] + p*y_d[n-1], run along the time axis of (B,T) complex.

    Sequential by construction -- this is the memory the DPD has to invert."""
    if p == 0:
        return yc
    b0 = 1.0 - p
    out = torch.empty_like(yc)
    state = torch.zeros(yc.shape[:-1], dtype=yc.dtype)
    for n in range(yc.shape[-1]):
        state = b0 * yc[..., n] + p * state
        out[..., n] = state
    return out


def _apply_zero(yc, q):
    """y_d[n] = (y[n] - q*y[n-1]) / (1-q).  One-tap FIR notch, |H(1)| = 1."""
    if q == 0:
        return yc
    prev = torch.zeros_like(yc)
    prev[..., 1:] = yc[..., :-1]
    return (yc - q * prev) / (1.0 - q)


def make_drift(pa, blk, n_blocks, step_blk, kind="memoryless", *,
               fs=FS_DEFAULT, r_max=None, df_max=POLE_DF_MAX):
    """Forward-only drifted PA operator: (B,T,2) -> (B,T,2).

    `kind='memoryless'` reproduces the historical `make_pa_fn` bit-for-bit.  The linear
    filters are applied AFTER the static nonlinearity: the matching network sits after
    the transistor, so the device's memoryless distortion is generated first, then
    filtered.  `r_max` defaults to POLE_R_MAX / ZERO_R_MAX per kind."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if kind == "thermal":
        raise NotImplementedError("M2 self-heating envelope memory not implemented yet")

    A, phi, b3, ampm = (NOMINAL if kind in MEMORY_KINDS
                        else drift_params(blk, n_blocks, step_blk))
    g = complex(A * np.cos(phi), A * np.sin(phi))
    c3 = complex(b3, ampm)
    p = q = 0
    if kind in ("pole", "both"):
        p = pole_params(blk, n_blocks, fs, r_max or POLE_R_MAX, df_max)
    if kind == "zero":
        q = zero_params(blk, n_blocks, fs, r_max or ZERO_R_MAX, df_max)

    def pa_fn(xpd):
        with torch.no_grad():
            y = pa(xpd)
        yc = y[..., 0] + 1j * y[..., 1]
        pw = yc.abs() ** 2
        pw = pw / (pw.mean() + 1e-12)                 # normalize power -> stable coeff scale
        yc = g * (yc + c3 * yc * pw)
        yc = _apply_pole(yc, p)
        yc = _apply_zero(yc, q)
        return torch.stack([yc.real, yc.imag], dim=-1)
    return pa_fn


def filter_response(p=0, q=0, fs=FS_DEFAULT, bw=200e6, n=9):
    """|H(f)| and arg H(f) of ((1-q z^-1)/(1-q)) * ((1-p)/(1 - p z^-1)) across the main
    channel.  A drift that barely tilts the in-band response cannot stress anything."""
    f = np.linspace(-bw / 2, bw / 2, n)
    z = np.exp(2j * np.pi * f / fs)
    H = np.ones_like(z)
    if p:
        H = H * (1 - p) / (1 - p / z)
    if q:
        H = H * (1 - q / z) / (1 - q)
    return f, np.abs(H), np.angle(H)
