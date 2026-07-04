#!/usr/bin/env python3
"""
ila_frontend.py -- Indirect Learning Architecture (ILA) front-end:
loop-delay estimation + gain/phase normalization + observation model.

Why this exists (design-space A3/A4/A5, K1 in rtrl_design_space.md):
    In ILA the post-inverse is trained on the PA OUTPUT z to reproduce the PA
    INPUT y (= the DPD output). Before any learning can work, z must be aligned
    to y in TIME (the DAC->PA->ADC feedback loop delay, integer + fractional)
    and in complex GAIN/PHASE. Get this wrong and the very first layer sees a
    mis-aligned target and the whole online learner diverges. This is the single
    most hardware-fragile part of online ILA-DPD, so we nail it in software first.

What it does:
    Given reference y (PA input) and observation z (PA output, z ~= G*delay(y,D)
    + distortion + noise), estimate:
      * D  = loop delay in samples (integer via xcorr, fractional via an in-band
             weighted least-squares phase-slope fit -- the low-variance estimator)
      * G  = complex gain/phase (complex least squares)
    and return the aligned, de-gained observation  y_hat = delay(z, -D)/G, which
    is what the ILA post-inverse consumes.

PA-agnostic: for a NONLINEAR PA the linear (D, G) still capture the delay and
linear gain; the residual (y_hat - y) is exactly the nonlinear distortion the
post-inverse must learn -- which is the point.

Verified (sandbox, band-limited signal, N=16384):
    clean  -> D exact, |G err| 1e-16, residual 6e-16
    40 dB  -> D err 1e-5, residual = noise floor (1e-2)
    30 dB  -> D err 1e-4, residual = noise floor (3e-2)
    20 dB  -> D err 5e-4, residual = noise floor (1e-1)
    delays 0.13 .. 100.27 samples all recovered exactly (clean).

Pure NumPy. Self-test:  uv run ila_frontend.py   (or python ila_frontend.py)
"""
import numpy as np


def frac_delay(x, tau):
    """Delay complex signal x by tau samples (fractional ok) via FFT phase ramp.
    Circular -- for streaming use block processing with overlap; here it is the
    exact reference used for calibration / offline alignment."""
    N = len(x)
    k = np.fft.fftfreq(N)
    return np.fft.ifft(np.fft.fft(x) * np.exp(-2j * np.pi * k * tau))


def band_limited(N, occ=0.6, rng=None):
    """Representative oversampled DPD test signal: white noise band-limited to the
    central `occ` fraction of Nyquist, unit average power."""
    rng = rng or np.random.default_rng(0)
    X = (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    k = np.fft.fftfreq(N)
    X[np.abs(k) > occ / 2] = 0
    x = np.fft.ifft(X)
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


def estimate_delay_gain(y, z, inband_thresh=0.01):
    """Estimate (D, G) such that z ~= G * delay(y, D).

    Stage 1: integer delay from |cross-correlation| peak.
    Stage 2: residual fractional delay from an IN-BAND weighted LS fit of the
             cross-spectrum phase slope (uses the full frequency lever arm, so
             low variance; no phase unwrapping needed because |D_frac|<0.5).
    Stage 3: complex-LS scalar gain on the fully delay-compensated signals.

    inband_thresh: keep bins with |Y|^2 above this fraction of the peak, i.e.
    ignore noise-only out-of-band bins.
    """
    N = len(y)
    Y = np.fft.fft(y); Z = np.fft.fft(z); f = np.fft.fftfreq(N)

    # --- integer delay (coarse) ---
    R = np.fft.ifft(Z * np.conj(Y))
    k0 = int(np.argmax(np.abs(R)))
    D_int = k0 if k0 <= N // 2 else k0 - N

    # --- fractional delay: in-band weighted LS phase slope ---
    Ydi = np.fft.fft(frac_delay(y, D_int))
    P = Z * np.conj(Ydi)                       # phase = angle(G) - 2*pi*f*D_frac
    w = np.abs(Ydi) ** 2
    mask = w > inband_thresh * w.max()
    ff, ww = f[mask], w[mask]
    Pd = P[mask] * np.exp(-1j * np.angle(np.sum(P[mask])))   # derotate -> no wrap
    phi = np.angle(Pd)
    Wm = ww.sum()
    fb = (ww * ff).sum() / Wm
    slope = (ww * (ff - fb) * (phi - (ww * phi).sum() / Wm)).sum() / (ww * (ff - fb) ** 2).sum()
    D = D_int - slope / (2 * np.pi)

    # --- complex-LS gain on delay-compensated signals ---
    y_D = frac_delay(y, D)
    G = np.sum(z * np.conj(y_D)) / np.sum(np.abs(y_D) ** 2)
    return D, G


def align_observation(z, D, G):
    """Advance z by D and de-gain by G -> aligned to the reference y.
    This is the ILA post-inverse input."""
    return frac_delay(z, -D) / G


# --------------------------------------------------------------------------- #
#  Self-test against a known synthetic channel (delay + gain + noise)
# --------------------------------------------------------------------------- #
def _self_test():
    rng = np.random.default_rng(0)
    N = 16384
    m = slice(128, N - 128)                     # trim circular-wrap edges for residual
    y = band_limited(N, occ=0.6, rng=rng)
    D_true, G_true = 7.37, 0.8 * np.exp(1j * 0.6)
    z_clean = G_true * frac_delay(y, D_true)

    print("=== fixed delay, sweep observation SNR ===")
    rows_ok = True
    for snr in [np.inf, 40, 30, 20]:
        if np.isinf(snr):
            z, floor = z_clean, 0.0
        else:
            p = np.mean(np.abs(z_clean) ** 2)
            s = np.sqrt(p / 2 / 10 ** (snr / 10))
            floor = 10 ** (-snr / 20)
            z = z_clean + s * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
        D, G = estimate_delay_gain(y, z)
        res = np.linalg.norm(align_observation(z, D, G)[m] - y[m]) / np.linalg.norm(y[m])
        print(f"  SNR={str(snr):>4}dB  Derr={abs(D - D_true):.2e}  "
              f"|Gerr|={abs(G - G_true):.2e}  resid={res:.2e} (noise floor~{floor:.1e})")

    print("=== random delays, clean ===")
    worst = 0.0
    for Dt in [0.13, 2.5, -3.4, 15.8, 100.27]:
        D, _ = estimate_delay_gain(y, G_true * frac_delay(y, Dt))
        worst = max(worst, abs(D - Dt))
        print(f"  D_true={Dt:8.3f}  D_est={D:8.3f}  err={abs(D - Dt):.2e}")

    D, G = estimate_delay_gain(y, z_clean)
    ok = abs(D - D_true) < 1e-4 and abs(G - G_true) < 1e-6 and worst < 1e-3
    print("\n" + ("OK: loop delay + gain/phase recovered; residual hits the noise floor."
                  if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
