#!/usr/bin/env python3
"""
pole_basis_check.py -- can the DPD's FIXED pole basis span what the drift demands?

Pricing Abar adaptation with an optimizer is confounded.  fixedpoint_drift.py measures
whether online SGD can MOVE Abar (it cannot: |dAbar| <= 0.017 over a run, against the
~0.29 the `zero` drift asks for) -- not whether a moved Abar would HELP.  And handing Abar
to Adam conflates two different things: "Abar as pole-matcher" and "Abar as 64 extra free
parameters".

Strip the optimizer and the nonlinearity out.  The S4D layer's linear transfer function is

    Y(z) = sum_n C_n / (1 - Abar_n z^-1)          (+ a direct feedthrough via the skip path)

so C_eff are free RESIDUES over a FIXED POLE BASIS {Abar_n}.  The question "does C_eff
alone suffice" is then exactly: how well does that fixed basis approximate the inverse of
the drifted PA's linear factor, over the 200 MHz signal band?  That is a complex linear
least-squares problem in C -- solvable exactly, in closed form, in milliseconds.

  drift `zero`: PA gains H(z) = (1 - q z^-1)/(1 - q).  DPD must realize 1/H, a POLE at q.
  drift `pole`: PA gains H(z) = (1 - p)/(1 - p z^-1).  DPD must realize 1/H, a ZERO at p.

A zero is trivially realizable by residues (the numerator is linear in C).  A pole at
q = 0.5 is not in the span of poles at [0.794, 0.999] -- unless, over a band that covers
only 200/983 of the unit circle, it effectively is.  That is the whole question, and it
does not need an optimizer to answer.

Reported: band-limited relative approximation error of the fixed-pole fit, versus a fit
where one pole is free to move to q (the "Abar adaptation" oracle).

Run: uv run experiments/pole_basis_check.py
"""
import os
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from online_adaptation import make_dpd, PA_TYPE
from rtrl_rflo_full import extract
import drift_models

FS = drift_models.FS_DEFAULT          # 983.04 MHz
BW = 200e6                            # main channel
NFREQ = 2048


def band_grid(bw=BW, n=NFREQ):
    f = np.linspace(-bw / 2, bw / 2, n)
    return np.exp(2j * np.pi * f / FS)          # z on the unit circle, in-band only


def basis(poles, z):
    """columns: 1/(1 - a z^-1) for each pole, plus a constant (the linear skip path)."""
    M = 1.0 / (1.0 - poles[None, :] / z[:, None])
    return np.concatenate([M, np.ones((len(z), 1))], axis=1)


def fit(poles, z, target):
    B = basis(poles, z)
    C, *_ = np.linalg.lstsq(B, target, rcond=None)
    resid = B @ C - target
    return C, 20 * np.log10(np.linalg.norm(resid) / np.linalg.norm(target) + 1e-300)


def free_pole_fit(poles, z, target, n_restart=6, iters=300, lr=0.02):
    """Let ONE pole move (gradient descent on its position), residues re-solved exactly
    at every step.  This is the best 'Abar adaptation' could ever do with one pole."""
    best = (None, 1e9)
    rng = np.random.default_rng(0)
    zt = torch.tensor(z)
    tt = torch.tensor(target)
    fixed = torch.tensor(poles)
    for r in range(n_restart):
        # start the movable pole at a random existing pole, or near the target region
        p0 = poles[rng.integers(len(poles))] if r else poles[np.argmin(np.abs(poles))]
        pr = torch.tensor(p0.real, requires_grad=True, dtype=torch.float64)
        pi = torch.tensor(p0.imag, requires_grad=True, dtype=torch.float64)
        opt = torch.optim.Adam([pr, pi], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            p = torch.complex(pr, pi)
            p = torch.where(p.abs() > 0.999, p / p.abs() * 0.999, p)
            cols = [1.0 / (1.0 - fixed[None, :] / zt[:, None]),
                    (1.0 / (1.0 - p / zt))[:, None],
                    torch.ones((len(z), 1), dtype=torch.complex128)]
            B = torch.cat(cols, dim=1)
            C = torch.linalg.lstsq(B, tt[:, None]).solution
            resid = (B @ C)[:, 0] - tt
            loss = (resid.abs() ** 2).sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            p = torch.complex(pr, pi)
            p = torch.where(p.abs() > 0.999, p / p.abs() * 0.999, p)
            allp = np.concatenate([poles, [complex(p)]])
            _, db = fit(allp, z, target)
            if db < best[1]:
                best = (complex(p), db)
    return best


def main():
    dpd = make_dpd(load_offline=True).cpu().eval()
    P = extract(dpd)
    z = band_grid()

    print("=== does the fixed pole basis span the drift's demand? ===")
    print(f"PA={PA_TYPE}   band +/-{BW/2e6:.0f} MHz of fs={FS/1e6:.2f} MHz "
          f"({BW/FS*100:.1f}% of the unit circle)\n")

    for layer in ('Ab1', 'Ab2'):
        poles = P[layer].detach().numpy().flatten().astype(np.complex128)
        print(f"--- layer {layer}: {len(poles)} poles, |a| in "
              f"[{np.abs(poles).min():.4f}, {np.abs(poles).max():.4f}] ---")

        for kind, r in (("zero", drift_models.ZERO_R_MAX), ("pole", drift_models.POLE_R_MAX)):
            w = 2 * np.pi * drift_models.POLE_DF_MAX / FS
            qp = r * np.exp(1j * w)
            if kind == "zero":
                # PA: H = (1 - q z^-1)/(1-q)  ->  DPD must realize 1/H (a POLE at q)
                target = (1 - qp) / (1 - qp / z)
            else:
                # PA: H = (1-p)/(1 - p z^-1)  ->  DPD must realize 1/H (a ZERO at p)
                target = (1 - qp / z) / (1 - qp)

            _, db_fixed = fit(poles, z, target)
            p_free, db_free = free_pole_fit(poles, z, target)
            gain = db_fixed - db_free
            print(f"  drift={kind:<5s} r={r}  DPD must realize a "
                  f"{'POLE' if kind == 'zero' else 'ZERO'} at |q|={abs(qp):.3f}")
            print(f"    fixed-pole basis (C_eff only) : {db_fixed:7.2f} dB rel. error")
            print(f"    one pole free    (Abar adapt) : {db_free:7.2f} dB rel. error   "
                  f"(moved to |a|={abs(p_free):.4f})")
            print(f"    -> Abar buys {gain:+.2f} dB of approximation error\n")

    # How severe must a notch be before the fixed basis stops covering it?  The DPD's own
    # accuracy floor is NMSE ~ -42.8 dB; once the basis error climbs above that, C_eff-only
    # adaptation becomes the limiting term and the trace channels start to pay for
    # themselves.  This is the number the hardware decision actually turns on.
    print("=== sensitivity: `zero` drift severity vs fixed-basis error ===")
    print("(DPD accuracy floor is NMSE ~ -42.8 dB; basis error above that begins to bind)\n")
    print("| notch |q| | Ab1 fixed-basis err | Ab2 fixed-basis err | binds? |")
    print("|---|---|---|---|")
    w = 2 * np.pi * drift_models.POLE_DF_MAX / FS
    for r in (0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
        qp = r * np.exp(1j * w)
        target = (1 - qp) / (1 - qp / z)
        dbs = []
        for layer in ('Ab1', 'Ab2'):
            poles = P[layer].detach().numpy().flatten().astype(np.complex128)
            dbs.append(fit(poles, z, target)[1])
        binds = "YES" if max(dbs) > -42.8 else "no"
        print(f"| {r:.2f} | {dbs[0]:7.2f} dB | {dbs[1]:7.2f} dB | {binds} |")
    print("\nThe `zero` arm of fixedpoint_drift.py uses r_max = "
          f"{drift_models.ZERO_R_MAX} at the last block.")


if __name__ == "__main__":
    main()
