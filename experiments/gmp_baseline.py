#!/usr/bin/env python3
"""
gmp_baseline.py -- the novelty litmus test: can a polynomial baseline do this too?

Traditional adaptive DPD = ILA + RLS on a GMP / memory-polynomial basis (the
textbook baseline; RLS with a forgetting factor tracks a time-varying PA). We run
it through the SAME nominal PA and the SAME drift as S4D-RTRL,
and compare on: linearization (ACLR/EVM), parameter count, and drift tracking.

The claim being tested: a complex diagonal SSM adapted by O(1) forward-mode RTRL
linearizes this wideband measured PA BETTER at comparable/fewer parameters than
RLS-GMP, and RLS is O(P^2)/sample vs RTRL's O(N).

RLS/GMP core validated on a synthetic Hammerstein PA (lin.NMSE -67.9 dB).
Run:  uv run experiments/gmp_baseline.py
"""
import numpy as np
import torch

from online_adaptation import (make_pa, target_gain_of, segment, DATASET, DEVICE,
                           FS, BW, NSUB, NPERSEG)
from drift_tracking import make_pa_fn, drift_params, N_BLOCKS, STEP_BLK
from modules.data_collector import load_dataset
from utils.metrics import NMSE, EVM, ACLR

# ----- GMP / complex-RLS (numpy) -----
def gmp_basis(x, M, K):
    """phi_{m,k}[n] = x[n-m]*|x[n-m]|^k, m<M, k<K -> (T, M*K) complex."""
    T = len(x); cols = []
    for m in range(M):
        xm = np.zeros(T, np.complex128); xm[m:] = x[:T - m]
        a = np.abs(xm)
        for k in range(K):
            cols.append(xm * a ** k)
    return np.stack(cols, axis=1)

def rls_stream(U, d, w, Pmat, lam):
    """Streaming complex RLS: fit d ~= w^H u, forgetting factor lam. In-place-ish."""
    for n in range(U.shape[0]):
        u = U[n]; pi = Pmat @ u
        k = pi / (lam + np.conj(u) @ pi)
        xi = d[n] - np.conj(w) @ u
        w = w + k * np.conj(xi)
        Pmat = (Pmat - np.outer(k, np.conj(u) @ Pmat)) / lam
    return w, Pmat

def apply_gmp(w, U):
    return np.conj(w) @ U.T                       # y[n] = w^H u[n]

def rls_init(P, delta=1e-2):
    return np.zeros(P, np.complex128), np.eye(P, dtype=np.complex128) / delta


# ----- torch PA forward on a numpy complex signal -----
def pafn_np(pa_fn, xc):
    xt = torch.tensor(np.stack([xc.real, xc.imag], -1)[None].astype(np.float32)).to(DEVICE)
    z = pa_fn(xt)[0].cpu().numpy()
    return z[:, 0] + 1j * z[:, 1]


def gmp_metrics(w, M, K, pa_fn, X_np, tg, nperseg=NPERSEG):
    xc = X_np[:, 0] + 1j * X_np[:, 1]
    y = apply_gmp(w, gmp_basis(xc, M, K))         # GMP DPD output
    z = pafn_np(pa_fn, y)                          # through PA
    yb = np.stack([z.real, z.imag], -1)
    n = (yb.shape[0] // nperseg) * nperseg
    yb = yb[:n].reshape(-1, nperseg, 2); gt = (tg * X_np[:n]).reshape(-1, nperseg, 2)
    return dict(ACLR=float(np.mean(ACLR(yb, fs=FS, nperseg=nperseg, bw_main_ch=BW, n_sub_ch=NSUB))),
                EVM=float(EVM(yb, gt, sample_rate=FS, bw_main_ch=BW, n_sub_ch=NSUB, nperseg=nperseg)))


def train_ila_rls(pa_fn, X_tr, M, K, lam=1.0, n_iter=3, Ttrain=15000):
    xc = X_tr[:Ttrain, 0] + 1j * X_tr[:Ttrain, 1]
    P = M * K; w, Pmat = rls_init(P)
    for it in range(n_iter):
        xpd = xc if it == 0 else apply_gmp(w, gmp_basis(xc, M, K))
        z = pafn_np(pa_fn, xpd)
        w, Pmat = rls_stream(gmp_basis(z, M, K), xpd, *rls_init(P), lam)
    return w, P


def main():
    torch.manual_seed(0)
    print(f"device = {DEVICE}  |  RLS-GMP vs S4D-RTRL")
    X_tr, y_tr, *_r, X_te, y_te = load_dataset(dataset_name=DATASET)
    X_tr = np.asarray(X_tr, np.float32); X_te = np.asarray(X_te, np.float32)
    tg = target_gain_of(X_tr, np.asarray(y_tr, np.float32))
    pa = make_pa()
    pa0 = make_pa_fn(pa, 0)                        # nominal PA (zero drift)

    # ---- 1) STATIC linearization vs GMP size ----
    print("\n=== static (nominal PA): RLS-GMP at several sizes ===")
    print(f"{'GMP (M,K)':>12} {'complex':>8} {'real params':>12} {'ACLR':>8} {'EVM':>8}")
    sizes = [(4, 3), (6, 4), (8, 5), (11, 7), (13, 9)]
    for M, K in sizes:
        w, P = train_ila_rls(pa0, X_tr, M, K, lam=1.0, n_iter=3)
        m = gmp_metrics(w, M, K, pa0, X_te, tg)
        print(f"{str((M,K)):>12} {P:>8} {2*P:>12} {m['ACLR']:>8.2f} {m['EVM']:>8.2f}")
    print("  S4D-RTRL (ILA) reference:  1048 real params   ACLR  -50.44   EVM  -47.22"
          "   (online_ila.py)")

    # ---- 2) DRIFT tracking: RLS-GMP (forgetting) vs S4D online ----
    Mg, Kg, lam = 11, 7, 0.9995                    # representative GMP + forgetting factor
    print(f"\n=== drift tracking: RLS-GMP (M={Mg},K={Kg}, {2*Mg*Kg} real params, lam={lam}) ===")
    xc_chunk = (X_tr[:6000, 0] + 1j * X_tr[:6000, 1])
    w, Pmat = rls_init(Mg * Kg)
    # warm-start on nominal so it begins linearized (like S4D from offline)
    for _ in range(3):
        xpd = apply_gmp(w, gmp_basis(xc_chunk, Mg, Kg)) if w.any() else xc_chunk
        w, Pmat = rls_stream(gmp_basis(pafn_np(pa0, xpd), Mg, Kg), xpd, *rls_init(Mg*Kg), 1.0)
    hist = {"ACLR": [], "EVM": []}
    print(f"{'blk':>3} {'ACLR':>8} {'EVM':>8}")
    for blk in range(N_BLOCKS):
        pa_fn = make_pa_fn(pa, blk)
        m = gmp_metrics(w, Mg, Kg, pa_fn, X_te, tg)
        hist["ACLR"].append(m["ACLR"]); hist["EVM"].append(m["EVM"])
        if blk % 4 == 0 or blk == STEP_BLK or blk == N_BLOCKS - 1:
            tag = "  <-- bias step" if blk == STEP_BLK else ""
            print(f"{blk:3d} {m['ACLR']:8.2f} {m['EVM']:8.2f}{tag}")
        # adapt (ILA + forgetting RLS) against current drifted PA
        xpd = apply_gmp(w, gmp_basis(xc_chunk, Mg, Kg))
        w, Pmat = rls_stream(gmp_basis(pafn_np(pa_fn, xpd), Mg, Kg), xpd, w, Pmat, lam)

    a = np.array(hist["ACLR"]); e = np.array(hist["EVM"])
    print(f"\n  RLS-GMP  drift: ACLR mean/final/worst {a.mean():.1f}/{a[-1]:.1f}/{a.max():.1f}  "
          f"EVM {e.mean():.1f}/{e[-1]:.1f}/{e.max():.1f}")
    print("  S4D-RTRL drift: ACLR mean/final/worst -41.3/-39.4/-28.2  EVM -33.2/-33.5/-14.7"
          "  (drift_tracking.py)")


if __name__ == "__main__":
    main()
