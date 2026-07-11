#!/usr/bin/env python3
"""
online_ila.py -- ILA variant: online adaptation WITHOUT a PA gradient.

This is the HARDWARE-FAITHFUL path. A real chip cannot backprop through the
physical PA, so DLA (online_adaptation.py) is not deployable as-is. ILA instead trains
a POST-INVERSE from the observed PA output back to its input:

    u   = x                      ideal transmit signal
    xpd = M(u)                   predistorter output  (M = the model we adapt)
    z   = PA(xpd)                observed PA output    (PA = black box, forward only)
    align z -> xpd via ila_frontend (remove PA loop delay + gain)     [decision A3/A4]
    train:  minimize || M(z_aligned) - xpd ||^2   w.r.t. M   (NO gradient through PA)
    deploy: the SAME M as predistorter (post-inverse ~= pre-inverse)

Key contrast with DLA: the gradient graph is M(z)->loss only. z and xpd are DATA
(detached). The PA is used purely forward. This is what a feedback-receiver DPD
loop does in silicon, and it is why the O(1) forward-mode RTRL gradient (verified
against autograd to machine precision) is the right hardware primitive.

Reuses online_adaptation.py's paper-faithful eval + models + ila_frontend aligner.
Run:  uv run experiments/online_ila.py
"""
import copy
import os
import numpy as np
import torch

from online_adaptation import (make_pa, make_dpd, evaluate, target_gain_of,
                           DATASET, DEVICE)
from modules.data_collector import load_dataset
from ila_frontend import estimate_delay_gain, frac_delay


def _cplx(a):
    a = np.asarray(a, np.float64)
    return a[..., 0] + 1j * a[..., 1]


def measure_loop(xpd_np, z_np):
    """Diagnostic only: estimate PA loop delay+gain (z ~= G*delay(xpd, D)).

    IMPORTANT (the ILA gain trap): the post-inverse must invert the PA's OWN
    gain/phase, normalizing by the TARGET gain (=target_gain=1), NOT by this
    estimated PA gain. Dividing z by the estimated |G| here would bake a constant
    scale error into the deployed predistorter (EVM floor ~ 20log10(|G|-1)).
    Since z = PA(xpd) directly (no separate feedback-receiver path in this
    PA-model sim), we feed z RAW and let M learn the inverse. On real silicon,
    this is where you would remove the FEEDBACK-PATH delay/gain (not the PA's)."""
    D, G = estimate_delay_gain(_cplx(xpd_np), _cplx(z_np))
    return float(D), complex(G)


def main():
    torch.manual_seed(0)
    print(f"device = {DEVICE}  |  ILA (no gradient through PA)")

    X_tr, y_tr, X_val, y_val, X_te, y_te = load_dataset(dataset_name=DATASET)
    X_tr = np.asarray(X_tr, np.float32); X_te = np.asarray(X_te, np.float32)
    tg = target_gain_of(X_tr, np.asarray(y_tr, np.float32))
    print(f"data: X_train {X_tr.shape}  X_test {X_te.shape}  target_gain={tg:.4f}")

    pa = make_pa()

    # baseline: frozen offline s4d_best (same paper-faithful eval)
    base = evaluate(make_dpd(load_offline=True), pa, X_te, tg)
    print(f"\n[offline s4d_best]  ACLR={base['ACLR']:.2f}  EVM={base['EVM']:.2f}  "
          f"NMSE={base['NMSE']:.2f} dB")

    # --- ILA online adaptation from a FRESH model ---
    M = make_dpd(load_offline=False)
    x_full = torch.tensor(X_tr[None]).to(DEVICE)         # (1, N, 2)
    N = x_full.shape[1]
    EPOCHS, fl, stride, lr0 = 40, 500, 500, 2e-3
    starts = list(range(0, N - fl, stride))
    opt = torch.optim.Adam(M.parameters(), lr=lr0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(starts), eta_min=lr0 * 0.02)
    lossf = torch.nn.MSELoss()
    rng = np.random.default_rng(0)

    print(f"\n[ILA online] {EPOCHS} epochs x {len(starts)} frames (fl={fl}), "
          f"post-inverse objective, PA forward-only")
    best = {"ACLR": 0.0}; best_state = None
    for epoch in range(EPOCHS):
        # ---- ILA outer step: predistort, observe, align (all detached) ----
        M.eval()
        with torch.no_grad():
            xpd = M(x_full)                              # predistorter output (1,N,2)
            z = pa(xpd)                                  # observed PA output (forward only)
        z_np = z[0].cpu().numpy()
        D, G = measure_loop(xpd[0].cpu().numpy(), z_np)  # diagnostic (PA gain/delay)
        z_al = z.detach()                                # post-inverse INPUT = observed z (raw)
        xpd_t = xpd.detach()                             # post-inverse TARGET (PA input)

        # ---- inner: train post-inverse M(z_al) -> xpd  (grad through M only) ----
        M.train()
        rng.shuffle(starts)
        for s in starts:
            pred = M(z_al[:, s:s + fl, :])
            loss = lossf(pred, xpd_t[:, s:s + fl, :])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        m = evaluate(M, pa, X_te, tg)                    # deploy M as predistorter, measure
        if m["ACLR"] < best["ACLR"]:
            best = m; best_state = copy.deepcopy(M.state_dict())
        if epoch % 4 == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch:2d}  loss={loss.item():.2e}  ACLR={m['ACLR']:.2f}  "
                  f"EVM={m['EVM']:.2f}  NMSE={m['NMSE']:.2f}  "
                  f"(align D={D:+.3f} |G|={abs(G):.3f})  lr={sched.get_last_lr()[0]:.1e}")

    if best_state is not None:
        M.load_state_dict(best_state)
    gap = best["ACLR"] - base["ACLR"]
    print(f"\n[result] ILA online best: ACLR={best['ACLR']:.2f}  EVM={best['EVM']:.2f}  NMSE={best['NMSE']:.2f} dB")
    print(f"         offline base     : ACLR={base['ACLR']:.2f}  EVM={base['EVM']:.2f}  NMSE={base['NMSE']:.2f} dB")
    print(f"         (DLA online ref  : ACLR -50.46  EVM -47.44 dB, online_adaptation.py)")
    verdict = "CONVERGED" if abs(gap) < 1.5 else f"gap {gap:+.2f} dB"
    print(f"         ACLR gap to offline = {gap:+.2f} dB  ({verdict})")
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)
    out_dir = os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(M.state_dict(), os.path.join(out_dir, "dpd_ila_best.pt"))
    print("  saved best ILA DPD -> results/dpd_ila_best.pt")


if __name__ == "__main__":
    main()
